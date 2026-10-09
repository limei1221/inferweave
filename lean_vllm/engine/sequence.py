import hashlib
import pickle
from copy import copy
from enum import Enum, auto
from itertools import count
from time import perf_counter

import xxhash

from lean_vllm.engine.output import RequestMetrics
from lean_vllm.sampling_params import SamplingParams


def _serialize(token_ids: tuple[int, ...], prefix: int) -> bytes:
    """Pickle, as vLLM does. Stable within a Python version, not across them."""
    return pickle.dumps((prefix, token_ids), protocol=pickle.HIGHEST_PROTOCOL)


def sha256_hash(token_ids: tuple[int, ...], prefix: int) -> int:
    """The default, and vLLM's. A collision would serve one tenant another's tokens."""
    return int.from_bytes(hashlib.sha256(_serialize(token_ids, prefix)).digest(), "big")


def xxhash_hash(token_ids: tuple[int, ...], prefix: int) -> int:
    """Faster, not cryptographic. Worth it only where every request is trusted."""
    return int.from_bytes(xxhash.xxh128(_serialize(token_ids, prefix)).digest(), "big")


HASH_ALGOS = {"sha256": sha256_hash, "xxhash": xxhash_hash}


class SequenceStatus(Enum):
    WAITING = auto()
    WAITING_FOR_REMOTE_KVS = auto()  # admitted, its blocks loading from a prefill instance
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # Set once from Config: Sequences are built where no config is at hand (warmup, TP workers).
    block_size = 16
    enable_prefix_caching = True
    hash_algo = "sha256"
    counter = count()

    def __init__(
        self, token_ids: list[int], sampling_params: SamplingParams | None = None, request_id: str | None = None
    ):
        sampling_params = sampling_params or SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.request_id = request_id if request_id is not None else f"req-{self.seq_id}"
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_pending_tokens = 0  # reserved by a launched step, not yet sampled here
        self.num_published_blocks = 0  # blocks already in the prefix cache
        self.num_scheduled_tokens = 0
        self.num_new_tokens = 0  # committed by the last reconcile, which the step's output carries
        self.spec_token_ids: list[int] = []  # drafts for the next step to verify
        self.is_prefill = True
        self.block_table: list[int] = []
        self.block_hashes: list[int] = []  # chained, one per full block, filled on demand
        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.stop_token_ids = sampling_params.stop_token_ids
        self.skip_special_tokens = sampling_params.skip_special_tokens
        self.priority = sampling_params.priority
        # Copied, as the connector marks it and generate() shares one SamplingParams across prompts.
        params = sampling_params.kv_transfer_params
        self.kv_transfer_params: dict | None = dict(params) if params is not None else None
        self.kv_transfer_result: dict | None = None  # what the connector hands back on finishing
        self.finish_reason: str | None = None
        self.num_preemptions = 0
        self.arrival_time = perf_counter()
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        self._extend_block_hashes()

    def metrics(self) -> RequestMetrics:
        return RequestMetrics(
            arrival_time=self.arrival_time,
            num_prompt_tokens=self.num_prompt_tokens,
            num_completion_tokens=self.num_completion_tokens,
            num_preemptions=self.num_preemptions,
            first_scheduled_time=self.first_scheduled_time,
            first_token_time=self.first_token_time,
            finish_time=self.finish_time,
        )

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_planned_tokens(self):
        """Length once every reserved token lands. What allocation must cover."""
        return self.num_tokens + self.num_pending_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_blocks(self):
        return (self.num_planned_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def _extend_block_hashes(self):
        """Hash every block that has just become full, chaining on the one before. Computed once, as it is final."""
        if not self.enable_prefix_caching:
            return
        algo = HASH_ALGOS[self.hash_algo]
        for i in range(len(self.block_hashes), self.num_tokens // self.block_size):
            prefix = self.block_hashes[-1] if self.block_hashes else -1
            self.block_hashes.append(algo(tuple(self.block(i)), prefix))

    def reserve_token(self, num_tokens: int = 1):
        """A launched step will sample here, or keep drafts here; the values are not known yet."""
        self.num_pending_tokens += num_tokens

    def commit_token(self, token_id: int):
        assert self.num_pending_tokens, "commit_token without a reservation"
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.num_pending_tokens -= 1
        self._extend_block_hashes()

    def append_token(self, token_id: int):
        self.reserve_token()
        self.commit_token(token_id)

    def drop_pending(self):
        """Give up reserved tokens, on preemption, abort or finish."""
        self.num_pending_tokens = 0

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.num_pending_tokens,
            self.is_prefill,
            self.block_table,
            self.spec_token_ids,
            last_state,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.num_pending_tokens,
            self.is_prefill,
            self.block_table,
            self.spec_token_ids,
            last_state,
        ) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
