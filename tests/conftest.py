"""A scheduler harness with no torch, no model and no GPU, so scheduling is testable on a laptop."""

import asyncio
from dataclasses import dataclass
from functools import wraps
from itertools import count
from time import perf_counter

import pytest

from lean_vllm.engine.metrics import Metrics
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.scheduler import QueueFull, Scheduler
from lean_vllm.engine.sequence import Sequence
from lean_vllm.kv_transfer import KVConnectorOutput
from lean_vllm.sampling_params import SamplingParams
from lean_vllm.spec_decode import split_sampled

EOS = 7


@dataclass
class FakeConfig:
    """The fields Scheduler reads. Config itself needs a model directory on disk."""

    num_kvcache_blocks: int = 64
    kvcache_block_size: int = 8
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 1024
    max_model_len: int = 4096
    eos: int = EOS
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "sha256"
    scheduling_policy: str = "fcfs"
    max_waiting_requests: int = 0
    request_timeout: float = 0.0
    long_prefill_token_threshold: int = 0
    async_scheduling: bool = False  # off, unlike Config: most tests count steps in sync order
    tensor_parallel_size: int = 1
    kv_transfer: object = None  # a KVTransferConfig turns the connector on
    speculative: object = None  # a SpeculativeConfig turns drafting on


class FakeSampledTokens:
    """What ModelRunner.run returns, without torch."""

    def __init__(self, token_ids: list[int]):
        self._token_ids = token_ids

    def tolist(self) -> list[int]:
        return self._token_ids


class FakeModelRunner:
    """Stands in for ModelRunner: same call() surface, deterministic tokens, every batch recorded."""

    def __init__(self, eos_after: dict[str, int] | None = None):
        self.eos_after = eos_after or {}
        self.batches: list[tuple[bool, list[tuple[str, int]]]] = []
        self.kv_metadata: list = []  # what each kv_connector_step was handed
        self.kv_outputs: list = []  # what the next ones report, oldest first

    def kv_connector_step(self, metadata):
        self.kv_metadata.append(metadata)
        return self.kv_outputs.pop(0) if self.kv_outputs else KVConnectorOutput()

    def call(self, method_name, *args):
        return getattr(self, method_name)(*args)

    def run(self, seqs: list[Sequence]) -> "FakeSampledTokens":
        is_prefill = any(seq.is_prefill for seq in seqs)
        self.batches.append((is_prefill, [(seq.request_id, seq.num_scheduled_tokens) for seq in seqs]))
        return FakeSampledTokens([self._token(seq) for seq in seqs if self._samples(seq)])

    @staticmethod
    def _samples(seq: Sequence) -> bool:
        return seq.num_cached_tokens + seq.num_scheduled_tokens >= seq.num_planned_tokens

    @staticmethod
    def _completion_index(seq: Sequence) -> int:
        """Which completion token this row is about to produce, read off the batch so a recomputed suffix repeats."""
        return seq.num_cached_tokens + seq.num_scheduled_tokens - seq.num_prompt_tokens

    def _token(self, seq: Sequence) -> int:
        limit = self.eos_after.get(seq.request_id)
        index = self._completion_index(seq)
        if limit is not None and index >= limit:
            return EOS
        return 1000 + seq.seq_id * 100 + index

    def exit(self):
        pass


class FakeEngine:
    """LLMEngine.step without the model. Mirrors it deliberately: change both together."""

    def __init__(self, config: FakeConfig, runner: FakeModelRunner):
        self.config = config
        self.scheduler = Scheduler(config)
        self.model_runner = runner
        self.metrics = Metrics()
        self.last_output = None
        self.in_flight = None
        self.profiler = None  # real LLMEngine exposes one; the async loop reads it

    def add(
        self, prompt: list[int], sampling_params: SamplingParams | None = None, request_id: str | None = None
    ) -> Sequence:
        seq = Sequence(prompt, sampling_params or SamplingParams(), request_id)
        try:
            self.scheduler.add(seq)
        except QueueFull:
            self.metrics.record_rejected()
            raise
        self.metrics.record_received()
        return seq

    def step(self) -> list[RequestOutput]:
        started = perf_counter()
        draining, self.in_flight = self.in_flight, None
        if self.config.async_scheduling:
            output = self._launch()
            stepped = self._reconcile(draining)
        else:
            stepped = self._reconcile(draining)
            output = self._launch()
        outputs = [
            RequestOutput(
                request_id=seq.request_id,
                token_ids=seq.token_ids[-seq.num_new_tokens :],
                finished=seq.is_finished,
                finish_reason=seq.finish_reason,
                metrics=seq.metrics() if seq.is_finished else None,
                kv_transfer_params=seq.kv_transfer_result,
            )
            for seq in stepped
        ]
        outputs += [
            RequestOutput(
                request_id=seq.request_id,
                token_ids=[],
                finished=True,
                finish_reason=seq.finish_reason,
                metrics=seq.metrics(),
            )
            for seq in output.dropped
            if seq not in stepped
        ]
        # attributed to the step this call launched
        self.metrics.record_step(self.scheduler, output, outputs, perf_counter() - started, "enforced")
        return outputs

    def _launch(self):
        output = self.last_output = self.scheduler.schedule()
        if output.kv_connector_metadata is not None:
            kv_output = self.model_runner.call("kv_connector_step", output.kv_connector_metadata)
            self.scheduler.update_from_kv_connector_output(kv_output)
            self.metrics.record_kv_transfer(kv_output)
        if output:
            pending = self.model_runner.call("run", output.scheduled)
            rows = self.scheduler.advance(output.scheduled)
            self.in_flight = (output, rows, pending)
        return output

    def _reconcile(self, draining) -> list[Sequence]:
        if draining is None:
            return []
        _, rows, pending = draining
        token_ids, draft_token_ids = pending.tolist(), None
        if self.config.speculative is not None:
            token_ids, draft_token_ids = split_sampled(token_ids, self.config.speculative.num_speculative_tokens)
            self.metrics.record_spec_decoding(rows, token_ids)
        return self.scheduler.reconcile(rows, token_ids, draft_token_ids)

    def is_finished(self):
        return self.in_flight is None and self.scheduler.is_finished()

    def run_to_completion(self, max_steps: int = 500) -> dict[str, list[int]]:
        """Completion token ids per request, as generate() would accumulate them."""
        outputs: dict[str, list[int]] = {}
        for _ in range(max_steps):
            if self.is_finished():
                return outputs
            for output in self.step():
                outputs.setdefault(output.request_id, []).extend(output.token_ids)
        raise AssertionError("engine did not finish; a sequence is stuck")


class FakeLLMEngine(FakeEngine):
    """LLMEngine's surface, so the core clients can drive it without a model."""

    tokenizer = None  # the async engine only needs one for str prompts

    def add_request(self, prompt: list[int], sampling_params=None, request_id: str | None = None) -> str:
        return self.add(prompt, sampling_params, request_id).request_id

    def abort_request(self, request_id: str, reason: str = "abort") -> bool:
        seq = self.scheduler.seqs.get(request_id)
        aborted = self.scheduler.abort(request_id, reason)
        if aborted:
            self.metrics.record_aborted(
                RequestOutput(
                    request_id=request_id,
                    token_ids=[],
                    finished=True,
                    finish_reason=seq.finish_reason,
                    metrics=seq.metrics(),
                )
            )
        return aborted

    def step(self) -> tuple[list[RequestOutput], int, int]:
        outputs = super().step()
        return outputs, self.last_output.num_prefill_tokens, self.last_output.num_decode_tokens


def asyncio_test(test):
    """No async plugin in the dev group, so each test drives its own loop."""

    @wraps(test)
    def wrapper(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return wrapper


@pytest.fixture(autouse=True)
def _reset_sequence_globals():
    """What Sequence takes from Config, plus the id counter, is class state."""
    saved = (Sequence.block_size, Sequence.enable_prefix_caching, Sequence.hash_algo, Sequence.counter)
    Sequence.counter = count()  # so seq ids are deterministic per test
    yield
    (Sequence.block_size, Sequence.enable_prefix_caching, Sequence.hash_algo, Sequence.counter) = saved


@pytest.fixture
def make_engine():
    def _make(eos_after: dict[int, int] | None = None, **overrides) -> FakeEngine:
        config = FakeConfig(**overrides)
        Sequence.counter = count()  # each engine gets its own seq ids, so two engines compare like-for-like
        Sequence.block_size = config.kvcache_block_size
        Sequence.enable_prefix_caching = config.enable_prefix_caching
        Sequence.hash_algo = config.prefix_caching_hash_algo
        return FakeEngine(config, FakeModelRunner(eos_after))

    return _make
