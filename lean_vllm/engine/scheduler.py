from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

from lean_vllm.config import Config
from lean_vllm.engine.block_manager import BlockManager
from lean_vllm.engine.policy import SchedulingPolicy
from lean_vllm.engine.sequence import Sequence, SequenceStatus
from lean_vllm.kv_transfer import KVConnectorMetadata, KVConnectorOutput, create_scheduler_connector
from lean_vllm.spec_decode.config import PLACEHOLDER_TOKEN_ID


@dataclass(slots=True)
class SchedulerOutput:
    """What one step should run. A sequence carries its own num_scheduled_tokens."""

    scheduled: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    dropped: list[Sequence] = field(default_factory=list)  # finished without ever sampling
    # Counted while scheduling: advance() clears num_scheduled_tokens.
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_queried_blocks: int = 0  # prefix cache, counted at admission
    num_cached_blocks: int = 0
    kv_connector_metadata: KVConnectorMetadata | None = None  # None without a connector

    def __bool__(self):
        return bool(self.scheduled)


@dataclass(slots=True)
class LaunchedRow:
    """One sampling row of a launched step. A requeue after launch voids its token."""

    seq: Sequence
    num_preemptions: int
    num_draft_tokens: int = 0  # verified beside its token; those rejected give back their slots


class QueueFull(Exception):
    """The waiting queue is at max_waiting_requests. The server answers 429."""


class InvalidRequest(Exception):
    """The request can never run as asked. The server answers 400."""


class DuplicateRequestId(InvalidRequest):
    """Another request with this id is still in flight, and seqs holds only one."""


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.max_waiting_requests = config.max_waiting_requests
        self.request_timeout = config.request_timeout
        self.long_prefill_token_threshold = config.long_prefill_token_threshold
        speculative = config.speculative
        self.num_speculative_tokens = speculative.num_speculative_tokens if speculative is not None else 0
        # Slots past a row's last token that the drafter writes, its first draft aside, as vLLM's lookahead.
        self.num_lookahead_tokens = max(self.num_speculative_tokens - 1, 0)
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_caching,
            recompute_last_hit=speculative is not None,
        )
        self.waiting = SchedulingPolicy.create(config.scheduling_policy)
        self.running: deque[Sequence] = deque()
        self.seqs: dict[str, Sequence] = {}  # live requests, for abort
        self.connector = create_scheduler_connector(config) if config.kv_transfer is not None else None
        self.recving: dict[str, Sequence] = {}  # admitted, their blocks loading from a prefill instance
        self.sending: dict[str, Sequence] = {}  # finished, their blocks held for a decode instance to read

    def is_finished(self):
        return not self.waiting and not self.running and not self.recving and not self.sending

    def add(self, seq: Sequence):
        # An aborted load or a held send still owns blocks under its id.
        if seq.request_id in self.seqs or seq.request_id in self.recving or seq.request_id in self.sending:
            raise DuplicateRequestId(f"{seq.request_id} is already in flight")
        if self.max_waiting_requests and len(self.waiting) >= self.max_waiting_requests:
            raise QueueFull(f"{len(self.waiting)} requests already waiting")
        self.seqs[seq.request_id] = seq
        self.waiting.add(seq)

    def abort(self, request_id: str, reason: str = "abort") -> bool:
        """Drop a request between steps. Returns False if it already finished."""
        seq = self.seqs.pop(request_id, None)
        if seq is None:
            return False
        if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS:
            self._finish(seq, reason)  # its blocks are being written; they are freed once the load ends
            return True
        queue = self.waiting if seq.status == SequenceStatus.WAITING else self.running
        queue.remove(seq)  # both queues expose remove()
        seq.drop_pending()
        self._finish(seq, reason)
        self._free(seq)
        return True

    def schedule(self) -> SchedulerOutput:
        """One token budget per step, running sequences first so decode is never starved."""
        dropped = self._expire_waiting()
        if not self.enable_chunked_prefill:
            output = self._schedule_whole_prompts()
            output.dropped = dropped + output.dropped
            if output:
                return self._with_connector_meta(output)  # prefill-only step
            dropped = output.dropped  # nothing to run, but the drops still owe an output
        output = SchedulerOutput(dropped=dropped)
        budget = self.max_num_batched_tokens
        still_running: deque[Sequence] = deque()
        # Most urgent first, so a victim is never one already served this step.
        self.running = self.waiting.by_urgency(self.running)

        while self.running:
            seq = self.running.popleft()
            if budget <= 0 or len(output.scheduled) >= self.max_num_seqs:
                still_running.append(seq)  # left untouched this step
                continue
            if seq.num_planned_tokens - seq.num_prompt_tokens >= seq.max_tokens:
                still_running.append(seq)  # its reserved tokens already reach the limit
                continue
            if not seq.is_prefill:  # decoding, so the cache grows
                # Drafts past the budget or the request's last token would be thrown away, so they never run.
                num_left = seq.max_tokens - (seq.num_planned_tokens - seq.num_prompt_tokens)
                del seq.spec_token_ids[max(min(budget, num_left) - 1, 0) :]
                num_lookahead = len(seq.spec_token_ids) + self.num_lookahead_tokens
                if not self._make_room(seq, still_running, output, num_lookahead):
                    continue
                self.block_manager.may_append(seq, num_lookahead)
            budget -= self._schedule(seq, budget, output)
            still_running.append(seq)
        self.running = still_running

        # Admitting new work while under memory pressure would only preempt again.
        if not output.preempted:
            while self.waiting and len(output.scheduled) < self.max_num_seqs and budget > 0:
                seq = self.waiting.peek()
                if seq.num_blocks > len(self.block_manager.blocks):
                    self.waiting.pop()  # impossible even with the entire cache free
                    self._drop(seq, "capacity", output)
                    continue
                num_cached_blocks = self.block_manager.can_allocate(seq, self.num_lookahead_tokens)
                if num_cached_blocks == -1:
                    break
                budget -= self._admit(seq, num_cached_blocks, budget, output)

        return self._with_connector_meta(output)

    def _with_connector_meta(self, output: SchedulerOutput) -> SchedulerOutput:
        if self.connector is not None:
            output.kv_connector_metadata = self.connector.build_connector_meta()
        return output

    def _expire_waiting(self) -> list[Sequence]:
        """Drop requests that waited past request_timeout without ever running. Preempted ones are kept."""
        if not self.request_timeout:
            return []
        deadline = perf_counter() - self.request_timeout
        expired = [seq for seq in self.waiting if seq.first_scheduled_time is None and seq.arrival_time < deadline]
        for seq in expired:
            self.waiting.remove(seq)
            self._drop(seq, "timeout")
        return expired

    def _schedule_whole_prompts(self) -> SchedulerOutput:
        """Chunked prefill disabled: whole prompts only, and never mixed with decode."""
        output = SchedulerOutput()
        budget = self.max_num_batched_tokens
        while self.waiting and len(output.scheduled) < self.max_num_seqs:
            seq = self.waiting.peek()
            if seq.num_blocks > len(self.block_manager.blocks):
                self.waiting.pop()
                self._drop(seq, "capacity", output)
                continue
            num_cached_blocks = self.block_manager.can_allocate(seq, self.num_lookahead_tokens)
            if num_cached_blocks == -1:
                break
            num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            if self._loads_remotely(seq):
                num_tokens = 0  # it waits for its blocks, then computes one token
            if num_tokens > self.max_num_batched_tokens:
                self.waiting.pop()  # will never fit in one step, and splitting is off
                self._drop(seq, "capacity", output)
                continue
            if num_tokens > budget:
                break
            budget -= self._admit(seq, num_cached_blocks, budget, output)
        return output

    def _loads_remotely(self, seq: Sequence) -> bool:
        params = seq.kv_transfer_params
        return self.connector is not None and bool(params and params.get("do_remote_prefill"))

    def _admit(self, seq: Sequence, num_cached_blocks: int, budget: int, output: SchedulerOutput) -> int:
        """Move the head of the waiting queue into the running set, or to wait for its blocks from elsewhere."""
        self.waiting.pop()
        self.block_manager.allocate(seq, num_cached_blocks, self.num_lookahead_tokens)
        output.num_queried_blocks += seq.num_blocks
        output.num_cached_blocks += num_cached_blocks
        if self.connector is not None:
            num_external_tokens, load_async = self.connector.get_num_new_matched_tokens(seq, seq.num_cached_tokens)
            if num_external_tokens:
                assert load_async, "only asynchronous loads are implemented"
                self.connector.update_state_after_alloc(seq, num_external_tokens)
                seq.status = SequenceStatus.WAITING_FOR_REMOTE_KVS
                self.recving[seq.request_id] = seq
                return 0
        seq.status = SequenceStatus.RUNNING
        num_tokens = self._schedule(seq, budget, output)
        self.running.append(seq)
        return num_tokens

    def _schedule(self, seq: Sequence, budget: int, output: SchedulerOutput) -> int:
        """Give seq its share of the budget: a prompt chunk, or one decoded token and the drafts to verify."""
        # A preempted request also prefills its generated suffix until it samples again.
        num_tokens = (
            min(seq.num_tokens - seq.num_cached_tokens, budget) if seq.is_prefill else 1 + len(seq.spec_token_ids)
        )
        if seq.is_prefill and self.enable_chunked_prefill and self.long_prefill_token_threshold:
            num_tokens = min(num_tokens, self.long_prefill_token_threshold)
        seq.num_scheduled_tokens = num_tokens
        if seq.first_scheduled_time is None:
            seq.first_scheduled_time = perf_counter()
        output.scheduled.append(seq)
        if seq.is_prefill:
            output.num_prefill_tokens += num_tokens
        else:
            output.num_decode_tokens += num_tokens
        return num_tokens

    def _make_room(
        self, seq: Sequence, still_running: deque[Sequence], output: SchedulerOutput, num_lookahead: int = 0
    ) -> bool:
        """Free blocks for one more decoded token and num_lookahead slots past it. False if seq itself gave way."""
        while not self.block_manager.can_append(seq, num_lookahead):
            if self.running:
                victim = self.waiting.victim(self.running)
                self.running.remove(victim)
                self._preempt(victim, output)
            elif still_running or output.scheduled:
                self._preempt(seq, output)
                return False
            elif seq.num_pending_tokens:
                still_running.append(seq)  # its in-flight token may stop it; decide after reconcile
                return False
            elif self.recving or self.sending:
                # Transfers still own blocks outside running. Wait until they
                # finish before deciding whether this request can ever fit.
                still_running.append(seq)
                return False
            else:
                # Alone in the cache and still short of a block: it can never fit.
                self.block_manager.deallocate(seq)
                self._drop(seq, "capacity", output)
                return False
        return True

    def _preempt(self, seq: Sequence, output: SchedulerOutput):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_preemptions += 1
        seq.drop_pending()  # the in-flight token is discarded and recomputed
        seq.spec_token_ids = []  # the step that samples again drafts anew
        self.block_manager.deallocate(seq)
        self.waiting.requeue(seq)
        output.preempted.append(seq)

    def _finish(self, seq: Sequence, reason: str):
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = perf_counter()

    def _drop(self, seq: Sequence, reason: str, output: SchedulerOutput | None = None):
        self._finish(seq, reason)
        seq.drop_pending()
        del self.seqs[seq.request_id]
        if output is not None:
            output.dropped.append(seq)  # the caller is still owed a final output

    def advance(self, seqs: list[Sequence]) -> list[LaunchedRow]:
        """Move bookkeeping forward with no token values. Returns the sampling rows, in the sampler's order."""
        rows = []
        for seq in seqs:
            # Optimistic, as vLLM's num_computed_tokens: reconcile takes back the slots of rejected drafts.
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            num_draft_tokens, seq.spec_token_ids = len(seq.spec_token_ids), []
            # Before the skip, so a chunked prefill publishes each block as it lands.
            self.block_manager.hash_blocks(seq, seq.num_cached_tokens)
            if seq.num_cached_tokens < seq.num_planned_tokens:
                continue  # prefill or recomputation unfinished, so this row samples nothing
            seq.is_prefill = False
            # Its next token and, optimistically, every draft it verifies, as vLLM's num_output_placeholders.
            seq.reserve_token(1 + num_draft_tokens)
            # The drafts it makes are on the device until reconcile; an async step schedules placeholders.
            seq.spec_token_ids = [PLACEHOLDER_TOKEN_ID] * self.num_speculative_tokens
            rows.append(LaunchedRow(seq, seq.num_preemptions, num_draft_tokens))
        return rows

    def reconcile(
        self,
        rows: list[LaunchedRow],
        token_ids: list[int] | list[list[int]],
        draft_token_ids: list[list[int]] | None = None,
    ) -> list[Sequence]:
        """Commit the sampled tokens and run the stop checks. Returns the rows that produced one.

        A speculative step gives each row its kept drafts and one more token, and the drafts for its next step.
        """
        stepped = []
        for i, row in enumerate(rows[: len(token_ids)]):
            seq, sampled = row.seq, token_ids[i]
            if seq.is_finished or seq.num_preemptions != row.num_preemptions:
                continue  # aborted, finished or requeued since the launch; the token is void
            new_token_ids = [sampled] if isinstance(sampled, int) else sampled
            # The rejected drafts' slots hold keys of the wrong tokens, so the next step writes over them, and their
            # reservations go. A step launched since was placed as if they were kept; the runner moved it back.
            num_rejected = row.num_draft_tokens - (len(new_token_ids) - 1)
            seq.num_cached_tokens -= num_rejected
            seq.num_pending_tokens -= num_rejected
            reason = None
            seq.num_new_tokens = 0
            for token_id in new_token_ids:
                seq.commit_token(token_id)
                seq.num_new_tokens += 1
                if (token_id == self.eos and not seq.ignore_eos) or token_id in seq.stop_token_ids:
                    reason = "stop"  # ignore_eos covers the eos token only, not client stop tokens
                elif seq.num_completion_tokens == seq.max_tokens:
                    reason = "length"
                if reason is not None:
                    break  # the tokens after a stop are dropped, as vLLM's
            if seq.first_token_time is None:
                seq.first_token_time = perf_counter()  # when the token reaches the host, not at launch
            stepped.append(seq)
            if reason is None:
                if draft_token_ids is not None and not seq.num_pending_tokens:  # else a launched step verifies them
                    seq.spec_token_ids = draft_token_ids[i]
                continue
            seq.drop_pending()  # a later step may already have reserved one
            self.running.remove(seq)
            self._drop(seq, reason)
            self._free(seq)
        return stepped

    def _free(self, seq: Sequence):
        """Release a finished request's blocks, unless the connector holds them for a decode instance to read."""
        if self.connector is not None:
            hold, seq.kv_transfer_result = self.connector.request_finished(seq)
            if hold:
                self.sending[seq.request_id] = seq
                return
        self.block_manager.deallocate(seq)

    def update_from_kv_connector_output(self, kv_output: KVConnectorOutput):
        """Free what was sent, and run what was loaded. A failed load prefills here, from the local prefix hits."""
        for request_id in kv_output.finished_sending:
            seq = self.sending.pop(request_id, None)
            if seq is not None:
                self.block_manager.deallocate(seq)
        for request_id in kv_output.finished_recving | kv_output.failed_recving:
            seq = self.recving.pop(request_id)
            if seq.is_finished:  # aborted while its blocks loaded
                self.block_manager.deallocate(seq)
                continue
            if request_id in kv_output.finished_recving:
                # As vLLM: the last prompt token recomputes, so this engine samples the first token itself.
                seq.num_cached_tokens = seq.num_prompt_tokens - 1
                self.block_manager.hash_blocks(seq, seq.num_cached_tokens)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
