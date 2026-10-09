"""The scheduler, against the fake runner in conftest: no model, no GPU.

TestChunkedPrefillDisabled covers `enable_chunked_prefill=False`, kept as the A/B baseline.
"""

import os
import subprocess
import sys

import pytest

from lean_vllm.engine import sequence
from lean_vllm.engine.scheduler import DuplicateRequestId, SchedulerOutput
from lean_vllm.sampling_params import SamplingParams

FOREVER = SamplingParams(max_tokens=64, ignore_eos=True)


def prompt(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


class TestGeneration:
    def test_every_request_returns_max_tokens(self, make_engine):
        engine = make_engine()
        seqs = [engine.add(prompt(8, s * 100), SamplingParams(max_tokens=5, ignore_eos=True)) for s in range(4)]
        outputs = engine.run_to_completion()
        assert len(outputs) == 4
        assert all(len(outputs[seq.request_id]) == 5 for seq in seqs)

    def test_eos_stops_generation_and_is_kept(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=64))
        engine.model_runner.eos_after[seq.request_id] = 3
        outputs = engine.run_to_completion()
        assert len(outputs[seq.request_id]) == 4  # three tokens, then the eos itself

    def test_ignore_eos_runs_to_max_tokens(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=5, ignore_eos=True))
        engine.model_runner.eos_after[seq.request_id] = 1
        assert len(engine.run_to_completion()[seq.request_id]) == 5

    def test_finished_sequences_free_their_blocks(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.run_to_completion()
        block_manager = engine.scheduler.block_manager
        assert not block_manager.used_block_ids
        assert len(block_manager.free_block_ids) == engine.config.num_kvcache_blocks


class TestStreaming:
    def test_a_token_is_reported_every_step(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.step()  # prefill launched; its token is still in flight
        assert [(o.token_ids, o.finished) for o in engine.step()] == [([1000], False)]
        assert [(o.token_ids, o.finished) for o in engine.step()] == [([1001], False)]
        assert engine.step()[0].finished

    def test_a_partial_prefill_reports_nothing(self, make_engine):
        engine = make_engine(max_num_batched_tokens=16)
        engine.add(prompt(40), FOREVER)
        assert engine.step() == []
        assert engine.step() == []
        assert engine.step() == []
        assert len(engine.step()) == 1  # fourth call drains the chunk that completed the prompt


class TestFinishReason:
    def test_max_tokens_is_length(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.run_to_completion()
        assert seq.finish_reason == "length"

    def test_eos_is_stop(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=64))
        engine.model_runner.eos_after[seq.request_id] = 2
        engine.run_to_completion()
        assert seq.finish_reason == "stop"

    def test_a_client_stop_token_is_honoured_even_with_ignore_eos(self, make_engine):
        """ignore_eos covers the eos token only, as in vLLM."""
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=64, ignore_eos=True, stop_token_ids=[1002]))
        engine.run_to_completion()
        assert seq.finish_reason == "stop" and seq.completion_token_ids == [1000, 1001, 1002]


class TestMetrics:
    def test_timestamps_are_ordered(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.run_to_completion()
        m = seq.metrics()
        assert m.arrival_time <= m.first_scheduled_time <= m.first_token_time <= m.finish_time
        assert m.queue_time >= 0 and m.ttft >= 0 and m.e2e >= m.ttft
        assert m.num_prompt_tokens == 8 and m.num_completion_tokens == 3

    def test_tpot_needs_more_than_one_token(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=1, ignore_eos=True))
        engine.run_to_completion()
        assert seq.metrics().tpot is None

    def test_preemptions_are_counted(self, make_engine):
        engine = make_engine(num_kvcache_blocks=2, kvcache_block_size=8)
        engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        engine.step()
        assert second.num_preemptions == 1


class TestAbort:
    def test_aborting_a_waiting_request_frees_nothing_and_drops_it(self, make_engine):
        engine = make_engine()
        first = engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), FOREVER)
        assert engine.scheduler.abort(second.request_id)
        engine.step()
        assert [request_id for request_id, _ in engine.model_runner.batches[0][1]] == [first.request_id]
        assert second.finish_reason == "abort"

    def test_aborting_a_running_request_frees_its_blocks(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(16), FOREVER)
        engine.step()
        blocks = list(seq.block_table)
        assert engine.scheduler.abort(seq.request_id)
        assert seq.block_table == []
        block_manager = engine.scheduler.block_manager
        assert not block_manager.used_block_ids.intersection(blocks)
        engine.step()  # drains the in-flight step launched before the abort
        assert engine.is_finished()

    def test_aborting_an_unknown_request_is_a_no_op(self, make_engine):
        engine = make_engine()
        assert not engine.scheduler.abort("nope")

    def test_aborting_a_finished_request_is_a_no_op(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.run_to_completion()
        assert not engine.scheduler.abort(seq.request_id)


class TestBatching:
    def test_prefill_batches_many_sequences(self, make_engine):
        engine = make_engine()
        for s in range(4):
            engine.add(prompt(8, s * 100), FOREVER)
        engine.step()
        is_prefill, batch = engine.model_runner.batches[0]
        assert is_prefill and [n for _, n in batch] == [8, 8, 8, 8]

    def test_decode_schedules_one_token_per_sequence(self, make_engine):
        engine = make_engine()
        for s in range(3):
            engine.add(prompt(8, s * 100), FOREVER)
        engine.step()
        engine.step()
        is_prefill, batch = engine.model_runner.batches[1]
        assert not is_prefill and [n for _, n in batch] == [1, 1, 1]

    def test_max_num_seqs_caps_the_batch(self, make_engine):
        engine = make_engine(max_num_seqs=2)
        for s in range(5):
            engine.add(prompt(8, s * 100), FOREVER)
        engine.step()
        assert len(engine.model_runner.batches[0][1]) == 2

    def test_an_arriving_prefill_mixes_with_running_decodes(self, make_engine):
        """An arrival must not stall decoding."""
        engine = make_engine()
        running = engine.add(prompt(8), FOREVER)
        engine.step()
        arriving = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        # one batch, both kinds of row in it
        assert engine.model_runner.batches[1] == (True, [(running.request_id, 1), (arriving.request_id, 8)])


class TestBudget:
    def test_running_sequences_are_scheduled_before_arrivals(self, make_engine):
        engine = make_engine(max_num_batched_tokens=1)
        running = engine.add(prompt(1), FOREVER)
        engine.step()
        arriving = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        assert engine.model_runner.batches[1] == (False, [(running.request_id, 1)])
        assert arriving in engine.scheduler.waiting  # the budget went to decode

    def test_budget_is_shared_across_prefill_and_decode(self, make_engine):
        engine = make_engine(max_num_batched_tokens=10)
        decoding = engine.add(prompt(8), FOREVER)
        prefilling = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(decoding.request_id, 8), (prefilling.request_id, 2)])
        engine.step()
        assert engine.model_runner.batches[1] == (True, [(decoding.request_id, 1), (prefilling.request_id, 6)])

    def test_no_admission_in_a_step_that_preempted(self, make_engine):
        """Admitting under memory pressure would only preempt again."""
        engine = make_engine(num_kvcache_blocks=3, kvcache_block_size=8)
        first = engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        waiting = engine.add(prompt(8, 200), FOREVER)
        engine.step()
        assert second.num_preemptions == 1
        assert engine.model_runner.batches[1] == (False, [(first.request_id, 1)])
        assert waiting in engine.scheduler.waiting


class TestChunkedPrefill:
    def test_long_prompt_is_split_across_steps(self, make_engine):
        engine = make_engine(max_num_batched_tokens=16)
        seq = engine.add(prompt(40), FOREVER)
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(seq.request_id, 16)])
        assert seq in engine.scheduler.running  # admitted, prompt still filling
        engine.step()
        engine.step()
        assert [n for _, n in engine.model_runner.batches[2][1]] == [8]
        engine.step()  # drains the chunk that completed the prompt
        assert seq.num_completion_tokens == 1  # only the final chunk samples
        assert engine.model_runner.batches[3] == (False, [(seq.request_id, 1)])

    def test_leftover_budget_partly_prefills_the_next_sequence(self, make_engine):
        engine = make_engine(max_num_batched_tokens=20)
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(40, 100), FOREVER)
        engine.step()
        assert [n for _, n in engine.model_runner.batches[0][1]] == [8, 12]  # budget fully spent


class TestPrefixCache:
    def test_repeated_prefix_is_not_recomputed(self, make_engine):
        engine = make_engine()
        first = engine.add(prompt(16), FOREVER)
        engine.step()
        second = engine.add(prompt(16), FOREVER)
        engine.step()
        # One full block hits; the trailing block is never a candidate, so 8 of 16 recompute.
        assert dict(engine.model_runner.batches[1][1])[second.request_id] == 8
        assert second.block_table[0] == first.block_table[0]
        assert engine.scheduler.block_manager.blocks[first.block_table[0]].ref_count == 2

    def test_disabling_it_recomputes_the_whole_prompt(self, make_engine):
        engine = make_engine(enable_prefix_caching=False)
        engine.add(prompt(16), FOREVER)
        engine.step()
        second = engine.add(prompt(16), FOREVER)
        engine.step()
        assert dict(engine.model_runner.batches[1][1])[second.request_id] == 16
        assert not engine.scheduler.block_manager.hash_to_block_id
        assert engine.metrics.prefix_cache_queries.total and not engine.metrics.prefix_cache_hits.total

    def test_a_block_is_hashed_on_arrival_and_never_again(self, make_engine, monkeypatch):
        """The queue head is re-queried every step, and rehashing it is the whole cost."""
        calls = []
        monkeypatch.setitem(sequence.HASH_ALGOS, "sha256", lambda *a: calls.append(1) or 1)
        engine = make_engine(num_kvcache_blocks=3)
        engine.add(prompt(16), FOREVER)
        waiting = engine.add(prompt(24, 100), FOREVER)  # never fits, so it is looked up each step
        assert len(waiting.block_hashes) == 3  # every full block, before it was ever scheduled
        hashed = len(calls)
        for _ in range(4):
            engine.step()
        assert waiting in engine.scheduler.waiting
        assert len(calls) == hashed

    def test_hashes_outlive_the_blocks_they_name(self, make_engine):
        """A preempted sequence is re-admitted against its own hashes, not a rehash."""
        engine = make_engine(num_kvcache_blocks=2)
        engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), FOREVER)  # newest, so it is the victim
        engine.step()
        hashes = list(second.block_hashes)
        engine.step()
        assert second.num_preemptions == 1
        assert second.block_table == []
        assert second.block_hashes == hashes and hashes

    def test_the_hash_chains_so_the_same_block_behind_a_different_head_misses(self, make_engine):
        engine = make_engine()
        first = engine.add(prompt(24), FOREVER)
        shifted = engine.add(prompt(16, 100) + prompt(8, 8), FOREVER)
        assert first.block_hashes[0] != shifted.block_hashes[0]
        assert first.block_hashes[1] != shifted.block_hashes[1]

    def test_xxhash_caches_the_same_prefix_as_sha256(self, make_engine):
        engine = make_engine(prefix_caching_hash_algo="xxhash")
        engine.add(prompt(16), FOREVER)
        engine.step()
        second = engine.add(prompt(16), FOREVER)
        engine.step()
        assert dict(engine.model_runner.batches[1][1])[second.request_id] == 8

    @pytest.mark.parametrize("algo", sorted(sequence.HASH_ALGOS))
    def test_every_hash_holds_across_processes(self, algo):
        """Hashes survive a restart, and TP workers agree with the parent."""
        code = f"from lean_vllm.engine.sequence import {algo}_hash as h; print(h((1, 2, 3), -1))"
        seen = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            ).stdout
            for seed in ("1", "2")
        }
        assert len(seen) == 1 and seen != {""}


class TestPreemption:
    def test_victim_is_preempted_and_requeued(self, make_engine):
        engine = make_engine(num_kvcache_blocks=2, kvcache_block_size=8)
        first = engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        engine.step()
        assert second.block_table == [] and second.num_cached_tokens == 0  # recompute, not swap
        assert second in engine.scheduler.waiting
        assert engine.model_runner.batches[1][1] == [(first.request_id, 1)]

    def test_preempted_sequence_still_completes(self, make_engine):
        engine = make_engine(num_kvcache_blocks=3, kvcache_block_size=8)
        for s in range(2):
            engine.add(prompt(8, s * 100), SamplingParams(max_tokens=12, ignore_eos=True))
        outputs = engine.run_to_completion()
        assert [len(o) for o in outputs.values()] == [12, 12]

    def test_a_sequence_the_cache_cannot_hold_is_dropped(self, make_engine):
        """Alone in the cache and still short of a block: dropped, not preempted forever."""
        engine = make_engine(num_kvcache_blocks=1, kvcache_block_size=8)
        seq = engine.add(prompt(8), FOREVER)
        engine.step()
        engine.step()
        assert seq.finish_reason == "capacity"
        assert engine.is_finished()
        assert not engine.scheduler.block_manager.used_block_ids


class TestPolicy:
    def test_fcfs_admits_in_arrival_order(self, make_engine):
        engine = make_engine(max_num_batched_tokens=8)
        first = engine.add(prompt(8), FOREVER)
        second = engine.add(prompt(8, 100), SamplingParams(max_tokens=64, ignore_eos=True, priority=-5))
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(first.request_id, 8)])
        assert second in engine.scheduler.waiting

    def test_priority_admits_the_urgent_request_first(self, make_engine):
        engine = make_engine(max_num_batched_tokens=8, scheduling_policy="priority")
        engine.add(prompt(8), FOREVER)
        urgent = engine.add(prompt(8, 100), SamplingParams(max_tokens=64, ignore_eos=True, priority=-5))
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(urgent.request_id, 8)])

    def test_priority_ties_break_on_arrival(self, make_engine):
        engine = make_engine(max_num_batched_tokens=8, scheduling_policy="priority")
        first = engine.add(prompt(8), FOREVER)
        engine.add(prompt(8, 100), FOREVER)
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(first.request_id, 8)])

    def test_fcfs_preempts_the_newest_sequence(self, make_engine):
        engine = make_engine(num_kvcache_blocks=2, kvcache_block_size=8)
        engine.add(prompt(8), FOREVER)
        newest = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        engine.step()
        assert newest.num_preemptions == 1

    def test_priority_preempts_the_least_urgent_sequence(self, make_engine):
        engine = make_engine(num_kvcache_blocks=2, kvcache_block_size=8, scheduling_policy="priority")
        expendable = engine.add(prompt(8), SamplingParams(max_tokens=64, ignore_eos=True, priority=5))
        engine.add(prompt(8, 100), FOREVER)
        engine.step()
        engine.step()
        assert expendable.num_preemptions == 1  # oldest, but least urgent

    def test_priority_spares_the_urgent_sequence_admitted_after_the_expendable_one(self, make_engine):
        engine = make_engine(num_kvcache_blocks=3, kvcache_block_size=8, scheduling_policy="priority")
        expendable = engine.add(prompt(8), SamplingParams(max_tokens=64, ignore_eos=True, priority=5))
        engine.step()
        urgent = engine.add(prompt(8, 100), FOREVER)  # admitted a step later, behind it in running
        for _ in range(20):
            engine.step()
            if engine.last_output.preempted:
                break
        assert engine.last_output.preempted == [expendable]
        assert urgent.num_preemptions == 0

    def test_an_unknown_policy_is_rejected(self, make_engine):
        with pytest.raises(ValueError, match="unknown scheduling policy"):
            make_engine(scheduling_policy="lifo")


class TestAdmission:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_oversized_prompt_is_dropped_without_blocking_smaller_requests(self, make_engine, chunked):
        engine = make_engine(num_kvcache_blocks=1, enable_chunked_prefill=chunked)
        oversized = engine.add(prompt(9), FOREVER)
        small = engine.add(prompt(8, 100), SamplingParams(max_tokens=1))

        dropped = {output.request_id: output for output in engine.step()}
        assert oversized.request_id in dropped
        assert dropped[oversized.request_id].finish_reason == "capacity"
        assert dropped[oversized.request_id].token_ids == []

        finished = {output.request_id: output for output in engine.step()}  # drains small's only step
        assert finished[small.request_id].finish_reason == "length"
        assert len(finished[small.request_id].token_ids) == 1
        assert engine.is_finished()
        assert not engine.scheduler.seqs
        assert not engine.scheduler.block_manager.used_block_ids
        assert engine.metrics.requests_finished.values == {"length": 1, "capacity": 1}

    def test_a_duplicate_request_id_is_refused(self, make_engine):
        """Both would queue behind one entry in seqs, and the second to finish would KeyError."""
        engine = make_engine()
        engine.add(prompt(8), FOREVER, request_id="dup")
        with pytest.raises(DuplicateRequestId):
            engine.add(prompt(8, 100), FOREVER, request_id="dup")
        assert len(engine.scheduler.waiting) == 1
        engine.run_to_completion()

    def test_an_id_is_reusable_once_its_request_finished(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=1, ignore_eos=True), request_id="reused")
        engine.run_to_completion()
        engine.add(prompt(8, 100), FOREVER, request_id="reused")


class TestLongPrompts:
    def test_one_prompt_cannot_take_the_whole_budget(self, make_engine):
        engine = make_engine(max_num_batched_tokens=64, long_prefill_token_threshold=8)
        first = engine.add(prompt(40), FOREVER)
        second = engine.add(prompt(40, 100), FOREVER)
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(first.request_id, 8), (second.request_id, 8)])


class TestAdvanceAndReconcile:
    def test_together_they_do_what_one_pass_did(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), FOREVER)
        scheduled = engine.scheduler.schedule().scheduled
        rows = engine.scheduler.advance(scheduled)
        assert [row.seq for row in rows] == [seq]
        assert seq.num_pending_tokens == 1
        assert seq.num_cached_tokens == 8
        stepped = engine.scheduler.reconcile(rows, [1234])
        assert stepped == [seq]
        assert seq.last_token == 1234
        assert seq.num_pending_tokens == 0

    def test_a_partial_prefill_reserves_nothing(self, make_engine):
        engine = make_engine(max_num_batched_tokens=4)
        seq = engine.add(prompt(8), FOREVER)
        rows = engine.scheduler.advance(engine.scheduler.schedule().scheduled)
        assert rows == []
        assert seq.num_pending_tokens == 0
        assert seq.is_prefill

    def test_a_chunk_publishes_its_blocks_before_the_prefill_finishes(self, make_engine):
        """Otherwise a concurrent request sharing the prefix gets no cache hit until the end."""
        engine = make_engine(max_num_batched_tokens=8)
        seq = engine.add(prompt(24), FOREVER)
        rows = engine.scheduler.advance(engine.scheduler.schedule().scheduled)
        assert rows == []  # first chunk only, nothing samples yet
        assert seq.num_published_blocks == 1
        assert engine.scheduler.block_manager.hash_to_block_id

    def test_a_row_preempted_since_the_launch_discards_its_token(self, make_engine):
        """Its token is recomputed after requeueing, so accepting it would double-count."""
        engine = make_engine()
        seq = engine.add(prompt(8), FOREVER)
        rows = engine.scheduler.advance(engine.scheduler.schedule().scheduled)
        engine.scheduler.running.remove(seq)  # as _make_room does before preempting
        engine.scheduler._preempt(seq, SchedulerOutput())
        assert seq.num_pending_tokens == 0
        assert engine.scheduler.reconcile(rows, [1234]) == []
        assert seq.last_token == 7  # prompt(8) is range(8), so the token never landed

    def test_a_row_aborted_since_the_launch_discards_its_token(self, make_engine):
        engine = make_engine()
        seq = engine.add(prompt(8), FOREVER)
        rows = engine.scheduler.advance(engine.scheduler.schedule().scheduled)
        assert engine.scheduler.abort(seq.request_id)
        assert engine.scheduler.reconcile(rows, [1234]) == []
        assert seq.finish_reason == "abort"

    def test_a_finished_row_gives_up_its_reservation(self, make_engine):
        """Otherwise the reserved token would be counted as a completion token."""
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=1))
        rows = engine.scheduler.advance(engine.scheduler.schedule().scheduled)
        engine.scheduler.reconcile(rows, [1234])
        assert seq.finish_reason == "length"
        assert seq.num_completion_tokens == 1
        assert seq.num_pending_tokens == 0


class TestBatchCounts:
    """advance() clears num_scheduled_tokens, so the counts must be taken at schedule time."""

    def test_counts_survive_advance(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), FOREVER)
        output = engine.scheduler.schedule()
        assert (output.num_prefill_tokens, output.num_decode_tokens) == (8, 0)
        engine.scheduler.reconcile(engine.scheduler.advance(output.scheduled), [1234])
        assert (output.num_prefill_tokens, output.num_decode_tokens) == (8, 0)

    def test_a_mixed_step_counts_both_kinds(self, make_engine):
        engine = make_engine(max_num_batched_tokens=10)
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(8, 100), FOREVER)
        engine.step()
        output = engine.scheduler.schedule()
        assert (output.num_prefill_tokens, output.num_decode_tokens) == (6, 1)


class TestChunkedPrefillDisabled:
    def test_a_prompt_is_never_split(self, make_engine):
        engine = make_engine(max_num_batched_tokens=16, enable_chunked_prefill=False)
        seq = engine.add(prompt(12), FOREVER)
        other = engine.add(prompt(12, 100), FOREVER)
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(seq.request_id, 12)])
        assert other in engine.scheduler.waiting  # no room for a whole prompt, so it waits

    def test_prefill_does_not_mix_with_decode(self, make_engine):
        engine = make_engine(enable_chunked_prefill=False)
        running = engine.add(prompt(8), FOREVER)
        engine.step()
        arriving = engine.add(prompt(8, 100), FOREVER)
        engine.step()
        assert engine.model_runner.batches[1] == (True, [(arriving.request_id, 8)])
        assert running.num_completion_tokens == 1  # starved this step, as a whole prompt does

    def test_a_prompt_larger_than_the_budget_is_dropped(self, make_engine):
        engine = make_engine(max_num_batched_tokens=16, enable_chunked_prefill=False)
        seq = engine.add(prompt(40), FOREVER)
        engine.step()
        assert seq.finish_reason == "capacity"

    def test_blocks_freed_by_a_drop_mid_step_admit_no_chunk(self, make_engine):
        """A decode dropped for capacity frees its blocks, but the prompt waiting on them still goes whole or not
        at all; this one is over the budget, so not at all."""
        engine = make_engine(num_kvcache_blocks=4, max_num_batched_tokens=16, enable_chunked_prefill=False)
        running = engine.add(prompt(8), FOREVER)  # 72 tokens, over the 32-slot cache
        while len(running.block_table) < 2:
            engine.step()
        waiting = engine.add(prompt(24, 100), FOREVER)  # 3 blocks, so it waits for the running one to go
        engine.run_to_completion()
        assert running.finish_reason == waiting.finish_reason == "capacity"
        assert all(r != waiting.request_id for _, rows in engine.model_runner.batches for r, _ in rows)

    def test_the_long_prefill_cap_does_not_split_a_prompt(self, make_engine):
        engine = make_engine(max_num_batched_tokens=64, enable_chunked_prefill=False, long_prefill_token_threshold=8)
        seq = engine.add(prompt(40), FOREVER)
        engine.step()
        assert engine.model_runner.batches[0] == (True, [(seq.request_id, 40)])


class TestTokenLimitGuard:
    def test_a_row_at_its_limit_is_not_scheduled_again(self, make_engine):
        """Its reserved tokens are already the last ones, so another step is wasted."""
        engine = make_engine()
        seq = engine.add(prompt(8), SamplingParams(max_tokens=2))
        engine.scheduler.advance(engine.scheduler.schedule().scheduled)  # prefill; reserves token 1
        engine.scheduler.advance(engine.scheduler.schedule().scheduled)  # decode; reserves token 2
        assert seq.num_pending_tokens == 2
        assert engine.scheduler.schedule().scheduled == []
        assert seq in engine.scheduler.running

    def test_the_guard_is_inert_without_reservations(self, make_engine):
        """With the flag off a finished row is dropped before the next schedule."""
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=2))
        assert engine.run_to_completion()
        assert not engine.scheduler.running

    def test_the_guard_works_with_chunked_prefill_disabled(self, make_engine):
        """Decode steps reach the guard even without chunked prefill."""
        engine = make_engine(enable_chunked_prefill=False)
        seq = engine.add(prompt(8), SamplingParams(max_tokens=2))
        engine.scheduler.advance(engine.scheduler.schedule().scheduled)  # prefill; reserves token 1
        engine.scheduler.advance(engine.scheduler.schedule().scheduled)  # decode; reserves token 2
        assert seq.num_pending_tokens == 2
        assert engine.scheduler.schedule().scheduled == []
        assert seq in engine.scheduler.running
