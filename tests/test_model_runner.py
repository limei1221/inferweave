"""Batch preparation on CPU, including the sequence state sent to TP workers."""

import pickle

import pytest
import torch

from lean_vllm.engine.model_runner import (
    ModelRunner, PIECEWISE_MAX_PAD, PIECEWISE_MAX_TOKENS, PIECEWISE_MIN_TOKENS)
from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams
from lean_vllm.utils.context import Context, get_context


@pytest.fixture
def runner():
    # Skip model loading and process-group startup; exercise real batch preparation.
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.device = torch.device("cpu")
    runner.block_size = 8
    runner._prev_tokens = runner._prev_rows = None
    return runner


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("decode", [False, True])
def test_batch_preparation_on_tensor_parallel_ranks(runner, rank, decode):
    seq = Sequence([10, 11, 12], SamplingParams(temperature=0.5))
    seq.num_scheduled_tokens = 3
    if decode:
        seq.append_token(13)
        seq.num_cached_tokens = 3
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
    runner.rank = rank
    if rank:
        seq = pickle.loads(pickle.dumps(seq))

    ids, positions, temperatures, context = runner.prepare_batch([seq])

    assert ids.tolist() == ([13] if decode else [10, 11, 12])
    assert positions.tolist() == ([3] if decode else [0, 1, 2])
    assert context["is_prefill"] == (not decode)
    if decode:
        assert context["logits_indices"] is None
    else:
        assert context["logits_indices"].tolist() == [2]
    if rank:
        assert temperatures is None
    else:
        assert temperatures.tolist() == [0.5]


def test_run_returns_tokens_that_are_not_yet_fetched(runner, make_engine):
    """The engine awaits them later, so run() must not block on the device."""
    from lean_vllm.engine.sampled_tokens import SampledTokens
    engine = make_engine()
    engine.add([10, 11, 12], SamplingParams(temperature=0.0))
    scheduled = engine.scheduler.schedule().scheduled
    runner.sampler = lambda logits, temperatures: torch.tensor([42], dtype=torch.int64)
    runner.run_model = lambda ids, positions, is_prefill: torch.zeros(len(ids), 8)
    runner.step_kind = "enforced"

    pending = runner.run(scheduled)

    assert isinstance(pending, SampledTokens)
    assert pending.tolist() == [42]


def test_a_pending_row_takes_its_input_from_the_previous_tokens(runner):
    """Its token was sampled by a step still in flight, so the host value is stale."""
    from lean_vllm.engine.sampled_tokens import SampledTokens
    seq = Sequence([10, 11, 12], SamplingParams(temperature=0.0))
    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = 3, 1, False
    seq.reserve_token()
    runner._prev_tokens = SampledTokens(torch.tensor([77], dtype=torch.int64), torch.device("cpu"))
    runner._prev_rows = {seq.seq_id: 0}

    ids, positions, temperatures, context = runner.prepare_batch([seq])

    assert ids.tolist() == [77]
    assert positions.tolist() == [3]


def test_the_fast_path_slices_a_previous_tensor_with_extra_rows(runner):
    """A request can finish between steps, so the previous tensor may outgrow this one."""
    from lean_vllm.engine.sampled_tokens import SampledTokens
    seq1 = Sequence([10, 11, 12], SamplingParams(temperature=0.0))
    seq1.num_cached_tokens, seq1.num_scheduled_tokens, seq1.is_prefill = 3, 1, False
    seq1.reserve_token()
    seq2 = Sequence([20, 21, 22], SamplingParams(temperature=0.0))
    seq2.num_cached_tokens, seq2.num_scheduled_tokens, seq2.is_prefill = 3, 1, False
    seq2.reserve_token()
    runner._prev_tokens = SampledTokens(torch.tensor([77, 88, 99], dtype=torch.int64), torch.device("cpu"))
    runner._prev_rows = {seq1.seq_id: 0, seq2.seq_id: 1}

    ids, positions, temperatures, context = runner.prepare_batch([seq1, seq2])

    assert ids.tolist() == [77, 88]


def test_a_reordered_mapping_takes_the_general_path(runner):
    """Rows need not line up with the previous tensor, so each looks up its own index."""
    from lean_vllm.engine.sampled_tokens import SampledTokens
    seq1 = Sequence([10, 11, 12], SamplingParams(temperature=0.0))
    seq1.num_cached_tokens, seq1.num_scheduled_tokens, seq1.is_prefill = 3, 1, False
    seq1.reserve_token()
    seq2 = Sequence([20, 21, 22], SamplingParams(temperature=0.0))
    seq2.num_cached_tokens, seq2.num_scheduled_tokens, seq2.is_prefill = 3, 1, False
    seq2.reserve_token()
    runner._prev_tokens = SampledTokens(torch.tensor([77, 88], dtype=torch.int64), torch.device("cpu"))
    runner._prev_rows = {seq1.seq_id: 1, seq2.seq_id: 0}

    ids, positions, temperatures, context = runner.prepare_batch([seq1, seq2])

    assert ids.tolist() == [88, 77]


def test_a_prefill_row_with_a_pending_token_is_refused(runner):
    """It would slice token_ids for a value that has not been sampled."""
    seq = Sequence([10, 11, 12], SamplingParams())
    seq.num_scheduled_tokens = 3
    seq.reserve_token()
    with pytest.raises(AssertionError, match="pending"):
        runner.prepare_batch([seq])


def test_a_pending_row_with_no_previous_row_is_refused(runner):
    """Every reserved token must be covered, or the batch silently uses a stale id."""
    seq = Sequence([10, 11, 12], SamplingParams())
    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = 3, 1, False
    seq.reserve_token()
    runner._prev_tokens = runner._prev_rows = None
    with pytest.raises(AssertionError, match="pending"):
        runner.prepare_batch([seq])


def test_an_all_greedy_batch_sends_no_temperatures(runner):
    """None is the sampler's fast path, and one transfer the step does not make."""
    seq = Sequence([10, 11, 12], SamplingParams(temperature=0.0))
    seq.append_token(13)
    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = 3, 1, False

    _, _, temperatures, _ = runner.prepare_batch([seq])

    assert temperatures is None


def test_keys_are_new_follows_the_cached_tokens(runner):
    """It decides which entry point a flash prefill takes, so it must not lag the batch."""
    cold = Sequence([10, 11, 12], SamplingParams())
    cold.num_scheduled_tokens = 3
    _, _, _, context = runner.prepare_batch([cold])
    assert context["keys_are_new"]

    resumed = Sequence([20, 21, 22, 23], SamplingParams())
    resumed.num_cached_tokens, resumed.num_scheduled_tokens = 2, 2
    _, _, _, context = runner.prepare_batch([cold, resumed])
    assert not context["keys_are_new"]


def test_a_failed_step_leaves_no_context_behind(runner):
    """The next step, or a graph capture, must not read this one's layout."""
    seq = Sequence([10, 11, 12], SamplingParams())
    seq.num_scheduled_tokens = 3

    def explode(input_ids, positions, is_prefill):
        assert get_context().is_prefill
        raise RuntimeError("boom")

    runner.run_model = explode
    with pytest.raises(RuntimeError, match="boom"):
        runner.run([seq])
    assert get_context() == Context()


class TestPiecewiseBuckets:

    def buckets(self, budget):
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = type("C", (), {"max_num_batched_tokens": budget})()
        return runner._piecewise_buckets()

    def test_the_grid_stops_at_the_cap(self):
        """Steps above it run eager: padding them costs more than the dispatch saves."""
        assert self.buckets(8192)[-1] == PIECEWISE_MAX_TOKENS
        assert self.buckets(16384)[-1] == PIECEWISE_MAX_TOKENS

    def test_a_budget_under_the_cap_is_the_top_bucket(self):
        assert self.buckets(300)[-1] == 300    # not a listed size, still covered

    def test_buckets_are_sorted_and_unique(self):
        sizes = self.buckets(5000)
        assert sizes == sorted(set(sizes))

    def test_a_budget_under_the_smallest_size_is_the_only_bucket(self):
        assert self.buckets(32) == [32]

    @pytest.mark.parametrize("budget", [256, 512, 8192, 16384])
    def test_no_step_pads_past_the_cap(self, budget):
        """The gap above a bucket is what a step one token past it pads through."""
        sizes = self.buckets(budget)
        assert sizes[0] == PIECEWISE_MIN_TOKENS
        for smaller, larger in zip(sizes, sizes[1:]):
            assert larger <= (smaller + 1) * (1 + PIECEWISE_MAX_PAD), f"{smaller} -> {larger}"


class TestStepKind:
    """Which capture, if any, a step is eligible for. Pure: shape and config."""

    @pytest.fixture
    def runner(self, runner):
        runner.cudagraph_mode = "full_and_piecewise"
        runner.graph_bs = [1, 2, 4, 8, 16]
        runner.piecewise_bs = [256, 512, 1024]
        return runner

    def test_a_small_decode_batch_replays_a_full_graph(self, runner):
        assert runner._step_kind(is_prefill=False, num_tokens=8) == "graph"

    def test_a_decode_batch_past_the_buckets_is_oversized(self, runner):
        assert runner._step_kind(is_prefill=False, num_tokens=17) == "oversized"

    def test_a_prefill_step_goes_piecewise(self, runner):
        assert runner._step_kind(is_prefill=True, num_tokens=512) == "piecewise"

    def test_a_prefill_step_past_the_buckets_stays_eager(self, runner):
        assert runner._step_kind(is_prefill=True, num_tokens=2048) == "prefill"

    def test_a_prefill_step_under_the_smallest_bucket_stays_eager(self, runner):
        """Padding 8 tokens up to 256 would cost more than the dispatch it saves."""
        assert runner._step_kind(is_prefill=True, num_tokens=8) == "prefill"

    def test_a_mode_without_piecewise_leaves_prefill_eager(self, runner):
        runner.cudagraph_mode = "full"
        assert runner._step_kind(is_prefill=True, num_tokens=512) == "prefill"

    def test_a_mode_without_full_sends_decode_piecewise(self, runner):
        runner.cudagraph_mode = "piecewise"
        assert runner._step_kind(is_prefill=False, num_tokens=512) == "piecewise"

    def test_none_is_enforced_eager(self, runner):
        runner.cudagraph_mode = "none"
        assert runner._step_kind(is_prefill=False, num_tokens=8) == "enforced"

    def test_a_mode_naming_a_capture_that_never_happened_falls_back(self, runner):
        """The mode is what was asked for; the bucket lists are what exists."""
        runner.piecewise_bs = []
        assert runner._step_kind(is_prefill=True, num_tokens=512) == "prefill"
        runner.graph_bs = []
        assert runner._step_kind(is_prefill=False, num_tokens=8) == "oversized"


class TestCudagraphMode:
    """What an MLA model's backend leaves capturable. Pure: config and backend."""

    def mode(self, mode, decodes_latents, mla=True, full_safe=True):
        backend = type("FakeBackend", (), {
            "supports_mla_decode": staticmethod(lambda: decodes_latents),
            "supports_full_cudagraph_mla_decode": staticmethod(lambda: full_safe),
        })
        return ModelRunner._cudagraph_mode(mode, mla, backend)

    def test_a_backend_without_mla_decode_loses_its_full_graphs(self):
        """They would capture attention expanding latents, which needs the step's host plan."""
        assert self.mode("full_and_piecewise", decodes_latents=False) == "piecewise"
        assert self.mode("full", decodes_latents=False) == "none"

    def test_a_decode_that_bakes_a_schedule_loses_its_full_graphs(self):
        """A backend that attends latents but bakes per-step state a replay cannot refresh."""
        assert self.mode("full_and_piecewise", decodes_latents=True, full_safe=False) == "piecewise"
        assert self.mode("full", decodes_latents=True, full_safe=False) == "none"

    def test_a_backend_with_replay_safe_mla_decode_keeps_the_mode(self):
        assert self.mode("full_and_piecewise", decodes_latents=True) == "full_and_piecewise"

    def test_a_model_without_mla_is_never_downgraded(self):
        assert self.mode("full", decodes_latents=False, mla=False) == "full"


@pytest.mark.parametrize("chunked", [False, True])
def test_preemption_recomputes_the_generated_suffix(runner, make_engine, chunked):
    engine = make_engine(num_kvcache_blocks=6, enable_chunked_prefill=chunked)
    first = engine.add(list(range(13)), SamplingParams(max_tokens=16, ignore_eos=True))
    second = engine.add(list(range(100, 108)), SamplingParams(max_tokens=16, ignore_eos=True))
    # The first request needs another block while the second is mid-block.
    for _ in range(16):
        engine.step()
    # Drain the in-flight step without launching another, so schedule() below sees fresh state.
    _, rows, pending = engine.in_flight
    engine.scheduler.reconcile(rows, pending.tolist())
    engine.in_flight = None
    assert first.is_finished
    assert second.num_preemptions == 1
    assert second.num_tokens == 20

    scheduled = engine.scheduler.schedule().scheduled
    ids, positions, temperatures, context = runner.prepare_batch(scheduled)

    assert context["is_prefill"]
    assert ids.tolist() == [1108, 1109, 1110, 1111]
    assert positions.tolist() == [16, 17, 18, 19]
    assert context["logits_indices"].tolist() == [3]
    assert temperatures.tolist() == [1.0]
    rows = engine.scheduler.advance(scheduled)
    stepped = engine.scheduler.reconcile(rows, [1112])
    assert stepped == [second]
    assert second.num_completion_tokens == 13
    assert second.num_cached_tokens == 20
    engine.run_to_completion()
    assert second.finish_reason == "length"
    assert not engine.scheduler.block_manager.used_block_ids


def test_recomputed_suffix_stays_prefill_across_chunks(runner, make_engine):
    engine = make_engine(num_kvcache_blocks=2, max_num_batched_tokens=2)
    seq = engine.add([10, 11], SamplingParams(max_tokens=8, ignore_eos=True))
    # A requeued sequence retains generated tokens, but may have no cached prefix.
    for token in [20, 21, 22, 23, 24, 25, 26]:
        seq.append_token(token)
    seq.num_preemptions = 1

    for expected_ids, expected_positions in [
        ([10, 11], [0, 1]),
        ([20, 21], [2, 3]),
        ([22, 23], [4, 5]),
        ([24, 25], [6, 7]),
        ([26], [8]),
    ]:
        scheduled = engine.scheduler.schedule().scheduled
        assert scheduled == [seq]
        ids, positions, temperatures, context = runner.prepare_batch(scheduled)
        assert context["is_prefill"]
        assert ids.tolist() == expected_ids
        assert positions.tolist() == expected_positions
        assert len(seq.block_table) == 2    # replay uses the blocks reserved at admission
        last = expected_positions == [8]
        # None until the chunk that samples: no row asks for a temperature before it.
        assert (temperatures.tolist() if last else temperatures) == ([1.0] if last else None)
        assert context["logits_indices"].tolist() == ([0] if last else [])
        rows = engine.scheduler.advance(scheduled)
        stepped = engine.scheduler.reconcile(rows, [27] if last else [])
        assert stepped == ([seq] if last else [])

    assert seq.completion_token_ids == [20, 21, 22, 23, 24, 25, 26, 27]
    assert seq.finish_reason == "length"
    assert engine.is_finished()
    assert not engine.scheduler.block_manager.used_block_ids
