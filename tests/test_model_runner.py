"""Batch preparation on CPU, including the sequence state sent to TP workers."""

import pickle
import random

import pytest
import torch

from lean_vllm.engine.model_runner import (
    ModelRunner, PIECEWISE_MAX_PAD, PIECEWISE_MAX_TOKENS, PIECEWISE_MIN_TOKENS)
from lean_vllm.attention import TorchAttention
from lean_vllm.engine.sequence import Sequence
from lean_vllm.layers.attention import Attention, MLAAttention
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


def test_one_query_rows_lead_the_batch_but_sample_in_the_schedulers_order(runner):
    """A splitting backend slices decode rows off the front; the scheduler reads tokens back by its own order."""
    prompt = Sequence([10, 11, 12], SamplingParams(temperature=0.5))
    prompt.num_scheduled_tokens = 3
    decoding = Sequence([20, 21], SamplingParams(temperature=0.25))
    decoding.append_token(22)
    decoding.num_cached_tokens, decoding.num_scheduled_tokens, decoding.is_prefill = 2, 1, False
    decoding.block_table = prompt.block_table = [0]    # any table, so the batch carries one

    ids, positions, temperatures, context = runner.prepare_batch([prompt, decoding])

    assert ids.tolist() == [22, 10, 11, 12]
    assert positions.tolist() == [2, 0, 1, 2]
    assert context["cu_seqlens_q_host"] == [0, 1, 4]
    assert context["logits_indices"].tolist() == [3, 0]
    assert temperatures.tolist() == [0.5, 0.25]
    assert runner._sampling_rows == [prompt, decoding]


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

    def test_a_decode_batch_past_the_buckets_runs_eager(self, runner):
        assert runner._step_kind(is_prefill=False, num_tokens=17) == "decode"

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

    def test_a_small_decode_under_piecewise_alone_runs_eager_as_decode(self, runner):
        """Not oversized: it is under the smallest piecewise bucket, and no full graph exists."""
        runner.cudagraph_mode = "piecewise"
        assert runner._step_kind(is_prefill=False, num_tokens=8) == "decode"

    def test_none_is_enforced_eager(self, runner):
        runner.cudagraph_mode = "none"
        assert runner._step_kind(is_prefill=False, num_tokens=8) == "enforced"

    def test_a_mode_naming_a_capture_that_never_happened_falls_back(self, runner):
        """The mode is what was asked for; the bucket lists are what exists."""
        runner.piecewise_bs = []
        assert runner._step_kind(is_prefill=True, num_tokens=512) == "prefill"
        runner.graph_bs = []
        assert runner._step_kind(is_prefill=False, num_tokens=8) == "decode"


class TestCudagraphMode:
    """What the layers' backends leave capturable. Pure: config and layers."""

    def mode(self, mode, *full_safe):
        layers = [type("FakeLayer", (), {"supports_full_cudagraph": lambda self, safe=safe: safe})() for safe in full_safe]
        return ModelRunner._cudagraph_mode(mode, layers)

    def test_one_layer_that_cannot_be_captured_costs_the_full_graphs(self):
        """A full graph holds every layer's attention, so one host-planned decode rules it out."""
        assert self.mode("full_and_piecewise", True, False) == "piecewise"
        assert self.mode("full", True, False) == "none"

    def test_capturable_layers_keep_the_mode(self):
        assert self.mode("full_and_piecewise", True, True) == "full_and_piecewise"
        assert self.mode("piecewise", False) == "piecewise"


class TestFullCudagraphSupport:
    """Which layers a full graph may hold, by kind and backend."""

    @staticmethod
    def backend(decodes_latents=True, full_safe=True, full=True):
        return type("FakeBackend", (), {
            "supports_mla_decode": staticmethod(lambda: decodes_latents),
            "supports_full_cudagraph_mla_decode": staticmethod(lambda: full_safe),
            "supports_full_cudagraph": staticmethod(lambda: full),
        })

    @staticmethod
    def mla_layer(backend):
        layer = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(layer)
        layer.backend = backend
        return layer

    def test_an_mla_layer_without_latent_decode_cannot_be_captured(self):
        """It would capture attention expanding latents, which needs the step's host plan."""
        assert not self.mla_layer(self.backend(decodes_latents=False)).supports_full_cudagraph()

    def test_an_mla_decode_that_bakes_a_schedule_cannot_be_captured(self):
        assert not self.mla_layer(self.backend(full_safe=False)).supports_full_cudagraph()

    def test_a_replay_safe_mla_decode_can_be_captured(self):
        assert self.mla_layer(self.backend()).supports_full_cudagraph()

    def test_a_plain_layer_asks_its_backend(self):
        for full in (True, False):
            layer = Attention(2, 8, 0.5, 1, backend=TorchAttention)
            layer.backend = self.backend(full=full)
            assert layer.supports_full_cudagraph() is full


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


class TestDummySamplerRun:
    """Warmup must reach the peak of sampling one row per sequence, as vLLM's profile run does."""

    @pytest.fixture
    def sampling_runner(self, runner):
        from types import SimpleNamespace
        runner.config = SimpleNamespace(
            max_num_seqs=6, max_num_batched_tokens=1024, hf_config=SimpleNamespace(hidden_size=8))
        runner.model = SimpleNamespace(compute_logits=lambda hidden: torch.zeros(hidden.size(0), 32))
        runner.sampled = []
        runner.sampler = lambda logits, temperatures: runner.sampled.append((logits.shape, temperatures))
        return runner

    def test_it_samples_one_row_per_sequence_off_the_greedy_path(self, sampling_runner):
        sampling_runner._dummy_sampler_run()
        (shape, temperatures), = sampling_runner.sampled
        assert shape == (6, 32)
        assert (temperatures > 0).all()

    def test_a_token_budget_under_max_num_seqs_caps_the_rows(self, sampling_runner):
        sampling_runner.config.max_num_batched_tokens = 4
        sampling_runner._dummy_sampler_run()
        assert sampling_runner.sampled[0][0] == (4, 32)

    def test_only_rank_zero_samples(self, sampling_runner):
        sampling_runner.rank = 1
        sampling_runner._dummy_sampler_run()
        assert sampling_runner.sampled == []

    def test_running_out_of_memory_names_the_knobs(self, sampling_runner):
        def oom(logits, temperatures):
            raise torch.OutOfMemoryError("CUDA out of memory")
        sampling_runner.sampler = oom
        with pytest.raises(RuntimeError, match="lower max_num_seqs or gpu_memory_utilization"):
            sampling_runner._dummy_sampler_run()


def reference_batch(seqs: list[Sequence], block_size: int, rank: int) -> dict:
    """The per-token loops prepare_batch replaced, kept as its oracle; pending tokens aside."""
    out = dict(input_ids=[], positions=[], slot_mapping=[], cu_seqlens_q=[0], cu_seqlens_k=[0], context_lens=[])
    last_tokens = {}
    for seq in ModelRunner.decodes_first(seqs):
        start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
        out["input_ids"] += seq[start:end] if seq.is_prefill else [seq.last_token]
        out["positions"] += range(start, end)
        out["cu_seqlens_q"].append(out["cu_seqlens_q"][-1] + seq.num_scheduled_tokens)
        out["cu_seqlens_k"].append(out["cu_seqlens_k"][-1] + end)
        out["context_lens"].append(end)
        if end == seq.num_planned_tokens:
            last_tokens[id(seq)] = out["cu_seqlens_q"][-1] - 1
        if seq.block_table:
            out["slot_mapping"] += [seq.block_table[p // block_size] * block_size + p % block_size
                                    for p in range(start, end)]
    sampling = [seq for seq in seqs if id(seq) in last_tokens]
    out["logits_indices"] = [last_tokens[id(seq)] for seq in sampling]
    temperatures = [seq.temperature for seq in sampling] if rank == 0 else []
    out["temperatures"] = None if all(t == 0 for t in temperatures) else temperatures
    tables = [seq.block_table for seq in ModelRunner.decodes_first(seqs)]
    width = max(map(len, tables))
    out["block_tables"] = [table + [-1] * (width - len(table)) for table in tables] if width else None
    return out


def random_batch(rng, block_size: int, with_tables: bool = True) -> list[Sequence]:
    """Decodes, cold prompts, resumed chunks and prefix-cache hits, in the scheduler's shuffled order."""
    next_block = iter(rng.sample(range(10_000), 2_000))    # scattered, so a wrong page lookup shows
    seqs = []
    for _ in range(rng.randint(1, 24)):
        kind = rng.choice(["decode", "prompt", "chunk"])
        prompt = [rng.randrange(1000) for _ in range(rng.randint(1, 90))]
        seq = Sequence(prompt, SamplingParams(temperature=rng.choice([0, 0, 0.7])))
        if kind == "decode":
            seq.append_token(rng.randrange(1000))
            seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = len(prompt), 1, False
        else:
            seq.num_cached_tokens = rng.randrange(len(prompt)) if kind == "chunk" else 0
            seq.num_scheduled_tokens = rng.randint(1, len(prompt) - seq.num_cached_tokens)
        if with_tables:
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            seq.block_table = [next(next_block) for _ in range((end + block_size - 1) // block_size)]
        seqs.append(seq)
    return seqs


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])    # cuda stages through pinned buffers


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [0, 1])
def test_the_numpy_batch_matches_the_per_token_loops(runner, device, rank):
    """Twenty random steps through one runner, so on cuda each overwrites the last one's buffers."""
    runner.rank, runner.device = rank, torch.device(device)
    rng = random.Random(rank)
    for step in range(20):
        seqs = random_batch(rng, runner.block_size, with_tables=step % 5 != 0)    # every fifth is warmup-like
        check_batch(runner, seqs, reference_batch(seqs, runner.block_size, rank))


def check_batch(runner, seqs: list[Sequence], want: dict):
    ids, positions, temperatures, context = runner.prepare_batch(seqs)

    assert ids.tolist() == want["input_ids"]
    assert positions.tolist() == want["positions"]
    for key in ("cu_seqlens_q", "cu_seqlens_k", "context_lens", "slot_mapping"):
        assert context[key].tolist() == want[key], key
    assert context["cu_seqlens_q_host"] == want["cu_seqlens_q"]
    assert context["cu_seqlens_k_host"] == want["cu_seqlens_k"]
    assert context["max_seqlen_q"] == max(b - a for a, b in zip(want["cu_seqlens_q"], want["cu_seqlens_q"][1:]))
    assert context["max_seqlen_k"] == max(want["context_lens"])
    assert context["keys_are_new"] == (want["cu_seqlens_q"] == want["cu_seqlens_k"])
    if context["is_prefill"]:
        assert context["logits_indices"].tolist() == want["logits_indices"]
    else:
        assert context["logits_indices"] is None and want["logits_indices"] == list(range(len(seqs)))
    assert (context["block_tables"] is None if want["block_tables"] is None
            else context["block_tables"].tolist() == want["block_tables"])
    assert (temperatures is None) == (want["temperatures"] is None)
    if temperatures is not None:
        assert temperatures.tolist() == pytest.approx(want["temperatures"])
    assert runner._sampling_rows == [seq for seq in seqs if seq.num_cached_tokens + seq.num_scheduled_tokens
                                     == seq.num_planned_tokens]
