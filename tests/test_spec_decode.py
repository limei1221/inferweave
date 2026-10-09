"""Speculative decoding without a model: the rejection sampler, the scheduler's drafts, and the config."""

import json
from itertools import count

import pytest
import torch
from conftest import EOS, FakeConfig, FakeEngine, FakeModelRunner, FakeSampledTokens

from lean_vllm.config import Config
from lean_vllm.engine.block_manager import BlockManager
from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams
from lean_vllm.spec_decode import RejectionSampler, SpeculativeConfig

WRONG_DRAFT = 3


def logits_for(tokens: list[int], vocab: int = 8) -> torch.Tensor:
    """One logits row per token, each with that token as its argmax."""
    logits = torch.zeros(len(tokens), vocab)
    logits[torch.arange(len(tokens)), torch.tensor(tokens)] = 5.0
    return logits


class TestRejectionSampler:
    def sample(self, argmaxes: list[list[int]], drafts: list[list[int]], temperatures=None, k: int = 2):
        """Each row's target argmax at its draft positions and its bonus position, and its drafts."""
        logits = logits_for([t for row in argmaxes for t in row])
        draft_ids = torch.tensor([t for row in drafts for t in row], dtype=torch.int64)
        counts = torch.tensor([len(row) for row in drafts])
        return RejectionSampler()(logits, draft_ids, counts, temperatures, k).tolist()

    def test_greedy_keeps_drafts_up_to_the_first_miss_then_takes_the_target(self):
        out = self.sample(
            argmaxes=[[4, 6, 1], [2], [5, 0, 3]],
            drafts=[[4, 7], [], [5, 0]],
        )
        assert out == [[4, 6, -1], [2, -1, -1], [5, 0, 3]]

    def test_a_greedy_row_among_sampled_ones_stays_greedy(self):
        temperatures = torch.tensor([0.0, 1.0])
        out = self.sample(argmaxes=[[4, 6, 1], [2, 2, 2]], drafts=[[4, 7], [2, 2]], temperatures=temperatures)
        assert out[0] == [4, 6, -1]
        assert len([t for t in out[1] if t >= 0]) >= 1

    def test_sampled_tokens_follow_the_target_whatever_the_drafts(self):
        """Leviathan et al.: verifying a draft leaves each position's distribution the target's."""
        torch.manual_seed(0)
        num_rows, k = 40000, 2
        p = torch.tensor([[0.1, 0.5, 0.15, 0.25], [0.4, 0.1, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25]])
        logits = p.log().repeat(num_rows, 1)
        drafts = torch.tensor([1, 2]).repeat(num_rows)  # the first a likely token, the second an unlikely one
        out = RejectionSampler()(logits, drafts, torch.full((num_rows,), k), torch.ones(num_rows), k)

        first = torch.bincount(out[:, 0], minlength=4) / num_rows
        torch.testing.assert_close(first, p[0], atol=0.01, rtol=0)
        # Given the first draft kept, the second token follows the second position's distribution.
        kept = out[:, 0] == 1
        second = torch.bincount(out[kept, 1], minlength=4) / kept.sum()
        torch.testing.assert_close(second, p[1], atol=0.015, rtol=0)
        # A row that keeps both drafts samples its bonus token from the last position.
        both = kept & (out[:, 1] == 2)
        assert (out[both, 2] >= 0).all() and (out[~both, 2] == -1).all()


class FakeDraftingRunner(FakeModelRunner):
    """Verifies drafts as greedy rejection does, and drafts the true tokens except where wrong(seq, index) says.

    As the runner does on the device, a row whose last step is in flight verifies that step's drafts, which the
    scheduler holds placeholders for, and moves back past those it rejected.
    """

    def __init__(self, k: int, eos_after=None, wrong=lambda seq, index: False):
        super().__init__(eos_after)
        self.k, self.wrong = k, wrong
        self.lookahead_short = 0  # rows whose blocks could not take the drafter's writes
        self.in_flight: dict[int, tuple[int, list[int]]] = {}  # by seq_id: the last run's rejected count and drafts

    def _truth(self, seq: Sequence, index: int) -> int:
        limit = self.eos_after.get(seq.request_id)
        return EOS if limit is not None and index >= limit else 1000 + seq.seq_id * 100 + index

    def run(self, seqs: list[Sequence]) -> FakeSampledTokens:
        self.batches.append(
            (any(seq.is_prefill for seq in seqs), [(seq.request_id, seq.num_scheduled_tokens) for seq in seqs])
        )
        rows = []
        in_flight, self.in_flight = self.in_flight, {}
        for seq in seqs:
            if not self._samples(seq):
                continue
            last = seq.num_cached_tokens + seq.num_scheduled_tokens - 1
            if len(seq.block_table) * seq.block_size < last + self.k:  # the last draft step's slot
                self.lookahead_short += 1
            spec = [] if seq.is_prefill else seq.spec_token_ids
            back = 0
            if seq.num_pending_tokens:
                back, made = in_flight[seq.seq_id]
                spec = made[: len(spec)]
            # The completion token this row produces first, after its input tokens but its drafts.
            index = seq.num_cached_tokens - back + seq.num_scheduled_tokens - len(spec) - seq.num_prompt_tokens
            tokens = []
            for draft in spec:
                if draft != self._truth(seq, index + len(tokens)):
                    break
                tokens.append(draft)
            tokens.append(self._truth(seq, index + len(tokens)))
            after = index + len(tokens)
            drafts = [WRONG_DRAFT if self.wrong(seq, i) else self._truth(seq, i) for i in range(after, after + self.k)]
            self.in_flight[seq.seq_id] = (len(spec) - (len(tokens) - 1), drafts)
            rows.append(tokens + [-1] * (self.k + 1 - len(tokens)) + drafts)
        return FakeSampledTokens(rows)


def make_drafting_engine(k: int, eos_after=None, wrong=lambda seq, index: False, **overrides) -> FakeEngine:
    config = FakeConfig(speculative=SpeculativeConfig(num_speculative_tokens=k), **overrides)
    Sequence.counter = count()  # the same seq ids as a plain engine's, so the same tokens
    Sequence.block_size = config.kvcache_block_size
    return FakeEngine(config, FakeDraftingRunner(k, eos_after, wrong))


def add_prompts(engine: FakeEngine, max_tokens: int = 20):
    for n in (5, 30, 11, 17):
        engine.add(list(range(n)), SamplingParams(max_tokens=max_tokens, ignore_eos=False))


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
class TestScheduler:
    @pytest.mark.parametrize("k", [1, 3])
    @pytest.mark.parametrize("num_kvcache_blocks", [64, 9], ids=["roomy", "preempting"])
    def test_completions_match_plain_decoding_in_fewer_steps(
        self, make_engine, k, num_kvcache_blocks, async_scheduling
    ):
        plain = make_engine(num_kvcache_blocks=num_kvcache_blocks, max_num_batched_tokens=24)
        add_prompts(plain)
        want = plain.run_to_completion()
        engine = make_drafting_engine(
            k,
            wrong=lambda seq, index: index % 5 == 2,
            num_kvcache_blocks=num_kvcache_blocks,
            max_num_batched_tokens=24,
            async_scheduling=async_scheduling,
        )
        add_prompts(engine)

        assert engine.run_to_completion() == want
        assert len(engine.model_runner.batches) < len(plain.model_runner.batches)
        assert engine.model_runner.lookahead_short == 0
        if num_kvcache_blocks == 9:
            assert engine.metrics.preemptions.total
        assert 0 < engine.metrics.spec_accepted_tokens.total < engine.metrics.spec_draft_tokens.total

    def test_a_stop_among_the_kept_drafts_ends_the_request_there(self, make_engine, async_scheduling):
        engine = make_drafting_engine(3, eos_after={"req-0": 6}, async_scheduling=async_scheduling)
        engine.add([1, 2, 3], SamplingParams(max_tokens=50), request_id="req-0")
        completion = engine.run_to_completion()["req-0"]
        assert completion[-1] == EOS and len(completion) == 7
        assert not engine.scheduler.block_manager.used_block_ids

    def test_drafts_never_run_past_max_tokens_or_the_budget(self, async_scheduling):
        engine = make_drafting_engine(3, max_num_batched_tokens=10, async_scheduling=async_scheduling)
        engine.add(list(range(8)), SamplingParams(max_tokens=6), request_id="a")
        engine.add(list(range(8)), SamplingParams(max_tokens=9), request_id="b")
        completions = engine.run_to_completion()
        assert [len(completions[r]) for r in "ab"] == [6, 9]
        for _, rows in engine.model_runner.batches:
            assert sum(n for _, n in rows) <= 10
        # A's first decode keeps its 3 drafts and one more, leaving it 1 token to go, so its next has no drafts.
        # Async, that next is placed as if the first kept them, which it does.
        a_decodes = [n for _, rows in engine.model_runner.batches for r, n in rows if r == "a"][1:]
        assert a_decodes == [4, 1]

    @pytest.mark.parametrize("enable_chunked_prefill", [True, False], ids=["chunked", "whole"])
    def test_a_prompt_that_fits_only_without_its_lookahead_is_dropped(self, async_scheduling, enable_chunked_prefill):
        """Its lookahead slots can never be free, so admission would otherwise wait on it forever."""
        engine = make_drafting_engine(
            2, num_kvcache_blocks=2, async_scheduling=async_scheduling, enable_chunked_prefill=enable_chunked_prefill
        )
        seq = engine.add(list(range(16)), SamplingParams(max_tokens=1))  # both blocks, then one lookahead slot
        assert engine.run_to_completion() == {seq.request_id: []}
        assert seq.finish_reason == "capacity"

    @pytest.mark.parametrize("wrong", [lambda seq, index: True, lambda seq, index: index % 3 == 1])
    def test_a_rejected_draft_gives_back_its_slot(self, monkeypatch, wrong, async_scheduling):
        """Only the newest token is left uncached after a step, however many drafts it kept; async, past the
        tokens a step in flight placed as if it keeps every draft."""
        engine = make_drafting_engine(2, wrong=wrong, async_scheduling=async_scheduling)
        engine.add(list(range(5)), SamplingParams(max_tokens=10))
        reconcile = engine.scheduler.reconcile
        checked = []

        def checking_reconcile(*args):
            stepped = reconcile(*args)
            checked.extend(
                seq.num_cached_tokens == seq.num_planned_tokens - 1 for seq in stepped if not seq.is_finished
            )
            return stepped

        monkeypatch.setattr(engine.scheduler, "reconcile", checking_reconcile)
        assert len(engine.run_to_completion()["req-0"]) == 10
        assert checked and all(checked)


def test_the_last_cached_block_recomputes_for_the_drafter():
    """Its last token's drafter cache was written beside the token after it, which another prompt may not share."""
    Sequence.block_size = 4
    for recompute_last_hit, want in ((False, 2), (True, 1)):
        manager = BlockManager(16, 4, recompute_last_hit=recompute_last_hit)
        first = Sequence(list(range(10)))
        manager.allocate(first, 0)
        manager.hash_blocks(first, 10)
        assert manager.can_allocate(Sequence(list(range(10)))) == want


class FakeHFConfig:
    max_position_embeddings = 4096
    architectures = ["DeepseekV3ForCausalLM"]
    num_nextn_predict_layers = 1


@pytest.fixture
def make_config(tmp_path, monkeypatch):
    from lean_vllm import config as config_module

    def make(hf_config=FakeHFConfig, **kwargs):
        monkeypatch.setattr(config_module.AutoConfig, "from_pretrained", lambda path: hf_config())
        return Config(str(tmp_path), **kwargs)

    return make


class TestConfig:
    def test_drafting_keeps_async_scheduling_on(self, make_config):
        config = make_config(speculative_config=json.dumps({"method": "mtp", "num_speculative_tokens": 2}))
        assert config.speculative == SpeculativeConfig("mtp", 2)
        assert config.async_scheduling

    def test_a_checkpoint_without_mtp_layers_is_refused(self, make_config):
        class NoMTP(FakeHFConfig):
            num_nextn_predict_layers = 0

        with pytest.raises(ValueError, match="multi-token prediction layers"):
            make_config(NoMTP, speculative_config='{"num_speculative_tokens": 1}')

    @pytest.mark.parametrize(
        "value, message",
        [
            ('{"method": "ngram"}', "not one of"),
            ('{"num_speculative_tokens": 0}', "at least 1"),
            ('{"model": "x"}', "unknown"),
        ],
    )
    def test_a_bad_speculative_config_is_refused(self, make_config, value, message):
        with pytest.raises(ValueError, match=message):
            make_config(speculative_config=value)
