"""DeepSeek-V3's MTP drafter end to end, on a tiny random checkpoint on CPU.

Greedy speculative decoding must give the tokens plain decoding gives, whatever the drafts. A drafter whose weights
make it predict the target exactly must have every draft kept; and the drafts the engine proposes step by step
must be the ones a single pass of the drafter over the finished sequence gives.
"""

import atexit
import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import DeepseekV3Config, PreTrainedTokenizerFast
from transformers import DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM

from lean_vllm.layers.attention import MLAAttention
from lean_vllm.sampling_params import SamplingParams
from lean_vllm.spec_decode.mtp_proposer import MTPProposer
from lean_vllm.utils.context import set_context

VOCAB = 128
NUM_LAYERS = 2
ENGINE_ARGS = dict(
    enforce_eager=True,
    num_kvcache_blocks=64,
    kvcache_block_size=16,
    max_model_len=256,
    max_num_batched_tokens=48,  # so the longer prompts prefill in chunks, beside other rows' decodes
    max_num_seqs=4,
)
PROMPTS = [[5, 9, 2, 77, 31], list(range(3, 70)), [100, 4] * 20, [64, 65]]


def write_checkpoint(path, seed: int = 0):
    """A V3 checkpoint as DeepSeek ships one: its MTP layer past the last, with copies of embed_tokens and lm_head."""
    torch.manual_seed(seed)
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=96,
        moe_intermediate_size=24,
        n_routed_experts=8,
        num_experts_per_tok=3,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        num_hidden_layers=NUM_LAYERS + 1,  # the last becomes the MTP layer's decoder block
        vocab_size=VOCAB,
        max_position_embeddings=256,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=8,
        v_head_dim=10,
        initializer_range=0.2,
        dtype="float32",
    )
    model = HFDeepseekV3ForCausalLM(config)
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.normal_(0, 0.2)
    model.save_pretrained(path)
    mtp = f"model.layers.{NUM_LAYERS}."
    hidden = config.hidden_size
    save_file(
        {
            mtp + "enorm.weight": 1 + 0.1 * torch.randn(hidden),
            mtp + "hnorm.weight": 1 + 0.1 * torch.randn(hidden),
            mtp + "eh_proj.weight": 0.2 * torch.randn(hidden, 2 * hidden),
            mtp + "shared_head.norm.weight": 1 + 0.1 * torch.randn(hidden),
            # Copies of the target's, which the drafter shares instead of loading.
            mtp + "embed_tokens.weight": model.model.embed_tokens.weight.detach().clone(),
            mtp + "shared_head.head.weight": model.lm_head.weight.detach().clone(),
        },
        path / "mtp.safetensors",
    )
    with open(path / "config.json") as f:
        saved = json.load(f)
    saved.update(num_hidden_layers=NUM_LAYERS, num_nextn_predict_layers=1)
    with open(path / "config.json", "w") as f:
        json.dump(saved, f)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(VOCAB)}, unk_token="t0"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", eos_token="t1").save_pretrained(path)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("deepseek_v3_mtp")
    write_checkpoint(path)
    return str(path)


@pytest.fixture
def make_engine(checkpoint, monkeypatch):
    monkeypatch.setenv("LEAN_VLLM_DEVICE", "cpu")
    monkeypatch.setenv("LEAN_VLLM_ATTENTION_BACKEND", "torch")
    from lean_vllm.engine.llm_engine import LLMEngine

    engines = []

    def close():
        """One process group at a time, so the engine before goes first."""
        while engines:
            engine = engines.pop()
            engine.exit()
            atexit.unregister(engine.exit)

    def make(num_speculative_tokens: int = 0, **kwargs):
        close()
        if num_speculative_tokens:
            kwargs["speculative_config"] = json.dumps(
                {"method": "mtp", "num_speculative_tokens": num_speculative_tokens}
            )
        engines.append(LLMEngine(checkpoint, **ENGINE_ARGS, **kwargs))
        return engines[-1]

    yield make
    close()


def generate(engine, prompts=PROMPTS, max_tokens: int = 24, temperature: float = 0.0) -> list[list[int]]:
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True)
    return [output["token_ids"] for output in engine.generate(prompts, params, use_tqdm=False)]


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_greedy_decoding_is_unchanged_by_the_drafts(make_engine, async_scheduling):
    want = generate(make_engine())
    for k in (1, 3):
        assert generate(make_engine(k, async_scheduling=async_scheduling)) == want, f"num_speculative_tokens={k}"


def predict_the_target(engine):
    """Weights under which the target's next token is the current one plus one, and the drafter predicts it exactly.

    No layer adds to the residual stream, so the target sees the current token's embedding alone, and the head scores
    each token by its predecessor's normalized embedding. The drafter sees the next token's embedding alone.
    """
    runner = engine.model_runner
    model = runner.model
    drafter = runner.proposer.drafter if runner.proposer is not None else None
    with torch.no_grad():
        for module in (model, drafter) if drafter is not None else (model,):
            for name, param in module.named_parameters():
                if "o_proj" in name or "down_proj" in name:
                    param.zero_()
        model.model.norm.weight.fill_(1)
        embed = model.model.embed_tokens.weight
        model.lm_head.weight.copy_((embed / embed.pow(2).mean(-1, keepdim=True).sqrt()).roll(1, dims=0))
        for layer in drafter.layers if drafter is not None else ():
            hidden = layer.enorm.weight.numel()
            layer.eh_proj.weight.copy_(torch.cat([torch.eye(hidden), torch.zeros(hidden, hidden)], dim=1))
            layer.enorm.weight.fill_(1)
            layer.shared_head.norm.weight.fill_(1)


@pytest.mark.parametrize("k", [1, 3])
def test_a_drafter_that_predicts_the_target_has_every_draft_kept(make_engine, k):
    engine = make_engine()
    predict_the_target(engine)
    want = generate(engine)
    plain_steps = engine.metrics.summary()["steps"]
    assert all(b == (a + 1) % VOCAB for tokens in want for a, b in zip(tokens, tokens[1:]))
    engine = make_engine(k)
    predict_the_target(engine)
    assert generate(engine) == want
    summary = engine.metrics.summary()
    assert summary["spec_decode"]["acceptance_rate"] == 1.0
    assert summary["steps"] < plain_steps


def no_cache_context(n: int) -> dict:
    """One row of n tokens attending each other alone, with nothing read from or written to the cache."""
    cu = torch.tensor([0, n], dtype=torch.int32)
    return dict(
        is_prefill=True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=n,
        max_seqlen_k=n,
        cu_seqlens_q_host=[0, n],
        cu_seqlens_k_host=[0, n],
        slot_mapping=torch.full((n,), -1, dtype=torch.int32),
    )


@torch.inference_mode()
def reference_drafts(engine, token_ids: list[int], k: int) -> list[int]:
    """The drafts after token_ids by passes over the whole sequence: the target's hidden states, then the drafter
    over every position with the next token beside it, then again with each draft and its hidden state appended."""
    runner = engine.model_runner
    model, drafter = runner.model, runner.proposer.drafter
    n = len(token_ids) - 1
    positions = torch.arange(n + k - 1)
    with set_context(**no_cache_context(n)):
        hidden_states = model(torch.tensor(token_ids[:-1]), positions[:n])
    inputs = list(token_ids[1:])
    drafts = []
    for step in range(k):
        m = len(inputs)
        with set_context(**no_cache_context(m)):
            out = drafter(model.model.embed_tokens(torch.tensor(inputs)), positions[:m], hidden_states, step)
            drafts.append(int(model.compute_logits(out[-1:]).argmax()))
        inputs.append(drafts[-1])
        hidden_states = torch.cat([hidden_states, out[-1:]])
    return drafts


def keep_drafts_at_random(logits, draft_token_ids, num_draft_tokens, temperatures, max_num_drafts):
    """Stands in for the rejection sampler, as vLLM's synthetic method: random weights would keep almost no draft.
    Each row keeps a random number of its drafts, then the target's token after them."""
    generator = torch.Generator().manual_seed(len(draft_token_ids) * 1000 + logits.size(0))
    output = torch.full((num_draft_tokens.numel(), max_num_drafts + 1), -1, dtype=torch.int64)
    first_draft = first_logits = 0
    for row, n in enumerate(num_draft_tokens.tolist()):
        kept = int(torch.randint(0, n + 1, (), generator=generator))
        output[row, :kept] = draft_token_ids[first_draft : first_draft + kept]
        output[row, kept] = logits[first_logits + kept].argmax()
        first_draft, first_logits = first_draft + n, first_logits + n + 1
    return output


class ReplayedPass:
    """A CUDA graph's stand-in on CPU: a replay runs the captured pass again, over the graph's buffers alone."""

    def __init__(self, run):
        self.run = run
        self.replays = 0

    def replay(self):
        self.replays += 1
        self.run()


def capture_draft_passes(engine, monkeypatch, sizes: list[int]) -> dict:
    """The drafter's graphs as the runner captures them on CUDA, at these batch sizes."""
    runner = engine.model_runner
    monkeypatch.setattr(MTPProposer, "_capture", staticmethod(lambda run, pool: ReplayedPass(run)))
    runner.proposer.capture_cudagraphs(sizes, runner._max_num_blocks(), None, runner.attention_backends)
    return runner.proposer.graphs


# Graphs of 4 rows pad every smaller step; graphs of up to 2 leave larger steps eager. A backend with no MLA decode
# expands latents, planned on the host from key lengths: a draft step reads them back, and an async step waits.
@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "graph_sizes, decode_latents",
    [(None, True), ([4], True), ([1, 2], True), (None, False)],
    ids=["eager", "padded_graphs", "graphs_and_eager", "expanded"],
)
def test_the_drafts_are_those_of_one_pass_over_the_sequence(
    make_engine, monkeypatch, graph_sizes, decode_latents, async_scheduling
):
    """Chunked prompts, kept and rejected drafts' slots, and the later draft positions all feed the drafter's cache,
    whether its single-token passes run eager or replay graphs, and whether a step is placed before the last
    one's drafts are verified."""
    k = 2
    engine = make_engine(k, async_scheduling=async_scheduling)
    runner = engine.model_runner
    monkeypatch.setattr(runner, "rejection_sampler", keep_drafts_at_random)
    if not decode_latents:
        for module in [*runner.model.modules(), *runner.proposer.drafter.modules()]:
            if isinstance(module, MLAAttention):
                monkeypatch.setattr(module.backend, "supports_mla_decode", lambda: False)
        monkeypatch.setattr(runner, "decodes_latents", False)
    graphs = capture_draft_passes(engine, monkeypatch, graph_sizes) if graph_sizes else {}
    proposed = []
    reconcile = engine.scheduler.reconcile

    def recording_reconcile(rows, token_ids, draft_token_ids=None):
        """Each row's drafts with the tokens before them; async, a step in flight holds them, not the sequence."""
        stepped = reconcile(rows, token_ids, draft_token_ids)
        for row, drafts in zip(rows, draft_token_ids or []):
            if row.seq in stepped and not row.seq.is_finished:
                proposed.append((list(row.seq.token_ids), list(drafts)))  # the scheduler trims its own
        return stepped

    monkeypatch.setattr(engine.scheduler, "reconcile", recording_reconcile)
    generate(engine, max_tokens=12)

    assert len(proposed) > len(PROMPTS)
    assert engine.metrics.summary()["spec_decode"]["accepted_per_position"].keys() == {"0", "1"}
    for token_ids, drafts in proposed:
        assert drafts == reference_drafts(engine, token_ids, k), f"after {len(token_ids)} tokens"
    assert sorted(graphs) == [(0, size) for size in graph_sizes or []]
    assert all(graph.replays for graph in graphs.values())


def test_sampled_decoding_runs_to_its_length(make_engine):
    engine = make_engine(2)
    params = [SamplingParams(temperature=t, max_tokens=17, ignore_eos=True) for t in (0.0, 0.7, 1.5, 0.7)]
    outputs = engine.generate(PROMPTS, params, use_tqdm=False)
    assert [len(output["token_ids"]) for output in outputs] == [17] * len(PROMPTS)
    assert all(0 <= t < VOCAB for output in outputs for t in output["token_ids"])


def test_tensor_parallel_ranks_draft_together(make_engine):
    want = generate(make_engine())
    assert generate(make_engine(2, tensor_parallel_size=2)) == want
