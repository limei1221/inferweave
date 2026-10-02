"""DeepSeek-V2 (MLA, MoE, YaRN) against transformers on a tiny random checkpoint, fp32 on the torch backend.

Paged steps go through the runner's real batch preparation, so mixed batches are compared too.
"""

import pytest
import torch
import torch.distributed as dist
from transformers import DeepseekV2Config
from transformers import DeepseekV2ForCausalLM as HFDeepseekV2ForCausalLM

from lean_vllm.attention import TorchAttention
from lean_vllm.engine.model_runner import ModelRunner
from lean_vllm.engine.sequence import Sequence
from lean_vllm.layers.attention import MLAAttention, plan_context_chunks, register_layers
from lean_vllm.models import get_model_class
from lean_vllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from lean_vllm.utils.context import reset_context, set_context
from lean_vllm.utils.loader import load_model

BLOCK_SIZE = 4
NUM_BLOCKS = 16

CONFIGS = {
    "lite": {},
    "q_lora": dict(q_lora_rank=24, attention_bias=True),    # biases load into the fused down-projection too
    "grouped_routing": dict(topk_method="group_limited_greedy", n_group=4, topk_group=2),
}


def tiny_config(**overrides) -> DeepseekV2Config:
    """DeepSeek-V2-Lite's shape in miniature: one dense layer, then MoE with a shared expert."""
    return DeepseekV2Config(**{
        **dict(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=4, intermediate_size=96,
            moe_intermediate_size=24, n_routed_experts=8, num_experts_per_tok=3, n_shared_experts=1,
            first_k_dense_replace=1, num_hidden_layers=3, vocab_size=128, max_position_embeddings=256,
            q_lora_rank=None, kv_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=8,
            v_head_dim=10,    # not the qk head size, so the value padding is exercised
            rope_parameters={
                "rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                "mscale": 1.0, "mscale_all_dim": 0.707,    # unequal, so cos and sin are scaled too
                "original_max_position_embeddings": 32, "beta_fast": 32, "beta_slow": 1,
            },
            initializer_range=0.1,
        ),
        **overrides,
    })


@pytest.fixture(scope="module")
def process_group():
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture(scope="module", params=list(CONFIGS))
def models(request, process_group, tmp_path_factory):
    """The transformers model and ours, loaded from the same checkpoint on disk."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp(request.param)
    HFDeepseekV2ForCausalLM(tiny_config(**CONFIGS[request.param])).save_pretrained(path)
    reference = HFDeepseekV2ForCausalLM.from_pretrained(path, attn_implementation="eager").eval()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LEAN_VLLM_ATTENTION_BACKEND", "torch")
        model = get_model_class(reference.config)(reference.config)
    load_model(model, str(path))
    return reference, model


@pytest.fixture
def runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.device = torch.device("cpu")
    runner.block_size = BLOCK_SIZE
    runner._prev_tokens = runner._prev_rows = None
    return runner


def reference_logits(reference, tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return reference(torch.tensor([tokens])).logits[0]


def row(tokens: list[int], num_cached: int, num_new: int, block_table: list[int], decode: bool = False) -> Sequence:
    """A sequence with num_cached tokens already in the cache and the next num_new scheduled."""
    seq = Sequence(tokens[:num_cached + num_new])
    seq.num_cached_tokens, seq.num_scheduled_tokens = num_cached, num_new
    seq.is_prefill = not decode
    seq.block_table = block_table
    return seq


def step(runner, model, seqs: list[Sequence]) -> list[torch.Tensor]:
    """Logits at every scheduled position, split by row."""
    register_layers(model)    # the op finds layers by name, and each config's model reuses the names
    input_ids, positions, _, context = runner.prepare_batch(seqs)
    context["logits_indices"] = None
    with torch.inference_mode(), set_context(**context):
        logits = model.compute_logits(model(input_ids, positions))
    return list(logits.split([seq.num_scheduled_tokens for seq in seqs]))


def test_a_prompt_matches_transformers(models, runner):
    reference, model = models
    tokens = torch.randint(0, 128, (13,)).tolist()

    (logits,) = step(runner, model, [row(tokens, 0, len(tokens), block_table=[])])

    torch.testing.assert_close(logits, reference_logits(reference, tokens), rtol=1e-4, atol=1e-4)


# 4 splits a's cached keys across chunks and shares one chunk between a's tail and b's head.
@pytest.mark.parametrize("max_context_chunk", [64, 4], ids=["one_chunk", "split_rows"])
# Pure decode attends the latents through mla_decode, or expands them as other steps do.
@pytest.mark.parametrize("latent_decode", [True, False], ids=["latent_decode", "expanded_decode"])
def test_paged_steps_match_transformers(models, runner, max_context_chunk, latent_decode, monkeypatch):
    """Chunks, a cold prompt beside a resumed one, pure decode, then decode mixed with a prompt."""
    monkeypatch.setattr(TorchAttention, "supports_mla_decode", staticmethod(lambda: latent_decode))
    reference, model = models
    layers = [module for module in model.modules() if isinstance(module, MLAAttention)]
    cache = torch.zeros(len(layers), *layers[0].kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE))
    for layer, layer_cache in zip(layers, cache):
        layer.bind_kv_cache(layer_cache)
        layer.max_context_chunk = max_context_chunk
    a, b, c = (torch.randint(0, 128, (n,)).tolist() for n in (11, 8, 3))
    table_a, table_b, table_c = [5, 2, 9], [7, 0], [3]    # scattered, so a wrong page walk shows

    want_a, want_b, want_c = (reference_logits(reference, tokens) for tokens in (a, b, c))

    steps = [
        [(row(a, 0, 5, table_a), want_a)],
        [(row(a, 5, 4, table_a), want_a), (row(b, 0, 7, table_b), want_b)],
        [(row(a, 9, 1, table_a, decode=True), want_a), (row(b, 7, 1, table_b, decode=True), want_b)],
        [(row(a, 10, 1, table_a, decode=True), want_a), (row(c, 0, 3, table_c), want_c)],
    ]
    for rows in steps:
        seqs = [seq for seq, _ in rows]
        for (seq, want), got in zip(rows, step(runner, model, seqs)):
            start = seq.num_cached_tokens
            torch.testing.assert_close(got, want[start:start + seq.num_scheduled_tokens], rtol=1e-4, atol=1e-4)


def test_context_chunks_split_long_rows_and_skip_empty_ones():
    # (first row, starts, lengths): row 0 splits three ways, row 1 has nothing cached, 2 and 3 share a chunk.
    assert plan_context_chunks([9, 0, 2, 3], budget=4) == [
        (0, [0], [4]), (0, [4], [4]), (0, [8], [1]), (2, [0, 0], [2, 2]), (3, [2], [1]),
    ]


@pytest.mark.parametrize("latent_decode", [True, False])
def test_mixed_rows_keep_latent_decode_and_original_order(models, runner, monkeypatch, latent_decode):
    """Decode must not expand its history when interleaved with resumed and cold prompts."""
    reference, model = models
    monkeypatch.setattr(TorchAttention, "supports_mla_decode", staticmethod(lambda: latent_decode))
    layers = [module for module in model.modules() if isinstance(module, MLAAttention)]
    for layer in layers:
        layer.bind_kv_cache(torch.zeros(*layer.kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE)))
        layer.max_context_chunk = 4
    a, b, c, d = (torch.randint(0, 128, (n,)).tolist() for n in (10, 7, 8, 1))
    tables = [[5, 2, 9], [7, 0], [3, 8], [6]]
    step(runner, model, [row(a, 0, 9, tables[0]), row(b, 0, 6, tables[1]), row(c, 0, 6, tables[2])])
    decoded = []
    original = TorchAttention.mla_decode

    def record_decode(self, q, cache, v_dim, context):
        decoded.append(context.context_lens.tolist())
        return original(self, q, cache, v_dim, context)

    monkeypatch.setattr(TorchAttention, "mla_decode", record_decode)
    expanded = []
    for layer in layers:
        expand = layer.expand

        def record_expand(latent, expand=expand):
            expanded.append(latent.size(0))
            return expand(latent)

        monkeypatch.setattr(layer, "expand", record_expand)
    seqs = [row(c, 6, 2, tables[2]), row(a, 9, 1, tables[0], decode=True),
            row(d, 0, 1, tables[3]), row(b, 6, 1, tables[1], decode=True)]
    for seq, tokens, got in zip(seqs, [c, a, d, b], step(runner, model, seqs)):
        want = reference_logits(reference, tokens)[seq.num_cached_tokens:]
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    assert decoded == ([[10, 7]] * len(layers) if latent_decode else [])
    if latent_decode:
        # Three new prefill tokens and six cached prefill tokens per layer.
        assert sum(expanded) == 9 * len(layers)


class FakeAttention(torch.nn.Module):
    """Deterministic and shaped like the real thing, so both paths see one value."""

    def __init__(self, v_head_dim: int):
        super().__init__()
        self.v_head_dim = v_head_dim

    def forward(self, q, latent):
        return q[..., :self.v_head_dim] * 0.5


def pieces(model) -> tuple:
    """A MoE layer, so the piece after attention carries routing, and what stands in for attention."""
    layer = model.model.layers[1]
    return layer, FakeAttention(layer.self_attn.v_head_dim), model.model.embed_tokens.weight.size(1)


def unsplit(layer, positions, hidden_states, residual, attend):
    """The unsplit layer, written out module for module."""
    if residual is None:
        hidden_states, residual = layer.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    attn = layer.self_attn
    if attn.q_lora_rank is None:
        q = attn.q_proj(hidden_states)
        kv_lora = attn.kv_a_proj_with_mqa(hidden_states)
    else:
        q_c, kv_lora = attn.fused_qkv_a_proj(hidden_states).split(
            [attn.q_lora_rank, attn.kv_lora_rank + attn.qk_rope_head_dim], dim=-1,
        )
        q = attn.q_b_proj(attn.q_a_layernorm(q_c))
    q = q.view(-1, attn.num_local_heads, attn.qk_head_dim)
    kv_c, k_pe = kv_lora.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    kv_c_normed = attn.kv_a_layernorm(kv_c)
    q[..., attn.qk_nope_head_dim:], k_pe = attn.rotary_emb(positions, q[..., attn.qk_nope_head_dim:], k_pe.unsqueeze(1))
    o = attend(q, torch.cat([kv_c_normed, k_pe.squeeze(1)], dim=-1))
    hidden_states = attn.o_proj(o.view(-1, attn.num_local_heads * attn.v_head_dim))
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    return layer.mlp(hidden_states), residual


@pytest.mark.parametrize("first_layer", [True, False])
def test_the_split_layer_matches_the_unsplit_one(models, first_layer):
    """Bitwise, on the same weights: the split piecewise capture takes must move no arithmetic."""
    _, model = models
    layer, attend, hidden_size = pieces(model)
    positions = torch.arange(5)
    hidden_states = torch.randn(5, hidden_size)
    residual = None if first_layer else torch.randn(5, hidden_size)

    with torch.inference_mode():
        # fp32 RMSNorm adds into its input, so each path gets its own copy
        q, latent, carried = layer.pre_attention(positions, hidden_states.clone(), residual)
        got, got_residual = layer.post_attention(attend(q, latent), carried)
        want, want_residual = unsplit(layer, positions, hidden_states.clone(), residual, attend)

    assert torch.equal(got, want)
    assert torch.equal(got_residual, want_residual)


def test_the_pieces_need_no_attention_context(models):
    """What a graph replays cannot depend on this step's sequence layout."""
    reset_context()    # any read of it would see an empty Context and misbehave
    _, model = models
    layer, attend, hidden_size = pieces(model)
    attn = layer.self_attn

    with torch.inference_mode():
        q, latent, residual = layer.pre_attention(torch.arange(5), torch.randn(5, hidden_size), None)
        hidden_states, residual = layer.post_attention(attend(q, latent), residual)

    assert q.shape == (5, attn.num_local_heads, attn.qk_head_dim)
    assert latent.shape == (5, attn.kv_lora_rank + attn.qk_rope_head_dim)
    assert hidden_states.shape == residual.shape == (5, hidden_size)


def test_forward_still_runs_the_pieces(models, monkeypatch):
    """forward is the composition, so the captured path cannot drift from it."""
    _, model = models
    layer, attend, hidden_size = pieces(model)
    monkeypatch.setattr(layer.self_attn, "mla_attn", attend)
    positions, hidden_states = torch.arange(5), torch.randn(5, hidden_size)

    with torch.inference_mode():
        got, got_residual = layer(positions, hidden_states.clone(), None)
        want, want_residual = unsplit(layer, positions, hidden_states.clone(), None, attend)

    assert torch.equal(got, want)
    assert torch.equal(got_residual, want_residual)


def test_the_cache_holds_one_latent_per_token(models):
    """The point of MLA: kv_lora_rank + rope dim per token and layer, not keys and values per head."""
    _, model = models
    layer = next(module for module in model.modules() if isinstance(module, MLAAttention))
    assert layer.kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE) == (1, NUM_BLOCKS, BLOCK_SIZE, 16 + 8)


def test_an_unknown_architecture_is_refused():
    config = DeepseekV2Config(architectures=["GPT2LMHeadModel"])
    with pytest.raises(ValueError, match="unsupported architectures"):
        get_model_class(config)


def test_the_registry_resolves_deepseek_v2():
    assert get_model_class(DeepseekV2Config(architectures=["DeepseekV2ForCausalLM"])) is DeepseekV2ForCausalLM
