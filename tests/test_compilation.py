"""The piecewise-compiled model against eager, on CPU: Inductor pieces, no graphs, the torch attention backend."""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from transformers import DeepseekV2Config, Qwen3Config

from lean_vllm.engine.compilation import Piece, compile_piecewise, mark_dynamic_tokens, weak_ref
from lean_vllm.engine.model_runner import ModelRunner
from lean_vllm.engine.sequence import Sequence
from lean_vllm.layers.attention import register_layers
from lean_vllm.models import get_model_class
from lean_vllm.utils.context import reset_context, set_context

CONFIGS = {
    "qwen3": Qwen3Config(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8, intermediate_size=64,
        num_hidden_layers=2, vocab_size=128, max_position_embeddings=64, architectures=["Qwen3ForCausalLM"],
    ),
    "deepseek_v2": DeepseekV2Config(    # MLA, then MoE after a dense layer
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4, intermediate_size=96,
        moe_intermediate_size=24, n_routed_experts=8, num_experts_per_tok=3, n_shared_experts=1,
        first_k_dense_replace=1, num_hidden_layers=3, vocab_size=128, max_position_embeddings=256,
        q_lora_rank=None, kv_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=8, v_head_dim=10,
        architectures=["DeepseekV2ForCausalLM"],
    ),
}


@pytest.fixture(scope="module")
def process_group():
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture(params=list(CONFIGS))
def model(request, process_group, monkeypatch):
    torch.manual_seed(0)
    torch._dynamo.reset()    # unguarded entries must not outlive the model they were traced for
    config = CONFIGS[request.param]
    monkeypatch.setenv("LEAN_VLLM_ATTENTION_BACKEND", "torch")
    with torch.inference_mode():
        model = get_model_class(config)(config)
        for param in model.parameters():
            param.normal_(0, 0.1)    # the weights are torch.empty until a checkpoint lands
    register_layers(model)
    yield model
    torch._dynamo.reset()
    reset_context()


@pytest.fixture
def runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.device = torch.device("cpu")
    runner.block_size = 16
    runner._prev_tokens = runner._prev_rows = None
    return runner


def prompt(runner, num_tokens: int) -> dict:
    """The context of one fresh prompt, with no cache, as warmup runs it."""
    seq = Sequence(torch.randint(0, 128, (num_tokens,)).tolist())
    seq.num_scheduled_tokens = num_tokens
    input_ids, positions, _, context = runner.prepare_batch([seq])
    return dict(input_ids=input_ids, positions=positions, context=context)


@torch.inference_mode()
def forward(model, input_ids, positions, context, traces=False, **extra) -> torch.Tensor:
    if traces:
        mark_dynamic_tokens(input_ids, positions)    # as the runner does before the call that traces
    with set_context(**context, **extra):
        return model(input_ids, positions)


def test_the_graph_splits_into_one_piece_per_stretch_between_attention_ops(model, runner):
    backend = compile_piecewise(model)
    forward(model, **prompt(runner, 5), traces=True)
    assert len(backend.pieces) == len(model.model.layers) + 1


def test_the_compiled_model_matches_eager_at_every_size_from_one_compile(model, runner):
    """Size 1 included: the trace must not have specialized on the size it saw first."""
    steps = [prompt(runner, n) for n in (7, 1, 2, 13)]
    want = [forward(model, **step) for step in steps]
    backend = compile_piecewise(model)

    got = [forward(model, **step, traces=i == 0) for i, step in enumerate(steps)]

    assert len(backend.pieces) == len(model.model.layers) + 1    # a recompile would add pieces
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-4)


def test_a_padded_step_matches_the_unpadded_one(model, runner):
    """What a piecewise replay runs: the bucket's rows through the pieces, the real rows through attention."""
    step = prompt(runner, 5)
    want = forward(model, **step)
    compile_piecewise(model)
    padded_ids = torch.cat([step["input_ids"], torch.randint(0, 128, (3,))])
    padded_positions = torch.cat([step["positions"], torch.arange(3)])

    got = forward(model, padded_ids, padded_positions, step["context"], traces=True, num_actual_tokens=5)

    torch.testing.assert_close(got[:5], want, rtol=1e-4, atol=1e-4)


def test_a_weak_ref_views_the_memory_without_owning_it():
    x = torch.arange(6.0).view(2, 3)
    view = weak_ref((x[1], 4))
    x[1, 0] = 42
    assert view[0].tolist() == [42.0, 4.0, 5.0] and view[1] == 4


class FakeGraph:
    replays = 0

    def replay(self):
        FakeGraph.replays += 1


def test_a_piece_captures_once_per_size_then_replays(monkeypatch):
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, pool: nullcontext())
    FakeGraph.replays = 0
    calls = []
    piece = Piece(lambda x: calls.append(x) or x * 2, pool=None)
    x = torch.ones(3)

    with set_context(True):
        assert torch.equal(piece(x), x * 2)    # no bucket: compiled code, no graph
    with set_context(True, piecewise_size=4):
        captured = piece(x)    # warmup, then capture; a real graph's pool would keep this memory
        replayed = piece(x)
    assert len(calls) == 3 and FakeGraph.replays == 1 and list(piece.graphs) == [4]
    assert replayed.data_ptr() == captured.data_ptr() and torch.equal(replayed, x * 2)
