"""The opaque attention custom op: its schema, fake, and layer lookup by name."""

import pytest
import torch

from lean_vllm.attention.torch_backend import TorchAttention
from lean_vllm.layers import attention as attention_module
from lean_vllm.layers.attention import Attention, register_layers
from lean_vllm.utils.context import get_context, reset_context, set_context

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE = 2, 1, 8, 4


@pytest.fixture
def model():
    """Two layers, so a lookup by the wrong name is visible in the output."""
    registry = dict(attention_module._LAYERS)
    model = torch.nn.Module()
    model.first = Attention(NUM_HEADS, HEAD_DIM, 0.5, NUM_KV_HEADS, backend=TorchAttention)
    model.second = Attention(NUM_HEADS, HEAD_DIM, 0.5, NUM_KV_HEADS, backend=TorchAttention)
    for layer in (model.first, model.second):
        layer.k_cache = torch.randn(4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
        layer.v_cache = torch.randn(4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    register_layers(model)
    yield model
    attention_module._LAYERS.clear()
    attention_module._LAYERS.update(registry)
    reset_context()


@pytest.fixture
def decode_step():
    """One decoding row, reading three cached tokens out of block 0."""
    torch.manual_seed(0)
    with set_context(
        False,
        slot_mapping=torch.tensor([-1], dtype=torch.int32),  # -1 skips the cache write
        context_lens=torch.tensor([3], dtype=torch.int32),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
    ):
        yield (
            torch.randn(1, NUM_HEADS, HEAD_DIM),
            torch.randn(1, NUM_KV_HEADS, HEAD_DIM),
            torch.randn(1, NUM_KV_HEADS, HEAD_DIM),
        )


def test_layers_are_named_by_module_path(model):
    assert model.first.layer_name == "first"
    assert attention_module._LAYERS["second"] is model.second


def run_op(q, k, v, layer_name: str) -> torch.Tensor:
    out = torch.empty_like(q)
    torch.ops.lean_vllm.attention(q, k, v, out, layer_name)
    return out


def test_the_op_routes_to_the_named_layer(model, decode_step):
    q, k, v = decode_step
    assert torch.equal(run_op(q, k, v, "first"), model.first.attend(q, k, v))
    assert not torch.equal(run_op(q, k, v, "first"), run_op(q, k, v, "second"))


def test_forward_goes_through_the_op(model, decode_step):
    """Not straight to attend(), or there would be nothing for the splitter to cut."""
    q, k, v = decode_step
    assert torch.equal(model.first(q, k, v), model.first.attend(q, k, v))


def test_rows_past_the_step_are_padding_and_left_alone(model, decode_step):
    """A piecewise bucket pads the step; attention reads and writes only the step's own rows."""
    q, k, v = decode_step
    padded = [torch.cat([t, torch.randn(2, *t.shape[1:])]) for t in (q, k, v)]
    get_context().num_actual_tokens = 1
    out = torch.full((3, NUM_HEADS, HEAD_DIM), 7.0)
    torch.ops.lean_vllm.attention(*padded, out, "first")
    assert torch.equal(out[:1], model.first.attend(q, k, v))
    assert (out[1:] == 7).all()


def test_an_unregistered_layer_fails_loudly(model, decode_step):
    q, k, v = decode_step
    with pytest.raises(KeyError, match="never registered"):
        run_op(q, k, v, "third")


def test_the_op_satisfies_its_schema(model, decode_step):
    """opcheck covers the fake against the real shape, which tracing relies on."""
    q, k, v = decode_step
    torch.library.opcheck(torch.ops.lean_vllm.attention.default, (q, k, v, torch.empty_like(q), "first"))
