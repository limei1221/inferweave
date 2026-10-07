"""EPLB on one rank: the policy, the routing op, and a tiny DeepSeek-V3 against transformers as its experts move.

Two ranks moving experts between them are in test_expert_parallel.py.
"""

import numpy as np
import pytest
import torch
import torch.distributed as dist
from test_compilation import forward, prompt
from test_deepseek_v2 import CONFIGS, reference_logits, row, step, tiny_config
from transformers import AutoConfig
from transformers import DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM

from lean_vllm.engine.compilation import compile_piecewise
from lean_vllm.engine.model_runner import ModelRunner
from lean_vllm.eplb import EplbConfig, EplbState, rebalance_experts
from lean_vllm.eplb.policy import keep_slots
from lean_vllm.layers.attention import register_layers
from lean_vllm.layers.moe import FusedMoE, eplb_map
from lean_vllm.models.deepseek_v2 import DeepseekV3ForCausalLM
from lean_vllm.utils.context import reset_context
from lean_vllm.utils.loader import load_model

NUM_REDUNDANT = 2


@pytest.fixture(scope="module")
def process_group():
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_the_busiest_expert_gets_the_spare_slots_and_every_rank_the_same_count():
    load = np.array([[100, 1, 1, 1, 1, 1, 1, 50]])
    placement = rebalance_experts(load, num_physical=12, num_ranks=4)
    assert placement.shape == (1, 12)
    counts = np.bincount(placement[0], minlength=8)
    assert counts.min() == 1  # every expert keeps a slot
    assert counts[0] >= counts[7] > 1  # the spares go where the load per replica is highest
    per_replica = load[0] / counts
    rank_load = per_replica[placement[0]].reshape(4, 3).sum(1)
    assert rank_load.max() < load.max()  # without copies, expert 0's rank would carry all of its 100


def test_a_kept_expert_keeps_its_slot():
    old = np.array([[0, 1, 2, 3, 4, 5]])
    new = np.array([[2, 1, 7, 5, 3, 4]])  # rank 0 keeps 1 and 2, rank 1 keeps all three
    assert keep_slots(new, old, num_ranks=2).tolist() == [[7, 1, 2, 3, 4, 5]]


def test_the_op_picks_a_replica_and_counts_only_real_rows():
    logical_to_physical = torch.tensor([[0, 3], [1, -1], [2, -1]])  # expert 0 has two slots, 0 and 3
    replica_count = torch.tensor([2, 1, 1])
    topk_ids = torch.tensor([[0, 1]] * 6 + [[2, 1]] * 2, dtype=torch.int32)
    load = torch.zeros(4, dtype=torch.int32)

    physical = eplb_map(topk_ids, logical_to_physical, replica_count, load, torch.tensor(6))

    assert physical.dtype == torch.int32
    assert set(physical[:6, 0].tolist()) == {0, 3}  # the hash spreads expert 0 over both replicas
    assert physical[:, 1].tolist() == [1] * 8 and physical[6:, 0].tolist() == [2, 2]
    # The last two rows are padding: expert 2 is picked but carries no load.
    assert load.tolist() == [int((physical[:6, 0] == 0).sum()), 6, 0, int((physical[:6, 0] == 3).sum())]


@pytest.fixture(scope="module")
def checkpoint(process_group, tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("eplb")
    config = tiny_config(**{**CONFIGS["v3"], "num_nextn_predict_layers": 0})
    HFDeepseekV3ForCausalLM(config).save_pretrained(path)
    reference = HFDeepseekV3ForCausalLM.from_pretrained(path, attn_implementation="eager").eval()
    return str(path), reference


def build(path: str, monkeypatch, num_redundant_experts: int = NUM_REDUNDANT, **eplb) -> tuple:
    monkeypatch.setenv("LEAN_VLLM_ATTENTION_BACKEND", "torch")
    eplb_config = EplbConfig(num_redundant_experts=num_redundant_experts, **eplb)
    model = DeepseekV3ForCausalLM(AutoConfig.from_pretrained(path), eplb_config=eplb_config)
    state = EplbState(model, eplb_config)
    load_model(model, path)
    register_layers(model)
    return model, state


@pytest.fixture
def runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = 0
    runner.device = torch.device("cpu")
    runner.block_size = 4
    runner._prev_tokens = runner._prev_rows = None
    return runner


def slots_hold_their_experts(path: str, state):
    """Each slot holds the checkpoint's weights of the logical expert the state says it does."""
    plain = DeepseekV3ForCausalLM(AutoConfig.from_pretrained(path))
    load_model(plain, path)
    plain_layers = [m for m in plain.modules() if isinstance(m, FusedMoE)]
    for layer, plain_layer, placement in zip(state.layers, plain_layers, state.physical_to_logical):
        assert torch.equal(layer.gate_up_proj, plain_layer.gate_up_proj[placement])
        assert torch.equal(layer.down_proj, plain_layer.down_proj[placement])


def test_redundant_slots_load_copies_and_route_like_transformers(checkpoint, runner, monkeypatch):
    path, reference = checkpoint
    model, state = build(path, monkeypatch)
    layer = state.layers[0]
    assert layer.gate_up_proj.size(0) == layer.num_experts + NUM_REDUNDANT
    slots_hold_their_experts(path, state)
    tokens = torch.randint(0, 128, (13,)).tolist()

    (logits,) = step(runner, model, [row(tokens, 0, len(tokens), block_table=[])])

    torch.testing.assert_close(logits, reference_logits(reference, tokens), rtol=1e-4, atol=1e-4)
    # Every pick of every token landed on some slot, and was counted once.
    assert state.expert_load_pass.sum(1).tolist() == [13 * layer.top_k] * len(state.layers)


def test_a_rearrangement_moves_weights_and_maps_together(checkpoint, runner, monkeypatch):
    path, reference = checkpoint
    model, state = build(path, monkeypatch, step_interval=2)
    tokens = torch.randint(0, 128, (11,)).tolist()
    step(runner, model, [row(tokens, 0, len(tokens), block_table=[])])
    state.step()
    before = state.physical_to_logical.copy()
    state.expert_load_window[0, :, 5] += 1000  # make expert 5 the hot one, so it takes the spare slots

    state.step()  # the second step of the interval rearranges

    assert not np.array_equal(state.physical_to_logical, before)
    assert (state.physical_to_logical == 5).sum(1).min() > 1
    assert state.steps == 0 and not state.expert_load_pass.any()
    slots_hold_their_experts(path, state)
    replicas = state.logical_replica_count[:, 5].tolist()
    assert replicas == (state.physical_to_logical == 5).sum(1).tolist()
    (logits,) = step(runner, model, [row(tokens, 0, len(tokens), block_table=[])])
    torch.testing.assert_close(logits, reference_logits(reference, tokens), rtol=1e-4, atol=1e-4)


def test_the_compiled_model_records_the_same_load_as_eager(checkpoint, runner, monkeypatch):
    """The op mutates its load in place; Inductor must neither drop nor double that write."""
    path, _ = checkpoint
    torch._dynamo.reset()
    try:
        model, state = build(path, monkeypatch)
        steps = [prompt(runner, n) for n in (7, 3)]
        want = [forward(model, **s) for s in steps]
        want_load = state.expert_load_pass.clone()
        state.reset()
        compile_piecewise(model)

        got = [forward(model, **s, traces=i == 0) for i, s in enumerate(steps)]

        for g, w in zip(got, want):
            torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-4)
        assert torch.equal(state.expert_load_pass, want_load)
    finally:
        torch._dynamo.reset()
        reset_context()
