"""Expert parallelism end to end: two gloo ranks on the CPU, on a tiny DeepSeek-V2 checkpoint, against transformers."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_deepseek_v2 import BLOCK_SIZE, reference_logits, row, tiny_config
from transformers import AutoConfig
from transformers import DeepseekV2ForCausalLM as HFDeepseekV2ForCausalLM

from lean_vllm.eplb import EplbConfig, EplbState
from lean_vllm.layers.attention import register_layers
from lean_vllm.layers.moe import FusedMoE
from lean_vllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from lean_vllm.utils.context import set_context
from lean_vllm.utils.loader import load_model

WORLD_SIZE = 2
NUM_REDUNDANT = 2  # 8 experts and 2 copies: 5 slots a rank
PROMPT = [5, 17, 99, 3, 64, 120, 8, 42, 77, 1, 30]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def prefill(runner, model) -> torch.Tensor | None:
    input_ids, positions, _, context = runner.prepare_batch([row(PROMPT, 0, len(PROMPT), block_table=[])])
    context["logits_indices"] = None  # every position, not just the last
    with torch.inference_mode(), set_context(**context):
        return model.compute_logits(model(input_ids, positions))  # None off rank 0


def run_rank(rank: int, port: int, path: str, mode: str, out: str):
    """One prefill on this rank's shard. Rank 0 saves the gathered logits and how its experts were split.

    With EPLB, expert 7 is then made hot, the ranks rearrange, and a second prefill runs on the moved experts.
    """
    from lean_vllm.engine.model_runner import ModelRunner

    os.environ["LEAN_VLLM_ATTENTION_BACKEND"] = "torch"
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=WORLD_SIZE)
    try:
        eplb_config = EplbConfig(num_redundant_experts=NUM_REDUNDANT) if mode == "ep_eplb" else None
        model = DeepseekV2ForCausalLM(
            AutoConfig.from_pretrained(path), enable_expert_parallel=mode != "tp", eplb_config=eplb_config
        )
        state = EplbState(model, eplb_config) if eplb_config is not None else None
        load_model(model, path)
        runner = ModelRunner.__new__(ModelRunner)
        runner.rank, runner.device, runner.block_size = rank, torch.device("cpu"), BLOCK_SIZE
        runner._prev_tokens = runner._prev_rows = None
        register_layers(model)
        logits = prefill(runner, model)
        result = {"logits": logits}
        if state is not None:
            state.expert_load_window[0, :, 7] += 1000  # slot 7 holds expert 7, on rank 1
            state.rearrange()
            result.update(after=prefill(runner, model), placement=state.physical_to_logical)
        if rank == 0:
            layers = [m for m in model.modules() if isinstance(m, FusedMoE)]
            split = [(m.ep_size, m.tp_size, m.gate_up_proj.size(0), m.gate_up_proj.size(1)) for m in layers]
            torch.save({**result, "split": split}, out)
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ep")
    HFDeepseekV2ForCausalLM(tiny_config()).save_pretrained(path)
    reference = HFDeepseekV2ForCausalLM.from_pretrained(path, attn_implementation="eager").eval()
    return str(path), reference_logits(reference, PROMPT)


@pytest.mark.parametrize("mode", ["ep", "tp", "ep_eplb"])
def test_two_ranks_match_transformers(checkpoint, tmp_path, mode):
    path, want = checkpoint
    out = str(tmp_path / "rank0.pt")
    mp.spawn(run_rank, args=(free_port(), path, mode, out), nprocs=WORLD_SIZE)
    got = torch.load(out, weights_only=False)

    config = tiny_config()
    # (ep_size, tp_size, experts held, gate_up rows): whole experts under EP, half of each under TP.
    if mode == "ep":
        expected = (2, 1, config.n_routed_experts // 2, 2 * config.moe_intermediate_size)
    elif mode == "ep_eplb":
        expected = (2, 1, (config.n_routed_experts + NUM_REDUNDANT) // 2, 2 * config.moe_intermediate_size)
    else:
        expected = (1, 2, config.n_routed_experts, config.moe_intermediate_size)
    assert got["split"] == [expected] * (config.num_hidden_layers - config.first_k_dense_replace)
    torch.testing.assert_close(got["logits"], want, rtol=1e-4, atol=1e-4)
    if mode == "ep_eplb":
        # Copies of the hot expert reached rank 0, sent by rank 1, and the model still computes the same.
        assert (got["placement"][:, : expected[2]] == 7).any(axis=1).all()
        torch.testing.assert_close(got["after"], want, rtol=1e-4, atol=1e-4)
