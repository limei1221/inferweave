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

from lean_vllm.layers.attention import register_layers
from lean_vllm.layers.moe import FusedMoE
from lean_vllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from lean_vllm.utils.context import set_context
from lean_vllm.utils.loader import load_model

WORLD_SIZE = 2
PROMPT = [5, 17, 99, 3, 64, 120, 8, 42, 77, 1, 30]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def run_rank(rank: int, port: int, path: str, enable_expert_parallel: bool, out: str):
    """One prefill on this rank's shard. Rank 0 saves the gathered logits and how its experts were split."""
    from lean_vllm.engine.model_runner import ModelRunner

    os.environ["LEAN_VLLM_ATTENTION_BACKEND"] = "torch"
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=WORLD_SIZE)
    try:
        model = DeepseekV2ForCausalLM(AutoConfig.from_pretrained(path), enable_expert_parallel=enable_expert_parallel)
        load_model(model, path)
        runner = ModelRunner.__new__(ModelRunner)
        runner.rank, runner.device, runner.block_size = rank, torch.device("cpu"), BLOCK_SIZE
        runner._prev_tokens = runner._prev_rows = None
        register_layers(model)
        input_ids, positions, _, context = runner.prepare_batch([row(PROMPT, 0, len(PROMPT), block_table=[])])
        context["logits_indices"] = None  # every position, not just the last
        with torch.inference_mode(), set_context(**context):
            logits = model.compute_logits(model(input_ids, positions))
        if rank == 0:
            layers = [m for m in model.modules() if isinstance(m, FusedMoE)]
            split = [(m.ep_size, m.tp_size, m.gate_up_proj.size(0), m.gate_up_proj.size(1)) for m in layers]
            torch.save({"logits": logits, "split": split}, out)
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ep")
    HFDeepseekV2ForCausalLM(tiny_config()).save_pretrained(path)
    reference = HFDeepseekV2ForCausalLM.from_pretrained(path, attn_implementation="eager").eval()
    return str(path), reference_logits(reference, PROMPT)


@pytest.mark.parametrize("enable_expert_parallel", [True, False], ids=["ep", "tp"])
def test_two_ranks_match_transformers(checkpoint, tmp_path, enable_expert_parallel):
    path, want = checkpoint
    out = str(tmp_path / "rank0.pt")
    mp.spawn(run_rank, args=(free_port(), path, enable_expert_parallel, out), nprocs=WORLD_SIZE)
    got = torch.load(out)

    config = tiny_config()
    # (ep_size, tp_size, experts held, gate_up rows): whole experts under EP, half of each under TP.
    if enable_expert_parallel:
        expected = (2, 1, config.n_routed_experts // 2, 2 * config.moe_intermediate_size)
    else:
        expected = (1, 2, config.n_routed_experts, config.moe_intermediate_size)
    assert got["split"] == [expected] * (config.num_hidden_layers - config.first_k_dense_replace)
    torch.testing.assert_close(got["logits"], want, rtol=1e-4, atol=1e-4)
