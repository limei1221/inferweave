"""Tune the Triton MoE's launch config per batch size, as vLLM's benchmarks/kernels/benchmark_moe.py --tune does.

Every config in the search space is timed on random routing for each batch size, and the fastest is written to
E=..,N=..,device_name=...json, the file lean_vllm/layers/fused_moe.py looks up at runtime. Without --tune, it
times the config the runtime would pick. One GPU; vLLM spreads batch sizes over GPUs with Ray.

    uv run python benchmarks/tune_moe.py --model ~/workspace/huggingface/DeepSeek-V2-Lite-Chat --tune
"""

import argparse
import gc
import json
import os
from datetime import datetime
from itertools import product

import torch
import triton
from tqdm import tqdm
from transformers import AutoConfig

from lean_vllm.layers.fused_moe import (
    CONFIG_DIR, fused_experts, get_config_file_name, silu_and_mul, try_get_optimal_moe_config,
)

BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
# vLLM's CUDA search space: 1,920 configs.
SEARCH_SPACE = dict(
    BLOCK_SIZE_M=[16, 32, 64, 128, 256],
    BLOCK_SIZE_N=[32, 64, 128, 256],
    BLOCK_SIZE_K=[64, 128, 256],
    GROUP_SIZE_M=[1, 16, 32, 64],
    num_warps=[4, 8],
    num_stages=[2, 3, 4, 5],
)
CACHE_CLEAR_INTERVAL = 50    # configs between clearing compiled kernels, which otherwise pile up


def search_space() -> list[dict]:
    keys, values = zip(*SEARCH_SPACE.items())
    return [dict(zip(keys, config)) for config in product(*values)]


def model_shape(path: str, tp_size: int, enable_expert_parallel: bool) -> tuple[int, int, int, int]:
    """(experts on a rank, intermediate size per expert on a rank, hidden size, top-k), as the layer shards them."""
    config = AutoConfig.from_pretrained(path)
    E, N = config.n_routed_experts, config.moe_intermediate_size
    if enable_expert_parallel:
        assert E % tp_size == 0, f"{E} experts do not split over {tp_size} ranks"
        E //= tp_size
    else:
        assert N % tp_size == 0, f"intermediate size {N} does not split over {tp_size} ranks"
        N //= tp_size
    return E, N, config.hidden_size, config.num_experts_per_tok


def benchmark_config(config: dict | None, num_tokens: int, E: int, N: int, hidden_size: int, top_k: int,
                     dtype: torch.dtype, num_iters: int) -> float:
    """Mean microseconds per call: 10 calls in one CUDA graph, replayed num_iters times on fresh routing."""
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    gate_up_proj = torch.randn(E, 2 * N, hidden_size, dtype=dtype)
    down_proj = torch.randn(E, hidden_size, N, dtype=dtype)
    gating_output = torch.randn(num_iters, num_tokens, E, dtype=torch.float32)
    # Graph inputs: each iteration copies its routing in before the replay. The dtypes select_experts returns.
    topk_weights = torch.empty(num_tokens, top_k, dtype=torch.float32)
    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.int32)

    def prepare(i: int):
        weights, ids = gating_output[i].softmax(dim=-1).topk(top_k, dim=-1)
        topk_weights.copy_(weights / weights.sum(dim=-1, keepdim=True))
        topk_ids.copy_(ids)

    def run():
        fused_experts(x, gate_up_proj, down_proj, topk_weights, topk_ids, silu_and_mul, config=config)

    prepare(0)
    run()    # compiles, and raises OutOfResources for a config that does not fit
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            run()
    torch.cuda.synchronize()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    latencies = []
    for i in range(num_iters):
        prepare(i)
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))
    graph.reset()
    return sum(latencies) / (num_iters * 10) * 1000


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def tune(num_tokens: int, shape: tuple, dtype: torch.dtype, configs: list[dict]) -> tuple[dict, float]:
    best_config, best_time = None, float("inf")
    for i, config in enumerate(tqdm(configs, desc=f"batch {num_tokens}", leave=False)):
        try:
            # 20 iterations, as vLLM's tuner: enough to rank, not to report.
            kernel_time = benchmark_config(config, num_tokens, *shape, dtype, num_iters=20)
        except triton.runtime.autotuner.OutOfResources:    # the path vLLM catches
            continue    # too much shared memory or too many registers for this GPU
        if kernel_time < best_time:
            best_config, best_time = config, kernel_time
        if i and i % CACHE_CLEAR_INTERVAL == 0:
            clear_cache()
    clear_cache()
    assert best_config is not None, f"no config in the search space fits batch size {num_tokens}"
    return best_config, best_time


def save_configs(configs: dict[int, dict], E: int, N: int, save_dir: str) -> str:
    path = os.path.join(save_dir, get_config_file_name(E, N))
    os.makedirs(save_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"triton_version": triton.__version__, **configs}, f, indent=4)
        f.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="a local checkpoint directory")
    parser.add_argument("--tp-size", "-tp", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", "-enable-ep", action="store_true")
    parser.add_argument("--batch-size", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--save-dir", default=CONFIG_DIR, help="where --tune writes; the shipped folder by default")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    shape = model_shape(os.path.expanduser(args.model), args.tp_size, args.enable_expert_parallel)
    E, N, hidden_size, top_k = shape
    print(f"E={E} N={N} hidden_size={hidden_size} top_k={top_k} on {torch.cuda.get_device_name()}")

    if not args.tune:
        for num_tokens in args.batch_size:
            config = try_get_optimal_moe_config(E, N, num_tokens)
            kernel_time = benchmark_config(None, num_tokens, *shape, dtype, num_iters=100)
            print(f"batch {num_tokens}: {kernel_time:.1f} us with {config}")
        return

    configs = search_space()
    print(f"tuning over {len(configs)} configs per batch size")
    best = {}
    for num_tokens in args.batch_size:
        best[num_tokens], kernel_time = tune(num_tokens, shape, dtype, configs)
        print(f"[{datetime.now().ctime()}] batch {num_tokens}: {kernel_time:.1f} us with {best[num_tokens]}")
    print(f"wrote {save_configs(best, E, N, args.save_dir)}")


if __name__ == "__main__":
    main()
