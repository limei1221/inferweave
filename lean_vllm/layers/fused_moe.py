"""A fused MoE built the way vLLM's Triton path builds it.

Pairs are sorted by expert and padded to whole row blocks, so each block reads one expert's weight.
No shape is decided on the host, so the layer stays capturable.
Tile sizes come from a tuned JSON file per shape and GPU, as vLLM's, or vLLM's defaults; benchmarks/tune_moe.py writes them.
"""

import functools
import json
import logging
import os
import re

import torch

from lean_vllm import envs

logger = logging.getLogger(__name__)

# Shipped tuned configs; $LEAN_VLLM_TUNED_CONFIG_FOLDER is searched first.
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "moe_configs")

_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
except ImportError as e:    # installed by the cuda extra
    _IMPORT_ERROR = e
else:

    @triton.jit
    def fused_moe_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        sorted_pairs_ptr,
        block_experts_ptr,
        num_rows_ptr,
        topk_weights_ptr,
        N,
        K,
        num_valid_pairs,
        num_blocks,
        stride_am,
        stride_ak,
        stride_be,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        TOP_K: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """One block of padded rows against one expert: C[pair] = A[pair // TOP_K] @ B[expert]."""
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        # Grouped order, as vLLM's: GROUP_SIZE_M row blocks in turn take each column tile, sharing it in L2.
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
        group_size_m = min(num_blocks - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m
        if pid_m * BLOCK_SIZE_M >= tl.load(num_rows_ptr): return    # a block the padding left empty

        offs_pair = tl.load(sorted_pairs_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        pair_mask = offs_pair < num_valid_pairs    # the tail of an expert's run overhangs its last block
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_pair[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = pair_mask[:, None] & (offs_cn < N)[None, :]
        expert = tl.load(block_experts_ptr + pid_m)
        if expert == -1:    # another EP rank's expert: its pairs add zero here, as in vLLM
            tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty), mask=c_mask)
            return

        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N    # wrapped, so only the store masks N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_pair // TOP_K)[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            k_mask = offs_k < K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=pair_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            acc = tl.dot(a, b, acc=acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            acc *= tl.load(topk_weights_ptr + offs_pair, mask=pair_mask, other=0.0)[:, None]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def use_triton(x: torch.Tensor) -> bool:
    """Triton when it can run here, unless $LEAN_VLLM_MOE_BACKEND asks for one by name."""
    name = envs.LEAN_VLLM_MOE_BACKEND
    if name not in (None, "triton", "torch"):
        raise ValueError(f"unknown moe backend {name!r}, expected one of ['torch', 'triton']")
    available = _IMPORT_ERROR is None and x.is_cuda
    if name == "triton" and not available:
        raise RuntimeError("moe backend 'triton' was requested but is not available on this machine")
    return available and name != "torch"


def get_config_file_name(E: int, N: int, device_name: str | None = None) -> str:
    """vLLM's name for bf16, so its tuned files load here too. N is the intermediate size per expert, after TP."""
    if device_name is None:
        device_name = re.sub(r"[\s/]+", "_", torch.cuda.get_device_name())
    if "H200" in device_name.split("_"):    # one file serves the H200 family, as in vLLM
        device_name = "NVIDIA_H200"
    return f"E={E},N={N},device_name={device_name}.json"


@functools.lru_cache
def get_moe_configs(E: int, N: int) -> dict[int, dict] | None:
    """Batch size -> launch config, from the first file found; None if there is none."""
    file_name = get_config_file_name(E, N)
    folders = [envs.LEAN_VLLM_TUNED_CONFIG_FOLDER, CONFIG_DIR]
    for path in (os.path.join(folder, file_name) for folder in folders if folder):
        if os.path.exists(path):
            logger.info("MoE launch configs from %s", path)
            with open(path) as f:
                configs = json.load(f)
            configs.pop("triton_version", None)
            return {int(m): config for m, config in configs.items()}
    logger.warning("no tuned MoE config %s, so vLLM's defaults; benchmarks/tune_moe.py writes one", file_name)
    return None


def get_default_config(M: int, E: int) -> dict:
    """vLLM's bf16 defaults: small batches are memory-bound and take tall K tiles, large ones big tiles and more warps."""
    block_m = 16 if M <= 32 else 32 if M <= 96 else 64 if M <= 512 else 128
    return dict(
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=64 if M <= 64 else 128,
        BLOCK_SIZE_K=128 if M <= 64 else 64,
        # Grouping only pays when an expert has enough row blocks to share a weight tile.
        GROUP_SIZE_M=16 if M // max(E, 1) > 128 else 1,
        num_warps=4 if M <= 128 else 8,
        num_stages=4 if M <= 32 else 3,
    )


def try_get_optimal_moe_config(E: int, N: int, M: int) -> dict:
    """The tuned config for the nearest batch size M (tokens, not pairs), else the default."""
    configs = get_moe_configs(E, N)
    if configs:
        config = configs[min(configs, key=lambda m: abs(m - M))]
        return {k: v for k, v in config.items() if k != "SPLIT_K"}    # vLLM writes it; the kernel has no split
    return get_default_config(M, E)


def align_blocks(topk_ids: torch.Tensor, num_experts: int, block_m: int) -> tuple[torch.Tensor, ...]:
    """Sort the token-expert pairs by expert, and pad each expert's run to a multiple of block_m.

    Returns each padded row's pair (out of range in an overhang), each block's expert, and the row count. No sync.
    """
    pairs = topk_ids.flatten()
    num_pairs = pairs.numel()
    experts = torch.arange(num_experts, device=pairs.device, dtype=pairs.dtype)
    expert_of_pair, order = pairs.sort()
    starts = torch.searchsorted(expert_of_pair, experts)
    counts = torch.searchsorted(expert_of_pair, experts, right=True) - starts
    padded = (counts + block_m - 1) // block_m * block_m
    padded_starts = padded.cumsum(0) - padded
    # Each pair keeps its rank within its expert's run, moved to where that run was padded to start.
    ranks = torch.arange(num_pairs, device=pairs.device) - starts[expert_of_pair]
    # An upper bound on the blocks: every expert wastes under one whole block.
    num_blocks = (num_pairs + block_m - 1) // block_m + num_experts
    sorted_pairs = torch.full((num_blocks * block_m,), num_pairs, dtype=torch.int32, device=pairs.device)
    sorted_pairs[padded_starts[expert_of_pair] + ranks] = order.to(torch.int32)
    # A block belongs to the last expert starting at or before it, so an empty expert owns none.
    blocks = torch.arange(num_blocks, device=pairs.device)
    block_experts = torch.searchsorted(padded_starts // block_m, blocks, right=True) - 1
    num_rows = (padded_starts[-1] + padded[-1]).to(torch.int32)
    return sorted_pairs, block_experts.to(torch.int32), num_rows


def fused_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act_fn: torch.nn.Module,
    expert_map: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """x through its top-k experts: sort into blocks, a GEMM either side of the activation, then sum.

    With expert_map, the weights hold this rank's experts, and blocks of other ranks' experts write zeros.
    config overrides the looked-up launch config, for the tuner.
    """
    num_tokens, _ = x.shape    # [T, D]
    num_experts, gate_up_size, hidden_size = gate_up_proj.shape    # [E, 2I, D]
    launch = config or try_get_optimal_moe_config(num_experts, down_proj.size(2), num_tokens)
    if expert_map is not None:
        num_experts = expert_map.numel()    # blocked by global id, as vLLM's moe_align_block_size
    top_k = topk_ids.size(1)
    num_pairs = num_tokens * top_k
    sorted_pairs, block_experts, num_rows = align_blocks(topk_ids, num_experts, launch["BLOCK_SIZE_M"])
    if expert_map is not None:
        block_experts = expert_map[block_experts]
    topk_weights = topk_weights.flatten().to(x.dtype)

    def gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, pairs_per_row: int, mul_routed_weight: bool):
        n, k = b.shape[1], b.shape[2]
        grid = (block_experts.numel() * triton.cdiv(n, launch["BLOCK_SIZE_N"]),)
        fused_moe_kernel[grid](
            a, b, c, sorted_pairs, block_experts, num_rows, topk_weights,
            n, k, num_pairs, block_experts.numel(),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), b.stride(2),
            c.stride(0), c.stride(1),
            TOP_K=pairs_per_row, MUL_ROUTED_WEIGHT=mul_routed_weight, **launch,
        )

    # The first GEMM writes every pair's row before the second reads it, so empty is safe.
    h = torch.empty(num_pairs, gate_up_size, device=x.device, dtype=x.dtype)    # [T*K, 2I]
    gemm(x, gate_up_proj, h, top_k, mul_routed_weight=False)    # x has one row per token
    h = act_fn(h)    # [T*K, 2I] -> [T*K, I]
    out = torch.empty(num_pairs, hidden_size, device=x.device, dtype=x.dtype)    # [T*K, D]
    gemm(h, down_proj, out, 1, mul_routed_weight=True)    # h is already one row per pair
    return out.view(num_tokens, top_k, hidden_size).sum(dim=1)    # [T, D]
