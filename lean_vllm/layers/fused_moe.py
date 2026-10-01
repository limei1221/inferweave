"""A fused MoE built the way vLLM's Triton path builds it.

Pairs are sorted by expert and padded to whole row blocks, so each block reads one expert's weight.
No shape is decided on the host, so the layer stays capturable.
"""

import torch

from lean_vllm import envs

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
        stride_am,
        stride_ak,
        stride_be,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        TOP_K: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One block of padded rows against one expert: C[pair] = A[pair // TOP_K] @ B[expert]."""
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
        if pid_m * BLOCK_M >= tl.load(num_rows_ptr): return    # a block the padding left empty

        offs_pair = tl.load(sorted_pairs_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
        pair_mask = offs_pair < num_valid_pairs    # the tail of an expert's run overhangs its last block
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_pair[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = pair_mask[:, None] & (offs_cn < N)[None, :]
        expert = tl.load(block_experts_ptr + pid_m)
        if expert == -1:    # another EP rank's expert: its pairs add zero here, as in vLLM
            tl.store(c_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=c_ptr.dtype.element_ty), mask=c_mask)
            return

        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N    # wrapped, so only the store masks N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_pair // TOP_K)[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=pair_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            acc = tl.dot(a, b, acc=acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

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


def config(num_pairs: int) -> dict:
    """vLLM tunes this per shape and dtype; a decode batch is a few rows, so the row block matters most."""
    return dict(BLOCK_M=16 if num_pairs < 256 else 64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=4)


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
) -> torch.Tensor:
    """x through its top-k experts: sort into blocks, a GEMM either side of the activation, then sum.

    With expert_map, the weights hold this rank's experts, and blocks of other ranks' experts write zeros.
    """
    num_tokens, _ = x.shape    # [T, D]
    num_experts, gate_up_size, hidden_size = gate_up_proj.shape    # [E, 2I, D]
    if expert_map is not None:
        num_experts = expert_map.numel()    # blocked by global id, as vLLM's moe_align_block_size
    top_k = topk_ids.size(1)
    num_pairs = num_tokens * top_k
    launch = config(num_pairs)
    sorted_pairs, block_experts, num_rows = align_blocks(topk_ids, num_experts, launch["BLOCK_M"])
    if expert_map is not None:
        block_experts = expert_map[block_experts]
    topk_weights = topk_weights.flatten().to(x.dtype)

    def gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, pairs_per_row: int, mul_routed_weight: bool):
        n, k = b.shape[1], b.shape[2]
        grid = (block_experts.numel() * triton.cdiv(n, launch["BLOCK_N"]),)
        fused_moe_kernel[grid](
            a, b, c, sorted_pairs, block_experts, num_rows, topk_weights,
            n, k, num_pairs,
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
