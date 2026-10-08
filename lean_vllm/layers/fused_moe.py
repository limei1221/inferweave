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
from typing import Callable

import torch

from lean_vllm import envs

logger = logging.getLogger(__name__)

# Shipped tuned configs; $LEAN_VLLM_TUNED_CONFIG_FOLDER is searched first.
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "moe_configs")

_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
except ImportError as e:  # installed by the cuda extra
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
        if pid_m * BLOCK_SIZE_M >= tl.load(num_rows_ptr):
            return  # a block the padding left empty

        offs_pair = tl.load(sorted_pairs_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        pair_mask = offs_pair < num_valid_pairs  # the tail of an expert's run overhangs its last block
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_pair[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = pair_mask[:, None] & (offs_cn < N)[None, :]
        expert = tl.load(block_experts_ptr + pid_m)
        if expert == -1:  # another EP rank's expert: its pairs add zero here, as in vLLM
            tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=c_ptr.dtype.element_ty), mask=c_mask)
            return

        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # wrapped, so only the store masks N
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

    @triton.jit
    def topk_softmax_kernel(
        logits_ptr,
        bias_ptr,
        weights_ptr,
        ids_ptr,
        num_tokens,
        scaling,
        stride_logits,
        NUM_EXPERTS: tl.constexpr,
        EXPERTS_POW2: tl.constexpr,
        TOP_K: tl.constexpr,
        TOP_K_POW2: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        GROUPS_POW2: tl.constexpr,
        TOPK_GROUP: tl.constexpr,
        SIGMOID: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        RENORMALIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """vLLM's topk_softmax and grouped_topk for BLOCK_T tokens: score, the best groups if grouped, top-k, all in
        registers. A correction bias steers the picks, not the weights."""
        rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = tl.arange(0, EXPERTS_POW2)
        row_mask = rows < num_tokens
        col_mask = cols < NUM_EXPERTS
        logits = tl.load(
            logits_ptr + rows[:, None] * stride_logits + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if SIGMOID:
            scores = tl.sigmoid(logits)
        else:
            logits = tl.where(col_mask[None, :], logits, float("-inf"))
            scores = tl.exp(logits - tl.max(logits, axis=1)[:, None])
            scores = scores / tl.sum(scores, axis=1)[:, None]
        choice = scores
        if HAS_BIAS:
            choice += tl.load(bias_ptr + cols, mask=col_mask, other=0.0)[None, :]
        choice = tl.where(col_mask[None, :], choice, float("-inf"))  # padding is never picked
        if NUM_GROUPS > 1:
            # A group scores its best expert, or its best two with a bias, as V3's; only the TOPK_GROUP best groups
            # stay eligible.
            group = cols // (NUM_EXPERTS // NUM_GROUPS)
            group_cols = tl.arange(0, GROUPS_POW2)
            group_scores = tl.full((BLOCK_T, GROUPS_POW2), float("-inf"), tl.float32)
            for g in tl.static_range(NUM_GROUPS):
                in_group = tl.where(group[None, :] == g, choice, float("-inf"))
                best, best_col = tl.max(in_group, axis=1, return_indices=True)
                if HAS_BIAS:
                    best += tl.max(tl.where(cols[None, :] == best_col[:, None], float("-inf"), in_group), axis=1)
                group_scores = tl.where(group_cols[None, :] == g, best[:, None], group_scores)
            eligible = tl.zeros((BLOCK_T, EXPERTS_POW2), dtype=tl.int32)
            for _ in tl.static_range(TOPK_GROUP):
                picked = tl.argmax(group_scores, axis=1)
                eligible = tl.where(group[None, :] == picked[:, None], 1, eligible)
                group_scores = tl.where(group_cols[None, :] == picked[:, None], float("-inf"), group_scores)
            choice = tl.where(eligible != 0, choice, float("-inf"))

        # Top-k by repeated argmax, as vLLM's kernel: k is small, and a tie goes to the lower expert.
        k_cols = tl.arange(0, TOP_K_POW2)
        weights = tl.zeros((BLOCK_T, TOP_K_POW2), dtype=tl.float32)
        ids = tl.zeros((BLOCK_T, TOP_K_POW2), dtype=tl.int32)
        for k in tl.static_range(TOP_K):
            expert = tl.argmax(choice, axis=1)
            picked = cols[None, :] == expert[:, None]
            weight = tl.sum(tl.where(picked, scores, 0.0), axis=1)
            weights = tl.where(k_cols[None, :] == k, weight[:, None], weights)
            ids = tl.where(k_cols[None, :] == k, expert[:, None], ids)
            choice = tl.where(picked, float("-inf"), choice)
        if RENORMALIZE:
            weights = weights / tl.sum(weights, axis=1)[:, None]
        weights = weights * scaling
        out = rows[:, None] * TOP_K + k_cols[None, :]
        out_mask = row_mask[:, None] & (k_cols < TOP_K)[None, :]
        tl.store(weights_ptr + out, weights, mask=out_mask)
        tl.store(ids_ptr + out, ids, mask=out_mask)

    # align_blocks in three launches. A program owns BLOCK pairs, so a pair's slot is its expert's padded start,
    # plus the pairs of that expert in earlier programs, plus its rank in its own: no atomics, and a fixed order.

    @triton.jit
    def count_experts_kernel(topk_ids_ptr, counts_ptr, num_pairs, EXPERTS_POW2: tl.constexpr, BLOCK: tl.constexpr):
        """counts[program, expert]: how many of the program's pairs go to the expert."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        ids = tl.load(topk_ids_ptr + offs, mask=offs < num_pairs, other=-1)
        experts = tl.arange(0, EXPERTS_POW2)
        counts = tl.sum((ids[:, None] == experts[None, :]).to(tl.int32), axis=0)
        tl.store(counts_ptr + pid * EXPERTS_POW2 + experts, counts)

    @triton.jit
    def scan_experts_kernel(
        counts_ptr,
        expert_starts_ptr,
        sorted_pairs_ptr,
        num_rows_ptr,
        num_programs,
        num_pairs,
        EXPERTS_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        ROWS: tl.constexpr,
        TAIL: tl.constexpr,
    ):
        """One program: counts becomes each program's offset within an expert's run, and runs are padded to BLOCK_M."""
        experts = tl.arange(0, EXPERTS_POW2)
        totals = tl.zeros((EXPERTS_POW2,), dtype=tl.int32)
        for start in range(0, num_programs, ROWS):
            rows = start + tl.arange(0, ROWS)
            ptrs = counts_ptr + rows[:, None] * EXPERTS_POW2 + experts[None, :]
            mask = (rows < num_programs)[:, None]
            counts = tl.load(ptrs, mask=mask, other=0)
            tl.store(ptrs, tl.cumsum(counts, axis=0) - counts + totals[None, :], mask=mask)  # exclusive
            totals += tl.sum(counts, axis=0)
        padded = (totals + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        starts = tl.cumsum(padded, axis=0) - padded
        tl.store(expert_starts_ptr + experts, starts)
        tl.store(num_rows_ptr, tl.sum(padded, axis=0))
        # The overhang of each run's last block: rows the GEMMs mask out.
        for offset in tl.static_range(0, BLOCK_M, TAIL):
            cols = offset + tl.arange(0, TAIL)
            overhang = totals[:, None] + cols[None, :]
            sentinel = tl.zeros((EXPERTS_POW2, TAIL), dtype=tl.int32) + num_pairs
            tl.store(sorted_pairs_ptr + starts[:, None] + overhang, sentinel, mask=overhang < padded[:, None])

    @triton.jit
    def scatter_pairs_kernel(
        topk_ids_ptr,
        counts_ptr,
        expert_starts_ptr,
        expert_map_ptr,
        sorted_pairs_ptr,
        block_experts_ptr,
        num_pairs,
        num_blocks,
        NUM_EXPERTS: tl.constexpr,
        EXPERTS_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_EXPERT_MAP: tl.constexpr,
    ):
        """Each pair to its slot, and each block's expert; program i does BLOCK of each."""
        pid = tl.program_id(0)
        experts = tl.arange(0, EXPERTS_POW2)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < num_pairs
        ids = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
        one_hot = (ids[:, None] == experts[None, :]).to(tl.int32)
        rank = tl.sum(tl.cumsum(one_hot, axis=0) * one_hot, axis=1) - 1  # among this program's pairs of the expert
        ids = tl.where(mask, ids, 0)
        slots = (
            tl.load(expert_starts_ptr + ids, mask=mask, other=0)
            + tl.load(counts_ptr + pid * EXPERTS_POW2 + ids, mask=mask, other=0)
            + rank
        )
        tl.store(sorted_pairs_ptr + slots, offs, mask=mask)

        # A block belongs to the last expert starting at or before it, so an empty expert owns none.
        blocks = pid * BLOCK + tl.arange(0, BLOCK)
        block_mask = blocks < num_blocks
        first_blocks = tl.load(expert_starts_ptr + experts) // BLOCK_M
        owners = (first_blocks[None, :] <= blocks[:, None]) & (experts < NUM_EXPERTS)[None, :]
        block_experts = tl.sum(owners.to(tl.int32), axis=1) - 1
        if HAS_EXPERT_MAP:
            block_experts = tl.load(expert_map_ptr + block_experts, mask=block_mask, other=-1)
        tl.store(block_experts_ptr + blocks, block_experts, mask=block_mask)

    @triton.jit
    def silu_and_mul_kernel(x_ptr, out_ptr, d, stride_x, stride_out, BLOCK: tl.constexpr):
        """out = silu(gate) * up for one row and BLOCK columns, in fp32."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < d
        gate = tl.load(x_ptr + row * stride_x + cols, mask=mask).to(tl.float32)
        up = tl.load(x_ptr + row * stride_x + d + cols, mask=mask).to(tl.float32)
        out = gate * tl.sigmoid(gate) * up
        tl.store(out_ptr + row * stride_out + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


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
    if "H200" in device_name.split("_"):  # one file serves the H200 family, as in vLLM
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
        return {k: v for k, v in config.items() if k != "SPLIT_K"}  # vLLM writes it; the kernel has no split
    return get_default_config(M, E)


def align_blocks(topk_ids: torch.Tensor, num_experts: int, block_m: int) -> tuple[torch.Tensor, ...]:
    """Sort the token-expert pairs by expert, and pad each expert's run to a multiple of block_m.

    Returns each padded row's pair (out of range in an overhang), each block's expert, and the row count. No sync.
    The reference for align_blocks_triton, which the Triton path runs.
    """
    pairs = topk_ids.flatten()
    num_pairs = pairs.numel()
    experts = torch.arange(num_experts, device=pairs.device, dtype=pairs.dtype)
    expert_of_pair, order = pairs.sort(stable=True)  # a run in pair order, as the kernels lay it out
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


def align_blocks_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """align_blocks in three launches where it takes about twenty, with expert_map applied to each block.

    Rows past num_rows are left unwritten, as no GEMM program reads them.
    """
    pairs = topk_ids.flatten()
    num_pairs = pairs.numel()
    experts_pow2 = triton.next_power_of_2(num_experts)
    tile = max(16, 8192 // experts_pow2)  # rows of a one-hot tile, so a tile stays near 8K elements
    num_programs = triton.cdiv(num_pairs, tile)
    num_blocks = triton.cdiv(num_pairs, block_m) + num_experts  # every expert wastes under one whole block
    device = pairs.device
    counts = torch.empty(num_programs, experts_pow2, dtype=torch.int32, device=device)
    expert_starts = torch.empty(experts_pow2, dtype=torch.int32, device=device)
    sorted_pairs = torch.empty(num_blocks * block_m, dtype=torch.int32, device=device)
    block_experts = torch.empty(num_blocks, dtype=torch.int32, device=device)
    num_rows = torch.empty(1, dtype=torch.int32, device=device)
    count_experts_kernel[(num_programs,)](pairs, counts, num_pairs, EXPERTS_POW2=experts_pow2, BLOCK=tile)
    scan_experts_kernel[(1,)](
        counts,
        expert_starts,
        sorted_pairs,
        num_rows,
        num_programs,
        num_pairs,
        EXPERTS_POW2=experts_pow2,
        BLOCK_M=block_m,
        ROWS=tile,
        TAIL=min(block_m, tile),
    )
    scatter_pairs_kernel[(max(num_programs, triton.cdiv(num_blocks, tile)),)](
        pairs,
        counts,
        expert_starts,
        block_experts if expert_map is None else expert_map,
        sorted_pairs,
        block_experts,
        num_pairs,
        num_blocks,
        NUM_EXPERTS=num_experts,
        EXPERTS_POW2=experts_pow2,
        BLOCK_M=block_m,
        BLOCK=tile,
        HAS_EXPERT_MAP=expert_map is not None,
    )
    return sorted_pairs, block_experts, num_rows


def topk_softmax(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scaling: float,
    num_groups: int = 1,
    topk_group: int = 1,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing in one launch: fp32 weights and int32 expert ids, [T, top_k] each, in descending order of pick."""
    num_tokens, num_experts = router_logits.shape
    assert num_experts % num_groups == 0, f"{num_experts} experts do not split into {num_groups} groups"
    weights = torch.empty(num_tokens, top_k, dtype=torch.float32, device=router_logits.device)
    ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=router_logits.device)
    experts_pow2 = triton.next_power_of_2(num_experts)
    block_t = max(1, 4096 // experts_pow2)
    topk_softmax_kernel[(triton.cdiv(num_tokens, block_t),)](
        router_logits,
        e_score_correction_bias,
        weights,
        ids,
        num_tokens,
        scaling,
        router_logits.stride(0),
        NUM_EXPERTS=num_experts,
        EXPERTS_POW2=experts_pow2,
        TOP_K=top_k,
        TOP_K_POW2=triton.next_power_of_2(top_k),
        NUM_GROUPS=num_groups,
        GROUPS_POW2=triton.next_power_of_2(num_groups),
        TOPK_GROUP=topk_group,
        SIGMOID=scoring_func == "sigmoid",
        HAS_BIAS=e_score_correction_bias is not None,
        RENORMALIZE=renormalize,
        BLOCK_T=block_t,
    )
    return weights, ids


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """[N, 2D] -> [N, D] in one launch. layers.activation's is torch.compiled: a Dynamo call per layer in an eager op."""
    num_rows, d = x.size(0), x.size(1) // 2
    out = torch.empty(num_rows, d, dtype=x.dtype, device=x.device)
    block = min(triton.next_power_of_2(d), 1024)
    silu_and_mul_kernel[(num_rows, triton.cdiv(d, block))](x, out, d, x.stride(0), out.stride(0), BLOCK=block)
    return out


def fused_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    expert_map: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """x through its top-k experts: sort into blocks, a GEMM either side of the activation, then sum.

    With expert_map, the weights hold this rank's experts, and blocks of other ranks' experts write zeros.
    config overrides the looked-up launch config, for the tuner.
    """
    num_tokens, _ = x.shape  # [T, D]
    num_experts, gate_up_size, hidden_size = gate_up_proj.shape  # [E, 2I, D]
    launch = config or try_get_optimal_moe_config(num_experts, down_proj.size(2), num_tokens)
    if expert_map is not None:
        num_experts = expert_map.numel()  # blocked by global id, as vLLM's moe_align_block_size
    top_k = topk_ids.size(1)
    num_pairs = num_tokens * top_k
    sorted_pairs, block_experts, num_rows = align_blocks_triton(
        topk_ids, num_experts, launch["BLOCK_SIZE_M"], expert_map
    )
    topk_weights = topk_weights.flatten()  # the kernel scales its fp32 accumulator by them, in their own dtype

    def gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, pairs_per_row: int, mul_routed_weight: bool):
        n, k = b.shape[1], b.shape[2]
        grid = (block_experts.numel() * triton.cdiv(n, launch["BLOCK_SIZE_N"]),)
        fused_moe_kernel[grid](
            a,
            b,
            c,
            sorted_pairs,
            block_experts,
            num_rows,
            topk_weights,
            n,
            k,
            num_pairs,
            block_experts.numel(),
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
            TOP_K=pairs_per_row,
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            **launch,
        )

    # The first GEMM writes every pair's row before the second reads it, so empty is safe.
    h = torch.empty(num_pairs, gate_up_size, device=x.device, dtype=x.dtype)  # [T*K, 2I]
    gemm(x, gate_up_proj, h, top_k, mul_routed_weight=False)  # x has one row per token
    h = act_fn(h)  # [T*K, 2I] -> [T*K, I]
    out = torch.empty(num_pairs, hidden_size, device=x.device, dtype=x.dtype)  # [T*K, D]
    gemm(h, down_proj, out, 1, mul_routed_weight=True)  # h is already one row per pair
    return out.view(num_tokens, top_k, hidden_size).sum(dim=1)  # [T, D]
