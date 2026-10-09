import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from lean_vllm import envs
from lean_vllm.layers import fused_moe
from lean_vllm.layers.activation import SiluAndMul
from lean_vllm.layers.linear import divide

silu_and_mul = SiluAndMul()
_aux_stream: torch.cuda.Stream | None = None


def aux_stream() -> torch.cuda.Stream:
    """One side stream per process, made on first use, which a warmup reaches before any graph captures."""
    global _aux_stream
    if _aux_stream is None:
        _aux_stream = torch.cuda.Stream()
    return _aux_stream


def shared_mlp(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, act_fn) -> torch.Tensor:
    """The shared experts, one gated MLP. Under TP its output is a partial sum, for the routed experts' all-reduce."""
    return F.linear(act_fn(F.linear(x, gate_up)), down)


def determine_expert_map(ep_size: int, ep_rank: int, num_experts: int) -> tuple[int, torch.Tensor | None]:
    """Linear placement: each rank gets a contiguous run, with one extra expert on the first ranks if needed.

    The map takes a global expert id to its local one, or -1 for another rank's. None when there is one rank.
    """
    if ep_size == 1:
        return num_experts, None
    base, remainder = divmod(num_experts, ep_size)
    local_num_experts = base + 1 if ep_rank < remainder else base
    start = ep_rank * base + min(ep_rank, remainder)
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cpu")
    expert_map[start : start + local_num_experts] = torch.arange(local_num_experts, dtype=torch.int32, device="cpu")
    return local_num_experts, expert_map


def torch_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """The portable path: one row per token-expert pair, two grouped matrix multiplies, scatter back."""
    num_experts, top_k = gate_up_proj.size(0), topk_ids.size(1)
    if expert_map is not None:
        # Another rank's pairs sort past every local run, so no group holds them.
        topk_ids = expert_map[topk_ids]
        topk_ids = topk_ids.masked_fill(topk_ids < 0, num_experts)
    expert_ids, order = topk_ids.flatten().sort()
    token_ids = order // top_k
    # Where each expert's run of sorted rows ends; searchsorted, unlike bincount, does not sync.
    experts = torch.arange(num_experts, device=x.device, dtype=expert_ids.dtype)
    offsets = torch.searchsorted(expert_ids, experts, right=True).to(torch.int32)
    h = F.grouped_mm(x[token_ids], gate_up_proj.transpose(1, 2), offs=offsets)
    h = F.grouped_mm(silu_and_mul(h), down_proj.transpose(1, 2), offs=offsets)
    h = h * topk_weights.flatten()[order].unsqueeze(1).to(h.dtype)
    if expert_map is not None:
        # grouped_mm leaves rows past the last offset undefined.
        h = torch.where((expert_ids < num_experts).unsqueeze(1), h, 0)
    return torch.zeros_like(x).index_add_(0, token_ids, h)


def torch_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scaling: float,
    num_groups: int = 1,
    topk_group: int = 1,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The portable routing: score, the best groups when grouped, then top-k. fp32 weights, int32 ids.

    With a correction bias (V3's noaux_tc) the bias steers which experts are picked, but not their weights.
    """
    scores = router_logits.softmax(dim=-1) if scoring_func == "softmax" else router_logits.sigmoid()
    # scores: [N, E]
    choice = scores if e_score_correction_bias is None else scores + e_score_correction_bias
    if num_groups > 1:
        # Only experts in the topk_group best groups stay eligible.
        num_token = choice.size(0)
        grouped = choice.view(num_token, num_groups, -1)
        if e_score_correction_bias is None:
            group_scores = grouped.max(dim=-1).values  # [N, G]
        else:  # a group scores its best two experts, as V3's
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)  # [N, G]
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [N, topk_group]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1).expand(num_token, num_groups, choice.size(-1) // num_groups).reshape(num_token, -1)
        )  # [N, E]
        choice = choice.masked_fill(~score_mask.bool(), float("-inf"))  # [N, E]
    topk_ids = torch.topk(choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if scaling != 1.0:
        topk_weights = topk_weights * scaling
    return topk_weights, topk_ids.to(torch.int32)


# One launch on the Triton path, where the traced softmax, topk and scaling took several.
@torch.library.custom_op("lean_vllm::select_experts", mutates_args=())
def select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scaling: float,
    num_groups: int,
    topk_group: int,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    args = (router_logits, top_k, renormalize, scaling, num_groups, topk_group, scoring_func, e_score_correction_bias)
    if fused_moe.use_triton(router_logits):
        return fused_moe.topk_softmax(*args)
    return torch_select_experts(*args)


@select_experts.register_fake
def _(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    scaling: float,
    num_groups: int,
    topk_group: int,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (router_logits.size(0), top_k)
    return router_logits.new_empty(shape, dtype=torch.float32), router_logits.new_empty(shape, dtype=torch.int32)


# Opaque to torch.compile: the Triton path picks its launch from the batch size, which a trace would fix.
@torch.library.custom_op("lean_vllm::moe_experts", mutates_args=())
def moe_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    shared_gate_up: torch.Tensor | None = None,
    shared_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """The routed experts, plus the shared ones when given."""
    if not fused_moe.use_triton(x):
        out = torch_experts(x, gate_up_proj, down_proj, topk_weights, topk_ids, expert_map)
        if shared_gate_up is not None and shared_down is not None:
            out = out + shared_mlp(x, shared_gate_up, shared_down, silu_and_mul)
        return out
    has_shared = shared_gate_up is not None
    # A small step leaves SMs idle, so the shared experts take them on a side stream.
    overlap = has_shared and x.size(0) <= envs.LEAN_VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
    if overlap:
        stream, current = aux_stream(), torch.cuda.current_stream()
        stream.wait_stream(current)
        # No record_stream for x or the output: the current stream waits on the side one before either is freed.
        with torch.cuda.stream(stream):
            shared = shared_mlp(x, shared_gate_up, shared_down, fused_moe.silu_and_mul)  # type: ignore[arg-type]
    out = fused_moe.fused_experts(
        x, gate_up_proj, down_proj, topk_weights, topk_ids, fused_moe.silu_and_mul, expert_map
    )
    if overlap:
        current.wait_stream(stream)
    elif has_shared:
        shared = shared_mlp(x, shared_gate_up, shared_down, fused_moe.silu_and_mul)  # type: ignore[arg-type]
    if has_shared:
        out += shared
    return out


@moe_experts.register_fake
def _(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    shared_gate_up: torch.Tensor | None = None,
    shared_down: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


class FusedMoE(nn.Module):
    """Routed experts, stacked per projection and run as two grouped matrix multiplies.

    Triton kernels from `fused_moe.py` on CUDA, `grouped_mm` elsewhere. TP shards each expert's intermediate size;
    with expert parallelism each rank holds whole experts instead. Either way the ranks'
    partial sums meet in one all-reduce.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        enable_expert_parallel: bool = False,
    ):
        super().__init__()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        use_ep = enable_expert_parallel and world_size > 1
        self.tp_rank, self.tp_size = (0, 1) if use_ep else (rank, world_size)
        self.ep_rank, self.ep_size = (rank, world_size) if use_ep else (0, 1)
        self.num_experts = num_experts  # global; the stacked weights hold local_num_experts
        self.top_k = top_k
        self.intermediate_size = divide(intermediate_size, self.tp_size)
        self.local_num_experts, expert_map = determine_expert_map(self.ep_size, self.ep_rank, num_experts)
        self.local_expert_ids = expert_map.tolist() if expert_map is not None else None  # for the loader
        self.gate_up_proj = nn.Parameter(torch.empty(self.local_num_experts, 2 * self.intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.local_num_experts, hidden_size, self.intermediate_size))
        self.gate_up_proj.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        self.down_proj.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        if expert_map is not None:
            expert_map = expert_map.to(self.gate_up_proj.device)
        self.register_buffer("expert_map", expert_map, persistent=False)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: tuple[int, str]):
        expert_id, proj = shard_id
        if self.local_expert_ids is not None:
            expert_id = self.local_expert_ids[expert_id]
            if expert_id == -1:  # another rank's expert
                return
        if proj == "down_proj":
            param.data[expert_id].copy_(loaded_weight.chunk(self.tp_size, 1)[self.tp_rank])
            return
        offset = 0 if proj == "gate_proj" else self.intermediate_size
        shard = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        param.data[expert_id].narrow(0, offset, self.intermediate_size).copy_(shard)

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_gate_up: torch.Tensor | None = None,
        shared_down: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """With the shared experts' weights, their output is added in, before the one all-reduce both need."""
        out = torch.ops.lean_vllm.moe_experts(
            x, self.gate_up_proj, self.down_proj, topk_weights, topk_ids, self.expert_map, shared_gate_up, shared_down
        )
        if self.tp_size > 1 or self.ep_size > 1:
            dist.all_reduce(out)
        return out
