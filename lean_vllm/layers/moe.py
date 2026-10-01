import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lean_vllm.layers import fused_moe
from lean_vllm.layers.activation import SiluAndMul
from lean_vllm.layers.linear import divide


silu_and_mul = SiluAndMul()


def determine_expert_map(ep_size: int, ep_rank: int, num_experts: int) -> tuple[int, torch.Tensor | None]:
    """vLLM's linear placement: each rank a contiguous run, the first ranks one extra when it does not divide.

    The map takes a global expert id to its local one, or -1 for another rank's. None when there is one rank.
    """
    if ep_size == 1:
        return num_experts, None
    base, remainder = divmod(num_experts, ep_size)
    local_num_experts = base + 1 if ep_rank < remainder else base
    start = ep_rank * base + min(ep_rank, remainder)
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cpu")
    expert_map[start:start + local_num_experts] = torch.arange(local_num_experts, dtype=torch.int32, device="cpu")
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


# Opaque to torch.compile, as vLLM's: the Triton path picks its launch from the batch size, which a trace would fix.
@torch.library.custom_op("lean_vllm::moe_experts", mutates_args=())
def moe_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    if fused_moe.use_triton(x):
        return fused_moe.fused_experts(x, gate_up_proj, down_proj, topk_weights, topk_ids, silu_and_mul, expert_map)
    return torch_experts(x, gate_up_proj, down_proj, topk_weights, topk_ids, expert_map)


@moe_experts.register_fake
def _(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty_like(x)


class FusedMoE(nn.Module):
    """Routed experts, stacked per projection and run as two grouped matrix multiplies.

    Triton kernels from `fused_moe.py` on CUDA, `grouped_mm` elsewhere. TP shards each expert's intermediate size;
    with expert parallelism each rank holds whole experts instead, as in vLLM without DP. Either way the ranks'
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
        self.num_experts = num_experts    # global; the stacked weights hold local_num_experts
        self.top_k = top_k
        self.intermediate_size = divide(intermediate_size, self.tp_size)
        self.local_num_experts, expert_map = determine_expert_map(self.ep_size, self.ep_rank, num_experts)
        self.local_expert_ids = expert_map.tolist() if expert_map is not None else None    # for the loader
        self.gate_up_proj = nn.Parameter(torch.empty(self.local_num_experts, 2 * self.intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.local_num_experts, hidden_size, self.intermediate_size))
        self.gate_up_proj.weight_loader = self.weight_loader
        self.down_proj.weight_loader = self.weight_loader
        if expert_map is not None:
            expert_map = expert_map.to(self.gate_up_proj.device)
        self.register_buffer("expert_map", expert_map, persistent=False)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: tuple[int, str]):
        expert_id, proj = shard_id
        if self.local_expert_ids is not None:
            expert_id = self.local_expert_ids[expert_id]
            if expert_id == -1:    # another rank's expert
                return
        if proj == "down_proj":
            param.data[expert_id].copy_(loaded_weight.chunk(self.tp_size, 1)[self.tp_rank])
            return
        offset = 0 if proj == "gate_proj" else self.intermediate_size
        shard = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        param.data[expert_id].narrow(0, offset, self.intermediate_size).copy_(shard)

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        out = torch.ops.lean_vllm.moe_experts(
            x, self.gate_up_proj, self.down_proj, topk_weights, topk_ids, self.expert_map)
        if self.tp_size > 1 or self.ep_size > 1:
            dist.all_reduce(out)
        return out
