import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lean_vllm.layers import fused_moe
from lean_vllm.layers.activation import SiluAndMul
from lean_vllm.layers.linear import divide


silu_and_mul = SiluAndMul()


def torch_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """The portable path: one row per token-expert pair, two grouped matrix multiplies, scatter back."""
    num_experts, top_k = gate_up_proj.size(0), topk_ids.size(1)
    expert_ids, order = topk_ids.flatten().sort()
    token_ids = order // top_k
    # Where each expert's run of sorted rows ends; searchsorted, unlike bincount, does not sync.
    experts = torch.arange(num_experts, device=x.device, dtype=expert_ids.dtype)
    offsets = torch.searchsorted(expert_ids, experts, right=True).to(torch.int32)
    h = F.grouped_mm(x[token_ids], gate_up_proj.transpose(1, 2), offs=offsets)
    h = F.grouped_mm(silu_and_mul(h), down_proj.transpose(1, 2), offs=offsets)
    h = h * topk_weights.flatten()[order].unsqueeze(1).to(h.dtype)
    return torch.zeros_like(x).index_add_(0, token_ids, h)


# Opaque to torch.compile, as vLLM's: the Triton path picks its launch from the batch size, which a trace would fix.
@torch.library.custom_op("lean_vllm::moe_experts", mutates_args=())
def moe_experts(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    if fused_moe.use_triton(x):
        return fused_moe.fused_experts(x, gate_up_proj, down_proj, topk_weights, topk_ids, silu_and_mul)
    return torch_experts(x, gate_up_proj, down_proj, topk_weights, topk_ids)


@moe_experts.register_fake
def _(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


class FusedMoE(nn.Module):
    """Routed experts, stacked per projection and run as two grouped matrix multiplies.

    Triton kernels from `fused_moe.py` on CUDA, `grouped_mm` elsewhere. TP shards each expert's intermediate size.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = divide(intermediate_size, self.tp_size)
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * self.intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, self.intermediate_size))
        self.gate_up_proj.weight_loader = self.weight_loader
        self.down_proj.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: tuple[int, str]):
        expert_id, proj = shard_id
        if proj == "down_proj":
            param.data[expert_id].copy_(loaded_weight.chunk(self.tp_size, 1)[self.tp_rank])
            return
        offset = 0 if proj == "gate_proj" else self.intermediate_size
        shard = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        param.data[expert_id].narrow(0, offset, self.intermediate_size).copy_(shard)

    def forward(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
        out = torch.ops.lean_vllm.moe_experts(x, self.gate_up_proj, self.down_proj, topk_weights, topk_ids)
        if self.tp_size > 1:
            dist.all_reduce(out)
        return out
