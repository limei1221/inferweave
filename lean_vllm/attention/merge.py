"""Attention over two disjoint key sets, from each one's output and log-sum-exp."""

import torch

from lean_vllm.attention import triton_merge


def merge_attention(o_a, lse_a, o_b, lse_b) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention over two disjoint key sets, from each one's output and log-sum-exp."""
    weight_b = torch.sigmoid(lse_b - lse_a).unsqueeze(-1)  # exp(lse_b) / (exp(lse_a) + exp(lse_b))
    o = torch.lerp(o_a.float(), o_b.float(), weight_b)
    return o.to(o_a.dtype), torch.logaddexp(lse_a, lse_b)


def merge_attention_(o_a, lse_a, o_b, lse_b, out=None) -> None:
    """merge_attention written over out, or else o_a, and lse_a: one Triton launch on CUDA, as vLLM's
    merge_attn_states."""
    if o_a.is_cuda and triton_merge._IMPORT_ERROR is None:
        triton_merge.merge_attn_states_(o_a, lse_a, o_b, lse_b, out)
        return
    o, lse = merge_attention(o_a, lse_a, o_b, lse_b)
    (o_a if out is None else out).copy_(o)
    lse_a.copy_(lse)
