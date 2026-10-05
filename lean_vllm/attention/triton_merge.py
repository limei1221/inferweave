"""vLLM's merge_attn_states in one Triton launch: attention over two disjoint key sets, from each one's output and
log-sum-exp, written over the first in place."""

import torch

_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
except ImportError as e:    # installed by the cuda extra
    _IMPORT_ERROR = e
else:

    @triton.jit
    def merge_attn_states_kernel(
        o_a_ptr,
        lse_a_ptr,
        o_b_ptr,
        lse_b_ptr,
        head_dim,
        stride_o_a_t,
        stride_o_a_h,
        stride_o_b_t,
        stride_o_b_h,
        stride_lse_a_t,
        stride_lse_a_h,
        stride_lse_b_t,
        stride_lse_b_h,
        BLOCK_D: tl.constexpr,
    ):
        """One (token, head): each side weighted by its share of the summed exp(lse), in fp32."""
        token = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        lse_a = tl.load(lse_a_ptr + token * stride_lse_a_t + head * stride_lse_a_h).to(tl.float32)
        lse_b = tl.load(lse_b_ptr + token * stride_lse_b_t + head * stride_lse_b_h).to(tl.float32)
        lse_max = tl.maximum(lse_a, lse_b)
        weight_a = tl.exp(lse_a - lse_max)
        weight_b = tl.exp(lse_b - lse_max)
        total = weight_a + weight_b
        dims = tl.arange(0, BLOCK_D)
        mask = dims < head_dim
        o_a_ptrs = o_a_ptr + token * stride_o_a_t + head * stride_o_a_h + dims
        o_a = tl.load(o_a_ptrs, mask=mask).to(tl.float32)
        o_b = tl.load(o_b_ptr + token * stride_o_b_t + head * stride_o_b_h + dims, mask=mask).to(tl.float32)
        o = (o_a * weight_a + o_b * weight_b) / total
        tl.store(o_a_ptrs, o.to(o_a_ptr.dtype.element_ty), mask=mask)
        tl.store(lse_a_ptr + token * stride_lse_a_t + head * stride_lse_a_h, lse_max + tl.log(total))


def merge_attn_states_(o_a: torch.Tensor, lse_a: torch.Tensor, o_b: torch.Tensor, lse_b: torch.Tensor) -> None:
    """o [tokens, heads, dim] and lse [tokens, heads], any strides but a unit-stride dim. Overwrites o_a and lse_a."""
    num_tokens, num_heads, head_dim = o_a.shape
    assert o_a.stride(-1) == 1 and o_b.stride(-1) == 1, "the head dim must be contiguous"
    merge_attn_states_kernel[(num_tokens, num_heads)](
        o_a, lse_a, o_b, lse_b, head_dim,
        o_a.stride(0), o_a.stride(1), o_b.stride(0), o_b.stride(1),
        lse_a.stride(0), lse_a.stride(1), lse_b.stride(0), lse_b.stride(1),
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
