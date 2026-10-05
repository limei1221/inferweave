"""vLLM's Triton MLA decode (triton_decode_attention.py, from SGLang and LightLLM): each row's query heads attend its
paged latents as one shared key head, its keys split in NUM_KV_SPLITS parts, which a second launch merges by
log-sum-exp."""

import torch

_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
except ImportError as e:    # installed by the cuda extra
    _IMPORT_ERROR = e
else:

    @triton.jit
    def mla_decode_stage1_kernel(
        q_ptr,
        cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        mid_ptr,
        sm_scale,
        stride_q_b,
        stride_q_h,
        stride_cache_slot,
        stride_block_tables_b,
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        num_heads,
        PAGE_SIZE: tl.constexpr,
        V_DIM: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
    ):
        """One row, BLOCK_H heads and one split of the row's keys: the split's output and log-sum-exp."""
        row = tl.program_id(0)
        heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        split = tl.program_id(2)
        head_mask = heads < num_heads

        seq_len = tl.load(context_lens_ptr + row)
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        start = split * split_len
        end = tl.minimum(start + split_len, seq_len)
        if end <= start: return    # stage 2 skips this split by the same arithmetic

        offs_v = tl.arange(0, V_DIM)    # the latent's leading entries, which are also its value
        offs_rope = V_DIM + tl.arange(0, ROPE_DIM)
        q_rows = q_ptr + row * stride_q_b + heads[:, None] * stride_q_h
        q = tl.load(q_rows + offs_v[None, :], mask=head_mask[:, None], other=0.0)
        q_rope = tl.load(q_rows + offs_rope[None, :], mask=head_mask[:, None], other=0.0)

        m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
        l = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, V_DIM], dtype=tl.float32)
        for n in range(start, end, BLOCK_N):
            offs_n = n + tl.arange(0, BLOCK_N)
            key_mask = offs_n < end
            pages = tl.load(block_tables_ptr + row * stride_block_tables_b + offs_n // PAGE_SIZE,
                            mask=key_mask, other=0)
            latents = cache_ptr + (pages.to(tl.int64) * PAGE_SIZE + offs_n % PAGE_SIZE) * stride_cache_slot
            k = tl.load(latents[None, :] + offs_v[:, None], mask=key_mask[None, :], other=0.0)
            k_rope = tl.load(latents[None, :] + offs_rope[:, None], mask=key_mask[None, :], other=0.0)
            scores = (tl.dot(q, k) + tl.dot(q_rope, k_rope)) * sm_scale
            scores = tl.where(head_mask[:, None] & key_mask[None, :], scores, float("-inf"))
            v = tl.load(latents[:, None] + offs_v[None, :], mask=key_mask[:, None], other=0.0)
            m_new = tl.maximum(m, tl.max(scores, 1))
            rescale = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])
            acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v)
            l = l * rescale + tl.sum(p, 1)
            m = m_new

        mid = mid_ptr + row * stride_mid_b + heads * stride_mid_h + split * stride_mid_s
        tl.store(mid[:, None] + offs_v[None, :], acc / l[:, None], mask=head_mask[:, None])
        tl.store(mid + V_DIM, m + tl.log(l), mask=head_mask)


    @triton.jit
    def mla_decode_stage2_kernel(
        mid_ptr,
        out_ptr,
        context_lens_ptr,
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        stride_out_b,
        stride_out_h,
        V_DIM: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
    ):
        """One (row, head): its splits weighted by their share of the summed exp(lse), in fp32."""
        row = tl.program_id(0)
        head = tl.program_id(1)
        seq_len = tl.load(context_lens_ptr + row)
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        offs_v = tl.arange(0, V_DIM)
        mid = mid_ptr + row * stride_mid_b + head * stride_mid_h

        m = float("-inf")
        l = 0.0
        acc = tl.zeros([V_DIM], dtype=tl.float32)
        for split in range(NUM_KV_SPLITS):
            if split * split_len < seq_len:
                o = tl.load(mid + split * stride_mid_s + offs_v)
                lse = tl.load(mid + split * stride_mid_s + V_DIM)
                m_new = tl.maximum(m, lse)
                rescale = tl.exp(m - m_new)
                weight = tl.exp(lse - m_new)
                acc = acc * rescale + weight * o
                l = l * rescale + weight
                m = m_new
        # A full graph's padding rows hold no keys, so write them zeros rather than 0 / 0.
        out = acc / tl.where(l > 0, l, 1.0)
        tl.store(out_ptr + row * stride_out_b + head * stride_out_h + offs_v, out.to(out_ptr.dtype.element_ty))


NUM_KV_SPLITS = 4    # as vLLM's TritonMLAImpl
BLOCK_H = 16
BLOCK_N = 16    # vLLM's for a 576-wide latent


def mla_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    v_dim: int,
    scale: float,
    num_kv_splits: int = NUM_KV_SPLITS,
) -> torch.Tensor:
    """q [batch, heads, latent_dim] over a cache [num_blocks, page_size, latent_dim]: [batch, heads, v_dim]."""
    batch, num_heads, latent_dim = q.shape
    page_size = cache.size(1)
    assert q.stride(-1) == 1 and cache.stride(-1) == 1 and cache.stride(0) == page_size * cache.stride(1)
    # Each split's output and, after it, its log-sum-exp.
    mid = torch.empty(batch, num_heads, num_kv_splits, v_dim + 1, dtype=torch.float32, device=q.device)
    out = q.new_empty(batch, num_heads, v_dim)
    mla_decode_stage1_kernel[(batch, triton.cdiv(num_heads, BLOCK_H), num_kv_splits)](
        q, cache, block_tables, context_lens, mid, scale,
        q.stride(0), q.stride(1), cache.stride(1), block_tables.stride(0),
        mid.stride(0), mid.stride(1), mid.stride(2), num_heads,
        PAGE_SIZE=page_size, V_DIM=v_dim, ROPE_DIM=latent_dim - v_dim, BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N,
        NUM_KV_SPLITS=num_kv_splits, num_warps=4, num_stages=2,
    )
    mla_decode_stage2_kernel[(batch, num_heads)](
        mid, out, context_lens, mid.stride(0), mid.stride(1), mid.stride(2), out.stride(0), out.stride(1),
        V_DIM=v_dim, NUM_KV_SPLITS=num_kv_splits, num_warps=4, num_stages=2,
    )
    return out
