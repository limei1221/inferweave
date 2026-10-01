import torch

from lean_vllm.attention.abstract import AttentionBackend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    import triton
    import triton.language as tl
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError as e:    # Hopper only, and built by the cuda extra
    _IMPORT_ERROR = e
else:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


    @triton.jit
    def store_latents_kernel(
        latent_ptr,
        latent_stride,
        cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        offsets = tl.arange(0, BLOCK)
        mask = offsets < D    # a latent is 576 wide for V2-Lite, so the block overhangs it
        latent = tl.load(latent_ptr + idx * latent_stride + offsets, mask=mask)
        tl.store(cache_ptr + slot * D + offsets, latent, mask=mask)


class FlashAttention3Backend(AttentionBackend):
    """FlashAttention-3 kernels with a Triton KV-cache scatter. Hopper only."""

    @staticmethod
    def get_name() -> str:
        return "flash_attn_3"

    @staticmethod
    def is_available() -> bool:
        # FA3 is built for Hopper (sm90) only.
        return (_IMPORT_ERROR is None and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 9)

    @staticmethod
    def supports_cuda_graph() -> bool:
        return True

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        num_tokens, num_heads, head_dim = key.shape
        dim = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == dim and v_cache.stride(1) == dim
        assert slot_mapping.numel() == num_tokens
        store_kvcache_kernel[(num_tokens,)](
            key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, dim
        )

    def store_latents(self, latent, latent_cache, slot_mapping) -> None:
        num_tokens, dim = latent.shape
        assert latent.stride(-1) == 1 and latent_cache.stride(-2) == dim
        assert slot_mapping.numel() == num_tokens
        store_latents_kernel[(num_tokens,)](
            latent, latent.stride(0), latent_cache, slot_mapping, dim, triton.next_power_of_2(dim)
        )

    def prefill(self, q, k, v, k_cache, v_cache, context: Context) -> torch.Tensor:
        if context.keys_are_new or context.block_tables is None:
            # k and v hold every key this batch attends (cold prompts), so skip the pages.
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale, causal=True,
            )
        # Some row reads cached keys, and FA3's varlen entry takes no page table.
        cache_seqlens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
        return flash_attn_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens, page_table=context.block_tables,
            cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
            softmax_scale=self.scale, causal=True,
        )

    def varlen_with_lse(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
        o, lse = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale, causal=causal, return_attn_probs=True,
        )
        return o, lse.transpose(0, 1)    # FA3's varlen lse is [heads, tokens]

    def decode(self, q, k_cache, v_cache, context: Context) -> torch.Tensor:
        o = flash_attn_with_kvcache(
            q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens, page_table=context.block_tables,
            softmax_scale=self.scale, causal=True,
        )
        return o.squeeze(1)    # match the [batch, heads, dim] contract
