import torch

from lean_vllm.attention import triton_cache
from lean_vllm.attention.abstract import AttentionBackend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError as e:  # Hopper only, and built by the cuda extra
    _IMPORT_ERROR = e


class FlashAttention3Backend(AttentionBackend):
    """FlashAttention-3 kernels with a Triton KV-cache scatter. Hopper only."""

    @staticmethod
    def get_name() -> str:
        return "flash_attn_3"

    @staticmethod
    def is_available() -> bool:
        # FA3 is built for Hopper (sm90) only.
        return (
            _IMPORT_ERROR is None
            and triton_cache._IMPORT_ERROR is None
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 9
        )

    @staticmethod
    def supports_cuda_graph() -> bool:
        return True

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256  # as vLLM's FlashAttention backend

    @staticmethod
    def supports_value_head_size(head_size: int, v_head_size: int) -> bool:
        # FA3 builds one mixed size, for DeepSeek's MLA prefill; vLLM skips the padding on Hopper for it too.
        return head_size == v_head_size or (head_size, v_head_size) == (192, 128)

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        triton_cache.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    def store_latents(self, latent, latent_cache, slot_mapping) -> None:
        triton_cache.store_latents(latent, latent_cache, slot_mapping)

    def prefill(self, q, k, v, k_cache, v_cache, context: Context) -> torch.Tensor:
        if context.block_tables is None:
            # No pages (warmup, or MLA's new tokens): k and v hold every key.
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        # Keys come from the pages, cold rows too, as vLLM's V1; FA3's varlen entry takes no page table.
        assert context.cu_seqlens_k is not None
        cache_seqlens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            page_table=context.block_tables,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_q=context.max_seqlen_q,
            softmax_scale=self.scale,
            causal=True,
        )

    def varlen_with_lse(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, host_cu_seqlens=None
    ):
        o, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=causal,
            return_attn_probs=True,
        )
        return o, lse.transpose(0, 1)  # FA3's varlen lse is [heads, tokens]

    def decode(self, q, k_cache, v_cache, context: Context) -> torch.Tensor:
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=context.context_lens,
            page_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        )
        return o.squeeze(1)  # match the [batch, heads, dim] contract
