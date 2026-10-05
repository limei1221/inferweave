import torch

from lean_vllm.attention import triton_cache
from lean_vllm.attention.abstract import AttentionBackend, LayerSpec
from lean_vllm.attention.flash_backend import FlashAttention3Backend
from lean_vllm.attention.flashinfer_backend import FlashInferBackend
from lean_vllm.utils.context import Context

# DeepSeek's latent: 512 compressed values, which are also the values attended, then the 64-wide rope key.
LATENT_DIM, LATENT_V_DIM = 576, 512


def prefill_backend() -> type[AttentionBackend] | None:
    """What expands MLA prefill, shared by every MLA decode kernel as vLLM's MLACommonImpl shares one:
    FlashAttention-3 on Hopper, else FlashInfer, which stands in for the FlashAttention-2 vLLM uses there."""
    return next((backend for backend in (FlashAttention3Backend, FlashInferBackend) if backend.is_available()), None)


class MLACommonBackend(AttentionBackend):
    """Base of a backend that brings only an MLA decode kernel: latents store with Triton, and prefill runs on
    prefill_backend()."""

    supported_kinds = ("mla",)

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
        self.expanded = prefill_backend()(num_heads, head_dim, scale, num_kv_heads)

    @staticmethod
    def supports_cuda_graph() -> bool:
        return True

    @staticmethod
    def supports_mla_decode() -> bool:
        return True

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        prefill = prefill_backend()
        # FlashInfer's prefill also builds DeepSeek's 192, which its decode kernels do not.
        return prefill is not None and (prefill.supports_head_size(head_size) or
                                        (prefill is FlashInferBackend and head_size == 192))

    @staticmethod
    def supports_value_head_size(head_size: int, v_head_size: int) -> bool:
        prefill = prefill_backend()
        return prefill is not None and prefill.supports_value_head_size(head_size, v_head_size)

    @classmethod
    def validate(cls, spec: LayerSpec) -> list[str]:
        reasons = super().validate(spec)
        if prefill_backend() is None:
            reasons.append("neither FlashAttention-3 nor FlashInfer is here to run its prefill")
        if spec.latent_dim and spec.latent_dim != LATENT_DIM:
            reasons.append(f"latent width {spec.latent_dim} is not the 512 + 64 its decode kernel takes")
        return reasons

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        raise NotImplementedError(f"the {self.get_name()} backend stores MLA latents only")

    def store_latents(self, latent, latent_cache, slot_mapping) -> None:
        triton_cache.store_latents(latent, latent_cache, slot_mapping)

    def prefill(self, q, k, v, k_cache, v_cache, context: Context) -> torch.Tensor:
        return self.expanded.prefill(q, k, v, k_cache, v_cache, context)

    def varlen_with_lse(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal,
                        host_cu_seqlens=None):
        return self.expanded.varlen_with_lse(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal,
                                             host_cu_seqlens)

    def decode(self, q, k_cache, v_cache, context: Context) -> torch.Tensor:
        raise NotImplementedError(f"the {self.get_name()} backend decodes MLA latents only")
