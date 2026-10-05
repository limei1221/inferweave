import torch

from lean_vllm.attention import triton_cache, triton_mla_decode
from lean_vllm.attention.mla_common import MLACommonBackend, prefill_backend
from lean_vllm.utils.context import Context


class TritonMLABackend(MLACommonBackend):
    """vLLM's Triton MLA decode over latents, at any page size, and the shared MLA prefill. sm80 and up."""

    @staticmethod
    def get_name() -> str:
        return "triton_mla"

    @staticmethod
    def is_available() -> bool:
        return (triton_mla_decode._IMPORT_ERROR is None and triton_cache._IMPORT_ERROR is None
                and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                and prefill_backend() is not None)

    @staticmethod
    def supports_full_cudagraph_mla_decode() -> bool:
        return True    # it reads lengths and pages on the device and plans nothing on the host

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        return triton_mla_decode.mla_decode(q, latent_cache, context.block_tables, context.context_lens, v_dim,
                                            self.scale)
