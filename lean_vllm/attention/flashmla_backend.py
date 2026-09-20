from einops import rearrange
import torch

from lean_vllm.attention.flash_backend import FlashAttention3Backend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
except ImportError as e:    # built from source, Hopper only
    _IMPORT_ERROR = e


class FlashMLABackend(FlashAttention3Backend):
    """FlashMLA's dense decode over MLA latents, and FlashAttention-3 for everything else. Hopper only."""

    @staticmethod
    def get_name() -> str:
        return "flashmla"

    @staticmethod
    def is_available() -> bool:
        # sm90 only, as FA3 is.
        return _IMPORT_ERROR is None and FlashAttention3Backend.is_available()

    @staticmethod
    def supports_mla_decode() -> bool:
        return True

    @staticmethod
    def supports_full_cudagraph_mla_decode() -> bool:
        # flash_mla_with_kvcache builds a tile schedule from this step's context_lens inside the kernel
        # (this build fuses the old get_mla_metadata into dense_decode_fwd). A full-graph replay reuses the
        # schedule captured for one set of lengths against later, different lengths and reads out of bounds,
        # so decode stays eager and only the rest of the model is captured piecewise.
        return False

    @staticmethod
    def mla_block_size() -> int | None:
        return 64

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        if context.mla_decode_metadata is None:
            # Scheduled by the first layer's call; the rest reuse it, as they share the step's lengths.
            context.mla_decode_metadata, _ = get_mla_metadata()
        o, _ = flash_mla_with_kvcache(
            rearrange(q, "b h d -> b 1 h d"), rearrange(latent_cache, "n p d -> n p 1 d"), context.block_tables, context.context_lens, v_dim,
            context.mla_decode_metadata, softmax_scale=self.scale, causal=True,
        )
        return rearrange(o, "b 1 h d -> b h d")    # match the [batch, heads, v_dim] contract
