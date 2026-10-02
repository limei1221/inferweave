import torch

from lean_vllm.attention.abstract import LayerSpec
from lean_vllm.attention.flash_backend import FlashAttention3Backend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
except ImportError as e:    # built from source, Hopper only
    _IMPORT_ERROR = e


class FlashMLABackend(FlashAttention3Backend):
    """FlashMLA's dense decode over MLA latents, and FlashAttention-3 for everything else. Hopper only."""

    supported_kinds = ("mla",)

    @staticmethod
    def get_name() -> str:
        return "flashmla"

    @staticmethod
    def is_available() -> bool:
        # sm90 only, as FA3 is.
        return _IMPORT_ERROR is None and FlashAttention3Backend.is_available()

    @classmethod
    def validate(cls, spec: LayerSpec) -> list[str]:
        reasons = super().validate(spec)    # FA3's, which runs the expanded prefill
        if spec.latent_dim and spec.latent_dim != 576:
            reasons.append(f"latent width {spec.latent_dim} is not the 512 + 64 FlashMLA decodes")
        return reasons

    @staticmethod
    def supports_mla_decode() -> bool:
        return True

    @staticmethod
    def supports_full_cudagraph_mla_decode() -> bool:
        # flash_mla_with_kvcache builds a tile schedule and split-KV workspace from context_lens inside the
        # kernel (this build fuses the old get_mla_metadata into dense_decode_fwd). The full graph is captured
        # with worst-case context_lens (max_model_len), so the baked schedule and workspace are sized for the
        # longest sequence; the kernel gates its KV loop on the context_lens each replay refreshes, so a replay
        # with shorter, different lengths stays in bounds.
        return True

    @staticmethod
    def mla_block_size() -> int | None:
        return 64

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        if context.mla_decode_metadata is None:
            # A holder the kernel fills with the schedule on the first layer's call; the rest reuse it.
            context.mla_decode_metadata, _ = get_mla_metadata()
        o, _ = flash_mla_with_kvcache(
            q.unsqueeze(1), latent_cache.unsqueeze(-2),    # add a query length and a head dim of 1
            context.block_tables, context.context_lens, v_dim,
            context.mla_decode_metadata, softmax_scale=self.scale, causal=True,
        )
        return o.squeeze(1)    # match the [batch, heads, v_dim] contract
