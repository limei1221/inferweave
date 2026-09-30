from abc import ABC, abstractmethod

import torch

from lean_vllm.utils.context import Context


class AttentionBackend(ABC):
    """Execution strategy for one attention layer. See docs/attention-backends.md."""

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Identifier matching LEAN_VLLM_ATTENTION_BACKEND."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Whether this backend can run on the current machine and build."""

    @staticmethod
    def supports_cuda_graph() -> bool:
        """False if decode branches on tensor values, so cannot be captured."""
        return False

    @staticmethod
    def supports_mla_decode() -> bool:
        """True if mla_decode attends MLA latents directly, so a decode step expands no keys."""
        return False

    @staticmethod
    def supports_full_cudagraph_mla_decode() -> bool:
        """False if mla_decode bakes per-step state a full-graph replay cannot refresh (e.g. a schedule built
        outside the kernel from one step's lengths), so decode must stay eager and only piecewise graphs apply."""
        return True

    @staticmethod
    def mla_block_size() -> int | None:
        """The page size mla_decode requires, or None for any."""
        return None

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int) -> tuple[int, ...]:
        """Per-layer shape of one of the key/value cache tensors."""
        return (num_blocks, block_size, num_kv_heads, head_dim)

    @abstractmethod
    def store_kvcache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter new keys/values into the paged cache in place. Slot -1 skips."""

    def store_latents(
        self,
        latent: torch.Tensor,
        latent_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter latents [num_tokens, latent_dim] into the paged cache in place. Slot -1 skips."""
        raise NotImplementedError(f"the {self.get_name()} backend stores no MLA latents")

    @abstractmethod
    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Causal attention over packed varlen sequences, bottom-right aligned.

        q is [num_tokens, num_heads, head_dim]; k and v hold new tokens only.
        Keys come from the paged cache when context.block_tables is set.
        """

    @abstractmethod
    def varlen_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uncached attention, bottom-right aligned, plus its log-sum-exp [num_tokens, num_heads] for merging."""

    @abstractmethod
    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Single-query attention against the paged cache. q is [batch, heads, dim]."""

    def mla_decode(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        v_dim: int,
        context: Context,
    ) -> torch.Tensor:
        """Single-query attention over a paged MLA latent cache, read as one key head shared by all.

        q is [batch, heads, latent_dim] and latent_cache [num_blocks, block_size, latent_dim].
        Values are each latent's first v_dim entries, so this returns [batch, heads, v_dim].
        """
        raise NotImplementedError(f"the {self.get_name()} backend has no MLA decode")
