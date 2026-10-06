from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from lean_vllm.utils.context import Context, split_decodes_and_prefills


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """What one attention layer asks of a backend; the selector picks a backend per layer by it."""

    head_size: int  # of queries and keys; an MLA layer's expanded prefill head
    num_heads: int
    num_kv_heads: int
    dtype: torch.dtype
    latent_dim: int = 0  # an MLA layer's cached latent width; 0 for a layer that caches keys and values

    @property
    def kind(self) -> str:
        return "mla" if self.latent_dim else "decoder"


class AttentionBackend(ABC):
    """Execution strategy for one attention layer. See docs/attention-backends.md."""

    # What the backend serves, checked by validate; vLLM's defaults are fp16 and bf16 too.
    supported_dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)
    supported_kinds: tuple[str, ...] = ("decoder", "mla")

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
        """False keeps a model using this backend eager: nothing compiled, no graphs."""
        return False

    @staticmethod
    def supports_full_cudagraph() -> bool:
        """False if decode cannot be captured in a full graph, e.g. it is planned on the host each step."""
        return True

    @classmethod
    def before_full_graph_replay(cls, context: Context, batch_size: int) -> None:
        """Refresh what the full graph captured at batch_size reads but cannot compute itself, from this step's
        pure-decode context, before it replays. vLLM's metadata builders run before every replay too."""

    @staticmethod
    def split_decodes() -> bool:
        """True if forward sends a step's one-query rows to decode and the rest to prefill, as vLLM's backends with
        reorder_batch_threshold = 1 do. False sends any step with a prompt row to prefill whole."""
        return False

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        return True

    @staticmethod
    def supports_value_head_size(head_size: int, v_head_size: int) -> bool:
        """Whether prefill takes values narrower than keys, as an MLA layer's are. False pads them to the keys' size."""
        return head_size == v_head_size

    @classmethod
    def validate(cls, spec: LayerSpec) -> list[str]:
        """Why this backend cannot serve the layer, or nothing if it can. As vLLM's validate_configuration."""
        reasons = []
        if spec.dtype not in cls.supported_dtypes:
            reasons.append(f"dtype {spec.dtype} is not one of {list(cls.supported_dtypes)}")
        if not cls.supports_head_size(spec.head_size):
            reasons.append(f"head size {spec.head_size} is not supported")
        if spec.num_heads % spec.num_kv_heads:
            reasons.append(f"{spec.num_heads} query heads do not group over {spec.num_kv_heads} key heads")
        if spec.kind not in cls.supported_kinds:
            reasons.append(f"{spec.kind} layers are not supported")
        return reasons

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
        host_cu_seqlens: tuple[list[int], list[int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uncached attention, bottom-right aligned, plus its log-sum-exp [num_tokens, num_heads] for merging.

        Only MLA layers call it, to merge chunks of expanded latents. host_cu_seqlens, the two on the host, spares
        a backend that plans on the host a sync.
        """
        raise NotImplementedError(f"the {self.get_name()} backend has no varlen_with_lse, so serves no MLA layer")

    @abstractmethod
    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """Single-query attention against the paged cache. q is [batch, heads, dim]."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """One step of a layer, after its keys are stored. A splitting backend runs the leading one-query rows
        through decode and the rest through prefill, as vLLM does; others hand a step with prompt rows to prefill."""
        if not context.is_prefill:
            return self.decode(q, k_cache, v_cache, context)
        if not self.split_decodes():
            return self.prefill(q, k, v, k_cache, v_cache, context)
        n, decodes, prefills = split_decodes_and_prefills(context)
        if decodes is None:
            assert prefills is not None
            return self.prefill(q, k, v, k_cache, v_cache, prefills)
        if prefills is None:
            return self.decode(q, k_cache, v_cache, decodes)
        out = torch.empty_like(q)
        out[:n] = self.decode(q[:n], k_cache, v_cache, decodes)
        out[n:] = self.prefill(q[n:], k[n:], v[n:], k_cache, v_cache, prefills)
        return out

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
