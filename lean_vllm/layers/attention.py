import dataclasses
from itertools import accumulate
from typing import Callable

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from lean_vllm.attention import AttentionBackend, get_attention_backend
from lean_vllm.utils import device as dev
from lean_vllm.utils.context import Context, get_context

# A custom op cannot take modules, so layers are looked up by name. One per process.
_LAYERS: dict[str, "Attention"] = {}


def register_layers(model: nn.Module):
    """Name every attention layer, so the `lean_vllm` attention ops can find it. Call before forward."""
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.layer_name = name
            _LAYERS[name] = module


def _layer(layer_name: str) -> "Attention":
    layer = _LAYERS.get(layer_name)
    if layer is None:
        raise KeyError(f"attention layer {layer_name!r} was never registered; call register_layers(model)")
    return layer


@torch.library.custom_op("lean_vllm::attention", mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str) -> torch.Tensor:
    return _layer(layer_name).attend(q, k, v)


@attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_name: str) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("lean_vllm::mla_attention", mutates_args=())
def mla_attention(q: torch.Tensor, latent: torch.Tensor, layer_name: str) -> torch.Tensor:
    return _layer(layer_name).attend(q, latent)


@mla_attention.register_fake
def _(q: torch.Tensor, latent: torch.Tensor, layer_name: str) -> torch.Tensor:
    return q.new_empty(q.size(0), q.size(1), _layer(layer_name).v_head_dim)


class Attention(nn.Module):
    """Paged causal attention for one layer, executed by the selected backend."""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        backend: type[AttentionBackend] | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        backend_cls = backend or get_attention_backend()
        self.backend = backend_cls(num_heads, head_dim, scale, num_kv_heads)
        self.k_cache = self.v_cache = torch.tensor([])
        self.layer_name = ""

    def kv_cache_shape(self, num_blocks: int, block_size: int) -> tuple[int, ...]:
        """This layer's cache: a key and a value tensor, stacked, in the backend's layout."""
        return (2, *self.backend.get_kv_cache_shape(num_blocks, block_size, self.num_kv_heads, self.head_dim))

    def bind_kv_cache(self, cache: torch.Tensor):
        self.k_cache, self.v_cache = cache[0], cache[1]

    def output_shape(self, num_tokens: int) -> tuple[int, ...]:
        """What attend returns for this many tokens; piecewise capture sizes its buffer with it."""
        return (num_tokens, self.num_heads, self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Through an opaque op, so a compiled model would split here; the runner's piecewise capture is manual.
        return torch.ops.lean_vllm.attention(q, k, v, self.layer_name)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """The op's body. Its KV cache write is undeclared, as the cache is module state."""
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.backend.store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            return self.backend.prefill(q, k, v, k_cache, v_cache, context)
        return self.backend.decode(q, k_cache, v_cache, context)


@dataclasses.dataclass(slots=True)
class ContextChunk:
    """Cached keys of consecutive rows, and the queries of those rows."""
    queries: slice    # the rows' tokens in the step's q
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    slots: torch.Tensor    # the cache slot of each key


def plan_context_chunks(context_lens: list[int], budget: int) -> list[tuple[int, list[int], list[int]]]:
    """Pack rows' cached keys into chunks of at most budget, as (first row, starts, lengths). Long rows split."""
    assert budget > 0
    chunks = []
    row = start = 0
    while row < len(context_lens):
        if not context_lens[row]:
            row += 1
            continue
        first, starts, lens, used = row, [], [], 0
        while row < len(context_lens) and context_lens[row] and used < budget:
            take = min(context_lens[row] - start, budget - used)
            starts.append(start)
            lens.append(take)
            used += take
            start += take
            if start == context_lens[row]:
                row, start = row + 1, 0
        chunks.append((first, starts, lens))
    return chunks


def context_chunks(context: Context, block_size: int, budget: int) -> list[ContextChunk]:
    """The step's cached keys in chunks, planned on the host once per step."""
    if context.context_chunks is None:
        cu_q, cu_k = context.cu_seqlens_q_host, context.cu_seqlens_k_host
        context_lens = [(cu_k[i + 1] - cu_k[i]) - (cu_q[i + 1] - cu_q[i]) for i in range(len(cu_q) - 1)]
        device = context.block_tables.device
        context.context_chunks = []
        for first, starts, lens in plan_context_chunks(context_lens, budget):
            last = first + len(lens)
            cu_seqlens_k = [0, *accumulate(lens)]
            num_keys = cu_seqlens_k[-1]
            key_rows = torch.repeat_interleave(dev.make_tensor(lens, torch.int64, device), output_size=num_keys)
            offsets = dev.make_tensor([s - c for s, c in zip(starts, cu_seqlens_k)], torch.int64, device)
            positions = torch.arange(num_keys, device=device) + offsets[key_rows]
            blocks = context.block_tables[first + key_rows, positions // block_size].long()
            context.context_chunks.append(ContextChunk(
                queries=slice(cu_q[first], cu_q[last]),
                cu_seqlens_q=dev.make_tensor([n - cu_q[first] for n in cu_q[first:last + 1]], torch.int32, device),
                cu_seqlens_k=dev.make_tensor(cu_seqlens_k, torch.int32, device),
                max_seqlen_q=max(cu_q[i + 1] - cu_q[i] for i in range(first, last)),
                max_seqlen_k=max(lens),
                slots=blocks * block_size + positions % block_size,
            ))
    return context.context_chunks


def merge_attention(o_a, lse_a, o_b, lse_b) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention over two disjoint key sets, from each one's output and log-sum-exp."""
    weight_b = rearrange(torch.sigmoid(lse_b - lse_a), "n h -> n h 1")    # exp(lse_b) / (exp(lse_a) + exp(lse_b))
    o = torch.lerp(o_a.float(), o_b.float(), weight_b)
    return o.to(o_a.dtype), torch.logaddexp(lse_a, lse_b)


def mla_partitions(context: Context) -> list[tuple[torch.Tensor, Context]]:
    """Build each phase's metadata once, without reading lengths back from the device."""
    if context.mla_partitions is None:
        cu_q, cu_k = context.cu_seqlens_q_host, context.cu_seqlens_k_host
        device = context.block_tables.device
        context.mla_partitions = []
        for is_prefill in (False, True):
            rows = [i for i, phase in enumerate(context.prefill_rows) if phase == is_prefill]
            q_lens = [cu_q[i + 1] - cu_q[i] for i in rows]
            k_lens = [cu_k[i + 1] - cu_k[i] for i in rows]
            tokens = dev.make_tensor([t for i in rows for t in range(cu_q[i], cu_q[i + 1])], torch.int64, device)
            row_indices = dev.make_tensor(rows, torch.int64, device)
            sub_q, sub_k = [0, *accumulate(q_lens)], [0, *accumulate(k_lens)]
            subset = Context(
                is_prefill=is_prefill,
                cu_seqlens_q=dev.make_tensor(sub_q, torch.int32, device),
                cu_seqlens_k=dev.make_tensor(sub_k, torch.int32, device),
                cu_seqlens_q_host=sub_q, cu_seqlens_k_host=sub_k,
                max_seqlen_q=max(q_lens), max_seqlen_k=max(k_lens),
                keys_are_new=sub_q == sub_k,
                block_tables=context.block_tables[row_indices],
                context_lens=context.context_lens[row_indices],
            )
            context.mla_partitions.append((tokens, subset))
    return context.mla_partitions


class MLAAttention(Attention):
    """Multi-head latent attention: the cache holds one compressed latent per token.

    Cached latents expand max_context_chunk at a time, merged by log-sum-exp; MLA decode backends skip expanding.
    """

    max_context_chunk = 8192    # the runner sets its step token budget

    def __init__(
        self,
        num_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        scale: float,
        latent_dim: int,
        expand: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        latent_projections: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        backend: type[AttentionBackend] | None = None,
    ):
        super().__init__(num_heads, qk_head_dim, scale, num_heads, backend or get_attention_backend(mla=True))
        self.v_head_dim = v_head_dim
        self.latent_dim = latent_dim
        self.expand = expand    # methods of the owning layer, so not a registered submodule
        self.latent_projections = latent_projections
        self.latent_cache = torch.tensor([])

    def kv_cache_shape(self, num_blocks: int, block_size: int) -> tuple[int, ...]:
        return (1, num_blocks, block_size, self.latent_dim)

    def bind_kv_cache(self, cache: torch.Tensor):
        self.latent_cache = cache[0]

    def output_shape(self, num_tokens: int) -> tuple[int, ...]:
        return (num_tokens, self.num_heads, self.v_head_dim)

    def forward(self, q: torch.Tensor, latent: torch.Tensor):
        return torch.ops.lean_vllm.mla_attention(q, latent, self.layer_name)

    def attend(self, q: torch.Tensor, latent: torch.Tensor):
        context = get_context()
        cache = self.latent_cache
        if cache.numel():
            self.backend.store_latents(latent, cache, context.slot_mapping)
            if not context.is_prefill and self.backend.supports_mla_decode():
                return self._decode_latents(q, context)    # pure decode
            if (context.is_prefill and context.prefill_rows is not None
                    and not all(context.prefill_rows) and self.backend.supports_mla_decode()):    # mixed step, split below
                out = q.new_empty(self.output_shape(q.size(0)))
                for tokens, subset in mla_partitions(context):
                    if subset.is_prefill:
                        part = self._prefill(q[tokens], latent[tokens], subset)
                    else:
                        part = self._decode_latents(q[tokens], subset)
                    out.index_copy_(0, tokens, part)
                return out
        return self._prefill(q, latent, context)    # expanded path

    def _prefill(self, q: torch.Tensor, latent: torch.Tensor, context: Context) -> torch.Tensor:
        cache = self.latent_cache
        k, v = self.expand(latent)
        if context.keys_are_new or context.block_tables is None:
            # Every key the step reads is new, so a plain prefill with no page table covers it.
            unpaged = dataclasses.replace(context, block_tables=None, keys_are_new=True)
            o = self.backend.prefill(q, k, self._pad(v), self.k_cache, self.v_cache, unpaged)
            return o[..., :self.v_head_dim].contiguous()
        # New tokens attend each other causally, then each chunk of cached keys unmasked.
        cu_seqlens_q, max_seqlen_q = context.cu_seqlens_q, context.max_seqlen_q
        o, lse = self.backend.varlen_with_lse(
            q, k, self._pad(v), cu_seqlens_q, cu_seqlens_q, max_seqlen_q, max_seqlen_q, causal=True,
        )
        del k, v
        latents = rearrange(cache, "b p d -> (b p) d")
        for chunk in context_chunks(context, cache.size(1), self.max_context_chunk):
            k, v = self.expand(latents[chunk.slots])
            rows = chunk.queries
            o_chunk, lse_chunk = self.backend.varlen_with_lse(
                q[rows], k, self._pad(v), chunk.cu_seqlens_q, chunk.cu_seqlens_k,
                chunk.max_seqlen_q, chunk.max_seqlen_k, causal=False,
            )
            del k, v
            o[rows], lse[rows] = merge_attention(o[rows], lse[rows], o_chunk, lse_chunk)
        return o[..., :self.v_head_dim].contiguous()

    def _decode_latents(self, q: torch.Tensor, context: Context) -> torch.Tensor:
        """Attention over the cached latents as they are, so nothing expands.

        q . W_k c = W_k^T q . c moves the query into latent space; W_v applies after, as attention is linear in values.
        """
        w_k, w_v = self.latent_projections()
        q_nope, q_pe = q.split([w_k.size(1), self.head_dim - w_k.size(1)], dim=-1)
        q = torch.cat([torch.einsum("bhn,hnl->bhl", q_nope, w_k), q_pe], dim=-1)
        o = self.backend.mla_decode(q, self.latent_cache, w_k.size(2), context)
        return torch.einsum("bhl,hvl->bhv", o, w_v)

    def _pad(self, v: torch.Tensor) -> torch.Tensor:
        # Backends take one head size, so values pad up to the keys' and are cut back after.
        return F.pad(v, (0, self.head_dim - self.v_head_dim))
