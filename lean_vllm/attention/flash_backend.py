import numpy as np
import torch
import torch.nn.functional as F

from lean_vllm.attention import triton_cache
from lean_vllm.attention.abstract import AttentionBackend, write_into
from lean_vllm.attention.merge import merge_attention_
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache, get_scheduler_metadata
except ImportError as e:  # Hopper only, and built by the cuda extra
    _IMPORT_ERROR = e

# vLLM's thresholds: below them, cascade's second kernel costs more than reading the prefix once saves.
CASCADE_MIN_PREFIX_LEN = 256
CASCADE_MIN_ROWS = 8


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def use_cascade_attention(
    common_prefix_len: int, query_lens: np.ndarray, num_heads: int, num_kv_heads: int, num_sms: int
) -> bool:
    """vLLM's heuristic, for FA's 128-wide tiles: cascade pays unless every row decodes and flash decoding, which
    splits each row's keys across CTAs, would fill the SMs in fewer waves over the prefix."""
    if common_prefix_len < CASCADE_MIN_PREFIX_LEN or len(query_lens) < CASCADE_MIN_ROWS:
        return False
    group = num_heads // num_kv_heads
    if group == 1 or not np.all(query_lens == 1):
        return True  # FA uses no flash decoding here
    num_rows, tile = len(query_lens), 128
    prefix_tiles = _cdiv(common_prefix_len, tile)
    cascade_time = _cdiv(num_heads * _cdiv(num_rows, tile), num_sms) * prefix_tiles
    flash_decoding_time = _cdiv(num_rows * num_kv_heads * _cdiv(group, tile) * prefix_tiles, num_sms)
    return cascade_time < flash_decoding_time


def _step_cache(context: Context) -> dict:
    if context.attn_metadata is None:
        context.attn_metadata = {}
    return context.attn_metadata


class FlashAttention3Backend(AttentionBackend):
    """FlashAttention-3 kernels with a Triton KV-cache scatter. Hopper only.

    Paged calls take a tile schedule made once per step (vLLM's AOT schedule), and rows sharing a long cached prefix
    attend it once for all (vLLM's cascade attention).
    """

    _graph_problems: dict[tuple, dict] = {}  # a full graph's decode schedule, by key: how to remake it before replay
    _graph_schedules: dict[tuple, torch.Tensor] = {}  # what full graphs read their schedule from, per layer shape
    _num_sms: int | None = None

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

    def use_cascade_attention(self, common_prefix_len: int, query_lens: np.ndarray) -> bool:
        num_sms = FlashAttention3Backend._num_sms
        if num_sms is None:
            num_sms = FlashAttention3Backend._num_sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        return use_cascade_attention(common_prefix_len, query_lens, self.num_heads, self.num_kv_heads, num_sms)

    def prefill(self, q, k, v, k_cache, v_cache, context: Context, out=None) -> torch.Tensor:
        # FA3's entry points take no out, so one given is copied into.
        if context.block_tables is None:
            # No pages (warmup, or MLA's new tokens): k and v hold every key.
            o = flash_attn_varlen_func(
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
            return write_into(out, o)
        # Keys come from the pages, cold rows too, as vLLM's V1; FA3's varlen entry takes no page table.
        if context.common_prefix_len:
            return self._cascade(q, k_cache, v_cache, context, out)
        assert context.cu_seqlens_k is not None
        cache_seqlens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
        o = self._paged(
            "prefill",
            q,
            k_cache,
            v_cache,
            context,
            cache_seqlens,
            context.block_tables,
            context.cu_seqlens_q,
            context.max_seqlen_q,
        )
        return write_into(out, o)

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

    def decode(self, q, k_cache, v_cache, context: Context, out=None) -> torch.Tensor:
        if context.common_prefix_len:
            return self._cascade(q, k_cache, v_cache, context, out)
        o = self._paged("decode", q.unsqueeze(1), k_cache, v_cache, context, context.context_lens, context.block_tables)
        return write_into(out, o.squeeze(1))  # match the [batch, heads, dim] contract

    def _cascade(self, q, k_cache, v_cache, context: Context, out) -> torch.Tensor:
        """vLLM's cascade_attention: every query attends the pages all rows share in one unmasked pass, then its own
        keys past them causally, and the two merge by log-sum-exp into out."""
        prefix_len = context.common_prefix_len
        num_prefix_pages = prefix_len // k_cache.size(1)
        tables, lens = context.block_tables, context.context_lens
        assert tables is not None and lens is not None and context.cu_seqlens_q is not None
        cache = _step_cache(context)
        if ("fa3", "cascade_lens") not in cache:
            prefix_lens = torch.full((1,), prefix_len, dtype=torch.int32, device=lens.device)
            cache["fa3", "cascade_lens"] = prefix_lens, lens - prefix_len
        prefix_lens, suffix_lens = cache["fa3", "cascade_lens"]
        # The step's queries as one row of a batch of one, so the prefix is read once.
        o_prefix, lse_prefix, *_ = self._paged(
            "prefix",
            q.unsqueeze(0),
            k_cache,
            v_cache,
            context,
            prefix_lens,
            tables[:1, :num_prefix_pages],
            causal=False,
            return_lse=True,
        )
        o_suffix, lse_suffix, *_ = self._paged(
            "suffix",
            q,
            k_cache,
            v_cache,
            context,
            suffix_lens,
            tables[:, num_prefix_pages:],
            context.cu_seqlens_q,
            context.max_seqlen_q,
            return_lse=True,
        )
        o_prefix = o_prefix[0]
        merge_attention_(o_prefix, lse_prefix[0].T, o_suffix, lse_suffix.T, out)  # FA3's lse is [heads, tokens]
        return o_prefix if out is None else out

    def _paged(
        self,
        kind: str,
        q,
        k_cache,
        v_cache,
        context: Context,
        cache_seqlens,
        page_table,
        cu_seqlens_q=None,
        max_seqlen_q=None,
        causal=True,
        return_lse=False,
    ):
        """FA3 over the pages, q batched or packed by cu_seqlens_q, on the step's schedule for this problem."""
        schedule = self._schedule(
            kind, q, k_cache, v_cache, context, cache_seqlens, page_table, cu_seqlens_q, max_seqlen_q, causal
        )
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=self.scale,
            causal=causal,
            scheduler_metadata=schedule,
            return_softmax_lse=return_lse,
        )

    def _schedule(
        self,
        kind: str,
        q,
        k_cache,
        v_cache,
        context: Context,
        cache_seqlens,
        page_table,
        cu_seqlens_q,
        max_seqlen_q,
        causal,
    ) -> torch.Tensor:
        """FA3's tile schedule for one problem, made by the step's first layer to pose it so the rest skip the
        in-kernel pass, as vLLM's AOT schedule. A full graph reads it from a buffer refilled before each replay."""
        page_size = k_cache.size(1)
        shape = (self.num_heads, self.num_kv_heads, self.head_dim, v_cache.size(-1), q.dtype, page_size)
        key = ("fa3", kind, *shape, context.full_graph_size)  # the step cache holds other backends' plans too
        cache = _step_cache(context)
        if key in cache:
            return cache[key]
        varlen = cu_seqlens_q is not None
        # Every argument the kernel derives its split count from must match, or it reads a schedule for another.
        problem = dict(
            batch_size=cu_seqlens_q.numel() - 1 if varlen else q.size(0),
            max_seqlen_q=max_seqlen_q if varlen else q.size(1),
            max_seqlen_k=page_table.size(1) * page_size,  # the kernel takes it from the page table's width
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            headdim=self.head_dim,
            qkv_dtype=q.dtype,
            headdim_v=v_cache.size(-1),
            cu_seqlens_q=cu_seqlens_q,
            page_size=page_size,
            causal=causal,
        )
        schedule = get_scheduler_metadata(cache_seqlens=cache_seqlens, **problem)
        if context.full_graph_size is not None:  # the capture's warmup; the capture finds it here
            assert not varlen, "a full graph holds decode only"
            FlashAttention3Backend._graph_problems[key] = problem
            schedule = _into_graph_schedule(key, schedule)
        cache[key] = schedule
        return schedule

    @classmethod
    def before_full_graph_replay(cls, context: Context, batch_size: int) -> None:
        """Remake the schedule each full graph at batch_size reads, for this step's lengths, as vLLM's builder does
        before every replay."""
        lens = None
        for key, problem in FlashAttention3Backend._graph_problems.items():
            if key[-1] != batch_size:
                continue
            if lens is None:  # the graph's rows past the step's hold no keys
                assert context.context_lens is not None
                lens = F.pad(context.context_lens, (0, batch_size - context.context_lens.numel()))
            _into_graph_schedule(key, get_scheduler_metadata(cache_seqlens=lens, **problem))


def _into_graph_schedule(key: tuple, schedule: torch.Tensor) -> torch.Tensor:
    """schedule copied into the buffer full graphs read, its tail zeroed as vLLM's, since a stale tail misdirects
    thread blocks. One buffer per layer shape, sized by the first capture, which is the largest batch."""
    buffers = FlashAttention3Backend._graph_schedules
    shape = key[:-1]
    if shape not in buffers:
        buffers[shape] = torch.zeros_like(schedule)
    buffer, n = buffers[shape], schedule.numel()
    assert n <= buffer.numel(), "full graphs must capture their largest batch size first"
    buffer[:n] = schedule
    buffer[n:] = 0
    return buffer[:n]
