from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None  # rows that sample; None means all of them
    cu_seqlens_q_host: list[int] | None = None  # cu_seqlens_q and _k kept on the host, so MLA plans without a sync
    cu_seqlens_k_host: list[int] | None = None
    context_chunks: list | None = None  # filled on first use by layers.attention.context_chunks
    mla_decode_metadata: object | None = None  # FlashMLA's schedule holder, made by the step's first layer
    decode_split: tuple | None = None  # filled on first use by split_decodes_and_prefills
    attn_metadata: dict | None = None  # what a backend plans on the step's first layer, e.g. FlashInfer's wrappers
    piecewise_size: int | None = None  # the bucket whose piecewise graphs this pass captures or replays
    full_graph_size: int | None = None  # the batch size whose full graph this pass captures
    num_actual_tokens: int | None = None  # the step's own rows, when a piecewise bucket pads it; None is all


def split_decodes_and_prefills(context: Context) -> tuple[int, Context | None, Context | None]:
    """A prefill step's leading one-query rows, as (their count, a decode context, a context for the rest).

    Either context is None when it has no rows. The runner puts one-query rows first, so this is a slice, built on
    the host once per step and shared by every layer, as vLLM's split_decodes_and_prefills. A row decodes by its
    shape, so a one-token prompt chunk does too: attention cannot tell it from a decode.
    """
    if context.block_tables is None or context.cu_seqlens_q_host is None:
        return 0, None, context  # nothing cached to decode against, or no host lengths to split by
    if context.decode_split is None:
        cu_q, cu_k = context.cu_seqlens_q_host, context.cu_seqlens_k_host
        assert cu_k is not None and context.cu_seqlens_q is not None and context.cu_seqlens_k is not None
        assert context.context_lens is not None
        num_rows = len(cu_q) - 1
        n = next((i for i in range(num_rows) if cu_q[i + 1] - cu_q[i] > 1), num_rows)
        decodes = prefills = None
        if n:
            decodes = Context(
                cu_seqlens_q=context.cu_seqlens_q[: n + 1],
                cu_seqlens_k=context.cu_seqlens_k[: n + 1],
                cu_seqlens_q_host=cu_q[: n + 1],
                cu_seqlens_k_host=cu_k[: n + 1],
                max_seqlen_q=1,
                max_seqlen_k=max(cu_k[i + 1] - cu_k[i] for i in range(n)),
                context_lens=context.context_lens[:n],
                block_tables=context.block_tables[:n],
            )
        if not n:
            prefills = context
        elif n < num_rows:
            sub_q, sub_k = [x - cu_q[n] for x in cu_q[n:]], [x - cu_k[n] for x in cu_k[n:]]
            prefills = Context(
                is_prefill=True,
                cu_seqlens_q=context.cu_seqlens_q[n:] - cu_q[n],
                cu_seqlens_k=context.cu_seqlens_k[n:] - cu_k[n],
                cu_seqlens_q_host=sub_q,
                cu_seqlens_k_host=sub_k,
                max_seqlen_q=max(b - a for a, b in zip(sub_q, sub_q[1:])),
                max_seqlen_k=max(b - a for a, b in zip(sub_k, sub_k[1:])),
                context_lens=context.context_lens[n:],
                block_tables=context.block_tables[n:],
            )
        context.decode_split = (n, decodes, prefills)
    return context.decode_split


_CONTEXT = Context()


def get_context():
    return _CONTEXT


@contextmanager
def set_context(is_prefill: bool, **kwargs):
    """The context for one forward pass; the previous one comes back on exit, even on error."""
    global _CONTEXT
    previous, _CONTEXT = _CONTEXT, Context(is_prefill, **kwargs)
    try:
        yield _CONTEXT
    finally:
        _CONTEXT = previous


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
