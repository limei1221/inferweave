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
    keys_are_new: bool = False    # no row carries cached keys, so k/v holds the whole batch
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None    # rows that sample; None means all of them
    cu_seqlens_q_host: list[int] | None = None    # cu_seqlens_q and _k kept on the host, so MLA plans without a sync
    cu_seqlens_k_host: list[int] | None = None
    context_chunks: list | None = None    # filled on first use by layers.attention.context_chunks
    mla_decode_metadata: object | None = None    # FlashMLA's schedule holder, made by the step's first layer
    prefill_rows: list[bool] | None = None    # explicit request phases; a one-token prefill is not decode
    mla_partitions: list | None = None    # (token indices, Context), shared across layers of a mixed step
    piecewise_size: int | None = None    # the bucket whose piecewise graphs this pass captures or replays
    num_actual_tokens: int | None = None    # the step's own rows, when a piecewise bucket pads it; None is all

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
