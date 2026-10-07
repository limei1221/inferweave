"""Backends checked against dense_attention, an independent oracle using no SDPA or paging."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lean_vllm.attention import (
    BACKENDS,
    FlashAttention3Backend,
    FlashInferBackend,
    FlashInferMLABackend,
    FlashMLABackend,
    LayerSpec,
    TorchAttention,
    TritonMLABackend,
    flash_backend,
    flashinfer_backend,
    get_attention_backend,
    triton_merge,
)
from lean_vllm.attention.merge import merge_attention, merge_attention_
from lean_vllm.layers.attention import Attention
from lean_vllm.utils.context import Context, set_context, split_decodes_and_prefills

torch.manual_seed(0)

NUM_HEADS = 8
NUM_KV_HEADS = 2  # exercises GQA head broadcasting
HEAD_DIM = 64  # a size every backend takes; FlashInfer's are 64, 128 and 256
SCALE = 0.137  # not head_dim**-0.5, so a dropped scale argument is detectable

# The engine's default page size, which every backend takes.
BLOCK_SIZE = 16
# Kernels are fp16/bf16 only.
DTYPE = {
    "torch": torch.float32,
    "flash_attn_3": torch.float16,
    "flashinfer": torch.float16,
    "flashmla": torch.float16,
    "triton_mla": torch.float16,
    "flashinfer_mla": torch.float16,
}

# Tolerances against the fp32 oracle; bf16 uses test_low_precision_no_worse_than_naive.
TOLERANCE = {torch.float32: 2e-3, torch.float16: 6e-3}


def _cases():
    """Every (backend, device) pair runnable here that serves a plain layer of the tests' shape."""
    cases = []
    for backend_cls in BACKENDS:
        spec = LayerSpec(HEAD_DIM, NUM_HEADS, NUM_KV_HEADS, DTYPE[backend_cls.get_name()])
        if not backend_cls.is_available() or backend_cls.validate(spec):
            continue
        if backend_cls.get_name() == "torch":
            devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        else:
            devices = ["cuda"]
        cases += [(backend_cls, d) for d in devices]
    return cases


CASES = _cases()


@pytest.fixture(params=CASES, ids=[f"{b.get_name()}-{d}" for b, d in CASES])
def case(request):
    backend_cls, device = request.param
    return backend_cls(NUM_HEADS, HEAD_DIM, SCALE, NUM_KV_HEADS), torch.device(device)


@pytest.fixture
def backend(case):
    return case[0]


@pytest.fixture
def device(case):
    return case[1]


@pytest.fixture
def block_size():
    return BLOCK_SIZE


@pytest.fixture
def dtype(backend):
    """Backend-specific dtype. Tests parametrizing `dtype` shadow this fixture."""
    return DTYPE[backend.get_name()]


@pytest.fixture
def tol(dtype):
    assert dtype in TOLERANCE, f"no fixed tolerance for {dtype}; check against naive arithmetic instead"
    return TOLERANCE[dtype]


def dense_scores(q, k, scale=SCALE, compute_dtype=torch.float32, causal=True):
    """Attention logits one head at a time, [H, lq, lk]. q is [lq, H, D], k [lk, Hkv, D] full sequence."""
    lq, num_heads, _ = q.shape
    lk, num_kv_heads, _ = k.shape
    k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)

    scores = torch.empty(num_heads, lq, lk, dtype=compute_dtype, device=q.device)
    for h in range(num_heads):
        scores[h] = (q[:, h, :].to(compute_dtype) @ k[:, h, :].to(compute_dtype).T) * scale
        for j in range(lq if causal else 0):
            scores[h, j, lk - lq + j + 1 :] = float("-inf")
    return scores


def dense_attention(q, k, v, scale=SCALE, compute_dtype=torch.float32, causal=True):
    """Attention one head at a time. q is [lq, H, D], k/v [lk, Hkv, D] full sequence."""
    num_heads = q.size(1)
    v = v.repeat_interleave(num_heads // v.size(1), dim=1)
    scores = dense_scores(q, k, scale, compute_dtype, causal)

    out = torch.empty_like(q)
    for h in range(num_heads):
        out[:, h, :] = (scores[h].softmax(dim=-1) @ v[:, h, :].to(compute_dtype)).to(q.dtype)
    return out


def make_cache(num_blocks, device, block_size, dtype):
    shape = (num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM)
    return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)


def slots_for(block_table, block_size, start, end):
    """Flat cache slots for token positions [start, end) of one sequence."""
    return [block_table[p // block_size] * block_size + p % block_size for p in range(start, end)]


def write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, num_cached):
    """Seed the cache as if computed on an earlier step."""
    # Explicit long dtype to avoid float indices from empty prefixes
    slots = torch.tensor(slots_for(block_table, block_size, 0, num_cached), dtype=torch.long, device=k_cache.device)
    k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots] = k_full[:num_cached]
    v_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots] = v_full[:num_cached]


def randn(*shape, device, dtype):
    """Generate random tensors in fp32 then cast to target dtype for reproducibility."""
    return torch.randn(*shape, device=device).to(dtype)


@pytest.mark.parametrize("mode", ["decode", "prefill", "mla_decode"])
@pytest.mark.parametrize("poison", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_torch_attention_ignores_unwritten_cache_slots(mode, poison):
    backend = TorchAttention(2, 4, 0.5, 1)
    # The second row forces padding in prefill too. Neither unused slots nor
    # block-table padding may contribute to the first row's attention.
    cache = torch.full((3, 2, 1, 4), poison)
    cache[1, 0] = 1
    cache[2] = 1
    cache[0, 0] = 1
    context = Context(
        is_prefill=mode == "prefill",
        block_tables=torch.tensor([[1, -1], [2, 0]], dtype=torch.int32),
        context_lens=torch.tensor([1, 3], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 1, 4], dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_k=3,
    )
    q = torch.ones(2, 2, 4)
    if mode == "mla_decode":
        out = backend.mla_decode(q, cache.squeeze(2), 2, context)
    elif mode == "prefill":
        out = backend.prefill(q, torch.empty(0), torch.empty(0), cache, cache, context)
    else:
        out = backend.decode(q, cache, cache, context)
    torch.testing.assert_close(out, torch.ones_like(out))


def test_prefill_without_cache(backend, device, dtype, tol):
    """Varlen causal prefill with no cached tokens."""
    seqlens = [5, 1, 12]  # no paging on this path
    qs = [randn(n, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]
    ks = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]
    vs = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]

    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    context = Context(
        is_prefill=True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(seqlens),
        max_seqlen_k=max(seqlens),
    )
    empty = torch.tensor([], device=device, dtype=dtype)
    out = backend.prefill(torch.cat(qs), torch.cat(ks), torch.cat(vs), empty, empty, context)

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(qs, ks, vs)])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "unmasked"])
def test_varlen_with_lse(backend, device, dtype, tol, causal):
    """Uncached attention and its log-sum-exp, by which MLA merges chunks of cached keys."""
    seqlens_q = [5, 1, 12]
    seqlens_k = seqlens_q if causal else [7, 4, 16]  # unmasked chunks hold more keys than queries
    qs = [randn(n, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens_q]
    ks = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens_k]
    vs = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens_k]

    def cumulative(seqlens):
        return torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)

    out, lse = backend.varlen_with_lse(
        torch.cat(qs),
        torch.cat(ks),
        torch.cat(vs),
        cumulative(seqlens_q),
        cumulative(seqlens_k),
        max(seqlens_q),
        max(seqlens_k),
        causal,
    )

    expected = torch.cat([dense_attention(q, k, v, causal=causal) for q, k, v in zip(qs, ks, vs)])
    expected_lse = torch.cat([dense_scores(q, k, causal=causal).logsumexp(-1).T for q, k in zip(qs, ks)])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(lse, expected_lse, atol=tol, rtol=tol)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "unmasked"])
def test_narrow_values_match_padded_ones(backend, device, dtype, tol, causal):
    """MLA prefill's shape, 192-wide keys and 128-wide values, taken as they are, as MLAAttention hands them over.

    Padded values are what every backend is checked on above, so they are the reference.
    """
    qk_dim, v_dim = 192, 128
    mla = type(backend)(NUM_HEADS, qk_dim, SCALE, NUM_HEADS)  # expanded MLA: a key head per query head
    seqlens_q = [5, 1, 12]
    seqlens_k = seqlens_q if causal else [7, 4, 16]
    q = randn(sum(seqlens_q), NUM_HEADS, qk_dim, device=device, dtype=dtype)
    k = randn(sum(seqlens_k), NUM_HEADS, qk_dim, device=device, dtype=dtype)
    # A view into a wider tensor, as MLA's split of kv_b_proj's output hands over.
    v = randn(sum(seqlens_k), NUM_HEADS, qk_dim, device=device, dtype=dtype)[..., :v_dim]
    padded_v = torch.nn.functional.pad(v, (0, qk_dim - v_dim))

    def cumulative(seqlens):
        return torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)

    args = (cumulative(seqlens_q), cumulative(seqlens_k), max(seqlens_q), max(seqlens_k), causal)
    out, lse = mla.varlen_with_lse(q, k, v, *args)
    want, want_lse = mla.varlen_with_lse(q, k, padded_v, *args)
    assert out.shape[-1] == v_dim
    torch.testing.assert_close(out, want[..., :v_dim], atol=tol, rtol=tol)
    torch.testing.assert_close(lse, want_lse, atol=tol, rtol=tol)
    if causal:
        context = Context(
            is_prefill=True,
            cu_seqlens_q=args[0],
            cu_seqlens_k=args[1],
            max_seqlen_q=args[2],
            max_seqlen_k=args[3],
        )
        empty = torch.tensor([], device=device, dtype=dtype)
        got = mla.prefill(q, k, v, empty, empty, context)
        torch.testing.assert_close(
            got, mla.prefill(q, k, padded_v, empty, empty, context)[..., :v_dim], atol=tol, rtol=tol
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton_merge._IMPORT_ERROR is not None,
    reason="the Triton merge needs a CUDA device and a Triton build",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("into", [False, True], ids=["in_place", "into_out"])
def test_the_triton_merge_matches_torch(dtype, into):
    """In place or into an output strided as a split's rows, with lse in FA3's transposed layout."""
    num_tokens, num_heads, head_dim = 37, 16, 128
    o_a = torch.randn(num_tokens, num_heads, head_dim, device="cuda", dtype=dtype)
    o_b = torch.randn(num_tokens, num_heads, head_dim, device="cuda", dtype=dtype)
    lse_a = (torch.randn(num_heads, num_tokens, device="cuda") * 3).T
    lse_b = (torch.randn(num_heads, num_tokens, device="cuda") * 3).T
    lse_a[0] = float("-inf")  # a row with no cached keys
    want_o, want_lse = merge_attention(o_a, lse_a, o_b, lse_b)
    before = o_a.clone()
    out = torch.empty(num_tokens + 3, num_heads, head_dim, device="cuda", dtype=dtype)[3:] if into else None
    merge_attention_(o_a, lse_a, o_b, lse_b, out)
    torch.testing.assert_close(o_a if out is None else out, want_o, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_a, want_lse)
    if into:
        assert torch.equal(o_a, before)


def test_prefill_of_a_cold_batch_with_pages(backend, device, block_size, dtype, tol):
    """A fresh prompt while serving: pages allocated but empty, so k/v and the pages must agree."""
    seqlens = [5, block_size + 3]
    block_tables_list = [[0, 1, -1], [2, 3, 4]]
    k_cache, v_cache = make_cache(6, device, block_size, dtype)

    qs = [randn(n, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]
    ks = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]
    vs = [randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) for n in seqlens]

    slot_mapping = []
    for table, n in zip(block_tables_list, seqlens):
        slot_mapping += slots_for(table, block_size, 0, n)
    k_new, v_new = torch.cat(ks), torch.cat(vs)
    backend.store_kvcache(
        k_new,
        v_new,
        k_cache,
        v_cache,
        torch.tensor(slot_mapping, dtype=torch.int32, device=device),
    )

    cu = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    context = Context(
        is_prefill=True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(seqlens),
        max_seqlen_k=max(seqlens),
        context_lens=torch.tensor(seqlens, dtype=torch.int32, device=device),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )
    out = backend.prefill(torch.cat(qs), k_new, v_new, k_cache, v_cache, context)

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(qs, ks, vs)])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


def test_prefill_with_prefix_cache(backend, device, block_size, dtype, tol):
    """Chunked prefill: seq 0 resumes mid-page after cached prefix, seq 1 starts cold."""
    num_cached = [2 * block_size + 2, 0]
    num_new = [6, 5]
    block_tables_list = [[0, 1, 2, 3, 4, 5], [6, 7, -1, -1, -1, -1]]
    k_cache, v_cache = make_cache(8, device, block_size, dtype)

    q_list, k_full, v_full = [], [], []
    slot_mapping = []
    for i, (cached, new) in enumerate(zip(num_cached, num_new)):
        total = cached + new
        k_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        v_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        q_list.append(randn(new, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype))
        write_prefix(k_cache, v_cache, k_full[i], v_full[i], block_tables_list[i], block_size, cached)
        slot_mapping += slots_for(block_tables_list[i], block_size, cached, total)

    # only new tokens reach the layer
    k_new = torch.cat([k[c:] for k, c in zip(k_full, num_cached)])
    v_new = torch.cat([v[c:] for v, c in zip(v_full, num_cached)])
    backend.store_kvcache(
        k_new,
        v_new,
        k_cache,
        v_cache,
        torch.tensor(slot_mapping, dtype=torch.int32, device=device),
    )

    totals = [c + n for c, n in zip(num_cached, num_new)]
    context = Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, *torch.tensor(num_new).cumsum(0).tolist()], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, *torch.tensor(totals).cumsum(0).tolist()], dtype=torch.int32, device=device),
        max_seqlen_q=max(num_new),
        max_seqlen_k=max(totals),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )
    out = backend.prefill(torch.cat(q_list), k_new, v_new, k_cache, v_cache, context)

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(q_list, k_full, v_full)])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


# The only shapes FlashMLA's dense decode takes: a 512 + 64 latent, 512 of it the value, 64-token pages.
LATENT_DIM, LATENT_V_DIM, MLA_BLOCK_SIZE = 576, 512, 64
MLA_CASES = [b for b in BACKENDS if b.supports_mla_decode() and b.is_available()]
# Each backend at the page sizes it reads: FlashMLA's one, or the engine's default and FlashMLA's for the rest.
MLA_PAGE_CASES = [(b, size) for b in MLA_CASES for size in ([b.mla_block_size()] if b.mla_block_size() else [16, 64])]


@pytest.mark.parametrize(
    "backend_cls, block_size", MLA_PAGE_CASES, ids=[f"{b.get_name()}-{size}" for b, size in MLA_PAGE_CASES]
)
def test_mla_decode(backend_cls, block_size):
    """Each row's query attends its cached latents as one shared key head, valued by their first v_dim entries."""
    context_lens = [block_size + 3, 3, 2 * block_size]  # mid-page, part-page, full
    check_mla_decode(backend_cls, block_size, context_lens, [[2, 0], [3, -1], [1, 4]])  # scattered


@pytest.mark.parametrize(
    "backend_cls, block_size", MLA_PAGE_CASES, ids=[f"{b.get_name()}-{size}" for b, size in MLA_PAGE_CASES]
)
def test_mla_decode_of_long_rows(backend_cls, block_size):
    """Rows long enough that split-KV kernels split them, each part over many pages, beside a one-key row."""
    context_lens = [21 * block_size - 5, 1, 9 * block_size + 1]
    pages = torch.randperm(32).tolist()
    check_mla_decode(
        backend_cls, block_size, context_lens, [pages[:21], pages[21:22] + [-1] * 20, pages[22:] + [-1] * 11]
    )


def check_mla_decode(backend_cls, block_size, context_lens, block_tables_list):
    name = backend_cls.get_name()
    device, dtype = torch.device("cpu" if name == "torch" else "cuda"), DTYPE[name]
    backend = backend_cls(NUM_HEADS, LATENT_DIM, SCALE, NUM_HEADS)
    num_blocks = max(max(table) for table in block_tables_list) + 1
    cache = torch.zeros(num_blocks, block_size, LATENT_DIM, device=device, dtype=dtype)
    latents = [randn(n, LATENT_DIM, device=device, dtype=dtype) for n in context_lens]
    for latent, table in zip(latents, block_tables_list):
        slots = torch.tensor(slots_for(table, block_size, 0, latent.size(0)), device=device)
        cache.view(-1, LATENT_DIM)[slots] = latent

    q = randn(len(context_lens), NUM_HEADS, LATENT_DIM, device=device, dtype=dtype)
    context = Context(
        context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )
    out = backend.mla_decode(q, cache, LATENT_V_DIM, context)

    expected = torch.cat(
        [
            (dense_scores(q[i : i + 1], latent.unsqueeze(1)).softmax(-1) @ latent[:, :LATENT_V_DIM].float()).transpose(
                0, 1
            )
            for i, latent in enumerate(latents)
        ]
    ).to(dtype)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


@pytest.mark.skipif(not TritonMLABackend.is_available(), reason="needs Triton and a GPU of sm80 or newer")
def test_triton_mla_decode_writes_zeros_for_a_row_with_no_keys():
    """A full graph's padding rows have no keys; they must come out zero, not 0 / 0, beside a real row."""
    from lean_vllm.attention import triton_mla_decode

    cache = randn(2, BLOCK_SIZE, LATENT_DIM, device="cuda", dtype=torch.float16)
    q = randn(2, NUM_HEADS, LATENT_DIM, device="cuda", dtype=torch.float16)
    block_tables = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([5, 0], dtype=torch.int32, device="cuda")
    out = triton_mla_decode.mla_decode(q, cache, block_tables, context_lens, LATENT_V_DIM, SCALE)
    assert out[0].isfinite().all() and (out[1] == 0).all()


@pytest.mark.parametrize("backend_cls", MLA_CASES, ids=[b.get_name() for b in MLA_CASES])
def test_store_latents_skips_negative_slots(backend_cls):
    """A full graph pads its batch, so slot -1 must leave that cache row untouched."""
    name = backend_cls.get_name()
    device, dtype = torch.device("cpu" if name == "torch" else "cuda"), DTYPE[name]
    backend = backend_cls(NUM_HEADS, LATENT_DIM, SCALE, NUM_HEADS)
    cache = torch.full((2, MLA_BLOCK_SIZE, LATENT_DIM), 7.0, device=device, dtype=dtype)
    latent = randn(3, LATENT_DIM, device=device, dtype=dtype)
    written = [0, MLA_BLOCK_SIZE + 5]

    backend.store_latents(latent, cache, torch.tensor([written[0], -1, written[1]], dtype=torch.int32, device=device))

    flat = cache.flatten(0, 1)
    torch.testing.assert_close(flat[written[0]], latent[0], atol=0, rtol=0)
    torch.testing.assert_close(flat[written[1]], latent[2], atol=0, rtol=0)
    untouched = [i for i in range(2 * MLA_BLOCK_SIZE) if i not in written]
    assert (flat[untouched] == 7.0).all(), "untouched slots were overwritten"


# What a layer asks for, by its kind: DeepSeek-V2-Lite's MLA and a Qwen3-8B-like plain layer.
MLA_SPEC = LayerSpec(192, 16, 16, torch.bfloat16, latent_dim=576)
PLAIN_SPEC = LayerSpec(128, 32, 8, torch.bfloat16)


@pytest.fixture
def all_available(monkeypatch):
    """Every backend reports itself available, so selection runs on any machine; nothing is executed."""
    monkeypatch.delenv("LEAN_VLLM_ATTENTION_BACKEND", raising=False)
    for backend in BACKENDS:
        monkeypatch.setattr(backend, "is_available", staticmethod(lambda: True))


def test_flashmla_is_preferred_for_mla_layers_only(all_available):
    assert get_attention_backend(MLA_SPEC) is FlashMLABackend
    assert get_attention_backend(PLAIN_SPEC) is FlashAttention3Backend


def test_each_layer_takes_the_first_backend_that_serves_it(all_available, monkeypatch):
    """By head size, dtype and kind, in priority order; torch takes whatever no kernel does."""
    assert get_attention_backend(LayerSpec(128, 32, 8, torch.float32)) is TorchAttention  # dtype
    assert get_attention_backend(LayerSpec(512, 8, 8, torch.bfloat16)) is TorchAttention  # head size
    assert get_attention_backend(LayerSpec(192, 16, 16, torch.bfloat16, latent_dim=512)) is FlashAttention3Backend
    for hopper_only in (FlashAttention3Backend, FlashMLABackend):  # an A100
        monkeypatch.setattr(hopper_only, "is_available", staticmethod(lambda: False))
    assert get_attention_backend(PLAIN_SPEC) is FlashInferBackend
    assert get_attention_backend(LayerSpec(96, 32, 8, torch.bfloat16)) is TorchAttention  # not 64, 128 or 256
    assert get_attention_backend(MLA_SPEC) is TritonMLABackend  # prefilling on FlashInfer


@pytest.mark.parametrize(
    "spec, reason",
    [
        (LayerSpec(128, 6, 4, torch.bfloat16), "6 query heads do not group over 4 key heads"),
        (MLA_SPEC, "mla layers are not supported"),
        (LayerSpec(128, 32, 8, torch.float32), "dtype torch.float32"),
    ],
    ids=["head_count", "kind", "dtype"],
)
def test_a_named_backend_that_cannot_serve_the_layer_says_why(all_available, monkeypatch, spec, reason):
    """Rather than fall back: a silent switch to a slower backend mid-benchmark is worse than a crash."""
    monkeypatch.setenv("LEAN_VLLM_ATTENTION_BACKEND", "flashinfer")
    with pytest.raises(RuntimeError, match=reason):
        get_attention_backend(spec)


def test_layers_of_one_model_can_take_different_backends(all_available):
    """Chosen at init from each layer's own shape, under the dtype the runner sets while building the model."""
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        wide, odd = Attention(32, 128, 0.1, 8), Attention(32, 96, 0.1, 8)
        torch.set_default_dtype(torch.float32)
        full = Attention(32, 128, 0.1, 8)
    finally:
        torch.set_default_dtype(default)
    assert [type(layer.backend) for layer in (wide, odd, full)] == [
        FlashAttention3Backend,
        FlashAttention3Backend,
        TorchAttention,
    ]


def test_mla_layers_take_triton_mla_where_flashmla_is_missing(all_available, monkeypatch):
    """As vLLM's MLA lists: FlashMLA, then Triton MLA. FlashInfer MLA serves when named; plain layers pass both by."""
    monkeypatch.setattr(FlashMLABackend, "is_available", staticmethod(lambda: False))
    assert get_attention_backend(MLA_SPEC) is TritonMLABackend
    assert get_attention_backend(MLA_SPEC, "flashinfer_mla") is FlashInferMLABackend
    assert get_attention_backend(PLAIN_SPEC) is FlashAttention3Backend


def test_mla_decode_backends_prefill_on_fa3_else_flashinfer(all_available, monkeypatch):
    from lean_vllm.attention import mla_common

    assert mla_common.prefill_backend() is FlashAttention3Backend
    monkeypatch.setattr(FlashAttention3Backend, "is_available", staticmethod(lambda: False))
    assert mla_common.prefill_backend() is FlashInferBackend
    # FlashInfer's prefill takes MLA's 192-wide keys and 128-wide values, though its decode takes neither.
    assert TritonMLABackend.validate(MLA_SPEC) == []
    monkeypatch.setattr(FlashInferBackend, "is_available", staticmethod(lambda: False))
    assert "neither FlashAttention-3 nor FlashInfer" in "; ".join(TritonMLABackend.validate(MLA_SPEC))


def test_decode(backend, device, block_size, dtype, tol):
    """One query per sequence against differing cached context lengths."""
    context_lens = [block_size + 3, 3, 4 * block_size]  # mid-page, part-page, full
    block_tables_list = [[0, 1, 2, 3], [4, 5, -1, -1], [6, 7, 8, 9]]
    k_cache, v_cache = make_cache(10, device, block_size, dtype)

    k_full, v_full = [], []
    for i, n in enumerate(context_lens):
        k_full.append(randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        v_full.append(randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        write_prefix(k_cache, v_cache, k_full[i], v_full[i], block_tables_list[i], block_size, n)

    q = randn(len(context_lens), NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    context = Context(
        is_prefill=False,
        context_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )
    out = backend.decode(q, k_cache, v_cache, context)

    expected = torch.cat([dense_attention(q[i : i + 1], k_full[i], v_full[i]) for i in range(len(context_lens))])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


def test_store_kvcache_skips_negative_slots(backend, device, block_size, dtype):
    """Slot -1 must leave that cache row untouched."""
    k_cache, v_cache = make_cache(2, device, block_size, dtype)
    k_cache.fill_(7.0)
    v_cache.fill_(7.0)

    key = randn(3, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    value = randn(3, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    slot_mapping = torch.tensor([0, -1, 5], dtype=torch.int32, device=device)
    backend.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    flat_k = k_cache.flatten(0, 1)
    torch.testing.assert_close(flat_k[0], key[0])
    torch.testing.assert_close(flat_k[5], key[2])
    assert (flat_k[1:5] == 7.0).all(), "untouched slots were overwritten"


def test_decode_matches_equivalent_prefill(backend, device, block_size, dtype, tol):
    """Decode must equal a 1-token prefill over the same context."""
    seqlen = 2 * block_size + 1
    block_table = [0, 1, 2]
    k_cache, v_cache = make_cache(3, device, block_size, dtype)
    k_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v_full = randn(seqlen, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, seqlen)

    q = randn(1, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    bt = torch.tensor([block_table], dtype=torch.int32, device=device)

    decoded = backend.decode(
        q,
        k_cache,
        v_cache,
        Context(
            is_prefill=False,
            context_lens=torch.tensor([seqlen], dtype=torch.int32, device=device),
            block_tables=bt,
        ),
    )
    prefilled = backend.prefill(
        q,
        k_full[-1:],
        v_full[-1:],
        k_cache,
        v_cache,
        Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
            max_seqlen_q=1,
            max_seqlen_k=seqlen,
            block_tables=bt,
        ),
    )
    torch.testing.assert_close(decoded, prefilled, atol=tol, rtol=tol)


def test_top_left_causal_alignment_would_be_wrong(backend, device, block_size, dtype, tol):
    """With lq < lk a top-left mask differs; fails if the test stops discriminating."""
    lq, lk = 4, 2 * block_size + 2
    block_table = [0, 1, 2]
    k_cache, v_cache = make_cache(3, device, block_size, dtype)
    k_full = randn(lk, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v_full = randn(lk, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, lk)

    q = randn(lq, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    out = backend.prefill(
        q,
        k_full[-lq:],
        v_full[-lq:],
        k_cache,
        v_cache,
        Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, lq], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, lk], dtype=torch.int32, device=device),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            block_tables=torch.tensor([block_table], dtype=torch.int32, device=device),
        ),
    )

    torch.testing.assert_close(out, dense_attention(q, k_full, v_full), atol=tol, rtol=tol)

    top_left = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1),
        k_full.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1).transpose(0, 1),
        v_full.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1).transpose(0, 1),
        is_causal=True,
        scale=SCALE,
    )
    top_left = top_left.transpose(0, 1)
    # well clear of the noise floor the assert_close above already allows
    assert not torch.allclose(out, top_left, atol=5 * tol), (
        "top-left and bottom-right masks agree; test is not discriminating"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_low_precision_no_worse_than_naive(backend, device, block_size, dtype):
    """Backend error against the fp32 oracle must not exceed naive arithmetic's in the same dtype."""
    num_cached, num_new = 2 * block_size + 2, 6
    total = num_cached + num_new
    block_table = [0, 1, 2, 3]
    k_cache, v_cache = make_cache(4, device, block_size, dtype)

    k_full = randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v_full = randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    q = randn(num_new, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)

    write_prefix(k_cache, v_cache, k_full, v_full, block_table, block_size, num_cached)
    backend.store_kvcache(
        k_full[num_cached:],
        v_full[num_cached:],
        k_cache,
        v_cache,
        torch.tensor(slots_for(block_table, block_size, num_cached, total), dtype=torch.int32, device=device),
    )
    out = backend.prefill(
        q,
        k_full[num_cached:],
        v_full[num_cached:],
        k_cache,
        v_cache,
        Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, num_new], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, total], dtype=torch.int32, device=device),
            max_seqlen_q=num_new,
            max_seqlen_k=total,
            block_tables=torch.tensor([block_table], dtype=torch.int32, device=device),
        ),
    )

    # identical inputs, so the only difference is the precision of the arithmetic
    ref = dense_attention(q.float(), k_full.float(), v_full.float())
    naive = dense_attention(q, k_full, v_full, compute_dtype=dtype).float()

    backend_err = (out.float() - ref).abs().max().item()
    naive_err = (naive - ref).abs().max().item()
    print(
        f"\n{dtype} backend_err={backend_err:.3e} naive_err={naive_err:.3e} "
        f"ratio={backend_err / max(naive_err, 1e-12):.2f}"
    )
    assert backend_err <= 2 * naive_err + 1e-6, (
        f"backend error {backend_err:.3e} exceeds twice naive {dtype} error {naive_err:.3e}"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_low_precision_cache_roundtrip_is_exact(backend, device, block_size, dtype):
    """store_kvcache must not perturb values; only attention arithmetic may lose precision."""
    block_table = [0, 1]
    k_cache, v_cache = make_cache(2, device, block_size, dtype)
    n = 2 * block_size
    key = randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    value = randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)

    slots = slots_for(block_table, block_size, 0, n)
    backend.store_kvcache(key, value, k_cache, v_cache, torch.tensor(slots, dtype=torch.int32, device=device))

    torch.testing.assert_close(k_cache.flatten(0, 1)[slots], key, atol=0, rtol=0)
    torch.testing.assert_close(v_cache.flatten(0, 1)[slots], value, atol=0, rtol=0)


def _mixed_batch(device, block_size, dtype, num_cached, num_new, block_tables_list, num_blocks):
    """Seed a cache and return (q, k_new, v_new, k_cache, v_cache, k_full, v_full)."""
    k_cache, v_cache = make_cache(num_blocks, device, block_size, dtype)
    q_list, k_full, v_full, slot_mapping = [], [], [], []
    for i, (cached, new) in enumerate(zip(num_cached, num_new)):
        total = cached + new
        k_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        v_full.append(randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype))
        q_list.append(randn(new, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype))
        write_prefix(k_cache, v_cache, k_full[i], v_full[i], block_tables_list[i], block_size, cached)
        slot_mapping += slots_for(block_tables_list[i], block_size, cached, total)
    k_new = torch.cat([k[c:] for k, c in zip(k_full, num_cached)])
    v_new = torch.cat([v[c:] for v, c in zip(v_full, num_cached)])
    return q_list, k_new, v_new, k_cache, v_cache, k_full, v_full, slot_mapping


def _mixed_context(device, num_cached, num_new, block_tables_list):
    totals = [c + n for c, n in zip(num_cached, num_new)]
    return Context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, *torch.tensor(num_new).cumsum(0).tolist()], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, *torch.tensor(totals).cumsum(0).tolist()], dtype=torch.int32, device=device),
        max_seqlen_q=max(num_new),
        max_seqlen_k=max(totals),
        block_tables=torch.tensor(block_tables_list, dtype=torch.int32, device=device),
    )


def test_mixed_batch_of_chunks_and_decodes(backend, device, block_size, dtype, tol):
    """One batch holding a decode row, a resumed chunk and a cold prefill."""
    num_cached = [2 * block_size + 5, block_size + 1, 0]
    num_new = [1, 7, 4]  # decode, resumed chunk, cold
    block_tables_list = [[0, 1, 2], [3, 4, -1], [5, -1, -1]]

    q_list, k_new, v_new, k_cache, v_cache, k_full, v_full, slots = _mixed_batch(
        device, block_size, dtype, num_cached, num_new, block_tables_list, 6
    )
    backend.store_kvcache(k_new, v_new, k_cache, v_cache, torch.tensor(slots, dtype=torch.int32, device=device))
    context = _mixed_context(device, num_cached, num_new, block_tables_list)
    out = backend.prefill(torch.cat(q_list), k_new, v_new, k_cache, v_cache, context)

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(q_list, k_full, v_full)])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)


def test_mixed_batch_matches_running_the_rows_separately(backend, device, block_size, dtype, tol):
    """One mixed call must equal the prefill call plus the decode call it replaces."""
    num_cached = [block_size + 1, 3 * block_size]
    num_new = [6, 1]  # a chunk, then a decode row; order must not matter
    block_tables_list = [[0, 1, -1, -1], [2, 3, 4, 5]]

    q_list, k_new, v_new, k_cache, v_cache, _, _, slots = _mixed_batch(
        device, block_size, dtype, num_cached, num_new, block_tables_list, 6
    )
    backend.store_kvcache(k_new, v_new, k_cache, v_cache, torch.tensor(slots, dtype=torch.int32, device=device))

    merged = backend.prefill(
        torch.cat(q_list),
        k_new,
        v_new,
        k_cache,
        v_cache,
        _mixed_context(device, num_cached, num_new, block_tables_list),
    )

    chunk = backend.prefill(
        q_list[0],
        k_new[: num_new[0]],
        v_new[: num_new[0]],
        k_cache,
        v_cache,
        Context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, num_new[0]], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, num_cached[0] + num_new[0]], dtype=torch.int32, device=device),
            max_seqlen_q=num_new[0],
            max_seqlen_k=num_cached[0] + num_new[0],
            block_tables=torch.tensor([block_tables_list[0]], dtype=torch.int32, device=device),
        ),
    )
    decoded = backend.decode(
        q_list[1],
        k_cache,
        v_cache,
        Context(
            is_prefill=False,
            context_lens=torch.tensor([num_cached[1] + num_new[1]], dtype=torch.int32, device=device),
            block_tables=torch.tensor([block_tables_list[1]], dtype=torch.int32, device=device),
        ),
    )
    torch.testing.assert_close(merged, torch.cat([chunk, decoded]), atol=tol, rtol=tol)


def _host_context(device, num_cached, num_new, block_tables_list):
    """A mixed step as the runner builds it, host lengths included, so a splitting backend can split it."""
    context = _mixed_context(device, num_cached, num_new, block_tables_list)
    context.cu_seqlens_q_host = context.cu_seqlens_q.tolist()
    context.cu_seqlens_k_host = context.cu_seqlens_k.tolist()
    context.context_lens = context.cu_seqlens_k.diff()
    return context


def test_the_split_slices_the_leading_one_query_rows(device):
    """Decode rows lead, as the runner orders them; the rest keep their lengths, offset to start at zero."""
    num_cached, num_new = [9, 0, 4, 2 * BLOCK_SIZE], [1, 1, 5, 3]  # the second row is a one-token prompt
    tables = [[0, -1, -1], [1, -1, -1], [2, -1, -1], [3, 4, 5]]
    context = _host_context(device, num_cached, num_new, tables)

    n, decodes, prefills = split_decodes_and_prefills(context)

    assert n == 2
    assert decodes.context_lens.tolist() == [10, 1]
    assert decodes.block_tables.tolist() == [[0, -1, -1], [1, -1, -1]]
    assert prefills.cu_seqlens_q.tolist() == prefills.cu_seqlens_q_host == [0, 5, 8]
    assert prefills.cu_seqlens_k.tolist() == prefills.cu_seqlens_k_host == [0, 9, 12 + 2 * BLOCK_SIZE]
    assert (prefills.max_seqlen_q, prefills.max_seqlen_k) == (5, 3 + 2 * BLOCK_SIZE)
    assert prefills.block_tables.tolist() == [[2, -1, -1], [3, 4, 5]]
    assert split_decodes_and_prefills(context)[1] is decodes  # built once, for every layer


def test_a_step_split_at_no_row_is_left_whole(device):
    context = _host_context(device, [0, 3], [4, 1], [[0], [1]])  # the one-query row trails the prompt
    assert split_decodes_and_prefills(context) == (0, None, context)


def test_forward_splits_a_mixed_step_into_decode_and_prefill(backend, device, block_size, dtype, tol, monkeypatch):
    """Rows of one query run through decode when the backend splits, each half writing into the step's output;
    together they match the reference."""
    num_cached = [2 * block_size + 5, 3, block_size + 1, 0]
    num_new = [1, 1, 7, 4]  # two decode rows lead, then a resumed chunk and a cold prompt
    block_tables_list = [[0, 1, 2], [3, -1, -1], [4, 5, -1], [6, -1, -1]]
    q_list, k_new, v_new, k_cache, v_cache, k_full, v_full, slots = _mixed_batch(
        device, block_size, dtype, num_cached, num_new, block_tables_list, 7
    )
    backend.store_kvcache(k_new, v_new, k_cache, v_cache, torch.tensor(slots, dtype=torch.int32, device=device))
    decoded, outs = [], []
    decode, prefill = backend.decode, backend.prefill

    def record_decode(q, *args, out=None):
        decoded.append(q.size(0))
        outs.append(out)
        return decode(q, *args, out=out)

    monkeypatch.setattr(backend, "decode", record_decode)
    monkeypatch.setattr(backend, "prefill", lambda *args, out=None: outs.append(out) or prefill(*args, out=out))

    out = backend.forward(
        torch.cat(q_list), k_new, v_new, k_cache, v_cache, _host_context(device, num_cached, num_new, block_tables_list)
    )

    expected = torch.cat([dense_attention(q, k, v) for q, k, v in zip(q_list, k_full, v_full)])
    torch.testing.assert_close(out, expected, atol=tol, rtol=tol)
    assert decoded == ([2] if backend.split_decodes() else [])
    if backend.split_decodes():
        assert all(o._base is out for o in outs)  # views of the step's output, so no copy joins the halves


def test_flashinfer_pages_list_each_rows_used_pages():
    """Its CSR page table: the pages a row's keys fill, not the padded table, and how full the last one is."""
    context = Context(
        cu_seqlens_k_host=[0, 5, 5 + 2 * BLOCK_SIZE, 6 + 2 * BLOCK_SIZE],
        block_tables=torch.tensor([[4, 9, -1], [7, 2, -1], [3, -1, -1]], dtype=torch.int32),
    )
    indptr, indices, last_page_len = FlashInferBackend._pages(context, BLOCK_SIZE)
    assert indptr.tolist() == [0, 1, 3, 4]
    assert indices.tolist() == [4, 7, 2, 3]
    assert last_page_len.tolist() == [5, BLOCK_SIZE, 1]


def test_flashinfer_pages_pad_a_graphs_rows_with_empty_ones():
    context = Context(
        cu_seqlens_k_host=[0, 5, 5 + BLOCK_SIZE], block_tables=torch.tensor([[4, -1], [7, -1]], dtype=torch.int32)
    )
    indptr, indices, last_page_len = FlashInferBackend._pages(context, BLOCK_SIZE, num_rows=4)
    assert indptr.tolist() == [0, 1, 2, 2, 2]
    assert indices.tolist() == [4, 7]
    assert last_page_len.tolist() == [5, BLOCK_SIZE, 1, 1]


class FakeDecodeWrapper:
    """Records what FlashInfer's decode wrapper is built with and planned with, and copies plans into its buffers
    under use_cuda_graph, as the real one does."""

    def __init__(
        self,
        workspace,
        layout,
        use_cuda_graph=False,
        use_tensor_cores=False,
        paged_kv_indptr_buffer=None,
        paged_kv_indices_buffer=None,
        paged_kv_last_page_len_buffer=None,
    ):
        self.use_cuda_graph = use_cuda_graph
        self.buffers = (paged_kv_indptr_buffer, paged_kv_indices_buffer, paged_kv_last_page_len_buffer)
        self.plans = []

    def plan(self, indptr, indices, last_page_len, num_heads, num_kv_heads, head_dim, page_size, **options):
        self.plans.append((indptr.tolist(), indices.tolist(), last_page_len.tolist()))
        if self.use_cuda_graph:
            assert len(last_page_len) == len(self.buffers[2]), "a graph's wrapper plans its own batch size only"
            self.buffers[0].copy_(indptr)
            self.buffers[1][: len(indices)].copy_(indices)
            self.buffers[2].copy_(last_page_len)

    def run(self, q, kv_cache, out=None):
        return torch.zeros_like(q) if out is None else out.zero_()


@pytest.fixture
def fake_flashinfer(monkeypatch):
    from lean_vllm.attention import flashinfer_backend

    monkeypatch.setattr(flashinfer_backend, "BatchDecodeWithPagedKVCacheWrapper", FakeDecodeWrapper, raising=False)
    monkeypatch.setattr(FlashInferBackend, "_wrappers", {})
    monkeypatch.setattr(FlashInferBackend, "_graph_pages", None)
    monkeypatch.setattr(FlashInferBackend, "_workspace", torch.zeros(1, dtype=torch.uint8))
    return FlashInferBackend(NUM_HEADS, HEAD_DIM, SCALE, NUM_KV_HEADS)


def test_flashinfer_full_graphs_get_a_wrapper_per_batch_size_replanned_before_replay(fake_flashinfer):
    """Captured as the runner captures, largest first with worst-case lengths, then re-planned for a real step."""
    backend, width = fake_flashinfer, 3
    k_cache = v_cache = torch.zeros(8, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    for bs in (4, 2):
        context = Context(
            context_lens=torch.full((bs,), width * BLOCK_SIZE, dtype=torch.int32),
            block_tables=torch.zeros(bs, width, dtype=torch.int32),
            full_graph_size=bs,
        )
        backend.decode(torch.zeros(bs, NUM_HEADS, HEAD_DIM), k_cache, v_cache, context)
    graphs = {key[-1]: wrapper for key, wrapper in FlashInferBackend._wrappers.items()}
    assert set(graphs) == {4, 2} and all(wrapper.use_cuda_graph for wrapper in graphs.values())
    indptr, indices, _ = FlashInferBackend._graph_pages
    assert graphs[2].buffers[1] is indices and graphs[2].buffers[0].data_ptr() == indptr.data_ptr()

    step = Context(
        cu_seqlens_k_host=[0, 5, 5 + BLOCK_SIZE, 6 + BLOCK_SIZE],
        block_tables=torch.tensor([[4, -1], [7, 2], [3, -1]], dtype=torch.int32),
    )
    FlashInferBackend.before_full_graph_replay(step, 4)

    assert graphs[4].plans[-1] == ([0, 1, 2, 3, 3], [4, 7, 3], [5, BLOCK_SIZE, 1, 1])
    assert len(graphs[2].plans) == 1  # another graph's wrapper is left alone
    assert indptr[:5].tolist() == [0, 1, 2, 3, 3] and indices[:3].tolist() == [4, 7, 3]

    backend.decode(torch.zeros(3, NUM_HEADS, HEAD_DIM), k_cache, v_cache, step)  # an eager step
    eager = [w for key, w in FlashInferBackend._wrappers.items() if key[-1] is None]
    assert len(eager) == 1 and not eager[0].use_cuda_graph


class FakeRaggedWrapper:
    """Records what FlashInfer's ragged prefill wrapper is planned with; runs to zeros and a base-2 lse of 2."""

    def __init__(self, workspace, layout):
        self.plans = []

    def plan(self, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, causal, **options):
        self.plans.append((qo_indptr.tolist(), kv_indptr.tolist(), head_dim_vo, causal))

    def run(self, q, k, v, return_lse=False, out=None):
        out = q.new_zeros(*q.shape[:2], v.size(-1)) if out is None else out.zero_()
        return (out, torch.full(q.shape[:2], 2.0)) if return_lse else out


@pytest.fixture
def fake_ragged(monkeypatch):
    monkeypatch.setattr(flashinfer_backend, "BatchPrefillWithRaggedKVCacheWrapper", FakeRaggedWrapper, raising=False)
    monkeypatch.setattr(FlashInferBackend, "_ragged", [])
    monkeypatch.setattr(FlashInferBackend, "_workspace", torch.zeros(1, dtype=torch.uint8))
    q = k = torch.zeros(6, NUM_HEADS, 192)
    return FlashInferBackend(NUM_HEADS, 192, SCALE, NUM_HEADS), q, k, torch.zeros(6, NUM_HEADS, 128)


def test_flashinfer_plans_each_ragged_problem_once_a_step(fake_ragged):
    """An MLA step's layers pose the same problems, its new tokens and each chunk of cached keys: each is planned
    once, on a wrapper of its own, and a later step posing one again reuses its plan."""
    backend, q, k, v = fake_ragged
    new, chunk = ([0, 2, 6], [0, 2, 6]), ([0, 6], [0, 5])
    with set_context(True):
        for _ in range(3):  # layers
            out, lse = backend.varlen_with_lse(q, k, v, None, None, 4, 4, True, host_cu_seqlens=new)
            backend.varlen_with_lse(q, k, v, None, None, 6, 5, False, host_cu_seqlens=chunk)
    assert [wrapper.plans for wrapper, _ in FlashInferBackend._ragged] == [
        [([0, 2, 6], [0, 2, 6], 128, True)],
        [([0, 6], [0, 5], 128, False)],
    ]
    assert out.shape == (6, NUM_HEADS, 128)
    torch.testing.assert_close(lse, torch.full((6, NUM_HEADS), 2 * math.log(2)))  # base e

    with set_context(True):
        backend.varlen_with_lse(q, k, v, None, None, 4, 4, True, host_cu_seqlens=new)
    assert len(FlashInferBackend._ragged[0][0].plans) == 1


def test_flashinfer_ragged_problems_past_the_pool_share_its_last_wrapper(fake_ragged, monkeypatch):
    backend, q, k, v = fake_ragged
    monkeypatch.setattr(flashinfer_backend, "MAX_RAGGED_PLANS", 2)
    problems = [([0, n], [0, n]) for n in (1, 2, 3)]
    with set_context(True):
        for _ in range(2):  # layers
            for host in problems:
                backend.varlen_with_lse(q, k, v, None, None, 1, 1, False, host_cu_seqlens=host)
    pool = FlashInferBackend._ragged
    assert len(pool) == 2 and len(pool[0][0].plans) == 1
    assert [plan[0] for plan in pool[1][0].plans] == [[0, 2], [0, 3], [0, 2], [0, 3]]  # re-planned as they alternate


class FakeMLAWrapper:
    """Records what FlashInfer's MLA wrapper is built and planned with, and copies plans into its buffers under
    use_cuda_graph, as the real one does."""

    def __init__(
        self, workspace, use_cuda_graph=False, qo_indptr=None, kv_indptr=None, kv_indices=None, kv_len_arr=None
    ):
        self.use_cuda_graph = use_cuda_graph
        self.buffers = (qo_indptr, kv_indptr, kv_indices, kv_len_arr)
        self.plans = []

    def plan(
        self, *, metadata, num_heads, head_dim_ckv, head_dim_kpe, page_size, causal, sm_scale, q_data_type, kv_data_type
    ):
        assert (head_dim_ckv, head_dim_kpe, causal) == (LATENT_V_DIM, LATENT_DIM - LATENT_V_DIM, False)
        self.plans.append(tuple(tensor.tolist() for tensor in metadata))
        if self.use_cuda_graph:
            assert len(metadata[3]) == len(self.buffers[3]), "a graph's wrapper plans its own batch size only"
            for buffer, tensor in zip(self.buffers, metadata):
                buffer[: len(tensor)].copy_(tensor)

    def run(self, *, query, kv_cache):
        return query.new_zeros(*query.shape[:2], LATENT_V_DIM)


@pytest.fixture
def fake_flashinfer_mla(monkeypatch):
    from lean_vllm.attention import flashinfer_mla_backend, mla_common

    monkeypatch.setattr(flashinfer_mla_backend, "BatchMLAPagedAttentionWrapper", FakeMLAWrapper, raising=False)
    monkeypatch.setattr(
        flashinfer_mla_backend, "MLAPlanMetadata", SimpleNamespace(csr=lambda *tensors: tensors), raising=False
    )
    monkeypatch.setattr(mla_common, "prefill_backend", lambda: TorchAttention)
    monkeypatch.setattr(FlashInferMLABackend, "_wrappers", {})
    monkeypatch.setattr(FlashInferMLABackend, "_graph_metadata", None)
    monkeypatch.setattr(FlashInferBackend, "_workspace", torch.zeros(1, dtype=torch.uint8))
    return FlashInferMLABackend(NUM_HEADS, 192, SCALE, NUM_HEADS)


def test_flashinfer_mla_metadata_lists_each_rows_pages_and_pads_with_one_key_rows():
    """Its CSR plan: one query per row, the pages a row's keys fill, and its key count; padding reads page 0."""
    context = Context(
        cu_seqlens_k_host=[0, 5, 5 + 2 * BLOCK_SIZE],
        block_tables=torch.tensor([[4, 9, -1], [7, 2, -1]], dtype=torch.int32),
    )
    metadata = FlashInferMLABackend._metadata(context, BLOCK_SIZE, num_rows=4)
    assert [tensor.tolist() for tensor in metadata] == [
        [0, 1, 2, 3, 4],
        [0, 1, 3, 4, 5],
        [4, 7, 2, 0, 0],
        [5, 2 * BLOCK_SIZE, 1, 1],
    ]
    assert all(tensor.dtype == torch.int32 for tensor in metadata)


def test_flashinfer_mla_full_graphs_get_a_wrapper_per_batch_size_replanned_before_replay(fake_flashinfer_mla):
    """Captured as the runner captures, largest first with worst-case lengths, then re-planned for a real step."""
    backend, width = fake_flashinfer_mla, 3
    cache = torch.zeros(8, BLOCK_SIZE, LATENT_DIM)
    for bs in (4, 2):
        context = Context(
            context_lens=torch.full((bs,), width * BLOCK_SIZE, dtype=torch.int32),
            block_tables=torch.zeros(bs, width, dtype=torch.int32),
            full_graph_size=bs,
        )
        for _ in range(2):  # layers, which share the step's plan
            out = backend.mla_decode(torch.zeros(bs, NUM_HEADS, LATENT_DIM), cache, LATENT_V_DIM, context)
    assert out.shape == (2, NUM_HEADS, LATENT_V_DIM)
    graphs = {key[-1]: wrapper for key, wrapper in FlashInferMLABackend._wrappers.items()}
    assert set(graphs) == {4, 2} and all(w.use_cuda_graph and len(w.plans) == 1 for w in graphs.values())
    qo_indptr, kv_indptr, kv_indices, kv_lens = FlashInferMLABackend._graph_metadata
    assert graphs[2].buffers[2] is kv_indices and graphs[2].buffers[3].data_ptr() == kv_lens.data_ptr()

    step = Context(
        cu_seqlens_k_host=[0, 5, 5 + BLOCK_SIZE + 1, 6 + BLOCK_SIZE + 1],
        block_tables=torch.tensor([[4, -1], [7, 2], [3, -1]], dtype=torch.int32),
    )
    FlashInferMLABackend.before_full_graph_replay(step, 4)

    assert graphs[4].plans[-1] == ([0, 1, 2, 3, 4], [0, 1, 3, 4, 5], [4, 7, 2, 3, 0], [5, BLOCK_SIZE + 1, 1, 1])
    assert len(graphs[2].plans) == 1  # another graph's wrapper is left alone
    assert kv_indptr[:5].tolist() == [0, 1, 3, 4, 5] and kv_lens[:4].tolist() == [5, BLOCK_SIZE + 1, 1, 1]

    backend.mla_decode(torch.zeros(3, NUM_HEADS, LATENT_DIM), cache, LATENT_V_DIM, step)  # an eager step
    eager = [w for key, w in FlashInferMLABackend._wrappers.items() if key[-1] is None]
    assert len(eager) == 1 and not eager[0].use_cuda_graph


def _paged_attention_with_lse(q, k_cache, v_cache, n: int, pages, causal: bool):
    """One row against its first n cached keys: output [lq, H, D] and log-sum-exp [H, lq]."""
    page_size = k_cache.size(1)
    pages = pages[: -(-n // page_size)].long()
    k, v = (cache[pages].flatten(0, 1)[:n].float() for cache in (k_cache, v_cache))
    group = q.size(1) // k.size(1)
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * SCALE
    if causal:  # bottom-right aligned
        lq = q.size(0)
        visible = torch.arange(n) <= (n - lq + torch.arange(lq)).unsqueeze(1)
        scores = scores.masked_fill(~visible, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).to(q.dtype), scores.logsumexp(-1)


class FakeFA3:
    """flash_attn_with_kvcache and get_scheduler_metadata on the CPU: the real arithmetic, and a schedule that
    records its problem, so a kernel call can be matched to the schedule it was handed."""

    def __init__(self):
        self.schedules, self.calls = [], []

    def get_scheduler_metadata(self, cache_seqlens, **problem):
        self.schedules.append(problem)
        return torch.tensor([problem["batch_size"], *cache_seqlens.tolist()], dtype=torch.int32)

    def flash_attn_with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        page_table,
        cu_seqlens_q=None,
        max_seqlen_q=None,
        softmax_scale=None,
        causal=False,
        scheduler_metadata=None,
        return_softmax_lse=False,
    ):
        assert softmax_scale == SCALE
        self.calls.append(scheduler_metadata)
        bounds = None if cu_seqlens_q is None else cu_seqlens_q.tolist()
        rows = list(q) if bounds is None else [q[a:b] for a, b in zip(bounds, bounds[1:])]
        results = [
            _paged_attention_with_lse(row, k_cache, v_cache, int(n), pages, causal)
            for row, n, pages in zip(rows, cache_seqlens, page_table)
        ]
        if bounds is None:  # batched: out [B, lq, H, D], lse [B, H, lq]
            out, lse = torch.stack([o for o, _ in results]), torch.stack([lse for _, lse in results])
        else:  # packed: out [N, H, D], lse [H, N]
            out, lse = torch.cat([o for o, _ in results]), torch.cat([lse for _, lse in results], dim=1)
        return (out, lse) if return_softmax_lse else out


@pytest.fixture
def fake_fa3(monkeypatch):
    from lean_vllm.attention import flash_backend

    fake = FakeFA3()
    monkeypatch.setattr(flash_backend, "flash_attn_with_kvcache", fake.flash_attn_with_kvcache, raising=False)
    monkeypatch.setattr(flash_backend, "get_scheduler_metadata", fake.get_scheduler_metadata, raising=False)
    monkeypatch.setattr(FlashAttention3Backend, "_graph_problems", {})
    monkeypatch.setattr(FlashAttention3Backend, "_graph_schedules", {})
    return FlashAttention3Backend(NUM_HEADS, HEAD_DIM, SCALE, NUM_KV_HEADS), fake


def _shared_prefix_step(decode: bool):
    """Three rows whose first two pages are the same blocks, then pages of their own, as a prefix-cache hit leaves
    them; each has cached keys past the shared pages, so the prefix covers no query."""
    torch.manual_seed(1)
    k_cache, v_cache = (torch.randn(12, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM) for _ in range(2))
    tables = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 6, 7]], dtype=torch.int32)
    starts, lens = [2 * BLOCK_SIZE + 3, 2 * BLOCK_SIZE + 9, 3 * BLOCK_SIZE], [1, 1, 1] if decode else [4, 1, 6]
    ends = [s + n for s, n in zip(starts, lens)]
    cu_q = [0, *torch.tensor(lens).cumsum(0).tolist()]
    context = Context(
        is_prefill=not decode,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, *torch.tensor(ends).cumsum(0).tolist()], dtype=torch.int32),
        max_seqlen_q=max(lens),
        context_lens=torch.tensor(ends, dtype=torch.int32),
        block_tables=tables,
    )
    q = torch.randn(cu_q[-1], NUM_HEADS, HEAD_DIM)
    expected = torch.cat(
        [
            _paged_attention_with_lse(q[a:b], k_cache, v_cache, n, pages, causal=True)[0]
            for a, b, n, pages in zip(cu_q, cu_q[1:], ends, tables)
        ]
    )
    return q, k_cache, v_cache, context, expected


@pytest.mark.parametrize("decode", [True, False], ids=["decode", "prefill"])
def test_fa3_cascade_matches_attending_each_row_whole(fake_fa3, decode):
    """The shared pages read once for every query, unmasked, merged with each row's own keys past them."""
    backend, fake = fake_fa3
    q, k_cache, v_cache, context, expected = _shared_prefix_step(decode)
    context.common_prefix_len = 2 * BLOCK_SIZE
    attend = backend.decode if decode else lambda *args, **kw: backend.prefill(q, None, None, *args[1:], **kw)
    out = torch.full_like(q, float("nan"))

    got = attend(q, k_cache, v_cache, context, out=out)

    assert got is out
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    prefix, suffix = fake.schedules
    assert (prefix["batch_size"], prefix["max_seqlen_q"], prefix["max_seqlen_k"]) == (1, q.size(0), 2 * BLOCK_SIZE)
    assert not prefix["causal"] and prefix["cu_seqlens_q"] is None
    assert (suffix["batch_size"], suffix["max_seqlen_k"], suffix["causal"]) == (3, 2 * BLOCK_SIZE, True)
    suffix_lens = (context.context_lens - 2 * BLOCK_SIZE).tolist()
    assert fake.calls[1].tolist() == [3, *suffix_lens]  # the suffix call ran on its schedule: each row's keys past it


def test_fa3_makes_one_schedule_per_problem_per_step(fake_fa3):
    """As vLLM's AOT schedule: the first layer to pose a problem makes it, and the rest are handed the same one."""
    backend, fake = fake_fa3
    q, k_cache, v_cache, context, expected = _shared_prefix_step(decode=True)
    for _ in range(3):  # layers
        torch.testing.assert_close(backend.decode(q, k_cache, v_cache, context), expected, atol=1e-5, rtol=1e-5)
    assert len(fake.schedules) == 1
    assert fake.schedules[0]["max_seqlen_k"] == 4 * BLOCK_SIZE  # the page table's width, as the kernel reads it
    assert all(schedule is fake.calls[0] for schedule in fake.calls)


def test_fa3_full_graphs_read_a_schedule_refilled_before_replay(fake_fa3):
    """Captured largest first; each graph bakes the shared buffer, which a replay refills for its step's lengths."""
    backend, fake = fake_fa3
    k_cache, v_cache = make_cache(4, "cpu", BLOCK_SIZE, torch.float32)
    width = 2
    for bs in (4, 2):
        context = Context(
            context_lens=torch.full((bs,), width * BLOCK_SIZE, dtype=torch.int32),
            block_tables=torch.zeros(bs, width, dtype=torch.int32),
            full_graph_size=bs,
        )
        for _ in range(2):  # layers
            backend.decode(torch.zeros(bs, NUM_HEADS, HEAD_DIM), k_cache, v_cache, context)
    (buffer,) = FlashAttention3Backend._graph_schedules.values()
    assert len(fake.schedules) == 2 and buffer.numel() == 5
    assert fake.calls[-1].data_ptr() == buffer.data_ptr()

    step = Context(context_lens=torch.tensor([5, 9], dtype=torch.int32))
    FlashAttention3Backend.before_full_graph_replay(step, 4)

    assert buffer.tolist() == [4, 5, 9, 0, 0]  # the graph's padding rows hold no keys
    assert fake.schedules[-1]["batch_size"] == 4 and fake.schedules[-1]["max_seqlen_k"] == width * BLOCK_SIZE
    FlashAttention3Backend.before_full_graph_replay(step, 2)
    assert buffer.tolist() == [2, 5, 9, 0, 0]  # a smaller graph's schedule, its stale tail zeroed


@pytest.mark.parametrize(
    "prefix, query_lens, want",
    [
        (255, [5] * 8, False),  # too short a prefix
        (1024, [5] * 7, False),  # too few rows
        (256, [5] * 8, True),  # prefill: no flash decoding to compete with
        (4096, [1] * 256, True),  # a wide decode batch: cascade reads the prefix in fewer waves
        (256, [1] * 8, False),  # a narrow one: flash decoding fills the SMs
    ],
)
def test_fa3_cascades_as_vllms_heuristic_decides(prefix, query_lens, want):
    """Qwen3-8B's 32 query heads over 8 key heads, on an H100's 132 SMs."""
    assert flash_backend.use_cascade_attention(prefix, np.array(query_lens), 32, 8, 132) is want
