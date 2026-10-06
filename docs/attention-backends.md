# Attention backends

Models say *what* attention to compute, and an attention backend decides *how*.
Model code never imports a kernel, so the same model runs on a laptop with plain
PyTorch and on an H100 with FlashAttention-3, and each backend is tested against
the same reference. Each layer picks its own backend when it is built.

```
Qwen3Attention / MLAAttention
      |
      v
layers.attention.Attention        # owns the layer's KV cache slice
      |
      v
AttentionBackend                  # the interface
      |
      +-- TorchAttention          # PyTorch SDPA, any device, the reference
      |
      +-- FlashInferBackend       # FlashInfer's paged kernels + the Triton scatter, Ampere and newer
      |
      +-- FlashAttention3Backend  # FlashAttention-3 + a Triton cache scatter, Hopper only
      |     |
      |     +-- FlashMLABackend   # FA3, plus FlashMLA's decode for MLA models
      |
      +-- MLACommonBackend        # MLA only: prefill on FA3 (Hopper) or FlashInfer (elsewhere)
            |
            +-- TritonMLABackend      # plus vLLM's Triton MLA decode, Ampere and newer
            +-- FlashInferMLABackend  # plus FlashInfer's MLA decode, Ampere and newer
```

## What is supported

### Backends

| Backend (name) | Runs on | Serves | KV page size | CUDA graphs | Splits decode from prefill |
|---|---|---|---|---|---|
| `torch` | Anything: CPU, Apple Silicon, any CUDA GPU | Any layer; fp32, fp16 or bf16 | Any multiple of 16 | No, always eager | Yes |
| `flashinfer` | sm80 and newer (A100, H100, ...), Linux | Plain layers; fp16/bf16; head size 64, 128 or 256 | Any multiple of 16 | Full + piecewise | Yes |
| `flash_attn_3` | H100 / H200 (sm90), Linux x86_64 | Plain and MLA layers; fp16/bf16; head size a multiple of 8, up to 256 | Any multiple of 16 | Full + piecewise | No |
| `flashmla` | H100 / H200, with FlashMLA built from source | MLA layers with 512 + 64 latents; as FA3 otherwise | 64 | Full + piecewise | MLA layers split themselves |
| `triton_mla` | sm80 and newer, with FA3 or FlashInfer for prefill | MLA layers with 512 + 64 latents; fp16/bf16 | Any | Full + piecewise | MLA layers split themselves |
| `flashinfer_mla` | sm80 and newer, Linux | As `triton_mla` | Any | Full + piecewise | MLA layers split themselves |

`torch` is written to be obviously correct, not fast. It is the backend for
development and tests, not for benchmarks. `torch`, `flashmla`, `triton_mla` and
`flashinfer_mla` decode MLA latents directly (`mla_decode`); `flash_attn_3`
expands them.

An A100 has a fast backend for plain layers, `flashinfer`, and for MLA layers,
`triton_mla`, which prefills there on FlashInfer. FlashAttention-2 is not
supported.

### Selection

Each layer chooses its backend in `__init__`, as vLLM's `Attention` does. It
describes itself as a `LayerSpec`: head size, query and key head counts, dtype
(the model's, which the runner sets as the default while building it), and kind:
a plain `decoder` layer, or `mla` when it caches latents. Then:

1. the name passed to `get_attention_backend()`, if any;
2. otherwise `LEAN_VLLM_ATTENTION_BACKEND`;
3. otherwise the first backend, in the order `flashmla`, `triton_mla`,
   `flashinfer_mla`, `flash_attn_3`, `flashinfer`, `torch`, that is available
   and whose `validate(spec)` returns no reason against the layer.

The MLA order is vLLM's on Hopper and Ampere: FlashMLA, then Triton MLA. So
`flashinfer_mla` is only reached by name, until measurements say otherwise.

`validate` is vLLM's `validate_configuration`: it checks `supported_dtypes`,
`supports_head_size`, that query heads group evenly over key heads, and
`supported_kinds`, and a backend can add its own checks (the MLA decode backends
require 576-wide latents). So an fp32 layer takes `torch` even on an H100, an MLA layer
never takes `flashinfer`, and `flashmla` never serves a plain layer. Layers of
one model may end up on different backends; the runner logs the count per
backend.

`torch` serves every layer, so automatic selection never fails. Naming a backend
that is unavailable, or that cannot serve a layer, raises an error listing the
reasons rather than falling back, because a silent switch to a much slower
backend in the middle of a benchmark is worse than a crash.

```bash
LEAN_VLLM_ATTENTION_BACKEND=torch uv run python example.py
```

When an MLA model's layers would take `flashmla`, `Config` switches the KV cache
to 64-token pages and logs a warning. It resolves the backend from the spec the
layers will build, before any layer exists.

Graphs depend on every layer: the runner stays eager if any layer's backend
reports `supports_cuda_graph()` false, and drops full graphs for piecewise if
any layer's decode cannot be captured (`Attention.supports_full_cudagraph()`).

### Installing the kernels

- **FlashAttention-3**: `uv sync --extra cuda`. Dao-AILab publishes no wheel,
  so this installs a third-party build pinned by URL and hash in
  `pyproject.toml`, for Linux on x86_64 against the pinned torch.
- **FlashInfer**: also in the `cuda` extra, as `flashinfer-python` on Linux.
  That wheel compiles its kernels on first use, so the machine needs `nvcc`.
- **FlashMLA**: build it from source; see
  [deepseek-v2.md](deepseek-v2.md#running-it).
- **Triton**: in the `cuda` extra. `triton_mla`'s kernels compile on first use.

## What has been verified

| Check | Where | Result |
|---|---|---|
| Every available backend against a dense reference: prefill, paged prefill, prefix-cache hits, decode, mixed batches, `varlen_with_lse` | Every machine the suite runs on | Pass |
| `mla_decode` against the reference at FlashMLA's shapes | CPU (`torch`); H100 (`flashmla`) | Pass on CPU; H100 result not recorded |
| The CUDA backend against the reference | A100, back when that backend was FlashAttention-2 | Pass |
| FlashAttention-3 suite | H100 | Not recorded, though both FA3 backends have served benchmarks there |
| FlashInfer suite | Any sm80+ GPU | **Never run.** The backend was written against the `flashinfer-python` 0.7.0 API and has not executed a kernel |
| Per-layer selection by dtype, head size, head count and kind; a forced backend that cannot serve a layer | Every machine | Pass |
| The decode/prefill split: its slices, and a split step against the reference | Every machine (`torch` splits) | Pass |
| FlashInfer's page table built from host lengths | CPU | Pass |
| FlashInfer's full-graph wrappers: one per batch size over shared buffers, padded rows, re-planned before replay | CPU, with a fake wrapper | Pass; never captured on a GPU |
| `triton_mla` and `flashinfer_mla` decode against the reference, at 16- and 64-token pages, short and split-length rows | Any sm80+ GPU | **Never run.** Written against Triton 3.6 and the `flashinfer-python` 0.7.0 API without a GPU |
| FlashInfer's ragged plans: one per problem per step, reused across steps, the overflow sharing one wrapper; its lse turned to base e | CPU, with a fake wrapper | Pass |
| FlashInfer MLA's CSR plan, its graph wrappers and re-planning before replay | CPU, with a fake wrapper | Pass; never captured on a GPU |
| MLA backend order, and their prefill on FA3 or else FlashInfer | Every machine | Pass |

The reference is `dense_attention` in `tests/test_attention_backends.py`: the
textbook formula, looped over heads, with no SDPA and no paging, so agreeing with
it means something. Tests are parametrized over the backends available on the
machine, so the same file runs on a laptop and on a GPU box.

The suite was mutation-tested: each backend was broken on purpose and the tests
were run again.

| Mutation | Outcome |
|---|---|
| Top-left causal alignment | Caught |
| Ignore `-1` slots in `store_kvcache` | Caught |
| Off-by-one in the page gather | Caught |
| Swap the k and v caches in the gather | Caught |
| Decode reads the wrong query row | Caught |
| Drop the `scale` argument | Survived at first, since the tests used SDPA's default scale; fixed with a non-default scale |
| `repeat` instead of `repeat_interleave` for GQA | Survived at first, since that path is dead on torch ≥ 2.5; fixed by forcing it in the test |
| Offset the prefill side's key lengths by the query count | Caught |
| Stop putting one-query rows first | Caught |
| Run the prefill side over the whole step's context | Caught |

## How it works

### The interface

| Method | Required | What it does |
|---|---|---|
| `store_kvcache` | Yes | Scatters this step's keys and values into their cache slots |
| `prefill` | Yes | Attends packed variable-length rows, reading cached keys when a row resumes |
| `decode` | Yes | One query per row against the paged cache; the path CUDA graphs capture |
| `forward` | Has a default | One step: decode for a pure-decode step, else prefill, or both halves when the backend splits |
| `validate` | Has a default | Why the backend cannot serve a `LayerSpec`; empty if it can |
| `get_kv_cache_shape` | Has a default | The cache layout, which is the backend's choice |
| `varlen_with_lse` | For MLA layers | Attention over given keys, no cache, also returning the log-sum-exp |
| `mla_decode`, `store_latents` | Optional | MLA decode over latents, and the latent cache's scatter |

The KV cache layout belongs to the backend, which is why `store_kvcache` and
`get_kv_cache_shape` live here. The runner asks each layer for its cache shape
and allocates each layer's cache on its own, so layers on different backends
may use different layouts: a plain `Attention` layer answers from its backend,
and `MLAAttention` answers with its own latent layout
([deepseek-v2.md](deepseek-v2.md)).

Capability flags tell the runner what a backend can do: `supports_cuda_graph()`,
`supports_full_cudagraph()`, `split_decodes()`, `supports_mla_decode()`,
`supports_value_head_size()`, `supports_full_cudagraph_mla_decode()` and
`mla_block_size()`. One hook,
`before_full_graph_replay(context, batch_size)`, runs on each layer's backend
class before a full graph replays, to refresh state the graph reads but the
host computes; it does nothing by default.

### Tensor shapes

Sequences are packed, not padded, and every backend takes and returns the same
shapes.

| | Shape |
|---|---|
| `prefill` q | `[num_tokens, num_heads, head_dim]` |
| `prefill` k, v | `[num_tokens, num_kv_heads, head_dim]`, new tokens only |
| `prefill` returns | `[num_tokens, num_heads, head_dim]` |
| `decode` q | `[batch_size, num_heads, head_dim]` |
| `decode` returns | `[batch_size, num_heads, head_dim]` |
| `varlen_with_lse` k, v | `[num_keys, num_kv_heads, head_dim]`, no cache |
| `varlen_with_lse` returns | output as `prefill`, and lse `[num_tokens, num_heads]` |
| `mla_decode` q | `[batch_size, num_heads, latent_dim]` |
| `mla_decode` returns | `[batch_size, num_heads, v_dim]` |
| `store_latents` latent | `[num_tokens, latent_dim]`, slot `-1` skips |

A slot of `-1` marks a row that a CUDA graph padded, and both store methods skip
it. FA3 returns its lse as `[num_heads, num_tokens]` and a singleton query axis
in decode, so the flash backend transposes and squeezes to match.

### Causal masking is bottom-right aligned

Under chunked prefill or prefix caching a row has fewer queries than keys, and
its queries are the *last* `lq` of its `lk` keys. Query `j` (0-indexed) attends
keys `0, …, lk - lq + j`.

`scaled_dot_product_attention(is_causal=True)` aligns top-left instead, and
when `lq != lk` it silently returns a plausible but wrong answer. So
`TorchAttention` builds the mask from absolute positions and never passes
`is_causal`. FlashAttention has aligned bottom-right since 2.1, so the two
agree. `test_top_left_causal_alignment_would_be_wrong` checks both that the
backend matches the reference and that the top-left answer differs, so the test
cannot quietly stop telling them apart.

### Mixed batches, and the split into decode and prefill

A step can hold prompt chunks and decode rows together. `prefill` already takes
packed rows with fewer queries than keys, so a decode row is simply a row with
one query, and the bottom-right mask already fits it. FA3 runs a mixed step that
way, in one call, as vLLM's FlashAttention backend does.

A backend whose `split_decodes()` is true instead runs the step's one-query rows
through `decode` and the rest through `prefill`, as vLLM's FlashInfer and MLA
backends do (their `reorder_batch_threshold` is 1). The pieces:

1. **Reorder.** `ModelRunner.decodes_first` puts one-query rows first, stably,
   like vLLM's `reorder_batch`. Only the token layout moves: `logits_indices`
   still lists sampling rows in the scheduler's order, which is the order it
   reads tokens back in.
2. **Split.** `split_decodes_and_prefills(context)` counts the leading
   one-query rows and slices the step into a decode context and a prefill
   context, from the host lengths, without a sync. It is built on the first
   layer's call and cached on the step's context, as are any plans made on it.
3. **Run.** `AttentionBackend.forward` calls `decode` on the first `n` rows
   and `prefill` on the rest, and writes both into one output.

`MLAAttention` splits the same way and with the same helper, since it plays the
part of vLLM's `MLACommonImpl`: its decode attends latents and its prefill
expands them.

A row decodes by its shape, so a one-token prompt chunk joins the decode side:
attention cannot tell it from a decode. Sampling is decided separately, so it
still does not sample unless it ends the prompt.

`decode` on its own remains the pure-decode path, because that is the only shape
a full CUDA graph can capture. The runner takes it only when **no** row is a
prompt chunk. Checking that every query length is 1 would be wrong there: a
prompt chunk can be one token long when the budget runs down to one, and unless
it ends the prompt it must not sample.

### FlashAttention-3's prefill reads the pages

As vLLM's V1, a prefill reads every key from the pages, cold rows included:
`attend` stores the step's keys first, then prefill calls
`flash_attn_with_kvcache` with `page_table`. FA3's `flash_attn_varlen_func`
takes no page table, so it serves only calls with no pages: warmup, before the
cache exists, and MLA's new tokens.

V0 skipped the pages when no row resumed from cached keys; V1 dropped that
second path, and so does this backend.

FA3 reads pages of any size, which is what made 16 tokens the default block
size. FA2 required multiples of 256.

### FlashInfer: planned once per step

FlashInfer splits each call into a host-side `plan()`, which builds the work
schedule from the step's lengths, and a `run()` per layer. Like vLLM's metadata
builder, the backend plans on the step's first layer and every later layer with
the same shape reuses the plan. The planned wrappers live in
`context.attn_metadata`, keyed by kernel and layer shape, so the next step's
fresh context plans again.

| Rows | Wrapper |
|---|---|
| Decode rows | `BatchDecodeWithPagedKVCacheWrapper`, on tensor cores when a key head serves more than 4 query heads, as vLLM chose |
| Prompt rows | `BatchPrefillWithPagedKVCacheWrapper`, causal (bottom-right aligned), cold rows too, as vLLM's V1 |
| No pages (warmup, MLA's new tokens) | `BatchPrefillWithRaggedKVCacheWrapper` on this step's k and v |
| `varlen_with_lse`, for MLA prefill | The same ragged wrapper, with its base-2 lse multiplied by ln 2 |

FlashInfer wants each row's pages packed (a CSR table), not the padded
`block_tables`. The index pointers and last-page lengths come from the host
lengths the runner already keeps, and the page gather runs on the device, so
planning costs no sync. All wrappers share one zeroed 256 MiB workspace.

Ragged plans are keyed by the problem itself, its host lengths and
causality, rather than by layer shape: an MLA step poses several, one for its
new tokens and one per chunk of cached keys, and every layer poses the same
ones. Each gets its own wrapper from a small pool, so it is planned once a step,
and a later step that poses it again reuses the plan. Past `MAX_RAGGED_PLANS`
problems in one step, the rest share the last wrapper and re-plan as they
alternate. `MLAAttention` passes the host lengths in, so this costs no sync.

The KV cache keeps the layout every backend uses (`[blocks, block_size, heads,
dim]`, FlashInfer's `NHD`), and keys are stored by the same Triton scatter as
FA3's.

Full graphs capture FlashInfer's decode as vLLM does. A replay runs no
Python inside the graph, so nothing would plan it; instead:

1. **Fixed buffers.** One page table (indptr, indices, last-page lengths) sized
   for the largest captured batch, allocated on the first capture, which is the
   largest. Each graph's wrapper is built with `use_cuda_graph=True` over slices
   of it, so the captured kernel reads those addresses.
2. **A wrapper per batch size.** Under CUDA graphs a wrapper's batch size is
   fixed, so each captured size gets its own, keyed by `full_graph_size` in the
   context. It is planned on the capture's warmup pass, with the capture's
   worst-case lengths, and the capture reuses that plan.
3. **Re-plan before replay.** `_replay_full` calls `before_full_graph_replay`,
   which plans that size's wrappers from the step's host lengths, padded to the
   graph's batch size with empty rows (last-page length 1, as vLLM pads). The
   plan copies into the fixed buffers, so the replay reads the new step.

This is safe because, in CUDA-graph mode, FlashInfer 0.7.0 decides split-KV
from the batch size alone (`scheduler.cuh`), so a re-plan never asks for kernels
the graph did not capture. Each wrapper also allocates its own int workspace,
so roughly 36 captured sizes cost a few hundred MiB on top of the KV cache.

### MLA decode: Triton and FlashInfer

Both backends take the latent cache as it is, `[blocks, block_size, 576]`, and
read any page size, so an MLA model keeps the 16-token default. Their prefill
expands latents, as every vLLM MLA backend's does, on FA3 on Hopper and on
FlashInfer's ragged prefill elsewhere (`mla_common.prefill_backend()`, standing
in for vLLM's FlashAttention-2).

`triton_mla` is vLLM's `TRITON_MLA` kernel (`triton_mla_decode.py`), ported to
this cache. A first launch splits each row's keys into 4 parts. Each program
attends 16 heads to one part, 16 keys at a time, and writes that part's output
and log-sum-exp. A second launch merges the parts. It plans nothing on the host
and reads lengths and pages on the device, so a full graph captures it as it
is. A row with no keys, as a graph's padding rows have, comes out zero.

`flashinfer_mla` is FlashInfer's `BatchMLAPagedAttentionWrapper`, its FA2/FA3
MLA kernel for Ampere and Hopper. vLLM's `FLASHINFER_MLA` is a different
kernel, trtllm-gen, which runs only on Blackwell. The wrapper takes the 576-wide
query and cache packed, so nothing is copied. It is planned once per step from a
CSR page table built from the host lengths, as `flashinfer` is. Full graphs
follow the same scheme: a wrapper per batch size, over reserved buffers sliced
from shared ones, re-planned before each replay. Padding rows hold one key on
page 0, so no planned row is empty.

### Trade-offs of the torch backend

`TorchAttention` pads every row to the step's longest, gathers all rows' pages
into one contiguous tensor, and makes one masked SDPA call, with no host sync.
Decode gathers the block table's full width, so it reads no lengths back either.
The padding and the gathered copy cost memory traffic that grows with batch
size and context length, and a step that mixes long chunks with decode rows
spends compute on padding. Each key head's query heads fold into its query axis
rather than going through `enable_gqa`, which takes a slow path on MPS once
batched. The backend still reports no CUDA graph support. That is the right
trade for a reference and for laptop development, and the wrong one for speed.

## Next steps

1. Run the suite with `flashinfer`, `triton_mla` and `flashinfer_mla` on an
   H100, then serve a Qwen3 model on `flashinfer` with FA3 disabled and compare
   against vLLM's FlashInfer backend.
2. Serve DeepSeek-V2-Lite on each MLA decode backend at loads 1 and 48, and
   order them by what they measure.
3. Capture a FlashInfer model's full graphs on a GPU and check its decode
   against eager, then compare decode throughput with piecewise only.
4. Let `decode` and `prefill` write into the split's output (`out=`), so a
   split step skips the copy.
