# Attention backends

Models say *what* attention to compute, and an attention backend decides *how*.
Model code never imports a kernel, so the same model runs on a laptop with plain
PyTorch and on an H100 with FlashAttention-3, and each backend is tested against
the same reference.

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
      +-- FlashAttention3Backend  # FlashAttention-3 + a Triton cache scatter, Hopper only
            |
            +-- FlashMLABackend   # FA3, plus FlashMLA's decode for MLA models
```

## What is supported

### Backends

| Backend (name) | Runs on | KV page size | CUDA graphs | MLA decode over latents |
|---|---|---|---|---|
| `torch` | Anything: CPU, Apple Silicon, any CUDA GPU | Any multiple of 16 | No, always eager | Yes, as the reference |
| `flash_attn_3` | H100 / H200 (sm90), Linux x86_64 | Any multiple of 16 | Full + piecewise | No; MLA decode expands latents |
| `flashmla` | H100 / H200, with FlashMLA built from source | 64 | Full + piecewise | Yes |

`torch` is written to be obviously correct, not fast. It is the backend for
development and tests, not for benchmarks.

There is no fast backend for GPUs other than Hopper: an A100 falls back to
`torch`. FlashInfer and FlashAttention-2 are not supported.

### Selection

The backend is chosen once, globally:

1. the name passed to `get_attention_backend()`, if any;
2. otherwise `LEAN_VLLM_ATTENTION_BACKEND`;
3. otherwise the first available of `flash_attn_3`, then `torch`. MLA models
   (DeepSeek-V2) try `flashmla` first.

`torch` is always available, so automatic selection never fails. Naming an
unavailable backend raises an error rather than falling back, because a silent
switch to a much slower backend in the middle of a benchmark is worse than a
crash.

```bash
LEAN_VLLM_ATTENTION_BACKEND=torch uv run python example.py
```

When `flashmla` is selected, `Config` switches the KV cache to 64-token pages and
logs a warning.

### Installing the kernels

- **FlashAttention-3**: `uv sync --extra cuda`. Dao-AILab publishes no wheel,
  so this installs a third-party build pinned by URL and hash in
  `pyproject.toml`, for Linux on x86_64 against the pinned torch.
- **FlashMLA**: build it from source; see
  [deepseek-v2.md](deepseek-v2.md#running-it).

## What has been verified

| Check | Where | Result |
|---|---|---|
| Every available backend against a dense reference: prefill, paged prefill, prefix-cache hits, decode, mixed batches, `varlen_with_lse` | Every machine the suite runs on | Pass |
| `mla_decode` against the reference at FlashMLA's shapes | CPU (`torch`); H100 (`flashmla`) | Pass on CPU; H100 result not recorded |
| The CUDA backend against the reference | A100, back when that backend was FlashAttention-2 | Pass |
| FlashAttention-3 suite | H100 | Not recorded, though both FA3 backends have served benchmarks there |

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

## How it works

### The interface

| Method | Required | What it does |
|---|---|---|
| `store_kvcache` | Yes | Scatters this step's keys and values into their cache slots |
| `prefill` | Yes | Attends packed variable-length rows, reading cached keys when a row resumes |
| `decode` | Yes | One query per row against the paged cache; the path CUDA graphs capture |
| `varlen_with_lse` | Yes | Attention over given keys, no cache, also returning the log-sum-exp |
| `get_kv_cache_shape` | Has a default | The cache layout, which is the backend's choice |
| `mla_decode`, `store_latents` | Optional | MLA decode over latents, and the latent cache's scatter |

The KV cache layout belongs to the backend, which is why `store_kvcache` and
`get_kv_cache_shape` live here. The runner asks each layer for its cache shape:
a plain `Attention` layer answers from its backend, and `MLAAttention` answers
with its own latent layout ([deepseek-v2.md](deepseek-v2.md)).

Capability flags tell the runner what a backend can do: `supports_cuda_graph()`,
`supports_mla_decode()`, `supports_full_cudagraph_mla_decode()` and
`mla_block_size()`.

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
its queries are the *last* `lq` of its `lk` keys. Query `j` attends keys
`0 ..= lk - lq + j`.

`scaled_dot_product_attention(is_causal=True)` aligns top-left instead, and
when `lq != lk` it silently returns a plausible but wrong answer. So
`TorchAttention` builds the mask from absolute positions and never passes
`is_causal`. FlashAttention has aligned bottom-right since 2.1, so the two
agree. `test_top_left_causal_alignment_would_be_wrong` checks both that the
backend matches the reference and that the top-left answer differs, so the test
cannot quietly stop telling them apart.

### Mixed batches

A step can hold prompt chunks and decode rows together. `prefill` already takes
packed rows with fewer queries than keys, so a decode row is simply a row with
one query, and the bottom-right mask already fits it.

`decode` remains as the pure-decode path, because that is the only shape a CUDA
graph can capture. The runner uses it only when **no** row is a prompt chunk.
Checking that every query length is 1 would be wrong: a prompt chunk can be
one token long when the budget runs down to one, and unless it ends the prompt
it must not sample.

### FlashAttention-3's two prefill calls

FA3 has two entry points, and which one a step uses depends on where its keys
are:

| Step | Call |
|---|---|
| No row resumes from cached keys | `flash_attn_varlen_func` on this step's k and v |
| Some row resumes | `flash_attn_with_kvcache` with `page_table` |

The runner answers this on the host as `keys_are_new`: cumulative query and key
lengths are equal exactly when no row starts from cached tokens. Reading it from
the tensors would cost a sync per layer.

Under chunked prefill, new prompts share a step with running decodes, so a
loaded server rarely takes the varlen path. Offline runs and steps with nothing
else running do. Whether that path is faster is unmeasured.

FA3 reads pages of any size, which is what made 16 tokens the default block
size. FA2 required multiples of 256.

### Trade-offs of the torch backend

`TorchAttention` loops over sequences in Python and gathers each one's pages
into a contiguous tensor before calling SDPA. That costs host syncs every
forward pass and memory traffic that grows with context length, and it makes
decode data-dependent, so the backend cannot be captured in a CUDA graph. That
is the right trade for a reference and for laptop development, and the wrong
one for speed.

## Next steps

1. Batch `TorchAttention`'s per-sequence loop before publishing any
   torch-versus-flash comparison, which would otherwise measure the Python
   loop.
2. Add a FlashInfer backend, then choose backends per layer (by head count,
   dtype, sequence length, or prefill versus decode) behind the same call.
