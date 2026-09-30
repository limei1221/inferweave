# DeepSeek-V2: MLA, MoE and YaRN

Status: `DeepseekV2ForCausalLM` loads DeepSeek-V2-Lite and runs on the torch,
FlashAttention-3 and FlashMLA backends, with CUDA graphs on. It is checked
against transformers on tiny random checkpoints. On an H100 the Triton MoE
matches `grouped_mm`, and the real V2-Lite-Chat weights have been served and
benchmarked against vLLM with both graph modes and FlashMLA decode inside the
full graph ([20 September report](benchmark-2026-09-20.md)). Not yet recorded:
FlashMLA decode against the torch reference on the GPU, and greedy output on the
real weights against vLLM.

## Running it

```bash
uv run hf download deepseek-ai/DeepSeek-V2-Lite-Chat --local-dir ~/workspace/huggingface/DeepSeek-V2-Lite-Chat
uv run lean-vllm serve ~/workspace/huggingface/DeepSeek-V2-Lite-Chat --served-model-name deepseek
```

The weights are 31 GB in bf16, so a 40 GB GPU is the floor. The config and
tokenizer load natively from transformers 4.56 on, with no `trust_remote_code`.
The runner picks the model class from `architectures` in `config.json`
(`lean_vllm/models/__init__.py`).

## MLA: the cache holds latents

Each token caches `kv_lora_rank + qk_rope_head_dim` values per layer. That is
the normalized compressed KV plus the shared rope key, with rope already
applied. Keys and values per head are not cached. For V2-Lite that is 576
values against the 16 × (192 + 128) = 5120 a plain KV cache would need, about
31 KB per token in bf16 across 27 layers.

`MLAAttention` (`layers/attention.py`) is an `Attention` layer with its own
layout. The runner asks every layer for `kv_cache_shape` and hands its slice
back through `bind_kv_cache`, rather than reading head counts off the config.
Each step it:

1. scatters the step's latents into their slots;
2. expands the step's own latents into keys and values with `kv_b_proj`. When
   no row resumes (`keys_are_new`) these are every key the step reads, so the
   backend's varlen `prefill` runs with no page table and that is all;
3. otherwise runs `varlen_with_lse` causally over the step's own keys. A decode
   row is simply a row whose query is one token long;
4. reads the cached context in chunks of at most `max_context_chunk` keys, as
   vLLM's chunked context does. Each chunk gathers its latents, expands them,
   attends them unmasked, since all cached keys precede the step's queries, and
   merges its output into the running one by log-sum-exp.

`plan_context_chunks` packs consecutive rows into a chunk and splits a row
longer than the budget across chunks. The plan and its slots are built once per
step from `cu_seqlens_q_host` and `cu_seqlens_k_host`, so nothing syncs. The
runner sets the budget to `max_num_batched_tokens`: warmup expands that many new
latents, so no chunk expands more than the KV-cache sizing already measured.
vLLM uses a separate workspace of up to 64k tokens and reserves it in its
profile run instead.

Values are zero-padded from `v_head_dim` (128) to the query/key head size (192),
so a backend still sees one head size. The padding leaves the scores alone and
yields zero output lanes, and the output is cut back to 128.

Chunking bounds the memory, not the compute: a step that goes this way
re-expands the prefill context in every layer, one chunk at a time. On a backend
with `mla_decode`, decode rows avoid expansion even in a mixed step. Backends
without it use the expanded path for every row.

## Decode over latents: FlashMLA

A key's nope part is `W_k c` for a latent `c`, so `q · W_k c = (W_kᵀ q) · c`.
For decode rows `MLAAttention` moves each head's nope query into latent
space with the key half of `kv_b_proj`, keeps the rope query as it is, and
attends the cached latents directly as one shared key head. The values are the
first `kv_lora_rank` entries of each latent. Attention is linear in them, so the
value half of `kv_b_proj` applies after it. Nothing expands. vLLM does the same
with `W_UK_T` and `W_UV`.

For mixed steps, `prepare_batch` records each request's prefill/decode phase.
After storing all new latents, attention gathers decode queries and runs them
against their own page tables, while only prefill rows expand their latents.
It scatters both outputs back into the original token order before the output
projection. The subset metadata, prefill context-chunk plan and FlashMLA decode
schedule are reused across layers within the step. A one-token prefill remains
a prefill. Mixed steps still use piecewise graphs or eager execution; the
full-graph path remains for pure decode.

`FlashMLABackend` runs this with FlashMLA's dense decode kernel and uses
FlashAttention-3 for every other step. The kernel takes bf16 or fp16, a latent
of 512 + 64 and 64-token pages, on Hopper only. V2-Lite's latent fits. When the
backend is selected, `Config` sets `kvcache_block_size` to 64. The first layer
of a step creates the schedule holder, the kernel fills it from `context_lens`,
and the rest of the step's layers reuse it.

FlashMLA publishes no wheel, and `flash-mla` on PyPI is an empty placeholder.
Build it into the project's environment from a checkout, against the pinned
torch. `uv sync` removes it again unless run with `--inexact`:

```bash
git clone --recursive https://github.com/deepseek-ai/FlashMLA.git && cd FlashMLA
VIRTUAL_ENV=~/workspace/lean-vllm/.venv uv pip install --no-build-isolation -v .
```

The backend targets FlashMLA's current interface, where `get_mla_metadata()`
takes no arguments and returns a `FlashMLASchedMeta`. `TorchAttention` also
implements `mla_decode`, as the reference, so the torch backend decodes the
same way.

## MoE

`FusedMoE` (`layers/moe.py`) stacks the routed experts into one `gate_up_proj`
of shape `[E, 2I, H]` and one `down_proj` of shape `[E, H, I]`. The loader maps
each checkpoint weight `experts.{e}.{proj}.weight` into its expert's row.
Routing follows the original V2 code: softmax scores, `greedy` or
`group_limited_greedy` selection, then either `routed_scaling_factor` or
`norm_topk_prob`. Tensor parallelism shards each expert's intermediate size.
Shared experts reuse the dense gated MLP.

Two paths run the experts, and both sort the token-expert pairs by expert with
`searchsorted` rather than `bincount`, which syncs on CUDA. On CUDA it is the
Triton kernel in `layers/fused_moe.py`, as vLLM's fused MoE does it:
`align_blocks` pads each expert's run of sorted rows to a whole number of
`BLOCK_M` rows, so `fused_moe_kernel` reads one expert's weight per block and
reuses it down the block. The kernel runs once per projection — `A[pair //
top_k] @ B[expert]`, with the routing weight folded into the second epilogue —
and the top-k rows are summed at the end. Everywhere else, and under
`LEAN_VLLM_MOE_BACKEND=torch`, two `F.grouped_mm` calls over the same sorted
rows do the same thing; that path is the reference the kernel is checked
against.

## YaRN

`rotary_embedding.py` computes YaRN's inverse frequencies and scales cos and sin
by `mscale / mscale_all_dim`, a ratio that is 1 for V2-Lite. The attention layer
multiplies the softmax scale by `yarn_get_mscale(factor, mscale_all_dim)²`,
about 1.59 at V2-Lite's factor of 40. DeepSeek's rope rotates adjacent pairs,
GPT-J style, so its rotary embedding is built with `is_neox_style=False`.

## CUDA graphs

The model reports `supports_cuda_graph = True`, and both modes took a change.

A full graph holds attention, so the step inside it must neither plan chunks on
the host nor expand latents — that is exactly what `mla_decode` avoids. The
runner drops full graphs for an MLA model on a backend without one, or whose
`supports_full_cudagraph_mla_decode` is false, keeping piecewise
(`_cudagraph_mode`). FlashMLA answers true: its kernel builds the tile schedule
and split-KV workspace from `context_lens`, so `capture_cudagraph` captures at
`max_model_len` and each replay gates the KV loop on the lengths it refreshes.
The warmup's schedule is cleared before capture, so the graph owns its own. A
graph also pads its batch to a captured size,
which is why the latent store is a backend call: `store_latents` skips slot -1
in the kernel, as `store_kvcache` already did, rather than masking on the host.

Piecewise capture no longer assumes q/k/v. `pre_attention` returns whatever this
model hands attention — a query and a latent here, three tensors for Qwen3 — and
the runner shapes one buffer per tensor from a real call at the largest bucket.
The buffer for attention's output comes from `Attention.output_shape`, which MLA
answers with `v_head_dim`. Attention itself still runs eager between the pieces,
on real rows only, so the chunked context path is untouched.

The MoE is inside the captured pieces, so everything feeding it must stay
launch-only. The padding makes the Triton path's block count depend on the
routing, which a graph cannot have: `align_blocks` sizes its output from the
batch shape instead, an upper bound of one wasted block per expert, and writes
how many rows survived to a device tensor. The kernel reads that tensor to drop
the blocks the padding left empty, and masks the rows overhanging the last block
of a run, so no count reaches the host. Nothing in the `grouped_mm` path syncs
either. Both graph modes ran in the 20 September benchmark on an H100, with no
step running as eager decode.

## Testing

`tests/test_deepseek_v2.py` loads one checkpoint on disk into both
transformers' eager implementation and this one, in fp32 on the torch backend.
It runs three configs: V2-Lite's shape, one with `q_lora_rank`, and one with
group-limited routing. It compares a whole prompt, then a sequence of paged
steps built by the runner's own `prepare_batch`: a chunk, a resumed chunk beside
a cold prompt, pure decode, and decode mixed with a prompt. The block tables are
scattered. The paged steps run twice: with the context in one chunk, and with a
budget of 4, which splits a row across chunks and puts the end of one row and
the start of the next in the same chunk. `test_varlen_with_lse` checks both
backends' output and lse against the dense oracle.

Mutations the suite catches: resumed rows reading only their new latents, a key
gather that reads only the first page, uncut value padding, a dropped softmax
mscale, a dropped cos/sin attention factor, and unscaled routing weights. For
the chunked context: unmerged chunks, chunks attended causally, split rows read
from position 0, swapped merge weights, and an lse that is not accumulated.

The paged steps also run with pure decode both over latents and expanded.
`test_mla_decode` checks each backend's `mla_decode` against the dense oracle
at FlashMLA's shapes. Mutations it and the paged steps catch: the query or value
projection using another head's weights, and values read from the latent's tail.

The mixed-row regression interleaves two decode requests with a resumed prompt
and a one-token cold prompt. It checks logits against transformers, verifies
that both decode requests use `mla_decode`, and counts expanded tokens to
ensure their cached context is excluded. It also checks the expanded fallback
when the backend does not support latent decode. These checks use the torch
backend. Mixed FlashMLA steps ran in the H100 benchmark, but their output has
not been compared against a reference there.

`tests/test_fused_moe.py` covers the Triton path's blocking without a GPU.
`blocked_moe` writes out in torch what the kernel does with `align_blocks` — the
same gather, one expert per block, the same masked scatter — and checks it
against `torch_experts`, at both row block sizes. The alignment itself is
checked for holding every pair exactly once, for never putting two experts in
one block, and for giving an expert with no tokens no block. On CUDA,
`test_the_triton_kernel_matches_grouped_mm_on_cuda` runs `fused_experts` in bf16
against `grouped_mm`, covering the arithmetic inside `tl.dot`; it passes on an
H100.

`test_mla_decode` covers FlashMLA on Hopper, and the flash backend's
`varlen_with_lse` has its own test. Both kernels ran in the H100 benchmark, but
no record says whether their tests ran there or skipped.

Writing it turned up an fp32 bug in `RMSNorm`. `.float()` and `.to()` alias an
fp32 tensor, so the in-place normalization rewrote the residual. bf16 always
copies, which is why Qwen3 never hit it.

End to end, `LLM.generate` ran greedy on a tiny random checkpoint with
V2-Lite's tokenizer, an 8-token step budget and repeated prompts. It matched
transformers' `generate` token for token.

## Next

1. Compare greedy output on the real weights with vLLM, and record it.
2. Record FlashMLA's decode against the torch reference on an H100, then add a
   Triton MLA decode off Hopper.
3. Time the Triton MoE at decode and prefill shapes. Its block sizes are a
   guess until then, where vLLM ships a tuned table per shape and dtype.

## Closing the gap with vLLM

Compressed MLA caching, latent-space decode, chunked context attention, MoE
and YaRN are already implemented. The remaining work is to validate the GPU
paths, benchmark mixed-batch execution, and measure which kernel
optimizations matter. Follow this order:

1. Finish validating the GPU paths. The Triton MoE matches `grouped_mm` on an
   H100, and eager, full graph and piecewise execution all ran in the
   benchmark. Still to record: FlashMLA decode against the torch reference,
   and logits and greedy generation on the real V2-Lite weights against a
   pinned vLLM version, with numerical tolerances and mismatches.
2. Extend the baseline. The [20 September report](benchmark-2026-09-20.md)
   measures serving curves against vLLM 0.26.0 on one H100. Still to measure
   separately: prefill, pure decode and mixed traffic, and peak memory.
3. Measure mixed latent decode. Attention separates prefill and decode subsets
   and restores the original token order, as vLLM's MLA path does, and mixed
   steps ran on FlashMLA in the benchmark. Check their output there, then
   measure decode latency while prompts arrive. The saved expansion work has
   not yet been measured as an end-to-end speedup.
4. Optimize the measured bottlenecks. Profile routing, expert GEMMs, latent
   projections, context gathering and attention merging. Tune the MoE launch
   configuration for actual shapes and dtypes. Evaluate prepared per-head
   projection layouts and batched matmuls for latent decode, fused routing,
   and shared-expert execution. For models with `q_lora_rank`, evaluate fusing
   `q_a_proj` with `kv_a_proj_with_mqa`, as vLLM's MLA path does; V2-Lite does
   not have that query-compression stage. Keep changes only when correctness
   checks pass and benchmarks show a benefit.
5. Extend support when a workload needs it. Weight and KV-cache quantization,
   expert and pipeline parallelism, expert load balancing, and optimized
   MLA decode beyond Hopper are separate feature gaps. Prioritize them from
   memory, hardware and scaling requirements rather than treating all of
   vLLM's features as prerequisites for efficient V2-Lite inference.

Upstream references: [DeepSeek model and MoE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py),
[MLA wrapper](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mla.py),
and [MLA execution](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py).
These links track `main`; pin the compared revision in benchmark results.
