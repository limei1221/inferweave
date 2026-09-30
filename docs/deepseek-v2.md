# DeepSeek-V2: MLA, MoE and YaRN

lean-vLLM runs DeepSeek-V2 checkpoints (`DeepseekV2ForCausalLM`) alongside
Qwen3. The model brings three things Qwen3 does not have, and each needed engine
work:

- **MLA** (multi-head latent attention): the KV cache stores one small latent
  per token instead of full keys and values.
- **MoE**: most layers route each token to a few of many experts.
- **YaRN**: rope scaled for a long context.

DeepSeek-V2-Lite-Chat has been served on an H100 and benchmarked against vLLM.
Its output has been checked against transformers only on tiny random
checkpoints, not yet on the real weights.

## What is supported

### Models

| Model | Status |
|---|---|
| DeepSeek-V2-Lite, V2-Lite-Chat (16B total, 2.4B active) | Runs; served and benchmarked on one H100 |
| DeepSeek-V2 (236B) | Its extra features (`q_lora_rank`, group-limited routing) are tested on tiny checkpoints; never run at full size |
| DeepSeek-V3 and later | Not supported: `DeepseekV3ForCausalLM` is not registered, and V3's sigmoid routing is refused |

The runner picks the model class from `architectures` in `config.json`
(`lean_vllm/models/__init__.py`). The config and tokenizer load with
transformers ≥ 4.56 and need no `trust_remote_code`.

### Hardware and attention backends

The backend is chosen automatically. For an MLA model the preference is
`flashmla`, then `flash_attn_3`, then `torch`. Set
`LEAN_VLLM_ATTENTION_BACKEND` to force one.

| Backend | Runs on | How decode reads the cache | CUDA graphs |
|---|---|---|---|
| `flashmla` | H100/H200, with FlashMLA built from source | FlashMLA kernel over the latents | full + piecewise |
| `flash_attn_3` | H100/H200 | expands latents into keys and values | piecewise only |
| `torch` | anything: CPU, Apple Silicon, any CUDA GPU | over the latents, in plain torch | none (eager) |

`flashmla` switches the KV cache to 64-token pages, the only size its kernel
reads. On a non-Hopper GPU only `torch` is available, so there is no fast MLA
path there yet.

The routed experts run a Triton kernel on CUDA and `F.grouped_mm` elsewhere.
Set `LEAN_VLLM_MOE_BACKEND=triton|torch` to force one.

### Engine features

| Feature | Status |
|---|---|
| Chunked prefill, mixed prefill + decode batches | Supported |
| Prefix caching (resuming from cached latents) | Supported |
| OpenAI-compatible server, streaming, metrics | Supported, as for Qwen3 ([online-serving.md](online-serving.md)) |
| bf16 | Used on GPU; fp32 is used in the tests |
| Tensor parallelism | Implemented (attention heads and expert width are sharded), but only run at TP=1 |
| Expert or pipeline parallelism, expert load balancing | Not supported |
| Weight or KV-cache quantization | Not supported |

## Running it

The bf16 weights are 31 GB, so you need a GPU with at least 40 GB.

```bash
uv sync --extra cuda
uv run hf download deepseek-ai/DeepSeek-V2-Lite-Chat --local-dir ~/workspace/huggingface/DeepSeek-V2-Lite-Chat
uv run lean-vllm serve ~/workspace/huggingface/DeepSeek-V2-Lite-Chat --served-model-name deepseek
```

This runs on `flash_attn_3`. For the fast decode path, install FlashMLA. It has
no wheel, and `flash-mla` on PyPI is an empty placeholder, so build it into the
project's environment against the pinned torch:

```bash
git clone --recursive https://github.com/deepseek-ai/FlashMLA.git && cd FlashMLA
VIRTUAL_ENV=~/workspace/lean-vllm/.venv uv pip install --no-build-isolation -v .
```

A later `uv sync` removes it again unless you pass `--inexact`. The backend
targets FlashMLA's current interface, where `get_mla_metadata()` takes no
arguments and returns a `FlashMLASchedMeta`.

## What has been verified

| Check | Where | Result |
|---|---|---|
| Logits match transformers: whole prompts, chunked and resumed prefill, decode, mixed batches, chunked context | CPU, fp32, tiny random checkpoints | Pass |
| Greedy `LLM.generate` matches transformers `generate` token for token | CPU, tiny checkpoint with V2-Lite's tokenizer | Pass |
| Triton MoE kernel matches `grouped_mm` in bf16 | H100 | Pass |
| Real V2-Lite-Chat weights served under load, both graph modes, FlashMLA decode in the full graph | H100 | Ran; see the benchmark below |
| FlashMLA decode against the torch reference | H100 | Not yet recorded |
| Mixed FlashMLA steps against a reference | H100 | Not yet recorded |
| Real-weight logits and greedy output against vLLM | H100 | Not yet recorded |

The tests are in `tests/test_deepseek_v2.py`, `tests/test_fused_moe.py` and
`tests/test_attention_backends.py` (`test_varlen_with_lse`, `test_mla_decode`).
They run on a laptop, and the GPU cases run only where their GPU and kernels are present. The
tiny checkpoints cover three configs: V2-Lite's shape, one with `q_lora_rank`,
and one with group-limited routing.

## Performance

The [20 September report](benchmark-2026-09-20.md) compares V2-Lite-Chat against
vLLM 0.26.0 on one H100. With no queueing, lean-vLLM's median time per output
token is 7.0 ms against vLLM's 4.4 ms. Under load, lean-vLLM plateaus at about
20 requests/s while vLLM reaches 31.8. Decode now runs from CUDA graphs, and
most of the remaining gap is eager prefill.

## How it works

### MLA: the cache holds latents

Each token caches `kv_lora_rank + qk_rope_head_dim` values per layer: the
normalized compressed KV plus the shared rope key, with rope already applied.
For V2-Lite that is 576 values, against the 16 × (192 + 128) = 5,120 a plain
KV cache would need, or about 31 KB per token in bf16 across 27 layers.

`MLAAttention` (`layers/attention.py`) owns this layout. The runner asks each
layer for its `kv_cache_shape` rather than reading head counts off the config.

**Prefill** expands latents back into per-head keys and values with
`kv_b_proj`, then attends them with an ordinary kernel:

1. The step's new tokens attend each other causally.
2. If a row resumes from cached context, that context is read in chunks of at
   most `max_num_batched_tokens` keys, expanded and attended. Each chunk's
   output is merged into the running result by log-sum-exp, as in vLLM's
   chunked context. This bounds memory, but the context is re-expanded in every
   layer.

Values are 128 wide and keys 192, so values are zero-padded to 192 for the
kernel and the output is cut back to 128.

**Decode** skips the expansion. Since `q · (W_k c) = (W_kᵀ q) · c`, each head's
query is projected into latent space and attends the cached latents directly
as one shared key head. The value projection is applied after attention. vLLM
does the same with `W_UK_T` and `W_UV`. `flashmla` and `torch` implement this
as `mla_decode`. On `flash_attn_3`, decode rows are expanded like prefill.

**Mixed batches** split into a decode subset, which uses `mla_decode`, and a
prefill subset, which expands. The outputs are put back in the original token
order before the output projection. The split and the chunk plan are computed
once per step and reused by every layer.

### MoE

`FusedMoE` (`layers/moe.py`) stacks the routed experts into one `gate_up_proj`
of shape `[E, 2I, H]` and one `down_proj` of shape `[E, H, I]`. Routing follows
vLLM's `grouped_topk`: softmax scores, `greedy` or `group_limited_greedy`
selection, then `norm_topk_prob` and `routed_scaling_factor`. Shared experts
reuse the dense gated MLP.

Both expert paths sort token-expert pairs by expert without a host sync. On
CUDA, the Triton kernel in `layers/fused_moe.py` pads each expert's rows to
whole blocks, so a block reads one expert's weights, as vLLM's fused MoE does.
Elsewhere, two `F.grouped_mm` calls do the same work. That path is the
reference the kernel is tested against.

### YaRN

`layers/rotary_embedding.py` computes YaRN's frequencies and cos/sin scaling.
The attention layer also multiplies the softmax scale by
`yarn_get_mscale(factor, mscale_all_dim)²`, about 1.59 for V2-Lite. DeepSeek
rotates adjacent pairs (GPT-J style), so its rope uses `is_neox_style=False`.

### CUDA graphs

- **Full graphs** capture the whole decode step, attention included. That needs
  a decode with no host-side planning or expansion, which only `flashmla`
  offers. FlashMLA builds its schedule on the GPU from the context lengths, so
  one graph captured at `max_model_len` serves any shorter lengths. On other
  backends the runner falls back to piecewise.
- **Piecewise graphs** capture everything around attention, and attention runs
  eager between the pieces. The MoE sits inside a piece, so the Triton path
  sizes its blocks from the batch shape rather than the routing, and never
  reads a count back to the host.

## Next steps

1. Record the missing correctness checks on an H100: FlashMLA decode against
   the torch reference, mixed FlashMLA steps, and real-weight logits and greedy
   output against a pinned vLLM version.
2. Measure prefill, pure decode, mixed traffic and peak memory separately, and
   the end-to-end gain of latent decode in mixed batches.
3. Profile and optimize: prefill cost, Triton MoE block sizes (vLLM ships a
   tuned table per shape and dtype), routing, latent projections and context
   gathering. For models with `q_lora_rank`, try fusing `q_a_proj` with
   `kv_a_proj_with_mqa`, as vLLM does.
4. Add features when a workload needs them: an MLA decode kernel beyond Hopper,
   quantization, and expert or pipeline parallelism.

Upstream references, tracking vLLM `main`:
[DeepSeek model and MoE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py),
[MLA wrapper](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mla.py),
[MLA execution](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py).
