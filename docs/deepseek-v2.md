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

Each MLA layer chooses its backend at init. The preference is `flashmla`, then
`flash_attn_3`, then `torch`; `flashinfer` serves no MLA layer, as it has no
`varlen_with_lse`. Set `LEAN_VLLM_ATTENTION_BACKEND` to force one.

| Backend | Runs on | How decode reads the cache | CUDA graphs |
|---|---|---|---|
| `flashmla` | H100/H200, with FlashMLA built from source | FlashMLA kernel over the latents | full + piecewise |
| `flash_attn_3` | H100/H200 | expands latents into keys and values | piecewise only |
| `torch` | anything: CPU, Apple Silicon, any CUDA GPU | over the latents, in plain torch | none (eager) |

`flashmla` switches the KV cache to 64-token pages, the only size its kernel
reads. On a non-Hopper GPU only `torch` is available, so there is no fast MLA
path there yet.

The routed experts run a Triton kernel on CUDA and `F.grouped_mm` elsewhere.
Set `LEAN_VLLM_MOE_BACKEND=triton|torch` to force one. The kernel's tile sizes
come from a tuned config file when one matches the shape and GPU, and from
vLLM's defaults otherwise; see [Tuning the MoE kernel](#tuning-the-moe-kernel).

### Engine features

| Feature | Status |
|---|---|
| Chunked prefill, mixed prefill + decode batches | Supported |
| Prefix caching (resuming from cached latents) | Supported |
| OpenAI-compatible server, streaming, metrics | Supported, as for Qwen3 ([online-serving.md](online-serving.md)) |
| bf16 | Used on GPU; fp32 is used in the tests |
| Tensor parallelism | Implemented (attention heads and expert width are sharded); checked at TP=2 on CPU, only run at TP=1 on a GPU |
| Expert parallelism | `--enable-expert-parallel`: each rank holds whole experts, as vLLM does without DP; checked at 2 ranks on CPU, not yet run on GPUs |
| Pipeline parallelism, data parallelism, all-to-all dispatch, expert load balancing | Not supported |
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

The [3 October report](benchmark-2026-10-03-DeepSeek-V2-Lite.md) compares V2-Lite-Chat against
vLLM 0.26.0 on one H100. With no queueing, lean-vLLM's median time per output
token is 6.5 ms against vLLM's 4.6 ms. Under load, lean-vLLM peaks at 23.2
requests/s at load 48 and falls to 21.1 at load 64, while vLLM reaches 31.2.
Large prefill steps now run compiled but outside graphs, and still take 55–63%
of step time at the plateau.

## How it works

### MLA: the cache holds latents

Each token caches `kv_lora_rank + qk_rope_head_dim` values per layer: the
normalized compressed KV plus the shared rope key, with rope already applied.
For V2-Lite that is 576 values, against the 16 × (192 + 128) = 5,120 a plain
KV cache would need, or about 31 KB per token in bf16 across 27 layers.

With `q_lora_rank` set (not V2-Lite), `q_a_proj` and `kv_a_proj_with_mqa` both
read the hidden states, so they load into one `fused_qkv_a_proj` and run as one
GEMM, as in vLLM.

`MLAAttention` (`layers/attention.py`) owns this layout. The runner asks each
layer for its `kv_cache_shape` rather than reading head counts off the config.

**Prefill** expands latents back into per-head keys and values with
`kv_b_proj`, then attends them with an ordinary kernel:

1. The step's new tokens attend each other causally.
2. If a row resumes from cached context, that context is read in chunks of at
   most `max_num_batched_tokens` keys, expanded and attended. Each chunk's
   output is merged into the running result by log-sum-exp, as in vLLM's
   chunked context. On CUDA the merge is one Triton launch, vLLM's
   `merge_attn_states`, written over the running result in place. This bounds
   memory, but the context is re-expanded in every layer.

Values are 128 wide and keys 192. FA3 on Hopper takes them as they are, as
vLLM's MLA prefill does there, and so does `torch`. A backend that takes one
head size (`supports_value_head_size` false, as `flashinfer` here) gets values
zero-padded to 192 and its output cut back to 128.

**Decode** skips the expansion. Since `q · (W_UK c) = (W_UKᵀ q) · c`, each head's
query is projected into latent space and attends the cached latents directly
as one shared key head. The value projection is applied after attention. vLLM
does the same with `W_UK_T` and `W_UV`. Both engines build them once the
weights load (`process_weights_after_loading`); here they are views of
`kv_b_proj`'s weight, whose strides `bmm` takes without a copy. `flashmla` and `torch` implement this
as `mla_decode`. On `flash_attn_3`, decode rows are expanded like prefill.

**Mixed batches** split into decode rows, which use `mla_decode`, and the rest,
which expand. The runner puts one-query rows first, so the split is a slice, as
in vLLM's MLA backends; a one-token prompt chunk decodes too, which attention
cannot tell apart. The split and the chunk plan are computed once per step and
reused by every layer.

### MoE

`FusedMoE` (`layers/moe.py`) stacks the routed experts into one `gate_up_proj`
of shape `[E, 2I, H]` and one `down_proj` of shape `[E, H, I]`. Routing follows
vLLM's `grouped_topk`: softmax scores, `greedy` or `group_limited_greedy`
selection, then `norm_topk_prob` and `routed_scaling_factor`. Shared experts
load into the dense gated MLP, but `FusedMoE` runs them, in the same op as the
routed experts. Their output is added before the all-reduce, so under TP a layer
reduces once rather than twice, as vLLM's `SharedFusedMoE` does. A step of at
most 256 tokens (`LEAN_VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`; 0 turns it
off) runs them on a side CUDA stream beside the routed experts, as in vLLM, so
a decode step's small GEMMs share the GPU. The side stream forks from and joins
the current one, so full decode graphs capture it.

Both expert paths sort token-expert pairs by expert without a host sync. On
CUDA, the Triton kernel in `layers/fused_moe.py` pads each expert's rows to
whole blocks, so a block reads one expert's weights, as vLLM's fused MoE does.
Elsewhere, two `F.grouped_mm` calls do the same work. That path is the
reference the kernel is tested against.

The glue around the two GEMMs is fused as vLLM's is, because a decode graph
replays it in every MoE layer. Routing is one launch after the fp32 gate
(vLLM's `topk_softmax`, extended to group-limited routing). Blocking the pairs
takes three launches in place of `align_blocks`' twenty or so torch ops (vLLM's
`moe_align_block_size`), with no atomics, so a run keeps its pairs in order.
SiLU-and-mul is one launch. Each kernel is tested against the torch code it
replaces.

### Tuning the MoE kernel

As in vLLM, tile sizes are tuned offline, not at runtime. For each batch size,
`benchmarks/tune_moe.py --tune` times vLLM's 1,920-config search space
(`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, warps, stages) on random routing, each
config as 10 calls in one CUDA graph, and keeps the fastest. It writes
`E={experts},N={intermediate size},device_name={GPU}.json`, with the shape as
one rank holds it under `--tp-size` and `--enable-expert-parallel`.

```bash
uv run python benchmarks/tune_moe.py --model ~/workspace/huggingface/DeepSeek-V2-Lite-Chat --tune
uv run python benchmarks/tune_moe.py --model ~/workspace/huggingface/DeepSeek-V2-Lite-Chat    # time what the runtime picks
```

At runtime the layer looks in `$LEAN_VLLM_TUNED_CONFIG_FOLDER`, then in
`lean_vllm/layers/moe_configs/`, and uses the entry for the nearest batch size
in tokens. With no file it logs a warning and falls back to vLLM's bf16
defaults. File names and keys are vLLM's, so a file vLLM's tuner wrote works
here too.

No tuned file ships yet. vLLM ships none for V2-Lite on an H100 either
(`E=64,N=1408` has only a B200 file), so the 20 September comparison ran both
engines on defaults; lean-vLLM's own defaults then were a fixed 16 or 64-row
tile, not vLLM's table.

`GROUP_SIZE_M` orders the launch as vLLM's kernel does: that many row blocks
take each column tile in turn, so they read the same weight tile while it is
in L2.

### Expert parallelism

`--enable-expert-parallel` with `--tensor-parallel-size N` follows vLLM with
no data parallelism: the expert-parallel group is the TP group. Attention, the
dense layers and the shared experts stay tensor-parallel. The routed experts
are not sliced; each rank holds a contiguous run of whole experts, vLLM's
`linear` placement, with the first ranks taking one extra when N does not divide
the expert count.

Every rank still sees every token and routes it with the replicated gate.
`expert_map` takes a global expert id to the rank's local one, or -1. Pairs are
blocked by global id, as in vLLM's `moe_align_block_size`, and a block whose
expert is -1 writes zeros. The ranks' partial sums meet in the all-reduce that
TP already does, so communication is unchanged; what changes is that each rank
runs full-width GEMMs for 1/N of the experts instead of 1/N-width GEMMs for all
of them. Without data parallelism there are no other ranks' tokens to exchange,
so vLLM's all-to-all backends do not apply.

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
  reads a count back to the host. It is an opaque op to `torch.compile`, as in
  vLLM, so the trace does not fix its launch to one batch size.

## Next steps

1. Record the missing correctness checks on an H100: FlashMLA decode against
   the torch reference, mixed FlashMLA steps, and real-weight logits and greedy
   output against a pinned vLLM version.
2. Measure prefill, pure decode, mixed traffic and peak memory separately, and
   the end-to-end gain of latent decode in mixed batches.
3. Profile and optimize: prefill cost, routing, latent projections and context
   gathering. Tune the Triton MoE on an H100 and ship the file.
4. Add features when a workload needs them: an MLA decode kernel beyond Hopper,
   quantization, pipeline parallelism, and data parallelism with all-to-all
   expert dispatch.

Upstream references, tracking vLLM `main`:
[DeepSeek model and MoE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py),
[MLA wrapper](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mla.py),
[MLA execution](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/mla_attention.py).
