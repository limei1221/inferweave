# lean-vLLM

A lightweight vLLM implementation used as a testbed for production inference
engineering: scheduling, attention kernels, model architectures, speculative
decoding, and distributed serving.

The goal is not feature parity with vLLM, but comparable numbers on the parts
that are implemented. Each project ships a design document, a correctness check
against a reference implementation, and a benchmark against vLLM on the same
hardware.

## Roadmap

| | Project | Status |
|---|---|---|
| 0 | Attention backend abstraction | interface, per-layer selection, decode/prefill split; Torch, FlashAttention-3 and FlashInfer backends done, plus Triton and FlashInfer MLA decode (FlashInfer and both MLA backends not yet run on a GPU) |
| 1 | Online serving + advanced scheduler | scheduler, async engine, OpenAI server, metrics and benchmark scripts done; [H100 numbers against vLLM](docs/benchmark-2026-10-03-Qwen3-8B.md) |
| 2 | DeepSeek-style model support: MLA + MoE + YaRN | DeepSeek-V2-Lite checked against transformers; served on an H100 with FlashMLA decode, the Triton MoE and both graph modes ([numbers](docs/benchmark-2026-10-03-DeepSeek-V2-Lite.md)); GPU reference checks still to record |
| 3 | Speculative decoding | DeepSeek-V3's MTP layers draft, verified by rejection sampling, as vLLM's MTP method; checked on CPU against plain decoding and a one-pass drafter on a tiny checkpoint; not yet run on a GPU |
| 4 | Disaggregated prefill / decode | vLLM's connector interface with NixlConnector's pull protocol over TCP, and a proxy; checked end to end on CPU against a single engine ([docs](docs/disaggregated-prefill.md)); not yet run on a GPU |

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # deps, dev tools, the server and the package, into .venv
uv sync --extra cuda     # add FlashAttention-3, FlashInfer, Triton and the torch vLLM pins (NVIDIA only)
uv sync --extra serve    # the server deps alone, for installing without the dev group
```

Dao-AILab publishes no FlashAttention-3 wheel, so the `cuda` extra installs a
third-party build of it, pinned by URL and hash. That build covers Linux on
x86_64, against the torch pinned beside it. The kernels are Hopper's, so the
backend reports itself unavailable on anything but an H100 or H200. The extra
also installs FlashInfer, which runs on Ampere and newer and compiles its
kernels on first use, so it needs `nvcc`. DeepSeek-V2 also decodes with FlashMLA when it is built from source; see
[docs/deepseek-v2.md](docs/deepseek-v2.md).

Without it, and without Triton, the engine runs on CPU via the
`torch` attention backend, at laptop speed — enough to develop and test the
scheduler and cache against a small model.

Only `cuda` and `cpu` are supported. The device is picked automatically (cuda,
then cpu) and can be forced with `LEAN_VLLM_DEVICE`. Off CUDA there is no `mem_get_info` to size the KV
cache from, so it comes from `kvcache_memory_gb` (default 2.0) instead of
`gpu_memory_utilization`.

## Quick start

```bash
uv run hf download Qwen/Qwen3-0.6B --local-dir ~/workspace/huggingface/Qwen3-0.6B
uv run python example.py
```

```python
from lean_vllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, lean-vLLM."], sampling_params)
outputs[0]["text"]
```

Each attention layer picks its backend automatically, and
`LEAN_VLLM_ATTENTION_BACKEND` forces one. A MoE model's routed experts run a Triton kernel
on CUDA, as vLLM's do, and `grouped_mm` elsewhere; `LEAN_VLLM_MOE_BACKEND`
forces either. The kernel's tile sizes come from a config tuned offline by
`benchmarks/tune_moe.py`, or from vLLM's defaults.

Qwen3 and DeepSeek-V2 checkpoints load, picked by `architectures` in
`config.json`. [docs/deepseek-v2.md](docs/deepseek-v2.md) covers DeepSeek-V2-Lite:
its latent KV cache, MoE layer and YaRN rope.

A DeepSeek-V3 checkpoint's multi-token prediction layers can draft tokens for
the model to verify, as vLLM's `--speculative-config` (in bf16: FP8 weights do
not load):

```bash
uv run lean-vllm serve /path/to/deepseek-v3-bf16 --tensor-parallel-size 8 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

The drafter shares the target's embedding and head, and runs after each step's
sampling. As vLLM's padded drafter batch, the kept-draft counts stay on the
device, so the host never waits between the two. With `async_scheduling`, as
vLLM's, the next step is placed as if every draft is kept, verifying
placeholders the device fills, and the device moves it back past those
rejected; only a row that expands latents (FA3) waits for the counts. Its
first pass, over the step's batch, is compiled piecewise and replays the
target's piecewise buckets, as vLLM's drafter; the single-token passes after it
replay full CUDA graphs whenever the target captures its own. A step whose
rows all verify their drafts replays the target's full graphs, which hold
1 + k queries per row, as vLLM's uniform decode.
`/metrics` reports drafts, draft tokens and accepted tokens, by position too.

## Serving

```bash
uv run lean-vllm serve ~/workspace/huggingface/Qwen3-0.6B --port 8000 --served-model-name qwen
```

An OpenAI-compatible server: `/v1/completions`, `/v1/chat/completions` (both
with SSE streaming), `/v1/models`, `/health`, and `/metrics` in Prometheus
format. Requests arrive at any time and
share one token budget per step, so a prompt being prefilled in chunks and a
batch of decoding requests run together; a client that hangs up frees its KV
blocks straight away.

```bash
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "prompt": "Hello, lean-vLLM.", "max_tokens": 32, "temperature": 0}'
```

Any OpenAI client works against it. `example_serving.py` is `example.py`'s two
prompts sent concurrently with the official SDK, reporting each one's
time-to-first-token:

```bash
uv run python example_serving.py
```

Prefill and decode can also run on separate servers behind `lean-vllm proxy`,
with the KV cache pulled from one to the other; see
[docs/disaggregated-prefill.md](docs/disaggregated-prefill.md).

Sampling parameters the engine does not implement (`top_p`, `seed`, penalties,
`n > 1`, and the rest) are refused with a 400 rather than ignored, and a `model`
the server does not serve is a 404 — `/v1/models` lists the name it answers to.
Every engine flag is a `Config` field; `lean-vllm serve --help` lists them.
[docs/online-serving.md](docs/online-serving.md) covers the endpoints, the
scheduling flags, and how `step` overlaps detokenization
and batch preparation with the forward pass.

## Benchmarks

vLLM is the baseline. Every project reports before/after numbers against it on
the same GPU, same model, same request trace — throughput, TTFT, TPOT, and
p50/p95/p99 latency, with the metrics that matter to that project called out.

`benchmarks/bench_offline.py` is the offline throughput number: 256 prompts
submitted at once.

```bash
uv run python benchmarks/bench_offline.py
```

`benchmarks/bench_serving.py` is the online one: requests arrive as a Poisson
process, and it reports goodput, TTFT, TPOT and end-to-end percentiles beside
the failure rate, and scrapes the server's `/metrics` before and after the run.
Point it at a running lean-vLLM or vLLM server.

```bash
uv run python benchmarks/bench_serving.py --dataset lognormal --request-rate 8
```

`benchmarks/sweep.py` runs it across a set of server configurations, starting
and stopping the server for each.

```bash
uv run python benchmarks/sweep.py --model ~/workspace/huggingface/Qwen3-8B \
    --suite rate --rates 1,2,4,8,16 --kvcache-tokens 131072 --out results/8b
```

Published reports, each against vLLM 0.26.0 on one H100:

- [3 October, Qwen3-8B](docs/benchmark-2026-10-03-Qwen3-8B.md): at parity below saturation, 3.7–3.9%
  behind on goodput at the plateau.
- [3 October, DeepSeek-V2-Lite-Chat](docs/benchmark-2026-10-03-DeepSeek-V2-Lite.md): median TPOT 6.5 ms against
  vLLM's 4.6 ms at low load, and peaks at 23.2 requests/s where vLLM reaches
  31.2.

[All 3 October numbers](docs/benchmark-results-2026-10-03.md) are also in one table.

[docs/benchmark-runbook.md](docs/benchmark-runbook.md) is the step-by-step for
producing numbers on a rented H100.

## Tests

```bash
uv run pytest tests/
```

The tests check every available backend against a dense reference that uses
neither SDPA nor paging, so the same suite runs on a laptop and on a GPU box.
Attention backend design and selection are covered in
[docs/attention-backends.md](docs/attention-backends.md).

## Credit

lean-vLLM is a fork of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
by Xingkai Yu, branched at
[`bb823b3`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8).
The scheduler, block manager, paged KV cache, and Qwen3 implementation are its
work. MIT licensed, as is this fork.
