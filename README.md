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
| 0 | Attention backend abstraction | interface + Torch/FlashAttention backends done |
| 1 | Online serving + advanced scheduler | scheduler, async engine, OpenAI server, metrics and benchmark scripts done; [H100 numbers against vLLM](docs/benchmark-2026-09-13.md) |
| 2 | DeepSeek-style model support: MLA + MoE + YaRN | DeepSeek-V2-Lite checked against transformers; served on an H100 with FlashMLA decode, the Triton MoE and both graph modes ([numbers](docs/benchmark-2026-09-20.md)); GPU reference checks still to record |
| 3 | Speculative decoding | |
| 4 | Disaggregated prefill / decode | |

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # deps, dev tools, the server and the package, into .venv
uv sync --extra cuda     # add FlashAttention-3, Triton and the torch vLLM pins (NVIDIA only)
uv sync --extra serve    # the server deps alone, for installing without the dev group
```

Dao-AILab publishes no FlashAttention-3 wheel, so the `cuda` extra installs a
third-party build of it, pinned by URL and hash. That build covers Linux on
x86_64, against the torch pinned beside it. The kernels are Hopper's, so the
backend reports itself unavailable on anything but an H100 or H200.
DeepSeek-V2 also decodes with FlashMLA when it is built from source; see
[docs/deepseek-v2.md](docs/deepseek-v2.md).

Without it, and without Triton, the engine runs on CPU and Apple Silicon via the
`torch` attention backend, at laptop speed — enough to develop and test the
scheduler and cache against a small model.

The device is picked automatically (cuda, then mps, then cpu) and can be forced
with `LEAN_VLLM_DEVICE`. Off CUDA there is no `mem_get_info` to size the KV
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

The attention backend is picked automatically and can be forced with
`LEAN_VLLM_ATTENTION_BACKEND`. A MoE model's routed experts run a Triton kernel
on CUDA, as vLLM's do, and `grouped_mm` elsewhere; `LEAN_VLLM_MOE_BACKEND`
forces either.

Qwen3 and DeepSeek-V2 checkpoints load, picked by `architectures` in
`config.json`. [docs/deepseek-v2.md](docs/deepseek-v2.md) covers DeepSeek-V2-Lite:
its latent KV cache, MoE layer and YaRN rope.

## Serving

```bash
uv run lean-vllm serve ~/workspace/huggingface/Qwen3-0.6B --port 8000 --served-model-name qwen
```

An OpenAI-compatible server: `/v1/completions`, `/v1/chat/completions` (both
with SSE streaming), `/v1/models`, `/health`, and `/metrics` in Prometheus
format (plus `/metrics.json` for the same numbers as a summary). Requests arrive at any time and
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

Sampling parameters the engine does not implement (`top_p`, `seed`, penalties,
`n > 1`, and the rest) are refused with a 400 rather than ignored, and a `model`
the server does not serve is a 404 — `/v1/models` lists the name it answers to.
Every engine flag is a `Config` field; `lean-vllm serve --help` lists them.
[docs/online-serving.md](docs/online-serving.md) covers the endpoints, the
scheduling flags, admission control, and how `step` overlaps detokenization
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
the rejection rate. Point it at a running lean-vLLM or vLLM server.

```bash
uv run python benchmarks/bench_serving.py --dataset lognormal --request-rate 8
```

`benchmarks/sweep.py` runs it across a set of server configurations, starting
and stopping the server for each.

```bash
uv run python benchmarks/sweep.py --model ~/workspace/huggingface/Qwen3-8B \
    --suite rate --rates 1,2,4,8,16 --num-kvcache-blocks 8192 --out results/8b
```

No benchmark numbers are published yet. The backends were verified for
numerical correctness against a dense reference on an A100 back when the CUDA
one was FlashAttention-2; the FlashAttention-3 backend that replaced it awaits
its first H100.
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
