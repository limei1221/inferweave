# Online serving

`lean-vllm serve` puts the engine behind an OpenAI-compatible HTTP API. Requests
arrive at any time and share one token budget per step, so prompt chunks and
decoding requests run in the same batch. Tokens stream back as they are
produced.

It serves every model the engine loads (Qwen3 and DeepSeek-V2, see
[deepseek-v2.md](deepseek-v2.md)), one model per server. On Qwen3-8B on an H100
it matches vLLM below saturation and trails it by 5–7% at the plateau
([13 September report](benchmark-2026-09-13.md)).

## What is supported

### Endpoints

| Endpoint | What it does |
| --- | --- |
| `POST /v1/completions` | Completes a prompt, given as a string or token ids |
| `POST /v1/chat/completions` | Completes `system` / `user` / `assistant` messages through the chat template |
| `GET /v1/models` | Lists the one model id this server answers to |
| `GET /health` | 200, or 503 once the engine thread has died |
| `GET /metrics` | Prometheus text |
| `GET /metrics.json` | The same numbers as a JSON summary |

Both completion endpoints stream over Server-Sent Events (SSE) with
`"stream": true`. No API key is checked.

### Request fields

| Field | Support |
| --- | --- |
| `model`, `prompt` / `messages` | Required |
| `max_tokens` | Default 64 |
| `temperature` | Default 1.0; 0 is greedy |
| `stream`, `stream_options.include_usage` | Supported |
| `stop` | A string or a list of strings |
| `n` (completions per prompt) | 1 only |
| `ignore_eos` (extra) | Generates the full `max_tokens` |
| `priority` (extra) | Lower runs first, under `--scheduling-policy priority` |
| `kv_transfer_params` (extra) | Disaggregated prefill, as vLLM's; see [disaggregated-prefill.md](disaggregated-prefill.md) |
| `top_p`, `top_k`, `min_p`, `seed`, penalties, `logprobs`, `logit_bias`, `tools`, `echo`, `suffix`, `best_of` | **Refused with a 400** |

Unsupported fields are refused rather than ignored, because ignoring them would
silently return the wrong output. A field set to its no-op value, such as
`"top_p": 1.0`, is accepted, since many clients send those by default.

### Errors

| Status | When |
| ---: | --- |
| 400 | An unsupported field, or prompt + `max_tokens` longer than the context |
| 404 | A `model` this server does not serve |
| 429 | The waiting queue is full (`--max-waiting-requests`) |
| 503 | The engine died, or the request can never fit in the KV cache |
| 504 | The request waited past `--request-timeout` without being scheduled |

Once a stream has sent its 200, a later error arrives as an SSE error event. A
client that disconnects frees its KV blocks straight away.

### Engine features

| Feature | Default | Notes |
| --- | --- | --- |
| Chunked prefill, mixed prefill + decode steps | On | Off runs whole prompts, never mixed with decode |
| Prefix caching | On | Reuses cached prompt blocks |
| Async scheduling | On | Schedules the next step while the GPU runs the current one; turns itself off under tensor parallelism |
| CUDA graphs | `full_and_piecewise` | Full graphs for decode, piecewise for small prefill and mixed steps |
| `torch.compile` | On, unless graphs are off | Inductor compiles the model between attention ops, as vLLM does |
| Priority scheduling | Off (`fcfs`) | `--scheduling-policy priority` |
| Admission control, queue timeout | Off | `--max-waiting-requests`, `--request-timeout` |
| Preemption | Recompute | The sequence goes back to the head of the queue; there is no swapping to CPU |
| Tensor parallelism | 1 | Up to 8 GPUs |
| Expert parallelism | Off | MoE models, over the tensor-parallel GPUs |

Not supported: serving several models from one server, restarting the engine in
place, and the sampling features refused above.

## Running it

```bash
uv sync --extra serve
uv run lean-vllm serve ~/workspace/huggingface/Qwen3-8B --port 8000 --served-model-name qwen
```

`--served-model-name` is the id the server answers to. Without it, the id is
the model path. Every `Config` field is a flag, and `lean-vllm serve --help`
lists them.

Any OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
completion = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "introduce yourself"}],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

`example_serving.py` streams two prompts concurrently and reports each one's
time to first token.

### Flags

| Flag | Default | What it controls |
| --- | ---: | --- |
| `--max-num-batched-tokens` | 8192 | Token budget per step |
| `--max-num-seqs` | 1024 | Sequences running at once |
| `--max-model-len` | 4096 | Context length, capped at the model's own |
| `--enable-chunked-prefill` | on | Mixing prompt chunks with decode |
| `--long-prefill-token-threshold` | 0 | Per-step token cap on one prompt, as in vLLM V1; 0 is none |
| `--scheduling-policy` | `fcfs` | Or `priority` |
| `--max-waiting-requests` | 0 | Queue length past which arrivals get a 429; 0 is none |
| `--request-timeout` | 0 | Seconds a request may wait unscheduled before a 504; 0 is none |
| `--gpu-memory-utilization` | 0.9 | Share of GPU memory for weights, activations and KV cache |
| `--num-kvcache-blocks` | profiled | Pins the cache size, to hold it fixed across runs |
| `--kvcache-block-size` | 16 | Tokens per KV block |
| `--enable-prefix-caching` | on | See [Prefix caching](#prefix-caching) |
| `--prefix-caching-hash-algo` | `sha256` | Or `xxhash` |
| `--async-scheduling` | on | See [Pipelined steps](#pipelined-steps) |
| `--cudagraph-mode` | `full_and_piecewise` | Or `full`, `piecewise`, `none`; see [CUDA graphs](#cuda-graphs) |
| `--enforce-eager` | off | Same as `--cudagraph-mode none` |
| `--tensor-parallel-size` | 1 | GPUs per model, up to 8 |
| `--enable-expert-parallel` | off | MoE models: each GPU holds whole experts, not a slice of each ([deepseek-v2.md](deepseek-v2.md#expert-parallelism)) |
| `--kv-transfer-config` | none | JSON; makes the server a prefill or decode instance ([disaggregated-prefill.md](disaggregated-prefill.md)) |

Admission control is off by default, as in vLLM, so overload shows up as p99
latency rather than rejections. Read goodput alongside p99.

## Performance

The [13 September report](benchmark-2026-09-13.md) compares lean-vLLM with
vLLM 0.26.0 on Qwen3-8B on one H100:

- Below saturation the two are at parity: identical goodput at load 1, and
  median TPOT 3% higher.
- At the plateau lean-vLLM reaches about 24.1–24.5 requests/s against vLLM's
  25.7–26.1.
- Async scheduling adds 7–9% goodput under load.
- The remaining gap was mostly large prefill steps, which ran eager here and
  compiled in vLLM. They now run compiled here too; the report predates that.

DeepSeek-V2-Lite has its own [20 September report](benchmark-2026-09-20.md).

## How it works

```text
     HTTP (FastAPI / uvicorn)      <- tokenize, stop strings, SSE
               |
         AsyncLLMEngine            <- per-request asyncio.Queue
               |  (event loop -> worker thread)
      step() on one worker         <- detokenize
               |
           Scheduler               <- one token budget per step
               |
          ModelRunner              <- one mixed batch
```

The engine is synchronous. Its loop runs on the asyncio event loop and hands
each `step()` to a single worker thread, since a step blocks for a whole forward
pass and would otherwise starve the HTTP handlers. New requests and aborts reach
the engine between steps. An abort that arrives during a step is applied when
the step returns.

If the engine raises, every outstanding request fails, `/health` turns 503, and
the process exits for a supervisor to restart.

### Scheduling

Each step first schedules the running sequences: one token for each decode, and
the next chunk for each unfinished prompt. Waiting requests are then admitted
into whatever budget is left. When the KV cache runs out, a sequence is
preempted: its blocks are freed, and it is recomputed later from the head of the
queue. A sequence that cannot fit even on its own is dropped with a 503.

### Pipelined steps

`step()` launches one step and then finishes the one before it, so host work
overlaps the GPU. `--async-scheduling` decides how much overlaps:

```text
off                              on
---                              --
await tokens of step k-1         schedule step k
reconcile step k-1               launch step k
schedule step k                  await tokens of step k-1
launch step k                    reconcile step k-1
detokenize step k-1              detokenize step k-1
```

With it off, only detokenization overlaps. With it on, scheduling and batch
preparation overlap too, but a stop is seen one step late. A request that ends
on EOS or a stop string therefore has one extra token computed and discarded.
Requests that end at `max_tokens` do not pay this.

Two details keep it safe:

- `SampledTokens` copies sampled tokens to pinned memory on a separate CUDA
  stream, so waiting for step k-1 does not block on step k's kernels.
- All other device work stays on one stream, so a step always runs after the
  previous step's KV writes. That makes it safe to free or share a KV block
  while a step is in flight.

Tensor parallelism turns it off, because the other ranks never see the sampled
tokens. On an H100 it cuts offline GPU idle time from 22.4% to 3.2%.

### CUDA graphs

| Step | Under `full_and_piecewise` (default) |
| --- | --- |
| Pure decode, up to 512 rows | One full graph |
| Prefill or mixed, 64–512 tokens | Piecewise graphs |
| Anything else | Compiled, no graph |

`full` runs all prefill and mixed steps compiled, without a graph. `piecewise`
also sends decode steps of 64–512 rows through piecewise graphs, so smaller
decode steps run without one. `none` compiles and captures nothing, so every
step runs eager.

As in vLLM, the model is traced once with `torch.compile` and split at each
attention op. Inductor compiles each piece between two attention ops, fusing
its elementwise work. Attention runs eager between the pieces, because its
inputs change shape every step. Each piece is then captured as a CUDA graph per
bucket.

A step is padded up to its graph's size, and the padding costs real compute.
Past 512 tokens that cost outweighs the launch overhead saved, so large steps
run the compiled pieces without a graph, as vLLM's do.

Things to know:

- The first start compiles the model. Inductor caches the result on disk, so
  later starts are faster.
- Greedy output can differ slightly from eager mode, because padding changes
  which cuBLAS kernel runs and Inductor fuses differently.
- The model must trace as one graph, with no Python branch on the token count;
  a trace that fixes the count fails at startup. Code inside a piece must not
  sync with the host (`.item()`, `.tolist()`, `.cpu()`). Re-check both after
  changing `layers/` or `models/`.
- Graph memory comes on top of the KV cache, so every mode gets the same cache
  size, but a tight GPU can run out of memory during capture.

### Prefix caching

A prompt is split into blocks, and each block is hashed together with the chain
of blocks before it. A request whose leading blocks are already cached reuses
them and prefills only the rest.

- Hashes are computed as tokens arrive, never inside a step.
- The last block is always recomputed, since attention needs at least one query
  token.
- Freed blocks are evicted deepest first, so shared prefixes outlive their
  suffixes.
- Caching changes what a request costs, never when it is scheduled.

The hash is `sha256` by default. `xxhash` is faster, but use it only when every
client is trusted: a collision could serve one tenant another's tokens. Either
way, the tokens behind a hit are compared before reuse.
`--no-enable-prefix-caching` turns caching off for A/B runs.

### Metrics

`/metrics.json` reports:

- steps: count, time, batch tokens, the prefill/decode split, and how each step
  ran;
- preemptions, prefix-cache hit rate, and peak KV usage;
- counts and means of TTFT, TPOT, queue delay and end-to-end latency.

Names mirror vLLM's under the `lean_vllm:` prefix, except that the prefix-cache
counters count blocks where vLLM's count tokens. Compute latency percentiles on
the client, as the histogram buckets are too coarse.

Each step is counted as `graph`, `piecewise`, or, when no graph covers it,
with a reason:

- `prefill`: a prefill or mixed step outside the piecewise range, run compiled;
- `decode`: a decode step no graph covers, run compiled;
- `enforced`: graphs and compilation are off, so the step ran eager.

For GPU utilization, trust `model_busy_fraction`, the share of wall-clock time
spent inside a forward pass. The nvidia-smi figure beside it counts any running
kernel as busy.

## Benchmarking

```bash
uv run python benchmarks/bench_serving.py --dataset lognormal --request-rate 8
uv run python benchmarks/sweep.py --model ~/workspace/huggingface/Qwen3-8B \
    --suite rate --rates 1,2,4,8,16 --kvcache-tokens 327680 --out results/8b
```

`bench_serving.py` sends Poisson arrivals through the OpenAI SDK, so it works
against vLLM unchanged. `sweep.py` runs it across server configurations,
starting a fresh server for each run. Only `--dataset prefix` shares prompt
prefixes, for `--suite prefix`.

[benchmark-runbook.md](benchmark-runbook.md) walks through a run on a fresh GPU
box.
