# Online serving

`lean-vllm serve` puts the engine behind an OpenAI-compatible HTTP API. Tokens
stream back as they are produced, and one token budget per step decides what
runs: prefill chunks and decode rows together.

## Start a server

```bash
uv sync --extra serve
uv run lean-vllm serve ~/workspace/huggingface/Qwen3-8B --port 8000 --served-model-name qwen
```

`--served-model-name` is the id the server answers to; without it, the model
path is. Every `Config` field is a flag; `lean-vllm serve --help` lists them.

Any OpenAI client works, and no key is checked:

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

## API

| endpoint | |
| --- | --- |
| `POST /v1/completions` | prompt as a string or token ids |
| `POST /v1/chat/completions` | messages, through the chat template |
| `GET /v1/models` | the id this server serves |
| `GET /health` | 503 once the engine thread dies |
| `GET /metrics`, `/metrics.json` | Prometheus text, or a JSON summary |

Supported fields: `model`, `prompt` / `messages`, `max_tokens` (default 64),
`temperature` (default 1.0; 0 is greedy), `stream`, `stream_options`, `stop`,
and `n` only as 1. Two extras: `ignore_eos` generates the full `max_tokens`,
and `priority` orders requests under `--scheduling-policy priority` (lower runs
first).

Every other field (`top_p`, `seed`, penalties, `logprobs`, `tools`, ...) is
**refused with a 400, not ignored**, since ignoring it returns wrong output
silently. Refusal is by value: `"top_p": 1.0` asks for nothing and passes.

| status | when |
| ---: | --- |
| 400 | an unsupported field, or prompt + `max_tokens` over the context |
| 404 | a `model` this server does not serve |
| 429 | the waiting queue is full (`--max-waiting-requests`) |
| 503 | the engine died, or the request can never fit in the cache |
| 504 | the request waited past `--request-timeout` without running |

Once a stream has sent its 200, later errors arrive as an SSE error event.

## Scheduling

Each step schedules running sequences first (one token per decode, the next
chunk per unfinished prefill), then admits waiting requests into the budget
left. Preemption recomputes rather than swaps, and puts the sequence back at
the head of the queue. A sequence that cannot fit even alone is dropped.

| flag | default | |
| --- | ---: | --- |
| `--max-num-batched-tokens` | 8192 | tokens per step |
| `--max-num-seqs` | 1024 | sequences running at once |
| `--enable-chunked-prefill` | on | off runs whole prompts, never mixed with decode |
| `--long-prefill-token-threshold` | 0 | per-step token cap on one prompt, as in vLLM V1; 0 is none |
| `--scheduling-policy` | `fcfs` | or `priority` |
| `--max-waiting-requests` | 0 | refuse arrivals with a 429 past this queue length; 0 is none |
| `--request-timeout` | 0 | drop a request never scheduled after this many seconds, with a 504; 0 is none |
| `--num-kvcache-blocks` | profiled | pin it to hold cache capacity fixed across runs |
| `--kvcache-block-size` | 16 | tokens per block |
| `--enable-prefix-caching` | on | see [Prefix caching](#prefix-caching) |
| `--async-scheduling` | on | see [Pipelined steps](#pipelined-steps) |
| `--cudagraph-mode` | `full_and_piecewise` | see [CUDA graphs](#cuda-graphs) |
| `--enforce-eager` | off | same as `--cudagraph-mode none` |

Admission control is off by default, as in vLLM, so overload shows up in p99
latency rather than a rejection rate. Read goodput alongside p99.

## Pipelined steps

`step()` launches one step and then drains the one before it, so the drain's
host work runs while the GPU computes. `--async-scheduling` decides how much
overlaps:

```text
off                              on
---                              --
await tokens of step k-1         schedule step k
reconcile step k-1               launch step k
schedule step k                  await tokens of step k-1
launch step k                    reconcile step k-1
detokenize step k-1              detokenize step k-1
```

Off, only detokenization overlaps. On, scheduling and batch preparation do too,
at a cost: a stop is seen one step late, so a request that hits EOS or a stop
string has one extra token computed and discarded. Requests ending at
`max_tokens` do not pay it.

Two details keep this safe:

- `SampledTokens` copies sampled tokens to pinned memory on a separate CUDA
  stream, so awaiting step k-1 does not block on step k's kernels.
- All other device work stays on one stream, so a later step always runs after
  an earlier step's KV writes. That is what makes it safe to free or share a KV
  block while a step is in flight.

On by default, as in vLLM; it turns itself off under tensor parallelism, where
other ranks never see the sampled tokens. On an H100 it cuts offline GPU idle
from 22.4% to 3.2% and adds 7–9% goodput on Qwen3-8B
([benchmark](benchmark-2026-09-13.md)).

## CUDA graphs

| step | `full_and_piecewise` (default) runs it as |
| --- | --- |
| pure decode, up to 512 rows | one full graph |
| prefill or mixed, 64–512 tokens | piecewise graphs |
| anything else | eager |

`full` leaves all prefill and mixed steps eager; `piecewise` also sends decode
through piecewise graphs; `none` captures nothing.

Piecewise means each decoder layer is captured in two pieces, before and after
attention. Attention stays eager because FlashAttention's varlen call depends
on the sequence layout, which changes every step. The pieces are captured with
`torch.cuda.CUDAGraph`, not torch.compile, so they save kernel launch cost but
fuse nothing.

A step pads up to its token bucket, and padding costs real compute. Past 512
tokens that cost exceeds the launch overhead saved, so large steps run eager;
vLLM compiles them instead. At plateau load on Qwen3-8B those eager steps take
about half the step time.

Things to know:

- Greedy output can differ slightly from eager, because padding changes which
  cuBLAS kernel runs. Replay matches the same pieces run eagerly at the same
  padded width bitwise.
- Nothing inside a piece may sync with the host (`.item()`, `.tolist()`,
  `.cpu()`); re-check after changing `layers/` or `models/`.
- Graph memory comes on top of the KV cache rather than out of it, so every mode
  gets the same cache, but a tight card can run out of memory at capture.

## Prefix caching

A prompt is split into blocks, each hashed with the chain of blocks before it.
A request whose leading blocks are cached reuses them and prefills only the
rest.

- Hashes are computed once, as tokens arrive, never inside a step.
- The last block is always recomputed, since attention needs a query token.
- Freed blocks are evicted deepest first, so shared prefixes outlive their
  suffixes.
- Caching changes what a request costs, never when it runs.

`--prefix-caching-hash-algo` is `sha256` by default, or `xxhash` for speed when
every client is trusted; a collision could serve one tenant another's tokens.
The tokens behind a hit are compared before reuse either way.
`--no-enable-prefix-caching` turns it all off, for A/B runs.

## Metrics

`/metrics.json` reports step count and time, batch tokens, the prefill/decode
split, steps by graph kind, preemptions, prefix-cache hit rate, peak KV usage,
and counts and means for TTFT, TPOT, queue delay and end-to-end latency. Names
mirror vLLM's under `lean_vllm:`, except that the prefix-cache counters count
blocks where vLLM's count tokens. Compute percentiles on the client; the
histogram buckets are too coarse.

Steps are counted as `graph`, `piecewise`, or an eager reason: `prefill`
(a prefill or mixed step outside the piecewise range), `decode` (a decode step
no graph covers: past the full-graph sizes, or under `piecewise` alone outside
its range) or `enforced` (`none`).

`model_busy_fraction`, the share of wall clock inside a forward pass, is the
utilization to trust; the nvidia-smi figure beside it counts any live kernel as
busy.

## Benchmarks

```bash
uv run python benchmarks/bench_serving.py --dataset lognormal --request-rate 8
uv run python benchmarks/sweep.py --model ~/workspace/huggingface/Qwen3-8B \
    --suite rate --rates 1,2,4,8,16 --kvcache-tokens 327680 --out results/8b
```

`bench_serving.py` sends open-loop Poisson arrivals through the OpenAI SDK and
works against vLLM unchanged; `sweep.py` runs it across arms with a fresh server
per run. Only `--dataset prefix` shares prompt prefixes, for `--suite prefix`.

See [benchmark-runbook.md](benchmark-runbook.md) for a fresh GPU box and
[the latest report](benchmark-2026-09-13.md) for numbers.

## How it fits together

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

The engine is synchronous. Its loop runs on the event loop and hands each
`step()` to a single worker thread, because a step blocks for a whole forward
pass and would starve the HTTP handlers. Requests and aborts reach the engine on
the event loop, between steps: an add waits for the step in flight, and an abort
that arrives during one is applied as it returns.

There is no in-process restart. An unhandled engine exception fails every
outstanding request, flips `/health` to 503, and exits for a supervisor to
restart.
