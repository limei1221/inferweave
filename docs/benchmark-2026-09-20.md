# Online serving benchmark — 20 September 2026

lean-vLLM against vLLM 0.26.0 on DeepSeek-V2-Lite-Chat (16B total, 2.4B active,
MLA + MoE + YaRN), one H100, with chunked prefill and full + piecewise CUDA
graphs on both engines — including MLA decode inside the full graph on both.
lean-vLLM ran two curves, async scheduling off and on; vLLM ran its default,
which is on. lean-vLLM results are in
[`results/20260920T152524Z/`](../results/20260920T152524Z/); the vLLM curve is
reused from [`results/20260920T125149Z/`](../results/20260920T125149Z/), since
the change measured here touches lean-vLLM alone.

The ten lean-vLLM runs completed — five offered loads per curve, 1,000 requests
each, no rejections, failures, or preemptions — replaying the same lognormal
trace (~661k prompt, ~177k generated tokens) as the reused vLLM curve.

This run follows a change that brings lean-vLLM's MLA decode into the full CUDA
graph; the earlier report on the eager-decode baseline is in
[`results/20260920T125149Z/`](../results/20260920T125149Z/). The main findings
are:

- **The eager decode floor is gone.** At load 1, where the engines share a
  workload with no queueing, lean-vLLM's median TPOT is 7.0 ms against vLLM's
  4.4 ms — 1.6× per generated token, down from 7.9× when decode ran eager. Under
  load the goodput gap is 12–36%, from 46–62% before.
- **lean-vLLM saturates around load 24 and plateaus at ~20 requests/s.** Its
  goodput levels off at ~17.6 requests/s by load 24 and ~20.3 by load 32, then
  climbs no further; vLLM is still rising at load 64, reaching 31.8 requests/s.
  At the plateau lean-vLLM produces ~3,570–3,590 output tokens/s to vLLM's
  ~5,300–5,600 — a step up from the ~2,100 it managed with eager decode.
- **Server counters confirm decode now replays a captured graph.** No step runs
  as eager decode anymore; the only eager kind left is prefill past the
  512-token grid. Captured graphs (full decode plus small piecewise pieces)
  carry 44–58% of step time; eager prefill carries the rest.
- **Async scheduling barely moves the online curve.** It adds only 1.7–5.3% over
  the off curve at loads 24–64, because lean-vLLM is GPU-bound here, not host-
  bound: its step loop is already busy ~94–95% of the time with async off.
- **What remains is prefill and an earlier knee.** Eager prefill steps run
  ~60–69 ms and now dominate eager time, and lean-vLLM saturates at ~20
  requests/s while vLLM scales to ~32 — so the remaining gap is prefill cost and
  headroom, not the decode step.

## Setup

### Hardware and software

| Component | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3, 81,559 MiB |
| NVIDIA driver | 580.126.09 |
| Linux kernel | `6.8.12-680-6063-coreweave-amd64-f81899c8` |
| Python | 3.12.11 |
| PyTorch, both engines | `2.11.0+cu130` |
| lean-vLLM | `feature/deepseek-v2-lite` with MLA decode captured in the full graph (`supports_full_cudagraph_mla_decode` on, `capture_cudagraph` sizing the schedule at `max_model_len`) |
| FlashMLA | commit `ba89a34` (deepseek-ai HEAD), built from source against the pinned torch |
| vLLM | 0.26.0 |
| MLA decode kernel | lean-vLLM: FlashMLA (`flashmla`); vLLM: `FLASH_ATTN_MLA` (prefill `FLASH_ATTN`) |
| Persistence mode | Enabled |
| Application graphics clock | 1,980 MHz, equal to the maximum |
| Power limit | 700 W (maximum 700 W) |

### Workload and server settings

| Setting | Both engines |
| --- | --- |
| Model | DeepSeek-V2-Lite-Chat, bf16, 27 layers |
| Requests per run | 1,000, plus 3 warmup requests |
| Workload | Lognormal lengths: input 512, output 128, σ = 0.8 |
| Sampling | Greedy, seed 0 |
| Batch token budget / maximum sequences | 8,192 / 256 |
| Chunked prefill | Enabled |
| Graph mode | Full + piecewise (requested) |
| KV cache | 327,680 tokens; lean-vLLM uses 64-token blocks, FlashMLA's page |
| Maximum model length | 4,096 |
| Offered loads | 1, 24, 32, 48, 64 requests/s |

vLLM's async scheduling was confirmed on in every server log
([`vllm-async-scheduling.txt`](../results/20260920T125149Z/vllm-async-scheduling.txt)),
so the like-for-like pair is lean-vLLM's `async-scheduling=True` curve against
vLLM. Both engines decoded with an MLA kernel, not an expanded fallback
([`vllm-mla-backend.txt`](../results/20260920T125149Z/vllm-mla-backend.txt)).
vLLM ran with prefix caching on, its default; lean-vLLM recorded a 0.0
prefix-cache hit rate on this trace, so prefix caching should not favour either
engine.

### How the two engines execute a step

Both engines now capture MLA decode inside the full graph; the remaining
difference is only in how prefill and mixed steps run.

| | lean-vLLM | vLLM 0.26.0 |
| --- | --- | --- |
| MLA decode (any batch up to `max_num_seqs`) | Inside the full CUDA graph (FlashMLA) | Inside the full CUDA graph |
| Prefill / mixed steps, 64–512 tokens | Piecewise graphs; attention eager | Inductor pieces |
| Prefill / mixed over 512 tokens | Eager | Inductor-compiled |
| Async scheduling | Off and on | On by default |

lean-vLLM runs `full_and_piecewise` with the decode step captured. The FlashMLA
HEAD here fuses the decode schedule into the kernel; rather than pin the older
split API, lean-vLLM warms up and captures at worst-case sequence lengths
(`context_lens = max_model_len`), so the tile schedule and split-KV workspace
baked into the graph are sized for the longest sequence, and each replay refreshes
`context_lens`/`block_tables` while the kernel gates its KV loop on them. Pure
decode batches (up to `max_num_seqs`) therefore replay the whole model, attention
included; prefill and mixed steps still fall to piecewise graphs or eager.
See [`docs/benchmark-runbook.md`](../docs/benchmark-runbook.md) for the mechanism.

### Reading the results

| Metric | Meaning | Better direction |
| --- | --- | --- |
| Goodput | Completed requests divided by total run time, including queue drain; no latency cutoff | Higher |
| TTFT | Time to first token | Lower |
| TPOT | Time per output token after the first | Lower |
| E2E | End-to-end request time | Lower |
| p50 / p99 | Median / 99th percentile | Lower for latency |

Goodput counts the drain after the last arrival, so it sits below the offered
load even when the engine keeps up. Compare engines at the same load, not
against the offered rate.

## 1. Throughput

| Offered load (req/s) | lean-vLLM off | lean-vLLM on | vLLM | on vs vLLM | on vs off |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.01 | 1.01 | 1.01 | −0.0% | +0.0% |
| 24 | 17.29 | 17.58 | 19.97 | **−12.0%** | +1.7% |
| 32 | 19.29 | 20.32 | 24.50 | −17.1% | +5.3% |
| 48 | 19.82 | 20.20 | 29.93 | −32.5% | +1.9% |
| 64 | 19.89 | 20.29 | **31.81** | **−36.2%** | +2.0% |

Goodput is in requests/s. Output token throughput follows goodput, since every
run generates the same tokens: at load 48 and 64 lean-vLLM on produces 3,572 and
3,588 tok/s, lean-vLLM off 3,506 and 3,518, and vLLM 5,292 and 5,625.

lean-vLLM reaches its plateau by load 32 — goodput moves only from 17.58 to
20.29 across loads 24–64 — so its saturation knee sits near the bottom of this
bracket, up from ~12 requests/s when decode ran eager. vLLM keeps climbing to
load 64. The gap still opens once the engines are asked for more than lean-vLLM
can serve, but it is far narrower: 12–36% here against 46–62% before.

Async scheduling adds little (+1.7% to +5.3%), unlike the 7–9% it was worth on
dense Qwen3-8B. lean-vLLM is GPU-bound here, not host-bound: pipelining host work
cannot help when the host is not the limit (section 3).

## 2. Latency

| Offered load (req/s) | p50 TTFT on / vLLM (ms) | p99 TTFT on / vLLM (s) | p50 TPOT on / vLLM (ms) | p99 E2E on / vLLM (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 49 / 37 | 0.16 / 0.09 | **7.0 / 4.4** | 7.99 / 4.25 |
| 24 | 167 / 98 | 1.05 / 1.40 | 30.7 / 17.4 | 24.56 / 15.14 |
| 32 | 276 / 130 | 1.14 / 0.41 | 38.0 / 19.2 | 26.19 / 15.51 |
| 48 | 3,217 / 416 | 10.18 / 1.02 | 50.5 / 22.4 | 31.97 / 16.04 |
| 64 | 6,310 / 1,331 | 14.96 / 3.59 | 48.8 / 28.1 | 33.29 / 18.92 |

Load 1 isolates per-step overhead, and it is where the change shows cleanest:
with no queue on either engine, lean-vLLM now spends 7.0 ms per generated token
to vLLM's 4.4 ms, 1.6×, down from 34.4 ms (7.9×) when decode ran eager. A
DeepSeek-V2-Lite decode step is a single token per sequence, so this is the cost
of the decode step now that MLA attention and the model around it replay from one
captured graph rather than dispatching eagerly.

Under load, median TPOT stays 1.7–2.3× vLLM's: lean-vLLM sustains a larger batch
(~20 vs ~12 requests/s before) rather than a faster per-token step, so the win
shows as throughput, not lower TPOT. The TTFT tail still blows out once lean-vLLM
is past its knee — p99 TTFT is 10.2 s at load 48 against vLLM's 1.0 s — because
requests queue behind a plateau that cannot drain them. The one place lean-vLLM
looks better, p99 TTFT at load 24 (1.05 s vs 1.40 s), is a noisy vLLM point —
vLLM's own p99 TTFT falls to 0.41 s at load 32 — not a lean-vLLM win.

Against its own off curve, async scheduling helps latency about as little as it
helps throughput: at load 48 it moves p50 TPOT from 51.0 ms to 50.5 ms.

## 3. Where the step time goes

lean-vLLM's server counters split each step by how it ran. `graph` is a
pure-decode batch replaying the whole model from one captured graph, attention
included; `piecewise` is a small prefill or mixed step replaying per-piece graphs
with attention eager between them; `prefill` is a prefill or mixed step past the
512-token grid, run eager. Figures are for the measured run, warmup subtracted:

| | load 24, on | load 48, off | load 48, on |
| --- | ---: | ---: | ---: |
| Steps | 2,805 | 2,152 | 2,146 |
| Captured (decode + piecewise) steps | 2,424 (86.4%) | 1,766 (82.1%) | 1,766 (82.3%) |
| Eager prefill steps | 381 (13.6%) | 386 (17.9%) | 380 (17.7%) |
| Full-graph decode step time | 29.0 s (53.8%) | 20.9 s (43.1%) | 19.8 s (42.2%) |
| Piecewise step time | 2.0 s (3.7%) | 1.2 s (2.4%) | 0.8 s (1.7%) |
| Eager prefill step time | 22.9 s (42.5%) | 26.3 s (54.5%) | 26.3 s (**56.0%**) |
| Mean prefill step | 60 ms | 68 ms | 69 ms |

Two things stand out. First, no step runs as eager decode anymore: the `graph`
kind — pure decode through the full model graph — is the single largest time
bucket at load 24 (54%) and still 42% at the plateau, and at load 1 the same step
is 7.0 ms (§2), where the eager path spent ~35 ms. Second, the only eager kind
left is prefill past the 512-token grid, ~60–69 ms per step, and as load rises it
grows to ~55% of step time. That makes the eager prefill step — not decode — the
dominant remaining cost, and the next lever on the gap to vLLM.

Captured graphs hold 82–86% of steps, and `model_busy_fraction` is ~0.94–0.95, so
the step loop is rarely idle between steps. That is why async scheduling, which
targets the gaps *between* steps, barely changes the online curve (§1): its off
and on step-time splits are within ~1.5 percentage points of each other.

(The counter reports full-graph decode and piecewise together as one captured
fraction; the time rows above separate them by kind.)

## Measurement limits

- **One run per point.** No point was repeated, so differences of a few percent
  between neighbouring loads are within run-to-run noise until repeated. The
  vLLM load-24 p99 TTFT is one such noisy point.
- **The vLLM curve is reused, not rerun.** It comes from an earlier session
  ([`20260920T125149Z`](../results/20260920T125149Z/)); since the change measured
  here touches lean-vLLM only, the vLLM numbers are unaffected and the workload
  and engine version match, but the two engines were not run back-to-back on the
  same day.
- **No device metrics this run.** `nvidia-smi` was not sampled for the lean-vLLM
  runs, so GPU utilization, power, and clock conditions are not reported; the step
  breakdown rests on server counters and client latencies alone. No online host
  trace or offline CUDA trace was captured this session either.
- **vLLM exposes no server counters here.** The step breakdown by kind is
  lean-vLLM only; vLLM's per-step cost is inferred from load-1 latency and its
  configuration, not measured.
- **vLLM is not the latest release.** 0.26.0 is the last release that pins torch
  2.11, which the FlashAttention-3 wheel links against; matching torch was
  chosen over a newer vLLM.

## Next experiments, in priority order

1. **Cut the eager prefill steps.** Prefill and mixed steps past the 512-token
   grid run eager at ~60–69 ms and now carry ~55% of step time at the plateau —
   the largest remaining cost. Extend the piecewise grid past 512, or compile the
   non-attention pieces, then rerun loads 48 and 64.
2. **Recenter the load bracket.** lean-vLLM now saturates around load 24–32, so
   1–64 still straddles its knee. Add loads between 1 and 32 to place the knee
   precisely, and keep the high loads for vLLM's.
3. **Recapture device metrics.** Rerun one point per arm with `nvidia-smi`
   sampling and the offline CUDA trace, now that decode is captured, to confirm
   the GPU-busy fraction inside steps rose and to re-measure the clock conditions
   against vLLM.
