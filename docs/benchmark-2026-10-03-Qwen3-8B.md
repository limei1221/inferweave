# Online serving benchmark — 3 October 2026

lean-vLLM against vLLM 0.26.0 on Qwen3-8B, one H100, with chunked prefill,
full + piecewise CUDA graphs, and Inductor compilation on both engines.
lean-vLLM ran two curves, async scheduling off and on; vLLM ran its default,
which is on. Results are in
[`results/20261003T095433Z/`](../results/20261003T095433Z/).

All fifteen serving runs completed: five offered loads per curve, 1,000 requests
each, no rejections, failures, or preemptions. Every run replayed the same trace
of 659,770 prompt tokens and 176,838 generated tokens.

The main findings are:

- **Below saturation the engines are at parity.** At load 1 goodput is identical
  and lean-vLLM's median TPOT is 0.2 ms (3%) above vLLM's. At load 24 lean-vLLM
  is 0.1% behind on goodput, and TTFT and TPOT are within 5%.
- **At the plateau lean-vLLM trails by 3.7–3.9%.** With async scheduling on it
  reaches 24.81 requests/s at load 48 and 25.06 at load 64, against vLLM's 25.76
  and 26.09.
- **Inductor compilation halved the gap.** The previous benchmark's eager prefill
  steps now run through Inductor-compiled pieces. Their share of step time at
  load 48 dropped from 52% to 43%, and the throughput gap narrowed from 5–7% to
  3.7–3.9%.
- **Async scheduling is worth 1.5–6.7%.** It adds 1.5% at load 24, 6.3% at
  load 48, and 6.7% at load 64 over the off curve, and lifts GPU utilization
  under load from 84–89% to 92–98%.
- **Pipelining removes almost all offline GPU idle.** The offline CUDA trace
  shows the GPU idle 26.5% of the step window with async scheduling off and 2.9%
  with it on, with the same kernel time in both.
- **Latency is within 5–12% of vLLM past load 1.** Median TPOT is 1.05–1.12×
  vLLM's from load 24 up, and p99 TTFT tracks within 6% up to load 32. At the
  plateau, p99 TTFT is 1.17–1.27× vLLM's.

## Setup

### Hardware and software

| Component | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3, 81,559 MiB |
| NVIDIA driver | 580.126.09 |
| Linux kernel | `6.8.12-680-6063-coreweave-amd64-f81899c8` |
| Python | 3.12.11 |
| PyTorch, both engines | `2.11.0+cu130` |
| lean-vLLM | commit `06e2725`; working tree differs only in `uv.lock` mirror URLs |
| vLLM | 0.26.0 |
| Attention | FlashAttention-3 on both engines |
| Persistence mode | Enabled |
| Application graphics clock | 1,980 MHz, equal to the maximum |
| Power limit | 700 W (maximum 700 W) |

### Workload and server settings

| Setting | Both engines |
| --- | --- |
| Requests per run | 1,000, plus 3 warmup requests |
| Workload | Lognormal lengths: input 512, output 128, σ = 0.8 |
| Sampling | Greedy, seed 0 |
| Batch token budget / maximum sequences | 8,192 / 256 |
| Chunked prefill | Enabled |
| Graph mode | Full + piecewise |
| KV cache | 327,680 tokens in 16-token blocks |
| Maximum model length | 4,096 |
| Offered loads | 1, 24, 32, 48, 64 requests/s |

vLLM's async scheduling was confirmed on in every server log,
so the like-for-like pair is lean-vLLM's `async-scheduling=True` curve against
vLLM. vLLM also ran with prefix caching on, its default. lean-vLLM recorded a
0.0 prefix-cache hit rate on this trace, so prefix caching should not favour
either engine.

### How the two engines execute a step

| | lean-vLLM | vLLM 0.26.0 |
| --- | --- | --- |
| Compilation | Inductor piecewise: the model is traced whole, split at attention ops, and each piece is compiled by Inductor | Inductor piecewise, same approach |
| Decode batches | Full CUDA graphs, 20 sizes | Full CUDA graphs, 35 sizes |
| Prefill and mixed steps, 64–512 tokens | Inductor-compiled piecewise graphs, 13 sizes up to 512 | Inductor-compiled piecewise graphs, 51 sizes up to 512 |
| Prefill and mixed steps above 512 or under 64 tokens | Inductor-compiled, no graph | Inductor-compiled, no graph above 512; captured under 64 |
| Async scheduling | Launch step k, then await and reconcile step k-1 | On by default |

With async scheduling off, lean-vLLM awaits step k-1's tokens before scheduling
step k, and only detokenization overlaps the forward pass. With it on,
scheduling and batch preparation also run ahead of the await
([`docs/online-serving.md`](../docs/online-serving.md#pipelined-steps)).

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
| 24 | 19.11 | 19.39 | 19.40 | −0.1% | +1.5% |
| 32 | 22.22 | 22.98 | 23.10 | −0.5% | +3.4% |
| 48 | 23.35 | 24.81 | **25.76** | −3.7% | +6.3% |
| 64 | 23.48 | 25.06 | **26.09** | −3.9% | +6.7% |

Goodput is in requests/s. Output token throughput follows goodput, since every
run generates the same tokens: at load 48 and 64 lean-vLLM on produces 4,387
and 4,432 tok/s, lean-vLLM off 4,129 and 4,152, and vLLM 4,556 and 4,614.

Both lean-vLLM curves level off from load 48: at 23.4 requests/s with async
scheduling off and at 24.8–25.1 with it on. vLLM is still climbing slightly at
load 64. The gap therefore opens only once the engines are capacity-bound, and
widens as vLLM's extra capacity is used.

Compared with the 13 September benchmark, the plateau gap narrowed from 5–7% to
3.7–3.9%. The 13 September run's prefill steps outside the piecewise graph range
ran eager; this run compiles them with Inductor. At load 48 their share of step
time dropped from 52% to 43%.

## 2. Latency

| Offered load (req/s) | p50 TTFT on / vLLM (ms) | p99 TTFT on / vLLM (s) | p50 TPOT on / vLLM (ms) | p50 E2E on / vLLM (s) | p99 E2E on / vLLM (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27 / 28 | 0.10 / 0.11 | 6.9 / 6.7 | 0.90 / 0.88 | 6.42 / 6.33 |
| 24 | 76 / 76 | 0.32 / 0.32 | 13.3 / 12.7 | 1.85 / 1.78 | 11.70 / 11.44 |
| 32 | 175 / 153 | 0.86 / 0.82 | 21.5 / 19.2 | 3.05 / 2.74 | 17.55 / 16.11 |
| 48 | 1,678 / 1,157 | 5.77 / 4.56 | 40.8 / 38.2 | 7.84 / 6.69 | 26.55 / 24.94 |
| 64 | 4,216 / 3,361 | 10.61 / 9.08 | 41.3 / 38.9 | 10.58 / 9.53 | 27.15 / 25.54 |

At load 1 the engines are indistinguishable except for a constant 0.2 ms per
token, the per-step overhead outside the graph. Past load 1 lean-vLLM trails
on every metric, but much less than in the 13 September run. Median TPOT is
1.05–1.12× vLLM's, down from 1.11–1.29×. The TTFT tail also narrowed: p99 TTFT
is within 6% of vLLM up to load 32, and 1.17–1.27× at the plateau, down from
3.3–3.7× at loads 24–32.

Async scheduling helps latency as much as throughput. Against the off curve at
load 48 it cuts p50 TTFT from 2.22 s to 1.68 s and p99 TPOT from 80 ms to
76 ms. At load 24 it cuts p50 TPOT by 14%, from 15.5 ms to 13.3 ms.

## 3. GPU utilization and clock conditions

Sampled once per second and aligned to each run's recorded start and finish:

| Offered load (req/s) | util off | util on | util vLLM | power off (W) | power on (W) | power vLLM (W) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 68% | 71% | 69% | 342 | 347 | 349 |
| 24 | 89% | 98% | 98% | 599 | 621 | 605 |
| 32 | 88% | 95% | 98% | 609 | 625 | 620 |
| 48 | 84% | 93% | 97% | 600 | 618 | 613 |
| 64 | 88% | 92% | 97% | 609 | 612 | 614 |

With async scheduling on, lean-vLLM keeps the GPU as busy as vLLM up to load
24 and draws the same power at every load. At loads 48 and 64 its utilization
is 4–5 points below vLLM's, in line with the 3.7–3.9% throughput gap.

| Offered load (req/s) | mean SM clock on (MHz) | mean SM clock vLLM (MHz) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 1,975 | 1,974 | +0.1% |
| 24 | 1,864 | 1,873 | −0.5% |
| 32 | 1,772 | 1,807 | −1.9% |
| 48 | 1,725 | 1,697 | +1.6% |
| 64 | 1,749 | 1,712 | +2.1% |

Clocks are averaged over busy samples. Under load both engines drop below the
1,980 MHz lock as `SwPowerCap` engages at the 700 W limit; no other throttle
reason appeared, and the GPU peaked at 65 °C. At load 64 lean-vLLM held the
higher clock and still trailed, so clock conditions do not explain the gap.

## 4. Where the step time goes

### Offline: pipelining removes the GPU idle

The offline CUDA trace records device activity, so it gives a true busy/idle
split. It is an offline `generate` on Qwen3-0.6B: 256 sequences with inputs and
outputs of 100–1,024 tokens, 200 steps captured after 200 skipped. GPU-busy
time is the union of all kernel and copy intervals over the captured window.

| | Async off | Async on |
| --- | ---: | ---: |
| Window, 200 steps | 2,478 ms | 1,893 ms |
| Step time | 12.4 ms | **9.5 ms** |
| Kernel-busy time | 1,817 ms | 1,833 ms |
| FlashAttention share of kernel time | 80.5% | 80.5% |
| GPU busy | 73.5% | **97.1%** |
| GPU idle | 26.5% | **2.9%** |
| Host `await_tokens` per step | 7.46 ms | 5.32 ms |
| Host `launch` per step | 3.66 ms | 3.15 ms |
| Host `detokenize` per step | 0.49 ms | 0.39 ms |

The device work is the same in both arms. The step is 23% faster with async
scheduling on only because batch preparation and launch now happen while the
GPU is still running the previous step, so the host blocks for less time in
`await_tokens`. At 2.9% idle there is little left to win offline from further
host overlap.

### Online at load 48: compiled steps still dominate

lean-vLLM's server counters break engine step time down by how each step ran.
Figures are for the measured run, with warmup subtracted:

| | load 24, on | load 48, off | load 48, on |
| --- | ---: | ---: | ---: |
| Steps | 4,431 | 2,180 | 2,231 |
| Compiled prefill steps (outside 64–512 tokens) | 433 (9.8%) | 387 (17.8%) | 369 (16.5%) |
| Mean compiled prefill step | 22.5 ms | 52.0 ms | **45.1 ms** |
| Mean full or piecewise graph step | 9.9 ms | 11.5 ms | 11.7 ms |
| Compiled share of step time | 19.7% | 49.4% | **43.3%** |

At the plateau, the steps that run outside the piecewise graph range still take
close to half the engine's time, even though they now run through Inductor
rather than eager. Async scheduling shortens them by 13%, from 52.0 ms to
45.1 ms, because their host dispatch overlaps the previous step. The graph steps
stay near 11 ms.

Both engines compile these large steps with Inductor and neither captures them
in graphs above 512 tokens. The plateau gap is no longer explained by an eager
versus compiled difference; it now sits in the graph steps and the number of
piecewise capture sizes: lean-vLLM captures 13 sizes (64–512) against vLLM's 51
(1–512), so lean-vLLM pads more when a step falls between sizes.

### Online host trace at load 48

A CPU-only step-loop trace was captured for one load-48 run per arm, 600 steps
each ([`profile-48/`](../results/20261003T095433Z/profile-48/)). The profiler
is lighter than in the 13 September run. It cut goodput by 9% with async
scheduling on (22.52 vs 24.81 requests/s) and by 10% with it off. Only the
proportions below are meaningful, not the absolute times:

| Host phase, mean per step | Async off | Async on |
| --- | ---: | ---: |
| `schedule` | 0.25 ms | 0.30 ms |
| `launch` | 15.67 ms | 21.05 ms |
| ↳ `prepare_batch` | 3.20 ms | 6.73 ms |
| ↳ `run_model` | 11.92 ms | 13.64 ms |
| `await_tokens` | 18.99 ms | 11.66 ms |
| `reconcile` | 0.35 ms | 0.32 ms |
| `detokenize` | 1.58 ms | 1.49 ms |
| Traced step | 39.2 ms | 37.3 ms |

The host's own work is dominated by `run_model`, the dispatch of the forward
pass. With async scheduling on, `prepare_batch` is higher than with it off
(6.73 ms vs 3.20 ms) because the batch is prepared while the previous step's
GPU work is still running, and the Inductor-compiled model's guard checks
execute inline. `await_tokens` drops correspondingly (11.66 ms vs 18.99 ms)
as less of the GPU work remains to wait for.
`detokenize` (about 1.5 ms) and `schedule` (about 0.3 ms) are steady and
already overlap the GPU.

## Measurement limits

- **One run per point.** No point was repeated, so differences of a few percent
  between neighbouring loads are within run-to-run noise until repeated.
- **The GPU busy/idle split is offline, on Qwen3-0.6B.** CUPTI cannot collect
  device activity from the online engine thread. The online 8B idle is inferred
  from sampled utilization, not measured.
- **The online host trace distorts what it measures.** In-process
  `record_function` tracing cost 9–10% of goodput and roughly doubled the
  traced step, and it inflates compiled-but-ungraphed steps most because they
  dispatch the most ops.
- **vLLM exposes no server counters here.** The step breakdown by kind is
  lean-vLLM only; vLLM's large-step cost is inferred from its configuration,
  not measured.
- **vLLM is not the latest release.** 0.26.0 is the last release that pins
  torch 2.11, which the FlashAttention-3 wheel links against; matching torch
  was chosen over a newer vLLM.

## Next experiments, in priority order

1. **Narrow the piecewise capture gap.** lean-vLLM captures 13 piecewise sizes
   (64–512) against vLLM's 51 (1–512). Adding sizes below 64 and closing the
   gaps between buckets would reduce padding waste in piecewise steps. Then
   rerun loads 48 and 64.
2. **Repeat every plateau point.** This puts error bars on the 3.7–3.9% gap and
   confirms the improvement over the 13 September run is real.
3. **Profile the online loop with less overhead.** Record per-phase timings
   with `perf_counter` counters instead of `record_function`, so the online
   host split can be read without the 9–10% distortion.
