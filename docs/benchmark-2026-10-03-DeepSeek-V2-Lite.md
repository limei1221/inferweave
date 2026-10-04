# Online serving benchmark — 3 October 2026

lean-vLLM against vLLM 0.26.0 on DeepSeek-V2-Lite-Chat (16B total, 2.4B active,
MLA + MoE + YaRN), one H100, with chunked prefill, full + piecewise CUDA
graphs, and Inductor compilation on both engines. Both engines ran with async
scheduling on. Results are in
[`results/20261003T164448Z/`](../results/20261003T164448Z/).

All ten serving runs completed — five offered loads per engine, 1,000 requests
each, no rejections, failures, or preemptions — replaying the same lognormal
trace (~661k prompt, ~177k generated tokens).

This run follows the addition of Inductor compilation to lean-vLLM; the earlier
report on the pre-Inductor baseline, `benchmark-2026-09-20.md`, was retired in
bbade78 and is in the git history.
The main findings are:

- **Inductor compilation lifted lean-vLLM's plateau from ~20 to ~23
  requests/s.** At load 48 goodput is 23.18 requests/s, up from 20.20 in
  the pre-Inductor run, narrowing the gap with vLLM from 32.5% to 23.1%.
  At load 24 the gap is just 3.6%.
- **lean-vLLM's load-1 TPOT improved to 6.5 ms (1.42× vLLM's 4.6 ms).**
  Down from 7.0 ms (1.6×) pre-Inductor. The decode step replays a captured
  graph on both engines; the remaining difference is in the non-attention
  piecewise pieces and per-step overhead.
- **lean-vLLM's goodput drops past load 48.** At load 64 goodput falls to
  21.14 requests/s, 9% below its load-48 peak, while vLLM keeps climbing to
  31.21. GPU utilization drops from 78% to 68% for lean-vLLM at these loads
  while vLLM holds 91–92%, pointing to host-side bottlenecks in the large
  compiled-but-ungraphed prefill steps.
- **Compiled prefill steps still dominate the plateau.** Steps outside the
  64–512-token piecewise grid now run through Inductor rather than eager, but
  they are still 55–63% of step time at loads 48–64, at 56–70 ms each. Captured
  graph steps run at 9–10 ms.
- **vLLM now ran fresh alongside lean-vLLM.** The earlier report reused a vLLM
  curve from a prior session; this run includes a fresh vLLM curve on the same
  day, with GPU metrics for both engines.

## Setup

### Hardware and software

| Component | Value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3, 81,559 MiB |
| NVIDIA driver | 580.126.09 |
| Linux kernel | `6.8.12-680-6063-coreweave-amd64-f81899c8` |
| Python | 3.12.11 |
| PyTorch, both engines | `2.11.0+cu130` |
| lean-vLLM | commit `06e2725` on `feature/deepseek-v2-lite`; Inductor piecewise compilation |
| FlashMLA | commit `ba89a34` (deepseek-ai HEAD), built from source against the pinned torch |
| vLLM | 0.26.0 |
| MLA decode kernel | lean-vLLM: FlashMLA (`flashmla`); vLLM: `FLASH_ATTN_MLA` (prefill `FLASH_ATTN`) |
| MoE | Neither engine tuned for this GPU; both use default tile sizes |
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
| Graph mode | Full + piecewise |
| KV cache | 327,680 tokens; lean-vLLM uses 64-token blocks, FlashMLA's page |
| Maximum model length | 4,096 |
| Offered loads | 1, 24, 32, 48, 64 requests/s |

vLLM's async scheduling was confirmed on in every server log
([`vllm-async-scheduling.txt`](../results/20261003T164448Z/vllm-async-scheduling.txt)).
Both engines decoded with an MLA kernel, not an expanded fallback
([`vllm-mla-backend.txt`](../results/20261003T164448Z/vllm-mla-backend.txt)).
vLLM ran with prefix caching on, its default; lean-vLLM recorded a 0.0
prefix-cache hit rate on this trace, so prefix caching should not favour either
engine.

### How the two engines execute a step

Both engines capture MLA decode inside the full graph and compile the model
piecewise with Inductor. The remaining difference is in the number of piecewise
capture sizes.

| | lean-vLLM | vLLM 0.26.0 |
| --- | --- | --- |
| Compilation | Inductor piecewise: the model is traced whole, split at attention ops, and each piece is compiled by Inductor | Inductor piecewise, same approach |
| MLA decode (any batch up to `max_num_seqs`) | Inside the full CUDA graph (FlashMLA) | Inside the full CUDA graph |
| Prefill / mixed steps, 64–512 tokens | Inductor-compiled piecewise graphs, 13 sizes | Inductor-compiled piecewise graphs, 51 sizes |
| Prefill / mixed over 512 tokens | Inductor-compiled, no graph | Inductor-compiled, no graph |
| Async scheduling | On | On by default |

lean-vLLM runs `full_and_piecewise` with the decode step captured. The FlashMLA
HEAD here fuses the decode schedule into the kernel; lean-vLLM warms up and
captures at worst-case sequence lengths (`context_lens = max_model_len`), so the
tile schedule and split-KV workspace baked into the graph are sized for the
longest sequence, and each replay refreshes `context_lens`/`block_tables` while
the kernel gates its KV loop on them. Pure decode batches (up to `max_num_seqs`)
therefore replay the whole model, attention included; prefill and mixed steps
fall to piecewise graphs or compiled-ungraphed dispatch.
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

| Offered load (req/s) | lean-vLLM | vLLM | lean vs vLLM |
| ---: | ---: | ---: | ---: |
| 1 | 1.01 | 1.01 | −0.0% |
| 24 | 19.04 | 19.76 | −3.6% |
| 32 | 22.36 | 24.57 | −9.0% |
| 48 | **23.18** | 30.13 | −23.1% |
| 64 | 21.14 | **31.21** | −32.3% |

Goodput is in requests/s. Output token throughput follows goodput: at load 48
lean-vLLM produces 4,100 tok/s against vLLM's 5,328; at load 64 lean-vLLM
falls to 3,739 tok/s while vLLM reaches 5,519.

lean-vLLM peaks at load 48 and then declines: load 64's goodput (21.14) is 9%
below load 48's (23.18). vLLM keeps climbing to load 64, so the gap widens from
23% to 32%. The decline is visible in the GPU metrics too — lean-vLLM's
utilization drops from 78% to 68% at loads 48–64 while vLLM holds 91–92%
(section 3), suggesting the bottleneck is in the host dispatch of compiled-but-
ungraphed prefill steps, which grow from 56 ms to 70 ms as the queue deepens.

Compared with the pre-Inductor run, lean-vLLM's plateau moved up from ~20 to
~23 requests/s. The gap at load 24 narrowed from 12% to 3.6%, and at load 48
from 32.5% to 23.1%. The improvement is concentrated in the compiled prefill
steps, which replaced eager dispatch.

## 2. Latency

| Offered load (req/s) | p50 TTFT lean / vLLM (ms) | p99 TTFT lean / vLLM (s) | p50 TPOT lean / vLLM (ms) | p99 E2E lean / vLLM (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 45 / 34 | 0.10 / 0.09 | **6.5 / 4.6** | 6.52 / 4.31 |
| 24 | 128 / 101 | 0.27 / 0.46 | 23.5 / 17.6 | 19.14 / 15.58 |
| 32 | 179 / 203 | 0.55 / 3.94 | 26.6 / 19.9 | 20.34 / 17.26 |
| 48 | 990 / 354 | 6.83 / 1.51 | 44.6 / 22.5 | 27.34 / 16.15 |
| 64 | 5,192 / 1,222 | 16.48 / 3.49 | 54.0 / 26.9 | 33.81 / 18.85 |

Load 1 isolates per-step overhead: lean-vLLM spends 6.5 ms per generated token
to vLLM's 4.6 ms (1.42×), down from 7.0 ms (1.6×) pre-Inductor. A
DeepSeek-V2-Lite decode step is a single token per sequence, so this is the cost
of the decode step now that MLA attention and the Inductor-compiled pieces around
it replay from one captured graph.

Under load, median TPOT is 1.3–2.0× vLLM's. The TTFT tail blows out once
lean-vLLM passes its knee: p99 TTFT is 6.83 s at load 48 against vLLM's 1.51 s,
because requests queue behind a plateau that cannot drain them. At loads 24–32
lean-vLLM's p99 TTFT is tighter than vLLM's (0.27–0.55 s vs 0.46–3.94 s);
vLLM's load-32 p99 TTFT of 3.94 s is a noisy point — its own p99 drops to
1.51 s at load 48.

## 3. GPU utilization and clock conditions

Sampled once per second and aligned to each run's recorded start and finish:

| Offered load (req/s) | util lean | util vLLM | power lean (W) | power vLLM (W) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 66% | 53% | 228 | 236 |
| 24 | 85% | 91% | 434 | 491 |
| 32 | 86% | 81% | 439 | 475 |
| 48 | 78% | 92% | 420 | 519 |
| 64 | 68% | 91% | 393 | 525 |

lean-vLLM's utilization peaks at loads 24–32 (85–86%) and drops at loads 48–64
(78% → 68%), matching the throughput decline. vLLM holds 91–92% at loads 48–64,
consistent with its still-rising goodput. The drop in lean-vLLM's utilization
suggests that the compiled-but-ungraphed prefill steps, which grow longer and
more numerous under heavier load, involve more host-side dispatch time during
which the GPU is idle.

Power draw reflects the utilization gap: lean-vLLM draws 393–434 W at loads
24–64 while vLLM draws 475–525 W. DeepSeek-V2-Lite activates only 2.4B
parameters per token, so both engines draw well below the 700 W limit and no
`SwPowerCap` throttling was observed during the lean-vLLM or vLLM runs at these
loads. The GPU peaked at 55 °C.

| Offered load (req/s) | mean SM clock lean (MHz) | mean SM clock vLLM (MHz) |
| ---: | ---: | ---: |
| 1 | 1,980 | 1,980 |
| 24 | 1,976 | 1,974 |
| 32 | 1,973 | 1,956 |
| 48 | 1,975 | 1,950 |
| 64 | 1,976 | 1,940 |

Both engines run at or near the 1,980 MHz clock lock. vLLM's clocks dip
slightly at loads 32–64 as it draws more power; lean-vLLM holds steady because
it draws less. Clock conditions do not explain the throughput gap — lean-vLLM
runs at higher clocks and still trails.

## 4. Where the step time goes

lean-vLLM's server counters split each step by how it ran. `graph` is a
pure-decode batch replaying the whole model from one captured graph, attention
included; `piecewise` is a small prefill or mixed step replaying per-piece graphs
with attention running between them; `prefill` is a prefill or mixed step past
the 512-token grid, running compiled through Inductor but without a graph
capture. Figures are for the measured run, warmup subtracted:

| | load 24 | load 32 | load 48 | load 64 |
| --- | ---: | ---: | ---: | ---: |
| Steps | 3,176 | 2,643 | 2,208 | 2,177 |
| Captured (decode + piecewise) steps | 2,758 (86.8%) | 2,331 (88.2%) | 1,806 (81.8%) | 1,783 (81.9%) |
| Compiled prefill steps | 418 (13.2%) | 312 (11.8%) | 402 (18.2%) | 394 (18.1%) |
| Full-graph decode step time | 26.6 s (53.0%) | 24.5 s (57.6%) | 17.6 s (43.2%) | 15.8 s (35.7%) |
| Piecewise step time | 2.8 s (5.5%) | 1.1 s (2.7%) | 0.8 s (1.9%) | 0.8 s (1.8%) |
| Compiled prefill step time | 20.9 s (**41.5%**) | 16.8 s (39.7%) | 22.5 s (**54.9%**) | 27.6 s (**62.5%**) |
| Mean compiled prefill step | 49.9 ms | 54.0 ms | 55.9 ms | **70.1 ms** |
| Mean captured step | 10.7 ms | 11.0 ms | 10.2 ms | 9.3 ms |

No step runs as eager decode; the only non-graph kind is the compiled prefill
past the 512-token grid. At loads 48–64 these compiled prefill steps carry
55–63% of step time at 56–70 ms each. As load increases, the prefill chunks grow
(the scheduler packs more pending prompt tokens into each step), so each step
dispatches more Inductor-compiled ops through the host, and the mean step time
rises from 50 ms at load 24 to 70 ms at load 64.

The goodput drop at load 64 correlates with this: the compiled prefill steps are
10 ms more expensive than at load 32, and they now take 63% of step time, up
from 40%. The captured graph steps are steady at 9–11 ms and decline as a share
because the engine spends proportionally more of its time on compiled prefill.

Compared with the pre-Inductor run, compiled prefill steps are faster than
the eager ones they replaced (~50–56 ms at loads 24–48 vs ~60–69 ms), and
lean-vLLM's plateau has risen by ~3 requests/s. The captured graph steps are
unchanged, as expected — the full graph replay path did not change.

## Measurement limits

- **One run per point.** No point was repeated, so differences of a few percent
  between neighbouring loads are within run-to-run noise until repeated. vLLM's
  load-32 p99 TTFT (3.94 s, worse than its load-48 value) is one such noisy
  point.
- **No async-off curve.** Only async scheduling on was run, so the pipelining
  gain is not measured this session.
- **No offline CUDA trace or host trace.** The GPU busy/idle split inside steps
  and the per-phase host breakdown are not measured; the step breakdown rests on
  server counters and client latencies alone.
- **Neither engine tuned its MoE.** Both use default tile sizes for the routed
  experts; tuning could change the absolute numbers but likely affects both
  engines equally.
- **vLLM exposes no server counters here.** The step breakdown by kind is
  lean-vLLM only; vLLM's per-step cost is inferred from load-1 latency and its
  configuration, not measured.
- **vLLM is not the latest release.** 0.26.0 is the last release that pins torch
  2.11, which the FlashAttention-3 wheel links against; matching torch was
  chosen over a newer vLLM.

## Next experiments, in priority order

1. **Cut the cost of compiled prefill steps past 512 tokens.** They are 55–63%
   of step time at the plateau at 56–70 ms each, and their growth causes
   lean-vLLM's goodput to drop at load 64. Extending the piecewise capture range
   past 512, or adding more capture sizes between 64 and 512 (lean-vLLM captures
   13 vs vLLM's 51), would move more of these steps into graph replay. Then
   rerun loads 48 and 64.
2. **Tune the MoE kernels.** Neither engine ran with tuned tile sizes for
   DeepSeek-V2-Lite's E=64, N=1408 experts on this H100. Tune both and rerun to
   see whether the gap changes.
3. **Recenter the load bracket.** lean-vLLM now saturates around load 32–48, so
   add loads between 32 and 48 to place the knee precisely and measure whether
   the goodput decline past the knee is a consistent pattern.
4. **Capture an offline CUDA trace and host trace.** Measure the GPU busy/idle
   fraction inside steps and the host per-phase breakdown, now that both Inductor
   compilation and the captured decode graph are in place.
