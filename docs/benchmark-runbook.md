# Benchmark runbook: lean-vLLM versus vLLM on DeepSeek-V2-Lite

Compare DeepSeek-V2-Lite (16B total, 2.4B active, MLA + MoE + YaRN) on one NVIDIA
H100-SXM5-80GB using the same workload and resource limits, with **chunked
prefill enabled** and **full + piecewise CUDA graphs** on both engines.
lean-vLLM decodes over latents with FlashMLA and runs the routed experts through
its Triton MoE. Both engines run **one curve each with async scheduling on**.

The five offered loads—**1, 24, 32, 48, and 64 requests/s**—are the starting
bracket, not a final one. V2-Lite activates only 2.4B parameters per token, so it
saturates at a higher load than a dense 8B; do a first pass, then recenter the
rates so they straddle the knee, where throughput plateaus and tail latency
rises. Load 1 stays because its TPOT isolates per-step overhead. Each point uses
1,000 requests. Run the engines sequentially on the same machine and compare
these fresh results.

The GPU paths this measures—FlashMLA decode, the Triton MoE, both graph
modes—must be validated for correctness before any timing is meaningful. Section
0 gates on that. Use the same Bash session for the commands below.

## 0. Validate correctness first

A fast curve over wrong numbers is worthless. Before benchmarking, confirm the
GPU paths agree with their references on this machine:

```bash
uv run pytest tests/test_attention_backends.py -k mla -v      # FlashMLA decode vs the fp32 oracle
uv run pytest tests/test_fused_moe.py -k cuda -v              # the Triton MoE kernel vs grouped_mm
uv run pytest tests/test_deepseek_v2.py -v                    # MLA + MoE + YaRN vs transformers
```

The first two must run, not skip: a skip means FlashMLA or Triton is not
available here, so the sweep would measure a fallback. Then compare greedy
generation on the real weights against the pinned vLLM below on a handful of
prompts, and record the token-level agreement and any tolerances. Only then run
the curves.

## 1. Install both engines

Clone once; for an existing checkout, start with `cd`:

```bash
git clone git@github.com:limei1221/lean-vllm.git ~/workspace/lean-vllm
cd ~/workspace/lean-vllm
git checkout feature/deepseek-v2-lite
uv sync --extra cuda
uv run hf download deepseek-ai/DeepSeek-V2-Lite-Chat --local-dir ~/workspace/huggingface/DeepSeek-V2-Lite-Chat
```

The sync pulls a prebuilt FlashAttention-3, about 400 MB, pinned by URL and hash
in `pyproject.toml`. FA3 alone is compiled by no one here, but FlashMLA is: it
publishes no wheel, so decode over latents needs a source build against the
pinned torch. That build wants `nvcc`, so unlike the FA3-only path this box needs
a CUDA toolkit. Build FlashMLA **after** `uv sync`, and never re-run a plain
`uv sync` afterward—it removes anything it did not install unless run with
`--inexact`:

```bash
git clone --recursive https://github.com/deepseek-ai/FlashMLA.git ~/workspace/FlashMLA
VIRTUAL_ENV=~/workspace/lean-vllm/.venv uv pip install --no-build-isolation -v ~/workspace/FlashMLA
```

Confirm both device paths the sweep will take:

```bash
uv run python -c 'import torch; from lean_vllm.attention import LayerSpec, get_attention_backend; print(get_attention_backend(LayerSpec(192, 16, 16, torch.bfloat16, latent_dim=576)).get_name())'
uv run python -c 'import torch; from lean_vllm.layers.fused_moe import use_triton; print("triton" if use_triton(torch.zeros(1, device="cuda")) else "torch")'
```

The first must print `flashmla`; `flash_attn_3` means the FlashMLA build did not
land and decode would expand latents every step—correct, but not the path being
measured. The second must print `triton`; `torch` means the Triton import failed
and the experts would fall back to `grouped_mm`.

Neither engine ships a tuned MoE config for V2-Lite on an H100, so both run
vLLM's default tiles. To compare tuned against tuned, tune both on this GPU
first: `benchmarks/tune_moe.py --tune` for lean-vLLM (see
[deepseek-v2.md](deepseek-v2.md#tuning-the-moe-kernel)), and vLLM's
`benchmarks/kernels/benchmark_moe.py --tune` with `VLLM_TUNED_CONFIG_FOLDER`
for vLLM. Each server logs which config file it loaded.

Keep vLLM in a separate environment because it manages its own PyTorch
dependencies. Version `0.26.0` is the last one that pins torch `2.11.0`, what the
`cuda` extra pins, so the curves compare engines rather than PyTorch releases:

```bash
uv venv ~/workspace/vllm-env --python 3.12
uv pip install --python ~/workspace/vllm-env/bin/python vllm==0.26.0
```

vLLM ships its own DeepSeek-V2 model and its own MLA backend; it selects one at
startup, recorded in section 4. Recent transformers loads V2-Lite's config and
tokenizer natively, so `--trust-remote-code` should not be needed; add it to the
vLLM server args if startup complains about the config.

## 2. Set the shared workload and server limits

```bash
export MODEL=~/workspace/huggingface/DeepSeek-V2-Lite-Chat
export RUN_RESULTS="results/$(date -u +%Y%m%dT%H%M%SZ)"
export RATES="1,24,32,48,64"
export KVTOKENS=327680
export PINNED="--max-num-batched-tokens 8192 --max-num-seqs 256 --enable-chunked-prefill"
export TRACE_ARGS="--dataset lognormal --input-len 512 --output-len 128 --sigma 0.8 --temperature 0 --warmup 3 --timeout 1200"
export LEAN_SERVER_ARGS="$PINNED --cudagraph-mode full_and_piecewise --async-scheduling"
export VLLM_SERVER_ARGS="$PINNED --compilation-config '{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'"
mkdir -p "$RUN_RESULTS"
```

| Setting | Both engines |
| --- | --- |
| Model | Same local DeepSeek-V2-Lite-Chat weights (31 GB bf16) |
| Input / output lengths | Lognormal distribution, medians 512 / 128 tokens, σ = 0.8 |
| Sampling | Greedy, seed 0 |
| Requests per point | 1,000 measured requests, plus 3 warmup requests |
| Maximum context | 4,096 tokens |
| KV cache capacity | 327,680 tokens |
| Batch token budget / maximum sequences | 8,192 / 256 |
| Chunked prefill | Enabled |
| CUDA graphs | Full + piecewise |
| Async scheduling | On for both engines |

The graph-mode option differs between engines: lean-vLLM uses
`--cudagraph-mode full_and_piecewise`; vLLM uses the equivalent
`FULL_AND_PIECEWISE` setting in `--compilation-config`.

Both engines now capture MLA decode inside the full graph. The FlashMLA build
here (deepseek-ai HEAD) fuses the tile schedule and split-KV workspace into
`dense_decode_fwd`, sizing them from `cache_seqlens`. Both lean-vLLM and vLLM
capture that construction once, at capture time, with worst-case sequence lengths,
so the baked schedule and workspace cover the longest sequence; the kernel then
gates its KV loop on the `context_lens` each replay refreshes, so a replay with
shorter, different lengths stays in bounds. lean-vLLM captures at `max_model_len`
(see `capture_cudagraph` in `lean_vllm/engine/model_runner.py`); a pure-decode
step that fits a captured batch size replays the whole model, attention included.
A decode batch past the largest captured size runs eager: the piecewise buckets
stop at 512 tokens too, so no graph covers it.

MLA caches one compressed latent per token per layer—about 31 KB per token
across V2-Lite's 27 layers, against the ~276 KB a plain paged cache of its heads
would take. The weights are 31 GB, leaving most of the 80 GB for cache, so
327,680 tokens fits with wide margin. Raise `KVTOKENS` if a first pass shows the
cache, not compute, capping concurrency; keep it identical across both engines.

`sweep.py` converts KV capacity to each engine's block count, so both get the
same number of cache tokens. When FlashMLA is selected lean-vLLM sets the cache
block size to 64, the kernel's page. Each point starts a fresh server. The
1,200-second client timeout allows queued requests to finish under heavy load.

## 3. Save metadata and monitor the GPU

Record the build and environment with the results:

```bash
{
  git rev-parse HEAD
  git status --short
  git -C ~/workspace/FlashMLA rev-parse HEAD
  uv run python -c 'import torch; from lean_vllm.attention import LayerSpec, get_attention_backend; print("lean-vLLM MLA backend:", get_attention_backend(LayerSpec(192, 16, 16, torch.bfloat16, latent_dim=576)).get_name())'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  uname -r
  uv run python --version
  uv run python -c 'import torch; print("lean-vLLM torch:", torch.__version__)'
  ~/workspace/vllm-env/bin/vllm --version
  ~/workspace/vllm-env/bin/python -c 'import torch; print("vLLM torch:", torch.__version__)'
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  date '+%Y-%m-%dT%H:%M:%S%z'
} | tee "$RUN_RESULTS/environment.txt"
git diff HEAD > "$RUN_RESULTS/working-tree.patch"
```

Save any untracked source files separately; they are not included in the patch.

If the host allows it, enable persistence mode and lock the clock. Both need
root, which the shell in a rented GPU container usually already has:

```bash
nvidia-smi -pm 1
nvidia-smi -lgc "$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)"
```

If the container refuses, continue with monitoring. Record settings and keep a
GPU log running through every run:

```bash
nvidia-smi --query-gpu=name,persistence_mode,clocks.applications.graphics,clocks.max.graphics,power.limit,power.max_limit \
  --format=csv | tee "$RUN_RESULTS/gpu-settings.csv"

nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active \
  --format=csv -l 1 > "$RUN_RESULTS/nvidia-smi.log" &
GPU_LOG_PID=$!
```

## 4. Run the curves

Make sure no other model server is running. The sweep handles startup, health
checks, warmup, and shutdown automatically.

### lean-vLLM

```bash
uv run python benchmarks/sweep.py \
  --model "$MODEL" --engine lean-vllm --suite rate \
  --rates "$RATES" --num-requests 1000 --seed 0 \
  --max-model-len 4096 --kvcache-tokens "$KVTOKENS" \
  --server-args "$LEAN_SERVER_ARGS" \
  --client-args "$TRACE_ARGS" --out "$RUN_RESULTS/lean-v2lite"
```

### vLLM

```bash
PATH="$HOME/workspace/vllm-env/bin:$PATH" uv run python benchmarks/sweep.py \
  --model "$MODEL" --engine vllm --suite rate \
  --rates "$RATES" --num-requests 1000 --seed 0 \
  --max-model-len 4096 --kvcache-tokens "$KVTOKENS" \
  --server-args "$VLLM_SERVER_ARGS" \
  --client-args "$TRACE_ARGS" --out "$RUN_RESULTS/vllm-v2lite"
```

Confirm vLLM ran with async scheduling on, and record it, along with the MLA
backend it chose so the comparison names both engines' decode kernel:

```bash
grep -h "Asynchronous scheduling is" "$RUN_RESULTS"/vllm-v2lite/rate/*.server.log | sort | uniq -c \
  | tee "$RUN_RESULTS/vllm-async-scheduling.txt"
grep -hiE "using.*mla|attention backend|flashmla" "$RUN_RESULTS"/vllm-v2lite/rate/*.server.log | sort | uniq -c \
  | tee "$RUN_RESULTS/vllm-mla-backend.txt"
```

If startup fails, read the corresponding `.server.log` and confirm the process
has stopped before retrying. The default startup timeout is 900 seconds.
Use a new output directory for a rerun to preserve the previous results.

## 5. Profile the step loop

Run one load-48 point with the step-loop profiler on. The server
inherits these variables from the sweep, and each arm's server log names its
trace file. This trace is CPU-only; `await_tokens` shows the host waiting on
the device.

```bash
LEAN_PROFILE_DIR="$RUN_RESULTS/profile-48" LEAN_PROFILE_STEPS=600 \
  uv run python benchmarks/sweep.py \
  --model "$MODEL" --engine lean-vllm --suite rate \
  --rates 48 --num-requests 1000 --seed 0 \
  --max-model-len 4096 --kvcache-tokens "$KVTOKENS" \
  --server-args "$LEAN_SERVER_ARGS" \
  --client-args "$TRACE_ARGS" --out "$RUN_RESULTS/profile-48"
```

The offline run adds device activity, which gives the GPU idle fraction.
`bench_offline.py` loads a fixed model path; point it at the V2-Lite-Chat
directory before running so the idle fraction reflects the model under test:

```bash
LEAN_PROFILE_DIR="$RUN_RESULTS/offline-on" LEAN_PROFILE_CUDA=1 \
  uv run python benchmarks/bench_offline.py --async-scheduling
```

## 6. Compare and archive

Print the same metrics for both engines:

```bash
jq -r '.engine as $engine | .rows[] | [$engine, .arm, .request_rate, .completed, .failure_rate, .goodput, .output_tok_s, .ttft_p99, .tpot_p50, .e2e_p99] | @tsv' \
  "$RUN_RESULTS/lean-v2lite/rate/summary.json" \
  "$RUN_RESULTS/vllm-v2lite/rate/summary.json"
```

Columns are engine, arm, offered load, completed requests, failure rate,
goodput, output tokens/s, p99 TTFT, median TPOT, and p99 E2E. Latency
values are in seconds.

| Metric | What to compare |
| --- | --- |
| Goodput | Completed requests per second, including queue drain time; higher is better |
| Output tokens/s | Generation throughput; higher is better |
| p99 TTFT | Time until the first token for slow requests; lower is better |
| Median TPOT | Time per output token after the first; use load 1 for low-load latency |
| p99 E2E | Time to finish slow requests; lower is better |

Before drawing conclusions:

- Check that each run completed all 1,000 requests without failures.
  Goodput has no latency cutoff, so read it alongside latency.
- Compare matching offered loads and workload settings. Both engines run with
  async scheduling on, so the curves are directly comparable. Find where throughput
  levels off and tail latency rises sharply—the saturation knee. If it sits at
  or beyond the top rate, extend `RATES` upward and rerun into a new directory.
- Confirm both engines decoded with an MLA kernel, not an expanded fallback, and
  name each in the report; a kernel-versus-fallback gap is not an engine gap.
- Check GPU-busy clocks and throttling for each pair. Aim for mean clocks within
  about 1%; report differences that could affect the comparison.

Per-run JSON contains client summaries and two scrapes of the server's
Prometheus `/metrics`, `server.before` and `server.after`, taken after warmup
and after the run. `server.run` holds what they moved by during the run: the
busy fraction, mean batch tokens and preemptions the sweep table shows. Each run records `started_at` and `finished_at` in UTC around the client
run, so align the GPU log to those rather than to file modification times.

Stop the logger and archive the session:

```bash
kill "$GPU_LOG_PID"
wait "$GPU_LOG_PID" || true
tar czf "${RUN_RESULTS}.tar.gz" "$RUN_RESULTS"
```

Write the report around the matched throughput and latency curves, with the
commit, the FlashMLA commit, engine versions, hardware, and clock conditions
alongside them. Report the MLA decode kernel of every curve, and the offline GPU
idle fraction.
