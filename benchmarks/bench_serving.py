r"""Open-loop serving benchmark: Poisson arrivals against lean-vLLM or vLLM.

    uv run python benchmarks/bench_serving.py --model ~/workspace/huggingface/Qwen3-8B \
        --dataset lognormal --num-requests 500 --request-rate 8

No request is retried, and percentiles cover completed requests only, so tables show the failure rate too.
Prompts are token ids on `/v1/completions`, so no chat template skews the counts.
"""

import argparse
import asyncio
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from time import perf_counter

import httpx
import numpy as np
from openai import AsyncOpenAI

PERCENTILES = (50, 90, 95, 99)


@dataclass
class Request:
    prompt_token_ids: list[int]
    output_len: int
    priority: int = 0
    label: str = "all"

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


@dataclass
class Result:
    index: int
    label: str
    priority: int
    prompt_len: int
    requested_output_len: int
    arrival: float  # seconds after the first send
    status: str  # ok | failed
    output_len: int = 0
    ttft: float | None = None
    latency: float | None = None
    itls: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def tpot(self) -> float | None:
        """Seconds per token after the first. Undefined for a one-token answer."""
        if self.latency is None or self.ttft is None or self.output_len < 2:
            return None
        return (self.latency - self.ttft) / (self.output_len - 1)


# ---------------------------------------------------------------- traces


def _prompt(rng: random.Random, num_tokens: int, vocab_size: int) -> list[int]:
    """Random ids below `--vocab-size`, so no prompts share a prefix or hit special tokens."""
    return [rng.randrange(vocab_size) for _ in range(num_tokens)]


def _lognormal(rng: random.Random, median: int, sigma: float, low: int, high: int) -> int:
    return max(low, min(high, round(rng.lognormvariate(math.log(median), sigma))))


def fixed_trace(rng: random.Random, args) -> list[Request]:
    """Every request the same shape: the cleanest read on a scheduling change."""
    return [Request(_prompt(rng, args.input_len, args.vocab_size), args.output_len) for _ in range(args.num_requests)]


def lognormal_trace(rng: random.Random, args) -> list[Request]:
    """ShareGPT-shaped lengths without ShareGPT: lognormal about the given medians."""
    trace = []
    for _ in range(args.num_requests):
        prompt_len = _lognormal(rng, args.input_len, args.sigma, 4, args.max_model_len - 8)
        output_len = _lognormal(rng, args.output_len, args.sigma, 1, args.max_model_len - prompt_len)
        trace.append(Request(_prompt(rng, prompt_len, args.vocab_size), output_len))
    return trace


def mixed_trace(rng: random.Random, args) -> list[Request]:
    """Short prompts beside long ones; read the `short` label's TTFT."""
    trace = []
    for _ in range(args.num_requests):
        if rng.random() < args.long_fraction:
            request = Request(
                _prompt(rng, args.long_input_len, args.vocab_size),
                args.long_output_len,
                args.long_priority,
                "long",
            )
        else:
            request = Request(_prompt(rng, args.input_len, args.vocab_size), args.output_len, 0, "short")
        trace.append(request)
    return trace


def prefix_trace(rng: random.Random, args) -> list[Request]:
    """A few shared system prompts, a unique question behind each: the only trace prefix caching helps."""
    prefixes = [_prompt(rng, args.prefix_len, args.vocab_size) for _ in range(args.num_prefixes)]
    trace = []
    for _ in range(args.num_requests):
        suffix_len = max(1, args.input_len - args.prefix_len)
        prompt = rng.choice(prefixes) + _prompt(rng, suffix_len, args.vocab_size)
        trace.append(Request(prompt, args.output_len))
    return trace


def sharegpt_trace(rng: random.Random, args) -> list[Request]:
    """Real conversations from a ShareGPT-format JSON, first turn and its answer."""
    from transformers import AutoTokenizer

    if not args.dataset_path:
        raise SystemExit("--dataset sharegpt needs --dataset-path")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    with open(args.dataset_path) as handle:
        conversations = json.load(handle)
    pairs = [
        (entry["conversations"][0]["value"], entry["conversations"][1]["value"])
        for entry in conversations
        if len(entry.get("conversations", ())) >= 2
    ]
    rng.shuffle(pairs)
    trace = []
    for prompt, answer in pairs:
        prompt_token_ids = tokenizer(prompt).input_ids
        output_len = len(tokenizer(answer).input_ids)
        # Skip degenerate turns, and over-long pairs that would be a 400.
        if len(prompt_token_ids) < 4 or output_len < 4:
            continue
        if len(prompt_token_ids) + output_len > args.max_model_len:
            continue
        trace.append(Request(prompt_token_ids, output_len))
        if len(trace) == args.num_requests:
            return trace
    raise SystemExit(f"{args.dataset_path} yielded {len(trace)} usable requests, wanted {args.num_requests}")


TRACES = {
    "fixed": fixed_trace,
    "lognormal": lognormal_trace,
    "mixed": mixed_trace,
    "prefix": prefix_trace,
    "sharegpt": sharegpt_trace,
}


def validate(trace: list[Request], args):
    """A request over the context is a 400, which would read as an engine failure."""
    over = [request for request in trace if request.prompt_len + request.output_len > args.max_model_len]
    if over:
        raise SystemExit(f"{len(over)} requests exceed --max-model-len {args.max_model_len}")


def arrival_offsets(rng: random.Random, num_requests: int, rate: float) -> list[float]:
    """Poisson arrivals are exponential gaps. An infinite rate is one burst."""
    if rate == float("inf"):
        return [0.0] * num_requests
    offsets, when = [], 0.0
    for _ in range(num_requests):
        offsets.append(when)
        when += rng.expovariate(rate)
    return offsets


# ---------------------------------------------------------------- the run


class Run:
    """What has finished so far, and whether the client should keep sending."""

    def __init__(self, total: int, max_failure_rate: float, quiet: bool):
        self.total = total
        self.budget = max(2, int(max_failure_rate * total))
        self.quiet = quiet
        self.done = 0
        self.failed = 0
        self.first_error: str | None = None
        self.aborted = False

    def finish(self, result: Result) -> Result:
        self.done += 1
        if result.status == "failed":
            self.failed += 1
            self.first_error = self.first_error or result.error
            if self.failed > self.budget and not self.aborted:
                self.aborted = True
                print(f"\naborting: {self.failed} failures, the first {self.first_error}", file=sys.stderr)
        if not self.quiet:
            print(f"\r{self.done}/{self.total} done, {self.failed} failed", end="", file=sys.stderr)
        return result


async def one_request(api: AsyncOpenAI, args, index: int, request: Request, t0: float, run: Run) -> Result:
    result = Result(
        index=index,
        label=request.label,
        priority=request.priority,
        prompt_len=request.prompt_len,
        requested_output_len=request.output_len,
        arrival=perf_counter() - t0,
        status="failed",
    )
    send = last = perf_counter()
    try:
        stream = await api.completions.create(
            model=args.model_name,
            prompt=request.prompt_token_ids,
            max_tokens=request.output_len,
            temperature=args.temperature,
            stream=True,
            stream_options={"include_usage": True},
            # Neither is in the OpenAI schema; both engines read them off the body.
            extra_body={"ignore_eos": True, "priority": request.priority},
        )
        async for chunk in stream:
            now = perf_counter()
            if chunk.usage:
                result.output_len = chunk.usage.completion_tokens
            for choice in chunk.choices:
                if not choice.text:
                    continue
                if result.ttft is None:
                    result.ttft = now - send
                else:
                    result.itls.append(now - last)
                last = now
                result.latency = now - send
    except Exception as error:
        # Anything else fails, including an error sent in the stream after a 200.
        result.error = f"{type(error).__name__}: {error}"
        return run.finish(result)
    if result.ttft is None:
        result.error = "the stream carried no tokens"
        return run.finish(result)
    # Detokenization holds bytes back, so chunks are not tokens: prefer the usage count.
    result.output_len = result.output_len or 1 + len(result.itls)
    result.status = "ok"
    return run.finish(result)


async def run_trace(
    api: AsyncOpenAI, args, trace: list[Request], offsets: list[float]
) -> tuple[list[Result], Run, float]:
    run = Run(len(trace), args.max_failure_rate, args.quiet)
    tasks: list[asyncio.Task] = []
    t0 = perf_counter()
    for index, (request, offset) in enumerate(zip(trace, offsets)):
        delay = offset - (perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        if run.aborted:
            break
        tasks.append(asyncio.create_task(one_request(api, args, index, request, t0, run)))
    if run.aborted:
        for task in tasks:
            task.cancel()
        finished = await asyncio.gather(*tasks, return_exceptions=True)
        results = [result for result in finished if isinstance(result, Result)]
    else:
        results = list(await asyncio.gather(*tasks))
    duration = perf_counter() - t0
    if not args.quiet:
        print(file=sys.stderr)
    return results, run, duration


async def warmup(api: AsyncOpenAI, args):
    """Short requests to pay for graph capture and allocator growth up front."""
    rng = random.Random(args.seed)
    run = Run(args.warmup, 1.0, quiet=True)
    for index in range(args.warmup):
        request = Request(_prompt(rng, args.input_len, args.vocab_size), 16)
        await one_request(api, args, index, request, perf_counter(), run)


async def resolve_model_name(api: AsyncOpenAI, args) -> str:
    """Whatever this server calls the model; both engines 404 any other name."""
    if args.model_name:
        return args.model_name
    try:
        return (await api.models.list()).data[0].id
    except Exception as error:
        raise SystemExit(f"could not read /v1/models ({error}); pass --model-name") from None


async def server_metrics(http: httpx.AsyncClient, base_url: str) -> dict[str, float] | None:
    """The server's Prometheus `/metrics`, as {"name{labels}": value}, or None if it has none."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/metrics"
    try:
        response = await http.get(url)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    samples = {}
    for line in response.text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            samples[name] = float(value)
    return samples


def server_run_stats(before: dict | None, after: dict | None, duration: float) -> dict:
    """What lean-vLLM's counters moved by during the run; None where a sample is missing, as on vLLM."""

    def delta(name: str) -> float | None:
        if before is None or after is None or name not in after:
            return None
        return after[name] - before.get(name, 0.0)

    busy = delta("lean_vllm:model_busy_seconds_total")
    tokens, steps = delta("lean_vllm:iteration_tokens_total_sum"), delta("lean_vllm:iteration_tokens_total_count")
    preemptions = delta("lean_vllm:num_preemptions_total")
    return {
        # Fraction of the run's wall clock in a forward pass; nvidia-smi's figure reads higher.
        "model_busy_fraction": busy / duration if busy is not None and duration else None,
        "mean_batch_tokens": tokens / steps if tokens is not None and steps else None,
        "preemptions": int(preemptions) if preemptions is not None else None,
    }


# ---------------------------------------------------------------- reporting


def distribution(values) -> dict | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        **{f"p{p}": float(value) for p, value in zip(PERCENTILES, np.percentile(values, PERCENTILES))},
        "max": max(values),
    }


def summarize(results: list[Result], duration: float, split_labels: bool = True) -> dict:
    completed = [result for result in results if result.status == "ok"]
    failed = sum(result.status == "failed" for result in results)
    total = len(results)
    output_tokens = sum(result.output_len for result in completed)
    prompt_tokens = sum(result.prompt_len for result in completed)
    summary = {
        "num_requests": total,
        "completed": len(completed),
        "failed": failed,
        # Percentiles cover completed requests only, so show this beside them.
        "failure_rate": failed / total if total else None,
        "duration_seconds": duration,
        "goodput_requests_per_second": len(completed) / duration,
        "attempted_requests_per_second": total / duration,
        "output_token_throughput": output_tokens / duration,
        "total_token_throughput": (prompt_tokens + output_tokens) / duration,
        "ttft_seconds": distribution(result.ttft for result in completed),
        "tpot_seconds": distribution(result.tpot for result in completed),
        "itl_seconds": distribution(itl for result in completed for itl in result.itls),
        "e2e_seconds": distribution(result.latency for result in completed),
    }
    labels = sorted({result.label for result in results})
    if split_labels and len(labels) > 1:
        summary["by_label"] = {
            label: summarize([r for r in results if r.label == label], duration, split_labels=False) for label in labels
        }
    return summary


def _row(name: str, summary: dict | None) -> str:
    if summary is None:
        return f"{name:<12} --"
    cells = " ".join(f"{key} {summary[key] * 1000:8.1f}" for key in ("mean", "p50", "p99"))
    return f"{name:<12} {cells}   (ms)"


def report(summary: dict, title: str = "") -> str:
    lines = [f"--- {title} ---" if title else "---"]
    lines.append(f"{summary['completed']} completed, {summary['failed']} failed in {summary['duration_seconds']:.1f}s")
    lines.append(
        f"goodput {summary['goodput_requests_per_second']:.2f} req/s, "
        f"offered {summary['attempted_requests_per_second']:.2f} req/s, "
        f"output {summary['output_token_throughput']:.0f} tok/s"
    )
    for name in ("ttft", "tpot", "itl", "e2e"):
        lines.append(_row(name, summary[f"{name}_seconds"]))
    for label, part in (summary.get("by_label") or {}).items():
        lines += ["", report(part, f"label: {label}")]
    return "\n".join(lines)


# ---------------------------------------------------------------- entry point


async def benchmark(args, api: AsyncOpenAI, http: httpx.AsyncClient) -> dict:
    rng = random.Random(args.seed)
    trace = TRACES[args.dataset](rng, args)
    validate(trace, args)
    offsets = arrival_offsets(rng, len(trace), args.request_rate)
    args.model_name = await resolve_model_name(api, args)
    if args.warmup:
        await warmup(api, args)
    before = await server_metrics(http, args.base_url)
    results, run, duration = await run_trace(api, args, trace, offsets)
    after = await server_metrics(http, args.base_url)
    return {
        "config": vars(args) | {"num_requests": len(trace)},
        "aborted": run.aborted,
        "summary": summarize(results, duration),
        "server": {"before": before, "after": after, "run": server_run_stats(before, after, duration)},
        "requests": [asdict(result) for result in results],
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="unused", help="neither engine checks it unless told to")
    parser.add_argument("--model", default="", help="model path, for the tokenizer under --dataset sharegpt")
    parser.add_argument("--model-name", default="", help="the id in the request body; read from /v1/models if unset")
    parser.add_argument("--tokenizer", default="", help="defaults to --model")

    trace = parser.add_argument_group("trace")
    trace.add_argument("--dataset", choices=sorted(TRACES), default="lognormal")
    trace.add_argument("--dataset-path", default="", help="ShareGPT-format JSON, for --dataset sharegpt")
    trace.add_argument("--num-requests", type=int, default=200)
    trace.add_argument("--request-rate", type=float, default=4.0, help="arrivals per second; inf sends one burst")
    trace.add_argument("--input-len", type=int, default=512, help="prompt tokens, a median under lognormal")
    trace.add_argument("--output-len", type=int, default=128, help="generated tokens, a median under lognormal")
    trace.add_argument("--sigma", type=float, default=0.8, help="lognormal spread")
    trace.add_argument("--long-fraction", type=float, default=0.2, help="--dataset mixed: share of long prompts")
    trace.add_argument("--long-input-len", type=int, default=2048)
    trace.add_argument("--long-output-len", type=int, default=32)
    trace.add_argument("--long-priority", type=int, default=0, help="0 leaves the policy nothing to act on")
    trace.add_argument("--num-prefixes", type=int, default=8, help="--dataset prefix: distinct shared prefixes")
    trace.add_argument("--prefix-len", type=int, default=1024, help="--dataset prefix: tokens each one shares")
    trace.add_argument("--max-model-len", type=int, default=4096, help="clamp, so no request is refused for length")
    trace.add_argument("--vocab-size", type=int, default=10000, help="ids are drawn below this")
    trace.add_argument("--seed", type=int, default=0)

    run = parser.add_argument_group("run")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--timeout", type=float, default=600.0, help="per-request, seconds")
    run.add_argument("--max-connections", type=int, default=8192, help="never let the client be the queue")
    run.add_argument("--max-failure-rate", type=float, default=0.05, help="failures that abort the run")
    run.add_argument("--output-json", default="")
    run.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def run(args) -> dict:
    """One trace against one server. `sweep.py` calls this, not the CLI."""

    async def go():
        # A pool smaller than the offered load would queue in the client and close the loop.
        limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=args.max_connections)
        async with httpx.AsyncClient(limits=limits, timeout=args.timeout) as http:
            api = AsyncOpenAI(
                base_url=args.base_url,
                api_key=args.api_key,
                timeout=args.timeout,
                # The SDK retries a 429 or 5xx twice by default, hiding failures as a queue.
                max_retries=0,
                http_client=http,
            )
            return await benchmark(args, api, http)

    return asyncio.run(go())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(report(result["summary"], f"{args.dataset} @ {args.request_rate}/s"))
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"wrote {args.output_json}")
    return 1 if result["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
