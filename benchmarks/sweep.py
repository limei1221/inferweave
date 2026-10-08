r"""Runs `bench_serving` across arms and rates, with a fresh server per run.

    uv run python benchmarks/sweep.py --model ~/workspace/huggingface/Qwen3-8B \
        --suite rate --rates 1,2,4,8,16 --kvcache-tokens 131072 --out results/8b

Pin `--kvcache-tokens`, or the profiled cache size moves with the token budget.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts, not a package

import bench_serving as bench


@dataclass
class Arm:
    """One server configuration, and the client flags the comparison needs."""

    name: str
    server: dict = field(default_factory=dict)
    client: dict = field(default_factory=dict)


def rate_suite(args) -> list[Arm]:
    return [Arm("default")]


def chunked_suite(args) -> list[Arm]:
    """Chunking on and off, each eager and with graphs; the eager pair isolates scheduling."""
    return [
        Arm(f"chunked={chunked}-eager={eager}", {"enable-chunked-prefill": chunked, "enforce-eager": eager})
        for chunked in (True, False)
        for eager in (True, False)
    ]


def budget_suite(args) -> list[Arm]:
    return [Arm(f"budget={budget}", {"max-num-batched-tokens": budget}) for budget in args.budgets]


def policy_suite(args) -> list[Arm]:
    """Long prompts arrive at priority 1, so only `priority` has anything to act on."""
    client = {"dataset": "mixed", "long-priority": 1}
    return [Arm(f"policy={policy}", {"scheduling-policy": policy}, client) for policy in ("fcfs", "priority")]


def prefix_suite(args) -> list[Arm]:
    """What prefix caching is worth, and what the slower hash costs. Runs on either engine."""
    client = {"dataset": "prefix"}
    return [
        Arm("prefix-caching=off", {"enable-prefix-caching": False}, client),
        Arm("hash=sha256", {"prefix-caching-hash-algo": "sha256"}, client),
        Arm("hash=xxhash", {"prefix-caching-hash-algo": "xxhash"}, client),
    ]


def starvation_suite(args) -> list[Arm]:
    """Does capping one prompt's share of a step protect short ones?"""
    client = {"dataset": "mixed"}
    return [
        Arm(f"long-prefill-threshold={threshold}", {"long-prefill-token-threshold": threshold}, client)
        for threshold in (0, 512)
    ]


def async_suite(args) -> list[Arm]:
    """Launching the next step before draining the last. vLLM takes the same flag."""
    return [Arm(f"async-scheduling={on}", {"async-scheduling": on}) for on in (False, True)]


SUITES = {
    "rate": rate_suite,
    "async": async_suite,
    "chunked": chunked_suite,
    "budget": budget_suite,
    "policy": policy_suite,
    "prefix": prefix_suite,
    "starvation": starvation_suite,
}

# vLLM's default, which FA3 also takes; pinned so block counts mean the same on both engines.
BLOCK_SIZE = 16


def cache_flags(args) -> dict:
    """The same block size and KV capacity on either engine, whatever it calls them."""
    lean = args.engine == "lean-vllm"
    flags = {"kvcache-block-size" if lean else "block-size": BLOCK_SIZE}
    if args.kvcache_tokens:
        flags["num-kvcache-blocks" if lean else "num-gpu-blocks-override"] = args.kvcache_tokens // BLOCK_SIZE
    return flags


def server_command(args, arm: Arm) -> list[str]:
    command = ["lean-vllm", "serve"] if args.engine == "lean-vllm" else ["vllm", "serve"]
    command += [args.model, "--host", args.host, "--port", str(args.port)]
    flags = dict(arm.server)
    flags.setdefault("max-model-len", args.max_model_len)
    flags |= cache_flags(args)
    for name, value in flags.items():
        if isinstance(value, bool):
            command.append(f"--{name}" if value else f"--no-{name}")
        else:
            command += [f"--{name}", str(value)]
    return command + args.server_args


def wait_for_health(base_url: str, process: subprocess.Popen, timeout: float, log: Path):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"the server exited with {process.returncode}; see {log}")
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise SystemExit(f"the server was not healthy within {timeout}s; see {log}")


class Server:
    """The engine under test, brought up and taken down around one arm."""

    def __init__(self, args, arm: Arm, log: Path):
        self.command = server_command(args, arm)
        self.base_url = f"http://{args.host}:{args.port}"
        self.timeout = args.startup_timeout
        self.log = log

    def __enter__(self):
        print(f"$ {' '.join(self.command)}", flush=True)
        self.handle = self.log.open("w")
        self.process = subprocess.Popen(self.command, stdout=self.handle, stderr=subprocess.STDOUT)
        try:
            wait_for_health(self.base_url, self.process, self.timeout, self.log)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exception):
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.handle.close()


def client_args(args, arm: Arm, rate: float):
    argv = [
        "--base-url",
        f"http://{args.host}:{args.port}/v1",
        "--model",
        args.model,
        "--request-rate",
        str(rate),
        "--num-requests",
        str(args.num_requests),
        "--max-model-len",
        str(args.max_model_len),
        "--seed",
        str(args.seed),
        "--quiet",
    ]
    for name, value in arm.client.items():
        argv += [f"--{name}", str(value)]
    return bench.parse_args(argv + args.client_args)  # the operator's flags win


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def row(arm: Arm, rate: float, result: dict) -> dict:
    summary = result["summary"]
    server = result["server"]["run"]

    def at(name, key):
        return (summary[name] or {}).get(key)

    return {
        "arm": arm.name,
        "request_rate": rate,
        "completed": summary["completed"],
        "failure_rate": summary["failure_rate"],
        "goodput": summary["goodput_requests_per_second"],
        "output_tok_s": summary["output_token_throughput"],
        "ttft_p50": at("ttft_seconds", "p50"),
        "ttft_p99": at("ttft_seconds", "p99"),
        "tpot_p50": at("tpot_seconds", "p50"),
        "tpot_p99": at("tpot_seconds", "p99"),
        "e2e_p99": at("e2e_seconds", "p99"),
        "model_busy_fraction": server.get("model_busy_fraction"),
        "mean_batch_tokens": server.get("mean_batch_tokens"),
        "preemptions": server.get("preemptions"),
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
    }


# (key, heading, format)
COLUMNS = [
    ("arm", "arm", "s"),
    ("request_rate", "rate", ".1f"),
    ("completed", "done", "d"),
    ("failure_rate", "failed", ".1%"),
    ("goodput", "goodput", ".2f"),
    ("output_tok_s", "tok/s", ".0f"),
    ("ttft_p50", "ttft_p50", ".3f"),
    ("ttft_p99", "ttft_p99", ".3f"),
    ("tpot_p50", "tpot_p50", ".4f"),
    ("e2e_p99", "e2e_p99", ".3f"),
    ("model_busy_fraction", "busy", ".2f"),
    ("preemptions", "preempt", "d"),
]


def _cell(entry: dict, key: str, spec: str) -> str:
    value = entry.get(key)
    return "--" if value is None else f"{value:{spec}}"


class Table:
    """Fixed widths, so a row printed as its run finishes lines up under the header."""

    def __init__(self, arms: list[Arm]):
        longest = max(len(arm.name) for arm in arms)
        self.widths = {key: max(len(heading), longest if key == "arm" else 9) for key, heading, _ in COLUMNS}

    def header(self) -> str:
        return "  ".join(f"{heading:>{self.widths[key]}}" for key, heading, _ in COLUMNS)

    def row(self, entry: dict) -> str:
        return "  ".join(f"{_cell(entry, key, spec):>{self.widths[key]}}" for key, _, spec in COLUMNS)

    def render(self, rows: list[dict]) -> str:
        return "\n".join([self.header()] + [self.row(entry) for entry in rows])


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to a local model directory")
    parser.add_argument("--engine", choices=("lean-vllm", "vllm"), default="lean-vllm")
    parser.add_argument("--suite", choices=sorted(SUITES), default="rate")
    parser.add_argument("--rates", default="1,2,4,8,16", help="arrivals per second, comma separated")
    parser.add_argument("--budgets", default="512,2048,8192", help="--suite budget: max_num_batched_tokens values")
    parser.add_argument("--num-requests", type=int, default=300)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--kvcache-tokens",
        type=int,
        default=0,
        help="KV cache capacity in tokens, converted to each engine's blocks; 0 profiles",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--out", default="results", help="directory for the per-run JSON and the summary")
    # One quoted string each, so their flags are not parsed as this parser's.
    parser.add_argument("--server-args", default="", help='extra serve flags, e.g. "--max-num-seqs 64"')
    parser.add_argument("--client-args", default="", help='extra bench_serving flags, e.g. "--dataset mixed"')
    args = parser.parse_args(argv)
    args.rates = [float(rate) for rate in args.rates.split(",")]
    args.budgets = [int(budget) for budget in args.budgets.split(",")]
    args.server_args = shlex.split(args.server_args)
    args.client_args = shlex.split(args.client_args)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.kvcache_tokens:
        blocks = args.kvcache_tokens // BLOCK_SIZE
        print(f"kv cache pinned to {blocks * BLOCK_SIZE} tokens ({blocks} blocks of {BLOCK_SIZE})", file=sys.stderr)
    else:
        print("warning: --kvcache-tokens is unpinned, so the cache size moves with the token budget", file=sys.stderr)
    out = Path(args.out) / args.suite
    out.mkdir(parents=True, exist_ok=True)
    arms = SUITES[args.suite](args)
    rows, aborted = [], False
    printer = Table(arms)
    print(printer.header(), flush=True)

    for arm in arms:
        for rate in args.rates:
            stem = f"{arm.name}-rate{rate:g}"
            with Server(args, arm, out / f"{stem}.server.log"):
                started_at = utc_now()  # the client run's wall-clock bounds, for the GPU log
                result = bench.run(client_args(args, arm, rate))
                result["started_at"], result["finished_at"] = started_at, utc_now()
            (out / f"{stem}.json").write_text(json.dumps(result, indent=2))
            entry = row(arm, rate, result)
            rows.append(entry)
            aborted |= result["aborted"]
            print(printer.row(entry), flush=True)

    summary = {
        "engine": args.engine,
        "suite": args.suite,
        "config": vars(args),
        "arms": {arm.name: {"server": arm.server, "client": arm.client} for arm in arms},
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + printer.render(rows))
    print(f"\nwrote {out}/summary.json")
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
