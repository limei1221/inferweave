"""The benchmark client against the HTTP layer and a fake engine, so no GPU.

Under test: a 429 is a never-retried rejection, a 503 a failure, and percentiles cover completions only.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the serve extra is not installed")

import httpx
from openai import AsyncOpenAI

from lean_vllm.engine.exceptions import EngineDeadError
from lean_vllm.engine.scheduler import QueueFull
from lean_vllm.entrypoints.api_server import build_app

from conftest import asyncio_test
from test_api_server import MODEL, FakeAsyncEngine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))    # scripts, not a package

import bench_serving as bench


def make_args(*extra: str):
    defaults = ["--warmup", "0", "--quiet", "--dataset", "fixed", "--input-len", "8", "--output-len", "4"]
    return bench.parse_args(defaults + list(extra))


@asynccontextmanager
async def clients(engine):
    """The SDK and the raw client, both bound to the app with no socket between."""
    transport = httpx.ASGITransport(app=build_app(engine, MODEL))
    async with httpx.AsyncClient(transport=transport, base_url="http://engine") as http:
        yield AsyncOpenAI(base_url="http://engine/v1", api_key="unused", max_retries=0, http_client=http), http


class TestTrace:

    def test_an_infinite_rate_is_one_burst(self):
        assert bench.arrival_offsets(bench.random.Random(0), 5, float("inf")) == [0.0] * 5

    def test_arrivals_are_ordered_and_average_the_requested_rate(self):
        offsets = bench.arrival_offsets(bench.random.Random(0), 4000, 8.0)
        assert offsets == sorted(offsets)
        assert 0.9 < (offsets[-1] / len(offsets)) * 8.0 < 1.1

    def test_prompts_are_the_requested_length_and_never_shared(self):
        trace = bench.fixed_trace(bench.random.Random(0), make_args("--num-requests", "3"))
        assert [request.prompt_len for request in trace] == [8, 8, 8]
        assert len({tuple(request.prompt_token_ids) for request in trace}) == 3

    def test_lognormal_lengths_stay_inside_the_context(self):
        args = make_args("--dataset", "lognormal", "--num-requests", "200", "--max-model-len", "128")
        trace = bench.lognormal_trace(bench.random.Random(0), args)
        assert all(request.prompt_len + request.output_len <= 128 for request in trace)
        assert len({request.prompt_len for request in trace}) > 1

    def test_a_trace_over_the_context_is_refused_before_the_run(self):
        args = make_args("--num-requests", "2", "--input-len", "64", "--max-model-len", "32")
        with pytest.raises(SystemExit, match="exceed --max-model-len"):
            bench.validate(bench.fixed_trace(bench.random.Random(0), args), args)

    def test_mixed_labels_and_prioritises_the_long_prompts(self):
        args = make_args(
            "--dataset", "mixed", "--num-requests", "200",
            "--long-fraction", "0.25", "--long-input-len", "64", "--long-priority", "1",
        )
        trace = bench.mixed_trace(bench.random.Random(0), args)
        long = [request for request in trace if request.label == "long"]
        assert 30 < len(long) < 70
        assert all(request.prompt_len == 64 and request.priority == 1 for request in long)
        assert all(request.priority == 0 for request in trace if request.label == "short")

    def test_prefix_shares_a_head_and_keeps_the_tails_distinct(self):
        args = make_args(
            "--dataset", "prefix", "--num-requests", "50",
            "--num-prefixes", "4", "--prefix-len", "16", "--input-len", "24",
        )
        trace = bench.prefix_trace(bench.random.Random(0), args)
        heads = {tuple(request.prompt_token_ids[:16]) for request in trace}
        assert len(heads) == 4
        assert all(request.prompt_len == 24 for request in trace)
        assert len({tuple(request.prompt_token_ids) for request in trace}) == 50


class TestStatistics:

    def test_percentiles_interpolate(self):
        summary = bench.distribution([4.0, 0.0, 3.0, 1.0, 2.0])
        assert summary["p50"] == 2.0
        assert summary["p99"] == pytest.approx(3.96)

    def test_a_single_sample_is_every_percentile(self):
        summary = bench.distribution([0.25])
        assert summary == {"count": 1, "mean": 0.25, "p50": 0.25, "p90": 0.25, "p95": 0.25, "p99": 0.25, "max": 0.25}

    def test_a_distribution_of_nothing_is_none_rather_than_zero(self):
        assert bench.distribution([None, None]) is None

    def test_percentiles_cover_completed_requests_only(self):
        results = [
            bench.Result(0, "all", 0, 4, 4, 0.0, "ok", output_len=4, ttft=0.1, latency=0.4),
            bench.Result(1, "all", 0, 4, 4, 0.0, "rejected"),
            bench.Result(2, "all", 0, 4, 4, 0.0, "failed"),
        ]
        summary = bench.summarize(results, duration=2.0)
        assert summary["ttft_seconds"]["count"] == 1
        assert summary["rejection_rate"] == pytest.approx(1 / 3)
        assert summary["failure_rate"] == pytest.approx(1 / 3)
        assert summary["goodput_requests_per_second"] == pytest.approx(0.5)
        assert summary["attempted_requests_per_second"] == pytest.approx(1.5)


class TestRun:

    def test_failures_past_the_budget_abort_the_run(self):
        run = bench.Run(total=100, max_failure_rate=0.05, quiet=True)
        for index in range(5):
            run.finish(bench.Result(index, "all", 0, 4, 4, 0.0, "failed", error="boom"))
        assert not run.aborted
        run.finish(bench.Result(5, "all", 0, 4, 4, 0.0, "failed", error="boom"))
        assert run.aborted and run.first_error == "boom"

    def test_rejections_never_abort_the_run(self):
        run = bench.Run(total=10, max_failure_rate=0.0, quiet=True)
        for index in range(10):
            run.finish(bench.Result(index, "all", 0, 4, 4, 0.0, "rejected"))
        assert not run.aborted


class TestAgainstTheServer:

    @asyncio_test
    async def test_every_request_completes_and_is_timed(self):
        engine = FakeAsyncEngine()
        args = make_args("--num-requests", "6", "--request-rate", "inf")
        async with clients(engine) as (api, http):
            result = await bench.benchmark(args, api, http)
        summary = result["summary"]
        assert summary["completed"] == 6 and summary["rejected"] == 0 and summary["failed"] == 0
        assert len(engine.requests) == 6
        assert all(record["ttft"] > 0 and record["latency"] >= record["ttft"] for record in result["requests"])
        # Two scripted pieces, so usage says two tokens and one inter-token gap.
        assert {record["output_len"] for record in result["requests"]} == {2}
        assert all(len(record["itls"]) == 1 for record in result["requests"])
        assert result["server"]["after"]["requests"]["received"] == 6

    @asyncio_test
    async def test_the_model_name_comes_off_the_server(self):
        engine = FakeAsyncEngine()
        args = make_args("--num-requests", "1", "--request-rate", "inf")
        async with clients(engine) as (api, http):
            result = await bench.benchmark(args, api, http)
        assert result["config"]["model_name"] == MODEL    # vLLM 404s anything else

    @asyncio_test
    async def test_an_explicit_model_name_wins(self):
        engine = FakeAsyncEngine()
        args = make_args("--num-requests", "1", "--request-rate", "inf", "--model-name", "chosen")
        async with clients(engine) as (api, http):
            result = await bench.benchmark(args, api, http)
        assert result["config"]["model_name"] == "chosen"

    @asyncio_test
    async def test_the_trace_sets_the_prompt_the_engine_sees(self):
        engine = FakeAsyncEngine()
        args = make_args("--num-requests", "2", "--request-rate", "inf", "--input-len", "11")
        async with clients(engine) as (api, http):
            await bench.benchmark(args, api, http)
        assert [len(prompt) for prompt, _, _ in engine.requests] == [11, 11]

    @asyncio_test
    async def test_a_429_is_a_rejection_and_is_not_retried(self):
        engine = FakeAsyncEngine()
        engine.admission_error = QueueFull("the queue is full")
        args = make_args("--num-requests", "4", "--request-rate", "inf")
        async with clients(engine) as (api, http):
            result = await bench.benchmark(args, api, http)
        assert result["summary"]["rejected"] == 4 and result["summary"]["failed"] == 0
        assert result["summary"]["rejection_rate"] == 1.0
        assert len(engine.requests) == 4, "a retry would make the offered load a lie"
        assert not result["aborted"]

    @asyncio_test
    async def test_a_503_is_a_failure_and_aborts_the_run(self):
        engine = FakeAsyncEngine()
        engine.admission_error = EngineDeadError("the engine thread died")
        args = make_args("--num-requests", "8", "--request-rate", "inf")
        async with clients(engine) as (api, http):
            result = await bench.benchmark(args, api, http)
        assert result["summary"]["failed"] == 8 and result["summary"]["rejected"] == 0
        assert result["aborted"]

    @asyncio_test
    async def test_a_mixed_trace_reports_each_class_on_its_own(self):
        engine = FakeAsyncEngine()
        args = make_args(
            "--dataset", "mixed", "--num-requests", "20", "--request-rate", "inf",
            "--long-fraction", "0.5", "--long-input-len", "16", "--long-priority", "1",
        )
        async with clients(engine) as (api, http):
            result = await bench.benchmark(args, api, http)
        by_label = result["summary"]["by_label"]
        assert set(by_label) == {"long", "short"}
        assert by_label["long"]["completed"] + by_label["short"]["completed"] == 20
        assert {params.priority for _, params, _ in engine.requests} == {0, 1}

    @asyncio_test
    async def test_an_error_event_mid_stream_is_a_failure(self):
        engine = FakeAsyncEngine(finish_reason="capacity")    # the engine dropped it
        args = make_args("--num-requests", "1", "--request-rate", "inf")
        async with clients(engine) as (api, http):
            result = await bench.benchmark(args, api, http)
        assert result["summary"]["failed"] == 1
        assert "capacity" in result["requests"][0]["error"]
