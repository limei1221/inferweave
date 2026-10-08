"""The registry's exposition format, and what the engine actually records."""

import pytest

from lean_vllm.engine.metrics import Counter, Gauge, Histogram, Metrics
from lean_vllm.sampling_params import SamplingParams

FOREVER = SamplingParams(max_tokens=64, ignore_eos=True)


def prompt(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


class TestExposition:
    def test_a_counter_renders_its_total(self):
        counter = Counter("lean_vllm:things_total", "Things.")
        counter.inc()
        counter.inc(3)
        assert counter.render()[-1] == "lean_vllm:things_total 4"

    def test_a_labelled_counter_renders_one_line_per_value(self):
        counter = Counter("lean_vllm:request_success_total", "Finishes.", label="finish_reason")
        counter.inc(label_value="stop")
        counter.inc(label_value="length")
        counter.inc(label_value="stop")
        assert counter.render()[-2:] == [
            'lean_vllm:request_success_total{finish_reason="length"} 1',
            'lean_vllm:request_success_total{finish_reason="stop"} 2',
        ]

    def test_a_gauge_renders_the_last_value_set(self):
        gauge = Gauge("lean_vllm:num_requests_running", "Running.")
        gauge.set(5)
        gauge.set(2)
        assert gauge.render()[-1] == "lean_vllm:num_requests_running 2"

    def test_histogram_buckets_are_cumulative(self):
        histogram = Histogram("lean_vllm:seconds", "Seconds.", (1.0, 10.0, float("inf")))
        for value in (0.5, 5.0, 50.0):
            histogram.observe(value)
        rendered = histogram.render()
        assert rendered[-5:-1] == [
            'lean_vllm:seconds_bucket{le="1"} 1',
            'lean_vllm:seconds_bucket{le="10"} 2',
            'lean_vllm:seconds_bucket{le="+Inf"} 3',
            "lean_vllm:seconds_sum 55.5",
        ]
        assert rendered[-1] == "lean_vllm:seconds_count 3"

    def test_every_metric_is_declared_before_it_is_sampled(self):
        """A sample with no preceding # TYPE is not scrapeable."""
        declared = set()
        for line in Metrics().render().splitlines():
            if line.startswith("# TYPE"):
                declared.add(line.split(" ")[2])
            elif not line.startswith("#"):
                name = line.split(" ")[0].split("{")[0]
                base = name.rsplit("_", 1)[0] if name.endswith(("_bucket", "_sum", "_count")) else name
                assert base in declared or name in declared, name


class TestEngineRecording:
    def test_a_finished_request_lands_in_the_latency_histograms(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.run_to_completion()
        metrics = engine.metrics
        assert metrics.ttft.count == 1 and metrics.e2e.count == 1
        assert metrics.queue_time.count == 1
        assert metrics.tpot.count == 1  # three tokens, so there is a per-token rate
        assert metrics.requests_finished.values == {"length": 1}

    def test_a_one_token_request_records_no_tpot(self, make_engine):
        """TPOT is the rate after the first token, and one token has no after."""
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=1, ignore_eos=True))
        engine.run_to_completion()
        assert engine.metrics.ttft.count == 1
        assert engine.metrics.tpot.count == 0

    def test_token_counters_split_prefill_from_decode(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        engine.run_to_completion()
        metrics = engine.metrics
        assert metrics.prefill_tokens.total == 8  # one chunk of the whole prompt
        assert metrics.decode_tokens.total == 2  # the prefill step sampled the first token
        assert metrics.prompt_tokens.total == 8 and metrics.generation_tokens.total == 3

    def test_queue_depths_and_kv_usage_are_gauged(self, make_engine):
        engine = make_engine(num_kvcache_blocks=2)
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(8, 100), FOREVER)
        engine.step()
        assert engine.metrics.running.value == 2
        assert engine.metrics.waiting.value == 0
        assert engine.metrics.kv_usage.value == 1.0  # both blocks taken

    def test_preemptions_are_counted(self, make_engine):
        engine = make_engine(num_kvcache_blocks=3, kvcache_block_size=8, max_num_seqs=2)
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(8, 100), FOREVER)
        for _ in range(4):
            engine.step()
        assert engine.metrics.preemptions.total > 0

    def test_the_prefix_cache_hit_rate_counts_blocks(self, make_engine):
        engine = make_engine()
        engine.add(prompt(16), FOREVER)
        engine.step()
        engine.add(prompt(16), FOREVER)  # same prompt, so one full block hits
        engine.step()
        metrics = engine.metrics
        assert metrics.prefix_cache_queries.total == 4  # two blocks each
        assert metrics.prefix_cache_hits.total == 1  # the trailing block is never a candidate

    def test_admitted_requests_are_counted(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), FOREVER)
        engine.add(prompt(8, 100), FOREVER)
        assert engine.metrics.requests_received.total == 2

    def test_a_step_that_scheduled_nothing_is_not_a_forward_pass(self, make_engine):
        """Otherwise an idle poll loop would inflate the step count and deflate the busy fraction."""
        engine = make_engine()
        engine.step()
        assert engine.metrics.steps.total == 0
        engine.add(prompt(8), FOREVER)
        engine.step()
        assert engine.metrics.steps.total == 1

    def test_each_step_is_counted_by_how_it_ran(self, make_engine):
        engine = make_engine()
        engine.add(prompt(8), SamplingParams(max_tokens=2, ignore_eos=True))
        engine.run_to_completion()
        metrics = engine.metrics
        assert metrics.requests_finished.values == {"length": 1}
        assert metrics.graph_steps.total == 0  # the fake runner captures no graphs
        assert metrics.eager_steps.values == {"enforced": metrics.steps.total}
        # the same clock as model_busy, split by kind
        assert metrics.step_seconds.values["enforced"] == pytest.approx(metrics.model_busy.total)
        assert metrics.e2e.count == 1
