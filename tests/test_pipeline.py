"""The pipelined step loop: one step in flight, the previous one draining."""

import pytest

from lean_vllm.sampling_params import SamplingParams

FOREVER = SamplingParams(max_tokens=64, ignore_eos=True)


def test_a_step_is_in_flight_when_step_returns(make_engine):
    engine = make_engine()
    engine.add([10, 11, 12], FOREVER)
    engine.step()
    assert not engine.is_finished()
    assert engine.in_flight is not None


def test_the_first_call_emits_no_tokens(make_engine):
    """It has nothing to drain; its own step is still in flight."""
    engine = make_engine()
    engine.add([10, 11, 12], FOREVER)
    assert engine.step() == []


def test_the_loop_drains_every_token(make_engine):
    """The drain is what stops generate() from losing the last output."""
    engine = make_engine()
    engine.add([10, 11, 12], SamplingParams(max_tokens=4, ignore_eos=True))
    collected = engine.run_to_completion()
    assert len(next(iter(collected.values()))) == 4


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_an_abort_against_an_in_flight_step_frees_its_blocks(make_engine, async_scheduling):
    """Intake is drained between calls, so an abort now lands on a launched step."""
    engine = make_engine(async_scheduling=async_scheduling)
    seq = engine.add([10, 11, 12], FOREVER)
    engine.step()
    assert engine.scheduler.abort(seq.request_id)
    engine.step()
    assert engine.is_finished()
    assert not engine.scheduler.block_manager.used_block_ids


def test_two_requests_finish_with_the_tokens_the_runner_sampled(make_engine):
    engine = make_engine()
    engine.add([10, 11, 12], SamplingParams(max_tokens=3, ignore_eos=True))
    engine.add([20, 21], SamplingParams(max_tokens=3, ignore_eos=True))
    collected = engine.run_to_completion()
    assert all(len(tokens) == 3 for tokens in collected.values())


class TestAsyncScheduling:
    def test_the_flag_launches_before_the_last_step_is_reconciled(self, make_engine):
        """Stopping on EOS costs one extra launched step, scheduled before the stop is known."""
        runs = {}
        for async_scheduling in (False, True):
            engine = make_engine(eos_after={"r": 2}, async_scheduling=async_scheduling)
            engine.add([10, 11, 12], SamplingParams(max_tokens=64), request_id="r")
            tokens = engine.run_to_completion()
            runs[async_scheduling] = (len(engine.model_runner.batches), tokens)

        sync_batches, sync_tokens = runs[False]
        async_batches, async_tokens = runs[True]
        assert async_tokens == sync_tokens  # the extra token is never emitted
        assert async_batches == sync_batches + 1  # but it was computed

    def test_the_flag_does_not_change_what_a_request_receives(self, make_engine):
        """The extra token is never emitted, so text and finish reasons match."""
        params = SamplingParams(max_tokens=6, ignore_eos=True)
        off = make_engine()
        off.add([10, 11, 12], params, request_id="r")
        sync_tokens = off.run_to_completion()

        on = make_engine(async_scheduling=True)
        on.add([10, 11, 12], params, request_id="r")
        async_tokens = on.run_to_completion()

        assert async_tokens == sync_tokens

    def test_an_eos_one_step_late_still_finishes_with_stop(self, make_engine):
        engine = make_engine(eos_after={"r": 2}, async_scheduling=True)
        seq = engine.add([10, 11, 12], SamplingParams(max_tokens=64), request_id="r")
        engine.run_to_completion()
        assert seq.finish_reason == "stop"
        assert seq.num_pending_tokens == 0

    def test_a_preemption_against_an_in_flight_step_keeps_the_text_right(self, make_engine):
        params = SamplingParams(max_tokens=16, ignore_eos=True)
        off = make_engine(num_kvcache_blocks=6)
        off.add(list(range(13)), params, request_id="a")
        off.add(list(range(100, 108)), params, request_id="b")
        sync_tokens = off.run_to_completion()

        on = make_engine(num_kvcache_blocks=6, async_scheduling=True)
        on.add(list(range(13)), params, request_id="a")
        on.add(list(range(100, 108)), params, request_id="b")
        assert on.run_to_completion() == sync_tokens

    def test_the_prefix_cache_hit_rate_matches(self, make_engine):
        """A block published over an unsampled token shows up here and nowhere else."""
        params = SamplingParams(max_tokens=8, ignore_eos=True)
        rates = []
        for async_scheduling in (False, True):
            engine = make_engine(async_scheduling=async_scheduling)
            for i in range(3):
                engine.add(list(range(24)), params, request_id=f"r{i}")
                engine.run_to_completion()
            rates.append((engine.metrics.prefix_cache_hits.total, engine.metrics.prefix_cache_queries.total))
        assert rates[0] == rates[1]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_an_eos_that_fills_the_cache_still_stops(make_engine, async_scheduling):
    """A launch must not drop a request for capacity while its token is in flight."""
    engine = make_engine(eos_after={"r": 0}, num_kvcache_blocks=1, async_scheduling=async_scheduling)
    seq = engine.add(list(range(8)), SamplingParams(max_tokens=4), request_id="r")
    engine.run_to_completion()
    assert seq.finish_reason == "stop"
    assert seq.num_completion_tokens == 1


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_a_capacity_drop_after_a_token_finishes_once(make_engine, async_scheduling):
    engine = make_engine(num_kvcache_blocks=1, async_scheduling=async_scheduling)
    seq = engine.add(list(range(7)), FOREVER, request_id="r")
    finals = []
    while not engine.is_finished():
        finals += [output for output in engine.step() if output.finished]
    assert [output.finish_reason for output in finals] == ["capacity"]
    assert engine.metrics.requests_finished.values == {"capacity": 1}
    assert seq.num_completion_tokens == 2


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_the_first_token_is_timed_when_it_is_read_back(make_engine, async_scheduling):
    engine = make_engine(async_scheduling=async_scheduling)
    seq = engine.add([10, 11, 12], FOREVER)
    engine.step()
    assert seq.first_token_time is None  # sampled, not yet read back
    engine.step()
    assert seq.first_token_time is not None
