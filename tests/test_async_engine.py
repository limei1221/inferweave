"""AsyncLLM's front end: streaming out, abort in, and what a dead engine owes its callers.

It runs over AsyncInprocClient below, which steps a fake engine on a thread of this process, so a test can gate
each step and read the scheduler directly. test_core_client.py covers the real client across a process.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, wait

import pytest

pytest.importorskip("zmq", reason="the serve extra is not installed")

from conftest import FakeConfig, FakeLLMEngine, FakeModelRunner, asyncio_test

from lean_vllm.engine.async_llm import AsyncLLM
from lean_vllm.engine.exceptions import EngineDeadError
from lean_vllm.engine.scheduler import DuplicateRequestId
from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams

FOREVER = SamplingParams(max_tokens=64, ignore_eos=True)


def prompt(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


class GatedModelRunner(FakeModelRunner):
    """One forward pass per release(), so a test is never racing the engine."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Semaphore(0)

    def run(self, seqs):
        self.gate.acquire()
        return super().run(seqs)

    def release(self, steps: int = 1):
        for _ in range(steps):
            self.gate.release()


class AsyncInprocClient:
    """AsyncMPClient's surface over an engine in this process, each step on a worker thread.

    A lock keeps adds and aborts off the engine while a step runs; aborts that arrive meanwhile land after it.
    """

    def __init__(self, engine: FakeLLMEngine):
        self.engine = engine
        self.error: BaseException | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._outputs: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._aborts: list[tuple[str, str]] = []
        self._has_work = asyncio.Event()
        self._stopped = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    def start(self):
        self._loop = asyncio.get_running_loop()
        self._task = self._loop.create_task(self._run())

    def shutdown(self, timeout: float | None = None):
        """Callable from any thread. Waits up to timeout for the step in flight."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._loop.call_soon_threadsafe(self._has_work.set)
        except (AttributeError, RuntimeError):
            pass  # never started, or the loop is closed
        done, _ = wait([self._executor.submit(lambda: None)], timeout)  # queued behind the step in flight
        self._executor.shutdown(wait=bool(done))

    async def add_request_async(self, request_id, prompt, sampling_params):
        async with self._lock:
            if self._stopped:
                raise EngineDeadError("the server is shutting down")
            if self.error is not None:
                raise dead_error(self.error)
            self.engine.add_request(prompt, sampling_params, request_id)
            self._has_work.set()

    def abort_request(self, request_id: str, reason: str = "abort"):
        if self.error is not None or self._stopped:
            return
        if self._lock.locked():
            self._aborts.append((request_id, reason))
        else:
            self.engine.abort_request(request_id, reason)

    async def get_output_async(self):
        item = await self._outputs.get()
        if isinstance(item, EngineDeadError):
            raise item
        return item

    async def render_metrics_async(self):
        return self.engine.metrics.render()

    async def _run(self):
        try:
            while not self._stopped:
                if self.engine.is_finished():
                    self._has_work.clear()
                    await self._has_work.wait()
                    continue
                async with self._lock:
                    outputs, num_prefill_tokens, num_decode_tokens = await self._loop.run_in_executor(
                        self._executor, self.engine.step
                    )
                    for request_id, reason in self._aborts:
                        self.engine.abort_request(request_id, reason)
                    self._aborts.clear()
                if outputs:
                    self._outputs.put_nowait(outputs)
                if not (outputs or num_prefill_tokens or num_decode_tokens):
                    await asyncio.sleep(0.005)
        except Exception as error:
            if not self._stopped:  # else a step refused by the shut-down executor
                self.error = error
                self._outputs.put_nowait(dead_error(error))


def dead_error(error: BaseException) -> EngineDeadError:
    dead = EngineDeadError(f"the engine thread died: {error!r}")
    dead.__cause__ = error
    return dead


@pytest.fixture
def make_async_engine():
    engines = []

    def _make(gated: bool = False, **overrides) -> AsyncLLM:
        config = FakeConfig(**overrides)
        Sequence.block_size = config.kvcache_block_size
        runner = GatedModelRunner() if gated else FakeModelRunner()
        engine = AsyncLLM(AsyncInprocClient(FakeLLMEngine(config, runner)))
        engine.start()
        engines.append((engine, runner))
        return engine

    yield _make
    for engine, runner in engines:
        if isinstance(runner, GatedModelRunner):
            runner.release(100)  # so a blocked step lets the stop finish
        engine.stop(timeout=5)


def fake_of(engine: AsyncLLM) -> FakeLLMEngine:
    return engine.engine_core.engine


def runner_of(engine: AsyncLLM) -> GatedModelRunner:
    return fake_of(engine).model_runner


async def collect(engine: AsyncLLM, tokens: int, prompt_ids: list[int]) -> list[int]:
    outputs = await engine.add_request(prompt_ids, SamplingParams(max_tokens=tokens, ignore_eos=True))
    return [token async for output in outputs for token in output.token_ids]


class TestStreaming:
    @asyncio_test
    async def test_every_token_reaches_the_caller(self, make_async_engine):
        engine = make_async_engine()
        assert len(await collect(engine, 5, prompt(8))) == 5

    @asyncio_test
    async def test_the_last_output_carries_the_reason_and_metrics(self, make_async_engine):
        engine = make_async_engine()
        outputs = await engine.add_request(prompt(8), SamplingParams(max_tokens=3, ignore_eos=True))
        last = [output async for output in outputs][-1]
        assert last.finished and last.finish_reason == "length"
        assert last.metrics.ttft is not None

    @asyncio_test
    async def test_a_step_feeds_every_stream_it_produced(self, make_async_engine):
        engine = make_async_engine()
        results = await asyncio.gather(*[collect(engine, 4, prompt(8, i * 100)) for i in range(3)])
        assert [len(tokens) for tokens in results] == [4, 4, 4]

    @asyncio_test
    async def test_the_loop_idles_between_requests(self, make_async_engine):
        """Not a spin loop: the runner is untouched while nothing is scheduled."""
        engine = make_async_engine()
        await asyncio.sleep(0.05)
        assert runner_of(engine).batches == []

    @asyncio_test
    async def test_the_last_token_of_a_request_is_not_lost(self, make_async_engine):
        """The pipeline holds a step past the queues emptying, so the loop must drain it."""
        engine = make_async_engine()
        stream = await engine.add_request([10, 11, 12], SamplingParams(max_tokens=3, ignore_eos=True))
        collected = [output async for output in stream]
        assert sum(len(output.token_ids) for output in collected) == 3
        assert collected[-1].finished


class TestAbort:
    @pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
    @pytest.mark.parametrize("disconnect_at", ["http.response.start", "http.response.body"])
    @asyncio_test
    async def test_chat_disconnect_before_tokens_frees_the_blocks(
        self,
        make_async_engine,
        spec_version,
        disconnect_at,
    ):
        pytest.importorskip("fastapi", reason="the serve extra is not installed")
        from starlette.requests import ClientDisconnect

        from lean_vllm.entrypoints.api_server import _serve
        from lean_vllm.entrypoints.protocol import ChatCompletionRequest

        engine = make_async_engine(gated=True)
        blocks = fake_of(engine).scheduler.block_manager
        free_before = len(blocks.free_block_ids)
        body = ChatCompletionRequest(
            model="fake",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        response = await _serve(engine, "fake", body, prompt(8), chat=True)
        disconnected = asyncio.Event()

        async def send(message):
            if message["type"] == disconnect_at:
                if spec_version == "2.4":
                    raise OSError("client disconnected")
                disconnected.set()
                await asyncio.Future()  # cancelled by the disconnect listener

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        scope = {"type": "http", "asgi": {"spec_version": spec_version}}
        if spec_version == "2.4":
            with pytest.raises(ClientDisconnect):
                await asyncio.wait_for(response(scope, receive, send), timeout=1)
        else:
            await asyncio.wait_for(response(scope, receive, send), timeout=1)

        runner_of(engine).release()  # drain the abort after the in-flight step
        await settle(engine)
        assert not engine.output_processor.request_states
        assert not fake_of(engine).scheduler.seqs
        assert len(blocks.free_block_ids) == free_before
        assert fake_of(engine).metrics.requests_aborted.total == 1

    @asyncio_test
    async def test_closing_the_generator_frees_the_blocks(self, make_async_engine):
        """This is what makes a client disconnect release KV, so it is the load-bearing test."""
        engine = make_async_engine(gated=True)
        blocks = fake_of(engine).scheduler.block_manager
        free_before = len(blocks.free_block_ids)

        outputs = await engine.add_request(prompt(8), FOREVER)
        runner_of(engine).release(2)  # the launch that emits nothing, then the one that drains it
        await outputs.__anext__()
        assert len(blocks.free_block_ids) < free_before

        await outputs.aclose()
        runner_of(engine).release()  # the abort lands once the step in flight returns
        await settle(engine)
        assert len(blocks.free_block_ids) == free_before
        assert not fake_of(engine).scheduler.seqs

    @asyncio_test
    async def test_cancelling_admission_leaves_nothing_running(self, make_async_engine):
        """Cancelled while waiting out the step in flight, so it never reaches the engine."""
        engine = make_async_engine(gated=True)
        blocks = fake_of(engine).scheduler.block_manager
        free_before = len(blocks.free_block_ids)
        await engine.add_request(prompt(8), FOREVER, "running")
        await wait_until(engine.engine_core._lock.locked)  # its step is blocked in the runner

        adding = asyncio.create_task(engine.add_request(prompt(8, 100), FOREVER, "cancelled"))
        await asyncio.sleep(0)  # now waiting on that step
        adding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await adding

        engine.abort("running")  # never iterated, so no generator finally will
        runner_of(engine).release()
        await settle(engine)
        assert not fake_of(engine).scheduler.seqs
        assert not engine.output_processor.request_states
        assert len(blocks.free_block_ids) == free_before

    @asyncio_test
    async def test_abort_of_an_unknown_request_is_harmless(self, make_async_engine):
        engine = make_async_engine()
        engine.abort("req-nobody")
        await settle(engine)
        assert not engine.is_dead

    @asyncio_test
    async def test_a_stop_string_finish_is_counted_as_a_stop(self, make_async_engine):
        """The server ends it through abort, but the client did not cancel it."""
        engine = make_async_engine(gated=True)
        outputs = await engine.add_request(prompt(8), FOREVER, "stopped")
        runner_of(engine).release(2)  # the launch that emits nothing, then the one that drains it
        await outputs.__anext__()

        engine.abort("stopped", "stop")
        await outputs.aclose()  # its own abort lands after, on a finished request
        runner_of(engine).release()
        await settle(engine)
        assert fake_of(engine).metrics.requests_aborted.total == 0
        assert fake_of(engine).metrics.requests_finished.values == {"stop": 1}
        assert fake_of(engine).metrics.ttft.count == 1


class TestAdmission:
    @asyncio_test
    async def test_closing_a_finished_stream_preserves_a_reused_request_id(self, make_async_engine):
        engine = make_async_engine(gated=True)
        first = await engine.add_request([10], SamplingParams(max_tokens=1), "reused")
        runner_of(engine).release()
        assert (await asyncio.wait_for(anext(first), 1)).finished
        second = await engine.add_request([20], SamplingParams(max_tokens=2), "reused")
        await first.aclose()
        runner_of(engine).release(2)

        async def read_second():
            return [output async for output in second]

        outputs = await asyncio.wait_for(read_second(), 1)
        assert sum(len(output.token_ids) for output in outputs) == 2
        assert outputs[-1].finish_reason == "length"

    @asyncio_test
    async def test_a_duplicate_id_is_refused_without_disturbing_the_original(self, make_async_engine):
        """Refused before the stream is replaced, or the live request is stranded holding blocks."""
        engine = make_async_engine(gated=True)
        outputs = await engine.add_request(prompt(8), FOREVER, "twice")

        with pytest.raises(DuplicateRequestId):
            # Refused here: a gated step holds the engine, so waiting on admission would hang.
            await asyncio.wait_for(engine.add_request(prompt(8, 100), FOREVER, "twice"), timeout=1)
        assert set(engine.output_processor.request_states) == {"twice"}

        runner_of(engine).release(2)  # the launch that emits nothing, then the one that drains it
        output = await asyncio.wait_for(outputs.__anext__(), timeout=1)  # a stranded stream hangs here
        assert output.token_ids

        await outputs.aclose()
        runner_of(engine).release()
        await settle(engine)

    @pytest.mark.parametrize("chunked", [False, True])
    @asyncio_test
    async def test_oversized_prompt_reports_capacity_and_engine_keeps_serving(self, make_async_engine, chunked):
        engine = make_async_engine(num_kvcache_blocks=1, enable_chunked_prefill=chunked)
        outputs = await engine.add_request(prompt(9), FOREVER)

        output = await asyncio.wait_for(anext(outputs), timeout=1)

        assert output.finished and output.finish_reason == "capacity"
        assert output.token_ids == []
        await outputs.aclose()
        assert len(await asyncio.wait_for(collect(engine, 2, prompt(1)), timeout=1)) == 2
        assert not engine.is_dead

    @asyncio_test
    async def test_a_prompt_that_cannot_fit_finishes_rather_than_hanging(self, make_async_engine):
        engine = make_async_engine(max_num_batched_tokens=16, enable_chunked_prefill=False)
        outputs = await engine.add_request(prompt(40), FOREVER)
        collected = [output async for output in outputs]
        assert [(output.finished, output.finish_reason) for output in collected] == [(True, "capacity")]


class TestEngineDeath:
    @asyncio_test
    async def test_a_live_stream_gets_the_error(self, make_async_engine):
        engine = make_async_engine(gated=True)
        outputs = await engine.add_request(prompt(8), FOREVER)
        runner_of(engine).release(2)  # the launch that emits nothing, then the one that drains it
        await outputs.__anext__()

        kill(engine)
        runner_of(engine).release()
        with pytest.raises(EngineDeadError):
            async for _ in outputs:
                pass
        assert engine.is_dead

    @asyncio_test
    async def test_new_requests_are_refused(self, make_async_engine):
        engine = make_async_engine(gated=True)
        await engine.add_request(prompt(8), FOREVER)
        kill(engine)
        runner_of(engine).release()
        await wait_until(lambda: engine.is_dead)
        with pytest.raises(EngineDeadError):
            await engine.add_request(prompt(8, 100), FOREVER)


class TestShutdown:
    @asyncio_test
    async def test_stop_ends_an_idle_loop(self):
        """Waiting for work rather than polling, so only the stop can end it."""
        fake = FakeLLMEngine(FakeConfig(), FakeModelRunner())
        checks = 0
        is_finished = fake.is_finished

        def counting_is_finished():
            nonlocal checks
            checks += 1
            return is_finished()

        fake.is_finished = counting_is_finished
        engine = AsyncLLM(AsyncInprocClient(fake))
        engine.start()
        await asyncio.sleep(0.05)
        assert checks <= 2
        engine.stop(timeout=1)
        assert not worker_alive(engine)
        await asyncio.wait_for(engine.engine_core._task, timeout=1)

    @asyncio_test
    async def test_stop_ends_a_loop_polling_on_outstanding_work(self):
        fake = FakeLLMEngine(FakeConfig(), FakeModelRunner())
        fake.is_finished = lambda: False
        fake.step = lambda: ([], 0, 0)  # work outstanding, but nothing can be stepped
        engine = AsyncLLM(AsyncInprocClient(fake))
        engine.start()
        await asyncio.sleep(0.05)
        engine.stop(timeout=1)
        assert not worker_alive(engine)
        await asyncio.wait_for(engine.engine_core._task, timeout=1)

    @asyncio_test
    async def test_a_live_stream_gets_the_shutdown_error(self, make_async_engine):
        engine = make_async_engine(gated=True)
        outputs = await engine.add_request(prompt(8), FOREVER)
        stopping = asyncio.create_task(asyncio.to_thread(engine.stop, 5))
        await asyncio.sleep(0.05)  # the stop waits on the step blocked in the runner
        runner_of(engine).release()
        await stopping
        with pytest.raises(EngineDeadError, match="shutting down"):
            async for _ in outputs:
                pass
        assert not worker_alive(engine)


def kill(engine: AsyncLLM):
    """Blow up the next step, the way an exception in the runner would."""

    def explode():
        raise RuntimeError("boom")

    fake_of(engine).step = explode


def worker_alive(engine: AsyncLLM) -> bool:
    return any(thread.is_alive() for thread in engine.engine_core._executor._threads)


async def settle(engine: AsyncLLM):
    await wait_until(fake_of(engine).is_finished)


async def wait_until(predicate, steps: int = 200):
    for _ in range(steps):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never held")
