"""The synchronous engine driven from the event loop, with each step on a worker thread.

Only the blocking `step()` leaves the loop; a lock keeps everything else off the engine meanwhile. The server runs
core_client.AsyncMPClient by default, which moves the engine to its own process; this is `--no-engine-process`.
"""

import asyncio
import logging
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import AsyncIterator, Callable
from uuid import uuid4

from lean_vllm.engine.llm_engine import LLMEngine
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.scheduler import DuplicateRequestId
from lean_vllm.sampling_params import SamplingParams

# Backoff after a step that ran nothing although work is outstanding, so the loop does not spin.
IDLE_POLL = 0.005


class EngineDeadError(RuntimeError):
    """The engine raised. Nothing can be served until the process restarts."""


class AsyncStream:
    """One request's outputs: fed on the event loop, read by its handler."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()

    def put(self, item: RequestOutput | Exception):
        self._queue.put_nowait(item)

    async def __aiter__(self) -> AsyncIterator[RequestOutput]:
        while True:
            item = await self._queue.get()
            if isinstance(item, Exception):
                raise item
            yield item
            if item.finished:
                return


class AsyncLLMEngine:

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.metrics = engine.metrics
        self.error: BaseException | None = None
        self.on_death: Callable[[], None] | None = None
        # One worker, so every step runs on the same thread: CUDA's current device and the profiler are per thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lean-vllm-engine")
        # The rest is event loop only.
        self._streams: dict[str, AsyncStream] = {}
        self._lock = asyncio.Lock()    # held while a step runs
        self._aborts: list[tuple[str, str]] = []    # arrived during a step, applied after it
        self._has_work = asyncio.Event()
        self._stopped = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    @classmethod
    def from_engine_args(cls, model: str, **kwargs) -> "AsyncLLMEngine":
        return cls(LLMEngine(model, **kwargs))

    @property
    def is_dead(self) -> bool:
        return self.error is not None

    def start(self):
        """Called from the event loop that will read the streams."""
        self._loop = asyncio.get_running_loop()
        self._task = self._loop.create_task(self._run())

    def stop(self, timeout: float = 30.0):
        """Callable from any thread. Waits up to timeout for the step in flight."""
        if self._stopped:
            return
        self._stopped = True
        self._call_soon(self._has_work.set)
        closed = self._close_profiler()    # queued behind the step in flight
        done, _ = wait([closed], timeout)
        self._executor.shutdown(wait=bool(done))
        self._call_soon(self._fail_streams, EngineDeadError("the server is shutting down"))

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncIterator[RequestOutput]:
        """Per-step outputs; closing the generator aborts. Admission settles first, so a status code is still choosable."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        request_id = request_id or f"req-{uuid4().hex}"
        if request_id in self._streams:
            # The scheduler would refuse it too, but only after this stream replaced the live one.
            raise DuplicateRequestId(f"{request_id} is already in flight")
        async with self._lock:    # a cancel while waiting here leaves nothing behind
            if self._stopped:
                raise EngineDeadError("the server is shutting down")
            if self.is_dead:
                raise self._dead_error()
            self.engine.add_request(prompt, sampling_params, request_id)
            stream = self._streams[request_id] = AsyncStream()
            self._has_work.set()
        return self._generate(request_id, stream)

    async def render_metrics(self) -> str:
        return self.metrics.render()

    async def metrics_summary(self) -> dict:
        return self.metrics.summary()

    async def _generate(self, request_id: str, stream: AsyncStream) -> AsyncIterator[RequestOutput]:
        try:
            async for output in stream:
                yield output
        finally:
            self.abort(request_id)    # this is what makes a disconnect free KV blocks

    def abort(self, request_id: str, reason: str = "abort"):
        """Non-blocking and never raising, so it is safe in a generator's finally."""
        self._streams.pop(request_id, None)
        if self.is_dead or self._stopped:
            return    # the blocks went with the engine
        if self._lock.locked():
            self._aborts.append((request_id, reason))
            return
        try:
            self.engine.abort_request(request_id, reason)
        except Exception as error:
            self._die(error)

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
                self._deliver(outputs)
                if not (outputs or num_prefill_tokens or num_decode_tokens):
                    await asyncio.sleep(IDLE_POLL)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._stopped:    # else a step refused by the shut-down executor
                self._die(error)

    def _deliver(self, outputs: list[RequestOutput]):
        for output in outputs:
            stream = self._streams.get(output.request_id)
            if stream is None:
                continue    # aborted since the step
            stream.put(output)
            if output.finished:
                del self._streams[output.request_id]

    def _fail_streams(self, error: Exception):
        streams, self._streams = self._streams, {}
        for stream in streams.values():
            stream.put(error)

    def _die(self, error: BaseException):
        """A status code cannot be retracted, so live streams get the error instead."""
        logging.getLogger(__name__).error("the engine thread died", exc_info=error)
        self.error = error
        self._fail_streams(self._dead_error())
        self._close_profiler()
        if self.on_death is not None:
            self.on_death()

    def _close_profiler(self) -> Future:
        """On the worker, since the profiler must be flushed by the thread that stepped it."""
        profiler = self.engine.profiler
        return self._executor.submit(profiler.close if profiler is not None else lambda: None)

    def _call_soon(self, callback: Callable, *args):
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except (AttributeError, RuntimeError):
            pass    # never started, or the loop is closed and its handlers with it

    def _dead_error(self) -> EngineDeadError:
        dead = EngineDeadError(f"the engine thread died: {self.error!r}")
        dead.__cause__ = self.error
        return dead
