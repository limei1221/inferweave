"""The server's engine, as vLLM's AsyncLLM: the front end of an engine core in another process.

It owns every request's state (its stream and detokenizer, in an OutputProcessor), so the client below it only
moves requests in and outputs out.
"""

import asyncio
from typing import AsyncGenerator, Callable
from uuid import uuid4

from lean_vllm.engine.core_client import AsyncMPClient
from lean_vllm.engine.exceptions import EngineDeadError
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.output_processor import AsyncStream, OutputProcessor
from lean_vllm.engine.scheduler import DuplicateRequestId
from lean_vllm.sampling_params import SamplingParams


class AsyncLLM:
    def __init__(self, engine_core: AsyncMPClient, tokenizer=None):
        """Without a tokenizer, prompts must be token ids and outputs keep the core's text."""
        self.engine_core = engine_core
        self.tokenizer = tokenizer
        self.output_processor = OutputProcessor(tokenizer)
        self.error: BaseException | None = None
        self.on_death: Callable[[], None] | None = None
        self._death = ""  # the message every later EngineDeadError carries
        self._stopped = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._output_handler: asyncio.Task | None = None

    @classmethod
    def from_engine_args(cls, model: str, **kwargs) -> "AsyncLLM":
        from lean_vllm.engine.core import make_engine
        from lean_vllm.engine.llm_engine import load_tokenizer

        return cls(AsyncMPClient(make_engine, (model,), kwargs), load_tokenizer(model))

    @property
    def is_dead(self) -> bool:
        return self.error is not None

    def start(self):
        """Called from the event loop that will read the streams."""
        self._loop = asyncio.get_running_loop()
        self.engine_core.start()
        self._output_handler = self._loop.create_task(self._handle_outputs())

    def stop(self, timeout: float | None = None):
        """Callable from any thread. Waits up to timeout for the step in flight."""
        if self._stopped:
            return
        self._stopped = True
        self.engine_core.shutdown(timeout)
        self._on_loop(self._shut_down_streams)

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Per-step outputs; closing the generator aborts. Admission settles first, so a status code is still choosable."""
        token_ids: list[int] = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        request_id = request_id or f"req-{uuid4().hex}"
        if request_id in self.output_processor.request_states:
            # The scheduler would refuse it too, but only after this stream replaced the live one.
            raise DuplicateRequestId(f"{request_id} is already in flight")
        if self._stopped:
            raise EngineDeadError("the server is shutting down")
        if self.is_dead:
            raise self._dead_error()
        # Ready before the add is sent: the first token may arrive before this coroutine resumes.
        stream = AsyncStream()
        self.output_processor.add_request(request_id, token_ids, sampling_params.skip_special_tokens, stream)
        try:
            await self.engine_core.add_request_async(request_id, token_ids, sampling_params)
        except asyncio.CancelledError:
            self.abort(request_id)  # the core may have taken it before the cancel
            raise
        except BaseException:  # refused, or the engine is gone
            self.output_processor.abort_request(request_id)
            raise
        return self._generate(request_id, stream)

    async def _generate(self, request_id: str, stream: AsyncStream) -> AsyncGenerator[RequestOutput, None]:
        try:
            async for output in stream:
                yield output
        finally:
            # A finished request's id may already belong to a newer stream.
            state = self.output_processor.request_states.get(request_id)
            if state is not None and state.stream is stream:
                self.abort(request_id)  # this is what makes a disconnect free KV blocks

    def abort(self, request_id: str, reason: str = "abort"):
        """Non-blocking and never raising, so it is safe in a generator's finally.

        reason is "stop" when the server ends it on a stop string, which counts as a finish.
        """
        self.output_processor.abort_request(request_id)
        if self.is_dead or self._stopped:
            return  # the blocks went with the engine
        self.engine_core.abort_request(request_id, reason)

    async def render_metrics(self) -> str:
        """The counters live with the engine, maybe in another process."""
        if self.is_dead:
            raise self._dead_error()
        return await self.engine_core.render_metrics_async()

    async def _handle_outputs(self):
        try:
            while True:
                self.output_processor.process_outputs(await self.engine_core.get_output_async())
        except EngineDeadError as dead:
            self._die(dead)

    def _die(self, dead: EngineDeadError):
        """A status code cannot be retracted, so live streams get the error instead."""
        if self.is_dead or self._stopped:
            return
        self.error = dead.__cause__ if dead.__cause__ is not None else dead
        self._death = str(dead)
        self.output_processor.fail_all(self._dead_error())
        if self.on_death is not None:
            self.on_death()

    def _shut_down_streams(self):
        if self._output_handler is not None:
            self._output_handler.cancel()
        self.output_processor.fail_all(EngineDeadError("the server is shutting down"))

    def _on_loop(self, callback: Callable, *args):
        """Run callback on the event loop: now if this is its thread, else soon."""
        if self._loop is None or self._loop.is_closed():
            return  # never started, or the loop is closed and its streams with it
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            callback(*args)
        else:
            self._loop.call_soon_threadsafe(callback, *args)

    def _dead_error(self) -> EngineDeadError:
        dead = EngineDeadError(self._death)
        dead.__cause__ = self.error
        return dead
