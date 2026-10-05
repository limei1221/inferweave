"""AsyncLLMEngine's surface over an engine core in another process, as vLLM's AsyncMPClient.

The event loop only sends requests, reads outputs and detokenizes; the step loop runs in the core, on its own GIL.
"""

import asyncio
import atexit
import logging
import multiprocessing as mp
import pickle
import shutil
import tempfile
from itertools import count
from multiprocessing.connection import wait
from typing import AsyncIterator, Callable
from uuid import uuid4

import zmq
import zmq.asyncio

from lean_vllm.engine.async_engine import AsyncStream, EngineDeadError
from lean_vllm.engine.core import make_engine, run_engine_core
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.scheduler import DuplicateRequestId
from lean_vllm.sampling_params import SamplingParams
from lean_vllm.utils.detokenizer import FastIncrementalDetokenizer

logger = logging.getLogger(__name__)


class AsyncMPClient:

    def __init__(
        self,
        engine_factory: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        tokenizer=None,
        shutdown_timeout: float = 30.0,
    ):
        """Starts the core and blocks until its engine is built. Without a tokenizer, outputs keep the core's text."""
        self.tokenizer = tokenizer
        self.error: BaseException | None = None
        self.on_death: Callable[[], None] | None = None
        self.shutdown_timeout = shutdown_timeout
        self._streams: dict[str, AsyncStream] = {}
        self._detokenizers: dict[str, FastIncrementalDetokenizer] = {}
        self._adding: dict[str, asyncio.Future] = {}    # sent, not yet acknowledged
        self._calls: dict[int, asyncio.Future] = {}
        self._call_ids = count()
        self._stopped = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: asyncio.Task | None = None

        # IPC sockets in a private directory, bound here so the core can connect as soon as it starts.
        self._ipc_dir = tempfile.mkdtemp(prefix="lean-vllm-")
        input_address, output_address = f"ipc://{self._ipc_dir}/in", f"ipc://{self._ipc_dir}/out"
        self._context = zmq.Context()
        self._input = self._context.socket(zmq.PUSH)
        self._input.setsockopt(zmq.SNDHWM, 0)    # a send never blocks the event loop
        self._input.bind(input_address)
        self._async_context = zmq.asyncio.Context()
        self._output = self._async_context.socket(zmq.PULL)
        self._output.bind(output_address)

        ctx = mp.get_context("spawn")
        ready, child_ready = ctx.Pipe(duplex=False)
        # Not a daemon: the core spawns the TP workers, and a daemon may have no children.
        self._process = ctx.Process(
            target=run_engine_core, name="EngineCore",
            args=(engine_factory, args, kwargs or {}, input_address, output_address, child_ready),
        )
        self._process.start()
        child_ready.close()
        atexit.register(self._shutdown_process)    # also when the server never got as far as its lifespan
        wait([ready, self._process.sentinel])
        try:
            status, error = ready.recv()
        except EOFError:    # it died before it could say why
            status, error = "exited", None
        ready.close()
        if status != "ready":
            self._shutdown_process()
            raise RuntimeError(f"the engine core failed to start (exit code {self._process.exitcode})") from error

    @classmethod
    def from_engine_args(cls, model: str, **kwargs) -> "AsyncMPClient":
        from lean_vllm.engine.llm_engine import load_tokenizer
        return cls(make_engine, (model,), kwargs, tokenizer=load_tokenizer(model))

    @property
    def is_dead(self) -> bool:
        return self.error is not None

    def start(self):
        """Called from the event loop that will read the streams."""
        self._loop = asyncio.get_running_loop()
        self._reader = self._loop.create_task(self._read_outputs())
        self._loop.add_reader(self._process.sentinel, self._on_process_exit)

    def stop(self, timeout: float | None = None):
        """From the event loop's thread: asks the core to finish its step and exit, killing it after timeout."""
        if self._stopped:
            return
        self._stopped = True
        if self._loop is not None and not self._loop.is_closed():
            self._loop.remove_reader(self._process.sentinel)
        self._shutdown_process(timeout)
        if self._reader is not None:
            self._reader.cancel()
        self._fail_all(EngineDeadError("the server is shutting down"))

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncIterator[RequestOutput]:
        """Per-step outputs; closing the generator aborts. The core answers first, so a status code is choosable."""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        request_id = request_id or f"req-{uuid4().hex}"
        if request_id in self._streams:
            raise DuplicateRequestId(f"{request_id} is already in flight")
        if self._stopped:
            raise EngineDeadError("the server is shutting down")
        if self.is_dead:
            raise self._dead_error()
        # Ready before the add is sent: the reader may hand out the first token before this coroutine resumes.
        stream = self._streams[request_id] = AsyncStream()
        if self.tokenizer is not None:
            self._detokenizers[request_id] = FastIncrementalDetokenizer(
                self.tokenizer, prompt, sampling_params.skip_special_tokens)
        added = self._adding[request_id] = self._loop.create_future()
        self._send(("add", request_id, prompt, sampling_params))
        try:
            error = await added
        except BaseException:    # cancelled while waiting, or the core died
            self._adding.pop(request_id, None)
            self.abort(request_id)
            raise
        if error is not None:
            self._streams.pop(request_id, None)
            self._detokenizers.pop(request_id, None)
            raise error
        return self._generate(request_id, stream)

    async def _generate(self, request_id: str, stream: AsyncStream) -> AsyncIterator[RequestOutput]:
        try:
            async for output in stream:
                yield output
        finally:
            # A finished request's id may already belong to a newer stream.
            if self._streams.get(request_id) is stream:
                self.abort(request_id)    # this is what makes a disconnect free KV blocks

    def abort(self, request_id: str, reason: str = "abort"):
        """Non-blocking and never raising, so it is safe in a generator's finally."""
        self._streams.pop(request_id, None)
        self._detokenizers.pop(request_id, None)
        if self.is_dead or self._stopped:
            return    # the blocks went with the engine
        self._send(("abort", request_id, reason))

    async def render_metrics(self) -> str:
        return await self._call_metrics("render")

    async def metrics_summary(self) -> dict:
        return await self._call_metrics("summary")

    async def _call_metrics(self, method: str):
        if self.is_dead:
            raise self._dead_error()
        call_id = next(self._call_ids)
        result = self._calls[call_id] = self._loop.create_future()
        self._send(("metrics", call_id, method))
        try:
            return await result
        finally:
            self._calls.pop(call_id, None)

    async def _read_outputs(self):
        try:
            while True:
                # Wait on poll, then take what is there: a non-blocking recv is done when made, so a cancel never
                # lands mid-message. recv_pyobj would chain its unpickling through the loop, which it could.
                await self._output.poll()
                while True:
                    try:
                        message = pickle.loads(await self._output.recv(zmq.NOBLOCK))
                    except zmq.Again:
                        break
                    if not self._handle(message):
                        return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._stopped:    # else the socket closed under a shutdown
                self._die(error)

    def _handle(self, message: tuple) -> bool:
        """False once the core has said it is dead."""
        kind = message[0]
        if kind == "outputs":
            self._deliver(message[1])
        elif kind == "added":
            _, request_id, error = message
            added = self._adding.pop(request_id, None)
            if added is not None and not added.done():    # else its caller gave up, and sent an abort
                added.set_result(error)
        elif kind == "metrics":
            _, call_id, value = message
            result = self._calls.get(call_id)
            if result is not None and not result.done():
                result.set_result(value)
        elif kind == "dead":
            self._die(message[1])
            return False
        return True

    def _send(self, message: tuple):
        """Never blocks the loop. With no peer to queue for, the core is gone: fail everything waiting on it."""
        try:
            self._input.send_pyobj(message, zmq.NOBLOCK)
        except zmq.Again:
            self._die(RuntimeError("the engine core is not connected"))

    def _deliver(self, outputs: list[RequestOutput]):
        for output in outputs:
            stream = self._streams.get(output.request_id)
            if stream is None:
                continue    # aborted since the step
            detokenizer = self._detokenizers.get(output.request_id)
            if detokenizer is not None:
                output.text = "".join(detokenizer.decode(token_id) for token_id in output.token_ids)
            stream.put(output)
            if output.finished:
                del self._streams[output.request_id]
                self._detokenizers.pop(output.request_id, None)

    def _on_process_exit(self):
        self._loop.remove_reader(self._process.sentinel)
        if not self._stopped:
            self._die(RuntimeError(f"the engine core process exited with code {self._process.exitcode}"))

    def _fail_all(self, error: Exception):
        streams, self._streams = self._streams, {}
        for stream in streams.values():
            stream.put(error)
        self._detokenizers.clear()
        for futures in (self._adding, self._calls):
            for future in futures.values():
                if not future.done():
                    future.set_exception(error)
            futures.clear()

    def _die(self, error: BaseException):
        """A status code cannot be retracted, so live streams get the error instead."""
        if self.error is not None:
            return
        logger.error("the engine core died", exc_info=error)
        self.error = error
        self._fail_all(self._dead_error())
        if self.on_death is not None:
            self.on_death()

    def _shutdown_process(self, timeout: float | None = None):
        """Idempotent; runs from stop() or at exit."""
        atexit.unregister(self._shutdown_process)
        process = self._process
        if process.is_alive():
            try:
                self._input.send_pyobj(("shutdown",), zmq.NOBLOCK)
            except zmq.ZMQError:
                pass    # never connected, or gone
            process.join(self.shutdown_timeout if timeout is None else timeout)
        if process.is_alive():
            logger.warning("the engine core did not stop in time, so it is terminated")
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        self._context.destroy(linger=0)
        self._async_context.destroy(linger=0)
        shutil.rmtree(self._ipc_dir, ignore_errors=True)

    def _dead_error(self) -> EngineDeadError:
        dead = EngineDeadError(f"the engine core died: {self.error!r}")
        dead.__cause__ = self.error
        return dead
