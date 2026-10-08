"""The engine in a process of its own, as vLLM's EngineCore, so the HTTP server's event loop never holds its GIL.

Requests come in and outputs go out over two ZMQ sockets, as pickled tuples:
in:  ("add", request_id, prompt_token_ids, sampling_params), ("abort", request_id, reason),
     ("metrics", call_id), ("shutdown",)
out: ("added", request_id, error or None), ("outputs", [RequestOutput]), ("metrics", call_id, text), ("dead", error)
"""

import logging
import pickle
import signal
from multiprocessing import parent_process
from multiprocessing.connection import Connection
from typing import Callable

import zmq

from lean_vllm.engine.scheduler import InvalidRequest

logger = logging.getLogger(__name__)

IDLE_POLL_MS = 5  # backoff after a step that ran nothing although work is outstanding
PARENT_POLL_MS = 1000  # how often an idle core checks that the client is still there


def make_engine(model: str, **kwargs):
    """The engine a core serves. Detokenization stays with the client, as in vLLM, so the step loop skips it."""
    from lean_vllm.engine.llm_engine import LLMEngine

    return LLMEngine(model, detokenize=False, **kwargs)


def picklable(error: BaseException) -> BaseException:
    """The error itself if the other side can rebuild it, else its repr in a RuntimeError."""
    try:
        pickle.loads(pickle.dumps(error))
        return error
    except Exception:
        return RuntimeError(repr(error))


class EngineCore:
    """vLLM's busy loop: take every queued request, step, send the outputs; block only when there is nothing to run."""

    def __init__(self, engine, input_socket: zmq.Socket, output_socket: zmq.Socket):
        self.engine = engine
        self.input = input_socket
        self.output = output_socket
        self.running = True

    def run(self):
        while self.running:
            self._handle_inputs(wait_ms=None if self.engine.is_finished() else 0)
            if not self.running or self.engine.is_finished():
                continue
            outputs, num_prefill_tokens, num_decode_tokens = self.engine.step()
            if outputs:
                self.output.send_pyobj(("outputs", outputs))
            if not (outputs or num_prefill_tokens or num_decode_tokens):
                self._handle_inputs(wait_ms=IDLE_POLL_MS)

    def _handle_inputs(self, wait_ms: int | None):
        """Everything queued. Waits up to wait_ms for the first, or, with None, for as long as the client lives."""
        if wait_ms is None:
            while not self.input.poll(PARENT_POLL_MS):
                parent = parent_process()
                if parent is not None and not parent.is_alive():
                    logger.warning("the client process is gone, so the engine core exits")
                    self.running = False
                    return
        elif wait_ms and not self.input.poll(wait_ms):
            return
        while self.running:
            try:
                message = self.input.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                return
            self._handle(message)

    def _handle(self, message: tuple):
        kind = message[0]
        if kind == "add":
            _, request_id, prompt, sampling_params = message
            error: BaseException | None = None
            try:
                self.engine.add_request(prompt, sampling_params, request_id)
            except InvalidRequest as refused:  # the client turns it into a 400
                error = refused
            except Exception as unexpected:  # the request's alone, so the engine serves on
                logger.exception("add_request failed for %s", request_id)
                error = picklable(unexpected)
            self.output.send_pyobj(("added", request_id, error))
        elif kind == "abort":
            _, request_id, reason = message
            self.engine.abort_request(request_id, reason)
        elif kind == "metrics":
            _, call_id = message
            self.output.send_pyobj(("metrics", call_id, self.engine.metrics.render()))
        elif kind == "shutdown":
            self.running = False
        else:
            raise ValueError(f"unknown message {kind!r}")


def _exit(signum, frame):
    raise SystemExit(0)


def run_engine_core(
    engine_factory: Callable,
    args: tuple,
    kwargs: dict,
    input_address: str,
    output_address: str,
    ready: Connection,
):
    """A spawned process's target: build the engine, report on ready, then serve until shutdown."""
    # Ctrl-C reaches the whole process group; the client decides when this process stops. SIGTERM exits cleanly,
    # so atexit still stops the TP workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _exit)
    try:
        engine = engine_factory(*args, **kwargs)
    except BaseException as error:
        ready.send(("failed", picklable(error)))
        raise
    context = zmq.Context()
    input_socket = context.socket(zmq.PULL)
    input_socket.connect(input_address)
    output_socket = context.socket(zmq.PUSH)
    output_socket.setsockopt(zmq.SNDHWM, 0)  # never block a step on a slow reader; outputs queue instead
    output_socket.connect(output_address)
    ready.send(("ready", None))
    ready.close()
    try:
        EngineCore(engine, input_socket, output_socket).run()
    except SystemExit:
        pass
    except BaseException as error:
        logger.exception("the engine core died")
        output_socket.send_pyobj(("dead", picklable(error)))
        raise
    finally:
        context.destroy(linger=1000)  # long enough to deliver a last message, never a hang at exit
