"""OpenAI-compatible HTTP server over `AsyncMPClient`, or `AsyncLLMEngine` in-process; both have one surface."""

import asyncio
import json
from contextlib import aclosing, asynccontextmanager
from time import time
from typing import AsyncIterator, Awaitable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel

from lean_vllm.engine.async_engine import AsyncLLMEngine, EngineDeadError
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.scheduler import InvalidRequest, QueueFull
from lean_vllm.entrypoints import protocol
from lean_vllm.entrypoints.protocol import (
    BaseRequest,
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
)
from lean_vllm.entrypoints.stop_checker import StopChecker
from lean_vllm.sampling_params import SamplingParams

DONE = "data: [DONE]\n\n"

# The engine's own reasons; the drops below are not completions and never appear here.
FINISH_REASONS = {"stop": "stop", "length": "length", "abort": "stop"}
DROP_STATUS = {"capacity": 503, "timeout": 504}


class _RequestStreamingResponse(StreamingResponse):
    """Own admission cleanup even if sending headers or the first chunk fails."""

    def __init__(self, stream, engine: AsyncLLMEngine, request_id: str):
        super().__init__(stream, media_type="text/event-stream")
        self.engine = engine
        self.request_id = request_id

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Closing an unstarted generator does not run its finally block.
            self.engine.abort(self.request_id)
            await self.body_iterator.aclose()


def build_app(engine: AsyncLLMEngine, model: str) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="lean-vLLM", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content=_error("invalid_request_error", _first_message(exc)))

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        kind = "server_error" if exc.status_code >= 500 else "invalid_request_error"
        return JSONResponse(status_code=exc.status_code, content=_error(kind, exc.detail))

    @app.get("/health")
    async def health():
        if engine.is_dead:
            raise HTTPException(503, f"the engine died: {engine.error!r}")
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        try:
            text = await engine.render_metrics()    # the counters live with the engine, maybe in another process
        except EngineDeadError as dead:
            raise HTTPException(503, str(dead))
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    @app.get("/metrics.json")
    async def metrics_json():
        try:
            return await engine.metrics_summary()
        except EngineDeadError as dead:
            raise HTTPException(503, str(dead))

    @app.get("/v1/models")
    async def models():
        return ModelList(data=[ModelCard(id=model)])

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        prompt_token_ids = body.prompt if isinstance(body.prompt, list) else engine.tokenizer.encode(body.prompt)
        return await _serve(engine, model, body, prompt_token_ids, chat=False, request=request)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        messages = [message.model_dump() for message in body.messages]
        # return_dict=False, or newer tokenizers hand back a BatchEncoding rather than ids.
        prompt_token_ids = engine.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        return await _serve(engine, model, body, prompt_token_ids, chat=True, request=request)

    return app


async def _serve(
    engine: AsyncLLMEngine,
    model: str,
    body: BaseRequest,
    prompt_token_ids: list[int],
    chat: bool,
    request: Request | None = None,
):
    if body.model != model:
        # As in OpenAI and vLLM, an unserved model name is a 404, not a field to ignore.
        raise HTTPException(404, f"the model {body.model!r} does not exist")
    if engine.is_dead:
        raise HTTPException(503, f"the engine died: {engine.error!r}")
    kv_transfer_params = body.kv_transfer_params
    if body.stream and kv_transfer_params and kv_transfer_params.get("do_remote_decode"):
        # A stream has nowhere to return the params, so the blocks would sit held until they expire.
        raise HTTPException(400, "do_remote_decode needs a non-streaming request")
    request_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid4().hex}"
    sampling_params = SamplingParams(
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        ignore_eos=body.ignore_eos,
        priority=body.priority,
        kv_transfer_params=kv_transfer_params,
    )
    try:
        outputs = await engine.add_request(prompt_token_ids, sampling_params, request_id)
    except InvalidRequest as invalid:
        raise HTTPException(400, str(invalid))
    except QueueFull as full:
        raise HTTPException(429, f"the engine is at capacity: {full}")
    except EngineDeadError as dead:
        raise HTTPException(503, str(dead))

    deltas = _deltas(outputs, StopChecker(body.stop_strings), engine, request_id)
    if body.stream:
        stream = _stream(deltas, request_id, model, body, len(prompt_token_ids), chat)
        return _RequestStreamingResponse(stream, engine, request_id)
    try:
        reply = await _unless_disconnected(request, _collect(deltas, request_id, model, len(prompt_token_ids), chat))
    except EngineDeadError as dead:
        raise HTTPException(503, str(dead))
    if reply is None:
        engine.abort(request_id)    # the generators may never have started, so no finally ran
        return Response(status_code=499)    # nobody is left to read it
    return reply


async def _unless_disconnected(request: Request | None, work: Awaitable):
    """Starlette only watches for a disconnect while streaming. None if the client left first."""
    if request is None:
        return await work
    task = asyncio.ensure_future(work)
    listener = asyncio.ensure_future(_disconnected(request))
    try:
        await asyncio.wait((task, listener), return_when=asyncio.FIRST_COMPLETED)
    finally:
        listener.cancel()
        task.cancel()    # a no-op once done
        await asyncio.wait((task,))
    return None if task.cancelled() else task.result()


async def _disconnected(request: Request):
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _deltas(
    outputs: AsyncIterator[RequestOutput], checker: StopChecker, engine: AsyncLLMEngine, request_id: str,
):
    """Yields (text, finish_reason, num_completion_tokens, kv_transfer_params); the last has a reason."""
    num_tokens = 0
    try:
        async for output in outputs:
            num_tokens += len(output.token_ids)
            text = checker.push(output.text)
            if checker.matched:
                engine.abort(request_id, "stop")    # frees the blocks, counted as a finish rather than a cancel
                yield text, "stop", num_tokens, None
                return
            if output.finished:
                if output.finish_reason not in FINISH_REASONS:
                    status = DROP_STATUS.get(output.finish_reason, 503)
                    raise HTTPException(status, f"the engine dropped the request: {output.finish_reason}")
                reason = FINISH_REASONS[output.finish_reason]
                yield text + checker.flush(), reason, num_tokens, output.kv_transfer_params
                return
            if text:
                yield text, None, num_tokens, None
    finally:
        await outputs.aclose()


async def _collect(deltas, request_id: str, model: str, num_prompt_tokens: int, chat: bool):
    text, finish_reason, num_tokens, kv_transfer_params = "", "stop", 0, None
    async for delta, reason, num_tokens, kv_transfer_params in deltas:
        text += delta
        finish_reason = reason or finish_reason
    usage = protocol.usage(num_prompt_tokens, num_tokens)
    if chat:
        message = protocol.ChatMessage(role="assistant", content=text)
        choice = protocol.ChatCompletionResponseChoice(index=0, message=message, finish_reason=finish_reason)
        return protocol.ChatCompletionResponse(
            id=request_id, model=model, choices=[choice], usage=usage, kv_transfer_params=kv_transfer_params,
        )
    choice = protocol.CompletionResponseChoice(index=0, text=text, finish_reason=finish_reason, logprobs=None)
    return protocol.CompletionResponse(
        id=request_id, model=model, choices=[choice], usage=usage, kv_transfer_params=kv_transfer_params,
    )


async def _stream(deltas, request_id: str, model: str, body: BaseRequest, num_prompt_tokens: int, chat: bool):
    created = int(time())
    if chat:
        response, kind = protocol.ChatCompletionStreamResponse, "chat.completion.chunk"
    else:
        response, kind = protocol.CompletionStreamResponse, "text_completion"

    def chunk(choices: list, **extra) -> str:
        return _event(response(id=request_id, object=kind, created=created, model=model, choices=choices, **extra))

    def choice(text: str, reason: str | None, role: str | None = None):
        if not chat:
            return protocol.CompletionResponseChoice(index=0, text=text, finish_reason=reason, logprobs=None)
        delta = protocol.DeltaMessage(role=role, content=text) if role else protocol.DeltaMessage(content=text)
        return protocol.ChatCompletionResponseStreamChoice(index=0, delta=delta, finish_reason=reason)

    async with aclosing(deltas):
        num_tokens = 0
        if chat:
            yield chunk([choice("", None, role="assistant")])
        try:
            async for delta, reason, num_tokens, _ in deltas:
                yield chunk([choice(delta, reason)])
        except (EngineDeadError, HTTPException) as error:
            # The 200 is already sent, so the error rides in the stream.
            yield _event(_error("server_error", str(getattr(error, "detail", error))))
            yield DONE
            return
        if body.include_usage:
            yield chunk([], usage=protocol.usage(num_prompt_tokens, num_tokens))
        yield DONE


def _event(payload: BaseModel | dict) -> str:
    data = payload.model_dump_json(exclude_unset=True) if isinstance(payload, BaseModel) else json.dumps(payload)
    return f"data: {data}\n\n"


def _error(kind: str, message: str) -> dict:
    return {"error": {"message": message, "type": kind}}


def _first_message(exc: RequestValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
    message = error.get("msg", "invalid request")
    message = message.removeprefix("Value error, ")
    return f"{location}: {message}" if location else message
