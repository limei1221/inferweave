"""Per-request state on the client side of the engine core, as vLLM's OutputProcessor.

LLMEngine runs one for offline generate, and AsyncLLM one for the server; the core itself only sends token ids.
"""

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from transformers import PreTrainedTokenizerFast

from lean_vllm.engine.output import RequestOutput
from lean_vllm.utils.detokenizer import FastIncrementalDetokenizer


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


@dataclass(slots=True)
class RequestState:
    detokenizer: FastIncrementalDetokenizer | None  # None without a tokenizer: outputs keep the core's text
    stream: AsyncStream | None  # None offline, where outputs are returned instead


class OutputProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerFast | None):
        self.tokenizer = tokenizer
        self.request_states: dict[str, RequestState] = {}

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        skip_special_tokens: bool,
        stream: AsyncStream | None = None,
    ):
        detokenizer = None
        if self.tokenizer is not None:
            detokenizer = FastIncrementalDetokenizer(self.tokenizer, prompt_token_ids, skip_special_tokens)
        self.request_states[request_id] = RequestState(detokenizer, stream)

    def abort_request(self, request_id: str) -> RequestState | None:
        return self.request_states.pop(request_id, None)

    def process_outputs(self, outputs: list[RequestOutput]) -> list[RequestOutput]:
        """Fill in each output's text and hand it to its stream. Returns those of requests without one."""
        unstreamed = []
        for output in outputs:
            state = self.request_states.get(output.request_id)
            if state is None:
                continue  # aborted since the step
            if state.detokenizer is not None:
                output.text = "".join(state.detokenizer.decode(token_id) for token_id in output.token_ids)
            if output.finished:
                del self.request_states[output.request_id]
            if state.stream is not None:
                state.stream.put(output)
            else:
                unstreamed.append(output)
        return unstreamed

    def fail_all(self, error: Exception):
        """Every live stream raises error, and every request is forgotten."""
        states, self.request_states = self.request_states, {}
        for state in states.values():
            if state.stream is not None:
                state.stream.put(error)
