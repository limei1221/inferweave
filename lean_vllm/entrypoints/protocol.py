"""OpenAI-compatible request and response bodies. Unimplemented fields are a 400, never ignored."""

from time import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Refused by name with a reason. Each allows its no-op values, which clients send by default.
UNSUPPORTED = {
    "top_p": ((1.0,), "lean-vLLM samples with temperature only"),
    "top_k": ((0, -1), "lean-vLLM samples with temperature only"),
    "min_p": ((0.0,), "lean-vLLM samples with temperature only"),
    "best_of": ((1,), "n > 1 is out of scope"),
    "logprobs": ((False, 0), "the sampler does not return logprobs"),
    "top_logprobs": ((0,), "the sampler does not return logprobs"),
    "presence_penalty": ((0.0,), "penalties are not implemented"),
    "frequency_penalty": ((0.0,), "penalties are not implemented"),
    "repetition_penalty": ((1.0,), "penalties are not implemented"),
    "seed": ((), "the sampler does not take a per-request seed"),
    "logit_bias": ((), "logit processors are not implemented"),
    "tools": ((), "tool calling is not implemented"),
    "tool_choice": (("none",), "tool calling is not implemented"),
    "functions": ((), "tool calling is not implemented"),
    "echo": ((False,), "the prompt is never echoed back"),
    "suffix": ((), "infilling is not implemented"),
}


def _asks_for_it(value, neutral: tuple) -> bool:
    return value is not None and value != [] and value != {} and value not in neutral


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class BaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = 64
    temperature: float = 1.0    # 0 is greedy
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    n: int = 1
    # Extras: the OpenAI schema has no field for these, so they ride in the body.
    ignore_eos: bool = False
    priority: int = 0
    kv_transfer_params: dict[str, Any] | None = None    # disaggregated prefill, as vLLM's; the proxy sets it

    @model_validator(mode="before")
    @classmethod
    def _refuse_unsupported(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for name, (neutral, reason) in UNSUPPORTED.items():
            if name not in data:
                continue
            if _asks_for_it(data[name], neutral):
                raise ValueError(f"{name} is not supported: {reason}")
            del data[name]    # a no-op the client filled in; extra="forbid" must not see it
        return data

    @model_validator(mode="after")
    def _check(self) -> "BaseRequest":
        if self.n != 1:
            raise ValueError("n must be 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        return self

    @property
    def stop_strings(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage


class CompletionRequest(BaseRequest):
    prompt: str | list[int]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseRequest):
    messages: list[ChatMessage] = Field(min_length=1)


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "lean-vllm"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


def usage(num_prompt_tokens: int, num_completion_tokens: int) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_completion_tokens,
        total_tokens=num_prompt_tokens + num_completion_tokens,
    )


# Chunks serialize with exclude_unset, so wire fields they carry have no defaults: one left out fails loudly.

class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None
    logprobs: None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo
    kv_transfer_params: dict[str, Any] | None = None    # from a prefill instance: where to pull its KV from


class CompletionStreamResponse(BaseModel):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    kv_transfer_params: dict[str, Any] | None = None


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = None
