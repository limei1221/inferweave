"""Prompt validation at admission: the guard both front doors share, without a model. And generate() and shutdown."""

from types import SimpleNamespace

import pytest

from lean_vllm.engine.llm_engine import LLMEngine, validate_request
from lean_vllm.engine.scheduler import InvalidRequest
from lean_vllm.sampling_params import SamplingParams

VOCAB, CONTEXT = 100, 16


def check(prompt: list[int], max_tokens: int = 1):
    validate_request(prompt, SamplingParams(max_tokens=max_tokens), VOCAB, CONTEXT)


class TestPromptValidation:
    @pytest.mark.parametrize("token_id", [VOCAB, VOCAB + 1, -1])
    def test_a_token_outside_the_vocabulary_is_refused(self, token_id):
        """Otherwise the embedding lookup raises, and that kills the engine thread."""
        with pytest.raises(InvalidRequest, match="outside the 100-token vocabulary"):
            check([1, 2, token_id])

    def test_an_empty_prompt_is_refused(self):
        with pytest.raises(InvalidRequest, match="empty"):
            check([])

    def test_a_prompt_that_fills_the_context_is_refused(self):
        with pytest.raises(InvalidRequest, match="no room in the 16-token context"):
            check(list(range(CONTEXT)))

    def test_max_tokens_that_overruns_the_context_is_refused(self):
        """The last position would index past the rotary cache."""
        with pytest.raises(InvalidRequest, match="plus max_tokens"):
            check(list(range(15)), max_tokens=4)

    def test_a_prompt_that_fits_with_its_output_is_accepted(self):
        check(list(range(12)), max_tokens=4)

    def test_the_edges_of_the_vocabulary_are_accepted(self):
        check([0, VOCAB - 1])


class TestExit:
    def test_a_second_exit_is_a_no_op(self):
        """atexit calls exit() again after a caller's own."""
        calls: list[str] = []
        engine = LLMEngine.__new__(LLMEngine)
        engine.profiler, engine.ps = None, []
        engine.model_runner = SimpleNamespace(call=calls.append)
        engine.exit()
        engine.exit()
        assert calls == ["exit"]


def admitting_engine(refuse: int | None = None) -> tuple[LLMEngine, list[str], list[str]]:
    """add_request refuses the prompt at index refuse; records what was added and aborted."""
    added: list[str] = []
    aborted: list[str] = []
    engine = LLMEngine.__new__(LLMEngine)

    def add_request(prompt, sampling_params):
        if len(added) == refuse:
            raise InvalidRequest("refused")
        added.append(f"req-{len(added)}")
        return added[-1]

    engine.add_request = add_request
    engine.abort_request = aborted.append
    return engine, added, aborted


class TestGenerate:
    def test_a_refused_prompt_aborts_those_admitted_before_it(self):
        """Left queued, their outputs would reach the next call, which has no slot for them."""
        engine, added, aborted = admitting_engine(refuse=1)
        with pytest.raises(InvalidRequest):
            engine.generate([[1], [2], [3]], SamplingParams(), use_tqdm=False)
        assert aborted == added == ["req-0"]

    def test_mismatched_sampling_params_are_refused_before_any_admission(self):
        engine, added, _ = admitting_engine()
        with pytest.raises(ValueError, match="2 prompts but 1 sampling params"):
            engine.generate([[1], [2]], [SamplingParams()], use_tqdm=False)
        assert not added
