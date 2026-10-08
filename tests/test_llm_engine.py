"""Prompt validation at admission: the guard both front doors share, without a model."""

import pytest

from lean_vllm.engine.llm_engine import validate_request
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
