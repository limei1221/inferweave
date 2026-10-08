"""Incremental detokenization must never split a multi-byte character."""

import os

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lean_vllm.engine import llm_engine
from lean_vllm.utils.detokenizer import FastIncrementalDetokenizer

MODEL = os.path.expanduser(os.getenv("LEAN_VLLM_TEST_MODEL", "~/workspace/huggingface/Qwen3-0.6B"))
SPECIAL = "<|s|>"
TEXTS = ["Hello, world!", "café 🙂 naïve", "日本語のテキスト", "def f(x):\n    return x"]
needs_model = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f"no tokenizer at {MODEL}")


@pytest.fixture(scope="module")
def byte_tokenizer() -> PreTrainedTokenizerFast:
    """One token per byte and no merges, so every non-ASCII character spans several tokens."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={byte: i for i, byte in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([SPECIAL])
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer)


def deltas(tokenizer, text: str, prompt: str = "", skip_special_tokens: bool = True) -> list[str]:
    detokenizer = FastIncrementalDetokenizer(tokenizer, tokenizer.encode(prompt), skip_special_tokens)
    return [detokenizer.decode(i) for i in tokenizer.encode(text)]


class TestFastIncrementalDetokenizer:
    def test_ascii_arrives_one_character_at_a_time(self, byte_tokenizer):
        assert deltas(byte_tokenizer, "hey") == ["h", "e", "y"]

    def test_two_byte_character_is_held_back_until_complete(self, byte_tokenizer):
        assert deltas(byte_tokenizer, "é") == ["", "é"]

    def test_four_byte_character_is_held_back_until_complete(self, byte_tokenizer):
        assert deltas(byte_tokenizer, "🙂") == ["", "", "", "🙂"]

    def test_deltas_concatenate_to_the_whole_string(self, byte_tokenizer):
        text = "héllo 🙂 wörld"
        assert "".join(deltas(byte_tokenizer, text)) == text

    def test_replacement_character_is_never_emitted(self, byte_tokenizer):
        assert "�" not in "".join(deltas(byte_tokenizer, "naïve 🚀"))

    def test_special_tokens_are_skipped(self, byte_tokenizer):
        assert deltas(byte_tokenizer, "hi" + SPECIAL) == ["h", "i", ""]

    def test_special_tokens_are_kept_when_asked(self, byte_tokenizer):
        assert deltas(byte_tokenizer, "hi" + SPECIAL, skip_special_tokens=False) == ["h", "i", SPECIAL]

    def test_prompt_tokens_are_context_not_output(self, byte_tokenizer):
        assert deltas(byte_tokenizer, "!", prompt="prompt") == ["!"]

    def test_invalid_token_id_emits_nothing(self, byte_tokenizer):
        assert FastIncrementalDetokenizer(byte_tokenizer, []).decode(-1) == ""

    def test_invalid_prefix_resets_the_stream(self, byte_tokenizer):

        class BrokenStream:
            def step(self, tokenizer, token_id):
                raise Exception("Invalid prefix encountered")

        detokenizer = FastIncrementalDetokenizer(byte_tokenizer, byte_tokenizer.encode("Hello"))
        detokenizer.stream = BrokenStream()
        assert "".join(detokenizer.decode(i) for i in byte_tokenizer.encode("hi there")) == "hi there"


class TestLoadTokenizer:
    def test_a_fast_tokenizer_is_accepted(self, byte_tokenizer, tmp_path):
        byte_tokenizer.save_pretrained(tmp_path)
        assert isinstance(llm_engine.load_tokenizer(str(tmp_path)), PreTrainedTokenizerFast)

    def test_a_tokenizer_that_is_not_fast_is_refused(self, monkeypatch):
        """Refused at startup, before workers spawn, rather than failing every request."""
        monkeypatch.setattr(llm_engine.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
        with pytest.raises(ValueError, match="no fast tokenizer"):
            llm_engine.load_tokenizer("some-model")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, use_fast=True)


@needs_model
class TestAgainstRealTokenizer:
    """The fake proves the holding back; this proves the contract against a real vocabulary."""

    @pytest.mark.parametrize("text", TEXTS)
    def test_matches_a_full_decode(self, tokenizer, text):
        token_ids = tokenizer.encode(text)
        detokenizer = FastIncrementalDetokenizer(tokenizer, [])
        assert "".join(detokenizer.decode(i) for i in token_ids) == tokenizer.decode(token_ids)

    @pytest.mark.parametrize("skip", [True, False])
    def test_special_tokens_follow_skip_special_tokens(self, tokenizer, skip):
        token_ids = tokenizer.encode("hi<|im_end|>")
        detokenizer = FastIncrementalDetokenizer(tokenizer, [], skip_special_tokens=skip)
        assert "".join(detokenizer.decode(i) for i in token_ids) == ("hi" if skip else "hi<|im_end|>")
