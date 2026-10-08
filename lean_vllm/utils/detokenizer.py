import logging

from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


class FastIncrementalDetokenizer:
    """Decodes one token at a time with DecodeStream, which holds back a split character rather than emit U+FFFD."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerFast, prompt_token_ids: list[int], skip_special_tokens: bool = True
    ):
        self.tokenizer = tokenizer._tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.stream = DecodeStream(ids=prompt_token_ids, skip_special_tokens=skip_special_tokens)

    def decode(self, token_id: int) -> str:
        """Append one token and return the text it completes, possibly empty."""
        try:
            return self.stream.step(self.tokenizer, token_id) or ""
        except (OverflowError, TypeError):
            logger.exception("invalid token id %r", token_id)
            return ""
        except Exception as e:
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                raise
            # non-monotonic UTF-8 output breaks the stream; start a fresh one
            logger.warning("invalid prefix while detokenizing, resetting the decode stream")
            self.stream = DecodeStream(skip_special_tokens=self.skip_special_tokens)
            return self.stream.step(self.tokenizer, token_id) or ""
