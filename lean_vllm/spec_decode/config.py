import json
from dataclasses import dataclass

# vLLM's names for a checkpoint's own multi-token prediction layers.
MTP_METHODS = ("mtp", "deepseek_mtp")
PLACEHOLDER_TOKEN_ID = -1  # past a row's last token, and a draft the host does not know yet


@dataclass(slots=True)
class SpeculativeConfig:
    """vLLM's --speculative-config, as JSON. Only a checkpoint's own MTP layers draft."""

    method: str = "mtp"
    num_speculative_tokens: int = 1  # drafts per request per step; past the MTP layers' count they repeat, as vLLM's

    def __post_init__(self):
        if self.method not in MTP_METHODS:
            raise ValueError(f"speculative_config method {self.method!r} is not one of {list(MTP_METHODS)}")
        if self.num_speculative_tokens < 1:
            raise ValueError("speculative_config num_speculative_tokens must be at least 1")

    @classmethod
    def parse(cls, value: "str | dict | SpeculativeConfig") -> "SpeculativeConfig":
        if isinstance(value, SpeculativeConfig):
            return value
        fields = json.loads(value or "{}") if isinstance(value, str) else dict(value)
        unknown = set(fields) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown speculative_config keys {sorted(unknown)}")
        return cls(**fields)


def split_sampled(rows: list[list[int]], num_speculative_tokens: int) -> tuple[list[list[int]], list[list[int]]]:
    """A speculative step's rows, as each one's new tokens and its drafts for the next step.

    The runner sends each row as num_speculative_tokens + 1 columns of tokens, -1 past the last, then the drafts.
    """
    width = num_speculative_tokens + 1
    return [[t for t in row[:width] if t >= 0] for row in rows], [row[width:] for row in rows]
