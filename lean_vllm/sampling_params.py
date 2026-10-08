from dataclasses import dataclass, field


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0  # 0 is greedy
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_token_ids: list[int] = field(default_factory=list)
    skip_special_tokens: bool = True
    priority: int = 0  # lower is scheduled sooner, under the priority policy
    kv_transfer_params: dict | None = None  # disaggregated prefill, as vLLM's: do_remote_decode, or what to pull

    def __post_init__(self):
        assert self.temperature >= 0
        assert self.max_tokens >= 1
