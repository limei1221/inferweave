from dataclasses import dataclass, field


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0  # 0 is greedy
    top_p: float = 1.0  # 1 keeps the whole vocab
    top_k: int = 0  # 0 or -1 keeps the whole vocab
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_token_ids: list[int] = field(default_factory=list)
    skip_special_tokens: bool = True
    priority: int = 0  # lower is scheduled sooner, under the priority policy
    kv_transfer_params: dict | None = None  # disaggregated prefill, as vLLM's: do_remote_decode, or what to pull

    def __post_init__(self):
        assert self.temperature >= 0
        assert self.max_tokens >= 1
        assert 0 < self.top_p <= 1
        assert self.top_k >= -1
        if self.temperature == 0:  # as vLLM's: greedy takes the argmax, whatever the truncation
            self.top_p, self.top_k = 1.0, 0
