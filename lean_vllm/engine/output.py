from dataclasses import dataclass


@dataclass(slots=True)
class RequestMetrics:
    """Timestamps are perf_counter seconds. Anything not reached yet is None."""
    arrival_time: float
    num_prompt_tokens: int
    num_completion_tokens: int = 0
    num_preemptions: int = 0
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def queue_time(self) -> float | None:
        return None if self.first_scheduled_time is None else self.first_scheduled_time - self.arrival_time

    @property
    def ttft(self) -> float | None:
        return None if self.first_token_time is None else self.first_token_time - self.arrival_time

    @property
    def e2e(self) -> float | None:
        return None if self.finish_time is None else self.finish_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        """Mean seconds per output token after the first."""
        if self.finish_time is None or self.first_token_time is None or self.num_completion_tokens < 2:
            return None
        return (self.finish_time - self.first_token_time) / (self.num_completion_tokens - 1)


@dataclass(slots=True)
class RequestOutput:
    """What one request produced in one step."""
    request_id: str
    token_ids: list[int]
    text: str = ""
    finished: bool = False
    finish_reason: str | None = None
    metrics: RequestMetrics | None = None
    kv_transfer_params: dict | None = None    # on a prefill instance's final output: where the decode pulls from
