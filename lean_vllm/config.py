import logging
import os
from dataclasses import dataclass
from transformers import AutoConfig

from lean_vllm.attention import get_attention_backend
from lean_vllm.engine.sequence import HASH_ALGOS

logger = logging.getLogger(__name__)


# Which steps may replay a graph: full for pure decode, piecewise for any step within its token buckets.
FULL_MODES = ("full", "full_and_piecewise")
PIECEWISE_MODES = ("piecewise", "full_and_piecewise")
CUDAGRAPH_MODES = ("none",) + FULL_MODES + ("piecewise",)


@dataclass(slots=True)
class Config:
    model: str
    # vLLM's server defaults on an H100 (tiered up from 2048/256 past an A100).
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 1024
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    kvcache_memory_gb: float = 2.0    # cpu/mps only; cuda uses gpu_memory_utilization
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    cudagraph_mode: str = "full_and_piecewise"    # none | full | piecewise | full_and_piecewise
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    enable_chunked_prefill: bool = True    # off never mixes prefill and decode, kept for the A/B
    enable_prefix_caching: bool = True     # off recomputes every prompt, kept for the A/B
    async_scheduling: bool = True    # schedule the next step before awaiting the last, as vLLM does
    prefix_caching_hash_algo: str = "sha256"    # or "xxhash", which is faster and not cryptographic
    scheduling_policy: str = "fcfs"    # or "priority"
    max_waiting_requests: int = 0      # 0 is unlimited
    request_timeout: float = 0.0       # seconds a request may wait unscheduled; 0 is none
    long_prefill_token_threshold: int = 0    # per-step token cap for one prompt; 0 is none

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert self.cudagraph_mode in CUDAGRAPH_MODES, f"unknown cudagraph_mode {self.cudagraph_mode!r}"
        assert self.prefix_caching_hash_algo in HASH_ALGOS, \
            f"unknown prefix_caching_hash_algo {self.prefix_caching_hash_algo!r}, expected one of {sorted(HASH_ALGOS)}"
        assert 1 <= self.tensor_parallel_size <= 8
        if self.async_scheduling and self.tensor_parallel_size > 1:
            # Ranks above zero never see the sampled tokens, so they could not follow.
            logger.warning("async_scheduling is off: tensor_parallel_size > 1 does not support it")
            self.async_scheduling = False
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if getattr(self.hf_config, "kv_lora_rank", None) is not None:    # an MLA model
            block_size = get_attention_backend(mla=True).mla_block_size()
            if block_size and block_size != self.kvcache_block_size:
                logger.warning("kvcache_block_size is %d: the MLA decode kernel reads no other page size", block_size)
                self.kvcache_block_size = block_size
