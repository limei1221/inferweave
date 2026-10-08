import logging
import os
import socket
from dataclasses import dataclass
from typing import Any

from transformers import AutoConfig

from lean_vllm.attention import LayerSpec, get_attention_backend
from lean_vllm.engine.sequence import HASH_ALGOS
from lean_vllm.eplb import EplbConfig
from lean_vllm.kv_transfer import KVTransferConfig, parse_kv_transfer_config
from lean_vllm.models import get_drafter_class, get_model_class
from lean_vllm.spec_decode import SpeculativeConfig

logger = logging.getLogger(__name__)


# Which steps may replay a graph: full for pure decode, piecewise for any step within its token buckets.
FULL_MODES = ("full", "full_and_piecewise")
PIECEWISE_MODES = ("piecewise", "full_and_piecewise")
CUDAGRAPH_MODES = ("none",) + FULL_MODES + ("piecewise",)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@dataclass(slots=True)
class Config:
    model: str
    # vLLM's server defaults on an H100 (tiered up from 2048/256 past an A100).
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 1024
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    kvcache_memory_gb: float = 2.0  # cpu only; cuda uses gpu_memory_utilization
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False  # MoE layers hold whole experts per rank, not slices of each; as vLLM
    enable_eplb: bool = False  # move and copy experts across ranks by their load, as vLLM's EPLB; needs the above
    eplb_config: str = ""  # JSON, as vLLM's --eplb-config: window_size, step_interval, num_redundant_experts, ...
    eplb: EplbConfig | None = None  # eplb_config, parsed when enable_eplb
    enforce_eager: bool = False
    cudagraph_mode: str = "full_and_piecewise"  # none | full | piecewise | full_and_piecewise
    hf_config: Any = None  # the checkpoint's transformers config, loaded in __post_init__
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    enable_chunked_prefill: bool = True  # off never mixes prefill and decode, kept for the A/B
    enable_prefix_caching: bool = True  # off recomputes every prompt, kept for the A/B
    async_scheduling: bool = True  # schedule the next step before awaiting the last, as vLLM does
    prefix_caching_hash_algo: str = "sha256"  # or "xxhash", which is faster and not cryptographic
    scheduling_policy: str = "fcfs"  # or "priority"
    max_waiting_requests: int = 0  # 0 is unlimited
    request_timeout: float = 0.0  # seconds a request may wait unscheduled; 0 is none
    long_prefill_token_threshold: int = 0  # per-step token cap for one prompt; 0 is none
    dist_port: int = 0  # rendezvous port for the ranks; 0 picks a free one
    kv_transfer_config: str = ""  # JSON, as vLLM's --kv-transfer-config; empty disaggregates nothing
    kv_transfer: KVTransferConfig | None = None  # kv_transfer_config, parsed
    speculative_config: str = ""  # JSON, as vLLM's --speculative-config: method ("mtp"), num_speculative_tokens
    speculative: SpeculativeConfig | None = None  # speculative_config, parsed

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert self.cudagraph_mode in CUDAGRAPH_MODES, f"unknown cudagraph_mode {self.cudagraph_mode!r}"
        assert self.prefix_caching_hash_algo in HASH_ALGOS, (
            f"unknown prefix_caching_hash_algo {self.prefix_caching_hash_algo!r}, expected one of {sorted(HASH_ALGOS)}"
        )
        assert 1 <= self.tensor_parallel_size <= 8
        if self.async_scheduling and self.tensor_parallel_size > 1:
            # Ranks above zero never see the sampled tokens, so they could not follow.
            logger.warning("async_scheduling is off: tensor_parallel_size > 1 does not support it")
            self.async_scheduling = False
        if self.kv_transfer_config and self.kv_transfer is None:
            self.kv_transfer = parse_kv_transfer_config(self.kv_transfer_config)
        if not self.dist_port:  # resolved here, so spawned workers get the same one
            self.dist_port = _free_port()
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.enable_expert_parallel and not get_model_class(self.hf_config).supports_expert_parallel:
            raise ValueError(f"enable_expert_parallel needs a MoE model, and {self.hf_config.architectures} has none")
        if self.enable_eplb:
            if not self.enable_expert_parallel:
                raise ValueError("enable_eplb needs enable_expert_parallel")
            self.eplb = EplbConfig.parse(self.eplb_config)
            num_physical = self.hf_config.n_routed_experts + self.eplb.num_redundant_experts
            if num_physical % self.tensor_parallel_size:
                raise ValueError(
                    f"EPLB needs the same slots on every rank: {num_physical} experts, with the redundant ones, "
                    f"do not split over {self.tensor_parallel_size} ranks"
                )
        elif self.eplb_config:
            raise ValueError("eplb_config needs enable_eplb")
        if self.speculative_config and self.speculative is None:
            self.speculative = SpeculativeConfig.parse(self.speculative_config)
        if self.speculative is not None:
            get_drafter_class(self.hf_config)  # raises for a checkpoint with no MTP layers
            if self.kv_transfer is not None or self.enable_eplb:
                raise ValueError("speculative decoding runs without kv_transfer_config and enable_eplb, for now")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if getattr(self.hf_config, "kv_lora_rank", None) is not None:  # an MLA model
            block_size = get_attention_backend(self._mla_layer_spec()).mla_block_size()
            if block_size and block_size != self.kvcache_block_size:
                logger.warning("kvcache_block_size is %d: the MLA decode kernel reads no other page size", block_size)
                self.kvcache_block_size = block_size

    def _mla_layer_spec(self) -> LayerSpec:
        """What each MLA layer of one rank will ask of a backend, so the page size follows the one it gets."""
        hf_config = self.hf_config
        num_heads = hf_config.num_attention_heads // self.tensor_parallel_size
        return LayerSpec(
            hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim,
            num_heads,
            num_heads,
            hf_config.dtype,
            latent_dim=hf_config.kv_lora_rank + hf_config.qk_rope_head_dim,
        )
