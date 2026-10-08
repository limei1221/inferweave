"""Speculative decoding with a checkpoint's own multi-token prediction layers, as vLLM's MTP method."""

from lean_vllm.spec_decode.config import SpeculativeConfig, split_sampled
from lean_vllm.spec_decode.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler

__all__ = ["PLACEHOLDER_TOKEN_ID", "RejectionSampler", "SpeculativeConfig", "split_sampled"]
