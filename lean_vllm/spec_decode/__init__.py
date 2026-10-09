"""Speculative decoding with a checkpoint's own multi-token prediction layers."""

from lean_vllm.spec_decode.config import PLACEHOLDER_TOKEN_ID, SpeculativeConfig, split_sampled
from lean_vllm.spec_decode.rejection_sampler import RejectionSampler

__all__ = ["PLACEHOLDER_TOKEN_ID", "RejectionSampler", "SpeculativeConfig", "split_sampled"]
