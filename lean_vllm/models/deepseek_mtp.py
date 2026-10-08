import re

import torch
from torch import nn
from transformers import PretrainedConfig

from lean_vllm.layers.layernorm import RMSNorm
from lean_vllm.layers.linear import ReplicatedLinear
from lean_vllm.models.deepseek_v2 import DeepseekV2DecoderLayer, packed_modules_mapping

# Only the first MTP layer's copies would load in vLLM, and it shares the target's anyway.
SHARED_WEIGHTS = ("embed_tokens.", "shared_head.head.")
OWN_WEIGHTS = ("enorm.", "hnorm.", "eh_proj.", "shared_head.")


class SharedHead(nn.Module):
    """The checkpoint's shared_head: its norm. Its head is the target's lm_head, as vLLM shares it."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    """One MTP module: the next token's embedding and the hidden state before it, through one decoder layer."""

    def __init__(self, config: PretrainedConfig, layer_idx: int, enable_expert_parallel: bool = False) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(config)
        self.mtp_block = DeepseekV2DecoderLayer(config, layer_idx, enable_expert_parallel)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Nothing comes before position 0, so its embedding is masked, as vLLM's.
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        hidden_states = self.eh_proj(torch.cat([self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1))
        hidden_states, residual = self.mtp_block(positions, hidden_states, None)
        # Normed once, both for the logits and for the next draft step, as vLLM's and SGLang's.
        hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        return hidden_states


class DeepSeekMTP(nn.Module):
    """DeepSeek-V3's multi-token prediction layers, which follow its last decoder layer in the checkpoint.

    The drafter of speculative decoding. It embeds and projects to logits with the target's own layers, so it
    holds neither, and takes the target's final hidden states (after its norm) as its input.
    """

    def __init__(self, config: PretrainedConfig, enable_expert_parallel: bool = False) -> None:
        super().__init__()
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleList(
            [
                DeepSeekMultiTokenPredictorLayer(config, self.mtp_start_layer_idx + i, enable_expert_parallel)
                for i in range(self.num_mtp_layers)
            ]
        )
        self.packed_modules_mapping = packed_modules_mapping(config)
        layer_ids = "|".join(str(self.mtp_start_layer_idx + i) for i in range(self.num_mtp_layers))
        self._weight_name = re.compile(rf"model\.layers\.({layer_ids})\.(.+)")

    def remap_weight_name(self, name: str) -> str | None:
        """model.layers.{num_hidden_layers + i}.X as this module's layers.{i}.X, or layers.{i}.mtp_block.X for the
        decoder layer's own weights. None for everything else, the target's and the copies it shares."""
        match = self._weight_name.fullmatch(name)
        if match is None or match[2].startswith(SHARED_WEIGHTS):
            return None
        layer, rest = int(match[1]) - self.mtp_start_layer_idx, match[2]
        return f"layers.{layer}.{rest}" if rest.startswith(OWN_WEIGHTS) else f"layers.{layer}.mtp_block.{rest}"

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.layers[spec_step_idx % self.num_mtp_layers](inputs_embeds, positions, hidden_states)
