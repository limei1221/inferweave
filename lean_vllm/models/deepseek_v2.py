import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PretrainedConfig

from lean_vllm.layers.attention import MLAAttention
from lean_vllm.layers.layernorm import RMSNorm
from lean_vllm.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear, divide
from lean_vllm.layers.moe import FusedMoE
from lean_vllm.layers.rotary_embedding import get_rope, rope_config, yarn_get_mscale
from lean_vllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from lean_vllm.models.qwen3 import Qwen3MLP as DeepseekV2MLP    # the same gated silu MLP


class DeepseekV2Attention(nn.Module):
    """MLA: keys and values come from a low-rank latent, with rope on a separate shared key."""

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        bias = getattr(config, "attention_bias", False)

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(hidden_size, self.total_num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = ReplicatedLinear(hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(self.q_lora_rank, self.total_num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = ReplicatedLinear(hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = RowParallelLinear(self.total_num_heads * self.v_head_dim, hidden_size, bias=bias)

        rope_theta, rope_scaling = rope_config(config)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
            rope_scaling=rope_scaling,
        )
        scaling = self.qk_head_dim ** -0.5
        if rope_scaling.get("mscale_all_dim"):    # YaRN also sharpens the softmax
            mscale = yarn_get_mscale(rope_scaling["factor"], rope_scaling["mscale_all_dim"])
            scaling *= mscale * mscale
        self.attn = MLAAttention(
            self.num_heads,
            self.qk_head_dim,
            self.v_head_dim,
            scaling,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.expand,
            self.latent_projections,
        )

    def expand(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values, [n, heads, dim] each, from cached latents [n, kv_lora_rank + qk_rope_head_dim]."""
        kv_c_normed, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_b_proj(kv_c_normed).view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = k_nope.new_empty((*k_nope.shape[:-1], self.qk_head_dim))
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe.unsqueeze(1)    # broadcast the shared rope key to every head
        return k, v

    def latent_projections(self) -> tuple[torch.Tensor, torch.Tensor]:
        """kv_b_proj per head: W_UK_T [heads, qk_nope_head_dim, kv_lora_rank] and W_UV [heads, kv_lora_rank, v_head_dim]."""
        kv_b_proj_weight = self.kv_b_proj.weight.T.view(
            self.kv_lora_rank, self.num_heads, self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # (L, N, P) -> (N, P, L) and (L, N, V) -> (N, L, V)
        return W_UK.permute(1, 2, 0), W_UV.transpose(0, 1)

    def project(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Up to attention: the query, and the latent this step caches."""
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(-1, self.num_heads, self.qk_head_dim)
        kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)    # add head dim of 1
        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(positions, q[..., self.qk_nope_head_dim:], k_pe)
        # The latent is cached normalized and with rope applied, so a read needs only kv_b_proj.
        return q, torch.cat([kv_c_normed, k_pe.squeeze(1)], dim=-1)

    def combine(self, o: torch.Tensor) -> torch.Tensor:
        """From attention's output back to the residual stream."""
        return self.o_proj(o.view(-1, self.num_heads * self.v_head_dim))


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        assert getattr(config, "scoring_func", "softmax") == "softmax"
        assert config.topk_method in ("greedy", "group_limited_greedy"), f"unsupported topk_method {config.topk_method!r}"
        self.top_k = config.num_experts_per_tok
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = ReplicatedLinear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = FusedMoE(config.n_routed_experts, self.top_k, config.hidden_size, config.moe_intermediate_size)
        self.shared_experts = None
        if config.n_shared_experts:
            self.shared_experts = DeepseekV2MLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act,
            )

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [N, config.hidden_size]
        scores = F.linear(x.float(), self.gate.weight.float()).softmax(dim=-1)
        # scores: [N, config.n_routed_experts]
        if self.topk_method == "group_limited_greedy":
            # Only experts in the topk_group best groups stay eligible.
            num_token = scores.size(0)
            group_scores = scores.view(num_token, self.num_group, -1).max(dim=-1).values    # [N, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]    # [N, topk_group]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (group_mask.unsqueeze(-1)
                          .expand(num_token, self.num_group, scores.size(-1) // self.num_group)
                          .reshape(num_token, -1))    # [N, n_routed_experts]
            scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        topk_weights, topk_ids = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.experts(x, *self.route(x))
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = DeepseekV2Attention(config)
        is_moe = (config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace
                  and layer_idx % (getattr(config, "moe_layer_freq", None) or 1) == 0)
        if is_moe:
            self.mlp = DeepseekV2MoE(config)
        else:
            self.mlp = DeepseekV2MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def pre_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The layer up to attention. Touches no KV cache, so it is capturable."""
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return self.self_attn.project(positions, hidden_states) + (residual,)

    def post_attention(
        self,
        attn_out: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer after attention. Capturable for the same reason."""
        hidden_states = self.self_attn.combine(attn_out)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, latent, residual = self.pre_attention(positions, hidden_states, residual)
        return self.post_attention(self.self_attn.attn(q, latent), residual)


class DeepseekV2Model(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):
    supports_cuda_graph = True
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.model = DeepseekV2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
