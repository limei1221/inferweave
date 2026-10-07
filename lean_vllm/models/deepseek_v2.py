import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from lean_vllm.eplb import EplbConfig
from lean_vllm.layers.attention import MLAAttention
from lean_vllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from lean_vllm.layers.layernorm import RMSNorm
from lean_vllm.layers.linear import (
    ColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
    divide,
)
from lean_vllm.layers.moe import FusedMoE
from lean_vllm.layers.rotary_embedding import get_rope, rope_config, yarn_get_mscale
from lean_vllm.models.qwen3 import Qwen3MLP as DeepseekV2MLP  # the same gated silu MLP


class DeepseekV2Attention(nn.Module):
    """MLA: keys and values come from a low-rank latent, with rope on a separate shared key."""

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_local_heads = divide(self.num_heads, tp_size)
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        bias = getattr(config, "attention_bias", False)

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(hidden_size, self.num_heads * self.qk_head_dim, bias=False)
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias
            )
        else:
            # q_a_proj and kv_a_proj_with_mqa read the same input, so one GEMM does both
            self.fused_qkv_a_proj = MergedReplicatedLinear(
                hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=bias,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim, hidden_size, bias=bias)

        rope_theta, rope_scaling = rope_config(config)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
            rope_scaling=rope_scaling,
        )
        self.scaling = self.qk_head_dim**-0.5
        mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        if mscale_all_dim:  # YaRN also sharpens the softmax
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        self._latent_projections: tuple[torch.Tensor, torch.Tensor] | None = None  # set once the weights load
        self.mla_attn = MLAAttention(
            self.num_local_heads,
            self.qk_head_dim,
            self.v_head_dim,
            self.scaling,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.expand,
            self.latent_projections,
        )

    def expand(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values, [n, heads, dim] each, from cached latents [n, kv_lora_rank + qk_rope_head_dim]."""
        kv_c_normed, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_b_proj(kv_c_normed).view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = k_nope.new_empty((*k_nope.shape[:-1], self.qk_head_dim))
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe.unsqueeze(1)  # broadcast the shared rope key to every head
        return k, v

    def latent_projections(self) -> tuple[torch.Tensor, torch.Tensor]:
        """kv_b_proj per head: W_UK_T [heads, qk_nope_head_dim, kv_lora_rank] and W_UV [heads, kv_lora_rank, v_head_dim]."""
        if self._latent_projections is not None:
            return self._latent_projections
        kv_b_proj_weight = self.kv_b_proj.weight.T.view(
            self.kv_lora_rank,
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # (L, N, P) -> (N, P, L) and (L, N, V) -> (N, L, V)
        # L kv_lora_rank, N num_local_heads, P qk_nope_head_dim, V v_head_dim
        return W_UK.permute(1, 2, 0), W_UV.transpose(0, 1)

    def process_weights_after_loading(self):
        """Views, so no memory and no copy: bmm takes their strides as they are. Spares each step their setup."""
        self._latent_projections = None
        self._latent_projections = self.latent_projections()

    def project(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Up to attention: the query, and the latent this step caches."""
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)
        else:
            q_c, kv_lora = self.fused_qkv_a_proj(hidden_states).split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q = self.q_b_proj(self.q_a_layernorm(q_c))
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)  # add head dim of 1
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(positions, q[..., self.qk_nope_head_dim :], k_pe)
        # The latent is cached normalized and with rope applied, so a read needs only kv_b_proj.
        return q, torch.cat([kv_c_normed, k_pe.squeeze(1)], dim=-1)

    def combine(self, attn_out: torch.Tensor) -> torch.Tensor:
        """From attention's output back to the residual stream."""
        return self.o_proj(attn_out.view(-1, self.num_local_heads * self.v_head_dim))


class DeepseekV2MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        enable_expert_parallel: bool = False,
        eplb_config: EplbConfig | None = None,
    ) -> None:
        super().__init__()
        # transformers' V3 config has neither field: its router is always noaux_tc over sigmoid scores
        self.topk_method = getattr(config, "topk_method", "noaux_tc")
        self.scoring_func = getattr(config, "scoring_func", "sigmoid" if self.topk_method == "noaux_tc" else "softmax")
        assert self.topk_method in ("greedy", "group_limited_greedy", "noaux_tc"), (
            f"unsupported topk_method {self.topk_method!r}"
        )
        assert self.scoring_func in ("softmax", "sigmoid"), f"unsupported scoring_func {self.scoring_func!r}"
        self.top_k = config.num_experts_per_tok
        self.num_expert_group = config.n_group
        self.topk_group = config.topk_group
        self.renormalize = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = ReplicatedLinear(config.hidden_size, config.n_routed_experts, bias=False)
        self.gate.weight.data = self.gate.weight.data.float()  # held in fp32, so no step casts it
        if self.topk_method == "noaux_tc":
            # Added to the scores to pick experts, as V3's auxiliary-loss-free balancing; the weights ignore it.
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
        self.experts = FusedMoE(
            config.n_routed_experts,
            self.top_k,
            config.hidden_size,
            config.moe_intermediate_size,
            enable_expert_parallel,
            enable_eplb=eplb_config is not None,
            num_redundant_experts=eplb_config.num_redundant_experts if eplb_config is not None else 0,
        )
        self.shared_experts = None
        if config.n_shared_experts:
            # Holds the weights only: FusedMoE runs them, beside the routed experts, and reduces both at once.
            self.shared_experts = DeepseekV2MLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act,
            )

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [N, D]; the gate runs in fp32, as transformers'
        router_logits = F.linear(hidden_states.float(), self.gate.weight)
        grouped = self.topk_method != "greedy"
        return torch.ops.lean_vllm.select_experts(
            router_logits,
            self.top_k,
            self.renormalize,
            float(self.routed_scaling_factor),
            self.num_expert_group if grouped else 1,
            self.topk_group if grouped else 1,
            self.scoring_func,
            getattr(self.gate, "e_score_correction_bias", None),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared = self.shared_experts
        shared_weights = () if shared is None else (shared.gate_up_proj.weight, shared.down_proj.weight)
        return self.experts(hidden_states, *self.route(hidden_states), *shared_weights)


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        enable_expert_parallel: bool = False,
        eplb_config: EplbConfig | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = DeepseekV2Attention(config)
        moe_layer_freq = getattr(config, "moe_layer_freq", None) or 1
        is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        if is_moe_layer:
            self.mlp: DeepseekV2MoE | DeepseekV2MLP = DeepseekV2MoE(config, enable_expert_parallel, eplb_config)
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
        return self.post_attention(self.self_attn.mla_attn(q, latent), residual)


class DeepseekV2Model(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        enable_expert_parallel: bool = False,
        eplb_config: EplbConfig | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, i, enable_expert_parallel, eplb_config)
                for i in range(config.num_hidden_layers)
            ]
        )
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
    supports_expert_parallel = True
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: PretrainedConfig,
        enable_expert_parallel: bool = False,
        eplb_config: EplbConfig | None = None,
    ) -> None:
        super().__init__()
        if config.q_lora_rank is not None:
            self.packed_modules_mapping = {
                **self.packed_modules_mapping,
                "q_a_proj": ("fused_qkv_a_proj", 0),
                "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
            }
        # The checkpoint's multi-token prediction layers follow the last one; nothing here runs them.
        self.skipped_weight_prefixes = tuple(
            f"model.layers.{config.num_hidden_layers + i}."
            for i in range(getattr(config, "num_nextn_predict_layers", None) or 0)
        )
        self.model = DeepseekV2Model(config, enable_expert_parallel, eplb_config)
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


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    """V2's layers; V3 differs in its router (noaux_tc over sigmoid scores), which the config selects."""
