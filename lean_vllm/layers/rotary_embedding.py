import math
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
) -> torch.Tensor:
    if not is_neox_style:
        # GPT-J style: rotate adjacent pairs rather than the two halves.
        x1, x2 = x.float()[..., ::2], x.float()[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2).to(x.dtype)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def rope_config(config) -> tuple[float, dict]:
    """rope_parameters from transformers 5, or rope_theta and rope_scaling before it."""
    params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    return params.get("rope_theta", getattr(config, "rope_theta", 10000)), params


def yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def yarn_inv_freq(rotary_dim: int, base: float, scaling: dict) -> torch.Tensor:
    """YaRN: long wavelengths interpolated by the factor, short ones kept, a linear ramp between."""
    factor = scaling["factor"]
    original_max_position = scaling["original_max_position_embeddings"]

    # index of the RoPE pair that completes num_rotations turns over the original context
    def correction_dim(num_rotations: float) -> float:
        return rotary_dim * math.log(original_max_position / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction_dim(scaling.get("beta_fast") or 32)), 0)
    high = min(math.ceil(correction_dim(scaling.get("beta_slow") or 1)), rotary_dim - 1)
    pos_freqs = base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    ramp = ((torch.arange(rotary_dim // 2, dtype=torch.float) - low) / (high - low if high != low else 0.001)).clamp(0, 1)
    return 1.0 / (factor * pos_freqs) * ramp + 1.0 / pos_freqs * (1 - ramp)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool = True,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        assert rotary_dim == head_size
        scaling_type = rope_scaling and (rope_scaling.get("rope_type") or rope_scaling.get("type")) or "default"
        if scaling_type == "yarn":
            inv_freq = yarn_inv_freq(rotary_dim, base, rope_scaling)
            factor = rope_scaling["factor"]
            mscale = rope_scaling.get("attention_factor") or (
                yarn_get_mscale(factor, rope_scaling.get("mscale", 1)) / yarn_get_mscale(factor, rope_scaling.get("mscale_all_dim", 0))
            )
        else:
            assert scaling_type == "default", f"unsupported rope scaling {scaling_type!r}"
            inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
            mscale = 1.0
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin, self.is_neox_style)
        key = apply_rotary_emb(key, cos, sin, self.is_neox_style)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool = True,
    rope_scaling: dict | None = None,
):
    # A dict cannot key the cache, so its items do.
    scaling = tuple(sorted(rope_scaling.items())) if rope_scaling else None
    return _get_rope(head_size, rotary_dim, max_position, base, is_neox_style, scaling)


@lru_cache(1)
def _get_rope(head_size, rotary_dim, max_position, base, is_neox_style, scaling):
    return RotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style, dict(scaling) if scaling else None)
