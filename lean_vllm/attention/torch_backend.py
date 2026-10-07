import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from lean_vllm.attention.abstract import AttentionBackend, write_into
from lean_vllm.utils.context import Context


class TorchAttention(AttentionBackend):
    """SDPA reference backend. Runs anywhere; optimized for clarity, not speed."""

    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)

    @staticmethod
    def get_name() -> str:
        return "torch"

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def supports_cuda_graph() -> bool:
        return False

    @staticmethod
    def supports_mla_decode() -> bool:
        return True

    @staticmethod
    def split_decodes() -> bool:
        return True  # so decode rows skip prefill's padding to the step's longest query

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        num_tokens = key.size(0)
        assert slot_mapping.numel() == num_tokens
        dim = self.num_kv_heads * self.head_dim

        slots = slot_mapping.long()
        keep = slots >= 0
        if not keep.all():  # syncs GPU with CPU; custom GPU kernels skip invalid slots on-device
            slots = slots[keep]
            key = key[keep]
            value = value[keep]

        k_cache.view(-1, dim)[slots] = key.flatten(1)
        v_cache.view(-1, dim)[slots] = value.flatten(1)

    def store_latents(self, latent, latent_cache, slot_mapping) -> None:
        dim = latent_cache.size(-1)
        assert slot_mapping.numel() == latent.size(0)

        slots = slot_mapping.long()
        keep = slots >= 0
        if not keep.all():  # syncs GPU with CPU; custom GPU kernels skip invalid slots on-device
            slots = slots[keep]
            latent = latent[keep]

        latent_cache.view(-1, dim)[slots] = latent

    def prefill(self, q, k, v, k_cache, v_cache, context: Context, out=None) -> torch.Tensor:
        cu_seqlens_q, cu_seqlens_k = context.cu_seqlens_q, context.cu_seqlens_k
        assert cu_seqlens_q is not None and cu_seqlens_k is not None
        max_seqlen_q, max_seqlen_k = context.max_seqlen_q, context.max_seqlen_k
        q_pad = self._pad_rows(q, cu_seqlens_q, max_seqlen_q)  # [B, Lq, H, D]
        if context.block_tables is not None:  # read every key back from the pages
            seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            k_pad = self._gather_pages(k_cache, context.block_tables, max_seqlen_k, seqlens_k)
            v_pad = self._gather_pages(v_cache, context.block_tables, max_seqlen_k, seqlens_k)
        else:
            k_pad = self._pad_rows(k, cu_seqlens_k, max_seqlen_k)
            v_pad = self._pad_rows(v, cu_seqlens_k, max_seqlen_k)
        mask = self._mask(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        o = self._sdpa(q_pad, k_pad, v_pad, mask)  # [B, Lq, H, D]
        return write_into(out, self._unpad_rows(o, cu_seqlens_q, q.size(0)))

    def varlen_with_lse(
        self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, host_cu_seqlens=None
    ):
        # Written out, since SDPA does not return the log-sum-exp.
        repeats = self.num_heads // self.num_kv_heads
        q_pad = self._pad_rows(q, cu_seqlens_q, max_seqlen_q).transpose(1, 2).float()  # [B, H, Lq, D]
        k_pad = self._pad_rows(k, cu_seqlens_k, max_seqlen_k).repeat_interleave(repeats, dim=2).transpose(1, 2).float()
        v_pad = self._pad_rows(v, cu_seqlens_k, max_seqlen_k).repeat_interleave(repeats, dim=2).transpose(1, 2).float()
        scores = q_pad @ k_pad.transpose(-1, -2) * self.scale  # [B, H, Lq, Lk]
        mask = self._mask(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal)
        scores = scores.masked_fill(~mask, float("-inf"))
        o = (scores.softmax(dim=-1) @ v_pad).transpose(1, 2).to(q.dtype)  # [B, Lq, H, D]
        lse = scores.logsumexp(dim=-1).transpose(1, 2)  # [B, Lq, H]
        return self._unpad_rows(o, cu_seqlens_q, q.size(0)), self._unpad_rows(lse, cu_seqlens_q, q.size(0))

    def decode(self, q, k_cache, v_cache, context: Context, out=None) -> torch.Tensor:
        block_tables, context_lens = context.block_tables, context.context_lens
        assert block_tables is not None and context_lens is not None
        # block_tables: [B, max_blocks_per_sequence]
        # k_cache: [num_blocks, block_size, ...]
        max_seqlen_k = block_tables.size(1) * k_cache.size(1)  # the table's width, so no length is read back
        k = self._gather_pages(k_cache, block_tables, max_seqlen_k, context_lens)  # [B, Lk, Hkv, D]
        v = self._gather_pages(v_cache, block_tables, max_seqlen_k, context_lens)  # [B, Lk, Hkv, D]
        mask = self._key_mask(context_lens, max_seqlen_k)  # [B, 1, 1, Lk]
        o = self._sdpa(rearrange(q, "b h d -> b 1 h d"), k, v, mask)
        return write_into(out, rearrange(o, "b 1 h d -> b h d"))

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        # q: [B, H, D], D = kv_lora_rank + rope_dim, and v_dim = kv_lora_rank
        block_tables, context_lens = context.block_tables, context.context_lens
        assert block_tables is not None and context_lens is not None
        max_seqlen_k = block_tables.size(1) * latent_cache.size(1)
        latent = self._gather_pages(latent_cache, block_tables, max_seqlen_k, context_lens)  # [B, Lk, D]
        kv = repeat(latent, "b lk d -> b h lk d", h=q.size(1))  # [B, H, Lk, D]
        mask = self._key_mask(context_lens, max_seqlen_k)
        o = F.scaled_dot_product_attention(
            rearrange(q, "b h d -> b h 1 d"), kv, kv[..., :v_dim], attn_mask=mask, scale=self.scale
        )
        return rearrange(o, "b h 1 d -> b h d")  # [B, H, v_dim]

    @staticmethod
    def _pad_rows(x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        """Packed rows [N, ...] to [B, max_seqlen, ...]. Past a row's end are other tokens, for the mask to hide."""
        index = cu_seqlens[:-1].long().unsqueeze(1) + torch.arange(max_seqlen, device=x.device)
        return x[index.clamp(max=x.size(0) - 1)]

    @staticmethod
    def _unpad_rows(x: torch.Tensor, cu_seqlens: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """[B, max_seqlen, ...] back to packed rows [num_tokens, ...]."""
        tokens = torch.arange(num_tokens, device=x.device, dtype=cu_seqlens.dtype)
        rows = torch.searchsorted(cu_seqlens, tokens, right=True) - 1
        return x[rows, tokens - cu_seqlens[rows]]

    @staticmethod
    def _gather_pages(
        cache: torch.Tensor,
        block_tables: torch.Tensor,
        seqlen: int,
        seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Each row's cached tokens, [B, seqlen, ...], with unused slots zeroed."""
        block_size = cache.size(1)  # [num_blocks, block_size, ...]
        num_blocks = (seqlen + block_size - 1) // block_size
        blocks = block_tables[:, :num_blocks].long().clamp(min=0)  # [B, num_blocks]
        gathered = cache[blocks].flatten(1, 2)[:, :seqlen]  # [B, num_blocks, block_size, ...] -> [B, seqlen, ...]
        # Slots past each row's length hold garbage, maybe NaN or inf. The mask only zeroes their
        # weights, and 0 * NaN is still NaN, so overwrite the garbage itself with 0.
        padding = torch.arange(gathered.size(1), device=cache.device) >= seqlens.unsqueeze(1)  # [B, seqlen]
        padding = padding.view(*padding.shape, *([1] * (gathered.ndim - 2)))  # [B, seqlen, 1, ...]
        return gathered.masked_fill_(padding, 0)

    @staticmethod
    def _key_mask(seqlens_k: torch.Tensor, max_seqlen_k: int) -> torch.Tensor:
        """[B, 1, 1, Lk]: each row's own keys."""
        k_pos = torch.arange(max_seqlen_k, device=seqlens_k.device)
        # [Lk] < [B, 1] -> [B, Lk] -> [B, 1, 1, Lk]
        return (k_pos < seqlens_k.unsqueeze(1)).view(-1, 1, 1, max_seqlen_k)

    @classmethod
    def _mask(cls, cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int, max_seqlen_k: int, causal: bool) -> torch.Tensor:
        """[B, 1, Lq, Lk]. Causal is bottom-right aligned, unlike SDPA's is_causal=True: query j sits at lk - lq + j."""
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        mask = cls._key_mask(seqlens_k, max_seqlen_k)  # [B, 1, 1, Lk]
        if not causal:
            return mask
        device = mask.device
        # [B, 1, 1, 1] + [Lq, 1] -> [B, 1, Lq, 1]
        q_pos = (seqlens_k - seqlens_q).view(-1, 1, 1, 1) + torch.arange(max_seqlen_q, device=device).view(-1, 1)
        return mask & (torch.arange(max_seqlen_k, device=device) <= q_pos)  # [B, 1, Lq, Lk]

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pack query heads that share a KV head into one longer query sequence.
        # This avoids duplicating keys and values or needing enable_gqa.
        # Query head h uses KV head h // group.
        Lq = q.size(1)
        group = self.num_heads // self.num_kv_heads
        q = rearrange(q, "b lq (hkv g) d -> b hkv (g lq) d", g=group)
        mask = repeat(mask, "b 1 lq lk -> b 1 (g lq) lk", g=group)
        o = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=self.scale)
        return rearrange(o, "b hkv (g lq) d -> b lq (hkv g) d", g=group, lq=Lq)
