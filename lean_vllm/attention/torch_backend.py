import torch
import torch.nn.functional as F

from lean_vllm.attention.abstract import AttentionBackend
from lean_vllm.utils.context import Context


class TorchAttention(AttentionBackend):
    """SDPA reference backend. Runs anywhere; optimized for clarity, not speed."""

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

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        num_tokens = key.size(0)
        assert slot_mapping.numel() == num_tokens
        dim = self.num_kv_heads * self.head_dim

        slots = slot_mapping.long()
        keep = slots >= 0
        if not keep.all():    # costs a host sync; kernels mask in-kernel instead
            slots = slots[keep]
            key = key[keep]
            value = value[keep]

        # Indexed assignment, as index_copy_ on MPS costs the whole cache's size per call.
        k_cache.view(-1, dim)[slots] = key.flatten(1)
        v_cache.view(-1, dim)[slots] = value.flatten(1)

    def store_latents(self, latent, latent_cache, slot_mapping) -> None:
        dim = latent_cache.size(-1)
        assert slot_mapping.numel() == latent.size(0)

        slots = slot_mapping.long()
        keep = slots >= 0
        if not keep.all():    # costs a host sync; kernels mask in-kernel instead
            slots = slots[keep]
            latent = latent[keep]

        latent_cache.view(-1, dim)[slots] = latent

    def prefill(self, q, k, v, k_cache, v_cache, context: Context) -> torch.Tensor:
        cu_seqlens_q, cu_seqlens_k = context.cu_seqlens_q, context.cu_seqlens_k
        max_seqlen_q, max_seqlen_k = context.max_seqlen_q, context.max_seqlen_k
        q_pad = self._pad_rows(q, cu_seqlens_q, max_seqlen_q)    # [B, Lq, H, D]
        if context.block_tables is not None:    # read every key back from the pages
            k_pad = self._gather_pages(k_cache, context.block_tables, max_seqlen_k)
            v_pad = self._gather_pages(v_cache, context.block_tables, max_seqlen_k)
        else:
            k_pad = self._pad_rows(k, cu_seqlens_k, max_seqlen_k)
            v_pad = self._pad_rows(v, cu_seqlens_k, max_seqlen_k)
        mask = self._mask(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
        return self._unpad_rows(self._sdpa(q_pad, k_pad, v_pad, mask), cu_seqlens_q, q.size(0))

    def varlen_with_lse(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
        # Written out, since SDPA does not return the log-sum-exp.
        repeats = self.num_heads // self.num_kv_heads
        q_pad = self._pad_rows(q, cu_seqlens_q, max_seqlen_q).transpose(1, 2).float()    # [B, H, Lq, D]
        k_pad = self._pad_rows(k, cu_seqlens_k, max_seqlen_k).repeat_interleave(repeats, dim=2).transpose(1, 2).float()
        v_pad = self._pad_rows(v, cu_seqlens_k, max_seqlen_k).repeat_interleave(repeats, dim=2).transpose(1, 2).float()
        scores = q_pad @ k_pad.transpose(-1, -2) * self.scale    # [B, H, Lq, Lk]
        mask = self._mask(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal)
        scores = scores.masked_fill(~mask, float("-inf"))
        o = (scores.softmax(dim=-1) @ v_pad).transpose(1, 2).to(q.dtype)    # [B, Lq, H, D]
        lse = scores.logsumexp(dim=-1).transpose(1, 2)    # [B, Lq, H]
        return self._unpad_rows(o, cu_seqlens_q, q.size(0)), self._unpad_rows(lse, cu_seqlens_q, q.size(0))

    def decode(self, q, k_cache, v_cache, context: Context) -> torch.Tensor:
        block_tables = context.block_tables
        seqlen = block_tables.size(1) * k_cache.size(1)    # the table's width, so no length is read back
        k = self._gather_pages(k_cache, block_tables, seqlen)
        v = self._gather_pages(v_cache, block_tables, seqlen)
        mask = self._key_mask(context.context_lens, seqlen)
        return self._sdpa(q.unsqueeze(1), k, v, mask).squeeze(1)

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        # q: [B, H, D], D = kv_lora_rank + rope_dim, and v_dim = kv_lora_rank
        block_tables = context.block_tables
        seqlen = block_tables.size(1) * latent_cache.size(1)
        latent = self._gather_pages(latent_cache, block_tables, seqlen)    # [B, Lk, D]
        kv = latent.unsqueeze(1).expand(-1, q.size(1), -1, -1)    # [B, H, Lk, D]
        mask = self._key_mask(context.context_lens, seqlen)
        o = F.scaled_dot_product_attention(q.unsqueeze(2), kv, kv[..., :v_dim], attn_mask=mask, scale=self.scale)
        return o.squeeze(2)[..., :v_dim]    # [B, H, v_dim]; MPS returns the keys' width for one query

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
    def _gather_pages(cache: torch.Tensor, block_tables: torch.Tensor, seqlen: int) -> torch.Tensor:
        """Each row's first seqlen cached tokens, [B, seqlen, ...]. A table's -1 padding reads block 0, for the mask to hide."""
        block_size = cache.size(1) # [num_blocks, block_size, ...]
        num_blocks = (seqlen + block_size - 1) // block_size
        blocks = block_tables[:, :num_blocks].long().clamp(min=0)
        return cache[blocks].flatten(1, 2)[:, :seqlen]

    @staticmethod
    def _key_mask(seqlens_k: torch.Tensor, max_seqlen_k: int) -> torch.Tensor:
        """[B, 1, 1, Lk]: each row's own keys."""
        k_pos = torch.arange(max_seqlen_k, device=seqlens_k.device)
        return (k_pos < seqlens_k.unsqueeze(1)).view(-1, 1, 1, max_seqlen_k)

    @classmethod
    def _mask(cls, cu_seqlens_q, cu_seqlens_k, max_seqlen_q: int, max_seqlen_k: int, causal: bool) -> torch.Tensor:
        """[B, 1, Lq, Lk]. Causal is bottom-right aligned, unlike SDPA's is_causal=True: query j sits at lk - lq + j."""
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        mask = cls._key_mask(seqlens_k, max_seqlen_k)
        if not causal:
            return mask
        device = mask.device
        q_pos = (seqlens_k - seqlens_q).view(-1, 1, 1, 1) + torch.arange(max_seqlen_q, device=device).view(-1, 1)
        return mask & (torch.arange(max_seqlen_k, device=device) <= q_pos)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Each key head's query heads fold into its query axis, so keys are never repeated and no enable_gqa
        # is needed, which takes a slow path on MPS once batched. Query head h reads key head h // group.
        B, Lq, _, D = q.shape
        group = self.num_heads // self.num_kv_heads
        q = q.view(B, Lq, self.num_kv_heads, group, D).permute(0, 2, 3, 1, 4).reshape(B, self.num_kv_heads, group * Lq, D)
        mask = mask.unsqueeze(2).expand(-1, -1, group, -1, -1).reshape(B, 1, group * Lq, -1)
        o = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=self.scale)
        return o.view(B, self.num_kv_heads, group, Lq, -1).permute(0, 3, 1, 2, 4).reshape(B, Lq, self.num_heads, -1)
