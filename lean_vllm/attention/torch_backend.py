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

        k_cache.view(-1, dim).index_copy_(0, slots, key.flatten(1))
        v_cache.view(-1, dim).index_copy_(0, slots, value.flatten(1))

    def store_latents(self, latent, latent_cache, slot_mapping) -> None:
        dim = latent_cache.size(-1)
        assert slot_mapping.numel() == latent.size(0)

        slots = slot_mapping.long()
        keep = slots >= 0
        if not keep.all():    # costs a host sync; kernels mask in-kernel instead
            slots = slots[keep]
            latent = latent[keep]

        latent_cache.view(-1, dim).index_copy_(0, slots, latent)

    def prefill(self, q, k, v, k_cache, v_cache, context: Context) -> torch.Tensor:
        cu_seqlens_q = context.cu_seqlens_q.tolist()
        cu_seqlens_k = context.cu_seqlens_k.tolist()
        block_tables = context.block_tables

        outputs = []
        for i in range(len(cu_seqlens_q) - 1):
            seqlen_q = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
            seqlen_k = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
            q_i = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
            if block_tables is not None:    # read every key back from the pages
                k_i = self._gather_pages(k_cache, block_tables[i], seqlen_k)
                v_i = self._gather_pages(v_cache, block_tables[i], seqlen_k)
            else:
                k_i = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                v_i = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
            mask = self._causal_mask(seqlen_q, seqlen_k, q.device)
            outputs.append(self._sdpa(q_i, k_i, v_i, mask))
        return torch.cat(outputs, dim=0)

    def varlen_with_lse(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
        # Written out, since SDPA does not return the log-sum-exp.
        cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        repeats = self.num_heads // self.num_kv_heads
        outputs, lses = [], []
        for i in range(len(cu_seqlens_q) - 1):
            q_i = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]].transpose(0, 1).float()    # [H, Lq, D]
            k_i = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].repeat_interleave(repeats, dim=1).transpose(0, 1).float()
            v_i = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].repeat_interleave(repeats, dim=1).transpose(0, 1).float()
            scores = q_i @ k_i.transpose(1, 2) * self.scale    # [H, Lq, Lk]
            mask = self._causal_mask(q_i.size(1), k_i.size(1), q.device) if causal else None
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            outputs.append((scores.softmax(dim=-1) @ v_i).transpose(0, 1).to(q.dtype))
            lses.append(scores.logsumexp(dim=-1).transpose(0, 1))
        return torch.cat(outputs), torch.cat(lses)

    def decode(self, q, k_cache, v_cache, context: Context) -> torch.Tensor:
        block_tables = context.block_tables
        outputs = []
        for i, seqlen_k in enumerate(context.context_lens.tolist()):
            k_i = self._gather_pages(k_cache, block_tables[i], seqlen_k)
            v_i = self._gather_pages(v_cache, block_tables[i], seqlen_k)
            outputs.append(self._sdpa(q[i:i + 1], k_i, v_i, None))
        return torch.cat(outputs, dim=0)

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        # q: [B, H, D], D = kv_lora_rank + rope_dim, and v_dim = kv_lora_rank
        block_tables = context.block_tables
        outputs = []
        for i, seqlen_k in enumerate(context.context_lens.tolist()):
            latent = self._gather_pages(latent_cache, block_tables[i], seqlen_k)    # [Lk, D]
            kv = latent.expand(q.size(1), -1, -1)    # [H, Lk, D]
            o = F.scaled_dot_product_attention(q[i].unsqueeze(1), kv, kv[..., :v_dim], scale=self.scale)    # [H, 1, v_dim]
            outputs.append(o.transpose(0, 1))    # [1, H, v_dim]
        return torch.cat(outputs, dim=0) # [B, H, v_dim]

    @staticmethod
    def _gather_pages(cache: torch.Tensor, block_table: torch.Tensor, seqlen: int) -> torch.Tensor:
        block_size = cache.size(1) # [num_blocks, block_size, ...]
        num_blocks = (seqlen + block_size - 1) // block_size
        blocks = block_table[:num_blocks].long()
        return cache[blocks].flatten(0, 1)[:seqlen] # [seq_len, ...]

    @staticmethod
    def _causal_mask(seqlen_q: int, seqlen_k: int, device: torch.device) -> torch.Tensor | None:
        # Bottom-right aligned, unlike SDPA's is_causal=True: query j sits at seqlen_k - seqlen_q + j.
        if seqlen_q == 1:
            return None
        q_pos = torch.arange(seqlen_k - seqlen_q, seqlen_k, device=device).unsqueeze(1)
        k_pos = torch.arange(seqlen_k, device=device).unsqueeze(0)
        return q_pos >= k_pos

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        q = q.transpose(0, 1) # [H, Lq, D]
        k = k.transpose(0, 1) # [H_kv, Lk, D]
        v = v.transpose(0, 1) # [H_kv, Lk, D]

        gqa = self.num_heads != self.num_kv_heads
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale, enable_gqa=gqa) # [H, Lq, D]
        return o.transpose(0, 1)
