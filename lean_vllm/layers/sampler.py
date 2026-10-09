import torch
from torch import nn


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """Masks each row's logits outside its top k, then outside its top p of probability.

    A k of 0 keeps the whole row, as does a p of 1. It sorts the vocab, so a batch with neither passes None.
    The logits are updated in place.
    """
    if k is None and p is None:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        vocab_size = logits.size(1)
        k = torch.where(k > 0, k, vocab_size).clamp_max(vocab_size)
        # The k-th largest value is the cutoff; ties with it stay in.
        cutoff = logits_sort.gather(1, (vocab_size - k.long()).unsqueeze(1))
        logits_sort.masked_fill_(logits_sort < cutoff, -float("inf"))
    if p is not None:
        # Ascending, so a token goes while the mass up to it is within 1 - p; the most likely one always stays.
        probs_sum = logits_sort.softmax(dim=-1, dtype=torch.float32).cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ):
        if temperatures is None:
            return logits.argmax(dim=-1)
        return self.sample(logits, temperatures, top_k, top_p)

    @torch.compile(dynamic=True)  # warmup's batch size is not serving's, so never recompile per size
    def sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
    ):
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits.div_(temperatures.clamp_min(1e-10).unsqueeze(dim=1))
        logits = apply_top_k_top_p(logits, top_k, top_p)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
