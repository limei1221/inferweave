import torch
from torch import nn

from lean_vllm.spec_decode.config import PLACEHOLDER_TOKEN_ID


class RejectionSampler(nn.Module):
    """Verifies each row's drafts against the target's logits, as vLLM's RejectionSampler (Leviathan et al. 2023).

    MTP drafts greedily, so a draft is its own whole distribution: it is kept with the target's probability of it,
    and the first one rejected is replaced by a sample from the target with that token taken out. A greedy row keeps
    a draft only if it is the target's argmax. A row that keeps every draft adds a bonus token from its last position.
    """

    def forward(
        self,
        logits: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        max_num_drafts: int,
    ) -> torch.Tensor:
        """The tokens each row produced, [rows, max_num_drafts + 1], PLACEHOLDER_TOKEN_ID past its last.

        logits holds each row's draft positions, then its bonus position, row after row: [sum(n + 1), vocab].
        draft_token_ids is every row's drafts in the same order, and num_draft_tokens how many each row has.
        """
        num_rows, num_drafts = num_draft_tokens.numel(), draft_token_ids.numel()
        device = logits.device
        rows = torch.arange(num_rows, device=device)
        num_logits = num_draft_tokens + 1
        first = torch.cumsum(num_logits, 0) - num_logits  # each row's first logits row
        # Each draft's row, its place in the row, and its logits row; no host sync, as the sizes are known.
        draft_rows = torch.repeat_interleave(rows, num_draft_tokens, output_size=num_drafts)
        draft_cols = (
            torch.arange(num_drafts, device=device) - (torch.cumsum(num_draft_tokens, 0) - num_draft_tokens)[draft_rows]
        )
        draft_index = first[draft_rows] + draft_cols

        greedy = logits.argmax(dim=-1)
        if temperatures is None:
            accepted = greedy[draft_index] == draft_token_ids
            candidates = greedy
        else:
            row_temperatures = torch.repeat_interleave(temperatures, num_logits, output_size=logits.size(0))
            probs = torch.softmax(logits.float() / row_temperatures.clamp_min(1e-10).unsqueeze(1), dim=-1)
            draft_probs = probs[draft_index, draft_token_ids]
            is_greedy = temperatures[draft_rows] == 0
            accepted = torch.where(
                is_greedy, greedy[draft_index] == draft_token_ids, torch.rand_like(draft_probs) < draft_probs
            )
            # Where a draft is rejected, the next token comes from what the target wanted beyond it.
            probs[draft_index, draft_token_ids] = 0
            sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            candidates = torch.where(row_temperatures == 0, greedy, sampled)

        # A row keeps its drafts up to the first rejection; the last column is never a draft, so the count stops there.
        keep = torch.zeros(num_rows, max_num_drafts + 1, dtype=torch.int32, device=device)
        keep[draft_rows, draft_cols] = accepted.int()
        num_accepted = keep.cumprod(dim=1).sum(dim=1)
        output = torch.full((num_rows, max_num_drafts + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int64, device=device)
        output[draft_rows, draft_cols] = draft_token_ids
        output.masked_fill_(
            torch.arange(max_num_drafts + 1, device=device) >= num_accepted.unsqueeze(1), PLACEHOLDER_TOKEN_ID
        )
        output[rows, num_accepted] = candidates[first + num_accepted]
        return output
