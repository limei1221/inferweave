from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from lean_vllm.engine.input_buffers import InputBuffers
from lean_vllm.utils.context import set_context


@dataclass(slots=True)
class DraftInputs:
    """What a step's batch preparation knows that the drafter needs, on the host and in the batch's row order."""

    token_ids: np.ndarray  # the target's input tokens
    positions: np.ndarray
    last_index: np.ndarray  # each row's last token in token_ids
    next_token_ids: np.ndarray  # the token after each row's chunk, for a row that does not sample; -1 for one that does
    sampling_rows: np.ndarray  # the batch row of each sampling row, in sampling order
    num_draft_tokens: np.ndarray  # drafts each sampling row verifies
    block_table: np.ndarray | None  # None in warmup, which has no cache


class MTPProposer:
    """Drafts num_speculative_tokens tokens per sampling row with the checkpoint's MTP layers, as vLLM's proposer.

    The first pass runs over the target's whole batch, each token beside the one after it, so the drafter's cache
    follows the target's; its rows end at the last kept token, beside the token sampled after it. Every further pass
    is one token per row: the draft before it, at the next position. The kept-draft counts are read on the host
    between the two, as vLLM's drafter does with disable_padded_drafter_batch.
    """

    def __init__(
        self,
        drafter: nn.Module,
        embed_tokens: Callable[[torch.Tensor], torch.Tensor],
        compute_logits: Callable[[torch.Tensor], torch.Tensor | None],
        num_speculative_tokens: int,
        block_size: int,
        max_model_len: int,
        rank: int,
        world_size: int,
        device: torch.device,
    ):
        self.drafter = drafter
        self.embed_tokens = embed_tokens
        self.compute_logits = compute_logits
        self.num_speculative_tokens = num_speculative_tokens
        self.block_size = block_size
        self.max_model_len = max_model_len
        self.rank, self.world_size = rank, world_size
        self.buffers = InputBuffers(device)  # its own, so a draft pass never rewrites the target's inputs

    def broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rank 0's tokens on every rank, as only it samples while every rank runs the drafter's layers."""
        if self.world_size > 1:
            dist.broadcast(tensor, src=0)
        return tensor

    def _draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy drafts, as vLLM's MTP drafts; rejection sampling then needs no draft probabilities."""
        logits = self.compute_logits(hidden_states)  # gathered on rank 0 only
        if logits is not None:
            tokens = logits.argmax(dim=-1)
        else:
            tokens = torch.empty(hidden_states.size(0), dtype=torch.int64, device=hidden_states.device)
        return self.broadcast(tokens)

    @torch.inference_mode()
    def propose(
        self, inputs: DraftInputs, context: dict, target_hidden_states: torch.Tensor, sampled: list[list[int]]
    ) -> torch.Tensor:
        """Drafts [sampling rows, num_speculative_tokens] after the target's step. sampled is each sampling row's
        tokens, -1 past its last, as the rejection sampler returns them; context is the target's step."""
        num_rows = len(sampled)
        num_accepted = np.array([sum(t >= 0 for t in row) - 1 for row in sampled], np.int64)
        token_ids = np.roll(inputs.token_ids, -1)
        not_sampling = inputs.next_token_ids >= 0
        token_ids[inputs.last_index[not_sampling]] = inputs.next_token_ids[not_sampling]
        # Each sampling row's first draft comes from its last kept token, beside the token sampled after it.
        sample_index = inputs.last_index[inputs.sampling_rows] - inputs.num_draft_tokens + num_accepted
        token_ids[sample_index] = [row[n] for row, n in zip(sampled, num_accepted.tolist())]

        buffers = self.buffers
        buffers.begin()
        input_ids = buffers.put("input_ids", token_ids, torch.int64)
        sample_index_t = buffers.put("sample_index", sample_index, torch.int64)
        positions = buffers.put("positions", inputs.positions, torch.int64)
        steps = [self._step_context(inputs, sample_index, step) for step in range(1, self.num_speculative_tokens)]
        buffers.end()

        with set_context(**{**context, "logits_indices": None}):
            hidden_states = self.drafter(self.embed_tokens(input_ids), positions, target_hidden_states)
            hidden_states = hidden_states[sample_index_t]
            drafts = [self._draft(hidden_states)]
        if not num_rows:
            return torch.empty(0, self.num_speculative_tokens, dtype=torch.int64, device=input_ids.device)
        for step, (step_positions, step_context) in enumerate(steps, 1):
            with set_context(**step_context):
                hidden_states = self.drafter(self.embed_tokens(drafts[-1]), step_positions, hidden_states, step)
                drafts.append(self._draft(hidden_states))
        return torch.stack(drafts, dim=1)

    def _step_context(self, inputs: DraftInputs, sample_index: np.ndarray, step: int) -> tuple[torch.Tensor, dict]:
        """One token per sampling row, step positions past its last kept one; as a decode step's context."""
        put = self.buffers.put
        num_rows = len(sample_index)
        # Past max_model_len only for drafts no step would schedule; clamped so the rope cache covers them.
        positions = np.minimum(inputs.positions[sample_index] + step, self.max_model_len - 1)
        cu_q = np.arange(num_rows + 1)
        context: dict = dict(
            is_prefill=False, max_seqlen_q=1, cu_seqlens_q=put(f"cu_seqlens_q_{step}", cu_q, torch.int32)
        )
        if inputs.block_table is None:  # warmup: no cache, so each token attends itself alone
            context.update(cu_seqlens_k=context["cu_seqlens_q"], max_seqlen_k=1)
            context.update(cu_seqlens_q_host=cu_q.tolist(), cu_seqlens_k_host=cu_q.tolist())
        else:
            table = inputs.block_table[inputs.sampling_rows]
            lens = positions + 1
            cu_k = np.zeros(num_rows + 1, np.int64)
            np.cumsum(lens, out=cu_k[1:])
            blocks = table[np.arange(num_rows), positions // self.block_size].astype(np.int64)
            context.update(
                cu_seqlens_k=put(f"cu_seqlens_k_{step}", cu_k, torch.int32),
                max_seqlen_k=int(lens.max()) if num_rows else 0,
                cu_seqlens_q_host=cu_q.tolist(),
                cu_seqlens_k_host=cu_k.tolist(),
                slot_mapping=put(
                    f"slot_mapping_{step}", blocks * self.block_size + positions % self.block_size, torch.int32
                ),
                context_lens=put(f"context_lens_{step}", lens, torch.int32),
                block_tables=put(f"block_tables_{step}", table, torch.int32),
            )
        return put(f"positions_{step}", positions, torch.int64), context
