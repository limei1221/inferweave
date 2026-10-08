from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist

from lean_vllm.engine.input_buffers import InputBuffers
from lean_vllm.models.deepseek_mtp import DeepSeekMTP
from lean_vllm.utils.context import Context, set_context


@dataclass(slots=True)
class DraftInputs:
    """What a step's batch preparation knows that the drafter needs, in the batch's row order."""

    input_ids: torch.Tensor  # the target's, on the device, where an async step fills what the host lacks
    positions: torch.Tensor  # the target's, on the device, where an async step moves rows back past rejected drafts
    max_position: int  # of a sampling row's last token, as if every draft was kept: vLLM's max_seq_len bound
    last_index: np.ndarray  # each row's last token in input_ids
    next_token_ids: np.ndarray  # the token after each row's chunk, for a row that does not sample; -1 for one that does
    sampling_rows: np.ndarray  # the batch row of each sampling row, in sampling order
    num_draft_tokens: np.ndarray  # drafts each sampling row verifies
    block_table: np.ndarray | None  # None in warmup, which has no cache


class MTPProposer:
    """Drafts num_speculative_tokens tokens per sampling row with the checkpoint's MTP layers, as vLLM's proposer.

    The first pass runs over the target's whole batch, each token beside the one after it, so the drafter's cache
    follows the target's; its rows end at the last kept token, beside the token sampled after it. Every further pass
    is one token per row: the draft before it, at the next position. As vLLM's padded drafter batch, the kept-draft
    counts stay on the device: rejected drafts' rows run in the first pass as padding, and the later passes find
    their positions there, so the host never waits on verification. Those single-token passes replay full CUDA
    graphs when the target captures its own.
    """

    graph_bs: list[int] = []  # captured batch sizes; none until capture_cudagraphs

    def __init__(
        self,
        drafter: DeepSeekMTP,
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
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}  # by MTP layer and batch size

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
        self,
        inputs: DraftInputs,
        context: dict,
        target_hidden_states: torch.Tensor,
        num_accepted: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Drafts [sampling rows, num_speculative_tokens] after the target's step, whose context this is. Each
        sampling row kept num_accepted of its drafts, then next_token_ids; both on the device."""
        num_rows = num_accepted.size(0)
        not_sampling = inputs.next_token_ids >= 0
        buffers = self.buffers
        buffers.begin()
        fixed_index = buffers.put("fixed_index", inputs.last_index[not_sampling], torch.int64)
        fixed_token_ids = buffers.put("fixed_token_ids", inputs.next_token_ids[not_sampling], torch.int64)
        first_draft_index = inputs.last_index[inputs.sampling_rows] - inputs.num_draft_tokens
        sample_index = buffers.put("first_draft_index", first_draft_index, torch.int64)
        cu_seqlens_q = buffers.put("cu_seqlens_q", np.arange(num_rows + 1), torch.int32)
        table = None
        if inputs.block_table is not None:  # None in warmup
            table = buffers.put("block_table", inputs.block_table[inputs.sampling_rows], torch.int32)
        buffers.end()

        # Each token beside the one after it, as vLLM's set_inputs_first_pass. Each sampling row's first draft comes
        # from its last kept token, beside the token sampled after it, as vLLM's eagle_prepare_inputs_padded_kernel;
        # the rows after it are padding.
        input_ids = inputs.input_ids.roll(-1)
        input_ids[fixed_index] = fixed_token_ids
        sample_index += num_accepted
        input_ids[sample_index] = next_token_ids
        positions = inputs.positions
        with set_context(**{**context, "logits_indices": None}):
            hidden_states = self.drafter(self.embed_tokens(input_ids), positions, target_hidden_states)
            hidden_states = hidden_states[sample_index]
            drafts = [self._draft(hidden_states)]
        if not num_rows:
            return torch.empty(0, self.num_speculative_tokens, dtype=torch.int64, device=input_ids.device)
        sample_positions = positions[sample_index]
        steps = [
            self._step_context(sample_positions, table, cu_seqlens_q, inputs.max_position, step)
            for step in range(1, self.num_speculative_tokens)
        ]
        graph_bs = next((size for size in self.graph_bs if size >= num_rows), None)
        for step, (step_positions, step_context) in enumerate(steps, 1):
            with set_context(**step_context) as step_ctx:
                if graph_bs is None or step_ctx.block_tables is None:  # past the largest graph, or warmup
                    hidden_states = self.drafter(self.embed_tokens(drafts[-1]), step_positions, hidden_states, step)
                else:
                    layer = step % self.drafter.num_mtp_layers
                    hidden_states = self._replay(layer, graph_bs, drafts[-1], step_positions, hidden_states, step_ctx)
                drafts.append(self._draft(hidden_states))
        return torch.stack(drafts, dim=1)

    @torch.inference_mode()
    def capture_cudagraphs(self, sizes: list[int], max_num_blocks: int, pool, backends: list) -> None:
        """A full graph of one single-token pass per batch size and MTP layer, where vLLM's drafter takes
        piecewise graphs only; the logits stay eager. backends are the attention backends whose replay hooks run
        before each replay."""
        layers = sorted({step % self.drafter.num_mtp_layers for step in range(1, self.num_speculative_tokens)})
        if not layers:
            return  # one draft per row, which the first pass makes
        weight = next(self.drafter.parameters())
        max_bs, hidden_size, device = sizes[-1], self.drafter.hidden_size, weight.device
        self.graph_vars = dict(
            input_ids=torch.zeros(max_bs, dtype=torch.int64, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int64, device=device),
            hidden_states=torch.zeros(max_bs, hidden_size, dtype=weight.dtype, device=device),
            slot_mapping=torch.zeros(max_bs, dtype=torch.int32, device=device),
            # FlashMLA bakes its schedule from the lengths at capture, so the worst case, as the target's capture.
            context_lens=torch.full((max_bs,), self.max_model_len, dtype=torch.int32, device=device),
            block_tables=torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=device),
            outputs=torch.zeros(max_bs, hidden_size, dtype=weight.dtype, device=device),
        )
        self.graph_backends = backends
        for bs in reversed(sizes):
            for layer in layers:
                self.graphs[(layer, bs)] = self._capture(partial(self._graph_pass, layer, bs), pool)
        self.graph_bs = sizes

    def _graph_pass(self, layer: int, bs: int) -> None:
        """One single-token pass over the graph's buffers, in a context of them alone."""
        graph_vars = self.graph_vars
        with set_context(
            False,
            slot_mapping=graph_vars["slot_mapping"][:bs],
            context_lens=graph_vars["context_lens"][:bs],
            block_tables=graph_vars["block_tables"][:bs],
            full_graph_size=bs,
        ):
            inputs_embeds = self.embed_tokens(graph_vars["input_ids"][:bs])
            hidden_states = graph_vars["hidden_states"][:bs]
            graph_vars["outputs"][:bs] = self.drafter(inputs_embeds, graph_vars["positions"][:bs], hidden_states, layer)

    @staticmethod
    def _capture(run: Callable[[], None], pool) -> torch.cuda.CUDAGraph:
        run()  # outside the graph first, where first launches may compile or allocate
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool):
            run()  # a fresh context, so FlashMLA's schedule holder lands in the graph's pool
        torch.cuda.synchronize()
        return graph

    def _replay(
        self,
        layer: int,
        bs: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        """The pass for these rows from the graph of bs rows, as the target's _replay_full; padding rows write no
        slot and attend nothing."""
        assert context.slot_mapping is not None and context.context_lens is not None
        assert context.block_tables is not None
        graph_vars = self.graph_vars
        n = input_ids.size(0)
        graph_vars["input_ids"][:n] = input_ids
        graph_vars["positions"][:n] = positions
        graph_vars["hidden_states"][:n] = hidden_states
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:n] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:n] = context.context_lens
        # Positions stop short of max_model_len, so blocks past the buffer's width are lookahead no pass reads.
        width = min(context.block_tables.size(1), graph_vars["block_tables"].size(1))
        graph_vars["block_tables"][:n, :width] = context.block_tables[:, :width]
        for backend in self.graph_backends:
            backend.before_full_graph_replay(context, bs)
        self.graphs[(layer, bs)].replay()
        return graph_vars["outputs"][:n]

    def _step_context(
        self,
        sample_positions: torch.Tensor,
        table: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        max_position: int,
        step: int,
    ) -> tuple[torch.Tensor, dict]:
        """One token per sampling row, step positions past its last kept one; as a decode step's context, built on
        the device, as vLLM's eagle_step_update_slot_mapping_and_metadata. Key lengths have only a bound on the host."""
        cu_q = list(range(cu_seqlens_q.numel()))
        # Past max_model_len only for drafts no step would schedule; clamped so the rope cache covers them.
        positions = (sample_positions + step).clamp_(max=self.max_model_len - 1)
        context: dict = dict(is_prefill=False, max_seqlen_q=1, cu_seqlens_q=cu_seqlens_q, cu_seqlens_q_host=cu_q)
        if table is None:  # warmup: no cache, so each token attends itself alone
            context.update(cu_seqlens_k=cu_seqlens_q, max_seqlen_k=1, cu_seqlens_k_host=cu_q)
        else:
            lens = positions + 1
            blocks = table.gather(1, (positions // self.block_size).unsqueeze(1)).squeeze(1).long()
            context.update(
                cu_seqlens_k=torch.nn.functional.pad(lens.cumsum(0), (1, 0)).int(),
                max_seqlen_k=min(max_position + step, self.max_model_len - 1) + 1,
                slot_mapping=(blocks * self.block_size + positions % self.block_size).int(),
                context_lens=lens.int(),
                block_tables=table,
            )
        return positions, context
