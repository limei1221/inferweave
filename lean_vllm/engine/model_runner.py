import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import record_function

from lean_vllm.config import FULL_MODES, PIECEWISE_MODES, Config
from lean_vllm.engine.compilation import PiecewiseBackend, compile_piecewise, mark_dynamic_tokens
from lean_vllm.engine.input_buffers import InputBuffers
from lean_vllm.engine.sampled_tokens import SampledTokens
from lean_vllm.engine.sequence import Sequence
from lean_vllm.eplb import EplbState
from lean_vllm.kv_transfer import KVConnectorMetadata, KVConnectorOutput, KVOutputAggregator, create_worker_connector
from lean_vllm.layers.attention import Attention, MLAAttention, register_layers
from lean_vllm.layers.sampler import Sampler
from lean_vllm.models import get_drafter_class, get_model_class
from lean_vllm.spec_decode import RejectionSampler
from lean_vllm.spec_decode.mtp_proposer import DraftInputs, MTPProposer
from lean_vllm.utils import device as dev
from lean_vllm.utils.context import get_context, set_context
from lean_vllm.utils.loader import load_model

logger = logging.getLogger(__name__)

# How long a TP worker may wait for its next call; gloo's 30-minute default would kill an idle server.
CALL_TIMEOUT = timedelta(days=365)


def cudagraph_capture_sizes(max_num_seqs: int, max_num_batched_tokens: int) -> list[int]:
    """vLLM's default capture sizes: 1, 2, 4, then every 8 below 256 and every 16 from there, up to twice the
    batch's sequences, 512 at most, and the token budget. The budget itself is captured if it fits.

    Piecewise graphs take every size, and full decode graphs those of at most max_num_seqs rows.
    """
    top = min(max_num_seqs * 2, 512, max_num_batched_tokens)
    sizes = [size for size in (1, 2, 4) if size <= top]
    sizes += range(8, min(top + 1, 256), 8)
    sizes += range(256, top + 1, 16)
    if max_num_batched_tokens <= top:
        sizes.append(max_num_batched_tokens)
    return sorted(set(sizes))


def decode_query_len(lens: np.ndarray, num_speculative_tokens: int) -> int:
    """The query length of the rows MLA decode takes this step: 1, or with drafts the commonest one up to 1 + drafts,
    as FlashMLA takes one per call. vLLM's FlashMLA likewise needs its decode rows uniform."""
    short = lens[lens <= 1 + num_speculative_tokens]
    return int(np.argmax(np.bincount(short))) if short.size else 1


@dataclass(slots=True)
class InFlightDrafts:
    """What a launched speculative step leaves on the device for the next, by sampling row: vLLM's
    prev_sampled_token_ids, _draft_token_ids and valid_sampled_token_count."""

    next_token_ids: torch.Tensor  # each row's newest token, after its kept drafts
    draft_token_ids: torch.Tensor  # [rows, num_speculative_tokens], for the next step to verify
    num_rejected: torch.Tensor  # drafts it verified and rejected, by which the next step's positions move back


class ModelRunner:
    cascade_layers: list[Attention] = []  # one layer per kind, backend and head count; each must agree to cascade
    eplb: EplbState | None = None  # with enable_eplb
    proposer: MTPProposer | None = None  # with a speculative_config
    num_speculative_tokens = 0
    in_flight_drafts: InFlightDrafts | None = None  # with a speculative_config, the last launched step's
    decodes_latents = True  # every MLA layer attends its latents, so no attention reads key lengths on the host

    def __init__(self, config: Config, rank: int):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        Sequence.block_size = self.block_size  # spawned workers never run LLMEngine.__init__
        self.device = dev.get_device()
        self.step_kind = "enforced"  # how the last step ran: see _step_kind
        self._prev_tokens: SampledTokens | None = None  # the step still in flight, if any
        self._prev_rows: dict[int, int] | None = None  # seq_id -> its row in those tokens
        model_cls = get_model_class(hf_config)
        self.graph_bs: list[int] = []  # captured batch sizes, full graphs
        self.piecewise_bs: list[int] = []  # captured token counts, piecewise graphs
        self.graphs: dict = {}
        self.graph_pool = None  # shared by both capture kinds
        self.compile_backend: PiecewiseBackend | None = None  # holds the pieces and their graphs
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        dist.init_process_group(
            dev.dist_backend(self.device), f"tcp://localhost:{config.dist_port}", world_size=self.world_size, rank=rank
        )
        if self.world_size > 1:
            # Calls go to the workers on the host, whatever device the default group runs on.
            self.call_group = dist.new_group(backend="gloo", timeout=CALL_TIMEOUT)
        dev.set_device(self.device, rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        model_kwargs: dict = {"enable_expert_parallel": True} if config.enable_expert_parallel else {}
        if config.eplb is not None:
            model_kwargs["eplb_config"] = config.eplb
        self.model = model_cls(hf_config, **model_kwargs)
        register_layers(self.model)  # before warmup_model, which runs the op
        drafter = None
        if config.speculative is not None:
            # Its own module, so the target's compile and graphs never hold it; it compiles and captures its own.
            drafter = get_drafter_class(hf_config)(hf_config, config.enable_expert_parallel)
            register_layers(drafter)
        # Binds the MoE layers' maps, which every forward reads, and holds a layer's worth of buffer before warmup.
        self.eplb = EplbState(self.model, config.eplb) if config.eplb is not None else None
        # Each layer chose its backend as it was built; graphs depend on all of them, the drafter's too.
        layers = [module for module in self.model.modules() if isinstance(module, Attention)]
        if drafter is not None:
            layers += [module for module in drafter.modules() if isinstance(module, Attention)]
        if rank == 0:
            counts = Counter(layer.backend.get_name() for layer in layers)
            logger.info("attention backends: %s", ", ".join(f"{name} ({n} layers)" for name, n in counts.items()))
        self.attention_backends = list(dict.fromkeys(type(layer.backend) for layer in layers))  # replay hooks
        groups = {(type(layer), type(layer.backend), layer.num_heads, layer.num_kv_heads): layer for layer in layers}
        self.decodes_latents = all(
            layer.backend.supports_mla_decode() for layer in layers if isinstance(layer, MLAAttention)
        )
        self.cascade_layers = list(groups.values())
        self.enforce_eager = (
            config.enforce_eager
            or self.device.type != "cuda"
            or not model_cls.supports_cuda_graph
            or not all(layer.backend.supports_cuda_graph() for layer in layers)
        )
        mode = "none" if self.enforce_eager else config.cudagraph_mode
        self.cudagraph_mode = self._cudagraph_mode(mode, layers)
        if rank == 0 and self.cudagraph_mode != mode:
            logger.info(
                "some attention layer cannot run in a full graph; cudagraph_mode is %r not %r",
                self.cudagraph_mode,
                mode,
            )
        for module in self.model.modules():
            if isinstance(module, MLAAttention):
                # Warmup expands a step's worth of new latents, so a chunk this size fits what it measured.
                module.max_context_chunk = config.max_num_batched_tokens
        load_model(self.model, config.model)
        if self.cudagraph_mode != "none":
            self.compile_model()
        self.sampler = Sampler()
        if drafter is not None:
            assert config.speculative is not None
            for module in drafter.modules():
                if isinstance(module, MLAAttention):
                    module.max_context_chunk = config.max_num_batched_tokens
            load_model(drafter, config.model)
            self.num_speculative_tokens = config.speculative.num_speculative_tokens
            self.rejection_sampler = RejectionSampler()
            self.proposer = MTPProposer(
                drafter,
                self.model.model.embed_tokens,
                self.model.compute_logits,
                self.num_speculative_tokens,
                self.block_size,
                config.max_model_len,
                rank,
                self.world_size,
                self.device,
            )
            if self.cudagraph_mode != "none":
                self.proposer.compile(self.graph_pool)
        self.warmup_model()
        self.allocate_kv_cache()
        self.kv_connector = None
        if config.kv_transfer is not None:
            self.kv_connector = create_worker_connector(config, rank)
            self.kv_connector.register_kv_caches(self.kv_cache)
            self.kv_aggregator = KVOutputAggregator(self.world_size)
        if self.cudagraph_mode in FULL_MODES:
            self.capture_cudagraph()
            if self.proposer is not None:
                self.proposer.capture_cudagraphs(
                    self.graph_bs, self._max_num_blocks(), self.graph_pool, self.attention_backends
                )
        if self.cudagraph_mode in PIECEWISE_MODES:
            self.capture_piecewise()
        if self.eplb is not None:
            self.eplb.reset()  # warmup's and capture's load is not the workload's
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1 and rank > 0:
            self.loop()

    def exit(self):
        if self.kv_connector is not None:
            self.kv_connector.shutdown()
        if self.world_size > 1:
            dist.barrier()  # sync ranks
        if self.cudagraph_mode != "none":
            for piece in self.compile_backend.pieces:
                piece.graphs.clear()
            if self.proposer is not None:
                self.proposer.graphs.clear()
                if self.proposer.compile_backend is not None:
                    for piece in self.proposer.compile_backend.pieces:
                        piece.graphs.clear()
            del self.graphs, self.graph_pool
        dev.synchronize(self.device)  # drain the device
        dist.destroy_process_group()  # drop the comms

    def loop(self):
        while True:
            call = [None, None]
            dist.broadcast_object_list(call, src=0, group=self.call_group)
            method_name, args = call
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            dist.broadcast_object_list([method_name, args], src=0, group=self.call_group)
        method = getattr(self, method_name, None)
        return method(*args)

    def kv_connector_step(self, metadata: KVConnectorMetadata) -> KVConnectorOutput | None:
        """Start this step's transfers on every rank, and return those finished on all of them (rank 0 only)."""
        assert self.kv_connector is not None
        self.kv_connector.start_load_kv(metadata)
        output = self.kv_connector.get_finished()
        if self.world_size == 1:
            return output
        outputs: list | None = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(output, outputs, dst=0, group=self.call_group)
        return self.kv_aggregator.aggregate(outputs) if outputs is not None else None

    def compile_model(self):
        """As vLLM: traced whole, split at attention, pieces compiled by Inductor. Warmup's step runs the compile."""
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.compile_backend = compile_piecewise(self.model, self.graph_pool)

    def warmup_model(self):
        dev.empty_cache(self.device)
        dev.reset_peak_memory_stats(self.device)
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs)
        self._dummy_sampler_run()
        dev.empty_cache(self.device)

    @torch.inference_mode()
    def _dummy_sampler_run(self):
        """As vLLM does: a step samples up to one row per sequence, far more than the warmup prefill, so measure that too.
        With drafts, each row has a logits row per draft beside its own."""
        num_logits = 1 + self.num_speculative_tokens
        num_rows = max(min(self.config.max_num_seqs * num_logits, self.config.max_num_batched_tokens) // num_logits, 1)
        # Random, as vLLM's: dummy hidden states could hold values that break the sampler.
        hidden_states = torch.rand(num_rows * num_logits, self.config.hf_config.hidden_size)
        with set_context(False):  # no logits_indices, so every row samples
            logits = self.model.compute_logits(hidden_states)
        if self.rank != 0:
            return  # only rank 0 gathers logits and samples
        temperatures = torch.full((num_rows,), 0.5, dtype=torch.float32)  # non-greedy, the costlier path
        try:
            if self.proposer is None:
                self.sampler(logits, temperatures)
            else:
                num_draft_tokens = torch.full((num_rows,), self.num_speculative_tokens, dtype=torch.int64)
                drafts = torch.zeros(num_rows * self.num_speculative_tokens, dtype=torch.int64)
                self.rejection_sampler(logits, drafts, num_draft_tokens, temperatures, self.num_speculative_tokens)
        except torch.OutOfMemoryError as error:
            raise RuntimeError(
                f"out of memory warming up the sampler with {num_rows} rows; "
                "lower max_num_seqs or gpu_memory_utilization"
            ) from error

    def allocate_kv_cache(self):
        config = self.config
        # Each layer names its cache layout, which its backend may choose: keys and values per head, or one MLA latent.
        layers = [module for module in self.model.modules() if isinstance(module, Attention)]
        if self.proposer is not None:  # the drafter's layers cache by the target's block tables
            layers += [module for module in self.proposer.drafter.modules() if isinstance(module, Attention)]
        block_numel = sum(math.prod(layer.kv_cache_shape(1, self.block_size)) for layer in layers)
        block_bytes = block_numel * config.hf_config.dtype.itemsize
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = dev.kvcache_bytes(self.device, config) // block_bytes
        assert config.num_kvcache_blocks > 0, "no memory left for the kv cache"
        self.kv_cache = [
            torch.empty(layer.kv_cache_shape(config.num_kvcache_blocks, self.block_size)) for layer in layers
        ]
        for layer, cache in zip(layers, self.kv_cache):
            layer.bind_kv_cache(cache)

    @property
    def input_buffers(self) -> InputBuffers:
        """Made on first use, so a runner built without __init__, as the tests build it, has one too."""
        buffers = self.__dict__.get("_input_buffers")
        if buffers is None:
            buffers = self._input_buffers = InputBuffers(self.device)
        return buffers

    def prepare_batch(self, seqs: list[Sequence]):
        """One batch for any mix of prompt chunks and decode rows, and the context to run it in.

        As vLLM's _prepare_inputs: one pass over the rows for their scalars and tokens, then numpy for everything
        per token, and one copy per tensor into buffers that outlive the step.
        """
        buffers = self.input_buffers
        buffers.begin()
        num_rows, block_size = len(seqs), self.block_size
        # Scheduler order first: it is the order the sampled tokens are read back in.
        lens = np.fromiter((seq.num_scheduled_tokens for seq in seqs), np.int64, num_rows)
        starts = np.fromiter((seq.num_cached_tokens for seq in seqs), np.int64, num_rows)
        planned = np.fromiter((seq.num_planned_tokens for seq in seqs), np.int64, num_rows)
        # A decoding row verifies its drafts beside its token, so it runs and samples one more position per draft.
        num_drafts = np.fromiter(
            (0 if seq.is_prefill else seq.num_scheduled_tokens - 1 for seq in seqs), np.int64, num_rows
        )
        query_len = decode_query_len(lens, self.num_speculative_tokens)
        order = np.argsort(lens != query_len, kind="stable")  # decodes_first, as indices
        batch = [seqs[i] for i in order]
        row_of = np.empty(num_rows, np.int64)
        row_of[order] = np.arange(num_rows)  # each scheduled sequence's row in the batch

        lens, starts = lens[order], starts[order]
        # A row with nothing left to prefill samples its last token, and each of its drafts' positions.
        sampling = np.flatnonzero(starts[row_of] + lens[row_of] >= planned)  # in scheduler order
        # Rows of an async step verifying drafts while the step before is in flight: placed as if it kept all of
        # its own, as vLLM's optimistic num_computed_tokens, and moved back on the device, so the host never waits.
        in_flight = self.in_flight_drafts
        moved = [i for i, seq in enumerate(batch) if seq.num_pending_tokens] if in_flight is not None else []
        moved_src = [self._prev_row(batch[i]) for i in moved]
        is_prefill = any(seq.is_prefill for seq in seqs) or bool(num_drafts.any())  # rows of several queries
        if moved and (not self.decodes_latents or (is_prefill and bool((lens[moved] != query_len).any()))):
            # Expanding latents plans keys on the host, so it waits for the step in flight: a sync.
            assert in_flight is not None
            num_rejected = in_flight.num_rejected.tolist()
            starts[moved] -= [num_rejected[src] for src in moved_src]
            moved = []
        ends = starts + lens
        cu_q = np.zeros(num_rows + 1, np.int64)
        np.cumsum(lens, out=cu_q[1:])
        cu_k = np.zeros(num_rows + 1, np.int64)
        np.cumsum(ends, out=cu_k[1:])
        token_rows = np.repeat(np.arange(num_rows), lens)
        positions = starts[token_rows] + np.arange(cu_q[-1]) - cu_q[token_rows]

        token_ids: list[int] = []
        pending_dst: list[int] = []
        pending_src: list[int] = []
        draft_dst: list[int] = []
        draft_src: list[int] = []  # into the in-flight drafts, flattened
        k = self.num_speculative_tokens
        for seq, start, end in zip(batch, starts.tolist(), ends.tolist()):
            if seq.is_prefill:
                assert not seq.num_pending_tokens, "a prefill row carries a pending token"
                token_ids.extend(seq[start:end])
            else:
                if seq.num_pending_tokens:
                    # Sampled by a step still in flight, its drafts too; the device copy fixes them below.
                    src = self._prev_row(seq)
                    pending_dst.append(len(token_ids))
                    pending_src.append(src)
                    if in_flight is not None:
                        n = len(seq.spec_token_ids)
                        draft_dst += range(len(token_ids) + 1, len(token_ids) + 1 + n)
                        draft_src += range(src * k, src * k + n)
                token_ids.append(seq.last_token)
                token_ids.extend(seq.spec_token_ids)

        sampling_drafts = num_drafts[sampling]
        last_index = cu_q[1:] - 1
        num_logits = sampling_drafts + 1
        offsets = np.arange(num_logits.sum()) - np.repeat(np.cumsum(num_logits) - num_logits, num_logits)
        logits_indices = np.repeat(last_index[row_of[sampling]] - sampling_drafts, num_logits) + offsets
        sampling_rows = [seqs[i] for i in sampling.tolist()]
        # Only the sampling rank owns sampling parameters.
        row_temperatures = [seq.temperature for seq in sampling_rows] if self.rank == 0 else []

        block_tables = slot_mapping = None
        common_prefix_len = 0
        tables = [seq.block_table for seq in batch]
        if any(tables):
            table = np.full((num_rows, max(map(len, tables))), -1, np.int32)
            for i, row in enumerate(tables):
                table[i, : len(row)] = row
            # Each token's slot is its block's, plus its offset in it. Rows with no blocks (warmup) store nothing.
            keep = np.fromiter(map(bool, tables), bool, num_rows)[token_rows]
            kept = positions[keep]
            blocks = table[token_rows[keep], kept // block_size].astype(np.int64)
            slot_mapping = blocks * block_size + kept % block_size
            block_tables = buffers.put("block_tables", table, torch.int32)
            common_prefix_len = self._cascade_prefix_len(table, starts, lens)
        cu_seqlens_q, cu_seqlens_k = cu_q.tolist(), cu_k.tolist()
        context = dict(
            is_prefill=is_prefill,
            cu_seqlens_q=buffers.put("cu_seqlens_q", cu_q, torch.int32),
            cu_seqlens_k=buffers.put("cu_seqlens_k", cu_k, torch.int32),
            max_seqlen_q=int(np.max(lens)),
            max_seqlen_k=int(np.max(ends)),
            cu_seqlens_q_host=cu_seqlens_q,
            cu_seqlens_k_host=cu_seqlens_k,
            slot_mapping=buffers.put("slot_mapping", [] if slot_mapping is None else slot_mapping, torch.int32),
            context_lens=buffers.put("context_lens", ends, torch.int32),
            block_tables=block_tables,
            # A pure-decode batch samples on every row, so the gather is skipped.
            logits_indices=buffers.put("logits_indices", logits_indices, torch.int64) if is_prefill else None,
            common_prefix_len=common_prefix_len,
            decode_query_len=query_len,
        )
        input_ids = buffers.put("input_ids", np.array(token_ids, np.int64), torch.int64)
        positions_t = buffers.put("positions", positions, torch.int64)
        all_greedy = all(temperature == 0 for temperature in row_temperatures)
        temperatures = None if all_greedy else buffers.put("temperatures", row_temperatures, torch.float32)
        if pending_dst:
            if in_flight is not None:
                prev = in_flight.next_token_ids
            else:
                assert self._prev_tokens is not None
                prev = self._prev_tokens.device_tokens()
            num_pending = len(pending_dst)
            if pending_dst == list(range(num_pending)) == pending_src:
                # Pending rows are the first n of both; prev may have more if a request finished.
                input_ids[:num_pending] = prev[:num_pending]
            else:
                dst = buffers.put("pending_dst", pending_dst, torch.int64)
                src = buffers.put("pending_src", pending_src, torch.int64)
                input_ids.index_copy_(0, dst, prev.index_select(0, src))
        if draft_dst:
            assert in_flight is not None
            dst = buffers.put("draft_dst", draft_dst, torch.int64)
            src = buffers.put("draft_src", draft_src, torch.int64)
            input_ids.index_copy_(0, dst, in_flight.draft_token_ids.flatten().index_select(0, src))
        if self.proposer is not None:
            num_draft_tokens = buffers.put("num_draft_tokens", sampling_drafts, torch.int64)
            draft_token_ids = None
            if self.rank == 0:  # it alone verifies; the drafts are the tokens after each sampling row's first
                is_draft = offsets < np.repeat(sampling_drafts, num_logits)
                draft_token_ids = input_ids[buffers.put("draft_index", logits_indices[is_draft] + 1, torch.int64)]
            self._spec_inputs = (draft_token_ids, num_draft_tokens)
            samples = np.zeros(num_rows, bool)
            samples[row_of[sampling]] = True
            next_token_ids = [
                -1 if row_samples else seq[end] for seq, end, row_samples in zip(batch, ends.tolist(), samples)
            ]
            self._draft_inputs = DraftInputs(
                input_ids=input_ids,
                positions=positions_t,
                max_position=int(positions[last_index[row_of[sampling]]].max(initial=0)),
                last_index=last_index,
                next_token_ids=np.array(next_token_ids, np.int64),
                sampling_rows=row_of[sampling],
                num_draft_tokens=sampling_drafts,
                block_table=table if block_tables is not None else None,
            )
        if moved:
            assert in_flight is not None and block_tables is not None
            self._move_back(context, positions_t, in_flight, token_rows, moved, moved_src, buffers)
        buffers.end()
        self._sampling_rows = sampling_rows
        return input_ids, positions_t, temperatures, context

    def _move_back(
        self,
        context: dict,
        positions: torch.Tensor,
        in_flight: InFlightDrafts,
        token_rows: np.ndarray,
        moved: list[int],
        moved_src: list[int],
        buffers: InputBuffers,
    ):
        """Move the moved rows back past the drafts the step in flight rejected, on the device: positions, slots and
        key lengths, as vLLM's update_num_computed_tokens_for_batch_change. The host keeps its lengths as bounds."""
        back = torch.zeros(len(context["context_lens"]), dtype=torch.int64, device=positions.device)
        rejected = in_flight.num_rejected.index_select(0, buffers.put("moved_src", moved_src, torch.int64))
        back.index_copy_(0, buffers.put("moved", moved, torch.int64), rejected)
        token_rows_t = buffers.put("token_rows", token_rows, torch.int64)
        positions -= back[token_rows_t]
        context_lens, block_size = context["context_lens"], self.block_size
        context_lens -= back.to(context_lens.dtype)
        context["cu_seqlens_k"][1:] = context_lens.cumsum(0)
        blocks = context["block_tables"][token_rows_t, positions // block_size].long()
        context["slot_mapping"].copy_(blocks * block_size + positions % block_size)

    def _cascade_prefix_len(self, table: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> int:
        """vLLM's _compute_cascade_attn_prefix_len: the pages every row shares, cut to the fewest cached tokens so no
        query falls inside them, in whole pages; 0 unless every kind of layer would cascade on it. vLLM counts
        blocks every request holds; this compares the step's own tables, so a request not scheduled cannot hide it."""
        if len(starts) < 2 or not self.cascade_layers:
            return 0
        num_pages = int(starts.min()) // self.block_size
        shared = (table[:, :num_pages] == table[0, :num_pages]).all(axis=0)
        common_prefix_len = (num_pages if shared.all() else int(shared.argmin())) * self.block_size
        if common_prefix_len and all(
            layer.use_cascade_attention(common_prefix_len, lens) for layer in self.cascade_layers
        ):
            return common_prefix_len
        return 0

    @staticmethod
    def decodes_first(seqs: list[Sequence], num_speculative_tokens: int = 0) -> list[Sequence]:
        """The batch's row order: decode rows first, one query each or, verifying drafts, decode_query_len, so a
        backend that splits a step slices them off, as vLLM's reorder_batch. Stable, and a no-op on a pure-decode
        step."""
        lens = np.array([seq.num_scheduled_tokens for seq in seqs], np.int64)
        query_len = decode_query_len(lens, num_speculative_tokens)
        return sorted(seqs, key=lambda seq: seq.num_scheduled_tokens != query_len)

    def _prev_row(self, seq: Sequence) -> int:
        """Where this sequence sampled in the step still in flight."""
        row = self._prev_rows.get(seq.seq_id) if self._prev_rows else None
        assert row is not None, "a pending token but no row in the launched step"
        return row

    @staticmethod
    def _cudagraph_mode(mode: str, layers: list[Attention]) -> str:
        """The mode these captures can serve. A full graph holds attention, so every layer's decode must be
        capturable; if one is not, full falls back to piecewise (attention runs eager, the rest is still captured)."""
        if mode in FULL_MODES and not all(layer.supports_full_cudagraph() for layer in layers):
            return "piecewise" if mode in PIECEWISE_MODES else "none"
        return mode

    def _step_kind(self, is_prefill: bool, num_tokens: int, cascade: bool = False) -> str:
        """How this step runs: "graph", "piecewise", or why no graph covers it. num_tokens is the batch size for decode.
        A cascade step takes no full graph, as in vLLM: the graphs hold the one-kernel decode."""
        if self.cudagraph_mode == "none":
            return "enforced"
        if not is_prefill and not cascade and self.cudagraph_mode in FULL_MODES and self.graph_bs:
            if num_tokens <= self.graph_bs[-1]:
                return "graph"
        if self.cudagraph_mode in PIECEWISE_MODES and self._piecewise_bucket(num_tokens):
            return "piecewise"
        # No graph covers the step, which runs compiled: "prefill" for a prefill or mixed step, "decode" for pure decode.
        return "prefill" if is_prefill else "decode"

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        self.step_kind = self._step_kind(is_prefill, input_ids.size(0), get_context().common_prefix_len > 0)
        if self.compile_backend is not None and not self.compile_backend.pieces:  # this call traces
            mark_dynamic_tokens(input_ids, positions)
        if self.eplb is not None:
            self.eplb.num_tokens.fill_(input_ids.size(0))  # a replay's padding rows record no load
        if self.step_kind == "graph":
            hidden_states = self._replay_full(input_ids, positions)
        elif self.step_kind == "piecewise":
            hidden_states = self._replay_piecewise(input_ids, positions)
        else:
            hidden_states = self.model(input_ids, positions)
        if self.proposer is not None:
            self._hidden_states = hidden_states  # the drafter's input, after sampling
        return self.model.compute_logits(hidden_states)

    def _replay_full(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """One graph for the whole model. Pure decode only: attention is inside it."""
        bs = input_ids.size(0)
        context = get_context()
        graph_bs = next(x for x in self.graph_bs if x >= bs)
        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
        for backend in self.attention_backends:  # e.g. FlashInfer re-plans the graph's decode
            backend.before_full_graph_replay(context, graph_bs)
        graph.replay()
        return graph_vars["outputs"][:bs]

    def _piecewise_bucket(self, num_tokens: int) -> int | None:
        """The bucket a step of this size replays in, or None. Shared so dispatch and replay agree."""
        bucket = next((size for size in self.piecewise_bs if size >= num_tokens), None)
        return None if bucket is None or num_tokens < self.piecewise_bs[0] else bucket

    def _replay_piecewise(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The compiled model padded to the step's bucket: pieces replay their graphs, attention runs eager on real rows."""
        num_tokens = input_ids.size(0)
        bucket = self._piecewise_bucket(num_tokens)
        buffers = self.piecewise_vars
        buffers["input_ids"][:num_tokens] = input_ids
        buffers["positions"][:num_tokens] = positions
        context = get_context()
        context.piecewise_size, context.num_actual_tokens = bucket, num_tokens
        return self.model(buffers["input_ids"][:bucket], buffers["positions"][:bucket])[:num_tokens]

    def run(self, seqs: list[Sequence]) -> SampledTokens | None:
        """Prepare, launch and sample. The tokens are not fetched here; the engine awaits them."""
        with record_function("prepare_batch"):
            input_ids, positions, temperatures, context = self.prepare_batch(seqs)
        with set_context(**context):
            with record_function("run_model"):
                logits = self.run_model(input_ids, positions, context["is_prefill"])
            with record_function("sample"):
                if self.proposer is not None:
                    verified = self._verify(logits, temperatures)
                    tokens = None
                else:
                    tokens = self.sampler(logits, temperatures) if self.rank == 0 else None
        if self.proposer is not None:
            with record_function("propose"):
                num_accepted = (verified >= 0).sum(dim=1) - 1
                next_token_ids = verified.gather(1, num_accepted.unsqueeze(1)).squeeze(1)
                drafts = self.proposer.propose(
                    self._draft_inputs, context, self._hidden_states, num_accepted, next_token_ids
                )
            self.in_flight_drafts = InFlightDrafts(next_token_ids, drafts, self._spec_inputs[1] - num_accepted)
            tokens = torch.cat([verified, drafts], dim=1) if self.rank == 0 else None
        if self.eplb is not None:
            self.eplb.step()  # every rank runs every step, so they rearrange together
        if tokens is None:
            return None
        pending = SampledTokens(tokens, self.device)
        self._prev_tokens = pending
        self._prev_rows = {seq.seq_id: i for i, seq in enumerate(self._sampling_rows)}
        return pending

    def _verify(self, logits: torch.Tensor | None, temperatures: torch.Tensor | None) -> torch.Tensor:
        """Each sampling row's kept drafts and next token, [rows, num_speculative_tokens + 1], -1 past the last; on
        every rank, as each runs the drafter on them."""
        assert self.proposer is not None
        if self.rank == 0:
            draft_token_ids, num_draft_tokens = self._spec_inputs
            assert logits is not None and draft_token_ids is not None
            tokens = self.rejection_sampler(
                logits, draft_token_ids, num_draft_tokens, temperatures, self.num_speculative_tokens
            )
        else:
            shape = (len(self._draft_inputs.sampling_rows), self.num_speculative_tokens + 1)
            tokens = torch.empty(shape, dtype=torch.int64, device=self.device)
        return self.proposer.broadcast(tokens)

    @torch.inference_mode()
    def capture_piecewise(self):
        """Run the compiled model once per bucket, largest first; each piece captures its graph as the run reaches it."""
        self.piecewise_bs = cudagraph_capture_sizes(self.config.max_num_seqs, self.config.max_num_batched_tokens)
        largest = self.piecewise_bs[-1]
        input_ids = torch.zeros(largest, dtype=torch.int64)
        positions = torch.zeros(largest, dtype=torch.int64)
        self.piecewise_vars = dict(input_ids=input_ids, positions=positions)
        for size in reversed(self.piecewise_bs):
            # Attention runs for real between the pieces, on one fresh prompt that writes no cache slot.
            seq = Sequence([0] * size)
            seq.num_scheduled_tokens = size
            _, _, _, context = self.prepare_batch([seq])
            context["slot_mapping"] = torch.full((size,), -1, dtype=torch.int32)
            with set_context(**context, piecewise_size=size):
                self.model(input_ids[:size], positions[:size])
                if self.proposer is not None:  # its first pass runs over the target's batch, so the same buckets
                    self.proposer.capture_piecewise(size)
            torch.cuda.synchronize()

    def _max_num_blocks(self) -> int:
        """The widest block table a step can hand a full graph: max_model_len, plus the slots the drafter writes
        past a row's last token, which can still be held from a step whose drafts were rejected."""
        return (self.config.max_model_len + self.num_speculative_tokens + self.block_size - 1) // self.block_size

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        sizes = cudagraph_capture_sizes(config.max_num_seqs, config.max_num_batched_tokens)
        self.graph_bs = [size for size in sizes if size <= config.max_num_seqs]  # a decode step has a row per token
        max_bs = self.graph_bs[-1]
        max_num_blocks = self._max_num_blocks()
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # FlashMLA bakes its tile schedule and split-KV workspace from context_lens at capture, so capture the
        # worst case: block 0 is valid, so a full block_tables of zeros holds max_model_len tokens per row. Every
        # replay refreshes context_lens/block_tables (see _replay_full), and the kernel gates on those lengths.
        context_lens.fill_(config.max_model_len)

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            with set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                full_graph_size=bs,
            ):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup, which also plans FlashInfer
                # The warmup scheduled MLA decode into the default pool; clear it so the capture reschedules
                # into the graph's own pool (else the graph bakes pointers freed with this context).
                get_context().mla_decode_metadata = None
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            self.graphs[bs] = graph
            torch.cuda.synchronize()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
