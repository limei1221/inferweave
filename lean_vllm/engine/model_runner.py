import logging
import math
from collections import Counter
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import record_function

from lean_vllm.config import Config, FULL_MODES, PIECEWISE_MODES
from lean_vllm.engine.compilation import PiecewiseBackend, compile_piecewise, mark_dynamic_tokens
from lean_vllm.engine.input_buffers import InputBuffers
from lean_vllm.engine.sampled_tokens import SampledTokens
from lean_vllm.engine.sequence import Sequence
from lean_vllm.kv_transfer import KVConnectorMetadata, KVConnectorOutput, KVOutputAggregator, create_worker_connector
from lean_vllm.models import get_model_class
from lean_vllm.layers.attention import Attention, MLAAttention, register_layers
from lean_vllm.layers.sampler import Sampler
from lean_vllm.utils.context import set_context, get_context
from lean_vllm.utils.loader import load_model
from lean_vllm.utils import device as dev

logger = logging.getLogger(__name__)

# How long a TP worker may wait for its next call; gloo's 30-minute default would kill an idle server.
CALL_TIMEOUT = timedelta(days=365)

# Piecewise buckets: step sizes worth capturing, minimum gap, and maximum replay padding.
PIECEWISE_MIN_TOKENS = 64
PIECEWISE_MAX_TOKENS = 512
PIECEWISE_MIN_GAP = 16
PIECEWISE_MAX_PAD = 0.25


class ModelRunner:

    def __init__(self, config: Config, rank: int):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        Sequence.block_size = self.block_size    # spawned workers never run LLMEngine.__init__
        self.device = dev.get_device()
        self.step_kind = "enforced"    # how the last step ran: see _step_kind
        self._prev_tokens: SampledTokens | None = None    # the step still in flight, if any
        self._prev_rows: dict[int, int] | None = None       # seq_id -> its row in those tokens
        model_cls = get_model_class(hf_config)
        self.graph_bs: list[int] = []          # captured batch sizes, full graphs
        self.piecewise_bs: list[int] = []      # captured token counts, piecewise graphs
        self.graphs: dict = {}
        self.graph_pool = None                 # shared by both capture kinds
        self.compile_backend: PiecewiseBackend | None = None    # holds the pieces and their graphs
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        dist.init_process_group(
            dev.dist_backend(self.device), f"tcp://localhost:{config.dist_port}", world_size=self.world_size, rank=rank)
        if self.world_size > 1:
            # Calls go to the workers on the host, whatever device the default group runs on.
            self.call_group = dist.new_group(backend="gloo", timeout=CALL_TIMEOUT)
        dev.set_device(self.device, rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        model_kwargs = {"enable_expert_parallel": True} if config.enable_expert_parallel else {}
        self.model = model_cls(hf_config, **model_kwargs)
        register_layers(self.model)    # before warmup_model, which runs the op
        # Each layer chose its backend as it was built; graphs depend on all of them.
        layers = [module for module in self.model.modules() if isinstance(module, Attention)]
        if rank == 0:
            counts = Counter(layer.backend.get_name() for layer in layers)
            logger.info("attention backends: %s", ", ".join(f"{name} ({n} layers)" for name, n in counts.items()))
        self.attention_backends = list(dict.fromkeys(type(layer.backend) for layer in layers))    # replay hooks
        self.enforce_eager = (config.enforce_eager or self.device.type != "cuda" or not model_cls.supports_cuda_graph
                              or not all(layer.backend.supports_cuda_graph() for layer in layers))
        mode = "none" if self.enforce_eager else config.cudagraph_mode
        self.cudagraph_mode = self._cudagraph_mode(mode, layers)
        if rank == 0 and self.cudagraph_mode != mode:
            logger.info("some attention layer cannot run in a full graph; cudagraph_mode is %r not %r",
                        self.cudagraph_mode, mode)
        for module in self.model.modules():
            if isinstance(module, MLAAttention):
                # Warmup expands a step's worth of new latents, so a chunk this size fits what it measured.
                module.max_context_chunk = config.max_num_batched_tokens
        load_model(self.model, config.model)
        if self.cudagraph_mode != "none":
            self.compile_model()
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        self.kv_connector = None
        if config.kv_transfer is not None:
            self.kv_connector = create_worker_connector(config, rank)
            self.kv_connector.register_kv_caches(self.kv_cache)
            self.kv_aggregator = KVOutputAggregator(self.world_size)
        if self.cudagraph_mode in FULL_MODES:
            self.capture_cudagraph()
        if self.cudagraph_mode in PIECEWISE_MODES:
            self.capture_piecewise()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1 and rank > 0:
            self.loop()

    def exit(self):
        if self.kv_connector is not None:
            self.kv_connector.shutdown()
        if self.world_size > 1:
            dist.barrier()    # sync ranks
        if self.cudagraph_mode != "none":
            for piece in self.compile_backend.pieces:
                piece.graphs.clear()
            del self.graphs, self.graph_pool
        dev.synchronize(self.device)    # drain the device
        dist.destroy_process_group()    # drop the comms

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
        self.kv_connector.start_load_kv(metadata)
        output = self.kv_connector.get_finished()
        if self.world_size == 1:
            return output
        outputs = [None] * self.world_size if self.rank == 0 else None
        dist.gather_object(output, outputs, dst=0, group=self.call_group)
        return self.kv_aggregator.aggregate(outputs) if self.rank == 0 else None

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
        """As vLLM does: a step samples up to one row per sequence, far more than the warmup prefill, so measure that too."""
        num_rows = min(self.config.max_num_seqs, self.config.max_num_batched_tokens)
        # Random, as vLLM's: dummy hidden states could hold values that break the sampler.
        hidden_states = torch.rand(num_rows, self.config.hf_config.hidden_size)
        with set_context(False):    # no logits_indices, so every row samples
            logits = self.model.compute_logits(hidden_states)
        if self.rank != 0:
            return    # only rank 0 gathers logits and samples
        try:
            self.sampler(logits, torch.full((num_rows,), 0.5, dtype=torch.float32))    # non-greedy, the costlier path
        except torch.OutOfMemoryError as error:
            raise RuntimeError(
                f"out of memory warming up the sampler with {num_rows} rows; "
                "lower max_num_seqs or gpu_memory_utilization"
            ) from error

    def allocate_kv_cache(self):
        config = self.config
        # Each layer names its cache layout, which its backend may choose: keys and values per head, or one MLA latent.
        layers = [module for module in self.model.modules() if isinstance(module, Attention)]
        block_numel = sum(math.prod(layer.kv_cache_shape(1, self.block_size)) for layer in layers)
        block_bytes = block_numel * config.hf_config.dtype.itemsize
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = dev.kvcache_bytes(self.device, config) // block_bytes
        assert config.num_kvcache_blocks > 0, "no memory left for the kv cache"
        self.kv_cache = [torch.empty(layer.kv_cache_shape(config.num_kvcache_blocks, self.block_size)) for layer in layers]
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
        order = np.argsort(lens > 1, kind="stable")    # decodes_first, as indices
        batch = [seqs[i] for i in order]
        row_of = np.empty(num_rows, np.int64)
        row_of[order] = np.arange(num_rows)    # each scheduled sequence's row in the batch

        lens, starts = lens[order], starts[order]
        ends = starts + lens
        cu_q = np.zeros(num_rows + 1, np.int64)
        np.cumsum(lens, out=cu_q[1:])
        cu_k = np.zeros(num_rows + 1, np.int64)
        np.cumsum(ends, out=cu_k[1:])
        token_rows = np.repeat(np.arange(num_rows), lens)
        positions = starts[token_rows] + np.arange(cu_q[-1]) - cu_q[token_rows]

        input_ids, pending_dst, pending_src = [], [], []
        for seq, start, end in zip(batch, starts.tolist(), ends.tolist()):
            if seq.is_prefill:
                assert not seq.num_pending_tokens, "a prefill row carries a pending token"
                input_ids.extend(seq[start:end])
            else:
                if seq.num_pending_tokens:
                    # Sampled by a step still in flight; the device copy fixes it below.
                    pending_dst.append(len(input_ids))
                    pending_src.append(self._prev_row(seq))
                input_ids.append(seq.last_token)

        # A row with nothing left to prefill samples its last token.
        sampling = np.flatnonzero(starts[row_of] + lens[row_of] == planned)    # in scheduler order
        logits_indices = cu_q[row_of[sampling] + 1] - 1
        sampling_rows = [seqs[i] for i in sampling.tolist()]
        # Only the sampling rank owns sampling parameters.
        temperatures = [seq.temperature for seq in sampling_rows] if self.rank == 0 else []

        block_tables = slot_mapping = None
        tables = [seq.block_table for seq in batch]
        if any(tables):
            table = np.full((num_rows, max(map(len, tables))), -1, np.int32)
            for i, row in enumerate(tables):
                table[i, :len(row)] = row
            # Each token's slot is its block's, plus its offset in it. Rows with no blocks (warmup) store nothing.
            keep = np.fromiter(map(bool, tables), bool, num_rows)[token_rows]
            kept = positions[keep]
            blocks = table[token_rows[keep], kept // block_size].astype(np.int64)
            slot_mapping = blocks * block_size + kept % block_size
            block_tables = buffers.put("block_tables", table, torch.int32)
        is_prefill = any(seq.is_prefill for seq in seqs)
        cu_seqlens_q, cu_seqlens_k = cu_q.tolist(), cu_k.tolist()
        context = dict(
            is_prefill=is_prefill,
            cu_seqlens_q=buffers.put("cu_seqlens_q", cu_q, torch.int32),
            cu_seqlens_k=buffers.put("cu_seqlens_k", cu_k, torch.int32),
            max_seqlen_q=int(lens.max()),
            max_seqlen_k=int(ends.max()),
            cu_seqlens_q_host=cu_seqlens_q,
            cu_seqlens_k_host=cu_seqlens_k,
            # No row reads cached keys, so the cache can be skipped.
            keys_are_new=not starts.any(),
            slot_mapping=buffers.put("slot_mapping", [] if slot_mapping is None else slot_mapping, torch.int32),
            context_lens=buffers.put("context_lens", ends, torch.int32),
            block_tables=block_tables,
            # A pure-decode batch samples on every row, so the gather is skipped.
            logits_indices=buffers.put("logits_indices", logits_indices, torch.int64) if is_prefill else None,
        )
        input_ids = buffers.put("input_ids", np.array(input_ids, np.int64), torch.int64)
        positions = buffers.put("positions", positions, torch.int64)
        all_greedy = all(temperature == 0 for temperature in temperatures)
        temperatures = None if all_greedy else buffers.put("temperatures", temperatures, torch.float32)
        if pending_dst:
            prev = self._prev_tokens.device_tokens()
            num_pending = len(pending_dst)
            if pending_dst == list(range(num_pending)) == pending_src:
                # Pending rows are the first n of both; prev may have more if a request finished.
                input_ids[:num_pending] = prev[:num_pending]
            else:
                dst = buffers.put("pending_dst", pending_dst, torch.int64)
                src = buffers.put("pending_src", pending_src, torch.int64)
                input_ids.index_copy_(0, dst, prev.index_select(0, src))
        buffers.end()
        self._sampling_rows = sampling_rows
        return input_ids, positions, temperatures, context

    @staticmethod
    def decodes_first(seqs: list[Sequence]) -> list[Sequence]:
        """The batch's row order: one-query rows first, so a backend that splits a step slices them off, as vLLM's
        reorder_batch. Stable, and a no-op on a pure-decode step."""
        return sorted(seqs, key=lambda seq: seq.num_scheduled_tokens > 1)

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

    def _step_kind(self, is_prefill: bool, num_tokens: int) -> str:
        """How this step runs: "graph", "piecewise", or why no graph covers it. num_tokens is the batch size for decode."""
        if self.cudagraph_mode == "none":
            return "enforced"
        if not is_prefill and self.cudagraph_mode in FULL_MODES and self.graph_bs:
            if num_tokens <= self.graph_bs[-1]:
                return "graph"
        if self.cudagraph_mode in PIECEWISE_MODES and self._piecewise_bucket(num_tokens):
            return "piecewise"
        # No graph covers the step, which runs compiled: "prefill" for a prefill or mixed step, "decode" for pure decode.
        return "prefill" if is_prefill else "decode"

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        self.step_kind = self._step_kind(is_prefill, input_ids.size(0))
        if self.compile_backend is not None and not self.compile_backend.pieces:    # this call traces
            mark_dynamic_tokens(input_ids, positions)
        if self.step_kind == "graph":
            return self.model.compute_logits(self._replay_full(input_ids, positions))
        if self.step_kind == "piecewise":
            return self.model.compute_logits(self._replay_piecewise(input_ids, positions))
        return self.model.compute_logits(self.model(input_ids, positions))

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
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        for backend in self.attention_backends:    # e.g. FlashInfer re-plans the graph's decode
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
                tokens = self.sampler(logits, temperatures) if self.rank == 0 else None
        if tokens is None:
            return None
        pending = SampledTokens(tokens, self.device)
        self._prev_tokens = pending
        self._prev_rows = {seq.seq_id: i for i, seq in enumerate(self._sampling_rows)}
        return pending

    def _piecewise_buckets(self) -> list[int]:
        """Token counts to capture at: small steps only, where launch overhead rivals compute, none padded past a quarter."""
        top = min(PIECEWISE_MAX_TOKENS, self.config.max_num_batched_tokens)
        sizes, size = [], PIECEWISE_MIN_TOKENS
        while size < top:
            sizes.append(size)
            gap = max(int(size * PIECEWISE_MAX_PAD), PIECEWISE_MIN_GAP)
            size += 1 << (gap.bit_length() - 1)    # a power of two, so sizes stay round
        return sorted(set(sizes) | {top})

    @torch.inference_mode()
    def capture_piecewise(self):
        """Run the compiled model once per bucket, largest first; each piece captures its graph as the run reaches it."""
        self.piecewise_bs = self._piecewise_buckets()
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
            torch.cuda.synchronize()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # FlashMLA bakes its tile schedule and split-KV workspace from context_lens at capture, so capture the
        # worst case: block 0 is valid, so a full block_tables of zeros holds max_model_len tokens per row. Every
        # replay refreshes context_lens/block_tables (see _replay_full), and the kernel gates on those lengths.
        context_lens.fill_(config.max_model_len)

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            with set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                             block_tables=block_tables[:bs], full_graph_size=bs):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup, which also plans FlashInfer
                # The warmup scheduled MLA decode into the default pool; clear it so the capture reschedules
                # into the graph's own pool (else the graph bakes pointers freed with this context).
                get_context().mla_decode_metadata = None
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
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
