import logging
import math
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.profiler import record_function

from lean_vllm.attention import AttentionBackend, get_attention_backend
from lean_vllm.config import Config, FULL_MODES, PIECEWISE_MODES
from lean_vllm.engine.sampled_tokens import SampledTokens
from lean_vllm.engine.sequence import Sequence
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
        mla = getattr(hf_config, "kv_lora_rank", None) is not None
        attention_backend = get_attention_backend(mla=mla)
        if rank == 0:
            logger.info("attention backend: %s", attention_backend.get_name())
        self.step_kind = "enforced"    # how the last step ran: see _step_kind
        self._prev_tokens: SampledTokens | None = None    # the step still in flight, if any
        self._prev_rows: dict[int, int] | None = None       # seq_id -> its row in those tokens
        model_cls = get_model_class(hf_config)
        self.enforce_eager = (config.enforce_eager or self.device.type != "cuda"
                              or not attention_backend.supports_cuda_graph() or not model_cls.supports_cuda_graph)
        mode = "none" if self.enforce_eager else config.cudagraph_mode
        self.cudagraph_mode = self._cudagraph_mode(mode, mla, attention_backend)
        if rank == 0 and self.cudagraph_mode != mode:
            logger.info("%s MLA decode cannot run in a full graph; cudagraph_mode is %r not %r",
                        attention_backend.get_name(), self.cudagraph_mode, mode)
        self.graph_bs: list[int] = []          # captured batch sizes, full graphs
        self.piecewise_bs: list[int] = []      # captured token counts, piecewise graphs
        self.graphs: dict = {}
        self.piecewise_graphs: dict = {}
        self.graph_pool = None                 # shared by both capture kinds
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        dist.init_process_group(dev.dist_backend(self.device), "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        if self.world_size > 1:
            # Calls go to the workers on the host, whatever device the default group runs on.
            self.call_group = dist.new_group(backend="gloo", timeout=CALL_TIMEOUT)
        dev.set_device(self.device, rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        self.model = model_cls(hf_config)
        register_layers(self.model)    # before warmup_model, which runs the op
        for module in self.model.modules():
            if isinstance(module, MLAAttention):
                # Warmup expands a step's worth of new latents, so a chunk this size fits what it measured.
                module.max_context_chunk = config.max_num_batched_tokens
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if self.cudagraph_mode in FULL_MODES:
            self.capture_cudagraph()
        if self.cudagraph_mode in PIECEWISE_MODES:
            self.capture_piecewise()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1 and rank > 0:
            self.loop()

    def exit(self):
        if self.world_size > 1:
            dist.barrier()
        if self.cudagraph_mode != "none":
            del self.graphs, self.piecewise_graphs, self.graph_pool
        dev.synchronize(self.device)
        dist.destroy_process_group()

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
        dev.empty_cache(self.device)

    def allocate_kv_cache(self):
        config = self.config
        # Each layer names its cache layout: keys and values per head, or one MLA latent.
        layers = [module for module in self.model.modules() if isinstance(module, Attention)]
        layer_shape = layers[0].kv_cache_shape(1, self.block_size)
        block_bytes = len(layers) * math.prod(layer_shape) * config.hf_config.dtype.itemsize
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = dev.kvcache_bytes(self.device, config) // block_bytes
        assert config.num_kvcache_blocks > 0, "no memory left for the kv cache"
        self.kv_cache = torch.empty(len(layers), *layers[0].kv_cache_shape(config.num_kvcache_blocks, self.block_size))
        for layer, cache in zip(layers, self.kv_cache):
            layer.bind_kv_cache(cache)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = dev.make_tensor(block_tables, torch.int32, self.device)
        return block_tables

    def prepare_batch(self, seqs: list[Sequence]):
        """One batch for any mix of prompt chunks and decode rows, and the context to run it in."""
        input_ids, positions, slot_mapping = [], [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q = max_seqlen_k = 0
        context_lens, logits_indices, temperatures = [], [], []
        pending_dst, pending_src, sampling_rows = [], [], []
        is_prefill = any(seq.is_prefill for seq in seqs)

        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            if seq.is_prefill:
                assert not seq.num_pending_tokens, "a prefill row carries a pending token"
                input_ids.extend(seq[start:end])
            else:
                if seq.num_pending_tokens:
                    # Sampled by a step still in flight; the device copy fixes it below.
                    pending_dst.append(len(input_ids))
                    pending_src.append(self._prev_row(seq))
                input_ids.append(seq.last_token)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seq.num_scheduled_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(seq.num_scheduled_tokens, max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)
            context_lens.append(end)
            if end == seq.num_planned_tokens:    # nothing left to prefill, so this row samples
                logits_indices.append(cu_seqlens_q[-1] - 1)
                sampling_rows.append(seq)
                if self.rank == 0:    # only the sampling rank owns sampling parameters
                    temperatures.append(seq.temperature)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        block_tables = self.prepare_block_tables(seqs) if any(seq.block_table for seq in seqs) else None
        context = dict(
            is_prefill=is_prefill,
            prefill_rows=[seq.is_prefill for seq in seqs],
            cu_seqlens_q=dev.make_tensor(cu_seqlens_q, torch.int32, self.device),
            cu_seqlens_k=dev.make_tensor(cu_seqlens_k, torch.int32, self.device),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q_host=cu_seqlens_q,
            cu_seqlens_k_host=cu_seqlens_k,
            # Equal cumulative lengths: no row reads cached keys, so the cache can be skipped.
            keys_are_new=cu_seqlens_k == cu_seqlens_q,
            slot_mapping=dev.make_tensor(slot_mapping, torch.int32, self.device),
            context_lens=dev.make_tensor(context_lens, torch.int32, self.device),
            block_tables=block_tables,
            # A pure-decode batch samples on every row, so the gather is skipped.
            logits_indices=dev.make_tensor(logits_indices, torch.int64, self.device) if is_prefill else None,
        )
        input_ids = dev.make_tensor(input_ids, torch.int64, self.device)
        if pending_dst:
            prev = self._prev_tokens.device_tokens()
            num_pending = len(pending_dst)
            if pending_dst == list(range(num_pending)) == pending_src:
                # Pending rows are the first n of both; prev may have more if a request finished.
                input_ids[:num_pending] = prev[:num_pending]
            else:
                dst = dev.make_tensor(pending_dst, torch.int64, self.device)
                src = dev.make_tensor(pending_src, torch.int64, self.device)
                input_ids.index_copy_(0, dst, prev.index_select(0, src))
        positions = dev.make_tensor(positions, torch.int64, self.device)
        all_greedy = all(temperature == 0 for temperature in temperatures)
        temperatures = None if all_greedy else dev.make_tensor(temperatures, torch.float32, self.device)
        self._sampling_rows = sampling_rows
        return input_ids, positions, temperatures, context

    def _prev_row(self, seq: Sequence) -> int:
        """Where this sequence sampled in the step still in flight."""
        row = self._prev_rows.get(seq.seq_id) if self._prev_rows else None
        assert row is not None, "a pending token but no row in the launched step"
        return row

    @staticmethod
    def _cudagraph_mode(mode: str, mla: bool, backend: type[AttentionBackend]) -> str:
        """The mode these captures can serve. A full graph holds attention, so MLA decode must both attend
        latents and be safe to replay; FlashMLA bakes a per-step schedule a replay cannot refresh, so it falls
        back to piecewise (decode runs eager, the rest of the model is still captured)."""
        full_safe = backend.supports_mla_decode() and backend.supports_full_cudagraph_mla_decode()
        if mla and mode in FULL_MODES and not full_safe:
            return "piecewise" if mode in PIECEWISE_MODES else "none"
        return mode

    def _step_kind(self, is_prefill: bool, num_tokens: int) -> str:
        """How this step runs: "graph", "piecewise", or why it must run eager. num_tokens is the batch size for decode."""
        if self.cudagraph_mode == "none":
            return "enforced"
        if not is_prefill and self.cudagraph_mode in FULL_MODES and self.graph_bs:
            if num_tokens <= self.graph_bs[-1]:
                return "graph"
        if self.cudagraph_mode in PIECEWISE_MODES and self._piecewise_bucket(num_tokens):
            return "piecewise"
        # "prefill" is a step past the piecewise grid; "oversized" a decode batch past the full-graph buckets.
        return "prefill" if is_prefill else "oversized"

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        self.step_kind = self._step_kind(is_prefill, input_ids.size(0))
        if self.step_kind == "graph":
            return self.model.compute_logits(self._replay_full(input_ids, positions))
        if self.step_kind == "piecewise":
            return self.model.compute_logits(self._replay_piecewise(input_ids, positions))
        return self.model.compute_logits(self.model(input_ids, positions))

    def _replay_full(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """One graph for the whole model. Pure decode only: attention is inside it."""
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return graph_vars["outputs"][:bs]

    def _piecewise_bucket(self, num_tokens: int) -> int | None:
        """The bucket a step of this size replays in, or None. Shared so dispatch and replay agree."""
        bucket = next((size for size in self.piecewise_bs if size >= num_tokens), None)
        return None if bucket is None or num_tokens < self.piecewise_bs[0] else bucket

    def _replay_piecewise(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """A graph per piece, with attention run eager on the real rows between them."""
        num_tokens = input_ids.size(0)
        bucket = self._piecewise_bucket(num_tokens)
        graphs, buffers = self.piecewise_graphs[bucket], self.piecewise_vars
        buffers["input_ids"][:num_tokens] = input_ids
        buffers["positions"][:num_tokens] = positions

        graphs["head"].replay()
        for layer, pre, post in zip(self.model.model.layers, graphs["pre"], graphs["post"]):
            pre.replay()
            # Real rows only: attention reads this step's sequence layout, which no graph can hold.
            attn_out = layer.self_attn.attn(*(buffer[:num_tokens] for buffer in buffers["attn_in"]))
            buffers["attn_out"][:num_tokens] = attn_out
            post.replay()
        graphs["tail"].replay()
        return buffers["output"][:num_tokens]

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
        """Capture the model either side of attention, one graph per piece per bucket."""
        hf_config = self.config.hf_config
        layers = self.model.model.layers
        self.piecewise_bs = self._piecewise_buckets()
        largest = self.piecewise_bs[-1]
        buffers = dict(
            input_ids=torch.zeros(largest, dtype=torch.int64),
            positions=torch.zeros(largest, dtype=torch.int64),
            hidden=torch.zeros(largest, hf_config.hidden_size),
            residual=torch.zeros(largest, hf_config.hidden_size),
            output=torch.zeros(largest, hf_config.hidden_size),
            attn_out=torch.zeros(layers[0].self_attn.attn.output_shape(largest)),
        )
        # What attention takes is the model's own: q, k and v for Qwen3, a query and a latent for MLA.
        *attn_inputs, _ = layers[0].pre_attention(buffers["positions"], buffers["hidden"], None)
        buffers["attn_in"] = [torch.zeros_like(tensor) for tensor in attn_inputs]
        self.piecewise_vars = buffers
        self.piecewise_graphs = {}

        def capture(run):
            """Warm up, then capture. The warmup pass is what allocates."""
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                run()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            torch.cuda.synchronize()
            return graph

        for size in reversed(self.piecewise_bs):
            def head(size=size):
                buffers["hidden"][:size].copy_(self.model.model.embed_tokens(buffers["input_ids"][:size]))

            def tail(size=size):
                normed, _ = self.model.model.norm(buffers["hidden"][:size], buffers["residual"][:size])
                buffers["output"][:size].copy_(normed)

            def pre(layer, first, size=size):
                # The first layer takes no residual in; its graph bakes that in.
                carried = None if first else buffers["residual"][:size]
                *attn_inputs, residual = layer.pre_attention(
                    buffers["positions"][:size], buffers["hidden"][:size], carried)
                for buffer, tensor in zip(buffers["attn_in"], attn_inputs):
                    buffer[:size].copy_(tensor)
                buffers["residual"][:size].copy_(residual)

            def post(layer, size=size):
                hidden, residual = layer.post_attention(buffers["attn_out"][:size], buffers["residual"][:size])
                buffers["hidden"][:size].copy_(hidden)
                buffers["residual"][:size].copy_(residual)

            self.piecewise_graphs[size] = {
                "head": capture(head),
                "pre": [capture(lambda layer=layer, first=i == 0: pre(layer, first)) for i, layer in enumerate(layers)],
                "post": [capture(lambda layer=layer: post(layer)) for layer in layers],
                "tail": capture(tail),
            }

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

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            with set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs]):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
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
