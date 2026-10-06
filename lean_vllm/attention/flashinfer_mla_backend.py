import torch

from lean_vllm.attention.flashinfer_backend import FlashInferBackend, _host_cumulative
from lean_vllm.attention.mla_common import MLACommonBackend, prefill_backend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
except ImportError as e:  # the cuda extra installs it on Linux
    _IMPORT_ERROR = e


class FlashInferMLABackend(MLACommonBackend):
    """FlashInfer's MLA decode over latents, planned once per step, and the shared MLA prefill. sm80 and up.

    Not vLLM's FLASHINFER_MLA, whose trtllm-gen kernel runs on Blackwell only: this is FlashInfer's FA2/FA3 MLA
    kernel, for Ampere and Hopper, as SGLang uses it.
    """

    _wrappers: dict[tuple, object] = {}  # by layer shape, page size and graph size; each holds one plan at a time
    _graph_metadata: tuple[torch.Tensor, ...] | None = None  # what full graphs read, sized at the largest

    @staticmethod
    def get_name() -> str:
        return "flashinfer_mla"

    @staticmethod
    def is_available() -> bool:
        return _IMPORT_ERROR is None and FlashInferBackend.is_available() and prefill_backend() is not None

    @staticmethod
    def supports_full_cudagraph_mla_decode() -> bool:
        return True  # full graphs re-plan decode before each replay, as FlashInferBackend's do

    def mla_decode(self, q, latent_cache, v_dim, context: Context) -> torch.Tensor:
        return self._planned(q, latent_cache, v_dim, context).run(query=q, kv_cache=latent_cache)

    def _planned(self, q: torch.Tensor, latent_cache: torch.Tensor, v_dim: int, context: Context):
        """This step's decode wrapper, planned by the first layer to ask and reused by every layer alike."""
        graph_size = context.full_graph_size
        page_size = latent_cache.size(1)
        key = (
            "mla_decode",
            self.num_heads,
            v_dim,
            latent_cache.size(-1) - v_dim,
            self.scale,
            q.dtype,
            latent_cache.dtype,
            page_size,
            graph_size,
        )
        if context.attn_metadata is None:
            context.attn_metadata = {}
        if key in context.attn_metadata:
            return context.attn_metadata[key]
        wrapper = FlashInferMLABackend._wrappers.get(key)
        if wrapper is None:
            wrapper = FlashInferMLABackend._wrappers[key] = self._make_wrapper(q.device, context, graph_size)
        _plan(wrapper, key, self._metadata(context, page_size, graph_size))
        context.attn_metadata[key] = wrapper
        return wrapper

    @classmethod
    def before_full_graph_replay(cls, context: Context, batch_size: int) -> None:
        """Re-plan the wrappers the graph at batch_size captured, for this step's rows, as vLLM's builder does.
        Their metadata buffers are the graph's, so the replay reads the new plan."""
        metadata = {}
        for key, wrapper in cls._wrappers.items():
            if key[-1] == batch_size:
                page_size = key[-2]
                if page_size not in metadata:
                    metadata[page_size] = cls._metadata(context, page_size, batch_size)
                _plan(wrapper, key, metadata[page_size])

    @classmethod
    def _make_wrapper(cls, device: torch.device, context: Context, graph_size: int | None):
        workspace = FlashInferBackend._workspace_for(device)
        if graph_size is None:
            return BatchMLAPagedAttentionWrapper(workspace)
        # One per captured batch size, over slices of buffers shared by all, as FlashInferBackend's.
        if cls._graph_metadata is None:
            assert context.block_tables is not None
            rows, width = context.block_tables.shape  # graphs capture largest first, so this bounds the rest
            cls._graph_metadata = tuple(
                torch.zeros(n, dtype=torch.int32, device=device) for n in (rows + 1, rows + 1, rows * width, rows)
            )
        qo_indptr, kv_indptr, kv_indices, kv_lens = cls._graph_metadata
        assert graph_size <= kv_lens.numel(), "full graphs must capture their largest batch size first"
        return BatchMLAPagedAttentionWrapper(
            workspace,
            use_cuda_graph=True,
            qo_indptr=qo_indptr[: graph_size + 1],
            kv_indptr=kv_indptr[: graph_size + 1],
            kv_indices=kv_indices,
            kv_len_arr=kv_lens[:graph_size],
        )

    @staticmethod
    def _metadata(context: Context, page_size: int, num_rows: int | None = None) -> tuple[torch.Tensor, ...]:
        """FlashInfer's CSR plan: query and page offsets and key lengths on the host, the used pages on the device.
        num_rows pads it to a graph's batch size with rows of one key on page 0, so no row is empty."""
        cu_k = _host_cumulative(context.cu_seqlens_k_host, context.cu_seqlens_k, context.context_lens)
        kv_lens = cu_k[1:] - cu_k[:-1]
        kv_indptr, kv_indices, _ = FlashInferBackend._pages(context, page_size)
        if num_rows is not None:
            pad = num_rows - kv_lens.numel()
            kv_lens = torch.cat([kv_lens, kv_lens.new_ones(pad)])
            kv_indptr = torch.cat([kv_indptr, kv_indptr[-1] + torch.arange(1, pad + 1, dtype=kv_indptr.dtype)])
            kv_indices = torch.cat([kv_indices, kv_indices.new_zeros(pad)])
        qo_indptr = torch.arange(kv_lens.numel() + 1, dtype=torch.int32)  # one query per row
        return qo_indptr, kv_indptr, kv_indices, kv_lens


def _plan(wrapper, key: tuple, metadata: tuple[torch.Tensor, ...]) -> None:
    """Plan a decode wrapper from its key alone, so a replay hook re-plans with no layer at hand."""
    _, num_heads, v_dim, rope_dim, scale, q_dtype, kv_dtype, page_size, _ = key
    wrapper.plan(
        metadata=MLAPlanMetadata.csr(*metadata),
        num_heads=num_heads,
        head_dim_ckv=v_dim,
        head_dim_kpe=rope_dim,
        page_size=page_size,
        causal=False,
        sm_scale=scale,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
