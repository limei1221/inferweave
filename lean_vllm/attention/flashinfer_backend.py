import torch

from lean_vllm.attention import triton_cache
from lean_vllm.attention.abstract import AttentionBackend
from lean_vllm.utils.context import Context

_IMPORT_ERROR: ImportError | None = None
try:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
except ImportError as e:    # the cuda extra installs it on Linux
    _IMPORT_ERROR = e

# Scratch for split-KV partial results, shared by every wrapper as vLLM shares one.
WORKSPACE_BYTES = 256 * 1024 * 1024


class FlashInferBackend(AttentionBackend):
    """FlashInfer's paged prefill and decode, planned once per step, with the Triton KV-cache scatter. sm80 and up."""

    supported_kinds = ("decoder",)    # no varlen_with_lse, which MLA layers need
    _workspace: torch.Tensor | None = None
    _wrappers: dict[tuple, object] = {}    # by kernel and layer shape; each holds one step's plan at a time

    @staticmethod
    def get_name() -> str:
        return "flashinfer"

    @staticmethod
    def is_available() -> bool:
        return (_IMPORT_ERROR is None and triton_cache._IMPORT_ERROR is None and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 8)

    @staticmethod
    def supports_cuda_graph() -> bool:
        return True    # piecewise, where attention runs eager between the pieces

    @staticmethod
    def supports_full_cudagraph() -> bool:
        return False    # decode is planned on the host each step, and replay has no hook to re-plan

    @staticmethod
    def split_decodes() -> bool:
        return True

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        return head_size in (64, 128, 256)    # as vLLM's FlashInfer backend

    def store_kvcache(self, key, value, k_cache, v_cache, slot_mapping) -> None:
        triton_cache.store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    def prefill(self, q, k, v, k_cache, v_cache, context: Context) -> torch.Tensor:
        if context.keys_are_new or context.block_tables is None:
            # k and v hold every key this batch attends (cold prompts), so skip the pages, as FA3 does.
            return self._planned("ragged", q, k.dtype, context).run(q, k, v)
        return self._planned("paged", q, k_cache.dtype, context, k_cache.size(1)).run(q, (k_cache, v_cache))

    def decode(self, q, k_cache, v_cache, context: Context) -> torch.Tensor:
        return self._planned("decode", q, k_cache.dtype, context, k_cache.size(1)).run(q, (k_cache, v_cache))

    def _planned(self, kind: str, q: torch.Tensor, kv_dtype: torch.dtype, context: Context, page_size: int = 0):
        """This step's wrapper for kind, planned by the first layer to ask and reused by every layer alike."""
        key = (kind, self.num_heads, self.num_kv_heads, self.head_dim, self.scale, q.dtype, kv_dtype, page_size)
        if context.attn_metadata is None:
            context.attn_metadata = {}
        if key in context.attn_metadata:
            return context.attn_metadata[key]
        wrapper = FlashInferBackend._wrappers.get(key)
        if wrapper is None:
            wrapper = FlashInferBackend._wrappers[key] = self._make_wrapper(kind, q.device)
        shape = (self.num_heads, self.num_kv_heads, self.head_dim)
        options = dict(sm_scale=self.scale, q_data_type=q.dtype, kv_data_type=kv_dtype)
        if kind == "decode":
            indptr, indices, last_page_len = self._pages(context, page_size)
            wrapper.plan(indptr, indices, last_page_len, *shape, page_size, **options)
        else:
            qo_indptr = _host_cumulative(context.cu_seqlens_q_host, context.cu_seqlens_q)
            if kind == "ragged":
                kv_indptr = _host_cumulative(context.cu_seqlens_k_host, context.cu_seqlens_k)
                wrapper.plan(qo_indptr, kv_indptr, *shape, causal=True, **options)
            else:    # FlashInfer's causal mask is bottom-right aligned, as the contract asks
                indptr, indices, last_page_len = self._pages(context, page_size)
                wrapper.plan(qo_indptr, indptr, indices, last_page_len, *shape, page_size, causal=True, **options)
        context.attn_metadata[key] = wrapper
        return wrapper

    def _make_wrapper(self, kind: str, device: torch.device):
        if FlashInferBackend._workspace is None:
            # Zeroed, as FlashInfer requires on its first use.
            FlashInferBackend._workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        workspace = FlashInferBackend._workspace
        if kind == "ragged":
            return BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        if kind == "paged":
            return BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        # Wide query groups decode on tensor cores, as vLLM chose; the CUDA-core kernel takes few group sizes.
        use_tensor_cores = self.num_heads // self.num_kv_heads > 4
        return BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=use_tensor_cores)

    @staticmethod
    def _pages(context: Context, page_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FlashInfer's page table: indptr and last-page lengths on the host, each row's used pages packed on the device."""
        cu_k = _host_cumulative(context.cu_seqlens_k_host, context.cu_seqlens_k, context.context_lens)
        kv_lens = cu_k[1:] - cu_k[:-1]
        num_pages = (kv_lens + page_size - 1) // page_size
        indptr = torch.zeros_like(cu_k)
        indptr[1:] = num_pages.cumsum(0)
        rows = torch.repeat_interleave(num_pages)
        pages = torch.arange(rows.numel()) - indptr[rows]
        block_tables = context.block_tables
        flat = (rows * block_tables.size(1) + pages).to(block_tables.device, non_blocking=True)
        indices = block_tables.flatten()[flat]
        return indptr, indices, kv_lens - (num_pages - 1) * page_size


def _host_cumulative(host: list[int] | None, device: torch.Tensor | None, lens: torch.Tensor | None = None):
    """Cumulative lengths as a host int32 tensor. A hand-built context may lack the host copy, so read the device."""
    if host is None:
        if device is None:    # a decode context carries its lengths alone
            device = torch.nn.functional.pad(lens.cumsum(0), (1, 0))
        host = device.tolist()
    return torch.tensor(host, dtype=torch.int32)
