from lean_vllm import envs
from lean_vllm.attention.abstract import AttentionBackend, LayerSpec
from lean_vllm.attention.flash_backend import FlashAttention3Backend
from lean_vllm.attention.flashinfer_backend import FlashInferBackend
from lean_vllm.attention.flashinfer_mla_backend import FlashInferMLABackend
from lean_vllm.attention.flashmla_backend import FlashMLABackend
from lean_vllm.attention.torch_backend import TorchAttention
from lean_vllm.attention.triton_mla_backend import TritonMLABackend

# In priority order, as vLLM's per-platform lists. The MLA backends serve MLA layers only, so a plain layer passes
# them by; as in vLLM, Triton MLA follows FlashMLA where FlashMLA is missing, and FlashInfer MLA is chosen by name.
# FA3 serves MLA layers too, by expanding latents even to decode. TorchAttention is last and serves any layer, so
# resolution cannot fail.
BACKENDS: tuple[type[AttentionBackend], ...] = (
    FlashMLABackend,
    TritonMLABackend,
    FlashInferMLABackend,
    FlashAttention3Backend,
    FlashInferBackend,
    TorchAttention,
)


def get_attention_backend(spec: LayerSpec, name: str | None = None) -> type[AttentionBackend]:
    """One layer's backend: explicit name, then $LEAN_VLLM_ATTENTION_BACKEND, then the first that can serve it."""
    name = name or envs.LEAN_VLLM_ATTENTION_BACKEND
    if name:
        by_name = {backend.get_name(): backend for backend in BACKENDS}
        if name not in by_name:
            raise ValueError(f"unknown attention backend {name!r}, expected one of {sorted(by_name)}")
        backend = by_name[name]
        reasons = _unsupported(backend, spec)
        if reasons:
            raise RuntimeError(
                f"attention backend {name!r} was requested but cannot serve {spec}: {'; '.join(reasons)}"
            )
        return backend
    return next(backend for backend in BACKENDS if not _unsupported(backend, spec))


def _unsupported(backend: type[AttentionBackend], spec: LayerSpec) -> list[str]:
    if not backend.is_available():
        return ["not available on this machine"]
    return backend.validate(spec)
