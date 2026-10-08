import torch

from lean_vllm import envs


def get_device() -> torch.device:
    if override := envs.LEAN_VLLM_DEVICE:
        device = torch.device(override)
        if device.type not in ("cpu", "cuda"):
            raise ValueError(f"unsupported device {override!r}; use cpu or cuda")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dist_backend(device: torch.device) -> str:
    return "nccl" if device.type == "cuda" else "gloo"


def set_device(device: torch.device, rank: int) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(rank)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def make_tensor(data, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # pinned staging only pays off, and only exists, on cuda
    if device.type == "cuda":
        return torch.tensor(data, dtype=dtype, pin_memory=True).cuda(non_blocking=True)
    return torch.tensor(data, dtype=dtype, device=device)


def kvcache_bytes(device: torch.device, config) -> int:
    """Bytes to spend on the KV cache, measured after warmup so its peak activations are set aside."""
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        used = total - free
        stats = torch.cuda.memory_stats()
        peak, current = stats["allocated_bytes.all.peak"], stats["allocated_bytes.all.current"]
        return int(total * config.gpu_memory_utilization) - used - peak + current
    # cpu shares memory with the OS, so take an explicit budget instead
    return int(config.kvcache_memory_gb * 2**30)
