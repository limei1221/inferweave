"""Every environment variable lean-vLLM reads, parsed in one place and read at access."""

import os
from typing import Any, Callable


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() not in ("", "0", "false")


environment_variables: dict[str, Callable[[], Any]] = {
    # Forces a device such as "cpu" or "mps"; unset picks cuda, then mps, then cpu.
    "LEAN_VLLM_DEVICE": lambda: os.getenv("LEAN_VLLM_DEVICE") or None,
    # Forces an attention backend by name; unset picks the first available.
    "LEAN_VLLM_ATTENTION_BACKEND": lambda: os.getenv("LEAN_VLLM_ATTENTION_BACKEND") or None,
    # Forces "triton" or "torch" for the routed experts; unset takes Triton where it runs.
    "LEAN_VLLM_MOE_BACKEND": lambda: os.getenv("LEAN_VLLM_MOE_BACKEND") or None,
    # A folder of tuned MoE launch configs, searched before the shipped ones; as vLLM's VLLM_TUNED_CONFIG_FOLDER.
    "LEAN_VLLM_TUNED_CONFIG_FOLDER": lambda: os.getenv("LEAN_VLLM_TUNED_CONFIG_FOLDER") or None,
    # Enables the step-loop profiler, which writes its trace here.
    "LEAN_PROFILE_DIR": lambda: os.getenv("LEAN_PROFILE_DIR") or None,
    # Steps skipped, then steps captured.
    "LEAN_PROFILE_SKIP": lambda: int(os.getenv("LEAN_PROFILE_SKIP", "200")),
    "LEAN_PROFILE_STEPS": lambda: int(os.getenv("LEAN_PROFILE_STEPS", "200")),
    # Adds CUDA activity to the trace; clean only for offline generate.
    "LEAN_PROFILE_CUDA": lambda: _bool("LEAN_PROFILE_CUDA", False),
}


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables)
