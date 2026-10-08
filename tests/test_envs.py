"""Environment variables: parsed in one place, and read when accessed rather than at import."""

import pytest

from lean_vllm import envs


def test_unset_variables_take_their_defaults(monkeypatch):
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)
    assert envs.LEAN_VLLM_DEVICE is None
    assert envs.LEAN_VLLM_ATTENTION_BACKEND is None
    assert envs.LEAN_VLLM_MOE_BACKEND is None
    assert envs.LEAN_PROFILE_DIR is None
    assert (envs.LEAN_PROFILE_SKIP, envs.LEAN_PROFILE_STEPS) == (200, 200)
    assert envs.LEAN_PROFILE_CUDA is False


def test_a_variable_is_read_at_access(monkeypatch):
    monkeypatch.setenv("LEAN_PROFILE_STEPS", "7")
    assert envs.LEAN_PROFILE_STEPS == 7


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", True),
        ("true", True),
        ("0", False),
        ("false", False),
        ("False", False),
        ("", False),
    ],
)
def test_booleans_parse_like_the_profiler_always_did(monkeypatch, value, expected):
    monkeypatch.setenv("LEAN_PROFILE_CUDA", value)
    assert envs.LEAN_PROFILE_CUDA is expected


def test_an_empty_string_counts_as_unset(monkeypatch):
    monkeypatch.setenv("LEAN_VLLM_DEVICE", "")
    assert envs.LEAN_VLLM_DEVICE is None


def test_an_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError, match="LEAN_NOT_A_VARIABLE"):
        _ = envs.LEAN_NOT_A_VARIABLE


def test_the_device_override_is_honoured(monkeypatch):
    from lean_vllm.utils.device import get_device

    monkeypatch.setenv("LEAN_VLLM_DEVICE", "cpu")
    assert get_device().type == "cpu"


def test_the_selector_reads_the_backend_variable(monkeypatch):
    import torch

    from lean_vllm.attention import LayerSpec
    from lean_vllm.attention.selector import get_attention_backend

    monkeypatch.setenv("LEAN_VLLM_ATTENTION_BACKEND", "nope")
    with pytest.raises(ValueError, match="unknown attention backend 'nope'"):
        get_attention_backend(LayerSpec(64, 4, 4, torch.float32))


def test_the_profiler_is_built_only_when_asked_for(monkeypatch, tmp_path):
    from lean_vllm.engine.llm_engine import _StepProfiler

    monkeypatch.delenv("LEAN_PROFILE_DIR", raising=False)
    assert _StepProfiler.from_env() is None
    monkeypatch.setenv("LEAN_PROFILE_DIR", str(tmp_path / "trace"))
    profiler = _StepProfiler.from_env()
    assert profiler.out_dir == str(tmp_path / "trace")
    assert (tmp_path / "trace").is_dir()
