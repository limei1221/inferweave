"""Config's defaults that depend on other fields, without a model on disk."""

import pytest
import torch

from lean_vllm import config as config_module
from lean_vllm.attention import LayerSpec
from lean_vllm.config import Config


class FakeHFConfig:
    max_position_embeddings = 4096


@pytest.fixture
def make_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module.AutoConfig, "from_pretrained", lambda path: FakeHFConfig())
    return lambda **kwargs: Config(str(tmp_path), **kwargs)


def test_async_scheduling_is_on_by_default(make_config):
    assert make_config().async_scheduling


def test_tensor_parallelism_turns_async_scheduling_off(make_config, caplog):
    """Ranks above zero never see the sampled tokens."""
    config = make_config(tensor_parallel_size=2)
    assert not config.async_scheduling
    assert "async_scheduling is off" in caplog.text


@pytest.mark.parametrize("architecture, supported", [("DeepseekV2ForCausalLM", True), ("Qwen3ForCausalLM", False)])
def test_expert_parallelism_needs_a_moe_model(make_config, monkeypatch, architecture, supported):
    monkeypatch.setattr(FakeHFConfig, "architectures", [architecture], raising=False)
    assert make_config().enable_expert_parallel is False  # off by default, as in vLLM
    if supported:
        assert make_config(enable_expert_parallel=True).enable_expert_parallel
    else:
        with pytest.raises(ValueError, match="enable_expert_parallel needs a MoE model"):
            make_config(enable_expert_parallel=True)


def test_an_mla_model_takes_the_page_size_of_its_decode_kernel(make_config, monkeypatch, caplog):
    from lean_vllm.attention import FlashMLABackend

    specs = []
    monkeypatch.setattr(config_module, "get_attention_backend", lambda spec: specs.append(spec) or FlashMLABackend)
    assert make_config().kvcache_block_size == 16  # not an MLA model

    # DeepSeek-V2-Lite's attention, so the spec is what its layers will ask for on each of two ranks.
    for name, value in dict(
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, num_attention_heads=16, dtype=torch.bfloat16
    ).items():
        monkeypatch.setattr(FakeHFConfig, name, value, raising=False)
    assert make_config(tensor_parallel_size=2).kvcache_block_size == 64
    assert specs == [LayerSpec(192, 8, 8, torch.bfloat16, latent_dim=576)]
    assert "kvcache_block_size is 64" in caplog.text


def test_the_rendezvous_port_is_picked_unless_given(make_config):
    """A fixed port would keep a second engine on the host from starting."""
    assert make_config().dist_port > 0
    assert make_config(dist_port=29500).dist_port == 29500


def test_eplb_needs_expert_parallelism_and_an_even_split(make_config, monkeypatch):
    monkeypatch.setattr(FakeHFConfig, "architectures", ["DeepseekV2ForCausalLM"], raising=False)
    monkeypatch.setattr(FakeHFConfig, "n_routed_experts", 64, raising=False)
    assert make_config().eplb is None
    with pytest.raises(ValueError, match="enable_eplb needs enable_expert_parallel"):
        make_config(enable_eplb=True)
    with pytest.raises(ValueError, match="eplb_config needs enable_eplb"):
        make_config(eplb_config='{"num_redundant_experts": 2}')
    with pytest.raises(ValueError, match="do not split over 4 ranks"):
        make_config(
            enable_expert_parallel=True,
            enable_eplb=True,
            tensor_parallel_size=4,
            eplb_config='{"num_redundant_experts": 2}',
        )
    with pytest.raises(ValueError, match="unknown eplb_config keys"):
        make_config(enable_expert_parallel=True, enable_eplb=True, eplb_config='{"window": 2}')

    config = make_config(
        enable_expert_parallel=True,
        enable_eplb=True,
        tensor_parallel_size=4,
        eplb_config='{"num_redundant_experts": 4}',
    )
    assert config.eplb.num_redundant_experts == 4 and config.eplb.step_interval == 3000  # vLLM's default
