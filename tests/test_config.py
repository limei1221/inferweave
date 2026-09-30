"""Config's defaults that depend on other fields, without a model on disk."""

import pytest

from lean_vllm import config as config_module
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


def test_an_mla_model_takes_the_page_size_of_its_decode_kernel(make_config, monkeypatch, caplog):
    from lean_vllm.attention import FlashMLABackend
    monkeypatch.setattr(config_module, "get_attention_backend", lambda mla: FlashMLABackend)
    assert make_config().kvcache_block_size == 16    # not an MLA model

    monkeypatch.setattr(FakeHFConfig, "kv_lora_rank", 512, raising=False)
    assert make_config().kvcache_block_size == 64
    assert "kvcache_block_size is 64" in caplog.text


def test_the_rendezvous_port_is_picked_unless_given(make_config):
    """A fixed port would keep a second engine on the host from starting."""
    assert make_config().dist_port > 0
    assert make_config(dist_port=29500).dist_port == 29500
