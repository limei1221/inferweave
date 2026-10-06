from transformers import PretrainedConfig

from lean_vllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from lean_vllm.models.qwen3 import Qwen3ForCausalLM

ModelClass = type[DeepseekV2ForCausalLM] | type[Qwen3ForCausalLM]

# Keyed by the architectures field of a checkpoint's config.json.
MODELS: dict[str, ModelClass] = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}


def get_model_class(hf_config: PretrainedConfig) -> ModelClass:
    architectures = getattr(hf_config, "architectures", None) or []
    for architecture in architectures:
        if architecture in MODELS:
            return MODELS[architecture]
    raise ValueError(f"unsupported architectures {architectures}, expected one of {sorted(MODELS)}")
