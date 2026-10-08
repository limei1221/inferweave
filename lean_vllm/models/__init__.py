from transformers import PretrainedConfig

from lean_vllm.models.deepseek_mtp import DeepSeekMTP
from lean_vllm.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
from lean_vllm.models.qwen3 import Qwen3ForCausalLM

ModelClass = type[DeepseekV2ForCausalLM] | type[Qwen3ForCausalLM]

# Keyed by the architectures field of a checkpoint's config.json.
MODELS: dict[str, ModelClass] = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}

# The checkpoint's own multi-token prediction layers, run as a drafter, by the same field.
DRAFTERS: dict[str, type[DeepSeekMTP]] = {
    "DeepseekV3ForCausalLM": DeepSeekMTP,
}


def get_model_class(hf_config: PretrainedConfig) -> ModelClass:
    architectures = getattr(hf_config, "architectures", None) or []
    for architecture in architectures:
        if architecture in MODELS:
            return MODELS[architecture]
    raise ValueError(f"unsupported architectures {architectures}, expected one of {sorted(MODELS)}")


def get_drafter_class(hf_config: PretrainedConfig) -> type[DeepSeekMTP]:
    architectures = getattr(hf_config, "architectures", None) or []
    drafter = next((DRAFTERS[a] for a in architectures if a in DRAFTERS), None)
    if drafter is None or not getattr(hf_config, "num_nextn_predict_layers", None):
        raise ValueError(f"MTP needs a checkpoint with multi-token prediction layers, and {architectures} has none")
    return drafter
