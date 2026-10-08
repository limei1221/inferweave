import os
import re
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

# Routed experts are stored one by one, and load into one stacked parameter per projection.
EXPERT_WEIGHT = re.compile(r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
STACKED_EXPERT_PARAMS = {"gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "down_proj": "down_proj"}


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    skipped_weight_prefixes = getattr(model, "skipped_weight_prefixes", ())
    remap_weight_name = getattr(model, "remap_weight_name", None)  # a drafter's name for a checkpoint weight, or None
    loaded: set[str] = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for checkpoint_name in f.keys():
                if checkpoint_name.startswith(skipped_weight_prefixes):
                    continue
                weight_name = checkpoint_name if remap_weight_name is None else remap_weight_name(checkpoint_name)
                if weight_name is None:
                    continue
                if expert := EXPERT_WEIGHT.fullmatch(weight_name):
                    prefix, expert_id, proj = expert.groups()
                    param_name = f"{prefix}.{STACKED_EXPERT_PARAMS[proj]}"
                    param = model.get_parameter(param_name)
                    param.weight_loader(param, f.get_tensor(checkpoint_name), (int(expert_id), proj))  # type: ignore[attr-defined]
                    loaded.add(param_name)
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = param.weight_loader  # type: ignore[attr-defined]
                        weight_loader(param, f.get_tensor(checkpoint_name), shard_id)
                        loaded.add(param_name)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(checkpoint_name))
                    loaded.add(weight_name)
    check_loaded(model, loaded, path)
    for module in model.modules():  # vLLM's hook, for what is derived from the weights once
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()  # type: ignore[operator]


def check_loaded(model: nn.Module, loaded: set[str], path: str):
    """A parameter no weight reached would run on uninitialized memory. A tied one shares a loaded one's storage."""
    loaded_storage = {model.get_parameter(name).data_ptr() for name in loaded}
    missing = [
        name
        for name, param in model.named_parameters()
        if name not in loaded and param.data_ptr() not in loaded_storage
    ]
    if missing:
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise ValueError(f"{path} has no weights for {len(missing)} parameters: {shown}")
