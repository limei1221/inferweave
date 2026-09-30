import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# Routed experts are stored one by one, and load into one stacked parameter per projection.
EXPERT_WEIGHT = re.compile(r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
STACKED_EXPERT_PARAMS = {"gate_proj": "gate_up_proj", "up_proj": "gate_up_proj", "down_proj": "down_proj"}


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    loaded: set[str] = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if expert := EXPERT_WEIGHT.fullmatch(weight_name):
                    prefix, expert_id, proj = expert.groups()
                    param_name = f"{prefix}.{STACKED_EXPERT_PARAMS[proj]}"
                    param = model.get_parameter(param_name)
                    param.weight_loader(param, f.get_tensor(weight_name), (int(expert_id), proj))
                    loaded.add(param_name)
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded.add(param_name)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded.add(weight_name)
    check_loaded(model, loaded, path)


def check_loaded(model: nn.Module, loaded: set[str], path: str):
    """A parameter no weight reached would run on uninitialized memory. A tied one shares a loaded one's storage."""
    loaded_storage = {model.get_parameter(name).data_ptr() for name in loaded}
    missing = [
        name for name, param in model.named_parameters()
        if name not in loaded and param.data_ptr() not in loaded_storage
    ]
    if missing:
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise ValueError(f"{path} has no weights for {len(missing)} parameters: {shown}")
