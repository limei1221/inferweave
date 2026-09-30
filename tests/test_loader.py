"""load_model must refuse a checkpoint that leaves a parameter uninitialized."""

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from lean_vllm.utils.loader import load_model


class Tiny(nn.Module):

    def __init__(self, tie: bool = False):
        super().__init__()
        self.embed = nn.Parameter(torch.empty(4, 2))
        self.head = nn.Parameter(torch.empty(4, 2))
        if tie:
            self.head.data = self.embed.data    # as the models tie lm_head


def test_a_missing_weight_is_an_error(tmp_path):
    save_file({"embed": torch.ones(4, 2)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="no weights for 1 parameters: head"):
        load_model(Tiny(), str(tmp_path))


def test_a_tied_weight_needs_no_entry_of_its_own(tmp_path):
    save_file({"embed": torch.ones(4, 2)}, tmp_path / "model.safetensors")
    model = Tiny(tie=True)
    load_model(model, str(tmp_path))
    assert torch.equal(model.head, torch.ones(4, 2))
