import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from lean_vllm.eplb.policy import rebalance_experts
from lean_vllm.layers.moe import FusedMoE

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EplbConfig:
    """vLLM's --eplb-config, as JSON."""

    window_size: int = 1000  # steps of load a rearrangement looks back on
    step_interval: int = 3000  # steps between rearrangements
    num_redundant_experts: int = 0  # extra slots per layer, each a copy of a busy expert
    log_balancedness: bool = False  # log each step's balance across ranks; reads the load back every step

    def __post_init__(self):
        if self.window_size <= 0 or self.step_interval <= 0:
            raise ValueError("eplb_config window_size and step_interval must be positive")
        if self.num_redundant_experts < 0:
            raise ValueError("eplb_config num_redundant_experts must not be negative")

    @classmethod
    def parse(cls, value: "str | dict | EplbConfig") -> "EplbConfig":
        if isinstance(value, EplbConfig):
            return value
        fields = json.loads(value or "{}") if isinstance(value, str) else dict(value)
        unknown = set(fields) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown eplb_config keys {sorted(unknown)}")
        return cls(**fields)


def replica_maps(physical_to_logical: np.ndarray, num_logical: int, max_replicas: int) -> tuple[np.ndarray, ...]:
    """Each logical expert's slots, -1 past its replicas, [layers, logical, max_replicas]; and how many it has."""
    num_layers = len(physical_to_logical)
    logical_to_physical = np.full((num_layers, num_logical, max_replicas), -1, dtype=np.int64)
    replica_count = np.zeros((num_layers, num_logical), dtype=np.int64)
    for layer, row in enumerate(physical_to_logical):
        for slot, expert in enumerate(row):
            logical_to_physical[layer, expert, replica_count[layer, expert]] = slot
            replica_count[layer, expert] += 1
    return logical_to_physical, replica_count


def move_experts(
    weights: list[torch.Tensor],
    buffers: list[torch.Tensor],
    old: np.ndarray,
    new: np.ndarray,
    rank: int,
) -> int:
    """vLLM's rearrange_expert_weights_inplace for one layer: each new slot fills from this rank's old slots where
    it can, otherwise from a rank that held the expert, point to point. The slots this rank rewrote."""
    num_local = weights[0].size(0)
    mine = slice(rank * num_local, (rank + 1) * num_local)
    old_local, new_local = old[mine].tolist(), new[mine].tolist()
    ops = []
    for expert in sorted(set(new.tolist())):
        held_by = {int(r) for r in (old == expert).nonzero()[0] // num_local}
        holders = sorted(held_by)
        needers = sorted({int(r) for r in (new == expert).nonzero()[0] // num_local} - held_by)
        for i, needer in enumerate(needers):  # spread over the holders, the same way on every rank
            sender = holders[i % len(holders)]
            if sender == rank:
                slot = old_local.index(expert)
                ops += [dist.P2POp(dist.isend, w[slot], needer) for w in weights]
            elif needer == rank:
                slot = new_local.index(expert)
                ops += [dist.P2POp(dist.irecv, b[slot], sender) for b in buffers]
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()
    changed = [slot for slot in range(num_local) if new_local[slot] != old_local[slot]]
    for slot in changed:
        expert = new_local[slot]
        if expert in old_local:  # held here before: copied out of the old slot before any is overwritten
            source = old_local.index(expert)
            for w, b in zip(weights, buffers):
                b[slot].copy_(w[source])
        else:  # received, into its first slot here
            first = new_local.index(expert)
            for b in buffers:
                b[slot].copy_(b[first])
    for slot in changed:
        for w, b in zip(weights, buffers):
            w[slot].copy_(b[slot])
    return len(changed)


class EplbState:
    """vLLM's EplbState, synchronous and on one node. Owns the load and the slot maps of every EPLB layer, the
    layers hold views of them. step() runs after each forward on every rank, so they rearrange together."""

    def __init__(self, model: nn.Module, config: EplbConfig):
        self.config = config
        self.layers = [m for m in model.modules() if isinstance(m, FusedMoE) and m.enable_eplb]
        assert self.layers, "EPLB needs a MoE layer built with enable_eplb"
        first = self.layers[0]
        device = first.gate_up_proj.device
        num_layers, self.num_logical, self.num_physical = (
            len(self.layers),
            first.num_experts,
            first.num_physical_experts,
        )
        self.ep_rank, self.ep_size = first.ep_rank, first.ep_size
        assert self.num_physical % self.ep_size == 0, "EPLB needs the same number of slots on every rank"
        self.max_replicas = self.num_physical - self.num_logical + 1
        # As the weights loaded: logical e in slot e, then the redundant slots cycle through the experts.
        self.physical_to_logical = np.tile(np.arange(self.num_physical) % self.num_logical, (num_layers, 1))
        self.logical_to_physical_map = torch.empty(
            num_layers, self.num_logical, self.max_replicas, dtype=torch.int64, device=device
        )
        self.logical_replica_count = torch.empty(num_layers, self.num_logical, dtype=torch.int64, device=device)
        self.expert_load_pass = torch.zeros(num_layers, self.num_physical, dtype=torch.int32, device=device)
        self.expert_load_window = torch.zeros(
            config.window_size, num_layers, self.num_physical, dtype=torch.int32, device=device
        )
        # Rows a step really has; a graph replays padding past them. Set by the runner, all rows until then.
        self.num_tokens = torch.full((), torch.iinfo(torch.int64).max, dtype=torch.int64, device=device)
        for i, layer in enumerate(self.layers):
            layer.logical_to_physical_map = self.logical_to_physical_map[i]
            layer.logical_replica_count = self.logical_replica_count[i]
            layer.expert_load = self.expert_load_pass[i]
            layer.num_tokens = self.num_tokens
        self._commit_maps()
        # One layer's experts: where a rearrangement assembles a layer's new slots.
        self.buffers = [torch.empty_like(first.gate_up_proj), torch.empty_like(first.down_proj)]
        self.reset()

    def reset(self):
        """Forget the load so far, as warmup and capture have none worth keeping."""
        self.expert_load_pass.zero_()
        self.expert_load_window.zero_()
        self.window_step = 0
        self.steps = 0  # since the last rearrangement

    def _commit_maps(self):
        logical_to_physical, replica_count = replica_maps(self.physical_to_logical, self.num_logical, self.max_replicas)
        self.logical_to_physical_map.copy_(torch.from_numpy(logical_to_physical))
        self.logical_replica_count.copy_(torch.from_numpy(replica_count))

    def step(self):
        """After each forward: its load goes into the window, and every step_interval steps the experts move."""
        if self.config.log_balancedness:
            self._log_balancedness()
        self.expert_load_window[self.window_step].copy_(self.expert_load_pass)
        self.expert_load_pass.zero_()
        self.window_step = (self.window_step + 1) % self.config.window_size
        self.steps += 1
        if self.steps >= self.config.step_interval:
            self.steps = 0
            self.rearrange()

    def _log_balancedness(self):
        """vLLM's balancedness: mean over max of the tokens each rank's experts took, summed over layers."""
        if self.ep_rank != 0:
            return
        per_rank = self.expert_load_pass.view(len(self.layers), self.ep_size, -1).sum(-1).float()
        avg, peak = per_rank.mean(1).sum().item(), per_rank.max(1).values.sum().item()
        logger.info("EPLB: avg_tokens=%.1f max_tokens=%.0f balancedness=%.4f", avg, peak, avg / peak if peak else 0.0)

    @torch.inference_mode()
    def rearrange(self):
        """Move the experts to balance the window's load, as every rank computes from the same all-reduced load."""
        start = time.perf_counter()
        physical_to_logical = torch.from_numpy(self.physical_to_logical).to(self.expert_load_window.device)
        logical_load = torch.zeros(
            len(self.layers), self.num_logical, dtype=torch.int64, device=physical_to_logical.device
        )
        logical_load.scatter_add_(1, physical_to_logical, self.expert_load_window.sum(0, dtype=torch.int64))
        if self.ep_size > 1:
            dist.all_reduce(logical_load)
        new = rebalance_experts(logical_load.cpu().numpy(), self.num_physical, self.ep_size, self.physical_to_logical)
        moved = 0
        for layer, old_row, new_row in zip(self.layers, self.physical_to_logical, new):
            moved += move_experts(
                [layer.gate_up_proj.data, layer.down_proj.data], self.buffers, old_row, new_row, self.ep_rank
            )
        self.physical_to_logical = new
        self._commit_maps()
        if self.ep_rank == 0:
            logger.info(
                "EPLB: rearranged experts, %d slots rewritten on rank 0, in %.2f s", moved, time.perf_counter() - start
            )
