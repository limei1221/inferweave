"""Where each physical expert slot goes, from the load each logical expert saw: vLLM's DefaultEplbPolicy, after
DeepSeek's EPLB, on one node. Arrays are [layers, ...] throughout, numpy on the host."""

import numpy as np


def balanced_packing(weight: np.ndarray, num_packs: int) -> tuple[np.ndarray, np.ndarray]:
    """Items into equal-count packs, heaviest first into the lightest pack. Each item's pack and rank in it."""
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs
    if items_per_pack == 1:
        pack_index = np.tile(np.arange(num_items, dtype=np.int64), (num_layers, 1))
        return pack_index, np.zeros_like(pack_index)
    pack_index = np.full((num_layers, num_items), -1, dtype=np.int64)
    rank_in_pack = np.full((num_layers, num_items), -1, dtype=np.int64)
    for layer, order in enumerate(np.argsort(-weight, axis=-1, kind="stable")):
        pack_weights = np.zeros(num_packs)
        pack_items = np.zeros(num_packs, dtype=np.int64)
        for item in order:
            pack = int(np.argmin(pack_weights))
            pack_index[layer, item] = pack
            rank_in_pack[layer, item] = pack_items[pack]
            pack_weights[pack] += weight[layer, item]
            pack_items[pack] += 1
            if pack_items[pack] == items_per_pack:
                pack_weights[pack] = np.inf  # full
    return pack_index, rank_in_pack


def replicate_experts(weight: np.ndarray, num_physical: int) -> tuple[np.ndarray, np.ndarray]:
    """One slot per logical expert, then each spare slot to whichever has the most load per replica.

    The logical expert in each slot, and how many replicas each logical expert got.
    """
    num_layers, num_logical = weight.shape
    assert num_physical >= num_logical
    physical_to_logical = np.tile(np.arange(num_physical, dtype=np.int64), (num_layers, 1))
    replica_count = np.ones((num_layers, num_logical), dtype=np.int64)
    layers = np.arange(num_layers)
    for slot in range(num_logical, num_physical):
        busiest = np.argmax(weight / replica_count, axis=-1)
        physical_to_logical[:, slot] = busiest
        replica_count[layers, busiest] += 1
    return physical_to_logical, replica_count


def keep_slots(new: np.ndarray, old: np.ndarray, num_ranks: int) -> np.ndarray:
    """vLLM's preserve_intragpu_slots: an expert a rank keeps stays in its slot there, so it is not copied."""
    out = new.copy()
    per_rank = new.shape[1] // num_ranks
    for layer in range(new.shape[0]):
        for start in range(0, new.shape[1], per_rank):
            incoming = new[layer, start : start + per_rank].tolist()
            kept: list[int | None] = []
            for expert in old[layer, start : start + per_rank].tolist():
                if expert in incoming:
                    incoming.remove(expert)
                    kept.append(expert)
                else:
                    kept.append(None)
            out[layer, start : start + per_rank] = [incoming.pop(0) if e is None else e for e in kept]
    return out


def rebalance_experts(
    weight: np.ndarray,
    num_physical: int,
    num_ranks: int,
    old_physical_to_logical: np.ndarray | None = None,
) -> np.ndarray:
    """The logical expert for each physical slot, [layers, num_physical], slots in rank order.

    vLLM's global policy: replicate the busiest experts, then pack the replicas so each rank's load is as even
    as it can be. With the old placement, kept experts keep their slots.
    """
    physical_to_logical, replica_count = replicate_experts(weight, num_physical)
    load_per_replica = np.take_along_axis(weight / replica_count, physical_to_logical, axis=1)
    pack_index, rank_in_pack = balanced_packing(load_per_replica, num_ranks)
    slot = pack_index * (num_physical // num_ranks) + rank_in_pack  # each replica's new slot
    replica_in_slot = np.empty_like(slot)
    np.put_along_axis(replica_in_slot, slot, np.arange(num_physical)[None, :].repeat(len(slot), 0), axis=1)
    new = np.take_along_axis(physical_to_logical, replica_in_slot, axis=1)
    return new if old_physical_to_logical is None else keep_slots(new, old_physical_to_logical, num_ranks)
