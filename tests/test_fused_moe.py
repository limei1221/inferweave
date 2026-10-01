"""The Triton MoE's blocking, checked on the CPU against the grouped_mm path.

`blocked_moe` writes the kernel's indexing out in torch, so only the `tl.dot` arithmetic needs a GPU.
"""

import pytest
import torch
import torch.distributed as dist

from lean_vllm.layers.fused_moe import align_blocks, fused_experts, use_triton
from lean_vllm.layers.moe import FusedMoE, determine_expert_map, silu_and_mul, torch_experts

HIDDEN, INTERMEDIATE = 32, 16
NUM_EXPERTS, TOP_K, TOKENS = 8, 3, 20
BLOCK_M = 16    # the kernel's smallest row block, so the padding is exercised at this size


@pytest.fixture(scope="module")
def process_group():
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture
def moe(process_group):
    torch.manual_seed(0)
    layer = FusedMoE(NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE)
    with torch.inference_mode():
        for param in layer.parameters():
            param.normal_(0, 0.1)    # the weights are torch.empty until a checkpoint lands
    return layer


@pytest.fixture
def batch():
    """A routing no expert dominates, and none is guaranteed a turn: empty runs are the edge case."""
    torch.manual_seed(1)
    x = torch.randn(TOKENS, HIDDEN)
    topk_weights, topk_ids = torch.rand(TOKENS, NUM_EXPERTS).topk(TOP_K, dim=-1)
    return x, topk_weights / topk_weights.sum(dim=-1, keepdim=True), topk_ids


def blocked_moe(gate_up_proj, down_proj, x, topk_weights, topk_ids, block_m: int, expert_map=None) -> torch.Tensor:
    """What the kernel computes, in torch: a block reads one expert, a row reads one pair."""
    num_experts = gate_up_proj.size(0) if expert_map is None else expert_map.numel()
    sorted_pairs, block_experts, num_rows = align_blocks(topk_ids, num_experts, block_m)
    if expert_map is not None:
        block_experts = expert_map[block_experts]
    num_pairs, weights, top_k = topk_ids.numel(), topk_weights.flatten(), topk_ids.size(1)
    h = torch.empty(num_pairs, gate_up_proj.size(1))
    out = torch.empty(num_pairs, x.size(1))
    for gate_up in (True, False):
        # h is one row per pair already, so the second gemm reads it without dividing.
        a, b, c, per = (x, gate_up_proj, h, top_k) if gate_up else (silu_and_mul(h), down_proj, out, 1)
        for block, expert in enumerate(block_experts.tolist()):
            if block * block_m >= num_rows: break
            pairs = sorted_pairs[block * block_m:(block + 1) * block_m]
            pairs = pairs[pairs < num_pairs].long()    # the mask the kernel applies to an overhanging block
            if expert == -1:    # another rank's expert
                c[pairs] = 0
                continue
            acc = a[pairs // per] @ b[expert].T
            c[pairs] = acc if gate_up else acc * weights[pairs].unsqueeze(1)
    return out.view(-1, top_k, x.size(1)).sum(dim=1)


def test_the_blocking_holds_every_pair_exactly_once(batch):
    _, _, topk_ids = batch
    sorted_pairs, _, _ = align_blocks(topk_ids, NUM_EXPERTS, BLOCK_M)
    held = sorted_pairs[sorted_pairs < topk_ids.numel()]
    assert sorted(held.tolist()) == list(range(topk_ids.numel()))


def test_a_block_reads_one_expert(batch):
    """The whole point of the padding: no block spans two experts' weights."""
    _, _, topk_ids = batch
    sorted_pairs, block_experts, _ = align_blocks(topk_ids, NUM_EXPERTS, BLOCK_M)
    pairs_expert = topk_ids.flatten()
    for block, expert in enumerate(block_experts.tolist()):
        held = sorted_pairs[block * BLOCK_M:(block + 1) * BLOCK_M]
        held = held[held < topk_ids.numel()].long()
        assert (pairs_expert[held] == expert).all()


def test_the_row_count_covers_each_padded_run(batch):
    _, _, topk_ids = batch
    _, _, num_rows = align_blocks(topk_ids, NUM_EXPERTS, BLOCK_M)
    counts = torch.bincount(topk_ids.flatten(), minlength=NUM_EXPERTS)
    assert num_rows == sum((count + BLOCK_M - 1) // BLOCK_M * BLOCK_M for count in counts.tolist())


def test_an_expert_with_no_tokens_owns_no_block():
    """An empty run is padded to nothing, so the next expert must start on the same block."""
    topk_ids = torch.zeros(4, 1, dtype=torch.long)    # every pair on expert 0
    topk_ids[:2] = NUM_EXPERTS - 1
    sorted_pairs, block_experts, num_rows = align_blocks(topk_ids, NUM_EXPERTS, BLOCK_M)
    assert num_rows == 2 * BLOCK_M
    assert block_experts[:2].tolist() == [0, NUM_EXPERTS - 1]
    assert (sorted_pairs[:2] < 4).all() and (sorted_pairs[2:BLOCK_M] == 4).all()


@pytest.mark.parametrize("block_m", [16, 64])
def test_the_blocked_path_matches_grouped_mm(moe, batch, block_m):
    """The kernel's indexing against the reference: a wrong gather, expert or scatter shows up here."""
    x, topk_weights, topk_ids = batch
    with torch.inference_mode():
        got = blocked_moe(moe.gate_up_proj, moe.down_proj, x, topk_weights, topk_ids, block_m)
        want = torch_experts(x, moe.gate_up_proj, moe.down_proj, topk_weights, topk_ids)
    torch.testing.assert_close(got, want)


requires_triton_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or use_triton.__globals__["_IMPORT_ERROR"] is not None,
    reason="the Triton MoE kernel needs a CUDA device and a Triton build",
)


@pytest.mark.parametrize("ep_size", [2, 3])    # 3 does not divide 8 experts
def test_placement_matches_vllm_linear(ep_size):
    """Contiguous runs, the remainder on the first ranks, and every expert on exactly one rank."""
    maps = [determine_expert_map(ep_size, rank, NUM_EXPERTS) for rank in range(ep_size)]
    counts = [count for count, _ in maps]
    assert counts == [NUM_EXPERTS // ep_size + (rank < NUM_EXPERTS % ep_size) for rank in range(ep_size)]
    owners = torch.stack([expert_map >= 0 for _, expert_map in maps]).int()
    assert owners.sum(0).tolist() == [1] * NUM_EXPERTS
    owned = [torch.nonzero(expert_map >= 0).flatten().tolist() for _, expert_map in maps]
    assert sum(owned, []) == list(range(NUM_EXPERTS))    # rank order is expert order
    for (count, expert_map), experts in zip(maps, owned):
        assert expert_map[experts].tolist() == list(range(count))


def test_one_rank_needs_no_map():
    assert determine_expert_map(1, 0, NUM_EXPERTS) == (NUM_EXPERTS, None)


def shards(moe: FusedMoE, ep_size: int):
    """Each rank's map and the experts it would hold, cut from one full layer."""
    for rank in range(ep_size):
        _, expert_map = determine_expert_map(ep_size, rank, NUM_EXPERTS)
        local = (expert_map >= 0).to(moe.gate_up_proj.device)
        yield expert_map, moe.gate_up_proj[local], moe.down_proj[local]


@pytest.mark.parametrize("ep_size", [2, 3])
@pytest.mark.parametrize("path", ["grouped_mm", "blocked"])
def test_expert_shards_sum_to_the_full_layer(moe, batch, ep_size, path):
    """What the all-reduce adds up: each rank's own experts, zeros for the rest."""
    x, topk_weights, topk_ids = batch
    with torch.inference_mode():
        want = torch_experts(x, moe.gate_up_proj, moe.down_proj, topk_weights, topk_ids)
        parts = [
            torch_experts(x, gate_up, down, topk_weights, topk_ids, expert_map) if path == "grouped_mm"
            else blocked_moe(gate_up, down, x, topk_weights, topk_ids, BLOCK_M, expert_map)
            for expert_map, gate_up, down in shards(moe, ep_size)
        ]
    torch.testing.assert_close(sum(parts), want)


def test_a_rank_with_no_routed_pairs_returns_zeros(moe, batch):
    x, topk_weights, _ = batch
    topk_ids = torch.zeros(TOKENS, TOP_K, dtype=torch.long)    # every pair on rank 0's expert
    expert_map, gate_up, down = list(shards(moe, 2))[1]
    with torch.inference_mode():
        assert torch.equal(torch_experts(x, gate_up, down, topk_weights, topk_ids, expert_map), torch.zeros_like(x))


@pytest.mark.parametrize("rank", [0, 1])
def test_expert_parallel_loads_whole_experts_for_its_rank(process_group, monkeypatch, rank):
    """EP replaces TP inside the layer: full intermediate size, and only this rank's experts."""
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    layer = FusedMoE(NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, enable_expert_parallel=True)
    assert (layer.tp_size, layer.ep_size, layer.ep_rank) == (1, 2, rank)
    assert layer.gate_up_proj.shape == (NUM_EXPERTS // 2, 2 * INTERMEDIATE, HIDDEN)
    layer.gate_up_proj.data.fill_(-1)
    for expert in range(NUM_EXPERTS):
        for proj in ("gate_proj", "up_proj"):
            weight = torch.full((INTERMEDIATE, HIDDEN), float(expert) + (proj == "up_proj") / 2)
            layer.weight_loader(layer.gate_up_proj, weight, (expert, proj))
    held = range(rank * NUM_EXPERTS // 2, (rank + 1) * NUM_EXPERTS // 2)
    for local, expert in enumerate(held):
        gate, up = layer.gate_up_proj[local].split(INTERMEDIATE)
        assert (gate == expert).all() and (up == expert + 0.5).all()


@pytest.mark.parametrize("rank", [0, 1])
def test_tensor_parallel_still_slices_every_expert(process_group, monkeypatch, rank):
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    layer = FusedMoE(NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE)
    assert (layer.tp_size, layer.ep_size, layer.expert_map) == (2, 1, None)
    assert layer.gate_up_proj.shape == (NUM_EXPERTS, INTERMEDIATE, HIDDEN)


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_the_triton_kernel_matches_grouped_mm_under_expert_parallel(moe, batch, dtype):
    """The kernel's zero blocks for other ranks' experts, with real arithmetic: shards must still sum to the layer."""
    layer = moe.to("cuda", dtype)
    x, topk_weights, topk_ids = batch
    x, topk_weights, topk_ids = x.to("cuda", dtype), topk_weights.to("cuda", dtype), topk_ids.cuda()
    with torch.inference_mode():
        want = torch_experts(x, layer.gate_up_proj, layer.down_proj, topk_weights, topk_ids)
        got = sum(
            fused_experts(x, gate_up, down, topk_weights, topk_ids, silu_and_mul, expert_map.cuda())
            for expert_map, gate_up, down in shards(layer, 3)
        )
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@requires_triton_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_the_triton_kernel_matches_grouped_mm_on_cuda(moe, batch, dtype):
    """The real tl.dot arithmetic against grouped_mm, in the dtype the model runs; the CPU test only reaches the indexing."""
    layer = moe.to("cuda", dtype)
    x, topk_weights, topk_ids = batch
    x, topk_weights, topk_ids = x.to("cuda", dtype), topk_weights.to("cuda", dtype), topk_ids.cuda()
    with torch.inference_mode():
        got = fused_experts(x, layer.gate_up_proj, layer.down_proj, topk_weights, topk_ids, silu_and_mul)
        want = torch_experts(x, layer.gate_up_proj, layer.down_proj, topk_weights, topk_ids)
    # bf16: reduction order differs between the kernel and grouped_mm, so allow bf16 rounding.
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


def test_forward_takes_the_torch_path_off_cuda(moe, batch):
    x, topk_weights, topk_ids = batch
    with torch.inference_mode():
        want = torch_experts(x, moe.gate_up_proj, moe.down_proj, topk_weights, topk_ids)
        assert torch.equal(moe(x, topk_weights, topk_ids), want)


def test_an_unknown_backend_is_refused(monkeypatch):
    monkeypatch.setenv("LEAN_VLLM_MOE_BACKEND", "nope")
    with pytest.raises(ValueError, match="unknown moe backend 'nope'"):
        use_triton(torch.zeros(1))


def test_triton_cannot_be_forced_where_it_does_not_run(monkeypatch):
    monkeypatch.setenv("LEAN_VLLM_MOE_BACKEND", "triton")
    if use_triton.__globals__["_IMPORT_ERROR"] is None and torch.cuda.is_available():
        pytest.skip("triton runs here")
    with pytest.raises(RuntimeError, match="not available"):
        use_triton(torch.zeros(1))
