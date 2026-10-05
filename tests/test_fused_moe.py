"""The Triton MoE's blocking, checked on the CPU against the grouped_mm path.

`blocked_moe` writes the kernel's indexing out in torch, so only the `tl.dot` arithmetic needs a GPU.
"""

import json

import pytest
import torch
import torch.distributed as dist

from lean_vllm.layers import fused_moe
from lean_vllm.layers.fused_moe import (
    align_blocks, fused_experts, get_config_file_name, get_default_config, get_moe_configs,
    try_get_optimal_moe_config, use_triton,
)
from lean_vllm.layers.moe import (
    FusedMoE, determine_expert_map, shared_mlp, silu_and_mul, torch_experts, torch_select_experts,
)

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


def test_a_run_holds_its_pairs_in_pair_order(batch):
    """The order the Triton kernels lay a run out in, so the two can be compared row for row."""
    _, _, topk_ids = batch
    sorted_pairs, block_experts, num_rows = align_blocks(topk_ids, NUM_EXPERTS, BLOCK_M)
    for block in range(int(num_rows) // BLOCK_M):
        held = sorted_pairs[block * BLOCK_M:(block + 1) * BLOCK_M]
        held = held[held < topk_ids.numel()].tolist()
        assert held == sorted(held)


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


def grouped_order(num_blocks: int, num_pid_n: int, group_size_m: int) -> list[tuple[int, int]]:
    """The kernel's pid -> (row block, column tile), in Python."""
    order = []
    for pid in range(num_blocks * num_pid_n):
        num_pid_in_group = group_size_m * num_pid_n
        first_pid_m = (pid // num_pid_in_group) * group_size_m
        size_m = min(num_blocks - first_pid_m, group_size_m)
        order.append((first_pid_m + (pid % num_pid_in_group) % size_m, (pid % num_pid_in_group) // size_m))
    return order


@pytest.mark.parametrize("group_size_m", [1, 4, 16, 64])
def test_grouped_order_covers_every_tile_once(group_size_m):
    """Any group size, including one past the block count or not dividing it, launches each tile exactly once."""
    order = grouped_order(num_blocks=13, num_pid_n=3, group_size_m=group_size_m)
    assert sorted(order) == [(m, n) for m in range(13) for n in range(3)]


def test_grouped_order_walks_a_column_tile_down_the_group():
    """Consecutive programs share a weight tile, which is the point of grouping."""
    assert grouped_order(num_blocks=4, num_pid_n=2, group_size_m=2)[:4] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert grouped_order(num_blocks=4, num_pid_n=2, group_size_m=1)[:2] == [(0, 0), (0, 1)]    # the old order


def test_config_file_names_match_vllm():
    assert get_config_file_name(64, 1408, "NVIDIA_H100_80GB_HBM3") == "E=64,N=1408,device_name=NVIDIA_H100_80GB_HBM3.json"
    assert get_config_file_name(64, 1408, "NVIDIA_H200_141GB") == "E=64,N=1408,device_name=NVIDIA_H200.json"


@pytest.fixture
def tuned_folder(tmp_path, monkeypatch):
    """A folder in $LEAN_VLLM_TUNED_CONFIG_FOLDER holding one vLLM-format file, on a pretend H100."""
    monkeypatch.setenv("LEAN_VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "NVIDIA H100 80GB HBM3")
    get_moe_configs.cache_clear()
    yield tmp_path
    get_moe_configs.cache_clear()


def tile(block_m: int) -> dict:
    return dict(BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3)


def test_a_tuned_file_maps_each_batch_to_its_nearest_entry(tuned_folder):
    configs = {"triton_version": "3.5.0", "1": tile(16), "64": tile(32), "512": {**tile(64), "SPLIT_K": 1}}
    (tuned_folder / "E=8,N=16,device_name=NVIDIA_H100_80GB_HBM3.json").write_text(json.dumps(configs))
    assert sorted(get_moe_configs(8, 16)) == [1, 64, 512]
    assert try_get_optimal_moe_config(8, 16, 3) == tile(16)
    assert try_get_optimal_moe_config(8, 16, 100) == tile(32)
    assert try_get_optimal_moe_config(8, 16, 4096) == tile(64)    # SPLIT_K dropped: the kernel has none


def test_no_file_falls_back_to_vllm_defaults(tuned_folder):
    assert get_moe_configs(8, 16) is None
    assert try_get_optimal_moe_config(8, 16, 7) == get_default_config(7, 8)


@pytest.mark.parametrize("M, E, want", [
    (1, 64, (16, 64, 128, 1, 4, 4)),
    (48, 64, (32, 64, 128, 1, 4, 3)),
    (256, 64, (64, 128, 64, 1, 8, 3)),
    (4096, 8, (128, 128, 64, 16, 8, 3)),    # 512 tokens per expert, so grouping pays
])
def test_defaults_are_vllms_bf16_table(M, E, want):
    config = get_default_config(M, E)
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages")
    assert tuple(config[key] for key in keys) == want


@requires_triton_gpu
@pytest.mark.parametrize("config", [
    dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4, num_stages=2),
    dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=16, num_warps=8, num_stages=3),
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128, GROUP_SIZE_M=64, num_warps=8, num_stages=2),
], ids=["small", "grouped", "large"])
def test_any_tuned_config_computes_the_same_layer(moe, batch, config):
    """Tiles and grouping change the schedule, never the result: what lets the tuner pick freely."""
    layer = moe.to("cuda", torch.bfloat16)
    x, topk_weights, topk_ids = batch
    x, topk_weights, topk_ids = x.to("cuda", torch.bfloat16), topk_weights.to("cuda", torch.bfloat16), topk_ids.cuda()
    with torch.inference_mode():
        got = fused_experts(x, layer.gate_up_proj, layer.down_proj, topk_weights, topk_ids, silu_and_mul, config=config)
        want = torch_experts(x, layer.gate_up_proj, layer.down_proj, topk_weights, topk_ids)
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


def dense_routing(weights: torch.Tensor, ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """[T, E] with each token's routing weights at its experts, so routings compare whatever order top-k left."""
    dense = torch.zeros(ids.size(0), num_experts, dtype=weights.dtype, device=weights.device)
    return dense.scatter_(1, ids.long(), weights)


@requires_triton_gpu
@pytest.mark.parametrize("num_tokens", [1, 20, 3000])    # 3000 * 6 pairs span many programs
@pytest.mark.parametrize("num_experts", [8, 64, 160])    # 160 is not a power of two
@pytest.mark.parametrize("block_m", [16, 64])
@pytest.mark.parametrize("ep", [False, True], ids=["one_rank", "ep"])
def test_triton_alignment_matches_the_reference(num_tokens, num_experts, block_m, ep):
    """Row for row up to num_rows, past which the kernels write nothing and no GEMM reads."""
    torch.manual_seed(0)
    top_k = 6
    topk_ids = torch.rand(num_tokens, num_experts, device="cuda").topk(top_k, dim=-1).indices.to(torch.int32)
    expert_map = determine_expert_map(3, 1, num_experts)[1].cuda() if ep else None
    want_pairs, want_experts, want_rows = align_blocks(topk_ids, num_experts, block_m)
    if expert_map is not None:
        want_experts = expert_map[want_experts]
    got_pairs, got_experts, got_rows = fused_moe.align_blocks_triton(topk_ids, num_experts, block_m, expert_map)
    num_rows = int(want_rows)
    assert int(got_rows) == num_rows
    assert torch.equal(got_pairs[:num_rows], want_pairs[:num_rows])
    assert torch.equal(got_experts[:num_rows // block_m], want_experts[:num_rows // block_m])


@requires_triton_gpu
@pytest.mark.parametrize("routing", [
    dict(num_groups=1, topk_group=1),
    dict(num_groups=8, topk_group=3),    # DeepSeek-V2's group-limited greedy
], ids=["greedy", "grouped"])
@pytest.mark.parametrize("renormalize, scaling", [(False, 1.0), (True, 2.5)])
@pytest.mark.parametrize("num_experts", [64, 160])
def test_triton_routing_matches_torch(routing, renormalize, scaling, num_experts):
    torch.manual_seed(0)
    router_logits = torch.randn(300, num_experts, device="cuda")
    args = (router_logits, 6, renormalize, scaling, routing["num_groups"], routing["topk_group"])
    got_weights, got_ids = fused_moe.topk_softmax(*args)
    want_weights, want_ids = torch_select_experts(*args)
    assert got_ids.dtype == want_ids.dtype == torch.int32
    torch.testing.assert_close(
        dense_routing(got_weights, got_ids, num_experts), dense_routing(want_weights, want_ids, num_experts))


@requires_triton_gpu
@pytest.mark.parametrize("width", [16, 1408, 3000])    # under one block, the V2-Lite expert, and three blocks
def test_triton_silu_and_mul_matches_torch(width):
    torch.manual_seed(0)
    x = torch.randn(37, 2 * width, device="cuda", dtype=torch.bfloat16)
    gate, up = x.float().chunk(2, dim=-1)
    torch.testing.assert_close(fused_moe.silu_and_mul(x), (torch.nn.functional.silu(gate) * up).to(torch.bfloat16))


def shared_weights(device="cpu", dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Two shared experts' worth, as DeepSeek-V2 merges them into one MLP."""
    torch.manual_seed(2)
    gate_up = torch.randn(4 * INTERMEDIATE, HIDDEN, device=device) * 0.1
    down = torch.randn(HIDDEN, 2 * INTERMEDIATE, device=device) * 0.1
    return gate_up.to(dtype), down.to(dtype)


def test_the_layer_adds_the_shared_experts(moe, batch):
    x, topk_weights, topk_ids = batch
    gate_up, down = shared_weights()
    with torch.inference_mode():
        want = torch_experts(x, moe.gate_up_proj, moe.down_proj, topk_weights, topk_ids)
        want = want + shared_mlp(x, gate_up, down, silu_and_mul)
        torch.testing.assert_close(moe(x, topk_weights, topk_ids, gate_up, down), want)


@requires_triton_gpu
@pytest.mark.parametrize("threshold", [0, 1024], ids=["one_stream", "side_stream"])
def test_the_side_stream_changes_nothing_but_timing(moe, batch, threshold, monkeypatch):
    """Same kernels on the same inputs, so the two orders agree to the bit."""
    monkeypatch.setenv("LEAN_VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", str(threshold))
    layer = moe.to("cuda", torch.bfloat16)
    x, topk_weights, topk_ids = batch
    x, topk_weights, topk_ids = x.to("cuda", torch.bfloat16), topk_weights.float().cuda(), topk_ids.int().cuda()
    gate_up, down = shared_weights("cuda", torch.bfloat16)
    with torch.inference_mode():
        routed = fused_experts(x, layer.gate_up_proj, layer.down_proj, topk_weights, topk_ids, fused_moe.silu_and_mul)
        want = routed + shared_mlp(x, gate_up, down, fused_moe.silu_and_mul)
        got = layer(x, topk_weights, topk_ids, gate_up, down)
    assert torch.equal(got, want)


@requires_triton_gpu
def test_the_side_stream_replays_in_a_cuda_graph(moe, batch, monkeypatch):
    """Forked from and joined back to the capturing stream, so a full decode graph can hold it."""
    monkeypatch.setenv("LEAN_VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "1024")
    layer = moe.to("cuda", torch.bfloat16)
    x, topk_weights, topk_ids = batch
    x, topk_weights, topk_ids = x.to("cuda", torch.bfloat16), topk_weights.float().cuda(), topk_ids.int().cuda()
    gate_up, down = shared_weights("cuda", torch.bfloat16)
    with torch.inference_mode():
        layer(x, topk_weights, topk_ids, gate_up, down)    # warmup, which also makes the side stream
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = layer(x, topk_weights, topk_ids, gate_up, down)
        x.copy_(torch.randn_like(x))    # new inputs in the captured buffers
        graph.replay()
        torch.cuda.synchronize()
        want = layer(x, topk_weights, topk_ids, gate_up, down)
    assert torch.equal(out, want)
