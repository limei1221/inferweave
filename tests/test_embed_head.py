"""The vocab-parallel LM head on two gloo ranks on the CPU: each rank's shard, gathered back in vocab order."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_expert_parallel import free_port

from lean_vllm.layers.embed_head import ParallelLMHead
from lean_vllm.utils.context import set_context

WORLD_SIZE = 2
VOCAB, HIDDEN, ROWS = 8, 4, 3


def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(VOCAB, HIDDEN, generator=generator), torch.randn(ROWS, HIDDEN, generator=generator)


def run_rank(rank: int, port: int, all_gather: bool, out: str):
    dist.init_process_group("gloo", f"tcp://localhost:{port}", world_size=WORLD_SIZE, rank=rank)
    try:
        weight, x = inputs()
        head = ParallelLMHead(VOCAB, HIDDEN)
        head.weight_loader(head.weight, weight)
        with torch.no_grad(), set_context(False):
            logits = head(x, all_gather)
        torch.save(logits, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("all_gather", [False, True], ids=["to_rank_0", "all_gather"])
def test_the_ranks_logits_are_the_whole_vocab_in_order(tmp_path, all_gather):
    out = str(tmp_path / "logits")
    mp.spawn(run_rank, args=(free_port(), all_gather, out), nprocs=WORLD_SIZE)
    weight, x = inputs()
    want = x @ weight.T
    got = [torch.load(f"{out}.{rank}") for rank in range(WORLD_SIZE)]
    torch.testing.assert_close(got[0], want)
    if all_gather:
        torch.testing.assert_close(got[1], want)
    else:
        assert got[1] is None  # only rank 0 samples
