"""The vocab-parallel LM head on two gloo ranks on the CPU: each rank's shard, gathered back in vocab order on both."""

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


def run_rank(rank: int, port: int, out: str):
    dist.init_process_group("gloo", f"tcp://localhost:{port}", world_size=WORLD_SIZE, rank=rank)
    try:
        weight, x = inputs()
        head = ParallelLMHead(VOCAB, HIDDEN)
        head.weight_loader(head.weight, weight)
        with torch.no_grad(), set_context(False):
            logits = head(x)
        torch.save(logits, f"{out}.{rank}")
    finally:
        dist.destroy_process_group()


def test_the_ranks_logits_are_the_whole_vocab_in_order(tmp_path):
    out = str(tmp_path / "logits")
    mp.spawn(run_rank, args=(free_port(), out), nprocs=WORLD_SIZE)
    weight, x = inputs()
    want = x @ weight.T
    got = [torch.load(f"{out}.{rank}") for rank in range(WORLD_SIZE)]
    for logits in got:
        torch.testing.assert_close(logits, want)  # every rank samples
