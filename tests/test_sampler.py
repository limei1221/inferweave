"""temperature == 0 means greedy, and must not divide the logits by zero."""

import torch

from lean_vllm.layers.sampler import Sampler

torch.manual_seed(0)


def logits(batch: int = 4, vocab: int = 32) -> torch.Tensor:
    return torch.randn(batch, vocab)


def test_zero_temperature_is_argmax():
    x = logits()
    temperatures = torch.zeros(x.size(0))
    assert torch.equal(Sampler()(x, temperatures), x.argmax(dim=-1))


def test_no_temperatures_is_the_greedy_fast_path():
    x = logits()
    assert torch.equal(Sampler()(x, None), x.argmax(dim=-1))


def test_greedy_and_random_rows_coexist_in_one_batch():
    x = logits()
    temperatures = torch.tensor([0.0, 1.0, 0.0, 1.0])
    tokens = Sampler()(x, temperatures)
    assert torch.equal(tokens[::2], x.argmax(dim=-1)[::2])
    assert ((tokens >= 0) & (tokens < x.size(1))).all()


def test_a_peaked_distribution_still_samples_its_peak():
    x = torch.full((2, 16), -20.0)
    x[:, 3] = 20.0
    tokens = Sampler()(x, torch.ones(2))
    assert torch.equal(tokens, torch.tensor([3, 3]))


def test_a_new_batch_size_does_not_recompile():
    """Warmup samples a couple of rows; serving any other count must not stall on a compile."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()  # forget the sizes earlier tests compiled
    sampler = Sampler()
    sampler(logits(batch=2), torch.ones(2))
    before = counters["stats"]["unique_graphs"]
    for batch in (3, 7):
        sampler(logits(batch=batch), torch.ones(batch))
    assert counters["stats"]["unique_graphs"] == before
