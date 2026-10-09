"""temperature == 0 means greedy, and must not divide the logits by zero; top-k and top-p truncate as vLLM's."""

import torch

from lean_vllm.layers.sampler import Sampler, apply_top_k_top_p

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


def test_top_k_one_is_argmax():
    x = logits(batch=64)
    top_k = torch.ones(x.size(0), dtype=torch.int64)
    assert torch.equal(Sampler()(x.clone(), torch.ones(x.size(0)), top_k, None), x.argmax(dim=-1))


def test_top_k_samples_only_from_each_rows_top_k():
    x = logits(batch=256)
    top_k = torch.tensor([1, 3, 0, 32]).repeat(64)  # 0 and the vocab size keep the whole row
    tokens = Sampler()(x.clone(), torch.full((x.size(0),), 5.0), top_k, None)
    rank = (x > x.gather(1, tokens.unsqueeze(1))).sum(dim=1)  # how many tokens beat the one sampled
    assert (rank[0::4] == 0).all() and (rank[1::4] < 3).all()
    assert rank[2::4].max() >= 3 and rank[3::4].max() >= 3


def test_top_p_keeps_the_smallest_set_reaching_p():
    """As vLLM's: a token goes only while the mass below it is within 1 - p, so the most likely always stays."""
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05]).repeat(3, 1)
    masked = apply_top_k_top_p(probs.log(), None, torch.tensor([0.75, 0.9, 0.1]))
    assert torch.isfinite(masked).tolist() == [
        [True, True, False, False],
        [True, True, True, False],
        [True, False, False, False],
    ]


def test_top_k_and_top_p_apply_together():
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    masked = apply_top_k_top_p(probs.log(), torch.tensor([3]), torch.tensor([0.5]))
    # Top-3 renormalizes to 4/9, 3/9, 2/9; the top 0.5 of that is the first two.
    assert torch.isfinite(masked).tolist() == [[True, True, False, False]]


def test_greedy_rows_ignore_top_k_and_top_p():
    x = logits()
    temperatures = torch.tensor([0.0, 1.0, 0.0, 1.0])
    tokens = Sampler()(x.clone(), temperatures, torch.zeros(4, dtype=torch.int64), torch.ones(4))
    assert torch.equal(tokens[::2], x.argmax(dim=-1)[::2])
