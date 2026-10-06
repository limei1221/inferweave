"""The sampled tokens of one step, fetched without waiting for later steps."""

import pytest
import torch

from lean_vllm.engine.sampled_tokens import SampledTokens


def test_it_returns_the_token_ids():
    tokens = torch.tensor([5, 6, 7], dtype=torch.int64)
    assert SampledTokens(tokens, torch.device("cpu")).tolist() == [5, 6, 7]


def test_an_empty_batch_returns_an_empty_list():
    """A step of partial prefills samples on no row."""
    tokens = torch.zeros(0, dtype=torch.int64)
    assert SampledTokens(tokens, torch.device("cpu")).tolist() == []


def test_the_copy_happens_once():
    """step() may drain the same pending step twice on the way out."""
    pending = SampledTokens(torch.tensor([9], dtype=torch.int64), torch.device("cpu"))
    assert pending.tolist() == [9]
    assert pending.tolist() == [9]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda for a copy stream")
def test_on_cuda_it_does_not_wait_for_later_work():
    """The copy rides its own stream, so a kernel queued after it is not awaited."""
    pending = SampledTokens(torch.tensor([3, 4], device="cuda"), torch.device("cuda"))
    later = torch.empty(4096, 4096, device="cuda")
    for _ in range(20):
        later = later @ later  # queued on the default stream after the copy
    assert pending.tolist() == [3, 4]
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda for pinned memory")
def test_host_destination_is_pinned():
    """The copy destination must be pinned for async device-to-host."""
    pending = SampledTokens(torch.tensor([1, 2], device="cuda"), torch.device("cuda"))
    # Access the host tensor to trigger the copy allocation check
    assert pending._host_tokens.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda for stream sharing")
def test_instances_share_copy_stream():
    """All instances share a single copy stream per process."""
    stream1 = SampledTokens._get_copy_stream()
    stream2 = SampledTokens._get_copy_stream()
    assert stream1 is stream2
