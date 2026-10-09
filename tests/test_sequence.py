"""Pending tokens: scheduled but not yet sampled on the host."""

import pickle

import pytest

from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def _block_size():
    Sequence.block_size = 4
    Sequence.enable_prefix_caching = True


def make(token_ids=(10, 11, 12)):
    return Sequence(list(token_ids), SamplingParams())


def test_a_reserved_token_lengthens_only_the_planned_count():
    seq = make()
    seq.reserve_token()
    assert seq.num_tokens == 3
    assert seq.num_planned_tokens == 4
    assert len(seq.token_ids) == 3


def test_committing_fills_the_reservation():
    seq = make()
    seq.reserve_token()
    seq.commit_token(99)
    assert (seq.num_tokens, seq.num_planned_tokens, seq.num_pending_tokens) == (4, 4, 0)
    assert seq.last_token == 99
    assert seq.token_ids == [10, 11, 12, 99]


def test_append_token_is_reserve_then_commit():
    seq, other = make(), make()
    other.reserve_token()
    other.commit_token(99)
    seq.append_token(99)
    assert (seq.num_tokens, seq.last_token) == (other.num_tokens, other.last_token)
    assert seq.num_pending_tokens == 0


def test_dropping_a_reservation_restores_the_planned_count():
    """A request that stops still had a token reserved for the next step."""
    seq = make()
    seq.reserve_token()
    seq.drop_pending()
    assert (seq.num_planned_tokens, seq.num_pending_tokens) == (3, 0)
    assert seq.num_completion_tokens == 0


def test_a_reserved_token_is_never_hashed():
    """A hash over an unsampled token would serve another request the wrong blocks."""
    seq = make([10, 11, 12])
    seq.reserve_token()  # would complete block 0 of size 4
    assert seq.block_hashes == []
    seq.commit_token(13)
    assert len(seq.block_hashes) == 1


def test_blocks_cover_the_reserved_token():
    """The reserved token needs a slot, so allocation counts it."""
    seq = make([10, 11, 12, 13])
    assert seq.num_blocks == 1
    seq.reserve_token()
    assert seq.num_blocks == 2


def test_pending_count_survives_pickle():
    """Rank 1 must agree with rank 0 about which rows sample."""
    seq = make()
    seq.reserve_token()
    restored = pickle.loads(pickle.dumps(seq))
    assert restored.num_pending_tokens == 1
    assert restored.num_planned_tokens == 4


def test_sampling_params_survive_pickle():
    """Every rank samples, so each needs the row's id and sampling params."""
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.7, top_k=5, top_p=0.9))
    restored = pickle.loads(pickle.dumps(seq))
    assert (restored.seq_id, restored.temperature, restored.top_k, restored.top_p) == (seq.seq_id, 0.7, 5, 0.9)


def test_commit_without_reservation_raises():
    """Prevent silent counter underflow."""
    seq = make()
    with pytest.raises(AssertionError, match="commit_token without a reservation"):
        seq.commit_token(99)


def test_max_tokens_must_be_at_least_one():
    """Prevent guard condition from hanging the request."""
    with pytest.raises(AssertionError):
        SamplingParams(max_tokens=0)


@pytest.mark.parametrize("bad", [dict(top_p=0.0), dict(top_p=1.5), dict(top_k=-2)])
def test_top_p_and_top_k_are_checked(bad):
    with pytest.raises(AssertionError):
        SamplingParams(**bad)


def test_greedy_drops_top_p_and_top_k():
    """The argmax needs no truncation, and the batch then skips the sort."""
    params = SamplingParams(temperature=0, top_p=0.5, top_k=3)
    assert (params.top_p, params.top_k) == (1.0, 0)
