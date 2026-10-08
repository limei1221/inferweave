"""The free queue's eviction order: what the cache gives up first when blocks run short."""

import pytest

from lean_vllm.engine.block_manager import BlockManager, FreeBlockQueue
from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams

BLOCK_SIZE = 4


@pytest.fixture
def manager():
    Sequence.block_size = BLOCK_SIZE
    return BlockManager(num_blocks=6, block_size=BLOCK_SIZE)


def cache(manager: BlockManager, token_ids: list[int]) -> list[int]:
    """Admit a sequence, hash its full blocks, release it, and report the blocks it held."""
    seq = Sequence(token_ids, SamplingParams())
    manager.allocate(seq, manager.can_allocate(seq))
    manager.hash_blocks(seq, seq.num_tokens)
    block_table = list(seq.block_table)
    manager.deallocate(seq)
    return block_table


class TestFreeQueue:
    def test_taking_a_block_out_of_the_middle_leaves_the_order_alone(self):
        """A prefix hit takes its blocks back from wherever they sit."""
        queue = FreeBlockQueue(range(5))
        queue.remove(2)
        assert list(queue) == [0, 1, 3, 4]
        queue.append(2)
        assert list(queue) == [0, 1, 3, 4, 2]
        assert queue.popleft() == 0


class TestEvictionOrder:
    def test_blocks_that_cache_nothing_are_spent_first(self, manager):
        cache(manager, list(range(12)))  # blocks 0, 1 and 2, then released
        assert list(manager.free_block_ids) == [3, 4, 5, 2, 1, 0]

    def test_a_sequence_gives_up_its_deepest_block_first(self, manager):
        """Which is what makes a shared prefix outlive the suffixes built on it."""
        held = cache(manager, list(range(12)))
        taken = [manager._allocate_block() for _ in range(6)]
        assert taken[3:] == list(reversed(held))

    def test_cached_contents_survive_until_the_block_is_claimed(self, manager):
        """Eviction is lazy: a freed block keeps its hash while it waits."""
        cache(manager, list(range(12)))
        surviving = []
        for _ in range(6):
            manager._allocate_block()
            surviving.append(len(manager.hash_to_block_id))
        assert surviving == [3, 3, 3, 2, 1, 0]


class TestPublishBound:
    def test_it_publishes_only_blocks_whose_kv_is_computed(self, manager):
        """A chunked prefill has tokens the step has not run yet."""
        seq = Sequence(list(range(12)), SamplingParams())
        manager.allocate(seq, manager.can_allocate(seq))
        manager.hash_blocks(seq, num_computed_tokens=4)
        assert len(manager.hash_to_block_id) == 1
        manager.hash_blocks(seq, num_computed_tokens=12)
        assert len(manager.hash_to_block_id) == 3

    def test_it_publishes_only_blocks_whose_tokens_are_known(self, manager):
        """A reserved token can put the computed count ahead of the real one."""
        seq = Sequence(list(range(4)), SamplingParams())
        manager.allocate(seq, manager.can_allocate(seq))
        seq.reserve_token()
        manager.hash_blocks(seq, num_computed_tokens=5)
        assert len(manager.hash_to_block_id) == 1  # block 0 is whole; there is no block 1 yet

    def test_publishing_twice_publishes_each_block_once(self, manager):
        seq = Sequence(list(range(8)), SamplingParams())
        manager.allocate(seq, manager.can_allocate(seq))
        manager.hash_blocks(seq, num_computed_tokens=8)
        manager.hash_blocks(seq, num_computed_tokens=8)
        assert seq.num_published_blocks == 2

    def test_deallocating_lets_a_requeued_sequence_publish_again(self, manager):
        """Preemption takes its block table away, so the new one must be published."""
        seq = Sequence(list(range(8)), SamplingParams())
        manager.allocate(seq, manager.can_allocate(seq))
        manager.hash_blocks(seq, num_computed_tokens=8)
        manager.deallocate(seq)
        assert seq.num_published_blocks == 0
