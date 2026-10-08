from collections import OrderedDict
from typing import Iterable

from lean_vllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class FreeBlockQueue:
    """Free blocks, least recently released first. An OrderedDict, as a prefix hit takes one from the middle."""

    def __init__(self, block_ids: Iterable[int]):
        self._ids: OrderedDict[int, None] = OrderedDict.fromkeys(block_ids)

    def popleft(self) -> int:
        return self._ids.popitem(last=False)[0]

    def append(self, block_id: int):
        self._ids[block_id] = None

    def remove(self, block_id: int):
        del self._ids[block_id]

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)


class BlockManager:
    def __init__(
        self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True, recompute_last_hit: bool = False
    ):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        # An MTP drafter's cache at a block's last token depends on the token after it, so that block recomputes,
        # as vLLM's use_eagle.
        self.recompute_last_hit = recompute_last_hit
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids = FreeBlockQueue(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @property
    def usage(self) -> float:
        return len(self.used_block_ids) / len(self.blocks) if self.blocks else 0.0

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _num_blocks(self, seq: Sequence, num_lookahead: int) -> int:
        """Blocks seq needs once every reserved token lands and num_lookahead more slots are written."""
        return (seq.num_planned_tokens + num_lookahead + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence, num_lookahead: int = 0) -> int:
        """Cached blocks seq would get, or -1 if the rest does not fit. The trailing block always recomputes."""
        num_cached_blocks = 0
        num_new_blocks = self._num_blocks(seq, num_lookahead)
        last_hit_held = False
        for i in range(seq.num_blocks - 1) if self.enable_prefix_caching else ():
            block_id = self.hash_to_block_id.get(seq.block_hashes[i], -1)
            if block_id == -1 or self.blocks[block_id].token_ids != seq.block(i):
                break
            num_cached_blocks += 1
            last_hit_held = block_id in self.used_block_ids
            if last_hit_held:
                num_new_blocks -= 1  # already held, so it costs no free block
        if self.recompute_last_hit and num_cached_blocks:
            num_cached_blocks -= 1
            num_new_blocks += last_hit_held
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int, num_lookahead: int = 0):
        assert not seq.block_table
        for i in range(num_cached_blocks):
            block_id = self.hash_to_block_id[seq.block_hashes[i]]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, self._num_blocks(seq, num_lookahead)):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        seq.num_published_blocks = num_cached_blocks

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_published_blocks = 0  # the new block table has published nothing
        seq.block_table.clear()

    def _num_missing_blocks(self, seq: Sequence, num_lookahead: int) -> int:
        return max(self._num_blocks(seq, num_lookahead) - len(seq.block_table), 0)

    def can_append(self, seq: Sequence, num_lookahead: int = 0) -> bool:
        """Whether the blocks for seq's next token, and num_lookahead slots past it (drafts), are free."""
        return len(self.free_block_ids) >= self._num_missing_blocks(seq, num_lookahead)

    def may_append(self, seq: Sequence, num_lookahead: int = 0):
        for _ in range(self._num_missing_blocks(seq, num_lookahead)):
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence, num_computed_tokens: int):
        """Publish each full block once its KV is computed and its tokens are known (a reserved token can lag)."""
        if not self.enable_prefix_caching:
            return
        end = min(num_computed_tokens, seq.num_tokens) // self.block_size
        for i in range(seq.num_published_blocks, end):
            block = self.blocks[seq.block_table[i]]
            block.update(seq.block_hashes[i], seq.block(i))
            self.hash_to_block_id[block.hash] = block.block_id
        seq.num_published_blocks = max(seq.num_published_blocks, end)
