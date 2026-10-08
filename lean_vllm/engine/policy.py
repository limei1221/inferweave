import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator

from lean_vllm.engine.sequence import Sequence


class SchedulingPolicy(ABC):
    """Owns the waiting queue and picks preemption victims."""

    @staticmethod
    def create(name: str) -> "SchedulingPolicy":
        policies = {policy.name: policy for policy in (Fcfs, Priority)}
        if name not in policies:
            raise ValueError(f"unknown scheduling policy {name!r}, expected one of {sorted(policies)}")
        return policies[name]()

    @abstractmethod
    def add(self, seq: Sequence):
        """A newly arrived request."""

    @abstractmethod
    def requeue(self, seq: Sequence):
        """A preempted request, which has already waited once."""

    @abstractmethod
    def peek(self) -> Sequence: ...

    @abstractmethod
    def pop(self) -> Sequence: ...

    @abstractmethod
    def remove(self, seq: Sequence): ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Sequence]: ...

    @abstractmethod
    def victim(self, running: deque[Sequence]) -> Sequence:
        """Which running sequence gives up its blocks first."""

    def by_urgency(self, running: deque[Sequence]) -> deque[Sequence]:
        """Running in the order victim() spares it, most spared first."""
        return running


class Fcfs(SchedulingPolicy):
    """Arrival order, with preempted sequences going back to the front."""

    name = "fcfs"

    def __init__(self):
        self.queue: deque[Sequence] = deque()

    def add(self, seq):
        self.queue.append(seq)

    def requeue(self, seq):
        self.queue.appendleft(seq)

    def peek(self):
        return self.queue[0]

    def pop(self):
        return self.queue.popleft()

    def remove(self, seq):
        self.queue.remove(seq)

    def victim(self, running):
        return running[-1]  # newest, so the oldest work is preserved

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return iter(self.queue)

    def __contains__(self, seq):
        return seq in self.queue


class Priority(SchedulingPolicy):
    """Client-supplied priority, lower first, then arrival. Preempted sequences re-enter by priority too."""

    name = "priority"

    def __init__(self):
        self.heap: list[tuple[int, float, int, Sequence]] = []

    @staticmethod
    def _key(seq: Sequence):
        return (seq.priority, seq.arrival_time, seq.seq_id, seq)

    def add(self, seq):
        heapq.heappush(self.heap, self._key(seq))

    requeue = add

    def peek(self):
        return self.heap[0][-1]

    def pop(self):
        return heapq.heappop(self.heap)[-1]

    def remove(self, seq):
        self.heap.remove(self._key(seq))
        heapq.heapify(self.heap)

    @staticmethod
    def _urgency(seq: Sequence):
        return (seq.priority, seq.arrival_time)

    def victim(self, running):
        return max(running, key=self._urgency)

    def by_urgency(self, running):
        return deque(sorted(running, key=self._urgency))

    def __len__(self):
        return len(self.heap)

    def __iter__(self):
        return (entry[-1] for entry in self.heap)

    def __contains__(self, seq):
        return any(entry[-1] is seq for entry in self.heap)
