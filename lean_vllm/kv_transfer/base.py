"""The KV connector interface, as vLLM's KVConnectorBase_V1: a scheduler half in the engine and a worker half per rank.

The scheduler half decides which requests load or hand over KV and holds their blocks meanwhile; each step it sends
the worker halves a KVConnectorMetadata, and they answer with the transfers that finished since.
"""

import json
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field

KV_ROLES = ("kv_producer", "kv_consumer", "kv_both")


@dataclass(slots=True)
class KVTransferConfig:
    """vLLM's --kv-transfer-config, as JSON. A producer serves prefilled KV, a consumer loads it, both does either."""

    kv_connector: str = "TcpConnector"
    kv_role: str = "kv_both"
    kv_ip: str = "127.0.0.1"  # where a producer listens, and what it tells consumers to dial
    kv_port: int = 14579  # rank r listens on kv_port + r
    engine_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        if self.kv_role not in KV_ROLES:
            raise ValueError(f"unknown kv_role {self.kv_role!r}, expected one of {list(KV_ROLES)}")

    @classmethod
    def parse(cls, value: "str | dict | KVTransferConfig") -> "KVTransferConfig":
        if isinstance(value, KVTransferConfig):
            return value
        fields = json.loads(value) if isinstance(value, str) else dict(value)
        unknown = set(fields) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown kv_transfer_config keys {sorted(unknown)}")
        return cls(**fields)

    @property
    def is_producer(self) -> bool:
        return self.kv_role in ("kv_producer", "kv_both")

    @property
    def is_consumer(self) -> bool:
        return self.kv_role in ("kv_consumer", "kv_both")


# What a decode instance needs to pull a remote prefill's blocks, as the producer's request_finished returns it.
REMOTE_PREFILL_FIELDS = {
    "remote_block_ids": list,
    "remote_request_id": str,
    "remote_engine_id": str,
    "remote_host": str,
    "remote_port": int,
}


def check_kv_transfer_params(params, kv_transfer: KVTransferConfig | None) -> str | None:
    """Why this engine cannot honour a request's kv_transfer_params, or None if it can."""
    if not isinstance(params, dict):
        return "kv_transfer_params must be an object"
    if kv_transfer is None:
        return "kv_transfer_params needs an engine started with kv_transfer_config"
    if params.get("do_remote_decode") and not kv_transfer.is_producer:
        return f"do_remote_decode needs kv_role kv_producer or kv_both, not {kv_transfer.kv_role}"
    if params.get("do_remote_prefill"):
        if not kv_transfer.is_consumer:
            return f"do_remote_prefill needs kv_role kv_consumer or kv_both, not {kv_transfer.kv_role}"
        for name, kind in REMOTE_PREFILL_FIELDS.items():
            if not isinstance(params.get(name), kind):
                return f"do_remote_prefill needs {name}, a {kind.__name__}"
        if not all(isinstance(block_id, int) for block_id in params["remote_block_ids"]):
            return "remote_block_ids must be integers"
    return None


@dataclass(slots=True)
class ReqToRecv:
    """A remote prefill's blocks to read into local ones, the two lists aligned."""

    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_request_id: str
    remote_engine_id: str
    remote_host: str
    remote_port: int


@dataclass(slots=True)
class KVConnectorMetadata:
    """One step's instructions to every worker half."""

    reqs_to_recv: dict[str, ReqToRecv] = field(default_factory=dict)
    reqs_to_send: dict[str, list[int]] = field(default_factory=dict)  # held blocks a remote decode may now read

    def __bool__(self):
        return bool(self.reqs_to_recv or self.reqs_to_send)


@dataclass(slots=True)
class KVTransferStats:
    """One load, as vLLM's NIXL stats record a transfer: from the read to the blocks landing."""

    seconds: float
    num_bytes: int
    transport: str  # "ipc" or "tcp"

    def merge(self, other: "KVTransferStats") -> "KVTransferStats":
        """Ranks load in parallel: the slowest sets the time, and the bytes add up."""
        transport = self.transport if self.transport == other.transport else "mixed"
        return KVTransferStats(max(self.seconds, other.seconds), self.num_bytes + other.num_bytes, transport)


@dataclass(slots=True)
class KVConnectorOutput:
    """Transfers a worker half finished since its last report. A failed load leaves its blocks to recompute."""

    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
    failed_recving: set[str] = field(default_factory=set)
    expired_sending: set[str] = field(default_factory=set)  # of finished_sending, freed with no consumer read
    recv_stats: dict[str, KVTransferStats] = field(default_factory=dict)  # of finished_recving

    def __bool__(self):
        return bool(self.finished_sending or self.finished_recving or self.failed_recving)


class KVOutputAggregator:
    """A transfer is over once every rank has reported it, as vLLM's; a load failed if any rank's did."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._sent: Counter[str] = Counter()
        self._recved: Counter[str] = Counter()
        self._failed: set[str] = set()
        self._expired: set[str] = set()
        self._stats: dict[str, KVTransferStats] = {}

    def aggregate(self, outputs: list[KVConnectorOutput]) -> KVConnectorOutput:
        result = KVConnectorOutput()
        for output in outputs:
            self._failed |= output.failed_recving
            self._expired |= output.expired_sending
            for request_id, stats in output.recv_stats.items():
                seen = self._stats.get(request_id)
                self._stats[request_id] = stats if seen is None else seen.merge(stats)
            for request_id in output.finished_sending:
                self._sent[request_id] += 1
                if self._sent[request_id] == self.world_size:
                    del self._sent[request_id]
                    result.finished_sending.add(request_id)
                    if request_id in self._expired:
                        self._expired.discard(request_id)
                        result.expired_sending.add(request_id)
            for request_id in output.finished_recving | output.failed_recving:
                self._recved[request_id] += 1
                if self._recved[request_id] == self.world_size:
                    del self._recved[request_id]
                    failed = request_id in self._failed
                    self._failed.discard(request_id)
                    stats = self._stats.pop(request_id, None)
                    if failed:
                        result.failed_recving.add(request_id)
                    else:
                        result.finished_recving.add(request_id)
                        if stats is not None:
                            result.recv_stats[request_id] = stats
        return result


class KVConnectorScheduler(ABC):
    """The engine's half. Every method runs on the engine thread, between steps."""

    @abstractmethod
    def get_num_new_matched_tokens(self, seq, num_computed_tokens: int) -> tuple[int, bool]:
        """Prompt tokens past num_computed_tokens whose KV can come from elsewhere, and whether it loads async."""

    @abstractmethod
    def update_state_after_alloc(self, seq, num_external_tokens: int):
        """seq now holds blocks for its prompt; queue the load into the ones not already cached here."""

    @abstractmethod
    def build_connector_meta(self) -> KVConnectorMetadata:
        """This step's metadata for the workers. Resets what it hands over."""

    @abstractmethod
    def request_finished(self, seq) -> tuple[bool, dict | None]:
        """Whether to hold seq's blocks until a worker reports them sent, and the kv_transfer_params to return."""


class KVConnectorWorker(ABC):
    """One rank's half: moves blocks between this rank's KV cache and another instance's."""

    @abstractmethod
    def register_kv_caches(self, kv_caches: list):
        """Each layer's cache, block index on dim 1, in model order. Called once, before any step."""

    @abstractmethod
    def start_load_kv(self, metadata: KVConnectorMetadata):
        """Start this step's loads and expose this step's sends. Must not block on the network."""

    @abstractmethod
    def get_finished(self) -> KVConnectorOutput:
        """Transfers finished on this rank since the last call."""

    def shutdown(self):
        pass
