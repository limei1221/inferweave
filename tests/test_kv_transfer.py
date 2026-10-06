"""The KV connector's worker halves over real sockets, a producer and a consumer in one process, on CPU caches."""

import socket
from dataclasses import dataclass
from time import monotonic, sleep

import pytest
import torch

from lean_vllm.kv_transfer import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVOutputAggregator,
    KVTransferConfig,
    ReqToRecv,
    TcpConnectorWorker,
    check_kv_transfer_params,
    parse_kv_transfer_config,
    tcp_connector,
)

NUM_BLOCKS, BLOCK = 12, 4


@dataclass
class WorkerConfig:
    kv_transfer: KVTransferConfig
    kvcache_block_size: int = BLOCK


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_caches(seed: int, latent_dim: int = 6) -> list[torch.Tensor]:
    """A paged key/value layer and an MLA latent layer, block index on dim 1 as the runner lays them out."""
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(2, NUM_BLOCKS, BLOCK, 2, 8, generator=generator).to(torch.bfloat16),
        torch.randn(1, NUM_BLOCKS, BLOCK, latent_dim, generator=generator).to(torch.bfloat16),
    ]


@pytest.fixture
def make_worker():
    workers = []

    def _make(role: str, caches: list[torch.Tensor], port: int = 0, engine_id: str = "engine", rank: int = 0):
        kv_transfer = KVTransferConfig(kv_role=role, kv_port=port or free_port(), engine_id=engine_id)
        worker = TcpConnectorWorker(WorkerConfig(kv_transfer), rank)
        worker.register_kv_caches(caches)
        workers.append(worker)
        return worker

    yield _make
    for worker in workers:
        worker.shutdown()


def recv(
    producer: TcpConnectorWorker,
    local: list[int],
    remote: list[int],
    request_id: str = "p-1",
    engine_id: str = "engine",
) -> ReqToRecv:
    kv = producer.kv_transfer
    return ReqToRecv(local, remote, request_id, engine_id, kv.kv_ip, kv.kv_port - producer.rank)


def wait_for(worker: TcpConnectorWorker, timeout: float = 10.0) -> KVConnectorOutput:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if output := worker.get_finished():
            return output
        sleep(0.01)
    raise AssertionError("no transfer finished")


class TestTransfer:
    def test_the_blocks_land_exactly_in_the_consumer_s_own_blocks(self, make_worker):
        producer = make_worker("kv_producer", make_caches(0))
        consumer = make_worker("kv_consumer", make_caches(1))
        before = [cache.clone() for cache in consumer.kv_caches]
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [7, 2, 9]}))
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producer, [0, 5, 3], [7, 2, 9])}))
        assert wait_for(consumer) == KVConnectorOutput(finished_recving={"d-1"})
        for theirs, ours, old in zip(producer.kv_caches, consumer.kv_caches, before):
            assert torch.equal(ours[:, [0, 5, 3]], theirs[:, [7, 2, 9]])
            untouched = [i for i in range(NUM_BLOCKS) if i not in (0, 5, 3)]
            assert torch.equal(ours[:, untouched], old[:, untouched])

    def test_the_consumer_s_done_releases_the_producer_s_blocks(self, make_worker):
        producer = make_worker("kv_producer", make_caches(0))
        consumer = make_worker("kv_consumer", make_caches(1))
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [1, 2]}))
        assert not producer.get_finished()
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producer, [3, 4], [1, 2])}))
        wait_for(consumer)
        assert wait_for(producer) == KVConnectorOutput(finished_sending={"p-1"})

    def test_a_tail_read_after_local_prefix_hits_is_released_whole(self, make_worker):
        """The consumer reads only the blocks it lacks, and done frees the request's every block."""
        producer = make_worker("kv_producer", make_caches(0))
        consumer = make_worker("kv_consumer", make_caches(1))
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [1, 2, 3]}))
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producer, [8], [3])}))
        wait_for(consumer)
        assert torch.equal(consumer.kv_caches[0][:, 8], producer.kv_caches[0][:, 3])
        assert wait_for(producer).finished_sending == {"p-1"}

    def test_a_read_that_beats_the_hand_over_waits_for_it(self, make_worker):
        """The HTTP reply carrying the params can leave before the engine's next step registers the blocks."""
        producer = make_worker("kv_producer", make_caches(0))
        consumer = make_worker("kv_consumer", make_caches(1))
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producer, [0], [4])}))
        sleep(0.2)
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
        assert wait_for(consumer).finished_recving == {"d-1"}

    def test_several_loads_from_one_producer_share_a_connection(self, make_worker):
        producer = make_worker("kv_producer", make_caches(0))
        consumer = make_worker("kv_consumer", make_caches(1))
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [1], "p-2": [2]}))
        consumer.start_load_kv(
            KVConnectorMetadata(
                reqs_to_recv={
                    "d-1": recv(producer, [5], [1], "p-1"),
                    "d-2": recv(producer, [6], [2], "p-2"),
                }
            )
        )
        finished = set()
        while len(finished) < 2:
            finished |= wait_for(consumer).finished_recving
        assert len(consumer._peers) == 1
        assert torch.equal(consumer.kv_caches[1][:, [5, 6]], producer.kv_caches[1][:, [1, 2]])

    def test_a_rank_talks_to_the_same_rank_of_the_producer(self, make_worker):
        port = free_port()
        producers = [make_worker("kv_producer", make_caches(rank), port, rank=rank) for rank in (0, 1)]
        consumer = make_worker("kv_consumer", make_caches(9), rank=1)
        for producer in producers:
            producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [3]}))
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producers[0], [0], [3])}))
        wait_for(consumer)
        assert torch.equal(consumer.kv_caches[0][:, 0], producers[1].kv_caches[0][:, 3])


class TestFailures:
    def test_a_request_never_handed_over_fails_the_load(self, make_worker, monkeypatch, caplog):
        monkeypatch.setattr(tcp_connector, "REGISTRATION_WAIT", 0.1)
        producer = make_worker("kv_producer", make_caches(0))
        consumer = make_worker("kv_consumer", make_caches(1))
        before = consumer.kv_caches[0].clone()
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producer, [0], [4])}))
        assert wait_for(consumer) == KVConnectorOutput(failed_recving={"d-1"})
        assert torch.equal(consumer.kv_caches[0], before)
        assert "holds no such blocks" in caplog.text

    def test_blocks_the_request_does_not_hold_are_refused(self, make_worker, monkeypatch):
        monkeypatch.setattr(tcp_connector, "REGISTRATION_WAIT", 0.1)
        producer = make_worker("kv_producer", make_caches(0))
        consumer = make_worker("kv_consumer", make_caches(1))
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producer, [0, 1], [4, 5])}))
        assert wait_for(consumer).failed_recving == {"d-1"}

    def test_no_producer_listening_fails_the_load(self, make_worker):
        consumer = make_worker("kv_consumer", make_caches(1))
        req = ReqToRecv([0], [0], "p-1", "engine", "127.0.0.1", free_port())
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": req}))
        assert wait_for(consumer).failed_recving == {"d-1"}

    def test_a_restarted_producer_is_not_read(self, make_worker, caplog):
        """Its block ids name someone else's KV now."""
        producer = make_worker("kv_producer", make_caches(0), engine_id="after-restart")
        consumer = make_worker("kv_consumer", make_caches(1))
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
        consumer.start_load_kv(
            KVConnectorMetadata(
                reqs_to_recv={
                    "d-1": recv(producer, [0], [4], engine_id="before-restart"),
                }
            )
        )
        assert wait_for(consumer).failed_recving == {"d-1"}
        assert "it restarted" in caplog.text

    def test_a_different_cache_layout_is_not_read(self, make_worker, caplog):
        producer = make_worker("kv_producer", make_caches(0, latent_dim=6))
        consumer = make_worker("kv_consumer", make_caches(1, latent_dim=10))
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
        consumer.start_load_kv(KVConnectorMetadata(reqs_to_recv={"d-1": recv(producer, [0], [4])}))
        assert wait_for(consumer).failed_recving == {"d-1"}
        assert "serve the same model, dtype and block size" in caplog.text

    def test_blocks_nobody_reads_are_freed_after_the_timeout(self, make_worker, monkeypatch, caplog):
        monkeypatch.setenv("LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT", "0")
        producer = make_worker("kv_producer", make_caches(0))
        producer.start_load_kv(KVConnectorMetadata(reqs_to_send={"p-1": [4]}))
        sleep(0.01)
        assert producer.get_finished() == KVConnectorOutput(finished_sending={"p-1"})
        assert "which no decode instance read" in caplog.text

    def test_a_taken_port_is_named(self, make_worker):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            with pytest.raises(RuntimeError, match="pick another kv_port"):
                make_worker("kv_producer", make_caches(0), port=taken.getsockname()[1])

    def test_a_consumer_serves_nothing(self, make_worker):
        consumer = make_worker("kv_consumer", make_caches(1))
        assert consumer._server is None


class TestAggregator:
    def test_a_transfer_finishes_once_every_rank_reports_it(self):
        aggregator = KVOutputAggregator(world_size=2)
        assert not aggregator.aggregate([KVConnectorOutput(finished_recving={"a"}), KVConnectorOutput()])
        result = aggregator.aggregate([KVConnectorOutput(), KVConnectorOutput(finished_recving={"a"})])
        assert result == KVConnectorOutput(finished_recving={"a"})

    def test_one_rank_failing_fails_the_load(self):
        aggregator = KVOutputAggregator(world_size=2)
        result = aggregator.aggregate(
            [KVConnectorOutput(failed_recving={"a"}), KVConnectorOutput(finished_recving={"a"})]
        )
        assert result == KVConnectorOutput(failed_recving={"a"})

    def test_sends_count_the_same_way(self):
        aggregator = KVOutputAggregator(world_size=2)
        assert not aggregator.aggregate([KVConnectorOutput(finished_sending={"a"}), KVConnectorOutput()])
        assert aggregator.aggregate(
            [KVConnectorOutput(), KVConnectorOutput(finished_sending={"a"})]
        ).finished_sending == {"a"}


class TestConfig:
    def test_the_json_flag_parses(self):
        kv_transfer = parse_kv_transfer_config('{"kv_role": "kv_consumer", "kv_port": 15000}')
        assert (kv_transfer.kv_connector, kv_transfer.kv_role, kv_transfer.kv_port) == (
            "TcpConnector",
            "kv_consumer",
            15000,
        )
        assert kv_transfer.is_consumer and not kv_transfer.is_producer

    @pytest.mark.parametrize(
        "value, message",
        [
            ('{"kv_role": "decoder"}', "unknown kv_role"),
            ('{"kv_connector": "NixlConnector"}', "unknown kv_connector"),
            ('{"kv_buffer_size": 1}', "unknown kv_transfer_config keys"),
        ],
    )
    def test_a_bad_flag_is_named(self, value, message):
        with pytest.raises(ValueError, match=message):
            parse_kv_transfer_config(value)

    def test_each_engine_gets_its_own_id(self):
        assert parse_kv_transfer_config("{}").engine_id != parse_kv_transfer_config("{}").engine_id


class TestRequestParams:
    BOTH = KVTransferConfig(kv_role="kv_both")
    PULL = dict(
        do_remote_prefill=True,
        remote_block_ids=[1],
        remote_request_id="p",
        remote_engine_id="e",
        remote_host="h",
        remote_port=1,
    )

    def test_well_formed_params_pass(self):
        assert check_kv_transfer_params({"do_remote_decode": True}, self.BOTH) is None
        assert check_kv_transfer_params(self.PULL, self.BOTH) is None

    @pytest.mark.parametrize(
        "params, kv_role, message",
        [
            ({"do_remote_decode": True}, None, "needs an engine started with kv_transfer_config"),
            ({"do_remote_decode": True}, "kv_consumer", "do_remote_decode needs kv_role"),
            (PULL, "kv_producer", "do_remote_prefill needs kv_role"),
            (PULL | {"remote_port": "1"}, "kv_both", "remote_port, a int"),
            (PULL | {"remote_block_ids": ["x"]}, "kv_both", "remote_block_ids must be integers"),
            ({k: v for k, v in PULL.items() if k != "remote_host"}, "kv_both", "remote_host"),
        ],
    )
    def test_what_the_engine_cannot_honour_is_named(self, params, kv_role, message):
        kv_transfer = KVTransferConfig(kv_role=kv_role) if kv_role else None
        assert message in check_kv_transfer_params(params, kv_transfer)
