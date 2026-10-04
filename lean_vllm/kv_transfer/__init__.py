from lean_vllm.kv_transfer.base import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVConnectorScheduler,
    KVConnectorWorker,
    KVOutputAggregator,
    KVTransferConfig,
    ReqToRecv,
    check_kv_transfer_params,
)
from lean_vllm.kv_transfer.tcp_connector import TcpConnectorScheduler, TcpConnectorWorker

# As vLLM's KVConnectorFactory: a name in kv_transfer_config, and the class for each half.
CONNECTORS = {"TcpConnector": (TcpConnectorScheduler, TcpConnectorWorker)}


def parse_kv_transfer_config(value: "str | dict | KVTransferConfig") -> KVTransferConfig:
    kv_transfer = KVTransferConfig.parse(value)
    if kv_transfer.kv_connector not in CONNECTORS:
        raise ValueError(f"unknown kv_connector {kv_transfer.kv_connector!r}, expected one of {sorted(CONNECTORS)}")
    return kv_transfer


def create_scheduler_connector(config) -> KVConnectorScheduler:
    return CONNECTORS[config.kv_transfer.kv_connector][0](config)


def create_worker_connector(config, rank: int) -> KVConnectorWorker:
    return CONNECTORS[config.kv_transfer.kv_connector][1](config, rank)
