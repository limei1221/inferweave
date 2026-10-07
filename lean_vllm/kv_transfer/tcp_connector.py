"""NixlConnector's protocol over TCP: the decode instance pulls a finished prefill's blocks, then releases them.

A producer holds a request's prompt blocks once it has sampled its one token, and serves them from kv_port + rank.
A consumer reads them into blocks it allocated, on a thread of its own, then sends "done" so the producer frees
them; blocks no consumer reads are freed after LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT. NIXL moves the same blocks GPU
to GPU over RDMA. Here a consumer that can see the producer's GPU maps its cache over CUDA IPC and copies the blocks
GPU to GPU; any other stages them through host memory, one layer at a time. The socket carries the control either way.
"""

import base64
import json
import logging
import queue
import socket
import socketserver
import struct
import threading
from collections import Counter
from contextlib import nullcontext
from time import monotonic

import torch

from lean_vllm import envs
from lean_vllm.kv_transfer.base import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVConnectorScheduler,
    KVConnectorWorker,
    KVTransferStats,
    ReqToRecv,
)

logger = logging.getLogger(__name__)

_HEADER = struct.Struct("!IQ")  # JSON length, then payload length
REGISTRATION_WAIT = 30.0  # how long a read waits for the engine to hand its request's blocks over
SOCKET_TIMEOUT = 60.0


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def send_message(sock: socket.socket, message: dict, nbytes: int = 0):
    """A JSON header announcing nbytes of payload, which the caller sends next."""
    header = json.dumps(message).encode()
    sock.sendall(_HEADER.pack(len(header), nbytes) + header)


def recv_message(sock: socket.socket) -> tuple[dict, int]:
    """A header and the payload bytes still to read."""
    header_len, nbytes = _HEADER.unpack(recv_exact(sock, _HEADER.size))
    return json.loads(recv_exact(sock, header_len)), nbytes


def recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:])
        if not k:
            raise ConnectionError("the peer closed the connection")
        got += k
    return buf


def _b64(data: bytes | None) -> str | None:
    return None if data is None else base64.b64encode(data).decode()


def _unb64(data: str | None) -> bytes | None:
    return None if data is None else base64.b64decode(data)


def export_ipc(caches: list[torch.Tensor]) -> dict | None:
    """What a process on this host needs to map each layer's cache, as torch shares a CUDA tensor; None off CUDA."""
    if not caches or not caches[0].is_cuda:
        return None
    layers = []
    for cache in caches:
        _, handle, size, offset, ref_handle, ref_offset, event_handle, event_sync = (
            cache.untyped_storage()._share_cuda_()
        )
        layers.append(
            dict(
                handle=_b64(handle),
                storage_size=size,
                storage_offset_bytes=offset,
                ref_counter_handle=_b64(ref_handle),
                ref_counter_offset=ref_offset,
                event_handle=_b64(event_handle),
                event_sync_required=event_sync,
                shape=list(cache.shape),
                stride=list(cache.stride()),
                storage_offset=cache.storage_offset(),
            )
        )
    return dict(device_uuid=str(torch.cuda.get_device_properties(caches[0].device).uuid), layers=layers)


def open_ipc(ipc: dict | None, caches: list[torch.Tensor]) -> list[torch.Tensor] | None:
    """The producer's caches mapped onto this GPU, or None if none was offered. Raises if its GPU is not mappable."""
    if ipc is None or not caches or not caches[0].is_cuda:
        return None
    visible = {str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())}
    if ipc["device_uuid"] not in visible:  # on another host, or hidden by CUDA_VISIBLE_DEVICES
        raise RuntimeError(f"its GPU {ipc['device_uuid']} is not visible here")
    device = caches[0].device
    peers = []
    for layer, cache in zip(ipc["layers"], caches):
        # Opened on this GPU, so the copies run here and read the producer's memory peer to peer.
        storage = torch.UntypedStorage._new_shared_cuda(
            device.index,
            _unb64(layer["handle"]),
            layer["storage_size"],
            layer["storage_offset_bytes"],
            _unb64(layer["ref_counter_handle"]),
            layer["ref_counter_offset"],
            _unb64(layer["event_handle"]),
            layer["event_sync_required"],
        )
        peer = torch.empty(0, dtype=cache.dtype, device=device)
        peers.append(peer.set_(storage, layer["storage_offset"], layer["shape"], layer["stride"]))
    return peers


class TcpConnectorScheduler(KVConnectorScheduler):
    def __init__(self, config):
        self.kv_transfer = config.kv_transfer
        self.block_size = config.kvcache_block_size
        self.tp_size = config.tensor_parallel_size
        self._reqs_to_recv: dict[str, ReqToRecv] = {}
        self._reqs_to_send: dict[str, list[int]] = {}

    def get_num_new_matched_tokens(self, seq, num_computed_tokens: int) -> tuple[int, bool]:
        params = seq.kv_transfer_params
        if not params or not params.get("do_remote_prefill"):
            return 0, False
        num_blocks = _cdiv(seq.num_prompt_tokens, self.block_size)
        problem = None
        if len(params["remote_block_ids"]) != num_blocks:
            problem = f"{len(params['remote_block_ids'])} remote blocks for a {num_blocks}-block prompt"
        elif params.get("tp_size", 1) != self.tp_size:
            problem = f"the prefill instance runs tp_size {params.get('tp_size', 1)}, this one {self.tp_size}"
        if problem:
            logger.warning("prefilling %s here instead: %s", seq.request_id, problem)
            params["do_remote_prefill"] = False
            return 0, False
        return seq.num_prompt_tokens - num_computed_tokens, True

    def update_state_after_alloc(self, seq, num_external_tokens: int):
        params = seq.kv_transfer_params
        params["do_remote_prefill"] = False  # one load per request, so a preempted one recomputes
        num_local_blocks = seq.num_cached_tokens // self.block_size  # prefix-cache hits here are not read
        self._reqs_to_recv[seq.request_id] = ReqToRecv(
            local_block_ids=seq.block_table[num_local_blocks:],
            remote_block_ids=params["remote_block_ids"][num_local_blocks:],
            remote_request_id=params["remote_request_id"],
            remote_engine_id=params["remote_engine_id"],
            remote_host=params["remote_host"],
            remote_port=params["remote_port"],
        )

    def build_connector_meta(self) -> KVConnectorMetadata:
        metadata = KVConnectorMetadata(self._reqs_to_recv, self._reqs_to_send)
        self._reqs_to_recv, self._reqs_to_send = {}, {}
        return metadata

    def request_finished(self, seq) -> tuple[bool, dict | None]:
        params = seq.kv_transfer_params
        # As vLLM: a prefill that hit its max_tokens or a stop token hands over; an abort ends there.
        if not params or not params.get("do_remote_decode") or seq.finish_reason not in ("length", "stop"):
            return False, None
        block_ids = seq.block_table[: _cdiv(seq.num_prompt_tokens, self.block_size)]
        self._reqs_to_send[seq.request_id] = block_ids
        kv = self.kv_transfer
        return True, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_request_id=seq.request_id,
            remote_engine_id=kv.engine_id,
            remote_block_ids=block_ids,
            remote_host=kv.kv_ip,
            remote_port=kv.kv_port,
            tp_size=self.tp_size,
        )


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class TcpConnectorWorker(KVConnectorWorker):
    def __init__(self, config, rank: int):
        self.kv_transfer = config.kv_transfer
        self.rank = rank
        self.block_size = config.kvcache_block_size
        self.kv_caches: list[torch.Tensor] = []
        self.layout: dict = {}
        self._lock = threading.Condition()  # guards the three below, shared with the server and loader threads
        self._reqs_to_send: dict[str, tuple[set[int], float]] = {}  # request -> (its blocks, when to give up)
        self._reading: Counter[str] = Counter()  # reads in progress, which expiry must wait out
        self._finished = KVConnectorOutput()
        self._recv_queue: queue.Queue = queue.Queue()
        # Loader thread only: a connection per producer rank, its engine id, and its caches if mapped over IPC.
        self._peers: dict[tuple[str, int], tuple[socket.socket, str, list[torch.Tensor] | None]] = {}
        self._server: _Server | None = None
        self._threads: list[threading.Thread] = []

    def register_kv_caches(self, kv_caches: list[torch.Tensor]):
        self.kv_caches = kv_caches
        # What a peer must match: the same blocks, in the same layout, layer for layer.
        self.layout = dict(
            block_size=self.block_size,
            layers=[[str(cache.dtype), [cache.size(0), *cache.shape[2:]]] for cache in kv_caches],
        )
        kv = self.kv_transfer
        if kv.is_producer:
            port = kv.kv_port + self.rank
            try:
                self._server = _Server((kv.kv_ip, port), self._handler())
            except OSError as error:
                raise RuntimeError(f"cannot serve KV on {kv.kv_ip}:{port}: {error}; pick another kv_port") from error
            self._start(self._server.serve_forever, "kv-server")
        if kv.is_consumer:
            self._start(self._load_loop, "kv-loader")

    def _start(self, target, name: str):
        thread = threading.Thread(target=target, name=f"lean-vllm-{name}-{self.rank}", daemon=True)
        thread.start()
        self._threads.append(thread)

    def start_load_kv(self, metadata: KVConnectorMetadata):
        if metadata.reqs_to_send:
            deadline = monotonic() + envs.LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT
            with self._lock:
                for request_id, block_ids in metadata.reqs_to_send.items():
                    self._reqs_to_send[request_id] = (set(block_ids), deadline)
                self._lock.notify_all()
        if metadata.reqs_to_recv:
            # Freshly allocated blocks may still be written by a step in flight; the loads wait for it.
            ready = torch.cuda.current_stream().record_event() if self._on_cuda else None
            for request_id, req in metadata.reqs_to_recv.items():
                self._recv_queue.put((request_id, req, ready))

    def get_finished(self) -> KVConnectorOutput:
        now = monotonic()
        with self._lock:
            for request_id, (_, deadline) in list(self._reqs_to_send.items()):
                if deadline < now and not self._reading[request_id]:
                    logger.warning(
                        "freeing the KV blocks of %s, which no decode instance read in %.0f s",
                        request_id,
                        envs.LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT,
                    )
                    self._release(request_id)
                    self._finished.expired_sending.add(request_id)
            finished, self._finished = self._finished, KVConnectorOutput()
        return finished

    def shutdown(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._recv_queue.put(None)
        for thread in self._threads:
            thread.join(timeout=5)
        for sock, _, _ in self._peers.values():
            sock.close()

    @property
    def _on_cuda(self) -> bool:
        return bool(self.kv_caches) and self.kv_caches[0].is_cuda

    def _thread_stream(self):
        """A side stream for this thread's copies, so they overlap the engine's steps."""
        if not self._on_cuda:
            return None
        torch.cuda.set_device(self.kv_caches[0].device)
        return torch.cuda.Stream()

    def _layer_nbytes(self, cache: torch.Tensor, num_blocks: int) -> int:
        return cache[:, 0].numel() * cache.element_size() * num_blocks

    def _release(self, request_id: str):
        """Lock held. The engine frees the blocks once every rank has reported them."""
        if self._reqs_to_send.pop(request_id, None) is not None:
            self._finished.finished_sending.add(request_id)

    # Producer: one server thread per connected consumer.

    def _handler(self):
        worker = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                stream = worker._thread_stream()
                while True:
                    try:
                        message, _ = recv_message(self.request)
                        worker._serve(self.request, message, stream)
                    except (ConnectionError, OSError):
                        return

        return Handler

    def _serve(self, sock: socket.socket, message: dict, stream):
        op = message.get("op")
        if op == "handshake":
            reply = dict(ok=True, engine_id=self.kv_transfer.engine_id, layout=self.layout)
            send_message(sock, dict(reply, ipc=export_ipc(self.kv_caches)))
        elif op == "read":
            self._serve_read(sock, message["request_id"], message["block_ids"], stream, message.get("direct", False))
        elif op == "done":
            with self._lock:
                self._release(message["request_id"])
            send_message(sock, dict(ok=True))
        else:
            send_message(sock, dict(ok=False, error=f"unknown op {op!r}"))

    def _serve_read(self, sock: socket.socket, request_id: str, block_ids: list[int], stream, direct: bool):
        with self._lock:
            # The engine hands the blocks over on the step after the request finishes, which can trail the reply.
            self._lock.wait_for(lambda: request_id in self._reqs_to_send, timeout=REGISTRATION_WAIT)
            held = self._reqs_to_send.get(request_id, (None,))[0]
            readable = held is not None and set(block_ids) <= held
            if readable:
                self._reading[request_id] += 1
        if not readable:
            error = f"{request_id} holds no such blocks here; it expired, or never prefilled here"
            send_message(sock, dict(ok=False, error=error))
            return
        try:
            if direct:
                # The consumer copies straight out of this cache; the read lasts until its next message, its done.
                send_message(sock, dict(ok=True))
                message, _ = recv_message(sock)
            else:
                num_bytes = sum(self._layer_nbytes(cache, len(block_ids)) for cache in self.kv_caches)
                send_message(sock, dict(ok=True), num_bytes)
                with torch.cuda.stream(stream) if stream is not None else nullcontext():
                    index = torch.tensor(block_ids, device=self.kv_caches[0].device)
                    for cache in self.kv_caches:
                        chunk = cache.index_select(1, index).cpu()  # blocking, so the bytes are there to send
                        sock.sendall(chunk.flatten().view(torch.uint8).numpy())
        finally:
            with self._lock:
                self._reading[request_id] -= 1
                if not self._reading[request_id]:
                    del self._reading[request_id]
        if direct:
            self._serve(sock, message, stream)

    # Consumer: one loader thread, reading requests in turn.

    def _load_loop(self):
        stream = self._thread_stream()
        while (item := self._recv_queue.get()) is not None:
            request_id, req, ready = item
            stats = None
            try:
                stats = self._load(req, ready, stream)
            except Exception as error:
                logger.warning(
                    "loading the KV of %s from %s:%d failed, so it prefills here: %s",
                    request_id,
                    req.remote_host,
                    req.remote_port + self.rank,
                    error,
                )
            with self._lock:
                if stats is None:
                    self._finished.failed_recving.add(request_id)
                else:
                    self._finished.finished_recving.add(request_id)
                    self._finished.recv_stats[request_id] = stats

    def _load(self, req: ReqToRecv, ready, stream) -> KVTransferStats:
        key = (req.remote_host, req.remote_port + self.rank)
        sock, peer_caches = self._connect(key, req.remote_engine_id)
        direct = peer_caches is not None
        try:
            started = monotonic()
            read = dict(op="read", request_id=req.remote_request_id, block_ids=req.remote_block_ids, direct=direct)
            send_message(sock, read)
            reply, num_bytes = recv_message(sock)
            if reply["ok"]:
                if direct:
                    self._copy_blocks(peer_caches, req, ready, stream)
                else:
                    self._receive_blocks(sock, req.local_block_ids, num_bytes, ready, stream)
                seconds = monotonic() - started
            # Done either way: blocks this engine will not read are no use held.
            send_message(sock, dict(op="done", request_id=req.remote_request_id))
            recv_message(sock)
        except BaseException:
            self._peers.pop(key, None)
            sock.close()  # its stream position is unknown
            raise
        if not reply["ok"]:
            raise RuntimeError(reply["error"])
        num_bytes = sum(self._layer_nbytes(cache, len(req.local_block_ids)) for cache in self.kv_caches)
        return KVTransferStats(seconds, num_bytes, "ipc" if direct else "tcp")

    def _copy_blocks(self, peer_caches: list[torch.Tensor], req: ReqToRecv, ready, stream):
        """GPU to GPU, out of the producer's mapped cache into this one."""
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if ready is not None:
                stream.wait_event(ready)
            device = self.kv_caches[0].device
            local, remote = (torch.tensor(ids, device=device) for ids in (req.local_block_ids, req.remote_block_ids))
            for cache, peer in zip(self.kv_caches, peer_caches):
                cache.index_copy_(1, local, peer.index_select(1, remote))
        # Before done, which lets the producer reuse the blocks.
        if stream is not None:
            stream.synchronize()

    def _receive_blocks(self, sock: socket.socket, block_ids: list[int], num_bytes: int, ready, stream):
        expected = sum(self._layer_nbytes(cache, len(block_ids)) for cache in self.kv_caches)
        if num_bytes != expected:
            raise RuntimeError(f"the producer sent {num_bytes} bytes for {len(block_ids)} blocks, not {expected}")
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if ready is not None:
                stream.wait_event(ready)
            index = torch.tensor(block_ids, device=self.kv_caches[0].device)
            for cache in self.kv_caches:
                data = recv_exact(sock, self._layer_nbytes(cache, len(block_ids)))
                chunk = torch.frombuffer(data, dtype=torch.uint8).view(cache.dtype)
                chunk = chunk.view(cache.size(0), len(block_ids), *cache.shape[2:])
                cache.index_copy_(1, index, chunk.to(cache.device))
        # The engine schedules the request as soon as this reports it.
        if stream is not None:
            stream.synchronize()

    def _connect(self, key: tuple[str, int], engine_id: str) -> tuple[socket.socket, list[torch.Tensor] | None]:
        """A connection to the producer rank at key, checked to hold engine_id's blocks; its caches if mapped over IPC."""
        if key in self._peers:
            sock, peer_engine_id, peer_caches = self._peers[key]
            if peer_engine_id == engine_id:
                return sock, peer_caches
            del self._peers[key]  # restarted since, so ask afresh
            sock.close()
        sock = socket.create_connection(key, timeout=SOCKET_TIMEOUT)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            send_message(sock, dict(op="handshake"))
            reply, _ = recv_message(sock)
            if reply["engine_id"] != engine_id:
                raise RuntimeError(f"{key[0]}:{key[1]} is engine {reply['engine_id']}, not {engine_id}; it restarted")
            if reply["layout"] != self.layout:
                raise RuntimeError(
                    f"{key[0]}:{key[1]} lays its KV cache out as {reply['layout']}, "
                    f"this engine as {self.layout}; serve the same model, dtype and block size"
                )
        except BaseException:
            sock.close()
            raise
        peer_caches = None
        try:
            peer_caches = open_ipc(reply.get("ipc"), self.kv_caches)
        except Exception as error:
            logger.info("reading KV from %s:%d over TCP, not CUDA IPC: %s", key[0], key[1], error)
        if peer_caches is not None:
            logger.info("reading KV from %s:%d over CUDA IPC", key[0], key[1])
        self._peers[key] = (sock, engine_id, peer_caches)
        return sock, peer_caches
