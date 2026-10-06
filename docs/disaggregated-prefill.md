# Disaggregated prefill / decode

Prefill and decode run on separate engines. A prefill instance computes a
prompt's KV cache and samples one token. A decode instance pulls that KV and
generates the rest. A proxy in front splits each request between them. Long
prompts then stop stalling other requests' decode steps, and each side can be
sized and batched for its own phase.

The design follows vLLM V1: a KV connector with a scheduler half and a worker
half (`KVConnectorBase_V1`), the request flow and `kv_transfer_params` of its
`NixlConnector`, and its `toy_proxy_server.py`. The difference is the transport.
NIXL moves blocks GPU to GPU over RDMA; lean-vLLM's `TcpConnector` stages them
through host memory over a TCP socket.

## What is supported

| | Support |
| --- | --- |
| Connector | `TcpConnector`: the decode instance pulls, as NIXL does |
| Roles | `kv_producer` (prefill), `kv_consumer` (decode), `kv_both` (either, per request) |
| Models | Every model the engine loads: Qwen3's key/value pages, DeepSeek-V2's MLA latents |
| Tensor parallelism | Equal sizes on both sides; rank *r* pulls from rank *r* |
| Prefix caching | On both sides. The decode instance reads only the blocks it does not already hold |
| Front door | `lean-vllm proxy`, round-robin over any number of prefill and decode servers |
| Failure | A load that fails, for any reason, falls back to prefilling on the decode instance |
| Devices | CPU verified end to end. The CUDA path (side streams, events) has not run on a GPU yet |

Not supported: RDMA or GPU-direct transfer, different tensor-parallel sizes on
the two sides, layer-by-layer streaming during the prefill (vLLM's
`save_kv_layer`), and a streamed prefill request.

## Running it

Two servers and a proxy. Here they share one host with a GPU each:

```bash
MODEL=~/workspace/huggingface/Qwen3-8B
CUDA_VISIBLE_DEVICES=0 uv run lean-vllm serve $MODEL --port 8100 --served-model-name qwen \
    --kv-transfer-config '{"kv_role": "kv_producer", "kv_port": 14579}'
CUDA_VISIBLE_DEVICES=1 uv run lean-vllm serve $MODEL --port 8200 --served-model-name qwen \
    --kv-transfer-config '{"kv_role": "kv_consumer"}'
uv run lean-vllm proxy --prefill http://127.0.0.1:8100 --decode http://127.0.0.1:8200 --port 8000
```

Clients talk to the proxy on port 8000, exactly as they would to one server.
`--prefill` and `--decode` each take several URLs. Across hosts, set `kv_ip` on
each prefill server to an address its decode servers can reach.

### `--kv-transfer-config`

JSON, as vLLM's flag. Every key is optional.

| Key | Default | What it sets |
| --- | --- | --- |
| `kv_connector` | `TcpConnector` | The only one |
| `kv_role` | `kv_both` | `kv_producer` serves KV, `kv_consumer` loads it, `kv_both` does either |
| `kv_ip` | `127.0.0.1` | Where a producer listens, and the address it tells consumers to dial |
| `kv_port` | 14579 | Rank *r* of a producer listens on `kv_port + r` |
| `engine_id` | random | Names this instance; a consumer refuses blocks from a producer that has restarted since |

`LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT` (default 480 seconds, as vLLM's
`VLLM_NIXL_ABORT_REQUEST_TIMEOUT`) is how long a producer holds blocks that no
consumer has read before freeing them.

### Without the proxy

The proxy only sets `kv_transfer_params`, a request field both endpoints accept:

1. Send the request to a prefill server with `"max_tokens": 1`, not streamed,
   and `"kv_transfer_params": {"do_remote_decode": true}`.
2. Its reply carries `kv_transfer_params`: the blocks, and where to pull them.
3. Send the original request to a decode server with those
   `kv_transfer_params`.

A server started without `--kv-transfer-config`, or in the wrong role, answers
a request carrying `kv_transfer_params` with a 400.

## What has been verified

- `tests/test_disagg_e2e.py` runs Qwen3-0.6B on CPU as two engines in two
  processes. The decode engine prefills one token itself, so the rest came over
  the socket. Its greedy tokens match a local run of the same prompt split the
  same way. Zeroing the received KV fails the test.
- Through the HTTP servers and proxy on CPU, greedy completions match a single
  server, streamed and not. One prompt differed, and was reproduced exactly by a
  single engine whose prompt was split into all-but-one tokens, then one. That
  is the recomputed last token below, not the transfer. After the decode
  instance reads its blocks, the prefill instance's KV usage returns to zero. A
  client that disconnects mid-stream aborts the decode.
- The transport, over real sockets on CPU caches: blocks land exactly in
  the consumer's own block ids, and nothing else is touched. Also covered: tail
  reads after local prefix hits, a read that arrives before the hand-over, the
  rank-to-rank pairing, and each failure below.
- No GPU run and no benchmark yet.

## How it works

### A request's path

```
client ──▶ proxy ──(max_tokens=1, do_remote_decode)──▶ prefill server
                                                         prefills, samples 1 token, finishes "length";
                                                         holds the prompt's blocks; replies with
                                                         kv_transfer_params {remote_block_ids, host, port, ...}
             proxy ──(original request + kv_transfer_params)──▶ decode server
                                                         allocates blocks, WAITING_FOR_REMOTE_KVS,
                                                         pulls the blocks ◀──TCP── prefill rank r
                                                         sends "done" ──▶ prefill frees its blocks
                                                         computes the last prompt token, samples, decodes
client ◀── proxy ◀── tokens, streamed or not
```

### Scheduler half

It lives in `Scheduler`, with vLLM's method names:

- `get_num_new_matched_tokens` is asked at admission, after the local prefix
  cache. With `do_remote_prefill`, every prompt token past the local hits comes
  from the producer. If the producer's block count does not fit this prompt, or
  its tensor-parallel size differs, the request prefills here instead.
- `update_state_after_alloc` queues the load. Local block ids pair with the
  producer's, skipping blocks already cached here. It also clears
  `do_remote_prefill`, so a request preempted later recomputes rather than
  reading blocks the producer has since freed.
- `build_connector_meta` collects each step's loads to start and blocks to
  expose, and goes to every rank with the step.
- `request_finished` decides, on the producer, whether to hold a finished
  request's blocks. Only a request that ran to its `max_tokens` is handed over,
  as in vLLM; a stop token or an abort ends it there. Held blocks stay in the
  prefix cache once freed.

A loading request sits in `Scheduler.recving` in `WAITING_FOR_REMOTE_KVS`. It
counts toward `num_requests_waiting`, as in vLLM, and takes no step budget.
Once loaded, it joins `running` with all but its last prompt token computed.
That token is recomputed against the pulled KV, so the decode instance samples
the first token itself, as vLLM's does. The cost is that its output matches a
local run split the same way, not one that prefills the prompt whole. In bf16
the two can pick different tokens at a near-tie.

The engine calls `kv_connector_step` every step, even one that schedules
nothing. Loads then start, and finished transfers come back, while the engine
is otherwise idle. `is_finished` stays false while a load or a held send is
open, so the server keeps polling.

### Worker half

One per rank, in `ModelRunner`. It registers the per-layer caches, whose block
index is dim 1 in every layout. Each step's `KVConnectorOutput` is gathered
from all ranks. As in vLLM's `KVOutputAggregator`, a transfer is over once
every rank reports it, and a load failed if any rank's did.

- **Producer**: a threaded TCP server on `kv_ip:kv_port + rank`. A read is
  served only for blocks the engine has handed over for that request. The
  hand-over comes with the step after the request finishes, so it can trail
  the HTTP reply; a read waits up to 30 s for it. Blocks are copied out one
  layer at a time and sent as they arrive. `done` releases them, and so does
  the timeout, though never during a read.
- **Consumer**: one loader thread with a connection per producer rank. A new
  connection handshakes first: the producer must be the `remote_engine_id` the
  request names, with the same per-layer shape, dtype and block size. Blocks
  land with `index_copy_` into the request's own blocks. Then it sends `done`,
  even after a refused read, since blocks it will not read are no use held.

The wire format is a fixed header (JSON length, payload length), a JSON
message, then raw bytes. No pickle crosses the socket.

### Ordering on a GPU

Transfers run on side streams. On the producer, a request's KV is complete
before it is handed over, because its token was synchronised back to the host.
On the consumer, freshly allocated blocks may still be written by a step in
flight under async scheduling: their previous owner's next token. So each load
waits on an event recorded when it is queued, and it synchronises its stream
before reporting itself done.

### Aborts and failures

| Event | What happens |
| --- | --- |
| Client aborts during the load | The request finishes at once; its blocks are freed when the load ends, since the loader is writing them |
| Producer unreachable, restarted, refused the read, or laid out differently | The load fails; the request prefills on the decode instance from its local prefix hits |
| No consumer reads the blocks | The producer frees them after `LEAN_VLLM_KV_ABORT_REQUEST_TIMEOUT` |
| Prefill stops on a stop token | No `kv_transfer_params`; the proxy sends the request on as is and the decode instance prefills it |

## Next steps

- Run on GPUs and benchmark against vLLM's `NixlConnector` and a single
  instance: TTFT, TPOT under prefill-heavy load, transfer time per block.
- Pull GPU to GPU: CUDA IPC within a host, NIXL or NCCL across hosts.
- Overlap loads with one another; the single loader thread serialises them.
- Heterogeneous tensor parallelism, and transfer metrics on `/metrics`.
