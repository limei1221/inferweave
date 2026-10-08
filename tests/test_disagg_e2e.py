"""Disaggregated prefill end to end: a prefill and a decode engine, each in its own process, on a real model on CPU.

The decode engine recomputes the last prompt token against the pulled KV, as vLLM's does, so its greedy tokens are
compared with the prefill engine running the same prompt split the same way: all but one token, then that one.
"""

import json
import multiprocessing as mp
import os
import socket
from time import sleep

import pytest

MODEL = os.path.expanduser(os.getenv("LEAN_VLLM_TEST_MODEL", "~/workspace/huggingface/Qwen3-0.6B"))
pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f"no model at {MODEL}")

# Five blocks and a bit; split whole, its greedy tokens differ from split n-1 + 1, so the comparison is exact.
PROMPT = "In 1905, Albert Einstein published four papers that " + "changed physics forever, and " * 12
MAX_TOKENS = 16
ENGINE_ARGS = dict(enforce_eager=True, kvcache_memory_gb=0.25, max_model_len=256, max_num_batched_tokens=256)


def _engine(**kwargs):
    os.environ["LEAN_VLLM_DEVICE"] = "cpu"
    from lean_vllm.engine.llm_engine import LLMEngine

    return LLMEngine(MODEL, **ENGINE_ARGS, **kwargs)


def _prefill(conn, kv_port: int):
    from transformers import AutoTokenizer

    from lean_vllm.sampling_params import SamplingParams

    num_prompt_tokens = len(AutoTokenizer.from_pretrained(MODEL).encode(PROMPT))
    engine = _engine(
        enable_prefix_caching=False,
        long_prefill_token_threshold=num_prompt_tokens - 1,
        kv_transfer_config=json.dumps(dict(kv_role="kv_producer", kv_port=kv_port)),
    )
    hand_off = SamplingParams(temperature=0, max_tokens=1, kv_transfer_params={"do_remote_decode": True})
    engine.add_request(PROMPT, hand_off, "p-1")
    params = None
    while params is None:
        params = next((output.kv_transfer_params for output in engine.step()[0] if output.finished), None)
    conn.send(params)
    while not engine.is_finished():  # until the decode engine has read the blocks and let them go
        engine.step()
        sleep(0.001)
    conn.send(engine.scheduler.block_manager.usage)
    reference = engine.generate([PROMPT], SamplingParams(temperature=0, max_tokens=MAX_TOKENS), use_tqdm=False)
    conn.send(reference[0]["token_ids"])


def _decode(conn):
    from lean_vllm.sampling_params import SamplingParams

    engine = _engine(kv_transfer_config=json.dumps(dict(kv_role="kv_consumer")))
    params = conn.recv()
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, kv_transfer_params=params)
    outputs = engine.generate([PROMPT], sampling_params, use_tqdm=False)
    conn.send((outputs[0]["token_ids"], engine.metrics.prefill_tokens.total))


def test_the_decode_engine_continues_from_the_prefill_engine_s_kv():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        kv_port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    prefill_conn, prefill_child = ctx.Pipe()
    decode_conn, decode_child = ctx.Pipe()
    processes = [
        ctx.Process(target=_prefill, args=(prefill_child, kv_port)),
        ctx.Process(target=_decode, args=(decode_child,)),
    ]
    for process in processes:
        process.start()
    try:
        assert prefill_conn.poll(300), "the prefill engine never handed off"
        params = prefill_conn.recv()
        assert params["do_remote_prefill"] and params["remote_port"] == kv_port
        decode_conn.send(params)
        assert decode_conn.poll(300), "the decode engine never finished"
        token_ids, num_prefill_tokens = decode_conn.recv()
        assert prefill_conn.poll(60)
        assert prefill_conn.recv() == 0.0  # every held block freed once read
        assert prefill_conn.poll(300)
        reference = prefill_conn.recv()
    finally:
        for process in processes:
            process.join(timeout=60)
            if process.is_alive():
                process.kill()
    assert num_prefill_tokens == 1  # only the last prompt token ran here; the rest was pulled
    assert token_ids == reference
