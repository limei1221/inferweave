"""The engine core in its own process: requests across, outputs back, and what a dead core owes its callers.

The core runs conftest's fake engine, built by the factories below in the spawned process.
"""

import asyncio
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import pytest

pytest.importorskip("zmq", reason="the serve extra is not installed")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from conftest import FakeConfig, FakeLLMEngine, FakeModelRunner, asyncio_test
from lean_vllm.engine.async_engine import EngineDeadError
from lean_vllm.engine.core_client import AsyncMPClient
from lean_vllm.engine.scheduler import QueueFull
from lean_vllm.engine.sequence import Sequence
from lean_vllm.sampling_params import SamplingParams

FOREVER = SamplingParams(max_tokens=64, ignore_eos=True)
MODEL = os.path.expanduser(os.getenv("LEAN_VLLM_TEST_MODEL", "~/workspace/huggingface/Qwen3-0.6B"))


class ScriptedModelRunner(FakeModelRunner):
    """Samples script[i] as a request's i-th token, so a real tokenizer can read the stream back."""

    def __init__(self, script: list[int]):
        super().__init__()
        self.script = script

    def _token(self, seq: Sequence) -> int:
        return self.script[self._completion_index(seq) % len(self.script)]


class ExplodingModelRunner(FakeModelRunner):

    def run(self, seqs):
        raise RuntimeError("boom")


# Factories run in the core process, so they must be importable from there: module level only.

def fake_engine(script: list[int] | None = None, explode: bool = False, **overrides) -> FakeLLMEngine:
    config = FakeConfig(**overrides)
    Sequence.block_size = config.kvcache_block_size
    runner = ExplodingModelRunner() if explode else ScriptedModelRunner(script) if script else FakeModelRunner()
    return FakeLLMEngine(config, runner)


def failing_engine():
    raise ValueError("no such model")


@pytest.fixture
def make_client():
    clients = []

    def _make(tokenizer=None, **kwargs) -> AsyncMPClient:
        client = AsyncMPClient(fake_engine, kwargs=kwargs, tokenizer=tokenizer, shutdown_timeout=10)
        client.start()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.stop()


async def collect(client: AsyncMPClient, prompt: list[int], params: SamplingParams):
    return [output async for output in await client.add_request(prompt, params)]


class TestAcrossTheProcess:

    @asyncio_test
    async def test_every_token_reaches_the_caller(self, make_client):
        client = make_client()
        outputs = await collect(client, list(range(8)), SamplingParams(max_tokens=5, ignore_eos=True))
        assert sum(len(output.token_ids) for output in outputs) == 5
        assert outputs[-1].finished and outputs[-1].finish_reason == "length"
        assert outputs[-1].metrics.ttft is not None

    @asyncio_test
    async def test_concurrent_requests_all_finish(self, make_client):
        client = make_client()
        params = SamplingParams(max_tokens=7, ignore_eos=True)
        results = await asyncio.gather(*(collect(client, list(range(n, n + 5)), params) for n in range(20)))
        assert [sum(len(o.token_ids) for o in outputs) for outputs in results] == [7] * 20

    @asyncio_test
    async def test_the_client_detokenizes(self, make_client):
        """The core sends ids only; the text is the client's, multi-byte characters included."""
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        backend = Tokenizer(models.BPE(vocab={byte: i for i, byte in enumerate(alphabet)}, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
        text = "café 🙂"
        script = tokenizer.encode(text)
        client = make_client(tokenizer=tokenizer, script=script)
        outputs = await collect(client, tokenizer.encode("hi"), SamplingParams(max_tokens=len(script), ignore_eos=True))
        assert "".join(output.text for output in outputs) == text

    @asyncio_test
    async def test_metrics_come_from_the_core(self, make_client):
        client = make_client()
        await collect(client, list(range(8)), SamplingParams(max_tokens=3, ignore_eos=True))
        summary = await client.metrics_summary()
        assert summary["requests"]["finished"] == {"length": 1}
        assert "lean_vllm" in await client.render_metrics()


class TestAdmissionAndAbort:

    @asyncio_test
    async def test_a_full_queue_is_refused_before_any_output(self, make_client):
        client = make_client(max_num_seqs=1, max_waiting_requests=1)
        running = await client.add_request(list(range(8)), FOREVER)
        await anext(running)    # scheduled, so the waiting queue is empty again
        waiting = await client.add_request(list(range(8)), FOREVER)
        with pytest.raises(QueueFull):
            await client.add_request(list(range(8)), FOREVER)
        await running.aclose()
        await waiting.aclose()

    @asyncio_test
    async def test_closing_the_stream_aborts_in_the_core(self, make_client):
        client = make_client()
        outputs = await client.add_request(list(range(8)), FOREVER)
        await anext(outputs)
        await outputs.aclose()
        for _ in range(100):
            summary = await client.metrics_summary()
            if summary["requests"]["aborted"]:
                break
            await asyncio.sleep(0.01)
        assert summary["requests"]["aborted"] == 1


class TestDeath:

    def test_a_core_that_cannot_build_its_engine_fails_the_constructor(self):
        with pytest.raises(RuntimeError, match="failed to start") as raised:
            AsyncMPClient(failing_engine)
        assert isinstance(raised.value.__cause__, ValueError)

    @asyncio_test
    async def test_a_step_that_raises_fails_live_streams(self, make_client):
        client = make_client(explode=True)
        died = asyncio.Event()
        client.on_death = died.set
        outputs = await client.add_request(list(range(8)), FOREVER)
        with pytest.raises(EngineDeadError, match="boom"):
            await anext(outputs)
        await asyncio.wait_for(died.wait(), 10)
        assert client.is_dead
        with pytest.raises(EngineDeadError):
            await client.add_request(list(range(8)), FOREVER)

    @asyncio_test
    async def test_a_killed_core_is_noticed(self, make_client):
        client = make_client()
        outputs = await client.add_request(list(range(8)), FOREVER)
        await anext(outputs)
        client._process.kill()
        with pytest.raises(EngineDeadError, match="exited"):
            async for _ in outputs:
                pass
        assert client.is_dead

    @asyncio_test
    async def test_stop_lets_the_core_exit_cleanly(self, make_client):
        client = make_client()
        await collect(client, list(range(8)), SamplingParams(max_tokens=2, ignore_eos=True))
        client.stop()
        assert client._process.exitcode == 0


ENGINE_ARGS = dict(enforce_eager=True, kvcache_memory_gb=0.25, max_model_len=256, max_num_batched_tokens=256)


def generate(prompt: str, params: SamplingParams) -> dict:
    """LLMEngine.generate, in a process of its own so its process group never meets this one's."""
    from lean_vllm.engine.llm_engine import LLMEngine
    return LLMEngine(MODEL, **ENGINE_ARGS).generate([prompt], params, use_tqdm=False)[0]


@pytest.mark.skipif(not os.path.isdir(MODEL), reason=f"no model at {MODEL}")
@asyncio_test
async def test_a_real_core_streams_what_the_engine_generates(monkeypatch):
    """LLMEngine with detokenize=False in the core, text from the client: the same as generate() in one process."""
    monkeypatch.setenv("LEAN_VLLM_DEVICE", "cpu")
    prompt, params = "The capital of France is", SamplingParams(temperature=0, max_tokens=12)
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        want = pool.submit(generate, prompt, params).result()
    client = AsyncMPClient.from_engine_args(MODEL, **ENGINE_ARGS)
    client.start()
    try:
        outputs = await collect(client, prompt, params)
    finally:
        client.stop()
    assert [token for output in outputs for token in output.token_ids] == want["token_ids"]
    assert "".join(output.text for output in outputs) == want["text"]
