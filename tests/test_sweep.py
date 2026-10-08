"""The sweep driver's command building, where an engine comparison can silently turn unfair."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))  # scripts, not a package

import sweep

MODEL = "/models/Qwen3-8B"


def args(*extra: str):
    return sweep.parse_args(["--model", MODEL, *extra])


class TestCacheFlags:
    def test_the_same_tokens_reach_either_engine(self):
        tokens = 320000
        lean = sweep.cache_flags(args("--kvcache-tokens", str(tokens)))
        vllm = sweep.cache_flags(args("--engine", "vllm", "--kvcache-tokens", str(tokens)))
        assert lean["num-kvcache-blocks"] * lean["kvcache-block-size"] == tokens
        assert vllm["num-gpu-blocks-override"] * vllm["block-size"] == tokens
        assert lean["num-kvcache-blocks"] == vllm["num-gpu-blocks-override"]

    def test_the_block_size_is_pinned_too_so_the_arithmetic_holds(self):
        assert sweep.cache_flags(args("--kvcache-tokens", "1024"))["kvcache-block-size"] == 16
        assert sweep.cache_flags(args("--engine", "vllm", "--kvcache-tokens", "1024"))["block-size"] == 16

    def test_unpinned_capacity_still_pins_the_block_size(self):
        assert sweep.cache_flags(args()) == {"kvcache-block-size": 16}
        assert sweep.cache_flags(args("--engine", "vllm")) == {"block-size": 16}


class TestServerCommand:
    def test_booleans_become_the_paired_flag(self):
        arm = sweep.Arm("a", {"enable-chunked-prefill": False, "enforce-eager": True})
        command = sweep.server_command(args(), arm)
        assert "--no-enable-chunked-prefill" in command
        assert "--enforce-eager" in command

    def test_the_operators_own_flags_come_last(self):
        command = sweep.server_command(args("--server-args", "--max-num-seqs 64"), sweep.Arm("a"))
        assert command[:3] == ["lean-vllm", "serve", MODEL]
        assert command[-2:] == ["--max-num-seqs", "64"]

    def test_vllm_is_served_by_its_own_binary(self):
        command = sweep.server_command(args("--engine", "vllm"), sweep.Arm("a"))
        assert command[:2] == ["vllm", "serve"]


class TestAsyncSuite:
    def test_both_arms_reach_either_engine_as_the_paired_flag(self):
        arms = sweep.async_suite(args())
        for engine in ("lean-vllm", "vllm"):
            off, on = (sweep.server_command(args("--engine", engine), arm) for arm in arms)
            assert "--no-async-scheduling" in off
            assert "--async-scheduling" in on
