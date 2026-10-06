"""Disaggregated prefill in the scheduler: what each side admits, holds and frees, with the transfers faked."""

import pytest

from lean_vllm.engine.scheduler import DuplicateRequestId
from lean_vllm.engine.sequence import SequenceStatus
from lean_vllm.kv_transfer import KVConnectorOutput, KVTransferConfig
from lean_vllm.sampling_params import SamplingParams

BLOCK = 8
PROMPT = list(range(100, 120))  # 20 tokens: three blocks, the last partial


def remote_prefill(num_blocks: int = 3, **overrides) -> dict:
    """What a producer's request_finished hands back, for blocks 50, 51, ..."""
    return (
        dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_request_id="p-1",
            remote_engine_id="prefill-engine",
            remote_block_ids=list(range(50, 50 + num_blocks)),
            remote_host="10.0.0.1",
            remote_port=14579,
            tp_size=1,
        )
        | overrides
    )


@pytest.fixture
def make_disagg_engine(make_engine):
    def _make(role: str = "kv_both", **overrides):
        kv_transfer = KVTransferConfig(kv_role=role, kv_ip="10.0.0.2", kv_port=15000, engine_id="this-engine")
        return make_engine(kvcache_block_size=BLOCK, kv_transfer=kv_transfer, **overrides)

    return _make


def recvs(engine) -> dict:
    """Every load the runner has been asked to start, by request."""
    return {
        request_id: req
        for metadata in engine.model_runner.kv_metadata
        for request_id, req in metadata.reqs_to_recv.items()
    }


def sends(engine) -> dict:
    return {
        request_id: block_ids
        for metadata in engine.model_runner.kv_metadata
        for request_id, block_ids in metadata.reqs_to_send.items()
    }


class TestDecodeSide:
    def test_a_remote_prefill_waits_for_its_blocks_without_running(self, make_disagg_engine):
        engine = make_disagg_engine()
        seq = engine.add(PROMPT, SamplingParams(max_tokens=4, kv_transfer_params=remote_prefill()), "d-1")
        engine.step()
        engine.step()
        assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
        assert engine.model_runner.batches == []
        assert not engine.is_finished()
        req = recvs(engine)["d-1"]
        assert req.local_block_ids == seq.block_table and len(seq.block_table) == 3
        assert req.remote_block_ids == [50, 51, 52]
        assert (req.remote_request_id, req.remote_host, req.remote_port) == ("p-1", "10.0.0.1", 14579)

    def test_once_loaded_it_computes_only_its_last_prompt_token(self, make_disagg_engine):
        """As vLLM: the decode instance recomputes one token so that it samples the first one itself."""
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=3, ignore_eos=True, kv_transfer_params=remote_prefill()), "d-1")
        engine.step()
        engine.model_runner.kv_outputs.append(KVConnectorOutput(finished_recving={"d-1"}))
        outputs = engine.run_to_completion()
        assert engine.model_runner.batches[0] == (True, [("d-1", 1)])
        assert len(outputs["d-1"]) == 3
        assert not engine.scheduler.block_manager.used_block_ids

    def test_local_prefix_hits_are_not_read_again(self, make_disagg_engine):
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1), "warm")
        engine.run_to_completion()
        seq = engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=remote_prefill()), "d-1")
        engine.step()
        req = recvs(engine)["d-1"]
        assert seq.num_cached_tokens == 2 * BLOCK  # the trailing block always recomputes
        assert req.local_block_ids == seq.block_table[2:]
        assert req.remote_block_ids == [52]

    def test_a_failed_load_prefills_here(self, make_disagg_engine):
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=remote_prefill()), "d-1")
        engine.step()
        engine.model_runner.kv_outputs.append(KVConnectorOutput(failed_recving={"d-1"}))
        outputs = engine.run_to_completion()
        assert engine.model_runner.batches[0] == (True, [("d-1", len(PROMPT))])
        assert len(outputs["d-1"]) == 1

    def test_a_block_count_that_does_not_fit_the_prompt_prefills_here(self, make_disagg_engine, caplog):
        """The prefill instance tokenized something else; its blocks would be the wrong KV."""
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=remote_prefill(num_blocks=2)), "d-1")
        engine.run_to_completion()
        assert recvs(engine) == {}
        assert engine.model_runner.batches[0] == (True, [("d-1", len(PROMPT))])
        assert "2 remote blocks for a 3-block prompt" in caplog.text

    def test_a_different_tensor_parallel_size_prefills_here(self, make_disagg_engine):
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=remote_prefill(tp_size=2)), "d-1")
        engine.run_to_completion()
        assert recvs(engine) == {}

    def test_an_abort_mid_load_frees_the_blocks_only_once_the_load_ends(self, make_disagg_engine):
        """The loader is still writing them."""
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=remote_prefill()), "d-1")
        engine.step()
        assert engine.scheduler.abort("d-1")
        assert len(engine.scheduler.block_manager.used_block_ids) == 3
        with pytest.raises(DuplicateRequestId):
            engine.add(PROMPT, SamplingParams(max_tokens=1), "d-1")
        engine.model_runner.kv_outputs.append(KVConnectorOutput(finished_recving={"d-1"}))
        engine.step()
        assert not engine.scheduler.block_manager.used_block_ids
        assert engine.is_finished()
        assert engine.model_runner.batches == []

    def test_a_preempted_request_recomputes_rather_than_loading_twice(self, make_disagg_engine):
        """The producer freed its blocks after the first load."""
        engine = make_disagg_engine()
        seq = engine.add(PROMPT, SamplingParams(max_tokens=4, kv_transfer_params=remote_prefill()), "d-1")
        engine.step()
        engine.model_runner.kv_outputs.append(KVConnectorOutput(finished_recving={"d-1"}))
        engine.step()
        engine.step()
        engine.scheduler.running.remove(seq)
        engine.scheduler._preempt(seq, engine.last_output)
        num_batches = len(engine.model_runner.batches)
        engine.run_to_completion()
        assert list(recvs(engine)) == ["d-1"]
        is_prefill, [(_, num_tokens)] = engine.model_runner.batches[num_batches]
        assert is_prefill and num_tokens > 1  # recomputed here, past its local prefix hits

    def test_whole_prompt_scheduling_admits_a_load_bigger_than_the_budget(self, make_disagg_engine):
        """Chunked prefill off refuses prompts over the budget, but a load computes one token."""
        engine = make_disagg_engine(enable_chunked_prefill=False, max_num_batched_tokens=16)
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=remote_prefill()), "d-1")
        engine.step()
        engine.model_runner.kv_outputs.append(KVConnectorOutput(finished_recving={"d-1"}))
        outputs = engine.run_to_completion()
        assert outputs["d-1"] and engine.model_runner.batches[0] == (True, [("d-1", 1)])

    def test_the_request_s_params_are_copied_not_marked(self, make_disagg_engine):
        """generate() shares one SamplingParams across prompts."""
        params = remote_prefill()
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=params), "d-1")
        engine.step()
        assert params["do_remote_prefill"] is True


class TestPrefillSide:
    def test_a_finished_prefill_holds_its_blocks_and_says_where_they_are(self, make_disagg_engine):
        engine = make_disagg_engine()
        params = {"do_remote_decode": True}
        seq = engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=params), "p-1")
        block_table = list(seq.block_table)
        final = [output for output in engine.step() + engine.step() if output.finished]
        assert len(final) == 1 and final[0].finish_reason == "length"
        assert final[0].kv_transfer_params == dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_request_id="p-1",
            remote_engine_id="this-engine",
            remote_block_ids=seq.block_table,
            remote_host="10.0.0.2",
            remote_port=15000,
            tp_size=1,
        )
        assert block_table == [] and len(seq.block_table) == 3  # allocated at admission, after the add
        assert len(engine.scheduler.block_manager.used_block_ids) == 3
        assert not engine.is_finished()
        engine.step()
        assert sends(engine) == {"p-1": seq.block_table}

    def test_the_blocks_are_freed_once_sent(self, make_disagg_engine):
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params={"do_remote_decode": True}), "p-1")
        engine.step()
        engine.step()
        engine.model_runner.kv_outputs.append(KVConnectorOutput(finished_sending={"p-1"}))
        engine.step()
        assert not engine.scheduler.block_manager.used_block_ids
        assert engine.is_finished()

    def test_the_freed_blocks_stay_in_the_prefix_cache(self, make_disagg_engine):
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params={"do_remote_decode": True}), "p-1")
        engine.step()
        engine.step()
        engine.model_runner.kv_outputs.append(KVConnectorOutput(finished_sending={"p-1"}))
        engine.step()
        engine.add(PROMPT, SamplingParams(max_tokens=1), "again")
        engine.step()
        assert engine.last_output.num_cached_blocks == 2

    def test_a_prefill_that_stops_early_hands_nothing_over(self, make_disagg_engine):
        """As vLLM: only a prefill capped by max_tokens is sent; a stop token ends the request there."""
        engine = make_disagg_engine(eos_after={"p-1": 0})
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params={"do_remote_decode": True}), "p-1")
        final = [output for output in engine.step() + engine.step() if output.finished]
        assert final[0].finish_reason == "stop" and final[0].kv_transfer_params is None
        assert engine.is_finished() and not engine.scheduler.block_manager.used_block_ids

    def test_an_aborted_prefill_hands_nothing_over(self, make_disagg_engine):
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params={"do_remote_decode": True}), "p-1")
        engine.step()
        assert engine.scheduler.abort("p-1")
        assert engine.scheduler.sending == {}

    def test_an_id_still_held_is_refused(self, make_disagg_engine):
        engine = make_disagg_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params={"do_remote_decode": True}), "p-1")
        engine.step()
        engine.step()
        with pytest.raises(DuplicateRequestId):
            engine.add(PROMPT, SamplingParams(max_tokens=1), "p-1")

    def test_without_a_connector_nothing_changes(self, make_engine):
        engine = make_engine()
        engine.add(PROMPT, SamplingParams(max_tokens=1))
        engine.run_to_completion()
        assert engine.model_runner.kv_metadata == []
        assert engine.last_output.kv_connector_metadata is None


def test_a_load_counts_as_waiting(make_disagg_engine):
    """As vLLM's num_requests_waiting, which includes WAITING_FOR_REMOTE_KVS."""
    engine = make_disagg_engine()
    engine.add(PROMPT, SamplingParams(max_tokens=1, kv_transfer_params=remote_prefill()), "d-1")
    engine.step()
    assert engine.metrics.waiting.value == 1 and engine.metrics.running.value == 0


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("transfer", ["sending", "recving"])
def test_decode_waits_for_blocks_held_by_a_transfer(make_disagg_engine, async_scheduling, transfer):
    engine = make_disagg_engine(num_kvcache_blocks=3, async_scheduling=async_scheduling)
    params = {"do_remote_decode": True} if transfer == "sending" else remote_prefill(num_blocks=2)
    remote = engine.add(list(range(30, 46)), SamplingParams(max_tokens=1, kv_transfer_params=params), "remote")
    for _ in range(3):
        engine.step()
    assert "remote" in getattr(engine.scheduler, transfer)

    local = engine.add(list(range(10, 18)), SamplingParams(max_tokens=2), "local")
    outputs = []
    for _ in range(4):
        outputs.extend(engine.step())
    assert not local.is_finished
    assert local.num_completion_tokens == 1

    done = (
        KVConnectorOutput(finished_sending={"remote"})
        if transfer == "sending"
        else KVConnectorOutput(finished_recving={"remote"})
    )
    engine.model_runner.kv_outputs.append(done)
    rest = engine.run_to_completion()
    tokens = [token for output in outputs if output.request_id == "local" for token in output.token_ids]
    assert len(tokens + rest["local"]) == 2
    assert local.finish_reason == remote.finish_reason == "length"
    assert not engine.scheduler.block_manager.used_block_ids
