import atexit
import collections.abc
import os
from dataclasses import dataclass, fields
from time import perf_counter

import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lean_vllm import envs
from lean_vllm.config import Config
from lean_vllm.engine.metrics import Metrics
from lean_vllm.engine.model_runner import ModelRunner
from lean_vllm.engine.output import RequestOutput
from lean_vllm.engine.output_processor import OutputProcessor
from lean_vllm.engine.sampled_tokens import SampledTokens
from lean_vllm.engine.scheduler import InvalidRequest, LaunchedRow, QueueFull, Scheduler, SchedulerOutput
from lean_vllm.engine.sequence import Sequence
from lean_vllm.kv_transfer import KVTransferConfig, check_kv_transfer_params
from lean_vllm.sampling_params import SamplingParams


def validate_request(
    prompt: list[int],
    sampling_params: SamplingParams,
    vocab_size: int,
    max_model_len: int,
    kv_transfer: KVTransferConfig | None = None,
):
    """Every front door comes through here, so nothing invalid reaches the runner."""
    if not prompt:
        raise InvalidRequest("the prompt is empty")
    bad = next((token_id for token_id in prompt if not 0 <= token_id < vocab_size), None)
    if bad is not None:
        raise InvalidRequest(f"token id {bad} is outside the {vocab_size}-token vocabulary")
    if len(prompt) >= max_model_len:
        raise InvalidRequest(f"prompt is {len(prompt)} tokens, leaving no room in the {max_model_len}-token context")
    if len(prompt) + sampling_params.max_tokens > max_model_len:
        raise InvalidRequest(
            f"prompt ({len(prompt)}) plus max_tokens ({sampling_params.max_tokens}) "
            f"is over the {max_model_len}-token context"
        )
    if sampling_params.kv_transfer_params is not None:
        problem = check_kv_transfer_params(sampling_params.kv_transfer_params, kv_transfer)
        if problem:
            raise InvalidRequest(problem)


def load_tokenizer(model: str) -> PreTrainedTokenizerFast:
    """Detokenization steps tokenizers' DecodeStream, so only a fast tokenizer will do."""
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(f"{model} has no fast tokenizer (a tokenizer.json), which detokenization needs")
    return tokenizer


class _StepProfiler:
    """torch.profiler over a window of engine steps, set by the LEAN_PROFILE_* variables.

    CPU only by default, since kineto refuses CUDA activity from the engine thread.
    """

    def __init__(self, out_dir: str, skip: int, steps: int, cuda: bool):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if cuda:
            activities.append(ProfilerActivity.CUDA)
        self.profile = profile(
            activities=activities,
            schedule=schedule(skip_first=skip, wait=0, warmup=1, active=steps, repeat=1),
            on_trace_ready=self._export,
        )
        self.started = False
        self.exported = False
        self.closed = False

    @classmethod
    def from_env(cls) -> "_StepProfiler | None":
        if not envs.LEAN_PROFILE_DIR:
            return None
        return cls(envs.LEAN_PROFILE_DIR, envs.LEAN_PROFILE_SKIP, envs.LEAN_PROFILE_STEPS, envs.LEAN_PROFILE_CUDA)

    def step(self):
        if self.started is False:
            self.profile.start()  # lazy, so kineto skips model load and warmup
            self.started = True
        self.profile.step()

    def _export(self, prof):
        path = os.path.join(self.out_dir, f"step-loop-{os.getpid()}.json")
        prof.export_chrome_trace(path)
        self.exported = True
        print(f"wrote step-loop profile to {path}", flush=True)

    def close(self):
        """Flush from the calling thread; must be the thread that ran step()."""
        if not self.started or self.closed:
            return
        self.closed = True
        try:
            self.profile.stop()  # flushes the window, partial or complete
        except RuntimeError:
            pass  # the window already completed and saved mid-run
        if not self.exported:
            print(
                f"warning: step-loop profile captured nothing in {self.out_dir}; "
                f"lower LEAN_PROFILE_SKIP below the run's step count",
                flush=True,
            )


@dataclass(slots=True)
class _InFlight:
    """A launched step, waiting for its tokens."""

    output: SchedulerOutput
    rows: list[LaunchedRow]
    pending: SampledTokens


class LLMEngine:
    def __init__(self, model, detokenize: bool = True, **kwargs):
        """detokenize=False leaves text empty, for a client that detokenizes itself, as an engine core's does."""
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        Sequence.enable_prefix_caching = config.enable_prefix_caching
        Sequence.hash_algo = config.prefix_caching_hash_algo
        self.tokenizer = load_tokenizer(config.model)  # before workers spawn, so a bad one fails fast
        config.eos = self.tokenizer.eos_token_id
        self.ps = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            process = ctx.Process(target=ModelRunner, args=(config, i))
            process.start()
            self.ps.append(process)
        self.model_runner = ModelRunner(config, 0)
        self.scheduler = Scheduler(config)
        self.metrics = Metrics()
        self.output_processor = OutputProcessor(self.tokenizer) if detokenize else None
        self.in_flight: _InFlight | None = None
        self.profiler = _StepProfiler.from_env()
        atexit.register(self.exit)

    def exit(self):
        if self.profiler is not None:
            self.profiler.close()
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self, prompt: str | list[int], sampling_params: SamplingParams, request_id: str | None = None
    ) -> str:
        token_ids: list[int] = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        config = self.config
        validate_request(
            token_ids, sampling_params, config.hf_config.vocab_size, config.max_model_len, config.kv_transfer
        )
        seq = Sequence(token_ids, sampling_params, request_id)
        try:
            self.scheduler.add(seq)  # before the detokenizer, so a refused request leaves nothing behind
        except QueueFull:
            self.metrics.record_rejected()
            raise
        self.metrics.record_received()
        if self.output_processor is not None:
            self.output_processor.add_request(seq.request_id, token_ids, seq.skip_special_tokens)
        return seq.request_id

    def abort_request(self, request_id: str, reason: str = "abort") -> bool:
        """reason is "stop" when the server ends it on a stop string, which counts as a finish."""
        seq = self.scheduler.seqs.get(request_id)
        aborted = self.scheduler.abort(request_id, reason)
        if self.output_processor is not None:
            self.output_processor.abort_request(request_id)
        if aborted:
            assert seq is not None
            self.metrics.record_aborted(self._dropped(seq))
        return aborted

    def step(self) -> tuple[list[RequestOutput], int, int]:
        """Launch one step and drain the previous one; async_scheduling launches first so the two overlap."""
        started = perf_counter()
        draining, self.in_flight = self.in_flight, None
        if self.config.async_scheduling:
            output = self._launch()
            stepped = self._reconcile(draining)
        else:
            stepped = self._reconcile(draining)
            output = self._launch()
        outputs = [self._output(seq) for seq in stepped]
        # A request dropped right after reconcile already has its final output.
        outputs += [self._dropped(seq) for seq in output.dropped if seq not in stepped]
        # attributed to the step this call launched
        self.metrics.record_step(
            self.scheduler,
            output,
            outputs,
            perf_counter() - started,
            self.model_runner.step_kind,
        )
        if self.output_processor is not None:
            with record_function("detokenize"):
                outputs = self.output_processor.process_outputs(outputs)
        if self.profiler is not None:
            self.profiler.step()
        return outputs, output.num_prefill_tokens, output.num_decode_tokens

    def _launch(self) -> SchedulerOutput:
        with record_function("schedule"):
            output = self.scheduler.schedule()
        if output.kv_connector_metadata is not None:
            with record_function("kv_connector"):
                # Every step, scheduled or not, so loads start and finished transfers come back while idle.
                kv_output = self.model_runner.call("kv_connector_step", output.kv_connector_metadata)
                self.scheduler.update_from_kv_connector_output(kv_output)
        if output:
            with record_function("launch"):
                pending = self.model_runner.call("run", output.scheduled)
                rows = self.scheduler.advance(output.scheduled)
            self.in_flight = _InFlight(output, rows, pending)
        return output

    def _reconcile(self, draining: "_InFlight | None") -> list[Sequence]:
        if draining is None:
            return []
        with record_function("await_tokens"):
            token_ids = draining.pending.tolist()
        with record_function("reconcile"):
            return self.scheduler.reconcile(draining.rows, token_ids)

    def _output(self, seq: Sequence) -> RequestOutput:
        return RequestOutput(
            request_id=seq.request_id,
            token_ids=[seq.last_token],
            finished=seq.is_finished,
            finish_reason=seq.finish_reason,
            metrics=seq.metrics() if seq.is_finished else None,
            kv_transfer_params=seq.kv_transfer_result,
        )

    def _dropped(self, seq: Sequence) -> RequestOutput:
        """Finished by the scheduler without ever sampling, so there is no token."""
        return RequestOutput(
            request_id=seq.request_id,
            token_ids=[],
            finished=True,
            finish_reason=seq.finish_reason,
            metrics=seq.metrics(),
        )

    def is_finished(self):
        return self.in_flight is None and self.scheduler.is_finished()

    def generate(
        self,
        prompts: collections.abc.Sequence[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Each prompt's {"text", "token_ids"}, in order."""
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        request_ids = [self.add_request(prompt, sp) for prompt, sp in zip(prompts, sampling_params)]
        collected: dict[str, dict] = {request_id: {"text": "", "token_ids": []} for request_id in request_ids}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            step_outputs, num_prefill_tokens, num_decode_tokens = self.step()
            elapsed = perf_counter() - t
            if num_prefill_tokens:
                prefill_throughput = num_prefill_tokens / elapsed
            if num_decode_tokens:
                decode_throughput = num_decode_tokens / elapsed
            pbar.set_postfix(
                {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
            )
            for output in step_outputs:
                collected[output.request_id]["text"] += output.text
                collected[output.request_id]["token_ids"] += output.token_ids
                if output.finished:
                    pbar.update(1)
        pbar.close()
        return [collected[request_id] for request_id in request_ids]
