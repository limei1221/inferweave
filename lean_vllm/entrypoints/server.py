"""Wires the engine, the app and uvicorn together."""

import uvicorn

from lean_vllm.engine.async_engine import AsyncLLMEngine
from lean_vllm.engine.core_client import AsyncMPClient
from lean_vllm.entrypoints.api_server import build_app


def run(args, engine_kwargs: dict):
    # The engine in its own process, as vLLM's, unless --no-engine-process keeps it on a thread here for the A/B.
    engine_cls = AsyncMPClient if args.engine_process else AsyncLLMEngine
    engine = engine_cls.from_engine_args(args.model, **engine_kwargs)
    app = build_app(engine, args.served_model_name or args.model)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level)
    server = uvicorn.Server(config)
    # CUDA state and TP workers die with the engine, so exit and let a supervisor restart.
    engine.on_death = lambda: setattr(server, "should_exit", True)
    server.run()
