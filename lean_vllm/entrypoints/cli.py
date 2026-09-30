"""`lean-vllm serve <model>`, with engine flags generated from `Config`."""

import argparse
from dataclasses import MISSING, fields

from lean_vllm.config import Config

# Not flags: the positional, and what the engine reads from the checkpoint.
INTERNAL = {"model", "hf_config", "eos"}


def add_engine_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("engine")
    for field in fields(Config):
        default = field.default
        if field.name in INTERNAL or default is MISSING or default is None:
            continue    # no default to take a type from
        flag = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            group.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            group.add_argument(flag, type=type(default), default=default)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="lean-vllm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("model", help="path to a local model directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default=None, help="the id reported by /v1/models")
    serve.add_argument("--log-level", default="info")
    add_engine_args(serve)
    args = parser.parse_args(argv)

    from lean_vllm.entrypoints.server import run    # imports fastapi, which is the `serve` extra

    engine_kwargs = {
        field.name: getattr(args, field.name)
        for field in fields(Config)
        if field.name not in INTERNAL and hasattr(args, field.name)
    }
    run(args, engine_kwargs)


if __name__ == "__main__":
    main()
