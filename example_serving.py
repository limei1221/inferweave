"""example.py's two prompts, over HTTP with the official OpenAI SDK.

uv run lean-vllm serve ~/workspace/huggingface/Qwen3-0.6B --served-model-name qwen
uv run python example_serving.py
"""

import asyncio
import os
from time import perf_counter

from openai import AsyncOpenAI

BASE_URL = os.getenv("LEAN_VLLM_URL", "http://127.0.0.1:8000/v1")


async def ask(client, model: str, prompt: str):
    # Only what the engine implements: top_p, seed, n > 1 and penalties are a 400.
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.6,
        stream=True,
    )
    start, ttft, text = perf_counter(), None, ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            ttft = ttft or perf_counter() - start
            text += delta
    return text, ttft, perf_counter() - start


async def main():
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    async with AsyncOpenAI(api_key="unused", base_url=BASE_URL) as client:  # no auth is checked
        model = (await client.models.list()).data[0].id  # whatever --served-model-name says
        outputs = await asyncio.gather(*(ask(client, model, prompt) for prompt in prompts))

    for prompt, (text, ttft, total) in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {text!r}")
        print(f"ttft {ttft:.2f}s, total {total:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
