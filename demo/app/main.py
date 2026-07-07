import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from redraft_client import (
    RedraftClient,
    collect_baseline,
    collect_redraft,
    cumulative_stats,
)

REDRAFT_BASE = os.environ.get("REDRAFT_BASE", "http://127.0.0.1:8080")
REDRAFT_MODEL = os.environ.get("REDRAFT_MODEL", "default")
DEFAULT_MAX_TOKENS = 512

APP_DIR = Path(__file__).parent
SCENARIOS = json.loads((APP_DIR / "scenarios.json").read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    http = httpx.AsyncClient(base_url=REDRAFT_BASE, timeout=httpx.Timeout(600.0))
    app.state.client = RedraftClient(http, REDRAFT_MODEL)
    yield
    await http.aclose()


app = FastAPI(lifespan=lifespan)


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def merge_tagged(streams: dict[str, AsyncIterator[dict]]) -> AsyncIterator[dict]:
    """Fan in several tagged async streams, yielding events as they arrive (not in
    round-robin order), so a live race view shows whichever arm is actually ahead.
    """
    queue: asyncio.Queue = asyncio.Queue()
    start = time.perf_counter()

    async def pump(arm: str, stream: AsyncIterator[dict]) -> None:
        async for event in stream:
            await queue.put({"arm": arm, "t": time.perf_counter() - start, **event})
        await queue.put({"arm": arm, "t": time.perf_counter() - start, "done": True})

    tasks = [asyncio.create_task(pump(arm, stream)) for arm, stream in streams.items()]
    remaining = len(tasks)
    try:
        while remaining:
            item = await queue.get()
            if item.get("done"):
                remaining -= 1
                continue
            yield item
    finally:
        await asyncio.gather(*tasks)


async def generate_once(
    client: RedraftClient, prompt_ids: list[int], max_tokens: int
) -> dict:
    events = [e async for e in client.stream_baseline(prompt_ids, max_tokens)]
    return collect_baseline(events)


class DraftRequest(BaseModel):
    context: str
    question: str
    max_tokens: int = DEFAULT_MAX_TOKENS


class EditRequest(BaseModel):
    context: str
    question: str
    old_output_tokens: list[int]
    max_tokens: int = DEFAULT_MAX_TOKENS


@app.get("/health")
async def health():
    client: RedraftClient = app.state.client
    resp = await client.http.get("/health")
    return resp.json()


@app.get("/scenarios")
async def scenarios():
    return SCENARIOS


@app.post("/draft")
async def draft(req: DraftRequest):
    client: RedraftClient = app.state.client
    template = await client.apply_template(req.context, req.question)
    prompt_ids = await client.tokenize(template)
    return await generate_once(client, prompt_ids, req.max_tokens)


@app.post("/edit")
async def edit(req: EditRequest):
    client: RedraftClient = app.state.client
    template = await client.apply_template(req.context, req.question)
    prompt_ids = await client.tokenize(template)
    eos = await client.eos_ids()

    async def gen() -> AsyncIterator[str]:
        streams = {
            "baseline": client.stream_baseline(prompt_ids, req.max_tokens),
            "redraft": client.stream_redraft(
                prompt_ids, req.old_output_tokens, eos, req.max_tokens
            ),
        }
        async for event in merge_tagged(streams):
            yield sse(event)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/batch")
async def batch():
    client: RedraftClient = app.state.client
    scenario = SCENARIOS["batch"]
    question = scenario["question"]
    step_contexts = scenario["steps"]

    async def gen() -> AsyncIterator[str]:
        eos = await client.eos_ids()
        prev_tokens: list[int] | None = None
        step_results = []

        for i, context in enumerate(step_contexts):
            template = await client.apply_template(context, question)
            prompt_ids = await client.tokenize(template)

            if prev_tokens is None:
                seed = await generate_once(client, prompt_ids, DEFAULT_MAX_TOKENS)
                prev_tokens = seed["tokens"]
                yield sse({"step": i, "seed": True, "text": seed["text"]})
                continue

            t0 = time.perf_counter()
            baseline_events = [
                e async for e in client.stream_baseline(prompt_ids, DEFAULT_MAX_TOKENS)
            ]
            baseline_wall_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            redraft_events = [
                e
                async for e in client.stream_redraft(
                    prompt_ids, prev_tokens, eos, DEFAULT_MAX_TOKENS
                )
            ]
            redraft_wall_s = time.perf_counter() - t0

            redraft = collect_redraft(redraft_events)
            prev_tokens = redraft["tokens"]

            step = {
                "step": i,
                "baseline_wall_s": baseline_wall_s,
                "redraft_wall_s": redraft_wall_s,
                "held_fraction": redraft["held_fraction"],
                "divergences": redraft["divergences"],
                "text": redraft["text"],
                "baseline_text": collect_baseline(baseline_events)["text"],
            }
            step_results.append(step)
            yield sse(step)

        yield sse({"final": True, **cumulative_stats(step_results)})

    return StreamingResponse(gen(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="static")
