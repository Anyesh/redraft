import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from redraft_client import RedraftClient, collect_baseline, collect_redraft

from daemon.config import Settings
from daemon.probe import probe_redraft_support
from daemon.sessions import DropStaleRunner, SessionStore


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


class CreateSessionRequest(BaseModel):
    instruction: str
    context: str
    max_tokens: int = 512


class RefreshRequest(BaseModel):
    context: str
    instruction: str | None = None


def unknown_session() -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "unknown_session"})


def _make_lifespan(settings: Settings, transport: httpx.AsyncBaseTransport | None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        http = httpx.AsyncClient(
            base_url=settings.redraft_base,
            timeout=httpx.Timeout(600.0),
            transport=transport,
        )
        client = RedraftClient(http, settings.redraft_model)
        app.state.settings = settings
        app.state.client = client
        app.state.store = SessionStore(cap=settings.session_cap)
        app.state.runner = DropStaleRunner()
        app.state.redraft_available = await probe_redraft_support(client)
        yield
        await http.aclose()

    return lifespan


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(lifespan=_make_lifespan(settings, transport))

    @app.get("/v1/health")
    async def health():
        client: RedraftClient = app.state.client
        try:
            resp = await client.http.get("/health")
            upstream_ok = resp.status_code == 200
        except httpx.HTTPError:
            upstream_ok = False
        return {
            "upstream_ok": upstream_ok,
            "redraft_available": app.state.redraft_available,
        }

    @app.post("/v1/sessions")
    async def create_session(req: CreateSessionRequest):
        client: RedraftClient = app.state.client
        store: SessionStore = app.state.store
        session_id, _ = store.create(req.instruction, req.context, req.max_tokens)

        async def gen() -> AsyncIterator[str]:
            template = await client.apply_template(req.context, req.instruction)
            prompt_ids = await client.tokenize(template)
            t0 = time.perf_counter()
            events = []
            async for event in client.stream_baseline(prompt_ids, req.max_tokens):
                events.append(event)
                yield sse({"content": event.get("content", "")})
            result = collect_baseline(events)
            wall_s = time.perf_counter() - t0
            store.update(session_id, text=result["text"], tokens=result["tokens"])
            yield sse({"session_id": session_id, "mode": "seed", "wall_s": wall_s})

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        session = app.state.store.get(session_id)
        if session is None:
            return unknown_session()
        return {
            "instruction": session.instruction,
            "max_tokens": session.max_tokens,
            "last_text": session.last_text,
            "updated_at": session.updated_at,
        }

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        await app.state.runner.cancel(session_id)
        if not app.state.store.delete(session_id):
            return unknown_session()
        return {"deleted": True}

    @app.post("/v1/sessions/{session_id}/refresh")
    async def refresh_session(session_id: str, req: RefreshRequest):
        store: SessionStore = app.state.store
        session = store.get(session_id)
        if session is None:
            return unknown_session()

        client: RedraftClient = app.state.client
        runner: DropStaleRunner = app.state.runner
        queue: asyncio.Queue = asyncio.Queue()

        async def run() -> None:
            try:
                instruction = req.instruction or session.instruction
                template = await client.apply_template(req.context, instruction)
                prompt_ids = await client.tokenize(template)
                t0 = time.perf_counter()
                events = []
                if app.state.redraft_available and session.tokens:
                    eos = await client.eos_ids()
                    async for event in client.stream_redraft(
                        prompt_ids, session.tokens, eos, session.max_tokens
                    ):
                        events.append(event)
                        await queue.put({"content": event.get("content", "")})
                    result = collect_redraft(events)
                    mode = "redraft"
                    extra = {
                        "held_fraction": result["held_fraction"],
                        "divergences": result["divergences"],
                    }
                else:
                    async for event in client.stream_baseline(
                        prompt_ids, session.max_tokens
                    ):
                        events.append(event)
                        await queue.put({"content": event.get("content", "")})
                    result = collect_baseline(events)
                    mode = "baseline"
                    extra = {}
                wall_s = time.perf_counter() - t0
                store.update(
                    session_id,
                    text=result["text"],
                    tokens=result["tokens"],
                    context=req.context,
                    instruction=instruction,
                )
                await queue.put(
                    {"session_id": session_id, "mode": mode, "wall_s": wall_s, **extra}
                )
            finally:
                await queue.put(None)

        await runner.replace(session_id, run())

        async def gen() -> AsyncIterator[str]:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield sse(item)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
