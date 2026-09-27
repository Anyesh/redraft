import asyncio
import copy
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from redraft_client import RedraftClient

from daemon.auth import AuthError, TokenTable
from daemon.config import Settings
from daemon.edits import BadRange
from daemon.engine import EngineState
from daemon.jobs import track
from daemon.prompts import TEMPLATE_VERSION
from daemon.refresh import Generation, ensure_draft, prepare
from daemon.sections import Change, SectionEdit, UnknownSource, apply_change
from daemon.sessions import (
    BadSessionId,
    DropStaleRunner,
    Session,
    SessionStore,
    Source,
    parse_session_id,
)
from daemon.slots import Busy, SlotPool

VERSION = version("redraftd")
log = logging.getLogger("redraftd")
RETRY_AFTER_MS = 2000
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class DaemonError(Exception):
    def __init__(self, status: int, body: dict, headers: dict | None = None):
        super().__init__(body.get("error"))
        self.status = status
        self.body = body
        self.headers = headers


class SourceIn(BaseModel):
    name: str = Field(min_length=1)
    text: str


class EditIn(BaseModel):
    target: Literal["source", "derived"]
    source: str | None = None
    start: int
    end: int
    lines: list[str]


class OpenRequest(BaseModel):
    instruction: str
    sources: list[SourceIn]
    derived: str | None = None
    pinned: list[str] = []
    max_tokens: int | None = Field(default=None, ge=1, le=8192)


class EditsRequest(BaseModel):
    base_revision: int | None = None
    edits: list[EditIn] = []
    pinned: list[str] | None = None


class RefreshRequest(EditsRequest):
    sources: list[SourceIn] | None = None
    instruction: str | None = None
    baseline: bool = False


def sse(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def to_sources(items: list[SourceIn]) -> list[Source]:
    names = [s.name for s in items]
    if len(set(names)) != len(names):
        raise DaemonError(
            422, {"error": "invalid_request", "detail": "source names must be unique"}
        )
    return [Source(s.name, s.text) for s in items]


def to_change(req: EditsRequest) -> Change:
    edits = [
        SectionEdit(e.target, e.source, e.start, e.end, e.lines) for e in req.edits
    ]
    change = Change(edits=edits, pinned=req.pinned)
    if isinstance(req, RefreshRequest):
        change.sources = to_sources(req.sources) if req.sources is not None else None
        change.instruction = req.instruction
    return change


def resolve_tokens(settings: Settings, tokens: TokenTable | None) -> TokenTable:
    if tokens is not None:
        return tokens
    if settings.tokens_file:
        return TokenTable.from_file(settings.tokens_file)
    if settings.no_auth and settings.host in LOOPBACK_HOSTS:
        return TokenTable.open()
    raise RuntimeError(
        "redraftd needs REDRAFTD_TOKENS_FILE; --no-auth is only allowed on a loopback bind"
    )


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    tokens: TokenTable | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    token_table = resolve_tokens(settings, tokens)
    engine = EngineState()
    store = SessionStore(settings.session_cap, settings.tenant_session_cap)
    runner = DropStaleRunner()
    pool = SlotPool(
        settings.slots, settings.queue_depth, settings.queue_timeout_ms / 1000
    )
    # one lock per open section; a lock is created only for sections that exist
    # (or are being opened) so unknown ids cannot grow this map
    locks: dict[str, asyncio.Lock] = {}
    state: dict[str, RedraftClient] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        http = httpx.AsyncClient(
            base_url=settings.redraft_base,
            timeout=httpx.Timeout(120.0, connect=2.0),
            transport=transport,
        )
        state["client"] = RedraftClient(http, settings.redraft_model)
        await engine.refresh(state["client"])
        yield
        await http.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(DaemonError)
    async def on_daemon_error(_: Request, exc: DaemonError):
        return JSONResponse(exc.body, status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError):
        detail = json.loads(json.dumps(exc.errors(), default=str))
        return JSONResponse(
            {"error": "invalid_request", "detail": detail}, status_code=422
        )

    def guard(request: Request, tenant: str, document: str, section: str) -> str:
        session_id = f"{tenant}/{document}/{section}"
        try:
            parse_session_id(session_id)
            token_table.authorize(request.headers.get("authorization"), tenant)
        except BadSessionId:
            raise DaemonError(400, {"error": "bad_session_id"}) from None
        except AuthError as exc:
            raise DaemonError(exc.status, {"error": exc.code}) from None
        return session_id

    def require(session_id: str) -> Session:
        session = store.get(session_id)
        if session is None:
            raise DaemonError(404, {"error": "unknown_session"})
        return session

    def lock_for(session_id: str) -> asyncio.Lock:
        return locks.setdefault(session_id, asyncio.Lock())

    def apply(session: Session, req: EditsRequest) -> None:
        if req.base_revision is not None and req.base_revision != session.revision:
            raise DaemonError(
                409, {"error": "stale_revision", "revision": session.revision}
            )
        try:
            apply_change(session, to_change(req))
        except BadRange as exc:
            raise DaemonError(422, {"error": "bad_range", "detail": str(exc)}) from None
        except UnknownSource as exc:
            raise DaemonError(
                422, {"error": "bad_range", "detail": f"unknown source {exc}"}
            ) from None

    async def ready_client() -> RedraftClient:
        client = state["client"]
        if not engine.ready:
            await engine.refresh(client)
        if not engine.ready:
            raise DaemonError(503, {"error": "engine_unavailable"})
        return client

    @app.get("/healthz")
    async def healthz():
        await engine.refresh(state["client"])
        if not engine.ready:
            status = "down"
        elif not engine.redraft or engine.total_slots != settings.slots:
            status = "degraded"
        else:
            status = "ok"
        body = {
            "status": status,
            "version": VERSION,
            "template_version": TEMPLATE_VERSION,
            "model": {
                "name": engine.model_name,
                "resident": engine.resident,
                "redraft": bool(engine.redraft),
            },
            "slots": pool.stats(),
            "sessions": len(store),
        }
        return JSONResponse(body, status_code=503 if status == "down" else 200)

    @app.put("/v1/sessions/{tenant}/{document}/{section}")
    async def open_section(
        tenant: str, document: str, section: str, req: OpenRequest, request: Request
    ):
        session_id = guard(request, tenant, document, section)
        async with lock_for(session_id):
            await runner.cancel(session_id)
            previous = store.get(session_id)
            session = Session(
                session_id=session_id,
                instruction=req.instruction,
                sources=to_sources(req.sources),
                derived=req.derived or "",
                pinned=list(req.pinned),
                max_tokens=req.max_tokens or settings.default_max_tokens,
                slot=previous.slot if previous else None,
                # continuing the count makes a client holding a revision from
                # before the re-open fail the revision check instead of editing
                # text it has never seen
                revision=previous.revision + 1 if previous else 1,
            )
            derived_tokens = None
            if session.derived and engine.ready:
                try:
                    draft = await ensure_draft(state["client"], session)
                    derived_tokens = len(draft.ids)
                except httpx.HTTPError:
                    session.draft = None
            store.put(session)
        return {
            "session_id": session_id,
            "revision": session.revision,
            "derived_tokens": derived_tokens,
        }

    @app.get("/v1/sessions/{tenant}/{document}/{section}")
    async def get_section(tenant: str, document: str, section: str, request: Request):
        session = require(guard(request, tenant, document, section))
        return {
            "session_id": session.session_id,
            "revision": session.revision,
            "instruction": session.instruction,
            "sources": [{"name": s.name, "text": s.text} for s in session.sources],
            "derived": session.derived,
            "pinned": session.pinned,
            "max_tokens": session.max_tokens,
            "updated_at": session.updated_at,
        }

    @app.delete("/v1/sessions/{tenant}/{document}/{section}")
    async def delete_section(
        tenant: str, document: str, section: str, request: Request
    ):
        session_id = guard(request, tenant, document, section)
        require(session_id)
        lock = lock_for(session_id)
        async with lock:
            await runner.cancel(session_id)
            deleted = store.delete(session_id)
        if not lock.locked():
            locks.pop(session_id, None)
        if not deleted:
            raise DaemonError(404, {"error": "unknown_session"})
        return {"deleted": True}

    @app.post("/v1/sessions/{tenant}/{document}/{section}/edits")
    async def edit_section(
        tenant: str, document: str, section: str, req: EditsRequest, request: Request
    ):
        session_id = guard(request, tenant, document, section)
        require(session_id)
        async with lock_for(session_id):
            session = require(session_id)
            await runner.cancel(session_id)
            apply(session, req)
        return {"revision": session.revision}

    @app.post("/v1/sessions/{tenant}/{document}/{section}/refresh")
    async def refresh_section(
        tenant: str, document: str, section: str, req: RefreshRequest, request: Request
    ):
        session_id = guard(request, tenant, document, section)
        require(session_id)
        async with lock_for(session_id):
            # cancel first: a refresh finishing while this one prepares would
            # otherwise write its output over the edits this request carries
            await runner.cancel(session_id)
            # all-or-nothing: work on a copy and commit it only once a slot is
            # held, so a refused refresh leaves the section exactly as it was
            session = copy.deepcopy(require(session_id))
            apply(session, req)
            client = await ready_client()
            try:
                prepared = await prepare(client, engine, session, req.baseline)
            except httpx.HTTPError:
                raise DaemonError(503, {"error": "engine_unavailable"}) from None
            limit = engine.n_ctx - session.max_tokens
            if len(prepared.prompt_ids) > limit:
                raise DaemonError(
                    413,
                    {
                        "error": "too_large",
                        "prompt_tokens": len(prepared.prompt_ids),
                        "limit": limit,
                    },
                )
            queued_at = time.perf_counter()
            try:
                slot = await pool.acquire(session.slot)
            except Busy as exc:
                raise DaemonError(
                    429,
                    {
                        "error": "busy",
                        "reason": exc.reason,
                        "retry_after_ms": RETRY_AFTER_MS,
                    },
                    headers={"Retry-After": str(RETRY_AFTER_MS // 1000)},
                ) from None
            queued_ms = round((time.perf_counter() - queued_at) * 1000, 1)
            slot_reused = session.slot == slot
            session.slot = slot
            store.put(session)
            queue: asyncio.Queue = asyncio.Queue()
            generation = Generation(client, engine, session, prepared, slot)
            opened = {
                "session_id": session_id,
                "revision": session.revision,
                "kind": prepared.kind,
                "slot": slot,
                "slot_reused": slot_reused,
                "queued_ms": queued_ms,
            }
            task = runner.start(session_id, produce(queue, generation, session, opened))
            track(task, queue, lambda: pool.release(slot))
        return StreamingResponse(consume(queue, task), media_type="text/event-stream")

    async def produce(
        queue: asyncio.Queue,
        generation: Generation,
        session: Session,
        opened: dict,
    ) -> None:
        try:
            await queue.put(("open", opened))
            async for event in generation.run():
                await queue.put(event)
            session.derived = generation.text
            session.draft = generation.draft
            session.derived_dirty = False
            session.touch()
            done = {
                "session_id": session.session_id,
                "revision": session.revision,
                "kind": opened["kind"],
                **generation.metrics,
                "queued_ms": opened["queued_ms"],
                "slot": opened["slot"],
                "slot_reused": opened["slot_reused"],
                "template_version": TEMPLATE_VERSION,
            }
            await queue.put(("done", done))
        except httpx.HTTPError as exc:
            log.warning("refresh %s failed: %r", session.session_id, exc)
            queue.put_nowait(("error", {"error": "engine_failed", "detail": str(exc)}))

    async def consume(queue: asyncio.Queue, task: asyncio.Task) -> AsyncIterator[str]:
        try:
            while (item := await queue.get()) is not None:
                yield sse(*item)
        finally:
            # a client that hangs up mid-stream must not keep holding an engine slot
            if not task.done():
                task.cancel()

    return app
