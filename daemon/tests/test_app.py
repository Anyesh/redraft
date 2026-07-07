import asyncio

import httpx
import pytest

from daemon.app import create_app
from daemon.config import Settings

from .fakes import FakeLlamaServer, parse_sse


@pytest.fixture
async def client_factory():
    lifespans = []
    clients = []

    async def make(fake: FakeLlamaServer, settings: Settings | None = None):
        settings = settings or Settings(session_cap=10)
        app = create_app(settings, transport=fake.transport())
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        lifespans.append(ctx)
        http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        clients.append(http)
        return http

    yield make

    for http in clients:
        await http.aclose()
    for ctx in lifespans:
        await ctx.__aexit__(None, None, None)


async def test_create_session_streams_seed_and_stores_it(client_factory):
    fake = FakeLlamaServer()
    client = await client_factory(fake)

    resp = await client.post(
        "/v1/sessions", json={"instruction": "summarize", "context": "doc"}
    )
    assert resp.status_code == 200
    events = parse_sse(resp.content)
    final = events[-1]
    assert final["mode"] == "seed"
    assert "wall_s" in final
    session_id = final["session_id"]

    resp = await client.get(f"/v1/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["instruction"] == "summarize"


async def test_refresh_unknown_session_returns_404(client_factory):
    fake = FakeLlamaServer()
    client = await client_factory(fake)

    resp = await client.post(
        "/v1/sessions/does-not-exist/refresh", json={"context": "edited"}
    )
    assert resp.status_code == 404
    assert resp.json() == {"error": "unknown_session"}


async def test_refresh_carries_emitted_tokens_forward_as_next_old_output(
    client_factory,
):
    fake = FakeLlamaServer(redraft_available=True)
    client = await client_factory(fake)

    seed = await client.post(
        "/v1/sessions", json={"instruction": "summarize", "context": "doc v1"}
    )
    session_id = parse_sse(seed.content)[-1]["session_id"]

    refresh1 = await client.post(
        f"/v1/sessions/{session_id}/refresh", json={"context": "doc v2"}
    )
    assert parse_sse(refresh1.content)[-1]["mode"] == "redraft"

    refresh2 = await client.post(
        f"/v1/sessions/{session_id}/refresh", json={"context": "doc v3"}
    )
    assert parse_sse(refresh2.content)[-1]["mode"] == "redraft"

    # the startup probe and the seed call precede these; the two refreshes'
    # redraft_stabilize bodies are the last two, and the second refresh's
    # old_output must be the first refresh's redraft_emitted
    redraft_bodies = [b for b in fake.completion_bodies if "redraft_stabilize" in b]
    assert len(redraft_bodies) == 3
    assert redraft_bodies[-1]["redraft_stabilize"]["old_output"] == [201, 202]


async def test_degraded_mode_falls_back_to_baseline_when_server_unpatched(
    client_factory,
):
    fake = FakeLlamaServer(redraft_available=False)
    client = await client_factory(fake)

    seed = await client.post(
        "/v1/sessions", json={"instruction": "summarize", "context": "doc"}
    )
    session_id = parse_sse(seed.content)[-1]["session_id"]

    health = await client.get("/v1/health")
    assert health.json()["redraft_available"] is False

    refresh = await client.post(
        f"/v1/sessions/{session_id}/refresh", json={"context": "doc edited"}
    )
    assert parse_sse(refresh.content)[-1]["mode"] == "baseline"


async def test_delete_session_then_get_is_404(client_factory):
    fake = FakeLlamaServer()
    client = await client_factory(fake)

    seed = await client.post(
        "/v1/sessions", json={"instruction": "summarize", "context": "doc"}
    )
    session_id = parse_sse(seed.content)[-1]["session_id"]

    resp = await client.delete(f"/v1/sessions/{session_id}")
    assert resp.json() == {"deleted": True}

    resp = await client.get(f"/v1/sessions/{session_id}")
    assert resp.status_code == 404


async def test_drop_stale_cancels_the_earlier_refresh_stream(client_factory):
    fake = FakeLlamaServer(redraft_available=True)
    client = await client_factory(fake)

    seed = await client.post(
        "/v1/sessions", json={"instruction": "summarize", "context": "doc v1"}
    )
    session_id = parse_sse(seed.content)[-1]["session_id"]

    fake.next_completion_delay = 0.3
    task1 = asyncio.create_task(
        client.post(f"/v1/sessions/{session_id}/refresh", json={"context": "doc v2"})
    )
    await asyncio.sleep(0.05)

    resp2 = await client.post(
        f"/v1/sessions/{session_id}/refresh", json={"context": "doc v3"}
    )
    resp1 = await task1

    assert resp1.status_code == 200
    assert parse_sse(resp1.content) == []

    assert resp2.status_code == 200
    assert parse_sse(resp2.content)[-1]["mode"] == "redraft"
