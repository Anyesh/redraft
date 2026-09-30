import asyncio
import hashlib

import httpx
import pytest

from daemon.app import create_app
from daemon.auth import TokenTable
from daemon.config import Settings

from .fakes import FakeLlamaServer, char_ids, parse_sse

SVC = {"Authorization": "Bearer svc"}
SID = "acme/doc-1/summary"
URL = f"/v1/sessions/{SID}"


def tokens_table(tmp_path) -> TokenTable:
    def line(token, scope):
        return f"sha256:{hashlib.sha256(token.encode()).hexdigest()} {scope}\n"

    path = tmp_path / "tokens"
    path.write_text(line("svc", "*") + line("acme-only", "acme"))
    return TokenTable.from_file(path)


@pytest.fixture
async def make(tmp_path):
    lifespans, clients = [], []

    async def factory(fake: FakeLlamaServer | None = None, **overrides):
        fake = fake or FakeLlamaServer()
        settings = Settings(**{"slots": 2, **overrides})
        app = create_app(
            settings, transport=fake.transport(), tokens=tokens_table(tmp_path)
        )
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        lifespans.append(ctx)
        http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=SVC
        )
        clients.append(http)
        return http, fake

    yield factory

    for http in clients:
        await http.aclose()
    for ctx in lifespans:
        await ctx.__aexit__(None, None, None)


def open_body(**overrides) -> dict:
    return {
        "instruction": "Summarize.",
        "sources": [{"name": "transcript", "text": "we ship friday"}],
        **overrides,
    }


def events_named(events, name):
    return [data for n, data in events if n == name]


def assert_spans_tile(old: str, new: str, spans: list[dict]) -> None:
    old_at = new_at = 0
    for span in spans:
        assert span["old"][0] == old_at and span["new"][0] == new_at
        old_at, new_at = span["old"][1], span["new"][1]
        if span["kind"] == "held":
            assert old[slice(*span["old"])] == new[slice(*span["new"])]
        else:
            assert span["text"] == new[slice(*span["new"])]
    assert (old_at, new_at) == (len(old), len(new))


async def test_healthz_reports_engine_and_slots_without_auth(make):
    client, _ = await make()
    resp = await client.get("/healthz", headers={"Authorization": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == {
        "name": "qwen2.5-3b-instruct-q4_k_m",
        "resident": True,
        "redraft": True,
    }
    assert body["slots"] == {"total": 2, "busy": 0, "queued": 0}
    assert body["version"] == "0.2.0"


async def test_healthz_is_503_while_the_engine_is_not_resident(make):
    fake = FakeLlamaServer()
    client, _ = await make(fake)
    fake.resident = False
    resp = await client.get("/healthz")
    assert resp.status_code == 503
    assert resp.json()["status"] == "down"
    assert resp.json()["model"]["resident"] is False


async def test_healthz_is_degraded_when_slot_counts_disagree(make):
    client, _ = await make(slots=3)
    assert (await client.get("/healthz")).json()["status"] == "degraded"


async def test_missing_token_is_401_and_foreign_tenant_is_403(make):
    client, _ = await make()
    resp = await client.put(URL, json=open_body(), headers={"Authorization": ""})
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}
    resp = await client.put(
        "/v1/sessions/globex/d/s",
        json=open_body(),
        headers={"Authorization": "Bearer acme-only"},
    )
    assert resp.status_code == 403
    assert resp.json() == {"error": "forbidden_tenant"}


async def test_bad_session_id_is_400(make):
    client, _ = await make()
    resp = await client.put("/v1/sessions/acme/d/s%20p", json=open_body())
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad_session_id"


async def test_validation_errors_use_the_error_shape(make):
    client, _ = await make()
    resp = await client.put(URL, json={"sources": []})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_request"


async def test_open_adopts_derived_text_and_get_returns_state(make):
    client, _ = await make()
    resp = await client.put(URL, json=open_body(derived="- ship friday"))
    assert resp.json() == {"session_id": SID, "revision": 1, "derived_tokens": 13}
    state = (await client.get(URL)).json()
    assert state["derived"] == "- ship friday"
    assert state["sources"] == [{"name": "transcript", "text": "we ship friday"}]
    assert state["revision"] == 1


async def test_refresh_of_unknown_section_is_404(make):
    client, _ = await make()
    resp = await client.post(f"{URL}/refresh", json={})
    assert resp.status_code == 404
    assert resp.json() == {"error": "unknown_session"}


async def test_first_refresh_without_a_draft_is_plain_generation(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    resp = await client.post(f"{URL}/refresh", json={})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.content)
    assert events[0][0] == "open"
    done = events_named(events, "done")[0]
    assert done["mode"] == "baseline"
    assert done["kind"] == "rederive"
    assert done["reused"] == 0.0
    assert done["revision"] == 2
    assert done["prompt_ms"] == 7.5
    spans = events_named(events, "span")
    assert_spans_tile("", fake.baseline_text, spans)
    assert (await client.get(URL)).json()["derived"] == fake.baseline_text


async def test_source_edit_rederives_with_the_old_draft_and_streams_spans(make):
    client, fake = await make()
    old = "- ship friday"
    await client.put(URL, json=open_body(derived=old))
    fake.redraft_edits = {7: "MON", 8: "", 9: "", 10: "", 11: "", 12: ""}
    resp = await client.post(
        f"{URL}/refresh",
        json={
            "base_revision": 1,
            "edits": [
                {
                    "target": "source",
                    "source": "transcript",
                    "start": 0,
                    "end": 1,
                    "lines": ["we ship monday"],
                }
            ],
        },
    )
    events = parse_sse(resp.content)
    body = fake.completion_bodies[-1]
    assert body["redraft_stabilize"]["old_output"] == char_ids(old)
    assert "we ship monday" in fake.templates[-1]
    assert "Current draft" not in fake.templates[-1]
    new = "".join(d["text"] for d in events_named(events, "delta"))
    assert new == "- ship MON"
    assert_spans_tile(old, new, events_named(events, "span"))
    done = events_named(events, "done")[0]
    assert done["mode"] == "redraft"
    assert done["kind"] == "rederive"
    assert done["reused"] == pytest.approx(7 / 10)
    assert done["reused_chars"] == pytest.approx(7 / 10)
    assert done["revision"] == 3
    assert done["template_version"] == 1


async def test_derived_edit_revises_from_the_edited_draft(make):
    client, fake = await make()
    await client.put(URL, json=open_body(derived="- a\n- b", pinned=["- a"]))
    resp = await client.post(
        f"{URL}/refresh",
        json={
            "edits": [{"target": "derived", "start": 1, "end": 2, "lines": ["- B!"]}]
        },
    )
    done = events_named(parse_sse(resp.content), "done")[0]
    assert done["kind"] == "revise"
    assert fake.completion_bodies[-1]["redraft_stabilize"]["old_output"] == char_ids(
        "- a\n- B!"
    )
    assert "## Current draft\n- a\n- B!" in fake.templates[-1]
    assert done["pinned_missing"] == []
    resp = await client.post(f"{URL}/refresh", json={})
    assert events_named(parse_sse(resp.content), "done")[0]["kind"] == "rederive"


async def test_missing_pinned_line_is_reported(make):
    client, fake = await make()
    await client.put(URL, json=open_body(derived="- a\n- b", pinned=["- b"]))
    fake.redraft_edits = {6: "c"}
    resp = await client.post(f"{URL}/refresh", json={})
    assert events_named(parse_sse(resp.content), "done")[0]["pinned_missing"] == ["- b"]


async def test_baseline_flag_forces_plain_generation(make):
    client, fake = await make()
    await client.put(URL, json=open_body(derived="- x"))
    resp = await client.post(f"{URL}/refresh", json={"baseline": True})
    assert events_named(parse_sse(resp.content), "done")[0]["mode"] == "baseline"
    assert "redraft_stabilize" not in fake.completion_bodies[-1]


async def test_stale_revision_is_409_and_nothing_applies(make):
    client, _ = await make()
    await client.put(URL, json=open_body(derived="- x"))
    resp = await client.post(
        f"{URL}/edits",
        json={
            "base_revision": 5,
            "edits": [{"target": "derived", "start": 0, "end": 1, "lines": ["y"]}],
        },
    )
    assert resp.status_code == 409
    assert resp.json() == {"error": "stale_revision", "revision": 1}
    assert (await client.get(URL)).json()["derived"] == "- x"


async def test_edits_endpoint_applies_without_generating(make):
    client, fake = await make()
    await client.put(URL, json=open_body(derived="- x"))
    resp = await client.post(
        f"{URL}/edits",
        json={
            "base_revision": 1,
            "edits": [{"target": "derived", "start": 0, "end": 1, "lines": ["- y"]}],
        },
    )
    assert resp.json() == {"revision": 2}
    assert fake.completion_bodies == [] or all(
        b["prompt"] == [0] for b in fake.completion_bodies
    )


async def test_out_of_range_edit_is_422_bad_range(make):
    client, _ = await make()
    await client.put(URL, json=open_body())
    resp = await client.post(
        f"{URL}/edits",
        json={
            "edits": [
                {
                    "target": "source",
                    "source": "transcript",
                    "start": 3,
                    "end": 4,
                    "lines": [],
                }
            ]
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "bad_range"


async def test_prompt_over_the_slot_context_is_413(make):
    fake = FakeLlamaServer()
    fake.n_ctx = 600
    client, _ = await make(fake)
    long_source = [{"name": "transcript", "text": "x" * 200}]
    await client.put(URL, json=open_body(max_tokens=512, sources=long_source))
    resp = await client.post(f"{URL}/refresh", json={})
    assert resp.status_code == 413
    body = resp.json()
    assert body["error"] == "too_large"
    assert body["limit"] == 88


async def test_section_returns_to_its_slot(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    first = events_named(
        parse_sse((await client.post(f"{URL}/refresh", json={})).content), "done"
    )[0]
    second = events_named(
        parse_sse((await client.post(f"{URL}/refresh", json={})).content), "done"
    )[0]
    assert second["slot"] == first["slot"]
    assert second["slot_reused"] is True
    assert fake.completion_bodies[-1]["id_slot"] == first["slot"]


async def test_full_queue_is_429_with_retry_after(make):
    client, fake = await make(slots=1, queue_depth=0)
    fake.total_slots = 1
    await client.put(URL, json=open_body())
    await client.put("/v1/sessions/acme/doc-1/other", json=open_body())
    fake.completion_delay = 0.3
    slow = asyncio.create_task(client.post(f"{URL}/refresh", json={}))
    await asyncio.sleep(0.1)
    resp = await client.post("/v1/sessions/acme/doc-1/other/refresh", json={})
    assert resp.status_code == 429
    assert resp.json()["error"] == "busy"
    assert resp.json()["reason"] == "queue_full"
    assert "retry-after" in resp.headers
    await slow


async def test_newer_refresh_supersedes_the_running_one(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    fake.completion_delay = 0.3
    first = asyncio.create_task(client.post(f"{URL}/refresh", json={}))
    await asyncio.sleep(0.1)
    fake.completion_delay = 0.0
    second = await client.post(f"{URL}/refresh", json={})
    first_events = parse_sse((await first).content)
    assert first_events[-1] == ("cancelled", {"reason": "superseded"})
    assert events_named(parse_sse(second.content), "done")


async def test_delete_removes_the_section(make):
    client, _ = await make()
    await client.put(URL, json=open_body())
    assert (await client.delete(URL)).json()["deleted"] is True
    assert (await client.get(URL)).status_code == 404


async def test_delete_of_an_absent_section_is_idempotent(make):
    client, _ = await make()
    await client.put(URL, json=open_body())
    assert (await client.delete(URL)).json()["deleted"] is True
    again = await client.delete(URL)
    assert again.status_code == 200
    assert again.json()["deleted"] is False


async def open_sections(client, *ids):
    for sid in ids:
        assert (
            await client.put(f"/v1/sessions/{sid}", json=open_body())
        ).status_code == 200


async def alive(client, *ids):
    return [
        sid
        for sid in ids
        if (await client.get(f"/v1/sessions/{sid}")).status_code == 200
    ]


async def test_delete_by_document_prefix_drops_only_that_document(make):
    client, _ = await make()
    ids = [
        "acme/doc-1/a",
        "acme/doc-1/b",
        "acme/doc-10/a",
        "acme/doc-2/a",
        "globex/doc-1/a",
    ]
    await open_sections(client, *ids)
    resp = await client.delete("/v1/sessions", params={"prefix": "acme/doc-1/"})
    assert resp.json()["deleted"] == 2
    assert await alive(client, *ids) == ids[2:]


async def test_delete_by_tenant_prefix_drops_every_section_of_the_tenant(make):
    client, _ = await make()
    ids = ["acme/doc-1/a", "acme/doc-2/a", "globex/doc-1/a"]
    await open_sections(client, *ids)
    resp = await client.delete("/v1/sessions", params={"prefix": "acme/"})
    assert resp.json()["deleted"] == 2
    assert await alive(client, *ids) == ["globex/doc-1/a"]
    again = await client.delete("/v1/sessions", params={"prefix": "acme/"})
    assert again.json()["deleted"] == 0


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"prefix": ""},
        {"prefix": "acme"},
        {"prefix": "acme/doc-1"},
        {"prefix": "acme/doc-1/a/"},
        {"prefix": "/"},
        {"prefix": "acme//"},
        {"prefix": "acme/d%20/"},
    ],
)
async def test_bad_prefix_is_400_and_deletes_nothing(make, params):
    client, _ = await make()
    await open_sections(client, SID)
    resp = await client.delete("/v1/sessions", params=params)
    assert resp.status_code == 400
    assert resp.json() == {"error": "bad_prefix"}
    assert await alive(client, SID) == [SID]


async def test_prefix_delete_is_scoped_to_the_callers_tenant(make):
    client, _ = await make()
    await open_sections(client, SID, "globex/doc-1/a")
    scoped = {"Authorization": "Bearer acme-only"}
    forbidden = await client.delete(
        "/v1/sessions", params={"prefix": "globex/"}, headers=scoped
    )
    assert (forbidden.status_code, forbidden.json()) == (
        403,
        {"error": "forbidden_tenant"},
    )
    anonymous = await client.delete(
        "/v1/sessions", params={"prefix": "acme/"}, headers={"Authorization": ""}
    )
    assert anonymous.status_code == 401
    assert len(await alive(client, SID, "globex/doc-1/a")) == 2
    own = await client.delete(
        "/v1/sessions", params={"prefix": "acme/"}, headers=scoped
    )
    assert own.json()["deleted"] == 1


async def test_prefix_delete_mid_stream_frees_the_slot_and_ends_the_stream(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    fake.completion_delay = 0.3
    running = asyncio.create_task(client.post(f"{URL}/refresh", json={}))
    await asyncio.sleep(0.1)
    resp = await client.delete("/v1/sessions", params={"prefix": "acme/doc-1/"})
    assert resp.json()["deleted"] == 1
    events = parse_sse((await running).content)
    assert events[-1] == ("cancelled", {"reason": "superseded"})
    assert (await client.get("/healthz")).json()["slots"]["busy"] == 0


async def test_done_reports_prompt_and_total_tokens(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    done = events_named(
        parse_sse((await client.post(f"{URL}/refresh", json={})).content), "done"
    )[0]
    assert done["prompt_tokens"] > 0
    assert done["total_tokens"] == done["prompt_tokens"] + done["tokens"]


async def refresh_once(client, sid):
    resp = await client.post(f"/v1/sessions/{sid}/refresh", json={})
    return events_named(parse_sse(resp.content), "done")[0]["slot"]


async def test_delete_erases_the_slot_that_last_served_the_section(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    slot = await refresh_once(client, SID)
    resp = await client.delete(URL)
    assert resp.json() == {"deleted": True, "slots_erased": 1, "slots_unerased": []}
    assert fake.erased_slots == [slot]
    assert (await client.get("/healthz")).json()["slots"]["busy"] == 0


async def test_delete_of_a_section_that_never_ran_erases_nothing(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    resp = await client.delete(URL)
    assert resp.json()["slots_erased"] == 0
    assert fake.erased_slots == []


async def test_slot_is_kept_while_a_live_section_was_the_last_to_use_it(make):
    client, fake = await make(slots=1)
    other = "acme/doc-1/other"
    await open_sections(client, SID, other)
    await refresh_once(client, SID)
    await refresh_once(client, other)
    assert (await client.delete(URL)).json()["slots_erased"] == 0
    assert fake.erased_slots == []
    assert (await client.delete(f"/v1/sessions/{other}")).json()["slots_erased"] == 1
    assert fake.erased_slots == [0]


async def test_prefix_delete_erases_every_slot_of_the_dropped_sections(make):
    client, fake = await make()
    ids = ["acme/doc-1/a", "acme/doc-1/b"]
    await open_sections(client, *ids)
    used = {await refresh_once(client, sid) for sid in ids}
    assert len(used) == 2
    resp = await client.delete("/v1/sessions", params={"prefix": "acme/"})
    assert resp.json() == {"deleted": 2, "slots_erased": 2, "slots_unerased": []}
    assert set(fake.erased_slots) == used


async def test_delete_mid_stream_erases_after_the_cancelled_refresh_lets_go(make):
    client, fake = await make(slots=1)
    await client.put(URL, json=open_body())
    fake.completion_delay = 0.3
    running = asyncio.create_task(client.post(f"{URL}/refresh", json={}))
    await asyncio.sleep(0.1)
    resp = await client.delete(URL)
    assert resp.json()["slots_erased"] == 1
    assert fake.erased_slots == [0]
    await running
    assert (await client.get("/healthz")).json()["slots"]["busy"] == 0


async def test_failed_erase_is_reported_and_retried_by_the_next_delete(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    slot = await refresh_once(client, SID)
    fake.erase_status = 500
    resp = await client.delete(URL)
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "slots_erased": 0, "slots_unerased": [slot]}
    assert (await client.get("/healthz")).json()["slots"]["busy"] == 0
    fake.erase_status = 200
    retry = await client.delete(URL)
    assert retry.json() == {"deleted": False, "slots_erased": 1, "slots_unerased": []}
    assert fake.erased_slots == [slot]


async def test_engine_without_the_patch_degrades_to_baseline(make):
    client, _ = await make(FakeLlamaServer(redraft_available=False))
    await client.put(URL, json=open_body(derived="- x"))
    resp = await client.post(f"{URL}/refresh", json={})
    assert events_named(parse_sse(resp.content), "done")[0]["mode"] == "baseline"
    assert (await client.get("/healthz")).json()["status"] == "degraded"


async def test_reopen_continues_the_revision_so_stale_clients_get_409(make):
    client, _ = await make()
    await client.put(URL, json=open_body(derived="- x"))
    resp = await client.put(URL, json=open_body(derived="- y"))
    assert resp.json()["revision"] == 2
    stale = await client.post(f"{URL}/edits", json={"base_revision": 1, "edits": []})
    assert stale.status_code == 409


async def test_a_refused_refresh_applies_none_of_its_edits(make):
    fake = FakeLlamaServer()
    fake.n_ctx = 600
    client, _ = await make(fake)
    await client.put(URL, json=open_body(max_tokens=512, derived="- x"))
    resp = await client.post(
        f"{URL}/refresh",
        json={
            "edits": [
                {
                    "target": "source",
                    "source": "transcript",
                    "start": 0,
                    "end": 1,
                    "lines": ["y" * 200],
                }
            ]
        },
    )
    assert resp.status_code == 413
    state = (await client.get(URL)).json()
    assert state["revision"] == 1
    assert state["sources"][0]["text"] == "we ship friday"


async def test_delete_mid_stream_frees_the_slot_and_ends_the_stream(make):
    client, fake = await make()
    await client.put(URL, json=open_body())
    fake.completion_delay = 0.3
    running = asyncio.create_task(client.post(f"{URL}/refresh", json={}))
    await asyncio.sleep(0.1)
    assert (await client.delete(URL)).status_code == 200
    events = parse_sse((await running).content)
    assert events[-1] == ("cancelled", {"reason": "superseded"})
    assert (await client.get("/healthz")).json()["slots"]["busy"] == 0
