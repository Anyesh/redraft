import asyncio

from fastapi.testclient import TestClient

from app.main import app, merge_tagged


async def _fake_stream(events: list[dict], delay: float = 0.0):
    for event in events:
        if delay:
            await asyncio.sleep(delay)
        yield event


async def test_merge_tagged_tags_and_forwards_every_event():
    streams = {
        "baseline": _fake_stream([{"content": "a"}, {"content": "b", "stop": True}]),
        "redraft": _fake_stream([{"content": "x", "stop": True}]),
    }
    events = [e async for e in merge_tagged(streams)]

    assert len(events) == 3
    assert all("arm" in e and "t" in e for e in events)
    assert all(not e.get("done") for e in events)
    by_arm = {"baseline": [], "redraft": []}
    for e in events:
        by_arm[e["arm"]].append(e)
    assert [e["content"] for e in by_arm["baseline"]] == ["a", "b"]
    assert [e["content"] for e in by_arm["redraft"]] == ["x"]


async def test_merge_tagged_empty_streams_yields_nothing():
    events = [e async for e in merge_tagged({"baseline": _fake_stream([])})]
    assert events == []


def test_scenarios_endpoint_serves_seeded_content():
    with TestClient(app) as client:
        resp = client.get("/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert "edits" in data and "batch" in data
        assert any(e["id"] == "doc-fix-punctuation" for e in data["edits"])
