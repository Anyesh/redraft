import json

import httpx

from replay.bundle import Unit
from replay.run import Redraftd, ReplayError, run_bundle, run_unit


class FakeRedraftd:
    def __init__(self, terminal: str = "done"):
        self.requests: list[tuple[str, str, dict | None]] = []
        self.terminal = terminal

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        if request.method in ("PUT", "DELETE"):
            return httpx.Response(200, json={"revision": 1})
        text = "baseline text" if body.get("baseline") else "redraft text"
        name = self.terminal
        stream = (
            'event: open\ndata: {"slot": 0}\n\n'
            f'event: delta\ndata: {json.dumps({"text": text, "kind": "new"})}\n\n'
            f'event: {name}\ndata: {json.dumps({"wall_ms": 10.0, "mode": "x"})}\n\n'
        )
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    def client(self) -> Redraftd:
        http = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handler), base_url="http://r"
        )
        return Redraftd(http, max_tokens=64)


def rederive_unit(uid="refresh:1") -> Unit:
    return Unit(
        unit_id=uid, kind="rederive", document_kind="plan", instruction="Summarize.",
        sources_before=[("t", "friday")], derived_before="- friday", pinned=[],
        sources_after=[("t", "monday")], reference={"wall_ms": 5},
    )


def timed_bodies(fake):
    refreshes = [b for m, p, b in fake.requests if p.endswith("/refresh")]
    return refreshes[1::2]


async def test_each_side_runs_from_a_restored_warm_state():
    fake = FakeRedraftd()
    record = await run_unit(fake.client(), rederive_unit(), redraft_first=True)
    kinds = [(m, p.rsplit("/", 1)[-1] if p.endswith("refresh") else m) for m, p, _ in fake.requests]
    one_side = [("PUT", "PUT"), ("POST", "refresh"), ("PUT", "PUT"), ("POST", "refresh")]
    assert kinds == one_side * 3 + [("DELETE", "DELETE")]
    warmups = [b for m, p, b in fake.requests if p.endswith("/refresh")][0::2]
    assert all(b == {"baseline": True} for b in warmups)
    puts = [b for m, p, b in fake.requests if m == "PUT"]
    assert all(b["derived"] == "- friday" for b in puts)
    assert all(b["sources"] == [{"name": "t", "text": "friday"}] for b in puts)
    assert record["baseline"]["text"] == "baseline text"
    assert record["redraft"]["text"] == "redraft text"
    assert record["baseline_repeat"]["text"] == "baseline text"
    assert record["sources"] == [["t", "monday"]]


async def test_order_follows_redraft_first():
    fake = FakeRedraftd()
    await run_unit(fake.client(), rederive_unit(), redraft_first=False)
    assert [b["baseline"] for b in timed_bodies(fake)] == [True, False, True]
    assert all(b["sources"] == [{"name": "t", "text": "monday"}] for b in timed_bodies(fake))


async def test_revise_units_send_the_persons_edit_as_line_ranges():
    fake = FakeRedraftd()
    unit = Unit(
        unit_id="correction:1", kind="revise", document_kind="reply",
        instruction="Reply.", sources_before=[("t", "x")], derived_before="a\nb",
        pinned=["a"], derived_after="a\nB",
    )
    record = await run_unit(fake.client(), unit, redraft_first=True)
    body = timed_bodies(fake)[0]
    assert body["edits"] == [{"target": "derived", "start": 1, "end": 2, "lines": ["B"]}]
    assert record["sources"] == [["t", "x"]]


async def test_a_stream_without_done_is_an_error():
    fake = FakeRedraftd(terminal="cancelled")
    try:
        await run_unit(fake.client(), rederive_unit(), redraft_first=True)
    except ReplayError as exc:
        assert "cancelled" in str(exc)
    else:
        raise AssertionError("expected ReplayError")


async def test_run_bundle_appends_jsonl_and_skips_units_already_done(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(json.dumps({"unit_id": "refresh:1"}) + "\n")
    fake = FakeRedraftd()
    await run_bundle(fake.client(), [rederive_unit("refresh:1"), rederive_unit("refresh:2")], out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["unit_id"] for r in rows] == ["refresh:1", "refresh:2"]
