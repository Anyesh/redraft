"""Drives redraftd through each replay unit (contract section 14)."""

import hashlib
import json
from pathlib import Path

import httpx

from replay.bundle import Sources, Unit, line_edits

TENANT = "replay"


class ReplayError(RuntimeError):
    pass


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events, name = [], "message"
    for line in body.splitlines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[len("data:") :].strip())))
    return events


PERSON_EDIT = "person edit"


def person_edit_source(unit: Unit) -> Sources:
    lines = [
        line
        for edit in line_edits(unit.derived_before, unit.derived_after)
        for line in edit["lines"]
    ]
    return [(PERSON_EDIT, "\n".join(lines))] if lines else []


def as_sources(sources: Sources) -> list[dict]:
    return [{"name": name, "text": text} for name, text in sources]


class Redraftd:
    def __init__(self, http: httpx.AsyncClient, max_tokens: int):
        self.http = http
        self.max_tokens = max_tokens

    async def put(self, path: str, body: dict) -> None:
        resp = await self.http.put(path, json=body)
        if resp.status_code != 200:
            raise ReplayError(f"PUT {path}: {resp.status_code} {resp.text[:200]}")

    async def delete(self, path: str) -> None:
        await self.http.delete(path)

    async def refresh(self, path: str, body: dict) -> tuple[str, dict]:
        resp = await self.http.post(f"{path}/refresh", json=body, timeout=None)
        if resp.status_code != 200:
            raise ReplayError(f"refresh {path}: {resp.status_code} {resp.text[:200]}")
        events = parse_sse(resp.text)
        name, data = events[-1] if events else ("empty", {})
        if name != "done":
            raise ReplayError(f"refresh {path} ended with {name}: {data}")
        text = "".join(d["text"] for n, d in events if n == "delta")
        return text, data


def section_path(unit_id: str) -> str:
    digest = hashlib.sha256(unit_id.encode()).hexdigest()[:16]
    return f"/v1/sessions/{TENANT}/{digest}/section"


async def run_unit(client: Redraftd, unit: Unit, redraft_first: bool) -> dict:
    path = section_path(unit.unit_id)
    setup = {
        "instruction": unit.instruction,
        "sources": as_sources(unit.sources_before),
        "derived": unit.derived_before,
        "pinned": unit.pinned,
        "max_tokens": client.max_tokens,
    }
    if unit.kind == "revise":
        change = {"edits": line_edits(unit.derived_before, unit.derived_after)}
        judged_sources = [*unit.sources_before, *person_edit_source(unit)]
    else:
        change = {"sources": as_sources(unit.sources_after)}
        judged_sources = unit.sources_after

    runs = {}
    order = ["redraft", "baseline"] if redraft_first else ["baseline", "redraft"]
    # a second baseline measures the engine's own run-to-run drift, the floor
    # redraft's fact failures are judged against
    for side in [*order, "baseline_repeat"]:
        await client.put(path, setup)
        # the warm-up leaves the before-state prompt cached on the section's slot,
        # as a real previous refresh would; the second PUT restores derived_before
        # as the draft, since the warm-up replaced it
        await client.refresh(path, {"baseline": True})
        await client.put(path, setup)
        text, done = await client.refresh(path, {**change, "baseline": side != "redraft"})
        runs[side] = {"text": text, "done": done}
    await client.delete(path)

    return {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "document_kind": unit.document_kind,
        "instruction": unit.instruction,
        "sources": [list(s) for s in judged_sources],
        "pinned": unit.pinned,
        "redraft_first": redraft_first,
        "baseline": runs["baseline"],
        "redraft": runs["redraft"],
        "baseline_repeat": runs["baseline_repeat"],
        "reference": unit.reference,
    }


def _done_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {json.loads(line)["unit_id"] for line in out.read_text().splitlines() if line}


async def run_bundle(client: Redraftd, units: list[Unit], out: Path) -> None:
    """Replay units in order, appending one JSON line each; units already in
    `out` are skipped so an interrupted window run resumes where it stopped."""
    done = _done_ids(out)
    with out.open("a") as fh:
        for i, unit in enumerate(units):
            if unit.unit_id in done:
                continue
            record = await run_unit(client, unit, redraft_first=i % 2 == 0)
            fh.write(json.dumps(record) + "\n")
            fh.flush()
