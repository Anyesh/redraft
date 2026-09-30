"""Live end-to-end check of a running redraftd in front of a patched llama-server,
optionally recording every exchange as a contract fixture. Not run by pytest,
since it needs a real model.

It opens a meeting-summary section, seeds it, refreshes after a source edit
(rederive) and after a draft edit with a pinned line (revise), provokes a stale
revision and a full queue, and checks the span invariants on every stream: the
spans tile the old and new text, held spans match byte for byte, and the deltas
concatenate to the new text.

Usage:
    REDRAFTD_URL=http://127.0.0.1:8787 REDRAFTD_TOKEN=... \\
        uv run python smoke.py [--record DIR] [--down-url URL]

--down-url points at a redraftd whose engine is unreachable, to record the 503
health body. The queue-full step needs redraftd started with --queue-depth 1.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

TENANT, DOC = "acme", "weekly-sync"
TRANSCRIPT = """Dana: Let's start with the beta. Are we still on for Friday?
Priya: Yes, the release checklist is done except the rollback drill.
Marco: The login bug is the last blocker. I can fix it by Wednesday.
Dana: Good. Priya, you own the checklist and the rollback drill.
Priya: Fine. I'll run the drill Thursday morning.
Dana: Budget review. Finance asked to push it.
Marco: Push it to next month, we won't have usage numbers before then.
Dana: Agreed, budget review moves to next month.
Priya: One more thing, support needs the beta FAQ before launch.
Dana: Marco, can you draft the FAQ? Tuesday is fine.
Marco: Sure, FAQ draft by Tuesday."""
INSTRUCTION = (
    "Summarize this meeting as two markdown lists under the headings "
    "'Decisions' and 'Action items'. Each action item names its owner and date."
)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events, name = [], "message"
    for line in body.splitlines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[len("data:") :].strip())))
    return events


def utf16(text: str) -> bytes:
    return text.encode("utf-16-le")


def check_stream(old: str, events: list[tuple[str, dict]]) -> str:
    names = [n for n, _ in events]
    assert names[0] == "open" and names[-1] == "done", names
    new = "".join(d["text"] for n, d in events if n == "delta")
    old16, new16 = utf16(old), utf16(new)
    old_at = new_at = 0
    for name, span in events:
        if name != "span":
            continue
        assert span["old"][0] == old_at and span["new"][0] == new_at, span
        old_at, new_at = span["old"][1], span["new"][1]
        new_part = new16[2 * span["new"][0] : 2 * span["new"][1]]
        if span["kind"] == "held":
            assert old16[2 * span["old"][0] : 2 * span["old"][1]] == new_part, span
        else:
            assert utf16(span["text"]) == new_part, span
    assert (old_at, new_at) == (len(old16) // 2, len(new16) // 2)
    return new


class Recorder:
    def __init__(self, directory: Path | None):
        self.directory = directory
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    def exchange(self, name: str, request: dict, resp: httpx.Response) -> None:
        if not self.directory:
            return
        keep = {"content-type", "retry-after"}
        record = {
            "request": request,
            "response": {
                "status": resp.status_code,
                "headers": {k: v for k, v in resp.headers.items() if k in keep},
                "body": resp.json(),
            },
        }
        (self.directory / f"{name}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )

    def stream(self, name: str, request: dict, resp: httpx.Response) -> None:
        if not self.directory:
            return
        (self.directory / f"{name}.sse").write_text(resp.text)
        (self.directory / f"{name}.request.json").write_text(
            json.dumps(request, indent=2) + "\n"
        )


async def refresh(http, section: str, body: dict) -> httpx.Response:
    return await http.post(
        f"/v1/sessions/{TENANT}/{DOC}/{section}/refresh", json=body, timeout=600
    )


def report(label: str, events: list[tuple[str, dict]]) -> dict:
    done = events[-1][1]
    spans = [d for n, d in events if n == "span"]
    print(
        f"{label}: mode={done['mode']} kind={done['kind']} reused={done['reused']:.2f} "
        f"reused_chars={done['reused_chars']:.2f} wall_ms={done['wall_ms']} "
        f"prompt_ms={done['prompt_ms']} prompt_tokens={done['prompt_tokens']} "
        f"total_tokens={done['total_tokens']} spans={len(spans)} "
        f"pinned_missing={done['pinned_missing']}"
    )
    return done


async def main(record: Path | None, down_url: str | None) -> None:
    url = os.environ.get("REDRAFTD_URL", "http://127.0.0.1:8787")
    token = os.environ["REDRAFTD_TOKEN"]
    rec = Recorder(record)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=30) as http:
        health = await http.get("/healthz")
        print("healthz:", health.json())
        assert health.json()["status"] == "ok", (
            "engine must be up with the redraft patch"
        )
        rec.exchange("healthz", {"method": "GET", "path": "/healthz"}, health)

        section = f"/v1/sessions/{TENANT}/{DOC}/summary"
        open_body = {
            "instruction": INSTRUCTION,
            "sources": [{"name": "transcript", "text": TRANSCRIPT}],
            "max_tokens": 256,
        }
        opened = await http.put(section, json=open_body)
        assert opened.status_code == 200, opened.text
        rec.exchange(
            "open", {"method": "PUT", "path": section, "json": open_body}, opened
        )

        seed = await refresh(http, "summary", {})
        seed_events = parse_sse(seed.text)
        draft = check_stream("", seed_events)
        report("seed", seed_events)
        rec.stream(
            "refresh_seed",
            {"method": "POST", "path": f"{section}/refresh", "json": {}},
            seed,
        )
        print("--- draft ---\n" + draft)

        lines = TRANSCRIPT.split("\n")
        source_edit = {
            "base_revision": seed_events[-1][1]["revision"],
            "edits": [
                {
                    "target": "source",
                    "source": "transcript",
                    "start": 0,
                    "end": 2,
                    "lines": [
                        "Dana: Let's start with the beta. Are we still on for Friday?",
                        "Priya: No, the rollback drill failed. Let's move the beta to Monday.",
                    ],
                },
                {
                    "target": "source",
                    "source": "transcript",
                    "start": 9,
                    "end": 11,
                    "lines": [
                        "Dana: Lena, can you draft the FAQ? Tuesday is fine.",
                        "Lena: Sure, FAQ draft by Tuesday.",
                    ],
                },
            ],
        }
        assert len(lines) == 11
        resp = await refresh(http, "summary", source_edit)
        events = parse_sse(resp.text)
        new = check_stream(draft, events)
        done = report("rederive", events)
        rec.stream(
            "refresh_rederive",
            {"method": "POST", "path": f"{section}/refresh", "json": source_edit},
            resp,
        )
        print("--- rederived ---\n" + new)

        derived_lines = new.split("\n")
        target = next(i for i, line in enumerate(derived_lines) if "FAQ" in line)
        pinned = derived_lines[0]
        revise_edit = {
            "base_revision": done["revision"],
            "pinned": [pinned],
            "edits": [
                {
                    "target": "derived",
                    "start": target,
                    "end": target + 1,
                    "lines": [
                        "- Lena: publish the beta FAQ on the support site by Tuesday"
                    ],
                },
            ],
        }
        resp = await refresh(http, "summary", revise_edit)
        events = parse_sse(resp.text)
        edited = "\n".join(
            derived_lines[:target]
            + ["- Lena: publish the beta FAQ on the support site by Tuesday"]
            + derived_lines[target + 1 :]
        )
        revised = check_stream(edited, events)
        done = report("revise", events)
        rec.stream(
            "refresh_revise",
            {"method": "POST", "path": f"{section}/refresh", "json": revise_edit},
            resp,
        )
        print("--- revised ---\n" + revised)

        stale = {"base_revision": 1, "edits": []}
        resp = await http.post(f"{section}/edits", json=stale)
        assert resp.status_code == 409, resp.text
        rec.exchange(
            "edits_stale",
            {"method": "POST", "path": f"{section}/edits", "json": stale},
            resp,
        )

        await queue_full(http, rec)
        await purge(http, rec)

    if down_url:
        async with httpx.AsyncClient(base_url=down_url, timeout=30) as http:
            resp = await http.get("/healthz")
            assert resp.status_code == 503, resp.text
            rec.exchange("healthz_down", {"method": "GET", "path": "/healthz"}, resp)
            print("healthz down:", resp.json())
    print("all invariants hold")


async def queue_full(http: httpx.AsyncClient, rec: Recorder) -> None:
    slots = (await http.get("/healthz")).json()["slots"]["total"]
    names = [f"filler-{i}" for i in range(slots + 1)]
    for name in names + ["overflow"]:
        body = {
            "instruction": INSTRUCTION,
            "sources": [{"name": "transcript", "text": TRANSCRIPT}],
            "max_tokens": 256,
        }
        await http.put(f"/v1/sessions/{TENANT}/{DOC}/{name}", json=body)
    fillers = [asyncio.create_task(refresh(http, n, {})) for n in names]
    await asyncio.sleep(1.0)
    path = f"/v1/sessions/{TENANT}/{DOC}/overflow/refresh"
    resp = await http.post(path, json={})
    assert resp.status_code == 429, (resp.status_code, resp.text[:200])
    rec.exchange("queue_full", {"method": "POST", "path": path, "json": {}}, resp)
    print("queue full:", resp.status_code, resp.json(), resp.headers.get("retry-after"))
    await asyncio.gather(*fillers)


async def purge(http: httpx.AsyncClient, rec: Recorder) -> None:
    """Runs last: the prefix delete drops every section the earlier steps made."""
    base = f"/v1/sessions/{TENANT}/{DOC}"
    ran = await http.get(f"{base}/filler-0")
    assert ran.status_code == 200, ran.text

    resp = await http.delete(f"{base}/overflow")
    assert resp.json() == {"deleted": True, "slots_erased": 0, "slots_unerased": []}
    rec.exchange(
        "delete_section", {"method": "DELETE", "path": f"{base}/overflow"}, resp
    )
    resp = await http.delete(f"{base}/overflow")
    assert resp.status_code == 200 and resp.json()["deleted"] is False, resp.text
    rec.exchange(
        "delete_section_absent", {"method": "DELETE", "path": f"{base}/overflow"}, resp
    )

    bad = f"/v1/sessions?prefix={TENANT}/{DOC}"
    resp = await http.delete(bad)
    assert resp.status_code == 400 and resp.json() == {"error": "bad_prefix"}
    rec.exchange("delete_prefix_bad", {"method": "DELETE", "path": bad}, resp)

    path = f"/v1/sessions?prefix={TENANT}/{DOC}/"
    resp = await http.delete(path)
    body = resp.json()
    assert body["deleted"] >= 2 and body["slots_unerased"] == [], body
    assert body["slots_erased"] >= 1, body
    rec.exchange("delete_prefix", {"method": "DELETE", "path": path}, resp)
    resp = await http.delete(path)
    assert resp.json() == {"deleted": 0, "slots_erased": 0, "slots_unerased": []}
    rec.exchange("delete_prefix_none", {"method": "DELETE", "path": path}, resp)
    health = (await http.get("/healthz")).json()
    assert health["slots"]["busy"] == 0 and health["sessions"] == 0, health
    print("purge:", body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path)
    parser.add_argument("--down-url")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.record, args.down_url))
    except AssertionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
