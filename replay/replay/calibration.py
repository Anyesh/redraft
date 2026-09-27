import json
from collections.abc import Awaitable, Callable
from pathlib import Path

CASES = Path(__file__).with_name("calibration.jsonl")

Diff = Callable[[str, str, list], Awaitable[dict]]


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def calibrate(diff: Diff, cases: list[dict]) -> tuple[bool, list[dict]]:
    """Run the fact diff on hand-labeled pairs. Verdicts on real data count only
    if every case comes out as labeled."""
    rows = []
    for case in cases:
        verdict = await diff(case["baseline"], case["redraft"], case["sources"])
        got = "fail" if verdict["failed"] else "pass"
        rows.append({"case_id": case["case_id"], "expect": case["expect"], "got": got,
                     "ok": got == case["expect"], "verdict": verdict})
    return all(r["ok"] for r in rows), rows
