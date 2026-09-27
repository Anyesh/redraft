import statistics
from collections import defaultdict


def edit_distance(a: str, b: str) -> float:
    """Character Levenshtein distance divided by the longer length."""
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def _group(rows: list[dict], verdicts: dict[str, dict], judged: bool) -> dict:
    speedups = [
        r["baseline"]["done"]["wall_ms"] / r["redraft"]["done"]["wall_ms"] for r in rows
    ]
    pinned = [(p, r["redraft"]["text"]) for r in rows for p in r["pinned"]]
    kept = sum(p.rstrip() in {x.rstrip() for x in t.split("\n")} for p, t in pinned)
    return {
        "units": len(rows),
        "median_speedup": round(statistics.median(speedups), 3),
        "median_reused": round(
            statistics.median(r["redraft"]["done"].get("reused", 0.0) for r in rows), 3
        ),
        "pinned_survival": round(kept / len(pinned), 3) if pinned else None,
        "median_edit_distance": round(
            statistics.median(
                edit_distance(r["redraft"]["text"], r["baseline"]["text"]) for r in rows
            ),
            3,
        ),
        "failed": (
            sum(verdicts.get(r["unit_id"], {}).get("failed", False) for r in rows)
            if judged
            else None
        ),
    }


def summarize(results: list[dict], verdicts: dict[str, dict], judged: bool) -> dict:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_kind[r["document_kind"]].append(r)
    failures = [
        {
            "unit_id": uid,
            "new_errors": [c["claim"] for c in v["new_errors"]],
            "lost_facts": [c["claim"] for c in v["lost_facts"]],
        }
        for uid, v in verdicts.items()
        if v.get("failed")
    ]
    return {
        "judged": judged,
        "overall": _group(results, verdicts, judged),
        "by_kind": {k: _group(rows, verdicts, judged) for k, rows in sorted(by_kind.items())},
        "failures": failures if judged else [],
    }
