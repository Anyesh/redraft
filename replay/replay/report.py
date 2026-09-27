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


MIN_SPEEDUP = 1.3
MIN_UNITS = 10


def _survival(rows: list[dict], side: str) -> float | None:
    pinned = [(p, r[side]["text"]) for r in rows for p in r["pinned"]]
    if not pinned:
        return None
    kept = sum(p.rstrip() in {x.rstrip() for x in t.split("\n")} for p, t in pinned)
    return round(kept / len(pinned), 3)


def _count(rows: list[dict], verdicts: dict[str, dict], key: str | None) -> int:
    total = 0
    for r in rows:
        v = verdicts.get(r["unit_id"], {})
        total += bool((v.get(key) or {}).get("failed") if key else v.get("failed"))
    return total


def _group(rows: list[dict], verdicts: dict[str, dict], judged: bool) -> dict:
    speedups = [
        r["baseline"]["done"]["wall_ms"] / r["redraft"]["done"]["wall_ms"] for r in rows
    ]
    return {
        "units": len(rows),
        "median_speedup": round(statistics.median(speedups), 3),
        "median_reused": round(
            statistics.median(r["redraft"]["done"].get("reused", 0.0) for r in rows), 3
        ),
        "pinned_survival": _survival(rows, "redraft"),
        "baseline_pinned_survival": _survival(rows, "baseline"),
        "median_edit_distance": round(
            statistics.median(
                edit_distance(r["redraft"]["text"], r["baseline"]["text"]) for r in rows
            ),
            3,
        ),
        "failed": _count(rows, verdicts, None) if judged else None,
        "control_failed": _count(rows, verdicts, "control") if judged else None,
    }


def _ship(group: dict, judged: bool, min_units: int) -> dict:
    """A refresh mode ships only if it is faster, no less factual than the
    engine's own run-to-run drift, and keeps pinned lines as well as baseline."""
    blockers = []
    if not judged:
        blockers.append("unjudged")
    if group["units"] < min_units:
        blockers.append(f"units {group['units']} < {min_units}")
    if group["median_speedup"] < MIN_SPEEDUP:
        blockers.append(f"median_speedup {group['median_speedup']} < {MIN_SPEEDUP}")
    if judged and group["failed"] > group["control_failed"]:
        blockers.append(f"failed {group['failed']} > control_failed {group['control_failed']}")
    kept, base = group["pinned_survival"], group["baseline_pinned_survival"]
    if kept is not None and base is not None and kept < base:
        blockers.append(f"pinned_survival {kept} < baseline {base}")
    return {**group, "ships": not blockers, "blockers": blockers}


def summarize(
    results: list[dict],
    verdicts: dict[str, dict],
    judged: bool,
    min_units: int = MIN_UNITS,
) -> dict:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_kind[r["document_kind"]].append(r)
        by_mode[r["kind"]].append(r)
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
        "by_mode": {
            k: _ship(_group(rows, verdicts, judged), judged, min_units)
            for k, rows in sorted(by_mode.items())
        },
        "failures": failures if judged else [],
    }
