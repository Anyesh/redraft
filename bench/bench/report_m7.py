import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def median_by(rows, key_wall="wall_s"):
    buckets = defaultdict(list)
    meta = {}
    for r in rows:
        k = (r["id"], r["arm"])
        buckets[k].append(float(r[key_wall]))
        meta[k] = (r["category"], r["correct"])
    med = {k: statistics.median(v) for k, v in buckets.items()}
    return med, meta


def arm_kind(arm):
    if arm == "baseline":
        return "baseline"
    if arm.startswith("redraft:"):
        return "redraft"
    return "stabilized"


def run(csv_path, canary_path):
    rows = list(csv.DictReader(Path(csv_path).open()))
    med, meta = median_by(rows)

    ids = sorted({r["id"] for r in rows})
    canary_ids = set()
    if Path(canary_path).exists():
        canary_ids = {c["id"] for c in json.loads(Path(canary_path).read_text())}

    arms = sorted({r["arm"] for r in rows})
    base_arm = next(a for a in arms if arm_kind(a) == "baseline")
    redraft_arm = next((a for a in arms if arm_kind(a) == "redraft"), None)
    stab_arm = next((a for a in arms if arm_kind(a) == "stabilized"), None)

    per_cat_redraft = defaultdict(list)
    per_cat_stab = defaultdict(list)
    regressions = []
    rows_out = []
    for cid in ids:
        cat = meta[(cid, base_arm)][0]
        b = med[(cid, base_arm)]
        sr = b / med[(cid, redraft_arm)] if redraft_arm else float("nan")
        ss = b / med[(cid, stab_arm)] if stab_arm else float("nan")
        per_cat_redraft[cat].append(sr)
        if stab_arm:
            per_cat_stab[cat].append(ss)
        base_ok = meta[(cid, base_arm)][1]
        red_ok = meta[(cid, redraft_arm)][1] if redraft_arm else ""
        if red_ok == "FAIL" and base_ok != "FAIL":
            regressions.append((cid, cat, base_ok, red_ok, cid in canary_ids))
        rows_out.append(
            (
                cid,
                cat,
                b,
                med.get((cid, redraft_arm)),
                sr,
                ss,
                red_ok,
                cid in canary_ids,
            )
        )

    print(
        f"{'id':34} {'cat':14} {'base_s':>7} {'redraft_s':>9} "
        f"{'redraft_x':>9} {'client_x':>8} {'correct':>7} canary"
    )
    for cid, cat, b, rs, sr, ss, ok, can in sorted(
        rows_out, key=lambda x: (x[1], x[0])
    ):
        print(
            f"{cid:34} {cat:14} {b:7.3f} {rs:9.3f} {sr:9.3f} {ss:8.3f} "
            f"{ok or 'n/a':>7} {'Y' if can else ''}"
        )

    print("\n=== per-category median speedup (baseline / arm) ===")
    print(f"{'category':16} {'n':>3} {'redraft_x':>10} {'faster':>7} {'client_x':>9}")
    for cat in sorted(per_cat_redraft):
        rr = per_cat_redraft[cat]
        rmed = statistics.median(rr)
        faster = sum(1 for x in rr if x > 1.0)
        cmed = (
            statistics.median(per_cat_stab[cat])
            if per_cat_stab.get(cat)
            else float("nan")
        )
        print(
            f"{cat:16} {len(rr):>3} {rmed:10.3f} {faster:>4}/{len(rr):<2} {cmed:9.3f}"
        )

    print("\n=== correctness regressions (baseline ok, redraft FAIL) ===")
    if regressions:
        for cid, cat, bok, rok, can in regressions:
            print(
                f"  {cid} [{cat}] baseline={bok} redraft={rok} canary={'Y' if can else 'N'}"
            )
    else:
        print("  none")

    print(f"\ncanary ids ({len(canary_ids)}): {sorted(canary_ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/m7_wall_14b.csv")
    ap.add_argument("--canary", default="cases/canary.json")
    args = ap.parse_args()
    run(args.csv, args.canary)


if __name__ == "__main__":
    main()
