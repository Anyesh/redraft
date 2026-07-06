import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

CATEGORY_ORDER = ["doc_summary", "fact_answer", "code_review"]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _excluded_ids(cal_csv_path: Path) -> set[str]:
    """Case ids whose own free-decode baseline already fails this model: they
    can't be blamed on a rule, so calibration must drop them before counting
    regressions or medians (the m4 exclusion rule this driver must replicate).
    """
    texts_path = cal_csv_path.with_suffix(".texts.json")
    entries = json.loads(texts_path.read_text())
    return {e["id"] for e in entries if e.get("new_free_correct") == "FAIL"}


def calibration_frontier(cal_paths: list[Path]) -> dict:
    """Per rule: regression count on each model (after excluding that model's own
    baseline failures) and median held fraction on the surviving cases. A rule is
    a zero-regression candidate only if every model it was run on shows zero.
    """
    per_model: dict[str, dict[str, list[dict]]] = {}
    for path in cal_paths:
        model = path.stem
        rows = _read_csv(path)
        excluded = _excluded_ids(path)
        by_rule: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["id"] in excluded:
                continue
            by_rule[row["rule"]].append(row)
        per_model[model] = by_rule

    rules = sorted({rule for by_rule in per_model.values() for rule in by_rule})
    frontier = {}
    for rule in rules:
        per_model_stats = {}
        zero_regression = True
        for model, by_rule in per_model.items():
            rows = by_rule.get(rule)
            if not rows:
                continue
            regressions = sum(1 for r in rows if r["correct"] == "FAIL")
            held = median(float(r["held_fraction"]) for r in rows)
            per_model_stats[model] = {
                "regressions": regressions,
                "n": len(rows),
                "median_held": round(held, 4),
            }
            if regressions > 0:
                zero_regression = False
        frontier[rule] = {
            "per_model": per_model_stats,
            "zero_regression": zero_regression,
            "median_held_overall": round(
                median(s["median_held"] for s in per_model_stats.values()), 4
            )
            if per_model_stats
            else None,
        }
    return frontier


def print_calibration_frontier(frontier: dict) -> None:
    print("=== calibration frontier ===")
    for rule, stats in sorted(frontier.items()):
        tag = "ZERO-REGRESSION" if stats["zero_regression"] else "regresses"
        print(f"{rule}: {tag}, median_held_overall={stats['median_held_overall']}")
        for model, s in sorted(stats["per_model"].items()):
            print(
                f"    {model}: regressions={s['regressions']}/{s['n']} "
                f"median_held={s['median_held']}"
            )

    candidates = sorted(
        (r for r, s in frontier.items() if s["zero_regression"]),
        key=lambda r: frontier[r]["median_held_overall"],
        reverse=True,
    )
    print("\nzero-regression candidates, best held first:")
    for rule in candidates:
        print(f"  {rule}: median_held_overall={frontier[rule]['median_held_overall']}")
    if not candidates:
        print("  (none)")


def wall_verdict(wall_rows: list[dict], arm_label: str) -> dict:
    """Four speedup rungs per case: measured (wall clock), server-only (server-
    reported prompt+predicted ms, strips HTTP/harness overhead), wasted-adjusted
    (server-only with chunk-overshoot decode time discounted out, a heuristic
    optimistic floor assuming uniform per-token decode cost), and analytic
    (forward-pass counting, batching free). wasted-adjusted is informational
    only; the DoD branch below is decided on server-only, which charges wasted
    decode time in full rather than assuming it away.
    """
    by_case_arm: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in wall_rows:
        by_case_arm[(row["id"], row["arm"])].append(row)

    per_case = {}
    for case_id in sorted({row["id"] for row in wall_rows}):
        base_rows = by_case_arm.get((case_id, "baseline"))
        arm_rows = by_case_arm.get((case_id, arm_label))
        if not base_rows or not arm_rows:
            continue

        base_wall = median(float(r["wall_s"]) for r in base_rows)
        arm_wall = median(float(r["wall_s"]) for r in arm_rows)
        base_server = median(
            float(r["server_prompt_ms"]) + float(r["server_predicted_ms"])
            for r in base_rows
        )
        arm_prompt = median(float(r["server_prompt_ms"]) for r in arm_rows)
        arm_predicted = median(float(r["server_predicted_ms"]) for r in arm_rows)
        arm_server = arm_prompt + arm_predicted

        arm_tokens = median(int(r["tokens"]) for r in arm_rows)
        arm_wasted = median(int(r["wasted_tokens"]) for r in arm_rows)
        total = arm_tokens + arm_wasted
        waste_frac = arm_wasted / total if total else 0.0
        arm_server_adjusted = arm_prompt + arm_predicted * (1 - waste_frac)

        analytic = median(float(r["analytic_speedup"]) for r in arm_rows)

        per_case[case_id] = {
            "category": arm_rows[0]["category"],
            "measured": base_wall / arm_wall if arm_wall else float("nan"),
            "server_only": base_server / arm_server if arm_server else float("nan"),
            "wasted_adjusted": base_server / arm_server_adjusted
            if arm_server_adjusted
            else float("nan"),
            "analytic": analytic,
        }
    return per_case


def print_wall_verdict(per_case: dict, arm_label: str) -> dict:
    print(f"\n=== wall verdict: {arm_label} vs baseline ===")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for stats in per_case.values():
        by_category[stats["category"]].append(stats)

    category_medians = {}
    categories = [c for c in CATEGORY_ORDER if c in by_category] + sorted(
        c for c in by_category if c not in CATEGORY_ORDER
    )
    for category in categories:
        rows = by_category[category]
        n = len(rows)
        faster = sum(1 for r in rows if r["measured"] > 1.0)
        medians = {
            metric: round(median(r[metric] for r in rows), 4)
            for metric in ("measured", "server_only", "wasted_adjusted", "analytic")
        }
        category_medians[category] = medians
        print(
            f"{category}: measured={medians['measured']}x ({faster}/{n} faster) "
            f"server_only={medians['server_only']}x "
            f"wasted_adjusted={medians['wasted_adjusted']}x "
            f"analytic={medians['analytic']}x"
        )
    return category_medians


def print_dod_branch(category_medians: dict) -> None:
    doc = category_medians.get("doc_summary")
    print("\n=== m5 DoD branch ===")
    if doc is None:
        print("no doc_summary rows in the wall CSV; cannot decide a branch")
        return

    measured, server_only = doc["measured"], doc["server_only"]
    if measured > 1.0:
        print(
            f"COMMIT: doc_summary measured speedup {measured}x > 1.0x under a "
            "zero-regression rule -> greenlight the in-engine port"
        )
    elif server_only < 1.0:
        print(
            f"KILL: doc_summary server-only ceiling {server_only}x < 1.0x even "
            "after both levers -> narrow product scope to fact/short-answer and "
            "cosmetic-edit surfaces"
        )
    elif server_only >= 1.2:
        print(
            f"MIDDLE: doc_summary measured {measured}x <= 1.0x but server-only "
            f"ceiling {server_only}x >= 1.2x -> the in-engine port is the one "
            "remaining lever; decide with the user"
        )
    else:
        print(
            f"MIDDLE (weak): doc_summary measured {measured}x <= 1.0x and "
            f"server-only ceiling {server_only}x is between 1.0x and 1.2x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="m5 verdict: calibration frontier and wall-clock DoD branch"
    )
    parser.add_argument(
        "--cal",
        type=Path,
        nargs="*",
        default=[],
        help="calibration CSVs, one per model (e.g. results/m5_cal_3b.csv ...)",
    )
    parser.add_argument("--wall", type=Path, default=None, help="14B wall-clock CSV")
    parser.add_argument(
        "--arm",
        default=None,
        help="stabilized arm label in --wall to compare against baseline "
        "(the selected rule's spec string); required if --wall is given",
    )
    args = parser.parse_args()

    if args.cal:
        frontier = calibration_frontier(args.cal)
        print_calibration_frontier(frontier)

    if args.wall:
        if not args.arm:
            parser.error("--arm is required when --wall is given")
        wall_rows = _read_csv(args.wall)
        per_case = wall_verdict(wall_rows, args.arm)
        category_medians = print_wall_verdict(per_case, args.arm)
        print_dod_branch(category_medians)


if __name__ == "__main__":
    main()
