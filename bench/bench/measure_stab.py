import argparse
import csv
import json
from pathlib import Path

from bench.batch_engine import BatchedStepper
from bench.free_align import free_salvage
from bench.rule_specs import parse_rule_specs
from bench.stab_engine import detokenize, eos_token_ids
from bench.stabilize import stab_passes, stabilize

DEFAULT_RULES = ["tau:0.0", "tau:1.5", "tau:3.0"]


def correctness(text: str, case: dict) -> str:
    """Case-insensitive: stabilized decode tends to echo a source document's
    capitalization (e.g. an all-caps status word) more literally than free
    decode's natural prose, so exact-case matching flags stylistic variation
    as a correctness failure even when the answer is semantically right.
    """
    if "expect_new" not in case and "forbid_new" not in case:
        return ""
    text_lower = text.lower()
    ok = all(s.lower() in text_lower for s in case.get("expect_new", [])) and not any(
        s.lower() in text_lower for s in case.get("forbid_new", [])
    )
    return "pass" if ok else "FAIL"


def load_cases(cases_dir: Path, start: int = 0, count: int | None = None) -> list[dict]:
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        cases.extend(json.loads(path.read_text()))
    end = start + count if count is not None else None
    return cases[start:end]


def run(
    base: str,
    model: str,
    cases_dir: Path,
    out_path: Path,
    max_tokens: int,
    anchor_len: int,
    window: int,
    reanchor_cost: int,
    rule_specs: list[str],
    horizon: int,
    start: int = 0,
    count: int | None = None,
) -> None:
    cases = load_cases(cases_dir, start=start, count=count)

    rules = parse_rule_specs(rule_specs)
    eos_ids = eos_token_ids(base, model)
    rows = []
    texts = []
    for case in cases:
        # both baselines and every rule arm roll out through a fresh BatchedStepper
        # (stabilize with no draft is chunked greedy): calibration must run on the
        # same serving path being measured for wall-clock, since llama.cpp batch
        # prefill and per-token stepping give numerically different decodes at
        # near-ties (m4's doc-change-key-fact); a fresh stepper per arm because
        # sharing cache state across rule arms would mix windowing numerics
        old_stepper = BatchedStepper(
            base, model, case["context_old"], case["question"], horizon=horizon
        )
        old_ids = stabilize(
            [], old_stepper, max_tokens=max_tokens, eos_ids=eos_ids
        ).emitted

        free_stepper = BatchedStepper(
            base, model, case["context_new"], case["question"], horizon=horizon
        )
        new_ids = stabilize(
            [], free_stepper, max_tokens=max_tokens, eos_ids=eos_ids
        ).emitted
        naive = round(free_salvage(old_ids, new_ids, reanchor_cost), 4)
        free_text = detokenize(base, model, new_ids)
        entry = {
            "id": case["id"],
            "old": detokenize(base, model, old_ids),
            "new_free": free_text,
            "new_free_correct": correctness(free_text, case),
            "stabilized": {},
        }

        for label, rule in rules:
            stepper = BatchedStepper(
                base, model, case["context_new"], case["question"], horizon=horizon
            )
            res = stabilize(
                old_ids,
                stepper,
                rule=rule,
                anchor_len=anchor_len,
                max_tokens=max_tokens,
                eos_ids=eos_ids,
            )
            passes = stab_passes(res.events, window, reanchor_cost)
            text = detokenize(base, model, res.emitted)
            entry["stabilized"][label] = text
            row = {
                "id": case["id"],
                "category": case["category"],
                "rule": label,
                "tokens": len(res.emitted),
                "held_fraction": round(res.held_fraction, 4),
                "divergences": res.divergences,
                "forward_passes": passes["forward_passes"],
                "speedup": passes["speedup"],
                "naive_salvage": naive,
                "correct": correctness(text, case),
                "http_calls": stepper.http_calls,
                "chunk_calls": stepper.chunk_calls,
                "wasted_tokens": stepper.wasted_tokens,
            }
            rows.append(row)
            print(
                f"{row['id']} rule={label}: held {row['held_fraction']} "
                f"(naive {naive}), div {row['divergences']}, "
                f"speedup {row['speedup']}x, correct={row['correct'] or 'n/a'}"
            )
        texts.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    texts_path = out_path.with_suffix(".texts.json")
    texts_path.write_text(json.dumps(texts, indent=1))
    print(f"wrote {len(rows)} rows to {out_path} and texts to {texts_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded-mode stabilized regeneration vs free decode after a context edit"
    )
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).parent.parent / "cases"
    )
    parser.add_argument("--out", type=Path, default=Path("results/stab.csv"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--anchor-len", type=int, default=3)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--reanchor-cost", type=int, default=4)
    parser.add_argument(
        "--rules",
        nargs="*",
        default=DEFAULT_RULES,
        help="rule specs, e.g. tau:0.0 confidence_gated:tau=3.0,p_cut=0.9 entropy_scaled:tau=3.0",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=64,
        help="BatchedStepper draft-verification window (matches measure_wall's default)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="skip this many cases (for batching a long run across server restarts)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="run only this many cases after --start (default: all remaining)",
    )
    args = parser.parse_args()
    run(
        args.base,
        args.model,
        args.cases,
        args.out,
        args.max_tokens,
        args.anchor_len,
        args.window,
        args.reanchor_cost,
        args.rules,
        args.horizon,
        args.start,
        args.count,
    )


if __name__ == "__main__":
    main()
