import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

from bench.batch_engine import BatchedStepper
from bench.rule_specs import parse_rule_specs
from bench.stab_engine import LlamaStepper, _post_retry, apply_template, eos_token_ids
from bench.stabilize import AcceptRule, EntropyFloorTau, stab_passes, stabilize

DEFAULT_RULES = ["tau:0.0", "tau:3.0"]
REPEATS = 3
WINDOW = 16
REANCHOR_COST = 4
ANCHOR_LEN = 3
HORIZON = 64


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


def erase_slot(base: str) -> None:
    req = urllib.request.Request(f"{base}/slots/0?action=erase", method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()


def warm_template(base: str, model: str, template_ids: list[int]) -> None:
    _post_retry(
        base,
        "/completion",
        {
            "model": model,
            "prompt": template_ids,
            "n_predict": 0,
            "cache_prompt": True,
        },
    )


def run_baseline(
    base: str, model: str, template_ids: list[int], max_tokens: int
) -> dict:
    erase_slot(base)
    warm_template(base, model, template_ids)

    t0 = time.perf_counter()
    data = _post_retry(
        base,
        "/completion",
        {
            "model": model,
            "prompt": template_ids,
            "n_predict": max_tokens,
            "temperature": 0,
            "cache_prompt": True,
            "return_tokens": True,
        },
    )
    wall_s = time.perf_counter() - t0

    timings = data.get("timings", {})
    return {
        "wall_s": wall_s,
        "server_prompt_ms": timings.get("prompt_ms", 0.0),
        "server_predicted_ms": timings.get("predicted_ms", 0.0),
        "http_calls": 1,
        "scoring_calls": 0,
        "chunk_calls": 0,
        "wasted_tokens": 0,
        "tokens": len(data["tokens"]),
        "held_fraction": "",
        "divergences": "",
        "analytic_speedup": 1.0,
        "text": data["content"],
    }


def run_stabilized(
    base: str,
    model: str,
    context_new: str,
    question: str,
    old_ids: list[int],
    rule: AcceptRule,
    eos_ids: set[int],
    max_tokens: int,
) -> dict:
    stepper = BatchedStepper(base, model, context_new, question, horizon=HORIZON)

    erase_slot(base)
    warm_template(base, model, stepper.template_ids)

    t0 = time.perf_counter()
    res = stabilize(
        old_ids,
        stepper,
        rule=rule,
        anchor_len=ANCHOR_LEN,
        max_tokens=max_tokens,
        eos_ids=eos_ids,
    )
    wall_s = time.perf_counter() - t0

    passes = stab_passes(res.events, WINDOW, REANCHOR_COST)
    text = _post_retry(base, "/detokenize", {"model": model, "tokens": res.emitted})[
        "content"
    ]
    return {
        "wall_s": wall_s,
        "server_prompt_ms": stepper.server_prompt_ms,
        "server_predicted_ms": stepper.server_predicted_ms,
        "http_calls": stepper.http_calls,
        "scoring_calls": stepper.scoring_calls,
        "chunk_calls": stepper.chunk_calls,
        "wasted_tokens": stepper.wasted_tokens,
        "tokens": len(res.emitted),
        "held_fraction": round(res.held_fraction, 4),
        "divergences": res.divergences,
        "analytic_speedup": passes["speedup"],
        "text": text,
    }


def run_redraft(
    base: str,
    model: str,
    template_ids: list[int],
    old_ids: list[int],
    tau: float,
    floor: float,
    eos_ids: set[int],
    max_tokens: int,
) -> dict:
    """In-engine stabilization: one /completion request carries the old output and
    the rule params, the server runs the whole stabilize loop, and returns the
    emitted ids plus its own compute timings. This is the path the port exists to
    enable, with no per-token HTTP round trips.
    """
    erase_slot(base)
    warm_template(base, model, template_ids)

    t0 = time.perf_counter()
    data = _post_retry(
        base,
        "/completion",
        {
            "model": model,
            "prompt": template_ids,
            # the in-engine loop streams through the normal token path, so n_predict
            # bounds it: it must match the redraft max_tokens or generation clips early.
            "n_predict": max_tokens,
            "temperature": 0,
            "cache_prompt": True,
            "redraft_stabilize": {
                "old_output": old_ids,
                "tau": tau,
                "floor": floor,
                "horizon": HORIZON,
                "anchor_len": ANCHOR_LEN,
                "max_tokens": max_tokens,
                "eos": sorted(eos_ids),
            },
        },
    )
    wall_s = time.perf_counter() - t0

    emitted = data.get("redraft_emitted", [])
    timings = data.get("timings", {})
    text = _post_retry(base, "/detokenize", {"model": model, "tokens": emitted})[
        "content"
    ]
    return {
        "wall_s": wall_s,
        "server_prompt_ms": timings.get("prompt_ms", 0.0),
        "server_predicted_ms": timings.get("predicted_ms", 0.0),
        "http_calls": 1,
        "scoring_calls": 0,
        "chunk_calls": 0,
        "wasted_tokens": 0,
        "tokens": len(emitted),
        "held_fraction": round(data.get("redraft_held_fraction", 0.0), 4),
        "divergences": data.get("redraft_divergences", 0),
        "analytic_speedup": "",
        "text": text,
    }


def run(
    base: str,
    model: str,
    cases_dir: Path,
    out_path: Path,
    max_tokens: int,
    ids: list[str] | None,
    rule_specs: list[str],
) -> None:
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        cases.extend(json.loads(path.read_text()))
    if ids:
        cases = [c for c in cases if c["id"] in ids]

    rules = parse_rule_specs(rule_specs)
    eos_ids = eos_token_ids(base, model)

    # one untimed full generation to page in experts and warm CUDA allocations
    # before any measurement starts
    warmup_case = cases[0]
    warmup_template_ids = _post_retry(
        base,
        "/tokenize",
        {
            "model": model,
            "content": apply_template(
                base, model, warmup_case["context_new"], warmup_case["question"]
            ),
        },
    )["tokens"]
    run_baseline(base, model, warmup_template_ids, max_tokens)

    rows = []
    for case in cases:
        old_stepper = LlamaStepper(base, model, case["context_old"], case["question"])
        old_ids = stabilize(
            [], old_stepper, max_tokens=max_tokens, eos_ids=eos_ids
        ).emitted

        new_template_ids = _post_retry(
            base,
            "/tokenize",
            {
                "model": model,
                "content": apply_template(
                    base, model, case["context_new"], case["question"]
                ),
            },
        )["tokens"]

        arms: list[tuple[str, str, AcceptRule | None]] = [
            ("baseline", "baseline", None)
        ]
        for name, rule in rules:
            arms.append((name, "stabilized", rule))
            # the in-engine path is defined for the entropy_floor rule the port ships
            if isinstance(rule, EntropyFloorTau):
                arms.append(("redraft:" + name, "redraft", rule))

        for repeat in range(REPEATS):
            for arm_name, kind, rule in arms:
                if kind == "baseline":
                    result = run_baseline(base, model, new_template_ids, max_tokens)
                elif kind == "redraft":
                    result = run_redraft(
                        base,
                        model,
                        new_template_ids,
                        old_ids,
                        rule.tau,
                        rule.floor,
                        eos_ids,
                        max_tokens,
                    )
                else:
                    result = run_stabilized(
                        base,
                        model,
                        case["context_new"],
                        case["question"],
                        old_ids,
                        rule,
                        eos_ids,
                        max_tokens,
                    )

                correct = correctness(result["text"], case)
                row = {
                    "id": case["id"],
                    "category": case["category"],
                    "arm": arm_name,
                    "repeat": repeat,
                    "wall_s": round(result["wall_s"], 4),
                    "server_prompt_ms": round(result["server_prompt_ms"], 2),
                    "server_predicted_ms": round(result["server_predicted_ms"], 2),
                    "http_calls": result["http_calls"],
                    "scoring_calls": result["scoring_calls"],
                    "chunk_calls": result["chunk_calls"],
                    "wasted_tokens": result["wasted_tokens"],
                    "tokens": result["tokens"],
                    "held_fraction": result["held_fraction"],
                    "divergences": result["divergences"],
                    "correct": correct,
                    "analytic_speedup": result["analytic_speedup"],
                }
                rows.append(row)
                print(
                    f"{row['id']} [{row['arm']}] rep{repeat}: wall={row['wall_s']}s "
                    f"server_compute={round(result['server_prompt_ms'] + result['server_predicted_ms'], 1)}ms "
                    f"http_calls={row['http_calls']} tokens={row['tokens']} "
                    f"held={row['held_fraction']} correct={correct or 'n/a'}"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wall-clock old-vs-new latency: baseline free decode vs bounded-mode stabilization"
    )
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).parent.parent / "cases"
    )
    parser.add_argument("--out", type=Path, default=Path("results/wall.csv"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument(
        "--rules",
        nargs="*",
        default=DEFAULT_RULES,
        help="rule specs, e.g. tau:0.0 confidence_gated:tau=3.0,p_cut=0.9 entropy_scaled:tau=3.0",
    )
    args = parser.parse_args()
    run(
        args.base,
        args.model,
        args.cases,
        args.out,
        args.max_tokens,
        args.ids,
        args.rules,
    )


if __name__ == "__main__":
    main()
