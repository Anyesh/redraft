import argparse
import json
import statistics
from pathlib import Path

from bench.measure_free import _post, tokenize
from bench.measure_wall import erase_slot, warm_template
from bench.stab_engine import _post_retry, apply_template

WIDTHS = (1, 8, 16, 32, 64)
REPEATS = 5
N_PROBS = 20


def fit_slope(widths: list[int], values: list[float]) -> tuple[float, float]:
    """Ordinary least-squares slope and intercept of values against widths. The
    slope is the marginal server-compute cost per additional verified draft
    token; the intercept is the fixed per-call overhead.
    """
    n = len(widths)
    mean_w = sum(widths) / n
    mean_v = sum(values) / n
    denom = sum((w - mean_w) ** 2 for w in widths)
    if denom == 0:
        raise ValueError("widths must not all be equal")
    slope = sum((w - mean_w) * (v - mean_v) for w, v in zip(widths, values)) / denom
    intercept = mean_v - slope * mean_w
    return slope, intercept


# Median held fractions measured on the 14B in m5 (results/m5_wall_14b.csv), used
# to project the microbenchmark's per-token verification cost onto real workloads.
M5_HELD = {"doc_summary": 0.728, "fact_answer": 0.806, "code_review": 0.218}
COMMIT_BAR = 1.2


def port_speedup(slope_over_decode: float, held: float) -> float:
    """Ceiling a clean in-engine batch approaches on a workload with the given
    held fraction. Each held token costs `slope_over_decode` of a serial decode
    (the fundamental per-draft verification-logit cost, once HTTP and per-call
    overhead are removed); each divergent token still costs a full serial decode.
    Ignores speculation waste and reanchor cost, so it is an optimistic ceiling,
    but it removes the client-harness overhead that dominated m5's server-only.
    """
    per_token = held * slope_over_decode + (1 - held)
    return 1.0 / per_token if per_token else float("nan")


def summarize(
    verify_by_width: dict[int, list[float]], decode_ms_per_token: float
) -> dict:
    """From per-width verification-call server-ms samples and the baseline serial
    decode cost, recover the marginal verification cost per draft token (the slope
    of verification cost vs width) and project the in-engine port speedup onto the
    m5 workloads. The port decision turns on the projected doc_summary speedup
    against the 1.2x commit bar, not on the slope alone: held fraction amplifies a
    modest per-token saving into a workload-level win.
    """
    widths = sorted(verify_by_width)
    medians = {w: statistics.median(verify_by_width[w]) for w in widths}
    slope, intercept = fit_slope(widths, [medians[w] for w in widths])
    ratio = slope / decode_ms_per_token if decode_ms_per_token else float("nan")
    projection = {cat: port_speedup(ratio, h) for cat, h in M5_HELD.items()}
    doc = projection["doc_summary"]
    if doc >= COMMIT_BAR:
        verdict = (
            f"COMMIT-LEANING: projected in-engine doc_summary {doc:.2f}x clears the "
            f"{COMMIT_BAR}x bar; a held draft token costs {ratio:.2f}x a serial decode, "
            f"far below what m5's server-only ceiling implied"
        )
    elif doc <= 1.05:
        verdict = (
            f"KILL-LEANING: projected doc_summary {doc:.2f}x at/below parity even "
            f"in-engine; server-only is the real ceiling and m5 MIDDLE (weak) stands"
        )
    else:
        verdict = (
            f"MARGINAL: projected doc_summary {doc:.2f}x between parity and the "
            f"{COMMIT_BAR}x bar; not a clean commit on speed alone"
        )
    return {
        "widths": widths,
        "verify_ms_median": medians,
        "slope_ms_per_token": slope,
        "intercept_ms": intercept,
        "decode_ms_per_token": decode_ms_per_token,
        "slope_over_decode": ratio,
        "port_projection": projection,
        "verdict": verdict,
    }


def _timings(data: dict) -> float:
    t = data.get("timings", {})
    return t.get("prompt_ms", 0.0) + t.get("predicted_ms", 0.0)


def make_draft(base: str, model: str, base_ids: list[int], length: int) -> list[int]:
    data = _post_retry(
        base,
        "/completion",
        {
            "model": model,
            "prompt": base_ids,
            "n_predict": length,
            "temperature": 0,
            "cache_prompt": True,
            "return_tokens": True,
            # keep generating past EOS: short answers (fact cases) would otherwise
            # stop early and starve the wide-W verification measurement of drafts.
            "ignore_eos": True,
        },
    )
    ids = data.get("tokens")
    if not ids:
        ids = [e["id"] for e in data.get("completion_probabilities", [])]
    if len(ids) < length:
        raise RuntimeError(f"draft generation returned {len(ids)} < {length} tokens")
    return ids[:length]


def verify_ms(
    base: str, model: str, base_ids: list[int], draft: list[int], w: int
) -> float:
    # erase and re-warm the base so the server holds only base_ids cached; the
    # verification call then prefills exactly the w draft tokens, isolating the
    # marginal per-width cost from cross-width prefix-cache contamination.
    erase_slot(base)
    warm_template(base, model, base_ids)
    data = _post_retry(
        base,
        "/completion",
        {
            "model": model,
            "prompt": base_ids + list(draft[:w]),
            "n_predict": 1,
            "n_probs": N_PROBS,
            "temperature": 0,
            "cache_prompt": True,
            "prompt_probs_tail": w,
        },
    )
    return _timings(data)


def decode_per_token(base: str, model: str, base_ids: list[int], length: int) -> float:
    erase_slot(base)
    warm_template(base, model, base_ids)
    data = _post_retry(
        base,
        "/completion",
        {
            "model": model,
            "prompt": base_ids,
            "n_predict": length,
            "temperature": 0,
            "cache_prompt": True,
            # ignore_eos so short-answer cases still generate a full sample; divide
            # by predicted_n (actual tokens), never the requested length, or an
            # early EOS silently halves the per-token cost.
            "ignore_eos": True,
        },
    )
    timings = data.get("timings", {})
    predicted_ms = timings.get("predicted_ms", 0.0)
    predicted_n = timings.get("predicted_n") or length
    return predicted_ms / predicted_n if predicted_n else float("nan")


def run(base: str, model: str, case: dict, repeats: int, out_path: Path) -> dict:
    prompt = apply_template(base, model, case["context_new"], case["question"])
    base_ids = tokenize(base, model, prompt)
    draft = make_draft(base, model, base_ids, max(WIDTHS))

    decode_samples = [
        decode_per_token(base, model, base_ids, 64) for _ in range(repeats)
    ]
    decode_ms = statistics.median(decode_samples)

    verify_by_width: dict[int, list[float]] = {w: [] for w in WIDTHS}
    for _ in range(repeats):
        for w in WIDTHS:
            verify_by_width[w].append(verify_ms(base, model, base_ids, draft, w))

    result = summarize(verify_by_width, decode_ms)
    result["case_id"] = case["id"]
    result["base_tokens"] = len(base_ids)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--case", required=True, help="path to a case JSON (single object or list)"
    )
    ap.add_argument("--case-id", default=None, help="id to pick when --case is a list")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", default="results/m6_verify_cost.json")
    args = ap.parse_args()

    raw = json.loads(Path(args.case).read_text())
    cases = raw if isinstance(raw, list) else [raw]
    if args.case_id:
        case = next(c for c in cases if c["id"] == args.case_id)
    else:
        case = cases[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.base, args.model, case, args.repeats, out_path)

    print(f"case={result['case_id']} base_tokens={result['base_tokens']}")
    print(f"decode_ms_per_token={result['decode_ms_per_token']:.2f}")
    for w in result["widths"]:
        print(f"  verify W={w:>2}: {result['verify_ms_median'][w]:.2f} ms")
    print(
        f"slope_ms_per_token={result['slope_ms_per_token']:.3f} "
        f"intercept={result['intercept_ms']:.2f}"
    )
    print(f"slope/decode={result['slope_over_decode']:.3f}")
    print("projected in-engine port speedup (m5 held fractions):")
    for cat, sp in result["port_projection"].items():
        print(f"  {cat}: {sp:.2f}x")
    print(result["verdict"])


if __name__ == "__main__":
    main()
