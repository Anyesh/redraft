import argparse
import json
import math
from pathlib import Path

from bench.batch_engine import BatchedStepper
from bench.stab_engine import eos_token_ids
from bench.stabilize import EntropyFloorTau, _find_anchor, _logprob_of

HORIZON = 64
ANCHOR_LEN = 3


def _effective_tau(top, tau, floor):
    k = len(top)
    if k <= 1:
        return floor
    probs = [math.exp(lp) for _, lp in top]
    total = sum(probs)
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    h_norm = entropy / math.log(k)
    return floor + (tau - floor) * h_norm


def load_case(cases_dir, case_id):
    for path in sorted(Path(cases_dir).glob("*.json")):
        for case in json.loads(path.read_text()):
            if case["id"] == case_id:
                return case
    raise SystemExit(f"case {case_id} not found")


def read_old_ids(input_path):
    for line in Path(input_path).read_text().splitlines():
        if line.startswith("old:"):
            return [int(x) for x in line[len("old:") :].split()]
    raise SystemExit("no old: line")


def run(base, model, cases_dir, input_dir, case_id, tau, floor, max_tokens):
    case = load_case(cases_dir, case_id)
    old_ids = read_old_ids(Path(input_dir) / f"{case_id}.txt")
    eos_ids = eos_token_ids(base, model)
    stepper = BatchedStepper(
        base, model, case["context_new"], case["question"], horizon=HORIZON
    )
    rule = EntropyFloorTau(tau=tau, floor=floor)

    log = []
    emitted, events = [], []
    draft = 0 if old_ids else None
    watermark = 0
    while len(emitted) < max_tokens:
        draft_rest = old_ids[draft:] if draft is not None else None
        top = stepper(emitted, draft_rest)
        argmax_id, argmax_lp = top[0]
        pos = len(emitted)
        chosen = None
        if draft is not None:
            d = old_ids[draft]
            d_lp = _logprob_of(d, top)
            eff = _effective_tau(top, tau, floor)
            gap = None if d_lp is None else argmax_lp - d_lp
            accept = rule(d, top)
            log.append(
                {
                    "pos": pos,
                    "draft": d,
                    "argmax": argmax_id,
                    "d_lp": d_lp,
                    "gap": gap,
                    "eff_tau": eff,
                    "margin": None if gap is None else eff - gap,
                    "decision": "held" if accept else "serial(diverge)",
                }
            )
            if accept:
                chosen = (d, "held")
                draft += 1
                watermark = max(watermark, draft)
                if draft >= len(old_ids):
                    draft = None
            else:
                draft = None
        else:
            log.append(
                {
                    "pos": pos,
                    "draft": None,
                    "argmax": argmax_id,
                    "decision": "serial(no-draft)",
                }
            )
        if chosen is None:
            if argmax_id in eos_ids:
                break
            chosen = (argmax_id, "serial")
        token, event = chosen
        if token in eos_ids:
            break
        emitted.append(token)
        events.append(event)
        if event == "serial":
            draft = _find_anchor(old_ids, emitted, ANCHOR_LEN, watermark)
            if draft is not None:
                watermark = draft
                if draft >= len(old_ids):
                    draft = None

    held = events.count("held") / len(events) if events else 0.0
    print(f"case={case_id} emitted={len(emitted)} held={held:.4f}")
    print("marginal decisions (|margin| <= 0.5 nats), all divergences:")
    for e in log:
        m = e.get("margin")
        is_div = e["decision"].startswith("serial(diverge)")
        near = m is not None and abs(m) <= 0.5
        if is_div or near:
            print(
                f"  pos={e['pos']:>3} draft={e['draft']} argmax={e['argmax']} "
                f"gap={_f(e.get('gap'))} eff_tau={_f(e.get('eff_tau'))} "
                f"margin={_f(m)} -> {e['decision']}"
            )


def _f(x):
    return "  None" if x is None else f"{x:6.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases-dir", default="cases")
    ap.add_argument("--input-dir", default="results/m7_inputs")
    ap.add_argument("--id", required=True)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--floor", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    run(
        args.base,
        args.model,
        args.cases_dir,
        args.input_dir,
        args.id,
        args.tau,
        args.floor,
        args.max_tokens,
    )


if __name__ == "__main__":
    main()
