import argparse
import json
import subprocess
from pathlib import Path

from bench.batch_engine import BatchedStepper
from bench.measure_free import tokenize
from bench.rule_specs import parse_rule_specs
from bench.stab_engine import LlamaStepper, _post_retry, apply_template, eos_token_ids
from bench.stabilize import stabilize

HORIZON = 64
ANCHOR_LEN = 3
RULE_SPEC = "entropy_floor:tau=3.0,floor=1.0"


def prep(base, model, cases_dir, ids, max_tokens, input_dir, manifest_path):
    """Server-up phase: for each case compute the old answer (free decode of the
    old context), the new prompt token ids, and the Python reference emitted
    sequence under the same rule the C++ port uses. Writes one pre-tokenized
    input file per case for the C++ binary plus a manifest of reference outputs.
    """
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        cases.extend(json.loads(path.read_text()))
    if ids:
        cases = [c for c in cases if c["id"] in ids]

    eos_ids = eos_token_ids(base, model)
    rule = parse_rule_specs([RULE_SPEC])[0][1]
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for case in cases:
        old_stepper = LlamaStepper(base, model, case["context_old"], case["question"])
        old_ids = stabilize(
            [], old_stepper, max_tokens=max_tokens, eos_ids=eos_ids
        ).emitted

        new_prompt_ids = _post_retry(
            base,
            "/tokenize",
            {
                "model": model,
                "content": apply_template(
                    base, model, case["context_new"], case["question"]
                ),
            },
        )["tokens"]

        stepper = BatchedStepper(
            base, model, case["context_new"], case["question"], horizon=HORIZON
        )
        ref = stabilize(
            old_ids,
            stepper,
            rule=rule,
            anchor_len=ANCHOR_LEN,
            max_tokens=max_tokens,
            eos_ids=eos_ids,
        )

        input_path = input_dir / f"{case['id']}.txt"
        input_path.write_text(
            f"prompt: {' '.join(map(str, new_prompt_ids))}\n"
            f"old: {' '.join(map(str, old_ids))}\n"
            f"tau: 3.0\nfloor: 1.0\nhorizon: {HORIZON}\n"
            f"max_tokens: {max_tokens}\n"
            f"eos: {' '.join(map(str, sorted(eos_ids)))}\n"
        )
        manifest.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": str(input_path),
                "emitted_py": ref.emitted,
                "held_py": round(ref.held_fraction, 4),
                "div_py": ref.divergences,
            }
        )
        print(
            f"prep {case['id']}: old={len(old_ids)} emitted_py={len(ref.emitted)} held={ref.held_fraction:.3f}"
        )

    Path(manifest_path).write_text(json.dumps(manifest, indent=2))


def compare(manifest_path, binary, model_gguf, ngl, ctx):
    """Server-down phase: run the C++ binary on each input file and diff its
    emitted sequence against the Python reference. Reports exact-match rate and,
    on mismatch, the first divergent position (expected only around the m4
    batched/quantized numeric near-ties, not structurally).
    """
    manifest = json.loads(Path(manifest_path).read_text())
    exact = 0
    for entry in manifest:
        out = subprocess.run(
            [
                binary,
                "-m",
                model_gguf,
                "-ngl",
                str(ngl),
                "-c",
                str(ctx),
                "--cache-type-k",
                "q4_0",
                "--cache-type-v",
                "q4_0",
            ],
            env={"REDRAFT_INPUT": entry["input"], "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("{")), "")
        cpp = json.loads(line) if line else {"emitted": [], "held_fraction": 0.0}
        emitted_cpp = cpp["emitted"]
        emitted_py = entry["emitted_py"]
        match = emitted_cpp == emitted_py
        exact += match
        first_div = next(
            (
                i
                for i in range(min(len(emitted_py), len(emitted_cpp)))
                if emitted_py[i] != emitted_cpp[i]
            ),
            min(len(emitted_py), len(emitted_cpp))
            if len(emitted_py) != len(emitted_cpp)
            else -1,
        )
        agree = sum(1 for a, b in zip(emitted_py, emitted_cpp) if a == b)
        denom = max(len(emitted_py), len(emitted_cpp), 1)
        print(
            f"{entry['id']}: {'MATCH' if match else 'DIFF'} "
            f"py={len(emitted_py)} cpp={len(emitted_cpp)} "
            f"held_py={entry['held_py']} held_cpp={cpp['held_fraction']:.4f} "
            f"token_agree={agree}/{denom} first_div={first_div}"
        )
    print(f"\nexact-match: {exact}/{len(manifest)} cases")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--base", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cases-dir", default="cases")
    p.add_argument("--ids", default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--input-dir", default="results/m7_inputs")
    p.add_argument("--manifest", default="results/m7_parity_manifest.json")

    c = sub.add_parser("compare")
    c.add_argument("--manifest", default="results/m7_parity_manifest.json")
    c.add_argument("--binary", required=True)
    c.add_argument("--model-gguf", required=True)
    c.add_argument("--ngl", type=int, default=99)
    c.add_argument("--ctx", type=int, default=8192)

    args = ap.parse_args()
    if args.phase == "prep":
        ids = args.ids.split(",") if args.ids else None
        prep(
            args.base,
            args.model,
            Path(args.cases_dir),
            ids,
            args.max_tokens,
            args.input_dir,
            args.manifest,
        )
    else:
        compare(args.manifest, args.binary, args.model_gguf, args.ngl, args.ctx)


if __name__ == "__main__":
    main()
