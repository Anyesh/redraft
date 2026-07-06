import argparse
import json
from pathlib import Path

from bench.batch_engine import BatchedStepper
from bench.stab_engine import eos_token_ids
from bench.stabilize import EntropyFloorTau, stabilize

HORIZON = 64
ANCHOR_LEN = 3


def read_input(input_path):
    prompt_ids, old_ids = None, None
    for line in Path(input_path).read_text().splitlines():
        if line.startswith("prompt:"):
            prompt_ids = [int(x) for x in line[len("prompt:") :].split()]
        elif line.startswith("old:"):
            old_ids = [int(x) for x in line[len("old:") :].split()]
    return prompt_ids, old_ids


def run(base, model, manifest_path, max_tokens):
    manifest = json.loads(Path(manifest_path).read_text())
    eos_ids = eos_token_ids(base, model)
    rule = EntropyFloorTau(tau=3.0, floor=1.0)

    same = 0
    for entry in manifest:
        prompt_ids, old_ids = read_input(entry["input"])
        stepper = BatchedStepper(base, model, "", "", horizon=HORIZON)
        stepper.template_ids = prompt_ids
        res = stabilize(
            old_ids,
            stepper,
            rule=rule,
            anchor_len=ANCHOR_LEN,
            max_tokens=max_tokens,
            eos_ids=eos_ids,
        )
        emitted_orig = entry["emitted_py"]
        match = res.emitted == emitted_orig
        same += match
        first_div = next(
            (
                i
                for i in range(min(len(emitted_orig), len(res.emitted)))
                if emitted_orig[i] != res.emitted[i]
            ),
            -1
            if len(emitted_orig) == len(res.emitted)
            else min(len(emitted_orig), len(res.emitted)),
        )
        print(
            f"{entry['id']}: {'SAME' if match else 'DRIFT'} "
            f"orig={len(emitted_orig)} now={len(res.emitted)} "
            f"held_orig={entry['held_py']} held_now={res.held_fraction:.4f} "
            f"first_div={first_div}"
        )
    print(f"\nprocess-stable: {same}/{len(manifest)} cases identical to 09:52 manifest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="results/m7_parity_manifest.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    run(args.base, args.model, args.manifest, args.max_tokens)


if __name__ == "__main__":
    main()
