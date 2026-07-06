import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from bench.spec_decode import spec_savings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speculative-decode forward-pass savings from a token dump"
    )
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--reanchor-cost", type=int, default=1)
    args = parser.parse_args()

    dump = json.loads(args.dump.read_text())
    by_category = defaultdict(list)
    print(f"window={args.window} reanchor_cost={args.reanchor_cost}\n")
    for case in dump:
        r = spec_savings(
            case["old_ids"], case["new_ids"], args.window, args.reanchor_cost
        )
        by_category[case["category"]].append(r["speedup"])
        print(
            f"{case['id']}: {r['forward_passes']} passes vs "
            f"{r['baseline_passes']} baseline, "
            f"verified {r['verified_tokens']} / serial {r['serial_tokens']}, "
            f"speedup {r['speedup']}x"
        )

    print("\nmedian speedup by category:")
    for cat, speedups in sorted(by_category.items()):
        print(f"  {cat}: {round(median(speedups), 2)}x  (n={len(speedups)})")


if __name__ == "__main__":
    main()
