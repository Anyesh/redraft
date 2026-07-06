import argparse
import json
from pathlib import Path

from bench.measure_free import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    case = next(c for c in json.loads(args.cases.read_text()) if c["id"] == args.id)
    old = generate(
        args.base, args.model, case["context_old"], case["question"], args.max_tokens
    )
    new = generate(
        args.base, args.model, case["context_new"], case["question"], args.max_tokens
    )
    print(f"--- {args.id} ---")
    print("OLD:", old)
    print("NEW:", new)


if __name__ == "__main__":
    main()
