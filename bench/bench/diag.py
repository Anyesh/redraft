import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.measure import greedy_output, teacher_forced_argmax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the divergence point for one case"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())
    case = next(c for c in cases if c["id"] == args.id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    old_out = greedy_output(
        model, tokenizer, case["context_old"], case["question"], args.max_new_tokens
    )
    new_argmax = teacher_forced_argmax(
        model, tokenizer, case["context_new"], case["question"], old_out
    )

    print(f"--- {case['id']} ---")
    print("OLD OUTPUT:")
    print(tokenizer.decode(old_out))
    for i, (o, n) in enumerate(zip(old_out, new_argmax)):
        if o != n:
            lo, hi = max(0, i - 3), min(len(old_out), i + 4)
            print(f"\nFIRST DIVERGENCE at position {i}:")
            print(f"  context before: {tokenizer.decode(old_out[lo:i])!r}")
            print(f"  old token:      {tokenizer.decode([o])!r}")
            print(f"  new argmax:     {tokenizer.decode([n])!r}")
            break
    else:
        print("\nNo divergence: entire output reproduced.")


if __name__ == "__main__":
    main()
