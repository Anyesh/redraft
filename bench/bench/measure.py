import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.align import accepted_prefix_len, salvage_fraction

RESYNC_COSTS = (0, 4, 16)


def load_cases(cases_dir: Path) -> list[dict]:
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        cases.extend(json.loads(path.read_text()))
    return cases


def prompt_ids(tokenizer, context: str, question: str) -> torch.Tensor:
    messages = [{"role": "user", "content": f"{context}\n\n{question}"}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return encoded["input_ids"]


@torch.no_grad()
def greedy_output(
    model, tokenizer, context: str, question: str, max_new_tokens: int
) -> list[int]:
    ids = prompt_ids(tokenizer, context, question).to(model.device)
    out = model.generate(
        ids,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out[0, ids.shape[1] :].tolist()


@torch.no_grad()
def teacher_forced_argmax(
    model, tokenizer, context: str, question: str, output_tokens: list[int]
) -> list[int]:
    ids = prompt_ids(tokenizer, context, question).to(model.device)
    prompt_len = ids.shape[1]
    full = torch.cat([ids, torch.tensor([output_tokens], device=model.device)], dim=1)
    logits = model(full).logits
    # logits at position i predict token i+1, so predictions for the output span
    # start at the last prompt position
    return logits[0, prompt_len - 1 : full.shape[1] - 1].argmax(dim=-1).tolist()


def run(model_id: str, cases_dir: Path, out_path: Path, max_new_tokens: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()

    cases = load_cases(cases_dir)
    if not cases:
        sys.exit(f"no cases found in {cases_dir}")

    rows = []
    for case in cases:
        old_out = greedy_output(
            model, tokenizer, case["context_old"], case["question"], max_new_tokens
        )
        new_argmax = teacher_forced_argmax(
            model, tokenizer, case["context_new"], case["question"], old_out
        )
        row = {
            "id": case["id"],
            "category": case["category"],
            "output_tokens": len(old_out),
            "accepted_prefix": accepted_prefix_len(old_out, new_argmax),
        }
        for cost in RESYNC_COSTS:
            # teacher-forced upper bound only, inflated by exposure bias; see docstring
            row[f"tf_upperbound_r{cost}"] = round(
                salvage_fraction(old_out, new_argmax, cost), 4
            )
        rows.append(row)
        print(
            f"{row['id']}: {row['output_tokens']} tokens, "
            f"accepted_prefix {row['accepted_prefix']} "
            f"(tf_upperbound_r0 {row['tf_upperbound_r0']})"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how much of a stale greedy output survives a context edit"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).parent.parent / "cases"
    )
    parser.add_argument("--out", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    run(args.model, args.cases, args.out, args.max_new_tokens)


if __name__ == "__main__":
    main()
