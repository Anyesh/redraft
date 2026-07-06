import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path

from bench.free_align import common_prefix_len, free_salvage

RESYNC_COSTS = (0, 4, 16)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def generate(
    base: str, model: str, context: str, question: str, max_tokens: int
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{context}\n\n{question}"}],
        "temperature": 0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = _post(base, "/v1/chat/completions", payload)
    content = data["choices"][0]["message"]["content"]
    return THINK_RE.sub("", content).strip()


def tokenize(base: str, model: str, text: str) -> list[int]:
    data = _post(base, "/tokenize", {"model": model, "content": text})
    tokens = data["tokens"]
    return [t["id"] if isinstance(t, dict) else t for t in tokens]


def run(
    base: str, model: str, cases_dir: Path, out_path: Path, max_tokens: int
) -> None:
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        cases.extend(json.loads(path.read_text()))

    rows = []
    dump = []
    for case in cases:
        old_text = generate(
            base, model, case["context_old"], case["question"], max_tokens
        )
        new_text = generate(
            base, model, case["context_new"], case["question"], max_tokens
        )
        old_ids = tokenize(base, model, old_text)
        new_ids = tokenize(base, model, new_text)
        row = {
            "id": case["id"],
            "category": case["category"],
            "old_tokens": len(old_ids),
            "new_tokens": len(new_ids),
            "accepted_prefix": common_prefix_len(old_ids, new_ids),
        }
        for cost in RESYNC_COSTS:
            row[f"salvage_r{cost}"] = round(free_salvage(old_ids, new_ids, cost), 4)
        rows.append(row)
        dump.append(
            {
                "id": case["id"],
                "category": case["category"],
                "old_ids": old_ids,
                "new_ids": new_ids,
            }
        )
        print(
            f"{row['id']}: old {row['old_tokens']} / new {row['new_tokens']} tok, "
            f"prefix {row['accepted_prefix']}, salvage_r4 {row['salvage_r4']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tokens_path = out_path.with_suffix(".tokens.json")
    tokens_path.write_text(json.dumps(dump))
    print(f"wrote {len(rows)} rows to {out_path} and token dump to {tokens_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Free-decode reuse ceiling: diff old vs new greedy outputs after a context edit"
    )
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).parent.parent / "cases"
    )
    parser.add_argument("--out", type=Path, default=Path("results/free.csv"))
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    run(args.base, args.model, args.cases, args.out, args.max_tokens)


if __name__ == "__main__":
    main()
