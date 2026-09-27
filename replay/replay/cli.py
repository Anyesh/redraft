import argparse
import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path

import httpx

from replay.bundle import load_units
from replay.calibration import calibrate, load_cases
from replay.judge import Judge, JudgeError, fact_diff
from replay.report import summarize
from replay.run import Redraftd, run_bundle


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _judge_client(args) -> httpx.AsyncClient:
    headers = {}
    if args.judge_key_env:
        headers["Authorization"] = f"Bearer {os.environ[args.judge_key_env]}"
    return httpx.AsyncClient(base_url=args.judge_url, headers=headers)


async def cmd_run(args) -> int:
    units = load_units(args.bundle)
    headers = {"Authorization": f"Bearer {os.environ[args.token_env]}"}
    async with httpx.AsyncClient(
        base_url=args.redraftd, headers=headers, timeout=60
    ) as http:
        await run_bundle(Redraftd(http, args.max_tokens), units, Path(args.out))
    print(f"replayed {len(units)} units into {args.out}")
    return 0


async def _calibrated(judge: Judge) -> bool:
    ok, rows = await calibrate(partial(fact_diff, judge), load_cases())
    for row in rows:
        mark = "ok  " if row["ok"] else "MISS"
        print(f"{mark} {row['case_id']}: expected {row['expect']}, got {row['got']}")
    return ok


async def cmd_calibrate(args) -> int:
    async with _judge_client(args) as http:
        return 0 if await _calibrated(Judge(http, args.judge_model)) else 1


async def cmd_judge(args) -> int:
    async with _judge_client(args) as http:
        judge = Judge(http, args.judge_model)
        if not await _calibrated(judge):
            print("judge failed calibration; results stay unjudged", file=sys.stderr)
            return 2
        out = Path(args.out)
        seen = {v["unit_id"] for v in _read_jsonl(out)}
        with out.open("a") as fh:
            for record in _read_jsonl(Path(args.results)):
                if record["unit_id"] in seen:
                    continue
                try:
                    verdict = await fact_diff(
                        judge,
                        record["baseline"]["text"],
                        record["redraft"]["text"],
                        record["sources"],
                    )
                except JudgeError as exc:
                    verdict = {"unjudged": True, "error": str(exc)}
                fh.write(json.dumps({"unit_id": record["unit_id"], **verdict}) + "\n")
                fh.flush()
    return 0


def cmd_report(args) -> int:
    results = _read_jsonl(Path(args.results))
    verdicts = (
        {v["unit_id"]: v for v in _read_jsonl(Path(args.verdicts))}
        if args.verdicts
        else {}
    )
    judged = bool(verdicts) and all(
        r["unit_id"] in verdicts and not verdicts[r["unit_id"]].get("unjudged")
        for r in results
    )
    print(json.dumps(summarize(results, verdicts, judged), indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="redraft-replay")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="replay a Loom bundle through redraftd")
    run.add_argument("--bundle", required=True)
    run.add_argument("--redraftd", required=True)
    run.add_argument("--token-env", default="REDRAFTD_TOKEN")
    run.add_argument("--out", required=True)
    run.add_argument("--max-tokens", type=int, default=512)

    for name in ("calibrate", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--judge-url", required=True, help="OpenAI-compatible base URL")
        p.add_argument("--judge-model", required=True)
        p.add_argument(
            "--judge-key-env", help="environment variable holding the API key"
        )
        if name == "judge":
            p.add_argument("--results", required=True)
            p.add_argument("--out", required=True)

    report = sub.add_parser("report")
    report.add_argument("--results", required=True)
    report.add_argument("--verdicts")

    args = parser.parse_args()
    if args.command == "report":
        sys.exit(cmd_report(args))
    handler = {"run": cmd_run, "calibrate": cmd_calibrate, "judge": cmd_judge}[
        args.command
    ]
    sys.exit(asyncio.run(handler(args)))


if __name__ == "__main__":
    main()
