import argparse
import json
import subprocess
import urllib.request
from pathlib import Path


def read_input(path):
    prompt, old, eos = [], [], []
    tau, floor, horizon, max_tokens = 3.0, 1.0, 64, 256
    for line in Path(path).read_text().splitlines():
        c = line.find(":")
        if c < 0:
            continue
        key, rest = line[:c], line[c + 1 :]
        vals = rest.split()
        if key == "prompt":
            prompt = [int(x) for x in vals]
        elif key == "old":
            old = [int(x) for x in vals]
        elif key == "eos":
            eos = [int(x) for x in vals]
        elif key == "tau":
            tau = float(rest)
        elif key == "floor":
            floor = float(rest)
        elif key == "horizon":
            horizon = int(rest)
        elif key == "max_tokens":
            max_tokens = int(rest)
    return prompt, old, eos, tau, floor, horizon, max_tokens


def via_server(base, model, entry):
    prompt, old, eos, tau, floor, horizon, max_tokens = read_input(entry["input"])
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0,
            "cache_prompt": True,
            "redraft_stabilize": {
                "old_output": old,
                "tau": tau,
                "floor": floor,
                "horizon": horizon,
                "anchor_len": 3,
                "max_tokens": max_tokens,
                "eos": eos,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return {
        "emitted": data.get("redraft_emitted", []),
        "held": round(data.get("redraft_held_fraction", 0.0), 4),
        "div": data.get("redraft_divergences", -1),
    }


def via_standalone(binary, model_gguf, entry):
    out = subprocess.run(
        [
            binary,
            "-m",
            model_gguf,
            "-ngl",
            "99",
            "-c",
            "8192",
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
    cpp = (
        json.loads(line)
        if line
        else {"emitted": [], "held_fraction": 0.0, "divergences": -1}
    )
    return {
        "emitted": cpp["emitted"],
        "held": round(cpp["held_fraction"], 4),
        "div": cpp["divergences"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["server", "standalone"], required=True)
    ap.add_argument("--manifest", default="results/m7_parity_manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="Q")
    ap.add_argument("--binary", default=None)
    ap.add_argument("--model-gguf", default=None)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    result = {}
    for entry in manifest:
        if args.mode == "server":
            r = via_server(args.base, args.model, entry)
        else:
            r = via_standalone(args.binary, args.model_gguf, entry)
        result[entry["id"]] = r
        print(f"{entry['id']}: len={len(r['emitted'])} held={r['held']} div={r['div']}")
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
