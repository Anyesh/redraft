#!/usr/bin/env bash
# Runs the demo FastAPI app locally against an already-running patched
# llama-server, with no Docker image build.
set -euo pipefail

cd "$(dirname "$0")"

export REDRAFT_BASE="${REDRAFT_BASE:-http://127.0.0.1:8080}"
export REDRAFT_MODEL="${REDRAFT_MODEL:-Qwen2.5-14B-Instruct-Q4_K_M.gguf}"

echo "REDRAFT_BASE=$REDRAFT_BASE"
uv run uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
