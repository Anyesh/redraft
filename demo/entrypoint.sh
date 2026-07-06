#!/usr/bin/env bash
set -euo pipefail

LLAMA_PORT=8080

/app/bin/llama-server \
    -hf "${MODEL_HF_REPO}:${MODEL_HF_QUANT}" \
    -ngl 99 \
    -c 8192 \
    --parallel 2 \
    --host 127.0.0.1 \
    --port "$LLAMA_PORT" \
    --slots \
    --cache-reuse 0 &
LLAMA_PID=$!

trap 'kill $LLAMA_PID 2>/dev/null || true' EXIT

echo "waiting for llama-server on :$LLAMA_PORT (model pull on first boot can take a while)..."
until curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "llama-server exited before becoming healthy" >&2
        wait "$LLAMA_PID"
        exit 1
    fi
    sleep 2
done
echo "llama-server ready"

cd /app
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 7860
