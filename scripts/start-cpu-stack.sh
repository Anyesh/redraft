#!/usr/bin/env bash
set -euo pipefail
WORK="${WORK:?set WORK to a scratch directory}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "$WORK/tokens" ]; then
  python3 -c "import secrets;print(secrets.token_urlsafe(24))" > "$WORK/token"
  echo "sha256:$(tr -d '\n' < "$WORK/token" | sha256sum | cut -d' ' -f1) *" > "$WORK/tokens"
fi
nohup "$WORK/llama.cpp/build/bin/llama-server" -m "$WORK/models/qwen2.5-3b-instruct-q4_k_m.gguf" \
  --host 127.0.0.1 --port 18080 --parallel 2 -c 8192 -t "${THREADS:-6}" > "$WORK/llama-server.log" 2>&1 &
echo $! > "$WORK/llama-server.pid"
for _ in $(seq 1 90); do curl -sf 127.0.0.1:18080/health > /dev/null && break; sleep 1; done
cd "$REPO/daemon"
nohup uv run redraftd --port 18787 --redraft-base http://127.0.0.1:18080 --slots 2 --queue-depth 1 \
  --tokens-file "$WORK/tokens" > "$WORK/redraftd.log" 2>&1 &
echo $! > "$WORK/redraftd.pid"
for _ in $(seq 1 30); do curl -sf 127.0.0.1:18787/healthz > /dev/null && break; sleep 1; done
curl -s 127.0.0.1:18787/healthz
echo
