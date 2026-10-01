#!/usr/bin/env bash
set -euo pipefail
WORK="${WORK:?set WORK to a scratch directory}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$WORK/models"
if [ ! -d "$WORK/llama.cpp" ]; then
  git init -q "$WORK/llama.cpp"
  git -C "$WORK/llama.cpp" remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C "$WORK/llama.cpp" fetch -q --depth 1 origin 9777256c3130fa3201327bfab44bae187f7caea2
  git -C "$WORK/llama.cpp" checkout -q FETCH_HEAD
  git -C "$WORK/llama.cpp" apply "$REPO/engine/llama-server-prompt-probs.patch"
  git -C "$WORK/llama.cpp" apply "$REPO/engine/llama-server-redraft.patch"
  for h in stabilize.hpp batched_stabilize.hpp redraft_stabilize.hpp; do
    ln -sf "$REPO/engine/stabilize/$h" "$WORK/llama.cpp/tools/server/$h"
  done
fi
cmake -S "$WORK/llama.cpp" -B "$WORK/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > "$WORK/cmake.log"
cmake --build "$WORK/llama.cpp/build" --config Release -t llama-server -j"$(nproc)" > "$WORK/build.log"
M="$WORK/models/qwen2.5-3b-instruct-q4_k_m.gguf"
[ -f "$M" ] || curl -sfL -o "$M" https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
echo "engine at $WORK/llama.cpp/build/bin/llama-server"
