#!/usr/bin/env bash
set -euo pipefail
WORK="${WORK:?set WORK to a scratch directory}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$WORK/patch-check"
command rm -rf "$SRC"
git init -q "$SRC"
git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$SRC" fetch -q --depth 1 origin 9777256c3130fa3201327bfab44bae187f7caea2
git -C "$SRC" checkout -q FETCH_HEAD
git -C "$SRC" apply --check "$REPO/engine/llama-server-prompt-probs.patch"
git -C "$SRC" apply "$REPO/engine/llama-server-prompt-probs.patch"
git -C "$SRC" apply --check "$REPO/engine/llama-server-redraft.patch"
for h in stabilize.hpp batched_stabilize.hpp redraft_stabilize.hpp; do
  test -f "$REPO/engine/stabilize/$h"
done
command rm -rf "$SRC"
echo "both patches apply to the pinned commit"
