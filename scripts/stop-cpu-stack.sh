#!/usr/bin/env bash
WORK="${WORK:?set WORK to a scratch directory}"
for f in redraftd llama-server; do
  [ -f "$WORK/$f.pid" ] && kill "$(cat "$WORK/$f.pid")" 2> /dev/null
  rm -f "$WORK/$f.pid"
done
