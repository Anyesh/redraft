#!/usr/bin/env bash
WORK="${WORK:?set WORK to a scratch directory}"
for f in llama-server redraftd; do
  if [ -f "$WORK/$f.pid" ] && kill -0 "$(cat "$WORK/$f.pid")" 2> /dev/null; then
    echo "$f running (pid $(cat "$WORK/$f.pid"))"
  else
    echo "$f stopped"
  fi
done
curl -s 127.0.0.1:18787/healthz || echo "redraftd not answering on 18787"
echo
