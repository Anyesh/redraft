#!/usr/bin/env bash
WORK="${WORK:?set WORK to a scratch directory}"
GRACE="${GRACE:-15}"
status=0
for f in redraftd llama-server; do
  pidfile="$WORK/$f.pid"
  if [ ! -f "$pidfile" ]; then
    echo "$f: not running"
    continue
  fi
  pid="$(cat "$pidfile")"
  if ! kill -0 "$pid" 2> /dev/null; then
    echo "$f: not running (stale pid $pid)"
    command rm -f "$pidfile"
    continue
  fi
  kill "$pid" 2> /dev/null
  waited=0
  while kill -0 "$pid" 2> /dev/null && [ "$waited" -lt "$GRACE" ]; do
    sleep 1
    waited=$((waited + 1))
  done
  if ! kill -0 "$pid" 2> /dev/null; then
    echo "$f: stopped (pid $pid)"
    command rm -f "$pidfile"
    continue
  fi
  kill -KILL "$pid" 2> /dev/null
  sleep 1
  if kill -0 "$pid" 2> /dev/null; then
    echo "$f: still running after SIGKILL (pid $pid)" >&2
    status=1
  else
    echo "$f: killed after ${GRACE}s (pid $pid)"
    command rm -f "$pidfile"
  fi
done
exit "$status"
