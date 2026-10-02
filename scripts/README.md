# CPU stack scripts

Run redraftd in front of a patched llama-server on a CPU-only machine, with a 3B model. All scripts take `WORK`, a scratch directory that holds the engine checkout, model, tokens, pid files and logs. `REPO` defaults to this checkout.

Host-specific GPU-box launch scripts are kept outside this repo.

## Build

    WORK=~/redraft-cpu scripts/build-cpu-engine.sh

Fetches llama.cpp at the pinned commit, applies the two patches from `engine/`, links the stabilizer headers, builds `llama-server` and downloads Qwen2.5-3B-Instruct Q4_K_M. Cmake and build output go to `$WORK/cmake.log` and `$WORK/build.log`.

## Start

    WORK=~/redraft-cpu THREADS=6 scripts/start-cpu-stack.sh

Starts llama-server on 127.0.0.1:18080 (2 slots, 8192 context) and redraftd on 127.0.0.1:18787 (2 slots, queue depth 1), then prints `/healthz`. The first run generates a bearer token in `$WORK/token` and its hash in `$WORK/tokens`.

## Check the engine patches

    WORK=~/redraft-cpu scripts/check-engine-patches.sh

Fetches the pinned llama.cpp commit into a throwaway checkout and verifies that both patches apply in order. Needs network access; run it after editing either patch.

## Status

    WORK=~/redraft-cpu scripts/status-cpu-stack.sh

## Stop

    WORK=~/redraft-cpu scripts/stop-cpu-stack.sh

Stops redraftd first, then llama-server. Each gets SIGTERM, then SIGKILL after `GRACE` seconds (default 15); the script prints what it stopped and exits non-zero if a process survives.

## Logs

`$WORK/llama-server.log` and `$WORK/redraftd.log`.

## Test

Unit tests run against a fake engine and need no stack:

    cd daemon && uv run pytest

The live end-to-end check needs the stack running. The daemon has queue depth 1, which the queue-full step requires:

    cd daemon
    REDRAFTD_URL=http://127.0.0.1:18787 REDRAFTD_TOKEN=$(cat $WORK/token) \
      REDRAFTD_TENANT=<tenant> uv run python smoke.py

See `daemon/smoke.py` for `--record` and `--down-url`.
