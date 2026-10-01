# redraftd

Session daemon in front of the patched llama-server (`engine/BUILD.md`). A caller keeps one
session per document section; each refresh reuses the section's previous text as a speculative
draft and streams the new text together with span events saying which parts were kept and which
were replaced.

## Install and run

redraftd needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). From `daemon/`, `uv sync`
creates the project environment and installs `clients/python/redraft_client` as a path dependency
(declared under `tool.uv.sources` in `pyproject.toml`), so the checkout must stay intact.

Start a patched llama-server (build and flags in `engine/BUILD.md`), then the daemon:

```bash
llama-server -m model.gguf --parallel 2 --slot-save-path ./slots --host 127.0.0.1 --port 8080
uv run redraftd --redraft-base http://127.0.0.1:8080 --slots 2 --tokens-file tokens
```

`--slots` must equal the engine's `--parallel`. `scripts/start-cpu-stack.sh` does both steps for a
CPU-only machine (`scripts/README.md`).

### Tokens file

One entry per line: `sha256:<hex digest of the token> <scope>`, where the scope is a tenant name
or `*` for every tenant. Blank lines and `#` comments are ignored; a malformed line stops the
daemon at startup with the file and line number. To create a token and its entry:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(24))" > token
echo "sha256:$(tr -d '\n' < token | sha256sum | cut -d' ' -f1) <tenant>" >> tokens
```

Callers send the token as `Authorization: Bearer <token>`. The tokens file is read once at start,
so restart the daemon after changing it. Keep both files out of version control. Without a tokens
file the daemon refuses to start, unless it gets `--no-auth` on a loopback bind (single-user local
use, e.g. an editor plugin with a fixed tenant such as `local`).

### Configuration

Every flag has an environment variable (`daemon/config.py`); a flag wins over the variable.

| Flag | Variable | Default | Meaning |
|---|---|---|---|
| `--host` | `REDRAFTD_HOST` | `127.0.0.1` | bind address; `--no-auth` is refused on a non-loopback bind |
| `--port` | `REDRAFTD_PORT` | `8787` | listen port |
| `--redraft-base` | `REDRAFT_BASE` | `http://127.0.0.1:8080` | base URL of the patched llama-server |
| `--redraft-model` | `REDRAFT_MODEL` | `default` | model name sent to the engine |
| `--slots` | `REDRAFTD_SLOTS` | `1` | engine slots; must equal llama-server `--parallel` |
| `--queue-depth` | `REDRAFTD_QUEUE_DEPTH` | `8` | refreshes allowed to wait for a slot |
| `--queue-timeout-ms` | `REDRAFTD_QUEUE_TIMEOUT_MS` | `5000` | how long a queued refresh waits before `429 queue_timeout` |
| `--session-cap` | `REDRAFTD_SESSION_CAP` | `256` | sections kept in memory, LRU |
| `--tenant-session-cap` | `REDRAFTD_TENANT_SESSION_CAP` | `64` | sections kept per tenant |
| not a flag | `REDRAFTD_DEFAULT_MAX_TOKENS` | `512` | `max_tokens` when a request omits it |
| `--tokens-file` | `REDRAFTD_TOKENS_FILE` | none | token scopes file |
| `--no-auth` | `REDRAFTD_NO_AUTH=1` | off | disable bearer auth |

## Operate

The daemon runs in the foreground and logs to stderr (uvicorn access lines plus `redraftd` warnings
and errors), so a service manager or a shell redirect decides where logs go. It has no pid file, no
reload and no log file option.

```bash
nohup uv run redraftd ... > redraftd.log 2>&1 & echo $! > redraftd.pid    # start in the background
curl -s http://127.0.0.1:8787/healthz                                      # status
tail -f redraftd.log                                                       # logs
kill "$(cat redraftd.pid)"                                                 # stop
```

Stopping loses every session, since they live in memory only; callers re-open sections with `PUT`.
The engine is a separate process and keeps running. Stop it the same way, and expect its slot
caches to go with it.

`/healthz` needs no token. `ok` means the engine answered, supports redraft mode and exposes as
many slots as `--slots`. `degraded` means the engine answered but lacks redraft support (an
unpatched llama-server) or its slot count differs from `--slots`. `down` (HTTP 503) means the
engine is unreachable or has no model loaded; the daemon recovers by itself when the engine comes
back, so it needs no restart.

Log lines worth knowing: `refresh <session> failed` is an engine error surfaced to the caller as
`engine_failed`, `refresh job failed` carries a traceback, and `erase of slot <n> failed` means a
delete could not clear that slot's cache (it is also reported in `slots_unerased`).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| daemon exits at start naming a tokens file line | the line is not `sha256:<hex> <scope>`; see Tokens file |
| daemon exits at start about auth | no `--tokens-file`; pass one, or use `--no-auth` on a loopback bind |
| `/healthz` is `down` | engine not running, wrong `--redraft-base`, or still loading the model; check the engine log and `curl <base>/health` |
| `/healthz` is `degraded` | engine is not the patched build, or `--parallel` differs from `--slots`; rebuild per `engine/BUILD.md` or align the two numbers |
| `401 unauthorized` | token missing or its digest is not in the tokens file; restart the daemon after editing the file |
| `403 forbidden_tenant` | the token's scope is another tenant; use a `*` token or the right tenant |
| `429 busy` | every slot is busy and the queue is full or timed out; retry after `Retry-After`, or raise `--queue-depth` and `--queue-timeout-ms` |
| `409 stale_revision` | another edit or refresh got there first; re-read the section and retry with its `revision` |
| `503 engine_unavailable` | the engine dropped mid-request; see `/healthz` |
| every delete reports `slots_unerased` | llama-server was started without `--slot-save-path <dir>` |
| `smoke.py` fails the queue-full step | the daemon was not started with `--queue-depth 1` |

## API

Sections are addressed as `/v1/sessions/<tenant>/<document>/<section>`, with each segment
matching `[A-Za-z0-9._~:@+-]{1,128}`. Every route except `/healthz` needs
`Authorization: Bearer <token>`, and the token's scope must cover the tenant.

| Route | Does |
|---|---|
| `GET /healthz` | `{status: ok/degraded/down, version, template_version, model: {name, resident, redraft}, slots: {total, busy, queued}, sessions}`; 503 when down |
| `PUT .../{section}` | open or replace: `{instruction, sources: [{name, text}], derived?, pinned?, max_tokens?}`; a given `derived` becomes the next refresh's draft |
| `GET .../{section}` | current state and `revision` |
| `DELETE .../{section}` | drop it, cancelling any in-flight refresh; idempotent, `{deleted, slots_erased, slots_unerased}` |
| `DELETE /v1/sessions?prefix=<tenant>/` or `<tenant>/<document>/` | drop every matching section; `{deleted: n, slots_erased, slots_unerased}`, `400 bad_prefix` on a malformed prefix |
| `POST .../{section}/edits` | `{base_revision?, edits, pinned?}`, applied without generating |
| `POST .../{section}/refresh` | `{base_revision?, edits?, sources?, instruction?, pinned?, baseline?}`, streams SSE |

An edit is `{target: "source"|"derived", source?, start, end, lines}`: it replaces lines
`[start, end)` of that text (split on `\n`) with `lines`. Edits apply in order and all-or-nothing.
After a derived edit the next refresh runs as `revise` (the edited draft is in the prompt);
otherwise it runs as `rederive`.
A refresh is all-or-nothing until its stream opens: a refused refresh applies none of its
edits. A re-`PUT` continues the section's revision count, so stale revisions keep failing.

The refresh stream is `open`, then `delta` (`{text, kind: held|new}`) and `span`
(`{kind: held|replaced, old: [a, b], new: [c, d], text?}`) events, then one terminal event:
`done` (numbers below), `cancelled` (a newer refresh, edit, PUT or DELETE superseded it) or
`error` (`engine_failed`). Span offsets are UTF-16 code units. The spans tile the old and the new
text in order, and a held span's old and new text are equal.

`done` carries `revision, kind, mode (redraft|baseline), reused` (held tokens over emitted),
`reused_chars, wall_ms, prompt_ms, queued_ms, tokens, prompt_tokens, total_tokens, divergences, slot, slot_reused,
pinned_missing, template_version`.

Errors are `{"error": code, ...}`: 400 `bad_session_id`, 401 `unauthorized`, 403
`forbidden_tenant`, 400 `bad_prefix`, 404 `unknown_session`, 409 `stale_revision`, 413 `too_large`, 422
`bad_range`/`invalid_request`, 429 `busy` (`queue_full` or `queue_timeout`, with `Retry-After`),
503 `engine_unavailable`. Refusals happen before the stream starts, so they are plain HTTP
statuses.

## Slots and memory

A section returns to the engine slot that served it last when that slot is free, which keeps the
prompt prefix cached; otherwise it takes any free slot. With every slot busy, requests wait FIFO
in a queue bounded by `--queue-depth` and `--queue-timeout-ms`. Sessions live in memory only, LRU
with a global cap and a per-tenant cap, so a restart loses them and callers re-open with `PUT`. Nothing is written to disk; a
delete also erases (`POST /slots/{id}?action=erase`) every engine slot whose last section is gone, and
reports slots it could not erase so the caller repeats the delete.
Erasure needs llama-server started with `--slot-save-path <dir>`; without it every slot action is 501
and each slot is reported in `slots_unerased`.

## Tests

`uv run pytest` in `daemon/` runs everything against a fake engine, offline, in a few seconds.

`smoke.py` drives a running daemon in front of a real patched engine and checks the span
invariants on real output. It opens a meeting-summary section, then runs a rederive refresh, a
revise refresh with a pinned line, a stale revision, a full queue and a purge, so the daemon must
run with `--queue-depth 1` and the engine with `--slot-save-path`.

```bash
REDRAFTD_URL=http://127.0.0.1:8787 REDRAFTD_TOKEN="$(cat token)" REDRAFTD_TENANT=<tenant> \
    uv run python smoke.py [--record DIR] [--down-url URL]
```

`REDRAFTD_TENANT` defaults to `acme` and must be a tenant the token covers. It prints the drafts it
built, ends with `all invariants hold` when every check passes, and exits non-zero on the first
failed check. `--record DIR` writes every exchange as a JSON
fixture, and `--down-url` points at a second daemon whose engine is unreachable, to record the 503
health body.
