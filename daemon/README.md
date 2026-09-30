# redraftd

Session daemon in front of the patched llama-server (`engine/BUILD.md`). A caller keeps one
session per document section; each refresh reuses the section's previous text as a speculative
draft and streams the new text together with span events saying which parts were kept and which
were replaced.

## Run

```bash
llama-server -m model.gguf --parallel 2 ...        # patched build
uv run redraftd --redraft-base http://127.0.0.1:8080 --slots 2 --tokens-file tokens
```

`--slots` must equal the engine's `--parallel`. The tokens file holds one
`sha256:<hex digest of the token> <tenant or *>` per line. Without a tokens file the daemon
refuses to start, unless it gets `--no-auth` on a loopback bind (single-user local use, e.g. an
editor plugin with a fixed tenant such as `local`). Every flag has a `REDRAFTD_*` environment
variable (`daemon/config.py`).

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

`uv run pytest` runs everything against a fake engine. `smoke.py` drives a running daemon in
front of a real patched engine, checks the span invariants on real output, and with `--record`
writes each exchange as a fixture.
