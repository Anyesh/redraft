# redraft
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21200826.svg)](https://doi.org/10.5281/zenodo.21200826)

Reactive LLM inference. When you edit an LLM's context, redraft recomputes only what the edit actually changed, on both sides of the call: the prompt (reuse the unedited KV prefix, known art) and the answer (salvage the parts of the previous output a fresh greedy decode would still produce, the part this project builds).

The mechanism: the stale answer is replayed as a self-speculative draft under the edited context, verified in batched windows, and re-anchored after each divergence. An entropy-scaled tolerance rule, calibrated across three models for zero correctness regressions, decides which old tokens still stand. Implemented inside llama.cpp as a streaming `/completion` mode.

Headline numbers (Qwen2.5-14B, 36-case edit suite, wall-clock vs full regeneration): document summaries 2.31x median, factual answers 2.76x, code review 1.66x, up to 11x on cosmetic edits. On a hybrid Gated DeltaNet 35B, where speculative rollback is impossible, a reprocess-on-divergence verifier with a parked post-template state gets long, mostly-stable answers to 1.9-3.6x.

Full writeup: [`paper/redraft-preprint.pdf`](paper/redraft-preprint.pdf).

## Layout

- `paper/` the preprint: method, acceptance-rule math, cost model, all results.
- `engine/` llama.cpp server patches (`prompt_probs_tail` batched draft scoring,
  `redraft_stabilize` in-engine mode), the C++ stabilizer core, and `BUILD.md` with build
  recipes and the bugs found along the way.
- `bench/` the measurement harness and research trail (milestones m0-m9):
  - `bench/bench/` stabilize loop, acceptance rules, steppers, measurement drivers, and
    per-milestone reports (`report_m5.py`, `report_m7.py` reproduce the paper's tables from
    the CSVs).
  - `bench/cases/` the annotated 36-case edit suite (expect/forbid checks, six fact-flip
    canaries).
  - `bench/results/` the measurement CSVs and token dumps behind every number in the paper.
- `daemon/` redraftd, the session daemon in front of the patched server: tenant-scoped section
  sessions, span events, slot management. Install, configuration, operation and troubleshooting
  are in `daemon/README.md`.
- `clients/python/` `redraft_client`, the Python client library the daemon depends on.
- `replay/` `redraft-replay`, which replays recorded edits through redraftd and judges redraft
  against plain generation fact by fact (`replay/README.md`).
- `scripts/` build, start, stop and status scripts for a CPU-only stack, plus a check that the
  engine patches apply (`scripts/README.md`).
- `demo/` interactive demo: a reactive document editor racing redraft against regeneration,
  over a FastAPI app in front of the patched server.

## Reproduce

```
cd bench
uv run pytest                                          # offline: loop, rules, cost model, parity fuzz
uv run python -m bench.measure_wall --help             # wall-clock arms against a patched llama-server
uv run python -m bench.report_m7 --csv results/m7_wall_14b.csv
```

The server patches apply to llama.cpp commit `9777256c` (b9354); see `engine/BUILD.md`.

## Run and operate

The shortest path to a running stack on a CPU-only machine is `scripts/README.md`: build the
patched engine, start it with redraftd, check `/healthz`, read the logs, stop. For your own model
and GPU, build per `engine/BUILD.md` and start the daemon per `daemon/README.md`, which also lists
every setting, the token file format and a troubleshooting table.

## Tests

| Tier | Needs | Command |
|---|---|---|
| `bench/` loop, acceptance rules, cost model, parity fuzz | nothing | `cd bench && uv run pytest` |
| `daemon/` redraftd against a fake engine | nothing | `cd daemon && uv run pytest` |
| `replay/` bundle, judge, report | nothing | `cd replay && uv run pytest` |
| `demo/` FastAPI app | nothing | `cd demo && uv run pytest` |
| engine patches apply to the pinned commit | network | `WORK=<scratch dir> scripts/check-engine-patches.sh` |
| live daemon plus real engine | running stack, token | `cd daemon && uv run python smoke.py` (`daemon/README.md`) |
| wall-clock and acceptance measurements | patched server, GPU | `bench/bench/measure_wall.py` (see Reproduce) |

## History

The idea started as the [incr](https://crates.io/crates/incr-compute) question (track dependencies, recompute only the delta) applied end to end to inference. Milestone zero found the obvious metric invalid (teacher-forcing the stale answer masks real divergence); the honest free-decode ceiling turned out low and edit-independent, which pointed at output stabilization as the actual primitive. The full milestone trail, including the negatives that redirected it (client-side placement loses, fixed tolerance bands do not transfer across models, recurrent memory silently corrupts under dense rollback), is in the paper and in `bench/`.
