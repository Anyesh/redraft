# redraft-replay

Replays Loom C1 edits through redraftd and judges redraft against plain generation fact by fact.
The input format, the replay procedure and the fact diff are specified in the Loom contract
(sections 11 and 14); this package implements them.

```bash
export REDRAFTD_TOKEN=...                      # a redraftd bearer token
uv run redraft-replay run --bundle BUNDLE_DIR --redraftd http://HOST:PORT --out results.jsonl
uv run redraft-replay calibrate --judge-url http://JUDGE --judge-model MODEL
uv run redraft-replay judge --judge-url http://JUDGE --judge-model MODEL \
    --results results.jsonl --out verdicts.jsonl
uv run redraft-replay report --results results.jsonl --verdicts verdicts.jsonl
```

- `run` turns every rederive refresh and every correction in the bundle into a unit. Each unit is
  timed from the same warm slot state three times: baseline and redraft in alternating order,
  then a repeat baseline that measures the engine's own run-to-run drift. Results append to the
  output file, so an interrupted run resumes.
- `judge` runs `calibrate` first and refuses to write verdicts if any hand-labeled case in
  `replay/calibration.jsonl` comes out wrong. The judge is any OpenAI-compatible endpoint with
  JSON-schema output (`--judge-key-env` names an environment variable holding its key). It must
  not be the model under test, and generation and judging run as separate phases so the two
  models never share the card.
- A pair fails on `lost_facts` (a claim the baseline got right that redraft contradicts) or
  `new_errors` (a redraft claim the sources contradict that the baseline does not make).
  Omissions are reported but do not fail it.
- `report` prints the median speedup, median `reused`, pinned-line survival and the edit distance
  between redraft and baseline, per document kind, per mode and overall, plus every failing
  claim. For each mode it gives a ship verdict: calibrated verdicts, at least 10 units, a median
  speedup of at least 1.3x, no more failing pairs than the baseline-repeat control, and pinned
  lines kept at least as often as baseline keeps them. Otherwise it lists the blockers.

## Install and test

From `replay/`, `uv sync` installs the package and `uv run pytest` runs the tests offline against
fakes. `uv run redraft-replay synth --help` builds a synthetic bundle for a dry run.

## Judge endpoint

`judge` and `calibrate` talk to an OpenAI-compatible chat endpoint that supports JSON-schema
structured output, for example a second `llama-server` started with a different model on its own
port. Run it only after the generation phase has finished and the redraft engine is stopped, so the
two models do not share one GPU. `--judge-key-env NAME` names the environment variable that holds an API key if
the endpoint needs one. A judge that fails calibration (`replay/calibration.jsonl`) stops the run
before any verdict is written.

## Troubleshooting

- `401` or `403` from `run`: `REDRAFTD_TOKEN` is unset, or the token's scope does not cover the
  tenant the bundle uses (see Tokens file in `daemon/README.md`).
- `run` stopped midway: rerun the same command, results append and finished units are skipped.
- `judge` refuses to write verdicts: calibration failed; try a larger judge model.
