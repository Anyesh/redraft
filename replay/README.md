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
  timed twice from the same warm slot state, once with `baseline: true` and once with redraft,
  alternating which goes first. Results append to the output file, so an interrupted run resumes.
- `judge` runs `calibrate` first and refuses to write verdicts if any hand-labeled case in
  `replay/calibration.jsonl` comes out wrong. The judge is any OpenAI-compatible endpoint with
  JSON-schema output (`--judge-key-env` names an environment variable holding its key). It must
  not be the model under test, and generation and judging run as separate phases so the two
  models never share the card.
- A pair fails on `lost_facts` (a claim the baseline got right that redraft contradicts) or
  `new_errors` (a redraft claim the sources contradict that the baseline does not make).
  Omissions are reported but do not fail it.
- `report` prints the median speedup, median `reused`, pinned-line survival and the edit distance
  between redraft and baseline, per document kind and overall, plus every failing claim.
  Without calibrated verdicts it reports `judged: false` and no failure counts.
