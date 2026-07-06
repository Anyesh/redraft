---
title: redraft
emoji: ✍️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# redraft: reactive inference demo

Edit an LLM's context, not the whole answer. `redraft` reuses the parts of a
stale output a fresh greedy decode would still produce and only recomputes the
span an edit actually disturbed, in-engine, over a patched llama.cpp server
(`engine/`, the `redraft_stabilize` completion mode).

Two halves, one engine:

- **Reactive document editor.** Generate a draft from a source document, edit
  the source (or type your own edit), and watch a live race: plain free-decode
  baseline vs. the redraft stabilizer, both streaming from the same patched
  server. High-held edits (typo fixes, cosmetic rewording) win outright; a
  low-held rewrite that changes most of the answer is shown honestly, no
  inflated numbers.
- **Agent-loop throughput.** Replays a scripted re-derivation loop (one field
  of a source table changes each step, the prior redraft output feeds forward
  as the next draft) and reports the real, measured cumulative compute saved
  across the run. This is the "billions of tiny re-derivations a day" shape,
  not the one-off editable-chat shape.

## The honest caveat

Speedup scales with `held_fraction x answer_length`. It fires on small,
meaning-preserving edits to long outputs. A rewrite that changes most of the
answer collapses `held_fraction` toward the same cost as baseline; the demo
shows that case too, not just the wins.

## Hardware

Ships on a single GPU Space (`sdk: docker`), sleep-on-idle. Model:
Qwen2.5-14B-Instruct (Q4_K_M GGUF, resolved via llama.cpp's own `-hf`
downloader at first boot, cached in the Space's persistent storage so a
sleep/wake cycle doesn't re-pull ~9GB). Default hardware is a small T4;
`CUDA_ARCH` build arg must match whatever GPU the Space actually runs on
(75 = T4, 86 = A10G, 89 = L4/Ada).

## Local development

No Docker needed: point the FastAPI app at any already-running patched
`llama-server` (see `engine/BUILD.md` for how that binary is built) and
iterate on the UI directly.

```
REDRAFT_BASE=http://<host>:<port> ./run_local.sh
```

## Container

```
docker build -f demo/Dockerfile --build-arg CUDA_ARCH=<75|86|89> -t redraft-space .
docker run --gpus all -p 7860:7860 -v redraft-data:/data redraft-space
```
