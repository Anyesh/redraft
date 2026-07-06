# llama.cpp patches: build and API reference

## Pinned source

Repo: `ggml-org/llama.cpp`, checked out at commit `9777256c3130fa3201327bfab44bae187f7caea2`
(build tag `b9354`). Both patches are `git diff`s against that exact commit with a clean
working tree.

## Build

Apply both patches (order matters, the second depends on the first) from the repo root at the
pinned commit, then copy the three stabilizer headers into `tools/server/` before building:

```bash
git apply llama-server-prompt-probs.patch
git apply llama-server-redraft.patch
cp stabilize/{stabilize.hpp,batched_stabilize.hpp,redraft_stabilize.hpp} tools/server/
```

For a CUDA build, configure with your GPU's compute capability (e.g. `89` for Ada Lovelace,
`86` for Ampere):

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release -t llama-server -j"$(nproc)"
```

For a CPU-only build, drop `-DGGML_CUDA=ON` and the architecture flag. See llama.cpp's own build
docs for other backends (Metal, ROCm, Vulkan).

## The `prompt_probs_tail` contract

`/completion` gains a new integer request field, `prompt_probs_tail` (default 0 = off). For a prompt of `M` tokens, setting it to `N` adds a `prompt_probabilities` array to the JSON response with (at most) `N` entries, reusing the exact same top-K/softmax/logprob JSON shape as the pre-existing `completion_probabilities` field (`id`, `token`, `bytes`, `logprob`, `top_logprobs`), so any code already parsing `completion_probabilities` can parse `prompt_probabilities` unchanged.

Entry `k` (0-indexed) describes prompt token `M-N+k`: it reports that token's id and its top-`n_probs` predicted-distribution, where the distribution is the one produced by the logits at position `M-N+k-1` (i.e. "what did the model think would come next, right before this token was fed in"). The very last position of the prompt (predicting the first generated token) is deliberately excluded, that's already covered by `completion_probabilities[0]` under the pre-existing `n_predict: 1` mechanism. If `N >= M`, the server clamps silently to `M-1` entries (position 0 can never be a target: there is no position `-1` to predict it from); it never crashes or 500s on an over-large `N`. If a `cache_prompt: true` request repeats an already-fully-cached prompt, the server forces a fresh re-decode of the trailing `N+1` positions so valid logits exist to harvest (cached KV positions never go through `llama_decode()` and therefore never produce fresh logits).

Implementation-wise, the tail-target predictor positions get their `batch.logits[]` flag set to true during prefill (same mechanism already used for embeddings/MTP), the server records which batch index maps to which target prompt position, and harvests `get_token_probabilities()` for each one immediately after the `llama_decode()` call that computed it, before the batch pointer advances (logits are only valid for the most recently completed decode call). See `tools/server/server-context.cpp` for the six edit sites (slot state, request parsing, prefill batch flagging, harvest loop, KV-cache clamp, and JSON serialization).

## Known numerical caveat (not a bug in this patch)

When two or more logit-output positions are computed within a single `llama_decode()` call (e.g. `prompt_probs_tail >= 2`, or the tail target coinciding with the standard next-token-prediction flag), the returned logprobs can diverge from what a single-output query for the same position would give, by up to roughly 0.1-0.2 nats on a 35B MoE model (much smaller, often exactly zero, on a small dense model). This is a pre-existing characteristic of llama.cpp's batched/quantized GPU kernel selection, not an indexing defect: token identities (top-1 id, target-position id) match exactly regardless. Anything downstream reading `prompt_probabilities` should tolerate this the same way it should already tolerate the well-known "batched prefill vs single-token decode aren't bit-identical" behavior of llama.cpp.

## Fixed bugs

Two bugs surfaced in `llama-server-prompt-probs.patch` during testing:

- **Crash on repeated `cache_prompt: true` requests with `prompt_probs_tail`.** The `n_past` clamp for the tail window ran after the context-checkpoint restore instead of before it. On recurrent/hybrid memory (e.g. Gated DeltaNet), the final KV trim then asked for a deeper rollback than the checkpoint machinery had reconciled, hitting `llama_memory_recurrent::seq_rm`'s rollback budget (`n_rs_seq`, zero without a draft model) and aborting; dense KV caches never hit this since `seq_rm` allows unconditional rollback there. Fix: move the clamp before the checkpoint search runs, so checkpoint-or-full-reset always reconciles against the final `n_past`.
- **Wrong token identity in `prompt_probabilities` past the first prefill chunk.** The harvest loop matched `prompt_probs_batch_idx` entries against a decode chunk's local batch index but never removed matched entries. A later chunk's batch-index range (reachable via the pre-existing checkpoint mechanism's early breaks in the fill loop) could then spuriously re-match and re-harvest a stale entry under the wrong logits. Fix: `std::stable_partition` the matched entries out of `prompt_probs_batch_idx` as they're harvested, so each entry is matched against exactly one decode call.

## The `redraft_stabilize` contract

`llama-server-redraft.patch` (applied on top of `prompt_probs_tail`) adds an optional
`redraft_stabilize` object to `/completion`:

```json
{"old_output": [ids], "tau": 3.0, "floor": 1.0, "horizon": 64, "anchor_len": 3, "max_tokens": 512, "eos": [ids]}
```

When present, the slot seeds a `redraft_session` from the prompt-prefill logits at the
`DONE_PROMPT -> GENERATING` transition and runs `batched_stabilize` (self-speculative replay of
`old_output` against the edited context, verified in batched windows) instead of the normal
sampler. Output flows through the standard `process_token` path, so SSE streaming
(`stream: true`), stop strings, and `n_predict` all apply unchanged; `n_predict` must be
`>= max_tokens` or generation clips early. The final chunk (in both stream and non-stream modes)
adds `redraft_emitted` (the token ids actually emitted), `redraft_held_fraction`, and
`redraft_divergences` alongside the usual `timings`.

Cooperative scheduling is at window granularity: a redraft session yields the tick back to the
scheduler between draft windows, so other slots make progress under `--parallel N`, but each
window is still a dedicated `llama_decode` the session issues for itself.

Standalone build: the same core also builds as a `llama.cpp` example independent of the server
patch (`stabilize.hpp`, `batched_stabilize.hpp`, `stabilize_llama.cpp`, `CMakeLists.txt` from
`engine/stabilize/` copied into `examples/redraft-stabilize/`, with
`add_subdirectory(redraft-stabilize)` added to `examples/CMakeLists.txt`). It reads a
pre-tokenized input file (`prompt:`, `old:`, `tau:`, `floor:`, `horizon:`, `max_tokens:`, `eos:`
lines of integer ids) and prints one JSON line `{"emitted":[...],"held_fraction":x,"divergences":n}`.

## Recurrent and hybrid memory

The dense rollback path issues `llama_memory_seq_rm` and decodes the correction, which assumes
cheap partial KV rollback. Recurrent (Mamba/RWKV) and hybrid (e.g. Gated DeltaNet) memory only
allows `seq_rm` to rewind within a small snapshot budget (`n_rs_seq`, zero without a draft model)
and otherwise no-ops, silently corrupting state on any divergent generation if left unhandled.

The verifier branches on `llama_model_is_recurrent(model) || llama_model_is_hybrid(model)`:

- On divergence, the recurrent path rebuilds committed state by forward reprocess instead of
  trusting `seq_rm` (forward decode is always correct on recurrent memory; only backward
  rollback is unsupported). Dense models keep the byte-identical `seq_rm` path.
- To avoid reprocessing the static template on every divergence, `init()` parks the
  post-template state once (`llama_state_seq_get_data`) and `rebuild()` restores it
  (`llama_state_seq_set_data`), reprocessing only the emitted suffix. Parking is best-effort; on
  failure it falls back to a full template+emitted reprocess.
