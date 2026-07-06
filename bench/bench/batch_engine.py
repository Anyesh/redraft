from collections.abc import Callable, Sequence

from bench.stab_engine import _post_retry, apply_template
from bench.measure_free import tokenize

Post = Callable[[str, str, dict], dict]


class BatchedStepper:
    """StepModel over llama-server: while a draft is available, one scoring call
    teacher-forces up to `horizon` draft tokens via `prompt_probs_tail` and caches
    all resulting distributions, so the next `horizon` steps are served from the
    cache with zero HTTP. Falls back to a plain per-token call (LlamaStepper-
    equivalent) whenever no draft is available to batch over.

    Cache validity follows directly from keying on the exact `emitted` prefix: a
    divergence emits a token the draft did not predict, so the next prefix never
    matches a cached continuation and the cache misses on its own, no explicit
    invalidation needed.
    """

    def __init__(
        self,
        base: str,
        model: str,
        context: str,
        question: str,
        n_probs: int = 20,
        horizon: int = 16,
        post: Post = _post_retry,
    ):
        self.base = base
        self.model = model
        self.n_probs = n_probs
        self.horizon = horizon
        self.post = post
        self.template_ids = tokenize(
            base, model, apply_template(base, model, context, question)
        )
        self._cache: dict[tuple[int, ...], list[tuple[int, float]]] = {}
        self._chunk_origin: set[tuple[int, ...]] = set()
        self._attempts = 0
        self.http_calls = 0
        self.scoring_calls = 0
        self.chunk_calls = 0
        self.wasted_tokens = 0
        self.server_prompt_ms = 0.0
        self.server_predicted_ms = 0.0

    def __call__(
        self, emitted: Sequence[int], draft_rest: Sequence[int] | None
    ) -> list[tuple[int, float]]:
        key = tuple(emitted)
        cached = self._cache.pop(key, None)
        if cached is not None:
            if key in self._chunk_origin:
                self._chunk_origin.discard(key)
                self.wasted_tokens -= 1
            else:
                # a hit that didn't come from the chunk currently being drained
                # (a batched-origin entry, or none active) means we're not mid
                # fallback-stretch, so the next chunk starts back at the small end
                self._attempts = 0
            return cached

        if not draft_rest:
            return self._score_chunk(emitted)

        self._attempts = 0
        window = list(draft_rest[: self.horizon])
        data = self.post(
            self.base,
            "/completion",
            {
                "model": self.model,
                "prompt": self.template_ids + list(emitted) + window,
                "n_predict": 1,
                "n_probs": self.n_probs,
                "temperature": 0,
                "cache_prompt": True,
                "prompt_probs_tail": len(window),
            },
        )
        self.http_calls += 1
        self.scoring_calls += 1
        self._accumulate_timings(data)
        dists = [_top_logprobs(entry) for entry in data["prompt_probabilities"]]
        dists.append(_top_logprobs(data["completion_probabilities"][0]))

        prefix = list(emitted)
        for offset, dist in enumerate(dists):
            self._cache[tuple(prefix + window[:offset])] = dist
        return self._cache.pop(key)

    def _score_chunk(self, emitted: Sequence[int]) -> list[tuple[int, float]]:
        """Replaces the old one-token-per-call fallback: request `k` tokens of
        greedy continuation in one call instead of one. At temperature 0 each
        chunk token equals the client-side argmax, so entry `i`'s distribution
        is exactly what a future call at that same prefix would return; caching
        it by content-keyed prefix (like the batched path) makes every serial
        step after the first free until the chunk runs out or the trajectory
        diverges (re-anchor, EOS), at which point the leftover entries simply
        never match a future key and sit unused (`wasted_tokens`).
        """
        k = min(32, 4 << self._attempts)
        self._attempts += 1
        data = self.post(
            self.base,
            "/completion",
            {
                "model": self.model,
                "prompt": self.template_ids + list(emitted),
                "n_predict": k,
                "n_probs": self.n_probs,
                "temperature": 0,
                "cache_prompt": True,
            },
        )
        self.http_calls += 1
        self.chunk_calls += 1
        self._accumulate_timings(data)
        dists = [_top_logprobs(entry) for entry in data["completion_probabilities"]]

        prefix = list(emitted)
        generated: list[int] = []
        for dist in dists:
            cache_key = tuple(prefix + generated)
            self._cache[cache_key] = dist
            self._chunk_origin.add(cache_key)
            self.wasted_tokens += 1
            generated.append(dist[0][0])

        key = tuple(prefix)
        self._chunk_origin.discard(key)
        self.wasted_tokens -= 1
        return self._cache.pop(key)

    def _accumulate_timings(self, data: dict) -> None:
        timings = data.get("timings", {})
        self.server_prompt_ms += timings.get("prompt_ms", 0.0)
        self.server_predicted_ms += timings.get("predicted_ms", 0.0)


def _top_logprobs(entry: dict) -> list[tuple[int, float]]:
    return [(t["id"], t["logprob"]) for t in entry["top_logprobs"]]
