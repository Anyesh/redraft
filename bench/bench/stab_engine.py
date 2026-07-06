import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence

from bench.measure_free import _post, tokenize


def _post_retry(base: str, path: str, payload: dict, attempts: int = 3) -> dict:
    """Per-token stepping issues thousands of requests per run; one transient stall
    must not kill an hour of measurement.
    """
    for attempt in range(attempts):
        try:
            return _post(base, path, payload)
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def apply_template(base: str, model: str, context: str, question: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{context}\n\n{question}"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return _post(base, "/apply-template", payload)["prompt"]


def detokenize(base: str, model: str, ids: Sequence[int]) -> str:
    return _post(base, "/detokenize", {"model": model, "tokens": list(ids)})["content"]


def eos_token_ids(base: str, model: str) -> set[int]:
    url = f"{base}/props?model={urllib.parse.quote(model)}"
    with urllib.request.urlopen(url, timeout=600) as resp:
        props = json.loads(resp.read())
    return set(tokenize(base, model, props["eos_token"]))


class LlamaStepper:
    """StepModel over llama-server: one /completion call per token with a token-array
    prompt. cache_prompt reuses the KV of the shared prefix across calls, so each step
    costs roughly one decode step rather than a full prefill.
    """

    def __init__(
        self, base: str, model: str, context: str, question: str, n_probs: int = 20
    ):
        self.base = base
        self.model = model
        self.n_probs = n_probs
        self.template_ids = tokenize(
            base, model, apply_template(base, model, context, question)
        )

    def __call__(
        self, emitted: Sequence[int], draft_rest: Sequence[int] | None
    ) -> list[tuple[int, float]]:
        data = _post_retry(
            self.base,
            "/completion",
            {
                "model": self.model,
                "prompt": self.template_ids + list(emitted),
                "n_predict": 1,
                "n_probs": self.n_probs,
                "temperature": 0,
                "cache_prompt": True,
            },
        )
        top = data["completion_probabilities"][0]["top_logprobs"]
        return [(t["id"], t["logprob"]) for t in top]
