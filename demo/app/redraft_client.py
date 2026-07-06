"""Self-contained llama-server client for the redraft demo Space.

Vendored from bench/bench/{stab_engine.py,measure_wall.py}'s request
shapes so this Space has no dependency on the bench tree. Ship rule is fixed to
entropy_floor:tau=3.0,floor=1.0 (the m5-selected, cross-model zero-regression rule).
"""

import json
import statistics
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence

import httpx

HORIZON = 64
ANCHOR_LEN = 3
TAU = 3.0
FLOOR = 1.0


def baseline_payload(model: str, prompt_ids: Sequence[int], max_tokens: int) -> dict:
    return {
        "model": model,
        "prompt": list(prompt_ids),
        "n_predict": max_tokens,
        "temperature": 0,
        "cache_prompt": True,
        "return_tokens": True,
        "stream": True,
    }


def redraft_payload(
    model: str,
    prompt_ids: Sequence[int],
    old_ids: Sequence[int],
    eos_ids: Iterable[int],
    max_tokens: int,
    tau: float = TAU,
    floor: float = FLOOR,
    horizon: int = HORIZON,
    anchor_len: int = ANCHOR_LEN,
) -> dict:
    return {
        "model": model,
        "prompt": list(prompt_ids),
        # the in-engine loop streams through the normal token path, so n_predict must
        # match redraft's max_tokens or generation clips early (m8 budget semantics).
        "n_predict": max_tokens,
        "temperature": 0,
        "cache_prompt": True,
        "stream": True,
        "redraft_stabilize": {
            "old_output": list(old_ids),
            "tau": tau,
            "floor": floor,
            "horizon": horizon,
            "anchor_len": anchor_len,
            "max_tokens": max_tokens,
            "eos": sorted(eos_ids),
        },
    }


def parse_sse_line(line: str) -> dict | None:
    """Parse one line of a `/completion` SSE stream (`data: {json}`).

    `/completion` uses TASK_RESPONSE_TYPE_NONE, which never emits a `[DONE]`
    sentinel (that's only sent for OAI-compatible endpoints); the stream just ends.
    Handled defensively anyway since it's a one-line cost and the RFC 8895 framing
    (blank-line-terminated events) is shared with every other SSE endpoint the
    server exposes.
    """
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def iter_sse_events(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        event = parse_sse_line(line)
        if event is not None:
            yield event


def is_final(event: dict) -> bool:
    return bool(event.get("stop"))


def collect_redraft(events: Sequence[dict]) -> dict:
    """Reduce a full redraft SSE stream (all chunks, not just the last) to a summary.

    In stream mode `content`/`tokens` on each chunk are that chunk's delta, not the
    accumulated total (the final chunk's own delta is typically empty), so the full
    text is the concatenation of every chunk's content. `redraft_emitted` /
    `redraft_held_fraction` / `redraft_divergences` are the exception: they are
    already the whole-generation totals and only appear on the final chunk.
    """
    text = "".join(e.get("content", "") for e in events)
    final = events[-1] if events else {}
    return {
        "text": text,
        "tokens": final.get("redraft_emitted", []),
        "held_fraction": final.get("redraft_held_fraction", 0.0),
        "divergences": final.get("redraft_divergences", 0),
    }


def collect_baseline(events: Sequence[dict]) -> dict:
    text = "".join(e.get("content", "") for e in events)
    tokens = [t for e in events for t in e.get("tokens", [])]
    return {"text": text, "tokens": tokens}


def speedup(baseline_wall_s: float, redraft_wall_s: float) -> float:
    if redraft_wall_s <= 0:
        raise ValueError("redraft_wall_s must be positive")
    return baseline_wall_s / redraft_wall_s


def cumulative_stats(steps: Sequence[dict]) -> dict:
    """Aggregate a batch of per-step {baseline_wall_s, redraft_wall_s, held_fraction}
    dicts (Half 2, the agent-loop throughput panel) into the headline numbers.
    """
    if not steps:
        return {
            "steps": 0,
            "total_baseline_s": 0.0,
            "total_redraft_s": 0.0,
            "cumulative_speedup": 0.0,
            "compute_saved_s": 0.0,
            "mean_held_fraction": 0.0,
        }
    total_baseline = sum(s["baseline_wall_s"] for s in steps)
    total_redraft = sum(s["redraft_wall_s"] for s in steps)
    return {
        "steps": len(steps),
        "total_baseline_s": total_baseline,
        "total_redraft_s": total_redraft,
        "cumulative_speedup": speedup(total_baseline, total_redraft),
        "compute_saved_s": total_baseline - total_redraft,
        "mean_held_fraction": statistics.mean(s["held_fraction"] for s in steps),
    }


class RedraftClient:
    def __init__(self, http: httpx.AsyncClient, model: str):
        self.http = http
        self.model = model

    async def tokenize(self, text: str) -> list[int]:
        resp = await self.http.post(
            "/tokenize", json={"model": self.model, "content": text}
        )
        resp.raise_for_status()
        tokens = resp.json()["tokens"]
        return [t["id"] if isinstance(t, dict) else t for t in tokens]

    async def detokenize(self, ids: Sequence[int]) -> str:
        resp = await self.http.post(
            "/detokenize", json={"model": self.model, "tokens": list(ids)}
        )
        resp.raise_for_status()
        return resp.json()["content"]

    async def apply_template(self, context: str, question: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"{context}\n\n{question}"}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = await self.http.post("/apply-template", json=payload)
        resp.raise_for_status()
        return resp.json()["prompt"]

    async def eos_ids(self) -> set[int]:
        resp = await self.http.get("/props", params={"model": self.model})
        resp.raise_for_status()
        return set(await self.tokenize(resp.json()["eos_token"]))

    async def _stream_completion(self, payload: dict) -> AsyncIterator[dict]:
        async with self.http.stream("POST", "/completion", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                event = parse_sse_line(line)
                if event is not None:
                    yield event

    def stream_baseline(
        self, prompt_ids: Sequence[int], max_tokens: int
    ) -> AsyncIterator[dict]:
        return self._stream_completion(
            baseline_payload(self.model, prompt_ids, max_tokens)
        )

    def stream_redraft(
        self,
        prompt_ids: Sequence[int],
        old_ids: Sequence[int],
        eos_ids: Iterable[int],
        max_tokens: int,
    ) -> AsyncIterator[dict]:
        return self._stream_completion(
            redraft_payload(self.model, prompt_ids, old_ids, eos_ids, max_tokens)
        )
