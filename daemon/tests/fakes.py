import asyncio
import json

import httpx

EOS = 0


def sse_body(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


def char_ids(text: str) -> list[int]:
    return [ord(c) for c in text]


class FakeLlamaServer:
    """Stands in for the patched llama-server with one token per character
    (token id = code point), so tests can reason about ids, pieces and offsets
    without a model.

    A redraft request replays `old_output` as held tokens, except where
    `redraft_edits` maps an old index to replacement text, which is emitted as
    serially decoded tokens in its place.
    """

    def __init__(self, redraft_available: bool = True):
        self.redraft_available = redraft_available
        self.resident = True
        self.total_slots = 2
        self.n_ctx = 4096
        self.baseline_text = "fresh text"
        self.redraft_edits: dict[int, str] = {}
        self.completion_delay = 0.0
        self.completion_bodies: list[dict] = []
        self.templates: list[str] = []
        self.erased_slots: list[int] = []
        self.erase_status = 200

    async def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            if self.resident:
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(503, json={"error": {"message": "Loading model"}})
        if path == "/props":
            return httpx.Response(
                200,
                json={
                    "eos_token": "<eos>",
                    "total_slots": self.total_slots,
                    "model_path": "/models/qwen2.5-3b-instruct-q4_k_m.gguf",
                    "default_generation_settings": {"n_ctx": self.n_ctx},
                },
            )
        if path.startswith("/slots/"):
            assert request.url.params.get("action") == "erase"
            if self.erase_status == 200:
                self.erased_slots.append(int(path.rsplit("/", 1)[1]))
            return httpx.Response(self.erase_status, json={})
        body = json.loads(request.content)
        if path == "/apply-template":
            prompt = body["messages"][0]["content"]
            self.templates.append(prompt)
            return httpx.Response(200, json={"prompt": prompt})
        if path == "/tokenize":
            content = body["content"]
            if content == "<eos>":
                return httpx.Response(200, json={"tokens": [EOS]})
            if body.get("with_pieces"):
                tokens = [{"id": ord(c), "piece": c} for c in content]
                return httpx.Response(200, json={"tokens": tokens})
            return httpx.Response(200, json={"tokens": char_ids(content)})
        if path == "/completion":
            if self.completion_delay:
                await asyncio.sleep(self.completion_delay)
            self.completion_bodies.append(body)
            stab = body.get("redraft_stabilize")
            if stab is not None and self.redraft_available:
                chunks = self._redraft_chunks(stab["old_output"])
            else:
                chunks = self._baseline_chunks()
            return httpx.Response(
                200,
                content=sse_body(*chunks),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"unexpected path requested: {path}")

    def _baseline_chunks(self) -> list[dict]:
        chunks = [
            {"content": c, "tokens": [ord(c)], "stop": False}
            for c in self.baseline_text
        ]
        chunks.append(
            {"content": "", "tokens": [], "stop": True, "timings": {"prompt_ms": 7.5}}
        )
        return chunks

    def _redraft_chunks(self, old: list[int]) -> list[dict]:
        chunks, emitted, held = [], [], 0
        for i, tok in enumerate(old):
            if i in self.redraft_edits:
                for c in self.redraft_edits[i]:
                    chunks.append({"content": c, "stop": False, "redraft_src": [-1]})
                    emitted.append(ord(c))
                continue
            chunks.append({"content": chr(tok), "stop": False, "redraft_src": [i]})
            emitted.append(tok)
            held += 1
        chunks.append(
            {
                "content": "",
                "stop": True,
                "timings": {"prompt_ms": 3.25},
                "redraft_emitted": emitted,
                "redraft_held_fraction": held / len(emitted) if emitted else 0.0,
                "redraft_divergences": len(self.redraft_edits),
            }
        )
        return chunks

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def parse_sse(content: bytes) -> list[tuple[str, dict]]:
    events, name = [], "message"
    for line in content.decode().splitlines():
        if line.startswith("event:"):
            name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[len("data:") :].strip())))
            name = "message"
    return events
