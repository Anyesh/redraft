import asyncio
import json

import httpx

BASELINE_PARTIAL_CHUNK = {
    "index": 0,
    "content": "hello",
    "tokens": [101, 102],
    "stop": False,
}
BASELINE_FINAL_CHUNK = {
    "index": 0,
    "content": "",
    "tokens": [],
    "stop": True,
    "stop_type": "eos",
}
REDRAFT_FINAL_CHUNK = {
    "index": 0,
    "content": "",
    "tokens": [],
    "stop": True,
    "stop_type": "eos",
    "redraft_emitted": [201, 202],
    "redraft_held_fraction": 0.9,
    "redraft_divergences": 1,
}


def sse_body(*events: dict) -> bytes:
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


class FakeLlamaServer:
    """Records requests and replays canned SSE responses, standing in for the
    real llama-server so daemon tests never need a live model.
    """

    def __init__(self, redraft_available: bool = True):
        self.redraft_available = redraft_available
        self.next_completion_delay = 0.0
        self.completion_bodies: list[dict] = []
        self.calls: list[str] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<template>"})
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        if path == "/props":
            return httpx.Response(200, json={"eos_token": "<eos>"})
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/completion":
            delay, self.next_completion_delay = self.next_completion_delay, 0.0
            if delay:
                await asyncio.sleep(delay)
            body = json.loads(request.content)
            self.completion_bodies.append(body)
            is_redraft_request = "redraft_stabilize" in body
            if is_redraft_request and self.redraft_available:
                final = REDRAFT_FINAL_CHUNK
            else:
                final = BASELINE_FINAL_CHUNK
            return httpx.Response(
                200,
                content=sse_body(BASELINE_PARTIAL_CHUNK, final),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"unexpected path requested: {path}")

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def parse_sse(content: bytes) -> list[dict]:
    events = []
    for line in content.decode().splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:") :].strip()))
    return events
