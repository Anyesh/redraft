import json

import httpx
import pytest

from app.redraft_client import (
    ANCHOR_LEN,
    FLOOR,
    HORIZON,
    TAU,
    RedraftClient,
    baseline_payload,
    collect_baseline,
    collect_redraft,
    cumulative_stats,
    is_final,
    iter_sse_events,
    parse_sse_line,
    redraft_payload,
    speedup,
)

# Recorded fixtures matching the wire format of tools/server/server-task.cpp
# (server_task_result_cmpl_partial/final::to_json_non_oaicompat) at the pinned
# llama.cpp commit, framed the way format_oai_sse() emits it: `data: {json}\n\n`.
# Confirmed live against the patched server: in stream mode, content/tokens on
# each chunk (including the final one) are that chunk's own delta, not a running
# total, so the final chunk's content/tokens are empty once generation is done.
PARTIAL_CHUNK_1 = {
    "index": 0,
    "content": "Hello",
    "tokens": [101],
    "stop": False,
    "id_slot": 0,
    "tokens_predicted": 1,
    "tokens_evaluated": 50,
}
PARTIAL_CHUNK_2 = {
    "index": 0,
    "content": " world",
    "tokens": [102],
    "stop": False,
    "id_slot": 0,
    "tokens_predicted": 2,
    "tokens_evaluated": 50,
}
REDRAFT_FINAL_CHUNK = {
    "index": 0,
    "content": "",
    "tokens": [],
    "id_slot": 0,
    "stop": True,
    "model": "qwen2.5-14b",
    "tokens_predicted": 2,
    "tokens_evaluated": 50,
    "stop_type": "eos",
    "stopping_word": "",
    "tokens_cached": 52,
    "timings": {"prompt_ms": 12.5, "predicted_ms": 40.2},
    "redraft_emitted": [101, 102],
    "redraft_held_fraction": 0.9,
    "redraft_divergences": 1,
}
BASELINE_FINAL_CHUNK = {
    "index": 0,
    "content": "",
    "tokens": [],
    "id_slot": 0,
    "stop": True,
    "model": "qwen2.5-14b",
    "tokens_predicted": 2,
    "tokens_evaluated": 50,
    "stop_type": "eos",
    "stopping_word": "",
    "tokens_cached": 52,
    "timings": {"prompt_ms": 12.5, "predicted_ms": 41.0},
}


def sse_lines(*events: dict) -> list[str]:
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    return lines


def sse_body(*events: dict) -> bytes:
    return ("\n".join(sse_lines(*events)) + "\n").encode()


def test_baseline_payload_shape():
    payload = baseline_payload("qwen2.5-14b", [1, 2, 3], 512)
    assert payload == {
        "model": "qwen2.5-14b",
        "prompt": [1, 2, 3],
        "n_predict": 512,
        "temperature": 0,
        "cache_prompt": True,
        "return_tokens": True,
        "stream": True,
    }


def test_redraft_payload_shape_uses_ship_rule_by_default():
    payload = redraft_payload("qwen2.5-14b", [1, 2, 3], [9, 8], {5, 3}, 512)
    assert payload["redraft_stabilize"] == {
        "old_output": [9, 8],
        "tau": TAU,
        "floor": FLOOR,
        "horizon": HORIZON,
        "anchor_len": ANCHOR_LEN,
        "max_tokens": 512,
        "eos": [3, 5],
    }
    assert payload["stream"] is True
    assert payload["n_predict"] == 512


def test_redraft_payload_custom_rule_params():
    payload = redraft_payload(
        "m", [1], [2], [7], 64, tau=1.5, floor=0.0, horizon=32, anchor_len=2
    )
    rs = payload["redraft_stabilize"]
    assert (rs["tau"], rs["floor"], rs["horizon"], rs["anchor_len"]) == (
        1.5,
        0.0,
        32,
        2,
    )


def test_parse_sse_line_data():
    line = f"data: {json.dumps(PARTIAL_CHUNK_1)}"
    assert parse_sse_line(line) == PARTIAL_CHUNK_1


@pytest.mark.parametrize("line", ["", "   ", "data: [DONE]", "event: ping"])
def test_parse_sse_line_returns_none_for_non_events(line):
    assert parse_sse_line(line) is None


def test_iter_sse_events_over_recorded_stream():
    lines = sse_lines(PARTIAL_CHUNK_1, PARTIAL_CHUNK_2, REDRAFT_FINAL_CHUNK)
    events = list(iter_sse_events(lines))
    assert events == [PARTIAL_CHUNK_1, PARTIAL_CHUNK_2, REDRAFT_FINAL_CHUNK]
    assert not is_final(events[0])
    assert is_final(events[-1])


def test_collect_redraft_accumulates_content_across_chunks():
    events = [PARTIAL_CHUNK_1, PARTIAL_CHUNK_2, REDRAFT_FINAL_CHUNK]
    summary = collect_redraft(events)
    assert summary == {
        "text": "Hello world",
        "tokens": [101, 102],
        "held_fraction": 0.9,
        "divergences": 1,
    }


def test_collect_baseline_accumulates_content_and_tokens_across_chunks():
    events = [PARTIAL_CHUNK_1, PARTIAL_CHUNK_2, BASELINE_FINAL_CHUNK]
    summary = collect_baseline(events)
    assert summary == {"text": "Hello world", "tokens": [101, 102]}


def test_collect_redraft_empty_stream():
    assert collect_redraft([]) == {
        "text": "",
        "tokens": [],
        "held_fraction": 0.0,
        "divergences": 0,
    }


def test_speedup():
    assert speedup(4.0, 2.0) == 2.0


def test_speedup_rejects_nonpositive_redraft_wall():
    with pytest.raises(ValueError):
        speedup(4.0, 0.0)


def test_cumulative_stats():
    steps = [
        {"baseline_wall_s": 2.0, "redraft_wall_s": 1.0, "held_fraction": 0.8},
        {"baseline_wall_s": 3.0, "redraft_wall_s": 1.5, "held_fraction": 0.6},
    ]
    stats = cumulative_stats(steps)
    assert stats["steps"] == 2
    assert stats["total_baseline_s"] == 5.0
    assert stats["total_redraft_s"] == 2.5
    assert stats["cumulative_speedup"] == 2.0
    assert stats["compute_saved_s"] == 2.5
    assert stats["mean_held_fraction"] == pytest.approx(0.7)


def test_cumulative_stats_empty():
    stats = cumulative_stats([])
    assert stats["steps"] == 0
    assert stats["cumulative_speedup"] == 0.0


def _client_with_handler(handler) -> RedraftClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://llama-server")
    return RedraftClient(http, "qwen2.5-14b")


async def test_tokenize_request_shape_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"tokens": [1, 2, 3]})

    client = _client_with_handler(handler)
    ids = await client.tokenize("hello")
    assert ids == [1, 2, 3]
    assert seen["url"].endswith("/tokenize")
    assert seen["body"] == {"model": "qwen2.5-14b", "content": "hello"}


async def test_tokenize_unwraps_dict_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tokens": [{"id": 5}, {"id": 6}]})

    client = _client_with_handler(handler)
    assert await client.tokenize("x") == [5, 6]


async def test_detokenize_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": "hello world"})

    client = _client_with_handler(handler)
    text = await client.detokenize([1, 2])
    assert text == "hello world"
    assert seen["body"] == {"model": "qwen2.5-14b", "tokens": [1, 2]}


async def test_apply_template_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"prompt": "<template>"})

    client = _client_with_handler(handler)
    prompt = await client.apply_template("context", "question")
    assert prompt == "<template>"
    assert seen["body"] == {
        "model": "qwen2.5-14b",
        "messages": [{"role": "user", "content": "context\n\nquestion"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def test_eos_ids_composes_props_and_tokenize():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/props":
            return httpx.Response(200, json={"eos_token": "<|im_end|>"})
        return httpx.Response(200, json={"tokens": [7]})

    client = _client_with_handler(handler)
    assert await client.eos_ids() == {7}
    assert calls == ["/props", "/tokenize"]


async def test_stream_baseline_yields_parsed_events():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(PARTIAL_CHUNK_1, BASELINE_FINAL_CHUNK),
            headers={"content-type": "text/event-stream"},
        )

    client = _client_with_handler(handler)
    events = [e async for e in client.stream_baseline([1, 2], 512)]
    assert events == [PARTIAL_CHUNK_1, BASELINE_FINAL_CHUNK]


async def test_stream_redraft_yields_parsed_events_with_redraft_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["redraft_stabilize"]["old_output"] == [9]
        return httpx.Response(
            200,
            content=sse_body(PARTIAL_CHUNK_1, REDRAFT_FINAL_CHUNK),
            headers={"content-type": "text/event-stream"},
        )

    client = _client_with_handler(handler)
    events = [e async for e in client.stream_redraft([1, 2], [9], {3}, 512)]
    assert is_final(events[-1])
    assert collect_redraft(events)["held_fraction"] == 0.9
