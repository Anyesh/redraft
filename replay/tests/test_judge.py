import json

import httpx
import pytest

from replay.judge import Claim, Judge, JudgeError, decide, fact_diff


def claim(text, verdict="supported"):
    return Claim(text, verdict, "")


def test_decide_counts_lost_facts_new_errors_and_omissions():
    b = [claim("lena drafts faq"), claim("beta monday"), claim("drill thursday"),
         claim("vague", "unverifiable")]
    r = [claim("lena drafts faq"), claim("priya drafts faq", "contradicted"),
         claim("beta tuesday", "contradicted")]
    verdict = decide(
        b, r,
        match_labels=["consistent", "contradicted", "absent"],
        same_labels=[False, False],
    )
    assert [c.claim for c in verdict["lost_facts"]] == ["beta monday"]
    assert [c.claim for c in verdict["new_errors"]] == ["priya drafts faq", "beta tuesday"]
    assert [c.claim for c in verdict["omissions"]] == ["drill thursday"]
    assert verdict["failed"] is True


def test_errors_the_baseline_shares_are_not_new():
    r = [claim("wrong owner", "contradicted")]
    verdict = decide([], r, match_labels=[], same_labels=[True])
    assert verdict["new_errors"] == []
    assert verdict["failed"] is False


def test_omissions_alone_pass():
    verdict = decide([claim("detail")], [], match_labels=["absent"], same_labels=[])
    assert verdict["failed"] is False
    assert len(verdict["omissions"]) == 1


class FakeJudgeServer:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        content = json.dumps(self.replies.pop(0))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    def judge(self) -> Judge:
        http = httpx.AsyncClient(transport=httpx.MockTransport(self.handler), base_url="http://j")
        return Judge(http, "judge-model")


async def test_fact_diff_runs_extract_match_and_same_calls():
    server = FakeJudgeServer([
        {"claims": [{"claim": "lena drafts faq", "verdict": "supported", "evidence": "l10"}]},
        {"claims": [{"claim": "lena drafts faq", "verdict": "supported", "evidence": "l10"},
                    {"claim": "priya drafts faq", "verdict": "contradicted", "evidence": "l10"}]},
        {"labels": [{"index": 0, "label": "consistent"}]},
        {"labels": [{"index": 0, "same": False}]},
    ])
    verdict = await fact_diff(server.judge(), "B", "R", [("transcript", "t")])
    assert verdict["failed"] is True
    assert [c["claim"] for c in verdict["new_errors"]] == ["priya drafts faq"]
    first = server.bodies[0]
    assert first["temperature"] == 0
    assert first["model"] == "judge-model"
    assert first["response_format"]["type"] == "json_schema"
    assert "## transcript\nt" in first["messages"][-1]["content"]


async def test_a_label_list_that_does_not_cover_every_claim_is_a_judge_error():
    server = FakeJudgeServer([
        {"claims": [{"claim": "a", "verdict": "supported", "evidence": ""},
                    {"claim": "b", "verdict": "supported", "evidence": ""}]},
        {"claims": []},
        {"labels": [{"index": 0, "label": "consistent"}]},
    ])
    with pytest.raises(JudgeError):
        await fact_diff(server.judge(), "B", "R", [])


async def test_identical_texts_pass_without_calling_the_judge():
    server = FakeJudgeServer([])
    verdict = await fact_diff(server.judge(), "same text", "same text", [])
    assert verdict["failed"] is False
    assert verdict["identical"] is True
    assert server.bodies == []
