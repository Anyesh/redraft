"""The fact diff (contract section 14): an LLM judge extracts claims, checks them
against the sources, and matches them across the baseline and redraft texts."""

import json
from dataclasses import asdict, dataclass

import httpx

from replay.bundle import Sources

FACT_DIFF_VERSION = 2

EXTRACT_SYSTEM = (
    "You check generated documents against their sources. List every atomic factual "
    "claim the TEXT makes: one checkable statement each, such as who owns a task, a date "
    "or deadline, a number, or a decision. Split compound sentences into separate claims "
    "and ignore headings and wording. Give each claim a verdict: 'supported' if the "
    "sources state or directly imply it, 'contradicted' if the sources state something "
    "incompatible with it, 'unverifiable' if the sources say nothing either way. Quote the "
    "source line the verdict rests on as evidence, or leave it empty. A source named "
    "'person edit' holds lines a person wrote by hand; it is authoritative and wins over "
    "every other source where they differ, so a claim that contradicts it is contradicted."
)
MATCH_SYSTEM = (
    "For each numbered CLAIM, read the TEXT and label it: 'consistent' if the text states "
    "the claim or something with the same meaning, 'contradicted' if the text states "
    "something incompatible with it (a different owner, date, number or decision for the "
    "same thing), 'absent' if the text does not address it. Return exactly one label per "
    "claim, with its index."
)
SAME_SYSTEM = (
    "For each numbered CLAIM, answer whether the TEXT makes the same claim, with the same "
    "meaning even if worded differently. Return exactly one answer per claim, with its index."
)

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"enum": ["supported", "contradicted", "unverifiable"]},
                    "evidence": {"type": "string"},
                },
                "required": ["claim", "verdict", "evidence"],
            },
        }
    },
    "required": ["claims"],
}


def _labels_schema(field: str, value: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}, field: value},
                    "required": ["index", field],
                },
            }
        },
        "required": ["labels"],
    }


MATCH_SCHEMA = _labels_schema("label", {"enum": ["consistent", "contradicted", "absent"]})
SAME_SCHEMA = _labels_schema("same", {"type": "boolean"})


class JudgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Claim:
    claim: str
    verdict: str
    evidence: str


def render_sources(sources: Sources) -> str:
    return "\n\n".join(f"## {name}\n{text}" for name, text in sources)


def numbered(claims: list[Claim]) -> str:
    return "\n".join(f"{i}. {c.claim}" for i, c in enumerate(claims))


class Judge:
    def __init__(self, http: httpx.AsyncClient, model: str):
        self.http = http
        self.model = model

    async def _json(self, system: str, user: str, schema: dict) -> dict:
        resp = await self.http.post(
            "/v1/chat/completions",
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": schema},
                },
            },
            timeout=600,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"judge returned non-JSON: {content[:200]}") from exc

    async def extract(self, text: str, sources: Sources) -> list[Claim]:
        user = f"SOURCES:\n{render_sources(sources)}\n\nTEXT:\n{text}"
        data = await self._json(EXTRACT_SYSTEM, user, EXTRACT_SCHEMA)
        return [Claim(c["claim"], c["verdict"], c.get("evidence", "")) for c in data["claims"]]

    async def _labels(
        self, system: str, schema: dict, field: str, claims: list[Claim], text: str
    ) -> list:
        if not claims:
            return []
        user = f"CLAIMS:\n{numbered(claims)}\n\nTEXT:\n{text}"
        data = await self._json(system, user, schema)
        by_index = {item["index"]: item[field] for item in data["labels"]}
        if sorted(by_index) != list(range(len(claims))):
            raise JudgeError(f"judge labeled {sorted(by_index)} for {len(claims)} claims")
        return [by_index[i] for i in range(len(claims))]

    async def match(self, claims: list[Claim], text: str) -> list[str]:
        return await self._labels(MATCH_SYSTEM, MATCH_SCHEMA, "label", claims, text)

    async def same(self, claims: list[Claim], text: str) -> list[bool]:
        return await self._labels(SAME_SYSTEM, SAME_SCHEMA, "same", claims, text)


def decide(
    baseline_claims: list[Claim],
    redraft_claims: list[Claim],
    match_labels: list[str],
    same_labels: list[bool],
) -> dict:
    """match_labels cover the baseline's supported claims read against the redraft
    text; same_labels cover the redraft's contradicted claims read against the
    baseline text, both in list order."""
    supported = [c for c in baseline_claims if c.verdict == "supported"]
    wrong = [c for c in redraft_claims if c.verdict == "contradicted"]
    lost = [c for c, label in zip(supported, match_labels) if label == "contradicted"]
    omitted = [c for c, label in zip(supported, match_labels) if label == "absent"]
    new = [c for c, shared in zip(wrong, same_labels) if not shared]
    return {
        "lost_facts": lost,
        "new_errors": new,
        "omissions": omitted,
        "failed": bool(lost or new),
    }


async def fact_diff(judge: Judge, baseline: str, redraft: str, sources: Sources) -> dict:
    if baseline == redraft:
        # identical texts cannot differ in any fact; asking the judge would only
        # add its own inconsistency
        return {"fact_diff_version": FACT_DIFF_VERSION, "failed": False, "identical": True,
                "lost_facts": [], "new_errors": [], "omissions": []}
    b_claims = await judge.extract(baseline, sources)
    r_claims = await judge.extract(redraft, sources)
    supported = [c for c in b_claims if c.verdict == "supported"]
    wrong = [c for c in r_claims if c.verdict == "contradicted"]
    match_labels = await judge.match(supported, redraft)
    same_labels = await judge.same(wrong, baseline)
    verdict = decide(b_claims, r_claims, match_labels, same_labels)
    as_dicts = {k: [asdict(c) for c in v] for k, v in verdict.items() if k != "failed"}
    return {
        "fact_diff_version": FACT_DIFF_VERSION,
        "failed": verdict["failed"],
        "identical": False,
        **as_dicts,
        "baseline_claims": [asdict(c) for c in b_claims],
        "redraft_claims": [asdict(c) for c in r_claims],
    }
