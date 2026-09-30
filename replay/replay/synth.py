"""Builds a synthetic Loom replay bundle (contract section 11) from templates.

The texts are template output, not model output, so the bundle proves the
replay pipeline end to end without standing in for real C1 edits.
"""

import hashlib
import json
import random
from pathlib import Path

KINDS = ("meeting_summary", "slide_outline", "plan", "reply")
PEOPLE = ["Priya", "Marco", "Lena", "Dana", "Tomas", "Aiko", "Ruben", "Sofia"]
TASKS = [
    "the release checklist",
    "the rollback drill",
    "the beta FAQ",
    "the login fix",
    "the usage report",
    "the vendor contract",
    "the onboarding guide",
    "the load test",
    "the pricing page",
    "the security review",
    "the data migration",
    "the launch email",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

INSTRUCTIONS = {
    "meeting_summary": "Summarize this meeting as two markdown lists under the headings 'Decisions' and 'Action items'. Each action item names its owner and date.",
    "slide_outline": "Turn these notes into a slide outline with one line per slide, naming the owner and date.",
    "plan": "Write a plan from this brief as a list of steps, each with an owner and a date.",
    "reply": "Draft a reply to this thread that confirms who owns what and by when.",
}


def _source(kind: str, facts: list[tuple[str, str, str]]) -> tuple[str, str]:
    if kind == "meeting_summary":
        lines = [
            f"{owner}: I'll take {task}, {day} works." for owner, task, day in facts
        ]
        return "transcript", "Dana: Let's go through the open items.\n" + "\n".join(
            lines
        )
    if kind == "slide_outline":
        lines = [f"- {task}: {owner} presents, {day}" for owner, task, day in facts]
        return "notes", "Offsite talk notes\n" + "\n".join(lines)
    if kind == "plan":
        lines = [
            f"{task} is owned by {owner} and due {day}." for owner, task, day in facts
        ]
        return "brief", "Project brief. " + " ".join(lines)
    lines = [f"> {owner} can do {task} by {day}?" for owner, task, day in facts]
    return "thread", "Subject: open items\n" + "\n".join(lines)


def _derived(kind: str, facts: list[tuple[str, str, str]]) -> str:
    head = {
        "meeting_summary": "Action items",
        "slide_outline": "Slides",
        "plan": "Steps",
        "reply": "Reply",
    }[kind]
    return "\n".join(
        [head] + [f"- {owner}: {task} by {day}" for owner, task, day in facts]
    )


class _Store:
    def __init__(self) -> None:
        self.revisions: list[dict] = []

    def add(self, input_id: str, input_kind: str, name: str, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        revision_id = f"rev_{digest[:12]}"
        if all(r["revision_id"] != revision_id for r in self.revisions):
            self.revisions.append(
                {
                    "revision_id": revision_id,
                    "input_id": input_id,
                    "input_kind": input_kind,
                    "name": name,
                    "text": text,
                    "sha256": digest,
                }
            )
        return revision_id


def _document(rng: random.Random, store: _Store, kind: str, index: int):
    doc = f"{kind}-{index:02d}"
    count = rng.randint(3, 4)
    owners = rng.sample(PEOPLE, count)
    tasks = rng.sample(TASKS, count)
    days = [rng.choice(DAYS) for _ in range(count)]
    facts = list(zip(owners, tasks, days))
    name, before_text = _source(kind, facts)

    changed = rng.randrange(count)
    new_owner = rng.choice([p for p in PEOPLE if p not in owners])
    new_day = rng.choice([d for d in DAYS if d != days[changed]])
    edited = list(facts)
    edited[changed] = (new_owner, tasks[changed], new_day)
    _, after_text = _source(kind, edited)

    src_id, ins_id = f"src_{doc}", f"ins_{doc}"
    before = {
        src_id: store.add(src_id, "source", name, before_text),
        "instruction": store.add(
            ins_id, "instruction", "instruction", INSTRUCTIONS[kind]
        ),
    }
    after = {**before, src_id: store.add(src_id, "source", name, after_text)}
    derived_before, derived_after = _derived(kind, facts), _derived(kind, edited)

    refresh = {
        "refresh_id": f"ref_{doc}",
        "ts": "2026-10-02T14:03:11Z",
        "document_id": doc,
        "document_kind": kind,
        "section_id": "main",
        "correction_id": None,
        "kind": "rederive",
        "mode": "plain",
        "input_revisions_before": before,
        "input_revisions_after": after,
        "source_order": [src_id],
        "pinned": [],
        "derived_before": derived_before,
        "derived_after": derived_after,
        "model_prior": "synthetic",
        "model_new": "synthetic",
        "wall_ms": rng.randint(4000, 9000),
        "prompt_ms": rng.randint(3000, 7000),
        "prompt_tokens": rng.randint(300, 900),
        "completion_tokens": rng.randint(60, 200),
        "fallback_reason": None,
    }

    lines = derived_before.split("\n")
    fixed = rng.randrange(1, len(lines))
    pinned_line = lines[1 + fixed % count]
    owner, task, day = facts[fixed - 1]
    person_day = rng.choice([d for d in DAYS if d != day])
    person_lines = list(lines)
    person_lines[fixed] = f"- {owner}: {task} by {person_day}"
    if person_lines[fixed] == pinned_line:
        pinned_line = lines[1 + fixed % count]
    correction = {
        "correction_id": f"cor_{doc}",
        "ts": "2026-10-02T15:10:00Z",
        "document_id": doc,
        "document_kind": kind,
        "section_id": "main",
        "actor": "person",
        "input_revisions": before,
        "source_order": [src_id],
        "pinned": [pinned_line],
        "derived_before": derived_before,
        "derived_after": "\n".join(person_lines),
        "re_edited_within_10min": False,
    }
    return refresh, correction


def build_bundle(out: Path, per_kind: int = 13, seed: int = 0) -> None:
    rng = random.Random(seed)
    store = _Store()
    refreshes, corrections = [], []
    for kind in KINDS:
        for index in range(per_kind):
            refresh, correction = _document(rng, store, kind, index)
            refreshes.append(refresh)
            corrections.append(correction)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("revisions", store.revisions),
        ("refreshes", refreshes),
        ("corrections", corrections),
    ]:
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
