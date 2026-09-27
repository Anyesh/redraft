"""Loads a Loom replay bundle (contract section 11) into self-contained units."""

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

Sources = list[tuple[str, str]]


@dataclass
class Unit:
    unit_id: str
    kind: str
    document_kind: str
    instruction: str
    sources_before: Sources
    derived_before: str
    pinned: list[str]
    sources_after: Sources | None = None
    derived_after: str | None = None
    reference: dict = field(default_factory=dict)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve(revisions: dict[str, dict], refs: dict[str, str], order: list[str]):
    instruction = revisions[refs["instruction"]]["text"]
    sources = [
        (revisions[refs[i]]["name"], revisions[refs[i]]["text"]) for i in order
    ]
    return instruction, sources


def load_units(bundle: str | Path) -> list[Unit]:
    root = Path(bundle)
    revisions = {r["revision_id"]: r for r in _rows(root / "revisions.jsonl")}
    units = []
    for row in _rows(root / "refreshes.jsonl"):
        # revise refreshes are replayed from their correction rows, which carry
        # the person's before and after text
        if row["kind"] != "rederive":
            continue
        instruction, before = _resolve(
            revisions, row["input_revisions_before"], row["source_order"]
        )
        _, after = _resolve(revisions, row["input_revisions_after"], row["source_order"])
        units.append(
            Unit(
                unit_id=f"refresh:{row['refresh_id']}",
                kind="rederive",
                document_kind=row["document_kind"],
                instruction=instruction,
                sources_before=before,
                derived_before=row["derived_before"],
                pinned=list(row.get("pinned", [])),
                sources_after=after,
                reference={
                    "derived_after": row.get("derived_after"),
                    "wall_ms": row.get("wall_ms"),
                    "mode": row.get("mode"),
                },
            )
        )
    for row in _rows(root / "corrections.jsonl"):
        instruction, sources = _resolve(
            revisions, row["input_revisions"], row["source_order"]
        )
        units.append(
            Unit(
                unit_id=f"correction:{row['correction_id']}",
                kind="revise",
                document_kind=row["document_kind"],
                instruction=instruction,
                sources_before=sources,
                derived_before=row["derived_before"],
                pinned=list(row.get("pinned", [])),
                derived_after=row["derived_after"],
            )
        )
    return units


def line_edits(before: str, after: str) -> list[dict]:
    """Derived-text range edits turning `before` into `after`.

    Emitted from the bottom of the text up, so each edit's line numbers are
    still valid after the ones before it in the list have been applied, which
    is the order redraftd applies them in.
    """
    old, new = before.split("\n"), after.split("\n")
    ops = difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes()
    edits = [
        {"target": "derived", "start": i1, "end": i2, "lines": new[j1:j2]}
        for tag, i1, i2, j1, j2 in ops
        if tag != "equal"
    ]
    return list(reversed(edits))
