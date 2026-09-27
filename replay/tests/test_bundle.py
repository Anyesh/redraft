import json

import pytest

from replay.bundle import line_edits, load_units


def write_bundle(tmp_path, revisions, refreshes, corrections):
    for name, rows in [
        ("revisions", revisions),
        ("refreshes", refreshes),
        ("corrections", corrections),
    ]:
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def rev(rid, input_id, kind, name, text):
    return {"revision_id": rid, "input_id": input_id, "input_kind": kind,
            "name": name, "text": text, "sha256": "x"}


REVISIONS = [
    rev("r1", "t", "source", "transcript", "ship friday"),
    rev("r2", "t", "source", "transcript", "ship monday"),
    rev("r3", "n", "source", "notes", "priya owns it"),
    rev("i1", "ins", "instruction", "instruction", "Summarize."),
]


def test_rederive_refresh_rows_become_units_with_before_and_after_sources(tmp_path):
    refresh = {
        "refresh_id": "f1", "document_id": "d", "document_kind": "plan",
        "section_id": "s", "correction_id": None, "kind": "rederive", "mode": "plain",
        "input_revisions_before": {"t": "r1", "n": "r3", "instruction": "i1"},
        "input_revisions_after": {"t": "r2", "n": "r3", "instruction": "i1"},
        "source_order": ["n", "t"], "pinned": [], "derived_before": "- friday",
        "derived_after": "- monday", "wall_ms": 900,
    }
    units = load_units(write_bundle(tmp_path, REVISIONS, [refresh], []))
    assert len(units) == 1
    u = units[0]
    assert (u.kind, u.document_kind, u.instruction) == ("rederive", "plan", "Summarize.")
    assert u.sources_before == [("notes", "priya owns it"), ("transcript", "ship friday")]
    assert u.sources_after == [("notes", "priya owns it"), ("transcript", "ship monday")]
    assert u.reference == {"derived_after": "- monday", "wall_ms": 900, "mode": "plain"}


def test_revise_refresh_rows_are_skipped_since_corrections_carry_them(tmp_path):
    refresh = {"refresh_id": "f2", "kind": "revise", "correction_id": "c1"}
    assert load_units(write_bundle(tmp_path, REVISIONS, [refresh], [])) == []


def test_corrections_become_revise_units(tmp_path):
    correction = {
        "correction_id": "c1", "document_id": "d", "document_kind": "reply",
        "section_id": "s", "actor": "person",
        "input_revisions": {"t": "r2", "instruction": "i1"}, "source_order": ["t"],
        "pinned": ["- keep"], "derived_before": "- a\n- keep", "derived_after": "- A\n- keep",
    }
    u = load_units(write_bundle(tmp_path, REVISIONS, [], [correction]))[0]
    assert u.kind == "revise"
    assert u.sources_before == [("transcript", "ship monday")]
    assert u.derived_after == "- A\n- keep"
    assert u.pinned == ["- keep"]


def test_a_missing_revision_fails_loudly(tmp_path):
    correction = {
        "correction_id": "c1", "document_id": "d", "document_kind": "reply",
        "section_id": "s", "input_revisions": {"t": "nope", "instruction": "i1"},
        "source_order": ["t"], "pinned": [], "derived_before": "", "derived_after": "",
    }
    with pytest.raises(KeyError):
        load_units(write_bundle(tmp_path, REVISIONS, [], [correction]))


def apply_in_order(text: str, edits: list[dict]) -> str:
    for e in edits:
        lines = text.split("\n")
        text = "\n".join(lines[: e["start"]] + e["lines"] + lines[e["end"] :])
    return text


@pytest.mark.parametrize(
    "before,after",
    [
        ("a\nb\nc", "a\nB\nc"),
        ("a\nb\nc", "a\nc"),
        ("a\nb", "x\na\nb\ny"),
        ("a\nb\nc\nd", "A\nb\nD"),
        ("", "new"),
        ("same", "same"),
    ],
)
def test_line_edits_reproduce_the_after_text_when_applied_in_order(before, after):
    edits = line_edits(before, after)
    assert all(e["target"] == "derived" for e in edits)
    assert apply_in_order(before, edits) == after
