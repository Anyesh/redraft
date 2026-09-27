import pytest

from daemon.edits import BadRange
from daemon.sections import Change, SectionEdit, UnknownSource, apply_change
from daemon.sessions import Session, Source


def section() -> Session:
    return Session(
        session_id="acme/d/s",
        instruction="Summarize.",
        sources=[Source("transcript", "a\nb\nc"), Source("notes", "n")],
        derived="- one\n- two",
        pinned=[],
        max_tokens=64,
    )


def test_source_edit_changes_the_source_and_bumps_revision():
    s = section()
    apply_change(s, Change(edits=[SectionEdit("source", "transcript", 1, 2, ["B"])]))
    assert s.sources[0].text == "a\nB\nc"
    assert s.revision == 2
    assert not s.derived_dirty


def test_derived_edit_marks_the_section_for_revise():
    s = section()
    apply_change(s, Change(edits=[SectionEdit("derived", None, 1, 2, ["- TWO"])]))
    assert s.derived == "- one\n- TWO"
    assert s.derived_dirty


def test_a_bad_edit_leaves_the_section_untouched():
    s = section()
    change = Change(
        edits=[
            SectionEdit("source", "transcript", 0, 1, ["A"]),
            SectionEdit("derived", None, 5, 6, ["x"]),
        ]
    )
    with pytest.raises(BadRange):
        apply_change(s, change)
    assert s.sources[0].text == "a\nb\nc"
    assert s.revision == 1


def test_edit_to_unknown_source_is_rejected():
    with pytest.raises(UnknownSource):
        apply_change(section(), Change(edits=[SectionEdit("source", "x", 0, 0, [])]))


def test_replacements_apply_after_edits():
    s = section()
    apply_change(
        s,
        Change(
            sources=[Source("transcript", "new")],
            instruction="Outline.",
            pinned=["- one"],
        ),
    )
    assert [x.text for x in s.sources] == ["new"]
    assert (s.instruction, s.pinned) == ("Outline.", ["- one"])


def test_an_empty_change_keeps_the_revision():
    s = section()
    apply_change(s, Change())
    assert s.revision == 1
