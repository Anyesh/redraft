from dataclasses import dataclass, field
from typing import Literal

from daemon.edits import LineEdit, apply_line_edit
from daemon.sessions import Session, Source


class UnknownSource(ValueError):
    pass


@dataclass(frozen=True)
class SectionEdit:
    target: Literal["source", "derived"]
    source: str | None
    start: int
    end: int
    lines: list[str]


@dataclass
class Change:
    edits: list[SectionEdit] = field(default_factory=list)
    sources: list[Source] | None = None
    instruction: str | None = None
    pinned: list[str] | None = None


def apply_change(session: Session, change: Change) -> None:
    """Apply edits in order, then replacements. All-or-nothing: every edit is
    validated against working copies before the section is touched."""
    sources = {s.name: s.text for s in session.sources}
    derived = session.derived
    derived_edited = False
    for edit in change.edits:
        line_edit = LineEdit(edit.start, edit.end, edit.lines)
        if edit.target == "derived":
            derived = apply_line_edit(derived, line_edit)
            derived_edited = True
        elif edit.source in sources:
            sources[edit.source] = apply_line_edit(sources[edit.source], line_edit)
        else:
            raise UnknownSource(edit.source)

    changed = bool(change.edits)
    session.sources = [Source(s.name, sources[s.name]) for s in session.sources]
    session.derived = derived
    session.derived_dirty = session.derived_dirty or derived_edited
    if change.sources is not None:
        session.sources = list(change.sources)
        changed = True
    if change.instruction is not None:
        session.instruction = change.instruction
        changed = True
    if change.pinned is not None:
        session.pinned = list(change.pinned)
        changed = True
    if changed:
        session.touch()
