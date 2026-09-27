from dataclasses import dataclass


class BadRange(ValueError):
    pass


@dataclass(frozen=True)
class LineEdit:
    start: int
    end: int
    lines: list[str]


def apply_line_edit(text: str, edit: LineEdit) -> str:
    """Replace lines [start, end) of `text`, where lines are `text.split("\\n")`."""
    lines = text.split("\n")
    if not 0 <= edit.start <= edit.end <= len(lines):
        raise BadRange(
            f"range [{edit.start}, {edit.end}) outside a text of {len(lines)} lines"
        )
    return "\n".join(lines[: edit.start] + list(edit.lines) + lines[edit.end :])
