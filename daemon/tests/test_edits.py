import pytest

from daemon.edits import BadRange, LineEdit, apply_line_edit


def edit(start, end, lines):
    return LineEdit(start=start, end=end, lines=lines)


def test_replace_a_range():
    assert apply_line_edit("a\nb\nc\nd", edit(1, 3, ["X"])) == "a\nX\nd"


def test_empty_lines_delete_the_range():
    assert apply_line_edit("a\nb\nc", edit(1, 2, [])) == "a\nc"


def test_insert_before_a_line():
    assert apply_line_edit("a\nb", edit(1, 1, ["X", "Y"])) == "a\nX\nY\nb"


def test_append_at_line_count():
    assert apply_line_edit("a\nb", edit(2, 2, ["c"])) == "a\nb\nc"


def test_trailing_newline_is_an_empty_last_line():
    assert apply_line_edit("a\n", edit(1, 2, ["b"])) == "a\nb"


@pytest.mark.parametrize("start,end", [(-1, 0), (2, 1), (0, 4), (4, 4)])
def test_out_of_range_is_rejected(start, end):
    with pytest.raises(BadRange):
        apply_line_edit("a\nb\nc", edit(start, end, []))
