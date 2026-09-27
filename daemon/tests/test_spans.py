from daemon.spans import SpanBuilder, piece_ends, utf16_len


def per_char_ends(text: str) -> list[int]:
    ends, n = [], 0
    for ch in text:
        n += utf16_len(ch)
        ends.append(n)
    return ends


def run(old: str, chunks: list[tuple[str, list[int] | None]]):
    builder = SpanBuilder(old, per_char_ends(old))
    events = []
    for text, srcs in chunks:
        events.extend(builder.feed(text, srcs))
    events.extend(builder.finish())
    return builder, events


def spans_of(events):
    return [data for name, data in events if name == "span"]


def assert_tiles(old: str, new: str, spans: list[dict]) -> None:
    old16 = old.encode("utf-16-le")
    new16 = new.encode("utf-16-le")
    old_at = new_at = 0
    for span in spans:
        assert span["old"][0] == old_at and span["new"][0] == new_at, span
        old_at, new_at = span["old"][1], span["new"][1]
        new_part = new16[2 * span["new"][0] : 2 * span["new"][1]].decode("utf-16-le")
        if span["kind"] == "held":
            old_part = old16[2 * span["old"][0] : 2 * span["old"][1]].decode(
                "utf-16-le"
            )
            assert old_part == new_part
        else:
            assert span["text"] == new_part
    assert old_at == len(old16) // 2
    assert new_at == len(new16) // 2


def test_divergence_in_the_middle_yields_held_replaced_held():
    builder, events = run(
        "ABCDEF",
        [("A", [0]), ("B", [1]), ("X", [-1]), ("D", [3]), ("E", [4]), ("F", [5])],
    )
    spans = spans_of(events)
    assert spans == [
        {"kind": "held", "old": [0, 2], "new": [0, 2]},
        {"kind": "replaced", "old": [2, 3], "new": [2, 3], "text": "X"},
        {"kind": "held", "old": [3, 6], "new": [3, 6]},
    ]
    assert builder.new_text == "ABXDEF"
    assert_tiles("ABCDEF", "ABXDEF", spans)


def test_truncated_output_deletes_the_old_tail():
    _, events = run("ABCDEF", [("A", [0]), ("B", [1])])
    spans = spans_of(events)
    assert spans[-1] == {"kind": "replaced", "old": [2, 6], "new": [2, 2], "text": ""}
    assert_tiles("ABCDEF", "AB", spans)


def test_insertion_between_held_runs_has_empty_old_range():
    _, events = run("ABC", [("A", [0]), ("X", [-1]), ("B", [1]), ("C", [2])])
    spans = spans_of(events)
    assert {"kind": "replaced", "old": [1, 1], "new": [1, 2], "text": "X"} in spans
    assert_tiles("ABC", "AXBC", spans)


def test_reanchor_jump_without_new_text_is_a_deletion():
    _, events = run("ABCDE", [("A", [0]), ("D", [3]), ("E", [4])])
    spans = spans_of(events)
    assert spans == [
        {"kind": "held", "old": [0, 1], "new": [0, 1]},
        {"kind": "replaced", "old": [1, 3], "new": [1, 1], "text": ""},
        {"kind": "held", "old": [3, 5], "new": [1, 3]},
    ]


def test_baseline_stream_is_one_replaced_span():
    builder, events = run("old text", [("new ", None), ("text", None)])
    spans = spans_of(events)
    assert spans == [
        {"kind": "replaced", "old": [0, 8], "new": [0, 8], "text": "new text"}
    ]
    assert builder.reused_chars == 0.0


def test_deltas_are_tagged_and_concatenate_to_the_new_text():
    builder, events = run("AB", [("A", [0]), ("Z", [-1])])
    deltas = [data for name, data in events if name == "delta"]
    assert deltas == [{"text": "A", "kind": "held"}, {"text": "Z", "kind": "new"}]
    assert "".join(d["text"] for d in deltas) == builder.new_text


def test_offsets_are_utf16_code_units():
    old = "a😀b"
    builder = SpanBuilder(old, [1, 1, 3, 4])
    events = []
    events += builder.feed("a", [0])
    events += builder.feed("😀", [1, 2])
    events += builder.feed("c", [-1])
    events += builder.finish()
    spans = spans_of(events)
    assert spans[0] == {"kind": "held", "old": [0, 3], "new": [0, 3]}
    assert_tiles(old, "a😀c", spans)
    assert builder.token_ends == [1, 1, 3, 4]


def test_held_span_whose_text_disagrees_is_downgraded_to_replaced():
    builder = SpanBuilder("ABC", [1, 2, 3])
    events = builder.feed("Q", [0]) + builder.finish()
    spans = spans_of(events)
    assert spans[0]["kind"] == "replaced"
    assert_tiles("ABC", "Q", spans)


def test_reused_chars_counts_new_text_inside_held_spans():
    builder, _ = run("ABCD", [("A", [0]), ("B", [1]), ("X", [-1]), ("Y", [-1])])
    assert builder.reused_chars == 0.5


def test_piece_ends_attribute_split_utf8_to_the_completing_token():
    smile = "😀".encode()
    pieces = ["a", list(smile[:2]), list(smile[2:]), "b"]
    assert piece_ends(pieces) == [1, 1, 3, 4]
