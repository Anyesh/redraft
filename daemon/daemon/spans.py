"""Turns the engine's per-token provenance into the contract's span events.

Every offset is in UTF-16 code units, because the consumer patches a browser
DOM with JavaScript string indices. The emitted spans tile both the old draft
and the new text in order with no gaps, and a held span's old and new text are
equal; the builder checks that and downgrades a held span to replaced when a
multi-byte character split across a divergence makes them disagree.
"""

import codecs
from collections.abc import Sequence

Event = tuple[str, dict]


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def piece_ends(pieces: Sequence[str | list[int]]) -> list[int]:
    """Cumulative UTF-16 end offset after each token of a `/tokenize` piece list.

    A piece is a string, or a byte list when it is not valid UTF-8 on its own.
    Bytes are decoded incrementally, so a character split across tokens counts
    toward the token that completes it, matching how llama-server holds back
    incomplete UTF-8 when streaming.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    ends, n = [], 0
    for piece in pieces:
        raw = piece.encode() if isinstance(piece, str) else bytes(piece)
        n += utf16_len(decoder.decode(raw))
        ends.append(n)
    return ends


class SpanBuilder:
    def __init__(self, old_text: str, old_ends: Sequence[int]):
        self._old16 = old_text.encode("utf-16-le")
        self._old_ends = list(old_ends)
        self._new16 = bytearray()
        self._old_cursor = 0
        self._held_chars = 0
        self._run: str | None = None
        self._run_new_start = 0
        self._held_first = 0
        self._held_last = 0
        self.token_ends: list[int] = []

    @property
    def new_text(self) -> str:
        return self._new16.decode("utf-16-le")

    @property
    def reused_chars(self) -> float:
        total = self._new_len()
        return self._held_chars / total if total else 0.0

    def feed(self, text: str, srcs: Sequence[int] | None) -> list[Event]:
        events: list[Event] = []
        at = self._new_len()
        self._new16 += text.encode("utf-16-le")
        end = self._new_len()
        if srcs:
            self.token_ends.extend([at] * (len(srcs) - 1) + [end])

        held = bool(srcs) and all(s >= 0 for s in srcs)
        if held and self._old_start(srcs[0]) < self._old_cursor:
            held = False

        if held:
            continues = self._run == "held" and srcs[0] == self._held_last + 1
            if continues:
                self._held_last = srcs[-1]
            else:
                events += self._start_held(srcs[0], srcs[-1], at)
        elif self._run != "new":
            if self._run == "held":
                events += self._close_held(at)
            self._run = "new"
            self._run_new_start = at

        if text:
            events.append(("delta", {"text": text, "kind": "held" if held else "new"}))
        return events

    def finish(self) -> list[Event]:
        events: list[Event] = []
        at = self._new_len()
        old_len = len(self._old16) // 2
        if self._run == "held":
            events += self._close_held(at)
            events += self._deletion(old_len, at)
        elif self._run == "new":
            events += self._close_new(at, old_len)
        else:
            events += self._deletion(old_len, at)
        self._run = None
        return events

    def _new_len(self) -> int:
        return len(self._new16) // 2

    def _old_start(self, token: int) -> int:
        return self._old_ends[token - 1] if token > 0 else 0

    def _start_held(self, first: int, last: int, at: int) -> list[Event]:
        events: list[Event] = []
        old_at = self._old_start(first)
        if self._run == "held":
            events += self._close_held(at)
        if self._run == "new":
            events += self._close_new(at, old_at)
        else:
            events += self._deletion(old_at, at)
        self._run = "held"
        self._run_new_start = at
        self._held_first, self._held_last = first, last
        return events

    def _close_held(self, at: int) -> list[Event]:
        old = [self._old_start(self._held_first), self._old_ends[self._held_last]]
        new = [self._run_new_start, at]
        self._run = None
        self._old_cursor = old[1]
        new_part = self._slice(self._new16, new)
        if self._slice(self._old16, old) != new_part:
            return [
                ("span", {"kind": "replaced", "old": old, "new": new, "text": new_part})
            ]
        self._held_chars += new[1] - new[0]
        return [("span", {"kind": "held", "old": old, "new": new})]

    def _close_new(self, at: int, old_until: int) -> list[Event]:
        old = [self._old_cursor, old_until]
        new = [self._run_new_start, at]
        self._run = None
        self._old_cursor = old_until
        if old[0] == old[1] and new[0] == new[1]:
            return []
        text = self._slice(self._new16, new)
        return [("span", {"kind": "replaced", "old": old, "new": new, "text": text})]

    def _deletion(self, old_until: int, at: int) -> list[Event]:
        if old_until <= self._old_cursor:
            return []
        old = [self._old_cursor, old_until]
        self._old_cursor = old_until
        return [("span", {"kind": "replaced", "old": old, "new": [at, at], "text": ""})]

    @staticmethod
    def _slice(buf: bytes | bytearray, rng: list[int]) -> str:
        return bytes(buf[2 * rng[0] : 2 * rng[1]]).decode("utf-16-le")
