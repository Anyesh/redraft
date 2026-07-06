from collections.abc import Sequence


def _check_lengths(old_tokens: Sequence[int], new_argmax: Sequence[int]) -> None:
    if len(old_tokens) != len(new_argmax):
        raise ValueError(
            f"token sequences must align position-wise, got {len(old_tokens)} vs {len(new_argmax)}"
        )


def accepted_prefix_len(old_tokens: Sequence[int], new_argmax: Sequence[int]) -> int:
    """Length of the leading span of the stale output that greedy decoding under the
    new context would reproduce exactly. Positions past the first mismatch are not
    counted even if they match again, because greedy would have diverged there.
    """
    _check_lengths(old_tokens, new_argmax)
    n = 0
    for old, new in zip(old_tokens, new_argmax):
        if old != new:
            break
        n += 1
    return n


def match_segments(
    old_tokens: Sequence[int], new_argmax: Sequence[int]
) -> list[tuple[bool, int]]:
    """Run-length segmentation of position-wise agreement: [(is_match, length), ...].

    Runs after the first mismatch are teacher-forced on the old continuation, so they
    estimate what a resync strategy could salvage rather than what plain greedy
    decoding would produce.
    """
    _check_lengths(old_tokens, new_argmax)
    segments: list[tuple[bool, int]] = []
    for old, new in zip(old_tokens, new_argmax):
        is_match = old == new
        if segments and segments[-1][0] == is_match:
            segments[-1] = (is_match, segments[-1][1] + 1)
        else:
            segments.append((is_match, 1))
    return segments


def salvage_fraction(
    old_tokens: Sequence[int], new_argmax: Sequence[int], resync_cost: int = 0
) -> float:
    """Teacher-forced UPPER BOUND on salvageable fraction, not an achievable number.

    Runs after the first mismatch are scored while the old tokens are still forced into
    the context, so this figure is inflated by exposure bias: a model conditioned on its
    own prior (stale) tokens keeps agreeing with them even when free decoding under the
    new context would have diverged. Milestone-zero measurement (2026-07-02) confirmed
    this directly. The only sound reuse metric here is `accepted_prefix_len`; measuring
    real segment-level salvage requires free re-decoding from each resync anchor, not
    teacher forcing. Kept for the upper-bound contrast only.
    """
    _check_lengths(old_tokens, new_argmax)
    if not old_tokens:
        return 0.0
    salvaged = sum(
        length
        for is_match, length in match_segments(old_tokens, new_argmax)
        if is_match and length > resync_cost
    )
    return salvaged / len(old_tokens)
