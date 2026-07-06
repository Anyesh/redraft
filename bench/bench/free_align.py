from collections.abc import Sequence
from difflib import SequenceMatcher


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Leading tokens shared by two freely-decoded sequences. This is the honest
    zero-machinery reuse: greedy decoding under the new context reproduces the old
    output up to here and no further, because at the first differing token the two
    decodes have diverged for real (no teacher forcing keeps them aligned).
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def matching_blocks(
    old_ids: Sequence[int], new_ids: Sequence[int]
) -> list[tuple[int, int, int]]:
    """Order-preserving contiguous matches between the old and new outputs as
    (old_start, new_start, size), dropping difflib's zero-size terminator.
    """
    sm = SequenceMatcher(a=list(old_ids), b=list(new_ids), autojunk=False)
    return [(i, j, n) for i, j, n in sm.get_matching_blocks() if n > 0]


def free_salvage(
    old_ids: Sequence[int], new_ids: Sequence[int], resync_cost: int = 0
) -> float:
    """Fraction of the NEW (target) output that could be spliced from the old output.

    Both sequences are produced by free greedy decoding, so a matched run genuinely
    appears in both. A run is only counted if it is longer than `resync_cost`, the
    token overhead of re-anchoring the decode onto a reused span; short runs are not
    worth splicing. This is the reuse ceiling for output-side incrementality: the
    achievable number is at or below it once KV-splice feasibility is accounted for.
    """
    if not new_ids:
        return 0.0
    salvaged = sum(
        size for _, _, size in matching_blocks(old_ids, new_ids) if size > resync_cost
    )
    return salvaged / len(new_ids)
