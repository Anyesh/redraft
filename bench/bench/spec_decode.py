from collections.abc import Sequence
from difflib import SequenceMatcher
from math import ceil


def spec_savings(
    old_ids: Sequence[int],
    new_ids: Sequence[int],
    window: int = 16,
    reanchor_cost: int = 1,
) -> dict:
    """Forward-pass cost of producing the new greedy output using the old output as a
    self-speculative draft, versus plain autoregressive decoding.

    Both sequences are true greedy outputs, so a SequenceMatcher `equal` block is a span
    the new decode actually reproduces from the old one: it is verified in parallel,
    `window` draft tokens per forward pass, all accepted. A divergent block (`replace`/
    `insert`) has no usable draft, so its new tokens are decoded serially, one pass each,
    which is where the exactness guarantee comes from (the corrected token is the true
    argmax, never a forced stale token). Each divergence also charges `reanchor_cost`
    passes to re-locate the draft window. `delete` blocks (old tokens absent from the new
    output) cost only the re-anchor. Baseline is one pass per new token.

    Returns forward_passes, baseline_passes, verified_tokens, serial_tokens, speedup.
    """
    baseline = len(new_ids)
    if baseline == 0:
        return {
            "forward_passes": 0,
            "baseline_passes": 0,
            "verified_tokens": 0,
            "serial_tokens": 0,
            "speedup": 0.0,
        }

    sm = SequenceMatcher(a=list(old_ids), b=list(new_ids), autojunk=False)
    passes = 0
    verified = 0
    serial = 0
    transitions = 0
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        span = j2 - j1
        if tag == "equal":
            passes += ceil(span / window)
            verified += span
        elif tag in ("replace", "insert"):
            passes += span
            serial += span
            transitions += 1
        elif tag == "delete":
            transitions += 1

    passes += transitions * reanchor_cost
    return {
        "forward_passes": passes,
        "baseline_passes": baseline,
        "verified_tokens": verified,
        "serial_tokens": serial,
        "speedup": round(baseline / passes, 4) if passes else 0.0,
    }
