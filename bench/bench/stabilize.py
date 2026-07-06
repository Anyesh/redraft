import math
from collections.abc import Callable, Sequence, Set
from dataclasses import dataclass, field
from math import ceil

StepModel = Callable[[Sequence[int], Sequence[int] | None], Sequence[tuple[int, float]]]
AcceptRule = Callable[[int, Sequence[tuple[int, float]]], bool]


def _logprob_of(draft_id: int, top: Sequence[tuple[int, float]]) -> float | None:
    return next((lp for tid, lp in top if tid == draft_id), None)


@dataclass
class TauRule:
    """Accept when the draft's logprob is within `tau` nats of the argmax. The
    bounded-mode contract's original rule; `tau=0` degenerates to exact greedy match.
    """

    tau: float = 0.0

    def __call__(self, draft_id: int, top: Sequence[tuple[int, float]]) -> bool:
        d_lp = _logprob_of(draft_id, top)
        if d_lp is None:
            return False
        argmax_lp = top[0][1]
        return argmax_lp - d_lp <= self.tau


@dataclass
class ConfidenceGatedTau:
    """Require an exact argmax match when the model is confident (`p_argmax >=
    p_cut`); below that confidence, fall back to the plain tau band. Targets the
    failure mode where a high-confidence position holds stale content that is
    outside the model's own top choice but still within a loose tau band.
    """

    tau: float = 0.0
    p_cut: float = 0.9

    def __call__(self, draft_id: int, top: Sequence[tuple[int, float]]) -> bool:
        d_lp = _logprob_of(draft_id, top)
        if d_lp is None:
            return False
        argmax_id, argmax_lp = top[0]
        if math.exp(argmax_lp) >= self.p_cut:
            return draft_id == argmax_id
        return argmax_lp - d_lp <= self.tau


@dataclass
class EntropyScaledTau:
    """Scale the tau band by the normalized entropy of the top-k distribution, so
    tolerance shrinks toward exactness as the distribution sharpens and widens
    toward `tau` as it flattens. Continuous version of `ConfidenceGatedTau`'s
    confidence gate.
    """

    tau: float = 0.0

    def __call__(self, draft_id: int, top: Sequence[tuple[int, float]]) -> bool:
        d_lp = _logprob_of(draft_id, top)
        if d_lp is None:
            return False
        argmax_lp = top[0][1]
        k = len(top)
        if k <= 1:
            effective_tau = 0.0
        else:
            probs = [math.exp(lp) for _, lp in top]
            total = sum(probs)
            probs = [p / total for p in probs]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            effective_tau = self.tau * entropy / math.log(k)
        return argmax_lp - d_lp <= effective_tau


@dataclass
class EntropyFloorTau:
    """Like `EntropyScaledTau`, but the tau band never fully closes: even a
    maximally sharp distribution still tolerates `floor` nats, so a single
    high-confidence miss doesn't force a reanchor. `floor=0` reproduces
    `EntropyScaledTau` exactly.
    """

    tau: float = 0.0
    floor: float = 0.0

    def __call__(self, draft_id: int, top: Sequence[tuple[int, float]]) -> bool:
        d_lp = _logprob_of(draft_id, top)
        if d_lp is None:
            return False
        argmax_lp = top[0][1]
        k = len(top)
        if k <= 1:
            effective_tau = self.floor
        else:
            probs = [math.exp(lp) for _, lp in top]
            total = sum(probs)
            probs = [p / total for p in probs]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            h_norm = entropy / math.log(k)
            effective_tau = self.floor + (self.tau - self.floor) * h_norm
        return argmax_lp - d_lp <= effective_tau


@dataclass
class GatedEntropyTau:
    """`ConfidenceGatedTau`'s exactness gate composed with `EntropyScaledTau`'s
    fallback: exact argmax match required when confident, otherwise the tau
    band scales with normalized entropy instead of staying flat.
    """

    tau: float = 0.0
    p_cut: float = 0.9

    def __call__(self, draft_id: int, top: Sequence[tuple[int, float]]) -> bool:
        d_lp = _logprob_of(draft_id, top)
        if d_lp is None:
            return False
        argmax_id, argmax_lp = top[0]
        if math.exp(argmax_lp) >= self.p_cut:
            return draft_id == argmax_id
        k = len(top)
        if k <= 1:
            effective_tau = 0.0
        else:
            probs = [math.exp(lp) for _, lp in top]
            total = sum(probs)
            probs = [p / total for p in probs]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            effective_tau = self.tau * entropy / math.log(k)
        return argmax_lp - d_lp <= effective_tau


@dataclass
class StabResult:
    emitted: list[int] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    divergences: int = 0

    @property
    def held_fraction(self) -> float:
        return self.events.count("held") / len(self.events) if self.events else 0.0


def _find_anchor(
    old_ids: Sequence[int], emitted: Sequence[int], anchor_len: int, watermark: int
) -> int | None:
    """Locate the last `anchor_len` emitted tokens inside the old output, at or past
    the watermark so drafting never re-consumes an old span (that would loop). Returns
    the old index right after the match, i.e. the next draft position.
    """
    if len(emitted) < anchor_len:
        return None
    gram = list(emitted[-anchor_len:])
    for start in range(watermark, len(old_ids) - anchor_len + 1):
        if list(old_ids[start : start + anchor_len]) == gram:
            return start + anchor_len
    return None


def stabilize(
    old_ids: Sequence[int],
    model: StepModel,
    tau: float = 0.0,
    anchor_len: int = 3,
    max_tokens: int = 512,
    eos_ids: Set[int] = frozenset(),
    rule: AcceptRule | None = None,
) -> StabResult:
    """Regenerate under the new context while holding spans of the old output.

    At each step the next old token is proposed as a draft; `rule` decides whether to
    emit (hold) it, so every held token satisfies whatever acceptance contract `rule`
    encodes. Defaults to `TauRule(tau)`, so passing only `tau` reproduces the original
    bounded-mode contract unchanged and `tau=0` degenerates to plain greedy decoding.
    On rejection the argmax is emitted and drafting re-anchors prompt-lookup style on
    the last `anchor_len` emitted tokens.
    """
    if rule is None:
        rule = TauRule(tau)
    result = StabResult()
    draft: int | None = 0 if old_ids else None
    watermark = 0

    while len(result.emitted) < max_tokens:
        draft_rest = old_ids[draft:] if draft is not None else None
        top = model(result.emitted, draft_rest)
        argmax_id, argmax_lp = top[0]

        chosen = None
        if draft is not None:
            d = old_ids[draft]
            if rule(d, top):
                chosen = (d, "held")
                draft += 1
                watermark = max(watermark, draft)
                if draft >= len(old_ids):
                    draft = None
            else:
                result.divergences += 1
                draft = None

        if chosen is None:
            if argmax_id in eos_ids:
                break
            chosen = (argmax_id, "serial")

        token, event = chosen
        if token in eos_ids:
            break
        result.emitted.append(token)
        result.events.append(event)

        if event == "serial":
            draft = _find_anchor(old_ids, result.emitted, anchor_len, watermark)
            if draft is not None:
                watermark = draft
                # a match ending at the end of the old output leaves nothing to draft
                if draft >= len(old_ids):
                    draft = None

    return result


def stab_passes(
    events: Sequence[str], window: int = 16, reanchor_cost: int = 1
) -> dict:
    """Forward-pass cost of the stabilized trajectory: held runs verify in parallel
    `window` tokens per pass, serial tokens cost one pass each, and each maximal
    serial run charges `reanchor_cost` to re-locate the draft. Baseline is one pass
    per emitted token, mirroring spec_savings in spec_decode.py.
    """
    baseline = len(events)
    if baseline == 0:
        return {"forward_passes": 0, "baseline_passes": 0, "speedup": 0.0}

    passes = 0
    run = 0
    prev = None
    for event in events:
        if event == "held":
            run += 1
        else:
            passes += 1
            if prev != "serial":
                passes += reanchor_cost
                if run:
                    passes += ceil(run / window)
                    run = 0
        prev = event
    if run:
        passes += ceil(run / window)
    return {
        "forward_passes": passes,
        "baseline_passes": baseline,
        "speedup": round(baseline / passes, 4),
    }
