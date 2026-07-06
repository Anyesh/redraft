import math

import pytest

from bench.stabilize import (
    ConfidenceGatedTau,
    EntropyFloorTau,
    EntropyScaledTau,
    GatedEntropyTau,
    TauRule,
    stabilize,
)

EOS = 999


def top(*pairs):
    return list(pairs)


def scripted(steps):
    def model(prefix_ids, draft_rest):
        return steps[len(prefix_ids)]

    return model


def test_tau_rule_accepts_within_band():
    rule = TauRule(tau=1.0)
    assert rule(5, top((8, -0.5), (5, -1.0))) is True


def test_tau_rule_rejects_outside_band():
    rule = TauRule(tau=0.2)
    assert rule(5, top((8, -0.5), (5, -1.0))) is False


def test_tau_rule_rejects_draft_below_topk():
    rule = TauRule(tau=10.0)
    assert rule(4, top((1, -0.1), (2, -0.5))) is False


def test_confidence_gated_tau_requires_exact_match_when_confident():
    # argmax prob = exp(-0.05) ~= 0.951, above p_cut, so only the exact
    # argmax id is acceptable even though the draft is within tau of it
    rule = ConfidenceGatedTau(tau=1.0, p_cut=0.9)
    assert rule(5, top((8, -0.05), (5, -0.3))) is False
    assert rule(8, top((8, -0.05), (5, -0.3))) is True


def test_confidence_gated_tau_allows_tau_band_when_unconfident():
    # argmax prob = exp(-2.0) ~= 0.135, below p_cut, so the tau band applies
    rule = ConfidenceGatedTau(tau=1.0, p_cut=0.9)
    assert rule(5, top((8, -2.0), (5, -2.5))) is True


def test_confidence_gated_tau_rejects_draft_below_topk():
    rule = ConfidenceGatedTau(tau=10.0, p_cut=0.9)
    assert rule(4, top((1, -0.1), (2, -0.5))) is False


def test_entropy_scaled_tau_shrinks_on_sharp_distribution():
    # same argmax-draft gap (1.0 nats) as the flat case below, but the tail
    # carries negligible mass so entropy is low and the effective tolerance
    # (tau * H/log(k) ~= 0.84) falls short of the gap
    sharp = top((1, -0.01), (2, -1.01), (3, -20.0), (4, -20.0))
    rule = EntropyScaledTau(tau=2.0)
    assert rule(2, sharp) is False


def test_entropy_scaled_tau_widens_on_flat_distribution():
    # identical argmax and draft logprobs to the sharp case, but the tail
    # carries real mass so entropy is higher and the effective tolerance
    # (tau * H/log(k) ~= 1.91) clears the same 1.0 nat gap
    flat = top((1, -0.01), (2, -1.01), (3, -0.02), (4, -0.03))
    rule = EntropyScaledTau(tau=2.0)
    assert rule(2, flat) is True


def test_entropy_scaled_tau_rejects_draft_below_topk():
    rule = EntropyScaledTau(tau=10.0)
    assert rule(4, top((1, -0.1), (2, -0.5))) is False


def test_entropy_scaled_tau_single_candidate_degenerates_to_exact():
    rule = EntropyScaledTau(tau=5.0)
    assert rule(1, top((1, -0.1))) is True
    assert rule(2, top((1, -0.1))) is False


def test_entropy_floor_tau_matches_entropy_scaled_when_floor_zero():
    sharp = top((1, -0.01), (2, -1.01), (3, -20.0), (4, -20.0))
    flat = top((1, -0.01), (2, -1.01), (3, -0.02), (4, -0.03))
    for dist in (sharp, flat):
        for tau in (0.0, 1.0, 2.0, 5.0):
            assert EntropyFloorTau(tau=tau, floor=0.0)(2, dist) == EntropyScaledTau(
                tau=tau
            )(2, dist)


def test_entropy_floor_tau_keeps_band_open_on_sharp_distribution():
    # same sharp distribution EntropyScaledTau(tau=2.0) rejects (effective tau
    # ~= 0.84 nats, short of the 1.0 nat gap); a floor above the gap holds even
    # though the distribution is sharp, unlike EntropyScaledTau which shrinks to 0
    sharp = top((1, -0.01), (2, -1.01), (3, -20.0), (4, -20.0))
    assert EntropyScaledTau(tau=2.0)(2, sharp) is False
    assert EntropyFloorTau(tau=2.0, floor=1.5)(2, sharp) is True


def test_entropy_floor_tau_rejects_draft_below_topk():
    rule = EntropyFloorTau(tau=10.0, floor=5.0)
    assert rule(4, top((1, -0.1), (2, -0.5))) is False


def test_entropy_floor_tau_single_candidate_uses_floor():
    rule = EntropyFloorTau(tau=5.0, floor=0.2)
    assert rule(1, top((1, -0.1))) is True
    assert rule(2, top((1, -0.1))) is False


def test_gated_entropy_tau_requires_exact_match_when_confident():
    rule = GatedEntropyTau(tau=1.0, p_cut=0.9)
    assert rule(5, top((8, -0.05), (5, -0.3))) is False
    assert rule(8, top((8, -0.05), (5, -0.3))) is True


def test_gated_entropy_tau_scales_band_by_entropy_when_unconfident():
    # argmax prob = exp(-2.0) ~= 0.135, below p_cut, so the entropy-scaled
    # band applies rather than a flat tau band; same 1.0 nat argmax-draft gap
    # in both, but the flatter tail's higher entropy (eff tau ~=1.91 vs 0.84)
    # is the only thing that flips accept/reject
    sharp = top((1, -2.0), (2, -3.0), (3, -25.0), (4, -25.0))
    flat = top((1, -2.0), (2, -3.0), (3, -2.01), (4, -2.02))
    rule = GatedEntropyTau(tau=2.0, p_cut=0.9)
    assert rule(2, sharp) is False
    assert rule(2, flat) is True


def test_gated_entropy_tau_rejects_draft_below_topk():
    rule = GatedEntropyTau(tau=10.0, p_cut=0.9)
    assert rule(4, top((1, -0.1), (2, -0.5))) is False


def test_stabilize_accepts_custom_rule():
    old = [1, 5]
    steps = [
        top((1, -0.1), (7, -2.0)),
        top((8, -0.05), (5, -0.3)),
        top(
            (EOS, -0.1),
        ),
    ]
    # ConfidenceGatedTau at high confidence forces the 5 draft to be rejected
    # even though plain TauRule(tau=1.0) would have held it
    r = stabilize(
        old, scripted(steps), rule=ConfidenceGatedTau(tau=1.0, p_cut=0.9), eos_ids={EOS}
    )
    assert r.emitted == [1, 8]
    assert r.events == ["held", "serial"]


def test_stabilize_default_rule_matches_bare_tau():
    old = [1, 5]
    steps = [
        top((1, -0.1), (7, -2.0)),
        top((8, -0.5), (5, -1.0)),
        top(
            (EOS, -0.1),
        ),
    ]
    by_tau = stabilize(old, scripted(steps), tau=1.0, eos_ids={EOS})
    by_rule = stabilize(old, scripted(steps), rule=TauRule(tau=1.0), eos_ids={EOS})
    assert by_tau == by_rule
