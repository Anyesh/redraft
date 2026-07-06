import pytest

from bench.rule_specs import parse_rule, parse_rule_specs
from bench.stabilize import (
    ConfidenceGatedTau,
    EntropyFloorTau,
    EntropyScaledTau,
    GatedEntropyTau,
    TauRule,
)


def test_parse_tau_rule():
    label, rule = parse_rule("tau:1.5")
    assert label == "tau:1.5"
    assert rule == TauRule(tau=1.5)


def test_parse_confidence_gated_rule():
    label, rule = parse_rule("confidence_gated:tau=3.0,p_cut=0.85")
    assert label == "confidence_gated:tau=3.0,p_cut=0.85"
    assert rule == ConfidenceGatedTau(tau=3.0, p_cut=0.85)


def test_parse_confidence_gated_rule_defaults():
    _, rule = parse_rule("confidence_gated:tau=2.0")
    assert rule == ConfidenceGatedTau(tau=2.0, p_cut=0.9)


def test_parse_entropy_scaled_rule():
    label, rule = parse_rule("entropy_scaled:tau=2.5")
    assert label == "entropy_scaled:tau=2.5"
    assert rule == EntropyScaledTau(tau=2.5)


def test_parse_entropy_floor_rule():
    label, rule = parse_rule("entropy_floor:tau=2.5,floor=0.5")
    assert label == "entropy_floor:tau=2.5,floor=0.5"
    assert rule == EntropyFloorTau(tau=2.5, floor=0.5)


def test_parse_entropy_floor_rule_defaults():
    _, rule = parse_rule("entropy_floor:tau=2.0")
    assert rule == EntropyFloorTau(tau=2.0, floor=0.0)


def test_parse_gated_entropy_rule():
    label, rule = parse_rule("gated_entropy:tau=2.5,p_cut=0.85")
    assert label == "gated_entropy:tau=2.5,p_cut=0.85"
    assert rule == GatedEntropyTau(tau=2.5, p_cut=0.85)


def test_parse_gated_entropy_rule_defaults():
    _, rule = parse_rule("gated_entropy:tau=2.0")
    assert rule == GatedEntropyTau(tau=2.0, p_cut=0.9)


def test_parse_unknown_kind_raises():
    with pytest.raises(ValueError):
        parse_rule("bogus:tau=1.0")


def test_parse_rule_specs_preserves_order():
    specs = parse_rule_specs(
        ["tau:0.0", "tau:3.0", "confidence_gated:tau=3.0,p_cut=0.9"]
    )
    assert [label for label, _ in specs] == [
        "tau:0.0",
        "tau:3.0",
        "confidence_gated:tau=3.0,p_cut=0.9",
    ]
    assert [rule for _, rule in specs] == [
        TauRule(tau=0.0),
        TauRule(tau=3.0),
        ConfidenceGatedTau(tau=3.0, p_cut=0.9),
    ]
