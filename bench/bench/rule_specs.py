from bench.stabilize import (
    AcceptRule,
    ConfidenceGatedTau,
    EntropyFloorTau,
    EntropyScaledTau,
    GatedEntropyTau,
    TauRule,
)


def parse_rule(spec: str) -> tuple[str, AcceptRule]:
    """Parse a rule spec string into (label, rule) for the calibration drivers.

    Formats: "tau:<tau>", "confidence_gated:tau=<tau>,p_cut=<p_cut>" (p_cut
    defaults to 0.9), "entropy_scaled:tau=<tau>", "entropy_floor:tau=<tau>,
    floor=<floor>" (floor defaults to 0.0), "gated_entropy:tau=<tau>,
    p_cut=<p_cut>" (p_cut defaults to 0.9). The label is the spec string
    itself, so CSV rows and printed output stay traceable to the exact params.
    """
    kind, _, params = spec.partition(":")
    if kind == "tau":
        return spec, TauRule(tau=float(params))

    kv = dict(p.split("=", 1) for p in params.split(",")) if params else {}
    if kind == "confidence_gated":
        return spec, ConfidenceGatedTau(
            tau=float(kv.get("tau", 0.0)), p_cut=float(kv.get("p_cut", 0.9))
        )
    if kind == "entropy_scaled":
        return spec, EntropyScaledTau(tau=float(kv.get("tau", 0.0)))
    if kind == "entropy_floor":
        return spec, EntropyFloorTau(
            tau=float(kv.get("tau", 0.0)), floor=float(kv.get("floor", 0.0))
        )
    if kind == "gated_entropy":
        return spec, GatedEntropyTau(
            tau=float(kv.get("tau", 0.0)), p_cut=float(kv.get("p_cut", 0.9))
        )

    raise ValueError(f"unknown rule kind {kind!r} in spec {spec!r}")


def parse_rule_specs(specs: list[str]) -> list[tuple[str, AcceptRule]]:
    return [parse_rule(spec) for spec in specs]
