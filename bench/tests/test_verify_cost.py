import pytest

from bench.measure_verify_cost import fit_slope, port_speedup, summarize


def test_fit_slope_recovers_known_line():
    widths = [1, 8, 16, 32, 64]
    values = [10 + 2.5 * w for w in widths]
    slope, intercept = fit_slope(widths, values)
    assert slope == pytest.approx(2.5)
    assert intercept == pytest.approx(10.0)


def test_fit_slope_rejects_degenerate_widths():
    with pytest.raises(ValueError):
        fit_slope([16, 16, 16], [1.0, 2.0, 3.0])


def test_port_speedup_no_holding_is_parity():
    assert port_speedup(0.5, held=0.0) == pytest.approx(1.0)


def test_port_speedup_full_holding_is_inverse_slope():
    assert port_speedup(0.4, held=1.0) == pytest.approx(2.5)


def test_summarize_cheap_verification_commits():
    # a held draft token costs a small fraction of a serial decode, so held
    # fraction amplifies it into a clear doc_summary win: commit the port.
    verify = {w: [10.0 + 4.0 * w] for w in (1, 8, 16, 32, 64)}
    result = summarize(verify, decode_ms_per_token=15.0)
    assert result["slope_over_decode"] < 0.35
    assert result["port_projection"]["doc_summary"] >= 1.2
    assert result["verdict"].startswith("COMMIT")


def test_summarize_per_token_verification_kills():
    # verification tracks serial decode one-for-one: no in-engine batch can beat
    # it, so server-only is the real ceiling and doc_summary stays at parity.
    verify = {w: [15.0 * w] for w in (1, 8, 16, 32, 64)}
    result = summarize(verify, decode_ms_per_token=15.0)
    assert result["slope_over_decode"] > 0.9
    assert result["verdict"].startswith("KILL")


def test_summarize_measured_slope_commits_via_held_fraction():
    # the actual 14B measurement: slope ~0.42x decode. doc_summary held 0.728
    # projects ~1.7x, clearing the commit bar even though the slope alone looks
    # unremarkable.
    verify = {w: [23.0 + 6.79 * w] for w in (1, 8, 16, 32, 64)}
    result = summarize(verify, decode_ms_per_token=16.0)
    assert 0.4 < result["slope_over_decode"] < 0.45
    assert result["port_projection"]["doc_summary"] == pytest.approx(1.72, abs=0.05)
    assert result["verdict"].startswith("COMMIT")
