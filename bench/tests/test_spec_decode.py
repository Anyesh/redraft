import pytest

from bench.spec_decode import spec_savings


def test_identical_outputs_one_pass_per_window():
    # 32 identical tokens, window 16 -> 2 verification passes, no serial decode
    ids = list(range(32))
    r = spec_savings(ids, ids, window=16, reanchor_cost=1)
    assert r["forward_passes"] == 2
    assert r["baseline_passes"] == 32
    assert r["verified_tokens"] == 32
    assert r["serial_tokens"] == 0
    assert r["speedup"] == pytest.approx(16.0)


def test_fully_divergent_no_speedup():
    old = [1, 2, 3, 4]
    new = [5, 6, 7, 8]
    r = spec_savings(old, new, window=16, reanchor_cost=0)
    # one replace block of 4 serial tokens, no verification
    assert r["forward_passes"] == 4
    assert r["serial_tokens"] == 4
    assert r["verified_tokens"] == 0
    assert r["speedup"] == pytest.approx(1.0)


def test_empty_new_is_zero():
    r = spec_savings([1, 2, 3], [], window=8)
    assert r["forward_passes"] == 0
    assert r["speedup"] == 0.0


def test_prefix_reused_tail_diverges():
    # equal-length divergent tail so difflib emits a single replace block
    old = list(range(20)) + [100, 101]
    new = list(range(20)) + [200, 201]
    r = spec_savings(old, new, window=16, reanchor_cost=1)
    # equal run 20 -> ceil(20/16)=2 passes, verified 20; replace 2 -> 2 serial + 1 reanchor
    assert r["verified_tokens"] == 20
    assert r["serial_tokens"] == 2
    assert r["forward_passes"] == 2 + 2 + 1
    assert r["speedup"] == pytest.approx(22 / 5)


def test_reanchor_cost_charged_per_divergence():
    old = [1, 2, 3, 4, 5, 6]
    new = [1, 2, 9, 4, 5, 6]
    # equal[1,2] (2), replace[9] (1 serial, 1 transition), equal[4,5,6] (3)
    r0 = spec_savings(old, new, window=16, reanchor_cost=0)
    r2 = spec_savings(old, new, window=16, reanchor_cost=2)
    assert r2["forward_passes"] == r0["forward_passes"] + 2


def test_window_reduces_verification_passes():
    ids = list(range(64))
    wide = spec_savings(ids, ids, window=64)
    narrow = spec_savings(ids, ids, window=8)
    assert wide["forward_passes"] == 1
    assert narrow["forward_passes"] == 8
