import pytest

from bench.align import accepted_prefix_len, match_segments, salvage_fraction


def test_accepted_prefix_full_match():
    assert accepted_prefix_len([1, 2, 3], [1, 2, 3]) == 3


def test_accepted_prefix_stops_at_first_mismatch():
    assert accepted_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2


def test_accepted_prefix_no_match():
    assert accepted_prefix_len([1, 2], [3, 4]) == 0


def test_accepted_prefix_empty():
    assert accepted_prefix_len([], []) == 0


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        accepted_prefix_len([1], [1, 2])
    with pytest.raises(ValueError):
        match_segments([1], [])
    with pytest.raises(ValueError):
        salvage_fraction([1], [1, 2])


def test_match_segments_runs():
    old = [1, 2, 3, 4, 5, 6]
    new = [1, 2, 9, 9, 5, 6]
    assert match_segments(old, new) == [(True, 2), (False, 2), (True, 2)]


def test_match_segments_single_run():
    assert match_segments([7, 7], [7, 7]) == [(True, 2)]
    assert match_segments([], []) == []


def test_salvage_counts_only_runs_longer_than_resync_cost():
    old = [1, 2, 3, 4, 5, 6, 7, 8]
    new = [1, 2, 3, 9, 5, 6, 9, 8]
    # runs: match 3, miss 1, match 2, miss 1, match 1
    assert salvage_fraction(old, new, resync_cost=0) == pytest.approx(6 / 8)
    assert salvage_fraction(old, new, resync_cost=1) == pytest.approx(5 / 8)
    assert salvage_fraction(old, new, resync_cost=2) == pytest.approx(3 / 8)
    assert salvage_fraction(old, new, resync_cost=3) == pytest.approx(0.0)


def test_salvage_empty_is_zero():
    assert salvage_fraction([], [], resync_cost=0) == 0.0
