import pytest

from bench.free_align import common_prefix_len, free_salvage, matching_blocks


def test_common_prefix_unequal_lengths():
    assert common_prefix_len([1, 2, 3], [1, 2, 9, 4, 5]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([5], [6]) == 0


def test_matching_blocks_drops_terminator():
    blocks = matching_blocks([1, 2, 3], [1, 2, 3])
    assert blocks == [(0, 0, 3)]


def test_matching_blocks_with_insertion():
    # new output inserts a token in the middle
    blocks = matching_blocks([1, 2, 3, 4], [1, 2, 9, 3, 4])
    assert (0, 0, 2) in blocks
    assert (2, 3, 2) in blocks


def test_free_salvage_identical():
    assert free_salvage([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_free_salvage_disjoint():
    assert free_salvage([1, 2, 3], [4, 5, 6]) == 0.0


def test_free_salvage_empty_new_is_zero():
    assert free_salvage([1, 2], []) == 0.0


def test_free_salvage_normalizes_by_new_length():
    # old has an extra trailing run the new target does not need; denominator is new
    old = [1, 2, 3, 7, 8, 9]
    new = [1, 2, 3]
    assert free_salvage(old, new) == pytest.approx(1.0)


def test_free_salvage_resync_cost_filters_short_runs():
    old = [1, 2, 3, 4, 5, 6]
    new = [1, 2, 9, 4, 5, 6]
    # matched runs against new: [1,2] (size 2) and [4,5,6] (size 3)
    assert free_salvage(old, new, resync_cost=0) == pytest.approx(5 / 6)
    assert free_salvage(old, new, resync_cost=2) == pytest.approx(3 / 6)
    assert free_salvage(old, new, resync_cost=3) == pytest.approx(0.0)
