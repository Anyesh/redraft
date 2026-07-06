import pytest

from bench.measure_stab import correctness as stab_correctness
from bench.measure_wall import correctness as wall_correctness

CORRECTNESS_FNS = [stab_correctness, wall_correctness]


@pytest.mark.parametrize("correctness", CORRECTNESS_FNS)
def test_expect_new_matches_regardless_of_case(correctness):
    case = {"expect_new": ["blocked"], "forbid_new": []}
    assert correctness("Atlas is currently BLOCKED for launch.", case) == "pass"


@pytest.mark.parametrize("correctness", CORRECTNESS_FNS)
def test_forbid_new_flags_regardless_of_case(correctness):
    case = {"expect_new": [], "forbid_new": ["on track"]}
    assert correctness("Atlas remains ON TRACK for launch.", case) == "FAIL"


@pytest.mark.parametrize("correctness", CORRECTNESS_FNS)
def test_missing_expect_new_fails(correctness):
    case = {"expect_new": ["blocked"], "forbid_new": []}
    assert correctness("Atlas remains on track for launch.", case) == "FAIL"


@pytest.mark.parametrize("correctness", CORRECTNESS_FNS)
def test_no_annotations_returns_empty(correctness):
    assert correctness("anything", {}) == ""
