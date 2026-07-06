from pathlib import Path

from bench.measure_stab import load_cases

CASES_DIR = Path(__file__).parent.parent / "cases"


def test_load_cases_no_slice_returns_all():
    cases = load_cases(CASES_DIR)
    assert len(cases) == 36


def test_load_cases_start_and_count_slices():
    all_cases = load_cases(CASES_DIR)
    sliced = load_cases(CASES_DIR, start=5, count=10)
    assert sliced == all_cases[5:15]


def test_load_cases_start_only_runs_to_end():
    all_cases = load_cases(CASES_DIR)
    sliced = load_cases(CASES_DIR, start=30)
    assert sliced == all_cases[30:]
