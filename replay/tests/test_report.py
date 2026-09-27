from replay.report import edit_distance, summarize


def record(uid, kind, b_ms, r_ms, reused, b_text, r_text, pinned=()):
    return {
        "unit_id": uid, "kind": "rederive", "document_kind": kind,
        "pinned": list(pinned),
        "baseline": {"text": b_text, "done": {"wall_ms": b_ms}},
        "redraft": {"text": r_text, "done": {"wall_ms": r_ms, "reused": reused}},
    }


def test_edit_distance_is_normalized():
    assert edit_distance("abc", "abc") == 0.0
    assert edit_distance("abc", "abd") == 1 / 3
    assert edit_distance("", "") == 0.0
    assert edit_distance("", "xy") == 1.0


def test_summary_per_kind_and_overall_with_verdicts():
    results = [
        record("a", "plan", 300, 100, 0.8, "x\nkeep", "x\nkeep", pinned=["keep"]),
        record("b", "plan", 200, 200, 0.5, "same", "same", pinned=["gone"]),
        record("c", "reply", 100, 50, 0.9, "hi", "ho"),
    ]
    verdicts = {
        "a": {"failed": False, "new_errors": [], "lost_facts": []},
        "b": {"failed": True, "new_errors": [{"claim": "bad"}], "lost_facts": []},
        "c": {"failed": False, "new_errors": [], "lost_facts": []},
    }
    s = summarize(results, verdicts, judged=True)
    plan = s["by_kind"]["plan"]
    assert plan["units"] == 2
    assert plan["median_speedup"] == 2.0
    assert plan["median_reused"] == 0.65
    assert plan["pinned_survival"] == 0.5
    assert plan["failed"] == 1
    assert s["failures"] == [{"unit_id": "b", "new_errors": ["bad"], "lost_facts": []}]
    assert s["overall"]["units"] == 3
    assert s["judged"] is True


def test_unjudged_summary_reports_no_failure_counts():
    s = summarize([record("a", "plan", 1, 1, 0.1, "a", "a")], {}, judged=False)
    assert s["judged"] is False
    assert s["overall"]["failed"] is None
