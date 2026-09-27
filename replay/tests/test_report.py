from replay.report import edit_distance, summarize


def record(uid, kind, b_ms, r_ms, reused, b_text, r_text, pinned=(), mode="rederive"):
    return {
        "unit_id": uid, "kind": mode, "document_kind": kind,
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


def verdict(failed):
    return {"failed": failed, "new_errors": [], "lost_facts": []}


def test_a_mode_ships_only_when_speed_facts_and_pins_all_hold():
    results = [
        record("a", "plan", 300, 100, 0.8, "x", "x", pinned=["x"]),
        record("b", "plan", 300, 150, 0.7, "y", "y"),
        record("c", "reply", 100, 120, 0.2, "z", "q", mode="revise"),
    ]
    verdicts = {
        "a": {**verdict(False), "control": verdict(False)},
        "b": {**verdict(True), "control": verdict(True)},
        "c": {**verdict(True), "control": verdict(False)},
    }
    s = summarize(results, verdicts, judged=True, min_units=1)
    rederive = s["by_mode"]["rederive"]
    assert rederive["failed"] == 1 and rederive["control_failed"] == 1
    assert rederive["ships"] is True
    revise = s["by_mode"]["revise"]
    assert revise["ships"] is False
    assert revise["blockers"] == ["median_speedup 0.833 < 1.3", "failed 1 > control_failed 0"]


def test_nothing_ships_unjudged():
    s = summarize([record("a", "plan", 300, 100, 0.8, "x", "x")], {}, judged=False)
    assert s["by_mode"]["rederive"]["ships"] is False
    assert "unjudged" in s["by_mode"]["rederive"]["blockers"]


def test_too_few_units_block_shipping():
    s = summarize([record("a", "plan", 300, 100, 0.8, "x", "x")], {"a": verdict(False)},
                  judged=True)
    assert "units 1 < 10" in s["by_mode"]["rederive"]["blockers"]
