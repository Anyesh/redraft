from replay.calibration import calibrate, load_cases


def test_shipped_cases_cover_both_outcomes():
    cases = load_cases()
    assert {c["expect"] for c in cases} == {"pass", "fail"}
    assert any(c["case_id"] == "stale_owner_held" for c in cases)


async def test_calibration_passes_only_when_every_case_matches():
    cases = [
        {"case_id": "a", "expect": "fail", "baseline": "B", "redraft": "R", "sources": []},
        {"case_id": "b", "expect": "pass", "baseline": "B", "redraft": "B", "sources": []},
    ]

    async def right(baseline, redraft, sources):
        return {"failed": baseline != redraft}

    async def wrong(baseline, redraft, sources):
        return {"failed": False}

    ok, rows = await calibrate(right, cases)
    assert ok and all(r["ok"] for r in rows)
    ok, rows = await calibrate(wrong, cases)
    assert not ok
    assert [r["case_id"] for r in rows if not r["ok"]] == ["a"]
