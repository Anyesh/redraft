import hashlib
import json
from collections import Counter

from replay.bundle import load_units
from replay.synth import KINDS, build_bundle


def test_synthetic_bundle_meets_the_contract_volume(tmp_path):
    build_bundle(tmp_path, per_kind=13, seed=7)
    units = load_units(tmp_path)
    corrections = [u for u in units if u.kind == "revise"]
    rederives = [u for u in units if u.kind == "rederive"]
    assert len(corrections) >= 50
    assert len(rederives) >= 50
    for kind in KINDS:
        assert Counter(u.document_kind for u in corrections)[kind] >= 10


def test_every_rederive_changes_a_source_and_every_revision_is_hashed(tmp_path):
    build_bundle(tmp_path, per_kind=2, seed=1)
    for u in load_units(tmp_path):
        if u.kind == "rederive":
            assert u.sources_before != u.sources_after
            assert u.derived_before
    rows = [
        json.loads(x) for x in (tmp_path / "revisions.jsonl").read_text().splitlines()
    ]
    assert all(
        r["sha256"] == hashlib.sha256(r["text"].encode()).hexdigest() for r in rows
    )


def test_corrections_edit_the_draft_and_pin_a_line(tmp_path):
    build_bundle(tmp_path, per_kind=2, seed=1)
    for u in load_units(tmp_path):
        if u.kind == "revise":
            assert u.derived_after != u.derived_before
            assert u.pinned and all(p in u.derived_before for p in u.pinned)


def test_the_same_seed_gives_the_same_bundle(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    build_bundle(a, per_kind=3, seed=5)
    build_bundle(b, per_kind=3, seed=5)
    for name in ("revisions", "refreshes", "corrections"):
        assert (a / f"{name}.jsonl").read_text() == (b / f"{name}.jsonl").read_text()
