import pytest

from bench.stabilize import stab_passes, stabilize

EOS = 999


def scripted(steps):
    def model(prefix_ids, draft_rest):
        return steps[len(prefix_ids)]

    return model


def top(*pairs):
    return list(pairs)


def test_tau_zero_matching_argmax_holds_everything():
    old = [1, 2, 3]
    steps = [
        top((1, -0.1), (7, -2.0)),
        top((2, -0.2), (7, -2.0)),
        top((3, -0.3), (7, -2.0)),
        top(
            (EOS, -0.1),
        ),
    ]
    r = stabilize(old, scripted(steps), tau=0.0, eos_ids={EOS})
    assert r.emitted == [1, 2, 3]
    assert r.events == ["held", "held", "held"]
    assert r.divergences == 0


def test_tolerance_accepts_near_argmax_draft():
    old = [1, 5]
    steps = [
        top((1, -0.1), (7, -2.0)),
        # argmax is 8, draft 5 is 0.5 nats behind
        top((8, -0.5), (5, -1.0)),
        top(
            (EOS, -0.1),
        ),
    ]
    loose = stabilize(old, scripted(steps), tau=1.0, eos_ids={EOS})
    assert loose.emitted == [1, 5]
    assert loose.events == ["held", "held"]
    strict = stabilize(old, scripted(steps), tau=0.0, eos_ids={EOS})
    assert strict.emitted == [1, 8]
    assert strict.events == ["held", "serial"]
    assert strict.divergences == 1


def test_draft_below_topk_is_rejected():
    old = [4]
    steps = [
        top((1, -0.1), (2, -0.5)),
        top(
            (EOS, -0.1),
        ),
    ]
    r = stabilize(old, scripted(steps), tau=10.0, eos_ids={EOS})
    assert r.emitted == [1]
    assert r.events == ["serial"]


def test_reject_then_reanchor_resumes_draft():
    old = [1, 2, 3, 4, 5]
    steps = [
        top((1, -0.1), (0, -9.0)),
        top((7, -0.1), (2, -9.0)),  # reject: emit 7, anchor [1,7] not in old
        top((3, -0.1), (0, -9.0)),  # serial 3, anchor [7,3] not in old
        top((4, -0.1), (0, -9.0)),  # serial 4, anchor [3,4] at old[2:4] -> draft=4
        top((5, -0.1), (0, -9.0)),  # draft 5 == argmax -> held
        top(
            (EOS, -0.1),
        ),
    ]
    r = stabilize(old, scripted(steps), tau=0.0, anchor_len=2, eos_ids={EOS})
    assert r.emitted == [1, 7, 3, 4, 5]
    assert r.events == ["held", "serial", "serial", "serial", "held"]
    assert r.divergences == 1


def test_reanchor_never_moves_backwards():
    # the anchor [5,6] only occurs before the watermark, so drafting must not resume
    old = [5, 6, 7]
    steps = [
        top(
            (5, -0.1),
        ),
        top(
            (6, -0.1),
        ),
        top(
            (7, -0.1),
        ),
        top(
            (5, -0.1),
        ),
        top(
            (6, -0.1),
        ),
        top(
            (EOS, -0.1),
        ),
    ]
    r = stabilize(old, scripted(steps), tau=0.0, anchor_len=2, eos_ids={EOS})
    assert r.events == ["held", "held", "held", "serial", "serial"]


def test_reanchor_at_end_of_old_output_does_not_crash():
    # anchor [2,3] matches old[1:3], leaving nothing to draft; must not index past old
    old = [1, 2, 3]
    steps = [
        top((9, -0.1), (1, -9.0)),
        top(
            (2, -0.1),
        ),
        top(
            (3, -0.1),
        ),
        top(
            (7, -0.1),
        ),
        top(
            (EOS, -0.1),
        ),
    ]
    r = stabilize(old, scripted(steps), tau=0.0, anchor_len=2, eos_ids={EOS})
    assert r.emitted == [9, 2, 3, 7]
    assert r.events == ["serial", "serial", "serial", "serial"]


def test_eos_argmax_stops_without_emitting():
    old = []
    steps = [top((EOS, -0.1), (1, -2.0))]
    r = stabilize(old, scripted(steps), tau=0.0, eos_ids={EOS})
    assert r.emitted == []
    assert r.events == []


def test_max_tokens_caps_generation():
    old = [1, 1, 1, 1]
    steps = [
        top(
            (1, -0.1),
        )
    ] * 10
    r = stabilize(old, scripted(steps), tau=0.0, max_tokens=3, eos_ids={EOS})
    assert r.emitted == [1, 1, 1]


def test_stab_passes_accounting():
    events = ["held"] * 20 + ["serial"] * 2 + ["held"] * 3
    r = stab_passes(events, window=16, reanchor_cost=1)
    # ceil(20/16)=2 verify + 2 serial + ceil(3/16)=1 verify + 1 reanchor
    assert r["forward_passes"] == 6
    assert r["baseline_passes"] == 25
    assert r["speedup"] == pytest.approx(25 / 6, abs=1e-4)


def test_stab_passes_empty():
    r = stab_passes([], window=16, reanchor_cost=1)
    assert r["forward_passes"] == 0
    assert r["speedup"] == 0.0
