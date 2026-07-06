import random

from bench.batch_engine import BatchedStepper
from bench.stabilize import (
    ConfidenceGatedTau,
    EntropyFloorTau,
    EntropyScaledTau,
    GatedEntropyTau,
    TauRule,
    stabilize,
)

EOS = 999


def top(*pairs):
    return list(pairs)


def table_dist(table):
    """A dist_fn over exact prefix content (unlike test_stabilize.py's scripted()
    helper, which indexes by call count: batched calls can request several
    positions from one call, so the oracle must be addressable by content).
    """

    def dist_fn(prefix):
        return table[tuple(prefix)]

    return dist_fn


class FakeServer:
    """Answers /completion the way the patched llama-server does: prompt_probs_tail
    entries describe the last N prompt positions as prediction targets, each scored
    from dist_fn(prefix-before-that-position); completion_probabilities scores the
    position right after the whole prompt. Defensively clamps N to the prompt length
    (BatchedStepper never sends N > len(prompt), but a well-behaved fake shouldn't
    assume its caller is well-behaved).
    """

    def __init__(self, dist_fn, n_probs=8, eos_id=None):
        self.dist_fn = dist_fn
        self.n_probs = n_probs
        self.eos_id = eos_id

    def post(self, base, path, payload):
        assert path == "/completion"
        prompt = payload["prompt"]
        n_probs = payload.get("n_probs", self.n_probs)
        n_predict = payload.get("n_predict", 1)

        if n_predict > 1:
            # greedy multi-token continuation, mirroring temperature=0 decode: each
            # entry is scored from the prefix built out of the *previous* entries'
            # own argmax, and generation halts early if that argmax is eos_id, same
            # as a real server stopping mid-chunk instead of padding to n_predict
            prefix = list(prompt)
            completion_probabilities = []
            for _ in range(n_predict):
                dist = self.dist_fn(tuple(prefix))[:n_probs]
                top_id = dist[0][0]
                completion_probabilities.append(
                    {
                        "id": top_id,
                        "top_logprobs": [{"id": i, "logprob": lp} for i, lp in dist],
                    }
                )
                if self.eos_id is not None and top_id == self.eos_id:
                    break
                prefix.append(top_id)
            return {"completion_probabilities": completion_probabilities}

        n_tail = min(payload.get("prompt_probs_tail", 0), len(prompt))
        m = len(prompt)

        prompt_probabilities = []
        for k in range(n_tail):
            target_pos = m - n_tail + k
            dist = self.dist_fn(tuple(prompt[:target_pos]))[:n_probs]
            prompt_probabilities.append(
                {
                    "id": prompt[target_pos],
                    "top_logprobs": [{"id": i, "logprob": lp} for i, lp in dist],
                }
            )

        final = self.dist_fn(tuple(prompt))[:n_probs]
        return {
            "prompt_probabilities": prompt_probabilities,
            "completion_probabilities": [
                {"top_logprobs": [{"id": i, "logprob": lp} for i, lp in final]}
            ],
        }


def batched_stepper(dist_fn, horizon, n_probs=8, eos_id=None):
    stepper = BatchedStepper.__new__(BatchedStepper)
    stepper.base = "fake://"
    stepper.model = "fake"
    stepper.n_probs = n_probs
    stepper.horizon = horizon
    stepper.post = FakeServer(dist_fn, n_probs=n_probs, eos_id=eos_id).post
    stepper.template_ids = []
    stepper._cache = {}
    stepper._chunk_origin = set()
    stepper._attempts = 0
    stepper.http_calls = 0
    stepper.scoring_calls = 0
    stepper.chunk_calls = 0
    stepper.wasted_tokens = 0
    stepper.server_prompt_ms = 0.0
    stepper.server_predicted_ms = 0.0
    return stepper


def per_token(dist_fn, n_probs=8):
    def model(emitted, draft_rest):
        return dist_fn(tuple(emitted))[:n_probs]

    return model


def test_emitted_empty_first_call_is_batched():
    table = {
        (): top((1, -0.1), (7, -3.0)),
        (1,): top((EOS, -0.1), (2, -3.0)),
    }
    dist_fn = table_dist(table)
    old = [1]
    per = stabilize(old, per_token(dist_fn), tau=0.0, eos_ids={EOS})
    batch = stabilize(old, batched_stepper(dist_fn, horizon=16), tau=0.0, eos_ids={EOS})
    assert per == batch
    assert per.emitted == [1]


def test_window_truncated_at_end_of_old_ids():
    table = {
        (): top((1, -0.1), (7, -3.0)),
        (1,): top((2, -0.1), (7, -3.0)),
        (1, 2): top((EOS, -0.1), (9, -3.0)),
    }
    dist_fn = table_dist(table)
    old = [1, 2]
    per = stabilize(old, per_token(dist_fn), tau=0.0, eos_ids={EOS})
    batch = stabilize(old, batched_stepper(dist_fn, horizon=16), tau=0.0, eos_ids={EOS})
    assert per == batch
    assert per.emitted == [1, 2]
    assert per.events == ["held", "held"]


def test_rejection_at_offset_zero():
    # horizon=16 teacher-forces the whole draft window [1,2,3] in one call, so the
    # oracle needs answers for every hypothetical continuation the batch scores
    # (1,), (1,2), (1,2,3), even though the real trajectory rejects at offset 0
    # and never consumes them.
    table = {
        (): top((9, -0.1), (1, -5.0)),
        (1,): top((2, -0.1), (7, -5.0)),
        (1, 2): top((3, -0.1), (7, -5.0)),
        (1, 2, 3): top((4, -0.1), (7, -5.0)),
        (9,): top(
            (EOS, -0.1),
        ),
    }
    dist_fn = table_dist(table)
    old = [1, 2, 3]
    per = stabilize(old, per_token(dist_fn), tau=0.0, anchor_len=2, eos_ids={EOS})
    batch = stabilize(
        old,
        batched_stepper(dist_fn, horizon=16, eos_id=EOS),
        tau=0.0,
        anchor_len=2,
        eos_ids={EOS},
    )
    assert per == batch
    assert per.emitted == [9]
    assert per.events == ["serial"]
    assert per.divergences == 1


def test_eos_as_cached_argmax():
    # the batch also scores the position after the full window (predicting a
    # hypothetical 4th token), even though EOS at (1, 2) stops generation first.
    table = {
        (): top((1, -0.1), (7, -5.0)),
        (1,): top((2, -0.1), (7, -5.0)),
        (1, 2): top((EOS, -0.1), (3, -5.0)),
        (1, 2, 3): top((4, -0.1), (7, -5.0)),
    }
    dist_fn = table_dist(table)
    old = [1, 2, 3]
    per = stabilize(old, per_token(dist_fn), tau=0.0, eos_ids={EOS})
    batch = stabilize(old, batched_stepper(dist_fn, horizon=16), tau=0.0, eos_ids={EOS})
    assert per == batch
    assert per.emitted == [1, 2]
    assert per.events == ["held", "held"]


def test_horizon_one_degenerate_batching():
    table = {
        (): top((1, -0.1), (7, -5.0)),
        (1,): top((2, -0.1), (7, -5.0)),
        (1, 2): top((3, -0.1), (7, -5.0)),
        (1, 2, 3): top((EOS, -0.1), (4, -5.0)),
    }
    dist_fn = table_dist(table)
    old = [1, 2, 3]
    per = stabilize(old, per_token(dist_fn), tau=0.0, eos_ids={EOS})
    batch = stabilize(old, batched_stepper(dist_fn, horizon=1), tau=0.0, eos_ids={EOS})
    assert per == batch
    assert per.emitted == [1, 2, 3]


def test_fake_server_chunked_stops_early_at_eos():
    def dist_fn(prefix):
        if len(prefix) == 2:
            return top((EOS, -0.01), (9999, -5.0))
        return top((len(prefix), -0.01), (9999, -5.0))

    server = FakeServer(dist_fn, n_probs=8, eos_id=EOS)
    data = server.post(
        "fake://", "/completion", {"prompt": [], "n_predict": 4, "n_probs": 8}
    )
    ids = [e["top_logprobs"][0]["id"] for e in data["completion_probabilities"]]
    assert ids == [0, 1, EOS]


def test_score_chunk_replaces_fallback_with_one_call_per_chunk():
    # old_ids empty: draft is never available, so every step falls back; with
    # k starting at 4 and the run only 3 tokens long, one chunk call covers it
    table = {
        (): top((1, -0.01), (9, -5.0)),
        (1,): top((2, -0.01), (9, -5.0)),
        (1, 2): top((3, -0.01), (9, -5.0)),
        (1, 2, 3): top((EOS, -0.01), (9, -5.0)),
    }
    dist_fn = table_dist(table)
    stepper = batched_stepper(dist_fn, horizon=16, eos_id=EOS)
    result = stabilize([], stepper, tau=0.0, eos_ids={EOS}, max_tokens=10)
    assert result.emitted == [1, 2, 3]
    assert stepper.http_calls == 1
    assert stepper.chunk_calls == 1


def test_score_chunk_matches_per_token_with_eos_mid_chunk():
    def dist_fn(prefix):
        if len(prefix) == 2:
            return top((EOS, -0.01), (9999, -5.0))
        return top((len(prefix), -0.01), (9999, -5.0))

    per = stabilize([], per_token(dist_fn), tau=0.0, eos_ids={EOS}, max_tokens=10)
    batch = stabilize(
        [],
        batched_stepper(dist_fn, horizon=16, eos_id=EOS),
        tau=0.0,
        eos_ids={EOS},
        max_tokens=10,
    )
    assert per == batch
    assert per.emitted == [0, 1]


def test_score_chunk_size_grows_across_uninterrupted_fallback_stretch():
    # every position emits a fresh id and EOS is never reachable, so the whole
    # run is one uninterrupted fallback stretch; each chunk that fully drains
    # doubles the next one (4, 8, 16) rather than resetting on its own hits
    def dist_fn(prefix):
        return top((len(prefix), -0.01), (9999, -5.0))

    stepper = batched_stepper(dist_fn, horizon=16)
    payloads = []
    real_post = stepper.post

    def spy(base, path, payload):
        payloads.append(payload)
        return real_post(base, path, payload)

    stepper.post = spy
    stabilize([], stepper, tau=0.0, eos_ids=set(), max_tokens=20)
    assert [p["n_predict"] for p in payloads] == [4, 8, 16]


def test_score_chunk_size_resets_after_batched_call():
    def dist_fn(prefix):
        return top((len(prefix), -0.01), (9999, -5.0))

    stepper = batched_stepper(dist_fn, horizon=4)
    payloads = []
    real_post = stepper.post

    def spy(base, path, payload):
        payloads.append(payload)
        return real_post(base, path, payload)

    stepper.post = spy

    # drain the first chunk completely: one HTTP call, three free cache hits
    stepper([], None)
    stepper([0], None)
    stepper([0, 1], None)
    stepper([0, 1, 2], None)
    assert [p["n_predict"] for p in payloads] == [4]
    assert stepper._attempts == 1

    # a draft becoming available on a key the drained chunk never touched
    # takes the batched branch, which resets the growth counter
    stepper([0, 1, 2, 3], [7, 8])

    # a fresh miss forces another fallback chunk: sized k=4 again, not 8
    stepper([0, 1, 2, 3, 42], None)
    assert [p["n_predict"] for p in payloads] == [4, 1, 4]


def test_score_chunk_tracks_wasted_tokens_left_unconsumed():
    # k=4 requested from a pure fallback stretch, but max_tokens stops
    # generation after 2 tokens, leaving 2 of the chunk's scored positions
    # (the ones covering hypothetical tokens 3 and 4) never looked up
    def dist_fn(prefix):
        return top((len(prefix), -0.01), (9999, -5.0))

    stepper = batched_stepper(dist_fn, horizon=16)
    result = stabilize([], stepper, tau=0.0, eos_ids=set(), max_tokens=2)
    assert result.emitted == [0, 1]
    assert stepper.chunk_calls == 1
    assert stepper.wasted_tokens == 2


def make_random_dist_fn(seed, old_ids, vocab_size, divergence_p, n_probs=8):
    """Pure function of (seed, prefix content): both step models query the same
    oracle, so any mismatch between per-token and batched results can only come
    from a bug in BatchedStepper's windowing/caching, not from the oracle itself.
    """

    def dist_fn(prefix):
        rng = random.Random(hash((seed, prefix)))
        pos = len(prefix)
        pool = {EOS}
        if pos < len(old_ids):
            pool.add(old_ids[pos])
        while len(pool) < n_probs:
            pool.add(rng.randrange(vocab_size))

        candidates = {tid: rng.uniform(-6.0, -0.5) for tid in pool}
        if pos < len(old_ids) and rng.random() > divergence_p:
            candidates[old_ids[pos]] = -0.01
        if pos >= len(old_ids) + 5:
            candidates[EOS] = -0.01

        return sorted(candidates.items(), key=lambda kv: -kv[1])

    return dist_fn


def make_random_rule(rng):
    tau = rng.choice([0.0, 0.5, 1.5, 3.0])
    kind = rng.choice(
        ["tau", "confidence_gated", "entropy_scaled", "entropy_floor", "gated_entropy"]
    )
    if kind == "tau":
        return TauRule(tau=tau)
    if kind == "confidence_gated":
        p_cut = rng.choice([0.3, 0.6, 0.9])
        return ConfidenceGatedTau(tau=tau, p_cut=p_cut)
    if kind == "entropy_scaled":
        return EntropyScaledTau(tau=tau)
    if kind == "entropy_floor":
        floor = rng.choice([0.0, 0.25, 0.5])
        return EntropyFloorTau(tau=tau, floor=floor)
    p_cut = rng.choice([0.3, 0.6, 0.9])
    return GatedEntropyTau(tau=tau, p_cut=p_cut)


def test_batched_matches_per_token_over_randomized_combos():
    rng = random.Random(20260703)
    combos = 0
    for _ in range(60):
        length = rng.randint(1, 25)
        old_ids = [rng.randrange(30) for _ in range(length)]
        divergence_p = rng.choice([0.0, 0.1, 0.3, 0.6, 1.0])
        seed = rng.randrange(1_000_000)
        rule = make_random_rule(rng)
        horizon = rng.choice([1, 2, 4, 16, 64])
        anchor_len = rng.choice([1, 2, 3])

        dist_fn = make_random_dist_fn(
            seed, old_ids, vocab_size=30, divergence_p=divergence_p
        )

        per = stabilize(
            old_ids,
            per_token(dist_fn, n_probs=8),
            rule=rule,
            anchor_len=anchor_len,
            max_tokens=80,
            eos_ids={EOS},
        )
        batch = stabilize(
            old_ids,
            batched_stepper(dist_fn, horizon=horizon, n_probs=8),
            rule=rule,
            anchor_len=anchor_len,
            max_tokens=80,
            eos_ids={EOS},
        )
        assert per == batch, (
            f"mismatch at length={length} divergence_p={divergence_p} seed={seed} "
            f"rule={rule} horizon={horizon} anchor_len={anchor_len}"
        )
        combos += 1

    assert combos == 60
