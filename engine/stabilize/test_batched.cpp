// GPU-free parity fuzz: batched_stabilize must produce the exact same emitted
// sequence, events, and divergence count as the single-step reference core on
// the same scripted model, across random old_ids, rules, and horizons. This is
// the analogue of the Python BatchedStepper-vs-LlamaStepper parity fuzz, and it
// isolates the batching control flow from any real-model numeric drift.

#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "batched_stabilize.hpp"
#include "stabilize.hpp"

using redraft::accept_rule;
using redraft::batched_stabilize;
using redraft::entropy_floor_tau;
using redraft::stab_result;
using redraft::stabilize;
using redraft::tau_rule;
using redraft::token_id;
using redraft::topk;
using redraft::verify_fn;

static const token_id EOS = 999;
static const int ALPHA = 6;

static uint64_t fnv(uint64_t salt, const std::vector<token_id> &prefix) {
    uint64_t h = 1469598103934665603ULL ^ salt;
    for (token_id t : prefix) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(t));
        h *= 1099511628211ULL;
    }
    return h;
}

// Deterministic scripted distribution as a pure function of the emitted prefix,
// keyed by a per-iteration salt so different fuzz rounds see different models.
static topk g(uint64_t salt, const std::vector<token_id> &prefix) {
    std::mt19937_64 rng(fnv(salt, prefix));
    int k = 3 + static_cast<int>(rng() % 4);
    topk t;
    std::set<token_id> used;
    double lp = -(0.01 + (rng() % 100) / 500.0);
    for (int i = 0; i < k; ++i) {
        token_id id = static_cast<token_id>(rng() % ALPHA);
        while (used.count(id)) {
            id = static_cast<token_id>(rng() % ALPHA);
        }
        used.insert(id);
        t.push_back({id, lp});
        lp -= 0.1 + (rng() % 100) / 100.0;
    }
    // ~10% of the time make the argmax EOS to exercise the serial-eos break path.
    if (rng() % 10 == 0) {
        t.front().id = EOS;
    }
    return t;
}

static int g_failures = 0;

static bool run_one(uint64_t salt, const std::vector<token_id> &old_ids,
                    const accept_rule &rule, size_t horizon) {
    redraft::step_model single = [salt](const std::vector<token_id> &emitted,
                                        const std::vector<token_id> *) {
        return g(salt, emitted);
    };
    verify_fn batched = [salt](const std::vector<token_id> &prefix,
                               const std::vector<token_id> &window) {
        std::vector<topk> out;
        if (window.empty()) {
            out.push_back(g(salt, prefix));
            return out;
        }
        std::vector<token_id> ctx = prefix;
        for (size_t i = 0; i < window.size(); ++i) {
            out.push_back(g(salt, ctx));
            ctx.push_back(window[i]);
        }
        return out;
    };

    stab_result ref = stabilize(old_ids, single, rule, 3, 64, {EOS});
    stab_result bat = batched_stabilize(old_ids, batched, rule, horizon, 3, 64, {EOS});
    bool ok = ref.emitted == bat.emitted && ref.events == bat.events &&
              ref.divergences == bat.divergences;

    // The resumable stepper drained step-by-step must reproduce batched_stabilize
    // exactly, and the concatenation of per-step deltas must equal emitted (the
    // invariant the streaming server path relies on).
    redraft::stab_stepper stepper(old_ids, rule, horizon, 3, 64, {EOS});
    std::vector<token_id> drained;
    while (!stepper.done) {
        std::vector<token_id> delta = stepper.step(batched);
        drained.insert(drained.end(), delta.begin(), delta.end());
    }
    ok = ok && stepper.result.emitted == bat.emitted &&
         stepper.result.events == bat.events &&
         stepper.result.divergences == bat.divergences && drained == bat.emitted;
    return ok;
}

int main() {
    std::printf("batched-vs-reference parity fuzz\n");
    std::vector<std::pair<const char *, accept_rule>> rules = {
        {"entropy_floor(3,1)", entropy_floor_tau(3.0, 1.0)},
        {"tau(1.0)", tau_rule(1.0)},
        {"tau(0.0)", tau_rule(0.0)},
    };
    std::vector<size_t> horizons = {1, 4, 64};

    int checked = 0;
    for (uint64_t salt = 0; salt < 3000; ++salt) {
        std::mt19937_64 rng(salt * 2654435761ULL + 12345);
        size_t n = 3 + rng() % 10;
        std::vector<token_id> old_ids;
        for (size_t i = 0; i < n; ++i) {
            old_ids.push_back(static_cast<token_id>(rng() % ALPHA));
        }
        const auto &rule = rules[salt % rules.size()];
        size_t horizon = horizons[(salt / rules.size()) % horizons.size()];
        if (!run_one(salt, old_ids, rule.second, horizon)) {
            if (g_failures < 5) {
                std::printf("  FAIL salt=%llu rule=%s horizon=%zu n=%zu\n",
                            (unsigned long long)salt, rule.first, horizon, n);
            }
            ++g_failures;
        }
        ++checked;
    }

    std::printf("checked %d scenarios\n", checked);
    if (g_failures) {
        std::printf("%d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("all parity holds\n");
    return 0;
}
