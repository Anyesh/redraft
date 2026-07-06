// Parity tests for the C++ reference core, mirroring tests/test_stabilize.py
// case for case. If these and the Python suite ever diverge on the same vectors,
// the port has drifted from the contract. Build: `make test` in this directory.

#include <cstdio>
#include <string>
#include <vector>

#include "stabilize.hpp"

using redraft::entropy_floor_tau;
using redraft::stab_result;
using redraft::stabilize;
using redraft::step_model;
using redraft::tau_rule;
using redraft::token_id;
using redraft::topk;

static int g_failures = 0;

static step_model scripted(std::vector<topk> steps) {
    return [steps](const std::vector<token_id> &emitted,
                   const std::vector<token_id> *) { return steps[emitted.size()]; };
}

static void check(const std::string &name, bool ok) {
    if (ok) {
        std::printf("  ok   %s\n", name.c_str());
    } else {
        std::printf("  FAIL %s\n", name.c_str());
        ++g_failures;
    }
}

static bool emitted_is(const stab_result &r, std::vector<token_id> want) {
    return r.emitted == want;
}

static bool events_is(const stab_result &r, std::vector<std::string> want) {
    return r.events == want;
}

static const token_id EOS = 999;
static const std::unordered_set<token_id> eos = {EOS};

static void test_tau_zero_holds_everything() {
    std::vector<token_id> old = {1, 2, 3};
    auto r = stabilize(old,
                       scripted({{{1, -0.1}, {7, -2.0}},
                                 {{2, -0.2}, {7, -2.0}},
                                 {{3, -0.3}, {7, -2.0}},
                                 {{EOS, -0.1}}}),
                       tau_rule(0.0), 3, 512, eos);
    check("tau0 holds everything: emitted", emitted_is(r, {1, 2, 3}));
    check("tau0 holds everything: events",
          events_is(r, {"held", "held", "held"}));
    check("tau0 holds everything: divergences", r.divergences == 0);
}

static void test_tolerance_accepts_near_argmax() {
    std::vector<token_id> old = {1, 5};
    std::vector<topk> steps = {{{1, -0.1}, {7, -2.0}},
                               {{8, -0.5}, {5, -1.0}},
                               {{EOS, -0.1}}};
    auto loose = stabilize(old, scripted(steps), tau_rule(1.0), 3, 512, eos);
    check("tolerance loose: emitted", emitted_is(loose, {1, 5}));
    check("tolerance loose: events", events_is(loose, {"held", "held"}));
    auto strict = stabilize(old, scripted(steps), tau_rule(0.0), 3, 512, eos);
    check("tolerance strict: emitted", emitted_is(strict, {1, 8}));
    check("tolerance strict: events", events_is(strict, {"held", "serial"}));
    check("tolerance strict: divergences", strict.divergences == 1);
}

static void test_draft_below_topk_rejected() {
    std::vector<token_id> old = {4};
    auto r = stabilize(old, scripted({{{1, -0.1}, {2, -0.5}}, {{EOS, -0.1}}}),
                       tau_rule(10.0), 3, 512, eos);
    check("below-topk rejected: emitted", emitted_is(r, {1}));
    check("below-topk rejected: events", events_is(r, {"serial"}));
}

static void test_reject_then_reanchor_resumes() {
    std::vector<token_id> old = {1, 2, 3, 4, 5};
    auto r = stabilize(old,
                       scripted({{{1, -0.1}, {0, -9.0}},
                                 {{7, -0.1}, {2, -9.0}},
                                 {{3, -0.1}, {0, -9.0}},
                                 {{4, -0.1}, {0, -9.0}},
                                 {{5, -0.1}, {0, -9.0}},
                                 {{EOS, -0.1}}}),
                       tau_rule(0.0), 2, 512, eos);
    check("reanchor resumes: emitted", emitted_is(r, {1, 7, 3, 4, 5}));
    check("reanchor resumes: events",
          events_is(r, {"held", "serial", "serial", "serial", "held"}));
    check("reanchor resumes: divergences", r.divergences == 1);
}

static void test_reanchor_never_backwards() {
    std::vector<token_id> old = {5, 6, 7};
    auto r = stabilize(old,
                       scripted({{{5, -0.1}},
                                 {{6, -0.1}},
                                 {{7, -0.1}},
                                 {{5, -0.1}},
                                 {{6, -0.1}},
                                 {{EOS, -0.1}}}),
                       tau_rule(0.0), 2, 512, eos);
    check("reanchor never backwards: events",
          events_is(r, {"held", "held", "held", "serial", "serial"}));
}

static void test_reanchor_at_end_no_crash() {
    std::vector<token_id> old = {1, 2, 3};
    auto r = stabilize(old,
                       scripted({{{9, -0.1}, {1, -9.0}},
                                 {{2, -0.1}},
                                 {{3, -0.1}},
                                 {{7, -0.1}},
                                 {{EOS, -0.1}}}),
                       tau_rule(0.0), 2, 512, eos);
    check("reanchor at end: emitted", emitted_is(r, {9, 2, 3, 7}));
    check("reanchor at end: events",
          events_is(r, {"serial", "serial", "serial", "serial"}));
}

static void test_eos_argmax_stops() {
    std::vector<token_id> old = {};
    auto r = stabilize(old, scripted({{{EOS, -0.1}, {1, -2.0}}}), tau_rule(0.0), 3,
                       512, eos);
    check("eos argmax stops: emitted empty", r.emitted.empty());
    check("eos argmax stops: events empty", r.events.empty());
}

static void test_max_tokens_caps() {
    std::vector<token_id> old = {1, 1, 1, 1};
    std::vector<topk> steps(10, {{1, -0.1}});
    auto r = stabilize(old, scripted(steps), tau_rule(0.0), 3, 3, eos);
    check("max_tokens caps: emitted", emitted_is(r, {1, 1, 1}));
}

// Ground-truth decisions computed from Python m0.stabilize.EntropyFloorTau(3,1)
// on the same five (draft, top-k) inputs; locks numeric parity of the selected
// rule across the two implementations.
static void test_entropy_floor_matches_python() {
    auto rule = entropy_floor_tau(3.0, 1.0);
    struct twant {
        token_id draft;
        topk top;
        bool want;
    };
    std::vector<twant> cases = {
        {2, {{1, -0.5}, {2, -1.5}, {3, -3.0}}, true},
        {2, {{1, -0.01}, {2, -5.0}, {3, -6.0}}, false},
        {3, {{1, -0.7}, {2, -0.9}, {3, -1.2}}, true},
        {9, {{1, -0.1}, {2, -2.0}}, false},
        {1, {{1, -0.2}, {2, -1.0}, {3, -2.0}}, true},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        bool got = rule(cases[i].draft, cases[i].top);
        check("entropy_floor parity case " + std::to_string(i),
              got == cases[i].want);
    }
}

int main() {
    std::printf("reference-core parity tests\n");
    test_entropy_floor_matches_python();
    test_tau_zero_holds_everything();
    test_tolerance_accepts_near_argmax();
    test_draft_below_topk_rejected();
    test_reject_then_reanchor_resumes();
    test_reanchor_never_backwards();
    test_reanchor_at_end_no_crash();
    test_eos_argmax_stops();
    test_max_tokens_caps();
    if (g_failures) {
        std::printf("%d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
