#pragma once

// Batched in-engine stabilization: the same emitted-token semantics as the
// single-step reference (stabilize.hpp), but verifying a whole draft window in
// one shot the way llama.cpp's speculative decode does. The window's per-position
// top-k comes from an injected verify_fn: a scripted one for the GPU-free parity
// fuzz against the reference core, and a llama_decode-backed one for real runs.
// Parity with the reference core on the same inputs is the correctness contract.

#include <algorithm>
#include <functional>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stabilize.hpp"

namespace redraft {

// verify(prefix, window) returns one top-k per window position: entry i is the
// model distribution predicting window[i], conditioned on prefix + window[0..i-1]
// (so entry 0 is the distribution right after `prefix`). For an empty window
// (a pure serial step) it returns exactly one entry, the distribution after
// `prefix`. This mirrors what a single batched llama_decode over the window
// yields via llama_get_logits_ith at each position.
using verify_fn = std::function<std::vector<topk>(const std::vector<token_id> &,
                                                  const std::vector<token_id> &)>;

// Resumable form of the batched stabilizer. One step() runs exactly one outer
// iteration (one verify() call over one draft window) and returns the tokens
// appended during that step; the server drives it one window per scheduler pass
// to stream and to yield the decode step back to other slots. batched_stabilize
// below is a thin drain-loop over this, so the emitted/events/divergences it
// produces are byte-identical to draining the stepper and remain guarded by the
// GPU-free parity fuzz.
struct stab_stepper {
    std::vector<token_id> old_ids;
    accept_rule rule;
    size_t horizon;
    size_t anchor_len;
    size_t max_tokens;
    std::unordered_set<token_id> eos_ids;

    stab_result result;
    std::optional<size_t> draft;
    size_t watermark = 0;
    bool done = false;

    stab_stepper(std::vector<token_id> old_ids_, accept_rule rule_,
                 size_t horizon_ = 64, size_t anchor_len_ = 3,
                 size_t max_tokens_ = 512,
                 std::unordered_set<token_id> eos_ids_ = {})
        : old_ids(std::move(old_ids_)), rule(std::move(rule_)), horizon(horizon_),
          anchor_len(anchor_len_), max_tokens(max_tokens_),
          eos_ids(std::move(eos_ids_)),
          draft(old_ids.empty() ? std::nullopt : std::optional<size_t>(0)) {}

    void reanchor_after_serial() {
        draft = find_anchor(old_ids, result.emitted, anchor_len, watermark);
        if (draft) {
            watermark = *draft;
            if (*draft >= old_ids.size()) {
                draft = std::nullopt;
            }
        }
    }

    std::vector<token_id> step(const verify_fn &verify) {
        size_t before = result.emitted.size();
        if (done) {
            return {};
        }
        if (result.emitted.size() >= max_tokens) {
            done = true;
            return {};
        }

        std::vector<token_id> window;
        if (draft) {
            size_t end = std::min(*draft + horizon, old_ids.size());
            window.assign(old_ids.begin() + *draft, old_ids.begin() + end);
        }
        std::vector<topk> tops = verify(result.emitted, window);

        if (!draft) {
            token_id argmax = tops.front().front().id;
            if (eos_ids.count(argmax)) {
                done = true;
                return {};
            }
            result.emitted.push_back(argmax);
            result.events.push_back("serial");
            reanchor_after_serial();
            return tail(before);
        }

        // Walk the verified window, holding drafts until the first reject, window
        // exhaustion, or the draft running off the end of old_ids. Each held token
        // advances the draft pointer; the top-k at position i was computed with
        // exactly the tokens held so far in context, so this matches the reference
        // core's step-by-step decisions up to the first divergence.
        for (size_t i = 0; i < window.size(); ++i) {
            if (result.emitted.size() >= max_tokens) {
                done = true;
                break;
            }
            const topk &top = tops[i];
            token_id d = window[i];
            token_id argmax = top.front().id;
            if (rule(d, top)) {
                if (eos_ids.count(d)) {
                    done = true;
                    break;
                }
                result.emitted.push_back(d);
                result.events.push_back("held");
                ++*draft;
                watermark = std::max(watermark, *draft);
                if (*draft >= old_ids.size()) {
                    draft = std::nullopt;
                    break;
                }
            } else {
                ++result.divergences;
                if (eos_ids.count(argmax)) {
                    done = true;
                    break;
                }
                result.emitted.push_back(argmax);
                result.events.push_back("serial");
                draft = std::nullopt;
                reanchor_after_serial();
                break;
            }
        }
        return tail(before);
    }

private:
    std::vector<token_id> tail(size_t before) const {
        return std::vector<token_id>(result.emitted.begin() + before,
                                     result.emitted.end());
    }
};

inline stab_result batched_stabilize(const std::vector<token_id> &old_ids,
                                     const verify_fn &verify,
                                     const accept_rule &rule, size_t horizon = 64,
                                     size_t anchor_len = 3, size_t max_tokens = 512,
                                     const std::unordered_set<token_id> &eos_ids = {}) {
    stab_stepper stepper(old_ids, rule, horizon, anchor_len, max_tokens, eos_ids);
    while (!stepper.done) {
        stepper.step(verify);
    }
    return stepper.result;
}

}  // namespace redraft
