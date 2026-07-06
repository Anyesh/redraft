#pragma once

// Reference C++ port of m0/stabilize.py: the bounded-mode output-stabilization
// state machine and its acceptance rules. This is the single-step semantic
// oracle. The in-engine batched decoder must emit a byte-identical token
// sequence to stabilize() here for any (old_ids, step_model, rule); that parity
// is the correctness contract, verified against the Python via shared vectors.

#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace redraft {

using token_id = int32_t;

struct logprob_entry {
    token_id id;
    double logprob;
};

using topk = std::vector<logprob_entry>;
using accept_rule = std::function<bool(token_id, const topk &)>;

// step_model receives the emitted prefix and, when a draft is live, the tail of
// the old output starting at the current draft position (nullptr otherwise, so
// the callee can decide whether and how far to batch). It returns the model's
// top-k (id, logprob) at the next position, argmax first.
using step_model =
    std::function<topk(const std::vector<token_id> &, const std::vector<token_id> *)>;

inline std::optional<double> logprob_of(token_id draft, const topk &top) {
    for (const auto &e : top) {
        if (e.id == draft) {
            return e.logprob;
        }
    }
    return std::nullopt;
}

// entropy of the normalized top-k distribution over natural log, matching
// stabilize.py: probs renormalized over the k retained entries, not the full
// vocabulary, so h_norm in [0, 1] measures how flat the retained head is.
inline double normalized_entropy(const topk &top) {
    double total = 0.0;
    for (const auto &e : top) {
        total += std::exp(e.logprob);
    }
    double entropy = 0.0;
    for (const auto &e : top) {
        double p = std::exp(e.logprob) / total;
        if (p > 0.0) {
            entropy -= p * std::log(p);
        }
    }
    return entropy / std::log(static_cast<double>(top.size()));
}

inline accept_rule tau_rule(double tau) {
    return [tau](token_id draft, const topk &top) {
        auto d_lp = logprob_of(draft, top);
        if (!d_lp) {
            return false;
        }
        return top.front().logprob - *d_lp <= tau;
    };
}

inline accept_rule entropy_scaled_tau(double tau) {
    return [tau](token_id draft, const topk &top) {
        auto d_lp = logprob_of(draft, top);
        if (!d_lp) {
            return false;
        }
        double eff = top.size() <= 1 ? 0.0 : tau * normalized_entropy(top);
        return top.front().logprob - *d_lp <= eff;
    };
}

// The m5-selected rule: entropy-scaled band that never closes below `floor`, so
// a single high-confidence miss does not force a reanchor. floor=0 reproduces
// entropy_scaled_tau.
inline accept_rule entropy_floor_tau(double tau, double floor) {
    return [tau, floor](token_id draft, const topk &top) {
        auto d_lp = logprob_of(draft, top);
        if (!d_lp) {
            return false;
        }
        double eff = top.size() <= 1
                         ? floor
                         : floor + (tau - floor) * normalized_entropy(top);
        return top.front().logprob - *d_lp <= eff;
    };
}

inline accept_rule confidence_gated_tau(double tau, double p_cut) {
    return [tau, p_cut](token_id draft, const topk &top) {
        auto d_lp = logprob_of(draft, top);
        if (!d_lp) {
            return false;
        }
        if (std::exp(top.front().logprob) >= p_cut) {
            return draft == top.front().id;
        }
        return top.front().logprob - *d_lp <= tau;
    };
}

inline accept_rule gated_entropy_tau(double tau, double p_cut) {
    return [tau, p_cut](token_id draft, const topk &top) {
        auto d_lp = logprob_of(draft, top);
        if (!d_lp) {
            return false;
        }
        if (std::exp(top.front().logprob) >= p_cut) {
            return draft == top.front().id;
        }
        double eff = top.size() <= 1 ? 0.0 : tau * normalized_entropy(top);
        return top.front().logprob - *d_lp <= eff;
    };
}

struct stab_result {
    std::vector<token_id> emitted;
    std::vector<std::string> events;
    int divergences = 0;

    double held_fraction() const {
        if (events.empty()) {
            return 0.0;
        }
        long held = 0;
        for (const auto &e : events) {
            if (e == "held") {
                ++held;
            }
        }
        return static_cast<double>(held) / static_cast<double>(events.size());
    }
};

// Locate the last anchor_len emitted tokens inside old_ids at or past watermark,
// returning the old index right after the match (the next draft position), or
// nullopt. The watermark guarantees drafting never re-consumes an old span,
// which would loop.
inline std::optional<size_t> find_anchor(const std::vector<token_id> &old_ids,
                                         const std::vector<token_id> &emitted,
                                         size_t anchor_len, size_t watermark) {
    if (emitted.size() < anchor_len || old_ids.size() < anchor_len) {
        return std::nullopt;
    }
    for (size_t start = watermark; start + anchor_len <= old_ids.size(); ++start) {
        bool match = true;
        for (size_t i = 0; i < anchor_len; ++i) {
            if (old_ids[start + i] != emitted[emitted.size() - anchor_len + i]) {
                match = false;
                break;
            }
        }
        if (match) {
            return start + anchor_len;
        }
    }
    return std::nullopt;
}

inline stab_result stabilize(const std::vector<token_id> &old_ids,
                             const step_model &model, const accept_rule &rule,
                             size_t anchor_len = 3, size_t max_tokens = 512,
                             const std::unordered_set<token_id> &eos_ids = {}) {
    stab_result result;
    std::optional<size_t> draft = old_ids.empty() ? std::nullopt
                                                  : std::optional<size_t>(0);
    size_t watermark = 0;

    while (result.emitted.size() < max_tokens) {
        std::vector<token_id> draft_rest;
        const std::vector<token_id> *draft_rest_ptr = nullptr;
        if (draft) {
            draft_rest.assign(old_ids.begin() + *draft, old_ids.end());
            draft_rest_ptr = &draft_rest;
        }
        topk top = model(result.emitted, draft_rest_ptr);
        token_id argmax_id = top.front().id;

        std::optional<std::pair<token_id, std::string>> chosen;
        if (draft) {
            token_id d = old_ids[*draft];
            if (rule(d, top)) {
                chosen = {d, "held"};
                ++*draft;
                watermark = std::max(watermark, *draft);
                if (*draft >= old_ids.size()) {
                    draft = std::nullopt;
                }
            } else {
                ++result.divergences;
                draft = std::nullopt;
            }
        }

        if (!chosen) {
            if (eos_ids.count(argmax_id)) {
                break;
            }
            chosen = {argmax_id, "serial"};
        }

        auto [token, event] = *chosen;
        if (eos_ids.count(token)) {
            break;
        }
        result.emitted.push_back(token);
        result.events.push_back(event);

        if (event == "serial") {
            draft = find_anchor(old_ids, result.emitted, anchor_len, watermark);
            if (draft) {
                watermark = *draft;
                if (*draft >= old_ids.size()) {
                    draft = std::nullopt;
                }
            }
        }
    }

    return result;
}

}  // namespace redraft
