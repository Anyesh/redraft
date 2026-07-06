#pragma once

// Server-side driver for the in-engine stabilizer. Wraps the same verifier the
// standalone example uses (a stateful llama_decode that holds accepted drafts in
// KV and rolls back rejected speculation via seq_rm), but parameterized on a
// sequence id and a KV base position so it runs inside a llama-server slot rather
// than a fresh seq-0 context. The stabilize control flow and rules are the frozen,
// fuzz-locked core (batched_stabilize.hpp / stabilize.hpp), reused unchanged.

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_set>
#include <vector>

#include "batched_stabilize.hpp"
#include "stabilize.hpp"

namespace redraft {

// Top-k logprobs at a set of full-vocab logits, matching the harness's n_probs=20
// view: entries sorted by logit desc, logprob = logit - logsumexp. Renormalization
// for the entropy rule happens inside the rule over these k.
inline topk top_from_logits(const float *logits, int n_vocab, int k) {
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        idx[i] = i;
    }
    int kk = std::min(k, n_vocab);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    double mx = logits[idx[0]];
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum += std::exp(static_cast<double>(logits[i]) - mx);
    }
    double lse = mx + std::log(sum);
    topk t;
    t.reserve(kk);
    for (int i = 0; i < kk; ++i) {
        t.push_back({static_cast<token_id>(idx[i]),
                     static_cast<double>(logits[idx[i]]) - lse});
    }
    return t;
}

// Stateful verify_fn over a live server context. Assumes the KV for `seq_id` holds
// exactly `template + emitted` at the top of each call, reconciled from the previous
// window's accept decisions. `template_len` is the KV base (the prompt length already
// prefilled by the server).
struct server_verifier {
    llama_context *ctx;
    llama_batch batch;
    int n_vocab;
    int k;
    int seq_id;
    size_t template_len;

    // Recurrent (Gated DeltaNet / Mamba) memory cannot roll back a window-sized span:
    // llama_memory_recurrent::seq_rm only rewinds within an n_rs_seq snapshot budget
    // (0 without a draft model) and otherwise no-ops. On such models a divergence is
    // reconciled by a full reset and forward reprocess of template + committed, so the
    // verifier needs the template ids and a reprocess chunk size. Dense models keep the
    // cheap seq_rm path.
    bool recurrent = false;
    std::vector<token_id> template_ids;
    int n_batch = 512;

    // Parked post-template state: recurrent rollback restores this once-saved snapshot
    // and reprocesses only the emitted suffix, instead of forward-reprocessing the
    // static template on every divergence.
    std::vector<uint8_t> template_state;
    bool parked = false;

    size_t n_committed = 0;
    topk committed_top;
    std::vector<token_id> spec_window;
    std::vector<topk> spec_tops;

    void decode_one(token_id tok, size_t pos) {
        common_batch_clear(batch);
        common_batch_add(batch, tok, static_cast<llama_pos>(pos), {seq_id}, true);
        llama_decode(ctx, batch);
    }

    // Forward-decode a contiguous token span at [start, start+toks.size()), flagging
    // logits only on the final token. Chunked by n_batch. Returns the batch index of
    // that final token's logits, or -1 for an empty span.
    int decode_span(const std::vector<token_id> &toks, size_t start) {
        int last_idx = -1;
        size_t i = 0;
        while (i < toks.size()) {
            size_t chunk = std::min(static_cast<size_t>(n_batch), toks.size() - i);
            common_batch_clear(batch);
            for (size_t j = 0; j < chunk; ++j) {
                bool is_last = (i + j + 1 == toks.size());
                common_batch_add(batch, toks[i + j],
                                 static_cast<llama_pos>(start + i + j), {seq_id}, is_last);
            }
            llama_decode(ctx, batch);
            if (i + chunk == toks.size()) {
                last_idx = static_cast<int>(chunk - 1);
            }
            i += chunk;
        }
        return last_idx;
    }

    // Snapshot the state as it stands (called once when the sequence holds exactly the
    // template, right after prefill) so rollback can restore it without recomputing the
    // template. Best-effort: on failure the rebuild path falls back to full reprocess.
    void park() {
        size_t sz = llama_state_seq_get_size(ctx, seq_id);
        template_state.resize(sz);
        size_t got = llama_state_seq_get_data(ctx, template_state.data(), sz, seq_id);
        parked = got > 0;
        if (!parked) {
            template_state.clear();
        }
    }

    // Recurrent rollback: restore committed state, then rebuild. With a parked template
    // snapshot this reprocesses only the emitted suffix; otherwise it falls back to a
    // full reprocess of template + emitted (the m3 tax, minus the HTTP round trips).
    void rebuild(const std::vector<token_id> &emitted) {
        int last = -1;
        llama_memory_seq_rm(llama_get_memory(ctx), seq_id, 0, -1);
        if (parked) {
            llama_state_seq_set_data(ctx, template_state.data(),
                                     template_state.size(), seq_id);
        } else {
            last = decode_span(template_ids, 0);
        }
        if (!emitted.empty()) {
            last = decode_span(emitted, template_len);
        }
        n_committed = emitted.size();
        if (last >= 0) {
            committed_top = top_from_logits(llama_get_logits_ith(ctx, last), n_vocab, k);
        }
    }

    void reconcile(const std::vector<token_id> &emitted) {
        if (recurrent) {
            reconcile_recurrent(emitted);
        } else {
            reconcile_dense(emitted);
        }
    }

    void reconcile_dense(const std::vector<token_id> &emitted) {
        size_t h = 0;
        while (h < spec_window.size() && n_committed + h < emitted.size() &&
               emitted[n_committed + h] == spec_window[h]) {
            ++h;
        }
        llama_memory_seq_rm(llama_get_memory(ctx), seq_id,
                            static_cast<llama_pos>(template_len + n_committed + h), -1);
        if (h < spec_tops.size()) {
            committed_top = spec_tops[h];
        }
        n_committed += h;
        while (n_committed < emitted.size()) {
            decode_one(emitted[n_committed], template_len + n_committed);
            committed_top = top_from_logits(llama_get_logits_ith(ctx, 0), n_vocab, k);
            ++n_committed;
        }
        spec_window.clear();
        spec_tops.clear();
    }

    void reconcile_recurrent(const std::vector<token_id> &emitted) {
        if (!spec_window.empty()) {
            size_t h = 0;
            while (h < spec_window.size() && n_committed + h < emitted.size() &&
                   emitted[n_committed + h] == spec_window[h]) {
                ++h;
            }
            if (h < spec_window.size()) {
                // divergence inside the window: state is ahead and cannot roll back
                rebuild(emitted);
                spec_window.clear();
                spec_tops.clear();
                return;
            }
            // whole window held: materialized state is still a correct prefix of emitted
            if (h < spec_tops.size()) {
                committed_top = spec_tops[h];
            }
            n_committed += h;
            spec_window.clear();
            spec_tops.clear();
        }
        // forward-extend any serial tokens beyond n_committed (forward decode is safe on
        // recurrent memory; only backward rollback is not)
        while (n_committed < emitted.size()) {
            decode_one(emitted[n_committed], template_len + n_committed);
            committed_top = top_from_logits(llama_get_logits_ith(ctx, 0), n_vocab, k);
            ++n_committed;
        }
    }

    std::vector<topk> operator()(const std::vector<token_id> &emitted,
                                 const std::vector<token_id> &window) {
        reconcile(emitted);
        std::vector<topk> out;
        if (window.empty()) {
            out.push_back(committed_top);
            spec_tops.push_back(committed_top);
            return out;
        }
        common_batch_clear(batch);
        for (size_t i = 0; i < window.size(); ++i) {
            common_batch_add(batch, window[i],
                             static_cast<llama_pos>(template_len + n_committed + i),
                             {seq_id}, true);
        }
        llama_decode(ctx, batch);

        out.push_back(committed_top);
        for (size_t i = 1; i < window.size(); ++i) {
            out.push_back(top_from_logits(llama_get_logits_ith(ctx, i - 1), n_vocab, k));
        }
        spec_window = window;
        spec_tops = out;
        // trailing distribution: predicts the token after the whole window, used
        // when every draft is held.
        spec_tops.push_back(
            top_from_logits(llama_get_logits_ith(ctx, window.size() - 1), n_vocab, k));
        return out;
    }
};

struct redraft_run_params {
    std::vector<token_id> old_output;
    double tau = 3.0;
    double floor = 1.0;
    size_t horizon = 64;
    size_t anchor_len = 3;
    size_t max_tokens = 512;
    std::unordered_set<token_id> eos;
    // recurrent-memory support: on a recurrent model the verifier rebuilds committed
    // state by reprocessing template_ids + emitted on divergence (see server_verifier).
    bool recurrent = false;
    std::vector<token_id> template_ids;
    int n_batch = 512;
};

struct redraft_run_result {
    std::vector<token_id> emitted;
    double held_fraction = 0.0;
    int divergences = 0;
};

// Resumable stabilize session for one slot. init() seeds committed_top from the
// prompt-prefill logits at seed_logits_idx (which must be read before any verifier
// decode invalidates it), then each step() advances the stabilizer by one draft
// window: it issues one verifier llama_decode and returns the tokens emitted in
// that window. The server holds one of these per slot and steps it once per
// scheduler pass so the loop streams and yields the decode step to other slots.
struct redraft_session {
    server_verifier v;
    std::optional<stab_stepper> stepper;
    bool batch_alloced = false;

    redraft_session() = default;
    redraft_session(const redraft_session &) = delete;
    redraft_session &operator=(const redraft_session &) = delete;

    void init(llama_context *ctx, int n_vocab, int seq_id, size_t template_len,
              int seed_logits_idx, const redraft_run_params &p) {
        v.ctx = ctx;
        v.recurrent = p.recurrent;
        v.template_ids = p.template_ids;
        v.n_batch = p.n_batch;
        // the reprocess path decodes up to n_batch tokens per chunk, so the batch must
        // hold that many; the window path needs horizon+1.
        int32_t batch_cap =
            static_cast<int32_t>(std::max<size_t>({p.horizon + 1, (size_t) p.n_batch, 1}));
        v.batch = llama_batch_init(batch_cap, 0, 1);
        batch_alloced = true;
        v.n_vocab = n_vocab;
        v.k = 20;
        v.seq_id = seq_id;
        v.template_len = template_len;
        v.committed_top =
            top_from_logits(llama_get_logits_ith(ctx, seed_logits_idx), n_vocab, 20);
        if (v.recurrent) {
            // the sequence holds exactly the template here (prefill just finished), so
            // snapshot it for cheap rollback restores.
            v.park();
        }
        stepper.emplace(p.old_output, entropy_floor_tau(p.tau, p.floor), p.horizon,
                        p.anchor_len, p.max_tokens, p.eos);
    }

    std::vector<token_id> step() { return stepper->step(std::ref(v)); }

    bool done() const { return stepper->done; }
    const std::vector<token_id> &emitted() const { return stepper->result.emitted; }
    double held_fraction() const { return stepper->result.held_fraction(); }
    int divergences() const { return stepper->result.divergences; }

    ~redraft_session() {
        if (batch_alloced) {
            llama_batch_free(v.batch);
        }
    }
};

// Runs the whole stabilize loop for one slot in one shot (non-streaming path).
// Thin drain over redraft_session, kept for the measurement path and the parity
// harness.
inline redraft_run_result run_redraft_stabilize(llama_context *ctx, int n_vocab,
                                                int seq_id, size_t template_len,
                                                int seed_logits_idx,
                                                const redraft_run_params &p) {
    redraft_session s;
    s.init(ctx, n_vocab, seq_id, template_len, seed_logits_idx, p);
    while (!s.done()) {
        s.step();
    }

    redraft_run_result out;
    out.emitted = s.emitted();
    out.held_fraction = s.held_fraction();
    out.divergences = s.divergences();
    return out;
}

}  // namespace redraft
