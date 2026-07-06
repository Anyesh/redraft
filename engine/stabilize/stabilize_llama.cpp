// llama.cpp-backed driver for the in-engine stabilizer. Reuses the reference
// core's rules and the proven batched control flow (batched_stabilize.hpp); the
// only new code is the stateful verifier that turns a draft window into one
// llama_decode, keeps accepted drafts in KV, and rolls back rejected speculation
// via seq_rm. Inputs are pre-tokenized (a text file: `prompt:`, `old:`, `tau:`,
// `floor:`, `horizon:`, `max_tokens:`, `eos:`) so tokenization parity with the
// Python harness is exact and isolated from the decode logic being validated.
// Output is one line of JSON: {"emitted":[...],"held_fraction":x,"divergences":n}.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "batched_stabilize.hpp"
#include "stabilize.hpp"

using redraft::accept_rule;
using redraft::batched_stabilize;
using redraft::entropy_floor_tau;
using redraft::token_id;
using redraft::topk;

struct inputs {
    std::vector<token_id> prompt;
    std::vector<token_id> old_ids;
    double tau = 3.0;
    double floor = 1.0;
    size_t horizon = 64;
    size_t max_tokens = 512;
    std::unordered_set<token_id> eos;
};

static std::vector<token_id> parse_ids(const std::string &rest) {
    std::vector<token_id> out;
    std::istringstream ss(rest);
    long v;
    while (ss >> v) {
        out.push_back(static_cast<token_id>(v));
    }
    return out;
}

static inputs read_inputs(const std::string &path) {
    inputs in;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, colon);
        std::string rest = line.substr(colon + 1);
        if (key == "prompt") {
            in.prompt = parse_ids(rest);
        } else if (key == "old") {
            in.old_ids = parse_ids(rest);
        } else if (key == "eos") {
            for (token_id t : parse_ids(rest)) {
                in.eos.insert(t);
            }
        } else if (key == "tau") {
            in.tau = std::stod(rest);
        } else if (key == "floor") {
            in.floor = std::stod(rest);
        } else if (key == "horizon") {
            in.horizon = static_cast<size_t>(std::stol(rest));
        } else if (key == "max_tokens") {
            in.max_tokens = static_cast<size_t>(std::stol(rest));
        }
    }
    return in;
}

// Top-k logprobs at a set of full-vocab logits, matching the Python harness's
// n_probs=20 view: entries sorted by logit desc, logprob = logit - logsumexp,
// renormalization for the entropy rule happens inside the rule over these k.
static topk top_from_logits(const float *logits, int n_vocab, int k) {
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

// Stateful verify_fn: assumes the KV holds exactly `template + emitted` at the
// top of each call, reconciled from the previous window's accept decisions.
struct llama_verifier {
    llama_context *ctx;
    llama_batch batch;
    int n_vocab;
    int k;
    size_t template_len;

    size_t n_committed = 0;
    topk committed_top;                 // predicts emitted[n_committed]
    std::vector<token_id> spec_window;  // last window decoded into KV
    std::vector<topk> spec_tops;        // size spec_window.size()+1, [i] predicts pos n_committed+i

    void decode_one(token_id tok, size_t pos) {
        common_batch_clear(batch);
        common_batch_add(batch, tok, static_cast<llama_pos>(pos), {0}, true);
        llama_decode(ctx, batch);
    }

    void reconcile(const std::vector<token_id> &emitted) {
        size_t h = 0;
        while (h < spec_window.size() && n_committed + h < emitted.size() &&
               emitted[n_committed + h] == spec_window[h]) {
            ++h;
        }
        // keep the h held drafts already in KV; drop the rest of the speculation
        llama_memory_seq_rm(llama_get_memory(ctx), 0,
                            static_cast<llama_pos>(template_len + n_committed + h), -1);
        if (h < spec_tops.size()) {
            committed_top = spec_tops[h];
        }
        n_committed += h;
        // any remaining emitted tokens are serial corrections not yet in KV;
        // decode each to advance the committed distribution past it.
        while (n_committed < emitted.size()) {
            decode_one(emitted[n_committed], template_len + n_committed);
            committed_top = top_from_logits(llama_get_logits_ith(ctx, 0), n_vocab, k);
            ++n_committed;
        }
        spec_window.clear();
        spec_tops.clear();
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
                             {0}, true);
        }
        llama_decode(ctx, batch);

        out.push_back(committed_top);  // predicts window[0]
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

int main(int argc, char **argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char *input_path = std::getenv("REDRAFT_INPUT");
    if (!input_path) {
        fprintf(stderr, "set REDRAFT_INPUT to the pre-tokenized input file\n");
        return 1;
    }
    inputs in = read_inputs(input_path);

    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params);
    auto *ctx = llama_init->context();
    auto *model = llama_init->model();
    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // prefill the prompt; the last decode leaves logits predicting the first
    // generated token in logits_ith(0).
    std::vector<llama_token> prompt(in.prompt.begin(), in.prompt.end());
    if (prompt.size() > 1) {
        llama_decode(ctx, llama_batch_get_one(prompt.data(), prompt.size() - 1));
    }
    llama_decode(ctx, llama_batch_get_one(&prompt.back(), 1));

    llama_verifier verifier;
    verifier.ctx = ctx;
    verifier.batch = llama_batch_init(std::max<size_t>(in.horizon, 1) + 1, 0, 1);
    verifier.n_vocab = n_vocab;
    verifier.k = 20;
    verifier.template_len = prompt.size();
    verifier.committed_top = top_from_logits(llama_get_logits_ith(ctx, 0), n_vocab, 20);

    accept_rule rule = entropy_floor_tau(in.tau, in.floor);
    auto result = batched_stabilize(
        in.old_ids, std::ref(verifier), rule, in.horizon, 3, in.max_tokens, in.eos);

    std::string ids;
    for (size_t i = 0; i < result.emitted.size(); ++i) {
        ids += (i ? "," : "") + std::to_string(result.emitted[i]);
    }
    printf("{\"emitted\":[%s],\"held_fraction\":%.6f,\"divergences\":%d}\n",
           ids.c_str(), result.held_fraction(), result.divergences);

    llama_batch_free(verifier.batch);
    llama_backend_free();
    return 0;
}
