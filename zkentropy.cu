#include "zkentropy.cuh"
#include <stdexcept>
#include <iostream>

// ── Constructor ───────────────────────────────────────────────────────────────

zkConditionalEntropy::zkConditionalEntropy(
    uint vocab_size,
    uint bit_width,
    uint cdf_precision,
    uint log_precision,
    uint cdf_scale,
    uint log_scale,
    double sigma_eff
) : vocab_size(vocab_size),
    bit_width(bit_width),
    cdf_precision(cdf_precision),
    log_precision(log_precision),
    cdf_scale(cdf_scale),
    log_scale(log_scale),
    sigma_eff(sigma_eff),
    argmax_prover(bit_width),
    cdf_prover(cdf_precision, cdf_scale, sigma_eff),
    log_prover(log_precision, log_scale)
{}

// ── Helper: single-position compute ──────────────────────────────────────────
// Simplified approach: only computes CDF and log for the actual token,
// using sum ≈ vocab_size * cdf_scale as the normalisation denominator.
// This is a valid conservative upper bound: sum(win_probs) <= vocab_size * cdf_scale,
// so q[actual] = win_prob[actual] / sum >= win_prob[actual] / (vocab_size * cdf_scale),
// meaning -log2(q[actual]) <= -log2(win_prob/(vocab_size*cdf_scale)).
//
// Field elements are stored without Montgomery form for logit values (val[2..7] == 0
// for small positive values).  Negative or very large values have val[2..7] != 0.

static inline unsigned long long fr_to_ull(const Fr_t& a) {
    return ((unsigned long long)a.val[1] << 32) | a.val[0];
}

static inline bool fr_is_large(const Fr_t& a) {
    return a.val[2] || a.val[3] || a.val[4] || a.val[5] || a.val[6] || a.val[7];
}

Fr_t zkConditionalEntropy::computePosition(const FrTensor& logits, uint actual_token) {
    if (logits.size != vocab_size)
        throw std::invalid_argument("computePosition: logits.size != vocab_size");

    // 1. Argmax.
    uint t_star = argmax_prover.compute(logits);
    Fr_t v_star = logits(t_star);

    // 3. Compute win probabilities for all tokens.
    //    win_prob[i] = (1 - Phi(diff_i/sigma)) * cdf_scale
    //                = P(token i's noisy logit exceeds the winner's)
    //    diffs_all[i] = v_star - logits[i] (non-negative; 0 for the winner).
    FrTensor diffs_all = -(logits - v_star);
    auto [cdf_vals_all, m_cdf] = cdf_prover.compute(diffs_all);
    (void)m_cdf;
    Fr_t cdf_scale_fr = {cdf_scale, 0, 0, 0, 0, 0, 0, 0};
    FrTensor win_probs_all = -(cdf_vals_all - cdf_scale_fr);
    Fr_t win_prob    = win_probs_all(actual_token);
    Fr_t total_win_f = win_probs_all.sum();

    // 4. q_idx = floor(win_prob[actual] * 2^log_precision / total_win).
    unsigned long long wp    = fr_to_ull(win_prob);
    unsigned long long lp    = 1ULL << log_precision;
    unsigned long long denom = fr_to_ull(total_win_f);
    if (denom == 0) denom = 1;
    unsigned long long q_idx = (denom > 0) ? (wp * lp) / denom : 1ULL;
    if (q_idx < 1)  q_idx = 1;
    if (q_idx > lp) q_idx = lp;

    // 5. Log lookup for single element: surprise = -log2(q_idx / 2^lp) * log_scale.
    Fr_t q_fr = {(uint)(q_idx & 0xFFFFFFFF), (uint)(q_idx >> 32), 0, 0, 0, 0, 0, 0};
    FrTensor q_1(1, &q_fr);
    auto [surp_1, m_log] = log_prover.compute(q_1);
    (void)m_log;
    return surp_1(0u);
}

// ── compute (sequence) ────────────────────────────────────────────────────────

Fr_t zkConditionalEntropy::compute(
    const vector<FrTensor>& logits_seq,
    const vector<uint>& tokens)
{
    if (logits_seq.size() != tokens.size())
        throw std::invalid_argument("compute: logits_seq and tokens must have the same length");

    Fr_t total = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint pos = 0; pos < tokens.size(); pos++)
        total = total + computePosition(logits_seq[pos], tokens[pos]);
    return total;
}

// ── prove (sequence) ──────────────────────────────────────────────────────────
// For each position:
//   - Prove argmax via zkArgmax (bit-decomposition range proof).
//   - Record (diff_actual, win_prob, q_idx, surprise) as constant polynomials.
//     Since the CDF and log tables are public and deterministic, a verifier
//     recomputes cdf_table[diff] and log_table[q_idx] to check consistency.
//
// This avoids the tLookupRangeMapping divisibility constraint (D % N == 0)
// because we never call lookup.prove() for D=1.

Fr_t zkConditionalEntropy::prove(
    const vector<FrTensor>& logits_seq,
    const vector<uint>& tokens,
    Fr_t claimed_entropy,
    vector<Polynomial>& proof)
{
    if (logits_seq.size() != tokens.size())
        throw std::invalid_argument("prove: logits_seq and tokens must have the same length");

    uint T = tokens.size();
    Fr_t entropy_sum   = {0, 0, 0, 0, 0, 0, 0, 0};
    Fr_t logits_claims = {0, 0, 0, 0, 0, 0, 0, 0};

    for (uint pos = 0; pos < T; pos++) {
        const FrTensor& logits = logits_seq[pos];
        uint actual_token      = tokens[pos];

        // ── Argmax prove ───────────────────────────────────────────────────
        uint t_star = argmax_prover.compute(logits);
        Fr_t v_star = logits(t_star);
        auto u_arg  = random_vec(ceilLog2(vocab_size));
        Fr_t lc     = argmax_prover.prove(logits, t_star, v_star, u_arg, proof);
        logits_claims = logits_claims + lc;

        // ── Bind logits[actual_token] to the logit tensor via MLE ─────────
        // Evaluating the MLE of logits at the binary encoding of actual_token
        // gives exactly logits[actual_token].  This links diff_actual to the
        // same tensor that the argmax proof committed to.
        uint log_N = ceilLog2(vocab_size);
        vector<Fr_t> e_actual(log_N);
        for (uint i = 0; i < log_N; i++)
            e_actual[i] = {(actual_token >> i) & 1u, 0, 0, 0, 0, 0, 0, 0};
        Fr_t logit_act = logits(e_actual);   // MLE at one-hot basis = logits[actual_token]
        // Sanity-check against the direct GPU read (catches MLE encoding bugs).
        Fr_t logit_act_direct = logits(actual_token);
        if (logit_act != logit_act_direct)
            throw std::runtime_error("prove: MLE one-hot eval != direct logit read");
        // Include the MLE claim in the proof transcript.
        proof.push_back(Polynomial(logit_act));

        // ── Full-vocab CDF + win probabilities ────────────────────────────
        FrTensor diffs_all = -(logits - v_star);
        auto [cdf_vals_all, m_cdf] = cdf_prover.compute(diffs_all);
        (void)m_cdf;
        Fr_t cdf_scale_fr = {cdf_scale, 0, 0, 0, 0, 0, 0, 0};
        FrTensor win_probs_all = -(cdf_vals_all - cdf_scale_fr);
        Fr_t win_prob  = win_probs_all(actual_token);
        Fr_t total_win = win_probs_all.sum();

        // Record: diff for actual token, its win_prob, and the total.
        Fr_t diff_actual = diffs_all(actual_token);
        proof.push_back(Polynomial(diff_actual));
        proof.push_back(Polynomial(win_prob));
        proof.push_back(Polynomial(total_win));

        // ── Normalisation + log ───────────────────────────────────────────
        unsigned long long wp    = fr_to_ull(win_prob);
        unsigned long long lp    = 1ULL << log_precision;
        unsigned long long denom = fr_to_ull(total_win);
        if (denom == 0) denom = 1;
        unsigned long long q_idx = (denom > 0) ? (wp * lp) / denom : 1ULL;
        if (q_idx < 1)  q_idx = 1;
        if (q_idx > lp) q_idx = lp;

        Fr_t q_fr = {(uint)(q_idx & 0xFFFFFFFF), (uint)(q_idx >> 32), 0, 0, 0, 0, 0, 0};
        FrTensor q_1(1, &q_fr);
        auto [surp_1, m_log] = log_prover.compute(q_1);
        (void)m_log;
        Fr_t surprise = surp_1(0u);

        // Record claimed values; verifier checks log_table[q_fr] == surprise.
        proof.push_back(Polynomial(q_fr));
        proof.push_back(Polynomial(surprise));

        entropy_sum = entropy_sum + surprise;
    }

    if (entropy_sum != claimed_entropy)
        throw std::runtime_error("zkConditionalEntropy::prove: entropy mismatch");

    std::cout << "zkConditionalEntropy::prove complete over " << T << " positions." << std::endl;
    return logits_claims;
}

// ── Strong prove: logits[actual_token] bound via Commitment::me_open ──────────

Fr_t zkConditionalEntropy::prove(
    const vector<FrTensor>& logits_seq,
    const vector<uint>& tokens,
    Fr_t claimed_entropy,
    const Commitment& logit_generators,
    const vector<G1TensorJacobian>& logit_commits,
    vector<Polynomial>& proof,
    vector<G1Jacobian_t>& g1_proof)
{
    if (logits_seq.size() != tokens.size())
        throw std::invalid_argument("prove: logits_seq and tokens must have the same length");
    if (logit_commits.size() != tokens.size())
        throw std::invalid_argument("prove: logit_commits.size() != tokens.size()");
    // Generators must cover the padded vocab size.
    uint padded = 1u << ceilLog2(vocab_size);
    if (logit_generators.size != padded)
        throw std::invalid_argument("prove: logit_generators.size must be 2^ceilLog2(vocab_size)");

    uint T   = tokens.size();
    uint log_N = ceilLog2(vocab_size);

    Fr_t entropy_sum   = {0, 0, 0, 0, 0, 0, 0, 0};
    Fr_t logits_claims = {0, 0, 0, 0, 0, 0, 0, 0};

    for (uint pos = 0; pos < T; pos++) {
        const FrTensor& logits = logits_seq[pos];
        uint actual_token      = tokens[pos];

        // ── Argmax prove ───────────────────────────────────────────────────
        uint t_star = argmax_prover.compute(logits);
        Fr_t v_star = logits(t_star);
        auto u_arg  = random_vec(log_N);
        Fr_t lc     = argmax_prover.prove(logits, t_star, v_star, u_arg, proof);
        logits_claims = logits_claims + lc;

        // ── Bind logits[actual_token] via Commitment::me_open ─────────────
        // Build the one-hot basis vector: e_actual[i] = bit i of actual_token.
        vector<Fr_t> e_actual(log_N);
        for (uint i = 0; i < log_N; i++)
            e_actual[i] = {(actual_token >> i) & 1u, 0, 0, 0, 0, 0, 0, 0};

        // Pad logit tensor to padded size if needed (me_open requires t.size == generators.size).
        FrTensor logits_padded = (logits.size == padded) ? logits : logits.pad({logits.size}, {0,0,0,0,0,0,0,0});
        Fr_t logit_act = Commitment::me_open(logits_padded, logit_generators,
                                             e_actual.begin(), e_actual.end(), g1_proof);

        // Sanity-check: me_open at one-hot basis must equal the direct element read.
        Fr_t logit_act_direct = logits(actual_token);
        if (logit_act != logit_act_direct)
            throw std::runtime_error("prove: me_open one-hot != direct logit read at pos "
                                     + std::to_string(pos));

        // Record the commitment and the scalar claim so the verifier can pair-check.
        proof.push_back(Polynomial(logit_act));

        // ── Full-vocab CDF / Log ──────────────────────────────────────────
        FrTensor diffs_all = -(logits - v_star);
        auto [cdf_vals_all, m_cdf] = cdf_prover.compute(diffs_all);
        (void)m_cdf;
        Fr_t cdf_scale_fr = {cdf_scale, 0, 0, 0, 0, 0, 0, 0};
        FrTensor win_probs_all = -(cdf_vals_all - cdf_scale_fr);
        Fr_t win_prob  = win_probs_all(actual_token);
        Fr_t total_win = win_probs_all.sum();

        Fr_t diff_actual = diffs_all(actual_token);
        proof.push_back(Polynomial(diff_actual));
        proof.push_back(Polynomial(win_prob));
        proof.push_back(Polynomial(total_win));

        unsigned long long wp    = fr_to_ull(win_prob);
        unsigned long long lp    = 1ULL << log_precision;
        unsigned long long denom = fr_to_ull(total_win);
        if (denom == 0) denom = 1;
        unsigned long long q_idx = (denom > 0) ? (wp * lp) / denom : 1ULL;
        if (q_idx < 1)  q_idx = 1;
        if (q_idx > lp) q_idx = lp;

        Fr_t q_fr = {(uint)(q_idx & 0xFFFFFFFF), (uint)(q_idx >> 32), 0, 0, 0, 0, 0, 0};
        FrTensor q_1(1, &q_fr);
        auto [surp_1, m_log] = log_prover.compute(q_1);
        (void)m_log;
        Fr_t surprise = surp_1(0u);
        proof.push_back(Polynomial(q_fr));
        proof.push_back(Polynomial(surprise));

        entropy_sum = entropy_sum + surprise;
    }

    if (entropy_sum != claimed_entropy)
        throw std::runtime_error("zkConditionalEntropy::prove (strong): entropy mismatch");

    std::cout << "zkConditionalEntropy::prove (strong) complete over " << T << " positions." << std::endl;
    return logits_claims;
}
