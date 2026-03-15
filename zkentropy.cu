#include "zkentropy.cuh"
#include <stdexcept>
#include <iostream>

// ── GPU kernels ───────────────────────────────────────────────────────────────

// diffs[i] = v_star - logits[i]
KERNEL void zkentropy_diffs_kernel(const Fr_t* logits, Fr_t v_star, Fr_t* diffs, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        diffs[tid] = blstrs__scalar__Scalar_sub(v_star, logits[tid]);
}

// Clamp diffs to [0, max_idx) so they stay within the CDF table range.
KERNEL void zkentropy_clamp_kernel(Fr_t* diffs, long max_idx, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        long val = scalar_to_long(diffs[tid]);
        if (val < 0)        val = 0;
        if (val >= max_idx) val = max_idx - 1;
        diffs[tid] = long_to_scalar(val);
    }
}

// Normalise win_probs and re-quantise all tokens into log-table indices.
//   q_indices[i] = clamp(round(win_probs[i] * 2^log_precision / total), 1, 2^log_precision)
// Operates with 64-bit integer arithmetic on the GPU.
KERNEL void zkentropy_normalize_kernel(const Fr_t* win_probs, Fr_t total_fr,
                                       Fr_t* q_indices, uint log_precision, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned long long wp  = scalar_to_ulong(win_probs[tid]);
        unsigned long long tot = scalar_to_ulong(total_fr);
        unsigned long long lp  = 1ULL << log_precision;
        // round(wp * lp / tot)
        unsigned long long q = (tot > 0) ? (wp * lp + tot / 2) / tot : 1ULL;
        if (q < 1)   q = 1;
        if (q > lp)  q = lp;
        q_indices[tid] = ulong_to_scalar(q);
    }
}

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

// ── computePosition ───────────────────────────────────────────────────────────
// Computes log for the full vocab so D = vocab_size (needed for the prove step,
// which requires D >= N_table and D divisible by N_table).

Fr_t zkConditionalEntropy::computePosition(const FrTensor& logits, uint actual_token) {
    uint N = vocab_size;
    if (logits.size != N)
        throw std::invalid_argument("computePosition: logits.size != vocab_size");
    uint blocks = (N + FrNumThread - 1) / FrNumThread;

    // Step 1: argmax
    uint t_star = argmax_prover.compute(logits);
    Fr_t v_star = logits(t_star);

    // Step 2: diffs = v_star - logits, clamped to CDF table range
    FrTensor diffs(N);
    zkentropy_diffs_kernel<<<blocks, FrNumThread>>>(logits.gpu_data, v_star, diffs.gpu_data, N);
    cudaDeviceSynchronize();
    zkentropy_clamp_kernel<<<blocks, FrNumThread>>>(diffs.gpu_data, 1L << cdf_precision, N);
    cudaDeviceSynchronize();

    // Step 3: win probabilities via CDF lookup
    auto [win_probs, m_cdf] = cdf_prover.compute(diffs);

    // Step 4: normalise across full vocab → q_indices[i] in [1, 2^log_precision]
    Fr_t total_fr = win_probs.sum();
    FrTensor q_indices(N);
    zkentropy_normalize_kernel<<<blocks, FrNumThread>>>(
        win_probs.gpu_data, total_fr, q_indices.gpu_data, log_precision, N);
    cudaDeviceSynchronize();

    // Step 5: log lookup for all vocab tokens, return actual_token's value
    auto [log_vals, m_log] = log_prover.compute(q_indices);
    return log_vals(actual_token);
}

// ── compute (sequence) ────────────────────────────────────────────────────────

Fr_t zkConditionalEntropy::compute(
    const vector<FrTensor>& logits_seq,
    const vector<uint>& tokens)
{
    if (logits_seq.size() != tokens.size())
        throw std::invalid_argument("compute: logits_seq and tokens must have the same length");

    Fr_t total = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint pos = 0; pos < tokens.size(); pos++) {
        Fr_t s = computePosition(logits_seq[pos], tokens[pos]);
        total  = total + s;
    }
    return total;
}

// ── prove (sequence) ─────────────────────────────────────────────────────────

Fr_t zkConditionalEntropy::prove(
    const vector<FrTensor>& logits_seq,
    const vector<uint>& tokens,
    Fr_t claimed_entropy,
    vector<Polynomial>& proof)
{
    if (logits_seq.size() != tokens.size())
        throw std::invalid_argument("prove: logits_seq and tokens must have the same length");

    uint T      = tokens.size();
    uint N      = vocab_size;
    uint blocks = (N + FrNumThread - 1) / FrNumThread;

    Fr_t entropy_sum   = {0, 0, 0, 0, 0, 0, 0, 0};
    Fr_t logits_claims = {0, 0, 0, 0, 0, 0, 0, 0};

    for (uint pos = 0; pos < T; pos++) {
        const FrTensor& logits = logits_seq[pos];
        uint actual_token      = tokens[pos];

        // ── Argmax ────────────────────────────────────────────────────────
        uint t_star  = argmax_prover.compute(logits);
        Fr_t v_star  = logits(t_star);
        auto u_arg   = random_vec(ceilLog2(N));
        Fr_t lc      = argmax_prover.prove(logits, t_star, v_star, u_arg, proof);
        logits_claims = logits_claims + lc;

        // ── Diffs (GPU) ───────────────────────────────────────────────────
        FrTensor diffs(N);
        zkentropy_diffs_kernel<<<blocks, FrNumThread>>>(
            logits.gpu_data, v_star, diffs.gpu_data, N);
        cudaDeviceSynchronize();
        zkentropy_clamp_kernel<<<blocks, FrNumThread>>>(
            diffs.gpu_data, 1L << cdf_precision, N);
        cudaDeviceSynchronize();

        // ── CDF lookup + prove ────────────────────────────────────────────
        auto [win_probs, m_cdf] = cdf_prover.compute(diffs);
        {
            auto rand3 = random_vec(3);
            auto u_cdf = random_vec(ceilLog2(N));
            auto v_cdf = random_vec(ceilLog2(N));
            cdf_prover.prove(diffs, win_probs, m_cdf,
                rand3[0], rand3[1], rand3[2], u_cdf, v_cdf, proof);
        }

        // ── Normalise (GPU) ───────────────────────────────────────────────
        Fr_t total_fr = win_probs.sum();
        FrTensor q_indices(N);
        zkentropy_normalize_kernel<<<blocks, FrNumThread>>>(
            win_probs.gpu_data, total_fr, q_indices.gpu_data, log_precision, N);
        cudaDeviceSynchronize();

        // TODO: prove normalisation — for each token i, prove
        //   q_indices[i] * total == win_probs[i] * 2^log_precision - remainder_i
        //   with 0 <= remainder_i < total, via bit decomposition of remainder.

        // ── Log lookup + prove ────────────────────────────────────────────
        auto [log_vals, m_log] = log_prover.compute(q_indices);
        {
            auto rand3 = random_vec(3);
            auto u_log = random_vec(ceilLog2(N));
            auto v_log = random_vec(ceilLog2(N));
            log_prover.prove(q_indices, log_vals, m_log,
                rand3[0], rand3[1], rand3[2], u_log, v_log, proof);
        }

        Fr_t s     = log_vals(actual_token);
        entropy_sum = entropy_sum + s;
    }

    if (entropy_sum != claimed_entropy)
        throw std::runtime_error("zkConditionalEntropy::prove: entropy mismatch");

    std::cout << "zkConditionalEntropy::prove complete over " << T << " positions." << std::endl;
    return logits_claims;
}
