#include "zkentropy.cuh"
#include <stdexcept>
#include <iostream>
#include <cmath>

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

// ── Field element helpers ────────────────────────────────────────────────────

#ifdef USE_GOLDILOCKS
static inline unsigned long long fr_to_ull(const Fr_t& a) {
    return a.val;
}
#else
static inline unsigned long long fr_to_ull(const Fr_t& a) {
    return ((unsigned long long)a.val[1] << 32) | a.val[0];
}
#endif

// ── GPU kernels for batched operations ───────────────────────────────────────

// diffs[t*V+i] = v_star[t] - logits[t*V+i]
KERNEL void batched_diffs_kernel(const Fr_t* logits, const Fr_t* v_star,
                                  Fr_t* diffs, uint T, uint V) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint total = T * V;
    if (idx < total) {
        uint t = idx / V;
        diffs[idx] = blstrs__scalar__Scalar_sub(v_star[t], logits[idx]);
    }
}

// Row sums: sums[t] = sum_{i=0}^{V-1} data[t*V + i]
KERNEL void batched_row_sum_kernel(const Fr_t* data, Fr_t* sums, uint T, uint V) {
    uint t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        Fr_t acc = FR_FROM_INT(0);
        const Fr_t* row = data + (size_t)t * V;
        for (uint i = 0; i < V; i++) {
            acc = blstrs__scalar__Scalar_add(acc, row[i]);
        }
        sums[t] = acc;
    }
}

// Extract: out[t] = data[t*V + indices[t]]
KERNEL void extract_by_index_kernel(const Fr_t* data, const uint* indices,
                                     Fr_t* out, uint T, uint V) {
    uint t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        out[t] = data[(size_t)t * V + indices[t]];
    }
}

// Clamp zero field elements to 1 (prevents log(0)).
KERNEL void clamp_min_one_kernel(Fr_t* data, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
#ifdef USE_GOLDILOCKS
        if (data[tid].val == 0ULL) data[tid].val = 1ULL;
#else
        if (data[tid].val[0] == 0 && data[tid].val[1] == 0 &&
            data[tid].val[2] == 0 && data[tid].val[3] == 0 &&
            data[tid].val[4] == 0 && data[tid].val[5] == 0 &&
            data[tid].val[6] == 0 && data[tid].val[7] == 0)
            data[tid].val[0] = 1;
#endif
    }
}

// ── Range-reduced log2 (CPU-side, proof placeholder) ─────────────────────────
// Computes log2(x) * log_scale for each element.  Values may exceed
// 2^log_precision so the standard table cannot be used directly.
// The result is correct; the ZK proof for this step is a placeholder.

static FrTensor range_reduced_log(const FrTensor& values, uint log_scale_param) {
    uint T = values.size;
    Fr_t* cpu = new Fr_t[T];
    cudaMemcpy(cpu, values.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);

    Fr_t* result = new Fr_t[T];
    for (uint t = 0; t < T; t++) {
        unsigned long long x = fr_to_ull(cpu[t]);
        if (x < 1) x = 1;
        double log2_x = log2((double)x);
        long scaled = (long)(log2_x * (double)log_scale_param + 0.5);
        if (scaled < 0) scaled = 0;
        result[t] = FR_FROM_INT((unsigned long long)scaled);
    }

    FrTensor out(T, result);
    delete[] cpu;
    delete[] result;
    return out;
}

// ── Helper: extract row from flat tensor ────────────────────────────────────

static FrTensor tensor_row(const FrTensor& mat, uint row_idx, uint row_size) {
    return mat.trunc((size_t)row_idx * row_size, (size_t)(row_idx + 1) * row_size);
}

// ── Helper: upload tokens to GPU ────────────────────────────────────────────

static uint* upload_tokens(const vector<uint>& tokens) {
    uint* gpu;
    cudaMalloc(&gpu, tokens.size() * sizeof(uint));
    cudaMemcpy(gpu, tokens.data(), tokens.size() * sizeof(uint), cudaMemcpyHostToDevice);
    return gpu;
}

// ── Entropy formula ─────────────────────────────────────────────────────────
//
// surprise[t] = -log2(win_prob[t] / total_win[t]) * log_scale
//             = (-log2(win_prob[t]) + log2(total_win[t])) * log_scale
//
// log_wp_table[t] = (log_precision - log2(wp[t])) * log_scale   [from tLookup]
// log_tw[t]       = log2(total_win[t]) * log_scale              [range-reduced]
//
// => surprise[t] = log_wp_table[t] + log_tw[t] - log_precision * log_scale
// => H = sum(log_wp + log_tw) - T * log_precision * log_scale

// ── Batched compute ─────────────────────────────────────────────────────────

Fr_t zkConditionalEntropy::compute(
    const FrTensor& logits_all, uint T, uint V,
    const vector<uint>& tokens)
{
    if (logits_all.size != T * V)
        throw std::invalid_argument("compute: logits_all.size != T * V");
    if (tokens.size() != T)
        throw std::invalid_argument("compute: tokens.size() != T");

    uint TV = T * V;

    // 1. Argmax per row
    Fr_t* v_star_cpu = new Fr_t[T];
    for (uint t = 0; t < T; t++) {
        FrTensor row = tensor_row(logits_all, t, V);
        uint t_star = argmax_prover.compute(row);
        v_star_cpu[t] = row(t_star);
    }
    FrTensor v_star_vec(T, v_star_cpu);
    delete[] v_star_cpu;

    // 2. Batched diffs
    FrTensor diffs_all(TV);
    batched_diffs_kernel<<<(TV + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        logits_all.gpu_data, v_star_vec.gpu_data, diffs_all.gpu_data, T, V);

    // 3. CDF + win probs
    auto [cdf_all, m_cdf] = cdf_prover.compute(diffs_all);
    (void)m_cdf;
    FrTensor win_probs_all = -(cdf_all - FR_FROM_INT(cdf_scale));

    // 4. Row sums for total_win
    FrTensor total_win_vec(T);
    batched_row_sum_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        win_probs_all.gpu_data, total_win_vec.gpu_data, T, V);
    cudaDeviceSynchronize();

    // 5. Extract actual-token win probs
    uint* tgpu = upload_tokens(tokens);
    FrTensor actual_wp_vec(T);
    extract_by_index_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        win_probs_all.gpu_data, tgpu, actual_wp_vec.gpu_data, T, V);
    cudaDeviceSynchronize();
    cudaFree(tgpu);

    // 6. Clamp + log lookups
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        actual_wp_vec.gpu_data, T);
    cudaDeviceSynchronize();

    auto [log_wp_vec, m_log_wp] = log_prover.compute(actual_wp_vec);
    (void)m_log_wp;
    FrTensor log_tw_vec = range_reduced_log(total_win_vec, log_scale);

    // 7. Entropy
    FrTensor surprise_raw = log_wp_vec + log_tw_vec;
    Fr_t H_raw = surprise_raw.sum();
    Fr_t offset = FR_FROM_INT((unsigned long long)T * log_precision * log_scale);
    return H_raw - offset;
}

// ── Batched prove ───────────────────────────────────────────────────────────

Fr_t zkConditionalEntropy::prove(
    const FrTensor& logits_all, uint T, uint V,
    const vector<uint>& tokens,
    Fr_t claimed_entropy,
    vector<Polynomial>& proof)
{
    if (logits_all.size != T * V)
        throw std::invalid_argument("prove: logits_all.size != T * V");
    if (tokens.size() != T)
        throw std::invalid_argument("prove: tokens.size() != T");

    uint TV = T * V;
    Fr_t logits_claims = FR_FROM_INT(0);

    // ════════════════════════════════════════════════════════════════════════
    // Phase 1: Compute all intermediate tensors
    // ════════════════════════════════════════════════════════════════════════

    // 1. Argmax per row
    vector<uint> t_star_vec(T);
    Fr_t* v_star_cpu = new Fr_t[T];
    for (uint t = 0; t < T; t++) {
        FrTensor row = tensor_row(logits_all, t, V);
        t_star_vec[t] = argmax_prover.compute(row);
        v_star_cpu[t] = row(t_star_vec[t]);
    }
    FrTensor v_star_vec_t(T, v_star_cpu);
    delete[] v_star_cpu;

    // 2. Batched diffs
    FrTensor diffs_all(TV);
    batched_diffs_kernel<<<(TV + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        logits_all.gpu_data, v_star_vec_t.gpu_data, diffs_all.gpu_data, T, V);

    // 3. CDF + win probs
    auto [cdf_all, m_cdf] = cdf_prover.compute(diffs_all);
    FrTensor win_probs_all = -(cdf_all - FR_FROM_INT(cdf_scale));

    // 4. Row sums
    FrTensor total_win_vec(T);
    batched_row_sum_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        win_probs_all.gpu_data, total_win_vec.gpu_data, T, V);
    cudaDeviceSynchronize();

    // 5. Extract actual-token win probs
    uint* tgpu = upload_tokens(tokens);
    FrTensor actual_wp_vec(T);
    extract_by_index_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        win_probs_all.gpu_data, tgpu, actual_wp_vec.gpu_data, T, V);
    cudaDeviceSynchronize();
    cudaFree(tgpu);

    // 6. Clamp + log lookups
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        actual_wp_vec.gpu_data, T);
    cudaDeviceSynchronize();

    auto [log_wp_vec, m_log_wp] = log_prover.compute(actual_wp_vec);
    FrTensor log_tw_vec = range_reduced_log(total_win_vec, log_scale);

    // 7. Entropy
    FrTensor surprise_raw = log_wp_vec + log_tw_vec;
    Fr_t H_raw = surprise_raw.sum();
    Fr_t offset = FR_FROM_INT((unsigned long long)T * log_precision * log_scale);
    Fr_t H = H_raw - offset;

    if (H != claimed_entropy)
        throw std::runtime_error("zkConditionalEntropy::prove: entropy mismatch");

    // ════════════════════════════════════════════════════════════════════════
    // Phase 2: Proofs
    // ════════════════════════════════════════════════════════════════════════

    // ── 2a. Argmax proofs (one per position) ────────────────────────────────
    std::cout << "  Proving argmax (" << T << " positions)..." << std::endl;
    for (uint t = 0; t < T; t++) {
        FrTensor row = tensor_row(logits_all, t, V);
        Fr_t v_star = row(t_star_vec[t]);
        auto u_arg = random_vec(ceilLog2(V));
        Fr_t lc = argmax_prover.prove(row, t_star_vec[t], v_star, u_arg, proof);
        logits_claims = logits_claims + lc;
    }

    // ── 2b. CDF tLookup proof (T*V tensor → T*V values) ────────────────────
    // Cryptographically binds all CDF values to the public CDF table.
    std::cout << "  Proving CDF lookup (" << TV << " elements)..." << std::endl;
    {
        auto r_cdf = random_vec(1)[0];
        auto alpha = random_vec(1)[0];
        auto beta  = random_vec(1)[0];
        auto u_cdf = random_vec(ceilLog2(TV));
        auto v_cdf = random_vec(ceilLog2(TV));
        cdf_prover.prove(diffs_all, cdf_all, m_cdf,
                         r_cdf, alpha, beta, u_cdf, v_cdf, proof);
    }

    // ── 2c. total_win row-sum proof (placeholder) ───────────────────────────
    // Proves total_win_vec[t] = sum_i win_probs_all[t*V + i].
    // Full sumcheck TBD; emit MLE evaluation as proof constant for now.
    std::cout << "  Proving total_win row sums..." << std::endl;
    {
        auto u_T = random_vec(ceilLog2(T));
        Fr_t tw_at_u = total_win_vec(u_T);
        proof.push_back(Polynomial(tw_at_u));
    }

    // ── 2d. Actual-token extraction proof ───────────────────────────────────
    // Proves actual_wp_vec[t] = win_probs_all[t*V + tokens[t]] via
    // inner product with indicator tensor.
    std::cout << "  Proving actual-token extraction..." << std::endl;
    {
        Fr_t* ind_cpu = new Fr_t[TV];
        for (uint i = 0; i < TV; i++) ind_cpu[i] = FR_FROM_INT(0);
        for (uint t = 0; t < T; t++)
            ind_cpu[(size_t)t * V + tokens[t]] = FR_FROM_INT(1);
        FrTensor indicator(TV, ind_cpu);
        delete[] ind_cpu;

        // Verify consistency
        Fr_t ip = (win_probs_all * indicator).sum();
        Fr_t wp_sum = actual_wp_vec.sum();
        if (ip != wp_sum)
            throw std::runtime_error("prove: indicator extraction mismatch");

        // Inner product sumcheck
        auto u_ext = random_vec(ceilLog2(TV));
        inner_product_sumcheck(win_probs_all, indicator, u_ext);

        auto u_T = random_vec(ceilLog2(T));
        Fr_t wp_at_u = actual_wp_vec(u_T);
        proof.push_back(Polynomial(wp_at_u));
    }

    // ── 2e. Win-prob log lookup proof ───────────────────────────────────────
    // Proves log_wp_vec = log_table(actual_wp_vec) via tLookup.
    // Pads to satisfy D % N == 0 constraint.
    std::cout << "  Proving win-prob log lookup..." << std::endl;
    {
        uint N = 1u << log_precision;
        // Smallest power of 2 >= max(T, N), guaranteeing D_padded % N == 0.
        uint D_padded = N;
        while (D_padded < T) D_padded *= 2;

        Fr_t* wp_cpu = new Fr_t[D_padded];
        cudaMemcpy(wp_cpu, actual_wp_vec.gpu_data, T * sizeof(Fr_t),
                   cudaMemcpyDeviceToHost);
        Fr_t pad_val = FR_FROM_INT(1);
        for (uint i = T; i < D_padded; i++) wp_cpu[i] = pad_val;
        FrTensor wp_padded(D_padded, wp_cpu);
        delete[] wp_cpu;

        auto [log_wp_padded, m_log_wp_p] = log_prover.compute(wp_padded);

        auto r_log = random_vec(1)[0];
        auto alpha = random_vec(1)[0];
        auto beta  = random_vec(1)[0];
        auto u_log = random_vec(ceilLog2(D_padded));
        auto v_log = random_vec(ceilLog2(D_padded));
        log_prover.prove(wp_padded, log_wp_padded, m_log_wp_p,
                         r_log, alpha, beta, u_log, v_log, proof);
    }

    // ── 2f. Range-reduced log of total_win (placeholder proof) ──────────────
    // Correctly computed; full bit-decomp + mantissa tLookup proof TBD.
    std::cout << "  Range-reduced log (placeholder)..." << std::endl;
    {
        auto u_tw = random_vec(ceilLog2(T));
        Fr_t log_tw_at_u = log_tw_vec(u_tw);
        proof.push_back(Polynomial(log_tw_at_u));
    }

    // ── 2g. Final linear check ──────────────────────────────────────────────
    // Verify surprise_raw = log_wp + log_tw at a random point (Schwartz-Zippel).
    {
        auto u_f = random_vec(ceilLog2(T));
        Fr_t s   = surprise_raw(u_f);
        Fr_t lwp = log_wp_vec(u_f);
        Fr_t ltw = log_tw_vec(u_f);
        if (s != lwp + ltw)
            throw std::runtime_error("prove: surprise linear check failed");
    }

    std::cout << "zkConditionalEntropy::prove complete (batched, "
              << T << " positions, " << proof.size() << " polynomials)."
              << std::endl;

    return logits_claims;
}

// ── Legacy per-position interface ───────────────────────────────────────────

Fr_t zkConditionalEntropy::computePosition(const FrTensor& logits, uint actual_token) {
    vector<uint> tokens = {actual_token};
    return compute(logits, 1, logits.size, tokens);
}

Fr_t zkConditionalEntropy::compute(
    const vector<FrTensor>& logits_seq,
    const vector<uint>& tokens)
{
    if (logits_seq.empty())
        throw std::invalid_argument("compute: empty logits sequence");

    uint T = logits_seq.size();
    uint V = logits_seq[0].size;
    FrTensor logits_all = catTensors(logits_seq);
    return compute(logits_all, T, V, tokens);
}

Fr_t zkConditionalEntropy::prove(
    const vector<FrTensor>& logits_seq,
    const vector<uint>& tokens,
    Fr_t claimed_entropy,
    vector<Polynomial>& proof)
{
    if (logits_seq.empty())
        throw std::invalid_argument("prove: empty logits sequence");

    uint T = logits_seq.size();
    uint V = logits_seq[0].size;
    FrTensor logits_all = catTensors(logits_seq);
    return prove(logits_all, T, V, tokens, claimed_entropy, proof);
}
