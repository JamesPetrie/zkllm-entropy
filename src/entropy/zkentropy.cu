#include "entropy/zkentropy.cuh"
#include <stdexcept>
#include <iostream>
#include <cmath>

// ── Constructor ───────────────────────────────────────────────────────────────

zkConditionalEntropy::zkConditionalEntropy(
    uint vocab_size,
    uint cdf_precision,
    uint log_precision,
    uint cdf_scale,
    uint log_scale,
    double sigma_eff
) : vocab_size(vocab_size),
    cdf_precision(cdf_precision),
    log_precision(log_precision),
    cdf_scale(cdf_scale),
    log_scale(log_scale),
    sigma_eff(sigma_eff),
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

// ── Host-side argmax scan ────────────────────────────────────────────────────

static bool fr_gt(const Fr_t& a, const Fr_t& b) {
#ifdef USE_GOLDILOCKS
    bool a_neg = a.val > (GOLDILOCKS_P >> 1);
    bool b_neg = b.val > (GOLDILOCKS_P >> 1);
    if (a_neg != b_neg) return b_neg;
    return a.val > b.val;
#else
    bool a_neg = a.val[2] || a.val[3] || a.val[4] || a.val[5] || a.val[6] || a.val[7];
    bool b_neg = b.val[2] || b.val[3] || b.val[4] || b.val[5] || b.val[6] || b.val[7];
    if (a_neg != b_neg) return b_neg;
    unsigned long long av = ((unsigned long long)a.val[1] << 32) | a.val[0];
    unsigned long long bv = ((unsigned long long)b.val[1] << 32) | b.val[0];
    return av > bv;
#endif
}

static uint find_argmax(const FrTensor& logits) {
    uint N = logits.size;
    Fr_t* cpu = new Fr_t[N];
    cudaMemcpy(cpu, logits.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    uint best = 0;
    for (uint i = 1; i < N; i++) {
        if (fr_gt(cpu[i], cpu[best])) best = i;
    }
    delete[] cpu;
    return best;
}

// ── GPU kernels for batched operations ───────────────────────────────────────

KERNEL void batched_diffs_kernel(const Fr_t* logits, const Fr_t* v_star,
                                  Fr_t* diffs, uint T, uint V) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint total = T * V;
    if (idx < total) {
        uint t = idx / V;
        diffs[idx] = blstrs__scalar__Scalar_sub(v_star[t], logits[idx]);
    }
}

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

KERNEL void extract_by_index_kernel(const Fr_t* data, const uint* indices,
                                     Fr_t* out, uint T, uint V) {
    uint t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        out[t] = data[(size_t)t * V + indices[t]];
    }
}

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

KERNEL void clamp_diffs_kernel(Fr_t* diffs, uint N, uint64_t max_val) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
#ifdef USE_GOLDILOCKS
        uint64_t v = diffs[tid].val;
        if (v >= (GOLDILOCKS_P >> 1)) diffs[tid] = FR_FROM_INT(0);
        else if (v > max_val) diffs[tid] = {max_val};
#else
        bool neg = diffs[tid].val[2] || diffs[tid].val[3] || diffs[tid].val[4] ||
                   diffs[tid].val[5] || diffs[tid].val[6] || diffs[tid].val[7];
        if (neg) diffs[tid] = FR_FROM_INT(0);
        else {
            uint64_t v = ((uint64_t)diffs[tid].val[1] << 32) | diffs[tid].val[0];
            if (v > max_val) { diffs[tid] = FR_FROM_INT(0); diffs[tid].val[0] = (uint32_t)max_val; diffs[tid].val[1] = (uint32_t)(max_val >> 32); }
        }
#endif
    }
}

KERNEL void entropy_bit_extract_kernel(const Fr_t* vals, Fr_t* bits_b, uint bit, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        long val = scalar_to_long(vals[tid]);
        bits_b[tid] = long_to_scalar((val >> bit) & 1L);
    }
}

// ── Bit-decomposition non-negativity proof ──────────────────────────────────
// Emits proof elements: vals(u), then bits_b(u) for each bit plane.

static void prove_nonneg(const FrTensor& vals, uint num_bits,
                          const vector<Fr_t>& u,
                          vector<FrTensor>& bit_planes,
                          FrTensor& combined_error,
                          uint& batch_idx,
                          vector<Polynomial>& proof) {
    uint N = vals.size;
    uint blocks = (N + FrNumThread - 1) / FrNumThread;

    uint base = bit_planes.size();
    for (uint b = 0; b < num_bits; b++) {
        bit_planes.emplace_back(N);
        entropy_bit_extract_kernel<<<blocks, FrNumThread>>>(
            vals.gpu_data, bit_planes.back().gpu_data, b, N);
    }
    cudaDeviceSynchronize();

    Fr_t vals_u = vals(u);
    Fr_t recon  = FR_FROM_INT(0);
    Fr_t pow2   = FR_FROM_INT(1);
    Fr_t two    = FR_FROM_INT(2);

    proof.push_back(Polynomial(vals_u));

    for (uint b = 0; b < num_bits; b++) {
        Fr_t bits_b_u = bit_planes[base + b](u);
        proof.push_back(Polynomial(bits_b_u));
        recon = recon + pow2 * bits_b_u;
        pow2  = pow2 * two;
    }
    if (recon != vals_u)
        throw std::runtime_error("prove_nonneg: bit reconstruction mismatch");

    auto r = random_vec(num_bits);
    for (uint b = 0; b < num_bits; b++) {
        combined_error += (bit_planes[base + b] * bit_planes[base + b]
                           - bit_planes[base + b]) * r[b];
    }
    batch_idx += num_bits;
}


// ── Serialize IP sumcheck proof to Polynomial vector ────────────────────────
// inner_product_sumcheck returns vector<Fr_t>: [3 values per round] + [a(u), b(u)]
// Convert to Polynomials: one degree-2 poly per round + 2 constant polys for finals.

static void serialize_ip_sumcheck(const vector<Fr_t>& ip_proof, uint num_rounds,
                                   vector<Polynomial>& proof) {
    for (uint i = 0; i < num_rounds; i++) {
        proof.push_back(Polynomial({ip_proof[3*i], ip_proof[3*i+1], ip_proof[3*i+2]}));
    }
    // Finals: a(u) and b(u)
    proof.push_back(Polynomial(ip_proof[3 * num_rounds]));
    proof.push_back(Polynomial(ip_proof[3 * num_rounds + 1]));
}

// ── Helpers ─────────────────────────────────────────────────────────────────

static FrTensor tensor_row(const FrTensor& mat, uint row_idx, uint row_size) {
    return mat.trunc((size_t)row_idx * row_size, (size_t)(row_idx + 1) * row_size);
}

static uint* upload_tokens(const vector<uint>& tokens) {
    uint* gpu;
    cudaMalloc(&gpu, tokens.size() * sizeof(uint));
    cudaMemcpy(gpu, tokens.data(), tokens.size() * sizeof(uint), cudaMemcpyHostToDevice);
    return gpu;
}

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

    Fr_t* v_star_cpu = new Fr_t[T];
    for (uint t = 0; t < T; t++) {
        FrTensor row = tensor_row(logits_all, t, V);
        uint t_star = find_argmax(row);
        v_star_cpu[t] = row(t_star);
    }
    FrTensor v_star_vec(T, v_star_cpu);
    delete[] v_star_cpu;

    FrTensor diffs_all(TV);
    batched_diffs_kernel<<<(TV + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        logits_all.gpu_data, v_star_vec.gpu_data, diffs_all.gpu_data, T, V);
    uint64_t cdf_max = (1ULL << cdf_precision) - 1;
    clamp_diffs_kernel<<<(TV + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        diffs_all.gpu_data, TV, cdf_max);

    auto [cdf_all, m_cdf] = cdf_prover.compute(diffs_all);
    (void)m_cdf;
    FrTensor win_probs_all = -(cdf_all - FR_FROM_INT(cdf_scale));

    FrTensor total_win_vec(T);
    batched_row_sum_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        win_probs_all.gpu_data, total_win_vec.gpu_data, T, V);
    cudaDeviceSynchronize();

    uint* tgpu = upload_tokens(tokens);
    FrTensor actual_wp_vec(T);
    extract_by_index_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        win_probs_all.gpu_data, tgpu, actual_wp_vec.gpu_data, T, V);
    cudaDeviceSynchronize();
    cudaFree(tgpu);

    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        actual_wp_vec.gpu_data, T);
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        total_win_vec.gpu_data, T);
    cudaDeviceSynchronize();

    uint table_size = 1u << log_precision;
    Fr_t* cpu_wp = new Fr_t[T];
    Fr_t* cpu_tw = new Fr_t[T];
    cudaMemcpy(cpu_wp, actual_wp_vec.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_tw, total_win_vec.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);

    uint D_padded = table_size * 2; // D must be > N for tLookup
    while (D_padded < T) D_padded *= 2;

    Fr_t* cpu_q = new Fr_t[D_padded];
    for (uint t = 0; t < T; t++) {
        unsigned long long wp_val = fr_to_ull(cpu_wp[t]);
        unsigned long long tw_val = fr_to_ull(cpu_tw[t]);
        unsigned long long q_val = (wp_val * (unsigned long long)table_size) / tw_val;
        cpu_q[t] = FR_FROM_INT(q_val);
    }
    for (uint i = T; i < D_padded; i++) cpu_q[i] = FR_FROM_INT(table_size);
    FrTensor q_padded(D_padded, cpu_q);
    delete[] cpu_wp;
    delete[] cpu_tw;
    delete[] cpu_q;

    auto [surprise_padded, m_surprise] = log_prover.compute(q_padded);
    (void)m_surprise;

    FrTensor surprise_vec = surprise_padded.trunc(0, T);
    return surprise_vec.sum();
}

// ── Batched prove ───────────────────────────────────────────────────────────

Fr_t zkConditionalEntropy::prove(
    const FrTensor& logits_all, uint T, uint V,
    const vector<uint>& tokens,
    Fr_t claimed_entropy,
    vector<Polynomial>& proof,
    vector<Claim>& claims)
{
    if (logits_all.size != T * V)
        throw std::invalid_argument("prove: logits_all.size != T * V");
    if (tokens.size() != T)
        throw std::invalid_argument("prove: tokens.size() != T");

    uint TV = T * V;

    // ════════════════════════════════════════════════════════════════════════
    // Phase 1: Compute all intermediate tensors
    // ════════════════════════════════════════════════════════════════════════

    Fr_t* v_star_cpu = new Fr_t[T];
    for (uint t = 0; t < T; t++) {
        FrTensor row = tensor_row(logits_all, t, V);
        uint t_star = find_argmax(row);
        v_star_cpu[t] = row(t_star);
    }
    FrTensor v_star_vec_t(T, v_star_cpu);
    delete[] v_star_cpu;

    FrTensor diffs_all(TV);
    batched_diffs_kernel<<<(TV + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        logits_all.gpu_data, v_star_vec_t.gpu_data, diffs_all.gpu_data, T, V);
    uint64_t cdf_max = (1ULL << cdf_precision) - 1;
    clamp_diffs_kernel<<<(TV + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        diffs_all.gpu_data, TV, cdf_max);

    auto [cdf_all, m_cdf] = cdf_prover.compute(diffs_all);
    FrTensor win_probs_all = -(cdf_all - FR_FROM_INT(cdf_scale));

    FrTensor total_win_vec(T);
    batched_row_sum_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        win_probs_all.gpu_data, total_win_vec.gpu_data, T, V);
    cudaDeviceSynchronize();

    uint* tgpu = upload_tokens(tokens);
    FrTensor actual_wp_raw(T);
    extract_by_index_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        win_probs_all.gpu_data, tgpu, actual_wp_raw.gpu_data, T, V);
    cudaDeviceSynchronize();
    cudaFree(tgpu);

    FrTensor actual_wp_vec(actual_wp_raw);
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        actual_wp_vec.gpu_data, T);
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        total_win_vec.gpu_data, T);
    cudaDeviceSynchronize();

    uint table_size = 1u << log_precision;
    Fr_t* cpu_wp = new Fr_t[T];
    Fr_t* cpu_tw = new Fr_t[T];
    cudaMemcpy(cpu_wp, actual_wp_vec.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_tw, total_win_vec.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);

    Fr_t* cpu_q = new Fr_t[T];
    Fr_t* cpu_r = new Fr_t[T];
    for (uint t = 0; t < T; t++) {
        unsigned long long wp_val = fr_to_ull(cpu_wp[t]);
        unsigned long long tw_val = fr_to_ull(cpu_tw[t]);
        unsigned long long wp_scaled = wp_val * (unsigned long long)table_size;
        unsigned long long q_val = wp_scaled / tw_val;
        unsigned long long r_val = wp_scaled - q_val * tw_val;
        cpu_q[t] = FR_FROM_INT(q_val);
        cpu_r[t] = FR_FROM_INT(r_val);
    }
    FrTensor q_vec(T, cpu_q);
    FrTensor r_vec(T, cpu_r);
    delete[] cpu_wp;
    delete[] cpu_tw;
    delete[] cpu_q;
    delete[] cpu_r;

    FrTensor wp_scaled_vec = actual_wp_vec * FR_FROM_INT(table_size);

    uint D_padded = table_size * 2; // D must be > N for tLookup
    while (D_padded < T) D_padded *= 2;

    Fr_t* cpu_q_pad = new Fr_t[D_padded];
    cudaMemcpy(cpu_q_pad, q_vec.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    for (uint i = T; i < D_padded; i++) cpu_q_pad[i] = FR_FROM_INT(table_size);
    FrTensor q_padded(D_padded, cpu_q_pad);
    delete[] cpu_q_pad;

    auto [surprise_padded, m_surprise] = log_prover.compute(q_padded);

    FrTensor surprise_vec = surprise_padded.trunc(0, T);
    Fr_t H = surprise_vec.sum();

    if (H != claimed_entropy)
        throw std::runtime_error("zkConditionalEntropy::prove: entropy mismatch");

    // ════════════════════════════════════════════════════════════════════════
    // Phase 2: Proofs
    // ════════════════════════════════════════════════════════════════════════

    // ── 2a. CDF tLookup proof ─────────────────────────────────────────────
    // Implicitly proves argmax: negative diffs cannot match CDF table entries.
    std::cout << "  Proving CDF lookup (" << TV << " elements)..." << std::endl;
    {
        uint cdf_N = 1u << cdf_precision;
        uint D_cdf = cdf_N * 2; // D must be > N for tLookup
        while (D_cdf < TV) D_cdf *= 2;

        Fr_t* d_cpu = new Fr_t[D_cdf];
        cudaMemcpy(d_cpu, diffs_all.gpu_data, TV * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        for (uint i = TV; i < D_cdf; i++) d_cpu[i] = FR_FROM_INT(0);
        FrTensor diffs_padded(D_cdf, d_cpu);
        delete[] d_cpu;

        auto [cdf_padded, m_cdf_padded] = cdf_prover.compute(diffs_padded);

        // Challenges via random_vec (dispatches to verifier in interactive mode)
        auto r_cdf = random_vec(1)[0];
        auto alpha = random_vec(1)[0];
        auto beta  = random_vec(1)[0];
        auto u_cdf = random_vec(ceilLog2(D_cdf));
        auto v_cdf = random_vec(ceilLog2(D_cdf));
        cdf_prover.prove(diffs_padded, cdf_padded, m_cdf_padded,
                         r_cdf, alpha, beta, u_cdf, v_cdf, proof);
    }

    // ── 2b. Diffs-to-logits linking ───────────────────────────────────────
    // Proves: MLE(diffs, u) + MLE(logits, u) = MLE(v_star_broadcast, u)
    // Returns a Claim on logits_all for upstream chaining.
    std::cout << "  Proving diffs-to-logits link..." << std::endl;
    {
        auto u_link = random_vec(ceilLog2(TV));

        uint log_V = ceilLog2(V);
        uint log_T = ceilLog2(TV) - log_V;
        vector<Fr_t> u_T(u_link.begin(), u_link.begin() + log_T);
        vector<Fr_t> u_V(u_link.begin() + log_T, u_link.end());

        Fr_t diffs_u  = diffs_all.pad({T, V})(u_link);
        Fr_t logits_u = logits_all.pad({T, V})(u_link);
        Fr_t vstar_u  = v_star_vec_t.pad({T})(u_T);

        // MLE(ones_V, u_V): verifier can compute this from public data
        Fr_t* ones_cpu = new Fr_t[1u << log_V];
        for (uint i = 0; i < V; i++) ones_cpu[i] = FR_FROM_INT(1);
        for (uint i = V; i < (1u << log_V); i++) ones_cpu[i] = FR_FROM_INT(0);
        FrTensor ones_V(1u << log_V, ones_cpu);
        delete[] ones_cpu;
        Fr_t ones_V_u = ones_V(u_V);

        // Emit evaluations for verifier
        proof.push_back(Polynomial(diffs_u));
        proof.push_back(Polynomial(logits_u));
        proof.push_back(Polynomial(vstar_u));
        proof.push_back(Polynomial(ones_V_u));

        // Verifier checks: diffs_u + logits_u == vstar_u * ones_V_u
        if (diffs_u + logits_u != vstar_u * ones_V_u)
            throw std::runtime_error("prove: diffs-to-logits link failed");

        // Return Claim on logits_all
        claims.push_back({logits_u,
                         vector<vector<Fr_t>>({u_T, u_V}),
                         vector<uint>({T, V})});
    }

    // ── 2c. total_win row-sum proof ────────────────────────────────────────
    std::cout << "  Proving total_win row sums..." << std::endl;
    {
        auto u_t = random_vec(ceilLog2(T));
        FrTensor wp_partial = win_probs_all.partial_me(u_t, V);

        Fr_t tw_claim = total_win_vec(u_t);
        Fr_t wp_partial_sum = wp_partial.sum();
        if (tw_claim != wp_partial_sum)
            throw std::runtime_error("prove: row-sum mismatch at challenge u_t");

        Fr_t* ones_cpu = new Fr_t[V];
        for (uint i = 0; i < V; i++) ones_cpu[i] = FR_FROM_INT(1);
        FrTensor ones_V(V, ones_cpu);
        delete[] ones_cpu;

        auto u_v = random_vec(ceilLog2(V));
        auto ip_rowsum = inner_product_sumcheck(wp_partial, ones_V, u_v);
        serialize_ip_sumcheck(ip_rowsum, ceilLog2(V), proof);
    }

    // ── 2d. Actual-token extraction proof ───────────────────────────────────
    std::cout << "  Proving actual-token extraction..." << std::endl;
    {
        Fr_t* ind_cpu = new Fr_t[TV];
        for (uint i = 0; i < TV; i++) ind_cpu[i] = FR_FROM_INT(0);
        for (uint t = 0; t < T; t++)
            ind_cpu[(size_t)t * V + tokens[t]] = FR_FROM_INT(1);
        FrTensor indicator(TV, ind_cpu);
        delete[] ind_cpu;

        Fr_t ip = (win_probs_all * indicator).sum();
        Fr_t wp_sum = actual_wp_raw.sum();
        if (ip != wp_sum)
            throw std::runtime_error("prove: indicator extraction mismatch");

        auto u_ext = random_vec(ceilLog2(TV));
        auto ip_extract = inner_product_sumcheck(win_probs_all, indicator, u_ext);
        serialize_ip_sumcheck(ip_extract, ceilLog2(TV), proof);

        auto u_T = random_vec(ceilLog2(T));
        Fr_t wp_at_u = actual_wp_raw(u_T);
        proof.push_back(Polynomial(wp_at_u));
    }

    // ── 2e. Quotient-remainder proof ───────────────────────────────────────
    std::cout << "  Proving quotient-remainder division..." << std::endl;
    {
        FrTensor q_tw_vec = q_vec * total_win_vec;

        auto u_qr = random_vec(ceilLog2(T));

        Fr_t q_tw_u = q_tw_vec(u_qr);
        Fr_t r_u  = r_vec(u_qr);
        Fr_t wp_scaled_u = wp_scaled_vec(u_qr);

        // Emit division relation for verifier
        proof.push_back(Polynomial(q_tw_u));
        proof.push_back(Polynomial(r_u));
        proof.push_back(Polynomial(wp_scaled_u));

        if (q_tw_u + r_u != wp_scaled_u)
            throw std::runtime_error("prove: division relation failed");

        uint q_bits = log_precision + 1;
        uint r_bits = 1;
        { unsigned long long max_tw = (unsigned long long)V * cdf_scale;
          while ((1ULL << r_bits) <= max_tw) r_bits++; }

        FrTensor gap = total_win_vec - r_vec - FR_FROM_INT(1);

        vector<FrTensor> bit_planes;
        FrTensor combined_error(T);
        cudaMemset(combined_error.gpu_data, 0, T * sizeof(Fr_t));
        uint batch_idx = 0;

        std::cout << "    q range proof (" << q_bits << " bits)..." << std::endl;
        prove_nonneg(q_vec, q_bits, u_qr, bit_planes, combined_error, batch_idx, proof);

        std::cout << "    r range proof (" << r_bits << " bits)..." << std::endl;
        prove_nonneg(r_vec, r_bits, u_qr, bit_planes, combined_error, batch_idx, proof);

        std::cout << "    gap range proof (" << r_bits << " bits)..." << std::endl;
        prove_nonneg(gap, r_bits, u_qr, bit_planes, combined_error, batch_idx, proof);

        Fr_t ce_u = combined_error(u_qr);
        proof.push_back(Polynomial(ce_u));
        if (ce_u != FR_FROM_INT(0))
            throw std::runtime_error("prove: batched binary check failed");
    }

    // ── 2f. Surprise log lookup proof ───────────────────────────────────────
    std::cout << "  Proving surprise log lookup..." << std::endl;
    {
        auto r_log = random_vec(1)[0];
        auto alpha = random_vec(1)[0];
        auto beta  = random_vec(1)[0];
        auto u_log = random_vec(ceilLog2(D_padded));
        auto v_log = random_vec(ceilLog2(D_padded));
        log_prover.prove(q_padded, surprise_padded, m_surprise,
                         r_log, alpha, beta, u_log, v_log, proof);
    }

    std::cout << "zkConditionalEntropy::prove complete (batched, "
              << T << " positions, " << proof.size() << " polynomials)."
              << std::endl;

    return H;
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
    vector<Polynomial>& proof,
    vector<Claim>& claims)
{
    if (logits_seq.empty())
        throw std::invalid_argument("prove: empty logits sequence");

    uint T = logits_seq.size();
    uint V = logits_seq[0].size;
    FrTensor logits_all = catTensors(logits_seq);
    return prove(logits_all, T, V, tokens, claimed_entropy, proof, claims);
}
