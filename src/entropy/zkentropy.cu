#include "entropy/zkentropy.cuh"
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

// ── Bit-extract kernel (same as zkargmax_bit_extract_kernel) ─────────────────
// bits_b[i] = (vals[i] >> bit) & 1
KERNEL void entropy_bit_extract_kernel(const Fr_t* vals, Fr_t* bits_b, uint bit, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        long val = scalar_to_long(vals[tid]);
        bits_b[tid] = long_to_scalar((val >> bit) & 1L);
    }
}

// ── Bit-decomposition non-negativity proof ──────────────────────────────────
// Proves that each element of `vals` lies in [0, 2^num_bits) by decomposing
// into `num_bits` bit-planes and verifying:
//   1. Reconstruction: sum_b 2^b * bits_b(u) == vals(u)
//   2. Each bit-plane is binary (accumulated into combined_error for batched check)
//
// bit_planes: output parameter; bit-planes are appended here
// combined_error: running batched binary check tensor (accumulated in-place)
// batch_idx: starting index for random coefficients in the batched check
static void prove_nonneg(const FrTensor& vals, uint num_bits,
                          const vector<Fr_t>& u,
                          vector<FrTensor>& bit_planes,
                          FrTensor& combined_error,
                          uint& batch_idx) {
    uint N = vals.size;
    uint blocks = (N + FrNumThread - 1) / FrNumThread;

    // 1. Extract bit planes
    uint base = bit_planes.size();
    for (uint b = 0; b < num_bits; b++) {
        bit_planes.emplace_back(N);
        entropy_bit_extract_kernel<<<blocks, FrNumThread>>>(
            vals.gpu_data, bit_planes.back().gpu_data, b, N);
    }
    cudaDeviceSynchronize();

    // 2. Reconstruction check at challenge u:
    //    vals(u) == sum_b 2^b * bits_b(u)
    Fr_t vals_u = vals(u);
    Fr_t recon  = FR_FROM_INT(0);
    Fr_t pow2   = FR_FROM_INT(1);
    Fr_t two    = FR_FROM_INT(2);
    for (uint b = 0; b < num_bits; b++) {
        Fr_t bits_b_u = bit_planes[base + b](u);
        recon = recon + pow2 * bits_b_u;
        pow2  = pow2 * two;
    }
    if (recon != vals_u)
        throw std::runtime_error("prove_nonneg: bit reconstruction mismatch");

    // 3. Accumulate binary check: r_k * bits_b * (bits_b - 1) for each bit plane
    auto r = random_vec(num_bits);
    for (uint b = 0; b < num_bits; b++) {
        combined_error += (bit_planes[base + b] * bit_planes[base + b]
                           - bit_planes[base + b]) * r[b];
    }
    batch_idx += num_bits;
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
//
// Quotient-remainder approach:
//   q[t] = floor(wp[t] * 2^log_precision / tw[t])   ∈ [1, 2^log_precision]
//   surprise[t] = log_table(q[t])
//               = (log_precision - log2(q[t])) * log_scale
//               ≈ -log2(wp[t]/tw[t]) * log_scale
//
// ZK proof of the division:
//   q[t]*tw[t] + r[t] = wp[t]*2^log_precision   where 0 ≤ r[t] < tw[t]
//   Non-negativity of q, r, (tw-r-1) via bit decomposition
//   H = sum(surprise)

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

    // 6. Clamp win probs and total_win to >= 1
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        actual_wp_vec.gpu_data, T);
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        total_win_vec.gpu_data, T);
    cudaDeviceSynchronize();

    // 7. Quotient: q[t] = floor(wp[t] * 2^log_precision / tw[t])
    //    Requires 2^log_precision >= max(total_win) for q >= 1 on all positions.
    //    In practice, log_precision >= ceil(log2(V * cdf_scale)).
    uint table_size = 1u << log_precision;
    Fr_t* cpu_wp = new Fr_t[T];
    Fr_t* cpu_tw = new Fr_t[T];
    cudaMemcpy(cpu_wp, actual_wp_vec.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_tw, total_win_vec.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);

    // Pad q to satisfy tLookup D%N==0: D_padded = smallest power of 2 >= max(T, N)
    uint D_padded = table_size;
    while (D_padded < T) D_padded *= 2;

    Fr_t* cpu_q = new Fr_t[D_padded];
    for (uint t = 0; t < T; t++) {
        unsigned long long wp_val = fr_to_ull(cpu_wp[t]);
        unsigned long long tw_val = fr_to_ull(cpu_tw[t]);
        unsigned long long q_val = (wp_val * (unsigned long long)table_size) / tw_val;
        cpu_q[t] = FR_FROM_INT(q_val);
    }
    // Pad with table_size (maps to surprise=0 in log table)
    for (uint i = T; i < D_padded; i++) cpu_q[i] = FR_FROM_INT(table_size);
    FrTensor q_padded(D_padded, cpu_q);
    delete[] cpu_wp;
    delete[] cpu_tw;
    delete[] cpu_q;

    // 8. Surprise lookup
    auto [surprise_padded, m_surprise] = log_prover.compute(q_padded);
    (void)m_surprise;

    // 9. Entropy = sum of surprise for the first T positions only
    //    (padded positions have surprise=0 by construction)
    FrTensor surprise_vec = surprise_padded.trunc(0, T);
    return surprise_vec.sum();
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

    // 6. Clamp win probs and total_win to >= 1
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        actual_wp_vec.gpu_data, T);
    clamp_min_one_kernel<<<(T + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        total_win_vec.gpu_data, T);
    cudaDeviceSynchronize();

    // 7. Quotient-remainder: q[t] = floor(wp[t] * 2^p / tw[t])
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

    // wp_scaled = actual_wp * 2^p  (on GPU, for the division relation proof)
    FrTensor wp_scaled_vec = actual_wp_vec * FR_FROM_INT(table_size);

    // 8. Surprise lookup (padded for tLookup D%N==0 constraint)
    uint D_padded = table_size;
    while (D_padded < T) D_padded *= 2;

    Fr_t* cpu_q_pad = new Fr_t[D_padded];
    cudaMemcpy(cpu_q_pad, q_vec.gpu_data, T * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    for (uint i = T; i < D_padded; i++) cpu_q_pad[i] = FR_FROM_INT(table_size);
    FrTensor q_padded(D_padded, cpu_q_pad);
    delete[] cpu_q_pad;

    auto [surprise_padded, m_surprise] = log_prover.compute(q_padded);

    // 9. Entropy = sum of first T surprise values
    FrTensor surprise_vec = surprise_padded.trunc(0, T);
    Fr_t H = surprise_vec.sum();

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
    // Pads to satisfy D % N == 0 constraint (N = 2^cdf_precision).
    std::cout << "  Proving CDF lookup (" << TV << " elements)..." << std::endl;
    {
        uint cdf_N = 1u << cdf_precision;
        uint D_cdf = cdf_N;
        while (D_cdf < TV) D_cdf *= 2;

        // Pad diffs with 0 (valid table input) and re-compute CDF for consistency
        Fr_t* d_cpu = new Fr_t[D_cdf];
        cudaMemcpy(d_cpu, diffs_all.gpu_data, TV * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        for (uint i = TV; i < D_cdf; i++) d_cpu[i] = FR_FROM_INT(0);
        FrTensor diffs_padded(D_cdf, d_cpu);
        delete[] d_cpu;

        auto [cdf_padded, m_cdf_padded] = cdf_prover.compute(diffs_padded);

        auto r_cdf = random_vec(1)[0];
        auto alpha = random_vec(1)[0];
        auto beta  = random_vec(1)[0];
        auto u_cdf = random_vec(ceilLog2(D_cdf));
        auto v_cdf = random_vec(ceilLog2(D_cdf));
        cdf_prover.prove(diffs_padded, cdf_padded, m_cdf_padded,
                         r_cdf, alpha, beta, u_cdf, v_cdf, proof);
    }

    // ── 2c. total_win row-sum proof ────────────────────────────────────────
    // Proves total_win_vec[t] = sum_i win_probs_all[t*V + i] via
    // partial MLE + inner product sumcheck with all-ones vector.
    std::cout << "  Proving total_win row sums..." << std::endl;
    {
        // Fix T dimension at random challenge u_t, producing a V-length tensor
        auto u_t = random_vec(ceilLog2(T));
        FrTensor wp_partial = win_probs_all.partial_me(u_t, V);

        // Claim: sum(wp_partial) == total_win_vec(u_t)
        Fr_t tw_claim = total_win_vec(u_t);
        Fr_t wp_partial_sum = wp_partial.sum();
        if (tw_claim != wp_partial_sum)
            throw std::runtime_error("prove: row-sum mismatch at challenge u_t");

        // Prove the sum via inner product with ones
        Fr_t* ones_cpu = new Fr_t[V];
        for (uint i = 0; i < V; i++) ones_cpu[i] = FR_FROM_INT(1);
        FrTensor ones_V(V, ones_cpu);
        delete[] ones_cpu;

        auto u_v = random_vec(ceilLog2(V));
        inner_product_sumcheck(wp_partial, ones_V, u_v);
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

    // ── 2e. Quotient-remainder proof ───────────────────────────────────────
    // Proves surprise[t] = log_table(q[t]) where q[t] = floor(wp[t]*2^p/tw[t]).
    // Division relation: q*tw + r = wp*2^p, with 0 <= q < 2^p, 0 <= r < tw.
    std::cout << "  Proving quotient-remainder division..." << std::endl;
    {
        auto u_qr = random_vec(ceilLog2(T));

        // 1. Division relation at random point: q(u)*tw(u) + r(u) = wp_scaled(u)
        Fr_t q_u  = q_vec(u_qr);
        Fr_t tw_u = total_win_vec(u_qr);
        Fr_t r_u  = r_vec(u_qr);
        Fr_t wp_scaled_u = wp_scaled_vec(u_qr);
        if (q_u * tw_u + r_u != wp_scaled_u)
            throw std::runtime_error("prove: division relation failed at challenge u");

        // 2. Non-negativity via bit decomposition
        //    - q in [0, 2^(log_precision+1)): log_precision+1 bits (q can equal 2^p)
        //    - r >= 0: r_bits bits (r < tw <= V * cdf_scale)
        //    - tw - r - 1 >= 0 (i.e. r < tw): r_bits bits
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
        prove_nonneg(q_vec, q_bits, u_qr, bit_planes, combined_error, batch_idx);

        std::cout << "    r range proof (" << r_bits << " bits)..." << std::endl;
        prove_nonneg(r_vec, r_bits, u_qr, bit_planes, combined_error, batch_idx);

        std::cout << "    gap range proof (" << r_bits << " bits)..." << std::endl;
        prove_nonneg(gap, r_bits, u_qr, bit_planes, combined_error, batch_idx);

        // 3. Batched binary check: combined_error(u) == 0
        Fr_t ce_u = combined_error(u_qr);
        if (ce_u != FR_FROM_INT(0))
            throw std::runtime_error("prove: batched binary check failed for q/r/gap");
    }

    // ── 2f. Surprise log lookup proof ───────────────────────────────────────
    // Proves surprise[t] = log_table(q[t]) via tLookup on the padded q tensor.
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
