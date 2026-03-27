#include "zkargmax.cuh"
#include <stdexcept>
#include <iostream>

zkArgmax::zkArgmax(uint bit_width) : bit_width(bit_width) {}

// ── Host-side helpers ─────────────────────────────────────────────────────────

// Read Fr_t as signed 64-bit integer (assumes value fits in lower 64 bits;
// negative numbers have val[2..7] non-zero and are treated as less than positives).
static bool fr_gt(const Fr_t& a, const Fr_t& b) {
#ifdef USE_GOLDILOCKS
    // In Goldilocks (p = 2^64 - 2^32 + 1), "negative" values (p-k for small k)
    // have val > p/2.
    bool a_neg = a.val > (GOLDILOCKS_P >> 1);
    bool b_neg = b.val > (GOLDILOCKS_P >> 1);
    if (a_neg != b_neg) return b_neg;
    return a.val > b.val;
#else
    // Check sign: a field element is "negative" (large mod-p value) when its
    // upper 192 bits (val[2..7]) are non-zero.
    bool a_neg = a.val[2] || a.val[3] || a.val[4] || a.val[5] || a.val[6] || a.val[7];
    bool b_neg = b.val[2] || b.val[3] || b.val[4] || b.val[5] || b.val[6] || b.val[7];
    if (a_neg != b_neg) return b_neg; // positive > negative
    unsigned long long av = ((unsigned long long)a.val[1] << 32) | a.val[0];
    unsigned long long bv = ((unsigned long long)b.val[1] << 32) | b.val[0];
    // Both positive: larger raw value is larger.
    // Both negative (p-k form): larger raw value means smaller k, i.e. less
    // negative, i.e. a greater signed value.  So av > bv in both cases.
    return av > bv;
#endif
}

uint zkArgmax::compute(const FrTensor& logits) {
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

// ── GPU kernels ───────────────────────────────────────────────────────────────

// diffs[i] = v_star - logits[i]
KERNEL void zkargmax_diffs_kernel(const Fr_t* logits, Fr_t v_star, Fr_t* diffs, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        diffs[tid] = blstrs__scalar__Scalar_sub(v_star, logits[tid]);
}

// bits_b[i] = (diffs[i] >> bit) & 1
KERNEL void zkargmax_bit_extract_kernel(const Fr_t* diffs, Fr_t* bits_b, uint bit, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        long val = scalar_to_long(diffs[tid]);
        bits_b[tid] = long_to_scalar((val >> bit) & 1L);
    }
}

// ── prove ─────────────────────────────────────────────────────────────────────

Fr_t zkArgmax::prove(const FrTensor& logits, uint t_star, Fr_t v_star,
                     const vector<Fr_t>& u, vector<Polynomial>& proof) {
    uint N = logits.size;
    if (u.size() != ceilLog2(N))
        throw std::invalid_argument("zkArgmax::prove: u.size() != ceilLog2(logits.size)");

    uint blocks = (N + FrNumThread - 1) / FrNumThread;

    // 1. Compute diffs[i] = v_star - logits[i] on GPU.
    FrTensor diffs(N);
    zkargmax_diffs_kernel<<<blocks, FrNumThread>>>(logits.gpu_data, v_star, diffs.gpu_data, N);
    cudaDeviceSynchronize();

    // 2. Diagnostic: check diff values and verify bit_width is sufficient.
    //    Gated behind ZKARGMAX_DIAGNOSTICS to avoid per-position GPU→CPU copies.
#ifdef ZKARGMAX_DIAGNOSTICS
    {
        Fr_t* cpu_diffs = new Fr_t[N];
        Fr_t* cpu_logits = new Fr_t[N];
        cudaMemcpy(cpu_diffs,  diffs.gpu_data,  N * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_logits, logits.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        unsigned long long max_diff_pos = 0;
        uint n_negative = 0, n_large64 = 0, first_neg_idx = N;
        for (uint i = 0; i < N; i++) {
            const Fr_t& d = cpu_diffs[i];
#ifdef USE_GOLDILOCKS
            bool is_neg = d.val > (GOLDILOCKS_P >> 1);
            if (is_neg) {
                n_negative++;
                if (first_neg_idx == N) first_neg_idx = i;
            } else {
                if (d.val > max_diff_pos) max_diff_pos = d.val;
                if (d.val >= (1ULL << 63)) n_large64++;
            }
#else
            bool has_high = d.val[2]||d.val[3]||d.val[4]||d.val[5]||d.val[6]||d.val[7];
            if (has_high) {
                n_negative++;
                if (first_neg_idx == N) first_neg_idx = i;
            } else {
                unsigned long long dv = ((unsigned long long)d.val[1] << 32) | d.val[0];
                if (d.val[1] >> 31) n_large64++;
                if (dv > max_diff_pos) max_diff_pos = dv;
            }
#endif
        }
        cerr << "[zkArgmax] n_negative_diffs=" << n_negative << "  n_large64=" << n_large64
             << "  max_positive_diff=" << max_diff_pos << "  N=" << N << endl;
        if (n_negative > 0) {
            cerr << "[zkArgmax] NEGATIVE DIFFS: v_star < logits[i] for " << n_negative
                 << " elements (argmax bug?). First at i=" << first_neg_idx << endl;
        }
        {
            uint needed = 0;
            unsigned long long tmp = max_diff_pos;
            while (tmp > 0) { needed++; tmp >>= 1; }
            cerr << "[zkArgmax] max_positive_diff needs " << needed << " bits (bit_width=" << bit_width << ")" << endl;
        }
        delete[] cpu_diffs;
        delete[] cpu_logits;
    }
#endif

    // 2. Bit-decompose each diff into bit_width bits.
    //    bits_vecs[b][i] = (diffs[i] >> b) & 1
    //    Launch all kernels, then sync once (avoids per-bit synchronization).
    vector<FrTensor> bits_vecs;
    bits_vecs.reserve(bit_width);
    for (uint b = 0; b < bit_width; b++) {
        bits_vecs.emplace_back(N);
        zkargmax_bit_extract_kernel<<<blocks, FrNumThread>>>(
            diffs.gpu_data, bits_vecs.back().gpu_data, b, N);
    }
    cudaDeviceSynchronize();

    // 3. Verify reconstruction at challenge point u:
    //    diffs(u) == sum_b 2^b * bits_b(u)
    Fr_t diffs_u = diffs(u);
    Fr_t recon   = FR_FROM_INT(0);
    Fr_t pow2    = FR_FROM_INT(1);
    Fr_t two     = FR_FROM_INT(2);
    for (uint b = 0; b < bit_width; b++) {
        Fr_t bits_b_u = bits_vecs[b](u);
        recon = recon + pow2 * bits_b_u;
        pow2  = pow2 * two;
    }
    if (recon != diffs_u)
        throw std::runtime_error("zkArgmax::prove: bit reconstruction mismatch at challenge u");

    // 4. Indicator vector: ind[t_star] = 1, all others 0.
    //    Proves v_star is actually in the logits tensor (soundness):
    //    - sum(ind) = 1: exactly one position selected
    //    - <ind, diffs> = 0: selected position has diff=0, so v_star = logits[t_star]
    //    - ind is binary: folded into the batched check below
    Fr_t* cpu_ind = new Fr_t[N];
    for (uint i = 0; i < N; i++) cpu_ind[i] = FR_FROM_INT(0);
    cpu_ind[t_star] = FR_FROM_INT(1);
    FrTensor ind(N, cpu_ind);
    delete[] cpu_ind;

    Fr_t ind_sum = ind.sum();
    if (ind_sum != FR_FROM_INT(1))
        throw std::runtime_error("zkArgmax::prove: indicator sum != 1");
    Fr_t ind_dot_diffs = (ind * diffs).sum();
    if (ind_dot_diffs != FR_FROM_INT(0))
        throw std::runtime_error("zkArgmax::prove: v_star not in logits tensor");

    // 5. Batched binary check via random linear combination.
    //    For K = bit_width + 1 tensors (bit planes + indicator), compute:
    //      combined_error[i] = sum_k r_k * a_k[i] * (a_k[i] - 1)
    //    If all a_k are binary, combined_error is the zero tensor.
    //    Verify by checking combined_error(u) = 0 (Schwartz-Zippel over |F| ~ 2^64).
    //    This replaces bit_width separate Fr_bin_sc calls (~30× fewer kernel launches).
    auto r = random_vec(bit_width + 1);
    FrTensor combined_error = (bits_vecs[0] * bits_vecs[0] - bits_vecs[0]) * r[0];
    for (uint b = 1; b < bit_width; b++) {
        combined_error += (bits_vecs[b] * bits_vecs[b] - bits_vecs[b]) * r[b];
    }
    combined_error += (ind * ind - ind) * r[bit_width];

    Fr_t ce_u = combined_error(u);
    if (ce_u != FR_FROM_INT(0))
        throw std::runtime_error("zkArgmax::prove: batched binary check failed");

    // 6. Add proof elements for indicator constraints.
    proof.push_back(Polynomial(ind_sum));
    proof.push_back(Polynomial(ind_dot_diffs));

    // 7. Return MLE claim on logits at u: logits(u) = v_star - diffs(u).
    return v_star - diffs_u;
}
