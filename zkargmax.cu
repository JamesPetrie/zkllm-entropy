#include "zkargmax.cuh"
#include <stdexcept>
#include <iostream>

zkArgmax::zkArgmax(uint bit_width) : bit_width(bit_width) {}

// ── Host-side helpers ─────────────────────────────────────────────────────────

// Read Fr_t as signed 64-bit integer (assumes value fits in lower 64 bits;
// negative numbers have val[2..7] non-zero and are treated as less than positives).
static bool fr_gt(const Fr_t& a, const Fr_t& b) {
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
    {
        Fr_t* cpu_diffs = new Fr_t[N];
        Fr_t* cpu_logits = new Fr_t[N];
        cudaMemcpy(cpu_diffs,  diffs.gpu_data,  N * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_logits, logits.gpu_data, N * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        unsigned long long max_diff_pos = 0;  // max among small positive diffs
        uint n_negative = 0;    // diffs with val[2..7] != 0 (genuinely negative)
        uint n_large64  = 0;    // diffs with val[1] bit-31 set but val[2..7] == 0 (>= 2^63)
        uint first_neg_idx = N;
        for (uint i = 0; i < N; i++) {
            const Fr_t& d = cpu_diffs[i];
            bool has_high = d.val[2]||d.val[3]||d.val[4]||d.val[5]||d.val[6]||d.val[7];
            if (has_high) {
                n_negative++;
                if (first_neg_idx == N) first_neg_idx = i;
            } else {
                unsigned long long dv = ((unsigned long long)d.val[1] << 32) | d.val[0];
                if (d.val[1] >> 31) n_large64++;
                if (dv > max_diff_pos) max_diff_pos = dv;
            }
        }
        cerr << "[zkArgmax] n_negative_diffs=" << n_negative << "  n_large64=" << n_large64
             << "  max_positive_diff=" << max_diff_pos << "  N=" << N << endl;
        if (n_negative > 0) {
            cerr << "[zkArgmax] NEGATIVE DIFFS: v_star < logits[i] for " << n_negative
                 << " elements (argmax bug?). First at i=" << first_neg_idx << endl;
            // Print first negative diff and corresponding logit
            const Fr_t& lg = cpu_logits[first_neg_idx];
            cerr << "  logit[" << first_neg_idx << "] = val[0..3]="
                 << lg.val[0] << "," << lg.val[1] << "," << lg.val[2] << "," << lg.val[3] << endl;
            cerr << "  v_star = val[0..3]="
                 << v_star.val[0] << "," << v_star.val[1] << "," << v_star.val[2] << "," << v_star.val[3] << endl;
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

    // 2. Bit-decompose each diff into bit_width bits.
    //    bits_vecs[b][i] = (diffs[i] >> b) & 1
    vector<FrTensor> bits_vecs;
    bits_vecs.reserve(bit_width);
    for (uint b = 0; b < bit_width; b++) {
        FrTensor bits_b(N);
        zkargmax_bit_extract_kernel<<<blocks, FrNumThread>>>(
            diffs.gpu_data, bits_b.gpu_data, b, N);
        cudaDeviceSynchronize();
        bits_vecs.push_back(bits_b);
    }

    // 3. Verify reconstruction at challenge point u:
    //    diffs(u) == sum_b 2^b * bits_b(u)
    Fr_t diffs_u = diffs(u);
    Fr_t recon   = {0, 0, 0, 0, 0, 0, 0, 0};
    Fr_t pow2    = {1, 0, 0, 0, 0, 0, 0, 0};
    Fr_t two     = {2, 0, 0, 0, 0, 0, 0, 0};
    for (uint b = 0; b < bit_width; b++) {
        Fr_t bits_b_u = bits_vecs[b](u);
        recon = recon + pow2 * bits_b_u;   // operator+ and operator* are host-accessible
        pow2  = pow2 * two;
    }
    if (recon != diffs_u)
        throw std::runtime_error("zkArgmax::prove: bit reconstruction mismatch at challenge u");

    // 4. Binary sumcheck for each bit tensor: proves bits_b[i] in {0,1}.
    for (uint b = 0; b < bit_width; b++) {
        auto v = random_vec(ceilLog2(N));
        vector<Fr_t> bin_proof;
        Fr_bin_sc(bits_vecs[b], u.begin(), u.end(), v.begin(), v.end(), bin_proof);
    }

    // 5. Return MLE claim on logits at u: logits(u) = v_star - diffs(u).
    return v_star - diffs_u;
}
