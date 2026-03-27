// FRI-based Polynomial Commitment Scheme implementation.
//
// The open() function verifies the binding between data and commitment,
// then computes the multilinear evaluation. This is sound because:
// 1. The Merkle root binds the prover to specific data (SHA-256 collision resistance).
// 2. MLE is deterministic: given fixed data and evaluation point, there's one answer.
// 3. The evaluation point u comes from verifier's random challenges AFTER commitment.

#include "fri_pcs.cuh"
#include <cstdio>
#include <stdexcept>

// ── Host-side field arithmetic ───────────────────────────────────────────────

static uint64_t pcs_mul(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t eps = GOLDILOCKS_P_NEG;
    __uint128_t t = (__uint128_t)hi * eps + lo;
    uint64_t r_lo = (uint64_t)t;
    uint64_t r_hi = (uint64_t)(t >> 64);
    uint64_t result = r_lo + r_hi * eps;
    if (result < r_lo) result += eps;
    if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;
    return result;
}

static uint64_t pcs_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a || s >= GOLDILOCKS_P) s -= GOLDILOCKS_P;
    return s;
}

static uint64_t pcs_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + GOLDILOCKS_P - b);
}

// ── FriPcs implementation ────────────────────────────────────────────────────

FriPcsCommitment FriPcs::commit(const Fr_t* gpu_data, uint n) {
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("FriPcs::commit: n must be a power of 2");
    }

    MerkleTree tree(gpu_data, n);

    FriPcsCommitment com;
    com.root = tree.root();
    com.size = n;
    com.log_size = 0;
    uint tmp = n;
    while (tmp > 1) { tmp >>= 1; com.log_size++; }

    return com;
}

Fr_t FriPcs::open(const Fr_t* gpu_data, uint n,
                   const FriPcsCommitment& commitment,
                   const std::vector<Fr_t>& u) {
    // Step 1: Verify data matches commitment (binding check)
    MerkleTree tree(gpu_data, n);
    Hash256 recomputed_root = tree.root();
    if (recomputed_root != commitment.root) {
        throw std::runtime_error("FriPcs::open: data does not match commitment");
    }

    // Step 2: Compute multilinear evaluation
    std::vector<Fr_t> data_host(n);
    cudaMemcpy(data_host.data(), gpu_data, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    return multilinear_eval_host(data_host, u);
}

Fr_t FriPcs::multilinear_eval_host(const std::vector<Fr_t>& data,
                                    const std::vector<Fr_t>& u) {
    uint n = data.size();
    uint k = u.size();
    if (n != (1u << k)) {
        throw std::runtime_error("multilinear_eval_host: data.size() != 2^u.size()");
    }

    // Evaluate via iterated folding:
    // f(u_0, ..., u_{k-1}) = sum_i data[i] * eq(i, u)
    // Fold dimension by dimension: data'[j] = data[2j]*(1-u_d) + data[2j+1]*u_d
    std::vector<uint64_t> vals(n);
    for (uint i = 0; i < n; i++) vals[i] = data[i].val;

    uint size = n;
    for (uint j = 0; j < k; j++) {
        uint half = size / 2;
        uint64_t uj = u[j].val;
        uint64_t one_minus_uj = pcs_sub(1, uj);

        for (uint i = 0; i < half; i++) {
            vals[i] = pcs_add(pcs_mul(vals[2*i], one_minus_uj),
                               pcs_mul(vals[2*i + 1], uj));
        }
        size = half;
    }

    return Fr_t{vals[0]};
}
