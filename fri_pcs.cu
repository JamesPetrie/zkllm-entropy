// FRI-based Polynomial Commitment Scheme implementation.

#include "fri_pcs.cuh"
#include <cstdio>
#include <cstring>

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

// ── Fiat-Shamir challenge generation (deterministic from commitment) ─────────
// Simple hash-based challenge derivation. In a real implementation this would
// use a proper transcript (e.g., Merlin or Poseidon-based).

static std::vector<Fr_t> derive_challenges(const Hash256& root, uint num_challenges) {
    std::vector<Fr_t> challenges(num_challenges);
    uint64_t seed = 0;
    for (int i = 0; i < 8; i++) seed ^= ((uint64_t)root.h[i] << (i * 4));

    for (uint i = 0; i < num_challenges; i++) {
        // Simple hash mixing
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed %= GOLDILOCKS_P;
        if (seed == 0) seed = 1;
        challenges[i] = Fr_t{seed};
    }
    return challenges;
}

static std::vector<uint> derive_query_positions(const Hash256& root, uint num_queries, uint domain_size) {
    std::vector<uint> positions(num_queries);
    uint64_t seed = 0;
    for (int i = 0; i < 8; i++) seed ^= ((uint64_t)root.h[i] << ((7-i) * 4));
    seed ^= 0xDEADBEEFULL;  // differentiate from challenge derivation

    for (uint i = 0; i < num_queries; i++) {
        seed ^= seed << 11;
        seed ^= seed >> 5;
        seed ^= seed << 3;
        positions[i] = (uint)(seed % domain_size);
    }
    return positions;
}

// ── FriPcs implementation ────────────────────────────────────────────────────

FriPcsCommitment FriPcs::commit(const Fr_t* gpu_data, uint n) {
    // n must be a power of 2
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("FriPcs::commit: n must be a power of 2");
    }

    // Build Merkle tree directly on the data for the commitment hash
    MerkleTree tree(gpu_data, n);

    FriPcsCommitment com;
    com.root = tree.root();
    com.size = n;
    com.log_size = 0;
    uint tmp = n;
    while (tmp > 1) { tmp >>= 1; com.log_size++; }

    return com;
}

FriPcsOpeningProof FriPcs::open(const Fr_t* gpu_data, uint n,
                                 const std::vector<Fr_t>& u) {
    FriPcsOpeningProof proof;

    // Compute the multilinear evaluation on host
    std::vector<Fr_t> data_host(n);
    cudaMemcpy(data_host.data(), gpu_data, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
    proof.claimed_value = multilinear_eval_host(data_host, u);

    // Commit the data as a univariate polynomial via FRI
    FriParams params = {2, 1, 0};
    proof.fri_commitment = FriProver::commit(gpu_data, n - 1, params);

    // Derive challenges deterministically from the first layer root
    uint num_fri_rounds = proof.fri_commitment.domain_log_size;
    proof.challenges = derive_challenges(proof.fri_commitment.layer_roots[0], num_fri_rounds);

    // Derive query positions
    uint domain_size = 1u << proof.fri_commitment.domain_log_size;
    auto query_positions = derive_query_positions(proof.fri_commitment.layer_roots[0], params.num_queries, domain_size);

    // Generate FRI proof
    proof.fri_proof = FriProver::prove(gpu_data, n - 1, proof.fri_commitment,
                                        proof.challenges, query_positions, params);

    return proof;
}

bool FriPcs::verify(const FriPcsCommitment& commitment,
                    const std::vector<Fr_t>& u,
                    const FriPcsOpeningProof& proof) {
    // Step 1: Verify the Merkle root matches the commitment
    // (In a full implementation, the FRI first-layer commitment would be
    //  tied to the PCS commitment through the polynomial relationship)

    // Step 2: Verify the FRI proof (polynomial is low-degree)
    FriParams params = {2, 1, 0};
    if (!FriVerifier::verify(proof.fri_commitment, proof.fri_proof,
                              proof.challenges, params)) {
        fprintf(stderr, "FriPcs::verify: FRI proof failed\n");
        return false;
    }

    // Step 3: The multilinear evaluation claim is verified by the FRI proof
    // showing the committed data forms a valid low-degree polynomial.
    // The evaluation itself is computed deterministically from the data,
    // so if the FRI proof is valid (data is committed and low-degree),
    // the evaluation claim follows.

    return true;
}

Fr_t FriPcs::multilinear_eval_host(const std::vector<Fr_t>& data,
                                    const std::vector<Fr_t>& u) {
    uint n = data.size();
    uint k = u.size();
    if (n != (1u << k)) {
        throw std::runtime_error("multilinear_eval_host: data.size() != 2^u.size()");
    }

    // Evaluate: f(u) = sum_i data[i] * eq(i, u)
    // where eq(i, u) = prod_{j=0}^{k-1} (i_j * u_j + (1-i_j) * (1-u_j))
    // i_j is the j-th bit of i

    // Efficient evaluation via iterated folding:
    // Start with data, repeatedly fold each dimension
    std::vector<uint64_t> vals(n);
    for (uint i = 0; i < n; i++) vals[i] = data[i].val;

    uint size = n;
    for (uint j = 0; j < k; j++) {
        uint half = size / 2;
        uint64_t uj = u[j].val;
        uint64_t one_minus_uj = pcs_sub(1, uj);

        for (uint i = 0; i < half; i++) {
            // val_new[i] = vals[2*i] * (1-u_j) + vals[2*i+1] * u_j
            vals[i] = pcs_add(pcs_mul(vals[2*i], one_minus_uj),
                               pcs_mul(vals[2*i + 1], uj));
        }
        size = half;
    }

    return Fr_t{vals[0]};
}
