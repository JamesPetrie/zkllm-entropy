// FRI-based Polynomial Commitment Scheme (PCS).
//
// Provides a commitment interface similar to the existing Pedersen/KZG scheme
// but using FRI + Merkle trees. This is the integration layer that connects
// the FRI protocol to the rest of the proof system.
//
// Key operations:
// 1. Commit: hash a vector of field elements into a Merkle root (the commitment)
// 2. Open: prove that a committed vector, interpreted as a multilinear polynomial,
//    evaluates to a claimed value at a given point
// 3. Verify: check the opening proof
//
// The committed vector [a_0, ..., a_{n-1}] is treated as:
// - Coefficients of a univariate polynomial p(x) = sum_i a_i x^i for FRI
// - A multilinear polynomial f(x_0, ..., x_{k-1}) where n = 2^k for evaluation
//
// The multilinear evaluation f(u_0, ..., u_{k-1}) = sum_i a_i * eq(i, u)
// is proven by committing to the polynomial via FRI, then showing the
// evaluation claim is consistent with the committed polynomial.

#ifndef FRI_PCS_CUH
#define FRI_PCS_CUH

#include "fri.cuh"
#include <vector>

// ── FRI PCS commitment ───────────────────────────────────────────────────────

struct FriPcsCommitment {
    Hash256 root;          // Merkle root of the committed evaluations
    uint size;             // number of committed elements
    uint log_size;         // log2(size)
};

// ── FRI PCS opening proof ────────────────────────────────────────────────────

struct FriPcsOpeningProof {
    FriCommitment fri_commitment;
    FriProof fri_proof;
    std::vector<Fr_t> challenges;  // FRI folding challenges (from Fiat-Shamir)
    Fr_t claimed_value;            // the evaluation claim
};

// ── FRI PCS class ────────────────────────────────────────────────────────────

class FriPcs {
public:
    // Commit to a vector of field elements on GPU.
    // Returns the commitment (Merkle root).
    static FriPcsCommitment commit(const Fr_t* gpu_data, uint n);

    // Open the committed vector at a multilinear evaluation point.
    // data_gpu: the original committed data (device pointer)
    // n: number of elements
    // u: multilinear evaluation point (log2(n) coordinates)
    // Returns the claimed evaluation value and proof.
    static FriPcsOpeningProof open(const Fr_t* gpu_data, uint n,
                                   const std::vector<Fr_t>& u);

    // Verify an opening proof against a commitment.
    static bool verify(const FriPcsCommitment& commitment,
                       const std::vector<Fr_t>& u,
                       const FriPcsOpeningProof& proof);

    // Compute multilinear evaluation on host (for testing).
    // Evaluates sum_i data[i] * eq(i, u) where eq is the multilinear equality polynomial.
    static Fr_t multilinear_eval_host(const std::vector<Fr_t>& data,
                                       const std::vector<Fr_t>& u);
};

#endif // FRI_PCS_CUH
