// FRI-based Polynomial Commitment Scheme (PCS).
//
// Provides a commitment interface replacing the Pedersen/KZG scheme,
// using Merkle trees for binding commitments and FRI for low-degree proofs.
//
// Architecture:
// - Commitment = Merkle root over field elements (binding, post-quantum).
// - Opening = multilinear evaluation verified against the commitment.
//
// Soundness model:
// The verifier has access to the committed data (same as the Pedersen case
// in this codebase, where verifyWeightClaim loads weights from disk).
// The Merkle root binds the data; the verifier recomputes MLE(data, u)
// and checks it matches the claim. Collision resistance of SHA-256
// ensures the prover can't swap data after committing.
//
// For a succinct third-party verifier (who doesn't have the data), a
// quotient polynomial evaluation argument would be needed on top of FRI.
// This is left for future work — the current interactive protocol doesn't
// need it.

#ifndef FRI_PCS_CUH
#define FRI_PCS_CUH

#include "commit/merkle.cuh"
#include <vector>
#include <string>

// ── FRI PCS commitment ───────────────────────────────────────────────────────

struct FriPcsCommitment {
    Hash256 root;          // Merkle root of the committed elements
    uint size;             // number of committed elements
    uint log_size;         // log2(size)

    void save(const std::string& filename) const;
    static FriPcsCommitment load(const std::string& filename);
};

// ── FRI PCS class ────────────────────────────────────────────────────────────

class FriPcs {
public:
    // Commit to a vector of field elements on GPU.
    // Returns the commitment (Merkle root).
    static FriPcsCommitment commit(const Fr_t* gpu_data, uint n);

    // Open and verify: compute MLE(data, u) and check data matches commitment.
    // Returns the evaluation value.
    // Throws if the Merkle root doesn't match the commitment.
    // If skip_binding_check is true, skips the Merkle tree rebuild
    // (use when the commitment was self-computed from the same data).
    static Fr_t open(const Fr_t* gpu_data, uint n,
                     const FriPcsCommitment& commitment,
                     const std::vector<Fr_t>& u,
                     bool skip_binding_check = false);

    // Compute multilinear evaluation on host.
    // Evaluates sum_i data[i] * eq(i, u) where eq is the multilinear equality polynomial.
    static Fr_t multilinear_eval_host(const std::vector<Fr_t>& data,
                                       const std::vector<Fr_t>& u);
};

#endif // FRI_PCS_CUH
