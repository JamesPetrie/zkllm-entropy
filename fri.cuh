// FRI (Fast Reed-Solomon Interactive Oracle Proof) polynomial commitment.
//
// Implements the FRI protocol for proving that a committed function is
// close to a low-degree polynomial. Used as the polynomial commitment
// scheme in our proof system (replacing Pedersen/KZG).
//
// Protocol overview:
// 1. Commit phase: evaluate polynomial on a coset domain, build Merkle tree
// 2. Folding phase: prover folds polynomial with verifier challenges, commits each layer
// 3. Query phase: verifier picks random positions, prover opens those positions
//    with Merkle proofs across all layers
// 4. Verify: check folding consistency and Merkle proofs
//
// Parameters:
// - blowup_factor: ratio of evaluation domain to polynomial degree (typically 2-8)
// - num_queries: number of query positions (determines soundness)
// - For interactive proofs with 64-bit field: ~1 query suffices for 2^-64 soundness

#ifndef FRI_CUH
#define FRI_CUH

#include "ntt.cuh"
#include "merkle.cuh"
#include <vector>
#include <memory>

// ── FRI parameters ───────────────────────────────────────────────────────────

struct FriParams {
    uint blowup_factor;   // domain size / poly degree (default: 2)
    uint num_queries;     // number of query positions (default: 1 for interactive)
    uint max_remainder_degree;  // stop folding when degree <= this (default: 0)
};

static const FriParams FRI_DEFAULT_PARAMS = {2, 1, 0};

// ── FRI commitment (output of commit phase) ──────────────────────────────────

struct FriCommitment {
    std::vector<Hash256> layer_roots;   // Merkle roots for each folding layer
    std::vector<Fr_t> remainder;        // final low-degree polynomial (coefficients)
    uint poly_degree;                   // degree of the committed polynomial
    uint domain_log_size;               // log2 of the evaluation domain
    Fr_t domain_offset;                 // coset shift
};

// ── FRI proof (output of query phase) ────────────────────────────────────────

struct FriQueryRound {
    Fr_t value;           // evaluation at the queried position
    Fr_t sibling_value;   // evaluation at the paired position
    MerkleProof proof;    // Merkle proof for the queried position
    MerkleProof sibling_proof;  // Merkle proof for the paired position
};

struct FriProof {
    std::vector<std::vector<FriQueryRound>> queries;  // [query_idx][round]
    std::vector<uint> query_positions;                 // initial query positions
};

// ── FRI prover ───────────────────────────────────────────────────────────────

class FriProver {
public:
    // Commit to a polynomial given as coefficients on GPU.
    // coeffs_gpu: device pointer to (degree+1) Fr_t coefficients
    // degree: polynomial degree
    // params: FRI parameters
    static FriCommitment commit(const Fr_t* coeffs_gpu, uint degree, const FriParams& params = FRI_DEFAULT_PARAMS);

    // Generate proof for given query positions.
    // Returns a FriProof that can be verified against the commitment.
    static FriProof prove(const Fr_t* coeffs_gpu, uint degree,
                          const FriCommitment& commitment,
                          const std::vector<Fr_t>& challenges,
                          const std::vector<uint>& query_positions,
                          const FriParams& params = FRI_DEFAULT_PARAMS);
};

// ── FRI verifier ─────────────────────────────────────────────────────────────

class FriVerifier {
public:
    // Verify a FRI proof against a commitment.
    static bool verify(const FriCommitment& commitment,
                       const FriProof& proof,
                       const std::vector<Fr_t>& challenges,
                       const FriParams& params = FRI_DEFAULT_PARAMS);
};

#endif // FRI_CUH
