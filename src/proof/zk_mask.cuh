// zk_mask.cuh — Zero-knowledge masking types and helpers
//
// Implements two ZK masking layers for sumcheck proofs:
// 1. Vanishing polynomial masking: Z_f(X) = f(X) + sum_i c_i * X_i(1 - X_i)
//    Hides polynomial evaluations at non-boolean points while preserving
//    sumcheck correctness (vanishing terms are zero on {0,1}^k).
// 2. Transcript masking (XZZ+19): p(X) = a_0 + sum_i p_i(X_i)
//    Hides round polynomials by adding rho * p to each round.

#ifndef ZK_MASK_CUH
#define ZK_MASK_CUH

#include "tensor/fr-tensor.cuh"
#include "poly/polynomial.cuh"
#include <vector>

// ── Vanishing polynomial masking config ─────────────────────────────────────
// For a polynomial with k sumcheck variables opened at m points,
// we need k > m masking coefficients for ZK.

struct ZkMaskConfig {
    bool enabled = false;
    std::vector<Fr_t> vanishing_coeffs;  // c_1..c_k, one per sumcheck variable
};

// Generate a random vanishing polynomial masking config.
// num_vars: number of sumcheck variables (k)
ZkMaskConfig generate_vanishing_mask(uint num_vars);

// Evaluate the vanishing correction at a point:
//   sum_{i=0}^{k-1} c_i * point[i] * (1 - point[i])
// Returns 0 for any boolean point (since x*(1-x) = 0 for x in {0,1}).
// The point vector must have at least coeffs.size() elements.
Fr_t vanishing_correction(const std::vector<Fr_t>& coeffs,
                          const std::vector<Fr_t>& point);

// ── Sumcheck transcript masking (XZZ+19 g+rho*p) ───────────────────────────
// Random polynomial p(X_1,...,X_b) = a_0 + sum_{i=1}^{b} p_i(X_i)
// where each p_i is a univariate of degree d (matching the round polynomial degree).

struct ZkTranscriptMask {
    Fr_t a0;                                    // constant term
    std::vector<std::vector<Fr_t>> p_univariates;  // p_univariates[i] = [coeff_1, ..., coeff_d] for p_i(X)
    Fr_t P_sum;                                 // precomputed: sum_{c in {0,1}^b} p(c)
    uint degree;                                // degree of each p_i
};

// Generate a random transcript masking polynomial.
// num_vars: number of sumcheck rounds (b)
// degree: degree of the round polynomial (typically 4 for masked sumcheck)
ZkTranscriptMask generate_transcript_mask(uint num_vars, uint degree);

// Evaluate the full masking polynomial p at a point:
//   p(point) = a_0 + sum_{i=0}^{b-1} p_i(point[i])
// where p_i(X) = sum_{k=1}^{d} coeff[k-1] * X^k
Fr_t eval_transcript_mask(const ZkTranscriptMask& mask,
                          const std::vector<Fr_t>& point);

// Get the univariate masking polynomial contribution for round j,
// given already-bound challenges alpha_0..alpha_{j-1}.
// Returns a Polynomial in X_j that equals:
//   (contribution from a_0 at this round) + p_j(X_j) + (sum of already-evaluated p_i(alpha_i))
// This is the term to scale by rho and add to the honest round polynomial.
//
// For round j, the masking polynomial p evaluated over the remaining variables is:
//   p(alpha_0,..,alpha_{j-1}, X_j, X_{j+1},..,X_{b-1})
// Summing over X_{j+1}..X_{b-1} in {0,1}^{b-j-1} gives:
//   2^{b-j-1} * [a_0 + sum_{i<j} p_i(alpha_i) + p_j(X_j)] + 2^{b-j-2} * sum_{i>j} (p_i(0) + p_i(1))
Polynomial transcript_mask_round_poly(const ZkTranscriptMask& mask,
                                      uint round_idx,
                                      const std::vector<Fr_t>& bound_challenges,
                                      uint total_rounds);

#endif // ZK_MASK_CUH
