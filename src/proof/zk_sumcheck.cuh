// zk_sumcheck.cuh — Zero-knowledge sumcheck prover functions
//
// ZK variants of the existing sumcheck provers. These add:
// 1. Vanishing polynomial masking (degree 2 per variable → degree 4 product)
// 2. XZZ+19 transcript masking (g + rho * p)
//
// The existing provers (zkip, inner_product_sumcheck, etc.) are untouched.
// These _zk variants produce 5 evaluations per round instead of 3.

#ifndef ZK_SUMCHECK_CUH
#define ZK_SUMCHECK_CUH

#include "proof/zk_mask.cuh"
#include "poly/polynomial.cuh"
#include "tensor/fr-tensor.cuh"
#include <vector>

// ── GPU kernel for degree-4 masked inner product ────────────────────────────
// Evaluates Za(t) * Zb(t) at t = 0,1,2,3,4 where:
//   Za(t) = a0 + (a1-a0)*t + c_a * t*(1-t)
//   Zb(t) = b0 + (b1-b0)*t + c_b * t*(1-t)
KERNEL void zkip_zk_poly_kernel(
    GLOBAL Fr_t *a, GLOBAL Fr_t *b,
    GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2,
    GLOBAL Fr_t *out3, GLOBAL Fr_t *out4,
    Fr_t c_a, Fr_t c_b,
    uint N_in, uint N_out);

// ── ZK inner product sumcheck (Polynomial output format) ────────────────────
// Proves: <a, b> = claim, with vanishing polynomial masking on a and/or b,
// and XZZ+19 transcript masking.
//
// Returns the final reduced claim. The proof vector receives one Polynomial
// per round (degree 4, stored as 5 coefficients).
//
// After all rounds, the prover appends:
//   proof.push_back(Polynomial(final_za));  // Z_a(s*)
//   proof.push_back(Polynomial(final_zb));  // Z_b(s*)
//   proof.push_back(Polynomial(p_final));   // p(s*)
//
// The verifier checks: final_za * final_zb + rho * p_final == last reduced claim

struct ZkIpResult {
    Fr_t final_za;   // masked a evaluation at terminal point
    Fr_t final_zb;   // masked b evaluation at terminal point
    Fr_t p_final;    // transcript masking polynomial at terminal point
};

Fr_t zkip_zk(
    const Fr_t& claim,
    const FrTensor& a, const FrTensor& b,
    const std::vector<Fr_t>& u,
    const ZkMaskConfig& mask_a, const ZkMaskConfig& mask_b,
    const ZkTranscriptMask& tmask, Fr_t rho,
    std::vector<Polynomial>& proof,
    ZkIpResult& result);

// ── ZK inner product sumcheck (flat Fr_t output, for zkentropy compat) ──────
// Same protocol but outputs to vector<Fr_t> in the format:
//   [5 evals per round] x num_rounds + [final_za, final_zb, p_final]
std::vector<Fr_t> inner_product_sumcheck_zk(
    const FrTensor& a, const FrTensor& b,
    std::vector<Fr_t> u,
    const ZkMaskConfig& mask_a, const ZkMaskConfig& mask_b,
    const ZkTranscriptMask& tmask, Fr_t rho);

// ── ZK stacked inner product sumcheck ──────────────────────────────────────
// Degree-4 masked version of zkip_stacked. Operates on A (N×D), B (N×D)
// tensors with separate challenge vectors for N and D dimensions.
// Falls back to zkip_zk when the N dimension is exhausted.
Fr_t zkip_stacked_zk(
    const Fr_t& claim,
    const FrTensor& A, const FrTensor& B,
    const std::vector<Fr_t>& uN, const std::vector<Fr_t>& uD,
    const std::vector<Fr_t> vN, uint N, uint D,
    const ZkMaskConfig& mask_a, const ZkMaskConfig& mask_b,
    const ZkTranscriptMask& tmask, Fr_t rho,
    std::vector<Polynomial>& proof,
    ZkIpResult& result);

#endif // ZK_SUMCHECK_CUH
