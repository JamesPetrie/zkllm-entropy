#include "proof/zk_round_commit.cuh"
#include "commit/commitment.cuh"
#include "poly/polynomial.cuh"

// Phase 3 subcomponent (2): commit-then-fold helpers for sumcheck
// round polynomials.  See zk_round_commit.cuh for the design rationale.
//
// All arithmetic is done on the host via existing operator overloads
// from src/poly/polynomial.cu (Fr_t +, -, *) and via Commitment-based
// size-1 wrappers for G1 ops (same pattern as src/proof/hyrax_sigma.cu
// and src/commit/commitment.cu).  No new device kernels are introduced;
// the per-round Σ-protocol work is small and CPU-bound.

static G1Jacobian_t g1_scalar_mul_host(G1Jacobian_t P, Fr_t s) {
    Commitment pp1(1, P);
    FrTensor S(1, &s);
    G1TensorJacobian out = pp1.commit(S);
    return out(0);
}

static G1Jacobian_t g1_add_host(G1Jacobian_t a, G1Jacobian_t b) {
    G1TensorJacobian A(1, a);
    G1TensorJacobian B(1, b);
    G1TensorJacobian C = A + B;
    return C(0);
}

// Build C = m · U + r · H using the size-1 commitment pipeline.  We
// avoid commit_hiding here because that path samples r internally; the
// caller of commit_round_poly needs to retain its own blindings.
static G1Jacobian_t commit_mU_rH(G1Jacobian_t U, G1Jacobian_t H,
                                 Fr_t m, Fr_t r) {
    return g1_add_host(g1_scalar_mul_host(U, m),
                       g1_scalar_mul_host(H, r));
}

RoundCommitment commit_round_poly(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const std::vector<Fr_t>& coeffs)
{
    if (coeffs.empty()) {
        throw std::runtime_error("commit_round_poly: empty coefficient vector");
    }

    RoundCommitment rc;
    rc.T.reserve(coeffs.size());
    rc.rho.reserve(coeffs.size());

    // One fresh blinding per coefficient.  FrTensor::random pulls
    // from the same RNG path as Phase 1's commit_hiding, so the
    // distinguisher tests in test_opening_distinguisher exercise the
    // same source.
    FrTensor rho_t = FrTensor::random(coeffs.size());
    for (uint k = 0; k < coeffs.size(); k++) {
        Fr_t rho_k = rho_t(k);
        rc.rho.push_back(rho_k);
        rc.T.push_back(commit_mU_rH(U, H, coeffs[k], rho_k));
    }
    return rc;
}

G1Jacobian_t fold_commitments_at(
    const std::vector<G1Jacobian_t>& T,
    Fr_t r)
{
    if (T.empty()) {
        throw std::runtime_error("fold_commitments_at: empty commitment vector");
    }

    // Σ_k r^k · T_k computed via successive scalar muls + adds.  At
    // typical sumcheck degrees (d ≤ 4 for multi-hadamard's K=4 case)
    // the loop is tiny; no need for an MSM kernel here.
    Fr_t pow = {1,0,0,0,0,0,0,0};
    G1Jacobian_t acc = g1_scalar_mul_host(T[0], pow);  // = T[0]
    for (uint k = 1; k < T.size(); k++) {
        pow = pow * r;  // host-side Fr_t multiply (poly/polynomial.cu)
        acc = g1_add_host(acc, g1_scalar_mul_host(T[k], pow));
    }
    return acc;
}

Fr_t fold_blindings_at(
    const std::vector<Fr_t>& rho,
    Fr_t r)
{
    if (rho.empty()) {
        throw std::runtime_error("fold_blindings_at: empty blinding vector");
    }
    Fr_t pow = {1,0,0,0,0,0,0,0};
    Fr_t acc = rho[0];
    for (uint k = 1; k < rho.size(); k++) {
        pow = pow * r;
        acc = acc + (rho[k] * pow);
    }
    return acc;
}

G1Jacobian_t sumcheck_identity_lhs(
    const std::vector<G1Jacobian_t>& T)
{
    if (T.empty()) {
        throw std::runtime_error("sumcheck_identity_lhs: empty commitment vector");
    }
    // 2·T_0 + T_1 + T_2 + … + T_d.  Equivalent to fold_at(0) + fold_at(1).
    Fr_t two = {2,0,0,0,0,0,0,0};
    G1Jacobian_t acc = g1_scalar_mul_host(T[0], two);
    for (uint k = 1; k < T.size(); k++) {
        acc = g1_add_host(acc, T[k]);
    }
    return acc;
}

Fr_t sumcheck_identity_blinding(
    const std::vector<Fr_t>& rho)
{
    if (rho.empty()) {
        throw std::runtime_error("sumcheck_identity_blinding: empty blinding vector");
    }
    Fr_t two = {2,0,0,0,0,0,0,0};
    Fr_t acc = two * rho[0];
    for (uint k = 1; k < rho.size(); k++) {
        acc = acc + rho[k];
    }
    return acc;
}
