#ifndef ZK_ROUND_COMMIT_CUH
#define ZK_ROUND_COMMIT_CUH

#include "tensor/fr-tensor.cuh"
#include "tensor/g1-tensor.cuh"
#include <vector>

// Phase 3 subcomponent (2): commitment to a sumcheck round polynomial.
//
// Hyrax §4 Protocol 3 (Wahby et al. 2018, eprint 2017/1132, p. 11–13)
// blinds each round of the sumcheck by Pedersen-committing the round
// polynomial g_j(X) instead of sending its coefficients in the clear.
// Per round the prover emits {Com(c_k; ρ_k)} for k = 0..d, where
// c_0, …, c_d are the coefficients of g_j(X) (degree d).
//
// Coefficient form is the natural shape for the existing kernels in
// src/proof/proof.cu (`Fr_ip_sc_step`, `Fr_hp_sc_step`, `Fr_bin_sc_step`)
// — they already emit `out_k` as coefficients of X^k.  Evaluation form
// would require a basis change per round; coefficient form lets the
// driver bolt on the commitment layer with no kernel rewrite.
//
// Two homomorphic operations are used by the §4 driver:
//
//   (a) eval_at(r):     Σ_k r^k · T_k        commits to g_j(r) under
//                       blinding Σ_k r^k · ρ_k.  Used for round-to-
//                       round handoff (T_j(r_j) is the input for j+1)
//                       and for the verifier's commitment-level eval.
//
//   (b) sumcheck_lhs(): T_0 + Σ_k T_k       commits to g_j(0) + g_j(1)
//                       under blinding ρ_0 + Σ_k ρ_k.  Equals 2T_0 +
//                       Σ_{k≥1} T_k after collecting the duplicated T_0
//                       term.  This is the LHS of the sumcheck identity
//                       at commitment level; the §A.1 proof-of-equality
//                       ties it to the previous round's eval commitment.
//
// Both operations are linear in the Pedersen homomorphism, so the
// scalar-side and commitment-side folds match identically:
//   fold({Com(c_k; ρ_k)}, r)  =  Com(p(r); ρ(r))
// where ρ(r) = Σ_k r^k · ρ_k.  The test suite verifies this byte-exactly.

struct RoundCommitment {
    // T[k] = Com(coef[k]; rho[k]) = coef[k] · U + rho[k] · H, k = 0..d.
    std::vector<G1Jacobian_t> T;

    // Blindings for each coefficient — prover-side only; the verifier
    // sees only T[].  Folded into the next round via fold_blindings_at.
    std::vector<Fr_t>         rho;

    uint degree() const { return T.size() - 1; }
};

// Prover: commit a round polynomial given as d+1 coefficients.
// Samples one fresh blinding ρ_k per coefficient; reuse across rounds
// or across calls would leak (the §4 zero-knowledge argument relies on
// independent ρ per emitted commitment).
RoundCommitment commit_round_poly(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const std::vector<Fr_t>& coeffs);

// Verifier-callable: homomorphic evaluation of a coefficient-form
// commitment vector {T_k} at point r:  Σ_k r^k · T_k.
G1Jacobian_t fold_commitments_at(
    const std::vector<G1Jacobian_t>& T,
    Fr_t r);

// Prover-callable: same fold for blindings:  Σ_k r^k · ρ_k.  The
// verifier never touches this — it's the new blinding for the next
// round's input commitment.
Fr_t fold_blindings_at(
    const std::vector<Fr_t>& rho,
    Fr_t r);

// Verifier-callable: LHS of the sumcheck identity at commitment level,
// i.e. Com(p(0) + p(1); ρ(0) + ρ(1)) computed from {T_k} alone.
//
// In coefficient form  p(0) = c_0  and  p(1) = Σ_k c_k, so
//   p(0) + p(1) = 2c_0 + c_1 + … + c_d
// and the corresponding commitment is 2·T_0 + T_1 + … + T_d.
G1Jacobian_t sumcheck_identity_lhs(
    const std::vector<G1Jacobian_t>& T);

// Prover-callable: blinding for sumcheck_identity_lhs, namely
//   2·ρ_0 + ρ_1 + … + ρ_d.
Fr_t sumcheck_identity_blinding(
    const std::vector<Fr_t>& rho);

#endif
