#ifndef ZK_SUMCHECK_CUH
#define ZK_SUMCHECK_CUH

#include "tensor/fr-tensor.cuh"
#include "tensor/g1-tensor.cuh"
#include "proof/hyrax_sigma.cuh"
#include "proof/zk_round_commit.cuh"
#include <vector>

// Phase 3 subcomponent (3): ZK sumcheck driver.
//
// Hyrax §4 Protocol 3 (Wahby et al. 2018, eprint 2017/1132, p. 11–13):
// the prover Pedersen-commits each sumcheck round polynomial g_j(X)
// instead of sending coefficients in the clear, and ties consecutive
// rounds together via the §A.1 Σ-protocols.  Phase 3 wraps the
// existing plain-sumcheck kernels (`Fr_ip_sc_step`, `Fr_hp_sc_step`,
// `Fr_bin_sc_step`) with this commitment + Σ-protocol layer; the GPU
// kernels themselves are reused unchanged.
//
// Per-round transcript, verbatim from §4 (p. 7):
//   "In round j of the sum-check, P commits to s_j(t) = c_{0,j} + c_{1,j}·t
//    + c_{2,j}·t² + c_{3,j}·t³, via δc_{0,j}←Com(c_{0,j}), …, and P and V
//    execute proof-of-opening for each one.  Now P convinces V that
//    s_j(0) + s_j(1) = s_{j-1}(r_{j-1})."
//
// Round-to-round chaining:
//   - LHS of sumcheck identity:  Com(g_j(0) + g_j(1)) computed verifier-side
//     by homomorphism as  Σ α_k · T_k  (for standard sumcheck α = (2,1,1,…);
//     for the Libra/Xie eq-factored HP variant α = (1, u_j, u_j)).
//   - RHS:  previous-round evaluation commitment T_{j-1}(r_{j-1}).
//   - Tie:  §A.1 proof-of-equality since the two commitments carry the
//     same scalar under different blindings.
//
// Top-level handoff (§4 "Step 1"):
//   "V computes C_0 = Com(Ṽ_y(q',q); 0)."
// The verifier constructs the initial claim commitment directly from
// the public claim S with blinding 0 — no prover commitment, no
// top-level proof-of-opening.  First round's equality proof ties
// "Σ α_k · T_1[k] (round 0's LHS)" to  S · U  (that C_0), at which
// point the chain starts.  This strictly binds T to S — a prover who
// tries to run the chain with a different S' cannot match the round-0
// equality because Σ α_k · T_1[k] would need to equal S' · U, not
// S · U.

// ─── Per-round transcript ─────────────────────────────────────────────
//
// `T` are the d+1 commitments to coefficients of g_j(X) in coefficient
// form.  `T_open[k]` is a §A.1 proof-of-opening for T[k] — Hyrax §4
// requires these per coefficient so the extractor can lift the round
// polynomial.  `eq_proof` is the §A.1 proof-of-equality tying
// Σ α_k · T_k (LHS) to the previous-round evaluation commitment.
struct ZKSumcheckRound {
    std::vector<G1Jacobian_t>      T;
    std::vector<SigmaOpeningProof> T_open;    // size T.size()
    SigmaEqualityProof             eq_proof;
};

// ─── Full transcript ──────────────────────────────────────────────────
//
// `rounds` carries the per-round transcripts.  `T_final` is the
// final-round evaluation commitment T_n(r_n) = Com(g_n(r_n); ρ_final),
// which the caller passes to Phase 2's verify_zk to discharge the
// final dot-product claim (Hyrax §4 Protocol 3 final step → §A.2
// Figure 6).  No top-level commitment to S: the verifier computes
// C_0 = S · U directly, per §4 Step 1.
struct ZKSumcheckProof {
    std::vector<ZKSumcheckRound>  rounds;
    G1Jacobian_t                  T_final;
};

// Prover-side handoff metadata not part of the proof.  Only ρ_final
// escapes — the caller needs it to discharge T_final via Phase 2's
// verify_zk (Phase 2's `r_tau` field).  Round-0's previous-eval
// blinding is identically 0 (C_0 = S·U has no hiding), so no ρ_0 to
// carry.
struct ZKSumcheckProverHandoff {
    Fr_t  rho_final;
};

// ─── Sigma-challenge layout ───────────────────────────────────────────
//
// Per round consumes (d + 2) Σ-protocol challenges:
//   - d+1 for per-coefficient proof-of-openings (Hyrax §4 requires one
//     per δc_{k,j}).
//   - 1 for the round-to-round proof-of-equality.
//
// For degree-2 sumchecks (IP, binary, HP): 4 challenges per round.
// For degree-(K+1) multi-Hadamard: K+3 per round.  The driver APIs
// below validate |sigma_challenges| against this layout.

// ─── Generic round-level helpers ──────────────────────────────────────
//
// Variant-specific drivers (inner-product, hadamard, binary,
// multi-hadamard) share this round-level glue.  They differ only in
// (a) the kernel that produces the round coefficients and (b) the
// state-update kernel that folds the underlying tensors with r_j.

// Prover: given the d+1 coefficients of g_j(X), the weights α for the
// sumcheck identity LHS (standard: (2,1,1,…); HP: (1,u_j,u_j,…)), the
// previous-round (T_eval, ρ_eval), and per-round Σ challenges (d+1 for
// openings + 1 for equality), emit the round bundle and the new
// (T_eval, ρ_eval) for round j+1.
struct ZKRoundOutput {
    ZKSumcheckRound   round;
    G1Jacobian_t      T_eval;       // T_j(r_j) — input for round j+1
    Fr_t              rho_eval;     // ρ_j(r_j) — prover-side
};

ZKRoundOutput emit_zk_round(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const std::vector<Fr_t>& coeffs,
    const std::vector<Fr_t>& alphas,              // size coeffs.size()
    std::vector<Fr_t>::const_iterator sigma_begin, // consumes d+2
    G1Jacobian_t T_prev_eval,
    Fr_t         rho_prev_eval,
    Fr_t         r_j);

// Verifier: given the round bundle, the weights α, the previous-round
// commitment T_prev_eval, and matching challenges, check per-
// coefficient openings + round-to-round equality and return the new
// T_eval for round j+1.  Throws with a specific message on failure so
// negative tests can pinpoint which check rejected.
G1Jacobian_t verify_zk_round(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const ZKSumcheckRound& round,
    const std::vector<Fr_t>& alphas,
    std::vector<Fr_t>::const_iterator sigma_begin,
    G1Jacobian_t T_prev_eval,
    Fr_t         r_j);

// ─── Per-variant drivers ──────────────────────────────────────────────
//
// Inner-product sumcheck.  `f(x) = a(x) · b(x)` for multilinear a, b
// over {0,1}^n.  Round polynomial degree d = 2.  The bottom of the
// recursion produces the contracted scalars a(r), b(r); the caller
// discharges those via Phase 2 verify_zk as in the plain code path.
//
// `eval_challenges` are the standard sumcheck challenges (size n);
// `sigma_challenges` are the per-round Σ-protocol challenges
// (size n·(d+2) = n·4 for degree-2 variants).  `final_a` and `final_b`
// are the contracted scalars a(r), b(r) — returned to the caller for
// the Phase 2 discharge.
ZKSumcheckProof prove_zk_inner_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const FrTensor& a,
    const FrTensor& b,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges,
    Fr_t&        final_a_out,
    Fr_t&        final_b_out,
    ZKSumcheckProverHandoff& handoff_out);

// Verifier: walks the same per-round sumcheck identity chain and
// returns true iff every check (per-coef openings + round-to-round
// equalities) accepts.  The final commitment T_final is left for the
// caller to feed into Phase 2 verify_zk (with the contracted scalars
// the prover sent).  Throws on rejection so the test can identify
// which check failed.
bool verify_zk_inner_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges);

// Hadamard sumcheck — eq-factored variant (Libra-style, Xie et al.
// 2019 Appendix A).  The claim is `S = (a∘b)(u)`, the multilinear
// evaluation of the elementwise product `a ⊙ b` at a public point u.
// Internally the sumcheck splits each round polynomial g_j as
// `g_j(X) = eq(X, u_j) · h_j(X)` with h_j degree 2.  The prover
// commits the 3 coefficients of h_j; the round-to-round check is the
// weighted identity
//   (1 - u_j)·h_j(0) + u_j·h_j(1) = h_{j-1}(v_{j-1})
// which in coefficient form becomes
//   c_0 + u_j · c_1 + u_j · c_2 = h_{j-1}(v_{j-1})
// and at commitment level reduces to a same-message equality between
// the weighted combination Σ α_k · T_k (α = (1, u_j, u_j)) and the
// previous-round evaluation commitment.
//
// Two independent challenge sequences are consumed:
//   * u_challenges (size n): the eq-factor point — this is where the
//     prover's claim is evaluated.
//   * v_challenges (size n): the sumcheck fold challenges.
//   * sigma_challenges (size n·4): per-round Σ-protocol challenges.
//
// Final-claim contraction: T_final commits to h_{n-1}(v) = a(v)·b(v),
// same as the inner-product driver — the caller discharges it via
// Phase 2 verify_zk.
ZKSumcheckProof prove_zk_hadamard_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const FrTensor& a,
    const FrTensor& b,
    const std::vector<Fr_t>& u_challenges,
    const std::vector<Fr_t>& v_challenges,
    const std::vector<Fr_t>& sigma_challenges,
    Fr_t&        final_a_out,
    Fr_t&        final_b_out,
    ZKSumcheckProverHandoff& handoff_out);

bool verify_zk_hadamard_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& u_challenges,
    const std::vector<Fr_t>& v_challenges,
    const std::vector<Fr_t>& sigma_challenges);

// Binary sumcheck (`f(x) = a(x)·(a(x) - 1)`).  Degree 2 in X.
ZKSumcheckProof prove_zk_binary(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const FrTensor& a,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges,
    Fr_t&        final_a_out,
    ZKSumcheckProverHandoff& handoff_out);

bool verify_zk_binary(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges);

// Multi-Hadamard sumcheck — degree-K eq-factored variant.  Proves
// S = Σ_x eq(x, u) · Π_k X_k(x) for K multilinear tensors.
// Mirrors `multi_hadamard_sumchecks` (src/proof/proof.cu:220) which
// consumes u and v back-to-front; this driver does the same so the
// swap at call sites (softmax) is mechanical.
//
// Round polynomial factors as g_j(X) = eq(X, u_last) · h_j(X), where
// h_j(X) = Π_k ((1-X)·X_k(x', 0) + X·X_k(x', 1)) contracted at u_ =
// u[0..n-1).  h_j has degree K, so the prover commits K+1
// coefficients (c_0, …, c_K).
//
// Round identity (Libra/Xie et al. 2019 Appendix A):
//   (1 - u_last) · h_j(0) + u_last · h_j(1) = h_{j-1}(v_{prev})
// In coefficient form:
//   c_0 + u_last · (c_1 + c_2 + … + c_K) = h_{j-1}(v_{prev})
// At commitment level: proof-of-equality between
//   Σ α_k · T_j[k]  with α = (1, u_last, u_last, …, u_last)
// of length K+1, and T_{j-1}(v_{prev}).
//
// Per-round sigma challenges consumed: K+2 (K+1 openings + 1 equality).
//
// Final claim: after n rounds of folding each X_k with v.back() (plain
// `hadamard_reduce_kernel`), each X_k collapses to a scalar X_k(v_reversed);
// T_final commits to the last round's h(v_last), which equals
// Π_k X_k(v_reversed) · (last round's trivial eq factor).  `final_Xs_out`
// exposes these scalars for the caller's Phase 2 discharge via
// proof-of-product or K-ary chaining of `verifyWeightClaimZK`.
ZKSumcheckProof prove_zk_multi_hadamard(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const std::vector<FrTensor>& Xs,
    const std::vector<Fr_t>& u_challenges,      // size n — consumed back-to-front
    const std::vector<Fr_t>& v_challenges,      // size n — consumed back-to-front
    const std::vector<Fr_t>& sigma_challenges,  // size n·(K+2), K = Xs.size()
    std::vector<Fr_t>&       final_Xs_out,      // size K (one per tensor)
    ZKSumcheckProverHandoff& handoff_out);

bool verify_zk_multi_hadamard(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    uint         K,
    const std::vector<Fr_t>& u_challenges,
    const std::vector<Fr_t>& v_challenges,
    const std::vector<Fr_t>& sigma_challenges);

#endif
