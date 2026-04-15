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
// The plain-sumcheck implementations in src/proof/proof.cu emit each
// round polynomial as 3 scalars (coefficients of X^0, X^1, X^2 for the
// degree-2 cases).  Each scalar is a linear function of the witness
// and leaks information about it.  The ZK driver replaces those
// scalars with Pedersen commitments T_j^{(k)} = Com(c_j^{(k)}; ρ_j^{(k)})
// for k = 0..d, and proves the per-round sumcheck identity at
// commitment level via a §A.1 proof-of-equality.

// ─── Per-round transcript ─────────────────────────────────────────────
//
// `T` are the d+1 commitments to coefficients of g_j(X) in coefficient
// form.  `eq_proof` is a Hyrax §A.1 proof-of-equality showing that the
// sum-at-{0,1} commitment 2·T[0] + T[1] + … + T[d] equals the
// previous-round evaluation commitment (or T0 for round 0).  Both
// commit to the same scalar — the per-round sumcheck identity says
// g_j(0) + g_j(1) = g_{j-1}(r_{j-1}) — but with different blindings,
// which is exactly the proof-of-equality statement.
struct ZKSumcheckRound {
    std::vector<G1Jacobian_t> T;
    SigmaEqualityProof        eq_proof;
};

// ─── Full transcript ──────────────────────────────────────────────────
//
// `T0` pins the claimed sum S via Com(S; ρ_0); `T0_open` is a §A.1
// proof-of-opening showing the prover knows (S, ρ_0).  `rounds` carries
// the per-round transcripts.  `T_final` is the final-round evaluation
// commitment T_n(r_n) = Com(g_n(r_n); ρ_final), which the caller
// passes to Phase 2's verify_zk to discharge the final dot-product
// claim (Hyrax §4 Protocol 3 final step → §A.2 Figure 6).
struct ZKSumcheckProof {
    G1Jacobian_t                  T0;
    SigmaOpeningProof             T0_open;
    std::vector<ZKSumcheckRound>  rounds;
    G1Jacobian_t                  T_final;
};

// Prover-side handoff metadata not part of the proof.  Needed for the
// Phase 2 verify_zk handoff (Phase 2's `r_tau` field).  Verifier
// derives nothing from this; it's witness-equivalent.
struct ZKSumcheckProverHandoff {
    Fr_t  S;                       // claimed sum (witness for T0 opening)
    Fr_t  rho0;                    // blinding for T0
    Fr_t  rho_final;               // blinding for T_final
};

// ─── Generic round-level helpers ──────────────────────────────────────
//
// Variant-specific drivers (inner-product, hadamard, binary,
// multi-hadamard) share this round-level glue.  They differ only in
// (a) the kernel that produces the round coefficients and (b) the
// state-update kernel that folds the underlying tensors with r_j.

// Prover: given the d+1 coefficients of g_j(X), the previous-round
// (T_eval, ρ_eval), and challenges (r_j for the multilinear fold,
// e_j for the Σ-protocol), emit the round bundle and the new
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
    G1Jacobian_t T_prev_eval,
    Fr_t         rho_prev_eval,
    Fr_t         r_j,
    Fr_t         e_j);

// Verifier: given the round bundle, the previous-round commitment
// T_prev_eval, and the same challenges (r_j, e_j), check the per-
// round sumcheck identity and return the new T_eval for round j+1.
// Throws std::runtime_error with a specific message on failure so
// negative tests can pinpoint which check rejected.
G1Jacobian_t verify_zk_round(
    G1Jacobian_t H,
    const ZKSumcheckRound& round,
    G1Jacobian_t T_prev_eval,
    Fr_t         r_j,
    Fr_t         e_j);

// ─── Top-level commit + opening proof ─────────────────────────────────
//
// Both callable by the prover as the very first step.  The verifier
// uses verify_top_open to validate the same against a claimed S.

struct ZKTopLevel {
    G1Jacobian_t       T0;
    Fr_t               rho0;          // prover-side
    SigmaOpeningProof  open_proof;
};

ZKTopLevel commit_and_open_top(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         S,
    Fr_t         e_top);

// Returns true iff the proof-of-opening accepts T0 against the claimed
// S.  The verifier knows S as the sumcheck claim (it's the public
// input to the protocol).
bool verify_top_open(
    G1Jacobian_t U,
    G1Jacobian_t H,
    G1Jacobian_t T0,
    Fr_t         S,
    const SigmaOpeningProof& open_proof,
    Fr_t         e_top);

// ─── Per-variant drivers ──────────────────────────────────────────────
//
// Inner-product sumcheck.  `f(x) = a(x) · b(x)` for multilinear a, b
// over {0,1}^n.  Round polynomial degree d = 2.  The bottom of the
// recursion produces the contracted scalars a(r), b(r); the caller
// discharges those via Phase 2 verify_zk as in the plain code path.
//
// `eval_challenges` are the standard sumcheck challenges (size n);
// `sigma_challenges` are the Σ-protocol challenges (size n+1, one
// extra for T0_open at the top).  `final_a` and `final_b` are the
// contracted scalars a(r), b(r) — returned to the caller for the
// Phase 2 discharge.
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
// returns true iff every check (top-level opening + n proof-of-
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

// Hadamard sumcheck.  Same shape as inner-product (degree 2) but the
// caller-supplied tensor folding follows the v-pattern from
// hadamard_product_sumcheck (sumcheck claim at the end is contracted
// via partial_me with two distinct challenges).  In Phase 3 the
// driver only handles the round-by-round transcript; the final claim
// shape stays variant-specific and is returned to the caller.
ZKSumcheckProof prove_zk_hadamard_product(
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

bool verify_zk_hadamard_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& eval_challenges,
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

#endif
