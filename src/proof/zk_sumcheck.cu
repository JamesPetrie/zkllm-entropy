#include "proof/zk_sumcheck.cuh"
#include "proof/proof.cuh"
#include "commit/commitment.cuh"
#include "poly/polynomial.cuh"
#include "field/bls12-381.cuh"

// Form-correction note (Phase 3 driver only):
//
// The plain sumcheck kernels in src/proof/proof.cu use Scalar_mul without
// the Scalar_mont wrap that the elementwise tensor ops apply
// (fr-tensor.cu:144).  That makes their multiplicative outputs land in
// "inverse-Montgomery" form (x·y/R) rather than standard form, which is
// fine for the plain protocol because the existing scalar callers stay
// internally consistent — no caller compares the raw kernel sums against
// a host-Fr_t-computed claim.  Phase 3 *does* compare: the public claim
// S is computed host-side via Fr_t operator*, which lands in standard
// form (polynomial.cu:73 wraps mul with mont).  So the driver has to
// normalize the kernel outputs back to standard form before committing.
//
// For purely multiplicative kernels (Fr_ip_sc_step) a single .mont() on
// the output tensor suffices: mont(x·y/R) = x·y.  But Fr_bin_sc_step
// mixes Scalar_mul outputs with raw additive corrections (a0² - a0,
// 2·a0·diff - diff), so the inverse-Mont scaling and the additive terms
// land in incompatible forms — no per-element transform fixes both.  We
// re-implement the binary step here with mont wraps inside the kernel.

KERNEL void Fr_bin_sc_step_zk(GLOBAL Fr_t *a,
                              GLOBAL Fr_t *out0,
                              GLOBAL Fr_t *out1,
                              GLOBAL Fr_t *out2,
                              uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;

    Fr_t a0 = (2 * gid     < in_size) ? a[2 * gid    ] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (2 * gid + 1 < in_size) ? a[2 * gid + 1] : blstrs__scalar__Scalar_ZERO;
    Fr_t one = {1, 0, 0, 0, 0, 0, 0, 0};

    // out0 = a0·(a0 - 1)
    Fr_t a0m1 = blstrs__scalar__Scalar_sub(a0, one);
    out0[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a0, a0m1));

    // out1 = (a1 - a0)·(2·a0 - 1)
    Fr_t diff = blstrs__scalar__Scalar_sub(a1, a0);
    Fr_t two_a0_m1 = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_double(a0), one);
    out1[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(diff, two_a0_m1));

    // out2 = (a1 - a0)²
    out2[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(diff, diff));
}

// Phase 3 subcomponent (3): ZK sumcheck driver implementation.
//
// Hyrax §4 Protocol 3 (Wahby et al. 2018, eprint 2017/1132, p. 11–13):
// per round, the prover Pedersen-commits the round polynomial
// coefficients (subcomponent (2)), emits a §A.1 proof-of-opening for
// each, and proves the sumcheck identity at commitment level via a
// §A.1 proof-of-equality (subcomponent (1)).  Existing GPU kernels
// (`Fr_ip_sc_step`, `Fr_bin_sc_step`, `Fr_me_step`) compute the round
// coefficients and fold the underlying tensors unchanged.
//
// Top-level handoff: per §4 Step 1 the verifier constructs C_0 =
// Com(S; 0) = S·U directly from the public claim.  No prover
// commitment, no top-level proof-of-opening.

// ── Host-side helpers (same pattern as hyrax_sigma.cu, kept local) ──

static G1Jacobian_t g1_scalar_mul_host(G1Jacobian_t P, Fr_t s) {
    Commitment pp1(1, P);
    FrTensor S(1, &s);
    return pp1.commit(S)(0);
}

// Hyrax §4 Step 1: verifier computes C_0 = Com(S; 0) = S·U.  Used by
// both prover and verifier as the round-0 "previous evaluation"
// commitment.  Blinding is 0 because S is public.
static G1Jacobian_t top_level_commitment(G1Jacobian_t U, Fr_t S) {
    return g1_scalar_mul_host(U, S);
}

// ── Per-round prover ──

ZKRoundOutput emit_zk_round(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const std::vector<Fr_t>& coeffs,
    const std::vector<Fr_t>& alphas,
    std::vector<Fr_t>::const_iterator sigma_begin,
    G1Jacobian_t T_prev_eval,
    Fr_t         rho_prev_eval,
    Fr_t         r_j)
{
    if (coeffs.empty())
        throw std::runtime_error("emit_zk_round: empty coefficient vector");
    if (alphas.size() != coeffs.size())
        throw std::runtime_error("emit_zk_round: |alphas| != |coeffs|");

    // Step 1: Pedersen-commit the d+1 coefficients with fresh blindings.
    RoundCommitment rc = commit_round_poly(U, H, coeffs);

    // Step 2: proof-of-opening for each δc_{k,j} (Hyrax §4 requires one
    // per coefficient so the extractor can lift the round polynomial).
    std::vector<SigmaOpeningProof> openings;
    openings.reserve(rc.T.size());
    for (uint k = 0; k < rc.T.size(); k++) {
        Fr_t e_open = *(sigma_begin + k);
        openings.push_back(prove_opening(U, H,
                                         rc.T[k],
                                         coeffs[k],
                                         rc.rho[k],
                                         e_open));
    }

    // Step 3: weighted LHS commitment + equality proof with the
    // previous-round evaluation commitment.
    //   C2     = Σ α_k · T[k]
    //   rho_C2 = Σ α_k · ρ[k]
    // Both commit to the same scalar Σ α_k · c_k (= g_j(0)+g_j(1) for
    // standard sumcheck α=(2,1,1,…); = c_0 + u_j·c_1 + u_j·c_2 for the
    // eq-factored HP identity).
    G1Jacobian_t C2     = combine_commitments_weighted(rc.T, alphas);
    Fr_t         rho_C2 = combine_blindings_weighted (rc.rho, alphas);

    Fr_t e_eq = *(sigma_begin + rc.T.size());
    SigmaEqualityProof eq = prove_equality(H,
                                           T_prev_eval,
                                           C2,
                                           rho_prev_eval,
                                           rho_C2,
                                           e_eq);

    ZKRoundOutput out;
    out.round.T        = rc.T;
    out.round.T_open   = std::move(openings);
    out.round.eq_proof = eq;
    out.T_eval         = fold_commitments_at(rc.T, r_j);
    out.rho_eval       = fold_blindings_at(rc.rho, r_j);
    return out;
}

// ── Per-round verifier ──

G1Jacobian_t verify_zk_round(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const ZKSumcheckRound& round,
    const std::vector<Fr_t>& alphas,
    std::vector<Fr_t>::const_iterator sigma_begin,
    G1Jacobian_t T_prev_eval,
    Fr_t         r_j)
{
    if (round.T.empty())
        throw std::runtime_error("verify_zk_round: round has no commitments");
    if (round.T_open.size() != round.T.size())
        throw std::runtime_error(
            "verify_zk_round: |T_open| != |T| (missing per-coef opening)");
    if (alphas.size() != round.T.size())
        throw std::runtime_error("verify_zk_round: |alphas| != |T|");

    // Per-coefficient proof-of-openings (Hyrax §4).
    for (uint k = 0; k < round.T.size(); k++) {
        Fr_t e_open = *(sigma_begin + k);
        if (!verify_opening(U, H, round.T[k], round.T_open[k], e_open)) {
            throw std::runtime_error(
                "verify_zk_round: per-coefficient proof-of-opening rejected");
        }
    }

    // Round-to-round proof-of-equality on Σ α_k · T_k vs T_prev_eval.
    G1Jacobian_t C2 = combine_commitments_weighted(round.T, alphas);
    Fr_t e_eq = *(sigma_begin + round.T.size());
    if (!verify_equality(H, T_prev_eval, C2, round.eq_proof, e_eq)) {
        throw std::runtime_error(
            "verify_zk_round: per-round proof-of-equality rejected");
    }
    // Verifier-side update: T_j(r_j) = Σ_k r_j^k · T_j[k].
    return fold_commitments_at(round.T, r_j);
}

// Standard-sumcheck weights: α = (2, 1, 1, …).  Matches plain
// `sumcheck_identity_lhs` from subcomponent (2).
static std::vector<Fr_t> standard_alphas(uint d_plus_one) {
    std::vector<Fr_t> a(d_plus_one, Fr_t{1,0,0,0,0,0,0,0});
    a[0] = Fr_t{2,0,0,0,0,0,0,0};
    return a;
}

// ── Inner-product driver ──
//
// Mirrors Fr_ip_sc / inner_product_sumcheck in src/proof/proof.cu.
// The kernel `Fr_ip_sc_step` produces per-pair (out0, out1, out2)
// where out_k is the coefficient of X^k in the per-pair product
// polynomial.  Summing across pairs gives the round polynomial g_j.
//
// Per-round layout consumes 4 sigma challenges: 3 for the coefficient
// openings and 1 for the round-to-round equality.

static constexpr uint kDegreeTwoSigmaPerRound = 4;  // 3 openings + 1 eq

static void zk_ip_sc_recurse(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const FrTensor& a,
    const FrTensor& b,
    std::vector<Fr_t>::const_iterator eval_begin,
    std::vector<Fr_t>::const_iterator eval_end,
    std::vector<Fr_t>::const_iterator sigma_begin,
    G1Jacobian_t T_prev_eval,
    Fr_t         rho_prev_eval,
    std::vector<ZKSumcheckRound>& rounds_out,
    G1Jacobian_t& T_final_out,
    Fr_t&         rho_final_out,
    Fr_t&         final_a_out,
    Fr_t&         final_b_out)
{
    if (a.size != b.size)
        throw std::runtime_error("zk_ip_sc_recurse: a.size != b.size");

    if (eval_begin >= eval_end) {
        // Bottom of recursion — tensors have collapsed to size 1.
        // T_prev_eval already commits to g_n(r_n) = a(r) * b(r); hand
        // it to the caller.
        T_final_out  = T_prev_eval;
        rho_final_out = rho_prev_eval;
        final_a_out  = a(0);
        final_b_out  = b(0);
        return;
    }

    auto in_size  = a.size;
    auto out_size = (in_size + 1) / 2;

    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data, b.gpu_data,
        out0.gpu_data, out1.gpu_data, out2.gpu_data,
        in_size, out_size);

    // Fr_ip_sc_step emits Scalar_mul(a,b) without the usual Scalar_mont wrap
    // that element-wise ops apply (fr-tensor.cu:144), so its outputs land in
    // inverse-Montgomery form (x·y/R).  .mont() re-multiplies by R,
    // restoring standard form so the commitments match the public claim S
    // that the caller computed with the host Fr_t operator*.
    out0.mont(); out1.mont(); out2.mont();
    std::vector<Fr_t> coeffs = { out0.sum(), out1.sum(), out2.sum() };

    Fr_t r_j = *eval_begin;

    ZKRoundOutput ro = emit_zk_round(U, H, coeffs,
                                     standard_alphas(coeffs.size()),
                                     sigma_begin,
                                     T_prev_eval, rho_prev_eval,
                                     r_j);
    rounds_out.push_back(ro.round);

    // Fold tensors for next round (matches plain Fr_ip_sc).
    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data, a_new.gpu_data, r_j, in_size, out_size);
    Fr_me_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        b.gpu_data, b_new.gpu_data, r_j, in_size, out_size);

    zk_ip_sc_recurse(U, H, a_new, b_new,
                     eval_begin + 1, eval_end,
                     sigma_begin + kDegreeTwoSigmaPerRound,
                     ro.T_eval, ro.rho_eval,
                     rounds_out,
                     T_final_out, rho_final_out,
                     final_a_out, final_b_out);
}

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
    ZKSumcheckProverHandoff& handoff_out)
{
    if (a.size != b.size)
        throw std::runtime_error("prove_zk_inner_product: |a| != |b|");
    if (sigma_challenges.size() != eval_challenges.size() * kDegreeTwoSigmaPerRound)
        throw std::runtime_error(
            "prove_zk_inner_product: sigma_challenges must have size n·4 "
            "(3 per-coef openings + 1 equality per round)");

    ZKSumcheckProof proof;

    // Round 0's T_prev_eval is C_0 = S·U (Hyrax §4 Step 1); ρ_0 = 0.
    G1Jacobian_t T_prev   = top_level_commitment(U, claimed_S);
    Fr_t         rho_prev = {0,0,0,0,0,0,0,0};

    zk_ip_sc_recurse(U, H, a, b,
                     eval_challenges.begin(), eval_challenges.end(),
                     sigma_challenges.begin(),
                     T_prev, rho_prev,
                     proof.rounds,
                     proof.T_final, handoff_out.rho_final,
                     final_a_out, final_b_out);
    return proof;
}

// Verify that the round-chain's final T matches the `T_final` the
// proof advertises.  Both are G1 points; compare by subtracting and
// checking z-limbs (same convention as verify_zk / hyrax_sigma).
static void assert_T_final_matches(G1Jacobian_t chain_tail,
                                   G1Jacobian_t proof_T_final,
                                   const char* label)
{
    G1TensorJacobian L(1, chain_tail), R(1, proof_T_final);
    G1Jacobian_t d = (L - R)(0);
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (d.z.val[i] != 0) {
            std::string msg = std::string(label) +
                ": T_final does not match the round-chain output";
            throw std::runtime_error(msg);
        }
    }
}

bool verify_zk_inner_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges)
{
    if (sigma_challenges.size() != eval_challenges.size() * kDegreeTwoSigmaPerRound)
        throw std::runtime_error(
            "verify_zk_inner_product: sigma_challenges must have size n·4");
    if (proof.rounds.size() != eval_challenges.size())
        throw std::runtime_error(
            "verify_zk_inner_product: proof has wrong number of rounds");

    // Per §4 Step 1: verifier constructs C_0 = S·U from public S.
    G1Jacobian_t T_prev = top_level_commitment(U, claimed_S);
    const auto alphas = standard_alphas(3);

    for (uint j = 0; j < proof.rounds.size(); j++) {
        T_prev = verify_zk_round(
            U, H,
            proof.rounds[j],
            alphas,
            sigma_challenges.begin() + j * kDegreeTwoSigmaPerRound,
            T_prev,
            eval_challenges[j]);
    }

    assert_T_final_matches(T_prev, proof.T_final, "verify_zk_inner_product");
    return true;
}

// ── Binary driver ──
//
// Mirrors Fr_bin_sc / binary_sumcheck.  Single-tensor variant; round
// polynomial degree 2 same as inner-product.  Same sigma layout.

static void zk_bin_sc_recurse(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const FrTensor& a,
    std::vector<Fr_t>::const_iterator eval_begin,
    std::vector<Fr_t>::const_iterator eval_end,
    std::vector<Fr_t>::const_iterator sigma_begin,
    G1Jacobian_t T_prev_eval,
    Fr_t         rho_prev_eval,
    std::vector<ZKSumcheckRound>& rounds_out,
    G1Jacobian_t& T_final_out,
    Fr_t&         rho_final_out,
    Fr_t&         final_a_out)
{
    if (eval_begin >= eval_end) {
        T_final_out   = T_prev_eval;
        rho_final_out = rho_prev_eval;
        final_a_out   = a(0);
        return;
    }

    auto in_size  = a.size;
    auto out_size = (in_size + 1) / 2;

    FrTensor out0(out_size), out1(out_size), out2(out_size);
    // ZK-corrected binary step (see form-correction note at top).
    Fr_bin_sc_step_zk<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data,
        out0.gpu_data, out1.gpu_data, out2.gpu_data,
        in_size, out_size);

    std::vector<Fr_t> coeffs = { out0.sum(), out1.sum(), out2.sum() };

    Fr_t r_j = *eval_begin;

    ZKRoundOutput ro = emit_zk_round(U, H, coeffs,
                                     standard_alphas(coeffs.size()),
                                     sigma_begin,
                                     T_prev_eval, rho_prev_eval,
                                     r_j);
    rounds_out.push_back(ro.round);

    FrTensor a_new(out_size);
    Fr_me_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data, a_new.gpu_data, r_j, in_size, out_size);

    zk_bin_sc_recurse(U, H, a_new,
                      eval_begin + 1, eval_end,
                      sigma_begin + kDegreeTwoSigmaPerRound,
                      ro.T_eval, ro.rho_eval,
                      rounds_out,
                      T_final_out, rho_final_out,
                      final_a_out);
}

ZKSumcheckProof prove_zk_binary(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const FrTensor& a,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges,
    Fr_t&        final_a_out,
    ZKSumcheckProverHandoff& handoff_out)
{
    if (sigma_challenges.size() != eval_challenges.size() * kDegreeTwoSigmaPerRound)
        throw std::runtime_error(
            "prove_zk_binary: sigma_challenges must have size n·4");

    ZKSumcheckProof proof;
    G1Jacobian_t T_prev   = top_level_commitment(U, claimed_S);
    Fr_t         rho_prev = {0,0,0,0,0,0,0,0};

    zk_bin_sc_recurse(U, H, a,
                      eval_challenges.begin(), eval_challenges.end(),
                      sigma_challenges.begin(),
                      T_prev, rho_prev,
                      proof.rounds,
                      proof.T_final, handoff_out.rho_final,
                      final_a_out);
    return proof;
}

bool verify_zk_binary(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges)
{
    if (sigma_challenges.size() != eval_challenges.size() * kDegreeTwoSigmaPerRound)
        throw std::runtime_error(
            "verify_zk_binary: sigma_challenges must have size n·4");
    if (proof.rounds.size() != eval_challenges.size())
        throw std::runtime_error(
            "verify_zk_binary: proof has wrong number of rounds");

    G1Jacobian_t T_prev = top_level_commitment(U, claimed_S);
    const auto alphas = standard_alphas(3);

    for (uint j = 0; j < proof.rounds.size(); j++) {
        T_prev = verify_zk_round(
            U, H,
            proof.rounds[j],
            alphas,
            sigma_challenges.begin() + j * kDegreeTwoSigmaPerRound,
            T_prev,
            eval_challenges[j]);
    }

    assert_T_final_matches(T_prev, proof.T_final, "verify_zk_binary");
    return true;
}

// ── Hadamard-product driver (eq-factored) ──
//
// The plain `hadamard_product_sumcheck` in src/proof/proof.cu is the
// eq-factored multilinear sumcheck (Libra-style, Xie et al. 2019
// Appendix A) proving S = (a∘b)(u) = Σ_x eq(x,u)·a(x)·b(x).  Each
// round's polynomial g_j(X) = Σ_{rest} eq((X, rest), u)·a·b factors
// as `g_j(X) = eq(X, u_j) · h_j(X)` with h_j degree 2; the prover
// sends only h_j's 3 coefficients (c_0, c_1, c_2).
//
// Round-to-round check.  After round j-1 the verifier holds
// h_{j-1}(v_{j-1}).  The sumcheck identity (1-u_j)·h_j(0) + u_j·h_j(1)
// = h_{j-1}(v_{j-1}) in coefficient form becomes
//   c_0 + u_j · c_1 + u_j · c_2 = h_{j-1}(v_{j-1}).
// At commitment level: proof-of-equality between Σ α_k · T_j[k]
// (α = (1, u_j, u_j)) and T_{j-1}(v_{j-1}).  Same primitive as IP, just
// HP-specific alphas instead of (2, 1, 1).
//
// First round's previous-eval is Com(S; 0) = S·U (Hyrax §4 Step 1).
// Consistent with the claim S being public.
//
// Bottom of recursion.  After n rounds of folding tensors at v, `a`
// and `b` have collapsed to singletons with values a(v) and b(v).  The
// sumcheck-reduced claim is h_{n-1}(v_{n-1}) = a(v)·b(v) — same final
// contraction as the inner-product driver.

// HP round weights: (1, u_j, u_j).  Not the standard (2, 1, 1).
static std::vector<Fr_t> hp_alphas(Fr_t u_j) {
    Fr_t one = {1,0,0,0,0,0,0,0};
    return { one, u_j, u_j };
}

static void zk_hp_sc_recurse(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const FrTensor& a,
    const FrTensor& b,
    std::vector<Fr_t>::const_iterator u_begin,
    std::vector<Fr_t>::const_iterator u_end,
    std::vector<Fr_t>::const_iterator v_begin,
    std::vector<Fr_t>::const_iterator v_end,
    std::vector<Fr_t>::const_iterator sigma_begin,
    G1Jacobian_t T_prev_eval,
    Fr_t         rho_prev_eval,
    std::vector<ZKSumcheckRound>& rounds_out,
    G1Jacobian_t& T_final_out,
    Fr_t&         rho_final_out,
    Fr_t&         final_a_out,
    Fr_t&         final_b_out)
{
    if (a.size != b.size)
        throw std::runtime_error("zk_hp_sc_recurse: a.size != b.size");
    if ((u_end - u_begin) != (v_end - v_begin))
        throw std::runtime_error("zk_hp_sc_recurse: |u| != |v|");

    if (v_begin >= v_end) {
        T_final_out  = T_prev_eval;
        rho_final_out = rho_prev_eval;
        final_a_out  = a(0);
        final_b_out  = b(0);
        return;
    }

    auto in_size  = a.size;
    auto out_size = (in_size + 1) / 2;

    // Same kernel as the inner-product driver — Fr_ip_sc_step emits
    // per-pair (out0, out1, out2) = coefficients of X^k in a·b.
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data, b.gpu_data,
        out0.gpu_data, out1.gpu_data, out2.gpu_data,
        in_size, out_size);

    // Normalise form (same rationale as IP driver — see form-correction
    // note at top of this file).
    out0.mont(); out1.mont(); out2.mont();

    // Partial multilinear evaluation at u_ = u[1..] gives h_j's
    // coefficients.  u_j = *u_begin is consumed as the eq-factor weight
    // for this round's round-identity check; u[1..] is the remaining
    // eq-factor point that gets absorbed into h_j per Libra's factoring.
    std::vector<Fr_t> u_(u_begin + 1, u_end);
    std::vector<Fr_t> coeffs = { out0(u_), out1(u_), out2(u_) };

    Fr_t v_j = *v_begin;
    Fr_t u_j = *u_begin;

    ZKRoundOutput ro = emit_zk_round(U, H, coeffs,
                                     hp_alphas(u_j),
                                     sigma_begin,
                                     T_prev_eval, rho_prev_eval,
                                     v_j);
    rounds_out.push_back(ro.round);

    // Fold tensors with v_j (matches plain Fr_hp_sc — fold-challenge v
    // is distinct from the eq-factor point u).
    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data, a_new.gpu_data, v_j, in_size, out_size);
    Fr_me_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        b.gpu_data, b_new.gpu_data, v_j, in_size, out_size);

    zk_hp_sc_recurse(U, H, a_new, b_new,
                     u_begin + 1, u_end,
                     v_begin + 1, v_end,
                     sigma_begin + kDegreeTwoSigmaPerRound,
                     ro.T_eval, ro.rho_eval,
                     rounds_out,
                     T_final_out, rho_final_out,
                     final_a_out, final_b_out);
}

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
    ZKSumcheckProverHandoff& handoff_out)
{
    if (a.size != b.size)
        throw std::runtime_error("prove_zk_hadamard_product: |a| != |b|");
    if (u_challenges.size() != v_challenges.size())
        throw std::runtime_error(
            "prove_zk_hadamard_product: |u| != |v|");
    if (sigma_challenges.size() != u_challenges.size() * kDegreeTwoSigmaPerRound)
        throw std::runtime_error(
            "prove_zk_hadamard_product: sigma_challenges must have size n·4");

    ZKSumcheckProof proof;
    G1Jacobian_t T_prev   = top_level_commitment(U, claimed_S);
    Fr_t         rho_prev = {0,0,0,0,0,0,0,0};

    zk_hp_sc_recurse(U, H, a, b,
                     u_challenges.begin(), u_challenges.end(),
                     v_challenges.begin(), v_challenges.end(),
                     sigma_challenges.begin(),
                     T_prev, rho_prev,
                     proof.rounds,
                     proof.T_final, handoff_out.rho_final,
                     final_a_out, final_b_out);
    return proof;
}

bool verify_zk_hadamard_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& u_challenges,
    const std::vector<Fr_t>& v_challenges,
    const std::vector<Fr_t>& sigma_challenges)
{
    if (u_challenges.size() != v_challenges.size())
        throw std::runtime_error(
            "verify_zk_hadamard_product: |u| != |v|");
    if (sigma_challenges.size() != u_challenges.size() * kDegreeTwoSigmaPerRound)
        throw std::runtime_error(
            "verify_zk_hadamard_product: sigma_challenges must have size n·4");
    if (proof.rounds.size() != u_challenges.size())
        throw std::runtime_error(
            "verify_zk_hadamard_product: proof has wrong number of rounds");

    G1Jacobian_t T_prev = top_level_commitment(U, claimed_S);

    for (uint j = 0; j < proof.rounds.size(); j++) {
        T_prev = verify_zk_round(
            U, H,
            proof.rounds[j],
            hp_alphas(u_challenges[j]),
            sigma_challenges.begin() + j * kDegreeTwoSigmaPerRound,
            T_prev,
            v_challenges[j]);
    }

    assert_T_final_matches(T_prev, proof.T_final, "verify_zk_hadamard_product");
    return true;
}
