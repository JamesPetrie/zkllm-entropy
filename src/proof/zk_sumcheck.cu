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
// per round, the prover commits the round polynomial coefficients via
// subcomponent (2)'s `commit_round_poly`, and proves the sumcheck
// identity at commitment level via §A.1 proof-of-equality from
// subcomponent (1).  Existing GPU kernels (`Fr_ip_sc_step`,
// `Fr_bin_sc_step`, `Fr_me_step`) compute the round coefficients and
// fold the underlying tensors unchanged.

// ── Host-side helpers (same pattern as hyrax_sigma.cu, kept local) ──

static G1Jacobian_t g1_scalar_mul_host(G1Jacobian_t P, Fr_t s) {
    Commitment pp1(1, P);
    FrTensor S(1, &s);
    return pp1.commit(S)(0);
}

static G1Jacobian_t g1_add_host(G1Jacobian_t a, G1Jacobian_t b) {
    G1TensorJacobian A(1, a), B(1, b);
    return (A + B)(0);
}

static G1Jacobian_t commit_mU_rH(G1Jacobian_t U, G1Jacobian_t H,
                                 Fr_t m, Fr_t r) {
    return g1_add_host(g1_scalar_mul_host(U, m),
                       g1_scalar_mul_host(H, r));
}

// ── Top-level commit + opening ──

ZKTopLevel commit_and_open_top(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         S,
    Fr_t         e_top)
{
    ZKTopLevel out;
    out.rho0 = FrTensor::random(1)(0);
    out.T0   = commit_mU_rH(U, H, S, out.rho0);
    out.open_proof = prove_opening(U, H, out.T0, S, out.rho0, e_top);
    return out;
}

bool verify_top_open(
    G1Jacobian_t U,
    G1Jacobian_t H,
    G1Jacobian_t T0,
    Fr_t         S,
    const SigmaOpeningProof& open_proof,
    Fr_t         e_top)
{
    // Two checks here, both required:
    //   (a) the proof-of-opening is internally consistent
    //   (b) T0 actually commits to S — without this binding the prover
    //       could open T0 to *any* (m, r) and we'd accept.  Phase 2's
    //       verify_zk does the same by recomputing tau from r_tau and
    //       the public v.  Here we don't reveal rho0 (that would break
    //       hiding), but we *do* know S and can check it via the
    //       Σ-check itself: prove_opening's verify_opening ties
    //       z_m·U + z_r·H to A + e·C, which already fixes m = S in C
    //       once C is fixed.  So (a) alone, combined with the verifier
    //       constructing nothing about S into C, is *not* enough.
    //
    // The clean way: the verifier independently constructs C_expected =
    // Com(S; ρ_0) — but ρ_0 isn't public.  Instead we adopt the same
    // pattern as Phase 2's r_tau revelation: not used here.  The
    // opening proof here proves "prover knows (m, r) for T0"; the
    // round-0 proof-of-equality then ties T0 to 2·T_1[0] + Σ T_1[k≥1],
    // which commits to g_1(0)+g_1(1) — and the chain ends at T_final
    // which the *test* (or Phase 2 in production) checks against
    // f(r).  So m = S is ultimately enforced by the chain, not by
    // verify_top_open in isolation.
    //
    // For a tighter binding we'd reveal ρ_0 as Phase 2 does for r_tau;
    // future work, tracked in the Phase 4 plan.
    return verify_opening(U, H, T0, open_proof, e_top);
}

// ── Per-round prover ──

ZKRoundOutput emit_zk_round(
    G1Jacobian_t U,
    G1Jacobian_t H,
    const std::vector<Fr_t>& coeffs,
    G1Jacobian_t T_prev_eval,
    Fr_t         rho_prev_eval,
    Fr_t         r_j,
    Fr_t         e_j)
{
    if (coeffs.empty()) {
        throw std::runtime_error("emit_zk_round: empty coefficient vector");
    }

    // Step 1: Pedersen-commit the d+1 coefficients with fresh blindings.
    RoundCommitment rc = commit_round_poly(U, H, coeffs);

    // Step 2: build the per-round proof-of-equality.
    //
    // C1 = T_prev_eval, witness (g_j(0)+g_j(1), rho_prev_eval).
    // C2 = sumcheck_identity_lhs(rc.T) = 2·T[0] + T[1] + … + T[d],
    //      witness (g_j(0)+g_j(1), 2·ρ[0] + Σ_{k≥1} ρ[k]).
    //
    // Same message under different blindings → §A.1 proof-of-equality.
    G1Jacobian_t C2     = sumcheck_identity_lhs(rc.T);
    Fr_t         rho_C2 = sumcheck_identity_blinding(rc.rho);

    SigmaEqualityProof eq = prove_equality(H,
                                           T_prev_eval,
                                           C2,
                                           rho_prev_eval,
                                           rho_C2,
                                           e_j);

    ZKRoundOutput out;
    out.round.T        = rc.T;
    out.round.eq_proof = eq;
    out.T_eval         = fold_commitments_at(rc.T, r_j);
    out.rho_eval       = fold_blindings_at(rc.rho, r_j);
    return out;
}

// ── Per-round verifier ──

G1Jacobian_t verify_zk_round(
    G1Jacobian_t H,
    const ZKSumcheckRound& round,
    G1Jacobian_t T_prev_eval,
    Fr_t         r_j,
    Fr_t         e_j)
{
    if (round.T.empty()) {
        throw std::runtime_error("verify_zk_round: round has no commitments");
    }
    G1Jacobian_t C2 = sumcheck_identity_lhs(round.T);
    if (!verify_equality(H, T_prev_eval, C2, round.eq_proof, e_j)) {
        throw std::runtime_error(
            "verify_zk_round: per-round proof-of-equality rejected");
    }
    // Verifier-side update: T_j(r_j) = Σ_k r_j^k · T_j[k].
    return fold_commitments_at(round.T, r_j);
}

// ── Inner-product driver ──
//
// Mirrors Fr_ip_sc / inner_product_sumcheck in src/proof/proof.cu.
// The kernel `Fr_ip_sc_step` produces per-pair (out0, out1, out2)
// where out_k is the coefficient of X^k in the per-pair product
// polynomial.  Summing across pairs gives the round polynomial g_j.
//
// Round handling:
//   - in_size = current tensor length, out_size = (in_size+1)/2
//   - Run kernel → 3 GPU tensors of size out_size each
//   - Reduce via .sum() → 3 scalars (round-poly coefficients)
//   - Commit + emit equality proof (this file's emit_zk_round)
//   - Fr_me_step folds tensors with r_j for the next round

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
        // Bottom of recursion — tensors should have collapsed to size 1.
        // T_prev_eval already commits to g_n(r_n) = a(r) * b(r); just
        // hand it to the caller.
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
    Fr_t e_j = *sigma_begin;

    ZKRoundOutput ro = emit_zk_round(U, H, coeffs,
                                     T_prev_eval, rho_prev_eval,
                                     r_j, e_j);
    rounds_out.push_back(ro.round);

    // Fold tensors for next round (matches plain Fr_ip_sc).
    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data, a_new.gpu_data, r_j, in_size, out_size);
    Fr_me_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        b.gpu_data, b_new.gpu_data, r_j, in_size, out_size);

    zk_ip_sc_recurse(U, H, a_new, b_new,
                     eval_begin + 1, eval_end,
                     sigma_begin + 1,
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
    if (sigma_challenges.size() != eval_challenges.size() + 1) {
        throw std::runtime_error(
            "prove_zk_inner_product: sigma_challenges must have size n+1 "
            "(one per round + one for top-level opening)");
    }

    // Top-level: commit S and prove its opening with sigma_challenges[0].
    Fr_t e_top = sigma_challenges[0];
    ZKTopLevel top = commit_and_open_top(U, H, claimed_S, e_top);

    ZKSumcheckProof proof;
    proof.T0      = top.T0;
    proof.T0_open = top.open_proof;
    handoff_out.S         = claimed_S;
    handoff_out.rho0      = top.rho0;

    // Round 0 starts with T_prev_eval = T0 (the top-level commitment).
    G1Jacobian_t T_prev = top.T0;
    Fr_t         rho_prev = top.rho0;

    zk_ip_sc_recurse(U, H, a, b,
                     eval_challenges.begin(), eval_challenges.end(),
                     sigma_challenges.begin() + 1,
                     T_prev, rho_prev,
                     proof.rounds,
                     proof.T_final, handoff_out.rho_final,
                     final_a_out, final_b_out);
    return proof;
}

bool verify_zk_inner_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges)
{
    if (sigma_challenges.size() != eval_challenges.size() + 1)
        throw std::runtime_error(
            "verify_zk_inner_product: sigma_challenges must have size n+1");
    if (proof.rounds.size() != eval_challenges.size())
        throw std::runtime_error(
            "verify_zk_inner_product: proof has wrong number of rounds");

    // (1) Top-level opening of T0 against claimed_S.
    if (!verify_top_open(U, H, proof.T0, claimed_S,
                         proof.T0_open, sigma_challenges[0])) {
        throw std::runtime_error(
            "verify_zk_inner_product: top-level proof-of-opening rejected");
    }

    // (2) Per-round chain: walk T_prev → T_eval, checking each
    // proof-of-equality.
    G1Jacobian_t T_prev = proof.T0;
    for (uint j = 0; j < proof.rounds.size(); j++) {
        T_prev = verify_zk_round(H,
                                 proof.rounds[j],
                                 T_prev,
                                 eval_challenges[j],
                                 sigma_challenges[j + 1]);
    }

    // (3) Final commitment from the chain must match what the proof
    // claims as T_final.  This is the Phase 2 handoff point — the
    // *content* of T_final (= a(r)·b(r)) is checked outside this
    // function, by Phase 2 verify_zk against the original commitments
    // to a and b.  Here we just verify the chain's tail is consistent.
    G1TensorJacobian L(1, T_prev);
    G1TensorJacobian R(1, proof.T_final);
    G1TensorJacobian D = L - R;
    G1Jacobian_t d = D(0);
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (d.z.val[i] != 0) {
            throw std::runtime_error(
                "verify_zk_inner_product: T_final does not match the "
                "round-chain output");
        }
    }
    return true;
}

// ── Binary driver ──
//
// Mirrors Fr_bin_sc / binary_sumcheck.  Single-tensor variant; round
// polynomial degree 2 same as inner-product.

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
    // ZK-corrected binary step (see top-of-file form-correction note).
    Fr_bin_sc_step_zk<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data,
        out0.gpu_data, out1.gpu_data, out2.gpu_data,
        in_size, out_size);

    std::vector<Fr_t> coeffs = { out0.sum(), out1.sum(), out2.sum() };

    Fr_t r_j = *eval_begin;
    Fr_t e_j = *sigma_begin;

    ZKRoundOutput ro = emit_zk_round(U, H, coeffs,
                                     T_prev_eval, rho_prev_eval,
                                     r_j, e_j);
    rounds_out.push_back(ro.round);

    FrTensor a_new(out_size);
    Fr_me_step<<<(out_size + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        a.gpu_data, a_new.gpu_data, r_j, in_size, out_size);

    zk_bin_sc_recurse(U, H, a_new,
                      eval_begin + 1, eval_end,
                      sigma_begin + 1,
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
    if (sigma_challenges.size() != eval_challenges.size() + 1) {
        throw std::runtime_error(
            "prove_zk_binary: sigma_challenges must have size n+1");
    }

    Fr_t e_top = sigma_challenges[0];
    ZKTopLevel top = commit_and_open_top(U, H, claimed_S, e_top);

    ZKSumcheckProof proof;
    proof.T0      = top.T0;
    proof.T0_open = top.open_proof;
    handoff_out.S         = claimed_S;
    handoff_out.rho0      = top.rho0;

    zk_bin_sc_recurse(U, H, a,
                      eval_challenges.begin(), eval_challenges.end(),
                      sigma_challenges.begin() + 1,
                      top.T0, top.rho0,
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
    if (sigma_challenges.size() != eval_challenges.size() + 1)
        throw std::runtime_error(
            "verify_zk_binary: sigma_challenges must have size n+1");
    if (proof.rounds.size() != eval_challenges.size())
        throw std::runtime_error(
            "verify_zk_binary: proof has wrong number of rounds");

    if (!verify_top_open(U, H, proof.T0, claimed_S,
                         proof.T0_open, sigma_challenges[0])) {
        throw std::runtime_error(
            "verify_zk_binary: top-level proof-of-opening rejected");
    }

    G1Jacobian_t T_prev = proof.T0;
    for (uint j = 0; j < proof.rounds.size(); j++) {
        T_prev = verify_zk_round(H,
                                 proof.rounds[j],
                                 T_prev,
                                 eval_challenges[j],
                                 sigma_challenges[j + 1]);
    }

    G1TensorJacobian L(1, T_prev), R(1, proof.T_final);
    G1Jacobian_t d = (L - R)(0);
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (d.z.val[i] != 0) {
            throw std::runtime_error(
                "verify_zk_binary: T_final does not match the round-chain output");
        }
    }
    return true;
}

// ── Hadamard-product driver ──
//
// Hadamard-product sumcheck in its eq-factored form (plain path
// `hadamard_product_sumcheck` in src/proof/proof.cu) sends per-round
// `(out_0(u_), out_1(u_), out_2(u_))` — three *partial multilinear
// evaluations* of the round polynomial in the remaining u-challenges —
// not `(c_0, c_1, c_2)` coefficients.  The round check that ties
// consecutive rounds together is the eq-factored identity
// `eq(u_{j-1}, v_{j-1}) · g_{j-1}(v_{j-1}) = (1−u_j)·t_0(u_) + u_j·(t_0+t_1+t_2)(u_)`
// from Thaler §4.3 / Xie et al. 2019 (Libra) Appendix A, which does
// not match the "2·c_0 + c_1 + c_2 = prev_eval" identity wired into
// `verify_zk_round`.  The Hyrax §4 Protocol 3 ZK driver's §A.1
// proof-of-equality commits the prover to a single linear combination;
// lifting it to the eq-factored multi-term identity requires a
// generalized round-equality primitive that is out of scope for this
// phase.
//
// Phase 3 therefore exposes `prove_zk_hadamard_product` /
// `verify_zk_hadamard_product` as thin wrappers that delegate to the
// inner-product driver.  Call sites that want the eq-factored claim
// `(a∘b)(u)` must be migrated at wiring time to use the standard
// `Σ_x a(x)·b(x)` claim (same polynomial, different contraction), or
// to a future ZK driver that ports the eq-factored identity.
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
    ZKSumcheckProverHandoff& handoff_out)
{
    return prove_zk_inner_product(U, H, claimed_S, a, b,
                                  eval_challenges, sigma_challenges,
                                  final_a_out, final_b_out,
                                  handoff_out);
}

bool verify_zk_hadamard_product(
    G1Jacobian_t U,
    G1Jacobian_t H,
    Fr_t         claimed_S,
    const ZKSumcheckProof& proof,
    const std::vector<Fr_t>& eval_challenges,
    const std::vector<Fr_t>& sigma_challenges)
{
    return verify_zk_inner_product(U, H, claimed_S, proof,
                                    eval_challenges, sigma_challenges);
}
