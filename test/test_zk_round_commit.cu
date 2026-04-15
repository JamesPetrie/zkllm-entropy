// Tests for Phase 3 subcomponent (2): zk_round_commit.
//
// Covers:
//   (1) Roundtrip: commit a degree-d random poly, recover Com(p(r); ρ(r))
//       two ways — by folding commitments vs. by computing scalar p(r),
//       ρ(r) and committing.  Byte-equality is the structural correctness
//       gate for the homomorphism.
//   (2) Sumcheck-identity sum: 2·T_0 + Σ_{k≥1} T_k matches Com(p(0)+p(1);
//       2ρ_0 + Σ_{k≥1} ρ_k) byte-exactly.
//   (3) Tampered T_k: flipping one coefficient commitment changes the
//       fold output.  Catches "unused commitment" bugs in the verifier
//       check equation.
//   (4) Blinding freshness: across two commit_round_poly calls on the
//       same coefficients, ρ vectors are byte-distinct (the §4 proof of
//       ZK relies on independent ρ per round).
//   (5) Coverage of all degrees the Phase 3 driver needs: d ∈ {1, 2, 3,
//       4} for inner-product / hadamard / binary / multi-hadamard-K=4.
//
// Hyrax §4 Protocol 3 (Wahby et al. 2018, eprint 2017/1132, p. 11):
//   "In each round, P sends a Pedersen commitment to the round
//    polynomial g_j and proves consistency with the previous round."
//
// These are structural correctness tests for the commitment layer.
// The Σ-protocol consistency check is exercised in test_zk_sumcheck.

#include "proof/zk_round_commit.cuh"
#include "commit/commitment.cuh"
#include "poly/polynomial.cuh"
#include <iostream>
#include <cstring>
#include <cstdlib>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

static bool fr_bytes_eq(const Fr_t& a, const Fr_t& b) {
    return memcmp(&a, &b, sizeof(Fr_t)) == 0;
}

// Group-element equality.  Different Jacobian addition paths produce
// identical affine points but distinct (X:Y:Z) limbs.  Compare via
// subtraction and test for the identity (Z == 0).  Same convention as
// Commitment::verify_zk and src/proof/hyrax_sigma.cu.
static bool g1_pt_eq(const G1Jacobian_t& a, const G1Jacobian_t& b) {
    G1TensorJacobian A(1, a);
    G1TensorJacobian B(1, b);
    G1TensorJacobian D = A - B;
    G1Jacobian_t d = D(0);
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (d.z.val[i] != 0) return false;
    }
    return true;
}

// Byte-exact equality — only meaningful for same-path output (e.g.
// "two commit calls produced byte-distinct points → blindings differ").
static bool g1_bytes_eq(const G1Jacobian_t& a, const G1Jacobian_t& b) {
    return memcmp(&a, &b, sizeof(G1Jacobian_t)) == 0;
}

// Build C = m · U + r · H using size-1 commitment pipeline.
static G1Jacobian_t commit_mU_rH(G1Jacobian_t U, G1Jacobian_t H,
                                 Fr_t m, Fr_t r) {
    Commitment ppU(1, U);
    Commitment ppH(1, H);
    FrTensor M(1, &m), R(1, &r);
    G1TensorJacobian mU = ppU.commit(M);
    G1TensorJacobian rH = ppH.commit(R);
    return (mU + rH)(0);
}

// Scalar-side polynomial evaluation in coefficient form: p(r) = Σ c_k r^k.
// We use the existing Polynomial class for this so the test computes the
// reference value through an independent code path from fold_commitments_at.
static Fr_t eval_poly_at(const vector<Fr_t>& coeffs, Fr_t r) {
    Polynomial p(coeffs);
    return p(r);
}

// Run the (1)+(2)+(3) gauntlet for a specific degree.
static void test_at_degree(uint d, G1Jacobian_t U, G1Jacobian_t H) {
    // Random poly: d+1 coefficients.
    FrTensor coef_t = FrTensor::random(d + 1);
    vector<Fr_t> coeffs;
    for (uint k = 0; k <= d; k++) coeffs.push_back(coef_t(k));

    RoundCommitment rc = commit_round_poly(U, H, coeffs);
    check(rc.T.size() == d + 1, "commit_round_poly: T has d+1 entries");
    check(rc.rho.size() == d + 1, "commit_round_poly: rho has d+1 entries");

    // ── (1) commitment-side fold == scalar-side eval, then commit ──
    Fr_t r = FrTensor::random(1)(0);
    G1Jacobian_t fold_com = fold_commitments_at(rc.T, r);
    Fr_t p_r = eval_poly_at(coeffs, r);
    Fr_t rho_r = fold_blindings_at(rc.rho, r);
    G1Jacobian_t expected = commit_mU_rH(U, H, p_r, rho_r);
    char buf[128];
    snprintf(buf, sizeof(buf),
             "deg=%u: fold_commitments_at(r) == Com(p(r); rho(r))",
             d);
    check(g1_pt_eq(fold_com, expected), buf);

    // ── (2) sumcheck-identity LHS ──
    G1Jacobian_t sum_com = sumcheck_identity_lhs(rc.T);
    Fr_t zero = {0,0,0,0,0,0,0,0};
    Fr_t one  = {1,0,0,0,0,0,0,0};
    Fr_t p_0 = eval_poly_at(coeffs, zero);
    Fr_t p_1 = eval_poly_at(coeffs, one);
    Fr_t sum_msg = p_0 + p_1;
    Fr_t sum_rho = sumcheck_identity_blinding(rc.rho);
    G1Jacobian_t sum_expected = commit_mU_rH(U, H, sum_msg, sum_rho);
    snprintf(buf, sizeof(buf),
             "deg=%u: sumcheck_identity_lhs == Com(p(0)+p(1); 2rho_0+sum rho_k)",
             d);
    check(g1_pt_eq(sum_com, sum_expected), buf);

    // Cross-check: sumcheck_identity_lhs == fold_at(0) + fold_at(1).
    G1Jacobian_t at0 = fold_commitments_at(rc.T, zero);
    G1Jacobian_t at1 = fold_commitments_at(rc.T, one);
    G1TensorJacobian A0(1, at0), A1(1, at1);
    G1TensorJacobian A_sum = A0 + A1;
    snprintf(buf, sizeof(buf),
             "deg=%u: sumcheck_identity_lhs == fold_at(0) + fold_at(1)",
             d);
    check(g1_pt_eq(sum_com, A_sum(0)), buf);

    // ── (3) tamper T[k]: fold result changes ──
    // Replace one mid-degree coefficient with a different commitment.
    if (d >= 1) {
        RoundCommitment bad = rc;
        // Flip one coordinate in the *message*: commit (coef + 1) instead.
        Fr_t bumped = coeffs[d / 2] + one;
        bad.T[d / 2] = commit_mU_rH(U, H, bumped, bad.rho[d / 2]);
        G1Jacobian_t bad_fold = fold_commitments_at(bad.T, r);
        snprintf(buf, sizeof(buf),
                 "deg=%u: tampered T[d/2] changes fold_at(r) result",
                 d);
        check(!g1_pt_eq(bad_fold, fold_com), buf);
    }
}

int main() {
    Commitment pp = Commitment::hiding_random(1);
    G1Jacobian_t U = pp.u_generator;
    G1Jacobian_t H = pp.hiding_generator;

    // ── Degrees the Phase 3 driver needs ──────────────────────────────────
    // d=1: inner-product (after kernel — actually deg 2; keep d=1 as the
    //      simplest sanity case).
    // d=2: inner-product / hadamard / binary kernels emit out0,out1,out2.
    // d=3: multi-hadamard with K=3.
    // d=4: multi-hadamard with K=4 (largest in the entropy pipeline).
    for (uint d : {1u, 2u, 3u, 4u}) {
        test_at_degree(d, U, H);
    }

    // ── (4) blinding freshness across calls on the same coefficients ──────
    {
        FrTensor coef_t = FrTensor::random(3);
        vector<Fr_t> coeffs = {coef_t(0), coef_t(1), coef_t(2)};
        RoundCommitment a = commit_round_poly(U, H, coeffs);
        RoundCommitment b = commit_round_poly(U, H, coeffs);

        // All ρ entries must differ across the two calls.  A bug like
        // "ρ table generated once at module init" would make these equal.
        for (uint k = 0; k < a.rho.size(); k++) {
            char buf[80];
            snprintf(buf, sizeof(buf),
                     "blinding freshness: rho[%u] differs across calls", k);
            check(!fr_bytes_eq(a.rho[k], b.rho[k]), buf);
        }
        // T's must also differ (since same message, different ρ → different
        // commitment with overwhelming probability).
        for (uint k = 0; k < a.T.size(); k++) {
            char buf[80];
            snprintf(buf, sizeof(buf),
                     "blinding freshness: T[%u] differs across calls", k);
            check(!g1_bytes_eq(a.T[k], b.T[k]), buf);
        }
    }

    // ── (5) edge case: degree 0 (constant polynomial) ─────────────────────
    // The driver shouldn't see d=0 in practice (inner-product is at least
    // d=2), but the API should still behave: fold_at returns Com(c_0; rho_0).
    {
        FrTensor coef_t = FrTensor::random(1);
        vector<Fr_t> coeffs = {coef_t(0)};
        RoundCommitment rc = commit_round_poly(U, H, coeffs);
        Fr_t r = FrTensor::random(1)(0);
        G1Jacobian_t fold = fold_commitments_at(rc.T, r);
        check(g1_pt_eq(fold, rc.T[0]),
              "deg=0: fold_at(r) == T[0] for all r (constant poly)");
        // sumcheck_identity_lhs for d=0 gives 2·T[0] (= Com(2c_0; 2ρ_0)).
        G1Jacobian_t lhs = sumcheck_identity_lhs(rc.T);
        Fr_t two = {2,0,0,0,0,0,0,0};
        G1Jacobian_t expected = commit_mU_rH(U, H, two * coeffs[0], two * rc.rho[0]);
        check(g1_pt_eq(lhs, expected),
              "deg=0: sumcheck_identity_lhs == 2·Com(c_0; rho_0)");
    }

    cout << "All zk_round_commit tests PASSED." << endl;
    return 0;
}
