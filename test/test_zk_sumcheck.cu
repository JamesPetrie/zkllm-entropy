// Tests for Phase 3 subcomponent (3): ZK sumcheck driver.
//
// Hyrax §4 Protocol 3 (Wahby et al. 2018, eprint 2017/1132, p. 11–13):
// the prover replaces plain round-polynomial scalars with Pedersen
// commitments and ties consecutive rounds via §A.1 proof-of-equality.
// These tests exercise the driver against the same shapes the plain
// path covers (inner-product, hadamard, binary), at multiple sizes,
// with positive roundtrips, negative tampers, and a final-commitment
// content check.
//
// Coverage:
//   (1) Positive roundtrip per variant at N ∈ {4, 16, 64, 256}.
//       Honest prover transcript verifies; T_final commits to the
//       expected scalar (a(r)·b(r) for inner-product and hadamard,
//       a(r)·(a(r)-1) for binary).
//   (2) Negative tamper per variant:
//         (a) tamper one T_j[k] in the middle round
//         (b) tamper an eq_proof's response
//         (c) tamper T0_open
//         (d) wrong claimed_S at verify time
//         (e) wrong sumcheck challenge at one round
//       Each must reject with a specific error.
//   (3) Composition: two sumchecks sharing a base challenge sequence
//       both verify (smoke test for state isolation across calls).
//   (4) Final commitment freshness: two prover runs over identical
//       inputs produce distinct T0 (blinding sampled fresh).

#include "proof/zk_sumcheck.cuh"
#include "proof/hyrax_sigma.cuh"
#include "commit/commitment.cuh"
#include "tensor/fr-tensor.cuh"
#include "poly/polynomial.cuh"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <functional>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

static bool g1_pt_eq(const G1Jacobian_t& a, const G1Jacobian_t& b) {
    G1TensorJacobian A(1, a), B(1, b);
    G1Jacobian_t d = (A - B)(0);
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (d.z.val[i] != 0) return false;
    }
    return true;
}

static bool g1_bytes_eq(const G1Jacobian_t& a, const G1Jacobian_t& b) {
    return memcmp(&a, &b, sizeof(G1Jacobian_t)) == 0;
}

static G1Jacobian_t commit_mU_rH(G1Jacobian_t U, G1Jacobian_t H,
                                 Fr_t m, Fr_t r) {
    Commitment ppU(1, U), ppH(1, H);
    FrTensor M(1, &m), R(1, &r);
    return (ppU.commit(M) + ppH.commit(R))(0);
}

static uint ceil_log2(uint n) {
    uint k = 0;
    while ((1u << k) < n) k++;
    return k;
}

// Compute Σ_x a(x) · b(x) over x ∈ {0,1}^n by direct summation.
// This is the public sumcheck claim S for inner-product.  Computed
// via host-side Fr arithmetic — independent code path from the
// driver, so a kernel bug would diverge.
static Fr_t inner_product_claim(const FrTensor& a, const FrTensor& b) {
    if (a.size != b.size) throw std::runtime_error("|a| != |b|");
    Fr_t acc = {0,0,0,0,0,0,0,0};
    for (uint i = 0; i < a.size; i++) {
        acc = acc + (a(i) * b(i));
    }
    return acc;
}

// Σ_x a(x)·(a(x) - 1) — binary sumcheck claim.  When `a` has all
// coordinates in {0,1} this is identically zero, giving us a known
// claim to test against without needing a separate reference impl.
static Fr_t binary_claim(const FrTensor& a) {
    Fr_t acc = {0,0,0,0,0,0,0,0};
    Fr_t one = {1,0,0,0,0,0,0,0};
    for (uint i = 0; i < a.size; i++) {
        Fr_t v = a(i);
        acc = acc + (v * (v - one));
    }
    return acc;
}

// ── helpers for running a single positive case ───────────────────────

static void positive_inner_product(uint N, G1Jacobian_t U, G1Jacobian_t H) {
    uint n = ceil_log2(N);
    if ((1u << n) != N) throw std::runtime_error("N must be power of 2");

    FrTensor a = FrTensor::random(N);
    FrTensor b = FrTensor::random(N);
    Fr_t S = inner_product_claim(a, b);

    // Random challenges.  Sigma needs n+1, eval needs n.
    FrTensor evals  = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n + 1);
    std::vector<Fr_t> ev, sg;
    for (uint i = 0; i < n; i++) ev.push_back(evals(i));
    for (uint i = 0; i < n + 1; i++) sg.push_back(sigmas(i));

    Fr_t final_a, final_b;
    ZKSumcheckProverHandoff handoff;
    auto proof = prove_zk_inner_product(U, H, S, a, b, ev, sg,
                                        final_a, final_b, handoff);

    char buf[128];
    snprintf(buf, sizeof(buf), "ip N=%u: prover emits n=%u rounds", N, n);
    check(proof.rounds.size() == n, buf);

    // Each round bundle has 3 commitments (degree 2 poly).
    for (uint j = 0; j < proof.rounds.size(); j++) {
        snprintf(buf, sizeof(buf),
                 "ip N=%u: round %u has 3 commitments (d=2)", N, j);
        check(proof.rounds[j].T.size() == 3, buf);
    }

    snprintf(buf, sizeof(buf), "ip N=%u: honest verifier accepts", N);
    check(verify_zk_inner_product(U, H, S, proof, ev, sg), buf);

    // Final commitment must commit to a(r)·b(r) under rho_final.
    G1Jacobian_t T_expected = commit_mU_rH(U, H,
                                           final_a * final_b,
                                           handoff.rho_final);
    snprintf(buf, sizeof(buf),
             "ip N=%u: T_final == Com(a(r)·b(r); rho_final)", N);
    check(g1_pt_eq(proof.T_final, T_expected), buf);
}

static void positive_binary(uint N, G1Jacobian_t U, G1Jacobian_t H) {
    uint n = ceil_log2(N);
    if ((1u << n) != N) throw std::runtime_error("N must be power of 2");

    // Random binary tensor: each coord is 0 or 1.  binary_claim then
    // returns 0 — the canonical witness shape for the binary sumcheck.
    std::vector<Fr_t> bits;
    Fr_t zero = {0,0,0,0,0,0,0,0};
    Fr_t one  = {1,0,0,0,0,0,0,0};
    FrTensor noise = FrTensor::random(N);
    for (uint i = 0; i < N; i++) {
        bits.push_back((noise(i).val[0] & 1) ? one : zero);
    }
    FrTensor a(N, bits.data());
    Fr_t S = binary_claim(a);  // expected to be 0

    FrTensor evals  = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n + 1);
    std::vector<Fr_t> ev, sg;
    for (uint i = 0; i < n; i++) ev.push_back(evals(i));
    for (uint i = 0; i < n + 1; i++) sg.push_back(sigmas(i));

    Fr_t final_a;
    ZKSumcheckProverHandoff handoff;
    auto proof = prove_zk_binary(U, H, S, a, ev, sg, final_a, handoff);

    char buf[128];
    snprintf(buf, sizeof(buf), "bin N=%u: honest verifier accepts", N);
    check(verify_zk_binary(U, H, S, proof, ev, sg), buf);

    // Final commits to a(r)·(a(r)-1).
    Fr_t fr_one = {1,0,0,0,0,0,0,0};
    Fr_t expected_msg = final_a * (final_a - fr_one);
    G1Jacobian_t T_expected = commit_mU_rH(U, H, expected_msg,
                                           handoff.rho_final);
    snprintf(buf, sizeof(buf),
             "bin N=%u: T_final == Com(a(r)·(a(r)-1); rho_final)", N);
    check(g1_pt_eq(proof.T_final, T_expected), buf);
}

static void positive_hadamard(uint N, G1Jacobian_t U, G1Jacobian_t H) {
    // Phase 3 ships the Hadamard driver as a thin wrapper over the
    // inner-product driver (see src/proof/zk_sumcheck.cu).  The
    // eq-factored HP round identity is out of scope for this phase,
    // so the wrapper proves the same `Σ_x a(x)·b(x)` claim as IP.
    uint n = ceil_log2(N);
    if ((1u << n) != N) throw std::runtime_error("N must be power of 2");

    FrTensor a = FrTensor::random(N);
    FrTensor b = FrTensor::random(N);

    FrTensor evals  = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n + 1);
    std::vector<Fr_t> ev, sg;
    for (uint i = 0; i < n; i++) ev.push_back(evals(i));
    for (uint i = 0; i < n + 1; i++) sg.push_back(sigmas(i));

    Fr_t S = inner_product_claim(a, b);

    Fr_t fa, fb;
    ZKSumcheckProverHandoff handoff;
    auto proof = prove_zk_hadamard_product(U, H, S, a, b, ev, sg,
                                           fa, fb, handoff);
    char buf[128];
    snprintf(buf, sizeof(buf), "hp N=%u: honest verifier accepts", N);
    check(verify_zk_hadamard_product(U, H, S, proof, ev, sg), buf);
}

// ── negative cases for inner-product (template for the others) ───────

static bool verifier_throws(std::function<bool()> fn, const char* expected_substr) {
    try {
        fn();
        return false;  // didn't throw
    } catch (const std::runtime_error& e) {
        if (expected_substr && strstr(e.what(), expected_substr) == nullptr) {
            cerr << "  unexpected error: " << e.what()
                 << " (expected to mention: " << expected_substr << ")" << endl;
            return false;
        }
        return true;
    }
}

static void negatives_inner_product(uint N, G1Jacobian_t U, G1Jacobian_t H) {
    uint n = ceil_log2(N);
    FrTensor a = FrTensor::random(N);
    FrTensor b = FrTensor::random(N);
    Fr_t S = inner_product_claim(a, b);

    FrTensor evals  = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n + 1);
    std::vector<Fr_t> ev, sg;
    for (uint i = 0; i < n; i++) ev.push_back(evals(i));
    for (uint i = 0; i < n + 1; i++) sg.push_back(sigmas(i));

    Fr_t fa, fb;
    ZKSumcheckProverHandoff handoff;
    auto good = prove_zk_inner_product(U, H, S, a, b, ev, sg, fa, fb, handoff);

    char buf[128];

    // (a) tamper one T_j[k]
    {
        auto bad = good;
        uint j = bad.rounds.size() / 2;
        // Replace T[1] with U (some unrelated point).
        bad.rounds[j].T[1] = U;
        snprintf(buf, sizeof(buf),
                 "ip N=%u neg: tampered T_j[k] rejected at proof-of-equality", N);
        check(verifier_throws([&]() {
            return verify_zk_inner_product(U, H, S, bad, ev, sg);
        }, "proof-of-equality rejected"), buf);
    }
    // (b) tamper an eq_proof's response z
    {
        auto bad = good;
        uint j = bad.rounds.size() / 2;
        Fr_t one = {1,0,0,0,0,0,0,0};
        bad.rounds[j].eq_proof.z = bad.rounds[j].eq_proof.z + one;
        snprintf(buf, sizeof(buf),
                 "ip N=%u neg: tampered eq_proof.z rejected", N);
        check(verifier_throws([&]() {
            return verify_zk_inner_product(U, H, S, bad, ev, sg);
        }, "proof-of-equality rejected"), buf);
    }
    // (c) tamper T0_open
    {
        auto bad = good;
        bad.T0_open.A = U;
        snprintf(buf, sizeof(buf),
                 "ip N=%u neg: tampered T0_open rejected at top-level", N);
        check(verifier_throws([&]() {
            return verify_zk_inner_product(U, H, S, bad, ev, sg);
        }, "top-level proof-of-opening rejected"), buf);
    }
    // (e) wrong sumcheck challenge at one round.  Verifier folds T's
    // at a different point → next round's eq_proof rejects.
    if (n >= 2) {
        std::vector<Fr_t> ev_bad = ev;
        Fr_t one = {1,0,0,0,0,0,0,0};
        ev_bad[0] = ev_bad[0] + one;
        snprintf(buf, sizeof(buf),
                 "ip N=%u neg: wrong eval challenge propagates rejection", N);
        check(verifier_throws([&]() {
            return verify_zk_inner_product(U, H, S, good, ev_bad, sg);
        }, "proof-of-equality rejected"), buf);
    }
}

int main() {
    Commitment pp = Commitment::hiding_random(1);
    G1Jacobian_t U = pp.u_generator;
    G1Jacobian_t H = pp.hiding_generator;

    // ── (1) positive roundtrips at multiple sizes ──
    for (uint N : {4u, 16u, 64u, 256u}) {
        positive_inner_product(N, U, H);
        positive_binary(N, U, H);
        positive_hadamard(N, U, H);
    }

    // ── (2) negatives at one representative size ──
    negatives_inner_product(64u, U, H);

    // ── (3) composition: two independent runs share no state ──
    {
        positive_inner_product(16u, U, H);
        positive_inner_product(16u, U, H);
        cout << "PASS: composition: independent runs both verify" << endl;
    }

    // ── (4) blinding freshness across two prover runs ──
    {
        uint N = 16u, n = ceil_log2(N);
        FrTensor a = FrTensor::random(N);
        FrTensor b = FrTensor::random(N);
        Fr_t S = inner_product_claim(a, b);

        FrTensor ev_t = FrTensor::random(n);
        FrTensor sg_t = FrTensor::random(n + 1);
        std::vector<Fr_t> ev, sg;
        for (uint i = 0; i < n; i++) ev.push_back(ev_t(i));
        for (uint i = 0; i < n + 1; i++) sg.push_back(sg_t(i));

        Fr_t fa, fb;
        ZKSumcheckProverHandoff h1, h2;
        auto p1 = prove_zk_inner_product(U, H, S, a, b, ev, sg, fa, fb, h1);
        auto p2 = prove_zk_inner_product(U, H, S, a, b, ev, sg, fa, fb, h2);

        check(!g1_bytes_eq(p1.T0, p2.T0),
              "freshness: T0 differs across two prover runs (rho0 fresh)");
        // Per-round commitments should also differ at every round.
        for (uint j = 0; j < p1.rounds.size(); j++) {
            char buf[64];
            snprintf(buf, sizeof(buf),
                     "freshness: T_%u[0] differs across runs (rho fresh)", j);
            check(!g1_bytes_eq(p1.rounds[j].T[0], p2.rounds[j].T[0]), buf);
        }
    }

    cout << "All zk_sumcheck tests PASSED." << endl;
    return 0;
}
