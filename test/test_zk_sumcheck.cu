// Tests for Phase 3 subcomponent (3): ZK sumcheck driver.
//
// Hyrax §4 Protocol 3 (Wahby et al. 2018, eprint 2017/1132, p. 11–13):
// the prover replaces plain round-polynomial scalars with Pedersen
// commitments and ties consecutive rounds via §A.1 proof-of-equality.
// Per §4 also requires one proof-of-opening per coefficient
// commitment; verifier-side Step 1 constructs C_0 = S·U directly from
// the public claim.
//
// Coverage:
//   (1) Positive roundtrip per variant at N ∈ {4, 16, 64, 256}.
//   (2) Negatives per variant — one per verifier check-point
//       (per-coef opening, round-to-round equality, C_0 binding).
//   (3) Composition and blinding-freshness smoke tests.

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

// Public claim Σ_x a(x)·b(x) for inner-product, computed host-side via
// the Fr_t operator* (independent of the GPU kernel, so a kernel bug
// would diverge).
static Fr_t inner_product_claim(const FrTensor& a, const FrTensor& b) {
    if (a.size != b.size) throw std::runtime_error("|a| != |b|");
    Fr_t acc = {0,0,0,0,0,0,0,0};
    for (uint i = 0; i < a.size; i++) {
        acc = acc + (a(i) * b(i));
    }
    return acc;
}

// Σ_x a(x)·(a(x)-1) — binary sumcheck claim.  Zero when a ∈ {0,1}^N.
static Fr_t binary_claim(const FrTensor& a) {
    Fr_t acc = {0,0,0,0,0,0,0,0};
    Fr_t one = {1,0,0,0,0,0,0,0};
    for (uint i = 0; i < a.size; i++) {
        Fr_t v = a(i);
        acc = acc + (v * (v - one));
    }
    return acc;
}

// Public claim S = (a∘b)(u) for eq-factored HP.
static Fr_t hadamard_claim(const FrTensor& a, const FrTensor& b,
                           const std::vector<Fr_t>& u) {
    if (a.size != b.size) throw std::runtime_error("|a| != |b|");
    std::vector<Fr_t> h(a.size);
    for (uint i = 0; i < a.size; i++) h[i] = a(i) * b(i);
    FrTensor h_t(a.size, h.data());
    return h_t(u);
}

// Sigma-challenge layout: per round consumes 4 for degree-2 variants
// (3 per-coef openings + 1 equality).  Total = n · 4.
static constexpr uint kSigmaPerRound = 4;

// ── positive roundtrips ──

static void positive_inner_product(uint N, G1Jacobian_t U, G1Jacobian_t H) {
    uint n = ceil_log2(N);
    if ((1u << n) != N) throw std::runtime_error("N must be power of 2");

    FrTensor a = FrTensor::random(N);
    FrTensor b = FrTensor::random(N);
    Fr_t S = inner_product_claim(a, b);

    FrTensor evals  = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n * kSigmaPerRound);
    std::vector<Fr_t> ev, sg;
    for (uint i = 0; i < n;                 i++) ev.push_back(evals(i));
    for (uint i = 0; i < n * kSigmaPerRound; i++) sg.push_back(sigmas(i));

    Fr_t final_a, final_b;
    ZKSumcheckProverHandoff handoff;
    auto proof = prove_zk_inner_product(U, H, S, a, b, ev, sg,
                                        final_a, final_b, handoff);

    char buf[128];
    snprintf(buf, sizeof(buf), "ip N=%u: prover emits n=%u rounds", N, n);
    check(proof.rounds.size() == n, buf);

    for (uint j = 0; j < proof.rounds.size(); j++) {
        snprintf(buf, sizeof(buf),
                 "ip N=%u: round %u has 3 commitments + 3 openings", N, j);
        check(proof.rounds[j].T.size() == 3 &&
              proof.rounds[j].T_open.size() == 3, buf);
    }

    snprintf(buf, sizeof(buf), "ip N=%u: honest verifier accepts", N);
    check(verify_zk_inner_product(U, H, S, proof, ev, sg), buf);

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

    std::vector<Fr_t> bits;
    Fr_t zero = {0,0,0,0,0,0,0,0};
    Fr_t one  = {1,0,0,0,0,0,0,0};
    FrTensor noise = FrTensor::random(N);
    for (uint i = 0; i < N; i++) {
        bits.push_back((noise(i).val[0] & 1) ? one : zero);
    }
    FrTensor a(N, bits.data());
    Fr_t S = binary_claim(a);  // == 0

    FrTensor evals  = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n * kSigmaPerRound);
    std::vector<Fr_t> ev, sg;
    for (uint i = 0; i < n;                  i++) ev.push_back(evals(i));
    for (uint i = 0; i < n * kSigmaPerRound; i++) sg.push_back(sigmas(i));

    Fr_t final_a;
    ZKSumcheckProverHandoff handoff;
    auto proof = prove_zk_binary(U, H, S, a, ev, sg, final_a, handoff);

    char buf[128];
    snprintf(buf, sizeof(buf), "bin N=%u: honest verifier accepts", N);
    check(verify_zk_binary(U, H, S, proof, ev, sg), buf);

    Fr_t fr_one = {1,0,0,0,0,0,0,0};
    Fr_t expected_msg = final_a * (final_a - fr_one);
    G1Jacobian_t T_expected = commit_mU_rH(U, H, expected_msg,
                                           handoff.rho_final);
    snprintf(buf, sizeof(buf),
             "bin N=%u: T_final == Com(a(r)·(a(r)-1); rho_final)", N);
    check(g1_pt_eq(proof.T_final, T_expected), buf);
}

static void positive_hadamard(uint N, G1Jacobian_t U, G1Jacobian_t H) {
    uint n = ceil_log2(N);
    if ((1u << n) != N) throw std::runtime_error("N must be power of 2");

    FrTensor a = FrTensor::random(N);
    FrTensor b = FrTensor::random(N);

    FrTensor u_t   = FrTensor::random(n);
    FrTensor v_t   = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n * kSigmaPerRound);
    std::vector<Fr_t> u, v, sg;
    for (uint i = 0; i < n;                  i++) u.push_back(u_t(i));
    for (uint i = 0; i < n;                  i++) v.push_back(v_t(i));
    for (uint i = 0; i < n * kSigmaPerRound; i++) sg.push_back(sigmas(i));

    Fr_t S = hadamard_claim(a, b, u);

    Fr_t fa, fb;
    ZKSumcheckProverHandoff handoff;
    auto proof = prove_zk_hadamard_product(U, H, S, a, b, u, v, sg,
                                           fa, fb, handoff);

    char buf[128];
    snprintf(buf, sizeof(buf), "hp N=%u: honest verifier accepts", N);
    check(verify_zk_hadamard_product(U, H, S, proof, u, v, sg), buf);

    Fr_t expected_msg = fa * fb;
    G1Jacobian_t expected = commit_mU_rH(U, H, expected_msg, handoff.rho_final);
    snprintf(buf, sizeof(buf),
             "hp N=%u: T_final == Com(a(v)·b(v); rho_final)", N);
    check(g1_pt_eq(proof.T_final, expected), buf);
}

// ── negative helpers ──

static bool verifier_throws(std::function<bool()> fn, const char* expected_substr) {
    try {
        fn();
        return false;
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
    FrTensor sigmas = FrTensor::random(n * kSigmaPerRound);
    std::vector<Fr_t> ev, sg;
    for (uint i = 0; i < n;                  i++) ev.push_back(evals(i));
    for (uint i = 0; i < n * kSigmaPerRound; i++) sg.push_back(sigmas(i));

    Fr_t fa, fb;
    ZKSumcheckProverHandoff handoff;
    auto good = prove_zk_inner_product(U, H, S, a, b, ev, sg, fa, fb, handoff);

    char buf[128];

    // (a) tamper one T_j[k] — the corresponding per-coef opening rejects.
    {
        auto bad = good;
        uint j = bad.rounds.size() / 2;
        bad.rounds[j].T[1] = U;
        snprintf(buf, sizeof(buf),
                 "ip N=%u neg: tampered T_j[k] rejected at per-coef opening", N);
        check(verifier_throws([&]() {
            return verify_zk_inner_product(U, H, S, bad, ev, sg);
        }, "proof-of-opening rejected"), buf);
    }
    // (b) tamper a T_open response — that coefficient's opening check fails.
    {
        auto bad = good;
        uint j = bad.rounds.size() / 2;
        Fr_t one = {1,0,0,0,0,0,0,0};
        bad.rounds[j].T_open[0].z_m = bad.rounds[j].T_open[0].z_m + one;
        snprintf(buf, sizeof(buf),
                 "ip N=%u neg: tampered T_open.z_m rejected", N);
        check(verifier_throws([&]() {
            return verify_zk_inner_product(U, H, S, bad, ev, sg);
        }, "proof-of-opening rejected"), buf);
    }
    // (c) tamper an eq_proof response.
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
    // (d) wrong claimed_S at verify time — round 0's equality ties
    // Σ α_k · T_1[k] to S·U; any S' ≠ S breaks it.  This is the G2
    // binding fix: S is now directly bound to the transcript via the
    // verifier-constructed C_0 = S·U.
    {
        Fr_t one = {1,0,0,0,0,0,0,0};
        Fr_t S_wrong = S + one;
        snprintf(buf, sizeof(buf),
                 "ip N=%u neg: wrong claimed_S rejected at round-0 equality", N);
        check(verifier_throws([&]() {
            return verify_zk_inner_product(U, H, S_wrong, good, ev, sg);
        }, "proof-of-equality rejected"), buf);
    }
    // (e) wrong sumcheck challenge at one round propagates rejection.
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

static void negatives_hadamard(uint N, G1Jacobian_t U, G1Jacobian_t H) {
    uint n = ceil_log2(N);
    if (n < 1) return;
    FrTensor a = FrTensor::random(N);
    FrTensor b = FrTensor::random(N);

    FrTensor u_t = FrTensor::random(n);
    FrTensor v_t = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n * kSigmaPerRound);
    std::vector<Fr_t> u, v, sg;
    for (uint i = 0; i < n;                  i++) u.push_back(u_t(i));
    for (uint i = 0; i < n;                  i++) v.push_back(v_t(i));
    for (uint i = 0; i < n * kSigmaPerRound; i++) sg.push_back(sigmas(i));

    Fr_t S = hadamard_claim(a, b, u);

    Fr_t fa, fb;
    ZKSumcheckProverHandoff handoff;
    auto good = prove_zk_hadamard_product(U, H, S, a, b, u, v, sg, fa, fb, handoff);

    char buf[128];
    // Swap u and v at verify — eq-factor weight (1, u_j, u_j) becomes
    // (1, v_j, v_j), breaking the round-to-round identity.
    {
        snprintf(buf, sizeof(buf),
                 "hp N=%u neg: swapped u/v at verify rejected", N);
        check(verifier_throws([&]() {
            return verify_zk_hadamard_product(U, H, S, good, v, u, sg);
        }, "proof-of-equality rejected"), buf);
    }
    // Flip u[0] — round 0's alpha weights mismatch between prover and
    // verifier.
    {
        std::vector<Fr_t> u_bad = u;
        Fr_t one = {1,0,0,0,0,0,0,0};
        u_bad[0] = u_bad[0] + one;
        snprintf(buf, sizeof(buf),
                 "hp N=%u neg: wrong u[0] rejected at round 0 eq-proof", N);
        check(verifier_throws([&]() {
            return verify_zk_hadamard_product(U, H, S, good, u_bad, v, sg);
        }, "proof-of-equality rejected"), buf);
    }
    // Wrong claimed_S: round 0 ties Σ α_k · T_1[k] to S·U, so any
    // S' ≠ S fails.  With the G2 fix (verifier-constructed C_0) the
    // claim is strictly bound to the transcript.
    {
        Fr_t one = {1,0,0,0,0,0,0,0};
        Fr_t S_wrong = S + one;
        snprintf(buf, sizeof(buf),
                 "hp N=%u neg: wrong claimed_S rejected at round-0 equality", N);
        check(verifier_throws([&]() {
            return verify_zk_hadamard_product(U, H, S_wrong, good, u, v, sg);
        }, "proof-of-equality rejected"), buf);
    }
}

// Public claim S = (Π_k X_k)(u) for multi-Hadamard.
static Fr_t multi_hadamard_claim(const std::vector<FrTensor>& Xs,
                                 const std::vector<Fr_t>& u) {
    uint N = Xs[0].size;
    std::vector<Fr_t> prod(N);
    for (uint i = 0; i < N; i++) {
        Fr_t acc = Xs[0](i);
        for (uint k = 1; k < Xs.size(); k++) acc = acc * Xs[k](i);
        prod[i] = acc;
    }
    FrTensor h(N, prod.data());
    return h(u);
}

static void positive_multi_hadamard(uint N, uint K,
                                    G1Jacobian_t U, G1Jacobian_t H) {
    uint n = ceil_log2(N);
    if ((1u << n) != N) throw std::runtime_error("N must be power of 2");
    const uint sigma_per_round = K + 2;

    std::vector<FrTensor> Xs;
    Xs.reserve(K);
    for (uint k = 0; k < K; k++) Xs.push_back(FrTensor::random(N));

    FrTensor u_t = FrTensor::random(n);
    FrTensor v_t = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n * sigma_per_round);
    std::vector<Fr_t> u, v, sg;
    for (uint i = 0; i < n;                  i++) u.push_back(u_t(i));
    for (uint i = 0; i < n;                  i++) v.push_back(v_t(i));
    for (uint i = 0; i < n * sigma_per_round; i++) sg.push_back(sigmas(i));

    Fr_t S = multi_hadamard_claim(Xs, u);

    std::vector<Fr_t> final_Xs;
    ZKSumcheckProverHandoff handoff;
    auto proof = prove_zk_multi_hadamard(U, H, S, Xs, u, v, sg,
                                         final_Xs, handoff);

    char buf[128];
    snprintf(buf, sizeof(buf),
             "mhp N=%u K=%u: prover emits n=%u rounds", N, K, n);
    check(proof.rounds.size() == n, buf);

    for (uint j = 0; j < proof.rounds.size(); j++) {
        snprintf(buf, sizeof(buf),
                 "mhp N=%u K=%u: round %u has K+1=%u commitments+openings",
                 N, K, j, K + 1);
        check(proof.rounds[j].T.size() == K + 1 &&
              proof.rounds[j].T_open.size() == K + 1, buf);
    }

    snprintf(buf, sizeof(buf),
             "mhp N=%u K=%u: honest verifier accepts", N, K);
    check(verify_zk_multi_hadamard(U, H, S, proof, K, u, v, sg), buf);

    // T_final commits to Π_k X_k(v) with blinding rho_final.  The
    // final_Xs returned by the prover are the contracted scalars
    // X_k(v) — Phase 2 discharge folds these via proof-of-product.
    check(final_Xs.size() == K, "mhp: final_Xs.size() == K");
    Fr_t prod = final_Xs[0];
    for (uint k = 1; k < K; k++) prod = prod * final_Xs[k];
    G1Jacobian_t expected = commit_mU_rH(U, H, prod, handoff.rho_final);
    snprintf(buf, sizeof(buf),
             "mhp N=%u K=%u: T_final == Com(Π_k X_k(v); rho_final)", N, K);
    check(g1_pt_eq(proof.T_final, expected), buf);
}

static void negatives_multi_hadamard(uint N, uint K,
                                     G1Jacobian_t U, G1Jacobian_t H) {
    uint n = ceil_log2(N);
    if (n < 1) return;
    const uint sigma_per_round = K + 2;

    std::vector<FrTensor> Xs;
    Xs.reserve(K);
    for (uint k = 0; k < K; k++) Xs.push_back(FrTensor::random(N));

    FrTensor u_t = FrTensor::random(n);
    FrTensor v_t = FrTensor::random(n);
    FrTensor sigmas = FrTensor::random(n * sigma_per_round);
    std::vector<Fr_t> u, v, sg;
    for (uint i = 0; i < n;                  i++) u.push_back(u_t(i));
    for (uint i = 0; i < n;                  i++) v.push_back(v_t(i));
    for (uint i = 0; i < n * sigma_per_round; i++) sg.push_back(sigmas(i));

    Fr_t S = multi_hadamard_claim(Xs, u);

    std::vector<Fr_t> final_Xs;
    ZKSumcheckProverHandoff handoff;
    auto good = prove_zk_multi_hadamard(U, H, S, Xs, u, v, sg,
                                        final_Xs, handoff);

    char buf[128];
    // Tamper a coefficient commitment → per-coef opening rejects.
    {
        auto bad = good;
        uint j = bad.rounds.size() / 2;
        bad.rounds[j].T[1] = U;
        snprintf(buf, sizeof(buf),
                 "mhp N=%u K=%u neg: tampered T_j[1] rejected at opening",
                 N, K);
        check(verifier_throws([&]() {
            return verify_zk_multi_hadamard(U, H, S, bad, K, u, v, sg);
        }, "proof-of-opening rejected"), buf);
    }
    // Tamper an eq-proof response → round-to-round equality rejects.
    {
        auto bad = good;
        uint j = bad.rounds.size() / 2;
        Fr_t one = {1,0,0,0,0,0,0,0};
        bad.rounds[j].eq_proof.z = bad.rounds[j].eq_proof.z + one;
        snprintf(buf, sizeof(buf),
                 "mhp N=%u K=%u neg: tampered eq_proof.z rejected", N, K);
        check(verifier_throws([&]() {
            return verify_zk_multi_hadamard(U, H, S, bad, K, u, v, sg);
        }, "proof-of-equality rejected"), buf);
    }
    // Wrong claimed_S → round 0 ties Σα·T to S·U; any S' ≠ S fails.
    {
        Fr_t one = {1,0,0,0,0,0,0,0};
        Fr_t S_wrong = S + one;
        snprintf(buf, sizeof(buf),
                 "mhp N=%u K=%u neg: wrong claimed_S rejected at round-0",
                 N, K);
        check(verifier_throws([&]() {
            return verify_zk_multi_hadamard(U, H, S_wrong, good, K, u, v, sg);
        }, "proof-of-equality rejected"), buf);
    }
    // Wrong K at verify → sigma-size precondition triggers.
    {
        snprintf(buf, sizeof(buf),
                 "mhp N=%u K=%u neg: wrong K at verify rejected", N, K);
        check(verifier_throws([&]() {
            return verify_zk_multi_hadamard(U, H, S, good, K + 1, u, v, sg);
        }, "sigma_challenges must have size"), buf);
    }
}

int main() {
    Commitment pp = Commitment::hiding_random(1);
    G1Jacobian_t U = pp.u_generator;
    G1Jacobian_t H = pp.hiding_generator;

    // ── positive roundtrips at multiple sizes ──
    for (uint N : {4u, 16u, 64u, 256u}) {
        positive_inner_product(N, U, H);
        positive_binary(N, U, H);
        positive_hadamard(N, U, H);
    }

    // ── multi-HP positives at K ∈ {2, 3, 4} ──
    for (uint K : {2u, 3u, 4u}) {
        for (uint N : {4u, 16u, 64u}) {
            positive_multi_hadamard(N, K, U, H);
        }
    }

    // ── negatives at one representative size ──
    negatives_inner_product(64u, U, H);
    negatives_hadamard(64u, U, H);
    for (uint K : {2u, 3u, 4u}) {
        negatives_multi_hadamard(64u, K, U, H);
    }

    // ── composition ──
    {
        positive_inner_product(16u, U, H);
        positive_inner_product(16u, U, H);
        cout << "PASS: composition: independent runs both verify" << endl;
    }

    // ── blinding freshness across two prover runs ──
    {
        uint N = 16u, n = ceil_log2(N);
        FrTensor a = FrTensor::random(N);
        FrTensor b = FrTensor::random(N);
        Fr_t S = inner_product_claim(a, b);

        FrTensor ev_t = FrTensor::random(n);
        FrTensor sg_t = FrTensor::random(n * kSigmaPerRound);
        std::vector<Fr_t> ev, sg;
        for (uint i = 0; i < n;                  i++) ev.push_back(ev_t(i));
        for (uint i = 0; i < n * kSigmaPerRound; i++) sg.push_back(sg_t(i));

        Fr_t fa, fb;
        ZKSumcheckProverHandoff h1, h2;
        auto p1 = prove_zk_inner_product(U, H, S, a, b, ev, sg, fa, fb, h1);
        auto p2 = prove_zk_inner_product(U, H, S, a, b, ev, sg, fa, fb, h2);

        // Per-round coefficient commitments use fresh blindings → differ
        // across runs.  Round 0's T[0] is a sufficient witness.
        check(!g1_bytes_eq(p1.rounds[0].T[0], p2.rounds[0].T[0]),
              "freshness: round 0 T[0] differs across runs (rho fresh)");
        for (uint j = 0; j < p1.rounds.size(); j++) {
            char buf[64];
            snprintf(buf, sizeof(buf),
                     "freshness: T_%u[0] differs across runs", j);
            check(!g1_bytes_eq(p1.rounds[j].T[0], p2.rounds[j].T[0]), buf);
        }
    }

    cout << "All zk_sumcheck tests PASSED." << endl;
    return 0;
}
