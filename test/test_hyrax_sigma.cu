// Tests for Phase 3 Σ-protocols (Hyrax §A.1 Figure 5).
//
// Covers:
//   (1) proof-of-opening roundtrip: prove then verify accepts.
//   (2) proof-of-opening negative: tampered A, z_m, z_r, or wrong e
//       each causes verify to reject.
//   (3) proof-of-opening soundness on wrong witness: prover who does
//       not know (m, r) cannot forge an accepting transcript.  We
//       simulate this by feeding a commitment to a different (m', r').
//   (4) proof-of-equality roundtrip: C1 = m·U + r1·H, C2 = m·U + r2·H.
//   (5) proof-of-equality negative: tampered A, z, or wrong e rejects.
//   (6) proof-of-equality soundness on non-equal messages: if C1 and
//       C2 commit to different m's, no (r1, r2) makes verify accept
//       for a fresh challenge.
//   (7) Mask freshness: two calls to prove_opening with the same
//       witness produce different A's (s_m, s_r sampling is fresh).
//
// Hyrax (Wahby et al. 2018, eprint 2017/1132), §A.1 Figure 5, p. 17.
// The interactive model matches Phase 2's open_zk/verify_zk: the
// verifier's challenge `e` is caller-supplied.  Phase 5 wraps this
// in Fiat-Shamir.

#include "proof/hyrax_sigma.cuh"
#include "commit/commitment.cuh"
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

static bool g1_bytes_eq(const G1Jacobian_t& a, const G1Jacobian_t& b) {
    return memcmp(&a, &b, sizeof(G1Jacobian_t)) == 0;
}

// Build a single-scalar Pedersen commitment C = m·U + r·H using the
// commit/commit_hiding primitives.  Returns (C, r) via out-params.
static G1Jacobian_t commit_mU_rH(G1Jacobian_t U, G1Jacobian_t H,
                                 Fr_t m, Fr_t r) {
    // m·U
    Commitment ppU(1, U);
    FrTensor M(1, &m);
    G1TensorJacobian mU = ppU.commit(M);
    // r·H
    Commitment ppH(1, H);
    FrTensor R(1, &r);
    G1TensorJacobian rH = ppH.commit(R);
    // Sum.
    G1TensorJacobian C = mU + rH;
    return C(0);
}

int main() {
    // ── Setup: hash-to-curve H, U from a hiding pp ────────────────────────
    // We only need U and H; the G_i slots are unused for scalar Σ-protocol
    // tests.  Any hiding pp provides both.
    Commitment pp = Commitment::hiding_random(1);
    G1Jacobian_t U = pp.u_generator;
    G1Jacobian_t H = pp.hiding_generator;

    // Random witness (m, r) and challenge e.
    FrTensor mt = FrTensor::random(1);
    FrTensor rt = FrTensor::random(1);
    FrTensor et = FrTensor::random(1);
    Fr_t m = mt(0);
    Fr_t r = rt(0);
    Fr_t e = et(0);

    G1Jacobian_t C = commit_mU_rH(U, H, m, r);

    // ── Test 1: proof-of-opening roundtrip ────────────────────────────────
    {
        auto proof = prove_opening(U, H, C, m, r, e);
        check(verify_opening(U, H, C, proof, e),
              "proof-of-opening: honest prover accepted");
    }

    // ── Test 2: proof-of-opening negative paths ───────────────────────────
    {
        auto proof = prove_opening(U, H, C, m, r, e);

        // Tamper A: replace with some other point (U itself works).
        {
            auto bad = proof;
            bad.A = U;
            check(!verify_opening(U, H, C, bad, e),
                  "proof-of-opening: tampered A rejected");
        }
        // Tamper z_m: add one (via FrTensor addition).
        {
            auto bad = proof;
            Fr_t one = {1,0,0,0,0,0,0,0};
            FrTensor Z(1, &bad.z_m);
            FrTensor O(1, &one);
            FrTensor Zp = Z + O;
            bad.z_m = Zp(0);
            check(!verify_opening(U, H, C, bad, e),
                  "proof-of-opening: tampered z_m rejected");
        }
        // Tamper z_r similarly.
        {
            auto bad = proof;
            Fr_t one = {1,0,0,0,0,0,0,0};
            FrTensor Z(1, &bad.z_r);
            FrTensor O(1, &one);
            FrTensor Zp = Z + O;
            bad.z_r = Zp(0);
            check(!verify_opening(U, H, C, bad, e),
                  "proof-of-opening: tampered z_r rejected");
        }
        // Wrong challenge: verifier uses a different e'.
        {
            Fr_t e_wrong = FrTensor::random(1)(0);
            // Guard against the astronomical coincidence e_wrong == e.
            if (!fr_bytes_eq(e_wrong, e)) {
                check(!verify_opening(U, H, C, proof, e_wrong),
                      "proof-of-opening: wrong challenge rejected");
            }
        }
    }

    // ── Test 3: proof-of-opening soundness on wrong witness ───────────────
    // A prover who holds (m', r') ≠ (m, r) and tries to open C (which
    // commits to m) produces a transcript that must fail verification.
    {
        Fr_t m_wrong = FrTensor::random(1)(0);
        Fr_t r_wrong = FrTensor::random(1)(0);
        auto bad_proof = prove_opening(U, H, C, m_wrong, r_wrong, e);
        check(!verify_opening(U, H, C, bad_proof, e),
              "proof-of-opening: wrong witness rejected");
    }

    // ── Test 4: proof-of-equality roundtrip ───────────────────────────────
    {
        // Same m under two fresh blindings r1, r2.
        Fr_t r1 = FrTensor::random(1)(0);
        Fr_t r2 = FrTensor::random(1)(0);
        G1Jacobian_t C1 = commit_mU_rH(U, H, m, r1);
        G1Jacobian_t C2 = commit_mU_rH(U, H, m, r2);

        auto proof = prove_equality(H, C1, C2, r1, r2, e);
        check(verify_equality(H, C1, C2, proof, e),
              "proof-of-equality: honest prover accepted");
        // Symmetric statement: swapping (C1,r1)↔(C2,r2) also verifies.
        auto proof_sym = prove_equality(H, C2, C1, r2, r1, e);
        check(verify_equality(H, C2, C1, proof_sym, e),
              "proof-of-equality: symmetric statement accepted");
    }

    // ── Test 5: proof-of-equality negative paths ──────────────────────────
    {
        Fr_t r1 = FrTensor::random(1)(0);
        Fr_t r2 = FrTensor::random(1)(0);
        G1Jacobian_t C1 = commit_mU_rH(U, H, m, r1);
        G1Jacobian_t C2 = commit_mU_rH(U, H, m, r2);
        auto proof = prove_equality(H, C1, C2, r1, r2, e);

        // Tamper A.
        {
            auto bad = proof;
            bad.A = H;
            check(!verify_equality(H, C1, C2, bad, e),
                  "proof-of-equality: tampered A rejected");
        }
        // Tamper z.
        {
            auto bad = proof;
            Fr_t one = {1,0,0,0,0,0,0,0};
            FrTensor Z(1, &bad.z);
            FrTensor O(1, &one);
            FrTensor Zp = Z + O;
            bad.z = Zp(0);
            check(!verify_equality(H, C1, C2, bad, e),
                  "proof-of-equality: tampered z rejected");
        }
        // Wrong challenge.
        {
            Fr_t e_wrong = FrTensor::random(1)(0);
            if (!fr_bytes_eq(e_wrong, e)) {
                check(!verify_equality(H, C1, C2, proof, e_wrong),
                      "proof-of-equality: wrong challenge rejected");
            }
        }
    }

    // ── Test 6: proof-of-equality soundness on non-equal messages ─────────
    // C1 commits to m1, C2 commits to m2 ≠ m1.  No honest prover-line
    // transcript under any (r1, r2) can make verify accept, because the
    // underlying statement (C1 - C2) = (r1-r2)·H is false.
    {
        Fr_t m1 = FrTensor::random(1)(0);
        Fr_t m2 = FrTensor::random(1)(0);
        // Paranoia: ensure m1 ≠ m2.  Collision probability is negligible.
        if (fr_bytes_eq(m1, m2)) {
            Fr_t one = {1,0,0,0,0,0,0,0};
            FrTensor A(1, &m2); FrTensor B(1, &one);
            FrTensor S = A + B;
            m2 = S(0);
        }
        Fr_t r1 = FrTensor::random(1)(0);
        Fr_t r2 = FrTensor::random(1)(0);
        G1Jacobian_t C1 = commit_mU_rH(U, H, m1, r1);
        G1Jacobian_t C2 = commit_mU_rH(U, H, m2, r2);
        auto bad_proof = prove_equality(H, C1, C2, r1, r2, e);
        check(!verify_equality(H, C1, C2, bad_proof, e),
              "proof-of-equality: unequal messages rejected");
    }

    // ── Test 7: mask freshness across prove_opening calls ─────────────────
    // Two calls with the same witness must produce different A's; reuse
    // of the mask would leak (m, r) under a rewinding extractor.
    {
        auto p1 = prove_opening(U, H, C, m, r, e);
        auto p2 = prove_opening(U, H, C, m, r, e);
        check(!g1_bytes_eq(p1.A, p2.A),
              "proof-of-opening: fresh masks produce distinct A across calls");
        // Same story for prove_equality: fresh s.
        Fr_t r2a = FrTensor::random(1)(0);
        G1Jacobian_t C2 = commit_mU_rH(U, H, m, r2a);
        auto q1 = prove_equality(H, C, C2, r, r2a, e);
        auto q2 = prove_equality(H, C, C2, r, r2a, e);
        check(!g1_bytes_eq(q1.A, q2.A),
              "proof-of-equality: fresh masks produce distinct A across calls");
    }

    cout << "All hyrax_sigma tests PASSED." << endl;
    return 0;
}
