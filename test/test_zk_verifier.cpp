// test_zk_verifier.cpp — CPU-only tests for ZK sumcheck verifier
//
// Tests the verifier logic without GPU. Constructs proof data manually
// using the CPU-side field arithmetic from verifier_utils.h.
//
// Build: make test_zk_verifier
// Usage: ./test_zk_verifier

#include "verifier_utils.h"
#include "sumcheck_verifier.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>

using namespace std;

// Simple PRNG for test data
static mt19937_64 rng(42);
static Fr_t rand_fr() {
    return fr_from_u64(rng() % GOLDILOCKS_P);
}

int main() {
    cout << "=== ZK Verifier Tests (CPU-only) ===" << endl;
    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: lagrange_from_evals roundtrip ───────────────────────────
    {
        // Known polynomial: p(X) = 2 + 3X + 5X^2 + X^3 + 4X^4
        vector<Fr_t> coeffs = {
            fr_from_u64(2), fr_from_u64(3), fr_from_u64(5),
            fr_from_u64(1), fr_from_u64(4)
        };
        Polynomial original{coeffs};

        // Evaluate at {0,1,2,3,4}
        vector<Fr_t> evals(5);
        for (uint32_t k = 0; k < 5; k++) {
            evals[k] = original.eval(fr_from_u64(k));
        }

        // Reconstruct
        Polynomial recon = lagrange_from_evals(evals);

        // Check at many points
        bool match = true;
        for (uint32_t x = 0; x < 20; x++) {
            Fr_t orig = original.eval(fr_from_u64(x));
            Fr_t rec = recon.eval(fr_from_u64(x));
            if (orig != rec) { match = false; break; }
        }
        check(match, "lagrange_from_evals roundtrip (degree 4)");
    }

    // ── Test 2: Valid ZK proof accepted ──────────────────────────────────
    // Manually construct a valid 2-round ZK IP sumcheck proof.
    // We simulate a simple inner product over 4 elements (k=2).
    {
        uint32_t k = 2;

        // Data: a = [1, 2, 3, 4], b = [5, 6, 7, 8]
        // T = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        vector<Fr_t> a_data = {fr_from_u64(1), fr_from_u64(2), fr_from_u64(3), fr_from_u64(4)};
        vector<Fr_t> b_data = {fr_from_u64(5), fr_from_u64(6), fr_from_u64(7), fr_from_u64(8)};

        Fr_t T = FR_ZERO;
        for (uint32_t i = 0; i < 4; i++) T = fr_add(T, fr_mul(a_data[i], b_data[i]));

        // No masking for this test (simple verification)
        Fr_t P_sum = FR_ZERO;
        Fr_t rho = FR_ZERO;

        Fr_t combined_claim = fr_add(T, fr_mul(rho, P_sum));

        // Round 0: bind variable 1 (high bit). The kernel splits a[0..1] / a[2..3].
        // g_0(X) = sum_{var0} a(var0,X) * b(var0,X)
        // a(0,X) = 1 + 2X, a(1,X) = 2 + 2X, b(0,X) = 5 + 2X, b(1,X) = 6 + 2X
        // g_0(X) = (1+2X)(5+2X) + (2+2X)(6+2X) = 17 + 28X + 8X^2
        // Evals: g_0(0)=17, g_0(1)=53, g_0(2)=105, g_0(3)=173, g_0(4)=257
        vector<Fr_t> evals_0 = {
            fr_from_u64(17), fr_from_u64(53), fr_from_u64(105),
            fr_from_u64(173), fr_from_u64(257)
        };
        Polynomial rp0 = lagrange_from_evals(evals_0);

        Fr_t alpha_0 = fr_from_u64(42);
        Fr_t claim_after_0 = rp0.eval(alpha_0);  // 17 + 28*42 + 8*42^2 = 15305

        // After binding var1 to alpha_0=42:
        // af[i] = a[i] + 42*(a[i+2]-a[i]), bf similar
        Fr_t af0 = fr_from_u64(85);   // 1 + 42*2
        Fr_t af1 = fr_from_u64(86);   // 2 + 42*2
        Fr_t bf0 = fr_from_u64(89);   // 5 + 42*2
        Fr_t bf1 = fr_from_u64(90);   // 6 + 42*2

        // Round 1: g_1(X) = (85+X)*(89+X) = 7565 + 174X + X^2
        // Evals: 7565, 7740, 7917, 8096, 8277
        vector<Fr_t> evals_1 = {
            fr_from_u64(7565), fr_from_u64(7740), fr_from_u64(7917),
            fr_from_u64(8096), fr_from_u64(8277)
        };
        Polynomial rp1 = lagrange_from_evals(evals_1);

        Fr_t alpha_1 = fr_from_u64(17);

        // Final values: za = 85+17=102, zb = 89+17=106
        Fr_t final_za = fr_from_u64(102);
        Fr_t final_zb = fr_from_u64(106);
        Fr_t final_p = FR_ZERO;  // no transcript masking

        // Build proof
        ZkIpSumcheckProof proof;
        proof.T = T;
        proof.P_sum = P_sum;
        proof.rho = rho;
        proof.round_polys = {rp0, rp1};
        proof.challenges = {alpha_0, alpha_1};
        proof.final_za = final_za;
        proof.final_zb = final_zb;
        proof.final_p = final_p;

        auto result = verify_zk_ip_sumcheck(proof);
        check(result.ok, "Valid ZK proof (no masking) accepted");
    }

    // ── Test 3: Tampered round polynomial rejected ──────────────────────
    {
        // Use the same setup as test 2 but tamper with round 0 evals
        vector<Fr_t> a_data = {fr_from_u64(1), fr_from_u64(2), fr_from_u64(3), fr_from_u64(4)};
        vector<Fr_t> b_data = {fr_from_u64(5), fr_from_u64(6), fr_from_u64(7), fr_from_u64(8)};
        Fr_t T = FR_ZERO;
        for (uint32_t i = 0; i < 4; i++) T = fr_add(T, fr_mul(a_data[i], b_data[i]));

        // Tampered: change g_0(0) from 17 to 18 (breaks p(0)+p(1)==claim)
        vector<Fr_t> evals_0 = {
            fr_from_u64(18), fr_from_u64(53), fr_from_u64(89),
            fr_from_u64(125), fr_from_u64(161)
        };
        Polynomial rp0 = lagrange_from_evals(evals_0);

        ZkIpSumcheckProof proof;
        proof.T = T;
        proof.P_sum = FR_ZERO;
        proof.rho = FR_ZERO;
        proof.round_polys = {rp0};
        proof.challenges = {fr_from_u64(42)};
        proof.final_za = FR_ONE;
        proof.final_zb = FR_ONE;
        proof.final_p = FR_ZERO;

        auto result = verify_zk_ip_sumcheck(proof);
        check(!result.ok, "Tampered round polynomial rejected");
    }

    // ── Test 4: Wrong final_p rejected ──────────────────────────────────
    {
        // Valid round polynomials but wrong p(s*) at the end
        Fr_t T = fr_from_u64(70);
        Fr_t P_sum = fr_from_u64(100);
        Fr_t rho = fr_from_u64(3);

        // combined = 70 + 3*100 = 370
        Fr_t combined = fr_add(T, fr_mul(rho, P_sum));

        // One round: g(X) = 185 + 0*X (constant — sum = 185+185=370)
        Fr_t half = fr_from_u64(185);
        vector<Fr_t> evals(5, half);
        Polynomial rp = lagrange_from_evals(evals);

        Fr_t alpha = fr_from_u64(7);
        Fr_t reduced = rp.eval(alpha);  // = 185

        // Correct final: za*zb + rho*p_final == 185
        // Let za=10, zb=15, p_final = (185 - 150)/3 = 35/3
        // That's messy. Let's pick: za=1, zb=1, p_final = (185-1)/3
        // = 184/3. Also messy in integer arithmetic.
        // Instead: za=185, zb=1, p_final=0 → 185*1 + 3*0 = 185. Good.
        Fr_t final_za = fr_from_u64(185);
        Fr_t final_zb = FR_ONE;
        Fr_t correct_p = FR_ZERO;
        Fr_t wrong_p = fr_from_u64(1);  // should be 0

        // Correct version
        ZkIpSumcheckProof proof_ok;
        proof_ok.T = T;
        proof_ok.P_sum = P_sum;
        proof_ok.rho = rho;
        proof_ok.round_polys = {rp};
        proof_ok.challenges = {alpha};
        proof_ok.final_za = final_za;
        proof_ok.final_zb = final_zb;
        proof_ok.final_p = correct_p;

        auto r1 = verify_zk_ip_sumcheck(proof_ok);
        check(r1.ok, "Correct p(s*) accepted");

        // Wrong version
        ZkIpSumcheckProof proof_bad = proof_ok;
        proof_bad.final_p = wrong_p;

        auto r2 = verify_zk_ip_sumcheck(proof_bad);
        check(!r2.ok, "Wrong p(s*) rejected");
    }

    // ── Test 5: parse_zk_ip_sumcheck from flat data ─────────────────────
    {
        // Create a flat proof: 2 rounds x 5 evals + 3 final values = 13 elements
        vector<Fr_t> flat(13);
        for (int i = 0; i < 13; i++) flat[i] = fr_from_u64(i + 1);

        Fr_t T = fr_from_u64(100);
        Fr_t P_sum = fr_from_u64(50);
        Fr_t rho = fr_from_u64(7);
        vector<Fr_t> challenges = {fr_from_u64(11), fr_from_u64(13)};

        auto proof = parse_zk_ip_sumcheck(flat, 0, 2, T, P_sum, rho, challenges);

        check(proof.T.val == 100, "parse: T correct");
        check(proof.rho.val == 7, "parse: rho correct");
        check(proof.round_polys.size() == 2, "parse: 2 round polys");
        check(proof.final_za.val == 11, "parse: final_za correct");
        check(proof.final_zb.val == 12, "parse: final_zb correct");
        check(proof.final_p.val == 13, "parse: final_p correct");

        // Verify round poly 0 evaluates correctly at 0
        Fr_t rp0_at_0 = proof.round_polys[0].eval(FR_ZERO);
        check(rp0_at_0.val == 1, "parse: round_poly[0](0) == flat[0]");
    }

    // ── Summary ─────────────────────────────────────────────────────────
    cout << endl;
    if (failures == 0) {
        cout << "All tests passed!" << endl;
    } else {
        cout << failures << " test(s) FAILED" << endl;
    }
    return failures;
}
