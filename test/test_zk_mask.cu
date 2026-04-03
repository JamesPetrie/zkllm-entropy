// test_zk_mask: Unit tests for ZK masking primitives (Phase 0)
//
// Tests:
// 1. Vanishing correction is zero on all boolean points
// 2. Vanishing correction is nonzero on random non-boolean points
// 3. P_sum matches brute-force sum over {0,1}^b
// 4. eval_transcript_mask matches brute-force at random point
// 5. transcript_mask_round_poly produces correct round sums
// 6. Polynomial::from_evaluations roundtrips correctly
//
// Build: make gold_test_zk_mask
// Usage: ./gold_test_zk_mask

#include "proof/zk_mask.cuh"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    cout << "=== ZK Masking Primitive Tests (Goldilocks) ===" << endl;
    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: Vanishing correction is zero on boolean points ──────────
    {
        uint k = 4;
        auto mask = generate_vanishing_mask(k);
        bool all_zero = true;
        for (uint c = 0; c < (1u << k); c++) {
            vector<Fr_t> point(k);
            for (uint i = 0; i < k; i++) {
                point[i] = ((c >> i) & 1) ? FR_ONE : FR_ZERO;
            }
            Fr_t corr = vanishing_correction(mask.vanishing_coeffs, point);
            if (corr != FR_ZERO) all_zero = false;
        }
        check(all_zero, "vanishing_correction == 0 on all boolean points (k=4)");
    }

    // ── Test 2: Vanishing correction is nonzero on random points ────────
    {
        uint k = 4;
        auto mask = generate_vanishing_mask(k);
        auto point = random_vec(k);  // random field elements, almost certainly non-boolean
        Fr_t corr = vanishing_correction(mask.vanishing_coeffs, point);
        check(corr != FR_ZERO, "vanishing_correction != 0 on random non-boolean point");
    }

    // ── Test 3: P_sum matches brute-force for b=3 ───────────────────────
    {
        uint b = 3;
        uint degree = 4;
        auto tmask = generate_transcript_mask(b, degree);

        // Brute force: sum p(c) over all c in {0,1}^b
        Fr_t brute_sum = FR_ZERO;
        for (uint c = 0; c < (1u << b); c++) {
            vector<Fr_t> point(b);
            for (uint i = 0; i < b; i++) {
                point[i] = ((c >> i) & 1) ? FR_ONE : FR_ZERO;
            }
            Fr_t val = eval_transcript_mask(tmask, point);
            brute_sum = brute_sum + val;
        }
        check(brute_sum == tmask.P_sum, "P_sum matches brute-force (b=3)");
    }

    // ── Test 4: P_sum matches brute-force for b=4 ───────────────────────
    {
        uint b = 4;
        uint degree = 4;
        auto tmask = generate_transcript_mask(b, degree);

        Fr_t brute_sum = FR_ZERO;
        for (uint c = 0; c < (1u << b); c++) {
            vector<Fr_t> point(b);
            for (uint i = 0; i < b; i++) {
                point[i] = ((c >> i) & 1) ? FR_ONE : FR_ZERO;
            }
            brute_sum = brute_sum + eval_transcript_mask(tmask, point);
        }
        check(brute_sum == tmask.P_sum, "P_sum matches brute-force (b=4)");
    }

    // ── Test 5: eval_transcript_mask at specific point ──────────────────
    {
        uint b = 2;
        uint degree = 2;
        auto tmask = generate_transcript_mask(b, degree);
        auto point = random_vec(b);

        // Manual evaluation: a0 + p_0(point[0]) + p_1(point[1])
        Fr_t manual = tmask.a0;
        for (uint i = 0; i < b; i++) {
            Fr_t xi = point[i];
            Fr_t xi_pow = xi;
            Fr_t pi = FR_ZERO;
            for (uint k = 0; k < tmask.p_univariates[i].size(); k++) {
                pi = pi + tmask.p_univariates[i][k] * xi_pow;
                xi_pow = xi_pow * xi;
            }
            manual = manual + pi;
        }
        Fr_t computed = eval_transcript_mask(tmask, point);
        check(computed == manual, "eval_transcript_mask matches manual computation");
    }

    // ── Test 6: transcript_mask_round_poly sum consistency ──────────────
    // For round j=0 with b=3: S_0(0) + S_0(1) should equal P_sum
    // (since P_sum = sum_{c in {0,1}^b} p(c) = S_0(0) + S_0(1) when j=0)
    {
        uint b = 3;
        uint degree = 4;
        auto tmask = generate_transcript_mask(b, degree);

        vector<Fr_t> bound_challenges;  // none bound yet at round 0
        Polynomial round_poly = transcript_mask_round_poly(tmask, 0, bound_challenges, b);

        Fr_t s0 = round_poly(FR_ZERO);
        Fr_t s1 = round_poly(FR_ONE);
        Fr_t round_sum = s0 + s1;

        check(round_sum == tmask.P_sum,
              "transcript_mask_round_poly: S_0(0) + S_0(1) == P_sum (b=3)");
    }

    // ── Test 7: transcript_mask_round_poly multi-round consistency ──────
    // After round 0 binds alpha_0, round 1's polynomial S_1 should satisfy:
    // S_1(0) + S_1(1) == S_0(alpha_0)
    {
        uint b = 3;
        uint degree = 4;
        auto tmask = generate_transcript_mask(b, degree);

        // Round 0
        vector<Fr_t> bound_0;
        Polynomial rp0 = transcript_mask_round_poly(tmask, 0, bound_0, b);
        auto alpha_0 = random_vec(1)[0];
        Fr_t claim_after_0 = rp0(alpha_0);

        // Round 1
        vector<Fr_t> bound_1 = {alpha_0};
        Polynomial rp1 = transcript_mask_round_poly(tmask, 1, bound_1, b);
        Fr_t sum_1 = rp1(FR_ZERO) + rp1(FR_ONE);

        check(sum_1 == claim_after_0,
              "transcript_mask_round_poly: S_1(0)+S_1(1) == S_0(alpha_0)");
    }

    // ── Test 8: Full multi-round chain for b=4 ─────────────────────────
    {
        uint b = 4;
        uint degree = 4;
        auto tmask = generate_transcript_mask(b, degree);

        Fr_t current_claim = tmask.P_sum;
        vector<Fr_t> bound;
        bool chain_ok = true;

        for (uint j = 0; j < b; j++) {
            Polynomial rp = transcript_mask_round_poly(tmask, j, bound, b);
            Fr_t s0 = rp(FR_ZERO);
            Fr_t s1 = rp(FR_ONE);
            if (s0 + s1 != current_claim) {
                cout << "    Round " << j << ": sum mismatch" << endl;
                chain_ok = false;
                break;
            }
            auto alpha = random_vec(1)[0];
            current_claim = rp(alpha);
            bound.push_back(alpha);
        }

        // After all rounds, current_claim should equal p(alpha_0,...,alpha_{b-1})
        Fr_t p_final = eval_transcript_mask(tmask, bound);
        if (current_claim != p_final) {
            cout << "    Final eval mismatch" << endl;
            chain_ok = false;
        }

        check(chain_ok, "Full transcript mask sumcheck chain (b=4)");
    }

    // ── Test 9: Polynomial::from_evaluations roundtrip ──────────────────
    {
        // Create a known degree-4 polynomial: p(X) = 3 + 2X + 5X^2 + X^3 + 7X^4
        vector<Fr_t> coeffs = {
            FR_FROM_INT(3), FR_FROM_INT(2), FR_FROM_INT(5),
            FR_FROM_INT(1), FR_FROM_INT(7)
        };
        Polynomial original(coeffs);

        // Evaluate at 0, 1, 2, 3, 4
        vector<Fr_t> evals(5);
        for (uint k = 0; k < 5; k++) {
            evals[k] = original(FR_FROM_INT(k));
        }

        // Reconstruct
        Polynomial reconstructed = Polynomial::from_evaluations(evals);

        // Check at several points
        bool match = true;
        for (uint x = 0; x < 10; x++) {
            Fr_t orig_val = original(FR_FROM_INT(x));
            Fr_t recon_val = reconstructed(FR_FROM_INT(x));
            if (orig_val != recon_val) {
                cout << "    Mismatch at x=" << x << endl;
                match = false;
                break;
            }
        }
        check(match, "Polynomial::from_evaluations roundtrip (degree 4)");
    }

    // ── Test 10: from_evaluations with degree 2 ────────────────────────
    {
        vector<Fr_t> coeffs = {FR_FROM_INT(1), FR_FROM_INT(4), FR_FROM_INT(6)};
        Polynomial original(coeffs);
        vector<Fr_t> evals(3);
        for (uint k = 0; k < 3; k++) {
            evals[k] = original(FR_FROM_INT(k));
        }
        Polynomial reconstructed = Polynomial::from_evaluations(evals);

        bool match = true;
        for (uint x = 0; x < 20; x++) {
            if (original(FR_FROM_INT(x)) != reconstructed(FR_FROM_INT(x))) {
                match = false;
                break;
            }
        }
        check(match, "Polynomial::from_evaluations roundtrip (degree 2)");
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
