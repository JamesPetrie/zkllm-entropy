// test_zk_sumcheck: Integration tests for ZK sumcheck (Phase 1)
//
// Tests:
// 1. zkip_zk with masking disabled matches zkip output
// 2. zkip_zk with masking: round checks pass (s(0)+s(1)==claim)
// 3. zkip_zk: final check za*zb + rho*p == last claim
// 4. Brute-force: sum over hypercube of masked products equals T
// 5. inner_product_sumcheck_zk produces valid proof
//
// Build: make gold_test_zk_sumcheck
// Usage: ./gold_test_zk_sumcheck

#include "proof/zk_sumcheck.cuh"
#include "proof/zk_mask.cuh"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    cout << "=== ZK Sumcheck Integration Tests (Goldilocks) ===" << endl;
    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: zkip_zk with masking disabled matches zkip ──────────────
    {
        // Create small test data: 8 elements (k=3 variables)
        uint N = 8;
        uint k = 3;
        vector<Fr_t> a_data(N), b_data(N);
        for (uint i = 0; i < N; i++) {
            a_data[i] = FR_FROM_INT(i + 1);
            b_data[i] = FR_FROM_INT(2 * i + 3);
        }

        FrTensor a(N, a_data.data()), b(N, b_data.data());

        // Compute honest T = <a, b>
        Fr_t T = (a * b).sum();

        auto u = random_vec(k);

        // Disabled masking: coefficients are zero
        ZkMaskConfig mask_a, mask_b;
        mask_a.enabled = false;
        mask_b.enabled = false;

        // Trivial transcript mask: all zeros (no effect when rho=0)
        ZkTranscriptMask tmask;
        tmask.a0 = FR_ZERO;
        tmask.degree = 4;
        tmask.p_univariates.resize(k);
        for (uint i = 0; i < k; i++) {
            tmask.p_univariates[i].assign(4, FR_ZERO);
        }
        tmask.P_sum = FR_ZERO;
        Fr_t rho = FR_ZERO;

        // Combined claim with rho=0: just T
        Fr_t combined_claim = T;

        vector<Polynomial> proof;
        ZkIpResult result;
        Fr_t final_claim = zkip_zk(combined_claim, a, b, u, mask_a, mask_b, tmask, rho, proof, result);

        // Verify final check: za * zb + rho * p == final_claim
        Fr_t check_val = result.final_za * result.final_zb + rho * result.p_final;
        check(check_val == final_claim, "zkip_zk disabled masking: final check passes");

        // The proof should have k round polys + 3 final values
        check(proof.size() == k + 3, "zkip_zk disabled masking: correct proof size");
    }

    // ── Test 2: zkip_zk with full masking: round checks ────────────────
    {
        uint N = 8;
        uint k = 3;
        vector<Fr_t> a_data(N), b_data(N);
        for (uint i = 0; i < N; i++) {
            a_data[i] = FR_FROM_INT(i * 3 + 7);
            b_data[i] = FR_FROM_INT(i * 2 + 1);
        }

        FrTensor a(N, a_data.data()), b(N, b_data.data());

        Fr_t T = (a * b).sum();
        auto u = random_vec(k);

        // Full masking
        auto mask_a = generate_vanishing_mask(k);
        auto mask_b = generate_vanishing_mask(k);
        auto tmask = generate_transcript_mask(k, 4);
        Fr_t rho = random_vec(1)[0];

        Fr_t combined_claim = T + rho * tmask.P_sum;

        vector<Polynomial> proof;
        ZkIpResult result;

        // This should not throw (round checks are internal)
        bool no_throw = true;
        try {
            zkip_zk(combined_claim, a, b, u, mask_a, mask_b, tmask, rho, proof, result);
        } catch (const exception& e) {
            cout << "    Exception: " << e.what() << endl;
            no_throw = false;
        }
        check(no_throw, "zkip_zk full masking: all rounds pass");

        // Verify final check: za * zb + rho * p == last reduced claim
        Fr_t final_claim = proof.size() > k ?
            result.final_za * result.final_zb + rho * result.p_final : FR_ZERO;

        // Replay the sumcheck to get the final claim
        Fr_t replayed_claim = combined_claim;
        for (uint j = 0; j < k; j++) {
            uint var_idx = k - 1 - j;
            replayed_claim = proof[j](u[var_idx]);
        }

        check(final_claim == replayed_claim,
              "zkip_zk full masking: final check za*zb + rho*p == claim");
    }

    // ── Test 3: Brute-force verify sum preservation ─────────────────────
    // sum_{c in {0,1}^k} Za(c) * Zb(c) should equal T
    // (since vanishing terms are zero on booleans)
    {
        uint N = 8;
        uint k = 3;
        vector<Fr_t> a_data(N), b_data(N);
        for (uint i = 0; i < N; i++) {
            a_data[i] = FR_FROM_INT(i + 10);
            b_data[i] = FR_FROM_INT(i * 5 + 2);
        }

        // Compute T on CPU
        Fr_t T_cpu = FR_ZERO;
        for (uint i = 0; i < N; i++) {
            T_cpu = T_cpu + a_data[i] * b_data[i];
        }

        // With masking, on booleans: Za(c) = a(c) + 0 = a(c), Zb(c) = b(c)
        // So sum should be the same.
        // This is a tautology for vanishing masking but let's verify the
        // vanishing correction is actually zero on all boolean points.
        auto mask_a = generate_vanishing_mask(k);
        auto mask_b = generate_vanishing_mask(k);

        Fr_t masked_sum = FR_ZERO;
        for (uint c = 0; c < N; c++) {
            vector<Fr_t> point(k);
            for (uint i = 0; i < k; i++) {
                point[i] = ((c >> i) & 1) ? FR_ONE : FR_ZERO;
            }
            Fr_t za = a_data[c] + vanishing_correction(mask_a.vanishing_coeffs, point);
            Fr_t zb = b_data[c] + vanishing_correction(mask_b.vanishing_coeffs, point);
            masked_sum = masked_sum + za * zb;
        }

        check(masked_sum == T_cpu, "Brute-force: masked sum == honest sum on hypercube");
    }

    // ── Test 4: inner_product_sumcheck_zk flat format ───────────────────
    {
        uint N = 16;
        uint k = 4;
        vector<Fr_t> a_data(N), b_data(N);
        for (uint i = 0; i < N; i++) {
            a_data[i] = FR_FROM_INT(i + 1);
            b_data[i] = FR_FROM_INT(N - i);
        }

        FrTensor a(N, a_data.data()), b(N, b_data.data());
        auto u = random_vec(k);
        auto mask_a = generate_vanishing_mask(k);
        ZkMaskConfig mask_b;  // public (no masking)
        mask_b.enabled = false;
        auto tmask = generate_transcript_mask(k, 4);
        Fr_t rho = random_vec(1)[0];

        auto flat_proof = inner_product_sumcheck_zk(a, b, u, mask_a, mask_b, tmask, rho);

        // Expected size: 5 evals per round * k rounds + 3 final values
        uint expected_size = 5 * k + 3;
        check(flat_proof.size() == expected_size,
              "inner_product_sumcheck_zk: correct flat proof size");

        // Verify round checks manually
        Fr_t T = (a * b).sum();
        Fr_t current_claim = T + rho * tmask.P_sum;
        bool rounds_ok = true;
        for (uint j = 0; j < k; j++) {
            Fr_t p0 = flat_proof[5 * j];      // p(0)
            Fr_t p1 = flat_proof[5 * j + 1];  // p(1)
            if (p0 + p1 != current_claim) {
                cout << "    Round " << j << ": sum mismatch" << endl;
                rounds_ok = false;
                break;
            }
            // Compute p(alpha) via Lagrange interpolation from 5 evals
            vector<Fr_t> evals(5);
            for (uint t = 0; t < 5; t++) evals[t] = flat_proof[5 * j + t];
            Polynomial rp = Polynomial::from_evaluations(evals);
            uint var_idx = k - 1 - j;
            current_claim = rp(u[var_idx]);
        }
        check(rounds_ok, "inner_product_sumcheck_zk: all round checks pass");

        // Final check
        Fr_t za = flat_proof[5 * k];
        Fr_t zb = flat_proof[5 * k + 1];
        Fr_t p_final = flat_proof[5 * k + 2];
        Fr_t final_val = za * zb + rho * p_final;
        check(final_val == current_claim,
              "inner_product_sumcheck_zk: final check za*zb + rho*p == claim");
    }

    // ── Test 5: Soundness — tampered data should break the proof ────────
    {
        uint N = 8;
        uint k = 3;
        vector<Fr_t> a_data(N), b_data(N);
        for (uint i = 0; i < N; i++) {
            a_data[i] = FR_FROM_INT(i + 1);
            b_data[i] = FR_FROM_INT(i + 1);
        }

        FrTensor a(N, a_data.data()), b(N, b_data.data());
        auto u = random_vec(k);
        auto mask_a = generate_vanishing_mask(k);
        ZkMaskConfig mask_b;
        mask_b.enabled = false;
        auto tmask = generate_transcript_mask(k, 4);
        Fr_t rho = random_vec(1)[0];

        // Use a WRONG claim (T + 1)
        Fr_t T = (a * b).sum();
        Fr_t wrong_claim = T + FR_ONE + rho * tmask.P_sum;

        vector<Polynomial> proof;
        ZkIpResult result;
        bool caught = false;
        try {
            zkip_zk(wrong_claim, a, b, u, mask_a, mask_b, tmask, rho, proof, result);
        } catch (const exception& e) {
            caught = true;
        }
        check(caught, "zkip_zk: wrong claim detected (throws at round 0)");
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
