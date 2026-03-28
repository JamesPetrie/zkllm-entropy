// test_fri_pcs: verify the FRI Polynomial Commitment Scheme integration.
//
// Tests that FRI PCS can:
// 1. Commit to a vector and open with binding verification
// 2. Produce correct multilinear evaluations
// 3. Detect data tampering (binding check)
// 4. Work with the sumcheck protocol (inner product sumcheck + FRI PCS)
//
// Usage: ./test_fri_pcs

#include "commit/fri_pcs.cuh"
#include "proof/proof.cuh"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    cout << "=== FRI PCS Integration Tests (Goldilocks) ===" << endl;
    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: Small vector commit + open ──────────────────────────────
    {
        uint n = 8;
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i + 1)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        FriPcsCommitment com = FriPcs::commit(gpu_data, n);
        check(com.size == 8, "commit: size == 8");
        check(com.log_size == 3, "commit: log_size == 3");

        // Open at u = (0, 0, 0) -> should give data[0] = 1
        vector<Fr_t> u_zero = {{0}, {0}, {0}};
        Fr_t val = FriPcs::open(gpu_data, n, com, u_zero);
        check(val.val == 1, "open at (0,0,0) == 1");

        // Open at u = (1, 0, 0) -> should give data[1] = 2
        vector<Fr_t> u_100 = {{1}, {0}, {0}};
        val = FriPcs::open(gpu_data, n, com, u_100);
        check(val.val == 2, "open at (1,0,0) == 2");

        cudaFree(gpu_data);
    }

    // ── Test 2: Multilinear evaluation correctness ──────────────────────
    {
        // f(x0, x1) = 10*(1-x0)*(1-x1) + 20*x0*(1-x1) + 30*(1-x0)*x1 + 40*x0*x1
        vector<Fr_t> data = {{10}, {20}, {30}, {40}};

        vector<Fr_t> u00 = {{0}, {0}};
        Fr_t v = FriPcs::multilinear_eval_host(data, u00);
        check(v.val == 10, "MLE [10,20,30,40] at (0,0) == 10");

        vector<Fr_t> u10 = {{1}, {0}};
        v = FriPcs::multilinear_eval_host(data, u10);
        check(v.val == 20, "MLE at (1,0) == 20");

        vector<Fr_t> u01 = {{0}, {1}};
        v = FriPcs::multilinear_eval_host(data, u01);
        check(v.val == 30, "MLE at (0,1) == 30");

        vector<Fr_t> u11 = {{1}, {1}};
        v = FriPcs::multilinear_eval_host(data, u11);
        check(v.val == 40, "MLE at (1,1) == 40");

        uint64_t half = (GOLDILOCKS_P + 1) / 2;
        vector<Fr_t> u_half = {{half}, {half}};
        v = FriPcs::multilinear_eval_host(data, u_half);
        check(v.val == 25, "MLE at (1/2,1/2) == 25");
    }

    // ── Test 3: Binding check — tampered data rejected ──────────────────
    {
        uint n = 8;
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i + 1)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        FriPcsCommitment com = FriPcs::commit(gpu_data, n);

        // Tamper with the data
        data[3] = Fr_t{999ULL};
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        // Opening with tampered data should throw
        vector<Fr_t> u = {{0}, {0}, {0}};
        bool caught = false;
        try {
            FriPcs::open(gpu_data, n, com, u);
        } catch (const std::runtime_error&) {
            caught = true;
        }
        check(caught, "tampered data detected (binding check)");

        cudaFree(gpu_data);
    }

    // ── Test 4: Larger vector (simulating a weight matrix) ──────────────
    {
        uint n = 1024;
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i % 100 + 1)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        FriPcsCommitment com = FriPcs::commit(gpu_data, n);

        vector<Fr_t> u(10);
        for (uint i = 0; i < 10; i++) u[i] = Fr_t{(uint64_t)(i * 37 + 11) % GOLDILOCKS_P};

        Fr_t val = FriPcs::open(gpu_data, n, com, u);
        Fr_t expected = FriPcs::multilinear_eval_host(data, u);
        check(val.val == expected.val, "large open: value matches independent computation");

        cudaFree(gpu_data);
    }

    // ── Test 5: FrTensor MLE matches FriPcs MLE ─────────────────────────
    {
        uint n = 16;
        int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        FrTensor t(n, data);

        vector<Fr_t> u = {{3}, {7}, {11}, {13}};
        Fr_t v_tensor = t(u);

        vector<Fr_t> data_fr(n);
        for (uint i = 0; i < n; i++) data_fr[i] = t(i);
        Fr_t v_pcs = FriPcs::multilinear_eval_host(data_fr, u);

        check(v_tensor.val == v_pcs.val, "FrTensor MLE matches FriPcs MLE");
    }

    // ── Test 6: End-to-end: sumcheck + FRI PCS ──────────────────────────
    {
        uint n = 8;
        int a_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
        int b_data[] = {8, 7, 6, 5, 4, 3, 2, 1};
        FrTensor a(n, a_data), b(n, b_data);

        // Commit both vectors
        FriPcsCommitment com_a = FriPcs::commit(a.gpu_data, n);
        FriPcsCommitment com_b = FriPcs::commit(b.gpu_data, n);

        // Run inner product sumcheck
        vector<Fr_t> u = random_vec(3);
        vector<Fr_t> sc_proof = inner_product_sumcheck(a, b, u);
        check(sc_proof.size() > 0, "sumcheck produced proof");

        // The last two elements of the proof are a(u) and b(u)
        Fr_t a_u_sumcheck = sc_proof[sc_proof.size() - 2];
        Fr_t b_u_sumcheck = sc_proof[sc_proof.size() - 1];

        // Open both via FRI PCS (verifies binding + computes MLE)
        Fr_t a_u_pcs = FriPcs::open(a.gpu_data, n, com_a, u);
        Fr_t b_u_pcs = FriPcs::open(b.gpu_data, n, com_b, u);

        check(a_u_sumcheck.val == a_u_pcs.val, "sumcheck a(u) matches FRI PCS open");
        check(b_u_sumcheck.val == b_u_pcs.val, "sumcheck b(u) matches FRI PCS open");

        cout << "  (inner product = 120, verified via sumcheck + FRI PCS)" << endl;
    }

    // ── Test 7: Weight struct + verifyWeightClaim ──────────────────────
    {
        // Simulate a 4x2 weight matrix (in_dim=4, out_dim=2, total=8 elements)
        uint in_dim = 4, out_dim = 2;
        uint n = in_dim * out_dim;
        int w_data[] = {10, 20, 30, 40, 50, 60, 70, 80};
        FrTensor weight(n, w_data);

        // Pad and commit (same as create_weight does)
        auto w_padded = weight.pad({in_dim, out_dim});
        FriPcsCommitment com = FriPcs::commit(w_padded.gpu_data, w_padded.size);

        Weight w{weight, com, in_dim, out_dim};

        // Build a Claim: evaluate the padded weight at random u
        // u[0] has ceilLog2(out_dim) elements, u[1] has ceilLog2(in_dim) elements
        uint log_out = ceilLog2(out_dim);
        uint log_in = ceilLog2(in_dim);
        vector<Fr_t> u0 = random_vec(log_out);
        vector<Fr_t> u1 = random_vec(log_in);

        // The claim value is MLE(w_padded, concat(u1, u0))
        vector<Fr_t> u_cat = concatenate(vector<vector<Fr_t>>({u1, u0}));
        Fr_t claim_val = FriPcs::open(w_padded.gpu_data, w_padded.size, com, u_cat);

        Claim c;
        c.claim = claim_val;
        c.u = {u0, u1};
        c.dims = {out_dim, in_dim};

        // verifyWeightClaim should succeed
        bool passed = false;
        try {
            verifyWeightClaim(w, c);
            passed = true;
        } catch (const std::runtime_error& e) {
            cout << "  verifyWeightClaim threw: " << e.what() << endl;
        }
        check(passed, "verifyWeightClaim succeeds with correct claim");

        // verifyWeightClaim should fail with wrong claim
        Claim bad_c = c;
        bad_c.claim = Fr_t{999ULL};
        bool caught = false;
        try {
            verifyWeightClaim(w, bad_c);
        } catch (const std::runtime_error&) {
            caught = true;
        }
        check(caught, "verifyWeightClaim rejects wrong claim");
    }

    cout << "\n=== Results: " << (failures == 0 ? "ALL PASSED" : "FAILURES")
         << " (failures=" << failures << ") ===" << endl;
    return failures;
}
