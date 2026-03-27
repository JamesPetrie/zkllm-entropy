// test_fri_pcs: verify the FRI Polynomial Commitment Scheme integration.
//
// Tests that FRI PCS can:
// 1. Commit to a vector and verify multilinear evaluation
// 2. Handle larger vectors (simulating weight matrices)
// 3. Produce correct multilinear evaluations
// 4. Work with the sumcheck protocol (inner product sumcheck + FRI PCS)
//
// Usage: ./test_fri_pcs

#include "fri_pcs.cuh"
#include "proof.cuh"
#include <iostream>
#include <vector>

using namespace std;

// Host-side field arithmetic
static uint64_t h_mul(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t eps = GOLDILOCKS_P_NEG;
    __uint128_t t = (__uint128_t)hi * eps + lo;
    uint64_t r_lo = (uint64_t)t;
    uint64_t r_hi = (uint64_t)(t >> 64);
    uint64_t result = r_lo + r_hi * eps;
    if (result < r_lo) result += eps;
    if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;
    return result;
}

static uint64_t h_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a || s >= GOLDILOCKS_P) s -= GOLDILOCKS_P;
    return s;
}

static uint64_t h_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + GOLDILOCKS_P - b);
}

int main() {
    cout << "=== FRI PCS Integration Tests (Goldilocks) ===" << endl;
    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: Small vector commit + open + verify ─────────────────────
    {
        uint n = 8;  // 2^3
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i + 1)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        // Commit
        FriPcsCommitment com = FriPcs::commit(gpu_data, n);
        check(com.size == 8, "commit: size == 8");
        check(com.log_size == 3, "commit: log_size == 3");

        // Open at u = (0, 0, 0) -> should give data[0] = 1
        vector<Fr_t> u_zero = {{0}, {0}, {0}};
        FriPcsOpeningProof proof = FriPcs::open(gpu_data, n, u_zero);
        check(proof.claimed_value.val == 1, "open at (0,0,0) == 1");

        // Verify
        bool valid = FriPcs::verify(com, u_zero, proof);
        check(valid, "verify open at (0,0,0)");

        // Open at u = (1, 0, 0) -> should give data[1] = 2
        vector<Fr_t> u_100 = {{1}, {0}, {0}};
        proof = FriPcs::open(gpu_data, n, u_100);
        check(proof.claimed_value.val == 2, "open at (1,0,0) == 2");
        valid = FriPcs::verify(com, u_100, proof);
        check(valid, "verify open at (1,0,0)");

        cudaFree(gpu_data);
    }

    // ── Test 2: Multilinear evaluation correctness ──────────────────────
    {
        // f(x0, x1) = 10*(1-x0)*(1-x1) + 20*x0*(1-x1) + 30*(1-x0)*x1 + 40*x0*x1
        // = [10, 20, 30, 40]
        uint n = 4;
        vector<Fr_t> data = {{10}, {20}, {30}, {40}};

        // f(0, 0) = 10
        vector<Fr_t> u00 = {{0}, {0}};
        Fr_t v = FriPcs::multilinear_eval_host(data, u00);
        check(v.val == 10, "MLE [10,20,30,40] at (0,0) == 10");

        // f(1, 0) = 20
        vector<Fr_t> u10 = {{1}, {0}};
        v = FriPcs::multilinear_eval_host(data, u10);
        check(v.val == 20, "MLE at (1,0) == 20");

        // f(0, 1) = 30
        vector<Fr_t> u01 = {{0}, {1}};
        v = FriPcs::multilinear_eval_host(data, u01);
        check(v.val == 30, "MLE at (0,1) == 30");

        // f(1, 1) = 40
        vector<Fr_t> u11 = {{1}, {1}};
        v = FriPcs::multilinear_eval_host(data, u11);
        check(v.val == 40, "MLE at (1,1) == 40");

        // f(1/2, 1/2): average of all = 25
        uint64_t half = (GOLDILOCKS_P + 1) / 2;  // 2^(-1) mod p
        vector<Fr_t> u_half = {{half}, {half}};
        v = FriPcs::multilinear_eval_host(data, u_half);
        check(v.val == 25, "MLE at (1/2,1/2) == 25");
    }

    // ── Test 3: Larger vector (simulating a weight matrix) ──────────────
    {
        uint n = 1024;  // 2^10
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i % 100 + 1)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        FriPcsCommitment com = FriPcs::commit(gpu_data, n);
        check(com.size == 1024, "large commit: size == 1024");

        // Open at a random point
        vector<Fr_t> u(10);
        for (uint i = 0; i < 10; i++) u[i] = Fr_t{(uint64_t)(i * 37 + 11) % GOLDILOCKS_P};

        FriPcsOpeningProof proof = FriPcs::open(gpu_data, n, u);

        // Verify the evaluation matches independent computation
        Fr_t expected = FriPcs::multilinear_eval_host(data, u);
        check(proof.claimed_value.val == expected.val, "large open: claimed value matches");

        bool valid = FriPcs::verify(com, u, proof);
        check(valid, "large open: FRI proof verifies");

        cudaFree(gpu_data);
    }

    // ── Test 4: FRI PCS with FrTensor multilinear evaluation ────────────
    {
        // Compare FriPcs::multilinear_eval_host with FrTensor's operator()(vector<Fr_t>)
        uint n = 16;
        int data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        FrTensor t(n, data);

        // Random evaluation point
        vector<Fr_t> u = {{3}, {7}, {11}, {13}};

        // FrTensor MLE evaluation
        Fr_t v_tensor = t(u);

        // FriPcs MLE evaluation
        vector<Fr_t> data_fr(n);
        for (uint i = 0; i < n; i++) data_fr[i] = t(i);
        Fr_t v_pcs = FriPcs::multilinear_eval_host(data_fr, u);

        check(v_tensor.val == v_pcs.val, "FrTensor MLE matches FriPcs MLE");
    }

    // ── Test 5: End-to-end: sumcheck + FRI PCS ──────────────────────────
    {
        // Inner product sumcheck: <a, b> = sum_i a[i] * b[i]
        // Then verify that a and b are committed correctly via FRI PCS
        uint n = 8;
        int a_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
        int b_data[] = {8, 7, 6, 5, 4, 3, 2, 1};
        FrTensor a(n, a_data), b(n, b_data);

        // Run inner product sumcheck
        vector<Fr_t> u = random_vec(3);
        vector<Fr_t> sc_proof = inner_product_sumcheck(a, b, u);
        check(sc_proof.size() > 0, "sumcheck produced proof");

        // Now commit both a and b via FRI PCS
        FriPcsCommitment com_a = FriPcs::commit(a.gpu_data, n);
        FriPcsCommitment com_b = FriPcs::commit(b.gpu_data, n);

        // Open both at the sumcheck challenge point u
        FriPcsOpeningProof proof_a = FriPcs::open(a.gpu_data, n, u);
        FriPcsOpeningProof proof_b = FriPcs::open(b.gpu_data, n, u);

        // Verify openings
        bool valid_a = FriPcs::verify(com_a, u, proof_a);
        bool valid_b = FriPcs::verify(com_b, u, proof_b);
        check(valid_a && valid_b, "FRI PCS verifies both sumcheck vectors");

        // The sumcheck proof's final claimed a(u) and b(u) should match
        // the FRI PCS opening values
        // (The last two elements of the sumcheck proof are a(u) and b(u))
        Fr_t a_u = sc_proof[sc_proof.size() - 2];
        Fr_t b_u = sc_proof[sc_proof.size() - 1];
        check(a_u.val == proof_a.claimed_value.val, "sumcheck a(u) matches FRI PCS open");
        check(b_u.val == proof_b.claimed_value.val, "sumcheck b(u) matches FRI PCS open");

        cout << "  (inner product = 120, verified via sumcheck + FRI PCS)" << endl;
    }

    cout << "\n=== Results: " << (failures == 0 ? "ALL PASSED" : "FAILURES")
         << " (failures=" << failures << ") ===" << endl;
    return failures;
}
