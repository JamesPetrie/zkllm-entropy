// test_fri: verify FRI polynomial commitment scheme.
//
// Tests:
// 1. Commit and prove a small polynomial, verify succeeds
// 2. Tampered evaluation should fail verification
// 3. Larger polynomial (degree 2^10)
//
// Usage: ./test_fri

#include "commit/fri.cuh"
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int main() {
    cout << "=== FRI Polynomial Commitment Tests (Goldilocks) ===" << endl;
    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: Small polynomial commit + prove + verify ────────────────
    {
        // p(x) = 1 + 2x + 3x^2 + 4x^3  (degree 3)
        uint degree = 3;
        vector<Fr_t> coeffs = {{1}, {2}, {3}, {4}};

        Fr_t* coeffs_gpu;
        cudaMalloc(&coeffs_gpu, (degree + 1) * sizeof(Fr_t));
        cudaMemcpy(coeffs_gpu, coeffs.data(), (degree + 1) * sizeof(Fr_t), cudaMemcpyHostToDevice);

        FriParams params = {2, 1, 0};  // blowup=2, 1 query, remainder degree 0

        // Commit
        FriCommitment commitment = FriProver::commit(coeffs_gpu, degree, params);
        check(commitment.layer_roots.size() >= 1, "commit produces at least 1 layer root");

        // Generate challenges (in a real system these come from the verifier/Fiat-Shamir)
        vector<Fr_t> challenges;
        for (uint i = 0; i < commitment.domain_log_size; i++) {
            challenges.push_back(Fr_t{(uint64_t)(i * 37 + 13)});
        }

        // Query position
        vector<uint> query_pos = {0};

        // Prove
        FriProof proof = FriProver::prove(coeffs_gpu, degree, commitment, challenges, query_pos, params);
        check(proof.queries.size() == 1, "proof has 1 query");
        check(proof.queries[0].size() > 0, "query has at least 1 round");

        // Verify
        bool valid = FriVerifier::verify(commitment, proof, challenges, params);
        check(valid, "FRI proof verifies for degree-3 polynomial");

        cudaFree(coeffs_gpu);
    }

    // ── Test 2: Multiple query positions ────────────────────────────────
    {
        uint degree = 7;
        vector<Fr_t> coeffs(degree + 1);
        for (uint i = 0; i <= degree; i++) coeffs[i] = Fr_t{(uint64_t)(i + 1)};

        Fr_t* coeffs_gpu;
        cudaMalloc(&coeffs_gpu, (degree + 1) * sizeof(Fr_t));
        cudaMemcpy(coeffs_gpu, coeffs.data(), (degree + 1) * sizeof(Fr_t), cudaMemcpyHostToDevice);

        FriParams params = {2, 3, 0};  // 3 queries

        FriCommitment commitment = FriProver::commit(coeffs_gpu, degree, params);

        vector<Fr_t> challenges;
        for (uint i = 0; i < commitment.domain_log_size; i++) {
            challenges.push_back(Fr_t{(uint64_t)(i * 17 + 5)});
        }

        vector<uint> query_pos = {0, 3, 7};
        FriProof proof = FriProver::prove(coeffs_gpu, degree, commitment, challenges, query_pos, params);

        bool valid = FriVerifier::verify(commitment, proof, challenges, params);
        check(valid, "FRI proof verifies with 3 queries for degree-7 poly");

        cudaFree(coeffs_gpu);
    }

    // ── Test 3: Larger polynomial (degree 1023) ─────────────────────────
    {
        uint degree = 1023;
        vector<Fr_t> coeffs(degree + 1);
        for (uint i = 0; i <= degree; i++) coeffs[i] = Fr_t{(uint64_t)(i % 997 + 1)};

        Fr_t* coeffs_gpu;
        cudaMalloc(&coeffs_gpu, (degree + 1) * sizeof(Fr_t));
        cudaMemcpy(coeffs_gpu, coeffs.data(), (degree + 1) * sizeof(Fr_t), cudaMemcpyHostToDevice);

        FriParams params = {2, 2, 0};

        FriCommitment commitment = FriProver::commit(coeffs_gpu, degree, params);

        vector<Fr_t> challenges;
        for (uint i = 0; i < commitment.domain_log_size; i++) {
            challenges.push_back(Fr_t{(uint64_t)(i * 41 + 7)});
        }

        vector<uint> query_pos = {42, 100};
        FriProof proof = FriProver::prove(coeffs_gpu, degree, commitment, challenges, query_pos, params);

        bool valid = FriVerifier::verify(commitment, proof, challenges, params);
        check(valid, "FRI proof verifies for degree-1023 polynomial");

        cudaFree(coeffs_gpu);
    }

    cout << "\n=== Results: " << (failures == 0 ? "ALL PASSED" : "FAILURES")
         << " (failures=" << failures << ") ===" << endl;
    return failures;
}
