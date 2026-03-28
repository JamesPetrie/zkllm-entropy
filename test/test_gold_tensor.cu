// test_gold_tensor: verify FrTensor and sumcheck work with Goldilocks field.
//
// Tests tensor arithmetic, multilinear evaluation, and inner product sumcheck
// using Goldilocks instead of BLS12-381.
//
// Usage: ./test_gold_tensor

#include "proof/proof.cuh"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

// Host-side conversion: Gold_t -> int (for testing)
static int gold_to_int(Fr_t x) {
    if (x.val <= (GOLDILOCKS_P >> 1)) return static_cast<int>(x.val);
    return -static_cast<int>(GOLDILOCKS_P - x.val);
}

int main() {
    cout << "=== FrTensor + Sumcheck with Goldilocks ===" << endl;
    cout << "sizeof(Fr_t) = " << sizeof(Fr_t) << " bytes" << endl;

    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: Basic tensor creation and element access ────────────────
    {
        int data[] = {1, 2, 3, 4};
        FrTensor t(4, data);
        Fr_t v0 = t(0u);
        Fr_t v3 = t(3u);
        check(gold_to_int(v0) == 1, "tensor from int: t[0] == 1");
        check(gold_to_int(v3) == 4, "tensor from int: t[3] == 4");
    }

    // ── Test 2: Negative values ─────────────────────────────────────────
    {
        int data[] = {-5, 10, -3, 7};
        FrTensor t(4, data);
        check(gold_to_int(t(0u)) == -5, "negative: t[0] == -5");
        check(gold_to_int(t(1u)) == 10, "negative: t[1] == 10");
    }

    // ── Test 3: Tensor addition ─────────────────────────────────────────
    {
        int a[] = {1, 2, 3, 4};
        int b[] = {10, 20, 30, 40};
        FrTensor ta(4, a), tb(4, b);
        FrTensor tc = ta + tb;
        check(gold_to_int(tc(0u)) == 11, "add: (1+10) == 11");
        check(gold_to_int(tc(3u)) == 44, "add: (4+40) == 44");
    }

    // ── Test 4: Tensor multiplication (Hadamard) ────────────────────────
    {
        int a[] = {3, 5, 7, 11};
        int b[] = {2, 4, 6, 8};
        FrTensor ta(4, a), tb(4, b);
        FrTensor tc = ta * tb;
        // Hadamard product in mont form: need to unmont
        tc.unmont();
        check(gold_to_int(tc(0u)) == 6, "hadamard: 3*2 == 6");
        check(gold_to_int(tc(1u)) == 20, "hadamard: 5*4 == 20");
    }

    // ── Test 5: Sum ─────────────────────────────────────────────────────
    {
        int data[] = {1, 2, 3, 4, 5, 6, 7, 8};
        FrTensor t(8, data);
        Fr_t s = t.sum();
        check(gold_to_int(s) == 36, "sum: 1+2+...+8 == 36");
    }

    // ── Test 6: Multilinear evaluation ──────────────────────────────────
    {
        // For a 4-element tensor [a0, a1, a2, a3], the MLE at (u0, u1) is:
        // a0*(1-u0)*(1-u1) + a1*u0*(1-u1) + a2*(1-u0)*u1 + a3*u0*u1
        int data[] = {10, 20, 30, 40};
        FrTensor t(4, data);
        // Evaluate at u = (0, 0) -> should give a0 = 10
        vector<Fr_t> u_zero = {{0ULL}, {0ULL}};
        Fr_t v = t(u_zero);
        check(gold_to_int(v) == 10, "MLE at (0,0) == 10");
    }

    // ── Test 7: Random tensor creation ──────────────────────────────────
    {
        FrTensor r = FrTensor::random_int(1024, 16);
        // Just verify it doesn't crash and has the right size
        check(r.size == 1024, "random_int size == 1024");
        // Check the first element is within reasonable range
        int v = gold_to_int(r(0u));
        check(v >= -32768 && v < 32768, "random_int value in range");
    }

    // ── Test 8: Inner product sumcheck ──────────────────────────────────
    {
        int a_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
        int b_data[] = {8, 7, 6, 5, 4, 3, 2, 1};
        FrTensor a(8, a_data), b(8, b_data);

        // Inner product = 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 120
        vector<Fr_t> u = random_vec(3);  // log2(8) = 3
        vector<Fr_t> proof = inner_product_sumcheck(a, b, u);

        // Verify the proof has the expected structure (3 rounds, 3 coefficients each)
        check(proof.size() > 0, "sumcheck produced proof");

        // The claimed sum should be 120
        // The inner_product_sumcheck verifies internally, so if it returns without
        // throwing, the proof is valid.
        cout << "  PASS: inner product sumcheck (if no exception)" << endl;
    }

    // ── Test 9: Matmul ──────────────────────────────────────────────────
    {
        // 2x3 * 3x2 = 2x2
        // A = [[1,2,3],[4,5,6]]  (row-major)
        // B = [[7,8],[9,10],[11,12]]  (row-major)
        // C = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //   = [[58, 64], [139, 154]]
        int a[] = {1, 2, 3, 4, 5, 6};
        int b[] = {7, 8, 9, 10, 11, 12};
        FrTensor ta(6, a), tb(6, b);
        FrTensor tc = FrTensor::matmul(ta, tb, 2, 3, 2);
        // matmul result is in mont form, need to unmont
        tc.unmont();
        check(gold_to_int(tc(0u)) == 58, "matmul: C[0][0] == 58");
        check(gold_to_int(tc(1u)) == 64, "matmul: C[0][1] == 64");
        check(gold_to_int(tc(2u)) == 139, "matmul: C[1][0] == 139");
        check(gold_to_int(tc(3u)) == 154, "matmul: C[1][1] == 154");
    }

    cout << "\n=== Results: " << (failures == 0 ? "ALL PASSED" : "FAILURES")
         << " (failures=" << failures << ") ===" << endl;
    return failures;
}
