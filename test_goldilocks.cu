// test_goldilocks: verify correctness of Goldilocks field arithmetic.
//
// Tests addition, subtraction, multiplication, squaring, inverse,
// Montgomery identity, and edge cases. Runs on GPU (device functions)
// and reports pass/fail.
//
// Usage: ./test_goldilocks

#include "goldilocks.cuh"
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

using namespace std;

// ── Test kernel: each thread tests one property ─────────────────────────────

struct TestResult {
    int passed;
    int failed;
    int test_id_failed;  // first failing test
};

__global__
void test_kernel(TestResult* result) {
    // Only run on thread 0
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int passed = 0;
    int failed = 0;
    int first_fail = -1;

    auto check = [&](bool cond, int id) {
        if (cond) { passed++; }
        else { failed++; if (first_fail < 0) first_fail = id; }
    };

    Gold_t zero = {0ULL};
    Gold_t one  = {1ULL};
    Gold_t two  = {2ULL};
    Gold_t p_minus_1 = {GOLDILOCKS_P - 1};
    Gold_t p_minus_2 = {GOLDILOCKS_P - 2};

    // ── Test 1: zero + zero = zero ──────────────────────────────────────
    check(gold_eq(gold_add(zero, zero), zero), 1);

    // ── Test 2: one + zero = one ────────────────────────────────────────
    check(gold_eq(gold_add(one, zero), one), 2);

    // ── Test 3: one + one = two ─────────────────────────────────────────
    check(gold_eq(gold_add(one, one), two), 3);

    // ── Test 4: (p-1) + 1 = 0 (mod p) ──────────────────────────────────
    check(gold_eq(gold_add(p_minus_1, one), zero), 4);

    // ── Test 5: (p-1) + (p-1) = p-2 (mod p) ────────────────────────────
    check(gold_eq(gold_add(p_minus_1, p_minus_1), p_minus_2), 5);

    // ── Test 6: zero - one = p-1 ────────────────────────────────────────
    check(gold_eq(gold_sub(zero, one), p_minus_1), 6);

    // ── Test 7: one - one = zero ────────────────────────────────────────
    check(gold_eq(gold_sub(one, one), zero), 7);

    // ── Test 8: one * zero = zero ───────────────────────────────────────
    check(gold_eq(gold_mul(one, zero), zero), 8);

    // ── Test 9: one * one = one ─────────────────────────────────────────
    check(gold_eq(gold_mul(one, one), one), 9);

    // ── Test 10: two * two = four ───────────────────────────────────────
    Gold_t four = {4ULL};
    check(gold_eq(gold_mul(two, two), four), 10);

    // ── Test 11: (p-1) * (p-1) = 1 (since -1 * -1 = 1) ────────────────
    check(gold_eq(gold_mul(p_minus_1, p_minus_1), one), 11);

    // ── Test 12: two * (p-1) = p-2 (since 2 * -1 = -2) ────────────────
    check(gold_eq(gold_mul(two, p_minus_1), p_minus_2), 12);

    // ── Test 13: sqr(two) = four ────────────────────────────────────────
    check(gold_eq(gold_sqr(two), four), 13);

    // ── Test 14: inverse(one) = one ─────────────────────────────────────
    check(gold_eq(gold_inverse(one), one), 14);

    // ── Test 15: inverse(two) * two = one ───────────────────────────────
    Gold_t inv2 = gold_inverse(two);
    check(gold_eq(gold_mul(inv2, two), one), 15);

    // ── Test 16: inverse(p-1) = p-1 (since (-1)^(-1) = -1) ────────────
    check(gold_eq(gold_inverse(p_minus_1), p_minus_1), 16);

    // ── Test 17: inverse of a random-ish value ──────────────────────────
    Gold_t x = {12345678901234567ULL};
    Gold_t inv_x = gold_inverse(x);
    check(gold_eq(gold_mul(x, inv_x), one), 17);

    // ── Test 18: mont/unmont are identity ───────────────────────────────
    check(gold_eq(gold_mont(x), x), 18);
    check(gold_eq(gold_unmont(x), x), 19);

    // ── Test 20: double = add to self ───────────────────────────────────
    check(gold_eq(gold_double(x), gold_add(x, x)), 20);

    // ── Test 21: associativity (a + b) + c = a + (b + c) ───────────────
    Gold_t a = {0xDEADBEEF12345678ULL % GOLDILOCKS_P};
    Gold_t b = {0x1234567890ABCDEFULL % GOLDILOCKS_P};
    Gold_t c = {0xFEDCBA9876543210ULL % GOLDILOCKS_P};
    check(gold_eq(gold_add(gold_add(a, b), c), gold_add(a, gold_add(b, c))), 21);

    // ── Test 22: commutativity a * b = b * a ────────────────────────────
    check(gold_eq(gold_mul(a, b), gold_mul(b, a)), 22);

    // ── Test 23: distributivity a * (b + c) = a*b + a*c ────────────────
    Gold_t lhs = gold_mul(a, gold_add(b, c));
    Gold_t rhs = gold_add(gold_mul(a, b), gold_mul(a, c));
    check(gold_eq(lhs, rhs), 23);

    // ── Test 24: multiplication with large values near p ────────────────
    Gold_t big1 = {GOLDILOCKS_P - 2};
    Gold_t big2 = {GOLDILOCKS_P - 3};
    // (p-2)*(p-3) = (-2)*(-3) = 6 mod p
    Gold_t six = {6ULL};
    check(gold_eq(gold_mul(big1, big2), six), 24);

    // ── Test 25: pow(two, 0) = one ──────────────────────────────────────
    check(gold_eq(gold_pow(two, zero), one), 25);

    // ── Test 26: pow(two, 10) = 1024 ────────────────────────────────────
    Gold_t ten = {10ULL};
    Gold_t k1024 = {1024ULL};
    check(gold_eq(gold_pow(two, ten), k1024), 26);

    // ── Test 27: Fermat's little theorem: a^(p-1) = 1 ──────────────────
    check(gold_eq(gold_pow(a, p_minus_1), one), 27);

    // ── Test 28: a^p = a ────────────────────────────────────────────────
    Gold_t p_val = {GOLDILOCKS_P};
    // Can't directly use p as exponent (it's 0 mod p), use p-1 then multiply
    Gold_t a_pow_p = gold_mul(gold_pow(a, p_minus_1), a);
    check(gold_eq(a_pow_p, a), 28);

    // ── Test 29: div(a, b) * b = a ──────────────────────────────────────
    check(gold_eq(gold_mul(gold_div(a, b), b), a), 29);

    // ── Test 30: bit operations ─────────────────────────────────────────
    Gold_t bits_val = {0b10110101ULL};
    check(gold_get_bit(bits_val, 0) == 1, 30);
    check(gold_get_bit(bits_val, 1) == 0, 31);
    check(gold_get_bit(bits_val, 2) == 1, 32);
    check(gold_get_bits(bits_val, 0, 4) == 0b0101, 33);

    // ── Test 34: sub then add = identity ────────────────────────────────
    check(gold_eq(gold_add(gold_sub(a, b), b), a), 34);

    // ── Test 35: stress test — chain of multiplies ──────────────────────
    Gold_t acc = one;
    Gold_t base = {7ULL};
    for (int i = 0; i < 100; i++) acc = gold_mul(acc, base);
    // 7^100 mod p — verify by computing inverse chain
    Gold_t inv_acc = one;
    Gold_t inv_base = gold_inverse(base);
    for (int i = 0; i < 100; i++) inv_acc = gold_mul(inv_acc, inv_base);
    check(gold_eq(gold_mul(acc, inv_acc), one), 35);

    // ── Test 36: 2^32 - 1 is the "epsilon" value ───────────────────────
    // Verify: (2^32)^2 = 2^64 ≡ 2^32 - 1 (mod p)
    Gold_t two32 = {1ULL << 32};
    Gold_t eps = {GOLDILOCKS_P_NEG};  // 2^32 - 1
    Gold_t two64_mod_p = gold_mul(two32, two32);
    check(gold_eq(two64_mod_p, eps), 36);

    result->passed = passed;
    result->failed = failed;
    result->test_id_failed = first_fail;
}

int main() {
    cout << "Goldilocks field arithmetic tests" << endl;
    cout << "  p = 2^64 - 2^32 + 1 = " << GOLDILOCKS_P << endl;
    cout << endl;

    TestResult* d_result;
    TestResult h_result;
    cudaMalloc(&d_result, sizeof(TestResult));
    cudaMemset(d_result, 0, sizeof(TestResult));

    test_kernel<<<1, 1>>>(d_result);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    cudaMemcpy(&h_result, d_result, sizeof(TestResult), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    cout << "Passed: " << h_result.passed << endl;
    cout << "Failed: " << h_result.failed << endl;
    if (h_result.failed > 0) {
        cout << "First failure: test " << h_result.test_id_failed << endl;
        return 1;
    }
    cout << "All tests passed!" << endl;
    return 0;
}
