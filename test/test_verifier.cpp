// test_verifier.cpp — Unit tests for CPU-only verifier components
//
// Tests field arithmetic, sumcheck verification, tLookup verification,
// SHA-256 / Merkle proofs, MLE evaluation, and proof parsing.
//
// Build: g++ -std=c++17 -O2 -o test_verifier test_verifier.cpp -lm
// Run:   ./test_verifier

#include "verifier_utils.h"
#include "sumcheck_verifier.h"
#include "tlookup_verifier.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

static int n_pass = 0;
static int n_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        n_pass++;
    } else {
        n_fail++;
        printf("  FAIL: %s\n", name);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Field arithmetic tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_field_arithmetic() {
    printf("=== Field Arithmetic ===\n");

    // Test 1: zero + zero = zero
    check(fr_eq(fr_add(FR_ZERO, FR_ZERO), FR_ZERO), "0 + 0 = 0");

    // Test 2: one + zero = one
    check(fr_eq(fr_add(FR_ONE, FR_ZERO), FR_ONE), "1 + 0 = 1");

    // Test 3: p-1 + 1 = 0 (wrap around)
    Fr_t p_minus_1 = {GOLDILOCKS_P - 1};
    check(fr_eq(fr_add(p_minus_1, FR_ONE), FR_ZERO), "(p-1) + 1 = 0");

    // Test 4: subtraction
    check(fr_eq(fr_sub(FR_ONE, FR_ONE), FR_ZERO), "1 - 1 = 0");

    // Test 5: 0 - 1 = p - 1
    check(fr_eq(fr_sub(FR_ZERO, FR_ONE), p_minus_1), "0 - 1 = p-1");

    // Test 6: multiplication by 1
    Fr_t val = {12345};
    check(fr_eq(fr_mul(val, FR_ONE), val), "x * 1 = x");

    // Test 7: multiplication by 0
    check(fr_eq(fr_mul(val, FR_ZERO), FR_ZERO), "x * 0 = 0");

    // Test 8: 2^32 * 2^32 = 2^32 - 1 (mod p) — the Goldilocks identity
    Fr_t two32 = {1ULL << 32};
    Fr_t eps = {GOLDILOCKS_P_NEG};
    check(fr_eq(fr_mul(two32, two32), eps), "2^32 * 2^32 = epsilon");

    // Test 9: inverse
    Fr_t a = {7};
    Fr_t a_inv = fr_inverse(a);
    check(fr_eq(fr_mul(a, a_inv), FR_ONE), "7 * 7^(-1) = 1");

    // Test 10: inverse of larger value
    Fr_t b = {123456789};
    Fr_t b_inv = fr_inverse(b);
    check(fr_eq(fr_mul(b, b_inv), FR_ONE), "123456789 * inv = 1");

    // Test 11: power
    Fr_t base = {3};
    Fr_t cubed = fr_pow(base, 3);
    check(fr_eq(cubed, Fr_t{27}), "3^3 = 27");

    // Test 12: larger power
    Fr_t pow10 = fr_pow(Fr_t{2}, 10);
    check(fr_eq(pow10, Fr_t{1024}), "2^10 = 1024");

    // Test 13: negation
    Fr_t neg_one = fr_neg(FR_ONE);
    check(fr_eq(fr_add(FR_ONE, neg_one), FR_ZERO), "1 + (-1) = 0");

    // Test 14: division
    Fr_t six = {6};
    Fr_t two = {2};
    Fr_t three = {3};
    check(fr_eq(fr_div(six, two), three), "6 / 2 = 3");

    // Test 15: associativity of multiplication
    Fr_t x = {999999};
    Fr_t y = {888888};
    Fr_t z = {777777};
    check(fr_eq(fr_mul(fr_mul(x, y), z), fr_mul(x, fr_mul(y, z))),
          "(x*y)*z = x*(y*z)");

    // Test 16: distributivity
    check(fr_eq(fr_mul(x, fr_add(y, z)), fr_add(fr_mul(x, y), fr_mul(x, z))),
          "x*(y+z) = x*y + x*z");

    // Test 17: from_u64 reduction
    Fr_t reduced = fr_from_u64(GOLDILOCKS_P);
    check(fr_eq(reduced, FR_ZERO), "from_u64(p) = 0");

    Fr_t reduced2 = fr_from_u64(GOLDILOCKS_P + 1);
    check(fr_eq(reduced2, FR_ONE), "from_u64(p+1) = 1");

    // Test 18: is_negative
    check(!fr_is_negative(FR_ZERO), "0 is not negative");
    check(!fr_is_negative(FR_ONE), "1 is not negative");
    check(fr_is_negative(p_minus_1), "p-1 is negative");
    check(fr_is_negative(fr_neg(FR_ONE)), "-1 is negative");

    // Test 19: operators
    check(FR_ONE + FR_ONE == Fr_t{2}, "operator+");
    check(Fr_t{5} - Fr_t{3} == Fr_t{2}, "operator-");
    check(Fr_t{3} * Fr_t{4} == Fr_t{12}, "operator*");
    check(FR_ONE != FR_ZERO, "operator!=");

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. SHA-256 tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_sha256() {
    printf("=== SHA-256 ===\n");

    // Test 1: deterministic hashing of field element
    Hash256 h1 = sha256_field_element(FR_ZERO);
    Hash256 h2 = sha256_field_element(FR_ZERO);
    check(h1 == h2, "hash(0) == hash(0) (deterministic)");

    // Test 2: different inputs give different hashes
    Hash256 h3 = sha256_field_element(FR_ONE);
    check(h1 != h3, "hash(0) != hash(1)");

    // Test 3: pair hashing is deterministic
    Hash256 hp1 = sha256_pair(h1, h3);
    Hash256 hp2 = sha256_pair(h1, h3);
    check(hp1 == hp2, "pair hash deterministic");

    // Test 4: pair hashing is order-dependent
    Hash256 hp3 = sha256_pair(h3, h1);
    check(hp1 != hp3, "pair hash order-dependent");

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Merkle proof tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_merkle() {
    printf("=== Merkle Proofs ===\n");

    // Build a small Merkle tree with 4 leaves
    Fr_t leaves[4] = {{10}, {20}, {30}, {40}};
    Hash256 lh[4];
    for (int i = 0; i < 4; i++) lh[i] = sha256_field_element(leaves[i]);

    Hash256 n01 = sha256_pair(lh[0], lh[1]);
    Hash256 n23 = sha256_pair(lh[2], lh[3]);
    Hash256 root = sha256_pair(n01, n23);

    // Test 1: verify leaf 0
    {
        std::vector<Hash256> path = {lh[1], n23};
        check(merkle_verify(root, lh[0], 0, path), "verify leaf 0");
    }

    // Test 2: verify leaf 1
    {
        std::vector<Hash256> path = {lh[0], n23};
        check(merkle_verify(root, lh[1], 1, path), "verify leaf 1");
    }

    // Test 3: verify leaf 2
    {
        std::vector<Hash256> path = {lh[3], n01};
        check(merkle_verify(root, lh[2], 2, path), "verify leaf 2");
    }

    // Test 4: verify leaf 3
    {
        std::vector<Hash256> path = {lh[2], n01};
        check(merkle_verify(root, lh[3], 3, path), "verify leaf 3");
    }

    // Test 5: wrong leaf should fail
    {
        std::vector<Hash256> path = {lh[1], n23};
        Hash256 wrong_leaf = sha256_field_element(Fr_t{99});
        check(!merkle_verify(root, wrong_leaf, 0, path), "wrong leaf rejected");
    }

    // Test 6: wrong index should fail
    {
        std::vector<Hash256> path = {lh[1], n23};
        check(!merkle_verify(root, lh[0], 1, path), "wrong index rejected");
    }

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. MLE evaluation tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_mle() {
    printf("=== Multilinear Evaluation ===\n");

    // Test 1: MLE of [a, b] at u = [0] gives a, at u = [1] gives b
    {
        std::vector<Fr_t> data = {{10}, {20}};
        check(fr_eq(mle_eval(data, {{0}}), Fr_t{10}), "MLE([10,20], [0]) = 10");
        check(fr_eq(mle_eval(data, {{1}}), Fr_t{20}), "MLE([10,20], [1]) = 20");
    }

    // Test 2: MLE of [a, b] at u = [1/2] (using inverse of 2)
    {
        Fr_t half = fr_inverse(Fr_t{2});
        std::vector<Fr_t> data = {{10}, {20}};
        // MLE at 1/2 = 10*(1-1/2) + 20*(1/2) = 5 + 10 = 15
        Fr_t result = mle_eval(data, {half});
        check(fr_eq(result, Fr_t{15}), "MLE([10,20], [1/2]) = 15");
    }

    // Test 3: MLE of 4 elements at binary points
    {
        std::vector<Fr_t> data = {{1}, {2}, {3}, {4}};
        // At (0,0) = data[0] = 1
        check(fr_eq(mle_eval(data, {{0}, {0}}), Fr_t{1}), "MLE(4, [0,0]) = 1");
        // At (1,0) = data[1] = 2
        // Note: index = u[0] + 2*u[1], so (u0=1, u1=0) → index 1
        check(fr_eq(mle_eval(data, {{1}, {0}}), Fr_t{2}), "MLE(4, [1,0]) = 2");
        // At (0,1) = data[2] = 3
        check(fr_eq(mle_eval(data, {{0}, {1}}), Fr_t{3}), "MLE(4, [0,1]) = 3");
        // At (1,1) = data[3] = 4
        check(fr_eq(mle_eval(data, {{1}, {1}}), Fr_t{4}), "MLE(4, [1,1]) = 4");
    }

    // Test 4: eq polynomial
    {
        std::vector<Fr_t> u = {{0}};
        std::vector<Fr_t> v = {{0}};
        check(fr_eq(eq_eval(u, v), FR_ONE), "eq([0],[0]) = 1");

        u = {{1}}; v = {{1}};
        check(fr_eq(eq_eval(u, v), FR_ONE), "eq([1],[1]) = 1");

        u = {{0}}; v = {{1}};
        check(fr_eq(eq_eval(u, v), FR_ZERO), "eq([0],[1]) = 0");

        u = {{1}}; v = {{0}};
        check(fr_eq(eq_eval(u, v), FR_ZERO), "eq([1],[0]) = 0");
    }

    // Test 5: eq polynomial at non-binary points
    {
        Fr_t half = fr_inverse(Fr_t{2});
        std::vector<Fr_t> u = {half};
        std::vector<Fr_t> v = {half};
        // eq(1/2, 1/2) = (1-1/2)(1-1/2) + (1/2)(1/2) = 1/4 + 1/4 = 1/2
        check(fr_eq(eq_eval(u, v), half), "eq([1/2],[1/2]) = 1/2");
    }

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Sumcheck verifier tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_sumcheck() {
    printf("=== Sumcheck Verifier ===\n");

    // Test 1: trivial 1-round sumcheck
    // Claim: sum = p(0) + p(1)
    // p(0) = 3, p(1) = 7, sum = 10
    {
        Fr_t claim = {10};
        std::vector<SumcheckRound> rounds = {{Fr_t{3}, Fr_t{7}, Fr_t{5}}};
        auto result = verify_sumcheck(claim, rounds);
        check(result.ok, "trivial sumcheck passes");
        check(fr_eq(result.final_claim, Fr_t{5}), "trivial sumcheck final claim = p(alpha)");
    }

    // Test 2: sumcheck with wrong claim should fail
    {
        Fr_t claim = {11};  // wrong
        std::vector<SumcheckRound> rounds = {{Fr_t{3}, Fr_t{7}, Fr_t{5}}};
        auto result = verify_sumcheck(claim, rounds);
        check(!result.ok, "wrong claim rejected");
    }

    // Test 3: 2-round sumcheck
    {
        // Round 1: p1(0)=4, p1(1)=6, claim=10, p1(alpha1)=8
        // Round 2: p2(0)=3, p2(1)=5, claim=8, p2(alpha2)=4
        Fr_t claim = {10};
        std::vector<SumcheckRound> rounds = {
            {Fr_t{4}, Fr_t{6}, Fr_t{8}},
            {Fr_t{3}, Fr_t{5}, Fr_t{4}}
        };
        auto result = verify_sumcheck(claim, rounds);
        check(result.ok, "2-round sumcheck passes");
        check(fr_eq(result.final_claim, Fr_t{4}), "2-round final claim");
    }

    // Test 4: Inner product sumcheck with known values
    // <a, b> where a = [2, 3], b = [5, 7]
    // <a, b> = 2*5 + 3*7 = 10 + 21 = 31
    {
        Fr_t claim = {31};
        // Simulate prover: 1 round for 2 elements
        // p(0) = a[0]*b[0] = 10, p(1) = a[1]*b[1] = 21
        // p(alpha) for some alpha, let's say alpha produces claim 15
        std::vector<SumcheckRound> rounds = {{Fr_t{10}, Fr_t{21}, Fr_t{15}}};
        IpSumcheckProof ip_proof;
        ip_proof.rounds = rounds;
        // After reduction: a(u) and b(u) such that a(u)*b(u) = 15
        ip_proof.final_a = Fr_t{3};
        ip_proof.final_b = Fr_t{5};
        auto result = verify_ip_sumcheck(claim, ip_proof);
        check(result.ok, "IP sumcheck: valid proof");
    }

    // Test 5: IP sumcheck with wrong final values should fail
    {
        Fr_t claim = {31};
        std::vector<SumcheckRound> rounds = {{Fr_t{10}, Fr_t{21}, Fr_t{15}}};
        IpSumcheckProof ip_proof;
        ip_proof.rounds = rounds;
        ip_proof.final_a = Fr_t{3};
        ip_proof.final_b = Fr_t{6};  // 3*6=18 != 15
        auto result = verify_ip_sumcheck(claim, ip_proof);
        check(!result.ok, "IP sumcheck: wrong final values rejected");
    }

    // Test 6: multi-round sumcheck with carry-through
    {
        // 3 rounds
        Fr_t claim = {100};
        std::vector<SumcheckRound> rounds = {
            {Fr_t{40}, Fr_t{60}, Fr_t{50}},
            {Fr_t{20}, Fr_t{30}, Fr_t{25}},
            {Fr_t{10}, Fr_t{15}, Fr_t{12}}
        };
        auto result = verify_sumcheck(claim, rounds);
        check(result.ok, "3-round sumcheck passes");
        check(fr_eq(result.final_claim, Fr_t{12}), "3-round final claim");
    }

    // Test 7: empty sumcheck (0 rounds)
    {
        Fr_t claim = {42};
        std::vector<SumcheckRound> rounds = {};
        auto result = verify_sumcheck(claim, rounds);
        check(result.ok, "0-round sumcheck passes");
        check(fr_eq(result.final_claim, claim), "0-round returns original claim");
    }

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. CDF and log table tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_tables() {
    printf("=== CDF / Log Tables ===\n");

    // Test 1: CDF at d=0 should be 0.5 * scale
    {
        uint64_t val = cdf_table_value(0, 5000.0, 65536);
        // Phi(0) = 0.5, so round(0.5 * 65536) = 32768
        check(val == 32768, "CDF(0) = 0.5 * scale");
    }

    // Test 2: CDF at very large d should approach scale
    {
        uint64_t val = cdf_table_value(100000, 5000.0, 65536);
        // Phi(100000/5000) = Phi(20) ≈ 1.0
        check(val == 65536 || val == 65535, "CDF(large) ≈ scale");
    }

    // Test 3: log table at q=1 should be max
    {
        uint64_t val = log_table_value(1, 15, 65536);
        // log_precision - log2(1) = 15 - 0 = 15
        // round(15 * 65536) = 983040
        check(val == 983040, "log(1) = precision * scale");
    }

    // Test 4: log table at q = 2^precision should be 0
    {
        uint64_t val = log_table_value(32768, 15, 65536);
        // log_precision - log2(32768) = 15 - 15 = 0
        check(val == 0, "log(2^p) = 0");
    }

    // Test 5: log table at q = 2^(precision-1) should be 1*scale
    {
        uint64_t val = log_table_value(16384, 15, 65536);
        // 15 - log2(16384) = 15 - 14 = 1
        check(val == 65536, "log(2^(p-1)) = scale");
    }

    // Test 6: build CDF table and spot check
    {
        auto table = build_cdf_table(4, 1000, 100.0);  // small table for testing
        check(table.size() == 16, "CDF table size = 2^4");
        check(table[0].val == 500, "CDF[0] = 500 (Phi(0)*1000)");
    }

    // Test 7: build log table and spot check
    {
        auto table = build_log_table(4, 1000);
        check(table.size() == 16, "log table size = 2^4");
        // Entry 0 is for input q=1: round((4 - log2(1)) * 1000) = 4000
        check(table[0].val == 4000, "log[0] = 4000");
        // Last entry is for input q=16: round((4 - log2(16)) * 1000) = 0
        check(table[15].val == 0, "log[15] = 0");
    }

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Polynomial evaluation tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_polynomial() {
    printf("=== Polynomial ===\n");

    // Test 1: constant polynomial
    {
        Polynomial p;
        p.coeffs = {{42}};
        check(fr_eq(p.eval(FR_ZERO), Fr_t{42}), "const poly eval(0) = 42");
        check(fr_eq(p.eval(FR_ONE), Fr_t{42}), "const poly eval(1) = 42");
        check(fr_eq(p.constant(), Fr_t{42}), "const poly constant() = 42");
    }

    // Test 2: linear polynomial p(x) = 3 + 5x
    {
        Polynomial p;
        p.coeffs = {{3}, {5}};
        check(fr_eq(p.eval(FR_ZERO), Fr_t{3}), "linear eval(0) = 3");
        check(fr_eq(p.eval(FR_ONE), Fr_t{8}), "linear eval(1) = 8");
        check(fr_eq(p.eval(Fr_t{2}), Fr_t{13}), "linear eval(2) = 13");
    }

    // Test 3: quadratic polynomial p(x) = 1 + 2x + 3x^2
    {
        Polynomial p;
        p.coeffs = {{1}, {2}, {3}};
        check(fr_eq(p.eval(FR_ZERO), Fr_t{1}), "quad eval(0) = 1");
        check(fr_eq(p.eval(FR_ONE), Fr_t{6}), "quad eval(1) = 6");
        // p(2) = 1 + 4 + 12 = 17
        check(fr_eq(p.eval(Fr_t{2}), Fr_t{17}), "quad eval(2) = 17");
    }

    // Test 4: empty polynomial
    {
        Polynomial p;
        check(fr_eq(p.eval(FR_ONE), FR_ZERO), "empty poly eval = 0");
        check(fr_eq(p.constant(), FR_ZERO), "empty poly constant = 0");
    }

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. ceil_log2 tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_ceil_log2() {
    printf("=== ceil_log2 ===\n");

    check(ceil_log2(1) == 0, "ceil_log2(1) = 0");
    check(ceil_log2(2) == 1, "ceil_log2(2) = 1");
    check(ceil_log2(3) == 2, "ceil_log2(3) = 2");
    check(ceil_log2(4) == 2, "ceil_log2(4) = 2");
    check(ceil_log2(5) == 3, "ceil_log2(5) = 3");
    check(ceil_log2(8) == 3, "ceil_log2(8) = 3");
    check(ceil_log2(32768) == 15, "ceil_log2(32768) = 15");
    check(ceil_log2(32769) == 16, "ceil_log2(32769) = 16");

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. End-to-end: write and read a synthetic proof file
// ═══════════════════════════════════════════════════════════════════════════

static void test_proof_roundtrip() {
    printf("=== Proof File Round-Trip ===\n");

    // Create a minimal synthetic proof file
    const char* tmpfile = "/tmp/test_verifier_proof.bin";
    {
        FILE* f = fopen(tmpfile, "wb");
        assert(f);

        // v2 header
        uint64_t magic = 0x5A4B454E54524F50ULL;
        uint64_t entropy_val = 1000;
        uint32_t T = 1;
        uint32_t vocab_size = 32000;
        double sigma_eff = 5223.0;
        uint32_t log_scale = 256;
        uint32_t cdf_precision = 15;
        uint32_t log_precision = 15;
        uint32_t cdf_scale = 65536;

        fwrite(&magic, 8, 1, f);
        fwrite(&entropy_val, 8, 1, f);
        fwrite(&T, 4, 1, f);
        fwrite(&vocab_size, 4, 1, f);
        fwrite(&sigma_eff, 8, 1, f);
        fwrite(&log_scale, 4, 1, f);
        fwrite(&cdf_precision, 4, 1, f);
        fwrite(&log_precision, 4, 1, f);
        fwrite(&cdf_scale, 4, 1, f);

        // 8 polynomials for 1 position
        uint32_t n_polys = 8;
        fwrite(&n_polys, 4, 1, f);

        // Write 8 constant polynomials (1 coefficient each)
        // Compute values that will pass verification:
        // diff_actual = 0 → CDF(0) = 32768, win_prob = 65536 - 32768 = 32768
        // total_win = 32768 (valid: >= win_prob, <= 32000*65536)
        // q_fr = clamp(32768 * 32768 / 32768, 1, 32768) = 32768
        // surprise = log_table_value(32768, 15, 256) = round((15 - 15)*256) = 0

        // But entropy_val = 1000, so surprise must = 1000.
        // Let's pick diff_actual = 10000
        uint64_t diff_actual = 10000;
        uint64_t cdf_val_computed = cdf_table_value(diff_actual, sigma_eff, cdf_scale);
        uint64_t win_prob_val = cdf_scale - cdf_val_computed;
        uint64_t total_win_val = win_prob_val;  // self-reported as win_prob for simplicity
        uint64_t q_fr_val = (win_prob_val * (1u << log_precision)) / total_win_val;
        if (q_fr_val < 1) q_fr_val = 1;
        if (q_fr_val > (1u << log_precision)) q_fr_val = (1u << log_precision);
        uint64_t surprise_val = log_table_value(q_fr_val, log_precision, log_scale);
        entropy_val = surprise_val;

        // Rewrite header with correct entropy_val
        fseek(f, 8, SEEK_SET);
        fwrite(&entropy_val, 8, 1, f);
        fseek(f, 0, SEEK_END);

        uint64_t poly_vals[8] = {
            1,              // ind_sum = 1
            0,              // ind_dot = 0
            42,             // logit_act (arbitrary)
            diff_actual,    // diff_actual
            win_prob_val,   // win_prob
            total_win_val,  // total_win
            q_fr_val,       // q_fr
            surprise_val    // surprise
        };

        for (int i = 0; i < 8; i++) {
            uint32_t n_coeffs = 1;
            fwrite(&n_coeffs, 4, 1, f);
            fwrite(&poly_vals[i], 8, 1, f);
        }

        fclose(f);
    }

    // Parse it back
    try {
        auto proof = parse_proof_file(tmpfile);
        check(proof.header.T == 1, "round-trip: T = 1");
        check(proof.header.vocab_size == 32000, "round-trip: vocab_size = 32000");
        check(proof.header.is_v2, "round-trip: detected v2 header");
        check(proof.header.cdf_precision == 15, "round-trip: cdf_precision = 15");
        check(proof.header.log_precision == 15, "round-trip: log_precision = 15");
        check(proof.header.cdf_scale == 65536, "round-trip: cdf_scale = 65536");
        check(proof.all_polys.size() == 8, "round-trip: 8 polynomials");
        check(proof.positions.size() == 1, "round-trip: 1 position");
        check(proof.positions[0].ind_sum.val == 1, "round-trip: ind_sum = 1");
        check(proof.positions[0].ind_dot.val == 0, "round-trip: ind_dot = 0");
    } catch (const std::exception& e) {
        printf("  EXCEPTION: %s\n", e.what());
        n_fail++;
    }

    // Clean up
    remove(tmpfile);

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. tLookup verifier structural test
// ═══════════════════════════════════════════════════════════════════════════

static void test_tlookup() {
    printf("=== tLookup Verifier ===\n");

    // Test 1: verify that phase round count calculation is correct
    {
        // D=16, N=4 → phase1 = ceil_log2(16/4) = 2, phase2 = ceil_log2(4) = 2
        uint32_t D = 16, N = 4;
        uint32_t p1 = ceil_log2(D / N);
        uint32_t p2 = ceil_log2(N);
        check(p1 == 2, "tLookup phase1 rounds (D=16,N=4) = 2");
        check(p2 == 2, "tLookup phase2 rounds (D=16,N=4) = 2");
    }

    // Test 2: combined table MLE computation
    {
        // Table: [0, 1, 2, 3] (range table with low=0)
        // Mapped: [10, 20, 30, 40]
        // Combined with r=2: [0+2*10, 1+2*20, 2+2*30, 3+2*40] = [20, 41, 62, 83]
        std::vector<Fr_t> mapped = {{10}, {20}, {30}, {40}};
        Fr_t r = {2};
        // At v = [0, 0]: index 0, value 20
        Fr_t val = compute_combined_table_mle(0, mapped, r, {{0}, {0}});
        check(fr_eq(val, Fr_t{20}), "combined table MLE at (0,0)");
        // At v = [1, 0]: index 1, value 41
        val = compute_combined_table_mle(0, mapped, r, {{1}, {0}});
        check(fr_eq(val, Fr_t{41}), "combined table MLE at (1,0)");
    }

    // Test 3: tLookup with wrong phase counts should fail
    {
        TLookupProof proof;
        proof.phase1_rounds = {{Fr_t{1}, Fr_t{2}, Fr_t{3}}};  // 1 round
        proof.phase2_rounds = {};  // 0 rounds
        proof.final_A = FR_ONE;
        proof.final_B = FR_ONE;
        proof.final_S = FR_ZERO;
        proof.final_T = FR_ZERO;
        proof.final_m = FR_ONE;

        auto result = verify_tlookup(Fr_t{5}, Fr_t{7}, 8, 4, proof);
        // D=8, N=4 → phase1 = 1, phase2 = 2
        // We gave 0 phase2 rounds, should fail
        check(!result.ok, "tLookup wrong phase2 count rejected");
    }

    printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    printf("test_verifier — CPU-only verifier component tests\n\n");

    test_field_arithmetic();
    test_sha256();
    test_merkle();
    test_mle();
    test_sumcheck();
    test_tables();
    test_polynomial();
    test_ceil_log2();
    test_proof_roundtrip();
    test_tlookup();

    printf("════════════════════════════════\n");
    printf("Total: %d passed, %d failed\n", n_pass, n_fail);

    if (n_fail == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
