// Tests for zkArgmax.
// Build: add to Makefile as a target (see Makefile).
// Run: ./test_zkargmax  (BLS) or ./gold_test_zkargmax (Goldilocks)

#include "zknn/zkargmax.cuh"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

// Helper: create FrTensor on GPU from a host vector of longs.
static FrTensor from_longs(const vector<long>& vals) {
    uint N = vals.size();
    Fr_t* cpu = new Fr_t[N];
    for (uint i = 0; i < N; i++) {
        long v = vals[i];
#ifdef USE_GOLDILOCKS
        if (v >= 0) {
            cpu[i] = {(uint64_t)v};
        } else {
            // Negative: represent as p - |v|
            cpu[i] = {GOLDILOCKS_P - (uint64_t)(-v)};
        }
#else
        if (v >= 0) {
            cpu[i] = {(uint)(v & 0xFFFFFFFF), (uint)((unsigned long)v >> 32),
                      0, 0, 0, 0, 0, 0};
        } else {
            unsigned long uv = (unsigned long)v;
            cpu[i] = {(uint)(uv & 0xFFFFFFFF), (uint)(uv >> 32), 0, 0, 0, 0, 0, 0};
        }
#endif
    }
    FrTensor t(N, cpu);
    delete[] cpu;
    return t;
}

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

int main() {
    zkArgmax am(16);  // 16-bit range proof

    // ── Test 1: correct argmax on small positive values ───────────────────
    {
        auto logits = from_longs({100, 500, 200, 50, 300});
        uint t = am.compute(logits);
        check(t == 1, "argmax of {100,500,200,50,300} == 1");
    }

    // ── Test 2: argmax on single element ─────────────────────────────────
    {
        auto logits = from_longs({42});
        uint t = am.compute(logits);
        check(t == 0, "argmax of {42} == 0");
    }

    // ── Test 3: argmax with all equal values picks first winner ──────────
    {
        auto logits = from_longs({7, 7, 7, 7});
        uint t = am.compute(logits);
        check(t < 4, "argmax of all-equal is a valid index");
    }

    // ── Test 4: prove accepts correct argmax ─────────────────────────────
    {
        auto logits = from_longs({10, 99, 55, 3});
        uint t_star = am.compute(logits);
        Fr_t v_star = logits(t_star);
        auto u = random_vec(ceilLog2(4u));
        vector<Polynomial> proof;
        Fr_t claim = am.prove(logits, t_star, v_star, u, proof);
        check(true, "prove does not throw for correct argmax");
        (void)claim;
    }

    // ── Test 5: prove rejects wrong v_star ───────────────────────────────
    {
        auto logits = from_longs({10, 99, 55, 3});
        uint t_star = 1;  // correct
        Fr_t wrong_v = logits(2u);
        auto u = random_vec(ceilLog2(4u));
        vector<Polynomial> proof;
        bool threw = false;
        try {
            am.prove(logits, t_star, wrong_v, u, proof);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "prove throws when v_star != logits[t_star]");
    }

    // ── Test 6: prove rejects wrong t_star ───────────────────────────────
    {
        auto logits = from_longs({10, 99, 55, 3});
        uint wrong_t = 2;  // not the argmax
        Fr_t v_star = logits(wrong_t);
        auto u = random_vec(ceilLog2(4u));
        vector<Polynomial> proof;
        bool threw = false;
        try {
            am.prove(logits, wrong_t, v_star, u, proof);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "prove throws when t_star is not the argmax (negative diff found)");
    }

    // ── Test 7: prove rejects inflated v_star (soundness of indicator) ───
    //    A cheating prover sets v_star above the true max.  All diffs are
    //    non-negative and fit in bit_width bits, but the indicator check
    //    (<ind, diffs> != 0) catches it.
    {
        auto logits = from_longs({10, 99, 55, 3});
        uint t_star = 1;  // true argmax index
        // Inflate v_star: use 150 instead of the real max (99)
        Fr_t inflated_v = FR_FROM_INT(150);
        auto u = random_vec(ceilLog2(4u));
        vector<Polynomial> proof;
        bool threw = false;
        try {
            am.prove(logits, t_star, inflated_v, u, proof);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "prove throws when v_star is inflated above true max");
    }

    cout << "\nAll zkArgmax tests passed." << endl;
    return 0;
}
