// Tests for zkNormalCDF.
// Build: add to Makefile as a target.
// Run: ./test_zknormalcdf

#include "zknn/zknormalcdf.cuh"
#include <iostream>
#include <cmath>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

static FrTensor from_long_val(long v) {
    unsigned long uv = (unsigned long)v;
    Fr_t f = FR_FROM_INT(uv);
    return FrTensor(1, &f);
}

static unsigned long fr_to_ul(const Fr_t& a) {
#ifdef USE_GOLDILOCKS
    return a.val;
#else
    return ((unsigned long)a.val[1] << 32) | a.val[0];
#endif
}

int main() {
    // sigma_eff = 1000 (field integer units), scale_out = 65536
    const uint precision = 16;
    const uint scale_out = 1u << 16;
    const double sigma_eff = 1000.0;

    zkNormalCDF cdf(precision, scale_out, sigma_eff);

    // ── Test 1: diff = 0 → Phi(0) = 0.5 ─────────────────────────────────
    {
        FrTensor diffs = from_long_val(0);
        auto [cdf_vals, m] = cdf.compute(diffs);
        Fr_t val = cdf_vals(0u);
        unsigned long result = fr_to_ul(val);
        double got      = (double)result / scale_out;
        double expected = 0.5;
        check(fabs(got - expected) < 0.01, "Phi(0) ≈ 0.5");
    }

    // ── Test 2: large positive diff → Phi ≈ 1.0 ─────────────────────────
    {
        // diff = 5*sigma_eff → Phi(5) ≈ 1.0
        long d = (long)(5.0 * sigma_eff);
        FrTensor diffs = from_long_val(d);
        auto [cdf_vals, m] = cdf.compute(diffs);
        Fr_t val = cdf_vals(0u);
        unsigned long result = fr_to_ul(val);
        double got = (double)result / scale_out;
        check(got > 0.99, "Phi(5*sigma) > 0.99");
    }

    // ── Test 3: diff = sigma_eff → Phi(1) ≈ 0.841 ───────────────────────
    {
        long d = (long)sigma_eff;  // diff = sigma → z = 1
        FrTensor diffs = from_long_val(d);
        auto [cdf_vals, m] = cdf.compute(diffs);
        Fr_t val = cdf_vals(0u);
        unsigned long result = fr_to_ul(val);
        double got      = (double)result / scale_out;
        double expected = 0.8413;
        check(fabs(got - expected) < 0.02, "Phi(sigma) ≈ 0.841");
    }

    // ── Test 4: monotonicity — larger diff → larger CDF value ────────────
    {
        long d1 = 100, d2 = 2000;
        FrTensor t1 = from_long_val(d1);
        FrTensor t2 = from_long_val(d2);
        auto [cv1, m1] = cdf.compute(t1);
        auto [cv2, m2] = cdf.compute(t2);
        unsigned long v1 = fr_to_ul(cv1(0u));
        unsigned long v2 = fr_to_ul(cv2(0u));
        check(v2 > v1, "larger diff gives larger CDF value");
    }

    // ── Test 5: clamping — diff beyond table range is clamped to max ─────
    {
        // Table range is [0, 2^precision) = [0, 65536). A diff of 70000 exceeds it.
        // After clamping (done by the caller in zkentropy), diff = 65535.
        // Just verify the compute doesn't crash on a max-range value.
        long d = (long)(1u << precision) - 1;  // max valid index
        FrTensor diffs = from_long_val(d);
        auto [cdf_vals, m] = cdf.compute(diffs);
        Fr_t val = cdf_vals(0u);
        unsigned long result = fr_to_ul(val);
        double got = (double)result / scale_out;
        check(got > 0.0 && got <= 1.0, "CDF at max table index is in [0, 1]");
    }

    cout << "\nAll zkNormalCDF tests passed." << endl;
    return 0;
}
