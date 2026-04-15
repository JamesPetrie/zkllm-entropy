// Phase 2 distinguisher test (plan task #15).
//
// Purpose: catch a future refactor that accidentally collapses the
// Figure 6 blinding scalars (d⃗, r_δ, r_β, r_τ) to zero or reuses them
// across openings.  That break is soundness-preserving (verifier still
// accepts) but destroys hiding — so no other test in the suite catches it.
//
// Method: fix pp, t, u; run N openings at the same u with fresh
// challenges (and thus fresh blindings).  Under the Hyrax hiding
// property (Theorem 11, eprint 2017/1132 p. 18) every transcript
// element should be statistically indistinguishable from uniform across
// runs.  We gate on two signals:
//
//   (a) Uniqueness: every run's transcript element is byte-distinct
//       from every other run's (collision probability under true
//       uniform on a 32-bit projection is <1e-70 for N=5000).
//   (b) χ² uniformity: project each element into B=20 bins via
//       low-order bits of a chosen limb; expected count per bin is
//       N/B = 250; reject if χ² > 43.82 (19 dof, p < 0.001).
//
// Elements tested: τ.x, δ.x, β.x, z[0], z_δ, z_β, r_τ.
// (x-coord of a G1 is a field element; using its low 32 bits as the
// hash is a reasonable uniform projection for a random curve point.)
//
// A regression that sets d⃗ = 0 makes δ deterministic → uniqueness
// fails in run 2.  A regression that reuses r_τ across openings makes
// τ deterministic → uniqueness fails.  A regression that biases the
// RNG makes χ² blow up.

#include "commit/commitment.cuh"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <unordered_set>
#include <cstdint>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

// Hash a 32-bit word to one of `bins`.
static uint32_t bin_of(uint32_t w, uint32_t bins) { return w % bins; }

// χ² statistic for observed counts against uniform expected.
static double chi_square(const vector<uint32_t>& counts, double expected) {
    double chi2 = 0.0;
    for (auto c : counts) {
        double d = (double)c - expected;
        chi2 += d * d / expected;
    }
    return chi2;
}

int main() {
    const uint N = 8;
    const uint NUM_RUNS = 5000;
    const uint BINS = 20;
    // χ²_{0.001, 19} ≈ 43.82 — reject if we exceed this (p < 0.001).
    const double CHI2_CRIT = 43.82;

    Commitment pp = Commitment::hiding_random(N);
    FrTensor t = FrTensor::random(N);
    auto hc = pp.commit_hiding(t);
    vector<Fr_t> u;
    {
        FrTensor u_rand = FrTensor::random(3);
        for (uint i = 0; i < 3; i++) u.push_back(u_rand(i));
    }

    // Collectors.
    unordered_set<uint64_t> seen_tau, seen_delta, seen_beta;
    unordered_set<uint64_t> seen_z0, seen_zdelta, seen_zbeta, seen_rtau;

    vector<uint32_t> bins_tau(BINS, 0), bins_delta(BINS, 0), bins_beta(BINS, 0);
    vector<uint32_t> bins_z0(BINS, 0), bins_zdelta(BINS, 0), bins_zbeta(BINS, 0), bins_rtau(BINS, 0);

    for (uint i = 0; i < NUM_RUNS; i++) {
        Fr_t c = FrTensor::random(1)(0);
        auto res = pp.open_zk(t, hc.r, hc.com, u, c);

        // 64-bit projection: concatenate two 32-bit limbs of the x-coord
        // (for G1) or the scalar (for Fr) — arbitrary but fixed choice.
        auto g1_key = [](const G1Jacobian_t& p) -> uint64_t {
            return ((uint64_t)p.x.val[0] << 32) | p.x.val[1];
        };
        auto fr_key = [](const Fr_t& s) -> uint64_t {
            return ((uint64_t)s.val[0] << 32) | s.val[1];
        };

        seen_tau.insert(g1_key(res.proof.tau));
        seen_delta.insert(g1_key(res.proof.delta));
        seen_beta.insert(g1_key(res.proof.beta));
        seen_z0.insert(fr_key(res.proof.z(0)));
        seen_zdelta.insert(fr_key(res.proof.z_delta));
        seen_zbeta.insert(fr_key(res.proof.z_beta));
        seen_rtau.insert(fr_key(res.proof.r_tau));

        bins_tau[bin_of(res.proof.tau.x.val[0], BINS)]++;
        bins_delta[bin_of(res.proof.delta.x.val[0], BINS)]++;
        bins_beta[bin_of(res.proof.beta.x.val[0], BINS)]++;
        bins_z0[bin_of(res.proof.z(0).val[0], BINS)]++;
        bins_zdelta[bin_of(res.proof.z_delta.val[0], BINS)]++;
        bins_zbeta[bin_of(res.proof.z_beta.val[0], BINS)]++;
        bins_rtau[bin_of(res.proof.r_tau.val[0], BINS)]++;
    }

    // Uniqueness: expect all NUM_RUNS values distinct (birthday bound).
    check(seen_tau.size()    == NUM_RUNS, "τ distinct across 5000 openings");
    check(seen_delta.size()  == NUM_RUNS, "δ distinct across 5000 openings");
    check(seen_beta.size()   == NUM_RUNS, "β distinct across 5000 openings");
    check(seen_z0.size()     == NUM_RUNS, "z[0] distinct across 5000 openings");
    check(seen_zdelta.size() == NUM_RUNS, "z_δ distinct across 5000 openings");
    check(seen_zbeta.size()  == NUM_RUNS, "z_β distinct across 5000 openings");
    check(seen_rtau.size()   == NUM_RUNS, "r_τ distinct across 5000 openings");

    // χ² uniformity.
    double exp_per_bin = (double)NUM_RUNS / BINS;
    struct { const char* name; vector<uint32_t>* bins; } tests[] = {
        {"τ",   &bins_tau},
        {"δ",   &bins_delta},
        {"β",   &bins_beta},
        {"z[0]", &bins_z0},
        {"z_δ", &bins_zdelta},
        {"z_β", &bins_zbeta},
        {"r_τ", &bins_rtau},
    };
    for (auto& tst : tests) {
        double chi2 = chi_square(*tst.bins, exp_per_bin);
        cout << "  χ²(" << tst.name << ") = " << chi2
             << " (crit = " << CHI2_CRIT << ")" << endl;
        if (chi2 >= CHI2_CRIT) {
            cerr << "FAIL: χ² uniformity rejected for " << tst.name << endl;
            exit(1);
        }
    }
    cout << "PASS: χ² uniformity gate (p > 0.001) for all 7 elements" << endl;

    cout << "All distinguisher tests PASSED." << endl;
    return 0;
}
