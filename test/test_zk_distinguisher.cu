// Phase 3 Step 4: distinguisher test for the ZK sumcheck drivers.
//
// Purpose: catch a future refactor that accidentally collapses any of
// the Σ-protocol blindings (round commit, proof-of-opening's (r_1, r_2),
// proof-of-equality's r) to zero, or reuses them across runs.  Such a
// break is soundness-preserving — the verifier still accepts — but
// destroys zero-knowledge, so no other test catches it.
//
// Method: same shape as test_opening_distinguisher.  Fix the statement
// (a, b, u, v); across NUM_RUNS runs, draw fresh σ-challenges.  Under
// the Hyrax §4 Protocol 3 zero-knowledge property (Theorem 3, Wahby et
// al. 2018 eprint 2017/1132 p. 12) every transcript element should be
// statistically indistinguishable from uniform across runs.
//
// Gates:
//   (a) Uniqueness: every run's transcript element is byte-distinct
//       from every other run's — regression to a deterministic
//       blinding makes uniqueness fail after two runs.
//   (b) χ² uniformity: bucket LSBs of each element, expect
//       NUM_RUNS / BINS per bin, reject at p < 0.001.
//
// Covered variants: IP (degree 2), degree-2 HP, multi-HP K=3.  Binary
// is excluded — it's declared in proof.cuh but has zero call sites in
// real code (same standing as before Phase 3), so no ZK driver exists.
//
// Representative elements sampled per run (round 0 of the first
// variant is sufficient — σ-challenges cascade through rounds, and
// any zeroed blinding in any round shows up in round 0):
//   T_final                      (G1)
//   rounds[0].T[0]               (G1; round commitment)
//   rounds[0].T_open[0].A        (G1; σ-opening commit)
//   rounds[0].T_open[0].z_m      (Fr; σ-opening response)
//   rounds[0].T_open[0].z_r      (Fr; σ-opening response)
//   rounds[0].eq_proof.A         (G1; σ-equality commit)
//   rounds[0].eq_proof.z         (Fr; σ-equality response)

#include "commit/commitment.cuh"
#include "proof/zk_sumcheck.cuh"
#include "tensor/fr-tensor.cuh"
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

static uint32_t bin_of(uint32_t w, uint32_t bins) { return w % bins; }

static double chi_square(const vector<uint32_t>& counts, double expected) {
    double chi2 = 0.0;
    for (auto c : counts) {
        double d = (double)c - expected;
        chi2 += d * d / expected;
    }
    return chi2;
}

// Collector bundle for a single variant's stats.
struct Collector {
    const char* variant;
    unordered_set<uint64_t> seen_T_final, seen_T0, seen_openA, seen_eqA;
    unordered_set<uint64_t> seen_zm, seen_zr, seen_eqz;
    vector<uint32_t> bins_T_final, bins_T0, bins_openA, bins_eqA;
    vector<uint32_t> bins_zm, bins_zr, bins_eqz;

    Collector(const char* v, uint bins)
        : variant(v),
          bins_T_final(bins, 0), bins_T0(bins, 0), bins_openA(bins, 0), bins_eqA(bins, 0),
          bins_zm(bins, 0),      bins_zr(bins, 0), bins_eqz(bins, 0) {}

    static uint64_t g1_key(const G1Jacobian_t& p) {
        return ((uint64_t)p.x.val[0] << 32) | p.x.val[1];
    }
    static uint64_t fr_key(const Fr_t& s) {
        return ((uint64_t)s.val[0] << 32) | s.val[1];
    }

    void record(const ZKSumcheckProof& pr, uint32_t bins) {
        seen_T_final.insert(g1_key(pr.T_final));
        bins_T_final[bin_of(pr.T_final.x.val[0], bins)]++;

        const auto& r0 = pr.rounds[0];
        seen_T0.insert(g1_key(r0.T[0]));
        bins_T0[bin_of(r0.T[0].x.val[0], bins)]++;

        seen_openA.insert(g1_key(r0.T_open[0].A));
        bins_openA[bin_of(r0.T_open[0].A.x.val[0], bins)]++;

        seen_zm.insert(fr_key(r0.T_open[0].z_m));
        bins_zm[bin_of(r0.T_open[0].z_m.val[0], bins)]++;
        seen_zr.insert(fr_key(r0.T_open[0].z_r));
        bins_zr[bin_of(r0.T_open[0].z_r.val[0], bins)]++;

        seen_eqA.insert(g1_key(r0.eq_proof.A));
        bins_eqA[bin_of(r0.eq_proof.A.x.val[0], bins)]++;
        seen_eqz.insert(fr_key(r0.eq_proof.z));
        bins_eqz[bin_of(r0.eq_proof.z.val[0], bins)]++;
    }

    void assert_hiding(uint NUM_RUNS, uint BINS, double CHI2_CRIT) {
        string pfx = string("[") + variant + "]";

        auto u_check = [&](const unordered_set<uint64_t>& s, const char* name) {
            string msg = pfx + " " + name + " distinct across runs";
            check(s.size() == NUM_RUNS, msg.c_str());
        };
        u_check(seen_T_final, "T_final");
        u_check(seen_T0,      "rounds[0].T[0]");
        u_check(seen_openA,   "rounds[0].T_open[0].A");
        u_check(seen_zm,      "rounds[0].T_open[0].z_m");
        u_check(seen_zr,      "rounds[0].T_open[0].z_r");
        u_check(seen_eqA,     "rounds[0].eq_proof.A");
        u_check(seen_eqz,     "rounds[0].eq_proof.z");

        double exp_per_bin = (double)NUM_RUNS / BINS;
        struct { const char* name; vector<uint32_t>* bins; } tests[] = {
            {"T_final",        &bins_T_final},
            {"rounds[0].T[0]", &bins_T0},
            {"T_open[0].A",    &bins_openA},
            {"T_open[0].z_m",  &bins_zm},
            {"T_open[0].z_r",  &bins_zr},
            {"eq_proof.A",     &bins_eqA},
            {"eq_proof.z",     &bins_eqz},
        };
        for (auto& t : tests) {
            double chi2 = chi_square(*t.bins, exp_per_bin);
            cout << "  " << pfx << " χ²(" << t.name << ") = " << chi2
                 << " (crit = " << CHI2_CRIT << ")" << endl;
            if (chi2 >= CHI2_CRIT) {
                cerr << "FAIL: " << pfx << " χ² uniformity rejected for "
                     << t.name << endl;
                exit(1);
            }
        }
    }
};

// ── Statement setup ───────────────────────────────────────────────────
// Tensors and eval point fixed across runs; only σ-challenges vary.
static const uint LOG_N = 6;          // N = 64 — 6 folding rounds
static const uint N     = 1u << LOG_N;

int main() {
    const uint NUM_RUNS   = 500;
    const uint BINS       = 10;
    // χ²_{0.001, 9} ≈ 27.88 — reject if we exceed this (p < 0.001).
    const double CHI2_CRIT = 27.88;

    // Fresh Pedersen (U, H) — independent random generators.
    Commitment sc_pp = Commitment::hiding_random(1);
    G1Jacobian_t U = sc_pp.u_generator;
    G1Jacobian_t H = sc_pp.hiding_generator;

    // Fixed witnesses + eval points — claim S is deterministic.
    FrTensor a_fixed = FrTensor::random(N);
    FrTensor b_fixed = FrTensor::random(N);
    vector<Fr_t> u_fixed, v_fixed;
    {
        FrTensor rand_uv = FrTensor::random(2 * LOG_N);
        for (uint i = 0; i < LOG_N; i++)       u_fixed.push_back(rand_uv(i));
        for (uint i = LOG_N; i < 2*LOG_N; i++) v_fixed.push_back(rand_uv(i));
    }

    // ── Variant 1: inner product ──────────────────────────────────────
    {
        Collector col("IP", BINS);
        Fr_t S_ip = (a_fixed * b_fixed).partial_me(u_fixed, N, 1)(0);
        for (uint i = 0; i < NUM_RUNS; i++) {
            FrTensor sig_rand = FrTensor::random(LOG_N * 4);
            vector<Fr_t> sigma(LOG_N * 4);
            for (uint k = 0; k < LOG_N * 4; k++) sigma[k] = sig_rand(k);

            Fr_t fa, fb;
            ZKSumcheckProverHandoff hand;
            ZKSumcheckProof pr = prove_zk_inner_product(
                U, H, S_ip, a_fixed, b_fixed,
                u_fixed, sigma, fa, fb, hand);
            col.record(pr, BINS);
        }
        col.assert_hiding(NUM_RUNS, BINS, CHI2_CRIT);
    }

    // ── Variant 2: degree-2 Hadamard product ──────────────────────────
    {
        Collector col("HP", BINS);
        // Claim S = (a∘b)(u) for Hadamard; same tensors.
        FrTensor had = a_fixed * b_fixed;
        Fr_t S_hp = had(u_fixed);
        for (uint i = 0; i < NUM_RUNS; i++) {
            FrTensor sig_rand = FrTensor::random(LOG_N * 4);
            vector<Fr_t> sigma(LOG_N * 4);
            for (uint k = 0; k < LOG_N * 4; k++) sigma[k] = sig_rand(k);

            Fr_t fa, fb;
            ZKSumcheckProverHandoff hand;
            ZKSumcheckProof pr = prove_zk_hadamard_product(
                U, H, S_hp, a_fixed, b_fixed,
                u_fixed, v_fixed, sigma, fa, fb, hand);
            col.record(pr, BINS);
        }
        col.assert_hiding(NUM_RUNS, BINS, CHI2_CRIT);
    }

    // ── Variant 3: multi-Hadamard, K = 3 ──────────────────────────────
    {
        Collector col("multi-HP K=3", BINS);
        FrTensor c_fixed = FrTensor::random(N);
        FrTensor prod = a_fixed * b_fixed * c_fixed;
        Fr_t S_mhp = prod(u_fixed);
        const uint K = 3;
        for (uint i = 0; i < NUM_RUNS; i++) {
            // Fresh copies each run — prover may fold in place.
            vector<FrTensor> Xs;
            Xs.push_back(a_fixed);
            Xs.push_back(b_fixed);
            Xs.push_back(c_fixed);

            FrTensor sig_rand = FrTensor::random(LOG_N * (K + 2));
            vector<Fr_t> sigma(LOG_N * (K + 2));
            for (uint k = 0; k < LOG_N * (K + 2); k++) sigma[k] = sig_rand(k);

            vector<Fr_t> final_Xs_out;
            ZKSumcheckProverHandoff hand;
            ZKSumcheckProof pr = prove_zk_multi_hadamard(
                U, H, S_mhp, Xs, u_fixed, v_fixed, sigma,
                final_Xs_out, hand);
            col.record(pr, BINS);
        }
        col.assert_hiding(NUM_RUNS, BINS, CHI2_CRIT);
    }

    cout << "\nAll ZK sumcheck distinguisher tests PASSED." << endl;
    return 0;
}
