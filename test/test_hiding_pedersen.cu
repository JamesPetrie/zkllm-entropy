// Tests for Phase 1 hiding Pedersen commitments.
//
// Covers:
//   (1) is_hiding() sanity: random()→false, hiding_random()→true
//   (2) save_hiding / load_hiding pp roundtrip with .h sidecar
//   (3) commit_hiding refuses non-hiding pp (fails loud, not silent)
//   (4) Collision resistance: N commitments of the same message with
//       fresh r are all byte-distinct.  Catches r=0 / fixed-r bugs that
//       are the #1 red-team target on a hiding commitment.
//   (5) r tensor freshness: the r values returned across calls are
//       byte-distinct.  Guards the RNG path specifically.
//   (6) Algebraic identity: for t = 0-vector, commit_hiding(t).com[row]
//       equals the direct size-1 commit of r[row] against H — i.e. the
//       implementation really is computing Σ tᵢGᵢ + r·H and not
//       something else.
//
// Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//   "Perfect hiding: For any m_0, m_1 ∈ {0,1}^λ and m_0 ≠ m_1:
//    {Com(m_0; r)}_{r←R} and {Com(m_1; r)}_{r←R} are identically
//    distributed."
//
// Perfect hiding is a distributional claim; the tests here are
// correctness/non-collision gates, not a statistical distinguisher.
// The simulator argument in docs/plans/phase-1-hiding-pedersen.md is
// the distributional statement; these tests are the structural gates
// that the simulator argument rests on.

#include "commit/commitment.cuh"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <unistd.h>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

// Byte-exact equality on a G1Jacobian_t.  We only use this for *equality
// of the same algebraic path's output* (deterministic arithmetic) or to
// confirm *inequality*; byte inequality between two distinct commitments
// is itself enough to prove the points differ as group elements.
static bool g1_bytes_eq(const G1Jacobian_t& a, const G1Jacobian_t& b) {
    return memcmp(&a, &b, sizeof(G1Jacobian_t)) == 0;
}

static bool fr_bytes_eq(const Fr_t& a, const Fr_t& b) {
    return memcmp(&a, &b, sizeof(Fr_t)) == 0;
}

int main() {
    // ── Test 1: is_hiding() sanity ────────────────────────────────────────
    {
        Commitment pp_nh = Commitment::random(4);
        check(!pp_nh.is_hiding(),
              "Commitment::random produces non-hiding pp (H == identity)");

        Commitment pp_h = Commitment::hiding_random(4);
        check(pp_h.is_hiding(),
              "Commitment::hiding_random produces hiding pp (H != identity)");
    }

    // ── Test 2: save_hiding / load_hiding roundtrip ───────────────────────
    {
        Commitment pp = Commitment::hiding_random(8);
        char tmpl[] = "/tmp/zke_pp_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) { cerr << "FAIL: mkstemp" << endl; return 1; }
        close(fd);
        string path = tmpl;
        pp.save_hiding(path);

        Commitment loaded = Commitment::load_hiding(path);
        check(loaded.size == pp.size, "load_hiding preserves size");
        check(loaded.is_hiding(), "load_hiding restores hiding_generator");
        check(g1_bytes_eq(loaded.hiding_generator, pp.hiding_generator),
              "load_hiding restores H byte-exactly");
        for (uint i = 0; i < pp.size; i++) {
            if (!g1_bytes_eq(loaded(i), pp(i))) {
                cerr << "FAIL: load_hiding G_i differs at i=" << i << endl;
                exit(1);
            }
        }
        cout << "PASS: load_hiding restores all G_i byte-exactly" << endl;

        // Cleanup
        remove(path.c_str());
        remove((path + ".h").c_str());
    }

    // ── Test 3: commit_hiding refuses non-hiding pp ───────────────────────
    {
        Commitment pp_nh = Commitment::random(4);
        FrTensor t = FrTensor::random(4);
        bool threw = false;
        try {
            auto hc = pp_nh.commit_hiding(t);
            (void)hc;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw,
              "commit_hiding on non-hiding pp throws (no silent loss of hiding)");
    }

    // ── Test 4: collision resistance on fixed message ─────────────────────
    // Commit the same message 16 times with fresh r; no two commitments
    // should be byte-equal.  A bug like "r sampled once at pp time" or
    // "r forced to 0 for perf" would collapse these to a single value.
    {
        Commitment pp = Commitment::hiding_random(4);
        FrTensor t = FrTensor::random(4);  // fixed message for this test
        const uint N = 16;
        vector<G1Jacobian_t> coms;
        coms.reserve(N);
        for (uint i = 0; i < N; i++) {
            auto hc = pp.commit_hiding(t);
            // m = t.size / size = 1 row, so com.size == 1
            check(hc.com.size == 1, "commit_hiding returns single-row com when t.size == size");
            check(hc.r.size == 1, "commit_hiding returns single-row r when t.size == size");
            coms.push_back(hc.com(0));
        }
        for (uint i = 0; i < N; i++) {
            for (uint j = i + 1; j < N; j++) {
                if (g1_bytes_eq(coms[i], coms[j])) {
                    cerr << "FAIL: commitments " << i << " and " << j
                         << " of the same message collide byte-exactly — "
                            "r sampling likely broken" << endl;
                    exit(1);
                }
            }
        }
        cout << "PASS: 16 hiding commits of the same message are all distinct" << endl;
    }

    // ── Test 5: r freshness across calls ──────────────────────────────────
    // Standalone guard on FrTensor::random for the blinding scalar path:
    // even if a future refactor routed commit_hiding's r through a
    // deterministic source, this test would fail.
    {
        Commitment pp = Commitment::hiding_random(4);
        FrTensor t = FrTensor::random(4);
        auto hc1 = pp.commit_hiding(t);
        auto hc2 = pp.commit_hiding(t);
        auto hc3 = pp.commit_hiding(t);
        check(!fr_bytes_eq(hc1.r(0), hc2.r(0)), "r across calls: 1 vs 2 differ");
        check(!fr_bytes_eq(hc1.r(0), hc3.r(0)), "r across calls: 1 vs 3 differ");
        check(!fr_bytes_eq(hc2.r(0), hc3.r(0)), "r across calls: 2 vs 3 differ");
    }

    // ── Test 6: algebraic identity C = Σ tᵢGᵢ + r·H on zero message ─────
    // For t = 0 vector, Σ tᵢ Gᵢ = 0, so commit_hiding(t).com[row] must
    // equal the size-1 commit of r[row] against H.  Both sides go through
    // the same commit pipeline (commit → rowwise_sum), so byte-equality
    // is a meaningful structural check.
    {
        Commitment pp = Commitment::hiding_random(4);

        // t = 0 vector of size 4 (one row, since t.size / pp.size == 1).
        Fr_t zeros[4] = {
            {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}
        };
        FrTensor t(4, zeros);
        auto hc = pp.commit_hiding(t);
        check(hc.com.size == 1, "zero-message commit has single row");

        // Independently compute r[0] · H via the size-1 commit path.
        Commitment h_as_commitment(1, pp.hiding_generator);
        G1TensorJacobian rH = h_as_commitment.commit(hc.r);
        check(rH.size == 1, "r·H computation has single row");

        check(g1_bytes_eq(hc.com(0), rH(0)),
              "zero-message commit equals r · H byte-exactly "
              "(C = Σ tᵢGᵢ + r·H with Σ tᵢGᵢ = 0)");
    }

    // ── Test 7: multi-row hiding commit ───────────────────────────────────
    // Exercise t.size > pp.size: each row gets its own r.  Checks the
    // per-row blinding structure end-to-end.
    {
        Commitment pp = Commitment::hiding_random(4);
        FrTensor t = FrTensor::random(4 * 3);  // 3 rows, 4 cols
        auto hc = pp.commit_hiding(t);
        check(hc.com.size == 3, "3-row message gives 3-row commitment");
        check(hc.r.size == 3, "3-row message gives 3-row blinding tensor");
        // All three r values should differ (statistical cert).
        check(!fr_bytes_eq(hc.r(0), hc.r(1)), "multi-row r: 0 vs 1 differ");
        check(!fr_bytes_eq(hc.r(0), hc.r(2)), "multi-row r: 0 vs 2 differ");
        check(!fr_bytes_eq(hc.r(1), hc.r(2)), "multi-row r: 1 vs 2 differ");
        // All three commitments should differ.
        check(!g1_bytes_eq(hc.com(0), hc.com(1)), "multi-row com: 0 vs 1 differ");
        check(!g1_bytes_eq(hc.com(0), hc.com(2)), "multi-row com: 0 vs 2 differ");
        check(!g1_bytes_eq(hc.com(1), hc.com(2)), "multi-row com: 1 vs 2 differ");
    }

    cout << "All hiding-Pedersen tests PASSED." << endl;
    return 0;
}
