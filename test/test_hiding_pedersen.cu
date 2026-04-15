// Tests for Phase 1 hiding Pedersen commitments.
//
// Covers:
//   (1) is_hiding() sanity: random()→false, hiding_random()→true
//   (2) save_hiding / load_hiding pp roundtrip (Phase 1.5 v2 format)
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
    // ── Test 1: is_hiding() / is_openable() sanity ────────────────────────
    {
        Commitment pp_nh = Commitment::random(4);
        check(!pp_nh.is_hiding(),
              "Commitment::random produces non-hiding pp (H == identity)");
        check(!pp_nh.is_openable(),
              "Commitment::random produces non-openable pp (U == identity)");

        Commitment pp_h = Commitment::hiding_random(4);
        check(pp_h.is_hiding(),
              "Commitment::hiding_random produces hiding pp (H != identity)");
        check(pp_h.is_openable(),
              "Commitment::hiding_random produces openable pp (U != identity)");
        // H and U must be independent points; equality would collapse the
        // Figure 6 Com(·; ·) into a degenerate form.
        check(!g1_bytes_eq(pp_h.hiding_generator, pp_h.u_generator),
              "hiding_random samples H and U as distinct points");
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
        check(loaded.is_openable(), "load_hiding restores u_generator");
        check(g1_bytes_eq(loaded.hiding_generator, pp.hiding_generator),
              "load_hiding restores H byte-exactly");
        check(g1_bytes_eq(loaded.u_generator, pp.u_generator),
              "load_hiding restores U byte-exactly");
        for (uint i = 0; i < pp.size; i++) {
            if (!g1_bytes_eq(loaded(i), pp(i))) {
                cerr << "FAIL: load_hiding G_i differs at i=" << i << endl;
                exit(1);
            }
        }
        cout << "PASS: load_hiding restores all G_i byte-exactly" << endl;

        // Cleanup (Phase 1.5 v2 format: single file, no sidecars)
        remove(path.c_str());
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

    // ── Test 6: algebraic identity C = Σ tᵢGᵢ + r·H ──────────────────────
    // (a) t = 0: Σ tᵢ Gᵢ = 0, so commit_hiding(t).com[row] must equal the
    //     size-1 commit of r[row] against H.
    // (b) t ≠ 0 (random): commit_hiding(t).com[row] must equal
    //     pp.commit(t)[row] + r·H[row], i.e. the plain Pedersen base plus
    //     the fresh blinding term.  This is the structural check that the
    //     hiding path really is Σ tᵢGᵢ + r·H — not, say, Σ tᵢGᵢ alone
    //     with r separately appended to the struct but ignored.
    {
        Commitment pp = Commitment::hiding_random(4);
        Commitment h_as_commitment(1, pp.hiding_generator);

        // (a) t = 0 case
        Fr_t zeros[4] = {
            {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0},
            {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}
        };
        FrTensor t_zero(4, zeros);
        auto hc_zero = pp.commit_hiding(t_zero);
        check(hc_zero.com.size == 1, "zero-message commit has single row");

        G1TensorJacobian rH_zero = h_as_commitment.commit(hc_zero.r);
        check(rH_zero.size == 1, "r·H computation has single row");
        check(g1_bytes_eq(hc_zero.com(0), rH_zero(0)),
              "zero-message commit equals r · H byte-exactly "
              "(C = Σ tᵢGᵢ + r·H with Σ tᵢGᵢ = 0)");

        // (b) random t ≠ 0 case: C = base + r·H where base = Σ tᵢGᵢ.
        // Both sides are computed through the same rowwise_sum pipeline,
        // so byte-equality is meaningful.
        FrTensor t_rand = FrTensor::random(4);
        auto hc_rand = pp.commit_hiding(t_rand);
        check(hc_rand.com.size == 1, "random-message commit has single row");

        G1TensorJacobian base = pp.commit(t_rand);  // Σ tᵢ Gᵢ
        G1TensorJacobian rH_rand = h_as_commitment.commit(hc_rand.r);
        G1TensorJacobian expected = base + rH_rand;
        check(g1_bytes_eq(hc_rand.com(0), expected(0)),
              "random-message commit equals base + r · H byte-exactly "
              "(C = Σ tᵢGᵢ + r·H with t random)");
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

    // ── Test 8: create_weight hiding overload — .r sidecar roundtrip ──────
    // Step 3 acceptance gate: save a hiding Weight's pp + com + r, then
    // load it back via the hiding create_weight and confirm r round-trips
    // byte-exactly and that the non-hiding overload stays non-throwing.
    {
        // Build a small hiding pp and a small int-valued weight.
        const uint in_dim = 1;
        const uint out_dim = 4;
        Commitment pp = Commitment::hiding_random(out_dim);
        int int_weight[4] = {17, -3, 42, 100};
        FrTensor weight(out_dim, int_weight);  // int ctor

        auto hc = pp.commit_int_hiding(weight);
        check(hc.com.size == in_dim, "create_weight smoke: com has in_dim rows");
        check(hc.r.size == in_dim, "create_weight smoke: r has in_dim rows");

        // Write pp + com + r + int-weight sidecars to /tmp.
        char ppt[] = "/tmp/zke_cw_pp_XXXXXX";
        char comt[] = "/tmp/zke_cw_com_XXXXXX";
        char intt[] = "/tmp/zke_cw_int_XXXXXX";
        int f1 = mkstemp(ppt);  close(f1);
        int f2 = mkstemp(comt); close(f2);
        int f3 = mkstemp(intt); close(f3);
        string pp_path = ppt;
        string com_path = comt;
        string int_path = intt;
        string r_path = com_path + ".r";

        pp.save_hiding(pp_path);
        hc.com.save(com_path);
        hc.r.save(r_path);
        weight.save_int(int_path);

        // Load via hiding create_weight.
        Weight w = create_weight(pp_path, int_path, com_path, r_path, in_dim, out_dim);
        check(w.generator.is_hiding(),
              "create_weight(hiding) restores is_hiding() == true");
        check(w.r.size == in_dim,
              "create_weight(hiding) restores r with correct row count");
        check(w.com.size == in_dim,
              "create_weight(hiding) restores com with correct row count");
        check(fr_bytes_eq(w.r(0), hc.r(0)),
              "create_weight(hiding) restores r[0] byte-exactly");
        check(g1_bytes_eq(w.com(0), hc.com(0)),
              "create_weight(hiding) restores com[0] byte-exactly");
        check(g1_bytes_eq(w.generator.hiding_generator, pp.hiding_generator),
              "create_weight(hiding) restores H byte-exactly");

        // Legacy overload: non-hiding Weight has r.size == 0.
        Weight w_legacy = create_weight(pp_path, int_path, com_path,
                                        in_dim, out_dim);
        check(w_legacy.r.size == 0,
              "create_weight(legacy) produces empty r (size 0)");

        // Cleanup (Phase 1.5 v2 format: single pp file, no sidecars)
        remove(pp_path.c_str());
        remove(com_path.c_str());
        remove(r_path.c_str());
        remove(int_path.c_str());
    }

    // ── Test 9: create_weight(hiding) throws on corrupted pp file ────────
    // Phase 1.5 equivalent of the old "missing .h sidecar" test: a
    // corrupted pp file (flip a byte inside a G_i slot) must fail loud
    // at load time via verify_pp, not silently degrade.  test_pp_format
    // has the fine-grained tamper coverage; this check ensures the
    // rejection propagates through the create_weight entry point too.
    {
        const uint in_dim = 1;
        const uint out_dim = 4;
        Commitment pp = Commitment::hiding_random(out_dim);
        int int_weight[4] = {1, 2, 3, 4};
        FrTensor weight(out_dim, int_weight);
        auto hc = pp.commit_int_hiding(weight);

        char ppt[] = "/tmp/zke_neg_h_pp_XXXXXX";
        char comt[] = "/tmp/zke_neg_h_com_XXXXXX";
        char intt[] = "/tmp/zke_neg_h_int_XXXXXX";
        int f1 = mkstemp(ppt);  close(f1);
        int f2 = mkstemp(comt); close(f2);
        int f3 = mkstemp(intt); close(f3);
        string pp_path = ppt, com_path = comt, int_path = intt;
        string r_path = com_path + ".r";

        pp.save_hiding(pp_path);
        hc.com.save(com_path);
        hc.r.save(r_path);
        weight.save_int(int_path);

        // Flip a byte somewhere in the middle of the pp file.  Exact
        // offset doesn't matter — any corruption inside the G_i / H / U
        // region will trip verify_pp.  We pick an offset past the DST
        // header (~80 bytes) to guarantee we hit a generator byte.
        {
            FILE* f = fopen(pp_path.c_str(), "r+b");
            fseek(f, 200, SEEK_SET);
            uint8_t b;
            fread(&b, 1, 1, f);
            fseek(f, 200, SEEK_SET);
            b ^= 0x01;
            fwrite(&b, 1, 1, f);
            fclose(f);
        }

        bool threw = false;
        try {
            Weight w = create_weight(pp_path, int_path, com_path, r_path,
                                     in_dim, out_dim);
            (void)w;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw,
              "create_weight(hiding) throws when pp file is corrupted");

        remove(pp_path.c_str());
        remove(com_path.c_str());
        remove(r_path.c_str());
        remove(int_path.c_str());
    }

    // ── Test 10: create_weight(hiding) throws on r.size != com.size ─────
    // The hiding overload cross-checks the per-row blinding tensor size
    // against the commitment row count; a mismatch (corrupted sidecar,
    // accidental re-use of a sibling weight's .r) must fail loud.
    {
        const uint in_dim = 1;
        const uint out_dim = 4;
        Commitment pp = Commitment::hiding_random(out_dim);
        int int_weight[4] = {7, -1, 11, 0};
        FrTensor weight(out_dim, int_weight);
        auto hc = pp.commit_int_hiding(weight);  // hc.com.size == 1

        char ppt[] = "/tmp/zke_neg_rs_pp_XXXXXX";
        char comt[] = "/tmp/zke_neg_rs_com_XXXXXX";
        char intt[] = "/tmp/zke_neg_rs_int_XXXXXX";
        int f1 = mkstemp(ppt);  close(f1);
        int f2 = mkstemp(comt); close(f2);
        int f3 = mkstemp(intt); close(f3);
        string pp_path = ppt, com_path = comt, int_path = intt;
        string r_path = com_path + ".r";

        pp.save_hiding(pp_path);
        hc.com.save(com_path);
        weight.save_int(int_path);

        // Write a mismatched r: 2 rows where the com has 1.
        FrTensor bad_r = FrTensor::random(2);
        bad_r.save(r_path);

        bool threw = false;
        try {
            Weight w = create_weight(pp_path, int_path, com_path, r_path,
                                     in_dim, out_dim);
            (void)w;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw,
              "create_weight(hiding) throws when r.size != com.size");

        remove(pp_path.c_str());
        remove(com_path.c_str());
        remove(r_path.c_str());
        remove(int_path.c_str());
    }

    cout << "All hiding-Pedersen tests PASSED." << endl;
    return 0;
}
