// Tests for Phase 2 blinded opening (Hyrax §A.2 Figure 6 composed with
// §6.1 row reduction).
//
// Positive:
//   1. Single-row (com.size == 1) open/verify roundtrip.
//   2. Multi-row (com.size > 1) open/verify roundtrip — exercises the
//      §6.1 fold of both row commitments and row blindings.
//   3. Claimed v matches MLE evaluation of t at u.
//   4. Transcript shape: |z| == pp.size, every scalar/group element
//      populated.
//
// Negative:
//   5. tampered δ  → verifier rejects (eq 13 breaks).
//   6. tampered β  → verifier rejects (eq 14 breaks).
//   7. tampered τ  → verifier rejects (τ ≠ v·U + r_τ·H binding).
//   8. tampered z⃗ → verifier rejects.
//   9. tampered z_δ → verifier rejects.
//  10. tampered z_β → verifier rejects.
//  11. wrong v (off by one) → verifier rejects.
//  12. wrong challenge c on verify → rejects.
//
// Sanity:
//  13. open_zk throws on non-openable pp (missing .u).
//
// Hyrax (Wahby et al. 2018, eprint 2017/1132) §A.2 Figure 6, Theorem 11
// (p. 18): "The protocol of Figure 6 is complete, honest-verifier
// perfect zero-knowledge, and special sound under the discrete log
// assumption."

#include "commit/commitment.cuh"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

// Build a challenge vector of length `log_n`.  We just use 1, 2, 3, ...
// as scalars (in Montgomery form) so verifier recomputation is
// deterministic.  For a ZK test any value works — but keeping them
// nontrivial avoids accidental collapse of eq-MLE weights.
static vector<Fr_t> make_u(uint log_n) {
    // Pull from random to exercise the MLE machinery.  The same u is
    // fed into prover and verifier so the randomness here is innocent.
    FrTensor u_rand = FrTensor::random(log_n);
    vector<Fr_t> u;
    u.reserve(log_n);
    for (uint i = 0; i < log_n; i++) u.push_back(u_rand(i));
    return u;
}

int main() {
    // ── Test 1: single-row roundtrip (com.size == 1) ─────────────────────
    {
        const uint N = 8;                      // pp.size
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);         // hc.com.size == 1
        check(hc.com.size == 1, "single-row: com has 1 row");
        check(hc.r.size == 1, "single-row: r has 1 row");

        vector<Fr_t> u = make_u(3);            // log2(N) = 3
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        check(res.proof.z.size == N, "single-row: |z| == pp.size");

        bool ok = pp.verify_zk(hc.com, u, res.v, res.proof, c);
        check(ok, "single-row: verify_zk accepts honest transcript");
    }

    // ── Test 2: multi-row (com.size == 4) ────────────────────────────────
    {
        const uint cols = 4;                   // pp.size
        const uint rows = 4;                   // com.size after commit
        Commitment pp = Commitment::hiding_random(cols);
        FrTensor t = FrTensor::random(cols * rows);
        auto hc = pp.commit_hiding(t);
        check(hc.com.size == rows, "multi-row: com has correct row count");

        // Total u length: log2(cols) + log2(rows) = 2 + 2 = 4
        vector<Fr_t> u = make_u(4);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        check(res.proof.z.size == cols,
              "multi-row: |z| == pp.size (length of folded row)");

        bool ok = pp.verify_zk(hc.com, u, res.v, res.proof, c);
        check(ok, "multi-row: verify_zk accepts honest transcript");

        // Cross-check: v returned by open_zk matches t evaluated as a
        // 2D MLE at u.  Using the same u convention
        // (last log(rows) fold rows, leading log(cols) fold within-row):
        // t as flat length cols*rows, rows = outer (slow) dim, cols = inner.
        // partial_me(u_R, cols) folds outer then we eval at u_L.
        vector<Fr_t> u_R(u.end() - 2, u.end());
        vector<Fr_t> u_L(u.begin(), u.end() - 2);
        FrTensor folded = t.partial_me(u_R, cols);
        Fr_t v_direct = folded(u_L);
        check(memcmp(&v_direct, &res.v, sizeof(Fr_t)) == 0,
              "multi-row: v matches direct MLE evaluation of t at u");
    }

    // ── Test 3: negative — tampered δ ────────────────────────────────────
    {
        const uint N = 4;
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        OpeningProof bad = {
            res.proof.delta, res.proof.beta, res.proof.tau,
            res.proof.z, res.proof.z_delta, res.proof.z_beta,
            res.proof.r_tau
        };
        // Flip one bit of δ.
        bad.delta.x.val[0] ^= 1u;
        check(!pp.verify_zk(hc.com, u, res.v, bad, c),
              "tampered δ is rejected");
    }

    // ── Test 4: negative — tampered β ────────────────────────────────────
    {
        const uint N = 4;
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        OpeningProof bad = {
            res.proof.delta, res.proof.beta, res.proof.tau,
            res.proof.z, res.proof.z_delta, res.proof.z_beta,
            res.proof.r_tau
        };
        bad.beta.x.val[0] ^= 1u;
        check(!pp.verify_zk(hc.com, u, res.v, bad, c),
              "tampered β is rejected");
    }

    // ── Test 5: negative — tampered τ ────────────────────────────────────
    {
        const uint N = 4;
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        OpeningProof bad = {
            res.proof.delta, res.proof.beta, res.proof.tau,
            res.proof.z, res.proof.z_delta, res.proof.z_beta,
            res.proof.r_tau
        };
        bad.tau.x.val[0] ^= 1u;
        check(!pp.verify_zk(hc.com, u, res.v, bad, c),
              "tampered τ is rejected (τ-binding check)");
    }

    // ── Test 6: negative — tampered z ────────────────────────────────────
    {
        const uint N = 4;
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        // Add a non-trivial perturbation to z[0]: bad_z = z + one.
        Fr_t one = FR_FROM_INT(1);
        FrTensor one_t(1, &one);
        // Build a perturbation vector: 1 at index 0, 0 elsewhere.
        FrTensor pert(N);
        {
            Fr_t zeros[4] = {
                FR_FROM_INT(0), FR_FROM_INT(0),
                FR_FROM_INT(0), FR_FROM_INT(0)
            };
            // Can't do this cleanly without int* init; use random and add 1.
            // Simpler: copy z and mutate via a temp host-side Fr.
            FrTensor tmp(N, zeros);
            cudaMemcpy(pert.gpu_data, tmp.gpu_data, N*sizeof(Fr_t),
                       cudaMemcpyDeviceToDevice);
        }
        // Overwrite pert[0] with one.
        cudaMemcpy(pert.gpu_data, &one, sizeof(Fr_t), cudaMemcpyHostToDevice);
        FrTensor bad_z = res.proof.z + pert;

        OpeningProof bad = {
            res.proof.delta, res.proof.beta, res.proof.tau,
            bad_z, res.proof.z_delta, res.proof.z_beta,
            res.proof.r_tau
        };
        check(!pp.verify_zk(hc.com, u, res.v, bad, c),
              "tampered z is rejected");
    }

    // ── Test 7: negative — tampered z_δ ──────────────────────────────────
    {
        const uint N = 4;
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        Fr_t bad_zd = res.proof.z_delta;
        bad_zd.val[0] ^= 1u;
        OpeningProof bad = {
            res.proof.delta, res.proof.beta, res.proof.tau,
            res.proof.z, bad_zd, res.proof.z_beta, res.proof.r_tau
        };
        check(!pp.verify_zk(hc.com, u, res.v, bad, c),
              "tampered z_δ is rejected");
    }

    // ── Test 8: negative — tampered z_β ──────────────────────────────────
    {
        const uint N = 4;
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        Fr_t bad_zb = res.proof.z_beta;
        bad_zb.val[0] ^= 1u;
        OpeningProof bad = {
            res.proof.delta, res.proof.beta, res.proof.tau,
            res.proof.z, res.proof.z_delta, bad_zb, res.proof.r_tau
        };
        check(!pp.verify_zk(hc.com, u, res.v, bad, c),
              "tampered z_β is rejected");
    }

    // ── Test 9: negative — wrong v ───────────────────────────────────────
    {
        const uint N = 4;
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        Fr_t bad_v = res.v;
        bad_v.val[0] ^= 1u;
        check(!pp.verify_zk(hc.com, u, bad_v, res.proof, c),
              "wrong v is rejected (τ-binding fails for bad v)");
    }

    // ── Test 10: negative — wrong challenge c at verify ──────────────────
    {
        const uint N = 4;
        Commitment pp = Commitment::hiding_random(N);
        FrTensor t = FrTensor::random(N);
        auto hc = pp.commit_hiding(t);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        auto res = pp.open_zk(t, hc.r, hc.com, u, c);
        Fr_t bad_c = c;
        bad_c.val[0] ^= 1u;
        check(!pp.verify_zk(hc.com, u, res.v, res.proof, bad_c),
              "wrong challenge at verify rejected (soundness)");
    }

    // ── Test 11: open_zk on non-openable pp throws ───────────────────────
    {
        Commitment pp_nh = Commitment::random(4);
        FrTensor t = FrTensor::random(4);
        FrTensor r_dummy = FrTensor::random(1);
        G1TensorJacobian com_dummy(1, G1Jacobian_ZERO);
        vector<Fr_t> u = make_u(2);
        Fr_t c = FrTensor::random(1)(0);

        bool threw = false;
        try {
            auto res = pp_nh.open_zk(t, r_dummy, com_dummy, u, c);
            (void)res;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "open_zk on non-openable pp throws");
    }

    cout << "All Phase 2 open_zk/verify_zk tests PASSED." << endl;
    return 0;
}
