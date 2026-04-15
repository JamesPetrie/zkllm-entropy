// Tests for verifyWeightClaimZK — the ZK opening variant of
// verifyWeightClaim that consumes the Hyrax §A.2 Figure 6 OpeningProof.
//
// Positive:
//   1. Honest Weight + correct claim → accepts.
//
// Negative:
//   2. Wrong claim (off-by-one) → throws.
//   3. Non-openable pp (legacy Commitment::random) → throws.
//   4. Weight with empty r (legacy create_weight path) → throws.

#include "commit/commitment.cuh"
#include "proof/proof.cuh"
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

static vector<Fr_t> make_u(uint log_n) {
    FrTensor u_rand = FrTensor::random(log_n);
    vector<Fr_t> u;
    u.reserve(log_n);
    for (uint i = 0; i < log_n; i++) u.push_back(u_rand(i));
    return u;
}

int main() {
    const uint in_dim = 4;   // rows (outer)
    const uint out_dim = 4;  // cols (inner == pp.size)

    // ── Test 1: honest accept ────────────────────────────────────────────
    {
        Commitment pp = Commitment::hiding_random(out_dim);
        FrTensor weight = FrTensor::random(in_dim * out_dim);
        auto hc = pp.commit_hiding(weight);

        Weight w{pp, weight, hc.com, std::move(hc.r), in_dim, out_dim};

        // Claim: u[0] folds inner (cols), u[1] folds outer (rows).
        // verifyWeightClaim builds u_cat = concat(u[1], u[0]) — outer
        // bits first, then inner bits — and evaluates w_padded at u_cat.
        Claim c;
        c.u.push_back(make_u(2));  // u[0] inner
        c.u.push_back(make_u(2));  // u[1] outer
        vector<Fr_t> u_cat;
        for (auto x : c.u[1]) u_cat.push_back(x);
        for (auto x : c.u[0]) u_cat.push_back(x);
        auto w_padded = w.weight.pad({w.in_dim, w.out_dim});
        c.claim = w_padded(u_cat);

        verifyWeightClaimZK(w, c);  // throws on failure
        cout << "PASS: honest Weight + correct claim accepts" << endl;
    }

    // ── Test 2: wrong claim rejected ─────────────────────────────────────
    {
        Commitment pp = Commitment::hiding_random(out_dim);
        FrTensor weight = FrTensor::random(in_dim * out_dim);
        auto hc = pp.commit_hiding(weight);
        Weight w{pp, weight, hc.com, std::move(hc.r), in_dim, out_dim};

        Claim c;
        c.u.push_back(make_u(2));
        c.u.push_back(make_u(2));
        vector<Fr_t> u_cat;
        for (auto x : c.u[1]) u_cat.push_back(x);
        for (auto x : c.u[0]) u_cat.push_back(x);
        auto w_padded = w.weight.pad({w.in_dim, w.out_dim});
        c.claim = w_padded(u_cat);
        c.claim.val[0] ^= 1u;  // flip a bit

        bool threw = false;
        try { verifyWeightClaimZK(w, c); } catch (const std::runtime_error&) { threw = true; }
        check(threw, "wrong claim rejected");
    }

    // ── Test 3: non-openable pp (legacy random) → throws ─────────────────
    {
        Commitment pp_nh = Commitment::random(out_dim);
        FrTensor weight = FrTensor::random(in_dim * out_dim);
        // Build a fake hc to construct Weight — the pp can't commit_hiding,
        // but we only need a Weight with is_openable()==false to trigger
        // the precondition check.
        G1TensorJacobian fake_com = pp_nh.commit(weight);
        FrTensor fake_r = FrTensor::random(in_dim);
        Weight w{pp_nh, weight, fake_com, std::move(fake_r), in_dim, out_dim};

        Claim c;
        c.u.push_back(make_u(2));
        c.u.push_back(make_u(2));
        c.claim = FrTensor::random(1)(0);

        bool threw = false;
        try { verifyWeightClaimZK(w, c); } catch (const std::runtime_error&) { threw = true; }
        check(threw, "non-openable pp rejected");
    }

    // ── Test 4: empty-r Weight (legacy create_weight path) → throws ──────
    {
        Commitment pp = Commitment::hiding_random(out_dim);
        FrTensor weight = FrTensor::random(in_dim * out_dim);
        auto hc = pp.commit_hiding(weight);
        Weight w{pp, weight, hc.com, FrTensor(0), in_dim, out_dim};  // empty r

        Claim c;
        c.u.push_back(make_u(2));
        c.u.push_back(make_u(2));
        c.claim = FrTensor::random(1)(0);

        bool threw = false;
        try { verifyWeightClaimZK(w, c); } catch (const std::runtime_error&) { threw = true; }
        check(threw, "empty-r Weight rejected");
    }

    cout << "All verifyWeightClaimZK tests PASSED." << endl;
    return 0;
}
