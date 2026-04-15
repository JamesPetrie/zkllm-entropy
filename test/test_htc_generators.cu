// Phase 1.5 generator-derivation tests.
//
// Gates on three properties of Commitment::hiding_random under the new
// RFC 9380 hash-to-curve derivation:
//
//   1. Deterministic — calling hiding_random(size) twice produces
//      byte-identical generators.  Guards against an accidental
//      RNG-dependency creeping back in.
//
//   2. Distinct — for a non-trivial size, every {G_i} is pairwise
//      distinct and distinct from H and U.  Under hash-to-curve's
//      uniform output distribution, the collision probability is
//      ~2^-255; any collision signals a domain-separation bug.
//
//   3. Non-identity — none of the generators is the G1 identity.
//      (Hash-to-curve with SSWU+cofactor-clearing never returns
//       identity; this is an integrity assertion on the wrapper.)

#include "commit/commitment.cuh"
#include "field/hash_to_curve.cuh"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

static bool is_identity(const G1Jacobian_t& p) {
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (p.z.val[i] != 0) return false;
    }
    return true;
}

// Byte-level G1 ordering so we can stuff points into a std::set.
struct G1Less {
    bool operator()(const G1Jacobian_t& a, const G1Jacobian_t& b) const {
        return memcmp(&a, &b, sizeof(G1Jacobian_t)) < 0;
    }
};

int main() {
    const uint N = 1024;

    // ── Test 1: determinism ──────────────────────────────────────────────
    {
        Commitment pp_a = Commitment::hiding_random(N);
        Commitment pp_b = Commitment::hiding_random(N);

        vector<G1Jacobian_t> ga(N), gb(N);
        cudaMemcpy(ga.data(), pp_a.gpu_data, N * sizeof(G1Jacobian_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(gb.data(), pp_b.gpu_data, N * sizeof(G1Jacobian_t),
                   cudaMemcpyDeviceToHost);

        check(memcmp(ga.data(), gb.data(), N * sizeof(G1Jacobian_t)) == 0,
              "G_i byte-identical across two hiding_random calls");
        check(memcmp(&pp_a.hiding_generator, &pp_b.hiding_generator,
                     sizeof(G1Jacobian_t)) == 0,
              "H byte-identical across two hiding_random calls");
        check(memcmp(&pp_a.u_generator, &pp_b.u_generator,
                     sizeof(G1Jacobian_t)) == 0,
              "U byte-identical across two hiding_random calls");
    }

    // ── Test 2: distinctness ─────────────────────────────────────────────
    {
        Commitment pp = Commitment::hiding_random(N);
        vector<G1Jacobian_t> g(N);
        cudaMemcpy(g.data(), pp.gpu_data, N * sizeof(G1Jacobian_t),
                   cudaMemcpyDeviceToHost);

        set<G1Jacobian_t, G1Less> seen;
        for (uint i = 0; i < N; i++) seen.insert(g[i]);
        check(seen.size() == N, "all G_i distinct");

        // H and U must not be in {G_i} and must differ from each other.
        check(seen.find(pp.hiding_generator) == seen.end(), "H not in {G_i}");
        check(seen.find(pp.u_generator)      == seen.end(), "U not in {G_i}");
        check(memcmp(&pp.hiding_generator, &pp.u_generator,
                     sizeof(G1Jacobian_t)) != 0, "H != U");
    }

    // ── Test 3: non-identity ─────────────────────────────────────────────
    {
        Commitment pp = Commitment::hiding_random(N);
        vector<G1Jacobian_t> g(N);
        cudaMemcpy(g.data(), pp.gpu_data, N * sizeof(G1Jacobian_t),
                   cudaMemcpyDeviceToHost);
        for (uint i = 0; i < N; i++) {
            if (is_identity(g[i])) {
                cerr << "FAIL: G_" << i << " is the identity" << endl;
                return 1;
            }
        }
        cout << "PASS: no G_i is the identity (checked " << N << ")" << endl;
        check(!is_identity(pp.hiding_generator), "H is not the identity");
        check(!is_identity(pp.u_generator),      "U is not the identity");
    }

    // ── Test 4: verify_pp accepts honest pp ──────────────────────────────
    {
        Commitment pp = Commitment::hiding_random(N);
        check(pp.verify_pp(ZKLLM_ENTROPY_PEDERSEN_DST_V1),
              "verify_pp accepts honest hiding_random output");
    }

    // ── Test 5: verify_pp rejects wrong DST ──────────────────────────────
    {
        Commitment pp = Commitment::hiding_random(N);
        bool threw = false;
        try {
            pp.verify_pp("ZKLLM-ENTROPY-PEDERSEN-V99_BLS12381G1_XMD:SHA-256_SSWU_RO_");
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "verify_pp rejects pp re-derived under a different DST");
    }

    cout << "All hash-to-curve generator tests PASSED." << endl;
    return 0;
}
