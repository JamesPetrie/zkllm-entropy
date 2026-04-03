// End-to-end test: prover generates v3 proof file, verifier reads and checks it.
// Build with Goldilocks: nvcc -arch=sm_90 -std=c++17 -O3 -DUSE_GOLDILOCKS ...
// Or just use the existing test infrastructure.

#include "entropy/zkentropy.cuh"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

static FrTensor make_flat_logits(uint T, uint V,
                                  const vector<uint>& winners,
                                  long winner_val, long other_val) {
    Fr_t* cpu = new Fr_t[T * V];
    for (uint t = 0; t < T; t++) {
        for (uint i = 0; i < V; i++) {
            long v = (i == winners[t]) ? winner_val : other_val;
            cpu[t * V + i] = FR_FROM_INT((unsigned long)v);
        }
    }
    FrTensor tensor(T * V, cpu);
    delete[] cpu;
    return tensor;
}

int main() {
    const uint vocab_size    = 32;
    const uint cdf_precision = 16;
    const uint log_precision = 16;
    const uint cdf_scale     = 1u << 16;
    const uint log_scale     = 1u << 16;
    const double sigma_eff   = 500.0;

    zkConditionalEntropy prover(vocab_size, cdf_precision, log_precision,
                                cdf_scale, log_scale, sigma_eff);

    uint T = 4;
    vector<uint> winners = {5, 3, 5, 5};
    auto logits = make_flat_logits(T, vocab_size, winners, 5000L, 100L);
    vector<uint> tokens = {5, 20, 5, 20};

    Fr_t claimed = prover.compute(logits, T, vocab_size, tokens);
    vector<Polynomial> proof;
    vector<Claim> claims;
    prover.prove(logits, T, vocab_size, tokens, claimed, proof, claims);

#ifdef USE_GOLDILOCKS
    unsigned long entropy_val = claimed.val;
#else
    unsigned long entropy_val =
        ((unsigned long)claimed.val[1] << 32) | claimed.val[0];
#endif

    // Write v3 proof file
    string proof_path = "/tmp/test_entropy_v3.proof";
    {
        ofstream f(proof_path, ios::binary);
        uint64_t magic = 0x5A4B454E54523033ULL;  // "ZKENTR03"
        uint32_t version = 3;
        f.write((char*)&magic, sizeof(magic));
        f.write((char*)&version, sizeof(version));
        f.write((char*)&entropy_val, sizeof(uint64_t));
        f.write((char*)&T, sizeof(uint32_t));
        f.write((char*)&vocab_size, sizeof(uint32_t));
        f.write((char*)&sigma_eff, sizeof(double));
        f.write((char*)&log_scale, sizeof(uint32_t));
        f.write((char*)&cdf_precision, sizeof(uint32_t));
        f.write((char*)&log_precision, sizeof(uint32_t));
        f.write((char*)&cdf_scale, sizeof(uint32_t));

        uint32_t n_polys = (uint32_t)proof.size();
        f.write((char*)&n_polys, sizeof(n_polys));
        for (const Polynomial& poly : proof) {
            int deg = poly.getDegree();
            uint32_t n_coeffs = (deg >= 0) ? (uint32_t)(deg + 1) : 0u;
            f.write((char*)&n_coeffs, sizeof(n_coeffs));
            for (uint32_t k = 0; k < n_coeffs; k++) {
                Fr_t xk = FR_FROM_INT(k);
                Fr_t yk = const_cast<Polynomial&>(poly)(xk);
                f.write((char*)&yk, sizeof(Fr_t));
            }
        }
    }

    cout << "Proof written to " << proof_path
         << " (" << proof.size() << " polynomials)" << endl;
    cout << "Entropy: " << (double)entropy_val / log_scale << " bits" << endl;
    cout << "Now run: verifier/verifier " << proof_path << endl;

    return 0;
}
