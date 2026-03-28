// zkllm_entropy: prove conditional entropy of an LLM output sequence.
//
// Proof structure (from committed weights to entropy bound):
//
//   committed W_norm  committed W_lm           public sigma_eff
//        |                  |                        |
//   zkRMSNorm(hidden, W_norm) → normed_hidden        |
//        |                                           |
//   zkFC(normed_hidden, W_lm) → logits              |
//        |                                           |
//   zkConditionalEntropy(logits, tokens, sigma_eff) → entropy bound H
//
// The hidden state (layer-31-output.bin) is taken from the existing zkLLM
// layer proofs.  The entropy proof links directly to the committed model
// weights without requiring any separate logit commitment.
//
// Usage:
//   ./zkllm_entropy <workdir> <tokens_file> <proof_output> <sigma_eff>
//                  [seq_len=1024] [hidden_size=4096] [vocab_size=32000]
//                  [cdf_precision=20] [log_precision=15]
//                  [cdf_scale=65536] [log_scale=65536]
//
// Required files in <workdir>:
//   layer-31-output.bin             final hidden state (from run_proofs.py)
//   final_norm-rms_inv.bin          per-position 1/rms  (from gen_entropy_inputs.py)
//   input_layernorm.weight-pp.bin   generators for norm weight (from llama-commit.py)
//   final_norm.weight-int.bin       quantised norm weight      (from commit_final_layers.py)
//   final_norm.weight-commitment.bin
//   lm_head-pp.bin                  generators for lm_head     (from ppgen)
//   lm_head-weight-int.bin          quantised lm_head weight   (from commit_final_layers.py)
//   lm_head-weight-commitment.bin

#include "entropy/zkentropy.cuh"
#include "zknn/zkfc.cuh"
#include "zknn/rescaling.cuh"
#include "proof/proof.cuh"
#ifndef USE_GOLDILOCKS
#include "commit/commitment.cuh"
#endif
#include "util/ioutils.cuh"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>

using namespace std;

static vector<uint> load_token_sequence(const string& filename) {
    ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open tokens file: " + filename);
    vector<uint> tokens;
    uint t;
    while (f >> t) tokens.push_back(t);
    return tokens;
}

// (tensor_row helper moved to zkentropy.cu — no longer needed here)

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " <workdir> <tokens_file> <proof_output> <sigma_eff>\n"
             << "       [seq_len=1024] [hidden_size=4096] [vocab_size=32000]\n"
             << "       [cdf_precision=20] [log_precision=15]\n"
             << "       [cdf_scale=65536] [log_scale=65536]\n";
        return 1;
    }

    string workdir     = argv[1];
    string tokens_file = argv[2];
    string proof_output= argv[3];
    double sigma_eff   = atof(argv[4]);

    uint seq_len      = argc > 5  ? (uint)atoi(argv[5])  : 1024u;
    uint hidden_size  = argc > 6  ? (uint)atoi(argv[6])  : 4096u;
    uint vocab_size   = argc > 7  ? (uint)atoi(argv[7])  : 32000u;
    uint cdf_precision= argc > 8  ? (uint)atoi(argv[8])  : 20u; // 2^20 covers diffs up to ~1M (logit gap ~16 in float units)
    uint log_precision= argc > 9  ? (uint)atoi(argv[9])  : 15u;
    uint cdf_scale    = argc > 10 ? (uint)atoi(argv[10]) : 65536u;
    uint log_scale    = argc > 11 ? (uint)atoi(argv[11]) : 65536u;

    auto path = [&](const string& f) { return workdir + "/" + f; };

    cout << "Loading inputs from " << workdir << " ..." << endl;
    cout << "  seq_len=" << seq_len << "  hidden_size=" << hidden_size
         << "  vocab_size=" << vocab_size << "  sigma_eff=" << sigma_eff << endl;

    // ── Load hidden state and rms_inv ─────────────────────────────────────────
    FrTensor hidden  = FrTensor::from_int_bin(path("layer-31-output.bin"));
    FrTensor rms_inv = FrTensor::from_int_bin(path("final_norm-rms_inv.bin"));

    if (hidden.size  != seq_len * hidden_size)
        throw std::runtime_error("hidden state size mismatch");
    if (rms_inv.size != seq_len)
        throw std::runtime_error("rms_inv size mismatch");

    // ── Load committed weights ────────────────────────────────────────────────
    // final_norm: in_dim=1, out_dim=hidden_size (same shape as per-layer norms)
#ifdef USE_GOLDILOCKS
    Weight final_norm_w = create_weight(
        path("final_norm.weight-int.bin"),
        path("final_norm.weight-gold-commitment.bin"),
        1, hidden_size);
#else
    Weight final_norm_w = create_weight(
        path("input_layernorm.weight-pp.bin"),
        path("final_norm.weight-int.bin"),
        path("final_norm.weight-commitment.bin"),
        1, hidden_size);
#endif

    // lm_head: in_dim=hidden_size, out_dim=vocab_size
#ifdef USE_GOLDILOCKS
    Weight lm_head_w = create_weight(
        path("lm_head-weight-int.bin"),
        path("lm_head-weight-gold-commitment.bin"),
        hidden_size, vocab_size);
#else
    Weight lm_head_w = create_weight(
        path("lm_head-pp.bin"),
        path("lm_head-weight-int.bin"),
        path("lm_head-weight-commitment.bin"),
        hidden_size, vocab_size);
#endif

    // ── Step 1: Final RMSNorm ─────────────────────────────────────────────────
    // Following the pattern in rmsnorm.cu:
    //   g(rms_inv) = diag(W_norm) × rms_inv  (zkFC with in_dim=1, out_dim=hidden_size)
    //   normed[t][d] = round(g_inv_rms[t][d] * hidden[t][d] / scale)
    cout << "Computing final RMSNorm..." << endl;
    Rescaling rs_norm1(1u << 16);  // rescale after weight × rms_inv
    Rescaling rs_norm2(1u << 16);  // rescale after Hadamard with hidden

    zkFC norm_fc(1, hidden_size, final_norm_w.weight);
    FrTensor g_inv_rms  = norm_fc(rms_inv);   // seq_len * hidden_size
    FrTensor g_inv_rms_ = rs_norm1(g_inv_rms);
    FrTensor normed     = g_inv_rms_ * hidden; // Hadamard
    FrTensor normed_    = rs_norm2(normed);

    // ── Step 2: lm_head ───────────────────────────────────────────────────────
    cout << "Computing lm_head..." << endl;
    Rescaling rs_lm(1u << 16);

    zkFC lm_fc(hidden_size, vocab_size, lm_head_w.weight);
    FrTensor logits_batch  = lm_fc(normed_);   // seq_len * vocab_size
    FrTensor logits_batch_ = rs_lm(logits_batch);

    // ── Step 3: Load tokens ─────────────────────────────────────────────────
    vector<uint> tokens = load_token_sequence(tokens_file);
    if (tokens.size() != seq_len)
        throw std::runtime_error("token count != seq_len");

    cout << "Building entropy lookup tables..." << endl;
    zkConditionalEntropy entropy_prover(
        vocab_size, cdf_precision, log_precision,
        cdf_scale, log_scale, sigma_eff);

    // ── Step 4: Compute entropy (batched, flat T×V tensor) ───────────────────
    cout << "Computing conditional entropy (batched)..." << endl;
    Fr_t total_entropy = entropy_prover.compute(logits_batch_, seq_len, vocab_size, tokens);

#ifdef USE_GOLDILOCKS
    unsigned long entropy_val = total_entropy.val;
#else
    unsigned long entropy_val =
        ((unsigned long)total_entropy.val[1] << 32) | total_entropy.val[0];
#endif
    double entropy_bits = (double)entropy_val / (double)log_scale;
    cout << "Conditional entropy bound : " << entropy_bits << " bits total" << endl;
    cout << "Average per token         : " << entropy_bits / seq_len << " bits/token" << endl;

    // ── Step 5: Prove entropy (batched) ──────────────────────────────────────
    cout << "Generating entropy proof..." << endl;
    vector<Polynomial> proof;
    entropy_prover.prove(logits_batch_, seq_len, vocab_size, tokens, total_entropy, proof);

    // ── Step 6: Prove lm_head — links logits to committed W_lm ───────────────
    cout << "Proving lm_head (zkFC)..." << endl;
    rs_lm.prove(logits_batch, logits_batch_);
    verifyWeightClaim(lm_head_w, lm_fc.prove(normed_, logits_batch)[0]);

    // ── Step 7: Prove final RMSNorm — links normed_hidden to committed W_norm ─
    cout << "Proving final RMSNorm..." << endl;
    rs_norm2.prove(normed, normed_);
    auto u_hp = random_vec(ceilLog2(normed.size));
    hadamard_product_sumcheck(g_inv_rms_, hidden, u_hp, random_vec(ceilLog2(normed.size)));
    rs_norm1.prove(g_inv_rms, g_inv_rms_);
    verifyWeightClaim(final_norm_w, norm_fc.prove(rms_inv, g_inv_rms)[0]);

    // ── Serialise proof ───────────────────────────────────────────────────────
    // Format (v4 — argmax merged into CDF, bit_width removed):
    //   [8 bytes]  magic "ZKENTROP"
    //   [8 bytes]  entropy_val (uint64, in log_scale units)
    //   [4 bytes]  seq_len
    //   [4 bytes]  vocab_size
    //   [8 bytes]  sigma_eff (double)
    //   [4 bytes]  log_scale
    //   [4 bytes]  cdf_precision
    //   [4 bytes]  log_precision
    //   [4 bytes]  cdf_scale
    //   [4 bytes]  n_polys
    //   For each polynomial:
    //     [4 bytes] n_coeffs
    //     [n_coeffs * sizeof(Fr_t) bytes] coefficients
    {
        ofstream f(proof_output, ios::binary);
        if (!f) { cerr << "Cannot write: " << proof_output << endl; return 1; }

        uint64_t magic = 0x5A4B454E54524F50ULL;
        f.write((char*)&magic,         sizeof(magic));
        f.write((char*)&entropy_val,   sizeof(entropy_val));
        f.write((char*)&seq_len,       sizeof(seq_len));
        f.write((char*)&vocab_size,    sizeof(vocab_size));
        f.write((char*)&sigma_eff,     sizeof(sigma_eff));
        f.write((char*)&log_scale,     sizeof(log_scale));
        f.write((char*)&cdf_precision, sizeof(cdf_precision));
        f.write((char*)&log_precision, sizeof(log_precision));
        f.write((char*)&cdf_scale,     sizeof(cdf_scale));

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
    cout << "Proof written to " << proof_output
         << " (" << proof.size() << " polynomials)" << endl;

    return 0;
}
