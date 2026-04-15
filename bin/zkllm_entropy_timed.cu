// zkllm_entropy_timed: instrumented version of zkllm_entropy that reports
// per-phase wall-clock times.  Used to identify bottlenecks and measure
// scaling with seq_len.
//
// Usage: same as zkllm_entropy (see that file for details).

#include "entropy/zkentropy.cuh"
#include "zknn/zkfc.cuh"
#include "zknn/rescaling.cuh"
#include "proof/proof.cuh"
#include "commit/commitment.cuh"
#include "util/ioutils.cuh"
#include "util/timer.hpp"
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

static FrTensor tensor_row(const FrTensor& mat, uint row_idx, uint row_size) {
    FrTensor row(row_size);
    cudaMemcpy(row.gpu_data,
               mat.gpu_data + (size_t)row_idx * row_size,
               row_size * sizeof(Fr_t),
               cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    return row;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " <workdir> <tokens_file> <proof_output> <sigma_eff>\n"
             << "       [seq_len=1024] [hidden_size=4096] [vocab_size=32000]\n"
             << "       [cdf_precision=12] [log_precision=15]\n"
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
    uint cdf_precision= argc > 8  ? (uint)atoi(argv[8])  : 15u;
    uint log_precision= argc > 9  ? (uint)atoi(argv[9]) : 15u;
    uint cdf_scale    = argc > 10 ? (uint)atoi(argv[10]) : 65536u;
    uint log_scale    = argc > 11 ? (uint)atoi(argv[11]) : 65536u;

    auto path = [&](const string& f) { return workdir + "/" + f; };

    Timer t_load, t_rmsnorm_compute, t_lmhead_compute, t_entropy_compute;
    Timer t_entropy_prove, t_lmhead_prove, t_rmsnorm_prove, t_serialize;
    Timer t_total;

    t_total.start();

    // ── Load ────────────────────────────────────────────────────────────────
    t_load.start();
    cout << "Loading inputs from " << workdir << " ..." << endl;
    cout << "  seq_len=" << seq_len << "  hidden_size=" << hidden_size
         << "  vocab_size=" << vocab_size << "  sigma_eff=" << sigma_eff << endl;

    FrTensor hidden  = FrTensor::from_int_bin(path("layer-31-output.bin"));
    FrTensor rms_inv = FrTensor::from_int_bin(path("final_norm-rms_inv.bin"));

    if (hidden.size  != seq_len * hidden_size)
        throw std::runtime_error("hidden state size mismatch");
    if (rms_inv.size != seq_len)
        throw std::runtime_error("rms_inv size mismatch");

    // Hiding Pedersen weights (Hyrax §3.1); see zkllm_entropy.cu for
    // the on-disk layout including the .h / .r sidecars.
    Weight final_norm_w = create_weight(
        path("input_layernorm.weight-pp.bin"),
        path("final_norm.weight-int.bin"),
        path("final_norm.weight-commitment.bin"),
        path("final_norm.weight-commitment.bin.r"),
        1, hidden_size);

    Weight lm_head_w = create_weight(
        path("lm_head-pp.bin"),
        path("lm_head-weight-int.bin"),
        path("lm_head-weight-commitment.bin"),
        path("lm_head-weight-commitment.bin.r"),
        hidden_size, vocab_size);

    vector<uint> tokens = load_token_sequence(tokens_file);
    if (tokens.size() != seq_len)
        throw std::runtime_error("token count != seq_len");
    cudaDeviceSynchronize();
    t_load.stop();

    // ── RMSNorm compute ─────────────────────────────────────────────────────
    t_rmsnorm_compute.start();
    cout << "Computing final RMSNorm..." << endl;
    Rescaling rs_norm1(1u << 16);
    Rescaling rs_norm2(1u << 16);
    zkFC norm_fc(1, hidden_size, final_norm_w.weight);
    FrTensor g_inv_rms  = norm_fc(rms_inv);
    FrTensor g_inv_rms_ = rs_norm1(g_inv_rms);
    FrTensor normed     = g_inv_rms_ * hidden;
    FrTensor normed_    = rs_norm2(normed);
    cudaDeviceSynchronize();
    t_rmsnorm_compute.stop();

    // ── lm_head compute ─────────────────────────────────────────────────────
    t_lmhead_compute.start();
    cout << "Computing lm_head..." << endl;
    Rescaling rs_lm(1u << 16);
    zkFC lm_fc(hidden_size, vocab_size, lm_head_w.weight);
    FrTensor logits_batch  = lm_fc(normed_);
    FrTensor logits_batch_ = rs_lm(logits_batch);
    cudaDeviceSynchronize();
    t_lmhead_compute.stop();

    // ── Split logits ────────────────────────────────────────────────────────
    cout << "Splitting logits by position..." << endl;
    vector<FrTensor> logits_seq;
    logits_seq.reserve(seq_len);
    for (uint pos = 0; pos < seq_len; pos++)
        logits_seq.push_back(tensor_row(logits_batch_, pos, vocab_size));

    // ── Entropy compute ─────────────────────────────────────────────────────
    t_entropy_compute.start();
    cout << "Building entropy lookup tables..." << endl;
    zkConditionalEntropy entropy_prover(
        vocab_size, cdf_precision, log_precision,
        cdf_scale, log_scale, sigma_eff);

    cout << "Computing conditional entropy..." << endl;
    Fr_t total_entropy = entropy_prover.compute(logits_seq, tokens);

    unsigned long entropy_val =
        ((unsigned long)total_entropy.val[1] << 32) | total_entropy.val[0];
    double entropy_bits = (double)entropy_val / (double)log_scale;
    cout << "Conditional entropy bound : " << entropy_bits << " bits total" << endl;
    cout << "Average per token         : " << entropy_bits / seq_len << " bits/token" << endl;
    cudaDeviceSynchronize();
    t_entropy_compute.stop();

    // ── Entropy prove ───────────────────────────────────────────────────────
    t_entropy_prove.start();
    cout << "Generating entropy proof..." << endl;
    vector<Polynomial> proof;
    vector<Claim> entropy_claims;
    vector<Fr_t> challenges;
    vector<FriPcsCommitment> commitments;
    entropy_prover.prove(logits_seq, tokens, total_entropy, proof, entropy_claims, challenges, commitments);
    cudaDeviceSynchronize();
    t_entropy_prove.stop();

    // ── lm_head prove ───────────────────────────────────────────────────────
    t_lmhead_prove.start();
    cout << "Proving lm_head (zkFC)..." << endl;
    rs_lm.prove(logits_batch, logits_batch_);
    verifyWeightClaimZK(lm_head_w, lm_fc.prove(normed_, logits_batch)[0]);
    cudaDeviceSynchronize();
    t_lmhead_prove.stop();

    // ── RMSNorm prove ───────────────────────────────────────────────────────
    t_rmsnorm_prove.start();
    cout << "Proving final RMSNorm..." << endl;
    rs_norm2.prove(normed, normed_);
    auto u_hp = random_vec(ceilLog2(normed.size));
    hadamard_product_sumcheck(g_inv_rms_, hidden, u_hp, random_vec(ceilLog2(normed.size)));
    rs_norm1.prove(g_inv_rms, g_inv_rms_);
    verifyWeightClaimZK(final_norm_w, norm_fc.prove(rms_inv, g_inv_rms)[0]);
    cudaDeviceSynchronize();
    t_rmsnorm_prove.stop();

    // ── Serialise ───────────────────────────────────────────────────────────
    t_serialize.start();
    {
        ofstream f(proof_output, ios::binary);
        if (!f) { cerr << "Cannot write: " << proof_output << endl; return 1; }

        uint64_t magic = 0x5A4B454E54524F50ULL;
        f.write((char*)&magic,       sizeof(magic));
        f.write((char*)&entropy_val, sizeof(entropy_val));
        f.write((char*)&seq_len,     sizeof(seq_len));
        f.write((char*)&vocab_size,  sizeof(vocab_size));
        f.write((char*)&sigma_eff,   sizeof(sigma_eff));
        f.write((char*)&log_scale,   sizeof(log_scale));

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
    t_serialize.stop();

    t_total.stop();

    // ── Print timing summary ────────────────────────────────────────────────
    cout << "\n===== TIMING BREAKDOWN =====" << endl;
    cout << "Load data + weights   : " << t_load.getTotalTime()           << " s" << endl;
    cout << "RMSNorm compute       : " << t_rmsnorm_compute.getTotalTime()<< " s" << endl;
    cout << "lm_head compute       : " << t_lmhead_compute.getTotalTime() << " s" << endl;
    cout << "Entropy compute       : " << t_entropy_compute.getTotalTime()<< " s" << endl;
    cout << "Entropy prove         : " << t_entropy_prove.getTotalTime()  << " s" << endl;
    cout << "lm_head prove         : " << t_lmhead_prove.getTotalTime()   << " s" << endl;
    cout << "RMSNorm prove         : " << t_rmsnorm_prove.getTotalTime()  << " s" << endl;
    cout << "Serialize proof       : " << t_serialize.getTotalTime()      << " s" << endl;
    cout << "TOTAL                 : " << t_total.getTotalTime()          << " s" << endl;
    cout << "============================" << endl;

    cout << "\nProof written to " << proof_output
         << " (" << proof.size() << " polynomials)" << endl;

    return 0;
}
