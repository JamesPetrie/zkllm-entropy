// zkllm_entropy: standalone conditional-entropy prover.
//
// Usage:
//   ./zkllm_entropy <logits_dir> <tokens_file> <proof_output> <sigma_eff>
//                  [bit_width=16] [cdf_precision=12] [log_precision=15]
//                  [cdf_scale=65536] [log_scale=65536]
//                  [--generators <pp_file>]   # enables strong commitment proof
//
// <logits_dir>   : directory containing logits_0.bin, logits_1.bin, ...
//                  (one vocab_size-element FrTensor per output position,
//                   produced by gen_logits.py)
// <tokens_file>  : greedy token ids, one uint per line (tokens.txt from gen_logits.py)
// <proof_output> : path to write binary proof file
// <sigma_eff>    : Gaussian noise std dev in field integer units
//                  (= sigma_real * logit_scale, e.g. 0.05 * 65536 = 3277)
//
// Strong prove (requires --generators):
//   Also loads logits_N-commitment.bin from <logits_dir> and performs
//   Commitment::me_open to bind each logits[actual_token] to the committed tensor.
//   Generate pp file with:  ./ppgen 32768 <pp_file>
//   Generate logit files with:  python gen_logits.py --generators <pp_file>

#include "zkentropy.cuh"
#include "ioutils.cuh"
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

int main(int argc, char* argv[]) {
    // Parse positional args and optional --generators flag.
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " <logits_dir> <tokens_file> <proof_output> <sigma_eff>"
             << " [bit_width=16] [cdf_precision=12] [log_precision=15]"
             << " [cdf_scale=65536] [log_scale=65536]"
             << " [--generators <pp_file>]" << endl;
        return 1;
    }

    string logits_dir   = argv[1];
    string tokens_file  = argv[2];
    string proof_output = argv[3];
    double sigma_eff    = atof(argv[4]);

    uint bit_width     = (argc > 5) ? (uint)atoi(argv[5]) : 16u;
    uint cdf_precision = (argc > 6) ? (uint)atoi(argv[6]) : 12u;
    uint log_precision = (argc > 7) ? (uint)atoi(argv[7]) : 15u;
    uint cdf_scale     = (argc > 8) ? (uint)atoi(argv[8]) : 65536u;
    uint log_scale     = (argc > 9) ? (uint)atoi(argv[9]) : 65536u;

    string generators_file;
    for (int i = 10; i < argc - 1; i++) {
        if (strcmp(argv[i], "--generators") == 0)
            generators_file = argv[i + 1];
    }

    bool strong_prove = !generators_file.empty();

    // Load token sequence.
    vector<uint> tokens = load_token_sequence(tokens_file);
    uint T = tokens.size();
    if (T == 0) { cerr << "No tokens loaded." << endl; return 1; }

    // Load logit tensors (logits_0.bin … logits_{T-1}.bin).
    vector<FrTensor> logits_seq;
    logits_seq.reserve(T);
    for (uint t = 0; t < T; t++) {
        string path = logits_dir + "/logits_" + to_string(t) + ".bin";
        uint fsize = findsize(path);
        if (fsize == 0) throw std::runtime_error("Cannot find/open: " + path);
        uint n_elements = fsize / sizeof(Fr_t);
        FrTensor lt(n_elements);
        loadbin(path, lt.gpu_data, fsize);
        logits_seq.push_back(lt);
    }

    uint vocab_size = logits_seq[0].size;
    cout << "Loaded " << T << " positions, vocab_size=" << vocab_size << endl;
    cout << "sigma_eff=" << sigma_eff
         << "  bit_width=" << bit_width
         << "  cdf_precision=" << cdf_precision
         << "  log_precision=" << log_precision
         << "  prove=" << (strong_prove ? "strong" : "weak") << endl;

    // Build prover.
    cout << "Building lookup tables..." << endl;
    zkConditionalEntropy prover(
        vocab_size, bit_width, cdf_precision, log_precision,
        cdf_scale, log_scale, sigma_eff);

    // Compute entropy.
    cout << "Computing conditional entropy..." << endl;
    Fr_t total_entropy = prover.compute(logits_seq, tokens);

    unsigned long entropy_val =
        ((unsigned long)total_entropy.val[1] << 32) | total_entropy.val[0];
    double entropy_bits = (double)entropy_val / (double)log_scale;
    cout << "Conditional entropy bound : " << entropy_bits << " bits total" << endl;
    cout << "Average per token         : " << entropy_bits / T << " bits/token" << endl;

    // Prove.
    cout << "Generating proof (" << (strong_prove ? "strong" : "weak") << ")..." << endl;
    vector<Polynomial>   proof;
    vector<G1Jacobian_t> g1_proof;

    if (strong_prove) {
        // Load generators.
        Commitment generators(generators_file);
        uint padded = 1u << ceilLog2(vocab_size);
        if (generators.size != padded) {
            cerr << "Generator size " << generators.size
                 << " != 2^ceilLog2(vocab_size)=" << padded << endl;
            return 1;
        }

        // Load per-position logit commitments.
        vector<G1TensorJacobian> logit_commits;
        logit_commits.reserve(T);
        for (uint t = 0; t < T; t++) {
            string com_path = logits_dir + "/logits_" + to_string(t) + "-commitment.bin";
            logit_commits.emplace_back(com_path);
        }

        prover.prove(logits_seq, tokens, total_entropy,
                     generators, logit_commits, proof, g1_proof);
    } else {
        prover.prove(logits_seq, tokens, total_entropy, proof);
    }

    // Serialise proof to binary file.
    // Format:
    //   [8 bytes]  magic 0x5A4B454E54524F50 ("ZKENTROP")
    //   [8 bytes]  entropy_scaled (uint64)
    //   [4 bytes]  T
    //   [4 bytes]  vocab_size
    //   [8 bytes]  sigma_eff (double)
    //   [4 bytes]  log_scale
    //   [1 byte]   strong_prove flag
    //   [4 bytes]  n_polys
    //   For each polynomial:
    //     [4 bytes] n_coeffs (degree+1)
    //     [n_coeffs * 32 bytes] evaluations at 0,1,...,deg
    //   [4 bytes]  n_g1  (number of G1Jacobian_t elements; 0 if weak prove)
    //   [n_g1 * sizeof(G1Jacobian_t) bytes] g1 proof elements
    {
        ofstream f(proof_output, ios::binary);
        if (!f) { cerr << "Cannot write: " << proof_output << endl; return 1; }

        uint64_t magic = 0x5A4B454E54524F50ULL;
        f.write((char*)&magic,       sizeof(magic));
        f.write((char*)&entropy_val, sizeof(entropy_val));
        f.write((char*)&T,           sizeof(T));
        f.write((char*)&vocab_size,  sizeof(vocab_size));
        f.write((char*)&sigma_eff,   sizeof(sigma_eff));
        f.write((char*)&log_scale,   sizeof(log_scale));
        uint8_t sp = strong_prove ? 1 : 0;
        f.write((char*)&sp, 1);

        uint32_t n_polys = (uint32_t)proof.size();
        f.write((char*)&n_polys, sizeof(n_polys));
        for (const Polynomial& poly : proof) {
            int deg = poly.getDegree();
            uint32_t n_coeffs = (deg >= 0) ? (uint32_t)(deg + 1) : 0u;
            f.write((char*)&n_coeffs, sizeof(n_coeffs));
            for (uint32_t k = 0; k < n_coeffs; k++) {
                Fr_t xk = {k, 0, 0, 0, 0, 0, 0, 0};
                Fr_t yk = const_cast<Polynomial&>(poly)(xk);
                f.write((char*)&yk, sizeof(Fr_t));
            }
        }

        // G1 proof elements.
        uint32_t n_g1 = (uint32_t)g1_proof.size();
        f.write((char*)&n_g1, sizeof(n_g1));
        if (n_g1 > 0) {
            // Copy G1 elements to host and write.
            vector<G1Jacobian_t> g1_host(n_g1);
            cudaMemcpy(g1_host.data(), g1_proof.data(),
                       n_g1 * sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
            f.write((char*)g1_host.data(), n_g1 * sizeof(G1Jacobian_t));
        }
    }
    cout << "Proof written to " << proof_output
         << " (" << proof.size() << " polynomials, "
         << g1_proof.size() << " G1 elements)" << endl;

    return 0;
}
