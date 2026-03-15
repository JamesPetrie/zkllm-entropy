// zkllm_entropy: standalone conditional-entropy prover.
//
// Usage:
//   ./zkllm_entropy <logits_dir> <tokens_file> <proof_output> <sigma_eff>
//                  [bit_width=16] [cdf_precision=18] [log_precision=15]
//                  [cdf_scale=65536] [log_scale=65536]
//
// <logits_dir>   : directory containing logits_0.bin, logits_1.bin, ...
//                  each file is a raw FrTensor saved with FrTensor::save().
// <tokens_file>  : one uint per line, the actual output token at each position.
// <proof_output> : path to write proof metadata (placeholder; full serialisation
//                  not yet implemented — see TODO in zkentropy.cu).
// <sigma_eff>    : Gaussian noise std dev in field integer units
//                  (= sigma_real * logit_scaling_factor, e.g. 0.05 * 65536 = 3277).
//
// Constraint: vocab_size must be divisible by 2^log_precision for the log-lookup
// prove to work.  For LLaMA (vocab=32000, log_precision=15): 32000 % 32768 != 0,
// but 32000 is padded to 32768 internally by tLookupRangeMapping::prove.
// Alternatively use log_precision <= 10 (table <= 1024) so 32000 / 1024 = 31.25;
// still not exact — easiest workaround is log_precision=5 (32 entries) giving
// 32000 % 32 == 0, or pad vocab_size to next power of 2.
// The compute() path works regardless of divisibility constraints.

#include "zkentropy.cuh"
#include "ioutils.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdlib>

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
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " <logits_dir> <tokens_file> <proof_output> <sigma_eff>"
             << " [bit_width=16] [cdf_precision=18] [log_precision=15]"
             << " [cdf_scale=65536] [log_scale=65536]" << endl;
        return 1;
    }

    string logits_dir   = argv[1];
    string tokens_file  = argv[2];
    string proof_output = argv[3];
    double sigma_eff    = atof(argv[4]);

    uint bit_width     = (argc > 5) ? (uint)atoi(argv[5]) : 16u;
    uint cdf_precision = (argc > 6) ? (uint)atoi(argv[6]) : 18u;
    uint log_precision = (argc > 7) ? (uint)atoi(argv[7]) : 15u;
    uint cdf_scale     = (argc > 8) ? (uint)atoi(argv[8]) : 65536u;
    uint log_scale     = (argc > 9) ? (uint)atoi(argv[9]) : 65536u;

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
         << "  log_precision=" << log_precision << endl;

    // Build the prover (pre-computes lookup tables on GPU).
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
    cout << "Generating proof..." << endl;
    vector<Polynomial> proof;
    prover.prove(logits_seq, tokens, total_entropy, proof);

    // Write a simple proof summary (full serialisation is TODO).
    {
        ofstream f(proof_output);
        if (!f) { cerr << "Cannot write proof output: " << proof_output << endl; return 1; }
        f << "entropy_scaled=" << entropy_val << "\n";
        f << "entropy_bits="   << entropy_bits << "\n";
        f << "tokens=" << T << "\n";
        f << "vocab_size=" << vocab_size << "\n";
        f << "sigma_eff=" << sigma_eff << "\n";
        f << "log_scale=" << log_scale << "\n";
        f << "proof_polynomials=" << proof.size() << "\n";
    }
    cout << "Proof summary written to " << proof_output << endl;

    return 0;
}
