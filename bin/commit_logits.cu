// commit_logits: commit a saved logit FrTensor to its polynomial commitment.
//
// Usage:
//   ./commit_logits <generators_file> <logits_file> <commitment_out>
//
// <generators_file> : Commitment (G1TensorJacobian) saved by ppgen,
//                     size must be 2^ceilLog2(vocab_size).
// <logits_file>     : Raw FrTensor binary (vocab_size Fr_t elements).
// <commitment_out>  : Output path for the G1TensorJacobian commitment.
//
// The logit tensor is padded to the generator size before committing so that
// Commitment::me_open can later open it at any challenge point.

#include "commit/commitment.cuh"
#include "util/ioutils.cuh"
#include <iostream>
#include <stdexcept>
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0]
             << " <generators_file> <logits_file> <commitment_out>" << endl;
        return 1;
    }

    string gen_file    = argv[1];
    string logits_file = argv[2];
    string out_file    = argv[3];

    // Load generators via hiding-pp path; if the `.h` sidecar is absent
    // this still succeeds and leaves is_hiding() == false (legacy pp).
    Commitment generators = Commitment::load_hiding(gen_file);
    uint padded = generators.size;

    // Load logit tensor.
    uint fsize = findsize(logits_file);
    if (fsize == 0) { cerr << "Cannot open: " << logits_file << endl; return 1; }
    uint n_elements = fsize / sizeof(Fr_t);
    if (n_elements > padded) {
        cerr << "Logit tensor size " << n_elements
             << " exceeds generator size " << padded << endl;
        return 1;
    }

    FrTensor logits(n_elements);
    loadbin(logits_file, logits.gpu_data, fsize);

    // Pad to generator size (fill with zeros).
    FrTensor logits_padded = (n_elements == padded)
        ? logits
        : logits.pad({n_elements}, FR_FROM_INT(0));

    // Commit: hiding if pp has H, legacy non-hiding otherwise.
    //
    // Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
    //   "We say that Com_pp(m; r) is a commitment to the message m with
    //    randomness r".
    if (generators.is_hiding()) {
        auto hc = generators.commit_hiding(logits_padded);
        hc.com.save(out_file);
        hc.r.save(out_file + ".r");
        cout << "Committed " << n_elements << " elements (padded to " << padded
             << ") -> " << out_file << " (+ " << out_file << ".r, hiding)" << endl;
    } else {
        G1TensorJacobian com = generators.commit(logits_padded);
        com.save(out_file);
        cout << "Committed " << n_elements << " elements (padded to " << padded
             << ") -> " << out_file << " (legacy non-hiding)" << endl;
    }
    return 0;
}
