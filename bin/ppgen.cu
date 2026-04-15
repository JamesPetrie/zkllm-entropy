// ppgen: generate public Pedersen commitment parameters for zkllm-entropy.
//
// Emits:
//   <out_file>      : {G_i} generators (G1TensorJacobian layout, unchanged)
//   <out_file>.h    : Hiding generator H (single G1Jacobian_t; sidecar)
//
// The H sidecar makes commitments hiding (Hyrax §3.1). The original file
// format is unchanged so existing non-hiding consumers keep working; they
// simply ignore the sidecar and get hiding_generator == identity.
//
// Reference, Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//   "Informally, a commitment scheme allows a sender to produce a message
//    C = Com(m) that hides m from a receiver but binds the sender to the
//    value m."
//
// Use --legacy to emit the old non-hiding pp only (useful for regression
// testing existing test fixtures during the Phase 1 migration).
//
// Usage: ./ppgen <size> <out_file> [--legacy]

#include "commit/commitment.cuh"
#include <iostream>
#include <string>
#include <cstring>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <size> <out_file> [--legacy]\n";
        return 1;
    }
    uint size = std::stoi(argv[1]);
    std::string filename = argv[2];
    bool legacy = (argc >= 4 && std::strcmp(argv[3], "--legacy") == 0);

    uint padded = 1u << ceilLog2(size);

    if (legacy) {
        Commitment pp = Commitment::random(padded);
        pp.save(filename);
        std::cout << "Emitted legacy non-hiding pp (" << padded
                  << " generators) -> " << filename << std::endl;
    } else {
        Commitment pp = Commitment::hiding_random(padded);
        pp.save_hiding(filename);
        std::cout << "Emitted hiding pp (" << padded
                  << " generators + H) -> " << filename
                  << " (+ " << filename << ".h)" << std::endl;
    }
    return 0;
}
