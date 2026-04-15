// ppgen: generate public Pedersen commitment parameters for zkllm-entropy.
//
// Phase 1.5: generators are derived from a public domain-separation tag
// via RFC 9380 hash-to-curve, so ppgen holds no secret scalars and the
// output pp has no pairwise-dlog toxic waste.
//
// Emits a single v2 file containing {G_i} + H + U and the DST used for
// derivation (see Commitment::save_hiding for the layout).  Receivers
// verify integrity by recomputing from the DST (Commitment::verify_pp /
// Commitment::load_hiding).
//
// Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//   "Informally, a commitment scheme allows a sender to produce a message
//    C = Com(m) that hides m from a receiver but binds the sender to the
//    value m."
//
// Use --legacy to emit the old non-hiding pp only (no hiding generators,
// no v2 wrapper) for regression testing against pre-Phase-1 fixtures.
//
// Usage: ./ppgen <size> <out_file> [--legacy]

#include "commit/commitment.cuh"
#include "field/hash_to_curve.cuh"
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
        std::cout << "Emitted hash-to-curve hiding pp (" << padded
                  << " generators + H + U, DST "
                  << ZKLLM_ENTROPY_PEDERSEN_DST_V1 << ") -> "
                  << filename << std::endl;
    }
    return 0;
}
