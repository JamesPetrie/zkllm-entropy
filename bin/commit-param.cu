#include "tensor/fr-tensor.cuh"

#include "commit/commitment.cuh"

// BLS12-381: Pedersen commitment (requires public parameters).
//
// If <pp_file> was produced by ppgen as a hiding pp (the default), this
// binary auto-detects the `.h` sidecar, uses Commitment::commit_int_hiding
// to produce C[row] = Σⱼ Gⱼ · t[row,j] + r[row] · H per Hyrax §3.1
// Pedersen, and writes the per-row blindings to <output_file>.r so
// downstream create_weight(hiding) can reload them.
//
// Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//   "We say that Com_pp(m; r) is a commitment to the message m with
//    randomness r".
//
// If <pp_file> is legacy (no `.h` sidecar), falls back to non-hiding
// commit_int and does not write a `.r` sidecar.  Callers that need the
// hiding property must run ppgen without --legacy.
//
// Usage: ./commit-param <pp_file> <param_file> <output_file> <in_dim> <out_dim>
int main(int argc, char *argv[])
{
    string generator_filename = argv[1];
    string param_filename = argv[2];
    string output_filename = argv[3];
    uint in_dim = std::stoi(argv[4]);
    uint out_dim = std::stoi(argv[5]);

    Commitment generator = Commitment::load_hiding(generator_filename);
    // generator.size has to be a power of 2
    if (generator.size != (1 << ceilLog2(generator.size))) throw std::runtime_error("Generator size has to be a power of 2");

    FrTensor param = FrTensor::from_int_bin(param_filename);
    cout << "Param size: " << param.size << endl;
    cout << "In dim: " << in_dim << endl;
    cout << "Out dim: " << out_dim << endl;
    auto param_padded = param.pad({in_dim, out_dim});

    if (generator.is_hiding()) {
        auto hc = generator.commit_int_hiding(param_padded);
        cout << "Generator size: " << generator.size << endl;
        cout << "Unpadded param size: " << param.size << endl;
        cout << "Padded param size: " << param_padded.size << endl;
        cout << "Commitment size: " << hc.com.size << "  (hiding)" << endl;
        cout << "Blinding tensor size: " << hc.r.size << endl;

        hc.com.save(output_filename);
        hc.r.save(output_filename + ".r");
    } else {
        auto com = generator.commit_int(param_padded);
        cout << "Generator size: " << generator.size << endl;
        cout << "Unpadded param size: " << param.size << endl;
        cout << "Padded param size: " << param_padded.size << endl;
        cout << "Commitment size: " << com.size << "  (legacy non-hiding)" << endl;
        com.save(output_filename);
    }
    return 0;
}
