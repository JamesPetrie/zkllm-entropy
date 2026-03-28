// Combined: RMSNorm (input) + Self-Attention Linear
// Single CUDA context init instead of two separate processes.
// Also computes rms_inv internally (no Python/torch dependency).
#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#ifndef USE_GOLDILOCKS
#include "commitment.cuh"
#endif
#include "rescaling.cuh"
#include "ioutils.cuh"
#include <string>
#include <cmath>

// Compute rms_inv from int32 input and save to file.
// rms_inv[i] = 1 / sqrt(mean(row_i^2) + epsilon), quantized to int32 with scale_factor.
static void compute_rms_inv(const std::string& input_file, int seq_len, int embed_dim,
                            double epsilon, int scale_factor, const std::string& output_file) {
    FILE* f = fopen(input_file.c_str(), "rb");
    int n = seq_len * embed_dim;
    int32_t* data = (int32_t*)malloc(n * sizeof(int32_t));
    fread(data, sizeof(int32_t), n, f);
    fclose(f);

    int32_t* out = (int32_t*)malloc(seq_len * sizeof(int32_t));
    for (int i = 0; i < seq_len; i++) {
        double sum_sq = 0.0;
        for (int j = 0; j < embed_dim; j++) {
            double val = (double)data[i * embed_dim + j] / scale_factor;
            sum_sq += val * val;
        }
        double rms_inv = 1.0 / sqrt(sum_sq / embed_dim + epsilon);
        // Round to nearest int32 with scale_factor
        out[i] = (int32_t)llround(rms_inv * scale_factor);
    }

    FILE* fo = fopen(output_file.c_str(), "wb");
    fwrite(out, sizeof(int32_t), seq_len, fo);
    fclose(fo);

    free(data);
    free(out);
}

int main(int argc, char *argv[]) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s <layer_input> <seq_len> <embed_dim> <workdir> <layer_prefix> <attn_input_out> <attn_output>\n", argv[0]);
        return 1;
    }

    std::string layer_input = argv[1];
    uint seq_len = std::stoi(argv[2]);
    uint embed_dim = std::stoi(argv[3]);
    std::string workdir = argv[4];
    std::string layer_prefix = argv[5];
    std::string attn_input_file = argv[6];
    std::string attn_output_file = argv[7];

    // Optional: variance_epsilon (default 1e-5 for Llama-2)
    double epsilon = (argc > 8) ? std::stod(argv[8]) : 1e-5;
    int scale_factor = 1 << 16;

    // -----------------------------------------------------------------------
    // Step 1: Compute rms_inv (CPU — trivial for 1024×4096)
    // -----------------------------------------------------------------------
    compute_rms_inv(layer_input, seq_len, embed_dim, epsilon, scale_factor, "rms_inv_temp.bin");

    // -----------------------------------------------------------------------
    // Step 2: RMSNorm proof
    // -----------------------------------------------------------------------
#ifdef USE_GOLDILOCKS
    auto rmsnorm_weight = create_weight(
        workdir + "/" + layer_prefix + "-input_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-input_layernorm.weight-gold-commitment.bin",
        1, embed_dim);
#else
    auto rmsnorm_weight = create_weight(
        workdir + "/input_layernorm.weight-pp.bin",
        workdir + "/" + layer_prefix + "-input_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-input_layernorm.weight-commitment.bin",
        1, embed_dim);
#endif

    FrTensor X = FrTensor::from_int_bin(layer_input);
    FrTensor rms_inv_temp = FrTensor::from_int_bin("rms_inv_temp.bin");

    Rescaling rs1(scale_factor), rs2(scale_factor);

    zkFC g = zkFC(1, embed_dim, rmsnorm_weight.weight);
    auto g_inv_rms = g(rms_inv_temp);
    auto g_inv_rms_ = rs1(g_inv_rms);
    auto Y = g_inv_rms_ * X;
    auto Y_ = rs2(Y);

    rs2.prove(Y, Y_);
    Y_.save_int(attn_input_file);
    hadamard_product_sumcheck(g_inv_rms_, X, random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)));
    rs1.prove(g_inv_rms, g_inv_rms_);
    verifyWeightClaim(rmsnorm_weight, g.prove(rms_inv_temp, g_inv_rms)[0]);

    // -----------------------------------------------------------------------
    // Step 3: Self-attention linear (Q, K, V projections)
    // -----------------------------------------------------------------------
#ifdef USE_GOLDILOCKS
    auto q_proj = create_weight(
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-gold-commitment.bin",
        embed_dim, embed_dim);
    auto k_proj = create_weight(
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-gold-commitment.bin",
        embed_dim, embed_dim);
    auto v_proj = create_weight(
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-gold-commitment.bin",
        embed_dim, embed_dim);
#else
    auto q_proj = create_weight(
        workdir + "/self_attn.q_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-commitment.bin",
        embed_dim, embed_dim);
    auto k_proj = create_weight(
        workdir + "/self_attn.k_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-commitment.bin",
        embed_dim, embed_dim);
    auto v_proj = create_weight(
        workdir + "/self_attn.v_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-commitment.bin",
        embed_dim, embed_dim);
#endif

    zkFC q_layer(embed_dim, embed_dim, q_proj.weight);
    zkFC k_layer(embed_dim, embed_dim, k_proj.weight);
    zkFC v_layer(embed_dim, embed_dim, v_proj.weight);
    Rescaling q_rescale(scale_factor);
    Rescaling k_rescale(scale_factor);
    Rescaling v_rescale(scale_factor);

    FrTensor input = FrTensor::from_int_bin(attn_input_file);
    auto Q = q_layer(input);
    auto Q_ = q_rescale(Q);
    auto K = k_layer(input);
    auto K_ = k_rescale(K);
    auto V = v_layer(input);
    auto V_ = v_rescale(V);

    q_rescale.prove(Q, Q_);
    k_rescale.prove(K, K_);
    v_rescale.prove(V, V_);

    verifyWeightClaim(k_proj, k_layer.prove(input, K)[0]);
    verifyWeightClaim(q_proj, q_layer.prove(input, Q)[0]);
    verifyWeightClaim(v_proj, v_layer.prove(input, V)[0]);

    Q_.save_int("temp_Q.bin");
    K_.save_int("temp_K.bin");
    V_.save_int("temp_V.bin");

    cout << "RMSNorm + QKV linear proof successfully verified!" << endl;

    // Clean up rms_inv_temp
    remove("rms_inv_temp.bin");

    return 0;
}
