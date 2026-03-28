// Combined: Self-Attn Attention + Skip + RMSNorm (post-attn) + FFN + Skip
// Single CUDA context init instead of three separate processes.
// Also computes rms_inv internally and does skip-connections in-process.
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

// CPU skip-connection: element-wise int32 addition
static void cpu_skip_connection(const std::string& in1, const std::string& in2, const std::string& out) {
    FILE* f1 = fopen(in1.c_str(), "rb");
    fseek(f1, 0, SEEK_END);
    long size = ftell(f1);
    fseek(f1, 0, SEEK_SET);
    int n = size / sizeof(int32_t);

    int32_t* a = (int32_t*)malloc(size);
    int32_t* b = (int32_t*)malloc(size);
    fread(a, sizeof(int32_t), n, f1);
    fclose(f1);

    FILE* f2 = fopen(in2.c_str(), "rb");
    fread(b, sizeof(int32_t), n, f2);
    fclose(f2);

    for (int i = 0; i < n; i++) a[i] += b[i];

    FILE* fo = fopen(out.c_str(), "wb");
    fwrite(a, sizeof(int32_t), n, fo);
    fclose(fo);
    free(a);
    free(b);
}

// Compute rms_inv from int32 input
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
        out[i] = (int32_t)llround(rms_inv * scale_factor);
    }

    FILE* fo = fopen(output_file.c_str(), "wb");
    fwrite(out, sizeof(int32_t), seq_len, fo);
    fclose(fo);
    free(data);
    free(out);
}

int main(int argc, char *argv[]) {
    if (argc < 10) {
        fprintf(stderr, "Usage: %s <attn_input> <seq_len> <embed_dim> <hidden_dim> <workdir> <layer_prefix> <layer_input> <attn_output> <layer_output>\n", argv[0]);
        return 1;
    }

    std::string attn_input = argv[1];
    uint seq_len = std::stoi(argv[2]);
    uint embed_dim = std::stoi(argv[3]);
    uint hidden_dim = std::stoi(argv[4]);
    std::string workdir = argv[5];
    std::string layer_prefix = argv[6];
    std::string layer_input = argv[7];
    std::string attn_output = argv[8];
    std::string layer_output = argv[9];

    double epsilon = (argc > 10) ? std::stod(argv[10]) : 1e-5;
    int scale_factor = 1 << 16;

    // Derived paths
    std::string post_attn = workdir + "/" + layer_prefix + "-post_attn.bin";
    std::string ffn_input = workdir + "/" + layer_prefix + "-ffn_input.bin";
    std::string ffn_output = workdir + "/" + layer_prefix + "-ffn_output.bin";

    // -----------------------------------------------------------------------
    // Step 1: Self-attention attention proof
    // -----------------------------------------------------------------------
    {
        auto Q = FrTensor::from_int_bin("temp_Q.bin");
        auto K = FrTensor::from_int_bin("temp_K.bin");
        auto V = FrTensor::from_int_bin("temp_V.bin");
        auto d = Q.size / seq_len;

        auto X_attn = FrTensor::matmul(Q, K.transpose(seq_len, d), seq_len, d, seq_len);

        zkSoftmax softmax({1<<8, 1<<20, 1<<20}, 1, 0, 1UL<<32, {1<<18, 1<<22}, seq_len, seq_len, d, 1);
        Rescaling rs1(1<<20), rs2(1<<20);

        FrTensor shift(seq_len), X_shifted(seq_len * seq_len);
        vector<FrTensor> X_segments, Y_segments, m_segments;
        FrTensor Y_soft = softmax.compute(X_attn, shift, X_shifted, X_segments, Y_segments, m_segments);
        Y_soft.save_long("temp_head_Y.bin");

        auto out = FrTensor::matmul(Y_soft, V, seq_len, seq_len, d);
        auto out_ = rs2(out);
        auto out__ = rs1(out_);
        out__.save_int("temp_head_out.bin");

        rs1.prove(out_, out__);
        rs2.prove(out, out_);
        auto temp_rand = random_vec(3);
        vector<Polynomial> proof;
        auto u1 = random_vec(ceilLog2(seq_len));
        auto u2 = random_vec(ceilLog2(d));
        auto ud = random_vec(ceilLog2(seq_len));
        auto claim = out.multi_dim_me({u1, u2}, {seq_len, d});
        auto final_claim = zkip(claim, Y_soft.partial_me(u1, seq_len, seq_len), V.partial_me(u2, d, 1), ud, proof);

        softmax.prove(Y_soft, X_attn, shift, X_shifted, X_segments, Y_segments, m_segments,
            random_vec(ceilLog2(Y_soft.size)), random_vec(ceilLog2(Y_soft.size)), temp_rand[0], temp_rand[1], temp_rand[2], proof);
        auto u1_ = random_vec(ceilLog2(seq_len));
        auto u2_ = random_vec(ceilLog2(seq_len));
        auto ud_ = random_vec(ceilLog2(d));
        auto claim_ = X_attn.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
        auto final_claim_ = zkip(claim_, Q.partial_me(u1_, seq_len, d), K.partial_me(u2_, seq_len, d), ud_, proof);
        cout << "Self attention proof verified." << endl;
    }

    // Clean up temp files
    remove("temp_Q.bin");
    remove("temp_K.bin");
    remove("temp_V.bin");
    remove("temp_head_Y.bin");
    remove("temp_head_out.bin");

    // -----------------------------------------------------------------------
    // Step 2: Skip connection 1 (CPU): layer_input + attn_output → post_attn
    // -----------------------------------------------------------------------
    cpu_skip_connection(layer_input, attn_output, post_attn);

    // -----------------------------------------------------------------------
    // Step 3: Compute rms_inv for post-attention layernorm (CPU)
    // -----------------------------------------------------------------------
    compute_rms_inv(post_attn, seq_len, embed_dim, epsilon, scale_factor, "rms_inv_temp.bin");

    // -----------------------------------------------------------------------
    // Step 4: RMSNorm (post-attention) proof
    // -----------------------------------------------------------------------
#ifdef USE_GOLDILOCKS
    auto rmsnorm_weight = create_weight(
        workdir + "/" + layer_prefix + "-post_attention_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-post_attention_layernorm.weight-gold-commitment.bin",
        1, embed_dim);
#else
    auto rmsnorm_weight = create_weight(
        workdir + "/post_attention_layernorm.weight-pp.bin",
        workdir + "/" + layer_prefix + "-post_attention_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-post_attention_layernorm.weight-commitment.bin",
        1, embed_dim);
#endif

    FrTensor X_post = FrTensor::from_int_bin(post_attn);
    FrTensor rms_inv_temp = FrTensor::from_int_bin("rms_inv_temp.bin");

    Rescaling rms_rs1(scale_factor), rms_rs2(scale_factor);
    zkFC g(1, embed_dim, rmsnorm_weight.weight);
    auto g_inv_rms = g(rms_inv_temp);
    auto g_inv_rms_ = rms_rs1(g_inv_rms);
    auto Y_rms = g_inv_rms_ * X_post;
    auto Y_rms_ = rms_rs2(Y_rms);

    rms_rs2.prove(Y_rms, Y_rms_);
    Y_rms_.save_int(ffn_input);
    hadamard_product_sumcheck(g_inv_rms_, X_post, random_vec(ceilLog2(Y_rms.size)), random_vec(ceilLog2(Y_rms.size)));
    rms_rs1.prove(g_inv_rms, g_inv_rms_);
    verifyWeightClaim(rmsnorm_weight, g.prove(rms_inv_temp, g_inv_rms)[0]);

    remove("rms_inv_temp.bin");

    // -----------------------------------------------------------------------
    // Step 5: FFN proof
    // -----------------------------------------------------------------------
#ifdef USE_GOLDILOCKS
    auto up_proj = create_weight(
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-gold-commitment.bin",
        embed_dim, hidden_dim);
    auto gate_proj = create_weight(
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-gold-commitment.bin",
        embed_dim, hidden_dim);
    auto down_proj = create_weight(
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-gold-commitment.bin",
        hidden_dim, embed_dim);
#else
    auto up_proj = create_weight(
        workdir + "/mlp.up_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-commitment.bin",
        embed_dim, hidden_dim);
    auto gate_proj = create_weight(
        workdir + "/mlp.gate_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-commitment.bin",
        embed_dim, hidden_dim);
    auto down_proj = create_weight(
        workdir + "/mlp.down_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-commitment.bin",
        hidden_dim, embed_dim);
#endif

    zkFC up_layer(embed_dim, hidden_dim, up_proj.weight);
    zkFC gate_layer(embed_dim, hidden_dim, gate_proj.weight);
    zkFC down_layer(hidden_dim, embed_dim, down_proj.weight);

    Rescaling up_rescale(scale_factor);
    Rescaling gate_rescale(1 << 20);
    Rescaling hidden_rescale(scale_factor);
    Rescaling down_rescale(scale_factor);

    FrTensor swiglu_values = FrTensor::from_int_bin("swiglu-table.bin");
    tLookupRangeMapping swiglu(-(1 << 21), 1 << 22, swiglu_values);

    FrTensor ffn_in = FrTensor::from_int_bin(ffn_input);
    auto up_out = up_layer(ffn_in);
    auto up_out_ = up_rescale(up_out);
    auto gate_out = gate_layer(ffn_in);
    auto gate_out_ = gate_rescale(gate_out);
    auto p = swiglu(gate_out_);
    auto &swiglu_out = p.first, &swiglu_m = p.second;

    auto temp_rand = random_vec(3);
    auto swiglu_u = random_vec(ceilLog2(seq_len * hidden_dim));
    auto swiglu_v = random_vec(ceilLog2(seq_len * hidden_dim));
    vector<Polynomial> swiglu_proof;

    auto down_in = swiglu_out * up_out_;
    auto down_in_ = hidden_rescale(down_in);
    auto down_out = down_layer(down_in_);
    auto down_out_ = down_rescale(down_out);
    down_out.save_int(ffn_output);

    down_rescale.prove(down_out, down_out_);
    verifyWeightClaim(down_proj, down_layer.prove(down_in_, down_out)[0]);
    hidden_rescale.prove(down_in, down_in_);
    swiglu.prove(gate_out_, swiglu_out, swiglu_m, temp_rand[0], temp_rand[1], temp_rand[2], swiglu_u, swiglu_v, swiglu_proof);
    cout << "SwiGLU proof complete." << endl;
    gate_rescale.prove(gate_out, gate_out_);
    verifyWeightClaim(gate_proj, gate_layer.prove(ffn_in, gate_out)[0]);
    up_rescale.prove(up_out, up_out_);
    verifyWeightClaim(up_proj, up_layer.prove(ffn_in, up_out)[0]);

    // -----------------------------------------------------------------------
    // Step 6: Skip connection 2 (CPU): post_attn + ffn_output → layer_output
    // -----------------------------------------------------------------------
    cpu_skip_connection(post_attn, ffn_output, layer_output);

    cout << "Layer complete (attn + skip + rmsnorm + ffn + skip)." << endl;
    return 0;
}
