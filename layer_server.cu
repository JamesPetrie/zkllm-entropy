// Persistent CUDA layer proof server.
// Initializes CUDA once, then reads commands from stdin.
// Eliminates ALL per-process CUDA context initialization overhead.
//
// Protocol:
//   stdin:  RMSNORM_LINEAR <layer_input> <seq_len> <embed_dim> <workdir> <layer_prefix> <attn_input_out> <attn_output_out>
//           POST_ATTN <attn_input> <seq_len> <embed_dim> <hidden_dim> <workdir> <layer_prefix> <layer_input> <attn_output> <layer_output>
//           QUIT
//   stdout: DONE (after each command completes)

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
#include <sstream>
#include <cmath>
#include <iostream>

// CPU skip-connection: element-wise int32 addition
static void cpu_skip_connection(const std::string& in1, const std::string& in2, const std::string& out) {
    FILE* f1 = fopen(in1.c_str(), "rb");
    fseek(f1, 0, SEEK_END);
    long size = ftell(f1);
    fseek(f1, 0, SEEK_SET);
    int n = size / sizeof(int32_t);
    int32_t* a = (int32_t*)malloc(size);
    int32_t* b = (int32_t*)malloc(size);
    fread(a, sizeof(int32_t), n, f1); fclose(f1);
    FILE* f2 = fopen(in2.c_str(), "rb");
    fread(b, sizeof(int32_t), n, f2); fclose(f2);
    for (int i = 0; i < n; i++) a[i] += b[i];
    FILE* fo = fopen(out.c_str(), "wb");
    fwrite(a, sizeof(int32_t), n, fo); fclose(fo);
    free(a); free(b);
}

// Compute rms_inv from int32 input
static void compute_rms_inv(const std::string& input_file, int seq_len, int embed_dim,
                            double epsilon, int scale_factor, const std::string& output_file) {
    FILE* f = fopen(input_file.c_str(), "rb");
    int n = seq_len * embed_dim;
    int32_t* data = (int32_t*)malloc(n * sizeof(int32_t));
    fread(data, sizeof(int32_t), n, f); fclose(f);
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
    fwrite(out, sizeof(int32_t), seq_len, fo); fclose(fo);
    free(data); free(out);
}

static void do_rmsnorm_linear(const std::string& layer_input, uint seq_len, uint embed_dim,
                               const std::string& workdir, const std::string& layer_prefix,
                               const std::string& attn_input_file, const std::string& attn_output_file) {
    int scale_factor = 1 << 16;
    compute_rms_inv(layer_input, seq_len, embed_dim, 1e-5, scale_factor, "rms_inv_temp.bin");

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
    zkFC g(1, embed_dim, rmsnorm_weight.weight);
    auto g_inv_rms = g(rms_inv_temp);
    auto g_inv_rms_ = rs1(g_inv_rms);
    auto Y = g_inv_rms_ * X;
    auto Y_ = rs2(Y);
    rs2.prove(Y, Y_);
    Y_.save_int(attn_input_file);
    hadamard_product_sumcheck(g_inv_rms_, X, random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)));
    rs1.prove(g_inv_rms, g_inv_rms_);
    verifyWeightClaim(rmsnorm_weight, g.prove(rms_inv_temp, g_inv_rms)[0]);
    remove("rms_inv_temp.bin");

#ifdef USE_GOLDILOCKS
    auto q_proj = create_weight(workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-gold-commitment.bin", embed_dim, embed_dim);
    auto k_proj = create_weight(workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-gold-commitment.bin", embed_dim, embed_dim);
    auto v_proj = create_weight(workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-gold-commitment.bin", embed_dim, embed_dim);
#else
    auto q_proj = create_weight(workdir + "/self_attn.q_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-commitment.bin", embed_dim, embed_dim);
    auto k_proj = create_weight(workdir + "/self_attn.k_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-commitment.bin", embed_dim, embed_dim);
    auto v_proj = create_weight(workdir + "/self_attn.v_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-commitment.bin", embed_dim, embed_dim);
#endif

    zkFC q_layer(embed_dim, embed_dim, q_proj.weight);
    zkFC k_layer(embed_dim, embed_dim, k_proj.weight);
    zkFC v_layer(embed_dim, embed_dim, v_proj.weight);
    Rescaling q_rescale(scale_factor), k_rescale(scale_factor), v_rescale(scale_factor);

    FrTensor input = FrTensor::from_int_bin(attn_input_file);
    auto Q = q_layer(input);  auto Q_ = q_rescale(Q);
    auto K = k_layer(input);  auto K_ = k_rescale(K);
    auto V = v_layer(input);  auto V_ = v_rescale(V);
    q_rescale.prove(Q, Q_); k_rescale.prove(K, K_); v_rescale.prove(V, V_);
    verifyWeightClaim(k_proj, k_layer.prove(input, K)[0]);
    verifyWeightClaim(q_proj, q_layer.prove(input, Q)[0]);
    verifyWeightClaim(v_proj, v_layer.prove(input, V)[0]);
    Q_.save_int("temp_Q.bin"); K_.save_int("temp_K.bin"); V_.save_int("temp_V.bin");
}

static void do_post_attn(const std::string& attn_input, uint seq_len, uint embed_dim, uint hidden_dim,
                          const std::string& workdir, const std::string& layer_prefix,
                          const std::string& layer_input, const std::string& attn_output,
                          const std::string& layer_output) {
    int scale_factor = 1 << 16;
    std::string post_attn = workdir + "/" + layer_prefix + "-post_attn.bin";
    std::string ffn_input = workdir + "/" + layer_prefix + "-ffn_input.bin";
    std::string ffn_output = workdir + "/" + layer_prefix + "-ffn_output.bin";

    // Self-attention proof
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
        auto out_ = rs2(out); auto out__ = rs1(out_);
        out__.save_int("temp_head_out.bin");
        rs1.prove(out_, out__); rs2.prove(out, out_);
        vector<Polynomial> proof;
        auto u1 = random_vec(ceilLog2(seq_len)); auto u2 = random_vec(ceilLog2(d));
        auto ud = random_vec(ceilLog2(seq_len));
        auto claim = out.multi_dim_me({u1, u2}, {seq_len, d});
        zkip(claim, Y_soft.partial_me(u1, seq_len, seq_len), V.partial_me(u2, d, 1), ud, proof);
        auto temp_rand = random_vec(3);
        softmax.prove(Y_soft, X_attn, shift, X_shifted, X_segments, Y_segments, m_segments,
            random_vec(ceilLog2(Y_soft.size)), random_vec(ceilLog2(Y_soft.size)),
            temp_rand[0], temp_rand[1], temp_rand[2], proof);
        auto u1_ = random_vec(ceilLog2(seq_len)); auto u2_ = random_vec(ceilLog2(seq_len));
        auto ud_ = random_vec(ceilLog2(d));
        auto claim_ = X_attn.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
        zkip(claim_, Q.partial_me(u1_, seq_len, d), K.partial_me(u2_, seq_len, d), ud_, proof);
    }
    remove("temp_Q.bin"); remove("temp_K.bin"); remove("temp_V.bin");
    remove("temp_head_Y.bin"); remove("temp_head_out.bin");

    // Skip connection 1
    cpu_skip_connection(layer_input, attn_output, post_attn);

    // RMSNorm post-attention
    compute_rms_inv(post_attn, seq_len, embed_dim, 1e-5, scale_factor, "rms_inv_temp.bin");

#ifdef USE_GOLDILOCKS
    auto rmsnorm_weight = create_weight(
        workdir + "/" + layer_prefix + "-post_attention_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-post_attention_layernorm.weight-gold-commitment.bin", 1, embed_dim);
#else
    auto rmsnorm_weight = create_weight(workdir + "/post_attention_layernorm.weight-pp.bin",
        workdir + "/" + layer_prefix + "-post_attention_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-post_attention_layernorm.weight-commitment.bin", 1, embed_dim);
#endif

    FrTensor X_post = FrTensor::from_int_bin(post_attn);
    FrTensor rms_inv_temp = FrTensor::from_int_bin("rms_inv_temp.bin");
    Rescaling rms_rs1(scale_factor), rms_rs2(scale_factor);
    zkFC g(1, embed_dim, rmsnorm_weight.weight);
    auto g_inv_rms = g(rms_inv_temp); auto g_inv_rms_ = rms_rs1(g_inv_rms);
    auto Y_rms = g_inv_rms_ * X_post; auto Y_rms_ = rms_rs2(Y_rms);
    rms_rs2.prove(Y_rms, Y_rms_);
    Y_rms_.save_int(ffn_input);
    hadamard_product_sumcheck(g_inv_rms_, X_post, random_vec(ceilLog2(Y_rms.size)), random_vec(ceilLog2(Y_rms.size)));
    rms_rs1.prove(g_inv_rms, g_inv_rms_);
    verifyWeightClaim(rmsnorm_weight, g.prove(rms_inv_temp, g_inv_rms)[0]);
    remove("rms_inv_temp.bin");

    // FFN
#ifdef USE_GOLDILOCKS
    auto up_proj = create_weight(workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-gold-commitment.bin", embed_dim, hidden_dim);
    auto gate_proj = create_weight(workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-gold-commitment.bin", embed_dim, hidden_dim);
    auto down_proj = create_weight(workdir + "/" + layer_prefix + "-mlp.down_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-gold-commitment.bin", hidden_dim, embed_dim);
#else
    auto up_proj = create_weight(workdir + "/mlp.up_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.up_proj.weight-commitment.bin", embed_dim, hidden_dim);
    auto gate_proj = create_weight(workdir + "/mlp.gate_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.gate_proj.weight-commitment.bin", embed_dim, hidden_dim);
    auto down_proj = create_weight(workdir + "/mlp.down_proj.weight-pp.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-int.bin",
        workdir + "/" + layer_prefix + "-mlp.down_proj.weight-commitment.bin", hidden_dim, embed_dim);
#endif

    zkFC up_layer(embed_dim, hidden_dim, up_proj.weight);
    zkFC gate_layer(embed_dim, hidden_dim, gate_proj.weight);
    zkFC down_layer(hidden_dim, embed_dim, down_proj.weight);
    Rescaling up_rescale(scale_factor), gate_rescale(1<<20), hidden_rescale(scale_factor), down_rescale(scale_factor);

    FrTensor swiglu_values = FrTensor::from_int_bin("swiglu-table.bin");
    tLookupRangeMapping swiglu(-(1<<21), 1<<22, swiglu_values);

    FrTensor ffn_in = FrTensor::from_int_bin(ffn_input);
    auto up_out = up_layer(ffn_in); auto up_out_ = up_rescale(up_out);
    auto gate_out = gate_layer(ffn_in); auto gate_out_ = gate_rescale(gate_out);
    auto p = swiglu(gate_out_);
    auto& swiglu_out = p.first; auto& swiglu_m = p.second;
    auto down_in = swiglu_out * up_out_; auto down_in_ = hidden_rescale(down_in);
    auto down_out = down_layer(down_in_); auto down_out_ = down_rescale(down_out);
    down_out.save_int(ffn_output);

    down_rescale.prove(down_out, down_out_);
    verifyWeightClaim(down_proj, down_layer.prove(down_in_, down_out)[0]);
    hidden_rescale.prove(down_in, down_in_);
    auto temp_rand = random_vec(3);
    auto swiglu_u = random_vec(ceilLog2(seq_len * hidden_dim));
    auto swiglu_v = random_vec(ceilLog2(seq_len * hidden_dim));
    vector<Polynomial> swiglu_proof;
    swiglu.prove(gate_out_, swiglu_out, swiglu_m, temp_rand[0], temp_rand[1], temp_rand[2], swiglu_u, swiglu_v, swiglu_proof);
    gate_rescale.prove(gate_out, gate_out_);
    verifyWeightClaim(gate_proj, gate_layer.prove(ffn_in, gate_out)[0]);
    up_rescale.prove(up_out, up_out_);
    verifyWeightClaim(up_proj, up_layer.prove(ffn_in, up_out)[0]);

    // Skip connection 2
    cpu_skip_connection(post_attn, ffn_output, layer_output);
}

int main(int argc, char* argv[]) {
    // Force CUDA initialization
    cudaFree(0);

    cerr << "CUDA initialized. Ready for commands." << endl;
    cout << "READY" << endl;
    cout.flush();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "QUIT") {
            break;
        }
        else if (cmd == "RMSNORM_LINEAR") {
            std::string layer_input, workdir, layer_prefix, attn_input_out, attn_output_out;
            uint seq_len, embed_dim;
            iss >> layer_input >> seq_len >> embed_dim >> workdir >> layer_prefix >> attn_input_out >> attn_output_out;
            do_rmsnorm_linear(layer_input, seq_len, embed_dim, workdir, layer_prefix, attn_input_out, attn_output_out);
            cout << "DONE" << endl;
            cout.flush();
        }
        else if (cmd == "POST_ATTN") {
            std::string attn_input, workdir, layer_prefix, layer_input, attn_output, layer_output;
            uint seq_len, embed_dim, hidden_dim;
            iss >> attn_input >> seq_len >> embed_dim >> hidden_dim >> workdir >> layer_prefix >> layer_input >> attn_output >> layer_output;
            do_post_attn(attn_input, seq_len, embed_dim, hidden_dim, workdir, layer_prefix, layer_input, attn_output, layer_output);
            cout << "DONE" << endl;
            cout.flush();
        }
        else {
            cerr << "Unknown command: " << cmd << endl;
            cout << "ERROR" << endl;
            cout.flush();
        }
    }
    return 0;
}
