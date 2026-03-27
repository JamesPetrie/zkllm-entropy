#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#ifndef USE_GOLDILOCKS
#include "commitment.cuh"
#endif
#include "rescaling.cuh"
#include <string>

int main(int argc, char *argv[])
{
    string which = argv[1];
    string input_file_name = argv[2];
    uint seq_len = std::stoi(argv[3]);
    uint embed_dim = std::stoi(argv[4]);
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];

#ifdef USE_GOLDILOCKS
    auto rmsnorm_weight = create_weight(
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-gold-commitment.bin",
        1, embed_dim, 1UL << 16
    );
#else
    auto rmsnorm_weight = create_weight(
        workdir + "/" + which + "_layernorm.weight-pp.bin",
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-commitment.bin",
        1, embed_dim
    );
#endif

    FrTensor X = FrTensor::from_int_bin(input_file_name);
    FrTensor rms_inv_temp = FrTensor::from_int_bin("rms_inv_temp.bin");

    // create an all 1 tensor with size embed_dim * embed_dim
    FrTensor all_one(seq_len);
    all_one *= FR_FROM_INT(0);
    all_one += FR_FROM_INT(1);

    Rescaling rs1(1 << 16), rs2(1 << 16);

    zkFC g = zkFC(1, embed_dim, rmsnorm_weight.weight, rmsnorm_weight.weight_fp16, rmsnorm_weight.scaling_factor);
    auto g_inv_rms = g(rms_inv_temp);
    auto g_inv_rms_ = rs1(g_inv_rms);

    auto Y = g_inv_rms_ * X;
    auto Y_ = rs2(Y);
    auto v0 = ceilLog2(seq_len);
    auto v1 = ceilLog2(embed_dim);

    rs2.prove(Y, Y_);
    Y_.save_int(output_file_name);
    hadamard_product_sumcheck(g_inv_rms_, X, random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)));
    rs1.prove(g_inv_rms, g_inv_rms_);
    verifyWeightClaim(rmsnorm_weight, g.prove(rms_inv_temp, g_inv_rms)[0]);
    return 0;
    
}