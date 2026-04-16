#include "zknn/zksoftmax.cuh"
#include "zknn/zkfc.cuh"
#include "tensor/fr-tensor.cuh"
#include "proof/proof.cuh"
#include "proof/zk_sumcheck.cuh"
#include "commit/commitment.cuh"
#include "zknn/rescaling.cuh"
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

    auto rmsnorm_weight = create_weight(
        workdir + "/" + which + "_layernorm.weight-pp.bin",
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-int.bin",
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-commitment.bin",
        workdir + "/" + layer_prefix + "-" + which + "_layernorm.weight-commitment.bin.r",
        1, embed_dim
    );

    FrTensor X = FrTensor::from_int_bin(input_file_name);
    FrTensor rms_inv_temp = FrTensor::from_int_bin("rms_inv_temp.bin");

    // create an all 1 tensor with size embed_dim * embed_dim
    FrTensor all_one(seq_len);
    all_one *= FR_FROM_INT(0);
    all_one += FR_FROM_INT(1);

    Rescaling rs1(1 << 16), rs2(1 << 16);

    zkFC g = zkFC(1, embed_dim, rmsnorm_weight.weight);
    auto g_inv_rms = g(rms_inv_temp);
    auto g_inv_rms_ = rs1(g_inv_rms);

    auto Y = g_inv_rms_ * X;
    auto Y_ = rs2(Y);
    auto v0 = ceilLog2(seq_len);
    auto v1 = ceilLog2(embed_dim);

    // Throwaway hiding Pedersen pp for ZK rescaling lookups.
    Commitment rm_pp = Commitment::hiding_random(1);
    vector<ZKSumcheckProof> rm_zk_sumchecks;
    vector<Fr_t> rm_challenges;
    rs2.prove(Y, Y_, rm_pp, rm_zk_sumchecks);
    Y_.save_int(output_file_name);
    {
        uint n_hp = ceilLog2(Y.size);
        auto u_hp = random_vec(n_hp);
        auto v_hp = random_vec(n_hp);
        auto sg_hp = random_vec(n_hp * 4);
        FrTensor h_hp = g_inv_rms_ * X;
        Fr_t S_hp = h_hp(u_hp);
        Fr_t fa, fb;
        ZKSumcheckProverHandoff handoff_hp;
        (void)prove_zk_hadamard_product(
            rmsnorm_weight.generator.u_generator, rmsnorm_weight.generator.hiding_generator,
            S_hp, g_inv_rms_, X,
            u_hp, v_hp, sg_hp,
            fa, fb, handoff_hp);
    }
    rs1.prove(g_inv_rms, g_inv_rms_, rm_pp, rm_zk_sumchecks);
    verifyWeightClaimZK(rmsnorm_weight, g.prove(
        rms_inv_temp, g_inv_rms,
        rmsnorm_weight.generator.u_generator, rmsnorm_weight.generator.hiding_generator,
        rm_zk_sumchecks, rm_challenges)[0]);
    return 0;
    
}