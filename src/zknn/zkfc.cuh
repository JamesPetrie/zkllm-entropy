#ifndef ZKFC_CUH
#define ZKFC_CUH

// #include <torch/torch.h>
// #include <torch/script.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "tensor/fr-tensor.cuh"
#include "proof/proof.cuh"
#include "proof/zk_sumcheck.cuh"
#include "commit/commitment.cuh"
#include "poly/polynomial.cuh"



// KERNEL void matrixMultiplyOptimized(Fr_t* A, Fr_t* B, Fr_t* C, int rowsA, int colsA, int colsB);


// This is for computing
class zkFC {
public:
    const uint inputSize, outputSize;
    bool has_bias;
    FrTensor weights, bias;
    __half* weights_fp16;              // compact fp16 storage for matmul (GPU, may be null)
    unsigned long scaling_factor;      // quantization scaling factor

    zkFC(uint input_size, uint output_size, const FrTensor& weight);
    zkFC(uint input_size, uint output_size, const FrTensor& weight,
         __half* weights_fp16, unsigned long scaling_factor);
    zkFC(uint input_size, uint output_size, const FrTensor& weight, const FrTensor& bias);
    FrTensor operator()(const FrTensor& X) const;
    // void prove(const FrTensor& X, const FrTensor& Z, Commitment& generators) const;

    // ZK variant of zkFC::prove (Phase 3 F1 closure).  Replaces the
    // previous plain-`zkip` path — which pushed degree-2 round
    // polynomials as raw coefficients — with the Hyrax §4 Protocol 3
    // ZK inner-product sumcheck (`prove_zk_inner_product`).  Per-round
    // coefficients are now Pedersen-committed; the transcript lands in
    // `zk_sumchecks` and the Σ-protocol randomness is appended to
    // `challenges` for Fiat-Shamir-style transcript continuity (same
    // pattern as src/entropy/zkentropy.cu::prove_ip_zk).
    //
    // `U` and `H` are the Pedersen bases for the inner-product
    // sumcheck (`U` = `Commitment::u_generator`, `H` =
    // `Commitment::hiding_generator`) — typically sourced from the
    // enclosing Weight's `generator`, which is openable per
    // `verifyWeightClaimZK`'s precondition.
    //
    // Returns the same weight-claim vector as before, so the call
    // pattern `verifyWeightClaimZK(w, fc.prove(..)[0])` is preserved
    // aside from the added generator + transcript arguments.
    vector<Claim> prove(const FrTensor& X, const FrTensor& Y,
                        G1Jacobian_t U, G1Jacobian_t H,
                        vector<ZKSumcheckProof>& zk_sumchecks,
                        vector<Fr_t>& challenges) const;

    static zkFC from_float_gpu_ptr (uint input_size, uint output_size, unsigned long scaling_factor, float* weight_ptr, float* bias_ptr);
    static zkFC from_float_gpu_ptr (uint input_size, uint output_size, unsigned long scaling_factor, float* weight_ptr);
    static FrTensor load_float_gpu_input(uint batch_size, uint input_size, unsigned long scaling_factor, float* input_ptr);

    // void attention(FrTensor &V, FrTensor &K, FrTensor &Q, FrTensor &out, uint rowsV, uint colsV, uint rowsK, uint colsK, uint rowsQ, uint colsQ);
};

// This is for proving
class zkFCStacked {
    public:
    bool has_bias;
    const uint num; 
    const uint batchSize;
    const uint inputSize;
    const uint outputSize;
    

    FrTensor X, W, b, Y; // num * batchSize * inputSize, num * inputSize * outputSize, num * outputSize, num * batchSize * outputSize

    zkFCStacked(bool has_bias, uint num, uint batch_size, uint input_size, uint output_size, const vector <zkFC>& layers, const vector <FrTensor>& Xs, const vector <FrTensor>& Ys);
    
    void prove(vector<Polynomial>& proof) const;

    void prove(const vector<Fr_t>& u_num, const vector<Fr_t>& v_num, const vector<Fr_t>& u_batch, const vector<Fr_t>& u_input, const vector<Fr_t>& u_output, vector<Polynomial>& proof) const;
};

// TODO: move this to somewhere else
// KERNEL void float_to_Fr_kernel(float* fs, Fr_t* frs, uint fs_num_window, uint frs_num_window, uint fs_window_size, uint frs_window_size);

// Note: `zkip` (plain inner-product sumcheck, degree-2 round
// polynomials in the clear) is kept here because it still backs the
// attention-only `zkFCStacked` / `zkip_stacked` path (src/zknn/zksoftmax.cu
// QK and Y·V matmuls, src/llm/self-attn.cu attention mode) — neither is
// part of the entropy pipeline's ZK claim graph.  Phase 3 F1 scope was
// specifically zkFC::prove, which has been migrated to
// `prove_zk_inner_product` (Hyrax §4 Protocol 3).  Migrating the
// attention path would be a separate future gap (not tracked by F1).
Fr_t zkip(const Fr_t& claim, const FrTensor& a, const FrTensor& b, const vector<Fr_t>& u, vector<Polynomial>& proof);

Fr_t zkip_stacked(const Fr_t& claim, const FrTensor& A, const FrTensor& B, const vector<Fr_t>& uN, const vector<Fr_t>& uD, const vector<Fr_t> vN, uint N, uint D, vector<Polynomial>& proof);


#endif  // ZKFC_CUH
//
