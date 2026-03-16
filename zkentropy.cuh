#ifndef ZKENTROPY_CUH
#define ZKENTROPY_CUH

#include "zkargmax.cuh"
#include "zknormalcdf.cuh"
#include "zklog.cuh"
#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "polynomial.cuh"

// Zero-knowledge conditional entropy calculator.
//
// For each output token position, computes surprise = -log2(q[actual_token])
// where q is a conservative probability: win_prob[actual] / (vocab_size * cdf_scale),
// with win_prob = Phi(diff_actual / sigma_eff) * cdf_scale.
//
// Prove path (strong, when logit_generators provided):
//   1. zkArgmax::prove  — bit-decomp range proof that t_star is the argmax
//   2. Commitment::me_open — G1 proof that logits[actual_token] is bound to the
//      committed logit tensor; diff_actual = v_star - logits[actual_token]
//   3. CDF/Log values recorded as constant polynomials (public table, verifier re-checks)
//
// Prove path (weak, no generators):
//   Steps 1 + 3 only; logits[actual_token] is self-reported.
//
// Parameters:
//   vocab_size    : number of tokens (e.g. 32000 for LLaMA)
//   bit_width     : bits for argmax diff range proof
//   cdf_precision : bits for CDF table input range [0, 2^cdf_precision)
//   log_precision : bits for log table input range [1, 2^log_precision]
//   cdf_scale     : fixed-point output scale for CDF values (e.g. 1<<16)
//   log_scale     : fixed-point output scale for log values (e.g. 1<<16)
//   sigma_eff     : Gaussian noise std dev in field integer units (sigma_real * logit_scale)
class zkConditionalEntropy {
public:
    uint vocab_size;
    uint bit_width;
    uint cdf_precision;
    uint log_precision;
    uint cdf_scale;
    uint log_scale;
    double sigma_eff;

    zkArgmax    argmax_prover;
    zkNormalCDF cdf_prover;
    zkLog       log_prover;

    zkConditionalEntropy(
        uint vocab_size,
        uint bit_width,
        uint cdf_precision,
        uint log_precision,
        uint cdf_scale,
        uint log_scale,
        double sigma_eff
    );

    // Compute surprise (-log2 q[actual_token]) for one position, in log_scale units.
    Fr_t computePosition(const FrTensor& logits, uint actual_token);

    // Compute total conditional entropy for a sequence (sum of per-position surprises).
    Fr_t compute(const vector<FrTensor>& logits_seq, const vector<uint>& tokens);

    // Weak prove: logits[actual_token] is self-reported (no commitment).
    Fr_t prove(const vector<FrTensor>& logits_seq, const vector<uint>& tokens,
               Fr_t claimed_entropy, vector<Polynomial>& proof);

    // Strong prove: logits[actual_token] is bound to the committed logit tensor via
    // Commitment::me_open.  logit_generators must have size == 2^ceilLog2(vocab_size).
    // logit_commits[pos] is the G1TensorJacobian commitment to logits_seq[pos].
    // G1 proof elements (log2(vocab_size) triplets per position) go into g1_proof.
    Fr_t prove(const vector<FrTensor>& logits_seq, const vector<uint>& tokens,
               Fr_t claimed_entropy,
               const Commitment& logit_generators,
               const vector<G1TensorJacobian>& logit_commits,
               vector<Polynomial>& proof,
               vector<G1Jacobian_t>& g1_proof);
};

#endif
