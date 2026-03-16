#ifndef ZKENTROPY_CUH
#define ZKENTROPY_CUH

#include "zkargmax.cuh"
#include "zknormalcdf.cuh"
#include "zklog.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "polynomial.cuh"

// Zero-knowledge conditional entropy calculator.
//
// For each output token position, proves surprise = -log2(q[actual_token])
// where q[actual] = win_prob[actual] / total_win_prob, with:
//   win_prob[i] = (1 - Phi((v_star - logits[i]) / sigma_eff)) * cdf_scale
//
// The logit tensor is an ephemeral value passed in from the caller, which is
// responsible for proving it derives from committed model weights (e.g. via
// zkFC on the final hidden state).  This class proves only the argmax and
// entropy calculation on top of whatever logits are provided.
//
// Parameters:
//   vocab_size    : number of tokens (e.g. 32000 for LLaMA)
//   bit_width     : bits for argmax diff range proof
//   cdf_precision : bits for CDF table input range [0, 2^cdf_precision)
//   log_precision : bits for log table input range [1, 2^log_precision]
//   cdf_scale     : fixed-point output scale for CDF values (e.g. 1<<16)
//   log_scale     : fixed-point output scale for log values (e.g. 1<<16)
//   sigma_eff     : Gaussian noise std dev in field integer units
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

    // Prove the entropy computation.  The logits are taken as given; the caller
    // is responsible for linking them to committed weights via zkFC.
    // Returns the sum of argmax MLE claims on the logit tensors (for chaining).
    Fr_t prove(const vector<FrTensor>& logits_seq, const vector<uint>& tokens,
               Fr_t claimed_entropy, vector<Polynomial>& proof);
};

#endif
