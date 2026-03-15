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
// For each output token position, computes the "surprise" -log2(q[actual_token])
// where q is the probability distribution derived from the pairwise win-probability
// model under Gaussian hardware noise. Summing over the sequence gives an upper
// bound on covert-channel capacity.
//
// Per-position pipeline:
//   1. zkArgmax:    verify t_star = argmax(logits), get v_star = logits[t_star]
//   2. GPU kernel:  diffs[i] = v_star - logits[i]  (all >= 0)
//   3. zkNormalCDF: win_probs[i] = Phi(diffs[i] / sigma_eff) * cdf_scale
//   4. Normalize:   q[actual] = win_probs[actual] / sum(win_probs)
//                   re-quantised to index in [1, 2^log_precision]
//   5. zkLog:       surprise = -log2(q[actual]) * log_scale
//
// Parameters:
//   vocab_size    : number of tokens
//   bit_width     : bits for argmax diff range proof (diffs in [0, 2^bit_width))
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

    // Prove the entropy computation for a sequence.
    // Returns summed MLE claim on logits at per-position challenge points.
    Fr_t prove(const vector<FrTensor>& logits_seq, const vector<uint>& tokens,
               Fr_t claimed_entropy, vector<Polynomial>& proof);
};

#endif
