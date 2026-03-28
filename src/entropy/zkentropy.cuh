#ifndef ZKENTROPY_CUH
#define ZKENTROPY_CUH

#include "zknn/zknormalcdf.cuh"
#include "zknn/zklog.cuh"
#include "tensor/fr-tensor.cuh"
#include "proof/proof.cuh"
#include "poly/polynomial.cuh"

// Zero-knowledge conditional entropy calculator (batched).
//
// Operates on a flat T x V logit tensor (all positions at once) instead of
// looping per position.  Only the aggregate entropy H is revealed; no
// per-token scalars (diff, win_prob, total_win, surprise) leak.
//
// Proof structure:
//   1. CDF tLookup proof  (T x V -> T x V, one proof)
//      — implicitly proves non-negativity of all diffs (negative diffs
//        cannot match any table entry in the LogUp identity)
//   2. total_win row-sum proof (partial MLE + inner product sumcheck)
//   3. Actual-token extraction proof (indicator inner product)
//   4. Quotient-remainder proof for surprise computation:
//      q*tw + r = wp*2^p, with bit-decomp range proofs on q, r, (tw-r-1)
//   5. Surprise log lookup proof (tLookup on padded q tensor)
//
// Non-negativity of diffs implies argmax correctness: if any logit[i] > v_star,
// then diff[i] = v_star - logit[i] wraps to near p (~1.8e19), which cannot
// match any CDF table entry (max = 2^cdf_precision - 1).  A prover who
// inflates v_star above the true max only increases the entropy bound,
// which is against their interest.
//
// Parameters:
//   vocab_size    : number of tokens (e.g. 32000 for LLaMA)
//   cdf_precision : bits for CDF table input range [0, 2^cdf_precision)
//   log_precision : bits for log table input range [1, 2^log_precision]
//   cdf_scale     : fixed-point output scale for CDF values (e.g. 1<<16)
//   log_scale     : fixed-point output scale for log values (e.g. 1<<16)
//   sigma_eff     : Gaussian noise std dev in field integer units
class zkConditionalEntropy {
public:
    uint vocab_size;
    uint cdf_precision;
    uint log_precision;
    uint cdf_scale;
    uint log_scale;
    double sigma_eff;

    zkNormalCDF cdf_prover;
    zkLog       log_prover;

    zkConditionalEntropy(
        uint vocab_size,
        uint cdf_precision,
        uint log_precision,
        uint cdf_scale,
        uint log_scale,
        double sigma_eff
    );

    // Compute total conditional entropy from a flat T x V logit tensor.
    // tokens[t] is the actual token at position t.
    // Returns entropy in log_scale fixed-point units.
    Fr_t compute(const FrTensor& logits_all, uint T, uint V,
                 const vector<uint>& tokens);

    // Prove the entropy computation on the flat T x V tensor.
    // Returns the sum of argmax MLE claims on the logit tensor (for chaining
    // with zkFC proofs on the lm_head weight).
    Fr_t prove(const FrTensor& logits_all, uint T, uint V,
               const vector<uint>& tokens,
               Fr_t claimed_entropy,
               vector<Polynomial>& proof);

    // ---- Legacy per-position interface (kept for test compatibility) --------

    // Compute surprise for one position.
    Fr_t computePosition(const FrTensor& logits, uint actual_token);

    // Compute total entropy from per-position logit vectors.
    Fr_t compute(const vector<FrTensor>& logits_seq, const vector<uint>& tokens);

    // Prove from per-position logit vectors (assembles flat tensor, delegates).
    Fr_t prove(const vector<FrTensor>& logits_seq, const vector<uint>& tokens,
               Fr_t claimed_entropy, vector<Polynomial>& proof);
};

#endif
