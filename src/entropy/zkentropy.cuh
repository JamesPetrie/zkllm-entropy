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
// Returns Claims on logits_all so the caller can chain to upstream proofs
// (e.g. zkFC on lm_head weight).
//
// Proof structure:
//   1. CDF tLookup proof  (T x V -> T x V, one proof)
//      -- implicitly proves argmax correctness (negative diffs cannot match
//         any CDF table entry in the LogUp identity)
//   2. Diffs-to-logits linking: proves diffs = v_star_broadcast - logits
//      at the CDF challenge point, yielding a Claim on logits_all
//   3. total_win row-sum proof (partial MLE + inner product sumcheck)
//   4. Actual-token extraction proof (indicator inner product)
//   5. Quotient-remainder proof for surprise computation:
//      q*tw + r = wp*2^p, with bit-decomp range proofs on q, r, (tw-r-1)
//   6. Surprise log lookup proof (tLookup on padded q tensor)
//
// Non-negativity of diffs implies argmax correctness: if any logit[i] > v_star,
// then diff[i] = v_star - logit[i] wraps to near p (~1.8e19), which cannot
// match any CDF table entry (max = 2^cdf_precision - 1).  A prover who
// inflates v_star above the true max only increases the entropy bound,
// which is against their interest.
//
// Clamping approximation: win probabilities are clamped to >= 1 to avoid
// log(0).  This affects only tokens with near-zero probability and results
// in a slight overestimate of entropy (at most log2(total_win) extra bits
// per affected position).
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
    // Appends proof polynomials and returns Claims on logits_all
    // (for chaining with upstream zkFC / rescaling proofs).
    // Returns the verified entropy value H.
    Fr_t prove(const FrTensor& logits_all, uint T, uint V,
               const vector<uint>& tokens,
               Fr_t claimed_entropy,
               vector<Polynomial>& proof,
               vector<Claim>& claims);

    // ---- Legacy per-position interface (kept for test compatibility) --------

    // Compute surprise for one position.
    Fr_t computePosition(const FrTensor& logits, uint actual_token);

    // Compute total entropy from per-position logit vectors.
    Fr_t compute(const vector<FrTensor>& logits_seq, const vector<uint>& tokens);

    // Prove from per-position logit vectors (assembles flat tensor, delegates).
    Fr_t prove(const vector<FrTensor>& logits_seq, const vector<uint>& tokens,
               Fr_t claimed_entropy, vector<Polynomial>& proof,
               vector<Claim>& claims);
};

#endif
