#ifndef ZKARGMAX_CUH
#define ZKARGMAX_CUH

#include "tensor/fr-tensor.cuh"
#include "proof/proof.cuh"
#include "poly/polynomial.cuh"

// Zero-knowledge argmax verification via bit-decomposition range proofs.
// Proves that v_star = logits[t_star] >= logits[i] for all i by showing that
// diffs[i] = v_star - logits[i] has a valid bit decomposition in [0, 2^bit_width).
class zkArgmax {
public:
    uint bit_width;

    zkArgmax(uint bit_width);

    // Find argmax index. Copies logits to host; assumes values fit in 64 bits
    // with negative numbers represented as large field elements (high val[7..2]).
    uint compute(const FrTensor& logits);

    // Prove t_star is the argmax of logits.
    //   v_star must equal logits(t_star) (caller copies this value from GPU).
    //   u: verifier challenge of length ceilLog2(logits.size).
    // Validates:
    //   - diffs = v_star - logits are all non-negative (bit decomposition exists)
    //   - reconstruction: sum_b 2^b * bits_b(u) == diffs(u)
    //   - each bits_b is binary (binary sumcheck)
    // Returns MLE claim on logits at u: v_star - diffs(u).
    Fr_t prove(const FrTensor& logits, uint t_star, Fr_t v_star,
               const vector<Fr_t>& u, vector<Polynomial>& proof);
};

#endif
