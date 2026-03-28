#ifndef ZKLOG_CUH
#define ZKLOG_CUH

#include "tensor/fr-tensor.cuh"
#include "zknn/tlookup.cuh"
#include "proof/proof.cuh"
#include "poly/polynomial.cuh"

// Zero-knowledge -log2 computation via tLookupRangeMapping.
//
// Input:  quantized probability indices as field elements in [1, 2^precision].
//         Index p represents probability p / 2^precision.
// Output: -log2(p / 2^precision) * scale_out, stored as a field integer.
//
// The table has 2^precision entries. Entry i (input index i+1) stores:
//   round(-log2((i+1) / 2^precision) * scale_out)
//     = round((precision - log2(i+1)) * scale_out)
class zkLog {
public:
    uint precision;   // bits of input range; table size = 2^precision
    uint scale_out;   // fixed-point scale for output values

    tLookupRangeMapping lookup;

    // precision: bits of input probability (table covers indices [1, 2^precision])
    // scale_out: output fixed-point scale (e.g. 1<<16)
    zkLog(uint precision, uint scale_out);

    // Compute -log2(p) for each element of probs (as quantized indices).
    // Returns {log_probs tensor, multiplicity counts} for use in prove().
    pair<FrTensor, FrTensor> compute(const FrTensor& probs);

    // Prove the lookup. Delegates to tLookupRangeMapping::prove.
    // Caller supplies random scalars r, alpha, beta and challenge vectors u, v
    // (each of length ceilLog2(probs.size)).
    Fr_t prove(const FrTensor& probs, const FrTensor& log_probs, const FrTensor& m,
               const Fr_t& r, const Fr_t& alpha, const Fr_t& beta,
               const vector<Fr_t>& u, const vector<Fr_t>& v,
               vector<Polynomial>& proof);
};

#endif
