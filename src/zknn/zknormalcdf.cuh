#ifndef ZKNORMALCDF_CUH
#define ZKNORMALCDF_CUH

#include "tensor/fr-tensor.cuh"
#include "zknn/tlookup.cuh"
#include "proof/proof.cuh"
#include "poly/polynomial.cuh"

// Zero-knowledge standard-normal CDF computation via tLookupRangeMapping.
//
// Models hardware noise as i.i.d. Gaussian on logits with standard deviation
// sigma_eff (in field integer units, i.e. sigma_real * logit_scale).
//
// Input:  non-negative integer diffs as field elements, d in [0, 2^precision).
//         d represents (v_star - logits[i]) in field integer units.
// Output: Phi(d / sigma_eff) * scale_out, stored as a field integer.
//         This approximates the probability that token t_star beats token i
//         under the Gaussian noise model (pairwise, not exact multinomial).
//
// Table has 2^precision entries covering [0, 2^precision).
class zkNormalCDF {
public:
    uint precision;    // bits of input range; table covers [0, 2^precision)
    uint scale_out;    // fixed-point scale for output CDF values
    double sigma_eff;  // noise std dev in field integer units

    tLookupRangeMapping lookup;

    // precision: bits of input range (table size = 2^precision)
    // scale_out: output fixed-point scale (e.g. 1<<16)
    // sigma_eff: std dev in field integer units (= sigma_real * logit_scale)
    zkNormalCDF(uint precision, uint scale_out, double sigma_eff);

    // Compute Phi(d / sigma_eff) for each diff element d.
    // Returns {cdf_values tensor, multiplicity counts} for use in prove().
    pair<FrTensor, FrTensor> compute(const FrTensor& diffs);

    // Prove the lookup. Same signature convention as zkLog::prove.
    Fr_t prove(const FrTensor& diffs, const FrTensor& cdf_values, const FrTensor& m,
               const Fr_t& r, const Fr_t& alpha, const Fr_t& beta,
               const vector<Fr_t>& u, const vector<Fr_t>& v,
               vector<Polynomial>& proof);
};

#endif
