#ifndef PROOF_CUH
#define PROOF_CUH

#include "tensor/fr-tensor.cuh"
#include <cuda_fp16.h>
#include "tensor/g1-tensor.cuh"
#include "commit/commitment.cuh"
#include "field/bls12-381.cuh"
#include "poly/polynomial.cuh"

#include <vector>
#include <random>

struct Claim {
    Fr_t claim;
    std::vector<std::vector<Fr_t>> u;
    std::vector<uint> dims;
};

struct Weight;
void verifyWeightClaim(const Weight& w, const Claim& c);

// ZK opening variant of verifyWeightClaim.  Consumes `w.r` (per-row
// blindings, populated by the hiding create_weight overload) and the
// `u_generator` on `w.generator`, so it requires a pp produced by
// `Commitment::hiding_random` and loaded via `load_hiding` (v2 pp file
// with embedded H/U from hash-to-curve, see src/field/hash_to_curve.*).
// Runs prover + verifier inline (Hyrax §A.2 Figure 6 composed with
// §6.1): samples a fresh Σ-protocol challenge, calls `open_zk` to
// produce an `OpeningProof`, calls `verify_zk`, and checks the claimed
// evaluation against `c.claim`.  Throws on any mismatch.
//
// The legacy `verifyWeightClaim` stays in place for call sites that
// haven't migrated to the hiding pipeline yet (Weights produced by the
// 5-arg `create_weight` that leaves `r` empty).
void verifyWeightClaimZK(const Weight& w, const Claim& c);

KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size);

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof);

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u);

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v);

KERNEL void Fr_bin_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size);

void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);

vector<Fr_t> binary_sumcheck(const FrTensor& a, vector<Fr_t> u, vector<Fr_t> v);


bool operator==(const Fr_t& a, const Fr_t& b);
bool operator!=(const Fr_t& a, const Fr_t& b);


Fr_t multi_hadamard_sumchecks(const Fr_t& claim, const vector<FrTensor>& Xs, const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof);

#endif
