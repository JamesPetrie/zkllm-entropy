#ifndef HASH_TO_CURVE_CUH
#define HASH_TO_CURVE_CUH

#include "tensor/g1-tensor.cuh"
#include <string>
#include <vector>
#include <cstdint>

// Phase 1.5: deterministic derivation of Pedersen generators via
// RFC 9380 hash-to-curve, so no party (in particular, whoever ran
// ppgen) knows any pairwise discrete log.
//
// RFC 9380 §3.1 (Faz-Hernández et al. 2023, p. 10):
//   "Domain separation tag (DST), a parameter to a hash function that
//    aims to keep the outputs of that function separate from those
//    produced using a different DST."
//
// Suite: BLS12381G1_XMD:SHA-256_SSWU_RO_  (RFC 9380 §8.8.1 / §8.8.2).
//
// ANY change to the generator derivation — including a wrapper bugfix —
// MUST bump this DST string (e.g., to _V2), because pp files keyed by
// DST v1 will otherwise silently diverge across the fix boundary and
// verify_pp will fire with no diagnostic context.
constexpr const char* ZKLLM_ENTROPY_PEDERSEN_DST_V1 =
    "ZKLLM-ENTROPY-PEDERSEN-V1_BLS12381G1_XMD:SHA-256_SSWU_RO_";

// Hash-to-curve onto BLS12-381 G1 via blst.  Host-side only; no GPU
// kernel calls into this.  Output is in codebase Jacobian coordinates
// with Montgomery-form limbs, directly usable as a Commitment generator.
G1Jacobian_t hash_to_curve_g1(const std::string& dst,
                              const std::vector<uint8_t>& msg);

// Convenience overload: treats `msg` as a byte string.
G1Jacobian_t hash_to_curve_g1(const std::string& dst,
                              const std::string& msg);

// Message encoding for the i-th vector generator:
//   "G_" || uint32_be(i)
// Big-endian 4-byte index so the byte encoding matches RFC 9380's
// convention for structured inputs and stays portable across hosts.
std::vector<uint8_t> msg_for_G_index(uint32_t i);

// Message encodings for the scalar generators H and U.
std::vector<uint8_t> msg_for_H();
std::vector<uint8_t> msg_for_U();

// Derive all Pedersen generators for a pp of the given size.
//
// Returns {G_0, G_1, ..., G_{size-1}} in `G`, and sets `H`, `U`.  All
// three are deterministic functions of (DST, size) alone — no RNG.
struct HashedGenerators {
    std::vector<G1Jacobian_t> G;
    G1Jacobian_t H;
    G1Jacobian_t U;
};
HashedGenerators hash_to_curve_generators(const std::string& dst, uint32_t size);

#endif
