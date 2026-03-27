// NTT (Number Theoretic Transform) over the Goldilocks field.
//
// The Goldilocks prime p = 2^64 - 2^32 + 1 has multiplicative group order
// p - 1 = 2^32 * (2^32 - 1), supporting NTTs up to size 2^32.
//
// Primitive root of the full multiplicative group: g = 7
// The 2^k-th root of unity is: omega_k = g^((p-1) / 2^k) mod p
//
// This module provides forward NTT, inverse NTT, and coset variants
// needed by the FRI polynomial commitment scheme.

#ifndef NTT_CUH
#define NTT_CUH

#include "fr-tensor.cuh"

#include <vector>
#include <stdexcept>

// ── Root of unity computation ────────────────────────────────────────────────

// Compute the principal 2^log_n-th root of unity in the Goldilocks field.
// Uses generator g=7: omega = 7^((p-1)/2^log_n) mod p.
Fr_t get_root_of_unity(uint log_n);

// ── NTT interface ────────────────────────────────────────────────────────────

// Forward NTT in-place on GPU data.
// data: device pointer to 2^log_n Fr_t elements
// log_n: log2 of the transform size
void ntt_forward(Fr_t* data, uint log_n);

// Inverse NTT in-place on GPU data.
// Applies forward NTT with inverse root, then scales by 1/n.
void ntt_inverse(Fr_t* data, uint log_n);

// Forward NTT on a coset: multiply by powers of shift, then NTT.
// shift: the coset generator (typically a root of unity of a larger domain)
void ntt_coset_forward(Fr_t* data, uint log_n, Fr_t shift);

// Inverse coset NTT: INTT then divide by powers of shift.
void ntt_coset_inverse(Fr_t* data, uint log_n, Fr_t shift);

#endif // NTT_CUH
