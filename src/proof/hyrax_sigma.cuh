#ifndef HYRAX_SIGMA_CUH
#define HYRAX_SIGMA_CUH

#include "tensor/fr-tensor.cuh"
#include "tensor/g1-tensor.cuh"

// Hyrax §A.1 Figure 5 Σ-protocols over Pedersen commitments of the form
//
//     C  =  m · U  +  r · H                         (additive notation)
//
// where U is the message generator (Commitment::u_generator, "g" in the
// paper) and H is the hiding generator (Commitment::hiding_generator,
// "h" in the paper).  Both come from the v2 pp that Phase 1.5 ships
// (hash-to-curve derived, no known pairwise dlog).
//
// Hyrax (Wahby et al. 2018, eprint 2017/1132), §A.1 Figure 5, p. 17,
// gives three sub-protocols: proof-of-opening, proof-of-equality, and
// proof-of-product.  Phase 3 uses the first two; proof-of-product is
// deferred pending audit A3 (the sumcheck recursion is linear-in-
// commitments, so the -product sub-protocol may not be needed here —
// see docs/plans/phase-3-zk-sumcheck.md "Open question").
//
// Interactive model: the verifier's challenge `e` is caller-supplied,
// matching Phase 2's open_zk/verify_zk convention.  Phase 5 wraps this
// in Fiat-Shamir.

// ─── proof-of-opening ──────────────────────────────────────────────────
//
// Statement:   given C, prover knows (m, r) such that C = m·U + r·H.
//
// Transcript (Hyrax §A.1 Figure 5, first sub-protocol):
//   1. P → V:  A = s_m · U + s_r · H     with fresh (s_m, s_r)
//   2. V → P:  challenge e
//   3. P → V:  z_m = s_m + e·m
//              z_r = s_r + e·r
//   V checks:  z_m · U + z_r · H  =?  A + e · C
struct SigmaOpeningProof {
    G1Jacobian_t A;
    Fr_t         z_m;
    Fr_t         z_r;
};

SigmaOpeningProof prove_opening(
    G1Jacobian_t U,
    G1Jacobian_t H,
    G1Jacobian_t C,          // statement (redundant for prover, included for symmetry)
    Fr_t         m,          // witness
    Fr_t         r,          // witness
    Fr_t         e);         // verifier challenge

bool verify_opening(
    G1Jacobian_t U,
    G1Jacobian_t H,
    G1Jacobian_t C,
    const SigmaOpeningProof& proof,
    Fr_t         e);

// ─── proof-of-equality ─────────────────────────────────────────────────
//
// Statement:   given C1, C2, prover knows (m, r1, r2) such that
//                C1 = m·U + r1·H   and   C2 = m·U + r2·H
//              (i.e. the same message under two blindings).
//
// Equivalent to proof-of-opening for the delta commitment
//   C1 - C2 = (r1 - r2) · H
// with message 0 in base U.  This collapses to a Schnorr in base H
// (single message coordinate), so the transcript has one response
// rather than two.
//
// Transcript (Hyrax §A.1 Figure 5, second sub-protocol):
//   1. P → V:  A = s · H        with fresh s
//   2. V → P:  challenge e
//   3. P → V:  z = s + e · (r1 - r2)
//   V checks:  z · H  =?  A + e · (C1 - C2)
struct SigmaEqualityProof {
    G1Jacobian_t A;
    Fr_t         z;
};

SigmaEqualityProof prove_equality(
    G1Jacobian_t H,
    G1Jacobian_t C1,
    G1Jacobian_t C2,
    Fr_t         r1,
    Fr_t         r2,
    Fr_t         e);

bool verify_equality(
    G1Jacobian_t H,
    G1Jacobian_t C1,
    G1Jacobian_t C2,
    const SigmaEqualityProof& proof,
    Fr_t         e);

#endif
