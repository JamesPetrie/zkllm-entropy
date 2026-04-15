#include "proof/hyrax_sigma.cuh"
#include "commit/commitment.cuh"

// Host-side wrappers for single-point G1 and single-scalar Fr arithmetic.
// Same pattern as src/commit/commitment.cu: bounce through size-1 tensor
// wrappers so we reuse the device kernels without a separate host
// implementation of BLS12-381 group / scalar ops.

static Fr_t fr_mul_host(Fr_t a, Fr_t b) {
    FrTensor A(1, &a);
    FrTensor B(1, &b);
    FrTensor C = A * B;
    return C(0);
}

static Fr_t fr_add_host(Fr_t a, Fr_t b) {
    FrTensor A(1, &a);
    FrTensor B(1, &b);
    FrTensor C = A + B;
    return C(0);
}

static G1Jacobian_t g1_scalar_mul_host(G1Jacobian_t P, Fr_t s) {
    Commitment pp1(1, P);
    FrTensor S(1, &s);
    G1TensorJacobian out = pp1.commit(S);
    return out(0);
}

static G1Jacobian_t g1_add_host(G1Jacobian_t a, G1Jacobian_t b) {
    G1TensorJacobian A(1, a);
    G1TensorJacobian B(1, b);
    G1TensorJacobian C = A + B;
    return C(0);
}

static G1Jacobian_t g1_sub_host(G1Jacobian_t a, G1Jacobian_t b) {
    G1TensorJacobian A(1, a);
    G1TensorJacobian B(1, b);
    G1TensorJacobian C = A - B;
    return C(0);
}

// G1 points computed via different Jacobian paths can share the same
// affine value but differ in (X:Y:Z) limbs.  Compare via subtraction
// and test whether the difference is the identity (Z == 0).  Matches
// the convention in Commitment::verify_zk.
static bool g1_eq_host(G1Jacobian_t a, G1Jacobian_t b) {
    G1Jacobian_t d = g1_sub_host(a, b);
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (d.z.val[i] != 0) return false;
    }
    return true;
}

// ─── proof-of-opening ──────────────────────────────────────────────────

SigmaOpeningProof prove_opening(
    G1Jacobian_t U,
    G1Jacobian_t H,
    G1Jacobian_t /*C*/,
    Fr_t         m,
    Fr_t         r,
    Fr_t         e)
{
    // Fresh masks (Figure 5 first sub-protocol, step 1).  Drawn
    // per-call; any reuse across invocations for the same statement
    // leaks the witness under a rewinding extractor.
    FrTensor s = FrTensor::random(2);
    Fr_t s_m = s(0);
    Fr_t s_r = s(1);

    // A = s_m · U + s_r · H
    G1Jacobian_t A = g1_add_host(g1_scalar_mul_host(U, s_m),
                                 g1_scalar_mul_host(H, s_r));

    // Responses (Figure 5 step 3): z_m = s_m + e·m, z_r = s_r + e·r.
    Fr_t z_m = fr_add_host(s_m, fr_mul_host(e, m));
    Fr_t z_r = fr_add_host(s_r, fr_mul_host(e, r));

    return {A, z_m, z_r};
}

bool verify_opening(
    G1Jacobian_t U,
    G1Jacobian_t H,
    G1Jacobian_t C,
    const SigmaOpeningProof& proof,
    Fr_t         e)
{
    // LHS: z_m · U + z_r · H
    G1Jacobian_t lhs = g1_add_host(g1_scalar_mul_host(U, proof.z_m),
                                   g1_scalar_mul_host(H, proof.z_r));
    // RHS: A + e · C
    G1Jacobian_t rhs = g1_add_host(proof.A, g1_scalar_mul_host(C, e));
    return g1_eq_host(lhs, rhs);
}

// ─── proof-of-equality ─────────────────────────────────────────────────

SigmaEqualityProof prove_equality(
    G1Jacobian_t H,
    G1Jacobian_t /*C1*/,
    G1Jacobian_t /*C2*/,
    Fr_t         r1,
    Fr_t         r2,
    Fr_t         e)
{
    // Fresh mask for the base-H Schnorr on (C1 - C2).
    FrTensor s_t = FrTensor::random(1);
    Fr_t s = s_t(0);

    // A = s · H
    G1Jacobian_t A = g1_scalar_mul_host(H, s);

    // Compute r_diff = r1 - r2 via additive-inverse identity in Fr.
    // Fr_t has no host-side subtraction helper; (r1 - r2) mod p =
    // r1 + (p - r2), but simplest is to bounce through FrTensor::-.
    FrTensor R1(1, &r1);
    FrTensor R2(1, &r2);
    FrTensor Rdiff = R1 - R2;
    Fr_t r_diff = Rdiff(0);

    // z = s + e · (r1 - r2)
    Fr_t z = fr_add_host(s, fr_mul_host(e, r_diff));

    return {A, z};
}

bool verify_equality(
    G1Jacobian_t H,
    G1Jacobian_t C1,
    G1Jacobian_t C2,
    const SigmaEqualityProof& proof,
    Fr_t         e)
{
    // LHS: z · H
    G1Jacobian_t lhs = g1_scalar_mul_host(H, proof.z);
    // RHS: A + e · (C1 - C2)
    G1Jacobian_t diff = g1_sub_host(C1, C2);
    G1Jacobian_t rhs = g1_add_host(proof.A, g1_scalar_mul_host(diff, e));
    return g1_eq_host(lhs, rhs);
}
