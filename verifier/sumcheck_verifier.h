// sumcheck_verifier.h — Sumcheck protocol verification (CPU-only)
//
// Verifies inner product, Hadamard product, and binary sumcheck proofs.
// Each verifier walks through the prover's round polynomials, checking
// that p_i(0) + p_i(1) == claim at each round, then reducing via
// p_i(challenge_i).

#ifndef SUMCHECK_VERIFIER_H
#define SUMCHECK_VERIFIER_H

#include "verifier_utils.h"
#include <array>
#include <string>

// ── Result type ─────────────────────────────────────────────────────────────

struct SumcheckResult {
    bool ok;
    Fr_t final_claim;   // reduced claim after all rounds
    std::string error;  // error message if !ok
};

// ── Generic sumcheck verifier ───────────────────────────────────────────────
// Verifies a sumcheck proof where each round provides a degree-d univariate
// polynomial p_i such that p_i(0) + p_i(1) == current_claim.
//
// For degree-1 (linear) sumchecks: each round has 2 values [p(0), p(1)]
// For degree-2 (quadratic) sumchecks: each round has 3 values [p(0), p(1), p(challenge)]
//
// The prover's Fr_ip_sc / Fr_hp_sc / Fr_bin_sc all produce 3 values per round
// in the proof vector. The third value is p_i evaluated at the round's challenge.

struct SumcheckRound {
    Fr_t p0;         // p_i(0)
    Fr_t p1;         // p_i(1)
    Fr_t p_alpha;    // p_i(challenge_i) — the reduced claim for next round
};

// Verify a sequence of sumcheck rounds.
// Returns the final reduced claim (after the last round's challenge evaluation).
// Caller must check this against committed evaluations.
static inline SumcheckResult verify_sumcheck(
    Fr_t initial_claim,
    const std::vector<SumcheckRound>& rounds
) {
    Fr_t current_claim = initial_claim;

    for (size_t i = 0; i < rounds.size(); i++) {
        Fr_t sum = fr_add(rounds[i].p0, rounds[i].p1);
        if (sum != current_claim) {
            return {false, FR_ZERO,
                    "Sumcheck round " + std::to_string(i) + ": p(0)+p(1)=" +
                    std::to_string(sum.val) + " != claim=" +
                    std::to_string(current_claim.val)};
        }
        current_claim = rounds[i].p_alpha;
    }

    return {true, current_claim, ""};
}

// ── Inner product sumcheck verifier ─────────────────────────────────────────
// Verifies: <a, b> = claim
//
// The prover's Fr_ip_sc produces, per round:
//   3 Fr_t values: [eval_at_0, eval_at_1, eval_at_challenge]
// After all rounds: 2 Fr_t values: [a(u), b(u)]
//
// The verifier checks:
//   1. Each round: p(0) + p(1) == current_claim
//   2. After reduction: final_claim == a(u) * b(u)

struct IpSumcheckProof {
    std::vector<SumcheckRound> rounds;
    Fr_t final_a;  // a(u) — MLE of a at challenge point
    Fr_t final_b;  // b(u) — MLE of b at challenge point
};

// Parse an inner product sumcheck proof from a flat Fr_t vector
// (as produced by Fr_ip_sc in proof.cu)
static inline IpSumcheckProof parse_ip_sumcheck(
    const std::vector<Fr_t>& proof_data,
    size_t offset,
    uint32_t num_rounds
) {
    IpSumcheckProof p;
    p.rounds.resize(num_rounds);
    size_t idx = offset;
    for (uint32_t i = 0; i < num_rounds; i++) {
        p.rounds[i].p0      = proof_data.at(idx++);
        p.rounds[i].p1      = proof_data.at(idx++);
        p.rounds[i].p_alpha = proof_data.at(idx++);
    }
    p.final_a = proof_data.at(idx++);
    p.final_b = proof_data.at(idx++);
    return p;
}

static inline SumcheckResult verify_ip_sumcheck(
    Fr_t claim,
    const IpSumcheckProof& proof
) {
    auto result = verify_sumcheck(claim, proof.rounds);
    if (!result.ok) return result;

    // Final check: reduced claim should equal a(u) * b(u)
    Fr_t expected = fr_mul(proof.final_a, proof.final_b);
    if (result.final_claim != expected) {
        return {false, result.final_claim,
                "IP sumcheck final: claim=" + std::to_string(result.final_claim.val) +
                " != a(u)*b(u)=" + std::to_string(expected.val)};
    }

    return {true, result.final_claim, ""};
}

// ── Hadamard product sumcheck verifier ──────────────────────────────────────
// Verifies: sum_i a[i] * b[i] * eq(i, v) = claim
//
// The prover's Fr_hp_sc produces per round:
//   3 Fr_t values: [out0(u'), out1(u'), out2(u')]
// After all rounds: 2 Fr_t values: [a(u), b(u)]
//
// After reduction: final_claim == a(u) * b(u) * eq(u, v)

struct HpSumcheckProof {
    std::vector<SumcheckRound> rounds;
    Fr_t final_a;
    Fr_t final_b;
};

static inline HpSumcheckProof parse_hp_sumcheck(
    const std::vector<Fr_t>& proof_data,
    size_t offset,
    uint32_t num_rounds
) {
    HpSumcheckProof p;
    p.rounds.resize(num_rounds);
    size_t idx = offset;
    for (uint32_t i = 0; i < num_rounds; i++) {
        p.rounds[i].p0      = proof_data.at(idx++);
        p.rounds[i].p1      = proof_data.at(idx++);
        p.rounds[i].p_alpha = proof_data.at(idx++);
    }
    p.final_a = proof_data.at(idx++);
    p.final_b = proof_data.at(idx++);
    return p;
}

static inline SumcheckResult verify_hp_sumcheck(
    Fr_t claim,
    const HpSumcheckProof& proof,
    const std::vector<Fr_t>& u,
    const std::vector<Fr_t>& v
) {
    auto result = verify_sumcheck(claim, proof.rounds);
    if (!result.ok) return result;

    // Final check: reduced claim should equal a(u) * b(u) * eq(u, v)
    Fr_t eq = eq_eval(u, v);
    Fr_t expected = fr_mul(fr_mul(proof.final_a, proof.final_b), eq);
    if (result.final_claim != expected) {
        return {false, result.final_claim,
                "HP sumcheck final: claim=" + std::to_string(result.final_claim.val) +
                " != a*b*eq=" + std::to_string(expected.val)};
    }

    return {true, result.final_claim, ""};
}

// ── Binary sumcheck verifier ────────────────────────────────────────────────
// Verifies: sum_i a[i] * (a[i] - 1) * eq(i, v) = 0
// (proves all a[i] are in {0, 1})
//
// After reduction: final_claim == a(u) * (a(u) - 1) * eq(u, v)

struct BinSumcheckProof {
    std::vector<SumcheckRound> rounds;
    Fr_t final_a;
};

static inline BinSumcheckProof parse_bin_sumcheck(
    const std::vector<Fr_t>& proof_data,
    size_t offset,
    uint32_t num_rounds
) {
    BinSumcheckProof p;
    p.rounds.resize(num_rounds);
    size_t idx = offset;
    for (uint32_t i = 0; i < num_rounds; i++) {
        p.rounds[i].p0      = proof_data.at(idx++);
        p.rounds[i].p1      = proof_data.at(idx++);
        p.rounds[i].p_alpha = proof_data.at(idx++);
    }
    p.final_a = proof_data.at(idx++);
    return p;
}

static inline SumcheckResult verify_bin_sumcheck(
    const BinSumcheckProof& proof,
    const std::vector<Fr_t>& u,
    const std::vector<Fr_t>& v
) {
    // Initial claim is 0 (all elements should be binary)
    auto result = verify_sumcheck(FR_ZERO, proof.rounds);
    if (!result.ok) return result;

    // Final check: claim == a(u) * (a(u) - 1) * eq(u, v)
    Fr_t a = proof.final_a;
    Fr_t a_minus_1 = fr_sub(a, FR_ONE);
    Fr_t eq = eq_eval(u, v);
    Fr_t expected = fr_mul(fr_mul(a, a_minus_1), eq);
    if (result.final_claim != expected) {
        return {false, result.final_claim,
                "Binary sumcheck final: claim=" + std::to_string(result.final_claim.val) +
                " != a*(a-1)*eq=" + std::to_string(expected.val)};
    }

    return {true, result.final_claim, ""};
}

// ── Multi-Hadamard sumcheck verifier ────────────────────────────────────────
// Verifies: sum_i prod_k X_k[i] * eq(i, v) = claim
// Used by multi_hadamard_sumchecks in proof.cu
//
// Each round produces a Polynomial (variable number of coefficients).
// The verifier checks p(0) + p(1) == claim and reduces via p(challenge).

struct MultiHadSumcheckProof {
    std::vector<Polynomial> round_polys;  // one polynomial per round
};

static inline SumcheckResult verify_multi_had_sumcheck(
    Fr_t claim,
    const MultiHadSumcheckProof& proof,
    const std::vector<Fr_t>& challenges  // one challenge per round
) {
    Fr_t current_claim = claim;

    for (size_t i = 0; i < proof.round_polys.size(); i++) {
        const auto& p = proof.round_polys[i];
        Fr_t p0 = p.eval(FR_ZERO);
        Fr_t p1 = p.eval(FR_ONE);
        Fr_t sum = fr_add(p0, p1);
        if (sum != current_claim) {
            return {false, FR_ZERO,
                    "Multi-Had round " + std::to_string(i) + ": p(0)+p(1)=" +
                    std::to_string(sum.val) + " != claim=" +
                    std::to_string(current_claim.val)};
        }
        if (i < challenges.size()) {
            current_claim = p.eval(challenges[i]);
        }
    }

    return {true, current_claim, ""};
}

// ── Lagrange interpolation from evaluations at {0,1,...,d} ──────────────────
// CPU-only version for the verifier (mirrors Polynomial::from_evaluations).

static inline Polynomial lagrange_from_evals(const std::vector<Fr_t>& evals) {
    if (evals.empty()) return Polynomial{{}};
    uint32_t d = evals.size() - 1;

    std::vector<Fr_t> result(d + 1, FR_ZERO);

    for (uint32_t j = 0; j <= d; j++) {
        // Denominator: prod_{k!=j} (j - k)
        Fr_t denom = FR_ONE;
        for (uint32_t k = 0; k <= d; k++) {
            if (k == j) continue;
            Fr_t diff;
            if (j > k) {
                diff = fr_from_u64(j - k);
            } else {
                diff = fr_neg(fr_from_u64(k - j));
            }
            denom = fr_mul(denom, diff);
        }
        Fr_t inv_denom = fr_inverse(denom);
        Fr_t scale = fr_mul(evals[j], inv_denom);

        // Expand prod_{k!=j} (X - k) into coefficients
        std::vector<Fr_t> basis(1, FR_ONE);
        for (uint32_t k = 0; k <= d; k++) {
            if (k == j) continue;
            Fr_t neg_k = (k == 0) ? FR_ZERO : fr_neg(fr_from_u64(k));
            std::vector<Fr_t> new_basis(basis.size() + 1, FR_ZERO);
            for (size_t m = 0; m < basis.size(); m++) {
                new_basis[m + 1] = fr_add(new_basis[m + 1], basis[m]);
                new_basis[m] = fr_add(new_basis[m], fr_mul(basis[m], neg_k));
            }
            basis = new_basis;
        }

        // Accumulate
        for (size_t m = 0; m < basis.size() && m <= d; m++) {
            result[m] = fr_add(result[m], fr_mul(scale, basis[m]));
        }
    }

    return Polynomial{result};
}

// ── ZK inner product sumcheck verifier ──────────────────────────────────────
// Verifies: <a, b> = T with vanishing polynomial masking + XZZ+19 transcript
// masking. The prover sends:
//   - T (honest claim)
//   - P_sum (transcript mask sum over {0,1}^b)
//   - rho (verifier challenge for transcript masking)
//   - b round polynomials, each degree 4 (5 evaluations)
//   - final_za, final_zb (masked evaluations at terminal point)
//   - final_p (transcript mask at terminal point)
//
// Verification:
//   1. combined_claim = T + rho * P_sum
//   2. Each round: p(0) + p(1) == current_claim, then claim = p(alpha_j)
//   3. Final: final_za * final_zb + rho * final_p == last reduced claim

struct ZkIpSumcheckProof {
    Fr_t T;             // honest claim (inner product)
    Fr_t P_sum;         // transcript mask polynomial sum
    Fr_t rho;           // verifier challenge
    std::vector<Polynomial> round_polys;  // degree-4 (5 coefficients each)
    std::vector<Fr_t> challenges;         // alpha_j for each round
    Fr_t final_za;      // Z_a(s*) — masked a evaluation
    Fr_t final_zb;      // Z_b(s*) — masked b evaluation
    Fr_t final_p;       // p(s*) — transcript mask at terminal point
};

// Parse a ZK IP sumcheck proof from a flat evaluation vector.
// Format: [5 evals per round] x num_rounds + [final_za, final_zb, final_p]
// The evals are at points {0, 1, 2, 3, 4}.
static inline ZkIpSumcheckProof parse_zk_ip_sumcheck(
    const std::vector<Fr_t>& proof_data,
    size_t offset,
    uint32_t num_rounds,
    Fr_t T,
    Fr_t P_sum,
    Fr_t rho,
    const std::vector<Fr_t>& challenges
) {
    ZkIpSumcheckProof p;
    p.T = T;
    p.P_sum = P_sum;
    p.rho = rho;
    p.challenges = challenges;

    size_t idx = offset;
    p.round_polys.resize(num_rounds);
    for (uint32_t i = 0; i < num_rounds; i++) {
        // Read 5 evaluations at t = 0, 1, 2, 3, 4
        std::vector<Fr_t> evals(5);
        for (uint32_t t = 0; t < 5; t++) {
            evals[t] = proof_data.at(idx++);
        }
        // Convert evaluations to coefficient form via Lagrange interpolation
        // For degree 4: p(X) interpolated from {(0,e0),(1,e1),(2,e2),(3,e3),(4,e4)}
        p.round_polys[i] = lagrange_from_evals(evals);
    }
    p.final_za = proof_data.at(idx++);
    p.final_zb = proof_data.at(idx++);
    p.final_p  = proof_data.at(idx++);
    return p;
}

static inline SumcheckResult verify_zk_ip_sumcheck(
    const ZkIpSumcheckProof& proof
) {
    // Step 1: Combined claim
    Fr_t current_claim = fr_add(proof.T, fr_mul(proof.rho, proof.P_sum));

    // Step 2: Verify each round
    for (size_t i = 0; i < proof.round_polys.size(); i++) {
        const auto& p = proof.round_polys[i];
        Fr_t p0 = p.eval(FR_ZERO);
        Fr_t p1 = p.eval(FR_ONE);
        Fr_t sum = fr_add(p0, p1);
        if (sum != current_claim) {
            return {false, FR_ZERO,
                    "ZK-IP round " + std::to_string(i) + ": p(0)+p(1)=" +
                    std::to_string(sum.val) + " != claim=" +
                    std::to_string(current_claim.val)};
        }
        if (i < proof.challenges.size()) {
            current_claim = p.eval(proof.challenges[i]);
        }
    }

    // Step 3: Final check — za * zb + rho * p(s*) == last reduced claim
    Fr_t expected = fr_add(fr_mul(proof.final_za, proof.final_zb),
                           fr_mul(proof.rho, proof.final_p));
    if (expected != current_claim) {
        return {false, current_claim,
                "ZK-IP final: za*zb+rho*p=" + std::to_string(expected.val) +
                " != claim=" + std::to_string(current_claim.val)};
    }

    return {true, current_claim, ""};
}

#endif // SUMCHECK_VERIFIER_H
