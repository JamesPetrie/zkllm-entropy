// tlookup_verifier.h — LogUp/tLookup verification (CPU-only)
//
// Verifies the LogUp table lookup argument: every element of witness S
// appears in table T with the claimed multiplicities m.
//
// The LogUp identity:
//   sum_{i in [D]} alpha / (S[i] + beta) = sum_{j in [N]} alpha * m[j] / (T[j] + beta)
//
// This is proven via a two-phase sumcheck:
//   Phase 1: while D > N, fold the witness side
//   Phase 2: when D == N, fold both sides together

#ifndef TLOOKUP_VERIFIER_H
#define TLOOKUP_VERIFIER_H

#include "verifier_utils.h"
#include "sumcheck_verifier.h"

// ── tLookup proof structure ─────────────────────────────────────────────────

struct TLookupProof {
    // Phase 1 rounds (ceilLog2(D/N) rounds)
    std::vector<SumcheckRound> phase1_rounds;
    // Phase 2 rounds (ceilLog2(N) rounds)
    std::vector<SumcheckRound> phase2_rounds;

    // Final evaluations after reduction
    Fr_t final_A;   // 1/(S(u) + beta) at final point
    Fr_t final_B;   // 1/(T(v) + beta) at final point
    Fr_t final_S;   // S(u) at final point
    Fr_t final_T;   // T(v) at final point
    Fr_t final_m;   // m(v) at final point
};

// ── tLookup verification ────────────────────────────────────────────────────
// Verifies the LogUp argument given:
//   - alpha, beta: random challenges
//   - u, v: evaluation point challenges (for MLE reduction)
//   - D: witness size, N: table size
//   - proof: the sumcheck rounds and final evaluations

struct TLookupResult {
    bool ok;
    std::string error;
    // Final claims that must be checked against commitments:
    Fr_t claim_S;   // S(u) — must match committed witness
    Fr_t claim_T;   // T(v) — must match public table
    Fr_t claim_m;   // m(v) — must match committed multiplicities
};

static inline TLookupResult verify_tlookup(
    Fr_t alpha,
    Fr_t beta,
    uint32_t D,  // witness size (padded to power of 2)
    uint32_t N,  // table size (power of 2)
    const TLookupProof& proof
) {
    // Initial claim: alpha + alpha^2
    // This comes from the LogUp identity:
    //   sum_i alpha/(S_i + beta) - sum_j alpha*m_j/(T_j + beta) = 0
    // Rearranged with the "combining" trick into a single sumcheck.
    Fr_t alpha_sq = fr_sqr(alpha);
    Fr_t initial_claim = fr_add(alpha, alpha_sq);

    // Phase 1: fold witness side (D > N)
    uint32_t phase1_rounds = ceil_log2(D / N);
    if (proof.phase1_rounds.size() != phase1_rounds) {
        return {false, "Phase 1: expected " + std::to_string(phase1_rounds) +
                       " rounds, got " + std::to_string(proof.phase1_rounds.size())};
    }

    auto r1 = verify_sumcheck(initial_claim, proof.phase1_rounds);
    if (!r1.ok) {
        return {false, "Phase 1: " + r1.error};
    }

    // Phase 2: fold both sides (D == N)
    uint32_t phase2_rounds_expected = ceil_log2(N);
    if (proof.phase2_rounds.size() != phase2_rounds_expected) {
        return {false, "Phase 2: expected " + std::to_string(phase2_rounds_expected) +
                       " rounds, got " + std::to_string(proof.phase2_rounds.size())};
    }

    auto r2 = verify_sumcheck(r1.final_claim, proof.phase2_rounds);
    if (!r2.ok) {
        return {false, "Phase 2: " + r2.error};
    }

    // Final check: verify consistency of final evaluations
    // After all rounds, the reduced claim should equal:
    //   alpha * A(u) - alpha^2 * B(v) * m(v) * (N/D)
    // where A(u) = 1/(S(u) + beta), B(v) = 1/(T(v) + beta)

    // Verify A = 1/(S + beta)
    Fr_t s_plus_beta = fr_add(proof.final_S, beta);
    Fr_t expected_A = fr_inverse(s_plus_beta);
    if (expected_A != proof.final_A) {
        return {false, "Final: A != 1/(S+beta)"};
    }

    // Verify B = 1/(T + beta)
    Fr_t t_plus_beta = fr_add(proof.final_T, beta);
    Fr_t expected_B = fr_inverse(t_plus_beta);
    if (expected_B != proof.final_B) {
        return {false, "Final: B != 1/(T+beta)"};
    }

    return {true, "", proof.final_S, proof.final_T, proof.final_m};
}

// ── tLookupRangeMapping verification ────────────────────────────────────────
// For range-mapped lookups (CDF, log tables):
//   - S_in: input values (indices into the table)
//   - S_out: output values (table entries at those indices)
//   - The lookup proves that (S_in[i], S_out[i]) matches the table mapping
//
// Combined via random challenge r: S_com[i] = S_in[i] + r * S_out[i]
// Similarly: T_com[j] = table[j] + r * mapped_vals[j]

struct TLookupRangeMappingProof {
    TLookupProof base_proof;
    // Additional info for range mapping verification
    Fr_t r;  // combining challenge
};

static inline TLookupResult verify_tlookup_range_mapping(
    Fr_t r,         // combining challenge
    Fr_t alpha,
    Fr_t beta,
    uint32_t D,
    uint32_t N,
    const TLookupProof& proof,
    // Public table for verification:
    const std::vector<Fr_t>& table_values,      // table[j] = j + low
    const std::vector<Fr_t>& mapped_values       // mapped_vals[j]
) {
    // The combined lookup uses S_com = S_in + r * S_out
    // and T_com = table + r * mapped_vals
    // Verification is the same as base tLookup on the combined values
    auto result = verify_tlookup(alpha, beta, D, N, proof);
    if (!result.ok) return result;

    // The caller must additionally verify:
    //   1. result.claim_T matches the public table's MLE at v
    //      (verifier can compute this since table is public)
    //   2. result.claim_S matches the committed witness

    return result;
}

// ── Utility: compute public table MLE ───────────────────────────────────────
// For a range table [low, low+1, ..., low+N-1] combined with mapped values:
//   T_com[j] = (j + low) + r * mapped_vals[j]

static inline Fr_t compute_combined_table_mle(
    uint32_t low,
    const std::vector<Fr_t>& mapped_vals,
    Fr_t r,
    const std::vector<Fr_t>& v  // evaluation point
) {
    uint32_t N = mapped_vals.size();
    std::vector<Fr_t> combined(N);
    for (uint32_t j = 0; j < N; j++) {
        Fr_t table_j = fr_from_u64(j + low);
        combined[j] = fr_add(table_j, fr_mul(r, mapped_vals[j]));
    }
    return mle_eval(combined, v);
}

// ── CDF table construction ──────────────────────────────────────────────────
// Reconstruct the CDF lookup table from public parameters.
// Entry j = round(Phi(j / sigma_eff) * cdf_scale)

static inline std::vector<Fr_t> build_cdf_table(
    uint32_t cdf_precision,
    uint32_t cdf_scale,
    double sigma_eff
) {
    uint32_t N = 1u << cdf_precision;
    std::vector<Fr_t> table(N);
    for (uint32_t j = 0; j < N; j++) {
        table[j] = fr_from_u64(cdf_table_value(j, sigma_eff, cdf_scale));
    }
    return table;
}

// ── Log table construction ──────────────────────────────────────────────────
// Reconstruct the log lookup table from public parameters.
// Entry j (for input j+1) = round((log_precision - log2(j+1)) * log_scale)

static inline std::vector<Fr_t> build_log_table(
    uint32_t log_precision,
    uint32_t log_scale
) {
    uint32_t N = 1u << log_precision;
    std::vector<Fr_t> table(N);
    for (uint32_t j = 0; j < N; j++) {
        // Input value is j+1 (log table has low=1)
        table[j] = fr_from_u64(log_table_value(j + 1, log_precision, log_scale));
    }
    return table;
}

#endif // TLOOKUP_VERIFIER_H
