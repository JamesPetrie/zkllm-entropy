// verifier.cpp — CPU-only verifier for zkllm-entropy proof files
//
// Reads proof files in the current v2 format and performs all arithmetic
// verification checks. This is a C++ port of verify_entropy.py with
// infrastructure for future cryptographic verification (sumcheck, tLookup,
// Merkle openings) once the prover serializes those proofs.
//
// Build: g++ -std=c++17 -O2 -o verifier verifier.cpp -lm
// Usage: ./verifier <proof_file> [--verbose]

#include "verifier_utils.h"
#include "sumcheck_verifier.h"
#include "tlookup_verifier.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Verification of a single position ───────────────────────────────────────

struct PosResult {
    bool ok;
    std::vector<std::string> errors;
    uint64_t surprise;
};

static PosResult verify_position(
    uint32_t pos,
    const PositionProof& pp,
    const ProofHeader& hdr
) {
    PosResult r;
    r.ok = true;
    r.surprise = 0;

    uint64_t diff_actual = pp.diff_actual.val;
    uint64_t win_prob    = pp.win_prob.val;
    uint64_t total_win   = pp.total_win.val;
    uint64_t q_fr        = pp.q_fr.val;
    uint64_t surprise    = pp.surprise.val;

    uint32_t cdf_len = 1u << hdr.cdf_precision;
    uint32_t log_len = 1u << hdr.log_precision;

    // Check 0: argmax proof — ind_sum must be 1, ind_dot must be 0
    if (pp.ind_sum.val != 1) {
        r.errors.push_back("argmax: indicator sum=" + std::to_string(pp.ind_sum.val) + ", expected 1");
        r.ok = false;
    }
    if (pp.ind_dot.val != 0) {
        r.errors.push_back("argmax: <ind, diffs>=" + std::to_string(pp.ind_dot.val) + ", expected 0");
        r.ok = false;
    }

    // Check 1: win_prob == cdf_scale - cdf_table[diff_actual]
    // Check for "negative" diff (field element > p/2) which would be invalid
    if (fr_is_negative(pp.diff_actual)) {
        r.errors.push_back("diff_actual is negative (field element > p/2): " +
                           std::to_string(diff_actual));
        r.ok = false;
    } else {
        uint64_t d_clamped = std::min(diff_actual, (uint64_t)(cdf_len - 1));
        uint64_t cdf_val = cdf_table_value(d_clamped, hdr.sigma_eff, hdr.cdf_scale);
        int64_t expected_win = (int64_t)hdr.cdf_scale - (int64_t)cdf_val;
        if (expected_win < 0) expected_win = 0;
        if (win_prob != (uint64_t)expected_win) {
            r.errors.push_back("win_prob mismatch: got " + std::to_string(win_prob) +
                               ", expected " + std::to_string(expected_win) +
                               " (diff=" + std::to_string(diff_actual) +
                               ", cdf_val=" + std::to_string(cdf_val) + ")");
            r.ok = false;
        }
    }

    // Check 1b: total_win bounds
    uint64_t max_total_win = (uint64_t)hdr.vocab_size * hdr.cdf_scale;
    if (total_win < win_prob) {
        r.errors.push_back("total_win (" + std::to_string(total_win) +
                           ") < win_prob (" + std::to_string(win_prob) + ")");
        r.ok = false;
    }
    if (total_win > max_total_win) {
        r.errors.push_back("total_win (" + std::to_string(total_win) +
                           ") > vocab_size*cdf_scale (" + std::to_string(max_total_win) + ")");
        r.ok = false;
    }

    // Check 2: q_fr == clamp(floor(win_prob * 2^log_precision / total_win), 1, 2^log_precision)
    uint64_t expected_q;
    if (total_win == 0) {
        expected_q = 1;
    } else {
        expected_q = (win_prob * log_len) / total_win;
        if (expected_q < 1) expected_q = 1;
        if (expected_q > log_len) expected_q = log_len;
    }
    if (q_fr != expected_q) {
        r.errors.push_back("q_fr mismatch: got " + std::to_string(q_fr) +
                           ", expected " + std::to_string(expected_q) +
                           " (win=" + std::to_string(win_prob) +
                           ", total=" + std::to_string(total_win) + ")");
        r.ok = false;
    }

    // Check 3: surprise == log_table[q_fr]
    if (q_fr < 1 || q_fr > log_len) {
        r.errors.push_back("q_fr=" + std::to_string(q_fr) +
                           " out of range [1, " + std::to_string(log_len) + "]");
        r.ok = false;
    } else {
        uint64_t expected_surprise = log_table_value(q_fr, hdr.log_precision, hdr.log_scale);
        if (surprise != expected_surprise) {
            r.errors.push_back("surprise mismatch: got " + std::to_string(surprise) +
                               ", expected " + std::to_string(expected_surprise) +
                               " (q_fr=" + std::to_string(q_fr) + ")");
            r.ok = false;
        }
    }

    // Check 4: consistency — win_prob <= total_win
    if (win_prob > total_win) {
        r.errors.push_back("win_prob (" + std::to_string(win_prob) +
                           ") > total_win (" + std::to_string(total_win) + ")");
        r.ok = false;
    }

    r.surprise = surprise;
    return r;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <proof_file> [--verbose]\n", argv[0]);
        return 1;
    }

    const char* proof_path = argv[1];
    bool verbose = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }

    // Parse proof file
    ParsedProof proof;
    try {
        proof = parse_proof_file(proof_path);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    const auto& hdr = proof.header;

    printf("Proof: T=%u, vocab_size=%u, sigma_eff=%.4f\n",
           hdr.T, hdr.vocab_size, hdr.sigma_eff);
    printf("       log_scale=%u, n_polys=%zu\n",
           hdr.log_scale, proof.all_polys.size());
    printf("Params: cdf_precision=%u, cdf_scale=%u, log_precision=%u%s\n",
           hdr.cdf_precision, hdr.cdf_scale, hdr.log_precision,
           hdr.is_v2 ? " (from proof header)" : " (defaults)");
    printf("\n");

    // Verify each position
    uint64_t entropy_sum = 0;
    uint32_t n_ok = 0, n_fail = 0;

    for (uint32_t pos = 0; pos < hdr.T; pos++) {
        auto result = verify_position(pos, proof.positions[pos], hdr);
        entropy_sum += result.surprise;

        if (result.ok) {
            n_ok++;
            if (verbose) {
                double bits = (double)result.surprise / hdr.log_scale;
                printf("  pos %4u: OK  surprise=%.3f bits  diff=%lu  q=%lu/%u  wp=%lu  tw=%lu\n",
                       pos, bits,
                       proof.positions[pos].diff_actual.val,
                       proof.positions[pos].q_fr.val,
                       1u << hdr.log_precision,
                       proof.positions[pos].win_prob.val,
                       proof.positions[pos].total_win.val);
            }
        } else {
            n_fail++;
            printf("  pos %4u: FAIL\n", pos);
            for (const auto& e : result.errors) {
                printf("    %s\n", e.c_str());
            }
        }
    }

    printf("\n");

    // Check entropy sum
    if (entropy_sum != hdr.entropy_val) {
        printf("FAIL: entropy_sum mismatch: computed %lu, claimed %lu\n",
               entropy_sum, hdr.entropy_val);
        n_fail++;
    } else {
        printf("OK: entropy_sum matches claimed value (%lu)\n", hdr.entropy_val);
    }

    // Summary
    printf("\n");
    double total_bits = (double)entropy_sum / hdr.log_scale;
    printf("Conditional entropy bound : %.4f bits total\n", total_bits);
    printf("Average per token         : %.4f bits/token\n", total_bits / hdr.T);
    printf("\n");
    printf("Positions checked: %u  OK: %u  FAIL: %u\n", hdr.T, n_ok, n_fail);

    // Soundness gap warnings
    printf("\n");
    printf("=== SOUNDNESS NOTES ===\n");
    printf("  [S1] Argmax bit-decomposition proofs: NOT VERIFIED (not yet serialized by prover)\n");
    printf("  [S2] total_win sum proof:             NOT VERIFIED (not yet serialized by prover)\n");
    printf("  [S3] Weight-binding proofs:           NOT VERIFIED (not yet serialized by prover)\n");
    printf("  Arithmetic consistency checks:        VERIFIED\n");

    if (n_fail == 0) {
        printf("\nVERIFICATION PASSED (arithmetic checks)\n");
        return 0;
    } else {
        printf("\nVERIFICATION FAILED\n");
        return 1;
    }
}
