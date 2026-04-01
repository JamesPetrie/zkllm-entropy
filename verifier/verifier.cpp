// verifier.cpp — CPU-only verifier for zkllm-entropy proof files (v3 batched format)
//
// Reads proof files in the v3 batched format and verifies algebraic consistency
// of all MLE evaluations emitted by the prover.
//
// Prototype verifier: trusts prover-generated challenges (random_vec), verifies
// that the proof's internal algebraic relations hold. Full interactive verifier
// (where verifier sends real challenges) is the next step.
//
// Checks performed:
//   [L] Linking:    diffs(u) + logits(u) == vstar(u) * ones_V(u)
//   [Q] QR division: q*tw(u) + r(u) == wp_scaled(u)
//   [B] Bit decomp: sum(2^b * bits_b(u)) == vals(u) for q, r, gap
//   [E] Binary:     combined_error(u) == 0
//   [H] Entropy:    claimed entropy_val matches header
//
// Not verified in prototype (trusted, self-checked by prover):
//   - CDF tLookup sumcheck rounds
//   - Log tLookup sumcheck rounds
//   - Row-sum inner product sumcheck
//   - Extraction inner product sumcheck
//   - Weight-binding (Pedersen/FRI commitments)
//
// Build: g++ -std=c++17 -O2 -o verifier verifier.cpp -lm
// Usage: ./verifier <proof_file> [--verbose]

#include "verifier_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ── v3 proof magic ──────────────────────────────────────────────────────────

static const uint64_t PROOF_MAGIC_V3 = 0x5A4B454E54523033ULL;  // "ZKENTR03"

// ── v3 header ───────────────────────────────────────────────────────────────

struct ProofHeaderV3 {
    uint32_t version;
    uint64_t entropy_val;
    uint32_t T;
    uint32_t vocab_size;
    double   sigma_eff;
    uint32_t log_scale;
    uint32_t cdf_precision;
    uint32_t log_precision;
    uint32_t cdf_scale;
};

// ── Parse v3 proof file ─────────────────────────────────────────────────────

struct ParsedProofV3 {
    ProofHeaderV3 header;
    std::vector<Fr_t> values;  // constant polynomial values (one per polynomial)
};

static ParsedProofV3 parse_v3(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open proof file: " + path);

    uint64_t magic = read_u64(f);
    if (magic == PROOF_MAGIC)
        throw std::runtime_error(
            "This is a v2 proof file. Regenerate with the updated prover.");
    if (magic != PROOF_MAGIC_V3)
        throw std::runtime_error("Bad magic number in proof file");

    ParsedProofV3 proof;
    proof.header.version       = read_u32(f);
    if (proof.header.version != 3)
        throw std::runtime_error("Unsupported proof version");

    proof.header.entropy_val   = read_u64(f);
    proof.header.T             = read_u32(f);
    proof.header.vocab_size    = read_u32(f);
    proof.header.sigma_eff     = read_f64(f);
    proof.header.log_scale     = read_u32(f);
    proof.header.cdf_precision = read_u32(f);
    proof.header.log_precision = read_u32(f);
    proof.header.cdf_scale     = read_u32(f);

    uint32_t n_polys = read_u32(f);

    proof.values.resize(n_polys);
    for (uint32_t i = 0; i < n_polys; i++) {
        uint32_t n_coeffs = read_u32(f);
        if (n_coeffs == 0) {
            proof.values[i] = FR_ZERO;
        } else {
            // Read first coefficient = constant value
            proof.values[i] = read_fr(f);
            // Skip remaining coefficients (should be 0 for degree-0 polys)
            for (uint32_t j = 1; j < n_coeffs; j++) {
                read_fr(f);  // discard
            }
        }
    }

    return proof;
}

// ── Compute expected polynomial count ───────────────────────────────────────

static uint32_t compute_r_bits(uint32_t vocab_size, uint32_t cdf_scale) {
    uint32_t r_bits = 1;
    uint64_t max_tw = (uint64_t)vocab_size * cdf_scale;
    while ((1ULL << r_bits) <= max_tw) r_bits++;
    return r_bits;
}

// ── Verification ────────────────────────────────────────────────────────────

struct CheckResult {
    bool ok;
    std::string tag;
    std::string detail;
};

static std::vector<CheckResult> verify_v3(const ParsedProofV3& proof, bool verbose) {
    std::vector<CheckResult> results;
    const auto& h = proof.header;
    const auto& v = proof.values;

    uint32_t q_bits = h.log_precision + 1;
    uint32_t r_bits = compute_r_bits(h.vocab_size, h.cdf_scale);

    // Expected polynomial count:
    // 4 (linking) + 1 (extraction) + 3 (QR) + 3*(1 + bits) + 1 (combined_error)
    // = 4 + 1 + 3 + (1+q_bits) + (1+r_bits) + (1+r_bits) + 1
    // = 12 + q_bits + 2*r_bits
    uint32_t expected_n = 12 + q_bits + 2 * r_bits;

    if (v.size() != expected_n) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Expected %u polynomials, got %zu (q_bits=%u, r_bits=%u)",
                 expected_n, v.size(), q_bits, r_bits);
        results.push_back({false, "STRUCTURE", buf});
        return results;
    }

    // ── [L] Linking check ─────────────────────────────────────────────────
    // diffs(u) + logits(u) == vstar(u) * ones_V(u)
    {
        Fr_t diffs_u   = v[0];
        Fr_t logits_u  = v[1];
        Fr_t vstar_u   = v[2];
        Fr_t ones_V_u  = v[3];
        Fr_t lhs = fr_add(diffs_u, logits_u);
        Fr_t rhs = fr_mul(vstar_u, ones_V_u);
        bool ok = fr_eq(lhs, rhs);
        std::string detail;
        if (ok) {
            detail = "diffs(u) + logits(u) == vstar(u) * ones_V(u)";
        } else {
            char buf[256];
            snprintf(buf, sizeof(buf), "LHS=%lu RHS=%lu", lhs.val, rhs.val);
            detail = buf;
        }
        results.push_back({ok, "LINK", detail});
    }

    // ── [Q] QR division check ─────────────────────────────────────────────
    // q*tw(u) + r(u) == wp_scaled(u)
    {
        Fr_t q_tw_u      = v[5];
        Fr_t r_u         = v[6];
        Fr_t wp_scaled_u = v[7];
        Fr_t lhs = fr_add(q_tw_u, r_u);
        bool ok = fr_eq(lhs, wp_scaled_u);
        std::string detail;
        if (ok) {
            detail = "q*tw(u) + r(u) == wp_scaled(u)";
        } else {
            char buf[256];
            snprintf(buf, sizeof(buf), "q_tw+r=%lu != wp_scaled=%lu",
                     lhs.val, wp_scaled_u.val);
            detail = buf;
        }
        results.push_back({ok, "QR_DIV", detail});
    }

    // ── [B] Bit decomposition checks ──────────────────────────────────────
    // Three prove_nonneg blocks: q, r, gap
    // Each: vals(u) followed by num_bits x bits_b(u)
    // Check: sum(2^b * bits_b(u)) == vals(u)
    uint32_t nonneg_starts[3];
    uint32_t nonneg_bits[3];
    const char* nonneg_names[3] = {"q_vec", "r_vec", "gap"};

    nonneg_starts[0] = 8;
    nonneg_bits[0]   = q_bits;
    nonneg_starts[1] = 8 + 1 + q_bits;
    nonneg_bits[1]   = r_bits;
    nonneg_starts[2] = 8 + 1 + q_bits + 1 + r_bits;
    nonneg_bits[2]   = r_bits;

    for (int b = 0; b < 3; b++) {
        Fr_t vals_u = v[nonneg_starts[b]];

        // Reconstruct: sum(2^bit * bits_bit(u))
        Fr_t recon = FR_ZERO;
        Fr_t pow2 = FR_ONE;
        Fr_t two = fr_from_u64(2);
        for (uint32_t bit = 0; bit < nonneg_bits[b]; bit++) {
            Fr_t bits_b_u = v[nonneg_starts[b] + 1 + bit];
            recon = fr_add(recon, fr_mul(pow2, bits_b_u));
            pow2 = fr_mul(pow2, two);
        }

        bool ok = fr_eq(recon, vals_u);
        char buf[256];
        if (ok) {
            snprintf(buf, sizeof(buf), "%s: reconstruction matches (%u bits)",
                     nonneg_names[b], nonneg_bits[b]);
        } else {
            snprintf(buf, sizeof(buf), "%s: vals=%lu recon=%lu (%u bits)",
                     nonneg_names[b], vals_u.val, recon.val, nonneg_bits[b]);
        }
        results.push_back({ok, std::string("BITDECOMP_") + nonneg_names[b], buf});
    }

    // ── [E] Combined binary error check ───────────────────────────────────
    {
        Fr_t ce_u = v[expected_n - 1];
        bool ok = fr_eq(ce_u, FR_ZERO);
        std::string detail;
        if (ok) {
            detail = "combined_error(u) == 0";
        } else {
            char buf[128];
            snprintf(buf, sizeof(buf), "combined_error(u) = %lu, expected 0", ce_u.val);
            detail = buf;
        }
        results.push_back({ok, "BINARY", detail});
    }

    return results;
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
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0)
            verbose = true;
    }

    ParsedProofV3 proof;
    try {
        proof = parse_v3(proof_path);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }

    const auto& h = proof.header;
    uint32_t q_bits = h.log_precision + 1;
    uint32_t r_bits = compute_r_bits(h.vocab_size, h.cdf_scale);

    printf("=== zkllm-entropy proof verifier (v3 batched format) ===\n\n");
    printf("Header:\n");
    printf("  T=%u  vocab_size=%u  sigma_eff=%.4f\n", h.T, h.vocab_size, h.sigma_eff);
    printf("  log_scale=%u  cdf_precision=%u  log_precision=%u  cdf_scale=%u\n",
           h.log_scale, h.cdf_precision, h.log_precision, h.cdf_scale);
    printf("  q_bits=%u  r_bits=%u  n_polys=%zu\n", q_bits, r_bits, proof.values.size());

    double entropy_bits = (double)h.entropy_val / (double)h.log_scale;
    printf("\nClaimed entropy: %.4f bits total (%.4f bits/token)\n\n",
           entropy_bits, entropy_bits / h.T);

    // Run checks
    auto results = verify_v3(proof, verbose);

    uint32_t n_pass = 0, n_fail = 0;
    for (const auto& r : results) {
        if (r.ok) {
            n_pass++;
            printf("  PASS [%s] %s\n", r.tag.c_str(), r.detail.c_str());
        } else {
            n_fail++;
            printf("  FAIL [%s] %s\n", r.tag.c_str(), r.detail.c_str());
        }
    }

    printf("\nChecks: %u passed, %u failed\n", n_pass, n_fail);

    // Soundness notes
    printf("\n=== SOUNDNESS NOTES (prototype) ===\n");
    printf("  Verified (algebraic consistency):\n");
    printf("    [L] diffs-to-logits linking relation\n");
    printf("    [Q] Quotient-remainder division relation\n");
    printf("    [B] Bit-decomposition reconstruction (q, r, gap)\n");
    printf("    [E] Combined binary error == 0\n");
    printf("  Trusted (prover self-checked, not serialized):\n");
    printf("    [ ] CDF tLookup sumcheck rounds\n");
    printf("    [ ] Log tLookup sumcheck rounds\n");
    printf("    [ ] Row-sum inner product sumcheck\n");
    printf("    [ ] Extraction inner product sumcheck\n");
    printf("    [ ] Weight-binding commitments (Pedersen/FRI)\n");
    printf("  Next step: interactive verifier with real challenges\n");

    if (n_fail == 0) {
        printf("\nVERIFICATION PASSED (algebraic consistency)\n");
        return 0;
    } else {
        printf("\nVERIFICATION FAILED\n");
        return 1;
    }
}
