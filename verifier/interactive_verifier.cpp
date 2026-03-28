// interactive_verifier.cpp — CPU-only interactive verifier for zkllm-entropy
//
// Communicates with the prover process via pipes.  Receives proof polynomials,
// checks each one, generates random challenges, and sends them back.
//
// Verification checks performed:
//   - Sumcheck round: p(0) + p(1) == claim
//   - IP sumcheck final: a * b == claim
//   - HP sumcheck: replay non-interactive rounds with pre-recorded challenges,
//     verify a * b * eq(u, v) == final_claim
//   - tLookup finals: A == 1/(S+beta), B == 1/(T+beta),
//     T matches public table MLE at challenge point
//
// Build: g++ -std=c++17 -O2 -DUSE_GOLDILOCKS -I verifier -o interactive_verifier verifier/interactive_verifier.cpp -lm
// Usage: ./interactive_verifier [--verbose] [--params T,V,H,cdf_prec,log_prec,cdf_scale,log_scale,sigma] -- <prover_command> [args...]

#include "verifier_utils.h"
#include "sumcheck_verifier.h"
#include "tlookup_verifier.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <random>
#include <climits>
#include <vector>
#include <string>
#include <algorithm>

// ── Wire protocol ───────────────────────────────────────────────────────────

static void read_all(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    while (len > 0) {
        ssize_t n = ::read(fd, p, len);
        if (n <= 0) {
            if (n == 0) throw std::runtime_error("read_all: EOF");
            throw std::runtime_error("read_all: error");
        }
        p += n;
        len -= n;
    }
}

static void write_all(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    while (len > 0) {
        ssize_t n = ::write(fd, p, len);
        if (n <= 0) throw std::runtime_error("write_all: error");
        p += n;
        len -= n;
    }
}

// ── Challenge generator ─────────────────────────────────────────────────────

static std::mt19937_64 g_rng(std::random_device{}());

static Fr_t generate_challenge() {
    std::uniform_int_distribution<uint64_t> dist(0, GOLDILOCKS_P - 1);
    return Fr_t{dist(g_rng)};
}

// ── Receive one 'P' message or 'E' ─────────────────────────────────────────

struct RecvResult {
    bool is_poly;
    std::vector<Fr_t> evals;
};

static RecvResult recv_from_prover(int read_fd) {
    char tag;
    read_all(read_fd, &tag, 1);
    if (tag == 'E') return {false, {}};
    if (tag != 'P')
        throw std::runtime_error("Unexpected tag: " + std::to_string((int)tag));

    uint32_t n;
    read_all(read_fd, &n, sizeof(n));
    std::vector<Fr_t> evals(n);
    if (n > 0) read_all(read_fd, evals.data(), n * sizeof(Fr_t));
    return {true, std::move(evals)};
}

static void send_challenge(int write_fd, Fr_t c) {
    write_all(write_fd, &c, sizeof(Fr_t));
}

// ── Proof parameters ────────────────────────────────────────────────────────

struct ProofParams {
    uint32_t T, V, hidden_size;
    uint32_t cdf_precision, log_precision;
    uint32_t cdf_scale, log_scale;
    double sigma_eff;
    uint32_t scaling_factor;
};

// ── Interaction sequence ────────────────────────────────────────────────────

enum class Tag {
    CHALLENGE_ONLY,
    SUMCHECK_ROUND,          // last poly is a sumcheck round
    SUMCHECK_ROUND_FIRST,    // first round of a new sumcheck (record claim, don't check)
    IP_FINALS_ONLY,          // batch contains exactly 2 IP final polys
    IP_FINALS_THEN_ROUND,    // batch starts with 2 IP finals, ends with new round poly
    HP_SUMCHECK_BATCH,       // batch has hp_rounds round polys + 2 finals
    TLOOKUP_FINALS,          // batch has 5 tLookup final evaluations (A, S, B, T, m)
};

struct Interaction {
    uint32_t polys_before;
    Tag tag;
    std::string label;
    uint32_t hp_rounds;      // only for HP_SUMCHECK_BATCH
};

// ── ProofSimulator ──────────────────────────────────────────────────────────

class ProofSimulator {
    uint32_t pending_ = 0;
    bool ip_finals_pending_ = false;
    bool hp_batch_pending_ = false;
    uint32_t hp_batch_rounds_ = 0;
    bool tlookup_finals_pending_ = false;

public:
    std::vector<Interaction> interactions;

    void push_poly() { pending_++; }

    void push_ip_finals() {
        push_poly();
        push_poly();
        ip_finals_pending_ = true;
    }

    void push_hp_batch(uint32_t rounds) {
        for (uint32_t i = 0; i < rounds; i++) push_poly();
        push_poly(); push_poly();  // finals
        hp_batch_pending_ = true;
        hp_batch_rounds_ = rounds;
    }

    void push_tlookup_finals() {
        for (int i = 0; i < 5; i++) push_poly();
        tlookup_finals_pending_ = true;
    }

    void emit(Tag tag, const std::string& label, uint32_t hp_rounds = 0) {
        interactions.push_back({pending_, tag, label, hp_rounds});
        pending_ = 0;
        ip_finals_pending_ = false;
        hp_batch_pending_ = false;
        tlookup_finals_pending_ = false;
    }

    void challenge_only(const std::string& label) {
        if (hp_batch_pending_) {
            emit(Tag::HP_SUMCHECK_BATCH, label, hp_batch_rounds_);
        } else if (tlookup_finals_pending_ && pending_ == 5) {
            emit(Tag::TLOOKUP_FINALS, label);
        } else if (ip_finals_pending_ && pending_ == 2) {
            emit(Tag::IP_FINALS_ONLY, label);
        } else {
            ip_finals_pending_ = false;
            tlookup_finals_pending_ = false;
            emit(Tag::CHALLENGE_ONLY, label);
        }
    }

    void sumcheck_round(const std::string& label, bool is_first) {
        if (ip_finals_pending_) {
            emit(Tag::IP_FINALS_THEN_ROUND, label);
        } else if (is_first) {
            emit(Tag::SUMCHECK_ROUND_FIRST, label);
        } else {
            emit(Tag::SUMCHECK_ROUND, label);
        }
    }

    // ── Simulate tLookup::prove_interactive ─────────────────────────────
    void simulate_tlookup(uint32_t D, uint32_t N, const std::string& s) {
        challenge_only(s + " alpha");
        challenge_only(s + " beta");

        uint32_t log_D = ceil_log2(D);
        for (uint32_t i = 0; i < log_D; i++)
            challenge_only(s + " u[" + std::to_string(i) + "]");

        uint32_t p1 = ceil_log2(D / N);
        for (uint32_t i = 0; i < p1; i++) {
            push_poly();
            sumcheck_round(s + " ph1 r" + std::to_string(i), i == 0);
        }

        uint32_t p2 = ceil_log2(N);
        for (uint32_t i = 0; i < p2; i++) {
            push_poly();
            sumcheck_round(s + " ph2 r" + std::to_string(i), i == 0 && p1 == 0);
        }

        // 5 final evaluations pushed at phase2 base case (no challenge)
        push_tlookup_finals();
    }

    void simulate_tlookup_range_mapping(uint32_t D, uint32_t N, const std::string& s) {
        challenge_only(s + " r");
        simulate_tlookup(D, N, s);
    }

    void simulate_ip_sumcheck(uint32_t rounds, const std::string& s) {
        for (uint32_t i = 0; i < rounds; i++) {
            push_poly();
            sumcheck_round(s + " r" + std::to_string(i), i == 0);
        }
        push_ip_finals();
    }

    void simulate_zkip(uint32_t rounds, const std::string& s) {
        for (uint32_t i = 0; i < rounds; i++) {
            push_poly();
            sumcheck_round(s + " r" + std::to_string(i), i == 0);
        }
    }

    void simulate_hp_sumcheck(uint32_t rounds, const std::string& s) {
        for (uint32_t i = 0; i < rounds; i++)
            challenge_only(s + " u[" + std::to_string(i) + "]");
        for (uint32_t i = 0; i < rounds; i++)
            challenge_only(s + " v[" + std::to_string(i) + "]");
        push_hp_batch(rounds);
    }

    void simulate_rescaling(uint32_t size, uint32_t sf, const std::string& s) {
        uint32_t log_sz = ceil_log2(size);
        for (uint32_t i = 0; i < log_sz; i++)
            challenge_only(s + " u[" + std::to_string(i) + "]");

        uint32_t rem_size = next_pow2(size);
        uint32_t table_N = next_pow2(sf);
        uint32_t D = table_N;
        while (D < rem_size) D *= 2;
        simulate_tlookup(D, table_N, s + " tL");
    }

    void simulate_zkfc(uint32_t batch, uint32_t in, uint32_t out, const std::string& s) {
        for (uint32_t i = 0; i < ceil_log2(batch); i++)
            challenge_only(s + " ub[" + std::to_string(i) + "]");
        for (uint32_t i = 0; i < ceil_log2(out); i++)
            challenge_only(s + " uo[" + std::to_string(i) + "]");
        simulate_zkip(ceil_log2(in), s + " zkip");
    }

    void simulate_prove_nonneg(uint32_t bits, const std::string& s) {
        for (uint32_t i = 0; i < bits; i++)
            challenge_only(s + " b" + std::to_string(i));
    }

    // ── Full proof flow ─────────────────────────────────────────────────
    void simulate_full_proof(const ProofParams& p) {
        uint32_t T = p.T, V = p.V, H = p.hidden_size;
        uint32_t TV = T * V;

        // §1: CDF tLookupRangeMapping
        uint32_t cdf_N = 1u << p.cdf_precision;
        uint32_t D_cdf = cdf_N;
        while (D_cdf < TV) D_cdf *= 2;
        simulate_tlookup_range_mapping(D_cdf, cdf_N, "CDF");

        // §2: Row-sum IP sumcheck
        for (uint32_t i = 0; i < ceil_log2(T); i++)
            challenge_only("rowsum u_t[" + std::to_string(i) + "]");
        simulate_ip_sumcheck(ceil_log2(V), "rowsum");

        // §3: Extraction IP sumcheck
        simulate_ip_sumcheck(ceil_log2(TV), "extract");

        // u_T challenges (first flushes 2 IP finals from extraction)
        for (uint32_t i = 0; i < ceil_log2(T); i++)
            challenge_only("extract u_T[" + std::to_string(i) + "]");
        // wp_at_u poly pushed, no challenge
        push_poly();

        // §4: Quotient-remainder
        for (uint32_t i = 0; i < ceil_log2(T); i++)
            challenge_only("qr u[" + std::to_string(i) + "]");

        uint32_t q_bits = p.log_precision + 1;
        uint32_t r_bits = 1;
        { uint64_t max_tw = (uint64_t)V * p.cdf_scale;
          while ((1ULL << r_bits) <= max_tw) r_bits++; }
        simulate_prove_nonneg(q_bits, "qr-q");
        simulate_prove_nonneg(r_bits, "qr-r");
        simulate_prove_nonneg(r_bits, "qr-gap");

        // §5: Surprise log tLookup
        uint32_t log_N = 1u << p.log_precision;
        uint32_t D_log = log_N;
        while (D_log < next_pow2(T)) D_log *= 2;
        simulate_tlookup_range_mapping(D_log, log_N, "log");

        // §6: lm_head Rescaling
        simulate_rescaling(TV, p.scaling_factor, "lm-rs");

        // §7: lm_head zkFC
        simulate_zkfc(T, H, V, "lm-fc");

        // §8: RMSNorm rescaling 2
        simulate_rescaling(T * H, p.scaling_factor, "n2-rs");

        // §9: Hadamard product sumcheck
        uint32_t hp_rounds = ceil_log2(next_pow2(T * H));
        simulate_hp_sumcheck(hp_rounds, "HP");

        // §10: RMSNorm rescaling 1 (first challenge flushes HP polys+finals)
        simulate_rescaling(T * H, p.scaling_factor, "n1-rs");

        // §11: Final norm zkFC
        simulate_zkfc(T, 1, H, "norm-fc");
    }
};

// ── Verification state ──────────────────────────────────────────────────────

struct Stats {
    uint32_t polys_received = 0;
    uint32_t challenges_sent = 0;
    uint32_t sc_checks_passed = 0;
    uint32_t sc_checks_failed = 0;
    uint32_t ip_checks_passed = 0;
    uint32_t ip_checks_failed = 0;
    uint32_t hp_checks_passed = 0;
    uint32_t hp_checks_failed = 0;
    uint32_t tl_checks_passed = 0;
    uint32_t tl_checks_failed = 0;
    std::vector<std::string> errors;

    void error(const std::string& msg) {
        errors.push_back(msg);
        fprintf(stderr, "  FAIL: %s\n", msg.c_str());
    }
};

// ── tLookup section state ──────────────────────────────────────────────────

struct TLookupState {
    Fr_t alpha = FR_ZERO;
    Fr_t beta = FR_ZERO;
    Fr_t r = FR_ZERO;        // combining challenge for range-mapped
    bool has_r = false;
    std::vector<Fr_t> phase2_challenges;
};

// ── Public table builders ──────────────────────────────────────────────────

// Build the combined table T_com[j] = table_j + r * mapped_j for range-mapped tLookup
static std::vector<Fr_t> build_combined_table(
    const std::string& section, Fr_t r, const ProofParams& params)
{
    if (section == "CDF") {
        auto mapped = build_cdf_table(params.cdf_precision, params.cdf_scale, params.sigma_eff);
        uint32_t N = mapped.size();
        std::vector<Fr_t> combined(N);
        for (uint32_t j = 0; j < N; j++) {
            Fr_t table_j = fr_from_u64(j);  // CDF range starts at 0
            combined[j] = fr_add(table_j, fr_mul(r, mapped[j]));
        }
        return combined;
    }
    if (section == "log") {
        auto mapped = build_log_table(params.log_precision, params.log_scale);
        uint32_t N = mapped.size();
        std::vector<Fr_t> combined(N);
        for (uint32_t j = 0; j < N; j++) {
            Fr_t table_j = fr_from_u64(j + 1);  // log range starts at 1
            combined[j] = fr_add(table_j, fr_mul(r, mapped[j]));
        }
        return combined;
    }
    return {};
}

// Build the plain range table for rescaling tLookup: [low, low+1, ..., low+N-1]
static std::vector<Fr_t> build_range_table(int32_t low, uint32_t N) {
    std::vector<Fr_t> table(N);
    for (uint32_t j = 0; j < N; j++) {
        int32_t val = (int32_t)j + low;
        // Convert signed to field element (wraps modulo p for negative values)
        if (val >= 0)
            table[j] = fr_from_u64((uint64_t)val);
        else
            table[j] = fr_neg(fr_from_u64((uint64_t)(-val)));
    }
    return table;
}

// Determine which section a tLookup belongs to and get its public table
static std::vector<Fr_t> get_public_table(
    const std::string& label, const TLookupState& tl, const ProofParams& params)
{
    // Range-mapped tLookups (have r challenge)
    if (tl.has_r) {
        if (label.find("CDF") != std::string::npos)
            return build_combined_table("CDF", tl.r, params);
        if (label.find("log") != std::string::npos)
            return build_combined_table("log", tl.r, params);
    }
    // Rescaling tLookups (plain range tables)
    uint32_t sf = params.scaling_factor;
    uint32_t N = next_pow2(sf);
    int32_t low = -(int32_t)(sf >> 1);
    return build_range_table(low, N);
}

// ── HP sumcheck verification ───────────────────────────────────────────────

static bool verify_hp_batch(
    const std::vector<std::vector<Fr_t>>& polys,
    uint32_t hp_rounds,
    const std::vector<Fr_t>& hp_u,
    const std::vector<Fr_t>& hp_v,
    Stats& stats,
    bool verbose,
    const std::string& label)
{
    if (polys.size() < hp_rounds + 2) {
        stats.error("HP: expected " + std::to_string(hp_rounds + 2) +
                    " polys, got " + std::to_string(polys.size()));
        return false;
    }

    // HP sumcheck claim equation: p(0)*(1-u[i]) + p(1)*u[i] == claim.
    // The HP polynomial is p(x) = sum_j f_j(x)*eq(j, u_remaining), where
    // f_j(x) = (a[2j]+x*Δa_j)*(b[2j]+x*Δb_j) is degree 2.  The eq weight
    // over the BOOLEAN hypercube gives: claim = p(0)*(1-u[i]) + p(1)*u[i],
    // NOT p(u[i]) (which would use the degree-2 extension, overcounting
    // cross terms).
    //
    // Round 0: no check — set claim = p_0(v[0])
    // Round i>0: check p(0)*(1-u[i]) + p(1)*u[i] == claim, advance via p(v[i])
    // Final: claim == a_final * b_final (no eq factor — it's absorbed)

    auto& e0 = polys[0];
    if (e0.size() < 2) {
        stats.error("HP: first round poly too short");
        return false;
    }
    Fr_t claim = lagrange_eval(e0, hp_v[0]);
    if (verbose) printf("  [HP r0] initial, advance to claim=%lu\n", claim.val);

    bool all_ok = true;
    for (uint32_t i = 1; i < hp_rounds; i++) {
        auto& e = polys[i];
        if (e.size() < 2) {
            stats.error("HP round " + std::to_string(i) + ": poly too short");
            return false;
        }
        // Check: p(0)*(1-u[i]) + p(1)*u[i] == claim
        Fr_t check = fr_add(fr_mul(e[0], fr_sub(FR_ONE, hp_u[i])),
                            fr_mul(e[1], hp_u[i]));
        if (check != claim) {
            stats.hp_checks_failed++;
            stats.error("HP round " + std::to_string(i) + ": eq_check=" +
                        std::to_string(check.val) +
                        " != claim=" + std::to_string(claim.val));
            all_ok = false;
        } else {
            stats.hp_checks_passed++;
            if (verbose) printf("  [HP r%u] eq_check==claim OK\n", i);
        }
        claim = lagrange_eval(e, hp_v[i]);
    }

    // Final check: claim == a_final * b_final
    Fr_t final_a = polys[hp_rounds].empty() ? FR_ZERO : polys[hp_rounds][0];
    Fr_t final_b = polys[hp_rounds + 1].empty() ? FR_ZERO : polys[hp_rounds + 1][0];
    Fr_t expected = fr_mul(final_a, final_b);

    if (expected == claim) {
        stats.hp_checks_passed++;
        if (verbose) printf("  [HP] final a*b==claim OK\n");
    } else {
        stats.hp_checks_failed++;
        stats.error("HP final: a*b=" + std::to_string(expected.val) +
                    " != claim=" + std::to_string(claim.val));
        all_ok = false;
    }

    return all_ok;
}

// ── tLookup final verification ─────────────────────────────────────────────

static bool verify_tlookup_finals(
    const std::vector<std::vector<Fr_t>>& polys,
    const TLookupState& tl,
    const ProofParams& params,
    Stats& stats,
    bool verbose,
    const std::string& label)
{
    if (polys.size() < 5) {
        stats.error("tLookup finals: expected 5 polys, got " + std::to_string(polys.size()));
        return false;
    }

    Fr_t final_A = polys[0].empty() ? FR_ZERO : polys[0][0];
    Fr_t final_S = polys[1].empty() ? FR_ZERO : polys[1][0];
    Fr_t final_B = polys[2].empty() ? FR_ZERO : polys[2][0];
    Fr_t final_T = polys[3].empty() ? FR_ZERO : polys[3][0];
    Fr_t final_m = polys[4].empty() ? FR_ZERO : polys[4][0];

    bool all_ok = true;

    // NOTE: A == 1/(S+beta) and B == 1/(T+beta) hold element-wise for the
    // original tensors, but NOT after MLE reduction (MLE is linear, inverse
    // is not).  These checks require a PCS to verify openings of the committed
    // polynomials.  Without PCS, we can only verify the public table T.

    if (verbose) {
        printf("  [%s] tL finals: A=%lu S=%lu B=%lu T=%lu m=%lu\n",
               label.c_str(), final_A.val, final_S.val, final_B.val,
               final_T.val, final_m.val);
    }

    // Check T matches public table MLE at phase2 challenge point.
    // Phase2 reduces highest bit first (pairs [j] with [j+N/2]), but
    // mle_eval folds lowest bit first (pairs [2j] with [2j+1]).
    // Reverse the challenge order to compensate.
    auto public_table = get_public_table(label, tl, params);
    if (!public_table.empty() && !tl.phase2_challenges.empty()) {
        auto rev_challenges = tl.phase2_challenges;
        std::reverse(rev_challenges.begin(), rev_challenges.end());
        Fr_t expected_T = mle_eval(public_table, rev_challenges);
        if (expected_T == final_T) {
            stats.tl_checks_passed++;
            if (verbose) printf("  [%s] tL T==table_MLE(v) OK\n", label.c_str());
        } else {
            stats.tl_checks_failed++;
            stats.error("tLookup T != table_MLE at " + label +
                        ": got " + std::to_string(final_T.val) +
                        " expected " + std::to_string(expected_T.val));
            all_ok = false;
        }
    }

    return all_ok;
}

// ── Main verification loop ──────────────────────────────────────────────────

static void run_verification(int read_fd, int write_fd, bool verbose,
                              const ProofParams& params) {
    ProofSimulator sim;
    sim.simulate_full_proof(params);
    auto& seq = sim.interactions;

    uint32_t total_polys = 0;
    for (auto& i : seq) total_polys += i.polys_before;
    printf("Interactive verifier started (%zu interactions, %u polys expected).\n",
           seq.size(), total_polys);

    Stats stats;
    Fr_t claim = FR_ZERO;
    bool claim_valid = false;

    // HP sumcheck state
    std::vector<Fr_t> hp_u, hp_v;

    // tLookup section state
    TLookupState current_tl;
    std::string current_tl_section;  // e.g. "CDF", "log", "lm-rs tL"
    bool in_tl_phase2 = false;

    for (size_t idx = 0; idx < seq.size(); idx++) {
        const auto& inter = seq[idx];

        // 1. Read the expected number of polynomials
        std::vector<std::vector<Fr_t>> polys;
        for (uint32_t i = 0; i < inter.polys_before; i++) {
            auto r = recv_from_prover(read_fd);
            if (!r.is_poly) {
                stats.error("Unexpected EOF at #" + std::to_string(idx) +
                            " (" + inter.label + ")");
                goto done;
            }
            polys.push_back(std::move(r.evals));
            stats.polys_received++;
        }

        // 2. Verification
        switch (inter.tag) {

        case Tag::CHALLENGE_ONLY:
            break;

        case Tag::SUMCHECK_ROUND_FIRST: {
            if (polys.empty() || polys.back().size() < 2) {
                stats.error("Missing poly for first round at " + inter.label);
                break;
            }
            auto& e = polys.back();
            claim = fr_add(e[0], e[1]);
            claim_valid = true;
            if (verbose) printf("  [%s] sumcheck claim=%lu\n", inter.label.c_str(), claim.val);
            break;
        }

        case Tag::SUMCHECK_ROUND: {
            if (polys.empty() || polys.back().size() < 2) {
                stats.error("Missing poly at " + inter.label);
                break;
            }
            auto& e = polys.back();
            Fr_t sum = fr_add(e[0], e[1]);
            if (claim_valid && sum == claim) {
                stats.sc_checks_passed++;
                if (verbose) printf("  [%s] p(0)+p(1)==claim OK\n", inter.label.c_str());
            } else if (claim_valid) {
                stats.sc_checks_failed++;
                stats.error("p(0)+p(1)=" + std::to_string(sum.val) +
                            " != claim=" + std::to_string(claim.val) +
                            " at " + inter.label);
            }
            break;
        }

        case Tag::IP_FINALS_ONLY: {
            if (polys.size() >= 2) {
                Fr_t a = polys[0].empty() ? FR_ZERO : polys[0][0];
                Fr_t b = polys[1].empty() ? FR_ZERO : polys[1][0];
                Fr_t ab = fr_mul(a, b);
                if (claim_valid && ab == claim) {
                    stats.ip_checks_passed++;
                    if (verbose) printf("  [%s] IP final a*b==claim OK\n", inter.label.c_str());
                } else if (claim_valid) {
                    stats.ip_checks_failed++;
                    stats.error("IP final: a*b=" + std::to_string(ab.val) +
                                " != claim=" + std::to_string(claim.val) +
                                " at " + inter.label);
                }
            }
            claim_valid = false;
            break;
        }

        case Tag::IP_FINALS_THEN_ROUND: {
            if (polys.size() >= 2) {
                Fr_t a = polys[0].empty() ? FR_ZERO : polys[0][0];
                Fr_t b = polys[1].empty() ? FR_ZERO : polys[1][0];
                Fr_t ab = fr_mul(a, b);
                if (claim_valid && ab == claim) {
                    stats.ip_checks_passed++;
                    if (verbose) printf("  [%s] IP final a*b==claim OK\n", inter.label.c_str());
                } else if (claim_valid) {
                    stats.ip_checks_failed++;
                    stats.error("IP final: a*b=" + std::to_string(ab.val) +
                                " != claim=" + std::to_string(claim.val) +
                                " at " + inter.label);
                }
            }
            if (!polys.empty() && polys.back().size() >= 2) {
                auto& e = polys.back();
                claim = fr_add(e[0], e[1]);
                claim_valid = true;
                if (verbose) printf("  [%s] new sumcheck claim=%lu\n",
                                     inter.label.c_str(), claim.val);
            }
            break;
        }

        case Tag::HP_SUMCHECK_BATCH: {
            verify_hp_batch(polys, inter.hp_rounds, hp_u, hp_v, stats, verbose, inter.label);
            claim_valid = false;
            break;
        }

        case Tag::TLOOKUP_FINALS: {
            verify_tlookup_finals(polys, current_tl, params, stats, verbose, current_tl_section);
            claim_valid = false;
            // Reset tLookup state for next section
            current_tl = TLookupState{};
            current_tl_section.clear();
            in_tl_phase2 = false;
            break;
        }

        } // switch

        // 3. Generate and send challenge
        Fr_t ch = generate_challenge();
        send_challenge(write_fd, ch);
        stats.challenges_sent++;

        // 4. Update claim = p(challenge) for sumcheck rounds
        if ((inter.tag == Tag::SUMCHECK_ROUND ||
             inter.tag == Tag::SUMCHECK_ROUND_FIRST ||
             inter.tag == Tag::IP_FINALS_THEN_ROUND) && !polys.empty()) {
            auto& e = polys.back();
            if (e.size() >= 2) {
                claim = lagrange_eval(e, ch);
                claim_valid = true;
            }
        }

        // 5. Track section-specific state from labels
        const auto& lbl = inter.label;

        // HP challenges
        if (lbl.find("HP u[") == 0) hp_u.push_back(ch);
        else if (lbl.find("HP v[") == 0) hp_v.push_back(ch);

        // tLookup challenges — detect section by label patterns
        // Range-mapped: "CDF r", "CDF alpha", "CDF ph2 r0", etc.
        // Plain: "lm-rs tL alpha", "lm-rs tL ph2 r0", etc.
        auto endswith = [](const std::string& s, const std::string& suffix) {
            return s.size() >= suffix.size() &&
                   s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

        if (endswith(lbl, " r") && lbl.find(" ph") == std::string::npos &&
            lbl.find(" r[") == std::string::npos) {
            // "CDF r" or "log r" — range mapping combining challenge
            current_tl_section = lbl.substr(0, lbl.size() - 2);
            current_tl.r = ch;
            current_tl.has_r = true;
        } else if (endswith(lbl, " alpha")) {
            current_tl.alpha = ch;
            if (current_tl_section.empty())
                current_tl_section = lbl.substr(0, lbl.size() - 6);
        } else if (endswith(lbl, " beta")) {
            current_tl.beta = ch;
        } else if (lbl.find(" ph2 r") != std::string::npos) {
            current_tl.phase2_challenges.push_back(ch);
        }
    }

    // Drain any remaining polys and read 'E'
    while (true) {
        char tag;
        read_all(read_fd, &tag, 1);
        if (tag == 'E') break;
        if (tag == 'P') {
            uint32_t n;
            read_all(read_fd, &n, sizeof(n));
            std::vector<Fr_t> tmp(n);
            if (n > 0) read_all(read_fd, tmp.data(), n * sizeof(Fr_t));
            stats.polys_received++;
        }
    }

done:
    printf("\nInteractive verification complete.\n");
    printf("  Polynomials received:     %u\n", stats.polys_received);
    printf("  Challenges sent:          %u\n", stats.challenges_sent);
    printf("  Sumcheck rounds checked:  %u passed, %u failed\n",
           stats.sc_checks_passed, stats.sc_checks_failed);
    printf("  IP final checks:          %u passed, %u failed\n",
           stats.ip_checks_passed, stats.ip_checks_failed);
    printf("  HP sumcheck checks:       %u passed, %u failed\n",
           stats.hp_checks_passed, stats.hp_checks_failed);
    printf("  tLookup final checks:     %u passed, %u failed\n",
           stats.tl_checks_passed, stats.tl_checks_failed);

    if (stats.errors.empty()) {
        printf("\nVERIFICATION PASSED (interactive)\n");
    } else {
        printf("\nVERIFICATION FAILED (%zu errors)\n", stats.errors.size());
        for (const auto& e : stats.errors)
            printf("  ERROR: %s\n", e.c_str());
    }
}

// ── Parse parameters ───────────────────────────────────────────────────────

static ProofParams parse_params_string(const char* s) {
    ProofParams p;
    p.scaling_factor = 1u << 16;
    // Format: T,V,H,cdf_prec,log_prec,cdf_scale,log_scale,sigma
    unsigned t, v, h, cp, lp, cs, ls;
    double sigma;
    int n = sscanf(s, "%u,%u,%u,%u,%u,%u,%u,%lf", &t, &v, &h, &cp, &lp, &cs, &ls, &sigma);
    if (n >= 3) { p.T = t; p.V = v; p.hidden_size = h; }
    if (n >= 4) p.cdf_precision = cp; else p.cdf_precision = 20;
    if (n >= 5) p.log_precision = lp; else p.log_precision = 15;
    if (n >= 6) p.cdf_scale = cs; else p.cdf_scale = 65536;
    if (n >= 7) p.log_scale = ls; else p.log_scale = 65536;
    if (n >= 8) p.sigma_eff = sigma; else p.sigma_eff = 3277.0;
    return p;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    bool verbose = false;
    int cmd_start = -1;
    const char* params_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--params") == 0 && i + 1 < argc) {
            params_str = argv[++i];
        } else if (strcmp(argv[i], "--") == 0) {
            cmd_start = i + 1;
            break;
        }
    }

    if (cmd_start < 0 || cmd_start >= argc) {
        fprintf(stderr, "Usage: %s [--verbose] [--params T,V,H,...] -- <prover_command> [args...]\n", argv[0]);
        fprintf(stderr, "\nParams format: T,V,H,cdf_prec,log_prec,cdf_scale,log_scale,sigma\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "  %s -v --params 1024,32000,4096,20,15,65536,65536,3277 -- ./gold_zkllm_entropy workdir tokens.txt proof.bin 1.0\n",
                argv[0]);
        return 1;
    }

    ProofParams params;
    if (params_str) {
        params = parse_params_string(params_str);
    } else {
        // Defaults
        params.T = 1024; params.V = 32000; params.hidden_size = 4096;
        params.cdf_precision = 20; params.log_precision = 15;
        params.cdf_scale = 65536; params.log_scale = 65536;
        params.sigma_eff = 3277.0;
        params.scaling_factor = 1u << 16;
    }

    printf("Proof parameters:\n");
    printf("  T=%u V=%u H=%u cdf_prec=%u log_prec=%u\n",
           params.T, params.V, params.hidden_size,
           params.cdf_precision, params.log_precision);
    printf("  cdf_scale=%u log_scale=%u sigma=%.1f sf=%u\n",
           params.cdf_scale, params.log_scale, params.sigma_eff, params.scaling_factor);

    {
        ProofSimulator preview;
        preview.simulate_full_proof(params);
        uint32_t total_polys = 0;
        for (auto& i : preview.interactions) total_polys += i.polys_before;
        printf("  Expected: %zu interactions, %u polys\n",
               preview.interactions.size(), total_polys);
    }

    int pipe_to_prover[2], pipe_from_prover[2];
    if (pipe(pipe_to_prover) != 0 || pipe(pipe_from_prover) != 0) {
        perror("pipe"); return 1;
    }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        close(pipe_to_prover[1]);
        close(pipe_from_prover[0]);

        char rfd[32], wfd[32];
        snprintf(rfd, sizeof(rfd), "%d", pipe_to_prover[0]);
        snprintf(wfd, sizeof(wfd), "%d", pipe_from_prover[1]);
        setenv("ZKLLM_CHALLENGE_READ_FD", rfd, 1);
        setenv("ZKLLM_CHALLENGE_WRITE_FD", wfd, 1);

        execvp(argv[cmd_start], &argv[cmd_start]);
        perror("execvp");
        _exit(1);
    }

    close(pipe_to_prover[0]);
    close(pipe_from_prover[1]);

    try {
        run_verification(pipe_from_prover[0], pipe_to_prover[1], verbose, params);
    } catch (const std::exception& e) {
        fprintf(stderr, "Verifier error: %s\n", e.what());
    }

    close(pipe_to_prover[1]);
    close(pipe_from_prover[0]);

    int status;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        printf("Prover exited with code %d\n", code);
        return code;
    } else {
        printf("Prover terminated abnormally\n");
        return 1;
    }
}
