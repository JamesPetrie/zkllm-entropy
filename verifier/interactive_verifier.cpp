// interactive_verifier.cpp — CPU-only interactive verifier for zkllm-entropy
//
// Communicates with the prover process via pipes.  Receives proof polynomials,
// checks each one, generates random challenges, and sends them back.
//
// The verifier knows the proof structure from public parameters and checks
// sumcheck round polynomials (p(0) + p(1) == claim) as they arrive.
//
// Build: g++ -std=c++17 -O2 -DUSE_GOLDILOCKS -I verifier -o interactive_verifier verifier/interactive_verifier.cpp -lm
// Usage: ./interactive_verifier [--verbose] -- ./gold_zkllm_entropy <args...>

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
// Each Interaction = one PipeChallengeSource::next() call = one challenge sent.
// polys_before = how many 'P' messages arrive before this challenge.
//
// For verification, we annotate each interaction:
//   CHALLENGE_ONLY:  no check, just send a challenge
//   SUMCHECK_ROUND:  check the LAST poly in the batch: p(0)+p(1)==claim
//   IP_FINALS:       check the FIRST 2 polys in the batch: claim==a*b
//                    (may have additional polys after which are the next round)

enum class Tag {
    CHALLENGE_ONLY,
    SUMCHECK_ROUND,          // last poly is a sumcheck round
    SUMCHECK_ROUND_FIRST,    // first round of a new sumcheck (record claim, don't check)
    IP_FINALS_ONLY,          // batch contains exactly 2 IP final polys
    IP_FINALS_THEN_ROUND,    // batch starts with 2 IP finals, ends with new round poly
};

struct Interaction {
    uint32_t polys_before;
    Tag tag;
    std::string label;
};

// ── ProofSimulator ──────────────────────────────────────────────────────────

class ProofSimulator {
    uint32_t pending_ = 0;
    bool ip_finals_pending_ = false;  // true if 2 IP final polys are in pending

public:
    std::vector<Interaction> interactions;

    void push_poly() { pending_++; }

    // Record that 2 IP final polys were just pushed (no challenge after them)
    void push_ip_finals() {
        push_poly();
        push_poly();
        ip_finals_pending_ = true;
    }

    void emit(Tag tag, const std::string& label) {
        interactions.push_back({pending_, tag, label});
        pending_ = 0;
        ip_finals_pending_ = false;
    }

    // Emit a challenge-only interaction (possibly flushing pending polys)
    void challenge_only(const std::string& label) {
        if (ip_finals_pending_ && pending_ == 2) {
            emit(Tag::IP_FINALS_ONLY, label);
        } else if (ip_finals_pending_ && pending_ > 2) {
            // IP finals + some other pending polys — shouldn't happen in practice
            emit(Tag::IP_FINALS_ONLY, label);
        } else {
            emit(Tag::CHALLENGE_ONLY, label);
        }
    }

    // Emit a sumcheck round (1 round poly was just pushed, then challenge)
    void sumcheck_round(const std::string& label, bool is_first) {
        if (ip_finals_pending_) {
            // The pending batch has: 2 IP finals + 1 round poly = 3
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
            // First round of phase2 is NOT first overall if phase1 had rounds
            sumcheck_round(s + " ph2 r" + std::to_string(i), i == 0 && p1 == 0);
        }
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
        // Non-interactive: rounds polys + 2 finals, all pending
        for (uint32_t i = 0; i < rounds; i++)
            push_poly();
        push_ip_finals();
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
    std::vector<std::string> errors;

    void error(const std::string& msg) {
        errors.push_back(msg);
        fprintf(stderr, "  FAIL: %s\n", msg.c_str());
    }
};

// ── Main verification loop ──────────────────────────────────────────────────

static void run_verification(int read_fd, int write_fd, bool verbose,
                              const ProofParams& params) {
    ProofSimulator sim;
    sim.simulate_full_proof(params);
    auto& seq = sim.interactions;

    printf("Interactive verifier started (%zu interactions expected).\n", seq.size());

    Stats stats;
    Fr_t claim = FR_ZERO;         // current sumcheck claim
    bool claim_valid = false;     // whether we have a claim to check against

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
            // No verification; just send challenge
            break;

        case Tag::SUMCHECK_ROUND_FIRST: {
            // First round of a new sumcheck — record the initial claim
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
            // Batch has 2 IP final polys (degree-0)
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
            // First 2 polys are IP finals, last poly is new sumcheck round
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

            // New sumcheck: last poly is the round poly
            if (!polys.empty() && polys.back().size() >= 2) {
                auto& e = polys.back();
                claim = fr_add(e[0], e[1]);
                claim_valid = true;
                if (verbose) printf("  [%s] new sumcheck claim=%lu\n",
                                     inter.label.c_str(), claim.val);
            }
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

    if (stats.errors.empty()) {
        printf("\nVERIFICATION PASSED (interactive)\n");
    } else {
        printf("\nVERIFICATION FAILED (%zu errors)\n", stats.errors.size());
        for (const auto& e : stats.errors)
            printf("  ERROR: %s\n", e.c_str());
    }
}

// ── Parse prover args to extract proof parameters ───────────────────────────

static ProofParams parse_prover_params(int argc, char* argv[], int cmd_start) {
    ProofParams p;
    p.scaling_factor = 1u << 16;

    int nargs = argc - cmd_start;
    if (nargs < 5) {
        fprintf(stderr, "Warning: using default proof parameters\n");
        p.T = 1024; p.V = 32000; p.hidden_size = 4096;
        p.cdf_precision = 20; p.log_precision = 15;
        p.cdf_scale = 65536; p.log_scale = 65536;
        p.sigma_eff = 3277.0;
        return p;
    }

    // prover args: binary workdir tokens proof sigma [seq_len] [hidden] [vocab] ...
    p.sigma_eff    = atof(argv[cmd_start + 4]);
    p.T            = nargs > 5  ? (uint32_t)atoi(argv[cmd_start + 5])  : 1024u;
    p.hidden_size  = nargs > 6  ? (uint32_t)atoi(argv[cmd_start + 6])  : 4096u;
    p.V            = nargs > 7  ? (uint32_t)atoi(argv[cmd_start + 7])  : 32000u;
    p.cdf_precision= nargs > 8  ? (uint32_t)atoi(argv[cmd_start + 8])  : 20u;
    p.log_precision= nargs > 9  ? (uint32_t)atoi(argv[cmd_start + 9])  : 15u;
    p.cdf_scale    = nargs > 10 ? (uint32_t)atoi(argv[cmd_start + 10]) : 65536u;
    p.log_scale    = nargs > 11 ? (uint32_t)atoi(argv[cmd_start + 11]) : 65536u;

    return p;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    bool verbose = false;
    int cmd_start = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--") == 0) {
            cmd_start = i + 1;
            break;
        }
    }

    if (cmd_start < 0 || cmd_start >= argc) {
        fprintf(stderr, "Usage: %s [--verbose] -- <prover_command> [args...]\n", argv[0]);
        fprintf(stderr, "\nExample:\n");
        fprintf(stderr, "  %s -v -- ./gold_zkllm_entropy workdir tokens.txt proof.bin 3277\n",
                argv[0]);
        return 1;
    }

    ProofParams params = parse_prover_params(argc, argv, cmd_start);

    printf("Proof parameters:\n");
    printf("  T=%u V=%u H=%u cdf_prec=%u log_prec=%u\n",
           params.T, params.V, params.hidden_size,
           params.cdf_precision, params.log_precision);
    printf("  cdf_scale=%u log_scale=%u sigma=%.1f sf=%u\n",
           params.cdf_scale, params.log_scale, params.sigma_eff, params.scaling_factor);

    {
        ProofSimulator preview;
        preview.simulate_full_proof(params);
        printf("  Expected interactions: %zu\n", preview.interactions.size());
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
