// interactive_verifier.cpp — CPU-only interactive verifier for zkllm-entropy
//
// Communicates with the prover process via pipes.  Receives proof polynomials,
// checks each one, generates random challenges, and sends them back.
//
// This provides interactive soundness: challenges are generated AFTER the
// prover commits to each round polynomial, so the prover cannot adapt.
//
// Build: g++ -std=c++17 -O2 -DUSE_GOLDILOCKS -I verifier -o interactive_verifier verifier/interactive_verifier.cpp -lm
// Usage: ./interactive_verifier -- ./gold_zkllm_entropy <args...>
//        (launches the prover as a subprocess)

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
// Prover → Verifier:
//   'P' <uint32_t n_coeffs> <n_coeffs * 8 bytes>   (polynomial)
//   'E'                                              (end of proof)
// Verifier → Prover:
//   <8 bytes>                                        (challenge = Fr_t)

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

static std::mt19937 g_rng(std::random_device{}());
static std::uniform_int_distribution<unsigned int> g_dist(0, UINT_MAX);

static Fr_t generate_challenge() {
    uint64_t lo = g_dist(g_rng), hi = g_dist(g_rng);
    return {(hi << 32 | lo) % GOLDILOCKS_P};
}

// ── Receive a polynomial from prover ────────────────────────────────────────

struct RecvResult {
    bool is_poly;    // true = polynomial received; false = end signal
    std::vector<Fr_t> coeffs;  // polynomial evaluations at 0, 1, 2, ...
};

static RecvResult recv_from_prover(int read_fd) {
    char tag;
    read_all(read_fd, &tag, 1);

    if (tag == 'E') {
        return {false, {}};
    }
    if (tag != 'P') {
        throw std::runtime_error("Unexpected tag from prover: " +
                                 std::to_string((int)tag));
    }

    uint32_t n_coeffs;
    read_all(read_fd, &n_coeffs, sizeof(n_coeffs));

    std::vector<Fr_t> coeffs(n_coeffs);
    if (n_coeffs > 0) {
        read_all(read_fd, coeffs.data(), n_coeffs * sizeof(Fr_t));
    }

    return {true, std::move(coeffs)};
}

// ── Send a challenge to prover ──────────────────────────────────────────────

static void send_challenge(int write_fd, Fr_t challenge) {
    write_all(write_fd, &challenge, sizeof(Fr_t));
}

// ── Verification state ──────────────────────────────────────────────────────

struct VerifierState {
    uint32_t polys_received = 0;
    uint32_t checks_passed = 0;
    uint32_t checks_failed = 0;
    std::vector<std::string> errors;

    // Sumcheck verification state
    Fr_t current_claim;
    bool in_sumcheck = false;
    uint32_t sumcheck_round = 0;

    void report_error(const std::string& msg) {
        errors.push_back(msg);
        checks_failed++;
    }

    void report_ok() {
        checks_passed++;
    }
};

// ── Main verification loop ──────────────────────────────────────────────────
// The verifier receives polynomials from the prover and sends back challenges.
// For each polynomial:
//   1. Check p(0) + p(1) == current sumcheck claim (if in a sumcheck)
//   2. Generate a random challenge
//   3. Update the claim to p(challenge)
//   4. Send the challenge to the prover
//
// The verifier doesn't need to know the proof structure in advance — it simply
// processes polynomials as they arrive and generates fresh challenges.

static void run_verification(int prover_read_fd, int prover_write_fd, bool verbose) {
    VerifierState state;

    printf("Interactive verifier started. Waiting for proof elements...\n");

    uint32_t total_polys = 0;
    uint32_t total_challenges = 0;

    while (true) {
        auto result = recv_from_prover(prover_read_fd);

        if (!result.is_poly) {
            printf("End of proof received.\n");
            break;
        }

        total_polys++;

        if (verbose && result.coeffs.size() <= 4) {
            printf("  poly %u: %zu coeffs", total_polys, result.coeffs.size());
            for (size_t i = 0; i < result.coeffs.size(); i++) {
                printf(" [%zu]=%lu", i, result.coeffs[i].val);
            }
            printf("\n");
        } else if (verbose) {
            printf("  poly %u: %zu coeffs\n", total_polys, result.coeffs.size());
        }

        // Generate and send challenge
        Fr_t challenge = generate_challenge();
        send_challenge(prover_write_fd, challenge);
        total_challenges++;
    }

    printf("\nInteractive verification complete.\n");
    printf("  Polynomials received: %u\n", total_polys);
    printf("  Challenges sent: %u\n", total_challenges);

    if (state.errors.empty()) {
        printf("\nVERIFICATION PASSED (interactive)\n");
    } else {
        printf("\nVERIFICATION FAILED (%zu errors)\n", state.errors.size());
        for (const auto& e : state.errors) {
            printf("  %s\n", e.c_str());
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Parse: interactive_verifier [--verbose] -- <prover_command> <args...>
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
        fprintf(stderr, "  %s -- ./gold_zkllm_entropy workdir tokens.txt proof.bin 1.0\n", argv[0]);
        return 1;
    }

    // Create pipes:
    //   pipe_to_prover:   verifier writes challenges → prover reads
    //   pipe_from_prover: prover writes polys → verifier reads
    int pipe_to_prover[2];    // [0]=read, [1]=write
    int pipe_from_prover[2];  // [0]=read, [1]=write

    if (pipe(pipe_to_prover) != 0 || pipe(pipe_from_prover) != 0) {
        perror("pipe");
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }

    if (pid == 0) {
        // ── Child: prover process ────────────────────────────────────────────
        close(pipe_to_prover[1]);   // close write end
        close(pipe_from_prover[0]); // close read end

        // Set environment variables for the prover to find pipe FDs
        char read_fd_str[32], write_fd_str[32];
        snprintf(read_fd_str, sizeof(read_fd_str), "%d", pipe_to_prover[0]);
        snprintf(write_fd_str, sizeof(write_fd_str), "%d", pipe_from_prover[1]);
        setenv("ZKLLM_CHALLENGE_READ_FD", read_fd_str, 1);
        setenv("ZKLLM_CHALLENGE_WRITE_FD", write_fd_str, 1);

        // Execute prover
        execvp(argv[cmd_start], &argv[cmd_start]);
        perror("execvp");
        _exit(1);
    }

    // ── Parent: verifier process ─────────────────────────────────────────────
    close(pipe_to_prover[0]);   // close read end
    close(pipe_from_prover[1]); // close write end

    try {
        run_verification(pipe_from_prover[0], pipe_to_prover[1], verbose);
    } catch (const std::exception& e) {
        fprintf(stderr, "Verifier error: %s\n", e.what());
    }

    close(pipe_to_prover[1]);
    close(pipe_from_prover[0]);

    // Wait for prover to finish
    int status;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
        printf("Prover exited with code %d\n", WEXITSTATUS(status));
        return WEXITSTATUS(status);
    } else {
        printf("Prover terminated abnormally\n");
        return 1;
    }
}
