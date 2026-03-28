// pipe_challenge.cuh — Pipe-based challenge source for interactive proving
//
// PipeChallengeSource sends proof polynomials to the verifier over a pipe
// and receives challenges back.  The verifier checks each polynomial before
// generating the next challenge.
//
// Wire protocol (binary, Goldilocks field elements are 8 bytes):
//   Prover → Verifier:
//     'P' <uint32_t n_coeffs> <n_coeffs * sizeof(Fr_t) bytes>   (polynomial)
//     'E'                                                        (end of proof)
//   Verifier → Prover:
//     <sizeof(Fr_t) bytes>                                       (challenge)

#ifndef PIPE_CHALLENGE_CUH
#define PIPE_CHALLENGE_CUH

#include "proof/challenge.cuh"
#include "poly/polynomial.cuh"
#include <unistd.h>
#include <stdexcept>
#include <cstring>

class PipeChallengeSource : public ChallengeSource {
    int read_fd;   // read challenges from verifier
    int write_fd;  // write proof elements to verifier
    std::vector<Polynomial>* proof_ptr;  // current proof vector
    size_t last_sent;  // number of polys already sent

    void flush_pending() {
        if (!proof_ptr) return;
        while (last_sent < proof_ptr->size()) {
            send_poly((*proof_ptr)[last_sent]);
            last_sent++;
        }
    }

    void send_poly(const Polynomial& p) {
        char tag = 'P';
        write_all(&tag, 1);

        int deg = p.getDegree();
        uint32_t n = (deg >= 0) ? (uint32_t)(deg + 1) : 0u;
        write_all(&n, sizeof(n));

        for (uint32_t k = 0; k < n; k++) {
            Fr_t xk = FR_FROM_INT(k);
            Fr_t yk = const_cast<Polynomial&>(p)(xk);
            write_all(&yk, sizeof(Fr_t));
        }
    }

    void write_all(const void* buf, size_t len) {
        const char* p = (const char*)buf;
        while (len > 0) {
            ssize_t n = ::write(write_fd, p, len);
            if (n <= 0) throw std::runtime_error("PipeChallengeSource: write failed");
            p += n;
            len -= n;
        }
    }

    void read_all(void* buf, size_t len) {
        char* p = (char*)buf;
        while (len > 0) {
            ssize_t n = ::read(read_fd, p, len);
            if (n <= 0) throw std::runtime_error("PipeChallengeSource: read failed");
            p += n;
            len -= n;
        }
    }

public:
    PipeChallengeSource(int read_fd, int write_fd)
        : read_fd(read_fd), write_fd(write_fd),
          proof_ptr(nullptr), last_sent(0) {}

    // Set the proof vector to monitor. Call this before each proof section.
    void set_proof(std::vector<Polynomial>* proof) {
        proof_ptr = proof;
        last_sent = proof ? proof->size() : 0;
    }

    Fr_t next() override {
        // Send any pending proof polynomials to verifier
        flush_pending();

        // Receive challenge from verifier
        Fr_t challenge;
        read_all(&challenge, sizeof(Fr_t));
        return challenge;
    }

    // Signal end of proof to verifier
    void signal_end() {
        flush_pending();
        char tag = 'E';
        write_all(&tag, 1);
    }
};

#endif // PIPE_CHALLENGE_CUH
