// challenge.cuh — Pluggable challenge generation for interactive proofs
//
// ChallengeSource is the abstract interface for generating verifier challenges.
// In testing / self-verification mode, LocalChallengeSource generates random
// challenges locally.  In interactive mode, PipeChallengeSource reads challenges
// from the verifier over a pipe (after sending proof elements).
//
// The global challenge source is used by random_vec() in fr-tensor.cu.
// When set, random_vec() delegates to the global source; when null, it uses
// the original std::random_device implementation.

#ifndef CHALLENGE_CUH
#define CHALLENGE_CUH

#include "tensor/fr-tensor.cuh"
#include <random>
#include <vector>

// ── Abstract challenge source ───────────────────────────────────────────────

class ChallengeSource {
public:
    // Generate a single random field element (challenge).
    virtual Fr_t next() = 0;

    // Generate n challenges.  Default implementation calls next() n times.
    virtual std::vector<Fr_t> next_vec(uint n) {
        std::vector<Fr_t> v(n);
        for (uint i = 0; i < n; i++) v[i] = next();
        return v;
    }

    virtual ~ChallengeSource() = default;
};

// ── Local (prover-side) challenge source ────────────────────────────────────
// Uses std::random_device + mt19937, same as the original random_vec().

class LocalChallengeSource : public ChallengeSource {
    std::mt19937 mt;
    std::uniform_int_distribution<unsigned int> dist;
public:
    LocalChallengeSource()
        : mt(std::random_device{}()), dist(0, UINT_MAX) {}

    Fr_t next() override {
#ifdef USE_GOLDILOCKS
        uint64_t lo = dist(mt), hi = dist(mt);
        return {(hi << 32 | lo) % GOLDILOCKS_P};
#else
        return {dist(mt), dist(mt), dist(mt), dist(mt),
                dist(mt), dist(mt), dist(mt), dist(mt) % 1944954707};
#endif
    }
};

// ── Global challenge source ─────────────────────────────────────────────────
// When set, random_vec() uses this source instead of creating its own RNG.
// Ownership is NOT transferred — caller must keep the source alive.

void set_challenge_source(ChallengeSource* src);
ChallengeSource* get_challenge_source();

#endif // CHALLENGE_CUH
