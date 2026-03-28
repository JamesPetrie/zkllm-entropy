// verifier_utils.h — Host-side Goldilocks field arithmetic and proof parsing
//
// CPU-only header for the verifier. No CUDA dependency.
// Provides Goldilocks field operations, proof file reading, and utility types.

#ifndef VERIFIER_UTILS_H
#define VERIFIER_UTILS_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

// ── Goldilocks field constants ──────────────────────────────────────────────

static const uint64_t GOLDILOCKS_P     = 0xFFFFFFFF00000001ULL;
static const uint64_t GOLDILOCKS_P_NEG = 0x00000000FFFFFFFFULL;  // 2^64 - P = 2^32 - 1
static const uint64_t GOLDILOCKS_P_HALF = GOLDILOCKS_P / 2;

// ── Field element type ──────────────────────────────────────────────────────

struct Fr_t {
    uint64_t val;
};

static const Fr_t FR_ZERO = {0ULL};
static const Fr_t FR_ONE  = {1ULL};

// ── Portable 64x64 → high 64 multiply ──────────────────────────────────────

static inline uint64_t umul64hi(uint64_t a, uint64_t b) {
    return (uint64_t)(((__uint128_t)a * b) >> 64);
}

// ── Field arithmetic ────────────────────────────────────────────────────────
// Mirrors goldilocks.cuh exactly, but without __host__ __device__ qualifiers.

static inline Fr_t fr_add(Fr_t a, Fr_t b) {
    uint64_t sum = a.val + b.val;
    uint64_t reduced = sum - GOLDILOCKS_P;
    bool overflow = (sum < a.val);
    bool ge_p = (sum >= GOLDILOCKS_P);
    return Fr_t{(overflow || ge_p) ? reduced : sum};
}

static inline Fr_t fr_sub(Fr_t a, Fr_t b) {
    uint64_t diff = a.val - b.val;
    bool borrow = (a.val < b.val);
    return Fr_t{borrow ? diff + GOLDILOCKS_P : diff};
}

static inline Fr_t fr_neg(Fr_t a) {
    if (a.val == 0) return a;
    return Fr_t{GOLDILOCKS_P - a.val};
}

static inline Fr_t fr_mul(Fr_t a, Fr_t b) {
    uint64_t lo = a.val * b.val;
    uint64_t hi = umul64hi(a.val, b.val);
    uint64_t eps = GOLDILOCKS_P_NEG;

    uint64_t t_lo = hi * eps;
    uint64_t t_hi = umul64hi(hi, eps);

    uint64_t sum = lo + t_lo;
    uint64_t carry1 = (sum < lo) ? 1ULL : 0ULL;
    uint64_t total_hi = t_hi + carry1;

    uint64_t r_lo = total_hi * eps;
    uint64_t r_hi = umul64hi(total_hi, eps);

    uint64_t result = sum + r_lo;
    uint64_t carry2 = (result < sum) ? 1ULL : 0ULL;
    uint64_t final_hi = r_hi + carry2;

    result += final_hi * eps;
    if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;
    return Fr_t{result};
}

static inline Fr_t fr_sqr(Fr_t a) { return fr_mul(a, a); }

static inline Fr_t fr_pow(Fr_t base, uint64_t exp) {
    Fr_t result = FR_ONE;
    Fr_t b = base;
    while (exp > 0) {
        if (exp & 1) result = fr_mul(result, b);
        b = fr_sqr(b);
        exp >>= 1;
    }
    return result;
}

static inline Fr_t fr_inverse(Fr_t a) {
    return fr_pow(a, GOLDILOCKS_P - 2);
}

static inline Fr_t fr_div(Fr_t a, Fr_t b) {
    return fr_mul(a, fr_inverse(b));
}

static inline bool fr_eq(Fr_t a, Fr_t b) { return a.val == b.val; }

static inline Fr_t fr_from_u64(uint64_t x) {
    return Fr_t{x % GOLDILOCKS_P};
}

// Check if a field element represents a "negative" value (> p/2)
static inline bool fr_is_negative(Fr_t a) {
    return a.val > GOLDILOCKS_P_HALF;
}

// ── Operators (convenience) ─────────────────────────────────────────────────

static inline Fr_t operator+(Fr_t a, Fr_t b) { return fr_add(a, b); }
static inline Fr_t operator-(Fr_t a, Fr_t b) { return fr_sub(a, b); }
static inline Fr_t operator*(Fr_t a, Fr_t b) { return fr_mul(a, b); }
static inline bool operator==(Fr_t a, Fr_t b) { return fr_eq(a, b); }
static inline bool operator!=(Fr_t a, Fr_t b) { return !fr_eq(a, b); }

// ── Polynomial type ─────────────────────────────────────────────────────────

struct Polynomial {
    std::vector<Fr_t> coeffs;

    Fr_t eval(Fr_t x) const {
        // Horner's method
        if (coeffs.empty()) return FR_ZERO;
        Fr_t result = coeffs.back();
        for (int i = (int)coeffs.size() - 2; i >= 0; i--) {
            result = fr_add(fr_mul(result, x), coeffs[i]);
        }
        return result;
    }

    // Constant (degree-0) value
    Fr_t constant() const {
        return coeffs.empty() ? FR_ZERO : coeffs[0];
    }
};

// ── Multilinear evaluation ──────────────────────────────────────────────────
// Evaluate multilinear extension of data[] at point u[].
// data has 2^n entries, u has n entries.
// MLE(u) = sum_{x in {0,1}^n} data[x] * prod_i ((1-u_i)(1-x_i) + u_i*x_i)

static inline Fr_t mle_eval(const std::vector<Fr_t>& data, const std::vector<Fr_t>& u) {
    if (data.empty()) return FR_ZERO;
    // Iterative folding: fold one dimension at a time
    std::vector<Fr_t> vals = data;
    for (size_t i = 0; i < u.size(); i++) {
        size_t half = vals.size() / 2;
        if (half == 0) break;
        std::vector<Fr_t> next(half);
        Fr_t ui = u[i];
        Fr_t one_minus_ui = fr_sub(FR_ONE, ui);
        for (size_t j = 0; j < half; j++) {
            // next[j] = vals[2j] * (1 - u_i) + vals[2j+1] * u_i
            next[j] = fr_add(fr_mul(vals[2*j], one_minus_ui), fr_mul(vals[2*j+1], ui));
        }
        vals = std::move(next);
    }
    return vals[0];
}

// ── eq polynomial ───────────────────────────────────────────────────────────
// eq(u, v) = prod_i ((1-u_i)(1-v_i) + u_i*v_i)

static inline Fr_t eq_eval(const std::vector<Fr_t>& u, const std::vector<Fr_t>& v) {
    Fr_t result = FR_ONE;
    size_t n = std::min(u.size(), v.size());
    for (size_t i = 0; i < n; i++) {
        // (1 - u_i)(1 - v_i) + u_i * v_i = 1 - u_i - v_i + 2*u_i*v_i
        Fr_t ui_vi = fr_mul(u[i], v[i]);
        Fr_t term = fr_add(fr_sub(fr_sub(FR_ONE, u[i]), v[i]), fr_add(ui_vi, ui_vi));
        result = fr_mul(result, term);
    }
    return result;
}

// ── SHA-256 (host-side, for Merkle verification) ────────────────────────────
// Minimal SHA-256 implementation for Merkle proof verification.

namespace sha256_impl {

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
static inline uint32_t sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
static inline uint32_t sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
static inline uint32_t gamma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
static inline uint32_t gamma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

// Compress a single 64-byte (16-word) block
static inline void compress(uint32_t state[8], const uint32_t block[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = block[i];
    for (int i = 16; i < 64; i++)
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

} // namespace sha256_impl

struct Hash256 {
    uint32_t words[8];
    bool operator==(const Hash256& o) const { return memcmp(words, o.words, 32) == 0; }
    bool operator!=(const Hash256& o) const { return !(*this == o); }
};

// Hash a single Goldilocks field element (8 bytes) with SHA-256
static inline Hash256 sha256_field_element(Fr_t val) {
    uint32_t block[16] = {};
    block[0] = (uint32_t)(val.val >> 32);
    block[1] = (uint32_t)(val.val & 0xFFFFFFFF);
    block[2] = 0x80000000u;  // padding
    block[15] = 64;          // message length in bits

    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    sha256_impl::compress(state, block);
    Hash256 h;
    memcpy(h.words, state, 32);
    return h;
}

// Hash two Hash256 values (internal Merkle node)
static inline Hash256 sha256_pair(Hash256 left, Hash256 right) {
    // 64 bytes of data = exactly one SHA-256 block after padding
    // But 64 bytes + padding > 64, so we need two blocks.
    // Actually: 64 bytes data + 1 byte 0x80 + padding + 8 bytes length = need 2 blocks.
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Block 1: 16 words = first hash (8 words) + second hash (8 words)
    uint32_t block1[16];
    memcpy(block1, left.words, 32);
    memcpy(block1 + 8, right.words, 32);
    sha256_impl::compress(state, block1);

    // Block 2: padding
    uint32_t block2[16] = {};
    block2[0] = 0x80000000u;
    block2[15] = 512;  // 64 bytes = 512 bits
    sha256_impl::compress(state, block2);

    Hash256 h;
    memcpy(h.words, state, 32);
    return h;
}

// ── Merkle proof verification ───────────────────────────────────────────────

struct MerkleProof {
    uint32_t leaf_index;
    std::vector<Hash256> path;  // sibling hashes from leaf to root
};

static inline bool merkle_verify(Hash256 root, Hash256 leaf_hash,
                                 uint32_t leaf_index,
                                 const std::vector<Hash256>& path) {
    Hash256 current = leaf_hash;
    uint32_t idx = leaf_index;
    for (size_t level = 0; level < path.size(); level++) {
        if (idx & 1) {
            current = sha256_pair(path[level], current);
        } else {
            current = sha256_pair(current, path[level]);
        }
        idx >>= 1;
    }
    return current == root;
}

// ── Proof file parsing ──────────────────────────────────────────────────────

static const uint64_t PROOF_MAGIC = 0x5A4B454E54524F50ULL;  // "ZKENTROP"

struct ProofHeader {
    uint64_t entropy_val;
    uint32_t T;             // sequence length
    uint32_t vocab_size;
    double   sigma_eff;
    uint32_t log_scale;
    // v2 fields (may be defaults if v1)
    uint32_t cdf_precision;
    uint32_t log_precision;
    uint32_t cdf_scale;
    bool     is_v2;
};

struct PositionProof {
    // 8 constant polynomial values per position
    Fr_t ind_sum;
    Fr_t ind_dot;
    Fr_t logit_act;
    Fr_t diff_actual;
    Fr_t win_prob;
    Fr_t total_win;
    Fr_t q_fr;
    Fr_t surprise;
};

struct ParsedProof {
    ProofHeader header;
    std::vector<PositionProof> positions;
    std::vector<Polynomial> all_polys;  // raw polynomial data
};

static inline Fr_t read_fr(std::ifstream& f) {
    uint64_t val;
    f.read(reinterpret_cast<char*>(&val), 8);
    if (!f.good()) throw std::runtime_error("Unexpected end of proof file reading Fr_t");
    return Fr_t{val};
}

static inline uint32_t read_u32(std::ifstream& f) {
    uint32_t val;
    f.read(reinterpret_cast<char*>(&val), 4);
    if (!f.good()) throw std::runtime_error("Unexpected end of proof file reading uint32");
    return val;
}

static inline uint64_t read_u64(std::ifstream& f) {
    uint64_t val;
    f.read(reinterpret_cast<char*>(&val), 8);
    if (!f.good()) throw std::runtime_error("Unexpected end of proof file reading uint64");
    return val;
}

static inline double read_f64(std::ifstream& f) {
    double val;
    f.read(reinterpret_cast<char*>(&val), 8);
    if (!f.good()) throw std::runtime_error("Unexpected end of proof file reading double");
    return val;
}

static inline Hash256 read_hash256(std::ifstream& f) {
    Hash256 h;
    f.read(reinterpret_cast<char*>(h.words), 32);
    if (!f.good()) throw std::runtime_error("Unexpected end of proof file reading Hash256");
    return h;
}

static inline ParsedProof parse_proof_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open proof file: " + path);

    ParsedProof proof;

    // Header
    uint64_t magic = read_u64(f);
    if (magic != PROOF_MAGIC) {
        throw std::runtime_error("Bad magic number in proof file");
    }

    proof.header.entropy_val = read_u64(f);
    proof.header.T           = read_u32(f);
    proof.header.vocab_size  = read_u32(f);
    proof.header.sigma_eff   = read_f64(f);
    proof.header.log_scale   = read_u32(f);

    // Try v2 header detection (same heuristic as verify_entropy.py)
    auto header_pos = f.tellg();
    uint32_t hdr_cdf_prec = 0, hdr_log_prec = 0, hdr_cdf_scale = 0;
    f.read(reinterpret_cast<char*>(&hdr_cdf_prec), 4);
    f.read(reinterpret_cast<char*>(&hdr_log_prec), 4);
    f.read(reinterpret_cast<char*>(&hdr_cdf_scale), 4);

    if (f.good() && hdr_cdf_prec >= 1 && hdr_cdf_prec <= 30 &&
        hdr_log_prec >= 1 && hdr_log_prec <= 30 && hdr_cdf_scale > 0) {
        proof.header.cdf_precision = hdr_cdf_prec;
        proof.header.log_precision = hdr_log_prec;
        proof.header.cdf_scale     = hdr_cdf_scale;
        proof.header.is_v2 = true;
    } else {
        f.seekg(header_pos);
        proof.header.cdf_precision = 15;
        proof.header.log_precision = 15;
        proof.header.cdf_scale     = 65536;
        proof.header.is_v2 = false;
    }

    uint32_t n_polys = read_u32(f);

    // Read all polynomials
    proof.all_polys.resize(n_polys);
    for (uint32_t i = 0; i < n_polys; i++) {
        uint32_t n_coeffs = read_u32(f);
        proof.all_polys[i].coeffs.resize(n_coeffs);
        for (uint32_t j = 0; j < n_coeffs; j++) {
            proof.all_polys[i].coeffs[j] = read_fr(f);
        }
    }

    // Extract per-position data (8 polynomials per position)
    const uint32_t POLYS_PER_POS = 8;
    uint32_t T = proof.header.T;
    proof.positions.resize(T);

    for (uint32_t pos = 0; pos < T; pos++) {
        uint32_t base = pos * POLYS_PER_POS;
        if (base + POLYS_PER_POS > n_polys) {
            throw std::runtime_error("Not enough polynomials for position " + std::to_string(pos));
        }
        proof.positions[pos].ind_sum     = proof.all_polys[base + 0].constant();
        proof.positions[pos].ind_dot     = proof.all_polys[base + 1].constant();
        proof.positions[pos].logit_act   = proof.all_polys[base + 2].constant();
        proof.positions[pos].diff_actual = proof.all_polys[base + 3].constant();
        proof.positions[pos].win_prob    = proof.all_polys[base + 4].constant();
        proof.positions[pos].total_win   = proof.all_polys[base + 5].constant();
        proof.positions[pos].q_fr        = proof.all_polys[base + 6].constant();
        proof.positions[pos].surprise    = proof.all_polys[base + 7].constant();
    }

    return proof;
}

// ── CDF and log table computation ───────────────────────────────────────────
// These are public deterministic functions — the verifier recomputes them.

static inline uint64_t cdf_table_value(uint64_t d, double sigma_eff, uint32_t cdf_scale) {
    if (sigma_eff <= 0.0) {
        return (d >= 0) ? cdf_scale : 0;
    }
    double x = (double)d / sigma_eff;
    double cdf = 0.5 * std::erfc(-x / std::sqrt(2.0));
    return (uint64_t)std::llround(cdf * (double)cdf_scale);
}

static inline uint64_t log_table_value(uint64_t q_fr, uint32_t log_precision, uint32_t log_scale) {
    if (q_fr <= 0) {
        return (uint64_t)std::llround((double)log_precision * (double)log_scale);
    }
    double val = (double)log_precision - std::log2((double)q_fr);
    return (uint64_t)std::llround(val * (double)log_scale);
}

// ── Utility ─────────────────────────────────────────────────────────────────

static inline uint32_t ceil_log2(uint32_t n) {
    if (n <= 1) return 0;
    uint32_t r = 0;
    n--;
    while (n > 0) { n >>= 1; r++; }
    return r;
}

#endif // VERIFIER_UTILS_H
