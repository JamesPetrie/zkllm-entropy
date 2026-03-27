// Goldilocks prime field: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
//
// This header provides field arithmetic with the same interface as bls12-381.cuh
// so that the rest of the codebase (fr-tensor, proof, etc.) can use either field
// via a typedef switch.

#ifndef GOLDILOCKS_CUH
#define GOLDILOCKS_CUH

#include <cstdint>

// ── Platform macros (shared with bls12-381.cuh) ─────────────────────────────
#ifndef DEVICE
#ifdef __NVCC__
  #define DEVICE __device__
  #define GLOBAL
  #define KERNEL extern "C" __global__
  #define LOCAL __shared__
  #define CONSTANT __constant__
  #define GET_GLOBAL_ID() (blockIdx.x * blockDim.x + threadIdx.x)
  #define GET_GROUP_ID() blockIdx.x
  #define GET_LOCAL_ID() threadIdx.x
  #define GET_LOCAL_SIZE() blockDim.x
  #define BARRIER_LOCAL() __syncthreads()
  typedef unsigned char uchar;
#endif
#endif

// ── Field constants ─────────────────────────────────────────────────────────

static const uint64_t GOLDILOCKS_P     = 0xFFFFFFFF00000001ULL;
static const uint64_t GOLDILOCKS_P_NEG = 0x00000000FFFFFFFFULL;  // 2^64 - P = 2^32 - 1

// ── Field element type ──────────────────────────────────────────────────────
// Single uint64_t, matching the BLS interface pattern of a struct with val[].
// We keep val[1] as a dummy so that code using val[0] and val[1] for
// reading 64-bit quantities (e.g. entropy_val construction) works unchanged.
// Arithmetic only uses val[0].

typedef struct {
    uint64_t val;
} Gold_t;

// ── Constants (device) ──────────────────────────────────────────────────────

extern CONSTANT Gold_t gold_ZERO;   // {0}
extern CONSTANT Gold_t gold_ONE;    // {1}
extern CONSTANT Gold_t gold_P;      // {GOLDILOCKS_P}

// ── Core arithmetic ─────────────────────────────────────────────────────────

DEVICE inline Gold_t gold_add(Gold_t a, Gold_t b) {
    uint64_t sum = a.val + b.val;
    // If sum wrapped around or sum >= P, reduce
    uint64_t reduced = sum - GOLDILOCKS_P;
    // Use carry: if sum < a.val (overflow) or sum >= P, use reduced
    bool overflow = (sum < a.val);
    bool ge_p = (sum >= GOLDILOCKS_P);
    return Gold_t{(overflow || ge_p) ? reduced : sum};
}

DEVICE inline Gold_t gold_sub(Gold_t a, Gold_t b) {
    uint64_t diff = a.val - b.val;
    // If a < b, we wrapped; add P back
    bool borrow = (a.val < b.val);
    return Gold_t{borrow ? diff + GOLDILOCKS_P : diff};
}

DEVICE inline Gold_t gold_double(Gold_t a) {
    return gold_add(a, a);
}

DEVICE inline Gold_t gold_mul(Gold_t a, Gold_t b) {
    // Full 128-bit product, then reduce mod p = 2^64 - 2^32 + 1
    // Since 2^64 ≡ 2^32 - 1 (mod p), we have:
    //   (hi * 2^64 + lo) ≡ hi * (2^32 - 1) + lo (mod p)
    unsigned long long lo = a.val * b.val;
    unsigned long long hi = __umul64hi(a.val, b.val);

    // Reduce: result = lo + hi * (2^32 - 1)  mod p
    // hi * (2^32 - 1) = hi * 2^32 - hi = (hi << 32) - hi
    uint64_t hi_lo = (uint32_t)hi;              // low 32 bits of hi
    uint64_t hi_hi = hi >> 32;                   // high 32 bits of hi

    // hi * (2^32 - 1) = hi_hi * 2^64 * (something)... let's be more careful.
    // Actually: hi < 2^64, so hi * (2^32 - 1) < 2^96.
    // We need to handle this in steps.
    //
    // Let hi = hi_hi * 2^32 + hi_lo
    // hi * 2^32 = hi_hi * 2^64 + hi_lo * 2^32
    // hi * (2^32 - 1) = hi_hi * 2^64 + hi_lo * 2^32 - hi
    //                  = hi_hi * 2^64 + (hi_lo * 2^32 - hi_hi - hi_lo) + carry adjustments
    //
    // This is getting complicated. Use a simpler approach:
    // result = lo + hi * EPSILON where EPSILON = 2^32 - 1
    // Since hi < p and EPSILON < 2^32, hi * EPSILON < 2^96
    // But we can split: hi * EPSILON = hi_hi_part * 2^64 + lo_part
    // And 2^64 ≡ EPSILON (mod p), so recursively reduce.

    // Step 1: compute hi * EPSILON as 128-bit
    uint64_t eps = GOLDILOCKS_P_NEG;  // 2^32 - 1
    uint64_t t_lo = hi * eps;
    uint64_t t_hi = __umul64hi(hi, eps);

    // Step 2: add lo + t_lo (with carry)
    uint64_t sum = lo + t_lo;
    uint64_t carry1 = (sum < lo) ? 1ULL : 0ULL;

    // Step 3: total high part = t_hi + carry1
    uint64_t total_hi = t_hi + carry1;

    // Step 4: reduce total_hi * 2^64 ≡ total_hi * EPSILON (mod p)
    // Since total_hi < 2^33 (because hi < 2^64 and eps < 2^32, so t_hi < 2^32,
    // plus carry is at most 1), total_hi * eps < 2^65.
    // So total_hi * eps fits in 128 bits but the high part is at most 1 bit.
    uint64_t r_lo = total_hi * eps;
    uint64_t r_hi = __umul64hi(total_hi, eps);  // 0 or very small

    // Step 5: sum + r_lo
    uint64_t result = sum + r_lo;
    uint64_t carry2 = (result < sum) ? 1ULL : 0ULL;
    uint64_t final_hi = r_hi + carry2;

    // Step 6: if there's still a high part, reduce once more
    // final_hi is at most ~2, so final_hi * eps fits in 64 bits
    result += final_hi * eps;

    // Final canonical reduction
    if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;

    return Gold_t{result};
}

DEVICE inline Gold_t gold_sqr(Gold_t a) {
    return gold_mul(a, a);
}

// ── Montgomery form ─────────────────────────────────────────────────────────
// Goldilocks doesn't need Montgomery form for efficiency (the modular
// reduction is already fast). We keep mont/unmont as identity operations
// for interface compatibility with code that does:
//   blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a, b))

DEVICE inline Gold_t gold_mont(Gold_t a) {
    return a;  // identity — no Montgomery form needed
}

DEVICE inline Gold_t gold_unmont(Gold_t a) {
    return a;  // identity
}

// ── Exponentiation and inverse ──────────────────────────────────────────────

DEVICE inline Gold_t gold_pow(Gold_t base, Gold_t exp) {
    Gold_t result = {1ULL};
    Gold_t b = base;
    uint64_t e = exp.val;
    while (e > 0) {
        if (e & 1) result = gold_mul(result, b);
        b = gold_sqr(b);
        e >>= 1;
    }
    return result;
}

DEVICE inline Gold_t gold_inverse(Gold_t a) {
    // Fermat's little theorem: a^(-1) = a^(p-2) mod p
    Gold_t exp = {GOLDILOCKS_P - 2};
    return gold_pow(a, exp);
}

// ── Division ────────────────────────────────────────────────────────────────

DEVICE inline Gold_t gold_div(Gold_t a, Gold_t b) {
    return gold_mul(a, gold_inverse(b));
}

// ── Comparison ──────────────────────────────────────────────────────────────

DEVICE inline bool gold_eq(Gold_t a, Gold_t b) {
    return a.val == b.val;
}

DEVICE inline bool gold_gte(Gold_t a, Gold_t b) {
    return a.val >= b.val;
}

// ── Bit operations ──────────────────────────────────────────────────────────

DEVICE inline uint gold_get_bit(Gold_t a, uint bit) {
    return (a.val >> bit) & 1;
}

DEVICE inline uint gold_get_bits(Gold_t a, uint start, uint count) {
    return (uint)((a.val >> start) & ((1ULL << count) - 1));
}

// ── Alias macros ────────────────────────────────────────────────────────────
// Map the blstrs__scalar__Scalar_* names to gold_* so consumer code compiles
// unchanged.

#define blstrs__scalar__Scalar_limb     uint64_t
#define blstrs__scalar__Scalar_LIMBS    1
#define blstrs__scalar__Scalar_LIMB_BITS 64
#define blstrs__scalar__Scalar_BITS     64
#define blstrs__scalar__Scalar_INV      0

#define blstrs__scalar__Scalar          Gold_t

#define blstrs__scalar__Scalar_ZERO     gold_ZERO
#define blstrs__scalar__Scalar_ONE      gold_ONE
#define blstrs__scalar__Scalar_P        gold_P
#define blstrs__scalar__Scalar_R2       gold_ONE

#define blstrs__scalar__Scalar_add      gold_add
#define blstrs__scalar__Scalar_sub      gold_sub
#define blstrs__scalar__Scalar_mul      gold_mul
#define blstrs__scalar__Scalar_sqr      gold_sqr
#define blstrs__scalar__Scalar_double   gold_double
#define blstrs__scalar__Scalar_mont     gold_mont
#define blstrs__scalar__Scalar_unmont   gold_unmont
#define blstrs__scalar__Scalar_pow      gold_pow
#define blstrs__scalar__Scalar_inverse  gold_inverse
#define blstrs__scalar__Scalar_div      gold_div
#define blstrs__scalar__Scalar_eq       gold_eq
#define blstrs__scalar__Scalar_gte      gold_gte
#define blstrs__scalar__Scalar_get_bit  gold_get_bit
#define blstrs__scalar__Scalar_get_bits gold_get_bits

// Reverse bits (used by FFT kernel in bls12-381.cu, may not be needed)
DEVICE inline uint bitreverse(uint n, uint bits) {
    uint r = 0;
    for (uint i = 0; i < bits; i++) {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}

#endif // GOLDILOCKS_CUH
