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
  #define DEVICE __host__ __device__
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


// Portable 64x64->high64 multiply (works on host and device)
#ifdef __CUDA_ARCH__
  #define UMUL64HI(a, b) __umul64hi((a), (b))
#else
  #define UMUL64HI(a, b) ((uint64_t)(((__uint128_t)(a) * (b)) >> 64))
#endif

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
    // Full 128-bit product: (hi, lo) = a * b
    uint64_t lo = a.val * b.val;
    uint64_t hi = UMUL64HI(a.val, b.val);

    // Reduce mod p = 2^64 - 2^32 + 1 using 2^64 ≡ (2^32 - 1) mod p
    // So: (hi * 2^64 + lo) ≡ lo + hi * (2^32 - 1) mod p
    //
    // Key trick: hi * (2^32 - 1) = (hi << 32) - hi, avoiding a second UMUL64HI.
    // Split hi into hi_hi (top 32 bits) and hi_lo (bottom 32 bits):
    //   hi << 32 = (hi_hi << 64) + (hi_lo << 32)
    //   2^64 ≡ (2^32 - 1) mod p, so hi_hi << 64 ≡ hi_hi * (2^32 - 1) mod p
    //
    // Since hi < p < 2^64, hi_hi < 2^32, so hi_hi * (2^32-1) < 2^64 — no overflow.

    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo = hi & 0xFFFFFFFFULL;

    // Compute: result = lo + (hi_lo << 32) - hi + hi_hi * (2^32 - 1)
    // Rewrite hi_hi * (2^32-1) = (hi_hi << 32) - hi_hi
    // So: result = lo + (hi_lo << 32) - hi + (hi_hi << 32) - hi_hi
    //            = lo + ((hi_lo + hi_hi) << 32) - hi - hi_hi
    //            = lo + (hi << 32 handled via parts) ... let's just be explicit:

    // Step 1: t = hi * (2^32 - 1) = (hi << 32) - hi
    // We split this to avoid 128-bit: hi << 32 = hi_lo << 32 (fits 64-bit) + hi_hi << 64 (overflow)
    uint64_t t_lo = (hi_lo << 32) - hi;  // This can underflow (borrow)
    bool borrow = (hi_lo << 32) < hi;
    // The overflow part: hi_hi contributes hi_hi words of 2^64, which ≡ hi_hi * (2^32-1)
    // Minus the borrow
    uint64_t t_hi = hi_hi - (borrow ? 1ULL : 0ULL);  // hi_hi is at most 2^32-1, so this is fine

    // Step 2: result = lo + t_lo (with carry)
    uint64_t result = lo + t_lo;
    uint64_t carry = (result < lo) ? 1ULL : 0ULL;

    // Step 3: add overflow: (t_hi + carry) * (2^32 - 1)
    // t_hi + carry < 2^32 + 1, so (t_hi + carry) * (2^32-1) < 2^64
    uint64_t overflow = (t_hi + carry) * (uint64_t)GOLDILOCKS_P_NEG;
    result += overflow;

    // One more carry possible
    if (result < overflow) {
        // Wrapped: add another (2^32 - 1) for the 2^64 overflow
        result += GOLDILOCKS_P_NEG;
    }

    // Final canonical reduction (at most one subtraction needed)
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

#define blstrs__scalar__Scalar_ZERO     Gold_t{0ULL}
#define blstrs__scalar__Scalar_ONE      Gold_t{1ULL}
#define blstrs__scalar__Scalar_P        Gold_t{GOLDILOCKS_P}
#define blstrs__scalar__Scalar_R2       Gold_t{1ULL}

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

// ── Initializer compatibility macros ────────────────────────────────────────
// BLS12-381 code uses {val, 0, 0, 0, 0, 0, 0, 0} to create Fr_t literals.
// These macros let the same pattern work for Goldilocks.
#define FR_ZERO Gold_t{0ULL}
#define FR_ONE  Gold_t{1ULL}
#define FR_FROM_U32(x) Gold_t{(uint64_t)(x)}
// Accept 8-arg initializer: use only first arg (cast to uint64_t)
#define FR_LITERAL(a, ...) Gold_t{(uint64_t)(a)}

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
