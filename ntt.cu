// NTT implementation for Goldilocks field.
//
// Uses iterative Cooley-Tukey radix-2 DIT with bit-reversal permutation.
// Each butterfly stage is a separate kernel launch for simplicity.

#include "ntt.cuh"
#include <cstdio>

// ── Constants ────────────────────────────────────────────────────────────────

// Goldilocks multiplicative group generator
static const uint64_t GOLDILOCKS_GENERATOR = 7ULL;

// ── Host-side modular arithmetic (for precomputation) ────────────────────────

static uint64_t host_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    if (sum < a || sum >= GOLDILOCKS_P) sum -= GOLDILOCKS_P;
    return sum;
}

static uint64_t host_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + GOLDILOCKS_P - b);
}

static uint64_t host_mul(uint64_t a, uint64_t b) {
    // Use __uint128_t for 128-bit product
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);

    // Reduce: hi * 2^64 + lo ≡ hi * (2^32 - 1) + lo (mod p)
    uint64_t eps = GOLDILOCKS_P_NEG;  // 2^32 - 1

    __uint128_t t = (__uint128_t)hi * eps + lo;
    uint64_t r_lo = (uint64_t)t;
    uint64_t r_hi = (uint64_t)(t >> 64);

    // One more reduction step
    uint64_t result = r_lo + r_hi * eps;
    if (result < r_lo) result += eps;  // overflow

    if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;
    return result;
}

static uint64_t host_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    base %= GOLDILOCKS_P;
    while (exp > 0) {
        if (exp & 1) result = host_mul(result, base);
        base = host_mul(base, base);
        exp >>= 1;
    }
    return result;
}

static uint64_t host_inverse(uint64_t a) {
    return host_pow(a, GOLDILOCKS_P - 2);
}

// ── Root of unity ────────────────────────────────────────────────────────────

Fr_t get_root_of_unity(uint log_n) {
    if (log_n > 32) {
        throw std::runtime_error("NTT: log_n > 32 not supported (Goldilocks has 2^32 roots of unity)");
    }
    // omega = g^((p-1) / 2^log_n) where g = 7
    uint64_t exponent = (GOLDILOCKS_P - 1) >> log_n;
    uint64_t omega = host_pow(GOLDILOCKS_GENERATOR, exponent);
    return Fr_t{omega};
}

// ── Bit-reversal permutation kernel ──────────────────────────────────────────

__global__ void ntt_bit_reverse_kernel(Fr_t* data, uint n, uint log_n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Compute bit-reverse of idx
    uint rev = 0;
    uint tmp = idx;
    for (uint i = 0; i < log_n; i++) {
        rev = (rev << 1) | (tmp & 1);
        tmp >>= 1;
    }

    // Only swap if rev > idx (to avoid double-swapping)
    if (rev > idx) {
        Fr_t t = data[idx];
        data[idx] = data[rev];
        data[rev] = t;
    }
}

// ── Butterfly kernel ─────────────────────────────────────────────────────────
// One butterfly stage of the Cooley-Tukey DIT NTT.
// stage s: butterfly groups of size 2^(s+1), half-group = 2^s
// twiddle factors: omega^(n/2^(s+1) * j) for j in [0, 2^s)

__global__ void ntt_butterfly_kernel(Fr_t* data, const Fr_t* twiddles, uint n, uint half_group) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint num_butterflies = n / 2;
    if (idx >= num_butterflies) return;

    uint group_size = half_group * 2;
    uint group = idx / half_group;
    uint j = idx % half_group;

    uint i0 = group * group_size + j;
    uint i1 = i0 + half_group;

    Fr_t u = data[i0];
    Fr_t v = blstrs__scalar__Scalar_mul(data[i1], twiddles[j]);

    data[i0] = blstrs__scalar__Scalar_add(u, v);
    data[i1] = blstrs__scalar__Scalar_sub(u, v);
}

// ── Scale kernel (for inverse NTT) ──────────────────────────────────────────

__global__ void ntt_scale_kernel(Fr_t* data, Fr_t scale, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = blstrs__scalar__Scalar_mul(data[idx], scale);
}

// ── Coset shift kernel ──────────────────────────────────────────────────────

__global__ void ntt_coset_shift_kernel(Fr_t* data, const Fr_t* shift_powers, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = blstrs__scalar__Scalar_mul(data[idx], shift_powers[idx]);
}

// ── NTT implementation ──────────────────────────────────────────────────────

static const uint NTT_THREADS = 256;

void ntt_forward(Fr_t* data, uint log_n) {
    uint n = 1u << log_n;

    // Bit-reversal permutation
    uint blocks = (n + NTT_THREADS - 1) / NTT_THREADS;
    ntt_bit_reverse_kernel<<<blocks, NTT_THREADS>>>(data, n, log_n);

    // Get the principal n-th root of unity
    uint64_t omega = get_root_of_unity(log_n).val;

    // Butterfly stages
    for (uint s = 0; s < log_n; s++) {
        uint half_group = 1u << s;
        uint group_size = half_group * 2;

        // Precompute twiddle factors for this stage: omega^(n/group_size * j) for j in [0, half_group)
        // = omega_stage^j where omega_stage = omega^(n/group_size)
        uint64_t omega_stage = host_pow(omega, n / group_size);

        std::vector<Fr_t> twiddles_host(half_group);
        uint64_t w = 1;
        for (uint j = 0; j < half_group; j++) {
            twiddles_host[j] = Fr_t{w};
            w = host_mul(w, omega_stage);
        }

        // Upload twiddles to GPU
        Fr_t* twiddles_gpu;
        cudaMalloc(&twiddles_gpu, half_group * sizeof(Fr_t));
        cudaMemcpy(twiddles_gpu, twiddles_host.data(), half_group * sizeof(Fr_t), cudaMemcpyHostToDevice);

        // Launch butterfly kernel
        uint num_butterflies = n / 2;
        uint bblocks = (num_butterflies + NTT_THREADS - 1) / NTT_THREADS;
        ntt_butterfly_kernel<<<bblocks, NTT_THREADS>>>(data, twiddles_gpu, n, half_group);
        cudaDeviceSynchronize();

        cudaFree(twiddles_gpu);
    }
}

void ntt_inverse(Fr_t* data, uint log_n) {
    uint n = 1u << log_n;

    // Bit-reversal permutation
    uint blocks = (n + NTT_THREADS - 1) / NTT_THREADS;
    ntt_bit_reverse_kernel<<<blocks, NTT_THREADS>>>(data, n, log_n);

    // Use the inverse root: omega_inv = omega^(-1)
    uint64_t omega = get_root_of_unity(log_n).val;
    uint64_t omega_inv = host_inverse(omega);

    // Butterfly stages with inverse root
    for (uint s = 0; s < log_n; s++) {
        uint half_group = 1u << s;
        uint group_size = half_group * 2;

        uint64_t omega_stage = host_pow(omega_inv, n / group_size);

        std::vector<Fr_t> twiddles_host(half_group);
        uint64_t w = 1;
        for (uint j = 0; j < half_group; j++) {
            twiddles_host[j] = Fr_t{w};
            w = host_mul(w, omega_stage);
        }

        Fr_t* twiddles_gpu;
        cudaMalloc(&twiddles_gpu, half_group * sizeof(Fr_t));
        cudaMemcpy(twiddles_gpu, twiddles_host.data(), half_group * sizeof(Fr_t), cudaMemcpyHostToDevice);

        uint num_butterflies = n / 2;
        uint bblocks = (num_butterflies + NTT_THREADS - 1) / NTT_THREADS;
        ntt_butterfly_kernel<<<bblocks, NTT_THREADS>>>(data, twiddles_gpu, n, half_group);
        cudaDeviceSynchronize();

        cudaFree(twiddles_gpu);
    }

    // Scale by 1/n
    uint64_t n_inv = host_inverse((uint64_t)n);
    Fr_t scale = Fr_t{n_inv};
    ntt_scale_kernel<<<blocks, NTT_THREADS>>>(data, scale, n);
    cudaDeviceSynchronize();
}

void ntt_coset_forward(Fr_t* data, uint log_n, Fr_t shift) {
    uint n = 1u << log_n;

    // Precompute shift powers: shift^0, shift^1, ..., shift^(n-1)
    std::vector<Fr_t> powers(n);
    uint64_t w = 1;
    for (uint i = 0; i < n; i++) {
        powers[i] = Fr_t{w};
        w = host_mul(w, shift.val);
    }

    Fr_t* powers_gpu;
    cudaMalloc(&powers_gpu, n * sizeof(Fr_t));
    cudaMemcpy(powers_gpu, powers.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

    // Multiply data by shift powers
    uint blocks = (n + NTT_THREADS - 1) / NTT_THREADS;
    ntt_coset_shift_kernel<<<blocks, NTT_THREADS>>>(data, powers_gpu, n);
    cudaDeviceSynchronize();

    cudaFree(powers_gpu);

    // Standard NTT
    ntt_forward(data, log_n);
}

void ntt_coset_inverse(Fr_t* data, uint log_n, Fr_t shift) {
    // Inverse NTT
    ntt_inverse(data, log_n);

    uint n = 1u << log_n;

    // Divide by shift powers: multiply by shift^(-i) = shift_inv^i
    uint64_t shift_inv = host_inverse(shift.val);
    std::vector<Fr_t> powers(n);
    uint64_t w = 1;
    for (uint i = 0; i < n; i++) {
        powers[i] = Fr_t{w};
        w = host_mul(w, shift_inv);
    }

    Fr_t* powers_gpu;
    cudaMalloc(&powers_gpu, n * sizeof(Fr_t));
    cudaMemcpy(powers_gpu, powers.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

    uint blocks = (n + NTT_THREADS - 1) / NTT_THREADS;
    ntt_coset_shift_kernel<<<blocks, NTT_THREADS>>>(data, powers_gpu, n);
    cudaDeviceSynchronize();

    cudaFree(powers_gpu);
}
