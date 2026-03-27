// bench_field_arith: microbenchmark comparing field multiplication throughput
// at different field sizes (255-bit BLS12-381, 64-bit Goldilocks, 31-bit Mersenne31).
//
// Measures raw multiply throughput on H100 CUDA cores to validate the
// claimed 10-25x speedup from smaller fields.
//
// Usage: ./bench_field_arith [num_elements=1000000] [iterations=100]

#include "bls12-381.cuh"
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// ═══════════════════════════════════════════════════════════════════════════
// Goldilocks field: p = 2^64 - 2^32 + 1
// ═══════════════════════════════════════════════════════════════════════════

typedef uint64_t Gold_t;
static const uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

__device__ __forceinline__
Gold_t gold_mul(Gold_t a, Gold_t b) {
    // Full 128-bit product, then reduce mod p = 2^64 - 2^32 + 1
    unsigned long long hi, lo;
    lo = a * b;  // low 64 bits
    hi = __umul64hi(a, b);  // high 64 bits

    // Reduce: value = hi * 2^64 + lo
    // 2^64 ≡ 2^32 - 1 (mod p)
    // So hi * 2^64 ≡ hi * (2^32 - 1) = hi * 2^32 - hi
    uint64_t hi_shifted = (hi << 32) - hi;  // hi * (2^32 - 1)

    // Now compute lo + hi_shifted mod p
    uint64_t sum = lo + hi_shifted;
    bool carry = (sum < lo);

    // If carry, add 2^64 mod p = 2^32 - 1
    if (carry) {
        sum += 0xFFFFFFFFULL; // 2^32 - 1
        if (sum < 0xFFFFFFFFULL) sum += 0xFFFFFFFFULL; // handle double carry
    }

    // Final reduction
    if (sum >= GOLDILOCKS_P) sum -= GOLDILOCKS_P;
    return sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Mersenne31 field: p = 2^31 - 1
// ═══════════════════════════════════════════════════════════════════════════

typedef uint32_t M31_t;
static const uint32_t MERSENNE31_P = 0x7FFFFFFFU; // 2^31 - 1

__device__ __forceinline__
M31_t m31_mul(M31_t a, M31_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    // Reduce mod 2^31 - 1: split into high and low 31-bit parts
    uint32_t lo = (uint32_t)(prod & MERSENNE31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t sum = lo + hi;
    if (sum >= MERSENNE31_P) sum -= MERSENNE31_P;
    return sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernels: repeated multiply-accumulate to measure throughput
// ═══════════════════════════════════════════════════════════════════════════

// BLS12-381 scalar field (255-bit)
__global__
void bench_bls_kernel(blstrs__scalar__Scalar* data, uint n, uint iters) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    blstrs__scalar__Scalar a = data[idx];
    blstrs__scalar__Scalar b = data[(idx + 1) % n];

    for (uint i = 0; i < iters; i++) {
        a = blstrs__scalar__Scalar_mul(a, b);
    }
    data[idx] = a;  // prevent optimization
}

// Goldilocks (64-bit)
__global__
void bench_gold_kernel(Gold_t* data, uint n, uint iters) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Gold_t a = data[idx];
    Gold_t b = data[(idx + 1) % n];

    for (uint i = 0; i < iters; i++) {
        a = gold_mul(a, b);
    }
    data[idx] = a;
}

// Mersenne31 (31-bit)
__global__
void bench_m31_kernel(M31_t* data, uint n, uint iters) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    M31_t a = data[idx];
    M31_t b = data[(idx + 1) % n];

    for (uint i = 0; i < iters; i++) {
        a = m31_mul(a, b);
    }
    data[idx] = a;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    uint n     = argc > 1 ? (uint)atoi(argv[1]) : 1000000u;
    uint iters = argc > 2 ? (uint)atoi(argv[2]) : 100u;

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    cout << "Field arithmetic benchmark" << endl;
    cout << "  Elements: " << n << "  Iterations: " << iters << endl;
    cout << "  Blocks: " << blocks << "  Threads/block: " << threads << endl;
    cout << "  Total multiplications per run: " << (double)n * iters << endl;
    cout << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // ── BLS12-381 (255-bit) ─────────────────────────────────────────────────
    {
        blstrs__scalar__Scalar* d_data;
        cudaMalloc(&d_data, n * sizeof(blstrs__scalar__Scalar));
        // Initialize with non-zero values
        blstrs__scalar__Scalar* h_data = new blstrs__scalar__Scalar[n];
        for (uint i = 0; i < n; i++) {
            for (int j = 0; j < 8; j++) h_data[i].val[j] = (i + 1) * 17 + j;
        }
        cudaMemcpy(d_data, h_data, n * sizeof(blstrs__scalar__Scalar), cudaMemcpyHostToDevice);

        // Warmup
        bench_bls_kernel<<<blocks, threads>>>(d_data, n, 10);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        bench_bls_kernel<<<blocks, threads>>>(d_data, n, iters);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        double total_muls = (double)n * iters;
        double muls_per_sec = total_muls / (ms / 1000.0);
        cout << "BLS12-381 (255-bit):" << endl;
        cout << "  Time: " << ms << " ms" << endl;
        cout << "  Throughput: " << muls_per_sec / 1e9 << " billion muls/sec" << endl;
        cout << endl;

        delete[] h_data;
        cudaFree(d_data);
    }

    // ── Goldilocks (64-bit) ─────────────────────────────────────────────────
    {
        Gold_t* d_data;
        cudaMalloc(&d_data, n * sizeof(Gold_t));
        Gold_t* h_data = new Gold_t[n];
        for (uint i = 0; i < n; i++) h_data[i] = (uint64_t)(i + 1) * 17 + 3;
        cudaMemcpy(d_data, h_data, n * sizeof(Gold_t), cudaMemcpyHostToDevice);

        bench_gold_kernel<<<blocks, threads>>>(d_data, n, 10);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        bench_gold_kernel<<<blocks, threads>>>(d_data, n, iters);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        double total_muls = (double)n * iters;
        double muls_per_sec = total_muls / (ms / 1000.0);
        cout << "Goldilocks (64-bit):" << endl;
        cout << "  Time: " << ms << " ms" << endl;
        cout << "  Throughput: " << muls_per_sec / 1e9 << " billion muls/sec" << endl;
        cout << endl;

        delete[] h_data;
        cudaFree(d_data);
    }

    // ── Mersenne31 (31-bit) ─────────────────────────────────────────────────
    {
        M31_t* d_data;
        cudaMalloc(&d_data, n * sizeof(M31_t));
        M31_t* h_data = new M31_t[n];
        for (uint i = 0; i < n; i++) h_data[i] = (i + 1) * 17 + 3;
        cudaMemcpy(d_data, h_data, n * sizeof(M31_t), cudaMemcpyHostToDevice);

        bench_m31_kernel<<<blocks, threads>>>(d_data, n, 10);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        bench_m31_kernel<<<blocks, threads>>>(d_data, n, iters);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        double total_muls = (double)n * iters;
        double muls_per_sec = total_muls / (ms / 1000.0);
        cout << "Mersenne31 (31-bit):" << endl;
        cout << "  Time: " << ms << " ms" << endl;
        cout << "  Throughput: " << muls_per_sec / 1e9 << " billion muls/sec" << endl;
        cout << endl;

        delete[] h_data;
        cudaFree(d_data);
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    cout << "Done." << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
