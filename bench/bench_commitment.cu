// bench_commitment: compare Pedersen commitment vs SHA-256 Merkle tree
// commitment throughput for the same vector of field elements.
//
// Pedersen: elliptic curve multi-scalar multiplication (current scheme)
// SHA-256 Merkle: hash each element, then binary tree of hashes (post-quantum)
//
// Usage: ./bench_commitment [num_elements=131072] [num_trials=3]

#include "commit/commitment.cuh"
#include "tensor/fr-tensor.cuh"
#include "util/timer.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

// ═══════════════════════════════════════════════════════════════════════════
// SHA-256 implementation (minimal, for benchmarking only)
// ═══════════════════════════════════════════════════════════════════════════

// SHA-256 constants
__device__ __constant__ uint32_t K_SHA[64] = {
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

struct SHA256Hash {
    uint32_t h[8];
};

__device__ __forceinline__
uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ __forceinline__
uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }

__device__ __forceinline__
uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

__device__ __forceinline__
uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }

__device__ __forceinline__
uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }

__device__ __forceinline__
uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }

__device__ __forceinline__
uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

// SHA-256 compress a single 64-byte block (already padded)
__device__
SHA256Hash sha256_compress(const uint32_t* block16) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = block16[i];
    for (int i = 16; i < 64; i++)
        w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K_SHA[i] + w[i];
        uint32_t t2 = ep0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    SHA256Hash out;
    out.h[0] = 0x6a09e667 + a; out.h[1] = 0xbb67ae85 + b;
    out.h[2] = 0x3c6ef372 + c; out.h[3] = 0xa54ff53a + d;
    out.h[4] = 0x510e527f + e; out.h[5] = 0x9b05688c + f;
    out.h[6] = 0x1f83d9ab + g; out.h[7] = 0x5be0cd19 + h;
    return out;
}

// Hash a single Fr_t element (32 bytes) with SHA-256 padding
__device__
SHA256Hash sha256_fr(const Fr_t& val) {
    // Fr_t is 32 bytes = 256 bits. One SHA-256 block is 64 bytes.
    // Message: 32 bytes data + 0x80 + 0x00...0x00 + 8 bytes length
    uint32_t block[16];
    // Copy 32 bytes of Fr_t (8 x 4-byte limbs)
    for (int i = 0; i < 8; i++) block[i] = val.val[i];
    // Padding: 0x80 byte followed by zeros
    block[8]  = 0x80000000u;
    for (int i = 9; i < 15; i++) block[i] = 0;
    // Length in bits: 256 = 0x100
    block[15] = 256;
    return sha256_compress(block);
}

// Hash two SHA256Hash values together (for Merkle tree internal nodes)
__device__
SHA256Hash sha256_pair(const SHA256Hash& left, const SHA256Hash& right) {
    // 64 bytes of data (two hashes) = exactly one block after we add another
    // block for padding. For simplicity, compress in two blocks.

    // Block 1: left.h[0..7] + right.h[0..7] = 64 bytes exactly
    uint32_t block1[16];
    for (int i = 0; i < 8; i++) block1[i] = left.h[i];
    for (int i = 0; i < 8; i++) block1[8 + i] = right.h[i];

    // First block compression with standard IV
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = block1[i];
    for (int i = 16; i < 64; i++)
        w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K_SHA[i] + w[i];
        uint32_t t2 = ep0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    uint32_t s0 = 0x6a09e667+a, s1 = 0xbb67ae85+b, s2 = 0x3c6ef372+c, s3 = 0xa54ff53a+d;
    uint32_t s4 = 0x510e527f+e, s5 = 0x9b05688c+f, s6 = 0x1f83d9ab+g, s7 = 0x5be0cd19+h;

    // Block 2: padding only (message was 512 bits = 64 bytes)
    uint32_t block2[16];
    block2[0] = 0x80000000u;
    for (int i = 1; i < 15; i++) block2[i] = 0;
    block2[15] = 512; // length in bits

    for (int i = 0; i < 16; i++) w[i] = block2[i];
    for (int i = 16; i < 64; i++)
        w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];

    a = s0; b = s1; c = s2; d = s3;
    e = s4; f = s5; g = s6; h = s7;
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + ep1(e) + ch(e, f, g) + K_SHA[i] + w[i];
        uint32_t t2 = ep0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    SHA256Hash out;
    out.h[0] = s0+a; out.h[1] = s1+b; out.h[2] = s2+c; out.h[3] = s3+d;
    out.h[4] = s4+e; out.h[5] = s5+f; out.h[6] = s6+g; out.h[7] = s7+h;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Merkle tree kernels
// ═══════════════════════════════════════════════════════════════════════════

// Leaf layer: hash each Fr_t element
__global__
void merkle_leaf_kernel(const Fr_t* data, SHA256Hash* leaves, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        leaves[idx] = sha256_fr(data[idx]);
    }
}

// Internal layer: hash pairs of children
__global__
void merkle_internal_kernel(const SHA256Hash* children, SHA256Hash* parents, uint n_parents) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_parents) {
        parents[idx] = sha256_pair(children[2 * idx], children[2 * idx + 1]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main benchmark
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    uint n      = argc > 1 ? (uint)atoi(argv[1]) : 131072u;  // default: 2^17
    uint trials = argc > 2 ? (uint)atoi(argv[2]) : 3u;

    // Round up to power of 2 for Merkle tree
    uint n_padded = 1;
    while (n_padded < n) n_padded <<= 1;

    int threads = 256;

    cout << "Commitment benchmark: Pedersen vs SHA-256 Merkle tree" << endl;
    cout << "  Elements: " << n << " (padded to " << n_padded << ")" << endl;
    cout << "  Element size: " << sizeof(Fr_t) << " bytes (BLS12-381 scalar)" << endl;
    cout << "  Total data: " << (double)n * sizeof(Fr_t) / (1024*1024) << " MB" << endl;
    cout << "  Trials: " << trials << endl;
    cout << endl;

    // ── Create test data ────────────────────────────────────────────────────
    FrTensor data = FrTensor::random_int(n, 16);

    // ── Benchmark Pedersen commitment ───────────────────────────────────────
    {
        cout << "=== Pedersen Commitment (EC multi-scalar mul) ===" << endl;
        Commitment generators = Commitment::random(n);

        // Warmup
        G1TensorJacobian result = generators.commit_int(data);
        cudaDeviceSynchronize();

        Timer timer;
        for (uint t = 0; t < trials; t++) {
            timer.start();
            G1TensorJacobian r = generators.commit_int(data);
            cudaDeviceSynchronize();
            timer.stop();
        }
        double avg = timer.getTotalTime() / trials;
        cout << "  Avg time: " << avg << " s" << endl;
        cout << "  Throughput: " << (double)n / avg / 1e6 << " M elements/s" << endl;
        cout << endl;
    }

    // ── Benchmark SHA-256 Merkle tree ───────────────────────────────────────
    {
        cout << "=== SHA-256 Merkle Tree ===" << endl;

        // Allocate leaf and internal node buffers
        // We need two buffers and ping-pong between them
        SHA256Hash* d_buf_a;
        SHA256Hash* d_buf_b;
        cudaMalloc(&d_buf_a, n_padded * sizeof(SHA256Hash));
        cudaMalloc(&d_buf_b, n_padded * sizeof(SHA256Hash));

        // If n < n_padded, zero-pad the extra leaves
        if (n < n_padded) {
            cudaMemset(d_buf_a + n, 0, (n_padded - n) * sizeof(SHA256Hash));
        }

        // Warmup
        {
            int blocks_leaf = (n_padded + threads - 1) / threads;
            merkle_leaf_kernel<<<blocks_leaf, threads>>>(data.gpu_data, d_buf_a, n);
            if (n < n_padded) {
                // Hash zero-padding for remaining leaves
                // (they're already zeroed, just need to be valid hashes)
            }
            cudaDeviceSynchronize();
            uint level_size = n_padded;
            SHA256Hash* src = d_buf_a;
            SHA256Hash* dst = d_buf_b;
            while (level_size > 1) {
                uint parents = level_size / 2;
                int blocks_int = (parents + threads - 1) / threads;
                merkle_internal_kernel<<<blocks_int, threads>>>(src, dst, parents);
                cudaDeviceSynchronize();
                level_size = parents;
                SHA256Hash* tmp = src; src = dst; dst = tmp;
            }
        }

        Timer timer;
        for (uint t = 0; t < trials; t++) {
            timer.start();

            // Leaf hashing
            int blocks_leaf = (n_padded + threads - 1) / threads;
            merkle_leaf_kernel<<<blocks_leaf, threads>>>(data.gpu_data, d_buf_a, n);
            cudaDeviceSynchronize();

            // Internal node hashing (tree reduction)
            uint level_size = n_padded;
            SHA256Hash* src = d_buf_a;
            SHA256Hash* dst = d_buf_b;
            while (level_size > 1) {
                uint parents = level_size / 2;
                int blocks_int = (parents + threads - 1) / threads;
                merkle_internal_kernel<<<blocks_int, threads>>>(src, dst, parents);
                cudaDeviceSynchronize();
                level_size = parents;
                SHA256Hash* tmp = src; src = dst; dst = tmp;
            }

            timer.stop();
        }
        double avg = timer.getTotalTime() / trials;
        cout << "  Avg time: " << avg << " s" << endl;
        cout << "  Throughput: " << (double)n / avg / 1e6 << " M elements/s" << endl;
        cout << endl;

        cudaFree(d_buf_a);
        cudaFree(d_buf_b);
    }

    cout << "Done." << endl;
    return 0;
}
