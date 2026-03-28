// Merkle tree implementation using SHA-256 on GPU.

#include "commit/merkle.cuh"
#include <cstring>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════════════════
// SHA-256 implementation (GPU device code)
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __constant__ uint32_t K_MERKLE[64] = {
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

__device__ __forceinline__
uint32_t mk_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ __forceinline__
uint32_t mk_ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }

__device__ __forceinline__
uint32_t mk_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

__device__ __forceinline__
uint32_t mk_ep0(uint32_t x) { return mk_rotr(x, 2) ^ mk_rotr(x, 13) ^ mk_rotr(x, 22); }

__device__ __forceinline__
uint32_t mk_ep1(uint32_t x) { return mk_rotr(x, 6) ^ mk_rotr(x, 11) ^ mk_rotr(x, 25); }

__device__ __forceinline__
uint32_t mk_sig0(uint32_t x) { return mk_rotr(x, 7) ^ mk_rotr(x, 18) ^ (x >> 3); }

__device__ __forceinline__
uint32_t mk_sig1(uint32_t x) { return mk_rotr(x, 17) ^ mk_rotr(x, 19) ^ (x >> 10); }

// Compress a single 64-byte block with given initial state
__device__
Hash256 sha256_compress_block(const uint32_t* block16, const Hash256& iv) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = block16[i];
    for (int i = 16; i < 64; i++)
        w[i] = mk_sig1(w[i-2]) + w[i-7] + mk_sig0(w[i-15]) + w[i-16];

    uint32_t a = iv.h[0], b = iv.h[1], c = iv.h[2], d = iv.h[3];
    uint32_t e = iv.h[4], f = iv.h[5], g = iv.h[6], h = iv.h[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + mk_ep1(e) + mk_ch(e, f, g) + K_MERKLE[i] + w[i];
        uint32_t t2 = mk_ep0(a) + mk_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    Hash256 out;
    out.h[0] = iv.h[0]+a; out.h[1] = iv.h[1]+b;
    out.h[2] = iv.h[2]+c; out.h[3] = iv.h[3]+d;
    out.h[4] = iv.h[4]+e; out.h[5] = iv.h[5]+f;
    out.h[6] = iv.h[6]+g; out.h[7] = iv.h[7]+h;
    return out;
}

__device__ const Hash256 SHA256_IV = {
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}
};

// Hash a single Goldilocks field element (8 bytes) with SHA-256
__device__
Hash256 sha256_hash_fr(const Fr_t& val) {
    uint32_t block[16];
    // Clear block
    for (int i = 0; i < 16; i++) block[i] = 0;

#ifdef USE_GOLDILOCKS
    // Goldilocks: 8 bytes (1 x uint64_t)
    block[0] = (uint32_t)(val.val >> 32);  // big-endian for SHA
    block[1] = (uint32_t)(val.val & 0xFFFFFFFF);
    // Padding: 0x80 byte after 8 bytes of data
    block[2] = 0x80000000u;
    // Length in bits at end: 64 bits = 0x40
    block[15] = 64;
#else
    // BLS12-381: 32 bytes (8 x uint32_t)
    for (int i = 0; i < 8; i++) block[i] = val.val[i];
    block[8] = 0x80000000u;
    block[15] = 256;
#endif

    return sha256_compress_block(block, SHA256_IV);
}

// Hash two Hash256 values together (for internal Merkle nodes)
// Input: 64 bytes of data = exactly one SHA-256 block, then a padding block
__device__
Hash256 sha256_hash_pair(const Hash256& left, const Hash256& right) {
    // Block 1: left || right (64 bytes)
    uint32_t block1[16];
    for (int i = 0; i < 8; i++) block1[i] = left.h[i];
    for (int i = 0; i < 8; i++) block1[8 + i] = right.h[i];

    Hash256 mid = sha256_compress_block(block1, SHA256_IV);

    // Block 2: padding for 512-bit (64-byte) message
    uint32_t block2[16];
    block2[0] = 0x80000000u;
    for (int i = 1; i < 15; i++) block2[i] = 0;
    block2[15] = 512;  // length in bits

    return sha256_compress_block(block2, mid);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Merkle tree kernels
// ═══════════════════════════════════════════════════════════════════════════════

__global__
void merkle_leaf_hash_kernel(const Fr_t* data, Hash256* hashes, uint n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        hashes[idx] = sha256_hash_fr(data[idx]);
    }
}

__global__
void merkle_internal_hash_kernel(const Hash256* children, Hash256* parents, uint n_parents) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_parents) {
        parents[idx] = sha256_hash_pair(children[2 * idx], children[2 * idx + 1]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MerkleTree class implementation
// ═══════════════════════════════════════════════════════════════════════════════

static const uint MERKLE_THREADS = 256;

MerkleTree::MerkleTree(const Fr_t* gpu_data, uint n) : n_(n) {
    // Verify n is power of 2
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("MerkleTree: n must be a power of 2");
    }

    log_n_ = 0;
    uint tmp = n;
    while (tmp > 1) { tmp >>= 1; log_n_++; }

    // Total tree nodes: n (leaves) + n/2 + n/4 + ... + 1 = 2n - 1
    uint total_nodes = 2 * n - 1;

    cudaMalloc(&tree_gpu_, total_nodes * sizeof(Hash256));

    // Hash leaves
    uint blocks = (n + MERKLE_THREADS - 1) / MERKLE_THREADS;
    merkle_leaf_hash_kernel<<<blocks, MERKLE_THREADS>>>(gpu_data, tree_gpu_, n);
    cudaDeviceSynchronize();

    // Build internal levels
    Hash256* current_level = tree_gpu_;
    uint level_size = n;
    uint offset = 0;

    for (uint level = 0; level < log_n_; level++) {
        uint parent_size = level_size / 2;
        Hash256* parent_level = tree_gpu_ + offset + level_size;

        uint pblocks = (parent_size + MERKLE_THREADS - 1) / MERKLE_THREADS;
        merkle_internal_hash_kernel<<<pblocks, MERKLE_THREADS>>>(current_level, parent_level, parent_size);
        cudaDeviceSynchronize();

        offset += level_size;
        current_level = parent_level;
        level_size = parent_size;
    }

    // Copy entire tree to host for proof generation
    tree_host_ = new Hash256[total_nodes];
    cudaMemcpy(tree_host_, tree_gpu_, total_nodes * sizeof(Hash256), cudaMemcpyDeviceToHost);
}

MerkleTree::~MerkleTree() {
    if (tree_gpu_) cudaFree(tree_gpu_);
    if (tree_host_) delete[] tree_host_;
}

Hash256 MerkleTree::root() const {
    // Root is the last element
    return tree_host_[2 * n_ - 2];
}

MerkleProof MerkleTree::prove(uint leaf_index) const {
    if (leaf_index >= n_) {
        throw std::runtime_error("MerkleTree::prove: leaf_index out of range");
    }

    MerkleProof proof;
    proof.leaf_index = leaf_index;

    uint idx = leaf_index;
    uint offset = 0;
    uint level_size = n_;

    for (uint level = 0; level < log_n_; level++) {
        // Sibling index
        uint sibling = (idx ^ 1);
        proof.path.push_back(tree_host_[offset + sibling]);

        // Move to parent
        idx >>= 1;
        offset += level_size;
        level_size >>= 1;
    }

    return proof;
}

// Host-side SHA-256 for verification
namespace {

static const uint32_t H_K[64] = {
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

inline uint32_t h_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t h_ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t h_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t h_ep0(uint32_t x) { return h_rotr(x, 2) ^ h_rotr(x, 13) ^ h_rotr(x, 22); }
inline uint32_t h_ep1(uint32_t x) { return h_rotr(x, 6) ^ h_rotr(x, 11) ^ h_rotr(x, 25); }
inline uint32_t h_sig0(uint32_t x) { return h_rotr(x, 7) ^ h_rotr(x, 18) ^ (x >> 3); }
inline uint32_t h_sig1(uint32_t x) { return h_rotr(x, 17) ^ h_rotr(x, 19) ^ (x >> 10); }

Hash256 host_sha256_compress(const uint32_t* block16, const Hash256& iv) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = block16[i];
    for (int i = 16; i < 64; i++)
        w[i] = h_sig1(w[i-2]) + w[i-7] + h_sig0(w[i-15]) + w[i-16];

    uint32_t a = iv.h[0], b = iv.h[1], c = iv.h[2], d = iv.h[3];
    uint32_t e = iv.h[4], f = iv.h[5], g = iv.h[6], h = iv.h[7];

    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + h_ep1(e) + h_ch(e, f, g) + H_K[i] + w[i];
        uint32_t t2 = h_ep0(a) + h_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    Hash256 out;
    out.h[0] = iv.h[0]+a; out.h[1] = iv.h[1]+b;
    out.h[2] = iv.h[2]+c; out.h[3] = iv.h[3]+d;
    out.h[4] = iv.h[4]+e; out.h[5] = iv.h[5]+f;
    out.h[6] = iv.h[6]+g; out.h[7] = iv.h[7]+h;
    return out;
}

Hash256 host_sha256_fr(const Fr_t& val) {
    const Hash256 iv = {{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    uint32_t block[16];
    for (int i = 0; i < 16; i++) block[i] = 0;

#ifdef USE_GOLDILOCKS
    block[0] = (uint32_t)(val.val >> 32);
    block[1] = (uint32_t)(val.val & 0xFFFFFFFF);
    block[2] = 0x80000000u;
    block[15] = 64;
#else
    for (int i = 0; i < 8; i++) block[i] = val.val[i];
    block[8] = 0x80000000u;
    block[15] = 256;
#endif

    return host_sha256_compress(block, iv);
}

Hash256 host_sha256_pair(const Hash256& left, const Hash256& right) {
    const Hash256 iv = {{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};

    uint32_t block1[16];
    for (int i = 0; i < 8; i++) block1[i] = left.h[i];
    for (int i = 0; i < 8; i++) block1[8 + i] = right.h[i];
    Hash256 mid = host_sha256_compress(block1, iv);

    uint32_t block2[16];
    block2[0] = 0x80000000u;
    for (int i = 1; i < 15; i++) block2[i] = 0;
    block2[15] = 512;
    return host_sha256_compress(block2, mid);
}

} // anonymous namespace

bool MerkleTree::verify(const Hash256& root, const Fr_t& leaf_value, const MerkleProof& proof, uint n) {
    // Compute leaf hash
    Hash256 current = host_sha256_fr(leaf_value);

    uint idx = proof.leaf_index;
    for (uint level = 0; level < proof.path.size(); level++) {
        if (idx & 1) {
            // current is right child
            current = host_sha256_pair(proof.path[level], current);
        } else {
            // current is left child
            current = host_sha256_pair(current, proof.path[level]);
        }
        idx >>= 1;
    }

    return current == root;
}
