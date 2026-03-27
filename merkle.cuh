// Merkle tree commitment for field elements.
//
// Builds a binary hash tree using SHA-256. Supports:
// - Building the full tree from a vector of field elements
// - Generating authentication paths (proofs) for individual leaves
// - Verifying authentication paths
//
// For FRI, we commit to polynomial evaluations and later open specific positions.

#ifndef MERKLE_CUH
#define MERKLE_CUH

#include "fr-tensor.cuh"
#include <vector>
#include <cstdint>

// ── Hash type ────────────────────────────────────────────────────────────────

struct Hash256 {
    uint32_t h[8];  // 256-bit hash
};

// Compare two hashes
inline bool operator==(const Hash256& a, const Hash256& b) {
    for (int i = 0; i < 8; i++) if (a.h[i] != b.h[i]) return false;
    return true;
}

inline bool operator!=(const Hash256& a, const Hash256& b) {
    return !(a == b);
}

// ── Merkle tree ──────────────────────────────────────────────────────────────

struct MerkleProof {
    std::vector<Hash256> path;  // sibling hashes, bottom to top
    uint leaf_index;
};

class MerkleTree {
public:
    // Build from GPU data (n elements, must be power of 2)
    MerkleTree(const Fr_t* gpu_data, uint n);
    ~MerkleTree();

    // Get the root hash
    Hash256 root() const;

    // Generate a proof for the given leaf index
    MerkleProof prove(uint leaf_index) const;

    // Verify a proof against a root and leaf value
    static bool verify(const Hash256& root, const Fr_t& leaf_value, const MerkleProof& proof, uint n);

    // Number of leaves
    uint num_leaves() const { return n_; }

private:
    uint n_;            // number of leaves (power of 2)
    uint log_n_;        // log2(n)
    Hash256* tree_gpu_; // full tree on GPU: tree[0..n-1] = leaves, tree[n..2n-2] = internal, tree[2n-2] = root
                        // Actually stored as: level 0 (leaves) = [0, n), level 1 = [n, n + n/2), etc.
    Hash256* tree_host_; // copy on host for proof generation
};

// ── GPU SHA-256 device functions (exposed for testing) ───────────────────────

__device__ Hash256 sha256_hash_fr(const Fr_t& val);
__device__ Hash256 sha256_hash_pair(const Hash256& left, const Hash256& right);

#endif // MERKLE_CUH
