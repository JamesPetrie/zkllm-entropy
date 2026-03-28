// test_merkle: verify Merkle tree commitment, proof generation, and verification.
//
// Tests:
// 1. Build tree from known data, verify root is deterministic
// 2. Generate and verify proofs for all leaves of a small tree
// 3. Verify that wrong leaf value fails verification
// 4. Verify that wrong index fails verification
// 5. Larger tree (2^16) proof round-trip
//
// Usage: ./test_merkle

#include "commit/merkle.cuh"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    cout << "=== Merkle Tree Tests (Goldilocks + SHA-256) ===" << endl;
    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: Deterministic root ──────────────────────────────────────
    {
        uint n = 8;
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i + 1)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        MerkleTree tree1(gpu_data, n);
        MerkleTree tree2(gpu_data, n);

        Hash256 r1 = tree1.root();
        Hash256 r2 = tree2.root();

        check(r1 == r2, "deterministic root: same data -> same root");

        // Modify one element and check root changes
        data[3] = Fr_t{999ULL};
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);
        MerkleTree tree3(gpu_data, n);
        Hash256 r3 = tree3.root();
        check(r1 != r3, "different data -> different root");

        cudaFree(gpu_data);
    }

    // ── Test 2: Verify proofs for all leaves (n=8) ──────────────────────
    {
        uint n = 8;
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i * 7 + 3)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        MerkleTree tree(gpu_data, n);
        Hash256 root = tree.root();

        bool all_ok = true;
        for (uint i = 0; i < n; i++) {
            MerkleProof proof = tree.prove(i);
            if (!MerkleTree::verify(root, data[i], proof, n)) {
                cout << "    Failed for leaf " << i << endl;
                all_ok = false;
            }
        }
        check(all_ok, "all 8 leaf proofs verify");

        // Check proof path length
        MerkleProof p = tree.prove(0);
        check(p.path.size() == 3, "proof path length == log2(8) == 3");

        cudaFree(gpu_data);
    }

    // ── Test 3: Wrong leaf value fails ──────────────────────────────────
    {
        uint n = 4;
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i + 10)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        MerkleTree tree(gpu_data, n);
        Hash256 root = tree.root();

        MerkleProof proof = tree.prove(2);
        // Correct value should verify
        check(MerkleTree::verify(root, data[2], proof, n), "correct value verifies");
        // Wrong value should fail
        Fr_t wrong = Fr_t{999ULL};
        check(!MerkleTree::verify(root, wrong, proof, n), "wrong value fails");

        cudaFree(gpu_data);
    }

    // ── Test 4: Wrong index fails ───────────────────────────────────────
    {
        uint n = 4;
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i + 20)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        MerkleTree tree(gpu_data, n);
        Hash256 root = tree.root();

        MerkleProof proof = tree.prove(0);
        // Verify with leaf 0's value but claim it's at index 1
        // (Tamper with the proof's leaf_index)
        MerkleProof tampered = proof;
        tampered.leaf_index = 1;
        check(!MerkleTree::verify(root, data[0], tampered, n), "wrong index fails");

        cudaFree(gpu_data);
    }

    // ── Test 5: Larger tree (n=2^16) ────────────────────────────────────
    {
        uint n = 1 << 16;
        vector<Fr_t> data(n);
        for (uint i = 0; i < n; i++) data[i] = Fr_t{(uint64_t)(i % 10007)};

        Fr_t* gpu_data;
        cudaMalloc(&gpu_data, n * sizeof(Fr_t));
        cudaMemcpy(gpu_data, data.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        MerkleTree tree(gpu_data, n);
        Hash256 root = tree.root();

        // Verify a few random positions
        bool all_ok = true;
        uint test_indices[] = {0, 1, 100, 1000, 32767, 65535};
        for (uint idx : test_indices) {
            MerkleProof proof = tree.prove(idx);
            if (!MerkleTree::verify(root, data[idx], proof, n)) {
                cout << "    Failed for leaf " << idx << endl;
                all_ok = false;
            }
        }
        check(all_ok, "2^16 tree: 6 random proofs verify");

        MerkleProof p = tree.prove(0);
        check(p.path.size() == 16, "proof path length == log2(2^16) == 16");

        cudaFree(gpu_data);
    }

    cout << "\n=== Results: " << (failures == 0 ? "ALL PASSED" : "FAILURES")
         << " (failures=" << failures << ") ===" << endl;
    return failures;
}
