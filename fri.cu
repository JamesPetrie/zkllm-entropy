// FRI polynomial commitment implementation.
//
// The FRI protocol works as follows:
//
// COMMIT:
// 1. Evaluate polynomial p(x) on coset domain D = {g * omega^i : i in [0, N)}
//    where N = blowup_factor * (degree + 1), rounded up to power of 2
// 2. Commit evaluations via Merkle tree
//
// FOLD (for each round r):
// 3. Given challenge alpha_r, fold: p_{r+1}(x^2) = (p_r(x) + p_r(-x))/2 + alpha * (p_r(x) - p_r(-x))/(2x)
//    In evaluation form: for each pair (f(w^i), f(-w^i)) = (f(w^i), f(w^{i+N/2})):
//      f_new(w^{2i}) = (f(w^i) + f(w^{i+N/2}))/2 + alpha/(2*w^i) * (f(w^i) - f(w^{i+N/2}))
// 4. Commit new evaluations via Merkle tree
// 5. Repeat until polynomial is constant (or below max_remainder_degree)
//
// QUERY:
// 6. For each query position q:
//    - Open position q and its pair q+N/2 in each layer with Merkle proofs
//
// VERIFY:
// 7. Check folding consistency at each query position across layers
// 8. Check final polynomial matches the remainder

#include "fri.cuh"
#include <cstdio>
#include <algorithm>

// ── Host-side Goldilocks arithmetic (same as in ntt.cu) ──────────────────────

static uint64_t fri_mul(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t eps = GOLDILOCKS_P_NEG;
    __uint128_t t = (__uint128_t)hi * eps + lo;
    uint64_t r_lo = (uint64_t)t;
    uint64_t r_hi = (uint64_t)(t >> 64);
    uint64_t result = r_lo + r_hi * eps;
    if (result < r_lo) result += eps;
    if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;
    return result;
}

static uint64_t fri_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a || s >= GOLDILOCKS_P) s -= GOLDILOCKS_P;
    return s;
}

static uint64_t fri_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + GOLDILOCKS_P - b);
}

static uint64_t fri_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result = fri_mul(result, base);
        base = fri_mul(base, base);
        exp >>= 1;
    }
    return result;
}

static uint64_t fri_inv(uint64_t a) {
    return fri_pow(a, GOLDILOCKS_P - 2);
}

// ── FRI folding kernel ───────────────────────────────────────────────────────
// Given evaluations f[0..N-1] on a domain, fold into f_new[0..N/2-1].
// f[i] and f[i + N/2] are evaluations at paired points (omega^i, -omega^i = omega^{i+N/2}).
// Domain point for index i: offset * omega^i  (where omega is the N-th root of unity)
//
// Folding formula:
//   f_new[i] = (f[i] + f[i+half])/2 + alpha * (f[i] - f[i+half]) / (2 * domain_point_i)
//
// We precompute inv_two_domain[i] = 1 / (2 * offset * omega^i) on the host.

__global__
void fri_fold_kernel(const Fr_t* f, Fr_t* f_new, Fr_t alpha, const Fr_t* inv_two_domain, uint half_n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_n) return;

    Fr_t f_pos = f[idx];
    Fr_t f_neg = f[idx + half_n];

    // sum = f_pos + f_neg
    Fr_t sum = blstrs__scalar__Scalar_add(f_pos, f_neg);
    // diff = f_pos - f_neg
    Fr_t diff = blstrs__scalar__Scalar_sub(f_pos, f_neg);

    // half_sum = sum / 2  (we'll multiply by inv_two later)
    // We need: sum/2 + alpha * diff / (2 * x)
    //        = (sum + alpha * diff / x) / 2
    // But it's cleaner to use: (sum/2) + alpha * (diff/(2*x))
    //   where inv_two_domain[i] = 1/(2*x_i)

    // diff_over_2x = diff * inv_two_domain[i]
    Fr_t diff_over_2x = blstrs__scalar__Scalar_mul(diff, inv_two_domain[idx]);

    // alpha_term = alpha * diff_over_2x
    Fr_t alpha_term = blstrs__scalar__Scalar_mul(alpha, diff_over_2x);

    // half_sum: we need sum/2. For Goldilocks, 2^(-1) mod p.
    // Precompute this as a constant. For now, use the field inverse.
    // Actually, let's restructure: result = (sum + alpha * diff * inv_domain) * inv_two
    // where inv_domain = 1/x and inv_two = 1/2.
    // But we already have inv_two_domain = 1/(2x), so:
    // result = sum/2 + alpha * diff * inv_two_domain
    // sum/2: multiply sum by inv_two (constant)

    // For Goldilocks: inv(2) = (p+1)/2
    Fr_t inv_two = {(GOLDILOCKS_P + 1) / 2};
    Fr_t half_sum = blstrs__scalar__Scalar_mul(sum, inv_two);

    f_new[idx] = blstrs__scalar__Scalar_add(half_sum, alpha_term);
}

// ── FRI Prover implementation ────────────────────────────────────────────────

static const uint FRI_THREADS = 256;

FriCommitment FriProver::commit(const Fr_t* coeffs_gpu, uint degree, const FriParams& params) {
    FriCommitment commitment;
    commitment.poly_degree = degree;

    // Compute domain size: next power of 2 >= (degree + 1) * blowup_factor
    uint min_domain = (degree + 1) * params.blowup_factor;
    uint domain_size = 1;
    uint domain_log = 0;
    while (domain_size < min_domain) { domain_size <<= 1; domain_log++; }

    commitment.domain_log_size = domain_log;

    // Coset offset: use the generator of a 2x larger domain
    Fr_t offset = get_root_of_unity(domain_log + 1);
    commitment.domain_offset = offset;

    // Copy coefficients into evaluation buffer (zero-padded to domain_size)
    Fr_t* evals;
    cudaMalloc(&evals, domain_size * sizeof(Fr_t));
    cudaMemset(evals, 0, domain_size * sizeof(Fr_t));
    cudaMemcpy(evals, coeffs_gpu, (degree + 1) * sizeof(Fr_t), cudaMemcpyDeviceToDevice);

    // Coset NTT: evaluate on the coset domain
    ntt_coset_forward(evals, domain_log, offset);

    // Build Merkle tree for the first layer
    MerkleTree* tree = new MerkleTree(evals, domain_size);
    commitment.layer_roots.push_back(tree->root());

    // Store the layer data and trees for proof generation later
    // (In a real implementation, we'd store these more efficiently)
    // For now, we just compute and store roots during commit.

    // We don't fold during commit — folding happens during prove() with challenges.
    // But we do need to store the initial evaluations.

    delete tree;
    cudaFree(evals);

    return commitment;
}

FriProof FriProver::prove(const Fr_t* coeffs_gpu, uint degree,
                          const FriCommitment& commitment,
                          const std::vector<Fr_t>& challenges,
                          const std::vector<uint>& query_positions,
                          const FriParams& params) {
    FriProof proof;
    proof.query_positions = query_positions;

    uint domain_log = commitment.domain_log_size;
    uint domain_size = 1u << domain_log;
    Fr_t offset = commitment.domain_offset;

    // Evaluate polynomial on coset domain
    Fr_t* evals;
    cudaMalloc(&evals, domain_size * sizeof(Fr_t));
    cudaMemset(evals, 0, domain_size * sizeof(Fr_t));
    cudaMemcpy(evals, coeffs_gpu, (degree + 1) * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    ntt_coset_forward(evals, domain_log, offset);

    // Store all layer evaluations and Merkle trees
    struct LayerData {
        Fr_t* evals_gpu;
        MerkleTree* tree;
        uint size;
        std::vector<Fr_t> evals_host;  // host copy for query responses
    };

    std::vector<LayerData> layers;

    // First layer
    {
        LayerData ld;
        ld.size = domain_size;
        ld.evals_gpu = evals;
        ld.tree = new MerkleTree(evals, domain_size);
        ld.evals_host.resize(domain_size);
        cudaMemcpy(ld.evals_host.data(), evals, domain_size * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        layers.push_back(std::move(ld));
    }

    // Folding rounds
    uint current_log = domain_log;
    uint64_t current_offset = offset.val;
    Fr_t* current_evals = evals;

    uint num_rounds = domain_log;  // fold down to size 1
    // But stop when we reach the remainder degree
    // After k folds, polynomial degree is (degree) / 2^k
    // We fold until domain_size / 2^k <= max(1, max_remainder_degree + 1) * blowup
    // Simplification: fold until size is small enough
    uint stop_size = std::max(1u, (params.max_remainder_degree + 1) * params.blowup_factor);
    // Round up to power of 2
    uint stop_log = 0;
    { uint tmp = stop_size; while ((1u << stop_log) < tmp) stop_log++; }
    if (stop_log > domain_log) stop_log = domain_log;
    num_rounds = domain_log - stop_log;

    if (challenges.size() < num_rounds) {
        throw std::runtime_error("FRI: not enough challenges for folding rounds");
    }

    for (uint round = 0; round < num_rounds; round++) {
        uint half_n = (1u << current_log) / 2;
        uint64_t omega = get_root_of_unity(current_log).val;

        // Precompute inv_two_domain[i] = 1 / (2 * current_offset * omega^i)
        std::vector<Fr_t> inv_two_domain(half_n);
        uint64_t two_inv = fri_inv(2);
        uint64_t w = 1;
        for (uint i = 0; i < half_n; i++) {
            uint64_t domain_pt = fri_mul(current_offset, w);
            inv_two_domain[i] = Fr_t{fri_mul(two_inv, fri_inv(domain_pt))};
            w = fri_mul(w, omega);
        }

        Fr_t* inv_two_domain_gpu;
        cudaMalloc(&inv_two_domain_gpu, half_n * sizeof(Fr_t));
        cudaMemcpy(inv_two_domain_gpu, inv_two_domain.data(), half_n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        Fr_t* new_evals;
        cudaMalloc(&new_evals, half_n * sizeof(Fr_t));

        uint blocks = (half_n + FRI_THREADS - 1) / FRI_THREADS;
        fri_fold_kernel<<<blocks, FRI_THREADS>>>(current_evals, new_evals, challenges[round], inv_two_domain_gpu, half_n);
        cudaDeviceSynchronize();

        cudaFree(inv_two_domain_gpu);

        // Build Merkle tree for this layer
        LayerData ld;
        ld.size = half_n;
        ld.evals_gpu = new_evals;
        ld.tree = new MerkleTree(new_evals, half_n);
        ld.evals_host.resize(half_n);
        cudaMemcpy(ld.evals_host.data(), new_evals, half_n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        layers.push_back(std::move(ld));

        current_evals = new_evals;
        current_log--;
        // After folding, the domain offset squares: offset' = offset^2
        current_offset = fri_mul(current_offset, current_offset);
    }

    // Store commitment layer roots (recomputed here — in practice these would be cached)
    // The commitment already has the first layer root; we add the rest
    FriCommitment& mut_commit = const_cast<FriCommitment&>(commitment);
    mut_commit.layer_roots.clear();
    for (auto& ld : layers) {
        mut_commit.layer_roots.push_back(ld.tree->root());
    }

    // Store remainder (final layer evaluations as polynomial)
    // The final layer evals are the evaluations of a low-degree polynomial
    // For simplicity, store them directly
    auto& last = layers.back();
    mut_commit.remainder = last.evals_host;

    // Generate query responses
    proof.queries.resize(query_positions.size());
    for (uint q = 0; q < query_positions.size(); q++) {
        uint pos = query_positions[q] % layers[0].size;

        for (uint round = 0; round < layers.size() - 1; round++) {
            uint layer_size = layers[round].size;
            uint half = layer_size / 2;
            uint idx = pos;
            uint paired = (idx < half) ? (idx + half) : (idx - half);

            FriQueryRound qr;
            qr.value = layers[round].evals_host[idx];
            qr.sibling_value = layers[round].evals_host[paired];
            qr.proof = layers[round].tree->prove(idx);
            qr.sibling_proof = layers[round].tree->prove(paired);
            proof.queries[q].push_back(qr);

            // Next round: the folded position is min(idx, paired)
            pos = (idx < half) ? idx : (idx - half);
        }
    }

    // Cleanup
    for (uint i = 0; i < layers.size(); i++) {
        delete layers[i].tree;
        cudaFree(layers[i].evals_gpu);
    }

    return proof;
}

// ── FRI Verifier implementation ──────────────────────────────────────────────

bool FriVerifier::verify(const FriCommitment& commitment,
                         const FriProof& proof,
                         const std::vector<Fr_t>& challenges,
                         const FriParams& params) {
    uint domain_log = commitment.domain_log_size;
    uint domain_size = 1u << domain_log;
    uint64_t current_offset = commitment.domain_offset.val;

    uint num_rounds = commitment.layer_roots.size() - 1;

    for (uint q = 0; q < proof.query_positions.size(); q++) {
        uint pos = proof.query_positions[q] % domain_size;
        uint64_t round_offset = current_offset;

        for (uint round = 0; round < num_rounds; round++) {
            uint layer_size = domain_size >> round;
            uint half = layer_size / 2;
            uint idx = pos;

            const auto& qr = proof.queries[q][round];

            // Verify Merkle proofs
            if (!MerkleTree::verify(commitment.layer_roots[round], qr.value, qr.proof, layer_size)) {
                fprintf(stderr, "FRI verify: Merkle proof failed for query %u round %u (value)\n", q, round);
                return false;
            }
            if (!MerkleTree::verify(commitment.layer_roots[round], qr.sibling_value, qr.sibling_proof, layer_size)) {
                fprintf(stderr, "FRI verify: Merkle proof failed for query %u round %u (sibling)\n", q, round);
                return false;
            }

            // Determine which is f(x) and which is f(-x)
            // Position idx < half means idx is the "positive" position
            uint pos_in_half = (idx < half) ? idx : (idx - half);
            uint64_t omega = get_root_of_unity(domain_log - round).val;
            uint64_t domain_pt = fri_mul(round_offset, fri_pow(omega, pos_in_half));

            uint64_t f_pos_val, f_neg_val;
            if (idx < half) {
                f_pos_val = qr.value.val;
                f_neg_val = qr.sibling_value.val;
            } else {
                f_pos_val = qr.sibling_value.val;
                f_neg_val = qr.value.val;
            }

            // Fold: (f_pos + f_neg)/2 + alpha * (f_pos - f_neg)/(2 * domain_pt)
            uint64_t sum = fri_add(f_pos_val, f_neg_val);
            uint64_t diff = fri_sub(f_pos_val, f_neg_val);
            uint64_t inv_two = fri_inv(2);
            uint64_t half_sum = fri_mul(sum, inv_two);
            uint64_t inv_two_x = fri_mul(inv_two, fri_inv(domain_pt));
            uint64_t alpha_term = fri_mul(challenges[round].val, fri_mul(diff, inv_two_x));
            uint64_t expected = fri_add(half_sum, alpha_term);

            // Check against next layer
            if (round < num_rounds - 1) {
                // The folded value should match what the prover claims at pos_in_half in the next layer
                const auto& next_qr = proof.queries[q][round + 1];
                // The prover's next round value should be at position pos_in_half
                if (next_qr.proof.leaf_index != pos_in_half &&
                    next_qr.sibling_proof.leaf_index != pos_in_half) {
                    fprintf(stderr, "FRI verify: position mismatch at query %u round %u->%u: "
                            "expected pos %u, got leaf_idx %u / sibling_idx %u\n",
                            q, round, round+1, pos_in_half,
                            next_qr.proof.leaf_index, next_qr.sibling_proof.leaf_index);
                    return false;
                }
                // Get the actual value at pos_in_half from next round's query data
                uint64_t actual;
                if (next_qr.proof.leaf_index == pos_in_half) {
                    actual = next_qr.value.val;
                } else {
                    actual = next_qr.sibling_value.val;
                }
                if (actual != expected) {
                    fprintf(stderr, "FRI verify: fold mismatch at query %u round %u: "
                            "expected %lu, got %lu\n", q, round, expected, actual);
                    return false;
                }
            } else {
                // Last folding round: check against remainder
                if (pos_in_half < commitment.remainder.size()) {
                    if (commitment.remainder[pos_in_half].val != expected) {
                        fprintf(stderr, "FRI verify: remainder mismatch at query %u round %u idx %u: "
                                "expected %lu, got %lu\n", q, round, pos_in_half,
                                expected, commitment.remainder[pos_in_half].val);
                        return false;
                    }
                }
            }

            // Update for next round
            pos = pos_in_half;
            round_offset = fri_mul(round_offset, round_offset);
        }
    }

    return true;
}
