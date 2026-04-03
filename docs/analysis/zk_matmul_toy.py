"""
zk_matmul_toy.py -- Concrete ZK proof of two-layer matrix multiplication.

GOAL: Demonstrate a complete ZK proof system for the statement
  "I know (W1, W2, x) such that W2 * (W1 * x) = y"
where the verifier has only commitments C_W1, C_W2, C_x and public output y.

Field:      F_257 (NTT-friendly prime: 257-1 = 256 = 2^8)
Matrices:   3x3 (padded to 4x4 = 2^2 x 2^2)
Vectors:    length 3 (padded to 4 = 2^2)

This file is self-contained and executable: python3 zk_matmul_toy.py

Each protocol message and field operation is explicit, allowing line-by-line
analysis of security and complexity.

Architecture (interleaved BaseFold):
  - RS-encoded witness polynomials committed via salted Merkle trees
  - Sumcheck and codeword folding happen simultaneously (same challenges)
  - No separate PCS opening step -- the sumcheck IS the opening
  - ZK mode: salted Merkle leaves + additive sumcheck masking
  - Cross-layer composition via sumcheck chaining
"""

import hashlib
import os
import random
import math
from dataclasses import dataclass, field as datafield
from typing import List, Tuple, Optional, Callable

# ==================================================================
# Part 1: Field Arithmetic (F_257)
# ==================================================================

P = 257

def fadd(a, b): return (a + b) % P
def fsub(a, b): return (a - b) % P
def fmul(a, b): return (a * b) % P
def finv(a):
    assert a % P != 0, "division by zero"
    return pow(a, P - 2, P)
def fneg(a): return (P - a) % P
def frand(): return random.randint(1, P - 1)

PRIMITIVE_ROOT = 3
INV2 = finv(2)  # 129

def root_of_unity(order):
    """Primitive order-th root of unity in F_257. order must be power of 2, dividing 256."""
    assert order > 0 and (order & (order - 1)) == 0 and 256 % order == 0
    return pow(PRIMITIVE_ROOT, 256 // order, P)

# ==================================================================
# Part 2: Multilinear Extensions (MLE)
# ==================================================================

def bits(i, k):
    """k-bit binary decomposition. bits(5, 4) = [1, 0, 1, 0]."""
    return [(i >> j) & 1 for j in range(k)]

def mle_eval(data, point):
    """Evaluate MLE of data at point using iterated folding. Cost: O(2^k) muls."""
    d = list(data)
    for j in range(len(point)):
        r = point[j]
        one_minus_r = fsub(1, r)
        half = len(d) // 2
        d = [fadd(fmul(one_minus_r, d[2*i]), fmul(r, d[2*i+1])) for i in range(half)]
    assert len(d) == 1
    return d[0]

def mle_fold(data, challenge):
    """One round of MLE folding: d'[i] = (1-r)*d[2i] + r*d[2i+1]."""
    r = challenge
    one_minus_r = fsub(1, r)
    half = len(data) // 2
    return [fadd(fmul(one_minus_r, data[2*i]), fmul(r, data[2*i+1])) for i in range(half)]

# ==================================================================
# Part 2.5: NTT (Number Theoretic Transform)
# ==================================================================

def bit_reverse(x, log_n):
    result = 0
    for _ in range(log_n):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

def ntt(coeffs, omega):
    """Evaluate polynomial at {omega^0, ..., omega^{N-1}}. N must be power of 2."""
    N = len(coeffs)
    log_n = int(math.log2(N))
    assert N == (1 << log_n)
    a = [coeffs[bit_reverse(i, log_n)] for i in range(N)]
    length = 2
    while length <= N:
        w_step = pow(omega, N // length, P)
        for start in range(0, N, length):
            w = 1
            for j in range(length // 2):
                u = a[start + j]
                v = fmul(a[start + j + length // 2], w)
                a[start + j] = fadd(u, v)
                a[start + j + length // 2] = fsub(u, v)
                w = fmul(w, w_step)
        length *= 2
    return a

def intt(evals, omega):
    """Inverse NTT: recover coefficients from evaluations."""
    N = len(evals)
    coeffs = ntt(evals, finv(omega))
    N_inv = finv(N)
    return [fmul(c, N_inv) for c in coeffs]

# ==================================================================
# Part 3: Merkle Commitment (with salt support)
# ==================================================================

def hash_leaf(value, salt=None):
    data = int(value).to_bytes(8, 'little')
    if salt is not None:
        data += salt
    return hashlib.sha256(data).digest()

def hash_pair(left, right):
    return hashlib.sha256(left + right).digest()

def generate_salts(n):
    """Generate n random 8-byte salts for Merkle leaf hiding."""
    return [os.urandom(8) for _ in range(n)]

def merkle_commit(data, salts=None):
    """Build Merkle tree. Returns (root, tree, salts_used)."""
    n = len(data)
    if salts is None:
        salts = [None] * n
    leaves = [hash_leaf(d, s) for d, s in zip(data, salts)]
    tree = [leaves]
    current = leaves
    while len(current) > 1:
        nxt = [hash_pair(current[i], current[i+1])
               for i in range(0, len(current) - 1, 2)]
        if len(current) % 2 == 1:
            nxt.append(current[-1])
        tree.append(nxt)
        current = nxt
    return current[0], tree, salts

def merkle_open(tree, index):
    proof = []
    idx = index
    for level in tree[:-1]:
        sibling = idx ^ 1
        if sibling < len(level):
            proof.append((level[sibling], idx % 2))
        idx //= 2
    return proof

def merkle_verify(root, index, value, proof, salt=None):
    h = hash_leaf(value, salt)
    for sibling_hash, side in proof:
        if side == 0:
            h = hash_pair(h, sibling_hash)
        else:
            h = hash_pair(sibling_hash, h)
    return h == root

# ==================================================================
# Part 4: RS Encoding and Coset-Aware Folding
# ==================================================================

BLOWUP = 2

def rs_encode(data, blowup=BLOWUP):
    """RS-encode: treat data as polynomial coefficients, evaluate on coset domain.
    Returns: (codeword, omega, coset)
    """
    n = len(data)
    N = n * blowup
    omega = root_of_unity(N)
    coset = PRIMITIVE_ROOT
    shifted = [fmul(data[i], pow(coset, i, P)) for i in range(n)]
    padded = shifted + [0] * (N - n)
    codeword = ntt(padded, omega)
    return codeword, omega, coset

def rs_fold(codeword, challenge, omega, coset):
    """Fold codeword with MLE-compatible formula.
    c'[j] = (1-r)*p_even(s_j) + r*p_odd(s_j)
    where p_even = (c[j]+c[j+N/2])/2, p_odd = (c[j]-c[j+N/2])/(2*coset*omega^j).
    Returns: (folded, new_omega, new_coset)
    """
    N = len(codeword)
    half = N // 2
    r = challenge
    one_minus_r = fsub(1, r)
    folded = []
    for j in range(half):
        c_lo = codeword[j]
        c_hi = codeword[j + half]
        omega_j = pow(omega, j, P)
        twiddle = finv(fmul(2, fmul(coset, omega_j)))
        p_even = fmul(fadd(c_lo, c_hi), INV2)
        p_odd = fmul(fsub(c_lo, c_hi), twiddle)
        folded.append(fadd(fmul(one_minus_r, p_even), fmul(r, p_odd)))
    return folded, fmul(omega, omega), fmul(coset, coset)

def rs_commit(data, blowup=BLOWUP, use_salts=False):
    """RS-encode data and Merkle-commit. Returns (root, codeword, tree, omega, coset, salts)."""
    codeword, omega, coset = rs_encode(data, blowup)
    salts = generate_salts(len(codeword)) if use_salts else [None] * len(codeword)
    root, tree, salts = merkle_commit(codeword, salts)
    return root, codeword, tree, omega, coset, salts

# ==================================================================
# Part 5: Lagrange Interpolation Helpers
# ==================================================================

def eval_degree2(evals, t):
    """Evaluate degree-2 poly given [f(0), f(1), f(2)] at t."""
    f0, f1, f2 = evals
    inv2 = INV2
    L0 = fmul(fmul(fsub(t, 1), fsub(t, 2)), inv2)
    L1 = fneg(fmul(t, fsub(t, 2)))
    L2 = fmul(fmul(t, fsub(t, 1)), inv2)
    return fadd(fadd(fmul(f0, L0), fmul(f1, L1)), fmul(f2, L2))

def lagrange_eval_deg3(evals, t):
    """Evaluate degree-3 poly given [f(0), f(1), f(2), f(3)] at t."""
    def L(i, nodes):
        num, den = 1, 1
        for j in nodes:
            if j != i:
                num = fmul(num, fsub(t, j))
                den = fmul(den, fsub(i, j))
        return fmul(num, finv(den))
    nodes = [0, 1, 2, 3]
    return sum_field([fmul(evals[i], L(i, nodes)) for i in range(4)])

def sum_field(vals):
    r = 0
    for v in vals:
        r = fadd(r, v)
    return r

# ==================================================================
# Part 6: Helpers
# ==================================================================

def pad_pow2(data, n=None):
    if n is None:
        n = 1 << math.ceil(math.log2(max(len(data), 1)))
    return data + [0] * (n - len(data))

def mat_flat(W, M, N, pM, pN):
    """Row-major flatten of M x N matrix, padded to pM x pN."""
    flat = []
    for i in range(pM):
        for j in range(pN):
            flat.append(W[i][j] if (i < M and j < N) else 0)
    return flat

def w_point(row_pt, col_pt):
    """MLE evaluation point for row-major matrix. col vars FIRST, then row vars."""
    return list(col_pt) + list(row_pt)

def matvec(W, x, M, N):
    return [sum(W[i][j] * x[j] for j in range(N)) % P for i in range(M)]

def make_zero_sum_mask(k):
    """Random rho: {0,1}^k -> F with sum = 0."""
    n = 1 << k
    values = [frand() for _ in range(n - 1)]
    total = sum(values) % P
    values.append(fneg(total))
    return values

# ==================================================================
# Part 7: Interleaved BaseFold Matmul Proof
# ==================================================================
#
# The core protocol. For a single layer y = W * x:
#
# COMMIT: RS-encode W_flat and x_pad, Merkle-commit with salts.
#
# INTERLEAVED SUMCHECK + FOLD (b rounds for column variables):
#   At each round j:
#     1. Prover sends sumcheck round polynomial s_j
#     2. Verifier sends challenge alpha_j
#     3. Prover folds BOTH W-codeword and x-codeword with alpha_j
#     4. Prover commits folded codewords, answers spot-check queries
#
# REMAINING FOLDS (a rounds for W's row variables):
#   W-codeword still has row variables. Fold with pre-chosen r challenges.
#   x-codeword has already collapsed to blowup constants.
#
# FINAL CHECK:
#   W_final * x_final == sumcheck final claim
#
# ZK mode: salted Merkle leaves, additive mask rho on sumcheck polynomials.

@dataclass
class FoldLayer:
    """One round of codeword folding for a single polynomial."""
    root: bytes
    tree: list
    codeword: list
    salts: list

@dataclass
class FoldQuery:
    """Spot-check query data for one polynomial at one fold round."""
    qi: int
    c_lo: int
    c_hi: int
    folded: int
    lo_proof: list
    hi_proof: list
    folded_proof: list
    lo_salt: bytes
    hi_salt: bytes
    folded_salt: bytes

def fold_and_commit(codeword, challenge, omega, coset, tree, salts,
                    num_queries=2, use_salts=False):
    """Fold a codeword and commit. Returns (folded_layer, queries, new_omega, new_coset)."""
    N = len(codeword)
    half = N // 2

    folded, new_omega, new_coset = rs_fold(codeword, challenge, omega, coset)
    new_salts = generate_salts(len(folded)) if use_salts else [None] * len(folded)
    folded_root, folded_tree, new_salts = merkle_commit(folded, new_salts)

    queries = []
    for _ in range(num_queries):
        qi = random.randint(0, half - 1)
        queries.append(FoldQuery(
            qi=qi,
            c_lo=codeword[qi], c_hi=codeword[qi + half],
            folded=folded[qi],
            lo_proof=merkle_open(tree, qi),
            hi_proof=merkle_open(tree, qi + half),
            folded_proof=merkle_open(folded_tree, qi),
            lo_salt=salts[qi] if salts[qi] else None,
            hi_salt=salts[qi + half] if salts[qi + half] else None,
            folded_salt=new_salts[qi] if new_salts[qi] else None,
        ))

    layer = FoldLayer(folded_root, folded_tree, folded, new_salts)
    return layer, queries, new_omega, new_coset

def verify_fold_queries(prev_root, curr_root, queries, challenge, omega, coset, N):
    """Verify fold consistency for a set of queries."""
    r = challenge
    one_minus_r = fsub(1, r)
    half = N // 2
    for q in queries:
        qi = q.qi
        # Merkle checks
        if not merkle_verify(prev_root, qi, q.c_lo, q.lo_proof, q.lo_salt):
            return False
        if not merkle_verify(prev_root, qi + half, q.c_hi, q.hi_proof, q.hi_salt):
            return False
        if not merkle_verify(curr_root, qi, q.folded, q.folded_proof, q.folded_salt):
            return False
        # Fold equation
        omega_qi = pow(omega, qi, P)
        twiddle = finv(fmul(2, fmul(coset, omega_qi)))
        p_even = fmul(fadd(q.c_lo, q.c_hi), INV2)
        p_odd = fmul(fsub(q.c_lo, q.c_hi), twiddle)
        expected = fadd(fmul(one_minus_r, p_even), fmul(r, p_odd))
        if q.folded != expected:
            return False
    return True

@dataclass
class InterleavedProof:
    """Proof for one layer of matmul (interleaved sumcheck + fold)."""
    # Sumcheck transcript
    round_polys: list          # [s_j(0), s_j(1), ...] per round
    challenges: list           # alpha_j per round

    # Per-polynomial fold data: dict mapping name -> list of (queries, root) per round
    poly_roots: dict           # name -> [root per fold layer]
    poly_queries: dict         # name -> [queries per fold round]
    poly_finals: dict          # name -> final codeword (blowup elements)

    # ZK mask
    mask_eval: Optional[int]   # rho_tilde(s) if ZK
    mask_root: Optional[bytes] # commitment to mask if ZK


def prove_matmul_interleaved(W_flat, x_pad, y_pad, r, b, a, use_zk=False,
                              num_queries=2, blowup=BLOWUP):
    """Interleaved BaseFold proof that W*x = y (one layer).

    r: verifier's row challenge (a elements)
    b: number of column variables (sumcheck variables)
    a: number of row variables

    Returns: (accept, proof, T)
    """
    pN = 1 << b
    pM = 1 << a

    # -- Commit --
    use_salts = use_zk
    W_root, W_cw, W_tree, W_omega, W_coset, W_salts = rs_commit(W_flat, blowup, use_salts)
    x_root, x_cw, x_tree, x_omega, x_coset, x_salts = rs_commit(x_pad, blowup, use_salts)

    T = mle_eval(y_pad, r)

    # -- ZK mask --
    mask = None
    mask_root = None
    mask_cw = mask_tree = mask_omega = mask_coset = mask_salts = None
    if use_zk:
        mask = make_zero_sum_mask(b)
        mask_root, mask_cw, mask_tree, mask_omega, mask_coset, mask_salts = \
            rs_commit(mask, blowup, use_salts)

    # -- Track MLE data for sumcheck evaluation --
    W_data = list(W_flat)
    x_data = list(x_pad)
    mask_data = mask

    # -- Interleaved sumcheck + fold --
    round_polys = []
    challenges = []
    current_claim = T

    W_roots = [W_root]
    x_roots = [x_root]
    W_all_queries = []
    x_all_queries = []
    mask_all_queries = []
    mask_roots = [mask_root] if use_zk else []

    cur_W_cw, cur_W_tree, cur_W_omega, cur_W_coset, cur_W_salts = W_cw, W_tree, W_omega, W_coset, W_salts
    cur_x_cw, cur_x_tree, cur_x_omega, cur_x_coset, cur_x_salts = x_cw, x_tree, x_omega, x_coset, x_salts
    if use_zk:
        cur_m_cw, cur_m_tree, cur_m_omega, cur_m_coset, cur_m_salts = mask_cw, mask_tree, mask_omega, mask_coset, mask_salts

    for j in range(b):
        remaining = b - j - 1

        # -- Sumcheck round polynomial --
        # W_data and x_data have already been folded by rounds 0..j-1,
        # so they have 2^(a + remaining + 1) and 2^(remaining + 1) elements.
        # Points use only the REMAINING variables (folded ones are baked in).
        n_eval = 4 if use_zk else 3
        s_vals = [0] * n_eval
        for X in range(n_eval):
            total = 0
            for tail in range(1 << remaining):
                c_rem = [X] + bits(tail, remaining)
                wp = w_point(r, c_rem)   # remaining col vars + row vars
                f_val = fmul(mle_eval(W_data, wp), mle_eval(x_data, c_rem))
                if use_zk:
                    f_val = fadd(f_val, mle_eval(mask_data, c_rem))
                total = fadd(total, f_val)
            s_vals[X] = total

        assert fadd(s_vals[0], s_vals[1]) == current_claim, \
            f"Round {j}: {s_vals[0]}+{s_vals[1]} != {current_claim}"

        round_polys.append(s_vals)

        # -- Verifier challenge --
        alpha = frand()
        challenges.append(alpha)
        if use_zk:
            current_claim = lagrange_eval_deg3(s_vals, alpha)
        else:
            current_claim = eval_degree2(s_vals, alpha)
        # -- Fold codewords with alpha --
        W_layer, W_q, cur_W_omega, cur_W_coset = fold_and_commit(
            cur_W_cw, alpha, cur_W_omega, cur_W_coset, cur_W_tree, cur_W_salts,
            num_queries, use_salts)
        cur_W_cw, cur_W_tree, cur_W_salts = W_layer.codeword, W_layer.tree, W_layer.salts
        W_roots.append(W_layer.root)
        W_all_queries.append(W_q)

        x_layer, x_q, cur_x_omega, cur_x_coset = fold_and_commit(
            cur_x_cw, alpha, cur_x_omega, cur_x_coset, cur_x_tree, cur_x_salts,
            num_queries, use_salts)
        cur_x_cw, cur_x_tree, cur_x_salts = x_layer.codeword, x_layer.tree, x_layer.salts
        x_roots.append(x_layer.root)
        x_all_queries.append(x_q)

        if use_zk:
            m_layer, m_q, cur_m_omega, cur_m_coset = fold_and_commit(
                cur_m_cw, alpha, cur_m_omega, cur_m_coset, cur_m_tree, cur_m_salts,
                num_queries, use_salts)
            cur_m_cw, cur_m_tree, cur_m_salts = m_layer.codeword, m_layer.tree, m_layer.salts
            mask_roots.append(m_layer.root)
            mask_all_queries.append(m_q)

        # Fold MLE data too (for next round's sumcheck evaluation)
        W_data = mle_fold(W_data, alpha)
        x_data = mle_fold(x_data, alpha)
        if use_zk:
            mask_data = mle_fold(mask_data, alpha)

    # -- After b sumcheck rounds: x-codeword should be collapsed --
    x_final = cur_x_cw  # blowup elements, should all equal x_tilde(s)

    # -- Continue folding W for a more rounds (row variables, using r) --
    for j in range(a):
        W_layer, W_q, cur_W_omega, cur_W_coset = fold_and_commit(
            cur_W_cw, r[j], cur_W_omega, cur_W_coset, cur_W_tree, cur_W_salts,
            num_queries, use_salts)
        cur_W_cw, cur_W_tree, cur_W_salts = W_layer.codeword, W_layer.tree, W_layer.salts
        W_roots.append(W_layer.root)
        W_all_queries.append(W_q)
        W_data = mle_fold(W_data, r[j])

    W_final = cur_W_cw  # blowup elements, should all equal W_tilde(r,s)

    # -- ZK mask final --
    mask_eval = None
    if use_zk:
        mask_eval = cur_m_cw[0]  # should all be equal = rho_tilde(s)

    proof = InterleavedProof(
        round_polys=round_polys,
        challenges=challenges,
        poly_roots={'W': W_roots, 'x': x_roots},
        poly_queries={'W': W_all_queries, 'x': x_all_queries},
        poly_finals={'W': W_final, 'x': x_final},
        mask_eval=mask_eval,
        mask_root=mask_root,
    )
    if use_zk:
        proof.poly_roots['mask'] = mask_roots
        proof.poly_queries['mask'] = mask_all_queries

    return proof, T, W_root, x_root


def verify_matmul_interleaved(proof, T, W_commit_root, x_commit_root, r, b, a,
                               use_zk=False, num_queries=2, blowup=BLOWUP):
    """Verify an interleaved BaseFold matmul proof."""
    pN = 1 << b

    # -- Verify sumcheck transcript --
    current_claim = T
    for j in range(b):
        s = proof.round_polys[j]
        if fadd(s[0], s[1]) != current_claim:
            return False, "sumcheck round poly sum mismatch"
        alpha = proof.challenges[j]
        if use_zk:
            current_claim = lagrange_eval_deg3(s, alpha)
        else:
            current_claim = eval_degree2(s, alpha)

    # -- Check commitment roots match --
    if proof.poly_roots['W'][0] != W_commit_root:
        return False, "W commitment mismatch"
    if proof.poly_roots['x'][0] != x_commit_root:
        return False, "x commitment mismatch"

    # -- Verify fold consistency for x (b rounds) --
    n_x = pN
    N_x = n_x * blowup
    x_omega = root_of_unity(N_x)
    x_coset = PRIMITIVE_ROOT
    for j in range(b):
        curr_N = N_x >> j
        if not verify_fold_queries(
                proof.poly_roots['x'][j], proof.poly_roots['x'][j+1],
                proof.poly_queries['x'][j], proof.challenges[j],
                x_omega, x_coset, curr_N):
            return False, f"x fold round {j} failed"
        x_omega = fmul(x_omega, x_omega)
        x_coset = fmul(x_coset, x_coset)

    # -- Verify fold consistency for W (b + a rounds) --
    n_W = (1 << a) * pN
    N_W = n_W * blowup
    W_omega = root_of_unity(N_W)
    W_coset = PRIMITIVE_ROOT
    all_W_challenges = list(proof.challenges) + list(r)
    for j in range(b + a):
        curr_N = N_W >> j
        if not verify_fold_queries(
                proof.poly_roots['W'][j], proof.poly_roots['W'][j+1],
                proof.poly_queries['W'][j], all_W_challenges[j],
                W_omega, W_coset, curr_N):
            return False, f"W fold round {j} failed"
        W_omega = fmul(W_omega, W_omega)
        W_coset = fmul(W_coset, W_coset)

    # -- Verify ZK mask fold consistency --
    if use_zk:
        n_m = pN
        N_m = n_m * blowup
        m_omega = root_of_unity(N_m)
        m_coset = PRIMITIVE_ROOT
        for j in range(b):
            curr_N = N_m >> j
            if not verify_fold_queries(
                    proof.poly_roots['mask'][j], proof.poly_roots['mask'][j+1],
                    proof.poly_queries['mask'][j], proof.challenges[j],
                    m_omega, m_coset, curr_N):
                return False, f"mask fold round {j} failed"
            m_omega = fmul(m_omega, m_omega)
            m_coset = fmul(m_coset, m_coset)

    # -- Check final codewords are constant --
    W_val = proof.poly_finals['W'][0]
    if not all(v == W_val for v in proof.poly_finals['W']):
        return False, "W final codeword not constant"

    x_val = proof.poly_finals['x'][0]
    if not all(v == x_val for v in proof.poly_finals['x']):
        return False, "x final codeword not constant"

    # -- Final product check --
    product = fmul(W_val, x_val)
    if use_zk:
        expected = fsub(current_claim, proof.mask_eval)
    else:
        expected = current_claim

    if product != expected:
        return False, f"product check failed: {W_val}*{x_val}={product} != {expected}"

    return True, "OK"


# ==================================================================
# Part 8: Single-Layer and Two-Layer Matmul Proofs
# ==================================================================

def prove_single_matmul(W, x, y, M, N, use_zk=False, label=""):
    """Full proof that W*x = y."""
    a = max(1, math.ceil(math.log2(M)))
    b = max(1, math.ceil(math.log2(N)))
    pM, pN = 1 << a, 1 << b

    W_flat = mat_flat(W, M, N, pM, pN)
    x_pad = pad_pow2(x, pN)
    y_pad = pad_pow2(y, pM)

    tag = f"[{label}] " if label else ""
    print(f"\n{'='*60}")
    print(f"{tag}MATMUL PROOF: {M}x{N} (padded {pM}x{pN}), ZK={'ON' if use_zk else 'OFF'}")

    # Verifier challenge
    r = [frand() for _ in range(a)]

    # Interleaved proof
    proof, T, W_root, x_root = prove_matmul_interleaved(
        W_flat, x_pad, y_pad, r, b, a, use_zk)

    print(f"  T = y_tilde(r) = {T}")
    print(f"  Sumcheck: {b} rounds, {len(proof.round_polys[0])} values/round")
    for j in range(b):
        print(f"    Round {j}: s = {proof.round_polys[j]}  alpha = {proof.challenges[j]}")
    print(f"  W final = {proof.poly_finals['W'][0]}, x final = {proof.poly_finals['x'][0]}")

    # Verify
    ok, msg = verify_matmul_interleaved(proof, T, W_root, x_root, r, b, a, use_zk)
    print(f"  Verify: {msg}")
    print(f"  RESULT: {'ACCEPT' if ok else 'REJECT'}")
    return ok


def prove_two_layer(W1, W2, x, y, N, use_zk=False):
    """Prove y = W2 * (W1 * x) with cross-layer composition."""
    a = max(1, math.ceil(math.log2(N)))
    b = a
    pN = 1 << a

    h = matvec(W1, x, N, N)

    W1_flat = mat_flat(W1, N, N, pN, pN)
    W2_flat = mat_flat(W2, N, N, pN, pN)
    x_pad = pad_pow2(x, pN)
    h_pad = pad_pow2(h, pN)
    y_pad = pad_pow2(y, pN)

    print(f"\n{'='*60}")
    print(f"TWO-LAYER PROOF: y = W2 * W1 * x  ({N}x{N} matrices)")
    print(f"  x = {x},  h = W1*x = {h},  y = W2*h = {y}")
    print(f"  ZK = {'ON' if use_zk else 'OFF'}")

    # ============================================================
    # LAYER 2: y_tilde(r) = sum_c W2_tilde(r,c) * h_tilde(c)
    # ============================================================
    print(f"\n  --- LAYER 2 ---")
    r = [frand() for _ in range(a)]
    T2 = mle_eval(y_pad, r)
    print(f"  r = {r}, T = {T2}")

    # Commit W2 and h
    W2_root, W2_cw, W2_tree, W2_omega, W2_coset, W2_salts = rs_commit(W2_flat, BLOWUP, use_zk)
    h_root, h_cw, h_tree, h_omega, h_coset, h_salts = rs_commit(h_pad, BLOWUP, use_zk)

    # Interleaved sumcheck + fold for layer 2
    proof2, _, _, _ = prove_matmul_interleaved(W2_flat, h_pad, y_pad, r, b, a, use_zk)

    # Extract h_tilde(s) from collapsed h-codeword
    s = proof2.challenges
    h_at_s = proof2.poly_finals['x'][0]  # 'x' slot holds h for layer 2
    W2_at_rs = proof2.poly_finals['W'][0]

    print(f"  Sumcheck: {b} rounds")
    for j in range(b):
        print(f"    Round {j}: s = {proof2.round_polys[j]}  alpha = {proof2.challenges[j]}")
    print(f"  W2_tilde(r,s) = {W2_at_rs}, h_tilde(s) = {h_at_s}")

    # Verify layer 2
    ok2, msg2 = verify_matmul_interleaved(
        proof2, T2, proof2.poly_roots['W'][0], proof2.poly_roots['x'][0],
        r, b, a, use_zk)
    print(f"  Layer 2 verify: {msg2}")

    # ============================================================
    # CROSS-LAYER JUNCTION
    # ============================================================
    print(f"\n  --- CROSS-LAYER JUNCTION ---")
    print(f"  h_tilde(s) = {h_at_s} becomes claim for layer 1.")

    # ============================================================
    # LAYER 1: h_tilde(s) = sum_c W1_tilde(s,c) * x_tilde(c)
    # ============================================================
    print(f"\n  --- LAYER 1 ---")

    # For layer 1, the "row challenge" is s (from layer 2's sumcheck)
    # The claim is h_at_s
    # We need to prove: sum_c W1_tilde(s,c) * x_tilde(c) = h_at_s

    # Build y_pad for layer 1 = h_pad, with the claim at point s
    proof1, T1, _, _ = prove_matmul_interleaved(W1_flat, x_pad, h_pad, s, b, a, use_zk)

    t = proof1.challenges
    W1_at_st = proof1.poly_finals['W'][0]
    x_at_t = proof1.poly_finals['x'][0]

    print(f"  Sumcheck: {b} rounds")
    for j in range(b):
        print(f"    Round {j}: s = {proof1.round_polys[j]}  alpha = {proof1.challenges[j]}")
    print(f"  W1_tilde(s,t) = {W1_at_st}, x_tilde(t) = {x_at_t}")

    # Verify layer 1
    ok1, msg1 = verify_matmul_interleaved(
        proof1, T1, proof1.poly_roots['W'][0], proof1.poly_roots['x'][0],
        s, b, a, use_zk)
    print(f"  Layer 1 verify: {msg1}")

    # Check cross-layer consistency
    assert T1 == h_at_s, f"Cross-layer mismatch: T1={T1} != h_at_s={h_at_s}"

    all_ok = ok2 and ok1
    print(f"\n  --- VERIFICATION ---")
    print(f"  Layer 2: {'OK' if ok2 else 'FAIL'}")
    print(f"  Layer 1: {'OK' if ok1 else 'FAIL'}")
    print(f"  Cross-layer: T1={T1} == h_at_s={h_at_s}: OK")
    print(f"  RESULT: {'ACCEPT' if all_ok else 'REJECT'}")

    return all_ok


# ==================================================================
# Part 9: Operation Counting at Scale
# ==================================================================

def count_ops(M, N, num_layers, num_queries=2, blowup=2):
    a = math.ceil(math.log2(M))
    b = math.ceil(math.log2(N))
    pM, pN = 1 << a, 1 << b
    n_W = pM * pN
    n_v = pN

    ops = {}

    # RS encoding (NTT)
    ops['rs_encode_muls'] = (num_layers * n_W * blowup * int(math.log2(n_W * blowup))
                             + (2 * num_layers) * n_v * blowup * int(math.log2(n_v * blowup)))

    # Commitments (Merkle over codewords)
    ops['commit_hashes'] = (num_layers * (2*n_W*blowup - 1) +
                            (2*num_layers) * (2*n_v*blowup - 1))

    # Sumcheck (unchanged)
    ops['sumcheck_muls_optimized'] = num_layers * 2 * (n_W + n_v)

    # Fold hashes (interleaved: one Merkle tree per fold round per polynomial)
    # W: (a+b) rounds, x: b rounds, each builds a Merkle tree
    ops['fold_hashes'] = num_layers * (
        sum(2 * n_W * blowup // (2**j) for j in range(1, a+b+1)) +
        sum(2 * n_v * blowup // (2**j) for j in range(1, b+1))
    )

    ops['total_hashes'] = ops['commit_hashes'] + ops['fold_hashes']
    ops['total_muls'] = ops['rs_encode_muls'] + ops['sumcheck_muls_optimized']

    return ops, {'a': a, 'b': b, 'pM': pM, 'pN': pN, 'n_W': n_W, 'n_v': n_v}

def print_scaling():
    print(f"\n{'='*70}")
    print(f"OPERATION COUNTS AT SCALE (interleaved BaseFold, blowup={BLOWUP})")
    print(f"{'='*70}")

    configs = [
        ("Toy (3x3, 2 layers)", 3, 3, 2),
        ("Small (64x64, 4 layers)", 64, 64, 4),
        ("GPT-2 head (768x768, 12 layers)", 768, 768, 12),
        ("LLaMA-7B (4096x4096, 32 layers)", 4096, 4096, 32),
        ("Entropy layer (4096x32000, 1 layer)", 4096, 32000, 1),
    ]

    for name, M, N, L in configs:
        ops, dims = count_ops(M, N, L)
        print(f"\n  {name}")
        print(f"    Padded: {dims['pM']}x{dims['pN']}, "
              f"n_W = {dims['n_W']:,}, n_v = {dims['n_v']:,}")
        print(f"    RS encode muls:  {ops['rs_encode_muls']:>15,}")
        print(f"    Commit hashes:   {ops['commit_hashes']:>15,}")
        print(f"    Fold hashes:     {ops['fold_hashes']:>15,}")
        print(f"    Total hashes:    {ops['total_hashes']:>15,}")
        print(f"    Sumcheck muls:   {ops['sumcheck_muls_optimized']:>15,}  (optimized)")
        print(f"    Total muls:      {ops['total_muls']:>15,}")

    # Cost estimates for entropy layer
    print(f"\n{'='*70}")
    print(f"ESTIMATED WALL CLOCK: Entropy layer (4096 x 32000, 1 layer)")
    print(f"{'='*70}")
    ops, dims = count_ops(4096, 32000, 1)

    mul_throughput = 13e9
    hash_low = 500e6
    hash_high = 2e9

    total_muls = ops['total_muls']
    total_hashes = ops['total_hashes']

    mul_time = total_muls / mul_throughput
    hash_time_low = total_hashes / hash_low
    hash_time_high = total_hashes / hash_high

    print(f"\n  Total field muls:  {total_muls:>15,}")
    print(f"  Total hashes:      {total_hashes:>15,}")
    print(f"")
    print(f"  Field mul time:    {mul_time*1000:>10.1f} ms  (at {mul_throughput/1e9:.0f} Gops/s)")
    print(f"  Hash time (low):   {hash_time_low*1000:>10.1f} ms  (at {hash_low/1e6:.0f} M/s)")
    print(f"  Hash time (high):  {hash_time_high*1000:>10.1f} ms  (at {hash_high/1e6:.0f} M/s)")
    print(f"")
    print(f"  TOTAL (conservative): {(mul_time + hash_time_low)*1000:.0f} ms")
    print(f"  TOTAL (optimistic):   {(mul_time + hash_time_high)*1000:.0f} ms")

    print(f"\n  For comparison (Pedersen/BLS12-381 at 1.4M MSM/s):")
    pedersen_time = dims['n_W'] / 1.4e6
    print(f"  Pedersen commit only:  {pedersen_time:.1f} s")
    print(f"  Pedersen commit+open:  ~{2*pedersen_time:.1f} s")

# ==================================================================
# Main
# ==================================================================

if __name__ == '__main__':
    random.seed(42)

    print("=" * 60)
    print("ZK MATMUL TOY -- Interleaved BaseFold proof")
    print(f"Field: F_{P} (NTT-friendly: {P}-1 = {P-1} = 2^{int(math.log2(P-1))})")
    print("=" * 60)

    # -- NTT self-test --
    print(f"\n>>> NTT self-test")
    test_data = [1, 2, 3, 4]
    omega4 = root_of_unity(4)
    evals = ntt(test_data, omega4)
    recovered = intt(evals, omega4)
    assert recovered == test_data, "NTT round-trip failed!"
    print(f"  Round-trip: OK")

    # -- RS encode + fold self-test --
    print(f"\n>>> RS encode + fold self-test")
    test_data = [5, 10, 15, 20]
    test_point = [frand(), frand()]
    expected = mle_eval(test_data, test_point)
    cw, omega_cw, coset_cw = rs_encode(test_data)
    folded1, omega1, coset1 = rs_fold(cw, test_point[0], omega_cw, coset_cw)
    folded2, omega2, coset2 = rs_fold(folded1, test_point[1], omega1, coset1)
    assert all(v == expected for v in folded2), f"RS fold failed!"
    print(f"  RS fold matches MLE eval: OK")

    # -- Example 1: Single 3x3 matmul (no ZK) --
    W = [[2, 3, 1],
         [1, 0, 2],
         [4, 1, 3]]
    x = [1, 2, 3]
    y = matvec(W, x, 3, 3)
    print(f"\n>>> Example 1: Single 3x3 matmul, y = {y}")
    prove_single_matmul(W, x, y, 3, 3, use_zk=False)

    # -- Example 2: Single 3x3 matmul (with ZK) --
    print(f"\n>>> Example 2: Same matmul with ZK")
    random.seed(42)
    prove_single_matmul(W, x, y, 3, 3, use_zk=True, label="ZK")

    # -- Example 3: Two-layer 3x3 (no ZK) --
    W1 = [[2, 3, 1],
          [1, 0, 2],
          [4, 1, 3]]
    W2 = [[1, 2, 0],
          [3, 1, 1],
          [0, 2, 4]]
    x = [1, 2, 3]
    h = matvec(W1, x, 3, 3)
    y = matvec(W2, h, 3, 3)
    print(f"\n>>> Example 3: Two-layer proof, h = {h}, y = {y}")
    random.seed(42)
    prove_two_layer(W1, W2, x, y, 3, use_zk=False)

    # -- Example 4: Two-layer 3x3 (with ZK) --
    print(f"\n>>> Example 4: Two-layer proof with ZK")
    random.seed(42)
    prove_two_layer(W1, W2, x, y, 3, use_zk=True)

    # -- Scaling analysis --
    print_scaling()
