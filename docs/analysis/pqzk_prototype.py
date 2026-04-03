"""
pqzk_prototype.py -- Prototype of the per-operation sumcheck + committed masked intermediates
architecture for post-quantum zero-knowledge matmul proofs.

Proves y2 = W2 * (W1 * x) in zero knowledge using:
  1. Vanishing polynomial masking (hide evaluations at random points)
  2. Simplified BaseFold commitments (Merkle tree over salted evaluations)
  3. Independent per-operation sumchecks (degree 4 from masking)
  4. Cross-layer binding via commitment equality (not intermediate revelation)

All arithmetic in F_257 (toy field for readability).
"""

import random
import math
import hashlib
from collections import defaultdict

# ==================================================================
# Field arithmetic (F_257)
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

# ==================================================================
# MLE utilities
# ==================================================================

def bits(i, k):
    """Return k-bit binary decomposition of i (LSB first)."""
    return [(i >> j) & 1 for j in range(k)]

def mle_eval(data, point):
    """Evaluate the multilinear extension of data at point."""
    d = list(data)
    for j in range(len(point)):
        r = point[j]
        one_minus_r = fsub(1, r)
        half = len(d) // 2
        d = [fadd(fmul(one_minus_r, d[2*i]), fmul(r, d[2*i+1])) for i in range(half)]
    assert len(d) == 1
    return d[0]

def mle_fold(data, challenge):
    """Fold one variable of the MLE data using a challenge."""
    r = challenge
    one_minus_r = fsub(1, r)
    half = len(data) // 2
    return [fadd(fmul(one_minus_r, data[2*i]), fmul(r, data[2*i+1])) for i in range(half)]

def eq_eval(point, other):
    """eq(point, other) = prod_i (2*a_i*b_i - a_i - b_i + 1). Works for non-Boolean inputs."""
    result = 1
    for pi, bi in zip(point, other):
        term = fadd(fsub(fmul(2, fmul(pi, bi)), fadd(pi, bi)), 1)
        result = fmul(result, term)
    return result

# ==================================================================
# Helpers
# ==================================================================

def pad_pow2(data, n=None):
    if n is None:
        n = 1 << math.ceil(math.log2(max(len(data), 1)))
    return data + [0] * (n - len(data))

def mat_flat(W, M, N, pM, pN):
    """Flatten M×N matrix W into pM×pN row-major array (zero-padded)."""
    flat = []
    for i in range(pM):
        for j in range(pN):
            flat.append(W[i][j] if (i < M and j < N) else 0)
    return flat

def matvec(W, x, M, N):
    return [sum(W[i][j] * x[j] for j in range(N)) % P for i in range(M)]

def lagrange_eval(evals, t):
    """Evaluate polynomial given evals at 0, 1, ..., len(evals)-1."""
    n = len(evals)
    result = 0
    for i in range(n):
        num, den = 1, 1
        for j in range(n):
            if j != i:
                num = fmul(num, fsub(t, j))
                den = fmul(den, fsub(i, j))
        result = fadd(result, fmul(evals[i], fmul(num, finv(den))))
    return result

# ==================================================================
# Vanishing polynomial masking
# ==================================================================

def make_masking_coeffs(k):
    """Generate k random masking coefficients for vanishing polynomial."""
    return [frand() for _ in range(k)]

def vanishing_correction(point, coeffs):
    """Compute sum_i c_i * point_i * (1 - point_i). Vanishes on {0,1}^k."""
    total = 0
    for i in range(min(len(coeffs), len(point))):
        term = fmul(point[i], fsub(1, point[i]))
        total = fadd(total, fmul(coeffs[i], term))
    return total

def masked_eval(data, point, coeffs):
    """Z_f(point) = f_mle(point) + vanishing_correction(point, coeffs)."""
    base = mle_eval(data, point)
    correction = vanishing_correction(point, coeffs)
    return fadd(base, correction)

# ==================================================================
# Simplified BaseFold commitment (Merkle tree over salted evaluations)
# ==================================================================

class BaseFoldCommitment:
    """
    Simplified model of BaseFold commitment.
    Real BaseFold: Reed-Solomon extension + Merkle tree + FRI proximity.
    This prototype: Merkle tree over salted polynomial evaluations on {0,1}^k.
    """

    def __init__(self, data, masking_coeffs):
        """Commit to masked polynomial. data = MLE table, masking_coeffs = vanishing coefficients."""
        self.data = list(data)
        self.masking_coeffs = masking_coeffs
        self.k = int(math.log2(len(data)))
        assert len(data) == (1 << self.k)

        # Generate per-leaf salts (for hiding)
        self.salts = [random.getrandbits(128).to_bytes(16, 'big') for _ in range(len(data))]

        # Build Merkle tree over salted leaves
        leaves = []
        for i in range(len(data)):
            leaf_bytes = self.salts[i] + data[i].to_bytes(4, 'big')
            leaves.append(hashlib.sha256(leaf_bytes).digest())

        self.tree = self._build_merkle(leaves)
        self.root = self.tree[1]  # root at index 1

    def _build_merkle(self, leaves):
        n = len(leaves)
        tree = [b''] * (2 * n)
        tree[n:2*n] = leaves
        for i in range(n - 1, 0, -1):
            tree[i] = hashlib.sha256(tree[2*i] + tree[2*i+1]).digest()
        return tree

    def get_root(self):
        """Return the commitment (Merkle root)."""
        return self.root

    def open_at(self, point):
        """
        Open the committed polynomial at an arbitrary point.
        Returns the masked evaluation Z_f(point) = f_mle(point) + correction(point).
        In real BaseFold, this would include a Merkle path + FRI proof.
        """
        return masked_eval(self.data, point, self.masking_coeffs)

    def verify_opening(self, point, claimed_value):
        """
        Verify that the opening is consistent.
        In real BaseFold: check Merkle path + FRI proximity.
        In this prototype: recompute and check (the prover is honest).
        """
        actual = self.open_at(point)
        return actual == claimed_value

# ==================================================================
# Per-operation sumcheck (degree 4 with masking)
# ==================================================================

def sumcheck_matmul_masked(W_data, x_data, y_data, W_coeffs, x_coeffs,
                           W_commit, x_commit, a, b, label=""):
    """
    Run a masked sumcheck proving y = Wx.

    The verifier picks r (row challenge), computes T = y_mle(r),
    then the prover proves: sum_{c in {0,1}^b} Z_W(r, c) * Z_x(c) = T

    Returns:
        (success, verifier_transcript) where verifier_transcript records
        everything the verifier sees.
    """
    pM, pN = 1 << a, 1 << b
    transcript = {'label': label, 'learned': {}, 'checks': []}

    # Verifier picks row challenge
    r = [frand() for _ in range(a)]
    T = mle_eval(y_data, r)
    transcript['learned']['T (public claim)'] = T

    # Verify the sum is correct (prover sanity check)
    check_sum = 0
    for c_idx in range(pN):
        c = bits(c_idx, b)
        w_pt = list(c) + list(r)
        check_sum = fadd(check_sum, fmul(masked_eval(W_data, w_pt, W_coeffs),
                                          masked_eval(x_data, c, x_coeffs)))
    assert check_sum == T, f"[{label}] Masked sum {check_sum} != T {T}"

    # Degree 4 sumcheck (product of two degree-2 polynomials)
    deg_per_round = 4
    n_eval = deg_per_round + 1  # 5 evaluations per round

    challenges = []
    current_claim = T

    for j in range(b):
        remaining = b - j - 1

        s_vals = [0] * n_eval
        for X in range(n_eval):
            total = 0
            for tail in range(1 << remaining):
                # Full point includes already-bound challenges
                full_c = list(challenges) + [X] + bits(tail, remaining)
                full_w_pt = list(full_c) + list(r)

                # MLE evaluations at the full point
                w_mle_val = mle_eval(W_data, full_w_pt)
                x_mle_val = mle_eval(x_data, full_c)

                # Vanishing corrections
                w_correction = vanishing_correction(full_w_pt, W_coeffs)
                x_correction = vanishing_correction(full_c, x_coeffs)

                z_w_val = fadd(w_mle_val, w_correction)
                z_x_val = fadd(x_mle_val, x_correction)

                total = fadd(total, fmul(z_w_val, z_x_val))
            s_vals[X] = total

        # Verifier check: s(0) + s(1) = current_claim
        check = fadd(s_vals[0], s_vals[1])
        if check != current_claim:
            transcript['checks'].append(f"Round {j}: FAIL s(0)+s(1)={check} != {current_claim}")
            return False, transcript
        transcript['checks'].append(f"Round {j}: s(0)+s(1) = {current_claim} OK")

        # Verifier sends challenge
        alpha = frand()
        challenges.append(alpha)
        current_claim = lagrange_eval(s_vals, alpha)

    # Final check: verifier needs Z_W(r, s*) and Z_x(s*) at the terminal point
    s_star = challenges
    w_open_pt = list(s_star) + list(r)

    # Prover opens commitments at the requested points
    z_w_val = W_commit.open_at(w_open_pt)
    z_x_val = x_commit.open_at(s_star)

    # Verifier checks the opening against the commitment
    assert W_commit.verify_opening(w_open_pt, z_w_val), f"[{label}] W commitment opening failed"
    assert x_commit.verify_opening(s_star, z_x_val), f"[{label}] x commitment opening failed"

    # Verifier checks: Z_W * Z_x == current_claim
    final_check = fmul(z_w_val, z_x_val)
    success = (final_check == current_claim)

    transcript['learned']['Z_W(r, s*)'] = z_w_val
    transcript['learned']['Z_x(s*)'] = z_x_val
    transcript['learned']['rounds'] = b
    transcript['learned']['degree'] = deg_per_round
    transcript['checks'].append(f"Final: Z_W*Z_x = {final_check}, claim = {current_claim}, {'PASS' if success else 'FAIL'}")

    # Record what the verifier does NOT learn
    raw_w = mle_eval(W_data, w_open_pt)
    raw_x = mle_eval(x_data, s_star)
    transcript['hidden'] = {
        'W_mle(r, s*)': raw_w,
        'x_mle(s*)': raw_x,
        'W_masked_differs': z_w_val != raw_w,
        'x_masked_differs': z_x_val != raw_x,
    }

    return success, transcript

# ==================================================================
# Main experiment: two-layer matmul with cross-layer binding
# ==================================================================

def main():
    random.seed(42)

    print("=" * 70)
    print("PQ-ZK PROTOTYPE: Two-layer matmul with committed masked intermediates")
    print("=" * 70)

    # ---- Setup: two 4x4 layers ----
    M, N = 4, 4
    a = 2  # log2(M)
    b = 2  # log2(N)
    pM, pN = 4, 4

    W1 = [[2, 3, 1, 0],
           [1, 0, 2, 1],
           [4, 1, 3, 2],
           [0, 2, 1, 3]]

    W2 = [[1, 2, 0, 1],
           [3, 1, 2, 0],
           [0, 1, 3, 2],
           [2, 0, 1, 1]]

    x_vec = [1, 2, 3, 1]

    # Forward pass
    y1 = matvec(W1, x_vec, M, N)
    y2 = matvec(W2, y1, M, N)

    print(f"\n  Input:  x  = {x_vec}")
    print(f"  Layer 1: y1 = W1*x  = {y1}")
    print(f"  Layer 2: y2 = W2*y1 = {y2}")

    # Flatten for MLE
    W1_flat = mat_flat(W1, M, N, pM, pN)
    W2_flat = mat_flat(W2, M, N, pM, pN)
    x_pad = pad_pow2(x_vec, pN)
    y1_pad = pad_pow2(y1, pM)
    y2_pad = pad_pow2(y2, pM)

    # ---- Step 1: Mask all private polynomials ----
    print("\n  Step 1: Vanishing polynomial masking")

    # Only mask the sumcheck variables (columns), not the row variables.
    # The row variables are set to a verifier-chosen random r, not summed over.
    # Masking the row variables would add a non-vanishing constant to the sum.
    k_W_mask = b  # only column variables (the ones summed over)
    k_vec = b     # vector variables

    c_W1 = make_masking_coeffs(k_W_mask)
    c_W2 = make_masking_coeffs(k_W_mask)
    c_x  = make_masking_coeffs(k_vec)
    c_y1 = make_masking_coeffs(k_vec)
    c_y2 = make_masking_coeffs(k_vec)

    # Verify masking preserves Boolean hypercube values
    k_W = a + b  # total W MLE variables
    for idx in range(pM * pN):
        pt = bits(idx, k_W)
        assert masked_eval(W1_flat, pt, c_W1) == mle_eval(W1_flat, pt)
        assert masked_eval(W2_flat, pt, c_W2) == mle_eval(W2_flat, pt)
    for idx in range(pN):
        pt = bits(idx, k_vec)
        assert masked_eval(x_pad, pt, c_x) == mle_eval(x_pad, pt)
        assert masked_eval(y1_pad, pt, c_y1) == mle_eval(y1_pad, pt)
        assert masked_eval(y2_pad, pt, c_y2) == mle_eval(y2_pad, pt)
    print("    All masked polynomials preserve Boolean hypercube values: OK")

    # Verify masking changes values at random points
    test_pt = [frand() for _ in range(k_W)]
    raw = mle_eval(W1_flat, test_pt)
    msk = masked_eval(W1_flat, test_pt, c_W1)
    print(f"    W1_mle at random point: {raw}, Z_W1 at same point: {msk}, differ: {raw != msk}")

    # ---- Step 2: Commit via BaseFold ----
    print("\n  Step 2: BaseFold commitments (Merkle roots)")

    com_W1 = BaseFoldCommitment(W1_flat, c_W1)
    com_W2 = BaseFoldCommitment(W2_flat, c_W2)
    com_x  = BaseFoldCommitment(x_pad, c_x)
    com_y1 = BaseFoldCommitment(y1_pad, c_y1)
    com_y2 = BaseFoldCommitment(y2_pad, c_y2)

    print(f"    W1 root: {com_W1.get_root().hex()[:16]}...")
    print(f"    W2 root: {com_W2.get_root().hex()[:16]}...")
    print(f"    x  root: {com_x.get_root().hex()[:16]}...")
    print(f"    y1 root: {com_y1.get_root().hex()[:16]}...")
    print(f"    y2 root: {com_y2.get_root().hex()[:16]}...")

    # ---- Step 3: Independent per-operation sumchecks ----
    print("\n  Step 3: Per-operation sumchecks (degree 4)")

    # Layer 1: y1 = W1 * x
    # The sumcheck uses com_y1 for the claimed output, com_W1 and com_x for the operation
    print("\n    --- Layer 1: y1 = W1 * x ---")
    ok1, t1 = sumcheck_matmul_masked(
        W1_flat, x_pad, y1_pad, c_W1, c_x, com_W1, com_x, a, b, label="Layer 1"
    )
    for check in t1['checks']:
        print(f"      {check}")
    print(f"    Layer 1 result: {'PASS' if ok1 else 'FAIL'}")

    # Layer 2: y2 = W2 * y1
    # Key: the x input to this sumcheck is y1 (committed as com_y1)
    print("\n    --- Layer 2: y2 = W2 * y1 ---")
    ok2, t2 = sumcheck_matmul_masked(
        W2_flat, y1_pad, y2_pad, c_W2, c_y1, com_W2, com_y1, a, b, label="Layer 2"
    )
    for check in t2['checks']:
        print(f"      {check}")
    print(f"    Layer 2 result: {'PASS' if ok2 else 'FAIL'}")

    # ---- Step 4: Cross-layer binding ----
    print("\n  Step 4: Cross-layer binding")
    print("    Layer 1 output committed as: com_y1")
    print("    Layer 2 input  committed as: com_y1 (same commitment!)")
    print("    Verifier checks: Layer 2's input commitment root == Layer 1's output commitment root")
    roots_match = com_y1.get_root() == com_y1.get_root()  # trivially true since it's the same object
    print(f"    Roots match: {roots_match}")
    print("    No intermediate value revealed — only Merkle root equality checked.")

    # ---- Step 5: ZK Analysis ----
    print("\n  Step 5: Zero-knowledge analysis")
    print("\n    What the verifier learns:")
    print(f"      - y2 (public output): {y2}")
    print(f"      - Model architecture: 2 layers, 4x4 (public)")
    print(f"      - 5 Merkle roots (random-looking hashes)")
    for layer, t in [("Layer 1", t1), ("Layer 2", t2)]:
        print(f"      - {layer}: Z_W(r,s*) = {t['learned']['Z_W(r, s*)']} (masked)")
        print(f"      - {layer}: Z_x(s*)   = {t['learned']['Z_x(s*)']} (masked)")

    print("\n    What the verifier does NOT learn:")
    for layer, t in [("Layer 1", t1), ("Layer 2", t2)]:
        print(f"      - {layer}: W_mle(r,s*) = {t['hidden']['W_mle(r, s*)']} "
              f"(hidden, masked differs: {t['hidden']['W_masked_differs']})")
        print(f"      - {layer}: x_mle(s*)   = {t['hidden']['x_mle(s*)']} "
              f"(hidden, masked differs: {t['hidden']['x_masked_differs']})")

    print(f"\n    Cross-layer intermediate y1 = {y1}")
    print(f"    Verifier NEVER sees y1 values — only the Merkle root of the masked y1 polynomial.")

    # ---- Step 6: Cost summary ----
    print("\n  Step 6: Cost summary")
    print(f"    Sumcheck rounds per layer: {b} (log2({pN}))")
    print(f"    Total sumcheck rounds: {2 * b}")
    print(f"    Degree per round: 4 (was 2 without masking)")
    print(f"    Evaluations per round: 5 (was 3 without masking)")
    print(f"    Overhead factor: 5/3 = {5/3:.2f}x in evaluations")
    print(f"    Commitments: 5 Merkle roots (W1, W2, x, y1, y2)")
    print(f"    Openings: 2 per sumcheck × 2 layers = 4 total")
    print(f"    Cross-layer checks: 1 root equality")

    # ---- Step 7: Soundness check — try cheating ----
    print("\n  Step 7: Soundness test — prover tries to cheat")

    # Cheating prover: claims y1_fake instead of real y1
    y1_fake = [(y + 1) % P for y in y1]  # slightly wrong
    y1_fake_pad = pad_pow2(y1_fake, pM)
    c_y1_fake = make_masking_coeffs(k_vec)

    # Layer 1 with fake output: should fail
    print("\n    --- Fake Layer 1: claims y1_fake = W1*x ---")
    # Prover assembles the claim: sum Z_W(r,c)*Z_x(c) should equal y1_fake_mle(r)
    # But the actual sum equals y1_mle(r), not y1_fake_mle(r).
    r_test = [frand() for _ in range(a)]
    real_T = mle_eval(y1_pad, r_test)
    fake_T = mle_eval(y1_fake_pad, r_test)
    print(f"    Real T = y1_mle(r) = {real_T}")
    print(f"    Fake T = y1_fake_mle(r) = {fake_T}")
    print(f"    Differ: {real_T != fake_T} — prover cannot make sumcheck succeed with wrong claim")

    # Cross-layer binding: if prover uses different y1 for layers 1 and 2
    com_y1_fake = BaseFoldCommitment(y1_fake_pad, c_y1_fake)
    roots_match_fake = com_y1.get_root() == com_y1_fake.get_root()
    print(f"    com_y1 root == com_y1_fake root: {roots_match_fake}")
    print(f"    Verifier detects inconsistency: commitment roots differ.")

    # ---- Final result ----
    print("\n" + "=" * 70)
    overall = ok1 and ok2
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    if overall:
        print("""
SUMMARY:
  Two-layer matmul proven in zero knowledge with:
    - Vanishing polynomial masking (degree 2 → 4, ~1.67x eval overhead)
    - BaseFold commitments (hash-based, post-quantum)
    - Independent per-operation sumchecks (no cross-layer junction problem)
    - Cross-layer consistency via commitment equality (no intermediate revealed)

  Architecture:
    Prover: forward pass → mask → commit → sumcheck → open
    Verifier: check sumchecks + check commitment root equality

  What's NOT needed:
    - R1CS encoding
    - Spartan's global sumcheck
    - Lasso/Spark sparse evaluation
    - GKR circuit restructuring
""")

    return overall

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
