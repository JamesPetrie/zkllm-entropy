"""
zk_approach_experiments.py -- Comparing three approaches to ZK matmul proofs.

Experiment 3: Vanishing polynomial masking on direct sumcheck
Experiment 1: GKR circuit representation for matmul
Experiment 2: R1CS + Spartan-style two sumchecks

All experiments use the same 3x3 matmul (padded to 4x4) in F_257.
Each experiment measures: field ops, sumcheck rounds, what the verifier learns.
"""

import random
import math
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
    return [(i >> j) & 1 for j in range(k)]

def mle_eval(data, point):
    d = list(data)
    for j in range(len(point)):
        r = point[j]
        one_minus_r = fsub(1, r)
        half = len(d) // 2
        d = [fadd(fmul(one_minus_r, d[2*i]), fmul(r, d[2*i+1])) for i in range(half)]
    assert len(d) == 1
    return d[0]

def mle_fold(data, challenge):
    r = challenge
    one_minus_r = fsub(1, r)
    half = len(data) // 2
    return [fadd(fmul(one_minus_r, data[2*i]), fmul(r, data[2*i+1])) for i in range(half)]

def eq_eval(point, other):
    """eq(point, other) = prod_i (point_i * other_i + (1-point_i)*(1-other_i))
    Works for both Boolean and non-Boolean inputs."""
    result = 1
    for pi, bi in zip(point, other):
        # pi*bi + (1-pi)*(1-bi) = 2*pi*bi - pi - bi + 1
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
    flat = []
    for i in range(pM):
        for j in range(pN):
            flat.append(W[i][j] if (i < M and j < N) else 0)
    return flat

def matvec(W, x, M, N):
    return [sum(W[i][j] * x[j] for j in range(N)) % P for i in range(M)]

def eval_degree2(evals, t):
    f0, f1, f2 = evals
    inv2 = finv(2)
    L0 = fmul(fmul(fsub(t, 1), fsub(t, 2)), inv2)
    L1 = fneg(fmul(t, fsub(t, 2)))
    L2 = fmul(fmul(t, fsub(t, 1)), inv2)
    return fadd(fadd(fmul(f0, L0), fmul(f1, L1)), fmul(f2, L2))

def lagrange_eval(evals, t):
    """Evaluate a polynomial given evals at 0, 1, ..., len(evals)-1."""
    n = len(evals)
    nodes = list(range(n))
    result = 0
    for i in range(n):
        num, den = 1, 1
        for j in nodes:
            if j != i:
                num = fmul(num, fsub(t, j))
                den = fmul(den, fsub(i, j))
        result = fadd(result, fmul(evals[i], fmul(num, finv(den))))
    return result

class OpCounter:
    def __init__(self):
        self.counts = defaultdict(int)
    def count(self, op, n=1):
        self.counts[op] += n
    def report(self):
        for k, v in sorted(self.counts.items()):
            print(f"    {k}: {v}")


# ######################################################################
#
# EXPERIMENT 3: Vanishing polynomial masking on direct sumcheck
#
# ######################################################################

def experiment_3():
    """
    Test adding X_i(1-X_i) vanishing polynomial masking to witness polynomials,
    plus proper g + rho*p sumcheck masking.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Vanishing polynomial masking on direct sumcheck")
    print("=" * 70)

    # Setup: 3x3 matmul, padded to 4x4
    W = [[2, 3, 1], [1, 0, 2], [4, 1, 3]]
    x_vec = [1, 2, 3]
    y_vec = matvec(W, x_vec, 3, 3)

    M, N = 3, 3
    a = max(1, math.ceil(math.log2(M)))  # 2
    b = max(1, math.ceil(math.log2(N)))  # 2
    pM, pN = 1 << a, 1 << b  # 4, 4

    W_flat = mat_flat(W, M, N, pM, pN)
    x_pad = pad_pow2(x_vec, pN)
    y_pad = pad_pow2(y_vec, pM)

    ops = OpCounter()

    # ---- Step 1: Vanishing polynomial masking ----
    # Replace W_flat with Z_W = W_flat + masking terms
    # For multilinear extension with k variables:
    #   Z(X_1,...,X_k) = f(X_1,...,X_k) + sum_{i=1}^{num_mask} c_i * X_i * (1 - X_i)
    #
    # X_i(1-X_i) vanishes on {0,1}, so Z agrees with f on the Boolean hypercube.
    # But Z is NOT multilinear (degree 2 in X_i).
    #
    # Problem: our MLE machinery assumes data is the multilinear extension.
    # X_i(1-X_i) cannot be represented as an MLE table entry.
    # We need to evaluate Z(point) = f(point) + sum c_i * point_i * (1 - point_i).

    # For W: k = a + b = 4 variables. We need 2 masking coefficients (Thaler says 2 suffice).
    k_W = a + b
    c_W = [frand(), frand()]  # masking coefficients for W

    # For x: k = b = 2 variables. We need 2 masking coefficients.
    k_x = b
    c_x = [frand(), frand()]

    def vanishing_correction(point, coeffs):
        """Compute sum_i c_i * point_i * (1 - point_i)."""
        total = 0
        for i in range(min(len(coeffs), len(point))):
            # X_i * (1 - X_i) at point_i
            term = fmul(point[i], fsub(1, point[i]))
            total = fadd(total, fmul(coeffs[i], term))
        return total

    def masked_eval_W(point):
        """Z_W(point) = W_mle(point) + vanishing_correction(point, c_W)"""
        base = mle_eval(W_flat, point)
        correction = vanishing_correction(point, c_W)
        return fadd(base, correction)

    def masked_eval_x(point):
        """Z_x(point) = x_mle(point) + vanishing_correction(point, c_x)"""
        base = mle_eval(x_pad, point)
        correction = vanishing_correction(point, c_x)
        return fadd(base, correction)

    # Verify masking doesn't change values on Boolean hypercube
    print("\n  Step 1: Vanishing polynomial masking")
    for idx in range(pM * pN):
        pt = bits(idx, k_W)
        assert masked_eval_W(pt) == mle_eval(W_flat, pt), f"Masking changed W at Boolean point {pt}"
    for idx in range(pN):
        pt = bits(idx, k_x)
        assert masked_eval_x(pt) == mle_eval(x_pad, pt), f"Masking changed x at Boolean point {pt}"
    print("    Masking preserves Boolean hypercube values: OK")

    # Check that masked evals at non-Boolean points differ
    test_pt = [frand() for _ in range(k_W)]
    raw = mle_eval(W_flat, test_pt)
    masked = masked_eval_W(test_pt)
    print(f"    W_mle at random point: {raw}, Z_W at same point: {masked}, differ: {raw != masked}")

    # ---- Step 2: Sumcheck with g + rho*p masking ----
    # The sumcheck proves: sum_c Z_W(r, c) * Z_x(c) = T
    # But wait: T = y_tilde(r) = sum_c W_mle(r, c) * x_mle(c)
    # And: sum_c Z_W(r, c) * Z_x(c) = sum_c [W_mle(r,c) + correction_W(r,c)] * [x_mle(c) + correction_x(c)]
    # This is NOT equal to T in general!
    #
    # The vanishing polynomial masking is applied to the COMMITTED polynomial,
    # but the sumcheck relation is over the Boolean hypercube where the masking vanishes.
    # So: sum_{c in {0,1}^b} Z_W(r, c) * Z_x(c) = sum_{c in {0,1}^b} W_mle(r, c) * x_mle(c) = T
    # because the corrections vanish on {0,1}^b.
    #
    # BUT: the sumcheck round polynomials evaluate at non-Boolean points (X = 0, 1, 2, ...),
    # and at those points the corrections are NON-ZERO. So the round polynomials
    # computed from Z_W and Z_x will differ from those computed from W_mle and x_mle.
    # The degree also increases (Z_W is degree 2 in the sumcheck variable, not degree 1).

    print("\n  Step 2: Sumcheck with vanishing-masked polynomials")

    # Verifier picks row challenge
    r = [frand() for _ in range(a)]
    T = mle_eval(y_pad, r)

    # Verify the sum over Boolean hypercube is still T with masked polynomials
    check_sum = 0
    for c_idx in range(pN):
        c = bits(c_idx, b)
        w_pt = list(c) + list(r)  # col vars first, then row vars
        check_sum = fadd(check_sum, fmul(masked_eval_W(w_pt), masked_eval_x(c)))
    assert check_sum == T, f"Masked sum {check_sum} != T {T}"
    print(f"    Sum over Boolean hypercube with masked polys = {T}: OK (corrections vanish)")

    # ---- Step 2b: g + rho*p sumcheck masking ----
    # Prover generates masking polynomial p = a_0 + sum_i p_i(x_i)
    # where each p_i is a univariate of degree d (= degree of the sumcheck poly in that variable).
    #
    # The sumcheck polynomial is g(c) = Z_W(r, c) * Z_x(c).
    # g has degree 2 in each variable (because Z_W is degree 2 in c_i, Z_x is degree 1 in c_i,
    # product is degree 3... actually let's check).
    #
    # Z_W(r, c_1, ..., c_b) = W_mle(r, c) + c_1*c_W[0]*(1-c_1) + c_2*c_W[1]*(1-c_2)
    # (if we mask variables c_1 and c_2, which are the sumcheck variables)
    # Z_W has degree 2 in c_1 (from c_1*(1-c_1) = c_1 - c_1^2) and degree 1 in c_i for i>2
    # (well, it also has degree 2 in c_2 from the second masking term)
    # Z_x(c) has degree 2 in c_1 and c_2 from the same masking.
    # Product Z_W * Z_x has degree up to 4 in c_1.
    #
    # This means the round polynomial degree jumps from 2 (without masking) to 4.
    # The prover needs to send 5 evaluations per round instead of 3.

    # Determine degree: Z_W is degree 2 in sumcheck var, Z_x is degree 2 in sumcheck var.
    # Product is degree 4. With the g + rho*p masking, p has the same degree, so still degree 4.
    deg_per_round = 4  # degree of round polynomial in the sumcheck variable
    n_eval = deg_per_round + 1  # 5 evaluations needed per round

    print(f"    Degree per round (with vanishing masking): {deg_per_round}")
    print(f"    Evaluations per round: {n_eval} (was 3 without masking)")

    # Generate masking polynomial p = a_0 + p_1(x_1) + p_2(x_2)
    # Each p_i is degree deg_per_round: p_i(x) = a_{i,1}*x + a_{i,2}*x^2 + ... + a_{i,d}*x^d
    p_coeffs = []  # [a_0, [a_{1,1},...,a_{1,d}], [a_{2,1},...,a_{2,d}]]
    p_a0 = frand()
    p_coeffs.append(p_a0)
    for _ in range(b):
        p_coeffs.append([frand() for _ in range(deg_per_round)])

    def eval_mask_poly_p(point):
        """Evaluate p(x_1,...,x_b) = a_0 + sum_i p_i(x_i)"""
        result = p_coeffs[0]
        for i in range(len(point)):
            p_i = p_coeffs[1 + i]
            xi = point[i]
            xi_pow = xi
            for coeff in p_i:
                result = fadd(result, fmul(coeff, xi_pow))
                xi_pow = fmul(xi_pow, xi)
        return result

    # Compute P = sum_{c in {0,1}^b} p(c)
    P_sum = 0
    for c_idx in range(pN):
        c = bits(c_idx, b)
        P_sum = fadd(P_sum, eval_mask_poly_p(c))
    ops.count('p_eval_for_sum', pN)

    # Prover commits to p (in real protocol). For the toy, we just note it.
    # Verifier picks rho
    rho = frand()

    # The combined claim: T + rho * P_sum
    combined_claim = fadd(T, fmul(rho, P_sum))
    print(f"    T = {T}, P = {P_sum}, rho = {rho}, combined claim = {combined_claim}")

    # ---- Run the sumcheck on g + rho*p ----
    print(f"\n  Step 3: Running sumcheck on g(c) + rho*p(c)")

    # We need to evaluate g(c) + rho*p(c) where g(c) = Z_W(r, c) * Z_x(c)
    # For the sumcheck, at round j, fix variables c_0,...,c_{j-1} and sum over c_{j+1},...,c_{b-1}

    # We'll track the folded W_data, x_data for the MLE part,
    # and compute corrections on the fly.
    W_data = list(W_flat)
    x_data = list(x_pad)

    current_claim = combined_claim
    challenges = []
    round_polys = []

    for j in range(b):
        remaining = b - j - 1

        s_vals = [0] * n_eval
        for X in range(n_eval):
            total = 0
            for tail in range(1 << remaining):
                c_rem = [X] + bits(tail, remaining)
                # W point: remaining col vars + row vars
                w_pt = list(c_rem) + list(r)

                # MLE part (using folded data)
                w_mle_val = mle_eval(W_data, w_pt)
                x_mle_val = mle_eval(x_data, c_rem)

                # Vanishing correction part
                # Build the full point for correction evaluation
                # The sumcheck variables already bound: challenges[0], ..., challenges[j-1]
                # Current variable: X
                # Remaining: bits(tail, remaining)
                full_c = list(challenges) + [X] + bits(tail, remaining)
                full_w_pt = list(full_c) + list(r)

                w_correction = vanishing_correction(full_w_pt, c_W)
                x_correction = vanishing_correction(full_c, c_x)

                z_w_val = fadd(w_mle_val, w_correction)
                z_x_val = fadd(x_mle_val, x_correction)

                g_val = fmul(z_w_val, z_x_val)

                # Masking polynomial p
                p_val = eval_mask_poly_p(full_c)

                total = fadd(total, fadd(g_val, fmul(rho, p_val)))
                ops.count('field_muls', 8)  # rough count per inner iteration
            s_vals[X] = total

        # Verify s(0) + s(1) = current_claim
        check = fadd(s_vals[0], s_vals[1])
        if check != current_claim:
            print(f"    Round {j}: SUMCHECK FAILED s(0)+s(1) = {check} != {current_claim}")
            return False
        round_polys.append(s_vals)

        # Verifier challenge
        alpha = frand()
        challenges.append(alpha)
        current_claim = lagrange_eval(s_vals, alpha)

        # Fold MLE data
        W_data = mle_fold(W_data, alpha)
        x_data = mle_fold(x_data, alpha)

        print(f"    Round {j}: s = {s_vals}, alpha = {alpha}")

    # ---- Final check ----
    # After b rounds, challenges = [alpha_0, ..., alpha_{b-1}] = s
    s = challenges
    full_w_point = list(s) + list(r)

    # The verifier needs Z_W(r, s) and Z_x(s) to check the final claim.
    # Z_W(r, s) = W_mle(r, s) + correction_W(r, s)
    # Z_x(s) = x_mle(s) + correction_x(s)
    #
    # The verifier learns Z_W(r, s) and Z_x(s), NOT W_mle(r,s) and x_mle(s).
    # The corrections depend on the masking coefficients c_W, c_x which the prover chose.
    # If the verifier doesn't know c_W, c_x, they can't extract the raw MLE values.

    z_W_final = masked_eval_W(full_w_point)
    z_x_final = masked_eval_x(s)

    # Also need p(s) for the masking check
    p_at_s = eval_mask_poly_p(s)

    expected = fadd(fmul(z_W_final, z_x_final), fmul(rho, p_at_s))
    print(f"\n  Step 4: Final check")
    print(f"    Z_W(r,s) = {z_W_final}, Z_x(s) = {z_x_final}, p(s) = {p_at_s}")
    print(f"    Z_W * Z_x + rho * p(s) = {expected}, current_claim = {current_claim}")
    final_ok = (expected == current_claim)
    print(f"    Final check: {'PASS' if final_ok else 'FAIL'}")

    # ---- Analysis: What does the verifier learn? ----
    print(f"\n  Step 5: ZK analysis")
    print(f"    Verifier learns:")
    print(f"      - Z_W(r,s) = {z_W_final} (masked, not raw W_mle(r,s)={mle_eval(W_flat, full_w_point)})")
    print(f"      - Z_x(s) = {z_x_final} (masked, not raw x_mle(s)={mle_eval(x_pad, s)})")
    print(f"      - p(s) = {p_at_s} (from commitment opening)")
    print(f"      - Round polynomials are masked by rho*p")
    print(f"    Verifier does NOT learn:")
    print(f"      - W_mle(r,s) or x_mle(s) (hidden behind vanishing masking)")
    print(f"      - c_W, c_x (masking coefficients, not revealed)")

    # Cross-layer problem:
    print(f"\n  Step 6: Cross-layer junction analysis")
    print(f"    At the junction, the verifier would need h_tilde(s) for layer 1's claim.")
    print(f"    With masking, they get Z_h(s) instead.")
    print(f"    Z_h(s) = h_mle(s) + correction_h(s)")
    print(f"    To set up layer 1: the claim is h_mle(s) = Z_h(s) - correction_h(s)")
    print(f"    The prover must reveal correction_h(s) so the verifier can compute h_mle(s).")
    print(f"    correction_h(s) = sum_i c_h_i * s_i * (1 - s_i)")
    print(f"    The prover can reveal correction_h(s) as a scalar without revealing c_h_i individually,")
    print(f"    IF c_h has more coefficients than evaluation points.")
    print(f"    With 2 coefficients and 1 evaluation point: the verifier learns one linear")
    print(f"    equation in 2 unknowns. This does NOT uniquely determine c_h.")
    print(f"    But the verifier DOES learn h_mle(s) = Z_h(s) - correction_h(s).")
    print(f"    This is the fundamental problem: the verifier needs h_mle(s) for the next layer.")

    # Degree increase analysis
    print(f"\n  Step 7: Cost analysis")
    print(f"    Without vanishing masking: degree 2 per round, 3 evals/round")
    print(f"    With vanishing masking: degree {deg_per_round} per round, {n_eval} evals/round")
    print(f"    Overhead: {n_eval/3:.1f}x more evaluations per round")
    print(f"    Sumcheck rounds: {b} (unchanged)")
    ops.count('sumcheck_rounds', b)
    ops.count('evals_per_round', n_eval * b)

    ops.report()
    return final_ok


# ######################################################################
#
# EXPERIMENT 1: GKR circuit representation for matmul
#
# ######################################################################

def experiment_1():
    """
    Represent 3x3 matmul as a layered arithmetic circuit and prove via GKR.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: GKR circuit representation for matmul")
    print("=" * 70)

    # Setup: y = W * x, 3x3 padded to 4x4
    W = [[2, 3, 1], [1, 0, 2], [4, 1, 3]]
    x_vec = [1, 2, 3]
    y_vec = matvec(W, x_vec, 3, 3)
    M, N = 3, 3
    pM, pN = 4, 4

    ops = OpCounter()

    # ---- Build the layered circuit ----
    # Layer 0 (output): y values, size pM = 4
    # Layer 1 (sum): partial sums, size pM * pN = 16 (4 rows * 4 cols)
    #   Actually for a binary reduction tree: we need log(pN) addition layers.
    #   But for simplicity, let's use the Thaler-style "one sumcheck per matmul"
    #   circuit which has:
    #
    # Layer 0 (output): y[0..pM-1], size pM
    # Layer 1 (multiplication): W[i][j] * x[j], size pM * pN
    # Layer 2 (input): [W_flat, x_pad], size pM*pN + pN
    #
    # The GKR sumcheck at layer 0->1 proves:
    #   V_0(g) = sum_{j in {0,1}^b} mult(g, j) * V_1(g, j) * V_2(j)
    # But this isn't standard GKR either. Let me use the actual GKR structure.
    #
    # Standard GKR for matmul y = Wx:
    #   Circuit depth = 1 + ceil(log2(N)) layers
    #   Layer d (input): W entries and x entries
    #   Layer d-1 (multiply): products W[i][j] * x[j] for each (i,j)
    #   Layers d-2 to 0: binary tree summing products per row

    print("\n  Step 1: Building layered circuit for 4x4 matmul")

    # Build the actual circuit values
    W_pad = [[W[i][j] if i < M and j < N else 0 for j in range(pN)] for i in range(pM)]
    x_pad = pad_pow2(x_vec, pN)
    y_pad = pad_pow2(y_vec, pM)

    # Layer 0 (output): y values, 4 gates
    layer_0 = list(y_pad)

    # Layer 1: partial sums after first addition.
    # For pN=4 columns, we need log2(4)=2 addition layers + 1 mult layer.
    # Mult layer: gate (i, j) = W[i][j] * x[j], total pM*pN = 16 gates
    mult_layer = [fmul(W_pad[i][j], x_pad[j]) for i in range(pM) for j in range(pN)]

    # Addition layer 1: pairs (i, 2k) and (i, 2k+1) summed
    add_layer_1 = []
    for i in range(pM):
        for k in range(pN // 2):
            add_layer_1.append(fadd(mult_layer[i * pN + 2*k], mult_layer[i * pN + 2*k + 1]))

    # Addition layer 2 (= output): pairs summed again
    add_layer_2 = []
    for i in range(pM):
        for k in range(pN // 4):
            add_layer_2.append(fadd(add_layer_1[i * (pN//2) + 2*k], add_layer_1[i * (pN//2) + 2*k + 1]))

    assert add_layer_2 == layer_0, f"Circuit output mismatch: {add_layer_2} != {layer_0}"

    layers = [layer_0, add_layer_1, mult_layer]
    # Input layer: [W_flat, x_pad] conceptually, but mult_layer takes direct products
    # For GKR, the input layer is the mult_layer's inputs = W and x

    print(f"    Layer 0 (output): {len(layer_0)} gates = {layer_0}")
    print(f"    Layer 1 (add):    {len(add_layer_1)} gates")
    print(f"    Layer 2 (add):    {len(mult_layer)} gates (actually mult)")
    print(f"    Depth: 3 layers (1 mult + 2 add)")

    # ---- GKR Protocol ----
    print("\n  Step 2: GKR protocol")

    # GKR starts at the output layer. Verifier picks random g and computes V_0(g).
    # Then runs sumcheck at each layer to reduce to input claims.

    num_output_vars = max(1, math.ceil(math.log2(len(layer_0))))  # 2
    g = [frand() for _ in range(num_output_vars)]
    V0_claim = mle_eval(layer_0, g)

    print(f"    Verifier picks g = {g}")
    print(f"    V_0(g) = {V0_claim}")

    # ---- Layer 0 -> Layer 1 sumcheck (addition layer) ----
    # V_0(g) = sum_{j in {0,1}} add_wiring(g, j) * [V_1(g, 2j) + V_1(g, 2j+1)]
    # For our binary tree: gate g in layer 0 takes inputs (g, 0) and (g, 1) from layer 1
    # V_0(g) = V_1(g||0) + V_1(g||1)
    # This is an addition gate with a trivial wiring.
    # The sumcheck is over 1 variable j: sum_j V_1(g || j)
    # where (g || j) means appending j to g.

    # Layer 1 has pM * (pN/2) = 4 * 2 = 8 gates, so 3 address bits
    num_L1_vars = max(1, math.ceil(math.log2(len(add_layer_1))))  # 3

    # For the binary tree addition, the sumcheck reduces V_0(g) to V_1(g, alpha_1)
    # via a 1-round sumcheck.

    print(f"\n    --- Layer 0->1 (addition tree, top) ---")
    # Data layout: add_layer_1[i*2 + k], so bit 0 = k (col pair), bits 1,2 = row.
    # Relation: layer_0[i] = add_layer_1[2i] + add_layer_1[2i+1]
    # In MLE terms: V_0(g) = sum_{j in {0,1}} V_1(j, g_0, g_1)
    # where j is prepended as variable 0 (the LSB = k axis).

    # Sumcheck: 1 round, degree 1 polynomial
    s_vals_01 = [0, 0]
    for j in range(2):
        pt = [j] + list(g)
        s_vals_01[j] = mle_eval(add_layer_1, pt)
    assert fadd(s_vals_01[0], s_vals_01[1]) == V0_claim
    alpha_01 = frand()
    V1_claim = fadd(fmul(fsub(1, alpha_01), s_vals_01[0]), fmul(alpha_01, s_vals_01[1]))
    u1 = [alpha_01] + list(g)
    ops.count('sumcheck_rounds', 1)
    ops.count('field_muls_sumcheck', 2 * len(add_layer_1))

    print(f"    1-round sumcheck: s = {s_vals_01}, alpha = {alpha_01}")
    print(f"    New claim: V_1({u1}) = {V1_claim}")

    # ---- Layer 1 -> Layer 2 sumcheck (second addition layer) ----
    # V_1(u1) = sum_{j in {0,1}} V_2(u1_row_bits, u1_col * 2 + j)
    # or more precisely: V_1 at point u1 (3 vars) reduces to V_2 at some point (4 vars)

    print(f"\n    --- Layer 1->2 (addition tree, bottom = mult layer) ---")
    num_L2_vars = max(1, math.ceil(math.log2(len(mult_layer))))  # 4

    # Data layout: mult_layer[i*pN + j], so bits 0,1 = col, bits 2,3 = row.
    # add_layer_1[m] = mult_layer[2m] + mult_layer[2m+1], same binary tree.
    # V_1(u1) = sum_{j in {0,1}} V_2(j, u1_0, u1_1, u1_2)
    # where j is prepended as variable 0 (the new LSB).

    s_vals_12 = [0, 0]
    for j in range(2):
        pt = [j] + list(u1)
        s_vals_12[j] = mle_eval(mult_layer, pt)
    assert fadd(s_vals_12[0], s_vals_12[1]) == V1_claim
    alpha_12 = frand()
    V2_claim = fadd(fmul(fsub(1, alpha_12), s_vals_12[0]), fmul(alpha_12, s_vals_12[1]))
    u2 = [alpha_12] + list(u1)
    ops.count('sumcheck_rounds', 1)
    ops.count('field_muls_sumcheck', 2 * len(mult_layer))

    print(f"    1-round sumcheck: s = {s_vals_12}, alpha = {alpha_12}")
    print(f"    New claim: V_2({u2}) = {V2_claim}")

    # ---- Layer 2 (mult) -> Input via sumcheck ----
    # MLE(mult_layer) != W_mle * x_mle at random points (product of MLEs ≠ MLE of products).
    # Standard GKR uses a sumcheck: V_mult(u) = sum_{b in {0,1}^k} eq(u, b) * W_mle(b) * x_ext_mle(b)
    # where x_ext(i,j) = x(j), so x_ext_mle depends only on column variables.

    print(f"\n    --- Layer 2 (mult) -> Input (sumcheck over wiring) ---")
    W_flat = mat_flat(W, M, N, pM, pN)
    # x_ext[i*pN + j] = x[j] — x replicated per row
    x_ext = [x_pad[j] for i in range(pM) for j in range(pN)]

    num_mult_vars = max(1, math.ceil(math.log2(len(mult_layer))))  # 4

    # Verify claim: sum_b eq(u2, b) * W_flat[b] * x_ext[b] should = V2_claim
    verify_sum = 0
    for b_idx in range(len(mult_layer)):
        b_bits = bits(b_idx, num_mult_vars)
        eq_val = eq_eval(u2, b_bits)
        verify_sum = fadd(verify_sum, fmul(eq_val, fmul(W_flat[b_idx], x_ext[b_idx])))
    assert verify_sum == V2_claim, f"Mult sumcheck setup: {verify_sum} != {V2_claim}"

    # Run sumcheck over num_mult_vars rounds
    # Polynomial degree per round: eq is deg 1, W_mle is deg 1, x_ext_mle is deg 1 → total deg 3
    mult_sc_challenges = []
    mult_sc_claim = V2_claim
    mult_ok = True

    for j in range(num_mult_vars):
        remaining = num_mult_vars - j - 1
        s_vals_mult = [0] * 4  # degree 3 → 4 evaluations

        for X in range(4):
            total = 0
            for tail in range(1 << remaining):
                full_bits = list(mult_sc_challenges) + [X] + bits(tail, remaining)
                eq_val = 1
                for k in range(num_mult_vars):
                    u_k = u2[k]; b_k = full_bits[k]
                    eq_val = fmul(eq_val, fadd(fmul(b_k, fsub(fmul(2, u_k), 1)), fsub(1, u_k)))
                w_val = mle_eval(W_flat, full_bits)
                x_val = mle_eval(x_ext, full_bits)
                total = fadd(total, fmul(eq_val, fmul(w_val, x_val)))
            s_vals_mult[X] = total

        check = fadd(s_vals_mult[0], s_vals_mult[1])
        if check != mult_sc_claim:
            print(f"    Mult sumcheck round {j}: FAIL s(0)+s(1)={check} != {mult_sc_claim}")
            mult_ok = False
            break

        alpha = frand()
        mult_sc_challenges.append(alpha)
        mult_sc_claim = lagrange_eval(s_vals_mult, alpha)
        ops.count('sumcheck_rounds', 1)
        ops.count('field_muls_sumcheck', 4 * (1 << remaining) * 3)

    if mult_ok:
        # Final: verifier checks eq(u2, r) * W_mle(r) * x_ext_mle(r)
        r_mult = mult_sc_challenges
        eq_final = eq_eval(u2, r_mult)
        W_at_r = mle_eval(W_flat, r_mult)
        # x_ext_mle only depends on column variables (bits 0,1)
        col_pt = r_mult[:2]
        x_at_r = mle_eval(x_pad, col_pt)
        final_product = fmul(eq_final, fmul(W_at_r, x_at_r))
        mult_ok = (final_product == mult_sc_claim)
        print(f"    Mult sumcheck: {num_mult_vars} rounds, degree 3")
        print(f"    Final: eq(u2,r)*W(r)*x(r_col) = {final_product}, claim = {mult_sc_claim}")
        print(f"    Final check: {'PASS' if mult_ok else 'FAIL'}")

    # ---- GKR ZK Analysis ----
    print(f"\n  Step 3: ZK analysis for GKR")
    print(f"    Verifier learns:")
    print(f"      - V_0(g) = {V0_claim} (output layer eval, public)")
    print(f"      - V_1(u1) = {V1_claim} (intermediate layer eval, leaks info)")
    print(f"      - V_2(u2) = {V2_claim} (mult layer eval, leaks info)")
    if mult_ok:
        print(f"      - W_mle(r) = {W_at_r} (weight eval, leaks info)")
        print(f"      - x_mle(r_col) = {x_at_r} (input eval, leaks info)")
    print(f"    Without ZK masking: multiple scalar evaluations are leaked.")
    print(f"    Libra's R_i masking would hide the intermediate V_1 and V_2 claims.")
    print(f"    But R_i needs TWO evaluation points per layer (u and v),")
    print(f"    which arise from GKR's wiring predicate structure.")
    print(f"    Our simplified GKR (binary tree) produces only ONE point per layer.")
    print(f"    For the full GKR with add/mult wiring predicates, we'd get two points.")

    # ---- Two-point structure ----
    # In standard GKR, the sumcheck polynomial involves both V_{i+1}(x) and V_{i+1}(y)
    # where x and y are the two input wire labels. This naturally produces
    # two claims: V_{i+1}(u) and V_{i+1}(v).
    # In our simplified version, each addition gate takes inputs from consecutive
    # positions, which means the wiring is simpler but we still get a sumcheck
    # over both input positions.
    #
    # For a proper GKR implementation, we'd need:
    # V_i(g) = sum_{x,y} [add_i(g,x,y)*(V_{i+1}(x) + V_{i+1}(y)) + mult_i(g,x,y)*(V_{i+1}(x)*V_{i+1}(y))]
    # After sumcheck, verifier gets claims at two points u,v for V_{i+1}.
    # These are combined via random linear combination.

    print(f"\n  Step 4: Cost analysis")
    total_gates = len(layer_0) + len(add_layer_1) + len(mult_layer)
    print(f"    Total circuit gates: {total_gates}")
    print(f"    Circuit depth: 3 (2 add layers + 1 mult layer)")
    print(f"    Sumcheck rounds: 2 (one per addition layer) + {num_mult_vars} (mult wiring)")
    print(f"    Total sumcheck rounds: {2 + num_mult_vars}")
    print(f"    For comparison, direct sumcheck: {max(1, math.ceil(math.log2(N)))} rounds")
    ops.count('total_gates', total_gates)

    ops.report()
    return mult_ok


# ######################################################################
#
# EXPERIMENT 2: R1CS + Spartan-style two sumchecks
#
# ######################################################################

def experiment_2():
    """
    Encode 3x3 matmul as R1CS and prove via Spartan's two-sumcheck protocol.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: R1CS + Spartan-style two sumchecks")
    print("=" * 70)

    # Setup
    W = [[2, 3, 1], [1, 0, 2], [4, 1, 3]]
    x_vec = [1, 2, 3]
    y_vec = matvec(W, x_vec, 3, 3)
    M, N = 3, 3
    pM, pN = 4, 4

    ops = OpCounter()

    # ---- Step 1: Build R1CS ----
    # R1CS: (A * z) ∘ (B * z) = C * z
    # z = (1, x_1, x_2, x_3, y_1, y_2, y_3, p_00, p_01, ..., p_22, ...)
    #
    # Variables:
    #   z[0] = 1 (constant)
    #   z[1..3] = x[0..2] (input)
    #   z[4..6] = y[0..2] (output, public)
    #   z[7..15] = p[i][j] = W[i][j] * x[j] (intermediate products)
    #
    # Constraints:
    #   Type 1: p[i][j] = W[i][j] * x[j]  (multiplication constraints)
    #     A row: W[i][j] at z[0] position (constant)... actually W[i][j] is a constant
    #     So: A selects W[i][j] * z[0] = W[i][j], B selects z[1+j] = x[j], C selects z[7+i*3+j] = p[i][j]
    #
    #   Type 2: y[i] = sum_j p[i][j]  (addition constraints)
    #     Can be expressed as: 1 * (sum_j p[i][j]) = y[i]
    #     A: z[0] = 1, B: sum_j z[7+i*3+j], C: z[4+i]
    #     But R1CS is A*B=C (Hadamard of matrix-vector products)
    #     So: A row selects 1, B row selects sum_j p[i][j], C row selects y[i]

    print("\n  Step 1: Building R1CS for 3x3 matmul")

    # Number of variables (padded to power of 2)
    # z = [1, x0, x1, x2, y0, y1, y2, p00, p01, p02, p10, p11, p12, p20, p21, p22]
    n_vars = 16  # padded
    z = [0] * n_vars
    z[0] = 1
    z[1], z[2], z[3] = x_vec[0], x_vec[1], x_vec[2]
    z[4], z[5], z[6] = y_vec[0], y_vec[1], y_vec[2]
    for i in range(M):
        for j in range(N):
            z[7 + i * N + j] = fmul(W[i][j], x_vec[j])

    # Number of constraints (padded to power of 2)
    # 9 mult constraints + 3 sum constraints = 12, pad to 16
    n_constraints = 16

    # Build A, B, C as sparse matrices (list of dicts)
    A = [{} for _ in range(n_constraints)]
    B = [{} for _ in range(n_constraints)]
    C = [{} for _ in range(n_constraints)]

    # Type 1: p[i][j] = W[i][j] * x[j]
    # A * z should give W[i][j], B * z should give x[j], C * z should give p[i][j]
    for i in range(M):
        for j in range(N):
            row = i * N + j
            A[row][0] = W[i][j]  # A selects constant W[i][j]
            B[row][1 + j] = 1     # B selects x[j]
            C[row][7 + i * N + j] = 1  # C selects p[i][j]

    # Type 2: y[i] = sum_j p[i][j]
    # A * z = 1, B * z = sum_j p[i][j], C * z = y[i]
    for i in range(M):
        row = M * N + i
        A[row][0] = 1
        for j in range(N):
            B[row][7 + i * N + j] = 1
        C[row][4 + i] = 1

    # Verify R1CS
    def sparse_mv(mat, vec):
        result = []
        for row in mat:
            val = 0
            for col, coeff in row.items():
                val = fadd(val, fmul(coeff, vec[col]))
            result.append(val)
        return result

    Az = sparse_mv(A, z)
    Bz = sparse_mv(B, z)
    Cz = sparse_mv(C, z)

    for i in range(n_constraints):
        if fmul(Az[i], Bz[i]) != Cz[i]:
            print(f"    R1CS constraint {i} FAILED: {Az[i]} * {Bz[i]} = {fmul(Az[i], Bz[i])} != {Cz[i]}")
            return False

    print(f"    R1CS: {n_constraints} constraints, {n_vars} variables")
    print(f"    z = {z}")
    print(f"    All constraints satisfied: OK")

    # ---- Step 2: Spartan Sumcheck 1 ----
    # Claim: sum_{x in {0,1}^s} eq(tau, x) * (A_tilde(x) * B_tilde(x) - C_tilde(x)) = 0
    # where A_tilde(x) = sum_y A_mle(x, y) * z_mle(y), etc.

    print("\n  Step 2: Spartan Sumcheck 1 (satisfiability)")

    s = max(1, math.ceil(math.log2(n_constraints)))  # log2(16) = 4
    n = max(1, math.ceil(math.log2(n_vars)))          # log2(16) = 4

    # Build dense A, B, C matrices for MLE
    A_dense = [[0] * n_vars for _ in range(n_constraints)]
    B_dense = [[0] * n_vars for _ in range(n_constraints)]
    C_dense = [[0] * n_vars for _ in range(n_constraints)]
    for i in range(n_constraints):
        for col, val in A[i].items():
            A_dense[i][col] = val
        for col, val in B[i].items():
            B_dense[i][col] = val
        for col, val in C[i].items():
            C_dense[i][col] = val

    # Flatten for MLE (row-major)
    A_flat = [A_dense[i][j] for i in range(n_constraints) for j in range(n_vars)]
    B_flat = [B_dense[i][j] for i in range(n_constraints) for j in range(n_vars)]
    C_flat = [C_dense[i][j] for i in range(n_constraints) for j in range(n_vars)]
    z_flat = list(z)

    # Verifier picks tau
    tau = [frand() for _ in range(s)]

    # Precompute A_tilde(x), B_tilde(x), C_tilde(x) for all x in {0,1}^s
    # A_tilde(x) = sum_{y in {0,1}^n} A_mle(x, y) * z_mle(y)
    def compute_tilde(mat_flat, z_vec, x_bits):
        """Compute sum_y mat_mle(x, y) * z_mle(y) = mat[x_int] dot z"""
        x_int = sum(b << i for i, b in enumerate(x_bits))
        total = 0
        for j in range(n_vars):
            total = fadd(total, fmul(mat_flat[x_int * n_vars + j], z_vec[j]))
        return total

    # Run sumcheck 1: sum_{x in {0,1}^s} eq(tau, x) * (A_t(x) * B_t(x) - C_t(x)) = 0
    current_claim = 0  # the sum should be 0

    # Verify the sum is indeed 0
    total_check = 0
    for x_idx in range(1 << s):
        x_bits = bits(x_idx, s)
        eq_val = eq_eval(tau, x_bits)
        a_t = compute_tilde(A_flat, z_flat, x_bits)
        b_t = compute_tilde(B_flat, z_flat, x_bits)
        c_t = compute_tilde(C_flat, z_flat, x_bits)
        term = fmul(eq_val, fsub(fmul(a_t, b_t), c_t))
        total_check = fadd(total_check, term)
    assert total_check == 0, f"R1CS check sum = {total_check}, expected 0"
    print(f"    Initial sum = 0: OK")

    # Run the actual sumcheck
    # The polynomial has degree 3 in each variable (eq is degree 1, A_t*B_t is degree 2 -> total 3)
    deg_sc1 = 3
    n_eval_sc1 = deg_sc1 + 1  # 4 evaluations per round

    challenges_sc1 = []
    # For the sumcheck, we partially bind variables one at a time
    # We need to track the partial binding of tau's eq polynomial and the A_t, B_t, C_t

    # Precompute eq(tau, .) table
    eq_table = [0] * (1 << s)
    for i in range(1 << s):
        eq_table[i] = eq_eval(tau, bits(i, s))

    # Precompute A_t, B_t, C_t tables
    At_table = [compute_tilde(A_flat, z_flat, bits(i, s)) for i in range(1 << s)]
    Bt_table = [compute_tilde(B_flat, z_flat, bits(i, s)) for i in range(1 << s)]
    Ct_table = [compute_tilde(C_flat, z_flat, bits(i, s)) for i in range(1 << s)]

    current_claim_sc1 = 0

    print(f"    Sumcheck 1: {s} rounds, degree {deg_sc1}, {n_eval_sc1} evals/round")

    for j in range(s):
        remaining = s - j - 1
        s_vals = [0] * n_eval_sc1

        for X in range(n_eval_sc1):
            # Brute force: for each X, sum over remaining variables
            total = 0
            for tail in range(1 << remaining):
                # Build full assignment: challenges so far, X, tail bits
                full_bits = list(challenges_sc1) + [X] + bits(tail, remaining)
                # Evaluate eq(tau, full_bits)
                eq_val = 1
                for k in range(s):
                    t_k = tau[k]
                    b_k = full_bits[k]
                    # eq term: t_k * b_k + (1 - t_k) * (1 - b_k)
                    # For non-Boolean b_k, this is: b_k * (2*t_k - 1) + (1 - t_k)
                    eq_val = fmul(eq_val, fadd(fmul(b_k, fsub(fmul(2, t_k), 1)), fsub(1, t_k)))
                    ops.count('field_muls_sc1', 4)

                # Evaluate A_t(x), B_t(x), C_t(x) via MLE:
                # A_t(x) = sum_y A_mle(x, y) * z(y)
                # MLE layout: A_flat[row*n_vars + col], bits 0..n-1 = col, bits n..n+s-1 = row.
                # Point ordering: (col_vars, row_vars) = (y_bits, x_bits).
                a_t_val = 0
                b_t_val = 0
                c_t_val = 0
                for y_idx in range(n_vars):
                    y_bits = bits(y_idx, n)
                    combined = list(y_bits) + list(full_bits)  # col first, row second
                    a_mle_val = mle_eval(A_flat, combined)
                    b_mle_val = mle_eval(B_flat, combined)
                    c_mle_val = mle_eval(C_flat, combined)
                    z_y = z_flat[y_idx]
                    a_t_val = fadd(a_t_val, fmul(a_mle_val, z_y))
                    b_t_val = fadd(b_t_val, fmul(b_mle_val, z_y))
                    c_t_val = fadd(c_t_val, fmul(c_mle_val, z_y))
                    ops.count('field_muls_sc1', 6)

                term = fmul(eq_val, fsub(fmul(a_t_val, b_t_val), c_t_val))
                total = fadd(total, term)
                ops.count('field_muls_sc1', 3)

            s_vals[X] = total

        check = fadd(s_vals[0], s_vals[1])
        if check != current_claim_sc1:
            print(f"    Round {j}: FAIL s(0)+s(1) = {check} != {current_claim_sc1}")
            return False

        alpha = frand()
        challenges_sc1.append(alpha)
        current_claim_sc1 = lagrange_eval(s_vals, alpha)

        print(f"    Round {j}: s = {s_vals}, alpha = {alpha}")
        ops.count('sumcheck_rounds_sc1', 1)

    # After sumcheck 1: verifier has point r_x = challenges_sc1
    r_x = challenges_sc1
    print(f"\n    After Sumcheck 1: r_x = {r_x}")
    print(f"    Verifier needs A_t(r_x), B_t(r_x), C_t(r_x)")

    # Compute A_t(r_x), B_t(r_x), C_t(r_x) the prover would send
    a_t_rx = 0
    b_t_rx = 0
    c_t_rx = 0
    for y_idx in range(n_vars):
        y_bits = bits(y_idx, n)
        combined = list(y_bits) + list(r_x)  # col first, row second
        a_t_rx = fadd(a_t_rx, fmul(mle_eval(A_flat, combined), z_flat[y_idx]))
        b_t_rx = fadd(b_t_rx, fmul(mle_eval(B_flat, combined), z_flat[y_idx]))
        c_t_rx = fadd(c_t_rx, fmul(mle_eval(C_flat, combined), z_flat[y_idx]))
    print(f"    A_t(r_x) = {a_t_rx}, B_t(r_x) = {b_t_rx}, C_t(r_x) = {c_t_rx}")

    # Oracle check: verify prover's tilde claims against sumcheck 1's final claim
    eq_tau_rx = eq_eval(tau, r_x)
    oracle_check = fmul(eq_tau_rx, fsub(fmul(a_t_rx, b_t_rx), c_t_rx))
    assert oracle_check == current_claim_sc1, \
        f"Oracle check FAILED: eq(tau,r_x)*(A_t*B_t - C_t) = {oracle_check} != {current_claim_sc1}"
    print(f"    Oracle check: eq(tau,r_x)*(A_t*B_t - C_t) = {oracle_check} == {current_claim_sc1}: OK")

    # ---- Step 3: Spartan Sumcheck 2 ----
    # Now verify A_t(r_x) = sum_y A_mle(r_x, y) * z_mle(y), and same for B, C.
    # Batch all three via random linear combination.

    print(f"\n  Step 3: Spartan Sumcheck 2 (batched inner products)")

    tau2_1 = frand()
    tau2_2 = frand()
    tau2_3 = frand()

    # Batched claim: tau2_1 * A_t(r_x) + tau2_2 * B_t(r_x) + tau2_3 * C_t(r_x)
    batched_claim = fadd(fadd(fmul(tau2_1, a_t_rx), fmul(tau2_2, b_t_rx)), fmul(tau2_3, c_t_rx))

    # Sumcheck on: sum_y [tau2_1 * A_mle(r_x, y) + tau2_2 * B_mle(r_x, y) + tau2_3 * C_mle(r_x, y)] * z_mle(y)
    # Degree 2 per round (product of two linear functions)
    deg_sc2 = 2
    n_eval_sc2 = deg_sc2 + 1  # 3

    challenges_sc2 = []
    current_claim_sc2 = batched_claim

    print(f"    Sumcheck 2: {n} rounds, degree {deg_sc2}, {n_eval_sc2} evals/round")
    print(f"    Batched claim = {batched_claim}")

    for j in range(n):
        remaining = n - j - 1
        s_vals = [0] * n_eval_sc2

        for X in range(n_eval_sc2):
            total = 0
            for tail in range(1 << remaining):
                y_full = list(challenges_sc2) + [X] + bits(tail, remaining)
                combined = list(y_full) + list(r_x)  # col first, row second

                a_val = mle_eval(A_flat, combined)
                b_val = mle_eval(B_flat, combined)
                c_val = mle_eval(C_flat, combined)

                mat_combined = fadd(fadd(fmul(tau2_1, a_val), fmul(tau2_2, b_val)), fmul(tau2_3, c_val))

                z_val = mle_eval(z_flat, y_full)
                total = fadd(total, fmul(mat_combined, z_val))
                ops.count('field_muls_sc2', 8)

            s_vals[X] = total

        check = fadd(s_vals[0], s_vals[1])
        if check != current_claim_sc2:
            print(f"    Round {j}: FAIL s(0)+s(1) = {check} != {current_claim_sc2}")
            return False

        alpha = frand()
        challenges_sc2.append(alpha)
        current_claim_sc2 = lagrange_eval(s_vals, alpha)

        print(f"    Round {j}: s = {s_vals}, alpha = {alpha}")
        ops.count('sumcheck_rounds_sc2', 1)

    # After sumcheck 2: verifier has r_y = challenges_sc2
    r_y = challenges_sc2

    # Verifier needs: A_mle(r_x, r_y), B_mle(r_x, r_y), C_mle(r_x, r_y), z_mle(r_y)
    combined_final = list(r_y) + list(r_x)  # col first, row second
    a_final = mle_eval(A_flat, combined_final)
    b_final = mle_eval(B_flat, combined_final)
    c_final = mle_eval(C_flat, combined_final)
    z_final = mle_eval(z_flat, r_y)

    # Verifier checks: (tau2_1*a + tau2_2*b + tau2_3*c) * z = current_claim_sc2
    mat_final = fadd(fadd(fmul(tau2_1, a_final), fmul(tau2_2, b_final)), fmul(tau2_3, c_final))
    expected = fmul(mat_final, z_final)

    print(f"\n    After Sumcheck 2: r_y = {r_y}")
    print(f"    A_mle(r_x,r_y) = {a_final}, B_mle = {b_final}, C_mle = {c_final}")
    print(f"    z_mle(r_y) = {z_final}")
    print(f"    Check: {expected} == {current_claim_sc2}: {'PASS' if expected == current_claim_sc2 else 'FAIL'}")

    final_ok = (expected == current_claim_sc2)

    # ---- ZK Analysis ----
    print(f"\n  Step 4: ZK analysis for Spartan")
    print(f"    Verifier learns:")
    print(f"      - A_mle(r_x, r_y) = {a_final} (public matrix, no privacy concern)")
    print(f"      - B_mle(r_x, r_y) = {b_final} (public matrix, no privacy concern)")
    print(f"      - C_mle(r_x, r_y) = {c_final} (public matrix, no privacy concern)")
    print(f"      - z_mle(r_y) = {z_final} (PRIVATE: contains witness)")
    print(f"    Only ONE private scalar needs PCS opening: z_mle(r_y)")
    print(f"    This evaluation is hidden by the PCS commitment opening.")
    print(f"    Additionally, the verifier sees:")
    print(f"      - A_t(r_x), B_t(r_x), C_t(r_x) (inner products of public matrices with private z)")
    print(f"      - Sumcheck round polynomials (encode information about z)")
    print(f"    Full Spartan masks these via sumcheck masking (g + rho*p) on both sumchecks.")
    print(f"    A, B, C are public (circuit structure), so their final MLE evals leak nothing.")

    print(f"\n  Step 5: Cross-layer analysis")
    print(f"    In R1CS, ALL layers are encoded in ONE constraint system.")
    print(f"    The intermediate values (products, partial sums) are in z")
    print(f"    but only z_mle(r_y) is ever evaluated — a SINGLE random evaluation")
    print(f"    of the entire witness vector, not individual layer outputs.")
    print(f"    There is NO cross-layer junction. The problem doesn't arise.")

    print(f"\n  Step 6: Cost analysis")
    print(f"    R1CS constraints: {n_constraints} (for {M}x{N} matmul)")
    print(f"    R1CS variables: {n_vars}")
    print(f"    Sumcheck 1: {s} rounds, degree {deg_sc1}")
    print(f"    Sumcheck 2: {n} rounds, degree {deg_sc2}")
    print(f"    Total sumcheck rounds: {s + n}")
    print(f"    For comparison:")
    print(f"      Direct sumcheck: {max(1, math.ceil(math.log2(N)))} rounds")
    print(f"      GKR: ~{1 + math.ceil(math.log2(pN))} rounds")
    ops.count('total_sumcheck_rounds', s + n)
    ops.count('r1cs_constraints', n_constraints)
    ops.count('r1cs_variables', n_vars)

    ops.report()
    return final_ok


# ######################################################################
#
# SCALING EXPERIMENTS
#
# ######################################################################

# ── Parameterized R1CS builder ────────────────────────────────────────

def build_r1cs_matmul(W_mat, x_vec, M, N):
    """
    Build R1CS for y = W·x.

    Constraints:
      - M*N multiplication constraints: p[i][j] = W[i][j] * x[j]
      - M addition constraints: y[i] = sum_j p[i][j]

    Variables in z:
      z[0] = 1
      z[1..N] = x[0..N-1]
      z[N+1..N+M] = y[0..M-1]
      z[N+M+1..N+M+M*N] = p[i][j] flattened

    Returns (A, B, C, z, n_constraints, n_vars) with all sizes padded to powers of 2.
    """
    raw_constraints = M * N + M
    raw_vars = 1 + N + M + M * N

    n_constraints = 1 << math.ceil(math.log2(max(raw_constraints, 2)))
    n_vars = 1 << math.ceil(math.log2(max(raw_vars, 2)))

    # Compute y and build z
    y_vec = [sum(W_mat[i][j] * x_vec[j] for j in range(N)) % P for i in range(M)]

    z = [0] * n_vars
    z[0] = 1
    for j in range(N):
        z[1 + j] = x_vec[j]
    for i in range(M):
        z[1 + N + i] = y_vec[i]
    for i in range(M):
        for j in range(N):
            z[1 + N + M + i * N + j] = fmul(W_mat[i][j], x_vec[j])

    # Build sparse A, B, C
    A = [{} for _ in range(n_constraints)]
    B = [{} for _ in range(n_constraints)]
    C = [{} for _ in range(n_constraints)]

    # Multiplication constraints: p[i][j] = W[i][j] * x[j]
    for i in range(M):
        for j in range(N):
            row = i * N + j
            A[row][0] = W_mat[i][j]         # constant W[i][j]
            B[row][1 + j] = 1               # x[j]
            C[row][1 + N + M + i * N + j] = 1  # p[i][j]

    # Addition constraints: y[i] = sum_j p[i][j]
    for i in range(M):
        row = M * N + i
        A[row][0] = 1
        for j in range(N):
            B[row][1 + N + M + i * N + j] = 1
        C[row][1 + N + i] = 1

    # Verify
    def sparse_mv(mat, vec):
        result = []
        for r in mat:
            val = 0
            for col, coeff in r.items():
                val = fadd(val, fmul(coeff, vec[col]))
            result.append(val)
        return result

    Az = sparse_mv(A, z)
    Bz = sparse_mv(B, z)
    Cz = sparse_mv(C, z)
    for i in range(n_constraints):
        assert fmul(Az[i], Bz[i]) == Cz[i], f"R1CS constraint {i} failed"

    return A, B, C, z, n_constraints, n_vars, y_vec


def run_spartan(A_sparse, B_sparse, C_sparse, z, n_constraints, n_vars, verbose=False):
    """
    Run the full Spartan two-sumcheck protocol.
    Returns (ok, stats) where stats is a dict of metrics.
    """
    s = max(1, math.ceil(math.log2(n_constraints)))
    n = max(1, math.ceil(math.log2(n_vars)))

    ops = OpCounter()

    # Build dense flats for MLE
    A_flat = [0] * (n_constraints * n_vars)
    B_flat = [0] * (n_constraints * n_vars)
    C_flat = [0] * (n_constraints * n_vars)
    for i in range(n_constraints):
        for col, val in A_sparse[i].items():
            A_flat[i * n_vars + col] = val
        for col, val in B_sparse[i].items():
            B_flat[i * n_vars + col] = val
        for col, val in C_sparse[i].items():
            C_flat[i * n_vars + col] = val
    z_flat = list(z)

    # ── Sumcheck 1 ──
    tau = [frand() for _ in range(s)]

    # Verify initial sum = 0
    def compute_tilde(mat_flat, z_vec, x_bits):
        x_int = sum(b << i for i, b in enumerate(x_bits))
        total = 0
        for j in range(n_vars):
            total = fadd(total, fmul(mat_flat[x_int * n_vars + j], z_vec[j]))
        return total

    total_check = 0
    for x_idx in range(1 << s):
        x_bits = bits(x_idx, s)
        eq_val = eq_eval(tau, x_bits)
        a_t = compute_tilde(A_flat, z_flat, x_bits)
        b_t = compute_tilde(B_flat, z_flat, x_bits)
        c_t = compute_tilde(C_flat, z_flat, x_bits)
        total_check = fadd(total_check, fmul(eq_val, fsub(fmul(a_t, b_t), c_t)))
    assert total_check == 0, f"Initial sum = {total_check}, expected 0"

    deg_sc1 = 3
    n_eval_sc1 = deg_sc1 + 1
    challenges_sc1 = []
    current_claim_sc1 = 0

    for j in range(s):
        remaining = s - j - 1
        s_vals = [0] * n_eval_sc1

        for X in range(n_eval_sc1):
            total = 0
            for tail in range(1 << remaining):
                full_bits = list(challenges_sc1) + [X] + bits(tail, remaining)
                # eq(tau, full_bits)
                eq_val = 1
                for k in range(s):
                    t_k = tau[k]; b_k = full_bits[k]
                    eq_val = fmul(eq_val, fadd(fmul(b_k, fsub(fmul(2, t_k), 1)), fsub(1, t_k)))

                # A_t, B_t, C_t via MLE
                a_t_val, b_t_val, c_t_val = 0, 0, 0
                for y_idx in range(n_vars):
                    y_bits = bits(y_idx, n)
                    combined = list(y_bits) + list(full_bits)
                    a_mle_val = mle_eval(A_flat, combined)
                    b_mle_val = mle_eval(B_flat, combined)
                    c_mle_val = mle_eval(C_flat, combined)
                    z_y = z_flat[y_idx]
                    a_t_val = fadd(a_t_val, fmul(a_mle_val, z_y))
                    b_t_val = fadd(b_t_val, fmul(b_mle_val, z_y))
                    c_t_val = fadd(c_t_val, fmul(c_mle_val, z_y))
                    ops.count('field_muls_sc1', 6)

                total = fadd(total, fmul(eq_val, fsub(fmul(a_t_val, b_t_val), c_t_val)))
                ops.count('field_muls_sc1', 3)

            s_vals[X] = total

        check = fadd(s_vals[0], s_vals[1])
        if check != current_claim_sc1:
            if verbose:
                print(f"    SC1 Round {j}: FAIL s(0)+s(1) = {check} != {current_claim_sc1}")
            return False, {}

        alpha = frand()
        challenges_sc1.append(alpha)
        current_claim_sc1 = lagrange_eval(s_vals, alpha)
        ops.count('sumcheck_rounds_sc1', 1)

    r_x = challenges_sc1

    # Oracle check
    a_t_rx, b_t_rx, c_t_rx = 0, 0, 0
    for y_idx in range(n_vars):
        y_bits = bits(y_idx, n)
        combined = list(y_bits) + list(r_x)
        a_t_rx = fadd(a_t_rx, fmul(mle_eval(A_flat, combined), z_flat[y_idx]))
        b_t_rx = fadd(b_t_rx, fmul(mle_eval(B_flat, combined), z_flat[y_idx]))
        c_t_rx = fadd(c_t_rx, fmul(mle_eval(C_flat, combined), z_flat[y_idx]))

    eq_tau_rx = eq_eval(tau, r_x)
    oracle_check = fmul(eq_tau_rx, fsub(fmul(a_t_rx, b_t_rx), c_t_rx))
    assert oracle_check == current_claim_sc1, "Oracle check failed"

    # ── Sumcheck 2 ──
    tau2_1, tau2_2, tau2_3 = frand(), frand(), frand()
    batched_claim = fadd(fadd(fmul(tau2_1, a_t_rx), fmul(tau2_2, b_t_rx)), fmul(tau2_3, c_t_rx))

    deg_sc2 = 2
    n_eval_sc2 = deg_sc2 + 1
    challenges_sc2 = []
    current_claim_sc2 = batched_claim

    for j in range(n):
        remaining = n - j - 1
        s_vals = [0] * n_eval_sc2

        for X in range(n_eval_sc2):
            total = 0
            for tail in range(1 << remaining):
                y_full = list(challenges_sc2) + [X] + bits(tail, remaining)
                combined = list(y_full) + list(r_x)

                a_val = mle_eval(A_flat, combined)
                b_val = mle_eval(B_flat, combined)
                c_val = mle_eval(C_flat, combined)

                mat_combined = fadd(fadd(fmul(tau2_1, a_val), fmul(tau2_2, b_val)), fmul(tau2_3, c_val))
                z_val = mle_eval(z_flat, y_full)
                total = fadd(total, fmul(mat_combined, z_val))
                ops.count('field_muls_sc2', 8)

            s_vals[X] = total

        check = fadd(s_vals[0], s_vals[1])
        if check != current_claim_sc2:
            if verbose:
                print(f"    SC2 Round {j}: FAIL s(0)+s(1) = {check} != {current_claim_sc2}")
            return False, {}

        alpha = frand()
        challenges_sc2.append(alpha)
        current_claim_sc2 = lagrange_eval(s_vals, alpha)
        ops.count('sumcheck_rounds_sc2', 1)

    r_y = challenges_sc2

    # Final check
    combined_final = list(r_y) + list(r_x)
    a_final = mle_eval(A_flat, combined_final)
    b_final = mle_eval(B_flat, combined_final)
    c_final = mle_eval(C_flat, combined_final)
    z_final = mle_eval(z_flat, r_y)

    mat_final = fadd(fadd(fmul(tau2_1, a_final), fmul(tau2_2, b_final)), fmul(tau2_3, c_final))
    expected = fmul(mat_final, z_final)
    ok = (expected == current_claim_sc2)

    stats = {
        'n_constraints': n_constraints,
        'n_vars': n_vars,
        'sc1_rounds': s,
        'sc2_rounds': n,
        'total_rounds': s + n,
        'sc1_degree': deg_sc1,
        'sc2_degree': deg_sc2,
        'field_muls_sc1': ops.counts.get('field_muls_sc1', 0),
        'field_muls_sc2': ops.counts.get('field_muls_sc2', 0),
    }
    return ok, stats


# ── Size scaling experiment ───────────────────────────────────────────

def experiment_size_scaling():
    """
    Run Spartan for increasing matrix sizes and measure how costs scale.
    """
    print("\n" + "=" * 70)
    print("SCALING EXPERIMENT: Spartan vs. matrix size")
    print("=" * 70)

    # Note: brute-force MLE evaluation makes large sizes impractical.
    # 8×8 takes ~9 min, 16×16 would take hours. Use formulaic projections instead.
    sizes = [(3, 3), (4, 4), (6, 6), (8, 8)]
    results = []

    for M, N in sizes:
        random.seed(42)
        # Random M×N weight matrix and N-vector input
        W_mat = [[random.randint(0, P - 1) for _ in range(N)] for _ in range(M)]
        x_vec = [random.randint(1, P - 1) for _ in range(N)]

        import time
        t0 = time.time()
        A, B, C, z, nc, nv, y = build_r1cs_matmul(W_mat, x_vec, M, N)
        ok, stats = run_spartan(A, B, C, z, nc, nv)
        elapsed = time.time() - t0

        raw_constraints = M * N + M
        raw_vars = 1 + N + M + M * N
        stats['M'] = M
        stats['N'] = N
        stats['raw_constraints'] = raw_constraints
        stats['raw_vars'] = raw_vars
        stats['elapsed_s'] = elapsed
        results.append(stats)

        status = "PASS" if ok else "FAIL"
        print(f"\n  {M}x{N}: {status}  ({elapsed:.2f}s)")
        print(f"    Raw constraints: {raw_constraints}, padded: {stats['n_constraints']}")
        print(f"    Raw variables:   {raw_vars}, padded: {stats['n_vars']}")
        print(f"    SC1: {stats['sc1_rounds']} rounds (deg {stats['sc1_degree']})")
        print(f"    SC2: {stats['sc2_rounds']} rounds (deg {stats['sc2_degree']})")
        print(f"    Total rounds: {stats['total_rounds']}")
        print(f"    Field muls SC1: {stats['field_muls_sc1']:,}")
        print(f"    Field muls SC2: {stats['field_muls_sc2']:,}")

        if not ok:
            print(f"    *** VERIFICATION FAILED ***")

    # ── Projection table ──
    print(f"\n  {'─' * 66}")
    print(f"  Projection to larger sizes (formulaic, not measured):")
    print(f"  {'─' * 66}")
    print(f"  {'Size':>10} {'Raw C':>8} {'Pad C':>8} {'Raw V':>8} {'Pad V':>8} {'SC1 rds':>8} {'SC2 rds':>8} {'Total':>6}")

    for M, N in [(64, 64), (128, 128), (256, 256), (512, 512),
                  (1024, 1024), (4096, 4096)]:
        raw_c = M * N + M
        raw_v = 1 + N + M + M * N
        pad_c = 1 << math.ceil(math.log2(max(raw_c, 2)))
        pad_v = 1 << math.ceil(math.log2(max(raw_v, 2)))
        sc1 = math.ceil(math.log2(pad_c))
        sc2 = math.ceil(math.log2(pad_v))
        print(f"  {M}x{N:>4}  {raw_c:>8,} {pad_c:>8,} {raw_v:>8,} {pad_v:>8,} {sc1:>8} {sc2:>8} {sc1+sc2:>6}")

    # ── Scaling relationships ──
    print(f"\n  Key scaling relationships for M×M matmul:")
    print(f"    Constraints ≈ M²    (exactly M² + M)")
    print(f"    Variables   ≈ M²    (exactly M² + 2M + 1)")
    print(f"    SC1 rounds  = ceil(log2(pad(M²+M)))  ≈ 2·log2(M)")
    print(f"    SC2 rounds  = ceil(log2(pad(M²+2M+1))) ≈ 2·log2(M)")
    print(f"    Total rounds ≈ 4·log2(M)")
    print(f"    For comparison, direct sumcheck = log2(M) rounds")
    print(f"    Overhead factor: ~4× in rounds")

    return results


# ── Multi-layer R1CS builder ─────────────────────────────────────────

def build_r1cs_multilayer(layers, x_vec):
    """
    Build R1CS for a chain of matmuls: h1 = W1·x, h2 = W2·h1, ..., y = Wk·h_{k-1}.

    layers: list of (W_mat, M, N) tuples, applied in order.
    x_vec: input vector.

    Variables in z:
      z[0] = 1
      z[1..N0] = x (input)
      z[...] = h1 (output of layer 1)
      z[...] = p1[i][j] (products for layer 1)
      z[...] = h2 (output of layer 2)
      z[...] = p2[i][j] (products for layer 2)
      ... etc.
    """
    # ── Layout planning ──
    # Offset 0: constant
    # Then for each layer: products first, then output
    offset = 1
    input_offset = 1  # x starts at z[1]
    input_size = len(x_vec)

    # Plan variable layout
    layer_info = []
    total_constraints = 0

    # Input
    var_offsets = {'input': 1}
    offset = 1 + input_size  # past x

    for layer_idx, (W_mat, M, N) in enumerate(layers):
        prod_offset = offset
        prod_count = M * N
        out_offset = offset + prod_count
        out_count = M

        layer_info.append({
            'W': W_mat, 'M': M, 'N': N,
            'prod_offset': prod_offset,
            'out_offset': out_offset,
            'input_offset': input_offset if layer_idx == 0 else layer_info[layer_idx - 1]['out_offset'],
        })

        offset = out_offset + out_count
        total_constraints += M * N + M

        # Next layer's input is this layer's output
        input_offset = out_offset
        input_size = M

    raw_vars = offset
    n_vars = 1 << math.ceil(math.log2(max(raw_vars, 2)))
    n_constraints = 1 << math.ceil(math.log2(max(total_constraints, 2)))

    # ── Build z ──
    z = [0] * n_vars
    z[0] = 1

    # Input
    for j in range(len(x_vec)):
        z[1 + j] = x_vec[j]

    # Compute forward pass and fill z
    current_input = list(x_vec)
    for info in layer_info:
        W_mat, M, N = info['W'], info['M'], info['N']
        inp_off = info['input_offset']

        # Products
        for i in range(M):
            for j in range(N):
                val = fmul(W_mat[i][j], z[inp_off + j])
                z[info['prod_offset'] + i * N + j] = val

        # Output
        for i in range(M):
            total = 0
            for j in range(N):
                total = fadd(total, z[info['prod_offset'] + i * N + j])
            z[info['out_offset'] + i] = total

    # ── Build A, B, C ──
    A = [{} for _ in range(n_constraints)]
    B = [{} for _ in range(n_constraints)]
    C = [{} for _ in range(n_constraints)]

    row = 0
    for info in layer_info:
        W_mat, M, N = info['W'], info['M'], info['N']
        inp_off = info['input_offset']

        # Multiplication constraints
        for i in range(M):
            for j in range(N):
                A[row][0] = W_mat[i][j]
                B[row][inp_off + j] = 1
                C[row][info['prod_offset'] + i * N + j] = 1
                row += 1

        # Addition constraints
        for i in range(M):
            A[row][0] = 1
            for j in range(N):
                B[row][info['prod_offset'] + i * N + j] = 1
            C[row][info['out_offset'] + i] = 1
            row += 1

    # Verify
    def sparse_mv(mat, vec):
        result = []
        for r in mat:
            val = 0
            for col, coeff in r.items():
                val = fadd(val, fmul(coeff, vec[col]))
            result.append(val)
        return result

    Az = sparse_mv(A, z)
    Bz = sparse_mv(B, z)
    Cz = sparse_mv(C, z)
    for i in range(n_constraints):
        assert fmul(Az[i], Bz[i]) == Cz[i], f"Multi-layer R1CS constraint {i} failed"

    # Final output
    last = layer_info[-1]
    y_vec = [z[last['out_offset'] + i] for i in range(last['M'])]

    return A, B, C, z, n_constraints, n_vars, y_vec, layer_info


def experiment_multilayer():
    """
    Encode multi-layer matmul chains as single R1CS and run Spartan.
    """
    print("\n" + "=" * 70)
    print("SCALING EXPERIMENT: Multi-layer Spartan")
    print("=" * 70)

    import time

    # Test configurations: (layer_sizes, description)
    # layer_sizes = [(M1,N1), (M2,N2), ...] where each layer is Mi×Ni
    # Note: brute-force MLE is expensive. Keep sizes small for measured data,
    # rely on formulaic projections for larger sizes.
    configs = [
        ([(4, 4)], "1 layer 4x4"),
        ([(4, 4), (4, 4)], "2 layers 4x4"),
        ([(4, 4), (4, 4), (4, 4)], "3 layers 4x4"),
        ([(4, 4), (4, 4), (4, 4), (4, 4)], "4 layers 4x4"),
        ([(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)], "8 layers 3x3"),
    ]

    results = []

    for layer_sizes, desc in configs:
        random.seed(42)

        # Build random weight matrices and input
        N0 = layer_sizes[0][1]
        x_vec = [random.randint(1, P - 1) for _ in range(N0)]

        layers = []
        for M, N in layer_sizes:
            W = [[random.randint(0, P - 1) for _ in range(N)] for _ in range(M)]
            layers.append((W, M, N))

        t0 = time.time()
        A, B, C, z, nc, nv, y, info = build_r1cs_multilayer(layers, x_vec)
        ok, stats = run_spartan(A, B, C, z, nc, nv)
        elapsed = time.time() - t0

        n_layers = len(layer_sizes)
        raw_c = sum(M * N + M for M, N in layer_sizes)
        raw_v = 1 + layer_sizes[0][1] + sum(M * N + M for M, N in layer_sizes)

        stats['desc'] = desc
        stats['n_layers'] = n_layers
        stats['raw_constraints'] = raw_c
        stats['raw_vars'] = raw_v
        stats['elapsed_s'] = elapsed
        results.append(stats)

        status = "PASS" if ok else "FAIL"
        print(f"\n  {desc}: {status}  ({elapsed:.2f}s)")
        print(f"    Raw constraints: {raw_c}, padded: {nc}")
        print(f"    Raw variables:   {raw_v}, padded: {nv}")
        print(f"    SC1: {stats['sc1_rounds']} rounds, SC2: {stats['sc2_rounds']} rounds")
        print(f"    Total rounds: {stats['total_rounds']}")
        print(f"    Field muls SC1: {stats['field_muls_sc1']:,}, SC2: {stats['field_muls_sc2']:,}")

        if not ok:
            print(f"    *** VERIFICATION FAILED ***")

    # ── Multi-layer projection ──
    print(f"\n  {'─' * 66}")
    print(f"  Multi-layer projection (formulaic):")
    print(f"  {'─' * 66}")
    print(f"  {'Config':>30} {'Raw C':>8} {'Pad C':>8} {'Raw V':>8} {'Pad V':>8} {'Rounds':>7}")

    projections = [
        ("2 layers 128x128", [(128, 128)] * 2),
        ("4 layers 128x128", [(128, 128)] * 4),
        ("8 layers 128x128", [(128, 128)] * 8),
        ("12 layers 128x128", [(128, 128)] * 12),
        ("2 layers 4096x4096", [(4096, 4096)] * 2),
        ("4 layers 4096x4096", [(4096, 4096)] * 4),
    ]

    for desc, layer_sizes in projections:
        raw_c = sum(M * N + M for M, N in layer_sizes)
        raw_v = 1 + layer_sizes[0][1] + sum(M * N + M for M, N in layer_sizes)
        pad_c = 1 << math.ceil(math.log2(max(raw_c, 2)))
        pad_v = 1 << math.ceil(math.log2(max(raw_v, 2)))
        sc1 = math.ceil(math.log2(pad_c))
        sc2 = math.ceil(math.log2(pad_v))
        print(f"  {desc:>30} {raw_c:>8,} {pad_c:>8,} {raw_v:>8,} {pad_v:>8,} {sc1+sc2:>7}")

    # ── Analysis ──
    print(f"\n  Key multi-layer observations:")
    print(f"    - Constraints grow LINEARLY with number of layers: C = L·(M²+M)")
    print(f"    - Sumcheck rounds grow LOGARITHMICALLY: ~2·log2(L·M²)")
    print(f"    - Doubling layers adds ~2 sumcheck rounds (from log2(2)=1 per sumcheck)")
    print(f"    - ALL layers share ONE witness z, so only z_mle(r_y) is revealed")
    print(f"    - No cross-layer junction: adding layers does NOT add new private reveals")

    return results


# ######################################################################
# MAIN
# ######################################################################

if __name__ == '__main__':
    random.seed(42)

    ok3 = experiment_3()
    random.seed(42)
    ok1 = experiment_1()
    random.seed(42)
    ok2 = experiment_2()

    print("\n" + "=" * 70)
    print("SUMMARY (original experiments)")
    print("=" * 70)
    print(f"  Experiment 3 (vanishing poly masking): {'PASS' if ok3 else 'FAIL'}")
    print(f"  Experiment 1 (GKR circuit):            {'PASS' if ok1 else 'FAIL'}")
    print(f"  Experiment 2 (R1CS + Spartan):         {'PASS' if ok2 else 'FAIL'}")

    # ── Scaling experiments ──
    size_results = experiment_size_scaling()
    multi_results = experiment_multilayer()
