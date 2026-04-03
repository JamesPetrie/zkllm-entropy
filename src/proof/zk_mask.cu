// zk_mask.cu — Zero-knowledge masking helpers (CPU-side, Goldilocks field)
//
// All functions here run on the host. The masking coefficients are small
// (one per sumcheck variable, typically 7-14 elements) so there's no
// benefit to GPU execution.

#include "proof/zk_mask.cuh"

#ifdef USE_GOLDILOCKS

// ── Vanishing polynomial masking ────────────────────────────────────────────

ZkMaskConfig generate_vanishing_mask(uint num_vars) {
    ZkMaskConfig config;
    config.enabled = true;
    config.vanishing_coeffs = random_vec(num_vars);
    return config;
}

Fr_t vanishing_correction(const std::vector<Fr_t>& coeffs,
                          const std::vector<Fr_t>& point) {
    // sum_{i=0}^{k-1} c_i * point[i] * (1 - point[i])
    Fr_t result = FR_ZERO;
    Fr_t one = FR_ONE;
    for (uint i = 0; i < coeffs.size() && i < point.size(); i++) {
        Fr_t xi = point[i];
        Fr_t one_minus_xi = one - xi;
        Fr_t term = coeffs[i] * xi * one_minus_xi;
        result = result + term;
    }
    return result;
}

// ── Transcript masking (XZZ+19) ─────────────────────────────────────────────

ZkTranscriptMask generate_transcript_mask(uint num_vars, uint degree) {
    ZkTranscriptMask mask;
    mask.degree = degree;

    // Random constant term
    auto a0_vec = random_vec(1);
    mask.a0 = a0_vec[0];

    // Random univariate polynomials p_i(X) = sum_{k=1}^{d} c_{i,k} * X^k
    // (no constant term — the constant is shared via a0)
    mask.p_univariates.resize(num_vars);
    for (uint i = 0; i < num_vars; i++) {
        mask.p_univariates[i] = random_vec(degree);  // coeffs for X^1, X^2, ..., X^d
    }

    // Precompute P_sum = sum_{c in {0,1}^b} p(c)
    //
    // p(X) = a_0 + sum_i p_i(X_i)
    // On the boolean hypercube, each variable X_i is 0 or 1 independently.
    //
    // sum_{c in {0,1}^b} p(c)
    //   = sum_{c} a_0 + sum_{c} sum_i p_i(c_i)
    //   = 2^b * a_0 + sum_i sum_{c} p_i(c_i)
    //
    // For each i, sum_{c} p_i(c_i) = 2^{b-1} * (p_i(0) + p_i(1))
    // since the other b-1 variables each contribute a factor of 2.
    //
    // p_i(0) = 0 (no constant term)
    // p_i(1) = sum_{k=1}^{d} c_{i,k}
    //
    // So: P = 2^b * a_0 + 2^{b-1} * sum_i p_i(1)

    Fr_t one = FR_ONE;
    Fr_t two = one + one;

    // Compute 2^b and 2^{b-1}
    Fr_t pow2_b = one;
    for (uint i = 0; i < num_vars; i++) pow2_b = pow2_b * two;
    Fr_t pow2_bm1 = (num_vars > 0) ? pow2_b / two : one;

    // sum_i p_i(1)
    Fr_t sum_pi_at_1 = FR_ZERO;
    for (uint i = 0; i < num_vars; i++) {
        Fr_t pi_1 = FR_ZERO;
        for (uint k = 0; k < degree; k++) {
            pi_1 = pi_1 + mask.p_univariates[i][k];
        }
        sum_pi_at_1 = sum_pi_at_1 + pi_1;
    }

    mask.P_sum = pow2_b * mask.a0 + pow2_bm1 * sum_pi_at_1;

    return mask;
}

Fr_t eval_transcript_mask(const ZkTranscriptMask& mask,
                          const std::vector<Fr_t>& point) {
    // p(point) = a_0 + sum_{i=0}^{b-1} p_i(point[i])
    // where p_i(X) = sum_{k=1}^{d} coeff[k-1] * X^k
    Fr_t result = mask.a0;
    for (uint i = 0; i < mask.p_univariates.size() && i < point.size(); i++) {
        const auto& coeffs = mask.p_univariates[i];
        Fr_t xi = point[i];
        Fr_t xi_pow = xi;  // X^1
        Fr_t pi_val = FR_ZERO;
        for (uint k = 0; k < coeffs.size(); k++) {
            pi_val = pi_val + coeffs[k] * xi_pow;
            xi_pow = xi_pow * xi;  // X^{k+2}
        }
        result = result + pi_val;
    }
    return result;
}

Polynomial transcript_mask_round_poly(const ZkTranscriptMask& mask,
                                      uint current_var,
                                      const std::vector<uint>& bound_var_indices,
                                      const std::vector<Fr_t>& bound_var_values,
                                      uint total_vars) {
    // Compute S(X_{current_var}) = sum over free variables of p(...)
    //
    // Free variables: those not in bound_var_indices and != current_var.
    // Bound variables: those in bound_var_indices (with known values).
    // Current variable: current_var (left as X).
    //
    // Since p(X) = a_0 + sum_i p_i(X_i) is separable:
    //
    // S(X_{cv}) = 2^R * [a_0 + Σ_{bound i} p_i(alpha_i) + p_{cv}(X_{cv})]
    //           + 2^{R-1} * Σ_{free i} (p_i(0) + p_i(1))
    //
    // where R = number of free variables (not bound and not current_var).

    // Build set of bound variables for quick lookup
    std::vector<bool> is_bound(total_vars, false);
    for (uint idx : bound_var_indices) {
        if (idx < total_vars) is_bound[idx] = true;
    }

    // Count free variables (not bound, not current_var)
    uint R = 0;
    for (uint i = 0; i < total_vars; i++) {
        if (i != current_var && !is_bound[i]) R++;
    }

    Fr_t one = FR_ONE;
    Fr_t two = one + one;

    // Compute 2^R
    Fr_t pow2_R = one;
    for (uint i = 0; i < R; i++) pow2_R = pow2_R * two;

    // Compute 2^{R-1}
    Fr_t pow2_Rm1 = (R > 0) ? pow2_R / two : one;

    // Constant part: 2^R * [a_0 + Σ_{bound i} p_i(alpha_i)]
    Fr_t constant = mask.a0;
    for (uint idx = 0; idx < bound_var_indices.size(); idx++) {
        uint var_i = bound_var_indices[idx];
        if (var_i >= mask.p_univariates.size()) continue;
        const auto& coeffs = mask.p_univariates[var_i];
        Fr_t alpha_i = bound_var_values[idx];
        Fr_t alpha_pow = alpha_i;
        Fr_t pi_alpha = FR_ZERO;
        for (uint k = 0; k < coeffs.size(); k++) {
            pi_alpha = pi_alpha + coeffs[k] * alpha_pow;
            alpha_pow = alpha_pow * alpha_i;
        }
        constant = constant + pi_alpha;
    }
    constant = pow2_R * constant;

    // Tail sum: 2^{R-1} * Σ_{free i} (p_i(0) + p_i(1))
    // p_i(0) = 0, p_i(1) = sum of coefficients
    Fr_t tail_sum = FR_ZERO;
    for (uint i = 0; i < total_vars; i++) {
        if (i == current_var || is_bound[i]) continue;  // skip bound and current
        if (i >= mask.p_univariates.size()) continue;
        const auto& coeffs = mask.p_univariates[i];
        Fr_t pi_1 = FR_ZERO;
        for (uint k = 0; k < coeffs.size(); k++) {
            pi_1 = pi_1 + coeffs[k];
        }
        tail_sum = tail_sum + pi_1;
    }
    tail_sum = pow2_Rm1 * tail_sum;

    // Build univariate polynomial in X_{current_var}:
    //   S(X) = (constant + tail_sum) + 2^R * p_{cv}(X)
    uint d = mask.degree;
    std::vector<Fr_t> poly_coeffs(d + 1);
    poly_coeffs[0] = constant + tail_sum;
    if (current_var < mask.p_univariates.size()) {
        const auto& pj_coeffs = mask.p_univariates[current_var];
        for (uint k = 0; k < d && k < pj_coeffs.size(); k++) {
            poly_coeffs[k + 1] = pow2_R * pj_coeffs[k];
        }
    }

    return Polynomial(poly_coeffs);
}

#endif // USE_GOLDILOCKS
