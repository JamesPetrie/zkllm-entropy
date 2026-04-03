// zk_sumcheck.cu — Zero-knowledge sumcheck prover implementations
//
// Degree-4 masked sumcheck kernels and prover functions.
// See zk_sumcheck.cuh for interface documentation.

#include "proof/zk_sumcheck.cuh"
#include "proof/zk_mask.cuh"

#ifdef USE_GOLDILOCKS

// ── Helper: evaluate Za(t) = a0 + da*t + c_a*t*(1-t) ───────────────────────
// where da = a1 - a0. This is degree 2 in t.
// t is a small integer (0..4), passed as Fr_t.
DEVICE inline Fr_t eval_masked_linear(Fr_t a0, Fr_t da, Fr_t c_a, Fr_t t) {
    // a0 + da*t + c_a * t * (1 - t)
    // = a0 + da*t + c_a*t - c_a*t^2
    Fr_t t2 = blstrs__scalar__Scalar_mul(t, t);
    Fr_t lin = blstrs__scalar__Scalar_add(
        blstrs__scalar__Scalar_mul(da, t),
        blstrs__scalar__Scalar_mul(c_a, t)
    );
    Fr_t quad = blstrs__scalar__Scalar_mul(c_a, t2);
    return blstrs__scalar__Scalar_sub(
        blstrs__scalar__Scalar_add(a0, lin),
        quad
    );
}

// ── Degree-4 masked inner product kernel ────────────────────────────────────
KERNEL void zkip_zk_poly_kernel(
    GLOBAL Fr_t *a, GLOBAL Fr_t *b,
    GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2,
    GLOBAL Fr_t *out3, GLOBAL Fr_t *out4,
    Fr_t c_a, Fr_t c_b,
    uint N_in, uint N_out)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;

    uint gid0 = gid;
    uint gid1 = gid + N_out;
    Fr_t a0 = (gid0 < N_in) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < N_in) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < N_in) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < N_in) ? b[gid1] : blstrs__scalar__Scalar_ZERO;

    Fr_t da = blstrs__scalar__Scalar_sub(a1, a0);
    Fr_t db = blstrs__scalar__Scalar_sub(b1, b0);

    // Evaluate Za(t) * Zb(t) at t = 0, 1, 2, 3, 4
    // t=0: Za = a0, Zb = b0 (vanishing terms are 0)
    out0[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a0, b0));

    // t=1: Za = a0 + da = a1, Zb = b0 + db = b1 (vanishing terms are 0)
    out1[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(a1, b1));

    // t=2: Za = a0 + 2*da + c_a*2*(1-2) = a0 + 2*da - 2*c_a
    //       Zb = b0 + 2*db - 2*c_b
    Fr_t t2 = {2ULL};
    Fr_t za2 = eval_masked_linear(a0, da, c_a, t2);
    Fr_t zb2 = eval_masked_linear(b0, db, c_b, t2);
    out2[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(za2, zb2));

    // t=3: Za = a0 + 3*da + c_a*3*(-2) = a0 + 3*da - 6*c_a
    Fr_t t3 = {3ULL};
    Fr_t za3 = eval_masked_linear(a0, da, c_a, t3);
    Fr_t zb3 = eval_masked_linear(b0, db, c_b, t3);
    out3[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(za3, zb3));

    // t=4: Za = a0 + 4*da + c_a*4*(-3) = a0 + 4*da - 12*c_a
    Fr_t t4 = {4ULL};
    Fr_t za4 = eval_masked_linear(a0, da, c_a, t4);
    Fr_t zb4 = eval_masked_linear(b0, db, c_b, t4);
    out4[gid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(za4, zb4));
}

// ── Single-round step: compute degree-4 round polynomial ────────────────────
static Polynomial zkip_zk_step_poly(
    const FrTensor& a, const FrTensor& b,
    Fr_t c_a, Fr_t c_b)
{
    if (a.size != b.size) throw std::runtime_error("zkip_zk_step_poly: a.size != b.size");
    uint N_in = a.size, N_out = (1 << ceilLog2(a.size)) >> 1;
    FrTensor out0(N_out), out1(N_out), out2(N_out), out3(N_out), out4(N_out);

    zkip_zk_poly_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        a.gpu_data, b.gpu_data,
        out0.gpu_data, out1.gpu_data, out2.gpu_data,
        out3.gpu_data, out4.gpu_data,
        c_a, c_b, N_in, N_out);

    // Sum each output array to get evaluations at t=0,1,2,3,4
    std::vector<Fr_t> evals = {
        out0.sum(), out1.sum(), out2.sum(), out3.sum(), out4.sum()
    };

    return Polynomial::from_evaluations(evals);
}

// ── Reduce kernel (duplicated from zkfc.cu for module independence) ─────────
// Folds a,b using the challenge v: new[i] = old[i] + v*(old[i+N_out] - old[i])
KERNEL void zkip_zk_reduce_kernel(
    GLOBAL Fr_t *a, GLOBAL Fr_t *b,
    GLOBAL Fr_t *new_a, GLOBAL Fr_t *new_b,
    Fr_t v, uint N_in, uint N_out)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;

    uint gid0 = gid;
    uint gid1 = gid + N_out;
    Fr_t v_mont = blstrs__scalar__Scalar_mont(v);
    Fr_t a0 = (gid0 < N_in) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < N_in) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < N_in) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < N_in) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    new_a[gid] = blstrs__scalar__Scalar_add(a0, blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(a1, a0)));
    new_b[gid] = blstrs__scalar__Scalar_add(b0, blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(b1, b0)));
}

// ── ZK inner product sumcheck (Polynomial format) ───────────────────────────
Fr_t zkip_zk(
    const Fr_t& claim,
    const FrTensor& a, const FrTensor& b,
    const std::vector<Fr_t>& u,
    const ZkMaskConfig& mask_a, const ZkMaskConfig& mask_b,
    const ZkTranscriptMask& tmask, Fr_t rho,
    std::vector<Polynomial>& proof,
    ZkIpResult& result)
{
    uint total_rounds = u.size();
    if (!total_rounds) {
        // Base case: no variables left. Compute final masked values.
        // a and b should each have 1 element.
        Fr_t a_val, b_val;
        cudaMemcpy(&a_val, a.gpu_data, sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b_val, b.gpu_data, sizeof(Fr_t), cudaMemcpyDeviceToHost);

        // The "terminal point" s* is the sequence of challenges already bound.
        // The vanishing correction at s* is computed by the caller since we
        // don't have the full challenge vector here. For the base case of the
        // recursion, we return a(0)*b(0) as the raw inner product value.
        // The masking corrections are applied in the wrapper.
        result.final_za = a_val;  // will be corrected by caller
        result.final_zb = b_val;
        return claim;
    }

    // Current round index (we process from the last variable backward)
    uint round_from_end = total_rounds;  // counts down
    // The plan processes u.back() first (same as zkip), so round_idx in the
    // forward direction is: total_original_rounds - rounds_remaining
    // We need to track this externally. For now, we use a helper approach:
    // the round index is inferred from how many challenges are left.

    // We'll implement this iteratively to track the round index properly.
    // (The recursive approach in zkip makes round tracking awkward.)

    Fr_t current_claim = claim;
    FrTensor cur_a = a;  // shallow copy (shares gpu_data)
    FrTensor cur_b = b;
    std::vector<Fr_t> bound_challenges;

    // Note: zkip processes u.back() first (last variable), but the masking
    // coefficients are indexed by variable order. The sumcheck variables in
    // the data layout are: variable 0 is the least-significant split (low/high),
    // processed last in u. The kernel splits by N_out = N/2, which corresponds
    // to the highest-indexed variable.
    //
    // zkip processes u = [u_0, u_1, ..., u_{b-1}] and binds u_{b-1} first.
    // So round j (0-indexed from start of sumcheck) binds variable b-1-j.
    //
    // For vanishing masking: coefficient index = variable index = b-1-j for round j.
    // For transcript masking: the round polynomial functions use forward indexing.

    uint b = total_rounds;

    for (uint j = 0; j < b; j++) {
        uint var_idx = b - 1 - j;  // variable being bound this round

        // Get vanishing coefficients for this round's variable
        Fr_t c_a_j = FR_ZERO;
        Fr_t c_b_j = FR_ZERO;
        if (mask_a.enabled && var_idx < mask_a.vanishing_coeffs.size()) {
            c_a_j = mask_a.vanishing_coeffs[var_idx];
        }
        if (mask_b.enabled && var_idx < mask_b.vanishing_coeffs.size()) {
            c_b_j = mask_b.vanishing_coeffs[var_idx];
        }

        // Compute honest round polynomial g(X) (degree 4)
        Polynomial g = zkip_zk_step_poly(cur_a, cur_b, c_a_j, c_b_j);

        // Compute transcript masking round polynomial p_round(X)
        Polynomial p_round = transcript_mask_round_poly(tmask, j, bound_challenges, b);

        // Combined: s(X) = g(X) + rho * p_round(X)
        Polynomial rho_poly(rho);
        Polynomial s = g + rho_poly * p_round;

        // Verify: s(0) + s(1) == current_claim
        Fr_t s0 = s(FR_ZERO);
        Fr_t s1 = s(FR_ONE);
        if (s0 + s1 != current_claim) {
            throw std::runtime_error("zkip_zk: s(0) + s(1) != claim at round " + std::to_string(j));
        }

        proof.push_back(s);

        // Bind this variable to challenge u[var_idx]
        Fr_t alpha = u[var_idx];
        current_claim = s(alpha);
        bound_challenges.push_back(alpha);

        // Fold a, b
        uint N_in = cur_a.size;
        uint N_out = (1 << ceilLog2(N_in)) >> 1;
        FrTensor new_a(N_out), new_b(N_out);
        zkip_zk_reduce_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            cur_a.gpu_data, cur_b.gpu_data,
            new_a.gpu_data, new_b.gpu_data,
            alpha, N_in, N_out);

        cur_a = new_a;
        cur_b = new_b;
    }

    // Final: compute masked evaluations at s*
    // s* = (u[b-1], u[b-2], ..., u[0]) in the order they were bound
    // = bound_challenges in the order [u[b-1], u[b-2], ..., u[0]]
    //
    // The raw values after folding are a(s*) and b(s*) (the honest MLE values).
    // The masked values are:
    //   Z_a(s*) = a(s*) + vanishing_correction(mask_a.coeffs, s*)
    //   Z_b(s*) = b(s*) + vanishing_correction(mask_b.coeffs, s*)
    //
    // We need s* in the original variable ordering (variable 0 first).
    // bound_challenges = [u[b-1], u[b-2], ..., u[0]] (bound in this order)
    // Original variable ordering: variable i was bound at round (b-1-i),
    // so s*[i] = bound_challenges[b-1-i] = u[i].
    // Simply: s* in original order = u (which is what we want).

    Fr_t raw_a, raw_b;
    cudaMemcpy(&raw_a, cur_a.gpu_data, sizeof(Fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&raw_b, cur_b.gpu_data, sizeof(Fr_t), cudaMemcpyDeviceToHost);

    // Vanishing corrections using u as the point (original variable order)
    Fr_t corr_a = FR_ZERO;
    Fr_t corr_b = FR_ZERO;
    if (mask_a.enabled) {
        corr_a = vanishing_correction(mask_a.vanishing_coeffs, u);
    }
    if (mask_b.enabled) {
        corr_b = vanishing_correction(mask_b.vanishing_coeffs, u);
    }

    result.final_za = raw_a + corr_a;
    result.final_zb = raw_b + corr_b;

    // Transcript mask at terminal point
    // The terminal point for the transcript mask is bound_challenges in forward order.
    result.p_final = eval_transcript_mask(tmask, bound_challenges);

    // Append final values to proof
    proof.push_back(Polynomial(result.final_za));
    proof.push_back(Polynomial(result.final_zb));
    proof.push_back(Polynomial(result.p_final));

    return current_claim;
}

// ── ZK inner product sumcheck (flat Fr_t format) ────────────────────────────
std::vector<Fr_t> inner_product_sumcheck_zk(
    const FrTensor& a, const FrTensor& b,
    std::vector<Fr_t> u,
    const ZkMaskConfig& mask_a, const ZkMaskConfig& mask_b,
    const ZkTranscriptMask& tmask, Fr_t rho)
{
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("inner_product_sumcheck_zk: size mismatch");
    if (a.size > (1u << log_size)) throw std::runtime_error("inner_product_sumcheck_zk: size too large");

    // T = <a, b> = sum of elementwise products (inner product on the hypercube)
    Fr_t T = (a * b).sum();

    // Combined claim: T + rho * P_sum
    Fr_t combined_claim = T + rho * tmask.P_sum;

    std::vector<Polynomial> poly_proof;
    ZkIpResult result;
    zkip_zk(combined_claim, a, b, u, mask_a, mask_b, tmask, rho, poly_proof, result);

    // Convert to flat format: 5 evals per round + 3 final values
    std::vector<Fr_t> flat_proof;
    uint num_rounds = u.size();
    for (uint j = 0; j < num_rounds; j++) {
        const Polynomial& rp = poly_proof[j];
        for (uint t = 0; t < 5; t++) {
            flat_proof.push_back(rp(FR_FROM_INT(t)));
        }
    }
    flat_proof.push_back(result.final_za);
    flat_proof.push_back(result.final_zb);
    flat_proof.push_back(result.p_final);

    return flat_proof;
}

#endif // USE_GOLDILOCKS
