#include "zknn/tlookup.cuh"
#include "proof/proof.cuh"
#include "proof/zk_sumcheck.cuh"
#include "proof/zk_round_commit.cuh"
#include "commit/commitment.cuh"

// Some utils


tLookup::tLookup(const FrTensor& table): table(table) {
}

KERNEL void tlookup_kernel(const uint* indices, const uint D, uint* counts){
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < D){
        atomicAdd(&counts[indices[tid]], 1U);
    }
}

KERNEL void count_to_m(uint* counts, Fr_t* m_ptr, uint N){
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) m_ptr[tid] = FR_FROM_INT(counts[tid]);
}

//
FrTensor tLookup::prep(const uint* indices, const uint D){
    // //copy indices to indices_cpu
    // uint indices_cpu[D];
    // cudaMemcpy(indices_cpu, indices, sizeof(uint) * D, cudaMemcpyDeviceToHost);
    // // for (uint i=0; i < D; ++ i) cout << indices_cpu[i] << " ";
    // // cout << endl;

    FrTensor m(table.size);
    uint* counts;
    cudaMalloc((void **)&counts, sizeof(uint) * table.size);
    cudaMemset(counts, 0, sizeof(uint) * table.size); // cnm

    tlookup_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(indices, D, counts);

    // //copy counts to cpu_counts
    // uint cpu_counts[table.size];
    // cudaMemcpy(cpu_counts, counts, sizeof(uint) * table.size, cudaMemcpyDeviceToHost);
    // // for (uint i=0; i < table.size; ++ i) cout << cpu_counts[i] << " ";
    // // cout << endl;

    count_to_m<<<(table.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(counts, m.gpu_data, table.size);
    cudaDeviceSynchronize();
    cudaFree(counts);
    return m;
}

KERNEL void tlookup_inv_kernel(Fr_t* in_data, Fr_t beta, Fr_t* out_data, uint N)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N){
        out_data[tid] = blstrs__scalar__Scalar_unmont(
            blstrs__scalar__Scalar_inverse(
                blstrs__scalar__Scalar_mont(
                    blstrs__scalar__Scalar_add(in_data[tid], beta)
                )
            )
        );
    }
}

KERNEL void half_tensor_kernel(const Fr_t* in_data, Fr_t* first_half_data, Fr_t* second_half_data, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        first_half_data[tid] = in_data[tid];
        second_half_data[tid] = in_data[tid + N_out];
    }
}

KERNEL void tLookup_phase1_poly_kernel(const Fr_t* A_data, const Fr_t* S_data, Fr_t alpha, Fr_t beta, Fr_t* out0, Fr_t* out1, Fr_t* out2, Fr_t* outA0, Fr_t* outA1, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t c00 = A_data[tid];
        Fr_t c01 = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
        Fr_t c10 = blstrs__scalar__Scalar_add(S_data[tid], beta);
        Fr_t c11 = blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid]);

        Fr_t alpha_mont = blstrs__scalar__Scalar_mont(alpha);
        out0[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c00, c10)));
        out1[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(c01, c10), blstrs__scalar__Scalar_mul(c00, c11))));
        out2[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c01, c11)));

        outA0[tid] = c00;
        outA1[tid] = c01;
    }
}

KERNEL void tLookup_phase1_reduce_kernel(const Fr_t* A_data, const Fr_t* S_data, Fr_t* new_A_data, Fr_t* new_S_data, Fr_t v, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t v_mont = blstrs__scalar__Scalar_mont(v);
        new_A_data[tid] = blstrs__scalar__Scalar_add(A_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid])));
        new_S_data[tid] = blstrs__scalar__Scalar_add(S_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid])));
    }
}

// BUGGY AND TOO HARD TO DEBUG
// KERNEL void tLookup_phase2_poly_kernel(const Fr_t* A_data, const Fr_t* S_data, const Fr_t* B_data, const Fr_t* T_data, const Fr_t* m_data,
//     Fr_t alpha_, Fr_t beta, uint N, uint D, Fr_t alpha_sq,
//     Fr_t* out_eval0, Fr_t* out_eval1, Fr_t* out_eval2, Fr_t* out_sum0, Fr_t* out_sum1, Fr_t* out_sum2, uint N_out)
// {
//     const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid < N_out)
//     {
//         Fr_t inv_ratio_mont = blstrs__scalar__Scalar_inverse(blstrs__scalar__Scalar_mont({D / N, 0, 0, 0, 0, 0, 0, 0}));
//         Fr_t inv_ratio = blstrs__scalar__Scalar_unmont(inv_ratio_mont);
//         Fr_t alpha__mont = blstrs__scalar__Scalar_mont(alpha_);
//         Fr_t alpha_sq_mont = blstrs__scalar__Scalar_mont(alpha_sq);
//         Fr_t inv_ratio_alpha_sq_mont = blstrs__scalar__Scalar_mul(inv_ratio_mont, alpha_sq_mont);

//         Fr_t c00 = A_data[tid];
//         Fr_t c01 = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
//         Fr_t c10 = blstrs__scalar__Scalar_add(S_data[tid], beta);
//         Fr_t c11 = blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid]);

//         Fr_t c00_ = B_data[tid];
//         Fr_t c01_ = blstrs__scalar__Scalar_sub(B_data[tid + N_out], B_data[tid]);
//         Fr_t c10_ = blstrs__scalar__Scalar_add(T_data[tid], beta);
//         Fr_t c11_ = blstrs__scalar__Scalar_sub(T_data[tid + N_out], T_data[tid]);

//         out_eval0[tid] = blstrs__scalar__Scalar_add(
//             blstrs__scalar__Scalar_mul(alpha__mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c00, c10))),
//             blstrs__scalar__Scalar_mul(inv_ratio_alpha_sq_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c00_, c10_)))
//         );
//         out_eval1[tid] = blstrs__scalar__Scalar_add(
//             blstrs__scalar__Scalar_mul(alpha__mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(c01, c10), blstrs__scalar__Scalar_mul(c00, c11)))),
//             blstrs__scalar__Scalar_mul(inv_ratio_alpha_sq_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(c01_, c10_), blstrs__scalar__Scalar_mul(c00_, c11_))))
//         );
//         out_eval2[tid] = blstrs__scalar__Scalar_add(
//             blstrs__scalar__Scalar_mul(alpha__mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c01, c11))),
//             blstrs__scalar__Scalar_mul(inv_ratio_alpha_sq_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c01_, c11_)))
//         );

//         Fr_t m0 = m_data[tid];
//         Fr_t m1 = blstrs__scalar__Scalar_sub(m_data[tid + N_out], m_data[tid]);
//         out_sum0[tid] = blstrs__scalar__Scalar_sub(
//             c00,
//             blstrs__scalar__Scalar_mul(inv_ratio_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(m0, c00_)))
//         );

//         out_sum1[tid] = blstrs__scalar__Scalar_sub(
//             c01,
//             blstrs__scalar__Scalar_mul(inv_ratio_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(m0, c01_), blstrs__scalar__Scalar_mul(m1, c00_))))
//         );

//         out_sum2[tid] = blstrs__scalar__Scalar_sub(
//             {0, 0, 0, 0, 0, 0, 0, 0},
//             blstrs__scalar__Scalar_mul(inv_ratio_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(m1, c11_)))
//         );
//     }
// }

KERNEL void tLookup_phase2_reduce_kernel(const Fr_t* A_data, const Fr_t* S_data, const Fr_t* B_data, const Fr_t* T_data, const Fr_t* m_data,
    Fr_t* new_A_data, Fr_t* new_S_data, Fr_t* new_B_data, Fr_t* new_T_data, Fr_t* new_m_data,
    Fr_t v, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t v_mont = blstrs__scalar__Scalar_mont(v);
        new_A_data[tid] = blstrs__scalar__Scalar_add(A_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid])));
        new_S_data[tid] = blstrs__scalar__Scalar_add(S_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid])));
        new_B_data[tid] = blstrs__scalar__Scalar_add(B_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(B_data[tid + N_out], B_data[tid])));
        new_T_data[tid] = blstrs__scalar__Scalar_add(T_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(T_data[tid + N_out], T_data[tid])));
        new_m_data[tid] = blstrs__scalar__Scalar_add(m_data[tid], blstrs__scalar__Scalar_mul(v_mont, blstrs__scalar__Scalar_sub(m_data[tid + N_out], m_data[tid])));
    }
}

// A.size == S.size == D
// u.size == ceilLog2(D)
// 0x39f6d3a994cebea4199cec0404d0ec02a9ded2017fff2dff7fffffff80000001
const Fr_t TWO_INV {2147483649, 2147483647, 2147429887, 2849952257, 80800770, 429714436, 2496577188, 972477353};
// const Fr_t TEMP_ZERO {0, 0, 0, 0, 0, 0, 0, 0};
// const Fr_t TEMP_ONE {1, 0, 0, 0, 0, 0, 0, 0};

Polynomial tLookup_phase1_step_poly(const FrTensor& A, const FrTensor& S,
    const Fr_t& alpha, const Fr_t& beta, const Fr_t& C, const vector<Fr_t>& u)
{
    if (A.size != S.size) throw std::runtime_error("A.size != S.size");
    uint D = A.size;
    FrTensor temp0(D >> 1), temp1(D >> 1), temp2(D >> 1), tempA0(D >> 1), tempA1(D >> 1);
    tLookup_phase1_poly_kernel<<<((D >> 1)+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, alpha, beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, tempA0.gpu_data, tempA1.gpu_data, D >> 1
    );
    vector<Fr_t> u_(u.begin(), u.end() - 1);

    Polynomial p0 ({temp0(u_), temp1(u_), temp2(u_)}); //
    Polynomial p1 ({tempA0.sum(), tempA1.sum()});
    p0 *= Polynomial::eq(u.back());
    return p0 + p1 + C * TWO_INV;
}

KERNEL void tLookup_phase2_poly_eval_kernel(const Fr_t* A_data, const Fr_t* S_data, Fr_t alpha, Fr_t beta, Fr_t* out0, Fr_t* out1, Fr_t* out2, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t c00 = A_data[tid];
        Fr_t c01 = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
        Fr_t c10 = blstrs__scalar__Scalar_add(S_data[tid], beta);
        Fr_t c11 = blstrs__scalar__Scalar_sub(S_data[tid + N_out], S_data[tid]);

        Fr_t alpha_mont = blstrs__scalar__Scalar_mont(alpha);
        out0[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c00, c10)));
        out1[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(c01, c10), blstrs__scalar__Scalar_mul(c00, c11))));
        out2[tid] = blstrs__scalar__Scalar_mul(alpha_mont, blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(c01, c11)));
    }
}

KERNEL void tLookup_phase2_poly_sum_kernel(const Fr_t* A_data, Fr_t* out0, Fr_t* out1, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        out0[tid] = A_data[tid];
        out1[tid] = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
    }
}

KERNEL void tLookup_phase2_poly_dotprod_kernel(const Fr_t* A_data, const Fr_t* B_data, Fr_t* out0, Fr_t* out1, Fr_t* out2, uint N_out)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N_out)
    {
        Fr_t A0 = A_data[tid];
        Fr_t A1 = blstrs__scalar__Scalar_sub(A_data[tid + N_out], A_data[tid]);
        Fr_t B0 = B_data[tid];
        Fr_t B1 = blstrs__scalar__Scalar_sub(B_data[tid + N_out], B_data[tid]);

        out0[tid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(A0, B0));
        out1[tid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(A1, B0), blstrs__scalar__Scalar_mul(A0, B1)));
        out2[tid] = blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(A1, B1));
    }
}

Polynomial tLookup_phase2_step_poly(const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr_t& alpha_, const Fr_t& beta, const Fr_t& inv_size_ratio, const Fr_t& alpha_sq,
    const vector<Fr_t>& u)
{
    uint N = m.size;
    uint N_out = N >> 1;
    vector<Fr_t> u_(u.begin(), u.end() - 1);

    FrTensor temp0(N_out), temp1(N_out), temp2(N_out);
    tLookup_phase2_poly_eval_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, alpha_, beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out
    );
    Polynomial p0 ({temp0(u_), temp1(u_), temp2(u_)});

    Fr_t coef = inv_size_ratio * alpha_sq;
    tLookup_phase2_poly_eval_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        B.gpu_data, T.gpu_data, coef, beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out
    );
    p0 += {{temp0(u_), temp1(u_), temp2(u_)}};

    tLookup_phase2_poly_sum_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, temp0.gpu_data, temp1.gpu_data, N_out
    );
    Polynomial p1 ({temp0.sum(), temp1.sum()});

    tLookup_phase2_poly_dotprod_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        m.gpu_data, B.gpu_data, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out
    );
    Polynomial p2 ({temp0.sum(), temp1.sum(), temp2.sum()});
    return Polynomial::eq(u.back()) * p0 + p1 - p2 * inv_size_ratio;
}

// Sigma challenges per round for tLookup: degree-3 polynomial → 4 coefficients
// → 4 openings + 1 equality = 5 challenges per round.
static const uint kTLookupSigmaPerRound = 5;

// Standard-sumcheck weights for degree-3 round polynomial: α = (2, 1, 1, 1).
static std::vector<Fr_t> tl_standard_alphas(uint d_plus_one) {
    std::vector<Fr_t> a(d_plus_one, Fr_t{1,0,0,0,0,0,0,0});
    a[0] = Fr_t{2,0,0,0,0,0,0,0};
    return a;
}

Fr_t tLookup_phase2(const Fr_t& claim, const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr_t& alpha_, const Fr_t& beta, const Fr_t& inv_size_ratio, const Fr_t& alpha_sq, const vector<Fr_t>& u, const vector<Fr_t>& v2,
    G1Jacobian_t U, G1Jacobian_t H,
    std::vector<Fr_t>::const_iterator sigma_begin,
    G1Jacobian_t T_prev_eval, Fr_t rho_prev_eval,
    std::vector<ZKSumcheckRound>& rounds_out,
    G1Jacobian_t& T_final_out, Fr_t& rho_final_out,
    vector<Polynomial>& proof)
{
    if (!v2.size()) {
        // Base case: push final evaluations for verifier
        proof.push_back(Polynomial(A(0)));  // A(u) = 1/(S(u)+beta)
        proof.push_back(Polynomial(S(0)));  // S(u)
        proof.push_back(Polynomial(B(0)));  // B(v) = 1/(T(v)+beta)
        proof.push_back(Polynomial(T(0)));  // T(v)
        proof.push_back(Polynomial(m(0)));  // m(v)
        T_final_out  = T_prev_eval;
        rho_final_out = rho_prev_eval;
        return claim;
    }
    auto p = tLookup_phase2_step_poly(A, S, B, T, m, alpha_, beta, inv_size_ratio, alpha_sq, u);
    FrTensor new_A(A.size >> 1), new_S(S.size >> 1), new_B(B.size >> 1), new_T(T.size >> 1), new_m(m.size >> 1);

    // Extract coefficients and commit via ZK round (Hyrax §4 Protocol 3).
    // tLookup round polynomials are degree 3 → 4 coefficients.
    // Standard sumcheck identity: claim == p(0) + p(1), α = (2,1,1,1).
    auto coeffs = p.getCoefficients();
    static const uint kTLookupSigmaPerRound = 5;  // 4 openings + 1 equality

    Fr_t r_j = v2.back();
    ZKRoundOutput ro = emit_zk_round(U, H, coeffs,
                                     tl_standard_alphas(coeffs.size()),
                                     sigma_begin,
                                     T_prev_eval, rho_prev_eval,
                                     r_j);
    rounds_out.push_back(ro.round);

    tLookup_phase2_reduce_kernel<<<((A.size >> 1)+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        A.gpu_data, S.gpu_data, B.gpu_data, T.gpu_data, m.gpu_data,
        new_A.gpu_data, new_S.gpu_data, new_B.gpu_data, new_T.gpu_data, new_m.gpu_data,
        v2.back(), A.size >> 1
    );

    return tLookup_phase2(p(v2.back()), new_A, new_S, new_B, new_T, new_m, alpha_ * Polynomial::eq(u.back(), v2.back()), beta, inv_size_ratio, alpha_sq * Polynomial::eq(u.back(), v2.back()), {u.begin(), u.end() - 1}, {v2.begin(), v2.end() - 1},
        U, H, sigma_begin + kTLookupSigmaPerRound,
        ro.T_eval, ro.rho_eval,
        rounds_out, T_final_out, rho_final_out,
        proof);
}

Fr_t tLookup_phase1(const Fr_t& claim, const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr_t& alpha, const Fr_t& beta, const Fr_t& C, const Fr_t& inv_size_ratio, const Fr_t& alpha_sq,
    const vector<Fr_t>& u, const vector<Fr_t>& v1, const vector<Fr_t>& v2,
    G1Jacobian_t U, G1Jacobian_t H,
    std::vector<Fr_t>::const_iterator sigma_begin,
    G1Jacobian_t T_prev_eval, Fr_t rho_prev_eval,
    std::vector<ZKSumcheckRound>& rounds_out,
    G1Jacobian_t& T_final_out, Fr_t& rho_final_out,
    vector<Polynomial>& proof)
{
    if (!v1.size())
    {
        return tLookup_phase2(claim, A, S, B, T, m, alpha, beta, inv_size_ratio, alpha_sq, u, v2,
            U, H, sigma_begin, T_prev_eval, rho_prev_eval,
            rounds_out, T_final_out, rho_final_out, proof);
    }
    else{
        auto p = tLookup_phase1_step_poly(A, S, alpha, beta, C, u);
        FrTensor new_A(A.size >> 1), new_S(S.size >> 1);

        // Extract coefficients and commit via ZK round (Hyrax §4 Protocol 3).
        auto coeffs = p.getCoefficients();
        Fr_t r_j = v1.back();

        ZKRoundOutput ro = emit_zk_round(U, H, coeffs,
                                         tl_standard_alphas(coeffs.size()),
                                         sigma_begin,
                                         T_prev_eval, rho_prev_eval,
                                         r_j);
        rounds_out.push_back(ro.round);

        tLookup_phase1_reduce_kernel<<<(A.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(
            A.gpu_data, S.gpu_data, new_A.gpu_data, new_S.gpu_data, v1.back(), A.size >> 1
        );
        return tLookup_phase1(p(v1.back()), new_A, new_S, B, T, m, alpha * Polynomial::eq(u.back(), v1.back()), beta, C * TWO_INV, inv_size_ratio, alpha_sq, {u.begin(), u.end() - 1}, {v1.begin(), v1.end() - 1}, v2,
            U, H, sigma_begin + kTLookupSigmaPerRound,
            ro.T_eval, ro.rho_eval,
            rounds_out, T_final_out, rho_final_out, proof);
    }
}



// Top-level commitment: C_0 = S·U (Hyrax §4 Step 1, blinding 0).
static G1Jacobian_t tl_top_level_commitment(G1Jacobian_t U, Fr_t S) {
    Commitment pp1(1, U);
    FrTensor Sv(1, &S);
    return pp1.commit(Sv)(0);
}

Fr_t tLookup::prove(const FrTensor& S, const FrTensor& m, const Fr_t& alpha, const Fr_t& beta,
    const vector<Fr_t>& u, const vector<Fr_t>& v,
    const Commitment& sc_pp,
    vector<Polynomial>& proof, vector<ZKSumcheckProof>& zk_sumchecks)
{
    const uint D = S.size;
    if (m.size != table.size) {
        throw std::runtime_error("m.size != table.size");
    }
    const uint N = m.size;

    if (D != 1 << ceilLog2(D) || N != 1 << ceilLog2(N) || D % N != 0) {
        throw std::runtime_error("D or N is not power of 2, or D is not divisible by N");
    }


    FrTensor A(D), B(N);
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        S.gpu_data,
        beta,
        A.gpu_data,
        D
    );

    tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        table.gpu_data,
        beta,
        B.gpu_data,
        N
    );

    if (u.size() != ceilLog2(D)) throw std::runtime_error("u.size() != ceilLog2(D)");
    if (v.size() != ceilLog2(D)) throw std::runtime_error("v.size() != ceilLog2(D)");

    vector<Fr_t> v1 = {v.begin(), v.begin() + ceilLog2(D / N)};
    vector<Fr_t> v2 = {v.begin() + ceilLog2(D / N), v.end()};

    Fr_t C = alpha * alpha - (B * m).sum();

    Fr_t alpha_sq = alpha * alpha;
    Fr_t claim = alpha + alpha_sq;
    Fr_t N_Fr = FR_FROM_INT(N);
    Fr_t D_Fr = FR_FROM_INT(D);

    // ZK setup: total rounds = v1.size() + v2.size() = v.size() = ceilLog2(D).
    // Sigma challenges: 5 per round (4 openings for degree-3 + 1 equality).
    G1Jacobian_t U_gen = sc_pp.u_generator;
    G1Jacobian_t H_gen = sc_pp.hiding_generator;
    uint total_rounds = v.size();
    auto sigma_tl = random_vec(total_rounds * kTLookupSigmaPerRound);

    // Hyrax §4 Step 1: C_0 = claim·U, blinding 0.
    G1Jacobian_t T_prev = tl_top_level_commitment(U_gen, claim);
    Fr_t rho_prev = {0,0,0,0,0,0,0,0};

    ZKSumcheckProof zk_proof;
    Fr_t result = tLookup_phase1(claim, A, S, B, table, m,
        alpha, beta, C, N_Fr / D_Fr, alpha_sq,
        u, v1, v2,
        U_gen, H_gen, sigma_tl.begin(),
        T_prev, rho_prev,
        zk_proof.rounds, zk_proof.T_final, rho_prev,
        proof);
    zk_sumchecks.push_back(std::move(zk_proof));
    return result;
}

KERNEL void tlookuprange_init_kernel(Fr_t* table_ptr, int low, uint len, uint table_size)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < table_size)
    {
        int val = (tid < len) ? static_cast<int>(tid) + low : static_cast<int>(len) + low - 1;
        table_ptr[tid] = int_to_scalar(val);
    }
}


// tLookup is a super class of tLookupRange. The length has to be padded to be a power of 2
tLookupRange::tLookupRange(int low, uint len) : low(low), tLookup(1 << ceilLog2(len))
{
    // Get the pointer to the super class's table
    Fr_t* table_ptr = table.gpu_data;
    // Initialize the table
    tlookuprange_init_kernel<<<(table.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(table_ptr, low, len, table.size);
}

KERNEL void lookuprange_prep_kernel(const int* vals, int low, uint* indices, uint N)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        indices[tid] = static_cast<uint>(vals[tid] - low);
    }
}

FrTensor tLookupRange::prep(const int* vals, const uint D){
    // assign uint indices pointer on gpu
    uint* indices;
    cudaMalloc((void **)&indices, sizeof(uint) * D);
    // convert vals (which should be on gpu) to indices
    lookuprange_prep_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(vals, low, indices, D);
    auto out = tLookup::prep(indices, D);
    cudaFree(indices);
    return out;
}

// table_bound: size of the target lookup table.  Indices are clamped to
// [0, table_bound-1] to prevent out-of-bounds GPU memory writes.
// Scalar_sub handles both positive and p-k (negative) field elements correctly
// because the subtraction is done in Fp: (p-k) - (p-low) = low-k (mod p),
// which gives the correct table index for valid negative values like remainders.
// Values that are genuinely out-of-range (raw >= table_bound) are clamped to
// the last entry, which is safe for CDF lookups where diff >> table exceeds Phi≈1.
KERNEL void lookuprange_tensor_prep_kernel(const Fr_t* vals, int low, uint* indices, uint N, uint table_bound)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        uint raw = blstrs__scalar__Scalar_sub(vals[tid], int_to_scalar(low)).val[0];
        indices[tid] = (raw < table_bound) ? raw : (table_bound - 1u);
    }
}

FrTensor tLookupRange::prep(const FrTensor& vals){
    // assign uint indices pointer on gpu
    uint* indices;
    cudaMalloc((void **)&indices, sizeof(uint) * vals.size);
    // convert vals (which should be on gpu) to indices, clamped to [0, table.size-1]
    lookuprange_tensor_prep_kernel<<<(vals.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(vals.gpu_data, low, indices, vals.size, table.size);
    auto out = tLookup::prep(indices, vals.size);
    cudaFree(indices);
    return out;
}

tLookupRangeMapping::tLookupRangeMapping(int low, uint len, const FrTensor& mvals):
    tLookupRange(low, len), mapped_vals(1 << ceilLog2(len))
{
    if (mvals.size != len) throw std::runtime_error("mvals.size != len");
    // fill mapp_vals with zeros
    cudaMemset(mapped_vals.gpu_data, 0, sizeof(Fr_t) * mapped_vals.size);
    mapped_vals += mvals(mvals.size - 1);
    // copy vals to mapped_vals
    cudaMemcpy(mapped_vals.gpu_data, mvals.gpu_data, sizeof(Fr_t) * mvals.size, cudaMemcpyDeviceToDevice);
}

KERNEL void lookuprangemapping_kernel(const uint* indices, const Fr_t* val_ptr, Fr_t* out_ptr, uint N)
{
    const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        out_ptr[tid] = val_ptr[indices[tid]];
    }
}



pair<FrTensor, FrTensor> tLookupRangeMapping::operator()(const int* vals, const uint D)
{
    uint* indices;
    cudaMalloc((void **)&indices, sizeof(uint) * D);
    // convert vals (which should be on gpu) to indices
    lookuprange_prep_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(vals, low, indices, D);
    auto m = tLookup::prep(indices, D);

    FrTensor y(D);
    lookuprangemapping_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(indices, mapped_vals.gpu_data, y.gpu_data, D);
    cudaDeviceSynchronize();
    cudaFree(indices);
    return {y, m};
}

pair<FrTensor, FrTensor> tLookupRangeMapping::operator()(const FrTensor& vals)
{
    uint* indices;
    cudaMalloc((void **)&indices, sizeof(uint) * vals.size);
    // convert vals (which should be on gpu) to indices, clamped to [0, mapped_vals.size-1]
    lookuprange_tensor_prep_kernel<<<(vals.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(vals.gpu_data, low, indices, vals.size, mapped_vals.size);
    auto m = tLookup::prep(indices, vals.size);
    FrTensor y(vals.size);
    lookuprangemapping_kernel<<<(vals.size+FrNumThread-1)/FrNumThread,FrNumThread>>>(indices, mapped_vals.gpu_data, y.gpu_data, vals.size);
    cudaDeviceSynchronize();
    cudaFree(indices);
    return {y, m};
}

KERNEL void tlookuprange_pad_m(Fr_t* m_ptr, uint index_padded, uint num_added)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        m_ptr[index_padded] = blstrs__scalar__Scalar_add(m_ptr[index_padded], FR_FROM_INT(num_added));
}

Fr_t tLookupRangeMapping::prove(const FrTensor& S_in, const FrTensor& S_out, const FrTensor& m,
        const Fr_t& r, const Fr_t& alpha, const Fr_t& beta,
        const vector<Fr_t>& u, const vector<Fr_t>& v,
        const Commitment& sc_pp,
        vector<Polynomial>& proof, vector<ZKSumcheckProof>& zk_sumchecks)
{
    const uint D = S_in.size;
    if (m.size != table.size) throw std::runtime_error("m.size != table.size");
    const uint N = m.size;

    if (D != 1 << ceilLog2(D))
    {
        auto S_in_ = S_in.pad({D}, table(0));
        auto S_out_ = S_out.pad({D}, mapped_vals(0));
        FrTensor m_(m);
        tlookuprange_pad_m<<<1,1>>>(m_.gpu_data, 0, (1 << ceilLog2(D)) - D);
        return prove(S_in_, S_out_, m_, r, alpha, beta, u, v, sc_pp, proof, zk_sumchecks);
    }

    if (N != 1 << ceilLog2(N) || D % N != 0) {
        throw std::runtime_error("N is not power of 2, or D is not divisible by N");
    }

    FrTensor A(D), B(N);
    auto S_com = S_in + S_out * r;
    auto T_com = table + mapped_vals * r;
    tlookup_inv_kernel<<<(D+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        S_com.gpu_data,
        beta,
        A.gpu_data,
        D
    );

    tlookup_inv_kernel<<<(N+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        T_com.gpu_data,
        beta,
        B.gpu_data,
        N
    );

    if (u.size() != ceilLog2(D)) throw std::runtime_error("u.size() != ceilLog2(D)");
    if (v.size() != ceilLog2(D)) throw std::runtime_error("v.size() != ceilLog2(D)");

    vector<Fr_t> v1 = {v.begin(), v.begin() + ceilLog2(D / N)};
    vector<Fr_t> v2 = {v.begin() + ceilLog2(D / N), v.end()};

    Fr_t C = alpha * alpha - (B * m).sum();

    Fr_t alpha_sq = alpha * alpha;
    Fr_t claim = alpha + alpha_sq;
    Fr_t N_Fr = FR_FROM_INT(N);
    Fr_t D_Fr = FR_FROM_INT(D);

    // ZK setup (same as tLookup::prove).
    G1Jacobian_t U_gen = sc_pp.u_generator;
    G1Jacobian_t H_gen = sc_pp.hiding_generator;
    uint total_rounds = v.size();
    auto sigma_tl = random_vec(total_rounds * kTLookupSigmaPerRound);

    G1Jacobian_t T_prev = tl_top_level_commitment(U_gen, claim);
    Fr_t rho_prev = {0,0,0,0,0,0,0,0};

    ZKSumcheckProof zk_proof;
    Fr_t result = tLookup_phase1(claim, A, S_com, B, T_com, m,
        alpha, beta, C, N_Fr / D_Fr, alpha_sq,
        u, v1, v2,
        U_gen, H_gen, sigma_tl.begin(),
        T_prev, rho_prev,
        zk_proof.rounds, zk_proof.T_final, rho_prev,
        proof);
    zk_sumchecks.push_back(std::move(zk_proof));
    return result;
}