#include "proof/proof.cuh"
#include "util/ioutils.cuh"
#include <cuda_fp16.h>
#include <fstream>


#ifdef USE_GOLDILOCKS
// Convert int32 weight data (on GPU) to fp16 representation
KERNEL void int_to_fp16_kernel(const int* int_data, __half* fp16_data,
                               float inv_scale, uint n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    fp16_data[gid] = __float2half((float)int_data[gid] * inv_scale);
}
#endif

#ifdef USE_GOLDILOCKS
void verifyWeightClaim(const Weight& w, const Claim& c)
{
    vector<Fr_t> u_cat = concatenate(vector<vector<Fr_t>>({c.u[1], c.u[0]}));
    auto w_padded = w.weight.pad({w.in_dim, w.out_dim});
    Fr_t opening = FriPcs::open(w_padded.gpu_data, w_padded.size, w.com, u_cat, true);
    if (opening != c.claim) throw std::runtime_error("verifyWeightClaim: opening != c.claim");
    cout << "Opening complete" << endl;
}

Weight create_weight(string weight_filename, string com_filename, uint in_dim, uint out_dim,
                     unsigned long scaling_factor)
{
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    auto w_padded = weight.pad({in_dim, out_dim});
    FriPcsCommitment com;
    // Load saved commitment if available, otherwise compute and save
    std::ifstream test(com_filename, std::ios::binary);
    if (test.good()) {
        test.close();
        com = FriPcsCommitment::load(com_filename);
    } else {
        com = FriPcs::commit(w_padded.gpu_data, w_padded.size);
        com.save(com_filename);
    }

    // Build fp16 weight copy for fast matmul (if scaling_factor provided)
    __half* weight_fp16 = nullptr;
    if (scaling_factor > 0) {
        uint n = weight.size;
        cudaMalloc(&weight_fp16, sizeof(__half) * n);
        // Load int data to GPU, then convert to fp16
        auto file_size = findsize(weight_filename);
        int* int_gpu;
        cudaMalloc(&int_gpu, file_size);
        loadbin(weight_filename, int_gpu, file_size);
        float inv_scale = 1.0f / (float)scaling_factor;
        int_to_fp16_kernel<<<(n + 255) / 256, 256>>>(int_gpu, weight_fp16, inv_scale, n);
        cudaFree(int_gpu);
    }

    return Weight{weight, com, in_dim, out_dim, weight_fp16, scaling_factor};
}
#else
void verifyWeightClaim(const Weight& w, const Claim& c)
{
    vector<Fr_t> u_cat = concatenate(vector<vector<Fr_t>>({c.u[1], c.u[0]}));
    auto w_padded = w.weight.pad({w.in_dim, w.out_dim});
    auto opening = w.generator.open(w_padded, w.com, u_cat);
    if (opening != c.claim) throw std::runtime_error("verifyWeightClaim: opening != c.claim");
    cout << "Opening complete" << endl;
}
#endif

KERNEL void Fr_ip_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;
    Fr_t a0 = (gid0 < in_size) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < in_size) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < in_size) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < in_size) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_mul(a0, b0);
    out1[gid] = blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(a0, blstrs__scalar__Scalar_sub(b1, b0)), 
        blstrs__scalar__Scalar_mul(b0, blstrs__scalar__Scalar_sub(a1, a0)));
    out2[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_sub(a1, a0), blstrs__scalar__Scalar_sub(b1, b0));
}

void Fr_ip_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (begin >= end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    // No sync needed: .sum() starts with cudaMemcpy which synchronizes
    proof.push_back(out0.sum());
    proof.push_back(out1.sum());
    proof.push_back(out2.sum());

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *begin, in_size, out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *begin, in_size, out_size);
    // No sync needed: kernels execute in stream order
    Fr_ip_sc(a_new, b_new, begin + 1, end, proof);
}

vector<Fr_t> inner_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u)
{
    vector<Fr_t> proof;
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_ip_sc(a, b, u.begin(), u.end(), proof);
    return proof;
}

void Fr_hp_sc(const FrTensor& a, const FrTensor& b, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 5");
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        proof.push_back(b(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    vector<Fr_t> u_(u_begin + 1, u_end);
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(b.gpu_data, b_new.gpu_data, *v_begin, in_size, out_size);
    Fr_hp_sc(a_new, b_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> hadamard_product_sumcheck(const FrTensor& a, const FrTensor& b, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions 1");
    uint log_size = u.size();
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions 2");
    if (a.size <= (1 << (log_size - 1))) throw std::runtime_error("Incompatible dimensions 3");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions 4");

    Fr_hp_sc(a, b, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

KERNEL void Fr_bin_sc_step(GLOBAL Fr_t *a, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_size, uint out_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_size) return;
    
    Fr_t a0 = (2 * gid < in_size) ? a[2 * gid] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (2 * gid + 1 < in_size) ? a[2 * gid + 1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(a0, a0), a0);
    Fr_t diff = blstrs__scalar__Scalar_sub(a1, a0);
    out1[gid] = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_double(a0), diff), diff);
    out2[gid] = blstrs__scalar__Scalar_sqr(diff);
}

void Fr_bin_sc(const FrTensor& a, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        proof.push_back(a(0));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_bin_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);
    vector<Fr_t> u_(u_begin + 1, u_end);
    proof.push_back(out0(u_));
    proof.push_back(out1(u_));
    proof.push_back(out2(u_));

    FrTensor a_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(a.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size);
    Fr_bin_sc(a_new, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

vector<Fr_t> binary_sumcheck(const FrTensor& a, vector<Fr_t> u, vector<Fr_t> v)
{
    vector<Fr_t> proof;
    if (u.size() != v.size()) throw std::runtime_error("Incompatible dimensions");
    uint log_size = u.size();
    if (a.size <= (1 << (log_size))/2) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1 << log_size)) throw std::runtime_error("Incompatible dimensions");

    Fr_bin_sc(a, u.begin(), u.end(), v.begin(), v.end(), proof);
    return proof;
}

// ── Interactive inner product sumcheck ───────────────────────────────────────
// Generates challenges per round via random_vec(1), after computing and
// recording the round polynomial.  This is essential for interactive soundness.

static void Fr_ip_sc_interactive(const FrTensor& a, const FrTensor& b,
    uint rounds_left, vector<Polynomial>& proof, vector<Fr_t>& u_out, Fr_t claim)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (rounds_left == 0) {
        // Base case: record final evaluations a(0), b(0)
        proof.push_back(Polynomial(a(0)));
        proof.push_back(Polynomial(b(0)));
        return;
    }

    auto in_size = a.size;
    auto out_size = (in_size + 1) / 2;
    FrTensor out0(out_size), out1(out_size), out2(out_size);
    Fr_ip_sc_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        a.gpu_data, b.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, in_size, out_size);

    Fr_t s0 = out0.sum(), s1 = out1.sum(), s2 = out2.sum();
    // Round polynomial p(x) with p(0) = s0, p(1) = s0+s1+s2
    // (coefficients: s0, s1, s2 in the basis 1, x, x^2)
    Polynomial p({s0, s1, s2});
    proof.push_back(p);

    Fr_t p0 = s0;
    Fr_t p1 = s0 + s1 + s2;
    if (p0 + p1 != claim)
        throw std::runtime_error("interactive ip sumcheck: p(0)+p(1) != claim");

    // Generate challenge AFTER polynomial is committed
    Fr_t r = random_vec(1)[0];
    u_out.push_back(r);

    FrTensor a_new(out_size), b_new(out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        a.gpu_data, a_new.gpu_data, r, in_size, out_size);
    Fr_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(
        b.gpu_data, b_new.gpu_data, r, in_size, out_size);

    Fr_ip_sc_interactive(a_new, b_new, rounds_left - 1, proof, u_out, p(r));
}

void inner_product_sumcheck_interactive(const FrTensor& a, const FrTensor& b,
    uint num_rounds, vector<Polynomial>& proof, vector<Fr_t>& u_out)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1u << num_rounds)) throw std::runtime_error("Incompatible dimensions");

    // Initial claim: <a, b>
    Fr_t claim = (a * b).sum();
    Fr_ip_sc_interactive(a, b, num_rounds, proof, u_out, claim);
}

// ── Interactive hadamard product sumcheck ────────────────────────────────────
// The HP sumcheck round polynomial depends on future u values (partial MLE
// evaluation), so u must be generated upfront. The v (folding) challenges are
// also generated upfront via random_vec(), which dispatches to the verifier's
// challenge source in interactive mode.
//
// The round polynomials and final openings are pushed to the proof vector for
// serialization and verification.

void hadamard_product_sumcheck_interactive(const FrTensor& a, const FrTensor& b,
    uint num_rounds, vector<Polynomial>& proof, vector<Fr_t>& u_out, vector<Fr_t>& v_out)
{
    if (a.size != b.size) throw std::runtime_error("Incompatible dimensions");
    if (a.size > (1u << num_rounds)) throw std::runtime_error("Incompatible dimensions");

    // Generate challenges via random_vec (dispatches to challenge source)
    u_out = random_vec(num_rounds);
    v_out = random_vec(num_rounds);

    // Run the HP sumcheck
    vector<Fr_t> hp_proof;
    Fr_hp_sc(a, b, u_out.begin(), u_out.end(), v_out.begin(), v_out.end(), hp_proof);

    // Convert Fr_t triples to Polynomial objects in the proof vector.
    // Each round produces 3 values; the base case appends 2 final openings.
    for (size_t i = 0; i + 2 < hp_proof.size(); i += 3) {
        proof.push_back(Polynomial({hp_proof[i], hp_proof[i+1], hp_proof[i+2]}));
    }
    if (hp_proof.size() % 3 == 2) {
        proof.push_back(Polynomial(hp_proof[hp_proof.size() - 2]));
        proof.push_back(Polynomial(hp_proof[hp_proof.size() - 1]));
    }
}

// TODO: DEPRECATE ABOVE


bool operator==(const Fr_t& a, const Fr_t& b)
{
#ifdef USE_GOLDILOCKS
    return a.val == b.val;
#else
    return (a.val[0] == b.val[0] && a.val[1] == b.val[1] && a.val[2] == b.val[2] && a.val[3] == b.val[3] && a.val[4] == b.val[4] && a.val[5] == b.val[5] && a.val[6] == b.val[6] && a.val[7] == b.val[7]);
#endif
}

bool operator!=(const Fr_t& a, const Fr_t& b)
{
    return !(a == b);
}

KERNEL void hadamard_split_kernel(const Fr_t* in_ptr, Fr_t* out0, Fr_t* out1, uint N_out)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    out0[gid] = in_ptr[gid];
    out1[gid] = blstrs__scalar__Scalar_sub(in_ptr[gid + N_out], in_ptr[gid]);
}

KERNEL void hadamard_reduce_kernel(const Fr_t* in_ptr, Fr_t v, Fr_t* out, uint N_out)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= N_out) return;
    auto v_mont = blstrs__scalar__Scalar_mont(v);
    auto temp0 = in_ptr[gid];
    auto temp1 = blstrs__scalar__Scalar_sub(in_ptr[gid + N_out], in_ptr[gid]);
    out[gid] = blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(v_mont, temp1), temp0);
}

// Hadamard of multiple ones, Y and Xs should have been padded if their shapes are not a multiple of 2
Fr_t multi_hadamard_sumchecks(const Fr_t& claim, const vector<FrTensor>& Xs, const vector<Fr_t>& u, const vector<Fr_t>& v, vector<Polynomial>& proof)
{   
    auto N = Xs[0].size;
    //ensure N is a power of 2
    if (N == 1) return claim;
    if ((N & (N - 1)) != 0) throw std::runtime_error("N is not a power of 2");
    uint N_out = N >> 1;
    vector<FrTensor> out;

    FrTensor X0(N_out), X1(N_out);
    hadamard_split_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(Xs[0].gpu_data, X0.gpu_data, X1.gpu_data, N_out);
    out.push_back(X0);
    out.push_back(X1);

    for (uint i = 1; i < Xs.size(); ++ i)
    {
        if (Xs[i].size != N) throw std::runtime_error("Xs[i] size is not N");
        FrTensor X0(N_out), X1(N_out);
        hadamard_split_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(Xs[i].gpu_data, X0.gpu_data, X1.gpu_data, N_out);

        out.push_back(out.back() * X1);
        for (int j = i; j >= 1; -- j)
        {
            out[j] *= X0;
            out[j] += out[j - 1] * X1;
        }
        out[0] *= X0;
    }

    vector<Fr_t> coefs;
    vector<Fr_t> u_(u.begin(), u.end() - 1);
    for (auto& x : out) coefs.push_back(x(u_));
    proof.push_back(Polynomial(coefs));
    auto p = proof.back() * Polynomial::eq(u.back());

#ifdef USE_GOLDILOCKS
    Fr_t fr_zero = {0ULL}, fr_one = {1ULL};
#else
    Fr_t fr_zero = {0, 0, 0, 0, 0, 0, 0, 0}, fr_one = {1, 0, 0, 0, 0, 0, 0, 0};
#endif
    if (claim != p(fr_zero) + p(fr_one)) throw std::runtime_error("multi_hadamard_sumchecks: claim != p(0) + p(1)");

    auto new_claim = proof.back()(v.back());
    vector<FrTensor> new_Xs;
    for (auto& X: Xs)
    {
        FrTensor new_X(N_out);
        hadamard_reduce_kernel<<<(N_out+FrNumThread-1)/FrNumThread,FrNumThread>>>(X.gpu_data, v.back(), new_X.gpu_data, N_out);
        new_Xs.push_back(new_X);
    }

    return multi_hadamard_sumchecks(new_claim, new_Xs, u_, {v.begin(), v.end() - 1}, proof);
}