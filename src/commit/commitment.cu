#include "commit/commitment.cuh"
#include "util/ioutils.cuh"

Commitment Commitment::random(uint size)
{
    Commitment out(size, G1Jacobian_generator);
    out *= FrTensor::random(size);
    return out;
}

// Pedersen hiding pp: G_i = s_i * G for i = 0..size-1, plus H = s_H * G.
// All scalars sampled locally via FrTensor::random.
//
// Reference: Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//   "Informally, a commitment scheme allows a sender to produce a message
//    C = Com(m) that hides m from a receiver but binds the sender to the
//    value m."
//
// Trust model: whoever runs this function knows every scalar {s_i, s_H}.
// Hiding against the verifier holds regardless — the verifier never sees
// the scalars.  Binding against an equivocating prover requires that the
// prover not know the cross-dlogs among {G_i, H}; in this codebase that's
// deferred to the same trusted-setup assumption that the non-hiding
// pp already makes (zkLLM §3.4).
Commitment Commitment::hiding_random(uint size)
{
    Commitment out(size, G1Jacobian_generator);
    out *= FrTensor::random(size);

    Commitment h_tmp(1, G1Jacobian_generator);
    h_tmp *= FrTensor::random(1);
    out.hiding_generator = h_tmp(0);

    // Sample U independently from the same generator — used as the
    // inner-product target in Hyrax §A.2 Figure 6.
    Commitment u_tmp(1, G1Jacobian_generator);
    u_tmp *= FrTensor::random(1);
    out.u_generator = u_tmp(0);

    if (!out.is_hiding()) {
        throw std::runtime_error(
            "Commitment::hiding_random: sampled H is the G1 identity; "
            "RNG bug or catastrophic coincidence");
    }
    if (!out.is_openable()) {
        throw std::runtime_error(
            "Commitment::hiding_random: sampled U is the G1 identity; "
            "RNG bug or catastrophic coincidence");
    }
    return out;
}

bool Commitment::is_hiding() const
{
    // In Jacobian coordinates (X, Y, Z), Z = 0 represents the identity.
    // Both identity conventions in this codebase (G1Jacobian_ZERO with
    // Y=0 and blstrs__g1__G1Affine_ZERO with Y=1) share Z = 0.
    // Fp has blstrs__fp__Fp_LIMBS = 12 u32 limbs; check every one.
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (hiding_generator.z.val[i] != 0) return true;
    }
    return false;
}

bool Commitment::is_openable() const
{
    if (!is_hiding()) return false;
    for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
        if (u_generator.z.val[i] != 0) return true;
    }
    return false;
}

void Commitment::save_hiding(const string& pp_file) const
{
    // Write the G_i vector with the inherited serializer — the legacy pp
    // file format is unchanged.
    this->save(pp_file);

    // Write H to a sidecar file.  One G1Jacobian_t on disk.
    G1Jacobian_t* d_h;
    cudaMalloc(&d_h, sizeof(G1Jacobian_t));
    cudaMemcpy(d_h, &hiding_generator, sizeof(G1Jacobian_t), cudaMemcpyHostToDevice);
    savebin(pp_file + ".h", d_h, sizeof(G1Jacobian_t));
    cudaFree(d_h);

    // Write U to its own sidecar.  Independent of the `.h` format so a
    // future reader with only `.h` continues to work in the hiding-but-
    // not-openable configuration.
    G1Jacobian_t* d_u;
    cudaMalloc(&d_u, sizeof(G1Jacobian_t));
    cudaMemcpy(d_u, &u_generator, sizeof(G1Jacobian_t), cudaMemcpyHostToDevice);
    savebin(pp_file + ".u", d_u, sizeof(G1Jacobian_t));
    cudaFree(d_u);
}

Commitment Commitment::load_hiding(const string& pp_file)
{
    // Load G_i via inherited ctor.  `Commitment` inherits G1TensorJacobian's
    // string ctor, which gives a non-hiding Commitment with
    // hiding_generator == identity.
    Commitment out(pp_file);

    string h_file = pp_file + ".h";
    FILE* probe = fopen(h_file.c_str(), "rb");
    if (!probe) {
        // Legacy pp without H sidecar.  Stays non-hiding.  Upstream callers
        // that need hiding should check is_hiding() and error out.
        return out;
    }
    fclose(probe);

    if (findsize(h_file) != sizeof(G1Jacobian_t)) {
        throw std::runtime_error(
            "Commitment::load_hiding: H sidecar file has unexpected size: " + h_file);
    }

    G1Jacobian_t* d_h;
    cudaMalloc(&d_h, sizeof(G1Jacobian_t));
    loadbin(h_file, d_h, sizeof(G1Jacobian_t));
    cudaMemcpy(&out.hiding_generator, d_h, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d_h);

    // U sidecar is optional: pp that was hiding-only (Phase 1) won't have
    // it, and load_hiding stays backward-compatible by leaving
    // u_generator at identity.  Phase 2 callers must check is_openable().
    string u_file = pp_file + ".u";
    FILE* u_probe = fopen(u_file.c_str(), "rb");
    if (!u_probe) return out;
    fclose(u_probe);

    if (findsize(u_file) != sizeof(G1Jacobian_t)) {
        throw std::runtime_error(
            "Commitment::load_hiding: U sidecar file has unexpected size: " + u_file);
    }

    G1Jacobian_t* d_u;
    cudaMalloc(&d_u, sizeof(G1Jacobian_t));
    loadbin(u_file, d_u, sizeof(G1Jacobian_t));
    cudaMemcpy(&out.u_generator, d_u, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(d_u);

    return out;
}

// KERNEL void com_sum_row_kernel(const G1Jacobian_t* arr, G1Jacobian_t* arr_out, uint m, uint n) {
//     auto row = GET_GLOBAL_ID();
//     if (row < m) {
//         G1Jacobian_t rowSum = arr[row * n];
//         for (uint i = 1; i < n; ++ i) {
//             rowSum = blstrs__g1__G1Affine_add(rowSum, arr[row * n + i]);
//         }
//         arr_out[row] = rowSum;
//     }
    
// }

G1TensorJacobian Commitment::commit(const FrTensor& t) const
{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit - Incompatible dimensions");

    uint m = t.size / size;
    G1TensorJacobian temp = (*this) * t;
    return temp.rowwise_sum(m, size);
}

DEVICE G1Jacobian_t commit_int_dev_func(G1Jacobian_t a, Fr_t s) {
    const int x = scalar_to_int(s);
    G1Jacobian_t out = blstrs__g1__G1Affine_ZERO;
    #pragma unroll
    for (uint i = 0; i < 31; ++ i) {
        if ((x >> i) & 1) out = blstrs__g1__G1Affine_add(out, a);
        a = blstrs__g1__G1Affine_double(a);
    }
    
    if (x < 0) out = blstrs__g1__G1Affine_add(out, G1Jacobian_minus(a));
    return out;
}

KERNEL void commit_int_kernel(const G1Jacobian_t* generators, const Fr_t* scalars, G1Jacobian_t* out, uint n, uint m) {
    const uint gid = GET_GLOBAL_ID();
    if (gid >= m * n) return;
    out[gid] = commit_int_dev_func(generators[gid % n], scalars[gid]);
}

G1TensorJacobian Commitment::commit_int (const FrTensor& t) const{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit_int - Incompatible dimensions");

    uint m = t.size / size;
    G1TensorJacobian temp(t.size);
    commit_int_kernel<<<(m*size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, temp.gpu_data, size, m);
    cudaDeviceSynchronize();
    return temp.rowwise_sum(m, size);
}

G1TensorJacobian Commitment::commit_int_multi(const vector<FrTensor>& ts) const{
    uint num_row = 0;
    for (auto& t : ts) {
        if (t.size % size != 0) throw std::runtime_error("Commitment::commit_int_multi - Incompatible dimensions");
        num_row += t.size / size;
    }

    G1TensorJacobian temp(num_row * size);
    auto temp_start = temp.gpu_data;
    for (auto& t: ts)
    {
        uint m = t.size / size;
        commit_int_kernel<<<(m*size+G1NumThread-1)/G1NumThread,G1NumThread>>>(gpu_data, t.gpu_data, temp_start, size, m);
        cudaDeviceSynchronize();
        temp_start += m * size;
    }
    return temp.rowwise_sum(temp.size / size, size);
}

// ─── Hiding commit variants ─────────────────────────────────────────────────
//
// Each returns (C, r) where C[row] = Σ_j G_j · t[row,j] + r[row] · H and
// r[row] ← F_r uniformly, independently per row.
//
// Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//   "We say that Com_pp(m; r) is a commitment to the message m with
//    randomness r".
//
// Requires pp to have been produced by `hiding_random`; we throw
// otherwise so a caller who forgot to migrate their pp doesn't
// silently get a non-hiding commitment.

// Shared helper: given a plain non-hiding row-wise commitment `base_com`
// (size m × 1 in field elements), produce `base_com + r · H` along with
// the fresh r tensor.
static Commitment::HidingCommit add_blinding(
    const Commitment& pp,
    G1TensorJacobian&& base_com,
    uint num_rows)
{
    if (!pp.is_hiding()) {
        throw std::runtime_error(
            "Commitment::commit*_hiding: pp was not produced by "
            "hiding_random(); hiding_generator is identity");
    }

    FrTensor r = FrTensor::random(num_rows);

    // Compute r[row] · H per row using the existing size-1 Commitment
    // scalar-mul path (H as the single generator).
    Commitment h_as_commitment(1, pp.hiding_generator);
    G1TensorJacobian rH = h_as_commitment.commit(r);  // size == num_rows

    return {base_com + rH, std::move(r)};
}

Commitment::HidingCommit Commitment::commit_hiding(const FrTensor& t) const
{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit_hiding - Incompatible dimensions");
    uint m = t.size / size;
    G1TensorJacobian base = this->commit(t);
    return add_blinding(*this, std::move(base), m);
}

Commitment::HidingCommit Commitment::commit_int_hiding(const FrTensor& t) const
{
    if (t.size % size != 0) throw std::runtime_error("Commitment::commit_int_hiding - Incompatible dimensions");
    uint m = t.size / size;
    G1TensorJacobian base = this->commit_int(t);
    return add_blinding(*this, std::move(base), m);
}

KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u, // always assume that scalars and u is in mont form
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint old_size, uint new_size)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= new_size) return;

    uint gid0 = 2 * gid;
    uint gid1 = 2 * gid + 1;

    if (gid1 >= old_size) {
        new_scalars[gid] = blstrs__scalar__Scalar_sub(scalars[gid0], 
            blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, scalars[gid0]))
        );
        new_generators[gid] = G1Jacobian_mul(generators[gid0], u);
        temp_out[gid] = G1Jacobian_mul(generators[gid0], scalars[gid0]);
        temp_out0[gid] = blstrs__g1__G1Affine_ZERO;
        temp_out1[gid] = blstrs__g1__G1Affine_ZERO;
        return;
    }


    new_scalars[gid] = blstrs__scalar__Scalar_add(scalars[gid0], blstrs__scalar__Scalar_mont(blstrs__scalar__Scalar_mul(u, blstrs__scalar__Scalar_sub(scalars[gid1], scalars[gid0]))));
    new_generators[gid] = blstrs__g1__G1Affine_add(generators[gid1], G1Jacobian_mul(blstrs__g1__G1Affine_add(generators[gid0], G1Jacobian_minus(generators[gid1])), u));
    temp_out[gid] = blstrs__g1__G1Affine_add(G1Jacobian_mul(generators[gid0], scalars[gid0]), G1Jacobian_mul(generators[gid1], scalars[gid1]));
    temp_out0[gid] = G1Jacobian_mul(generators[gid1], scalars[gid0]);
    temp_out1[gid] = G1Jacobian_mul(generators[gid0], scalars[gid1]);
}

Fr_t Commitment::me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof)
{
    if (t.size != generators.size) throw std::runtime_error("Commitment::me_open - Incompatible dimensions "+ std::to_string(t.size) + " " + std::to_string(generators.size));
    if (begin >= end)
    {
        proof.push_back(generators(0));
        return t(0);
    }
    uint new_size = (t.size + 1) / 2;
    FrTensor new_scalars(new_size);
    Commitment new_generators(new_size);
    G1TensorJacobian temp(new_size), temp0(new_size), temp1(new_size);
    me_open_step<<<(new_size+G1NumThread-1)/G1NumThread,G1NumThread>>>(t.gpu_data, generators.gpu_data, *begin, 
    new_scalars.gpu_data, new_generators.gpu_data, temp.gpu_data, temp0.gpu_data, temp1.gpu_data, 
    t.size, new_size);
    cudaDeviceSynchronize();
    proof.push_back(temp.sum());
    proof.push_back(temp0.sum());
    proof.push_back(temp1.sum());
    return me_open(new_scalars, new_generators, begin + 1, end, proof);
}



// ─── Host-side wrappers for DEVICE-only scalar / group arithmetic ─────
//
// Fr and G1Jacobian arithmetic in this codebase lives in DEVICE kernels
// (bls12-381.cu), so scalar ops on the host bounce through the size-1
// FrTensor / Commitment wrappers.  One round-trip per op is fine because
// the Figure 6 path touches O(1) scalars and O(1) single-point group ops;
// the O(n) work (Σ dᵢ·Gᵢ, c·x̂, etc.) goes through the bulk tensor ops.

static Fr_t fr_mul_host(Fr_t a, Fr_t b) {
    FrTensor A(1, &a);
    FrTensor B(1, &b);
    FrTensor C = A * B;
    return C(0);
}

static Fr_t fr_add_host(Fr_t a, Fr_t b) {
    FrTensor A(1, &a);
    FrTensor B(1, &b);
    FrTensor C = A + B;
    return C(0);
}

static G1Jacobian_t g1_scalar_mul_host(G1Jacobian_t P, Fr_t s) {
    Commitment pp1(1, P);
    FrTensor S(1, &s);
    G1TensorJacobian out = pp1.commit(S);  // Σ sᵢ·Pᵢ with |S| = |pp1| = 1
    return out(0);
}

static G1Jacobian_t g1_add_host(G1Jacobian_t a, G1Jacobian_t b) {
    G1TensorJacobian A(1, a);
    G1TensorJacobian B(1, b);
    G1TensorJacobian C = A + B;
    return C(0);
}

// Convert a vector<Fr_t> from the plain-form storage convention (as used
// by Fr_me / partial_me) to the Montgomery-form convention that
// G1_me_step expects on input.  G1_me_step unmonts its scalar argument
// before feeding it to G1Jacobian_mul (which consumes bits as plain);
// so to have com(u) evaluate the MLE at the same point u that
// partial_me(u, ...) evaluates at, we mont-convert u here.
static vector<Fr_t> mont_vec_host(const vector<Fr_t>& v) {
    if (v.empty()) return v;
    FrTensor t(v.size(), v.data());
    t.mont();
    vector<Fr_t> out;
    out.reserve(v.size());
    for (uint i = 0; i < v.size(); i++) out.push_back(t(i));
    return out;
}

// ─── Hyrax §A.2 Figure 6 opening (with §6.1 row reduction) ────────────

Commitment::OpeningResult Commitment::open_zk(
    const FrTensor& t,
    const FrTensor& row_blindings,
    const G1TensorJacobian& com,
    const vector<Fr_t>& u,
    Fr_t c) const
{
    if (!is_openable()) {
        throw std::runtime_error(
            "Commitment::open_zk: pp is not openable — hiding_generator "
            "or u_generator is identity (missing .h or .u sidecar)");
    }
    if (t.size != com.size * size) {
        throw std::runtime_error(
            "Commitment::open_zk: t.size must equal com.size * pp.size");
    }
    if (row_blindings.size != com.size) {
        throw std::runtime_error(
            "Commitment::open_zk: row_blindings.size != com.size");
    }
    uint log_rows = ceilLog2(com.size);
    if (u.size() < log_rows) {
        throw std::runtime_error("Commitment::open_zk: u too short");
    }
    vector<Fr_t> u_R(u.end() - log_rows, u.end());     // folds rows  (§6.1)
    vector<Fr_t> u_L(u.begin(), u.end() - log_rows);   // folds within-row

    // §6.1 reduction.  x̂ = M · ẽq(·, u_R) has length pp.size;
    // r_ξ = Σ ẽq(bits(i), u_R) · r_{ξ,i}.
    FrTensor x_hat = (com.size == 1) ? t : t.partial_me(u_R, size);
    Fr_t r_xi = (com.size == 1) ? row_blindings(0) : row_blindings(u_R);

    // Fresh masks (Figure 6 step 1 preamble; Hyrax p. 18: "P chooses
    // d⃗ ← F^n, r_δ, r_β ← F uniformly").
    FrTensor d = FrTensor::random(size);
    FrTensor r_scalars = FrTensor::random(3);  // [r_δ, r_β, r_τ]
    Fr_t r_delta = r_scalars(0);
    Fr_t r_beta  = r_scalars(1);
    Fr_t r_tau   = r_scalars(2);

    // δ = Σ dᵢ·Gᵢ + r_δ·H   (eq 11).  commit() here runs the bulk MSM.
    G1TensorJacobian d_com = this->commit(d);            // size 1
    G1Jacobian_t delta = g1_add_host(d_com(0),
                                     g1_scalar_mul_host(hiding_generator, r_delta));

    // v = ⟨x̂, â⟩ = x̂ evaluated as an MLE at u_L.  (Standard identity:
    // for â_j = ẽq(bits(j), u_L), ⟨x̂, â⟩ equals the multilinear
    // extension of x̂ at u_L.)
    Fr_t v = (u_L.empty()) ? x_hat(0) : x_hat(u_L);
    Fr_t ad_dot = (u_L.empty()) ? d(0) : d(u_L);  // ⟨â, d⃗⟩

    // β = ⟨â,d⃗⟩·U + r_β·H   (eq 12).
    G1Jacobian_t beta = g1_add_host(g1_scalar_mul_host(u_generator, ad_dot),
                                    g1_scalar_mul_host(hiding_generator, r_beta));

    // τ = v·U + r_τ·H.  Binds v (see OpeningProof doc).
    G1Jacobian_t tau = g1_add_host(g1_scalar_mul_host(u_generator, v),
                                   g1_scalar_mul_host(hiding_generator, r_tau));

    // Σ-protocol responses (Figure 6 step 3).
    FrTensor z = x_hat * c + d;   // c·x̂ + d⃗, both length pp.size
    Fr_t z_delta = fr_add_host(fr_mul_host(c, r_xi), r_delta);
    Fr_t z_beta  = fr_add_host(fr_mul_host(c, r_tau), r_beta);

    OpeningProof proof{
        delta, beta, tau,
        std::move(z), z_delta, z_beta,
        r_tau
    };
    return {std::move(proof), v};
}

bool Commitment::verify_zk(
    const G1TensorJacobian& com,
    const vector<Fr_t>& u,
    Fr_t v,
    const OpeningProof& proof,
    Fr_t c) const
{
    if (!is_openable()) {
        throw std::runtime_error(
            "Commitment::verify_zk: pp is not openable — hiding_generator "
            "or u_generator is identity");
    }
    if (proof.z.size != size) {
        throw std::runtime_error(
            "Commitment::verify_zk: proof.z.size != pp.size");
    }
    uint log_rows = ceilLog2(com.size);
    if (u.size() < log_rows) return false;
    vector<Fr_t> u_R(u.end() - log_rows, u.end());
    vector<Fr_t> u_L(u.begin(), u.end() - log_rows);

    // ξ = Σ ẽq(bits(i), u_R) · Cᵢ   (§6.1, verifier-recomputes).
    // G1TensorJacobian MLE eval (G1_me_step) unmonts its u argument
    // before multiplication; the Fr-side partial_me / Fr_me kernels
    // mont their u argument.  Pass u_R in Montgomery form here so the
    // G1 MLE evaluates at the same point u_R that the prover used
    // for the §6.1 fold.
    G1Jacobian_t xi = (com.size == 1) ? com(0) : com(mont_vec_host(u_R));

    // G1 points computed via different Jacobian paths can have the same
    // affine value but different (X:Y:Z) limbs.  Compare via A - B and
    // test whether the difference is the identity (Z == 0).
    auto g1_eq = [](G1Jacobian_t a, G1Jacobian_t b) {
        G1TensorJacobian A(1, a);
        G1TensorJacobian B(1, b);
        G1TensorJacobian D = A - B;
        G1Jacobian_t d = D(0);
        for (uint i = 0; i < blstrs__fp__Fp_LIMBS; i++) {
            if (d.z.val[i] != 0) return false;
        }
        return true;
    };

    // τ binding: recompute τ_expected = v·U + r_τ·H.
    G1Jacobian_t tau_expected = g1_add_host(
        g1_scalar_mul_host(u_generator, v),
        g1_scalar_mul_host(hiding_generator, proof.r_tau));
    if (!g1_eq(tau_expected, proof.tau)) return false;

    // eq 13:  c·ξ + δ  =?  Σ zᵢ·Gᵢ + z_δ·H.
    G1Jacobian_t lhs13 = g1_add_host(g1_scalar_mul_host(xi, c), proof.delta);
    G1TensorJacobian zG = this->commit(proof.z);
    G1Jacobian_t rhs13 = g1_add_host(zG(0),
                                     g1_scalar_mul_host(hiding_generator, proof.z_delta));
    if (!g1_eq(lhs13, rhs13)) return false;

    // eq 14:  c·τ + β  =?  ⟨z⃗,â⟩·U + z_β·H.
    Fr_t za_dot = (u_L.empty()) ? proof.z(0) : proof.z(u_L);
    G1Jacobian_t lhs14 = g1_add_host(g1_scalar_mul_host(proof.tau, c),
                                     proof.beta);
    G1Jacobian_t rhs14 = g1_add_host(
        g1_scalar_mul_host(u_generator, za_dot),
        g1_scalar_mul_host(hiding_generator, proof.z_beta));
    if (!g1_eq(lhs14, rhs14)) return false;

    return true;
}

Fr_t Commitment::open(const FrTensor& t, const G1TensorJacobian& com, const vector<Fr_t>& u) const
{
    const vector<Fr_t> u_out(u.end() - ceilLog2(com.size), u.end());
    const vector<Fr_t> u_in(u.begin(), u.end() - ceilLog2(com.size));
    auto g_temp = (com.size == 1)? com(0) : com(u_out);
    // if (size != (1 << u_in.size())) throw std::runtime_error("Incompatible dimensions");
    vector<G1Jacobian_t> proof;
    return me_open(t.partial_me(u_out, t.size / com.size), *this, u_in.begin(), u_in.end(), proof);
}

Weight create_weight(string generator_filename, string weight_filename,
                     string com_filename,
                     uint in_dim, uint out_dim) {
    Commitment generator(generator_filename);
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    G1TensorJacobian com(com_filename);
    return {generator, weight, com, FrTensor(0), in_dim, out_dim};
}

// Hiding create_weight: loads generators via the hiding-pp sidecar
// path (so `generator.hiding_generator` is populated) and additionally
// loads the per-row blinding tensor `r` from `r_filename`.
//
// Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//   "We say that Com_pp(m; r) is a commitment to the message m with
//    randomness r".
//
// The produced Weight satisfies `r.size == com.size` and
// `generator.is_hiding() == true`; these are the preconditions the
// Phase 2 blinded opening will rely on.
Weight create_weight(string generator_filename, string weight_filename,
                     string com_filename, string r_filename,
                     uint in_dim, uint out_dim) {
    Commitment generator = Commitment::load_hiding(generator_filename);
    if (!generator.is_hiding()) {
        throw std::runtime_error(
            "create_weight(hiding): generator pp is missing its H sidecar: "
            + generator_filename + ".h");
    }
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    G1TensorJacobian com(com_filename);
    FrTensor r(r_filename);
    if (r.size != com.size) {
        throw std::runtime_error(
            "create_weight(hiding): blinding tensor size " + std::to_string(r.size)
            + " does not match commitment row count " + std::to_string(com.size));
    }
    return {generator, weight, com, r, in_dim, out_dim};
}