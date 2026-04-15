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

    if (!out.is_hiding()) {
        throw std::runtime_error(
            "Commitment::hiding_random: sampled H is the G1 identity; "
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

Commitment::HidingCommit Commitment::commit_int_multi_hiding(const vector<FrTensor>& ts) const
{
    uint num_row = 0;
    for (auto& t : ts) {
        if (t.size % size != 0) throw std::runtime_error("Commitment::commit_int_multi_hiding - Incompatible dimensions");
        num_row += t.size / size;
    }
    G1TensorJacobian base = this->commit_int_multi(ts);
    return add_blinding(*this, std::move(base), num_row);
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