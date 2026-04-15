#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "tensor/fr-tensor.cuh"
#include "tensor/g1-tensor.cuh"
#include "proof/proof.cuh"

// Pedersen commitment generators.
//
// Inherits G1TensorJacobian for the vector {G_i} of message generators.
// Adds `hiding_generator` H used for blinding:
//
//     Com(t; r) = sum_i t_i * G_i + r * H        (additive notation)
//
// Reference (anchoring the hiding property to the paper we cite):
//
//   Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//     "Perfect hiding: For any m_0, m_1 in {0,1}^lambda and m_0 != m_1:
//      {Com(m_0; r)}_{r<-R} and {Com(m_1; r)}_{r<-R} are identically
//      distributed."
//
// H is stored in a sidecar file next to the main pp file so the existing
// G1TensorJacobian save/load path stays untouched. H is identity
// (G1Jacobian_t zero) for a non-hiding Commitment produced by the legacy
// `random()` factory; use `hiding_random()` to get a Commitment with a
// real H.  Callers can check `is_hiding()` to distinguish the two.
class Commitment: public G1TensorJacobian
{
    public:
    using G1TensorJacobian::G1TensorJacobian;

    using G1TensorJacobian::operator+;
    using G1TensorJacobian::operator-;
    using G1TensorJacobian::operator*;
    using G1TensorJacobian::operator*=;

    // Additional generator for Pedersen hiding blinding.  Lives on the host
    // (single point, no reason to keep it on the GPU).  Initialized to the
    // G1 identity for legacy non-hiding Commitments.
    G1Jacobian_t hiding_generator = G1Jacobian_ZERO;

    bool is_hiding() const;

    G1TensorJacobian commit(const FrTensor& t) const;
    G1TensorJacobian commit_int (const FrTensor& t) const;
    G1TensorJacobian commit_int_multi(const vector<FrTensor>& t) const;

    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u) const;

    // Legacy non-hiding pp factory.  Kept for backwards compatibility with
    // existing test fixtures and binaries that don't need hiding.
    static Commitment random(uint size);

    // New hiding pp factory.  Samples the `size` message generators
    // {G_i = s_i * G} and additionally samples `H = s_H * G` where
    // s_H <- F_r uniformly.  All scalars are generated locally by whoever
    // runs this factory (same trusted-setup model as the non-hiding
    // version; zkLLM §3.4 assumes "trusted public parameters").
    static Commitment hiding_random(uint size);

    // Serialization for hiding pp.  Writes the G_i vector to `pp_file`
    // (inherited G1TensorJacobian layout, unchanged) and writes H to a
    // sidecar `pp_file + ".h"` file.  Loading from a path without an
    // ".h" sidecar leaves hiding_generator at identity.
    void save_hiding(const string& pp_file) const;
    static Commitment load_hiding(const string& pp_file);

    static Fr_t me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof);
};

struct Weight {
    Commitment generator;
    FrTensor weight;
    G1TensorJacobian com;
    uint in_dim;
    uint out_dim;
};

Weight create_weight(string generator_filename, string weight_filename, string com_filename, uint in_dim, uint out_dim);
// KERNEL void sum_axis_n_optimized(GLOBAL G1Jacobian_t* arr, GLOBAL G1Jacobian_t* arr_out, uint n, uint m);


KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u, // always assume that scalars and u is in mont form
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint old_size, uint new_size);

#endif