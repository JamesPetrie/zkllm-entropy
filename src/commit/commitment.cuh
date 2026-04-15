#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "tensor/fr-tensor.cuh"
#include "tensor/g1-tensor.cuh"
#include "proof/proof.cuh"

// Transcript of a Hyrax §A.2 Figure 6 `proof-of-dot-prod` opening,
// composed with the §6.1 matrix-layout reduction so that a single
// OpeningProof attests to a multilinear-polynomial evaluation.
//
// Field names map to Hyrax 2017/1132 Figure 6 verbatim.  Hyrax p. 18:
//   "δ ← Com_{g⃗}(d⃗; r_δ), β ← Com(⟨â,d⃗⟩; r_β)" (step 1; eqs 11, 12).
//   "z⃗ ← c·x̂ + d⃗, z_δ ← c·r_ξ + r_δ, z_β ← c·r_τ + r_β" (step 3).
//
// `tau` commits the claimed evaluation `v = f̃(u)` with fresh blinding
// `r_tau`.  Phase 2's interactive-verifier model sends `r_tau` alongside
// the Σ-protocol responses so the verifier can check `τ = v·U + r_tau·H`,
// binding the Figure-6 protocol's internal y to the public v.  The r_tau
// field deviates from the Phase-2 plan's initial struct listing; see
// docs/plans/phase-2-blinded-opening.md "Value binding" (added after
// implementation revealed Figure 6 alone does not bind τ to a public v).
struct OpeningProof {
    // Figure 6 step 1 (P → V), eqs 11 & 12.
    G1Jacobian_t delta;       // δ = h^{r_δ} ⊙ Π gᵢ^{dᵢ}
    G1Jacobian_t beta;        // β = g^{⟨â,d⃗⟩} ⊙ h^{r_β}

    // Commitment to the claimed evaluation.
    G1Jacobian_t tau;         // τ = g^v ⊙ h^{r_τ}

    // Figure 6 step 3 (P → V): n + 2 field elements.
    FrTensor     z;           // z⃗ = c·x̂ + d⃗   (length pp.size)
    Fr_t         z_delta;     // z_δ = c·r_ξ + r_δ
    Fr_t         z_beta;      // z_β = c·r_τ + r_β

    // Extra field for binding τ to the public evaluation value v.
    Fr_t         r_tau;       // r_τ revealed so V recomputes τ = v·U + r_τ·H
};

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

    // Additional generator used as the inner-product target in the Hyrax
    // §A.2 Figure 6 ZK dot-product protocol (the "g" in
    // "τ = g^{y} ⊙ h^{r_τ}", Hyrax p. 17).  Distinct from both the message
    // generators {G_i} and the hiding generator H.  Identity for legacy
    // non-hiding Commitments; populated by hiding_random() and persisted
    // in the `.u` sidecar.
    G1Jacobian_t u_generator = G1Jacobian_ZERO;

    bool is_hiding() const;
    // True iff both hiding_generator (H) and u_generator (U) are non-identity;
    // i.e. this pp is usable by the Figure 6 open_zk / verify_zk path.
    bool is_openable() const;

    G1TensorJacobian commit(const FrTensor& t) const;
    G1TensorJacobian commit_int (const FrTensor& t) const;
    G1TensorJacobian commit_int_multi(const vector<FrTensor>& t) const;

    // Hiding variants of commit / commit_int / commit_int_multi.
    //
    // Each committed row gets its own fresh blinding scalar r_row; the
    // returned pair holds the row-wise commitment tensor C together with
    // the FrTensor r of per-row blindings.  C[row] = sum_j G_j * t[row,j]
    // + r[row] * H.
    //
    // Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
    //   "We say that Com_pp(m; r) is a commitment to the message m with
    //    randomness r, and sometimes do the same for the opening".
    //
    // These methods require is_hiding() == true; they throw otherwise
    // to prevent silent loss of the hiding property.
    struct HidingCommit {
        G1TensorJacobian com;
        FrTensor r;
    };
    HidingCommit commit_hiding(const FrTensor& t) const;
    HidingCommit commit_int_hiding(const FrTensor& t) const;

    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u) const;

    // ZK opening per Hyrax §A.2 Figure 6 composed with §6.1 row
    // reduction.  Produces a transcript that reveals nothing about the
    // committed witness beyond the public evaluation value.  Requires
    // `is_openable() == true` (hiding_generator and u_generator both
    // populated); throws otherwise.
    //
    //   t            length == row_blindings.size * size (padded)
    //   row_blindings length == com.size (the per-row r from Phase 1)
    //   com          the row commitments {C_i} (Phase 1 output)
    //   u            evaluation point; last ceilLog2(com.size) coords
    //                fold rows, leading coords fold within row
    //   c            verifier-supplied challenge (Figure 6 step 2)
    struct OpeningResult { OpeningProof proof; Fr_t v; };
    OpeningResult open_zk(const FrTensor& t,
                          const FrTensor& row_blindings,
                          const G1TensorJacobian& com,
                          const vector<Fr_t>& u,
                          Fr_t c) const;

    bool verify_zk(const G1TensorJacobian& com,
                   const vector<Fr_t>& u,
                   Fr_t v,
                   const OpeningProof& proof,
                   Fr_t c) const;

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

// A committed weight tensor.
//
// `r` holds the per-row blinding scalars that were used when `com` was
// produced via Commitment::commit_int_hiding / commit_hiding.  For
// legacy non-hiding commitments produced by the 5-arg create_weight
// overload, `r` is empty (size == 0); downstream Phase 2 consumers
// that require hiding must check `r.size == com.size` and error out
// otherwise.
//
// Hyrax §3.1 (Wahby et al. 2018, eprint 2017/1132, p. 4):
//   "We say that Com_pp(m; r) is a commitment to the message m with
//    randomness r".
struct Weight {
    Commitment generator;
    FrTensor weight;
    G1TensorJacobian com;
    FrTensor r;
    uint in_dim;
    uint out_dim;
};

// Legacy non-hiding create_weight.  Produces a Weight with an empty
// `r` tensor (size 0).  Kept so call sites that haven't migrated to
// the hiding pipeline (the per-layer proofs in src/llm/*.cu) keep
// compiling and behaving identically.
Weight create_weight(string generator_filename, string weight_filename,
                     string com_filename,
                     uint in_dim, uint out_dim);

// Hiding create_weight: additionally loads the per-row blinding
// tensor `r` from `r_filename`.  The file must contain exactly
// (in_dim_rows) Fr_t elements — one blinding per committed row — in
// the on-disk format produced by FrTensor::save.  Throws if the file
// is missing or the wrong size.  Callers who want a hiding commitment
// must use this overload.
Weight create_weight(string generator_filename, string weight_filename,
                     string com_filename, string r_filename,
                     uint in_dim, uint out_dim);
// KERNEL void sum_axis_n_optimized(GLOBAL G1Jacobian_t* arr, GLOBAL G1Jacobian_t* arr_out, uint n, uint m);


KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators, Fr_t u, // always assume that scalars and u is in mont form
    GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
    GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0, GLOBAL G1Jacobian_t* temp_out1, 
    uint old_size, uint new_size);

#endif