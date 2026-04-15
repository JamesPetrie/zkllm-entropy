# Phase 1.5 — Hash-to-Curve Generators (Remove Toxic Waste)

**Date:** 2026-04-15
**Status:** Draft / proposed
**Branch:** `phase-1.5-hash-to-curve` (to be created)
**Parent plan:** `docs/plans/plan-production-readiness.md`
**Estimated effort:** 1–2 days

## Goal

Derive every Pedersen generator — `{Gᵢ}`, `H`, `U` — from a public
domain-separation tag via RFC 9380 hash-to-curve, so that nobody
(including whoever ran `ppgen`) knows any pairwise discrete log. This
closes the "toxic waste" gap flagged in the Phase 1 alternative-
considered note and brings the implementation into line with what
Hyrax actually assumes.

> **Wahby et al. 2018, eprint 2017/1132, §3.1 (p. 4):** the Pedersen
> commitment's binding property rests on the discrete-log assumption
> applied to generators with no known relations. Hyrax's §A.1 Fig. 5
> and §A.2 Fig. 6 soundness reductions both rely on this.

> **zkLLM (Sun, Li, Zhang 2024) §3.4 (p. 5):** "Hyrax … is used as an
> instantiation of the polynomial commitment scheme." — zkLLM inherits
> Hyrax's generator-independence assumption.

The current codebase samples `Gᵢ = sᵢ · G_base` and `H = s_H · G_base`,
`U = s_U · G_base` with fresh random scalars (`src/commit/commitment.cu:25`
onward, `hiding_random`). Whoever ran `ppgen` knows every scalar and
can therefore equivocate on commitments. This is not the trust model
Hyrax describes; it's a research-prototype shortcut inherited from the
upstream zkLLM implementation.

## Non-goals

- No change to `open_zk` / `verify_zk` / `commit` / `commit_hiding` /
  the MSM kernels. Generators are opaque G1 points to all of them.
- No change to the field arithmetic or GPU kernels at all.
  Hash-to-curve is a host-side, setup-time operation.
- No MPC / ceremony tooling. The whole point of hash-to-curve is that
  no ceremony is needed.
- No change to Fiat-Shamir, transcript hashing, or any challenge
  derivation (those are SHA-256 → F_r, not hash-to-curve).

## Approach: blst for hash-to-curve

Use the audited `blst` C library's `blst_hash_to_g1` (RFC 9380 suite
`BLS12381G1_XMD:SHA-256_SSWU_RO_`) as a host-side dependency.
Wrap its output into the codebase's `G1Jacobian_t` representation and
expose a single helper:

```cpp
// src/field/hash_to_curve.cuh
G1Jacobian_t hash_to_curve_g1(const std::string& dst,
                              const std::string& msg);
```

**Why blst over a custom CUDA port:**

- blst's `blst_hash_to_g1` is ~0 LOC of new crypto for us — the code
  is already written, audited, and battle-tested in production
  (filecoin, eth2, many others).
- Hash-to-curve is a setup-time, host-side operation. No GPU kernel
  ever calls it. So a host C library is a strict superset of what
  we need — we're not giving anything up by not porting to CUDA.
- A custom SSWU+isogeny implementation is ~400–600 LOC with subtle
  edge cases (sqrt branch, `xi` choice for the 11-isogeny, iso_map
  constants, cofactor clearing) where bugs produce "valid G1 points
  computed wrong." RFC 9380 Appendix J vectors catch those, but the
  review burden is real — and would consume crypto-review budget on
  something that's already a commodity.

**Why `blst` specifically (not blstrs, not arkworks):**

- Pure C, single-call API for our use case, no C++ runtime.
- Well-packaged in `pkg-config` and conda-forge (`blst` package).
- Header-compatible with the rest of the C++/CUDA build.

### Domain separation tag

```
DST = "ZKLLM-ENTROPY-PEDERSEN-V1_BLS12381G1_XMD:SHA-256_SSWU_RO_"
```

Follows RFC 9380 §3.1 DST construction: application-specific prefix
(`ZKLLM-ENTROPY-PEDERSEN-V1`) + suite identifier. Any change to the
generator derivation (even a bugfix) MUST bump this to `V2` — this is
a hard forward-compat boundary and a code comment on the constant
should say so explicitly.

### Generator derivation

```
G_i = hash_to_curve_g1(DST, "G_" || uint32_be(i))   for i in 0..size
H   = hash_to_curve_g1(DST, "H")
U   = hash_to_curve_g1(DST, "U")
```

Deterministic function of `(DST, size)`. No RNG, no scalars stored,
no secret anywhere in `ppgen`'s execution.

### pp file format (cached + verify-on-load)

Store the generators as today (fast prover startup, no per-load
recomputation), plus the DST. On `load_hiding`, recompute generators
from DST and byte-compare against what the file contains; reject on
mismatch.

```
magic     : 8 bytes  = "ZKEPP\x00v2"
version   : uint32   = 2
flags     : uint32   = bit 0 ⇒ hiding (H present)
                       bit 1 ⇒ openable (U present)
                       bit 2 ⇒ htc-derived (DST present; v2 requires this)
dst_len   : uint32
dst       : dst_len bytes
size      : uint32
G_i       : size * sizeof(G1Jacobian_t)
H         : sizeof(G1Jacobian_t)    (if hiding)
U         : sizeof(G1Jacobian_t)    (if openable)
```

Phase 1 shipped `v1` (no DST, scalars-from-RNG). Phase 1.5 is `v2`.
The v1 read path stays for backward compatibility *but rejects any
call that tries to use it for `open_zk`* — i.e. the ZK pipeline
refuses non-htc pp files. Non-ZK legacy paths (`verifyWeightClaim`
without `ZK`) keep reading v1 until Phase 3 retires them.

## Files to add / modify

1. **`src/field/hash_to_curve.cuh`, `src/field/hash_to_curve.cu`** (new):
   blst wrapper. `hash_to_curve_g1(dst, msg) → G1Jacobian_t`. The
   wrapper converts from blst's internal point representation to the
   codebase's `G1Jacobian_t` layout.

2. **`src/commit/commitment.cu`**: rewrite `Commitment::hiding_random`
   to derive `{Gᵢ}`, `H`, `U` via `hash_to_curve_g1`. Delete the
   `FrTensor::random` calls for generator scalars. Add
   `Commitment::verify_pp(const std::string& dst) const` that
   recomputes and byte-compares. Legacy `Commitment::random` stays but
   is marked deprecated.

3. **`src/commit/commitment.cu`**: update `save_hiding` / `load_hiding`
   to v2 format. `load_hiding` calls `verify_pp` unconditionally on
   v2 files; on mismatch, throw with the exact index that differs (so
   a truncation or single-point corruption is diagnosable).

4. **`bin/ppgen.cu`**: call the new `hiding_random` (no code change
   needed if the factory signature stays the same); writes v2 format.

5. **`Makefile`**: add `-lblst` to link flags. Add `INCLUDES +=
   -I$(CONDA_PREFIX)/include/blst` (conda install path) or whatever
   the `pkg-config --cflags libblst` route resolves to. Add blst as
   a prereq in the build instructions.

6. **`CLAUDE.md` / README / build docs**: document the new dependency
   (`apt install libblst-dev` or `conda install -c conda-forge blst`).

## Tests

### Positive

1. **`test_hash_to_curve_rfc9380`**: RFC 9380 Appendix J.9.1
   test-vector harness. Hard-code the spec's `(msg, DST) →
   (P.x, P.y)` tuples and byte-compare. Non-negotiable: this is
   the gate that says "we're really implementing RFC 9380 and not
   some byte-swapped variant." The RFC ships ~10 vectors for
   `BLS12381G1_XMD:SHA-256_SSWU_RO_`.

2. **`test_htc_generators_deterministic`**: `hiding_random(size)`
   called twice returns byte-identical `{Gᵢ, H, U}`. Guards against
   an accidental RNG-dependency creeping back in.

3. **`test_htc_generators_distinct`**: for `size = 2^15`, assert
   `Gᵢ ≠ Gⱼ` for `i ≠ j`, and `Gᵢ ≠ H ≠ U ≠ Gᵢ`. Birthday bound is
   `2^(-111)`; a collision signals a domain-separation bug.

4. **`test_pp_verify_accepts_honest`**: `save_hiding` then
   `load_hiding` round-trips; `load_hiding` returns without throwing
   and `verify_pp(DST)` returns true.

### Negative

5. **`test_pp_verify_rejects_tampered_G`**: write a v2 file, flip one
   byte in `G_5`, reload → `load_hiding` throws with a message
   naming index 5.

6. **`test_pp_verify_rejects_wrong_DST`**: write with DST "V1", load
   while demanding DST "V2" → throws.

7. **`test_open_zk_rejects_v1_pp`**: load a v1 pp (Phase 1 format)
   and call `open_zk` → throws with a clear "pp is not htc-derived"
   message.

### Regression

8. All Phase 1 hiding-Pedersen tests pass against v2 pp (regenerate
   fixtures, re-run `test_hiding_pedersen`).

9. All Phase 2 tests pass against v2 pp (`test_open_zk`,
   `test_verify_weight_zk`, `test_opening_distinguisher`).

10. End-to-end `zkllm_entropy` dry run with a tiny model, v2 pp,
    confirms all `verifyWeightClaim` / `verifyWeightClaimZK` calls
    succeed. This is the "did we break anything downstream" gate.

## Migration plan for existing pp files

Every v1 pp file in the tree becomes inert for the ZK pipeline once
Phase 1.5 lands. Steps:

1. **Identify all pp-writing paths.** Known set: `bin/ppgen.cu`,
   `bin/commit-param.cu`. Grep for `save_hiding` / `save`.
2. **Identify all pp fixtures in tests.** If any test loads from a
   pre-committed pp blob (vs. generating in-memory), it needs
   regeneration.
3. **Add a one-shot regeneration script** `scripts/regen_pp.sh`:
   iterates over pp files, reads size, calls the new `ppgen` with
   v2 format + DST. Commits the regenerated files.
4. **CI gate:** a lint step that rejects any committed pp file whose
   magic is not `ZKEPP\x00v2`, so we can't accidentally re-introduce
   a v1 file.

## Risks

1. **blst API mismatch.** `blst_hash_to_g1` returns a `blst_p1` in
   Jacobian coordinates with blst's internal limb layout; the
   codebase's `G1Jacobian_t` has a specific limb order that may or
   may not match. Mitigation: the wrapper test (test #1, RFC 9380
   vectors) catches any layout mismatch immediately because it
   compares against paper-specified `(x, y)` coordinates.

2. **DST versioning drift.** If a future change to the generator
   derivation (e.g. a bugfix in the wrapper) silently keeps the v1
   DST, pp files generated before and after that change collide in
   filename but not content, and `verify_pp` fires. Mitigation: the
   DST constant carries a comment "ANY change to generator derivation
   bumps this" and a changelog entry in the header. Consider
   encoding the DST in the file's hex-dump comment for human-
   inspection sanity.

3. **Build-system brittleness from adding blst.** CUDA + C + new lib
   = new link-error surface. Mitigation: pin the blst version
   (conda-forge `blst=0.3.11` or whatever is current) in CI, and
   document the exact install command in the build instructions.

4. **Performance regression on ppgen.** `hiding_random(2^15)` goes
   from ~N scalar-mults to ~N hash-to-curve calls. Per-generator
   cost is ~100k CPU cycles for hash-to-curve vs ~80k for scalar
   mult; net overhead ~25% on a one-off call that already takes ~1s.
   Acceptable. If it's not, `hash_to_curve_g1` parallelizes trivially
   across cores.

5. **blst not available in the jpetrieamodo environment.** Ubuntu
   24.04 ships `libblst-dev` in `universe`; if that's not installed,
   `apt install libblst-dev` is a one-liner. conda-forge has `blst`
   as a fallback.

## Strengthens prior work

The Phase 2 `r_τ` value-binding mechanism (`τ = v·U + r_τ·H` with
`r_τ` revealed) relies on `U`, `H`, and the `{Gᵢ}` having no known
dlog relations. Under the current `sᵢ·G` scheme that holds only
*relative to someone who doesn't know the sᵢ*. Under hash-to-curve
it holds unconditionally. So Phase 1.5 upgrades the Phase 2 soundness
argument from "assuming trusted setup" to "assuming hash-to-curve +
DL" — which is strictly what Hyrax claims and what zkLLM cites.

## Out of scope

- **Phase 5+ full verifier integrity check.** The verifier also needs
  to validate pp integrity on its side; that's natural to fold into
  verifier bring-up once the ZK pipeline and Fiat-Shamir are wired.
  Phase 1.5 delivers the primitive (`verify_pp`); Phase 5+ wires it
  into the verifier's load path.
- **Key-committing AEAD for pp distribution.** If pp files are
  shipped over a network, integrity + authenticity are table stakes;
  but Phase 1.5 already makes integrity publicly recomputable (any
  receiver can run `verify_pp` without trusting the sender), so this
  is belt-and-suspenders.
- **MPC ceremony.** With hash-to-curve there's no ceremony needed for
  vector generators. If a future pairing-based extension requires
  powers-of-tau, that's a separate setup and a separate plan.

## References

- **RFC 9380** (Faz-Hernández et al. 2023), "Hashing to Elliptic
  Curves." Appendix J.9.1 has the BLS12-381 G1 test vectors we gate
  on. §3.1 has the DST construction.
- **blst** C library: https://github.com/supranational/blst . API
  docs: `bindings/blst.h`. `blst_hash_to_g1` at a glance:
  ```c
  void blst_hash_to_g1(blst_p1 *out,
                       const byte *msg, size_t msg_len,
                       const byte *DST, size_t DST_len,
                       const byte *aug, size_t aug_len);
  ```
- **Hyrax** (Wahby et al. 2018, eprint 2017/1132), §3.1, §A.1, §A.2
  — the generator-independence assumption Phase 1.5 honors.
- **Phase 1 plan** `docs/plans/phase-1-hiding-pedersen.md` lines
  101–111 — the "alternative considered and rejected" note that
  flagged this as deferred work.
