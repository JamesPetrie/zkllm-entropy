# Claims Convention

**Status:** v1, M1 (spec + walker + worked example on §2 Pedersen).
**Scope:** `docs/proof-layer-analysis.md` — the claim-graph overlay format,
the walker's semantics, and CI integration.

This document is the authoritative source for the claim-block syntax
and walker behaviour. The walker implementation is
`tools/check_claims.py`; if the code and this spec disagree, the spec
wins and the code gets fixed.

## Why

`docs/proof-layer-analysis.md` is a ~1500-line reference manual that
asserts per-layer properties: soundness, completeness, ZK, cost.
Phase 3 shipped over-claimed: the `zkFC` plain-sumcheck gap and the
Figure 6 final-reduction gap were *documented in prose* but were not
mechanically tied to the composed end-to-end ZK claim, so the
over-claim slipped through review.

This convention turns the reference manual into a claim graph:

- Every asserted property has a stable ID (e.g. `C-PED-HIDE`).
- Every claim declares what justifies it: a combinator over other
  claims, or a typed leaf.
- The walker backchains from headline claims (e.g. `C-END2END-ZK`)
  and reports transitive unjustified leaves. Any claim marked
  `justified` with an `UNJUSTIFIED` in its transitive support fails
  CI.

The convention is annotation-only: it does not rewrite the prose and
does not introduce a parallel docs tree. Claim blocks live inside
`docs/proof-layer-analysis.md` alongside the properties they describe.

## Block syntax

Each asserted property ends with a fenced block with the info string
`claim`:

    ```claim
    id: C-PED-HIDE
    statement: Pedersen commitment is perfectly hiding.
    justifiedBy:
      - PAPER(Hyrax Definition 4, Wahby et al. 2018 p.4)
      - CODE(src/commit/commitment.cuh + test/test_hiding_pedersen.cu)
    status: justified
    ```

A claim block is YAML-like but is parsed by a minimal line-oriented
parser (no YAML library dependency). Supported structure:

| Field | Required? | Meaning |
|---|---|---|
| `id` | yes | Stable immutable claim ID; see ID convention below. |
| `statement` | recommended | One-line English statement for `CLAIMS.md`. |
| `combinator` | internal claims only | One of `AND_OF`, `THEOREM`, `SEQUENTIAL_COMPOSITION`. Absence → leaf. |
| `justifiedBy` | yes | List of claim IDs (internal) or leaf tags (leaf). |
| `status` | yes | `justified` or `open`. |
| `superseded_by` | on retirement | Another claim ID that replaces this one. |
| `measurement` | perf claims only | Block with `hardware`, `commit`, `date`, `result`, `threshold`, `depends_on`. |

### Leaf tags

A leaf claim's `justifiedBy` list is a non-empty sequence of tag
invocations. There are exactly four leaf tags:

| Tag | Meaning | Example |
|---|---|---|
| `PAPER(<citation>)` | Backed by a direct quote from a cited paper | `PAPER(Hyrax Theorem 7, Wahby et al. 2018 Appendix A)` |
| `CODE(<file> + <test>)` | Backed by a specific implementation file plus a regression test | `CODE(src/commit/commitment.cuh + test/test_hiding_pedersen.cu)` |
| `ASSUMPTION(<statement>)` | Axiomatic relative to this work; community-accepted | `ASSUMPTION(DL hard in BLS12-381 G1 at λ=128)` |
| `UNJUSTIFIED(<reason>)` | Explicit todo; the property is not claimed yet | `UNJUSTIFIED(src/zknn/zkfc.cu:142-152 still uses plain zkip)` |

The distinction between `ASSUMPTION` and `UNJUSTIFIED` is
load-bearing. Collapsing them is the single largest failure mode of
informal audits, and was the proximate cause of Phase 3's over-claim.

### Combinators

Each internal (non-leaf) claim has exactly one combinator:

| Combinator | Semantics |
|---|---|
| `AND_OF` | Trivial conjunction — all subclaims must hold. |
| `THEOREM(<name>)` | The combination of subclaims implies the claim via a named theorem; the theorem itself must be a `PAPER` leaf (typically via a subclaim) or its own claim node. |
| `SEQUENTIAL_COMPOSITION` | Composes under a stated composition theorem. The composition theorem is itself referenced as a subclaim. |

`justifiedBy` for an internal claim is a list of **claim IDs only** —
leaf evidence cannot be mixed in. If a claim is "mostly subclaims
plus one paper quote," the paper quote gets its own sibling leaf
claim and the parent `AND_OF`s them. This is explicit to prevent the
"partial evidence dressed up as a leaf" failure mode.

### Status field

- `justified` — the author asserts this claim holds. The walker
  *checks* this by verifying no `UNJUSTIFIED` leaf is in the
  transitive support. A `justified` claim with an `UNJUSTIFIED`
  transitive leaf is a walker error.
- `open` — the author acknowledges this claim is not yet supported;
  expected to backchain to at least one `UNJUSTIFIED` leaf.

## ID convention

Pattern: `C-<LAYER>-<PROPERTY>` where

- `<LAYER>` is a short uppercase tag matching a section or
  subcomponent of `docs/proof-layer-analysis.md` (`PED`, `SIGMA`,
  `OPEN`, `SC`, `TLOOKUP`, `ZKFC`, `RESCALING`, `ARGMAX`,
  `NORMALCDF`, `LOG`, `ENTROPY`, `END2END`). Also `CRY` for generic
  crypto assumptions that aren't per-layer.
- `<PROPERTY>` is one of `COMPLETE`, `SOUND`, `BIND`, `HIDE`, `ZK`,
  `HVZK`, `HOMO`, `COST`, or a specific suffix (e.g.
  `C-OPEN-FINAL-REDUCTION`).

**IDs are immutable.** A claim that is substantively restructured
gets a new ID; the old ID is retired with `status: open` and
`superseded_by: <new-id>`. The walker warns on references to
superseded IDs from live claims.

## Walker behaviour

`tools/check_claims.py` reads `docs/proof-layer-analysis.md`, builds
the claim graph, validates it, and reports.

### Validation rules

1. Every `id` is unique.
2. Every claim ID referenced in `justifiedBy` resolves to a defined
   claim.
3. No cycles in the `justifiedBy` graph.
4. Every leaf has exactly one type of tag per entry, chosen from the
   four leaf tags. (Multiple leaf entries per claim are allowed —
   they `AND` implicitly: all must be valid.)
5. Every `CODE(<file> + <test>)` leaf names files that exist (soft
   check — line numbers are advisory, the test must exist because it
   is a regression test).
6. No live claim has `superseded_by` set to a live claim ID.
7. For perf claims with `measurement`: if any path in `depends_on`
   has changed since `commit` (via `git log <commit>..HEAD --
   <paths>`), the claim is demoted to `UNJUSTIFIED` with reason
   "stale: <path> changed in <sha>."

### Propagation

A claim is **blocked** if any transitive leaf in its `justifiedBy`
DAG is `UNJUSTIFIED` (including ones demoted from stale perf claims).

A `justified` claim that is `blocked` is a walker error (exit
non-zero).

An `open` claim is expected to be `blocked`. The walker does not
error on `open + blocked`; it only errors on `justified + blocked`.

### Report

The walker prints:

1. A summary: total claims, justified, open, leaves, UNJUSTIFIED
   leaves.
2. For each `UNJUSTIFIED` leaf: its reason and the set of upstream
   claims it transitively blocks (the "unjustified-upstream"
   report).
3. Any validation errors (unresolved IDs, cycles, bad leaf tags,
   missing CODE files, superseded-references, `justified` claims
   with `UNJUSTIFIED` transitive leaves).

Exit code: `0` on clean, non-zero on any error from (3).

### `CLAIMS.md` generation

`--emit-claims-md` regenerates `docs/CLAIMS.md`: a flat index with
ID, section anchor, one-line statement, and status (justified /
open / blocked).

## Invocation

```
python3 tools/check_claims.py             # validate + report, exit non-zero on errors
python3 tools/check_claims.py --emit-claims-md  # regenerate docs/CLAIMS.md
make check-claims                          # CI entry point
```

## CI integration

`make check-claims` is wired as a required CI check. A failing
walker (any validation error or any `justified + blocked` claim)
fails the build.

## Maintenance rule

Any commit that changes proof-affecting code (under `src/` or under
`docs/proof-layer-analysis.md`) must also update the relevant claim
block(s) in the same commit. This keeps the claim graph in lockstep
with the code.

Two specific guardrails follow from this rule:

- Adding an `UNJUSTIFIED` leaf is always allowed — honest admission.
- Promoting an `UNJUSTIFIED` leaf to `PAPER` / `CODE` / `ASSUMPTION`
  is a claim upgrade and should be reviewable in its own diff.

## Non-goals

- **No formal-proof-assistant export.** This is informal markdown
  with a lightweight walker, not Lean/Coq.
- **No automated PAPER-quote verification.** Spot-checked by audits.
- **No performance-benchmark harness.** `depends_on` tracks
  in-tree drift; out-of-tree drift (CUDA driver, firmware, toolchain)
  is not tracked — benchmarks are re-run when discrepancies are
  noticed.
