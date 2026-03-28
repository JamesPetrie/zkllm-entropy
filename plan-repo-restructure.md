# Repo Restructuring Plan

**Status:** Proposal only — no files have been moved yet.
**Date:** 2026-03-27

## Problem

The repo currently has ~50 `.cu`, `.cpp`, and `.h` files in a single flat directory.
This makes it hard to understand the architecture, find related files, or onboard
new contributors. Files serving very different purposes (field arithmetic, neural
network layers, benchmarks, tests, build targets) are all interleaved.

## Proposed Directory Structure

```
zkllm-entropy/
├── src/
│   ├── field/              # Finite field arithmetic
│   │   ├── bls12-381.cu    #   BLS12-381 field & curve ops
│   │   ├── bls12-381.cuh
│   │   ├── goldilocks.cu   #   Goldilocks field (p = 2^64 - 2^32 + 1)
│   │   └── goldilocks.cuh
│   │
│   ├── poly/               # Polynomial operations
│   │   ├── polynomial.cu   #   Polynomial evaluation, interpolation
│   │   ├── polynomial.cuh
│   │   ├── ntt.cu          #   Number Theoretic Transform
│   │   └── ntt.cuh
│   │
│   ├── commit/             # Commitment schemes
│   │   ├── commitment.cu   #   KZG / Pedersen commitments (BLS12-381)
│   │   ├── commitment.cuh
│   │   ├── merkle.cu       #   SHA-256 Merkle tree (Goldilocks/FRI)
│   │   ├── merkle.cuh
│   │   ├── fri.cu          #   FRI low-degree testing
│   │   ├── fri.cuh
│   │   ├── fri_pcs.cu      #   FRI polynomial commitment scheme
│   │   └── fri_pcs.cuh
│   │
│   ├── proof/              # Proof infrastructure
│   │   ├── proof.cu        #   Sumcheck protocols (ip, hp, bin)
│   │   └── proof.cuh
│   │
│   ├── tensor/             # GPU tensor types
│   │   ├── fr-tensor.cu    #   Field-element tensor
│   │   ├── fr-tensor.cuh
│   │   ├── g1-tensor.cu    #   G1 group-element tensor (BLS12-381)
│   │   └── g1-tensor.cuh
│   │
│   ├── zknn/               # ZK neural-network layer proofs
│   │   ├── zkrelu.cu       #   ReLU
│   │   ├── zkfc.cu         #   Fully connected
│   │   ├── zksoftmax.cu    #   Softmax
│   │   ├── zkargmax.cu     #   Argmax
│   │   ├── zklog.cu        #   Log lookup
│   │   ├── zknormalcdf.cu  #   Normal CDF lookup
│   │   ├── rescaling.cu    #   Fixed-point rescaling
│   │   └── tlookup.cu      #   Table lookup (LogUp argument)
│   │
│   ├── entropy/            # Entropy-specific proving logic
│   │   ├── zkentropy.cu    #   Entropy proof generation
│   │   └── zkentropy.cuh
│   │
│   ├── llm/                # LLM layer proofs
│   │   ├── self-attn.cu
│   │   ├── ffn.cu
│   │   ├── rmsnorm.cu
│   │   └── skip-connection.cu
│   │
│   └── util/               # Shared utilities
│       ├── ioutils.cu      #   File I/O helpers
│       ├── ioutils.cuh
│       ├── timer.cpp        #   Wall-clock timer
│       └── timer.h
│
├── verifier/               # CPU-only verifier (no CUDA dependency)
│   ├── verifier.cpp        #   Main verifier entry point
│   ├── verifier_utils.h    #   Field arithmetic, parsing, SHA-256
│   ├── sumcheck_verifier.h #   Sumcheck protocol verification
│   └── tlookup_verifier.h  #   tLookup verification
│
├── bin/                    # Build-target entry points (main() functions)
│   ├── main.cu
│   ├── ppgen.cu
│   ├── commit-param.cu
│   ├── commit_logits.cu
│   ├── zkllm_entropy.cu
│   ├── zkllm_entropy_timed.cu
│   └── layer_server.cu
│
├── test/                   # Test programs
│   ├── test_goldilocks.cu
│   ├── test_gold_tensor.cu
│   ├── test_ntt.cu
│   ├── test_merkle.cu
│   ├── test_fri.cu
│   ├── test_fri_pcs.cu
│   ├── test_zkargmax.cu
│   ├── test_zklog.cu
│   ├── test_zknormalcdf.cu
│   ├── test_zkentropy.cu
│   └── test_verifier.cpp   #   CPU-only verifier tests
│
├── bench/                  # Benchmark programs
│   ├── bench_field.cu
│   ├── bench_field_arith.cu
│   ├── bench_commitment.cu
│   └── bench_matmul.cu
│
├── scripts/                # Build and run scripts
│   └── ...
│
├── python/                 # Python verification / analysis tools
│   ├── verify_entropy.py
│   └── ...
│
├── docs/                   # Documentation and plans
│   ├── security-review.md
│   ├── plan-full-verifier.md
│   ├── plan-fp16-weights.md
│   └── plan-repo-restructure.md  # (this file)
│
├── Makefile
└── README.md
```

## Rationale

| Concern | Current state | After restructure |
|---------|--------------|-------------------|
| Finding related code | Scroll through 50+ files | Navigate to the relevant `src/` subdirectory |
| Understanding architecture | Read file names and guess | Directory names map to architectural layers |
| Build targets vs libraries | Mixed together | `bin/` for executables, `src/` for libraries |
| Tests and benchmarks | Mixed with source | `test/` and `bench/` separated |
| CPU verifier | Mixed with GPU code | `verifier/` stands alone, no CUDA dependency |
| Documentation | Top-level clutter | `docs/` directory |

## Key Design Decisions

1. **`src/` subdirectories by architectural layer**, not by file type. Field arithmetic,
   polynomial ops, commitments, proofs, and neural network layers each get their own
   directory. This mirrors how developers think about the system.

2. **`verifier/` is separate from `src/`** because it has zero CUDA dependency and
   can be built with just `g++`. Keeping it outside `src/` makes this independence
   visible.

3. **`bin/` for entry points** — files whose sole purpose is providing a `main()`
   function. This separates "what can be built" from "reusable library code."

4. **Goldilocks vs BLS12-381 variants** of the same file stay in the same directory.
   The `#ifdef USE_GOLDILOCKS` / `gold_` prefix pattern continues to work — the
   Makefile just needs updated paths.

5. **Header files stay next to their `.cu` files.** No separate `include/` directory,
   since most headers are tightly coupled to a single `.cu` file.

## Migration Notes

- **Makefile must be updated** to reference new paths. The pattern rules (`%.o: %.cu`)
  will need `VPATH` or explicit per-directory rules.
- **`#include` paths** will change. Use `-I src/` in compiler flags so includes like
  `#include "field/goldilocks.cuh"` work.
- **Do this in one atomic PR** so there's no half-migrated state.
- **Run full test suite after migration** — `make -j64 all` plus all `test_*` and
  `gold_*` targets.
- **Coordinate with other contributors** — check that no one has in-flight branches
  that would be disrupted by mass file moves.

## What NOT to Do

- Don't move files piecemeal across multiple PRs — that creates merge conflicts.
- Don't rename files during the move (keep `fr-tensor.cu`, not `field_tensor.cu`).
  Renames can happen in a follow-up PR if desired.
- Don't change any code during the move — the diff should be purely path changes
  and Makefile updates.
- Don't restructure until active feature branches (`goldilocks-fri`, etc.) are
  merged or coordinated, to avoid painful rebases.
