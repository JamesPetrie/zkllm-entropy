# Repo Restructuring Plan

**Status:** Proposal only — no files have been moved yet.
**Date:** 2026-03-28 (updated)

## Problem

The repo currently has 70+ source files (`.cu`, `.cpp`, `.cuh`, `.h`) plus
20+ Python scripts, 10+ shell scripts, and 15+ documentation files — all in a
single flat directory. This makes it hard to understand the architecture, find
related files, or onboard new contributors.

## Current File Inventory

Source files by category (as of 2026-03-28):

- **Field arithmetic (4):** `bls12-381.cu/.cuh`, `goldilocks.cu/.cuh`
- **Polynomial (4):** `polynomial.cu/.cuh`, `ntt.cu/.cuh`
- **Commitment (8):** `commitment.cu/.cuh`, `merkle.cu/.cuh`, `fri.cu/.cuh`, `fri_pcs.cu/.cuh`
- **Proof infra (2):** `proof.cu/.cuh`
- **Tensors (4):** `fr-tensor.cu/.cuh`, `g1-tensor.cu/.cuh`
- **ZK neural-net layers (16):** `zkrelu.cu/.cuh`, `zkfc.cu/.cuh`, `zksoftmax.cu/.cuh`,
  `zkargmax.cu/.cuh`, `zklog.cu/.cuh`, `zknormalcdf.cu/.cuh`, `rescaling.cu/.cuh`,
  `tlookup.cu/.cuh`
- **Entropy (2):** `zkentropy.cu/.cuh`
- **LLM layers (5):** `self-attn.cu`, `ffn.cu`, `rmsnorm.cu`, `rmsnorm_linear.cu`,
  `post_attn.cu`, `skip-connection.cu`
- **Utilities (3):** `ioutils.cu/.cuh`, `timer.cpp/.hpp`
- **CPU verifier (4):** `verifier.cpp`, `verifier_utils.h`, `sumcheck_verifier.h`,
  `tlookup_verifier.h`
- **CPU support (1):** `skip_connection_cpu.cpp`
- **Entry points (7):** `main.cu`, `ppgen.cu`, `commit-param.cu`, `commit_logits.cu`,
  `zkllm_entropy.cu`, `zkllm_entropy_timed.cu`, `layer_server.cu`
- **Tests (10):** `test_goldilocks.cu`, `test_gold_tensor.cu`, `test_ntt.cu`,
  `test_merkle.cu`, `test_fri.cu`, `test_fri_pcs.cu`, `test_zkargmax.cu`,
  `test_zklog.cu`, `test_zknormalcdf.cu`, `test_zkentropy.cu`, `test_verifier.cpp`
- **Benchmarks (4):** `bench_field.cu`, `bench_field_arith.cu`, `bench_commitment.cu`,
  `bench_matmul.cu`
- **Python scripts (14):** `verify_entropy.py`, `gen_entropy_inputs.py`, `gen_logits.py`,
  `gen_initial_input.py`, `calibrate_sigma.py`, `quantization_accuracy.py`,
  `overflow_check.py`, `run_proofs.py`, `download-models.py`, `fileio_utils.py`,
  `commit_final_layers.py`, `generate_swiglu_table.py`, `llama-*.py` (6 files)
- **Shell scripts (9):** `build_zkllm.sh`, `run_setup.sh`, `run_e2e.sh`,
  `run_e2e_resume.sh`, `run_proofs.sh`, `run_ppgen_logits.sh`, `run_test_entropy.sh`,
  `run_tests.sh`, `run_calibrate.sh`
- **Documentation (15):** `README.md`, `security-review.md`, `plan-full-verifier.md`,
  `plan-fp16-weights.md`, `plan-goldilocks-fri.md`, `plan-entropy-proof-redesign.md`,
  `plan-repo-restructure.md` (this file), `plan.md`, `plan2.md`, `design-goals.md`,
  `contributions.md`, `status.md`, `gpu-latency-reduction-plan.md`,
  `improvement-opportunities.md`, `int32-throughput-analysis.md`,
  `zkllm-entropy-scaling-analysis.md`, `zkml-efficiency-comparison.md`,
  `collect_nondeterminism_instructions.md`, `report-nondeterminism-sigma.md`,
  `bench-goldilocks-results.md`, `bench-results-2026-03-27.md`,
  `bench-results-2026-03-28.md`, `references.md`

## Proposed Directory Structure

```
zkllm-entropy/
├── src/
│   ├── field/              # Finite field arithmetic
│   │   ├── bls12-381.cu
│   │   ├── bls12-381.cuh
│   │   ├── goldilocks.cu
│   │   └── goldilocks.cuh
│   │
│   ├── poly/               # Polynomial operations
│   │   ├── polynomial.cu
│   │   ├── polynomial.cuh
│   │   ├── ntt.cu
│   │   └── ntt.cuh
│   │
│   ├── commit/             # Commitment schemes
│   │   ├── commitment.cu   #   KZG / Pedersen (BLS12-381)
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
│   │   ├── zkrelu.cu/.cuh
│   │   ├── zkfc.cu/.cuh
│   │   ├── zksoftmax.cu/.cuh
│   │   ├── zkargmax.cu/.cuh
│   │   ├── zklog.cu/.cuh
│   │   ├── zknormalcdf.cu/.cuh
│   │   ├── rescaling.cu/.cuh
│   │   └── tlookup.cu/.cuh
│   │
│   ├── entropy/            # Entropy-specific proving logic
│   │   ├── zkentropy.cu
│   │   └── zkentropy.cuh
│   │
│   ├── llm/                # LLM layer proofs
│   │   ├── self-attn.cu
│   │   ├── ffn.cu
│   │   ├── rmsnorm.cu
│   │   ├── rmsnorm_linear.cu
│   │   ├── post_attn.cu
│   │   ├── skip-connection.cu
│   │   └── skip_connection_cpu.cpp
│   │
│   └── util/               # Shared utilities
│       ├── ioutils.cu
│       ├── ioutils.cuh
│       ├── timer.cpp
│       └── timer.hpp
│
├── verifier/               # CPU-only verifier (no CUDA dependency)
│   ├── verifier.cpp        #   Main entry point
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
│   └── test_verifier.cpp
│
├── bench/                  # Benchmark programs
│   ├── bench_field.cu
│   ├── bench_field_arith.cu
│   ├── bench_commitment.cu
│   └── bench_matmul.cu
│
├── scripts/                # Build and run scripts
│   ├── build_zkllm.sh
│   ├── run_setup.sh
│   ├── run_e2e.sh
│   ├── run_e2e_resume.sh
│   ├── run_proofs.sh
│   ├── run_ppgen_logits.sh
│   ├── run_test_entropy.sh
│   ├── run_tests.sh
│   └── run_calibrate.sh
│
├── python/                 # Python tools
│   ├── verify_entropy.py   #   Proof verification
│   ├── gen_entropy_inputs.py
│   ├── gen_logits.py
│   ├── gen_initial_input.py
│   ├── calibrate_sigma.py
│   ├── quantization_accuracy.py
│   ├── overflow_check.py
│   ├── run_proofs.py
│   ├── download-models.py
│   ├── fileio_utils.py
│   ├── commit_final_layers.py
│   ├── generate_swiglu_table.py
│   ├── llama-commit.py
│   ├── llama-ffn.py
│   ├── llama-ppgen.py
│   ├── llama-rmsnorm.py
│   ├── llama-self-attn.py
│   └── llama-skip-connection.py
│
├── docs/                   # Documentation and plans
│   ├── plans/
│   │   ├── plan-full-verifier.md
│   │   ├── plan-fp16-weights.md
│   │   ├── plan-goldilocks-fri.md
│   │   ├── plan-entropy-proof-redesign.md
│   │   ├── plan-repo-restructure.md   # (this file)
│   │   ├── plan.md
│   │   ├── plan2.md
│   │   └── gpu-latency-reduction-plan.md
│   ├── analysis/
│   │   ├── security-review.md
│   │   ├── improvement-opportunities.md
│   │   ├── int32-throughput-analysis.md
│   │   ├── zkllm-entropy-scaling-analysis.md
│   │   ├── zkml-efficiency-comparison.md
│   │   └── report-nondeterminism-sigma.md
│   ├── benchmarks/
│   │   ├── bench-goldilocks-results.md
│   │   ├── bench-results-2026-03-27.md
│   │   └── bench-results-2026-03-28.md
│   ├── design-goals.md
│   ├── contributions.md
│   ├── status.md
│   ├── collect_nondeterminism_instructions.md
│   └── references.md
│
├── Makefile
└── README.md
```

## Rationale

| Concern | Current state | After restructure |
|---------|--------------|-------------------|
| Finding related code | Scroll through 70+ files | Navigate to the relevant `src/` subdirectory |
| Understanding architecture | Read file names and guess | Directory names map to architectural layers |
| Build targets vs libraries | Mixed together | `bin/` for executables, `src/` for libraries |
| Tests and benchmarks | Mixed with source | `test/` and `bench/` separated |
| CPU verifier | Mixed with GPU code | `verifier/` stands alone, no CUDA dependency |
| Python tools | Mixed with C++/CUDA | `python/` directory |
| Shell scripts | Mixed with everything | `scripts/` directory |
| Documentation | 20+ .md files at top level | `docs/` with subdirectories |

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

6. **`docs/` has subdirectories** for plans, analysis, and benchmark results. This
   keeps the 20+ documentation files organized without requiring readers to scan
   a flat list.

7. **`skip_connection_cpu.cpp` goes in `src/llm/`** alongside the GPU version. It's
   a CPU fallback for the same layer, so co-location makes sense.

## Migration Notes

- **Makefile must be updated** to reference new paths. The pattern rules (`%.o: %.cu`)
  will need `VPATH` or explicit per-directory rules.
- **`#include` paths** will change. Use `-I src/` in compiler flags so includes like
  `#include "field/goldilocks.cuh"` work.
- **Python imports:** `fileio_utils.py` is imported by other scripts. After moving to
  `python/`, update `sys.path` or add an `__init__.py`.
- **Shell script paths:** Scripts reference binaries and data files by relative path.
  These need updating, or the scripts should `cd` to the repo root.
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
