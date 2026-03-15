# Implementation Status

## What is built and compiles

All new source files compile cleanly. The following binaries were successfully produced
in the last build (job 1071, 2026-03-15):

| Binary | Status | Notes |
|---|---|---|
| `zkllm_entropy` | **Built** | Main entropy prover entry point |
| `test_zkargmax` | Not linked | Job timed out during linking (30 min limit) |
| `test_zklog` | Not linked | Job timed out |
| `test_zknormalcdf` | Not linked | Job timed out |
| `test_zkentropy` | Not linked | Job timed out |

All four object files (`zkargmax.o`, `zklog.o`, `zknormalcdf.o`, `zkentropy.o`) compiled
without errors. The test binaries need a longer build job (increase `--time` in
`build_zkllm.sh` to e.g. `01:00:00`).

## New files added

| File | Description |
|---|---|
| `zkargmax.cuh/cu` | ZK argmax via bit-decomposition range proofs |
| `zklog.cuh/cu` | ZK −log₂ via tLookupRangeMapping |
| `zknormalcdf.cuh/cu` | ZK normal CDF via tLookupRangeMapping |
| `zkentropy.cuh/cu` | Full conditional entropy pipeline (compute + prove) |
| `zkllm_entropy.cu` | Standalone binary: loads saved logit tensors, runs prover |
| `test_zkargmax.cu` | Unit tests for zkArgmax |
| `test_zklog.cu` | Unit tests for zkLog |
| `test_zknormalcdf.cu` | Unit tests for zkNormalCDF |
| `test_zkentropy.cu` | Integration tests for zkConditionalEntropy |

## Architecture summary

```
zkllm_entropy.cu
  └─ zkentropy.cuh/cu (zkConditionalEntropy)
       ├─ zkargmax.cuh/cu   — argmax + bit-decomp range proof
       ├─ zknormalcdf.cuh/cu — Phi(d/sigma) lookup (tLookupRangeMapping)
       └─ zklog.cuh/cu       — −log2(q) lookup (tLookupRangeMapping)
```

Per-position pipeline:
1. `zkArgmax::compute` — find greedy token t_star
2. GPU kernel: `diffs[i] = v_star − logits[i]`, clamped to CDF table range
3. `zkNormalCDF::compute` — win_probs[i] = Φ(diffs[i] / σ) × cdf_scale
4. GPU kernel: normalise → q_indices[i] in [1, 2^log_precision]
5. `zkLog::compute` — log_vals[i] = −log₂(q_indices[i] / 2^log_precision) × log_scale
6. Return `log_vals[actual_token]` as surprise for this position

## Known TODOs / limitations

1. **Normalisation proof missing**: `zkentropy.cu` has a `TODO` comment at the
   normalisation step. The compute is correct; the ZK proof that
   `q_idx * total == win_probs[actual] * 2^log_precision − remainder` (with
   bit-decomposed remainder) is not yet implemented. This is the main remaining
   cryptographic gap.

2. **Log-table size constraint**: `tLookupRangeMapping::prove` requires
   `vocab_size % table_size == 0`. For LLaMA (vocab=32000) with `log_precision=15`
   (table=32768) this fails. Workaround: use `log_precision ≤ 5` (table=32, divides
   32000) or pad `vocab_size` to the next power of 2. The `compute()` path has
   no such constraint.

3. **Test binaries not yet linked**: Increase `--time` in `build_zkllm.sh` to
   `01:00:00` and resubmit `sbatch build_zkllm.sh` to finish linking the test
   binaries.

4. **Proof serialisation**: `vector<Polynomial>& proof` is passed through all
   prove functions but not yet serialised to disk. `zkllm_entropy` writes a
   plaintext summary only.

## How to run

### Build test binaries
Edit `build_zkllm.sh`: change `--time=00:30:00` → `--time=01:00:00`, then:
```bash
sbatch /mnt/sharefs/user50/zk/build_zkllm.sh
```

### Run tests (once built, on a GPU node)
```bash
srun --gpus=1 --pty bash
cd /mnt/sharefs/user50/zk/zkllm-ccs2024
./test_zkargmax
./test_zklog
./test_zknormalcdf
./test_zkentropy
```

### Run the entropy prover
```bash
./zkllm_entropy <logits_dir> <tokens_file> <proof_output> <sigma_eff> \
    [bit_width=16] [cdf_precision=18] [log_precision=15] \
    [cdf_scale=65536] [log_scale=65536]
```
`logits_dir` should contain `logits_0.bin`, `logits_1.bin`, … saved via
`FrTensor::save()`. `sigma_eff` = σ_real × logit_scaling_factor (e.g. for
σ=0.05 and logit_scale=65536: sigma_eff=3277).

### Calibrate sigma
Run the model twice on the same input (temperature 0) and measure the fraction
of tokens that match exactly. Find σ such that the noise model predicts this
match rate. Target: ~97% exact match.
