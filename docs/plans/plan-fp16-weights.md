# Plan: Store Weights as fp16, Convert to Field Elements On-the-Fly

## Motivation

Weight matrices are currently stored as 8-byte Goldilocks field elements in GPU
memory. The source data is natively fp16 (2 bytes), quantized to int32 via
`round(fp16_to_float(w) * 65536)`, then converted to field elements. This plan
keeps weights in their fp16 representation and converts inside the matmul kernel.

**Benefits:**
- **4× less GPU memory** for weights (2B vs 8B per element)
- **L2 cache improvement** for 7B-scale models: a 4096×4096 weight matrix drops
  from 128 MB (exceeds H100 L2) to 32 MB (fits in H100's 50 MB L2). This
  eliminates redundant HBM reloads across activation row-tiles.
- **No proof changes** — the commitment is computed offline over the full field
  elements; the prover produces the same field elements on the fly. Transcript,
  verifier, and commitments are all identical.

**Verified:** fp16 → float → `* 65536` → round → int32 is bit-exact with the
current quantization pipeline. Tested on all 16.7M elements of Llama-2-7b
`q_proj.weight` with zero mismatches.

## Architecture

```
Current:
  disk (int32 4B) → GPU (int32 4B) → int_to_scalar kernel → GPU (Fr_t 8B) → matmul kernel

Proposed:
  disk (fp16 2B) → GPU (fp16 2B) → matmul kernel (converts inline)
```

The key insight is that `Weight.weight` (an `FrTensor`) can be replaced with a
compact `FP16Tensor` that stores raw `__half` values. The matmul kernel loads
`__half`, converts to `Fr_t` in registers (~4 instructions), and proceeds with
field arithmetic as before.

---

## Phase 1: FP16 Weight Tensor Class

**File:** `fr-tensor.cuh` / `fr-tensor.cu`

Add a lightweight GPU tensor for fp16 storage:

```cpp
class FrFP16Weights {
public:
    __half* gpu_data;
    const uint size;

    FrFP16Weights(uint size);
    FrFP16Weights(const FrFP16Weights& t);
    ~FrFP16Weights();

    // Load from fp16 binary file (2 bytes per element)
    static FrFP16Weights from_fp16_bin(const std::string& filename);

    // Load from int32 binary file (for backward compat), store as fp16
    static FrFP16Weights from_int_bin(const std::string& filename, float inv_scale);
};
```

`from_int_bin` provides backward compatibility: loads existing int32 files,
divides by the scaling factor, and stores as fp16. This avoids regenerating
weight files.

`from_fp16_bin` loads raw fp16 weights directly (2 bytes per element).

---

## Phase 2: Matmul Kernel with fp16 Weight Input

**File:** `fr-tensor.cu`

New kernel alongside `matrixMultiplyOptimized`:

```cuda
__device__ inline Fr_t fp16_to_field(const __half w, unsigned long scaling_factor) {
    float f = __half2float(w);
    long q = __float2ll_rn(f * (float)scaling_factor);
    return long_to_scalar(q);
}

KERNEL void matmul_fp16_weights(
    const Fr_t* A,           // Activations: M×N, full field elements
    const __half* B_fp16,    // Weights: N×P, stored as fp16
    Fr_t* C,                 // Output: M×P, full field elements
    int M, int N, int P,
    unsigned long scaling_factor)
{
    __shared__ Fr_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ Fr_t B_tile[TILE_WIDTH][TILE_WIDTH];  // converted in shared mem

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    Fr_t sum = blstrs__scalar__Scalar_ZERO;

    for (int t = 0; t < (N - 1)/TILE_WIDTH + 1; ++t) {
        // Load A tile (already field elements)
        if (row < M && t*TILE_WIDTH + threadIdx.x < N)
            A_tile[threadIdx.y][threadIdx.x] = A[row*N + t*TILE_WIDTH + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;

        // Load B tile: fp16 → field element conversion in registers
        if (t*TILE_WIDTH + threadIdx.y < N && col < P) {
            __half w = B_fp16[(t*TILE_WIDTH + threadIdx.y)*P + col];
            B_tile[threadIdx.y][threadIdx.x] = fp16_to_field(w, scaling_factor);
        } else {
            B_tile[threadIdx.y][threadIdx.x] = blstrs__scalar__Scalar_ZERO;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum = blstrs__scalar__Scalar_add(sum,
                  blstrs__scalar__Scalar_mul(A_tile[threadIdx.y][k], B_tile[k][threadIdx.x]));

        __syncthreads();
    }

    if (row < M && col < P)
        C[row*P + col] = blstrs__scalar__Scalar_mont(sum);
}
```

The fp16→field conversion happens during the B-tile load. Each thread converts
one element. The conversion is ~4 instructions (`__half2float`, float multiply,
`__float2ll_rn`, `long_to_scalar`), compared to the ~270 instructions in the
TILE_WIDTH=16 inner loop (16 field multiply-adds). Overhead: ~1.5%.

Add a static method:

```cpp
static FrTensor matmul_fp16(const FrTensor& activations,
                            const FrFP16Weights& weights,
                            uint M, uint N, uint P,
                            unsigned long scaling_factor);
```

---

## Phase 3: Integrate into Weight Struct and zkFC

**File:** `proof.cuh`

Extend `Weight` to support fp16:

```cpp
struct Weight {
    FrTensor weight;           // kept for commitment opening (padded full-precision)
    FrFP16Weights weight_fp16; // compact storage for matmul
    FriPcsCommitment com;
    uint in_dim;
    uint out_dim;
    unsigned long scaling_factor;
};
```

**File:** `proof.cu`

Update `create_weight` to also produce fp16:

```cpp
Weight create_weight(string weight_filename, string com_filename,
                     uint in_dim, uint out_dim, unsigned long scaling_factor)
{
    FrTensor weight = FrTensor::from_int_bin(weight_filename);
    auto w_padded = weight.pad({in_dim, out_dim});
    FriPcsCommitment com = /* load or compute */;

    // Also create fp16 version for matmul
    float inv_scale = 1.0f / (float)scaling_factor;
    FrFP16Weights weight_fp16 = FrFP16Weights::from_int_bin(weight_filename, inv_scale);

    return Weight{weight, weight_fp16, com, in_dim, out_dim, scaling_factor};
}
```

**File:** `zkfc.cu`

Update `zkFC::operator()` to dispatch to the fp16 kernel:

```cpp
FrTensor zkFC::operator()(const FrTensor& X) const {
    uint batchSize = X.size / inputSize;
    FrTensor out(batchSize * outputSize);
    // Use fp16 kernel when fp16 weights are available
    matmul_fp16_weights<<<grid, block>>>(
        X.gpu_data, weights_fp16.gpu_data, out.gpu_data,
        batchSize, inputSize, outputSize, scaling_factor);
    return out;
}
```

**Important:** The full `FrTensor weight` is still needed for `verifyWeightClaim`
(polynomial opening). Only the matmul uses the fp16 path. A future optimization
(Phase 5) can eliminate the full FrTensor entirely.

---

## Phase 4: Python Pipeline — Generate fp16 Weight Files

**File:** `llama-commit.py`

Add `--fp16-weights` flag:

```python
if args.fp16_weights:
    # Save raw fp16 weights (transposed, matching C++ layout)
    w_orig.half().cpu().numpy().tofile(fp16_bin_path)
```

**File:** `run_proofs.py`

Pass fp16 weight paths to layer binaries when available.

**Backward compatibility:** The C++ `from_int_bin` with `inv_scale` path means
existing int32 weight files still work. New fp16 files are optional.

---

## Phase 5 (Future): Eliminate Full FrTensor for Weights

Currently `create_weight` loads the full FrTensor (8B/elem) for two purposes:
1. The matmul (replaced by fp16 in Phase 2)
2. `verifyWeightClaim` → `FriPcs::open` needs the padded field elements

To fully eliminate the 8B copy, `FriPcs::open` would need to accept fp16 data
and convert on the fly (similar to the matmul kernel). This is a deeper change
to the PCS layer and can be done later.

With Phase 3 alone, memory usage increases slightly (fp16 + full FrTensor). The
immediate benefit is speed (L2 cache effects in matmul). Memory savings come in
Phase 5.

**Alternative for Phase 3:** Free the full FrTensor after commitment loading,
reconstruct it only when `verifyWeightClaim` is called. This recovers most of
the memory benefit without touching the PCS layer.

---

## Testing Plan

1. **Unit test:** Write `bench_matmul_fp16.cu` that:
   - Creates random weights as fp16
   - Runs both `matrixMultiplyOptimized` (with pre-expanded Fr_t) and
     `matmul_fp16_weights` on the same inputs
   - Verifies output is bit-identical
   - Reports throughput for both

2. **Integration test:** Run entropy proof with fp16 matmul path, compare
   entropy value and proof output against current code.

3. **Single-layer test:** `run_proofs.py --goldilocks --num_layers 1` with
   fp16 weights, verify all layer proofs pass.

4. **Full proof:** 32-layer proof, compare timing.

---

## Expected Impact

| Metric | Current | With fp16 weights |
|--------|---------|-------------------|
| Weight memory per layer (7B) | 1.6 GB | 0.4 GB |
| Weight matrix in L2 (4096²) | 128 MB (miss) | 32 MB (hit) |
| Matmul kernel speed | baseline | **TBD — needs benchmark** |
| Total weight memory (1T) | 8 TB (42 B200s) | 2 TB (10 B200s) |

## File Changes

| File | Change |
|------|--------|
| `fr-tensor.cuh` | Add `FrFP16Weights` class declaration |
| `fr-tensor.cu` | Add `FrFP16Weights` impl, `matmul_fp16_weights` kernel |
| `proof.cuh` | Add `weight_fp16` and `scaling_factor` to `Weight` struct |
| `proof.cu` | Update `create_weight` to produce fp16 |
| `zkfc.cu` | `operator()` dispatches to fp16 matmul kernel |
| `self-attn.cu` | Pass scaling factor to `create_weight` |
| `ffn.cu` | Pass scaling factor to `create_weight` |
| `rmsnorm.cu` | Pass scaling factor to `create_weight` |
| `llama-commit.py` | Optional `--fp16-weights` flag |
| `Makefile` | No changes (same compilation units) |
