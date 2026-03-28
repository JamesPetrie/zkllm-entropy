# INT32 Throughput on H100 and Blackwell GPUs

**Date:** 2026-03-27
**Purpose:** Understand the hardware ceiling for field arithmetic throughput, since
Goldilocks (64-bit) field operations decompose into INT32/INT64 instructions.

---

## INT32 Compute Throughput

### Per-SM Architecture

**Hopper (H100, compute capability 9.0):**
- 128 FP32 cores + 64 INT32 cores per SM (separate hardware)
- INT32 IMAD throughput: 64 ops/clock/SM
- FP32 FFMA throughput: 128 ops/clock/SM
- INT32 = exactly half of FP32 throughput
- The 64 INT32 units are not an independent datapath — they cannot run concurrently
  with FP32 at full throughput (unlike Turing, which had truly independent INT/FP paths)

Sources: [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/);
NVIDIA staff confirmation on [Developer Forums](https://forums.developer.nvidia.com/t/calculating-tops-and-tflops-in-h100/300345).

**Blackwell (B100/B200/GB200, compute capability 10.0):**
- 128 CUDA cores per SM, fully unified for FP32 and INT32
- Each core can execute either FP32 or INT32 per cycle, not both simultaneously
- INT32 throughput doubles vs Hopper (128 ops/clock/SM, matching FP32)
- Mixed INT/FP kernels share the 128-core budget

Source: [RTX Blackwell GPU Architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf);
[Developer Forums](https://forums.developer.nvidia.com/t/blackwell-integer/320578).

Note: NVIDIA has not yet published the official CUDA arithmetic instruction throughput
table for compute capability 10.0. The INT32=FP32 parity is inferred from the unified
core architecture and confirmed for consumer Blackwell (RTX 5090: 104.8 INT32 TOPS =
FP32 TFLOPS). Datacenter Blackwell is expected to follow the same pattern.

### GPU-Level INT32 Throughput

| GPU | SMs | INT32 ops/clk/SM | INT32 TOPS | FP32 TFLOPS | INT32/FP32 |
|-----|-----|------------------|-----------|-------------|------------|
| **H100 SXM** | 132 (114 enabled) | 64 | **30** | 60-67 | 0.5x |
| **H100 PCIe** | 114 | 64 | **~26** | 51 | 0.5x |
| **B200** | ~160 (est.) | 128 | **~75-90** | 75-90 | 1.0x |
| **GB200** (per die) | ~160 (est.) | 128 | **~80** | ~80 | 1.0x |

H100 SXM numbers (30 INT32 TOPS, 60 FP32 TFLOPS) are from the
[Hopper Architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).
H100 PCIe FP32 (51 TFLOPS) is from the
[H100 product page](https://www.nvidia.com/en-us/data-center/h100/); INT32 derived at 0.5x.
B200 FP32 (75 TFLOPS per GPU) is from the
[DGX B200 page](https://www.nvidia.com/en-us/data-center/dgx-b200/) (600 TFLOPS / 8 GPUs).
GB200 FP32 (160 TFLOPS per superchip = ~80 per GPU) is from the
[GB200 NVL72 page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).

---

## Memory Bandwidth

| GPU | Memory | Capacity | Bandwidth |
|-----|--------|----------|-----------|
| **H100 SXM** | HBM3 | 80 GB | **3.35 TB/s** |
| **H100 PCIe** | HBM2e | 80 GB | **2.0 TB/s** |
| **B200** | HBM3e | 192 GB | **8.0 TB/s** |
| **GB200** (per GPU) | HBM3e | 192 GB | **8.0 TB/s** |

Sources: [H100 product page](https://www.nvidia.com/en-us/data-center/h100/),
[GB200 NVL72 page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).

---

## Arithmetic Intensity Ceiling (INT32 ops per byte from HBM)

| GPU | INT32 TOPS | Bandwidth (TB/s) | Ops/byte |
|-----|-----------|-------------------|----------|
| **H100 SXM** | 30 | 3.35 | **9.0** |
| **H100 PCIe** | ~26 | 2.0 | **~13** |
| **B200** | ~75-90 | 8.0 | **~9-11** |
| **GB200** (per GPU) | ~80 | 8.0 | **~10** |

A kernel must perform at least ~9-13 INT32 ops per byte loaded from HBM to be
compute-bound. Below this ratio, the kernel is memory-bandwidth-limited.

---

## Implications for Goldilocks Field Arithmetic

### Current measured throughput vs hardware ceiling

Our Goldilocks multiply benchmark measures **13.0 Gop/s** (field multiplies per second)
on H100 PCIe (33M elements, `bench-goldilocks-results.md`).

Each `gold_mul` (see `goldilocks.cuh:73-117`) performs 3 × `__umul64hi` calls (each
decomposing into ~4 INT32 multiplies on hardware), 3 regular 64-bit multiplies, carry
logic, comparisons, and a conditional subtract — roughly **30-40 INT32 ops** per field
multiply. This gives:

| | Ceiling | Achieved | Utilization |
|---|---|---|---|
| **Compute** | ~26 TOPS / ~35 INT32 ops per field mul ≈ **740 Gop/s** | 13 Gop/s | **1.8%** |
| **Bandwidth** | 2.0 TB/s / 24 bytes per field mul ≈ **83 Gop/s** | 13 Gop/s | **16%** |

We achieve 16% of the bandwidth ceiling but only 1.8% of the compute ceiling. The
binding constraint is **memory bandwidth** — the GPU can compute field multiplies ~9×
faster than it can feed them with data from HBM. Each element-wise multiply reads two
8-byte operands and writes one 8-byte result (24 bytes total).

The 16% bandwidth utilization (rather than closer to 100%) is typical for simple
element-wise kernels that don't fully hide memory latency. Higher occupancy,
instruction-level parallelism, or kernel fusion would improve this.

### Projection for Blackwell

| Metric | H100 PCIe | B200 | Speedup |
|--------|-----------|------|---------|
| INT32 TOPS | ~26 | ~75-90 | 2.9-3.5x |
| Memory bandwidth | 2.0 TB/s | 8.0 TB/s | 4.0x |
| Bandwidth-limited Goldilocks mul ceiling | ~83 Gop/s | ~333 Gop/s | 4.0x |
| Compute-limited Goldilocks mul ceiling | ~740 Gop/s | ~2,100-2,600 Gop/s | 2.9-3.5x |

Since Goldilocks field arithmetic is memory-bandwidth-limited, the B200's 4x bandwidth
advantage translates to roughly **4x higher throughput** for element-wise field operations
(add, multiply, etc.). For operations with higher arithmetic intensity (e.g., NTT,
sumcheck polynomial evaluation), the 2.9-3.5x compute improvement becomes relevant.

### Sumcheck arithmetic intensity

The sumcheck inner loop (e.g., `Fr_ip_sc_step` in `proof.cu`) reads 4 elements per
pair (A[2i], A[2i+1], B[2i], B[2i+1] = 32 bytes for Goldilocks), computes 3 field
multiplies + ~6 field adds, and writes 2 reduced elements for the next round (16 bytes).

In INT32 terms: 3 × ~35 + 6 × ~3 ≈ **123 INT32 ops per 48 bytes** = **2.6 INT32
ops/byte**. This is well below the ~10-13 ops/byte crossover point. Sumcheck rounds
are firmly memory-bandwidth-limited.

This means:
1. **Goldilocks on B200 should be ~4x faster** than H100 PCIe for sumcheck-dominated
   proofs (proportional to bandwidth ratio, not compute ratio).
2. **Kernel fusion** (combining multiple sumcheck rounds or field operations into a
   single kernel pass to reduce memory traffic) would have more impact than faster
   arithmetic.
3. **Mersenne31** (31-bit field, 4 bytes vs 8 bytes per element) would roughly **double**
   effective bandwidth throughput by halving element size, in addition to cheaper arithmetic.

---

## Primary Sources

1. [NVIDIA H100 Product Page](https://www.nvidia.com/en-us/data-center/h100/) — FP32 TFLOPS, memory specs
2. [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — SM layout (128 FP32 + 64 INT32), peak 30 INT32 TOPS
3. [NVIDIA Developer Forums: TOPS/TFLOPS in H100](https://forums.developer.nvidia.com/t/calculating-tops-and-tflops-in-h100/300345) — NVIDIA staff confirmation of IMAD=64/clk/SM
4. [NVIDIA RTX Blackwell Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) — Unified INT32/FP32, doubled INT32
5. [NVIDIA Developer Forums: Blackwell Integer](https://forums.developer.nvidia.com/t/blackwell-integer/320578) — cc10.0 throughput table not yet published
6. [NVIDIA GB200 NVL72 Page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) — 160 TFLOPS FP32 per superchip, 8 TB/s per GPU
7. [NVIDIA DGX B200 Page](https://www.nvidia.com/en-us/data-center/dgx-b200/) — B200 system specs
