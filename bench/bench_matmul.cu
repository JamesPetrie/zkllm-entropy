// bench_matmul: microbenchmark for the FrTensor::matmul kernel.
// Times a (M×K) × (K×N) Goldilocks field matrix multiply.
// Also compares the original kernel vs the optimized V2 kernel.
//
// Usage: ./gold_bench_matmul [M=1024] [K=4096] [N=4096]

#include "tensor/fr-tensor.cuh"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// Declare both kernels (defined in fr-tensor.cu)
KERNEL void matrixMultiplyOptimized(Fr_t* A, Fr_t* B, Fr_t* C,
                                     int rowsA, int colsA, int colsB);
#ifdef USE_GOLDILOCKS
KERNEL void matrixMultiplyGoldV2(Fr_t* A, Fr_t* B, Fr_t* C,
                                  int rowsA, int colsA, int colsB);
#endif

struct BenchResult {
    float avg_ms;
    float best_ms;
    double gops;
};

BenchResult run_bench(const char* name, FrTensor& A, FrTensor& B,
                      uint M, uint K, uint N, int kernel_id) {
    double total_field_ops = (double)M * K * N * 2;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid((N-1)/TILE_WIDTH + 1, (M-1)/TILE_WIDTH + 1);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    // Warmup
    FrTensor warmup(M * N);
    if (kernel_id == 0) {
        matrixMultiplyOptimized<<<grid, block>>>(A.gpu_data, B.gpu_data,
                                                  warmup.gpu_data, M, K, N);
    }
#ifdef USE_GOLDILOCKS
    else {
        matrixMultiplyGoldV2<<<grid, block>>>(A.gpu_data, B.gpu_data,
                                               warmup.gpu_data, M, K, N);
    }
#endif
    cudaDeviceSynchronize();

    int num_runs = 5;
    float total_ms = 0;
    float best_ms = 1e9;

    for (int r = 0; r < num_runs; r++) {
        FrTensor C(M * N);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        if (kernel_id == 0) {
            matrixMultiplyOptimized<<<grid, block>>>(A.gpu_data, B.gpu_data,
                                                      C.gpu_data, M, K, N);
        }
#ifdef USE_GOLDILOCKS
        else {
            matrixMultiplyGoldV2<<<grid, block>>>(A.gpu_data, B.gpu_data,
                                                   C.gpu_data, M, K, N);
        }
#endif
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
        if (ms < best_ms) best_ms = ms;
        cout << "  " << name << " run " << r << ": " << ms << " ms" << endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float avg_ms = total_ms / num_runs;
    double gops = total_field_ops / (best_ms / 1000.0) / 1e9;
    return {avg_ms, best_ms, gops};
}

// Correctness check: compare outputs of both kernels element-by-element
#ifdef USE_GOLDILOCKS
bool verify_correctness(FrTensor& A, FrTensor& B, uint M, uint K, uint N) {
    dim3 grid((N-1)/TILE_WIDTH + 1, (M-1)/TILE_WIDTH + 1);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

    FrTensor C_orig(M * N);
    FrTensor C_v2(M * N);

    matrixMultiplyOptimized<<<grid, block>>>(A.gpu_data, B.gpu_data,
                                              C_orig.gpu_data, M, K, N);
    matrixMultiplyGoldV2<<<grid, block>>>(A.gpu_data, B.gpu_data,
                                           C_v2.gpu_data, M, K, N);
    cudaDeviceSynchronize();

    // Copy both to host and compare
    uint64_t* h_orig = new uint64_t[M * N];
    uint64_t* h_v2   = new uint64_t[M * N];
    cudaMemcpy(h_orig, C_orig.gpu_data, M * N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v2,   C_v2.gpu_data,   M * N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (uint i = 0; i < M * N && mismatches < 10; i++) {
        if (h_orig[i] != h_v2[i]) {
            cout << "  MISMATCH at [" << i / N << "][" << i % N << "]: "
                 << "orig=" << h_orig[i] << " v2=" << h_v2[i] << endl;
            mismatches++;
        }
    }

    delete[] h_orig;
    delete[] h_v2;

    if (mismatches == 0) {
        cout << "  Correctness: PASS (all " << M * N << " elements match)" << endl;
    } else {
        cout << "  Correctness: FAIL (" << mismatches << " mismatches)" << endl;
    }
    return mismatches == 0;
}
#endif

int main(int argc, char* argv[]) {
    uint M = argc > 1 ? (uint)atoi(argv[1]) : 1024;
    uint K = argc > 2 ? (uint)atoi(argv[2]) : 4096;
    uint N = argc > 3 ? (uint)atoi(argv[3]) : 4096;

    cout << "Matmul benchmark: (" << M << " x " << K << ") x (" << K << " x " << N << ")" << endl;
    double total_muls = (double)M * K * N;
    cout << "Total multiply-adds: " << total_muls / 1e9 << " billion" << endl;
    cout << endl;

    FrTensor A(M * K);
    FrTensor B(K * N);

    // --- Original kernel ---
    cout << "=== Original kernel (matrixMultiplyOptimized) ===" << endl;
    auto r_orig = run_bench("orig", A, B, M, K, N, 0);
    cout << "  Avg: " << r_orig.avg_ms << " ms  Best: " << r_orig.best_ms
         << " ms  (" << r_orig.gops << " Gfield-ops/s)" << endl << endl;

#ifdef USE_GOLDILOCKS
    // --- V2 kernel ---
    cout << "=== V2 kernel (matrixMultiplyGoldV2: 2-way ILP + double buffer) ===" << endl;
    auto r_v2 = run_bench("v2", A, B, M, K, N, 1);
    cout << "  Avg: " << r_v2.avg_ms << " ms  Best: " << r_v2.best_ms
         << " ms  (" << r_v2.gops << " Gfield-ops/s)" << endl << endl;

    // --- Comparison ---
    float speedup = r_orig.best_ms / r_v2.best_ms;
    cout << "=== Comparison ===" << endl;
    cout << "  Original: " << r_orig.best_ms << " ms" << endl;
    cout << "  V2:       " << r_v2.best_ms << " ms" << endl;
    cout << "  Speedup:  " << speedup << "x" << endl << endl;

    // --- Correctness check ---
    cout << "=== Correctness check ===" << endl;
    verify_correctness(A, B, M, K, N);
#endif

    // Context: estimated layer matmul time
    double best_gops = r_orig.gops;
#ifdef USE_GOLDILOCKS
    if (r_v2.gops > best_gops) best_gops = r_v2.gops;
#endif
    double layer_field_ops = 216e9 * 2;  // 7B model
    double est_layer_matmul_time = layer_field_ops / (best_gops * 1e9);
    cout << endl;
    cout << "Estimated matmul-only time for one 7B layer: " << est_layer_matmul_time << " sec" << endl;

    return 0;
}
