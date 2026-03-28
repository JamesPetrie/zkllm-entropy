// bench_matmul: microbenchmark for the FrTensor::matmul kernel.
// Times a (M×K) × (K×N) Goldilocks field matrix multiply.
//
// Usage: ./gold_bench_matmul [M=1024] [K=4096] [N=4096]

#include "tensor/fr-tensor.cuh"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

int main(int argc, char* argv[]) {
    uint M = argc > 1 ? (uint)atoi(argv[1]) : 1024;
    uint K = argc > 2 ? (uint)atoi(argv[2]) : 4096;
    uint N = argc > 3 ? (uint)atoi(argv[3]) : 4096;

    cout << "Matmul benchmark: (" << M << " x " << K << ") x (" << K << " x " << N << ")" << endl;
    double total_muls = (double)M * K * N;
    double total_field_ops = total_muls * 2; // mul + add per element
    cout << "Total multiply-adds: " << total_muls / 1e9 << " billion" << endl;
    cout << "Total field ops (mul+add): " << total_field_ops / 1e9 << " billion" << endl;
    cout << endl;

    // Create random-ish input tensors
    FrTensor A(M * K);
    FrTensor B(K * N);

    // Initialize with simple pattern (on GPU, faster than host init)
    // The FrTensor constructor zero-inits on GPU, so fill via a simple kernel
    // or just use the zero-init — throughput won't depend on values
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cout << "Warmup..." << flush;
    FrTensor warmup = FrTensor::matmul(A, B, M, K, N);
    cout << " done." << endl;

    // Timed runs
    int num_runs = 5;
    float total_ms = 0;
    float best_ms = 1e9;

    for (int r = 0; r < num_runs; r++) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        FrTensor C = FrTensor::matmul(A, B, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
        if (ms < best_ms) best_ms = ms;
        cout << "  Run " << r << ": " << ms << " ms" << endl;
    }

    float avg_ms = total_ms / num_runs;
    double gops_avg = total_field_ops / (avg_ms / 1000.0) / 1e9;
    double gops_best = total_field_ops / (best_ms / 1000.0) / 1e9;

    cout << endl;
    cout << "Average: " << avg_ms << " ms  (" << gops_avg << " Gfield-ops/sec)" << endl;
    cout << "Best:    " << best_ms << " ms  (" << gops_best << " Gfield-ops/sec)" << endl;
    
    // Context: what fraction of a layer proof is this?
    // One QKV projection = 3 * (1024 x 4096) x (4096 x 4096) 
    // One FFN = 3 * (1024 x 4096) x (4096 x 11008)
    // Total per layer ≈ 216B multiply-adds
    double layer_field_ops = 216e9 * 2;  // 7B model
    double est_layer_matmul_time = layer_field_ops / (gops_best * 1e9);
    cout << endl;
    cout << "Estimated matmul-only time for one 7B layer: " << est_layer_matmul_time << " sec" << endl;
    cout << "(Actual layer proof time includes sumcheck, commitments, etc.)" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
