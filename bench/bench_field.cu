// Minimal field arithmetic benchmark: times FrTensor ops for both builds.
// Build as BLS:  make bench_field
// Build as Gold: make gold_bench_field

#include "tensor/fr-tensor.cuh"
#include <iostream>
#include <chrono>
#include <cstdlib>

using namespace std;
using hrc = chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    uint N = (argc > 1) ? (uint)atoi(argv[1]) : (1u << 20);  // ~1M elements default

    cout << "Field benchmark: N = " << N << " elements" << endl;
#ifdef USE_GOLDILOCKS
    cout << "Field: Goldilocks (64-bit)" << endl;
#else
    cout << "Field: BLS12-381 (256-bit)" << endl;
#endif

    // Create random tensors
    auto t0 = hrc::now();
    FrTensor a = FrTensor::random(N);
    FrTensor b = FrTensor::random(N);
    cudaDeviceSynchronize();
    auto t1 = hrc::now();
    double init_ms = chrono::duration<double,milli>(t1-t0).count();
    cout << "Init (2x random): " << init_ms << " ms" << endl;

    // Warmup
    { FrTensor c = a + b; cudaDeviceSynchronize(); }
    { FrTensor c = a * b; cudaDeviceSynchronize(); }

    int REPS = 10;

    // Element-wise add
    auto ta0 = hrc::now();
    for (int i = 0; i < REPS; i++) {
        FrTensor c = a + b;
        cudaDeviceSynchronize();
    }
    auto ta1 = hrc::now();
    double add_ms = chrono::duration<double,milli>(ta1-ta0).count() / REPS;

    // Element-wise multiply (Hadamard)
    auto tm0 = hrc::now();
    for (int i = 0; i < REPS; i++) {
        FrTensor c = a * b;
        cudaDeviceSynchronize();
    }
    auto tm1 = hrc::now();
    double mul_ms = chrono::duration<double,milli>(tm1-tm0).count() / REPS;

    // Sum (reduction)
    auto ts0 = hrc::now();
    for (int i = 0; i < REPS; i++) {
        Fr_t s = a.sum();
        cudaDeviceSynchronize();
    }
    auto ts1 = hrc::now();
    double sum_ms = chrono::duration<double,milli>(ts1-ts0).count() / REPS;

    cout << "\nResults (avg of " << REPS << " reps, N=" << N << "):" << endl;
    cout << "  Add (a+b):  " << add_ms << " ms" << endl;
    cout << "  Mul (a*b):  " << mul_ms << " ms" << endl;
    cout << "  Sum:        " << sum_ms << " ms" << endl;
    cout << "  Add tput:   " << (double)N / add_ms / 1e6 << " Gop/s" << endl;
    cout << "  Mul tput:   " << (double)N / mul_ms / 1e6 << " Gop/s" << endl;

    return 0;
}
