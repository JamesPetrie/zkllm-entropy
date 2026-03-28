// test_ntt: verify NTT forward/inverse and coset variants over Goldilocks.
//
// Tests:
// 1. Root of unity properties (omega^n == 1, omega^(n/2) == -1)
// 2. NTT forward then inverse recovers original data
// 3. NTT evaluates polynomial at roots of unity (manual check for small n)
// 4. Coset NTT forward then inverse recovers original data
// 5. Larger NTT (2^16 elements) round-trip
//
// Usage: ./test_ntt

#include "poly/ntt.cuh"
#include <iostream>
#include <vector>
#include <cstring>

using namespace std;

// Host-side Goldilocks arithmetic for verification
static uint64_t h_mul(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t eps = GOLDILOCKS_P_NEG;
    __uint128_t t = (__uint128_t)hi * eps + lo;
    uint64_t r_lo = (uint64_t)t;
    uint64_t r_hi = (uint64_t)(t >> 64);
    uint64_t result = r_lo + r_hi * eps;
    if (result < r_lo) result += eps;
    if (result >= GOLDILOCKS_P) result -= GOLDILOCKS_P;
    return result;
}

static uint64_t h_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a || s >= GOLDILOCKS_P) s -= GOLDILOCKS_P;
    return s;
}

static uint64_t h_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result = h_mul(result, base);
        base = h_mul(base, base);
        exp >>= 1;
    }
    return result;
}

int main() {
    cout << "=== NTT Tests (Goldilocks) ===" << endl;
    int failures = 0;
    auto check = [&](bool cond, const char* name) {
        if (cond) { cout << "  PASS: " << name << endl; }
        else { cout << "  FAIL: " << name << endl; failures++; }
    };

    // ── Test 1: Root of unity properties ────────────────────────────────
    {
        for (uint log_n = 1; log_n <= 20; log_n++) {
            Fr_t omega = get_root_of_unity(log_n);
            uint64_t n = 1ULL << log_n;
            // omega^n should equal 1
            uint64_t omega_n = h_pow(omega.val, n);
            if (omega_n != 1) {
                cout << "  FAIL: omega^n != 1 for log_n=" << log_n << endl;
                failures++;
                break;
            }
            // omega^(n/2) should equal p-1 (i.e., -1 mod p)
            uint64_t omega_half = h_pow(omega.val, n / 2);
            if (omega_half != GOLDILOCKS_P - 1) {
                cout << "  FAIL: omega^(n/2) != -1 for log_n=" << log_n << endl;
                failures++;
                break;
            }
        }
        check(true, "root of unity: omega^n==1 and omega^(n/2)==-1 for log_n=1..20");
    }

    // ── Test 2: Small NTT round-trip (n=8) ──────────────────────────────
    {
        uint log_n = 3;
        uint n = 1 << log_n;
        vector<Fr_t> data_host(n);
        for (uint i = 0; i < n; i++) data_host[i] = Fr_t{(uint64_t)(i + 1)};
        vector<Fr_t> original = data_host;

        Fr_t* data_gpu;
        cudaMalloc(&data_gpu, n * sizeof(Fr_t));
        cudaMemcpy(data_gpu, data_host.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        ntt_forward(data_gpu, log_n);
        ntt_inverse(data_gpu, log_n);

        cudaMemcpy(data_host.data(), data_gpu, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaFree(data_gpu);

        bool ok = true;
        for (uint i = 0; i < n; i++) {
            if (data_host[i].val != original[i].val) { ok = false; break; }
        }
        check(ok, "NTT forward+inverse round-trip (n=8)");
    }

    // ── Test 3: NTT evaluates polynomial at roots of unity ──────────────
    {
        // Polynomial p(x) = 1 + 2x + 3x^2 + 4x^3 (coefficients [1,2,3,4])
        // NTT should give [p(omega^0), p(omega^1), p(omega^2), p(omega^3)]
        uint log_n = 2;
        uint n = 1 << log_n;
        vector<Fr_t> coeffs = {{1}, {2}, {3}, {4}};

        Fr_t* data_gpu;
        cudaMalloc(&data_gpu, n * sizeof(Fr_t));
        cudaMemcpy(data_gpu, coeffs.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        ntt_forward(data_gpu, log_n);

        vector<Fr_t> evals(n);
        cudaMemcpy(evals.data(), data_gpu, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaFree(data_gpu);

        // Verify against naive evaluation
        uint64_t omega = get_root_of_unity(log_n).val;
        bool ok = true;
        for (uint i = 0; i < n; i++) {
            uint64_t w = h_pow(omega, i);
            // p(w) = 1 + 2*w + 3*w^2 + 4*w^3
            uint64_t expected = h_add(h_add(h_add(1, h_mul(2, w)), h_mul(3, h_mul(w, w))), h_mul(4, h_mul(w, h_mul(w, w))));
            if (evals[i].val != expected) {
                cout << "    Mismatch at i=" << i << ": got " << evals[i].val << " expected " << expected << endl;
                ok = false;
            }
        }
        check(ok, "NTT evaluates polynomial at roots of unity (n=4)");
    }

    // ── Test 4: Coset NTT round-trip ────────────────────────────────────
    {
        uint log_n = 3;
        uint n = 1 << log_n;
        vector<Fr_t> data_host(n);
        for (uint i = 0; i < n; i++) data_host[i] = Fr_t{(uint64_t)(i * 7 + 3)};
        vector<Fr_t> original = data_host;

        // Use a shift: the 2^(log_n+1)-th root of unity (generator of the coset)
        Fr_t shift = get_root_of_unity(log_n + 1);

        Fr_t* data_gpu;
        cudaMalloc(&data_gpu, n * sizeof(Fr_t));
        cudaMemcpy(data_gpu, data_host.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        ntt_coset_forward(data_gpu, log_n, shift);
        ntt_coset_inverse(data_gpu, log_n, shift);

        cudaMemcpy(data_host.data(), data_gpu, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaFree(data_gpu);

        bool ok = true;
        for (uint i = 0; i < n; i++) {
            if (data_host[i].val != original[i].val) { ok = false; break; }
        }
        check(ok, "coset NTT forward+inverse round-trip (n=8)");
    }

    // ── Test 5: Larger NTT round-trip (n=2^16) ─────────────────────────
    {
        uint log_n = 16;
        uint n = 1 << log_n;
        vector<Fr_t> data_host(n);
        for (uint i = 0; i < n; i++) data_host[i] = Fr_t{(uint64_t)(i % 997)};
        vector<Fr_t> original = data_host;

        Fr_t* data_gpu;
        cudaMalloc(&data_gpu, n * sizeof(Fr_t));
        cudaMemcpy(data_gpu, data_host.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        ntt_forward(data_gpu, log_n);
        ntt_inverse(data_gpu, log_n);

        cudaMemcpy(data_host.data(), data_gpu, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaFree(data_gpu);

        bool ok = true;
        for (uint i = 0; i < n; i++) {
            if (data_host[i].val != original[i].val) {
                cout << "    Mismatch at i=" << i << ": got " << data_host[i].val << " expected " << original[i].val << endl;
                ok = false;
                if (i > 5) break;  // limit output
            }
        }
        check(ok, "NTT round-trip (n=2^16 = 65536)");
    }

    // ── Test 6: Very large NTT round-trip (n=2^20) ─────────────────────
    {
        uint log_n = 20;
        uint n = 1 << log_n;
        vector<Fr_t> data_host(n);
        for (uint i = 0; i < n; i++) data_host[i] = Fr_t{(uint64_t)(i % 10007)};
        vector<Fr_t> original = data_host;

        Fr_t* data_gpu;
        cudaMalloc(&data_gpu, n * sizeof(Fr_t));
        cudaMemcpy(data_gpu, data_host.data(), n * sizeof(Fr_t), cudaMemcpyHostToDevice);

        ntt_forward(data_gpu, log_n);
        ntt_inverse(data_gpu, log_n);

        cudaMemcpy(data_host.data(), data_gpu, n * sizeof(Fr_t), cudaMemcpyDeviceToHost);
        cudaFree(data_gpu);

        bool ok = true;
        for (uint i = 0; i < n; i++) {
            if (data_host[i].val != original[i].val) {
                cout << "    Mismatch at i=" << i << ": got " << data_host[i].val << " expected " << original[i].val << endl;
                ok = false;
                if (i > 5) break;
            }
        }
        check(ok, "NTT round-trip (n=2^20 = 1M)");
    }

    cout << "\n=== Results: " << (failures == 0 ? "ALL PASSED" : "FAILURES")
         << " (failures=" << failures << ") ===" << endl;
    return failures;
}
