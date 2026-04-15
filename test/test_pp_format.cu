// Phase 1.5 pp file v2 round-trip and tamper-rejection tests.
//
//   1. Round-trip:  save_hiding → load_hiding succeeds and every
//      generator survives byte-exactly.
//   2. Tamper G_i:  flip one byte inside a G_i slot and confirm
//      load_hiding throws, naming the exact index.
//   3. Tamper H:    flip one byte in the H slot → load throws.
//   4. Tamper U:    flip one byte in the U slot → load throws.
//   5. Truncation:  truncate after the header → load throws cleanly
//      rather than reading garbage.
//   6. Bad magic:   a non-v2 file (e.g. an old-format pp blob) is
//      rejected with a message steering the user at ppgen.

#include "commit/commitment.cuh"
#include "field/hash_to_curve.cuh"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

// Read the whole file into a byte vector.
static vector<uint8_t> slurp(const string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { cerr << "slurp: cannot open " << path << endl; exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<uint8_t> out(n);
    fread(out.data(), 1, n, f);
    fclose(f);
    return out;
}

static void write_bytes(const string& path, const vector<uint8_t>& bytes) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { cerr << "write_bytes: cannot open " << path << endl; exit(1); }
    fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
}

// Computed offsets into the v2 header.
//   magic(8) + version(4) + flags(4) + dst_len(4) + dst + size(4) + ...
struct Layout {
    size_t dst_offset;
    size_t dst_len;
    size_t size_offset;
    size_t g_start;
    size_t h_offset;
    size_t u_offset;
};

static Layout compute_layout(const vector<uint8_t>& bytes, uint32_t n) {
    Layout L{};
    L.dst_offset = 8 + 4 + 4 + 4;
    memcpy(&L.dst_len, &bytes[8 + 4 + 4], sizeof(uint32_t));
    L.size_offset = L.dst_offset + L.dst_len;
    L.g_start = L.size_offset + 4;
    L.h_offset = L.g_start + n * sizeof(G1Jacobian_t);
    L.u_offset = L.h_offset + sizeof(G1Jacobian_t);
    return L;
}

int main() {
    const uint N = 64;
    const string path = "/tmp/pp_phase15_test.bin";

    // ── Test 1: round-trip ───────────────────────────────────────────────
    {
        Commitment pp = Commitment::hiding_random(N);
        pp.save_hiding(path);

        Commitment loaded = Commitment::load_hiding(path);
        check(loaded.size == N,                  "size round-trips");
        check(loaded.is_hiding(),                "loaded pp is hiding");
        check(loaded.is_openable(),              "loaded pp is openable");
        check(loaded.verify_pp(ZKLLM_ENTROPY_PEDERSEN_DST_V1),
              "loaded pp verify_pp accepts");

        vector<G1Jacobian_t> ga(N), gb(N);
        cudaMemcpy(ga.data(), pp.gpu_data,     N * sizeof(G1Jacobian_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(gb.data(), loaded.gpu_data, N * sizeof(G1Jacobian_t),
                   cudaMemcpyDeviceToHost);
        check(memcmp(ga.data(), gb.data(), N * sizeof(G1Jacobian_t)) == 0,
              "G_i byte-identical after round-trip");
        check(memcmp(&pp.hiding_generator, &loaded.hiding_generator,
                     sizeof(G1Jacobian_t)) == 0,
              "H byte-identical after round-trip");
        check(memcmp(&pp.u_generator, &loaded.u_generator,
                     sizeof(G1Jacobian_t)) == 0,
              "U byte-identical after round-trip");
    }

    // Reference serialization for tamper tests.
    Commitment pp = Commitment::hiding_random(N);
    pp.save_hiding(path);
    vector<uint8_t> ref = slurp(path);
    Layout L = compute_layout(ref, N);

    // ── Test 2: tamper G_5 ───────────────────────────────────────────────
    {
        vector<uint8_t> bad = ref;
        const uint tamper_index = 5;
        bad[L.g_start + tamper_index * sizeof(G1Jacobian_t) + 0] ^= 0x01;
        write_bytes(path, bad);

        bool threw = false;
        string what;
        try { Commitment::load_hiding(path); }
        catch (const std::runtime_error& e) { threw = true; what = e.what(); }
        check(threw, "tampered G_5 rejected");
        // Error message should name index 5.  Don't hard-require substring
        // because the exact phrasing may evolve, but do sanity check.
        check(what.find("5") != string::npos,
              "tampered G_5 error mentions index 5");
    }

    // ── Test 3: tamper H ─────────────────────────────────────────────────
    {
        vector<uint8_t> bad = ref;
        bad[L.h_offset + 0] ^= 0x01;
        write_bytes(path, bad);

        bool threw = false;
        try { Commitment::load_hiding(path); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "tampered H rejected");
    }

    // ── Test 4: tamper U ─────────────────────────────────────────────────
    {
        vector<uint8_t> bad = ref;
        bad[L.u_offset + 0] ^= 0x01;
        write_bytes(path, bad);

        bool threw = false;
        try { Commitment::load_hiding(path); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "tampered U rejected");
    }

    // ── Test 5: truncation ──────────────────────────────────────────────
    {
        vector<uint8_t> bad(ref.begin(), ref.begin() + L.g_start + 16);
        write_bytes(path, bad);

        bool threw = false;
        try { Commitment::load_hiding(path); }
        catch (const std::runtime_error&) { threw = true; }
        check(threw, "truncated pp file rejected");
    }

    // ── Test 6: bad magic (v1-like blob) ─────────────────────────────────
    {
        // Simulate a legacy v1 file: raw G1TensorJacobian serialization,
        // no header.  First 8 bytes will not match "ZKEPP\0v2".
        vector<uint8_t> bad(ref.size(), 0xAB);
        write_bytes(path, bad);

        bool threw = false;
        string what;
        try { Commitment::load_hiding(path); }
        catch (const std::runtime_error& e) { threw = true; what = e.what(); }
        check(threw, "non-v2 file rejected");
        check(what.find("v2") != string::npos || what.find("ppgen") != string::npos,
              "non-v2 error steers the user at ppgen / v2");
    }

    // Restore the good file so subsequent test runs start clean.
    write_bytes(path, ref);

    cout << "All pp-format tests PASSED." << endl;
    return 0;
}
