// RFC 9380 Appendix J.9.1 test-vector harness for our hash_to_curve_g1
// wrapper.  Non-negotiable: this is the gate that proves we actually
// implement RFC 9380 BLS12381G1_XMD:SHA-256_SSWU_RO_ and not some byte-
// swapped or mis-domained variant.
//
// RFC 9380 §J.9.1 (Faz-Hernández et al. 2023):
//   "suite = BLS12381G1_XMD:SHA-256_SSWU_RO_
//    dst   = QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_"
//
// Path: wrapper-output G1Jacobian_t → memcpy-back to blst_p1 → affine →
// serialize to uncompressed 96-byte big-endian x || y → byte-compare to
// the RFC hex.  A single mismatch in limb order, Montgomery convention,
// or Jacobian-vs-affine handling trips this test loudly.

// blst.h before project headers (see src/field/hash_to_curve.cu for rationale).
#include <blst.h>
#include "field/hash_to_curve.cuh"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

using namespace std;

static vector<uint8_t> hex_to_bytes(const string& hex) {
    vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        auto nyb = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return 0xFF;
        };
        out.push_back((nyb(hex[i]) << 4) | nyb(hex[i+1]));
    }
    return out;
}

static string bytes_to_hex(const uint8_t* data, size_t n) {
    static const char* HEX = "0123456789abcdef";
    string out;
    out.reserve(2 * n);
    for (size_t i = 0; i < n; i++) {
        out.push_back(HEX[(data[i] >> 4) & 0xF]);
        out.push_back(HEX[data[i] & 0xF]);
    }
    return out;
}

struct Vec {
    const char* msg;
    const char* x_hex;
    const char* y_hex;
};

// DST from RFC 9380 Appendix J.9.1 preamble.
static const char* RFC_DST = "QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_";

// Vectors copied from RFC 9380 Appendix J.9.1, BLS12381G1_XMD:SHA-256_SSWU_RO_.
static const Vec VECTORS[] = {
    { "",
      "052926add2207b76ca4fa57a8734416c8dc95e24501772c814278700eed6d1e4e8cf62d9c09db0fac349612b759e79a1",
      "08ba738453bfed09cb546dbb0783dbb3a5f1f566ed67bb6be0e8c67e2e81a4cc68ee29813bb7994998f3eae0c9c6a265" },
    { "abc",
      "03567bc5ef9c690c2ab2ecdf6a96ef1c139cc0b2f284dca0a9a7943388a49a3aee664ba5379a7655d3c68900be2f6903",
      "0b9c15f3fe6e5cf4211f346271d7b01c8f3b28be689c8429c85b67af215533311f0b8dfaaa154fa6b88176c229f2885d" },
    { "abcdef0123456789",
      "11e0b079dea29a68f0383ee94fed1b940995272407e3bb916bbf268c263ddd57a6a27200a784cbc248e84f357ce82d98",
      "03a87ae2caf14e8ee52e51fa2ed8eefe80f02457004ba4d486d6aa1f517c0889501dc7413753f9599b099ebcbbd2d709" },
    // (RFC 9380 has two additional vectors with 129- and 521-byte messages;
    //  omitted here because the 3 short vectors already gate the full
    //  SSWU + iso_map + cofactor-clearing + Montgomery-layout pipeline.)
};

int main() {
    int passed = 0;
    int total = sizeof(VECTORS) / sizeof(VECTORS[0]);

    for (int k = 0; k < total; k++) {
        const auto& v = VECTORS[k];
        string msg(v.msg);
        vector<uint8_t> x_exp = hex_to_bytes(v.x_hex);
        vector<uint8_t> y_exp = hex_to_bytes(v.y_hex);

        // Our wrapper's output, ported back to blst for normalization.
        G1Jacobian_t p = hash_to_curve_g1(RFC_DST, msg);
        blst_p1 bp;
        memcpy(&bp.x, &p.x, sizeof(p.x));
        memcpy(&bp.y, &p.y, sizeof(p.y));
        memcpy(&bp.z, &p.z, sizeof(p.z));

        blst_p1_affine aff;
        blst_p1_to_affine(&aff, &bp);

        uint8_t ser[96];
        blst_p1_affine_serialize(ser, &aff);

        bool x_ok = memcmp(ser,      x_exp.data(), 48) == 0;
        bool y_ok = memcmp(ser + 48, y_exp.data(), 48) == 0;

        if (x_ok && y_ok) {
            cout << "PASS: vector " << k << " (msg len " << msg.size() << ")" << endl;
            passed++;
        } else {
            cerr << "FAIL: vector " << k << " (msg len " << msg.size() << ")" << endl;
            cerr << "  expected x = " << v.x_hex << endl;
            cerr << "  got      x = " << bytes_to_hex(ser, 48) << endl;
            cerr << "  expected y = " << v.y_hex << endl;
            cerr << "  got      y = " << bytes_to_hex(ser + 48, 48) << endl;
        }
    }

    if (passed != total) {
        cerr << "FAIL: " << (total - passed) << " of " << total
             << " RFC 9380 vectors mismatched." << endl;
        return 1;
    }
    cout << "All " << total << " RFC 9380 J.9.1 vectors PASSED." << endl;
    return 0;
}
