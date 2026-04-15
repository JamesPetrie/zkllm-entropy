// blst.h must be included before any project header: the project's
// g1-tensor.cuh pulls `using namespace std` into the global namespace,
// and C++17's std::byte then collides with blst's global `typedef
// unsigned char byte;`.  Including blst first lets its declarations
// resolve `byte` to the C typedef before `using namespace std` drags
// in std::byte.
#include <blst.h>
#include "field/hash_to_curve.cuh"
#include <cstring>
#include <stdexcept>

// blst_fp is {limb_t l[6]} with limb_t = uint64_t, stored in Montgomery
// form with R = 2^384 mod p (see blst/src/exports.c:195 `blst_fp_to`
// calls `mul_fp(ret, a, BLS12_381_RR)`).
//
// Codebase Fp_t (blstrs__fp__Fp) is {uint val[12]} with uint = uint32_t,
// also Montgomery form with R = 2^384 mod p (same R, since both
// representations pack 384 bits total).
//
// On a little-endian host the byte layouts are identical — 12 × 32-bit
// little-endian limbs are the same bytes as 6 × 64-bit little-endian
// limbs.  So `memcpy(sizeof=48)` faithfully transfers the Montgomery
// representation.  The `test_hash_to_curve_rfc9380` harness catches
// any layout mismatch by comparing against RFC 9380 Appendix J.9.1
// affine (x, y) hex vectors.
static_assert(sizeof(blst_fp) == sizeof(Fp_t),
              "blst_fp and Fp_t must have identical byte size");

static G1Jacobian_t blst_p1_to_G1Jacobian(const blst_p1& p)
{
    G1Jacobian_t out;
    std::memcpy(&out.x, &p.x, sizeof(Fp_t));
    std::memcpy(&out.y, &p.y, sizeof(Fp_t));
    std::memcpy(&out.z, &p.z, sizeof(Fp_t));
    return out;
}

G1Jacobian_t hash_to_curve_g1(const std::string& dst,
                              const std::vector<uint8_t>& msg)
{
    blst_p1 p;
    blst_hash_to_g1(&p,
                    msg.data(), msg.size(),
                    reinterpret_cast<const uint8_t*>(dst.data()), dst.size(),
                    nullptr, 0);
    return blst_p1_to_G1Jacobian(p);
}

G1Jacobian_t hash_to_curve_g1(const std::string& dst,
                              const std::string& msg)
{
    std::vector<uint8_t> bytes(msg.begin(), msg.end());
    return hash_to_curve_g1(dst, bytes);
}

std::vector<uint8_t> msg_for_G_index(uint32_t i)
{
    std::vector<uint8_t> out;
    out.reserve(2 + 4);
    out.push_back('G');
    out.push_back('_');
    out.push_back(static_cast<uint8_t>((i >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((i >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((i >>  8) & 0xFF));
    out.push_back(static_cast<uint8_t>( i        & 0xFF));
    return out;
}

std::vector<uint8_t> msg_for_H() { return {'H'}; }
std::vector<uint8_t> msg_for_U() { return {'U'}; }

HashedGenerators hash_to_curve_generators(const std::string& dst, uint32_t size)
{
    HashedGenerators out;
    out.G.reserve(size);
    for (uint32_t i = 0; i < size; i++) {
        out.G.push_back(hash_to_curve_g1(dst, msg_for_G_index(i)));
    }
    out.H = hash_to_curve_g1(dst, msg_for_H());
    out.U = hash_to_curve_g1(dst, msg_for_U());
    return out;
}
