// Tests for zkConditionalEntropy.
// Build: add to Makefile as a target.
// Run: ./test_zkentropy

#include "zkentropy.cuh"
#include <iostream>
#include <cmath>

using namespace std;

static void check(bool cond, const char* msg) {
    if (!cond) { cerr << "FAIL: " << msg << endl; exit(1); }
    cout << "PASS: " << msg << endl;
}

// Build a vocab_size tensor where logits[winner] is high and all others are low.
static FrTensor make_logits(uint vocab_size, uint winner, long winner_val, long other_val) {
    vector<long> vals(vocab_size, other_val);
    vals[winner] = winner_val;
    Fr_t* cpu = new Fr_t[vocab_size];
    for (uint i = 0; i < vocab_size; i++) {
        unsigned long uv = (unsigned long)vals[i];
        cpu[i] = {(uint)(uv & 0xFFFFFFFF), (uint)(uv >> 32), 0, 0, 0, 0, 0, 0};
    }
    FrTensor t(vocab_size, cpu);
    delete[] cpu;
    return t;
}

int main() {
    const uint vocab_size    = 32;   // small vocab for testing
    const uint bit_width     = 16;
    const uint cdf_precision = 16;
    const uint log_precision = 5;    // 32 entries; 32 % 32 == 0 ✓
    const uint cdf_scale     = 1u << 16;
    const uint log_scale     = 1u << 16;
    const double sigma_eff   = 500.0;  // noise in field integer units

    zkConditionalEntropy prover(vocab_size, bit_width, cdf_precision, log_precision,
                                cdf_scale, log_scale, sigma_eff);

    // ── Test 1: argmax correctly identifies winner ─────────────────────────
    {
        auto logits = make_logits(vocab_size, /*winner=*/5, 1000L, 100L);
        uint t = prover.argmax_prover.compute(logits);
        check(t == 5, "argmax identifies correct winner");
    }

    // ── Test 2: greedy token (max prob) → low surprise ────────────────────
    //    Winner has much higher logit → win_prob[winner] ≈ sum(win_probs)
    //    → q[winner] ≈ 1 → surprise ≈ 0
    {
        auto logits = make_logits(vocab_size, 5, 5000L, 100L);
        Fr_t s = prover.computePosition(logits, /*actual_token=*/5);
        unsigned long sv = ((unsigned long)s.val[1] << 32) | s.val[0];
        double surprise_bits = (double)sv / log_scale;
        cout << "  greedy surprise = " << surprise_bits << " bits" << endl;
        check(surprise_bits < 2.0, "greedy token has surprise < 2 bits");
    }

    // ── Test 3: unlikely token → high surprise ────────────────────────────
    //    Winner has much higher logit but we claim an unlikely token was chosen.
    {
        auto logits = make_logits(vocab_size, 5, 5000L, 100L);
        // actual_token=20 has low win_prob → high surprise
        Fr_t s = prover.computePosition(logits, /*actual_token=*/20);
        unsigned long sv = ((unsigned long)s.val[1] << 32) | s.val[0];
        double surprise_bits = (double)sv / log_scale;
        cout << "  unlikely token surprise = " << surprise_bits << " bits" << endl;
        check(surprise_bits > 1.0, "unlikely token has surprise > 1 bit");
    }

    // ── Test 4: uniform logits → moderate surprise ────────────────────────
    //    All logits equal → win_probs ≈ equal → q ≈ 1/vocab_size
    //    → surprise ≈ log2(vocab_size)
    {
        Fr_t one_val = {1000, 0, 0, 0, 0, 0, 0, 0};
        // Build uniform logits
        Fr_t* cpu = new Fr_t[vocab_size];
        for (uint i = 0; i < vocab_size; i++) cpu[i] = one_val;
        FrTensor logits(vocab_size, cpu);
        delete[] cpu;

        Fr_t s = prover.computePosition(logits, /*actual_token=*/0);
        unsigned long sv = ((unsigned long)s.val[1] << 32) | s.val[0];
        double surprise_bits = (double)sv / log_scale;
        double expected = log2((double)vocab_size);
        cout << "  uniform surprise = " << surprise_bits << " bits (expected ≈ "
             << expected << ")" << endl;
        check(fabs(surprise_bits - expected) < 2.0,
              "uniform distribution surprise ≈ log2(vocab_size)");
    }

    // ── Test 5: compute() sums per-position surprises ────────────────────
    {
        vector<FrTensor> seq;
        // Use one greedy token (0 surprise) and one unlikely token (high surprise)
        // so the total is non-zero.
        seq.push_back(make_logits(vocab_size, 5, 5000L, 100L));
        seq.push_back(make_logits(vocab_size, 3, 4000L, 200L));
        vector<uint> tokens = {5, 20};  // token 20 is unlikely at position 1

        Fr_t total = prover.compute(seq, tokens);
        unsigned long tv = ((unsigned long)total.val[1] << 32) | total.val[0];
        check(tv > 0, "compute() returns non-zero total entropy for unlikely token");
    }

    // ── Test 6: prove() validates consistency ────────────────────────────
    {
        vector<FrTensor> seq;
        seq.push_back(make_logits(vocab_size, 5, 5000L, 100L));
        vector<uint> tokens = {5};

        Fr_t claimed = prover.compute(seq, tokens);
        vector<Polynomial> proof;
        // Should not throw.
        bool ok = true;
        try {
            prover.prove(seq, tokens, claimed, proof);
        } catch (const std::exception& e) {
            cerr << "  prove threw: " << e.what() << endl;
            ok = false;
        }
        check(ok, "prove() does not throw for correct claimed entropy");
    }

    cout << "\nAll zkConditionalEntropy tests passed." << endl;
    return 0;
}
