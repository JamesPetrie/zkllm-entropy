# Claims Index

Auto-generated from `docs/proof-layer-analysis.md` by `tools/check_claims.py`.
Do not edit by hand.

| ID | Section | Status | Statement |
|---|---|---|---|
| `C-CRY-DL-G1` | [`§`](proof-layer-analysis.md#25-claims) | justified | Discrete logarithm is hard in BLS12-381 G1 at the λ ≈ 128 security level. |
| `C-OPEN-COMPLETE` | [`§`](proof-layer-analysis.md#45-claims) | justified | Hyrax §A.2 Figure 6 opening protocol is complete — honest prover's four verification checks all pass. |
| `C-OPEN-HVZK` | [`§`](proof-layer-analysis.md#45-claims) | justified | Hyrax §A.2 Figure 6 opening is perfect honest-verifier zero-knowledge — the simulator picks z, z_δ, z_β uniformly and computes δ, β from the verification equations; the distribution matches a real transcript because d is a uniform mask. |
| `C-OPEN-SOUND` | [`§`](proof-layer-analysis.md#45-claims) | justified | Hyrax §A.2 Figure 6 opening is special-sound — from two accepting transcripts with c ≠ c', the extractor recovers the witness vector, blinding, and the committed evaluation. |
| `C-PED-BIND` | [`§`](proof-layer-analysis.md#25-claims) | justified | Pedersen commitment is computationally binding in the random-oracle model. |
| `C-PED-COMPLETE` | [`§`](proof-layer-analysis.md#25-claims) | justified | Pedersen commitment scheme is complete — honest Commit produces a valid commitment for every message and blinding. |
| `C-PED-HIDE` | [`§`](proof-layer-analysis.md#25-claims) | justified | Pedersen commitment is perfectly hiding — the commitment distribution is identical under any message, taken over the blinding. |
| `C-PED-HOMO` | [`§`](proof-layer-analysis.md#25-claims) | justified | Pedersen commitment is additively homomorphic — C(t1;ρ1) + C(t2;ρ2) = C(t1+t2; ρ1+ρ2). |
| `C-PED-SETUP-NO-DLOG` | [`§`](proof-layer-analysis.md#25-claims) | justified | Generators {G_i}, H, U are derived by RFC 9380 hash-to-curve, so no party knows any pairwise discrete log. |
| `C-SC-ZK-COMPLETE` | [`§`](proof-layer-analysis.md#58-claims) | justified | ZK sumcheck (Hyrax §4 Protocol 3) is complete — honest commitments satisfy the round-to-round identity check by Pedersen homomorphism. |
| `C-SC-ZK-FORMAL` | [`§`](proof-layer-analysis.md#58-claims) | open (blocked) | A written formal simulator exists that, given only public inputs and the commitment oracle, produces a transcript distribution computationally indistinguishable from the real protocol — following the Real-vs-Ideal structure of zkLLM Theorem 7.4. |
| `C-SC-ZK-INFORMAL` | [`§`](proof-layer-analysis.md#58-claims) | justified | ZK sumcheck hides the round polynomials under Pedersen perfect hiding, and the Σ-protocol responses are HVZK — so the transcript informally reveals nothing beyond the public claim S and the verifier's challenges. This is the claim supported by Hyrax Lemma 4 plus the §5.2 informal argument; a full formal simulator is tracked separately as C-SC-ZK-FORMAL. |
| `C-SC-ZK-SOUND` | [`§`](proof-layer-analysis.md#58-claims) | justified | ZK sumcheck has total soundness error ≤ n(2d+2)/\|F\| — composition of per-coefficient opening soundness (Σ-protocol), round-to-round equality soundness (Σ-protocol), and extracted-coefficient sumcheck soundness (Schwartz-Zippel). |
| `C-SIGMA-EQ-COMPLETE` | [`§`](proof-layer-analysis.md#34-claims) | justified | Hyrax §A.1 proof-of-equality is complete. |
| `C-SIGMA-EQ-HVZK` | [`§`](proof-layer-analysis.md#34-claims) | justified | Hyrax §A.1 proof-of-equality is perfect honest-verifier zero-knowledge. |
| `C-SIGMA-EQ-SOUND` | [`§`](proof-layer-analysis.md#34-claims) | justified | Hyrax §A.1 proof-of-equality is special-sound. |
| `C-SIGMA-OPEN-COMPLETE` | [`§`](proof-layer-analysis.md#34-claims) | justified | Hyrax §A.1 Figure 5 proof-of-opening is complete — honest prover passes verification. |
| `C-SIGMA-OPEN-HVZK` | [`§`](proof-layer-analysis.md#34-claims) | justified | Hyrax §A.1 proof-of-opening is perfect honest-verifier zero-knowledge — the simulator outputs a transcript identically distributed to the real one. |
| `C-SIGMA-OPEN-SOUND` | [`§`](proof-layer-analysis.md#34-claims) | justified | Hyrax §A.1 proof-of-opening is special-sound — two accepting transcripts with distinct challenges extract the witness. |
