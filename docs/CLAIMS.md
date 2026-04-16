# Claims Index

Auto-generated from `docs/proof-layer-analysis.md` by `tools/check_claims.py`.
Do not edit by hand.

| ID | Section | Status | Statement |
|---|---|---|---|
| `C-CRY-DL-G1` | [`§`](proof-layer-analysis.md#25-claims) | justified | Discrete logarithm is hard in BLS12-381 G1 at the λ ≈ 128 security level. |
| `C-PED-BIND` | [`§`](proof-layer-analysis.md#25-claims) | justified | Pedersen commitment is computationally binding in the random-oracle model. |
| `C-PED-COMPLETE` | [`§`](proof-layer-analysis.md#25-claims) | justified | Pedersen commitment scheme is complete — honest Commit produces a valid commitment for every message and blinding. |
| `C-PED-HIDE` | [`§`](proof-layer-analysis.md#25-claims) | justified | Pedersen commitment is perfectly hiding — the commitment distribution is identical under any message, taken over the blinding. |
| `C-PED-HOMO` | [`§`](proof-layer-analysis.md#25-claims) | justified | Pedersen commitment is additively homomorphic — C(t1;ρ1) + C(t2;ρ2) = C(t1+t2; ρ1+ρ2). |
| `C-PED-SETUP-NO-DLOG` | [`§`](proof-layer-analysis.md#25-claims) | justified | Generators {G_i}, H, U are derived by RFC 9380 hash-to-curve, so no party knows any pairwise discrete log. |
