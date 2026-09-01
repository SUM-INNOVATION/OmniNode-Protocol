# R0 closure record — candidate benchmark evidence (non-production)

Closure artifact for **SUM-INNOVATION/OmniNode-Protocol#101** (R0 research benchmark: candidate SP1 & RISC Zero guests + bounded transformer unit + `SelectToken` + object/manifest commitments). Machine-auditable companion: [`r0-closure.v1.json`](./r0-closure.v1.json), bound in CI by `tools/r0-zkvm-bench/tests/r0_closure_record.rs`.

**Research only. No production or integration claim.** No new proofs were run and no frozen rule was changed to produce this record; it binds the finished official B0-FINAL measurement.

## Bound official evidence (complete, untruncated)

| Field | Value |
|---|---|
| Measurement head (sum-chain main) | `9cccaa5ee6e038fb9dcb45af44ecb3cbdc2f48c6` |
| B0-PRE spec hash | `e933e7325c2639a48d8e25f20746d0f8abc822dee9fcfa87c2e6cdec226cf2a2` |
| Official deterministic seal (BLAKE3) | `60ace32cc2775fd38c3a4b9ea81f49686121cdd25a38db7a5ca5a0f4580bd600` |
| VEC9 package id | `80ab5ecfbe7a24d96d02dad78db2e4aee712ea0b29a071c19893b0d83dd0f11b` |
| Records-derived guest set | `11d059c7dbc37b3d80f0a0c1fcaee96ad5e0ba1916ba08bdb0f37e1a7d76401a` |
| SP1/x86_64 program_id / toolchain | `001c9c55…fad6` / `4367170f…5f5c` |
| SP1/aarch64 program_id (identity-only) | `001c9c55…fad6` |
| RISC0/x86_64 program_id / image | `f09c454a…b6cb` / `9945fcb4…82b3` |

Durable, read-back-verified evidence: sum-chain release [`b0-final-evidence-60ace32c`](https://github.com/SUM-INNOVATION/sum-chain/releases/tag/b0-final-evidence-60ace32c) (tag → `9cccaa5e…`), holding all 18 official members; the canonical manifest + `CHECKSUMS.txt` are committed alongside it in sum-chain `docs/b0-final/`.

## CORRECTION / SUPERSESSION (owner ruling — also posted on #101)

1. **Architecture.** The obsolete AC "both candidate guests prove on **both architectures**" is superseded by the ratified **two-cell** model: terminal native measurement is **x86_64-only** for both candidates (SP1/x86_64 + RISC0/x86_64). **SP1/aarch64** is **identity-only reconciliation**; **SP1/aarch64 terminal Groth16** is **ratified-unsupported** (no first-party linux/arm64 gnark backend); **RISC0/aarch64** is **unsupported/refused**. ARM identity reconciliation was **required and satisfied**; ARM performance was **never measured or implied**.
2. **Golden-fixture type names.** `StateObjectV1` / `UnitOutputManifestV1` are obsolete; the ratified types are **`ObjectCommitmentV1` / `OutputManifestV1` / `InputManifestV1`** (`sumchain-wire::b0`).

## Acceptance criteria (corrected) — confirmation

- **Both eligible x86_64 cells completed** (SP1/x86_64 + RISC0/x86_64); the bounded transformer-layer-group unit and `SelectToken` were proven and verified (fragments `113958582e…` / `b6090b05…`).
- **Statement / program / lock / toolchain bindings** recorded (records-authoritative, per-candidate).
- **Golden roots present + hash-bound.** The five cases (empty object, single-chunk object, two-slot output manifest, three-slot input manifest, three-chunk/chunk-boundary root) are committed at `sum-chain docs/b0-pre/fixtures/encoding-golden/vectors.json` (SHA-256 `26a6338e3572384adfc4e0aa379f4501cb1c350a4195ce85f8056b2f378875c1`) and cross-validated byte-for-byte against the SNIP producer in `Storage-Node-Interface-Protocol/crates/sum-store/tests/object_manifest_golden.rs`. The host reference executor's object/manifest commitments are covered by `tools/r0-zkvm-bench/tests/guest_output_equivalence.rs`.
- **Malformed / wrong-statement proofs deterministically refused** — `tools/r0-zkvm-bench/tests/adversarial_binding.rs` + `verifier_matrix.rs`.
- **Determinism** — `tools/r0-zkvm-bench/tests/guest_output_equivalence.rs`.
- **Raw metrics + reproducibility retrievable and hash-bound** — via the durable release + committed manifest.
- **No production/integration claim.**

## Closure

Tracked under sub-issue #113 (sibling: sum-chain#194). #101 closes only after this PR merges and the final line-by-line administrative re-audit passes.
