# Stage 11d.2 Review Packet — Production Fixed-Point MLP

**Status**: populated for Stage 11d.2 (first production-grade proof class candidate). This packet is the audit input for the SUM Chain review board ahead of the Stage 11d.3 allowlist PR. **It does not itself allowlist anything** — `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` stays empty through Stage 11d.2.

Sibling to [`stage11d-review-packet.md`](stage11d-review-packet.md) (the criteria template). Section numbering matches the template so each entry maps 1:1 to a criteria field. Fields marked `TBD` are explicitly deferred to Stage 11d.3 per the Stage 11d.2 plan §11 decisions #5 and #6.

---

## Packet metadata

| Field | Value |
|---|---|
| Packet author | OmniNode core (Stage 11d.2 PR) |
| Packet version | `0.1.0` (first revision; bumps on substantive content change) |
| Proposed proof system | `ProofSystem::Stage11dProductionFixedPointMlp` |
| Proposed `backend_id` | `production-fixedpoint-mlp-v1` |
| Proposed `circuit_id_hex` | `593d027df3778bc582f9ec40bf453e757a1be6a9b6961243f2dfdf38fb4ea95d` (bare BLAKE3 of `vk_canonical_bytes(vk)` — pinned in `shared::EXPECTED_CIRCUIT_ID_HEX`; verifier enforces drift) |
| Proposed `model_hash` | `EXPECTED_PRODUCTION_SPEC_HASH` (BLAKE3 of `crates/omni-proofs-halo2-production-mlp/assets/canonical_spec.json`; build-time-pinned via `build.rs`) |
| Proposed `verification_key_hash_hex` | `2ec18faed223a28a23155492459c507a2672b9ff495c1df566103a19638655a9` (`mainnet_vk_hash(vk_canonical_bytes(vk))` per criteria §1.7 — pinned in `shared::EXPECTED_VK_HASH_HEX`; verifier enforces drift) |
| Proposed `model_format` | `ModelFormat::ProductionFixedPointMlp` (off-chain proof metadata only) |
| Proposed `model_framework` | `ModelFramework::FrameworkAgnostic` (canonical spec is framework-neutral; RUMUS / PyTorch / TensorFlow / Caffe are equal-status primary compatibility targets) |
| `review_board_roster_ref` | `TBD` (Stage 11d.3) |
| `benchmark_record_ref` | [`stage11d.2-benchmark-record.md`](stage11d.2-benchmark-record.md) |
| Stage 11d.2 implementation PR | this PR |
| Target Stage 11d.3 allowlist PR | `TBD` |
| Date submitted to SUM Chain review board | `TBD` |
| Date of external cryptographer sign-off | `TBD` (deferred per Stage 11d.2 plan §11 decision #6) |
| Chain-team approval status | `Not yet submitted` |

---

## Chain compatibility clarification

Per Stage 11d.2 plan §0 and the chain-team correction:

- **No chain wire / Stage 7b tx / `InferenceAttestationDigest` / RPC / validator-side verification changes.** The on-chain footprint is unchanged: the existing `proof_root` commitment is the only chain-visible surface.
- `ModelFormat::ProductionFixedPointMlp` is **off-chain proof metadata only**. The chain never inspects this variant.
- The R9 optional chain digest+signature roundtrip is a developer-host scaffold (gated behind the `r9-chain-digest-roundtrip` feature, `#[ignore]` by default) that exercises the existing `canonical_digest_bytes` / `sign_chain_attestation_digest` primitives without touching chain wire or signing keys.
- **Mainnet posture**: `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` through 11d.2. Every artifact under this proof class is refused by `check_mainnet_eligible` layer 6.

---

## Workload one-liner

> Deterministic small-model classification.

Per Stage 11d.2 plan §4.1. The architecture (`16 → 32 → 16 → 8` with ReLU, int16 fixed-point) is sized to that workload: dense logits over an 8-class output domain with 16-dimensional integer feature inputs. The bounded weight ranges keep `|with_bias|` strictly under 2^23 for every i16-valued input, so no dense-layer output saturates and the gadget chain has full coverage.

---

## 1. Soundness (criteria 1.1)

### 1.1.1 Underlying scheme

- Scheme name: **halo2 (Pasta IPA, halo2_proofs 0.3.2)** — Zcash, MIT OR Apache-2.0.
- Published soundness analysis: halo2 book (Zcash); accumulator-based IPA from the original Halo paper.
- Security parameter (claimed): **≥ 128 bits** for the Pasta curves' IPA argument (Pallas / Vesta target ~128-bit security against DLOG; matches criteria §1.1's ≥ 128-bit requirement). External cryptographer sign-off in Stage 11d.3 is the gate that turns this claim into the criteria's S2 evidence.
- Adversary model: malicious prover, no shared secret, public coin. The verifier is deterministic given fixed params + VK.

### 1.1.2 OmniNode-specific soundness argument

- **Assumption A1**: `halo2_proofs 0.3.2` Pasta IPA argument is sound at ≥ 128 bits under the discrete-log assumption on Pallas / Vesta. (External cryptographer sign-off is the criteria §1.1 S2 gate; deferred to Stage 11d.3.)
- **Assumption A2**: `BLAKE3` is collision-resistant at ≥ 128 bits (used for `input_hash`, `response_hash`, `model_hash`, `proof_root`).
- **Invariant I1**: For every i16 public input, the gadget chain (dense identity → RHAZ → 3-branch saturation → ReLU sign-bit → bit-decomposition range checks) admits **exactly one** consistent witness assignment, producing `canonical_evaluate(input) == output`.
  - *RHAZ uniqueness*: the Euclidean division `abs_w + S/2 = q_abs · S + r_pos` with `r_pos ∈ [0, S)` is unique; tie cases (`with_bias = ±S/2 · (2k+1)`) are pinned to the canonical round-half-AWAY branch.
  - *Saturation uniqueness*: `b_lo + b_in + b_hi = 1` with branch-correctness aux witnesses deterministically maps `q_unsat` to `q_sat = saturate_i16(q_unsat)` regardless of in/out-of-range status.
  - *Range checks*: bit decomposition at widths 8u/15u/16s/16u/17u/23u rejects any slack absorption.
- **Claim C1**: A halo2 proof verified against the committed VK + the (input, output) instance column attests that `canonical_evaluate(input) == output` under the frozen `production-fixedpoint-mlp-v1 / spec_version: 1` numeric contract.
- **Claim C2**: The verifier rejects any artifact whose `metadata.public_inputs` does not BLAKE3-hash to `metadata.input_hash` / `metadata.response_hash` (steps 7 + 7.5 of the verify pipeline; see `crates/omni-proofs-halo2-production-mlp/src/verifier.rs`).
- **Defense in depth**: step 7.5 of the verifier independently runs `canonical_evaluate(input)` and refuses any artifact whose claimed output disagrees, *before* the SNARK verifier runs.

### 1.1.3 Adversarial regressions (in-tree)

- `tampered_q_abs_fails_range_check` — bumping `q_abs` and adjusting `r_pos` to keep the Euclidean equation balanced is rejected by the 8-bit range check on `r_pos`.
- `tampered_saturation_branch_fails` — forcing `b_hi = 1` when the canonical branch was `b_in` is rejected by the 17-bit range check on `hi_aux` and the output rule.
- `tampered_saturation_sum_fails` — setting both `b_lo` and `b_in` (sum = 2) fails the `sum_one` constraint.
- `wrong_output_fails_mock_prover` — any drift in the public output instance fails verification.
- `tie_input` semantics pinned via the `rhaz_helper_handles_pos/neg_half_tie` + `pos/neg_three_half_tie` unit tests.

### 1.1.4 External cryptographer review

`TBD` (Stage 11d.3 gate per Stage 11d.2 plan §11 decision #6).

### R4 reproducer

`TBD` (Stage 11d.3 gate per Stage 11d.2 plan §11 decision #5).

---

## 1.2 Determinism

- **Prover**: byte-deterministic. RNG is seeded with the fixed constant `PROVER_RNG_SEED = *b"OmniNode/Stage11d.2/prover-rngv1"` (`crates/omni-proofs-halo2-production-mlp/src/prover.rs`); verified by the `prove_canonical_is_byte_deterministic` unit test.
- **VK derivation**: deterministic given params + circuit definition; re-derived at verifier construction time via `keygen_vk(&params, &ProductionMlpCircuit::default())`.
- **Params**: `Params::<EqAffine>::new(HALO2_K)` produces byte-identical params for a given `HALO2_K = 11`.
- **Spec hash**: `EXPECTED_PRODUCTION_SPEC_HASH` is build-time-pinned from `assets/canonical_spec.json` via `build.rs`; any drift fails compilation.

---

## 1.3 Off-chain isolation

- The crate is **workspace-included** but **not in the default `omni-node` build graph**. Operator opt-in is via `cargo build -p omni-node --features stage11d-production-verify`.
- CI's `default-tree-check` job rejects any default build that pulls `omni-proofs-halo2-production-mlp` or `halo2_proofs`.
- The regen tool (`tools/halo2_production_mlp_regen/`) is a **standalone Cargo package** excluded from the workspace; the operator binary's compile graph cannot reach it transitively.
- The prover surface is gated behind the `prove` feature on the production crate; `--features verify` (CI / operator) pulls neither `rand_chacha` nor any prover code path.

---

## 1.4 Backward compatibility

- New `ProofSystem::Stage11dProductionFixedPointMlp` variant; new `ModelFormat::ProductionFixedPointMlp` variant. Both are inert in default builds.
- `Stage11bHalo2Reference` remains permanently testnet/dev-only and cannot be reused or promoted (criteria §1.4 hard rule).
- `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` and `MAINNET_APPROVED_PROOF_SYSTEMS = &[]`. No allowlist surface change in this PR.

---

## 1.5 Public-input binding

- The artifact's `metadata.public_inputs` JSON carries the raw i16 input + output arrays. The verifier rejects any artifact where `BLAKE3(LE(input))` ≠ `metadata.input_hash` or `BLAKE3(LE(output))` ≠ `metadata.response_hash`.
- The three-hash `PublicInputs` shape is insufficient; the verifier's `verify(&[u8], &PublicInputs)` returns `RequiresArtifactDispatch` to force callers through `verify_artifact(&ProofArtifactBody)`.

---

## 1.6 Distinguishability hard rules

- **H1 (separate verifier struct)**: `Halo2ProductionMlpVerifier` is a distinct type from `Halo2ReferenceVerifier` and lives in a separate crate.
- **H2 (separate spec-hash const)**: `EXPECTED_PRODUCTION_SPEC_HASH` (not `EXPECTED_SPEC_HASH`); name collision is impossible.
- **H3 (distinct canonical spec)**: `crates/omni-proofs-halo2-production-mlp/assets/canonical_spec.json`, distinct from the reference's spec.

---

## 1.7 VK hash scheme

- Computed via `omni_zkml::mainnet_vk_hash(canonical_vk_bytes)` = `BLAKE3("OMNINODE-VK:v1:" || canonical_vk_bytes)` per criteria §1.7. Recorded as `verification_key_hash_hex` on the eventual Stage 11d.3 `AllowlistEntry` and as `metadata.verification_key_hex` on every Stage 11d.2 proof artifact (audit field; pinned in `shared::EXPECTED_VK_HASH_HEX`).
- `canonical_vk_bytes` for halo2_proofs 0.3.2 is `format!("{:?}", vk.pinned()).into_bytes()` — the library-blessed canonical representation, mirroring how halo2_proofs derives `transcript_repr` internally (`halo2_proofs/src/plonk.rs` `VerifyingKey::from_parts`). Byte-stable given a fixed `halo2_proofs` version + circuit definition + `HALO2_K`. See `verifier::vk_canonical_bytes`.
- Drift detection: `Halo2ProductionMlpVerifier::from_params_bytes` rejects construction if the live VK's `(circuit_id_hex, vk_hash_hex)` pair does not equal the pinned constants. A `halo2_proofs` version bump requires an explicit fixture-regen PR.

---

## 2. Performance

See [`stage11d.2-benchmark-record.md`](stage11d.2-benchmark-record.md) for numbers and methodology.

Approved targets (Stage 11d.2 plan §11 decision #3):
- **Verifier**: p95 ≤ 1000 ms.
- **Prover**: < 60 s release-mode on dev host.
- **HALO2_K ceiling**: `k ≤ 13`; halt and report at `k ≥ 14`.

Current state: `HALO2_K = 11`; `params.bin = 131_140 bytes`; `proof.bin = 7_744 bytes`. All within budget.

---

## 3. Outstanding gates before Stage 11d.3 PR

| Gate | Owner | Status |
|---|---|---|
| External cryptographer review | proposing engineer | `TBD` (decision #6) |
| R4 reproducer document | proposing engineer | `TBD` (decision #5) |
| Chain-team digest/signature roundtrip exercise (full sign-off, not scaffold) | proposing engineer + chain team | `TBD` (R9 sign-off is 11d.3) |
| `review_board_roster_ref` | proposing engineer | `TBD` |
| `verification_key_hash_hex` | proposing engineer | **Pinned at Stage 11d.2** in `shared::EXPECTED_VK_HASH_HEX` (`2ec18faed223a28a23155492459c507a2672b9ff495c1df566103a19638655a9`). Stage 11d.3 PR copies into the `AllowlistEntry`. |
| `circuit_id_hex` | proposing engineer | **Pinned at Stage 11d.2** in `shared::EXPECTED_CIRCUIT_ID_HEX` (`593d027df3778bc582f9ec40bf453e757a1be6a9b6961243f2dfdf38fb4ea95d`). Stage 11d.3 PR copies into the `AllowlistEntry`. |
