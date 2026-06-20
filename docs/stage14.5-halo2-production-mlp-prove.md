# Stage 14.5 — halo2 production-MLP prover reachable from `omni-node`

**Status:** delivered as an additive feature on top of Stage 11d.2 + Stage 14.1. Closes the production-prover gap: Stage 11d.2 shipped the production verifier shell ([`Halo2ProductionMlpVerifier`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs)) with **no operator-reachable prover sibling**. Stage 14.5 adds the prover sibling, mirroring the Stage 14.1 pattern that did the same for the reference verifier.

## Scope

Pair the existing `omni-proofs-halo2-production-mlp` verifier (Stage 11d.2) with its already-in-tree prover sibling, expose the prover through a new `omni-node` feature `stage11d-production-prove`, and add a CLI subcommand `operator generate-production-mlp-proof` that produces a `ProofArtifactBody` JSON consumable by the existing `operator verify-proof` dispatch.

### Constraints honored

| Constraint | How |
| --- | --- |
| `crates/omni-zkml/src/error.rs` untouched | Existing `ProofBackendError::BackendInternal` covers all new validation failures. |
| `ProofSystem` enum untouched | Reuses the already-shipped `ProofSystem::Stage11dProductionFixedPointMlp` variant. |
| `ModelFormat` enum untouched | Reuses `ModelFormat::ProductionFixedPointMlp`. |
| Mainnet allowlist unchanged | `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` stays empty; **Stage 11d.3 separate track** for the chain-team-reviewed allowlist entry. |
| Verifier API untouched | `Halo2ProductionMlpVerifier::verify_artifact` consumes the new artifact byte-for-byte through its existing dispatch — recon verified the contract is satisfiable without expanding the verifier. |
| `ProofArtifactBody` schema untouched | All metadata fields the generator writes already exist (Stage 11b.0 + 11d.2). |
| `omni-sumchain` untouched | No chain RPCs added. |
| `omni-contributor` untouched | Stage 12.0 lean-crate invariant preserved. |
| `omni-proofs-halo2-reference` untouched | Stage 14.1 reference path unaffected. |
| No new `reason=` tag strings | The generator emits one new `event=halo2_production_mlp_proof_generated` line; failures use existing `OperatorError` variants. |
| Default build pulls zero halo2 / pasta / production-MLP deps | Existing Stage 11d.2 default-tree isolation guard covers (asserts `no omni-proofs-halo2-production-mlp`). |
| No live-chain CI tests | All Stage 14.5 tests are hermetic. |

## Critical contract diff vs Stage 14.1 (reference prover)

The Stage 14.5 production-MLP path mirrors Stage 14.1's structure but the artifact contract is **stricter and shaped differently**. A future refactor that conflates the two paths must regress visibly against the test suite.

| Aspect | Stage 14.1 reference | **Stage 14.5 production** |
| --- | --- | --- |
| `proof_system` | `Stage11bHalo2Reference` | `Stage11dProductionFixedPointMlp` |
| `model_format` | `Halo2ReferenceMlp` | `ProductionFixedPointMlp` |
| `backend_id` | `"halo2-reference-mlp-v1"` | `"production-fixedpoint-mlp-v1"` |
| `testnet_or_dev_only` | `Some(true)` | **`Some(false)`** ← production-shape contract |
| `circuit_id_hex` | Optional (None tolerated by verifier) | **Required**, must equal pinned `EXPECTED_CIRCUIT_ID_HEX = "593d027d…"` |
| `verification_key_hex` | None (verifier doesn't check) | **Required**, must equal pinned `EXPECTED_VK_HASH_HEX = "2ec18fae…"` |
| Mainnet refusal layers | 1 + 3 + 6 (three independent gates) | **6 only** (sole gate — empty allowlist) |
| Input shape | `[i16; 4]` (8 bytes LE) | `[i16; 16]` (32 bytes LE) |
| Output shape | `[i16; 4]` (8 bytes LE) | `[i16; 8]` (16 bytes LE) |
| `HALO2_K` | 10 (1 024 rows) | 11 (2 048 rows) |
| `PROVER_RNG_SEED` | `*b"OmniNode/Stage11b.1.b/prover-rng"` | `*b"OmniNode/Stage11d.2/prover-rngv1"` (distinct) |
| CI `RUST_MIN_STACK` | default | **64 MB** required (constraint-system walker needs headroom for the wider circuit) |

The production refusal landing at **layer 6 only** is the key Stage 11d.x design: production-shape artifacts pass layer 1 (they correctly declare `testnet_or_dev_only=Some(false)`), and the only thing keeping them off mainnet is the empty allowlist that Stage 11d.3 will fill via a chain-team-reviewed PR. Stage 14.5 does **not** modify the allowlist; the production prover ships under the existing empty-allowlist refusal.

## Surface map

| Surface | Stage 14.5 change |
| --- | --- |
| [`crates/omni-proofs-halo2-production-mlp/src/prover.rs`](../crates/omni-proofs-halo2-production-mlp/src/prover.rs) | New `Halo2ProductionMlpProofBackend` struct + `impl omni_zkml::ProofBackend`; 6 unit tests. |
| [`crates/omni-proofs-halo2-production-mlp/src/lib.rs`](../crates/omni-proofs-halo2-production-mlp/src/lib.rs) | Re-export `Halo2ProductionMlpProofBackend` behind `feature = "prove"`. |
| [`crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs`](../crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs) | New hermetic integration test: 4 cases (canonical roundtrip, artifact-level determinism, tamper-rejection, production-shape contract pin). Gated by `feature = "prove" + feature = "verify"`. |
| [`crates/omni-node/Cargo.toml`](../crates/omni-node/Cargo.toml) | New `stage11d-production-prove` feature: `["stage11d-production-verify", "omni-proofs-halo2-production-mlp/prove"]`. |
| [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) | New `OperatorCmd::GenerateProductionMlpProof` variant + `GenerateProductionMlpProofArgs` + `generate_production_mlp_proof_core` + `parse_input_16xi16` + 3 new `OperatorError` variants. New tracing label `generate-production-mlp-proof`. Six new test functions (4 parse + 1 end-to-end roundtrip + 1 input-parse refusal + 1 mainnet-posture pin + 1 coexistence pin). |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | New gate `stage11d-production-prove tree — must contain halo2_proofs + rand_chacha + production-mlp`; new build-test job `--features stage11d-production-prove` (with `RUST_MIN_STACK=67108864`). |
| [`docs/operator-runbook.md`](operator-runbook.md) | New Stage 14.5 sub-section. |
| This doc | Engineering doc. |

**Zero changes** to: `crates/omni-zkml`, `crates/omni-sumchain`, `crates/omni-contributor`, `crates/omni-proofs-halo2-reference`, mainnet allowlist, `ProofSystem` enum, `ModelFormat` enum, `ProofArtifactBody` schema, `Halo2ProductionMlpVerifier` API.

## Test inventory

### Unit tests (in `prover.rs`, gated `feature = "prove"`) — 6 new

1. `adapter_backend_id_and_proof_system_match_committed_constants` — pins trait answers.
2. `adapter_produces_proof_for_canonical_triple` — happy path + determinism cross-check against `prove_canonical`.
3. `adapter_refuses_when_model_bytes_do_not_match_spec_hash` — guards step 1 of validation.
4. `adapter_refuses_when_input_byte_length_is_wrong` — guards step 2 (16 × i16 = 32 bytes).
5. `adapter_refuses_when_output_does_not_match_canonical_evaluator` — guards step 3.
6. `adapter_proof_is_byte_deterministic_across_two_calls` — adapter does not introduce non-determinism.

Plus 2 pre-existing tests inside `prover.rs::tests` (`prove_canonical_produces_proof`, `prove_canonical_is_byte_deterministic`).

### Integration tests (gated `feature = "prove" + feature = "verify"`) — 4 new

In [`tests/halo2_production_mlp_prove_verify_roundtrip.rs`](../crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs):

1. `prove_then_verify_round_trip_succeeds_for_canonical_input` — closes the loop on the canonical input.
2. `adapter_proof_bytes_are_deterministic_across_two_artifact_builds` — full `ProofArtifactBody`-level determinism.
3. `tampered_proof_bytes_are_rejected_by_verifier` — sanity: tamper does not verify.
4. `artifact_carries_production_shape_testnet_or_dev_only_false` — **production-shape contract pin** (D-level invariant for Stage 14.5): asserts `testnet_or_dev_only=Some(false)`, `circuit_id_hex` and `verification_key_hex` equal the pinned constants. Distinguishes the Stage 14.5 contract from Stage 14.1.

### CLI tests (in `omni-node/src/operator.rs::tests`, gated `feature = "stage11d-production-prove"`) — 8 new

1. `parse_input_16xi16_accepts_canonical_value` — 16-value canonical string round-trips.
2. `parse_input_16xi16_tolerates_whitespace_around_separators` — whitespace tolerance.
3. `parse_input_16xi16_refuses_wrong_arity` — typed `ProductionMlpProofInputParse`.
4. `parse_input_16xi16_refuses_non_i16_value` — typed `ProductionMlpProofInputParse`.
5. `generate_production_mlp_proof_writes_artifact_that_verify_proof_accepts` — end-to-end CLI roundtrip.
6. `generate_production_mlp_proof_refuses_when_input_parse_fails_before_invoking_prover` — error path; no artifact written.
7. `generated_production_mlp_artifact_is_mainnet_refused_at_layer_6_only` — **mainnet-posture pin**: `testnet_or_dev_only=Some(false)` (layer 1 passes); `check_mainnet_eligible(meta).is_err()` (layer 6 fires).
8. `mock_artifact_verify_path_still_accepts_under_stage11d_production_prove_feature` — coexistence pin.

## CI gates added

1. **`stage11d-production-prove-build-test` job**: runs `cargo build -p omni-node --features stage11d-production-prove`, full `cargo test -p omni-proofs-halo2-production-mlp --features prove --features verify` (includes 6 new adapter tests + 4 integration tests + pre-existing 42+6 suite), and `cargo test -p omni-node --features stage11d-production-prove --bins`. All test runs use `RUST_MIN_STACK=67108864`.
2. **`stage11d-production-prove-tree-check` job**: asserts `cargo tree -p omni-node --features stage11d-production-prove` contains `halo2_proofs`, `rand_chacha`, AND `omni-proofs-halo2-production-mlp`. Catches silent feature wiring breaks. Uses bash glob matching to avoid the SIGPIPE-under-pipefail pattern that Stage 14.1 first hit.

Existing gates unchanged: default-tree (already isolates `omni-proofs-halo2-production-mlp` since Stage 11d.2), submit-tree, halo2-reference-verify, halo2-reference-prove, stage11d-production-verify, contributor, Stage 6 byte-stability, Stage 11a byte-stability.

## Mainnet posture

The Stage 14.5 production prover ships with the **same** mainnet refusal as Stage 11d.2's verifier: every artifact is refused on `chain_id == 1` because `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]`. Lifting that refusal requires:

1. **Stage 11d.3 review packet** — chain-team-reviewed mainnet eligibility criteria for `ProofSystem::Stage11dProductionFixedPointMlp`.
2. **Stage 11d.3 allowlist PR** — adds an `AllowlistEntry` with the pinned `circuit_id_hex` and `verification_key_hash_hex` already present in [`shared.rs`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs).
3. **External-cryptographer sign-off** (per the Stage 11d.0 criteria).

Stage 14.5 does **not** unblock or touch any of these. The production prover is shippable today under the existing testnet-or-dev posture; Stage 11d.3 lands separately whenever the chain-team-side review concludes.

## Future outlook

- **Stage 14.6+** — operator UX hardening: cross-input batch proving, per-circuit performance docs, regen-tool consolidation.
- **Stage 11d.3** — chain-team-reviewed mainnet allowlist entry. Separate track, not blocked by Stage 14.5.
- **Stage 14.x track end-state** — once both the reference and production halo2 paths are operator-reachable for prove + verify + contributor sidecar, Phase 5 transitions to the chain-side tokenomics work. The Stage 14.x track does **not** include staking / slashing / reward distribution — those require chain-team contract specs.

EZKL remains rejected (license/supply-chain posture unchanged from the 2026-05-22 spike); Stage 14.4 honestly reported that and pivoted to this production-MLP slice. Any future EZKL revisit requires upstream license changes.
