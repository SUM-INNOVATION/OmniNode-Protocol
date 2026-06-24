# Stage 14.1 — halo2-reference prover reachable from `omni-node`

**Status:** delivered as an additive feature on top of Stage 11b.1.b. First end-to-end real proof loop in the operator binary; closes the Phase 5 prover-side gap that the Stage 14.0 design packet identified.

## Scope

Pair the existing `omni-proofs-halo2-reference` verifier (Stage 11b.1.b) with its already-in-tree prover sibling, expose the prover through a new `omni-node` feature, and add a CLI subcommand that produces a `ProofArtifactBody` JSON consumable by the existing `operator verify-proof` dispatch.

### Constraints honored

| Constraint | How |
| --- | --- |
| `crates/omni-zkml/src/error.rs` untouched | Existing `ProofBackendError::BackendInternal` covers the new validation failures. |
| `ProofSystem` enum untouched | Reuses the already-shipped `ProofSystem::Stage11bHalo2Reference` variant. |
| Mainnet eligibility registry unchanged | `MAINNET_APPROVED_PROOF_SYSTEMS` stays empty; the new artifact sets `testnet_or_dev_only = Some(true)`. |
| Verifier API untouched | `Halo2ReferenceVerifier::verify_artifact` consumes the new artifact byte-for-byte through its existing dispatch. |
| `ProofArtifactBody` schema untouched | All metadata fields the generator writes already exist (Stage 11b.0 + 11b.1.a). |
| `omni-sumchain` untouched | No chain RPCs added; submission flow unchanged. |
| No new `reason=` tag strings | The generator emits one new `event=halo2_reference_proof_generated` line; failures use existing `OperatorError` variants. |
| Default build pulls zero halo2 / pasta / `omni-proofs-halo2-reference` deps | Two CI tree gates (existing Stage 11b.1.b `no halo2_proofs` + new Stage 14.1 `no omni-proofs-halo2-reference`) enforce this. |
| No live-chain CI tests | All Stage 14.1 tests are hermetic. |

## Surface map

| Surface | Stage 14.1 change |
| --- | --- |
| [`crates/omni-proofs-halo2-reference/src/prover.rs`](../crates/omni-proofs-halo2-reference/src/prover.rs) | New `Halo2ReferenceProofBackend` struct + `impl omni_zkml::ProofBackend`; 6 unit tests. |
| [`crates/omni-proofs-halo2-reference/src/lib.rs`](../crates/omni-proofs-halo2-reference/src/lib.rs) | Re-export `Halo2ReferenceProofBackend` behind `feature = "prove"`. |
| [`crates/omni-proofs-halo2-reference/tests/halo2_reference_prove_verify_roundtrip.rs`](../crates/omni-proofs-halo2-reference/tests/halo2_reference_prove_verify_roundtrip.rs) | New hermetic integration test: 4 cases (canonical + non-canonical roundtrip, determinism, tamper). Gated by `feature = "prove"` + `feature = "verify"`. |
| [`crates/omni-node/Cargo.toml`](../crates/omni-node/Cargo.toml) | New `halo2-reference-prove` feature: `["halo2-reference-verify", "omni-proofs-halo2-reference/prove"]`. |
| [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) | New `OperatorCmd::GenerateReferenceProof` variant + `GenerateReferenceProofArgs` + `generate_reference_proof_core` + three new `OperatorError` variants (all gated). New tracing label `generate-reference-proof`. Six new test functions (4 parse-input unit tests + 1 roundtrip end-to-end + 1 coexistence pin). |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | New gate `halo2-reference-prove tree — must contain halo2_proofs + rand_chacha`; new build-test job `--features halo2-reference-prove`. New default-tree isolation guard `no omni-proofs-halo2-reference`. |
| [`docs/operator-runbook.md`](operator-runbook.md) | New section "Phase 5 — real proof generation" with Stage 14.1 entry. |
| This doc | Engineering doc. |

## Test inventory

### Unit tests (in `prover.rs`, gated `feature = "prove"`)

1. `prove_canonical_produces_proof` — pre-existing.
2. `prove_canonical_is_byte_deterministic` — pre-existing.
3. `adapter_backend_id_and_proof_system_match_committed_constants` — pins trait answers.
4. `adapter_produces_proof_for_canonical_triple` — happy path + determinism cross-check against `prove_canonical`.
5. `adapter_refuses_when_model_bytes_do_not_match_spec_hash` — guards step 1 of validation.
6. `adapter_refuses_when_input_byte_length_is_wrong` — guards step 2 of validation.
7. `adapter_refuses_when_output_does_not_match_canonical_evaluator` — guards step 3 of validation.
8. `adapter_proof_is_byte_deterministic_across_two_calls` — adapter does not introduce non-determinism in the wrapping.

### Integration tests (gated `feature = "prove" + feature = "verify"`)

In [`tests/halo2_reference_prove_verify_roundtrip.rs`](../crates/omni-proofs-halo2-reference/tests/halo2_reference_prove_verify_roundtrip.rs):

1. `prove_then_verify_round_trip_succeeds_for_canonical_input` — closes the loop on the committed canonical input.
2. `prove_then_verify_round_trip_succeeds_for_a_non_canonical_input` — exercises the seam beyond the canonical fixture.
3. `adapter_proof_bytes_are_deterministic_across_two_artifact_builds` — full `ProofArtifactBody`-level determinism.
4. `tampered_proof_bytes_are_rejected_by_verifier` — sanity: tamper does not verify.

### CLI tests (in `omni-node/src/operator.rs::tests`, gated `feature = "halo2-reference-prove"`)

1. `parse_input_i16_accepts_canonical_value` — `-5,10,20,-100` round-trips to `CANONICAL_INPUT`.
2. `parse_input_i16_tolerates_whitespace_around_separators` — whitespace tolerance.
3. `parse_input_i16_refuses_wrong_arity` — typed `ReferenceProofInputParse`.
4. `parse_input_i16_refuses_non_i16_value` — typed `ReferenceProofInputParse`.
5. `generate_reference_proof_writes_artifact_that_verify_proof_accepts` — end-to-end CLI roundtrip.
6. `generate_reference_proof_refuses_when_input_parse_fails_before_invoking_prover` — error path; no artifact written.
7. `mock_artifact_verify_path_still_accepts_under_halo2_reference_prove_feature` — **coexistence pin**: existing Mock-backend flow stays green under the new feature combination.

## CI gates added

1. **Default-tree isolation guard** (`default-tree-check` job): asserts `cargo tree -p omni-node` does NOT contain `omni-proofs-halo2-reference`. The existing Stage 11b.1.b `no halo2_proofs / pasta_curves` guard covers the heavy deps; this guard adds an explicit isolation pin on the crate itself.
2. **`halo2-reference-prove-build-test` job**: runs `cargo build -p omni-node --features halo2-reference-prove`, full `cargo test -p omni-proofs-halo2-reference --features prove --features verify` (8 unit tests + 4 integration tests), and `cargo test -p omni-node --features halo2-reference-prove --bins` (entire omni-node bin test surface, including the 7 new tests).
3. **`halo2-reference-prove-tree-check` job**: asserts `cargo tree -p omni-node --features halo2-reference-prove` DOES contain both `halo2_proofs` and `rand_chacha`. Catches a silent feature wiring break.

Existing gates unchanged: default-tree, submit-tree, halo2-reference-verify tree + build, stage11d-production-verify tree + build, contributor build, Stage 6 byte-stability, Stage 11a byte-stability.

## Notes on naming reconciliation

The Stage 14.0 design packet's APPROVE locks specified `ProofSystem::Halo2Reference`. The recon report (logged before implementation) flagged that the existing variant is `Stage11bHalo2Reference`. User confirmed reuse of the existing variant to avoid enum / eligibility registry / classifier / verifier dispatch churn. Stage 14.1 reuses `ProofSystem::Stage11bHalo2Reference` end-to-end.

The "concrete, not parameterised" intent of the lock is satisfied — the variant has been concrete since Stage 11b.1, ships in `MAINNET_APPROVED_PROOF_SYSTEMS` as not-approved, and is dispatched-on by name in the verifier.

## Future outlook

- **Stage 14.2** (candidate): ezkl-backed proof generation for a single-op ONNX model. Same trait seam (`ProofBackend`), new crate `omni-proofs-ezkl-reference`. Brings ONNX runtime + ezkl tooling deps; requires its own CI tree gate. Mainnet posture identical (testnet-or-dev-only + bounded-reference refusal).
- **Stage 14.3** (candidate): wire the new prover into `omni-contributor::run-job` so each contributor inference output carries a real proof. Requires a chain-side opinion on whether the Stage 12.25 signed report's wrapper should embed proof bytes verbatim or reference a SNIP V2 store entry — chain-team review point.
- **Out of scope for Stage 14.x**: real prover for the production MLP (Stage 11d.2 has only the verifier shell; production prover is a separate stage); RISC Zero zkVM integration; staking/slashing/reward economy (chain-team contract dependency).

The Stage 14.x track will continue until the dual-prover surface (halo2 reference + ezkl reference) is operator-reachable for both small reference circuits and a real one-op model. After that, Phase 5 moves to chain-side tokenomics work — chain-team dependent.