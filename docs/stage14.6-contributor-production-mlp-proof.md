# Stage 14.6 — contributor production-MLP sidecar proof emission

**Status:** delivered as an additive extension on top of Stage 14.5 (operator-side production prover) and Stage 14.2/14.3 (contributor reference sidecar). Stage 14.6 wires the Stage 14.5 [`Halo2ProductionMlpProofBackend`](../crates/omni-proofs-halo2-production-mlp/src/prover.rs) into the contributor `run-job` flow, mirroring how Stage 14.2/14.3 wired the reference prover. Both StubRunner and ExternalCommandRunner are supported in this single slice.

## Scope (locked v1 design packet, Q1 + Q2 + Q3)

Stage 14.6 is **strictly additive on top of Stages 14.5 + 14.2 + 14.3.** No new feature flag, no new `ProofSystem`/`ModelFormat` variants, no schema additions, no chain RPCs, no mainnet allowlist changes, no `omni-contributor` modifications, no `omni-zkml` modifications.

### Locks honored

| Lock | How |
| --- | --- |
| **Q1 — bundle StubRunner + ExternalCommandRunner in one slice** | Both paths wired through `emit_production_mlp_proof_sidecar` (file-based) and `emit_production_mlp_proof_sidecar_from_bytes` (bytes-based via the existing Stage 14.3 `ByteCapturingRunner`). |
| **Q2 — accept ~3.5 min added CI prover time for test clarity** | All 11 tests prover-driven; no shared-proof caching. Per-test runtime acceptable as long as the existing `stage11d-production-prove-build-test` job stays under ~10 min wall clock. |
| **Q3 — `conflicts_with` on both emit flags for defensive symmetry** | Declared on both `emit_halo2_reference_proof` and `emit_production_mlp_proof` via `#[cfg_attr(other-feature, arg(...))]` so single-feature builds compile cleanly; both-feature builds get reciprocal clap-layer mutual exclusion. |
| **D1 Alpha runtime check (Stage 14.3) extended** | The early `run_run_job` guard refusing missing `--stub-input` now fires for **either** emit flag with `--runner stub`. |
| **`--stub-input` reused (not duplicated)** | Same flag as Stage 14.2/14.3. Shape-agnostic at the CLI layer; production-shape enforced by `Halo2ProductionMlpProofBackend` (32-byte input, 16-byte output) plus the hash bindings inside `assemble_and_write_production_sidecar`. |
| **D6 — semantic `ContributorResult` equality only** | The new helpers take `&ContributorResult` (immutable); test 2 pins the on-disk result file is byte-untouched even when sidecar emission succeeds. |
| **D7 — `InferenceRunner::run` only** | The shared Stage 14.3 `ByteCapturingRunner` is unchanged; activation handoff has no proof binding. |
| **Production metadata contract matches Stage 14.5 exactly** | `proof_system = Stage11dProductionFixedPointMlp`, `model_format = ProductionFixedPointMlp`, `circuit_id_hex = EXPECTED_CIRCUIT_ID_HEX` (required), `verification_key_hex = EXPECTED_VK_HASH_HEX` (required), `testnet_or_dev_only = Some(false)`. |
| **Mainnet refusal at layer 6 only** | Test 8 pins `testnet_or_dev_only = Some(false)` (layer 1 passes); `check_mainnet_eligible(meta).is_err()` (layer 6 fires from empty `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`). |

### Constraints honored

| Constraint | Check |
| --- | --- |
| `omni-zkml` untouched | `git diff main -- crates/omni-zkml/` empty |
| `omni-sumchain` untouched | `git diff main -- crates/omni-sumchain/` empty |
| `omni-contributor` untouched | `git diff main -- crates/omni-contributor/` empty (Stage 12.0 lean-crate invariant preserved) |
| `omni-proofs-halo2-reference` untouched | `git diff main -- crates/omni-proofs-halo2-reference/` empty |
| `omni-proofs-halo2-production-mlp` untouched | `git diff main -- crates/omni-proofs-halo2-production-mlp/` empty (Stage 14.5 adapter reused verbatim) |
| No `Cargo.toml` changes | Stage 14.6 reuses the existing `stage11d-production-prove` feature |
| No CI gate changes | Existing `stage11d-production-prove-build-test` runs `cargo test -p omni-node --features stage11d-production-prove --bins` and picks up the 11 new tests automatically |
| Default-build dep tree unchanged | Stage 14.5 default-tree isolation gate still asserts |
| Mainnet allowlist unchanged | `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` stays empty; Stage 11d.3 remains the separate chain-team-reviewed allowlist track |
| Stage 14.2/14.3 reference sidecar tests pass byte-equivalent | All 21 reference-path tests still green |

## Contract diff vs Stage 14.2/14.3

The production sidecar is structurally parallel to the reference sidecar but with the **Stage 14.5 strict production contract**:

| Aspect | Stage 14.2/14.3 reference | **Stage 14.6 production** |
| --- | --- | --- |
| Feature | `halo2-reference-prove` | `stage11d-production-prove` |
| CLI flag | `--emit-halo2-reference-proof <PATH>` | `--emit-production-mlp-proof <PATH>` |
| Canonical spec | `omni-proofs-halo2-reference/assets/canonical_spec.json` | `omni-proofs-halo2-production-mlp/assets/canonical_spec.json` |
| Spec-hash constant | `EXPECTED_SPEC_HASH` | `EXPECTED_PRODUCTION_SPEC_HASH` |
| Input bytes | 8 (4 × i16 LE) | **32 (16 × i16 LE)** |
| Output bytes | 8 (4 × i16 LE) | **16 (8 × i16 LE)** |
| `proof_system` | `Stage11bHalo2Reference` | `Stage11dProductionFixedPointMlp` |
| `model_format` | `Halo2ReferenceMlp` | `ProductionFixedPointMlp` |
| `backend_id` | `"halo2-reference-mlp-v1"` | `"production-fixedpoint-mlp-v1"` |
| `testnet_or_dev_only` | `Some(true)` | **`Some(false)`** |
| `circuit_id_hex` | from `backend.circuit_id()`, optional | **required, must equal `EXPECTED_CIRCUIT_ID_HEX`** |
| `verification_key_hex` | `None` | **required, must equal `EXPECTED_VK_HASH_HEX`** |
| Mainnet refusal layers | 1 + 3 + 6 | **6 only** |
| `--stub-input` | required for `--runner stub` | required for `--runner stub` (same flag, different byte-length validation) |
| Emit-flag mutual exclusion | `conflicts_with = "emit_production_mlp_proof"` | `conflicts_with = "emit_halo2_reference_proof"` |

`assemble_and_write_production_sidecar` is a **parallel sibling** to Stage 14.2/14.3's `assemble_and_write_sidecar` — they don't share an inner because the metadata contracts diverge in too many load-bearing places (different spec, different testnet flag, different required fields, different input/output sizes).

## Surface map

| Surface | Stage 14.6 change |
| --- | --- |
| [`crates/omni-node/src/contributor_cli.rs`](../crates/omni-node/src/contributor_cli.rs) | One new feature-gated field on `RunJobArgs` (`emit_production_mlp_proof: Option<PathBuf>`); existing `emit_halo2_reference_proof` field gains a `cfg_attr` for the symmetric `conflicts_with`; `stub_input` field cfg broadened to `any(halo2-reference-prove, stage11d-production-prove)`. New helpers `emit_production_mlp_proof_sidecar`, `emit_production_mlp_proof_sidecar_from_bytes`, `assemble_and_write_production_sidecar`. `ByteCapturingRunner` cfg broadened to the same `any(…)`. `run_run_job` runtime check + dispatch extended to cover the production emit flag (Stub + External branches). New `mod stage_14_6_production_mlp_sidecar_proof` with 11 hermetic tests inside the existing `tests` module. |
| [`docs/operator-runbook.md`](operator-runbook.md) | New Stage 14.6 sub-section under "Phase 5 — real proof generation". |
| This doc | Engineering doc. |

**Zero changes** to: `crates/omni-zkml`, `crates/omni-sumchain`, `crates/omni-contributor`, `crates/omni-proofs-halo2-reference`, `crates/omni-proofs-halo2-production-mlp`, any `Cargo.toml`, `.github/workflows/ci.yml`.

## Test inventory (11 hermetic; gated `feature = "stage11d-production-prove"`)

In [`contributor_cli::tests::stage_14_6_production_mlp_sidecar_proof`](../crates/omni-node/src/contributor_cli.rs):

1. `stub_runner_with_emit_flag_writes_production_sidecar_artifact_under_canonical_production_spec_job` — happy path; sidecar parses with `proof_system=Stage11dProductionFixedPointMlp` and `testnet_or_dev_only=Some(false)`.
2. `emit_production_sidecar_does_not_mutate_contributor_result_bytes_on_disk` — D6 pin.
3. `run_job_refuses_emit_production_flag_when_job_model_hash_is_not_canonical_production_spec` — non-canonical model hash → no sidecar.
4. `run_job_refuses_emit_production_flag_when_stub_input_hash_mismatches_job_input_hash` — hash-binding refusal (correct length, wrong content).
5. `run_job_refuses_emit_production_flag_when_stub_input_byte_length_is_wrong` — production-shape arity refusal (8-byte reference-shape file is refused under production path; no sidecar).
6. `production_mlp_verifier_accepts_contributor_emitted_sidecar` — end-to-end stitch.
7. `clap_refuses_both_emit_flags_simultaneously` — **Q3** clap mutual-exclusion pin. Gated `#[cfg(all(halo2-reference-prove, stage11d-production-prove))]` — the conflict only exists when both features are simultaneously enabled.
8. `generated_production_sidecar_artifact_is_mainnet_refused_at_layer_6_only` — mainnet posture pin: `testnet_or_dev_only=Some(false)` (layer 1 passes); `check_mainnet_eligible(meta).is_err()` (layer 6 fires).
9. `production_sidecar_public_inputs_carry_contributor_job_id` — D2 presence + 16/8 array-length pin.
10. `production_verifier_tolerates_extra_contributor_job_id_key_in_public_inputs` — D2 regression pin for the production verifier (extra-key tolerance).
11. `external_command_runner_real_subprocess_end_to_end_writes_verifiable_production_sidecar` — **`#[cfg(unix)]`** real `ExternalCommandRunner` subprocess + shell-script fixture (emits valid `ExternalRunnerEnvelope` with canonical 16-byte production response bytes) + `ByteCapturingRunner` + `emit_production_mlp_proof_sidecar_from_bytes` + verifier accepts.

Regression-side: all 21 Stage 14.2/14.3 reference-path tests, all 8 Stage 14.5 CLI tests, all `omni-proofs-halo2-production-mlp` suites pass byte-equivalent.

## Mainnet posture

Identical to Stage 14.5. The sidecar artifact:
- `proof_system = Stage11dProductionFixedPointMlp` → refused at `check_mainnet_eligible` **layer 6** (empty `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`)
- `testnet_or_dev_only = Some(false)` → layer 1 does **not** fire
- `model_format = ProductionFixedPointMlp` → also refused at layer 6

**Layer 6 is the sole mainnet gate.** Stage 11d.3 lands the chain-team-reviewed allowlist entry that lifts it; Stage 14.6 does not touch the allowlist.

## Future outlook

- **Stage 14.7+** — operator UX hardening for the production sidecar story: per-contributor proof signing, batch proving across multiple jobs, regen-tool consolidation.
- **Stage 11d.3** — chain-team-reviewed mainnet allowlist entry for `Stage11dProductionFixedPointMlp`. Separate track, not blocked by Stage 14.6.
- **Out of scope for Stage 14.x:** production prover for any other proof system, EZKL (license still blocked per Stage 14.4), staking / slashing / reward distribution, chain-side proof verification, `Evidence` enum bump.

The Stage 14.x track is approaching its natural end-state: both reference (Stage 14.1 + 14.2 + 14.3) and production (Stage 14.5 + 14.6) paths are now operator-reachable for prove + verify + contributor sidecar. After Stage 14.x closes, Phase 5 transitions to chain-side tokenomics work — chain-team dependent.
