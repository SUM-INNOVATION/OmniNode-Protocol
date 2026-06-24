# Stage 14.3 — ExternalCommandRunner sidecar proof emission

**Status:** delivered as an additive extension on top of Stage 14.2. The Stage 14.2 `--emit-halo2-reference-proof` flag — previously StubRunner-only — now also works with `--runner external`. Bytes are captured at the `InferenceRunner::run` trait boundary by a thin wrapper inside the CLI; the operator does not have to redundantly supply `--stub-input` for the external path.

## Scope (locked v2 design packet, D1–D7)

Stage 14.3 stays **strictly additive on top of Stage 14.1 + 14.2.** No new feature flag, no new `ProofSystem` variant, no schema additions, no chain RPCs, no `omni-contributor` modifications.

### Locks honored

| Lock | How |
| --- | --- |
| **D1 Alpha** — drop static `requires = "stub_input"` from clap; replace with an early runtime check inside `run_run_job` | Field attribute changed to `#[arg(long)]`. Runtime guard at the top of `run_run_job` fires before any work for `--runner stub && emit-flag && stub-input missing`. Stage 14.2 user-observable contract preserved: command fails fast, error message mentions `--stub-input`. |
| **D2** — Unix shell-script fixture pattern | Two test layers: (1) the `ByteCapturingRunner` D6 lifecycle pins use an in-process `ScriptedRunner` (deterministic, no subprocess); (2) `external_command_runner_real_subprocess_end_to_end_writes_verifiable_sidecar` spawns a real `ExternalCommandRunner` against a `#[cfg(unix)]` shell script that emits a valid `ExternalRunnerEnvelope`, drives the bytes through `ByteCapturingRunner`, calls the sidecar helper, and verifies. Mirrors the repo's existing `#[cfg(unix)]` + `set_permissions(0o755)` pattern at [`integrity_evidence_bundle.rs:446`](../crates/omni-contributor/tests/integrity_evidence_bundle.rs#L446). CI is Linux. |
| **D3** — ignore `--stub-input` for ExternalRunner | The `--stub-input` field is consulted only when `args.runner == RunnerChoice::Stub`. For External, captured bytes are used directly; no defensive cross-check against an operator-supplied `--stub-input` (if one is passed by mistake, it is silently ignored). |
| **D4** — no new failure event format | All refusals propagate as `anyhow::Error` and flatten through the existing `OperatorError::ContributorWorkflow(String)` catch-all. Tests assert "no sidecar written"; they do **not** assert any specific `event=…_failed` line. |
| **D5** — `ContributorResult` invariant is semantic, not byte-identical | Acceptance bar: same `schema_version`, same `Evidence::AttestationOnly`, same model/input/response hashes, same accounting. `produced_at_utc` may differ across runs. No byte-equality assertion. |
| **D6** — `ByteCapturingRunner` clears slots at start; captures output only after successful inner return | The wrapper's `run` impl: (1) clears both `captured_input` and `captured_output` to `None` at the start, (2) sets `captured_input = Some(input_bytes.to_vec())` **before** the inner call, (3) sets `captured_output = Some(output.response_bytes.clone())` **only after** the inner `Ok(_)`. Tests 7, 8, 9 pin each step independently. |
| **D7** — Stage 14.3 covers `InferenceRunner::run` only | The wrapper does **not** override `run_with_activations`. The default-impl forward-through-`run` chain is incidental, not relied upon. Stage 12.4 activation handoff paths have no proof binding in Stage 14.3. Test 10 is a documentation pin. |

### Constraints honored

| Constraint | Check |
| --- | --- |
| `crates/omni-zkml/src/error.rs` untouched | `git diff main` empty |
| `ProofSystem` enum untouched | Reuses `Stage11bHalo2Reference` |
| Mainnet eligibility registry unchanged | `MAINNET_APPROVED_PROOF_SYSTEMS` stays empty |
| Verifier API untouched | `Halo2ReferenceVerifier` not modified |
| `ProofArtifactBody` schema untouched | Only the opaque `public_inputs` JSON value carries the Stage 14.2 `contributor_job_id` extra key (already-pinned tolerance) |
| `Evidence` enum / `ContributorResult` schema unchanged | `schema_version: 1`, `Evidence::AttestationOnly` only |
| `omni-contributor` `Cargo.toml` and `src/` untouched | Stage 12.0 lean-crate invariant preserved (verified by empty diff) |
| `omni-sumchain` untouched | No chain RPCs added |
| No `Cargo.toml` changes | Stage 14.3 reuses the existing `halo2-reference-prove` feature |
| No `.github/workflows/ci.yml` changes | Existing Stage 14.1 `halo2-reference-prove-build-test` runs `cargo test -p omni-node --features halo2-reference-prove --bins`, which picks up new tests automatically |
| No live-chain CI tests | All Stage 14.3 tests are hermetic |
| Default-build dep tree unchanged | Stage 14.1 default-tree isolation gate already asserts |
| StubRunner behavior from Stage 14.2 unchanged | All 10 Stage 14.2 tests pass; only test 5 was reshaped to assert the runtime refusal that replaces the clap-layer one (same user-observable contract) |

## Surface map

| Surface | Stage 14.3 change |
| --- | --- |
| [`crates/omni-node/src/contributor_cli.rs`](../crates/omni-node/src/contributor_cli.rs) | (1) field attribute on `emit_halo2_reference_proof`: `requires = "stub_input"` → `#[arg(long)]` (no static requires). (2) New early runtime check in `run_run_job` (D1 Alpha). (3) `emit_halo2_reference_proof_sidecar` refactored: shared `assemble_and_write_sidecar` inner + Stage 14.2 file-based wrapper + Stage 14.3 bytes-based wrapper. (4) New `ByteCapturingRunner<'a, R>` struct + `InferenceRunner` impl with D6 lifecycle and the D7 "standard `run` only" carve-out. (5) Wiring in `run_run_job`'s External branch: when emit-flag is set, runner is wrapped and bytes are extracted post-`run_job`. (6) `stage_14_2_sidecar_proof::clap_refuses_emit_flag_without_stub_input` renamed and reshaped to `run_run_job_refuses_emit_flag_without_stub_input_for_stub_runner_via_runtime_check`. (7) `stage_14_2_sidecar_proof::emit_sidecar_refuses_external_runner` renamed to `file_based_emit_helper_refuses_non_stub_runner` (same body — pins that the file-based wrapper is StubRunner-only). (8) New `mod stage_14_3_external_runner_sidecar_proof` with 10 hermetic tests. |
| [`docs/operator-runbook.md`](operator-runbook.md) | New Stage 14.3 sub-section under "Phase 5 — real proof generation". |
| This doc | Engineering doc. |

**Zero changes** to: `crates/omni-zkml`, `crates/omni-sumchain`, `crates/omni-contributor`, `crates/omni-proofs-halo2-reference`, `crates/omni-node/Cargo.toml`, `.github/workflows/ci.yml`.

## Byte-capture lifecycle (D6)

```text
   ┌─────────────────────────────────────────────────────────────────────┐
   │ ByteCapturingRunner::run(manifest_path, input_bytes)                │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                     │
   │  1. captured_input  := None       ← D6: clear at start              │
   │     captured_output := None                                         │
   │                                                                     │
   │  2. captured_input  := Some(input_bytes.to_vec())                   │
   │                                   ← captured BEFORE inner call      │
   │                                                                     │
   │  3. inner.run(manifest_path, input_bytes) →                         │
   │       Ok(output)  → captured_output := Some(output.response_bytes)  │
   │       Err(e)      → captured_output stays None ← D6 invariant       │
   │                       (no sidecar emission possible)                │
   │                                                                     │
   └─────────────────────────────────────────────────────────────────────┘

After a successful inner run, run_run_job extracts both slots and hands
them to `emit_halo2_reference_proof_sidecar_from_bytes(...)`. After an
inner Err, `take_captured_output()` returns None and the post-run guard
in run_run_job bails before any sidecar work runs.
```

## Test inventory (10 hermetic tests; gated `feature = "halo2-reference-prove"`)

In [`contributor_cli.rs::tests::stage_14_3_external_runner_sidecar_proof`](../crates/omni-node/src/contributor_cli.rs):

1. `external_path_bytes_helper_writes_sidecar_for_canonical_triple` — happy path on the bytes-based helper.
2. `external_path_sidecar_input_hash_equals_captured_input_bytes_hash` — tautological hash binding pin.
3. `external_path_sidecar_output_hash_equals_envelope_response_bytes_hash` — same for output.
4. `external_path_sidecar_verifies_under_halo2_reference_verifier` — end-to-end stitch.
5. `external_path_refuses_when_envelope_response_bytes_do_not_match_canonical_evaluator_output` — the adapter's defense-in-depth check fires; no sidecar written.
6. `no_sidecar_is_written_when_byte_capturing_runner_inner_returns_err` — D4 + D6: inner Err → captured_output None → post-run guard bails → no sidecar.
7. `byte_capturing_runner_clears_both_slots_at_start_of_each_run` — **D6 #1**: reuse across two runs (Ok then Err) leaves no leaked output bytes.
8. `byte_capturing_runner_does_not_capture_output_on_inner_error` — **D6 #3**: single Err leaves output None.
9. `byte_capturing_runner_captures_input_before_inner_call_runs` — **D6 #2**: input captured even when inner errors.
10. `byte_capturing_runner_does_not_override_run_with_activations_per_d7` — **D7** documentation pin: wraps only `run`.
11. `external_command_runner_real_subprocess_end_to_end_writes_verifiable_sidecar` — **`#[cfg(unix)]`** spawns a real `ExternalCommandRunner` against a shell-script fixture (`chmod 0o755`, valid `ExternalRunnerEnvelope` JSON with the canonical 8-byte `response_b64`); drives the bytes through `ByteCapturingRunner`; calls `emit_halo2_reference_proof_sidecar_from_bytes`; verifier accepts. Closes the operator-promise loop: `--runner external --external-command <script> --emit-halo2-reference-proof <path>` works end-to-end through the real subprocess driver. A tiny inline base64 encoder avoids pulling `base64` into `omni-node`'s deps; correctness is implicit (a buggy encoder would make the script's response_bytes decode to garbage, the canonical-evaluator pre-check would refuse, and this test would fail).

In [`contributor_cli.rs::tests::stage_14_2_sidecar_proof`](../crates/omni-node/src/contributor_cli.rs) (Stage 14.2 module, updated by Stage 14.3):

- `run_run_job_refuses_emit_flag_without_stub_input_for_stub_runner_via_runtime_check` (formerly `clap_refuses_emit_flag_without_stub_input`) — clap parse now succeeds; runtime check refuses with the same user-observable contract.
- `file_based_emit_helper_refuses_non_stub_runner` (formerly `emit_sidecar_refuses_external_runner`) — same body, renamed to reflect that this pins the file-based wrapper's StubRunner-only refusal (the External path goes through the bytes-based wrapper).
- All other 8 Stage 14.2 tests pass byte-equivalent.

**Total Stage 14.x tests:** 21 (10 Stage 14.2 + 11 Stage 14.3, including the `#[cfg(unix)]` real-subprocess end-to-end), all green under `--features halo2-reference-prove`.

## Mainnet posture

Identical to Stage 14.1 + 14.2. Sidecar artifact metadata:
- `proof_system = Stage11bHalo2Reference` → refused at `check_mainnet_eligible` layer 3 (BoundedReference class) and layer 6 (empty eligibility registry)
- `testnet_or_dev_only = Some(true)` → refused at layer 1

Three independent refusal layers fire on the same artifact. `--allow-mainnet-submit` cannot override.

## Future outlook

- **Stage 14.4** — ezkl-backed sidecar emission for a single-op ONNX model. Adds a second `ProofSystem` variant (likely `Stage11bEzklReference`) and either a parallel CLI flag (`--emit-ezkl-reference-proof`) or a refactor to a generic `--emit-proof <system>:<path>` shape. The latter would also need to fit Stage 14.3's bytes-based path. Out of scope for 14.3.
- **Stage 14.5+** — `run_with_activations` proof binding (Stage 12.4 live handoff path). Would extend `ByteCapturingRunner` to capture `upstream_activation_bytes` and the output activation; proof would bind the full pipelined trace. Significantly larger surface; chain-team review point for activation-attribution.
- **Out of scope for the Stage 14.x track:** production-MLP prover wiring, staking/slashing/reward distribution, chain-side proof verification, `Evidence` enum bump (the Stage 12.0 schema-migration roadmap covers this as `schema_version: 2`).

Stage 14.x will continue until the dual-prover surface (halo2 reference + ezkl reference) is operator-reachable across both `operator generate-reference-proof` (standalone) and `contributor run-job --emit-…-proof` (sidecar). After that, Phase 5 moves to chain-side tokenomics — chain-team dependent.
