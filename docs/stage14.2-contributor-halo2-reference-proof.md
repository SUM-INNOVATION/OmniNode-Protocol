# Stage 14.2 — contributor sidecar proof emission for halo2-reference

**Status:** delivered as an additive feature on top of Stage 14.1. Closes the operator-side gap where `contributor run-job` and `operator generate-reference-proof` were two separate invocations; Stage 14.2 lets a contributor produce both a `ContributorResult` and a `Halo2Reference` proof artifact in a single CLI call.

## Scope (locked v2 design packet, D1–D6)

Stage 14.2 is **strictly additive on top of Stage 14.1.** No new feature flag, no new `ProofSystem` variant, no schema additions, no chain RPCs, no `omni-contributor` modifications.

### Locks honored

| Lock | How |
| --- | --- |
| **D1** — single-purpose `--emit-halo2-reference-proof <PATH>` CLI flag | Added on `RunJobArgs`, gated `#[cfg(feature = "halo2-reference-prove")]`. No generic proof-system routing. |
| **D2** — `public_inputs` adds `contributor_job_id` as extra JSON key; verifier tolerance regression-pinned | Sidecar emits `{ "input": [...], "output": [...], "contributor_job_id": "<64-hex>" }`. Test 10 plants an additional unknown key and confirms `Halo2ReferenceVerifier::verify_artifact` still accepts. |
| **D3** — StubRunner-only; explicit `--stub-input <PATH>` required-with `--emit-halo2-reference-proof`; no SNIP fetch | Clap `requires = "stub_input"` enforces the pairing **before** any work runs. The helper reads both files verbatim and binds them to the contributor's committed hashes. |
| **D4** — ExternalCommandRunner proof emission out of scope | Helper refuses at runtime when `runner != RunnerChoice::Stub`. |
| **D5** — no new reason tags / closed operator taxonomy | All refusals go through `anyhow::Error` and flatten via the existing `OperatorError::ContributorWorkflow(String)` catch-all. Tests assert message substrings, NOT closed reason-tag strings. |
| **D6** — `ContributorResult` invariant is semantic, not byte-identical | Acceptance bar: same `schema_version`, `Evidence::AttestationOnly`, `job_id`, `model_hash`, `input_hash`, `response_hash`, `measured_accounting`. `produced_at_utc` may differ (no clock injection in Stage 14.2). Test 2 pins the **file-on-disk** is not mutated by sidecar emission. |
| **Q1** — clap-layer pairing enforcement | `#[arg(long, requires = "stub_input")]` on `emit_halo2_reference_proof`. Clap surfaces the missing-flag error before `run_job` starts. Test 5 pins this. |

### Constraints honored

| Constraint | Check |
| --- | --- |
| `crates/omni-zkml/src/error.rs` untouched | `git diff main` empty |
| `ProofSystem` enum untouched | Reuses `Stage11bHalo2Reference` |
| Mainnet allowlist unchanged | `MAINNET_APPROVED_PROOF_SYSTEMS` stays empty |
| Verifier API untouched | `Halo2ReferenceVerifier` not modified |
| `ProofArtifactBody` schema untouched | Only the opaque `public_inputs` JSON value grows a new key (extra-key tolerance pinned by Test 10) |
| `Evidence` enum / `ContributorResult` schema unchanged | `schema_version: 1`, `Evidence::AttestationOnly` only |
| `omni-contributor` `Cargo.toml` and `src/` untouched | Stage 12.0 lean-crate invariant preserved |
| `omni-sumchain` untouched | No chain RPCs added |
| No new CI gates | Existing Stage 14.1 `halo2-reference-prove-build-test` runs `cargo test -p omni-node --features halo2-reference-prove --bins`, which picks up the 10 new tests automatically |
| Default-build dep tree unchanged | Stage 14.1 default-tree isolation gate already asserts; Stage 14.2 adds nothing to default build |
| No `Cargo.toml` changes | Stage 14.2 reuses the existing feature `halo2-reference-prove` |
| No live-chain CI tests | All Stage 14.2 tests are hermetic |

## Surface map

| Surface | Stage 14.2 change |
| --- | --- |
| [`crates/omni-node/src/contributor_cli.rs`](../crates/omni-node/src/contributor_cli.rs) | Two new feature-gated fields on `RunJobArgs` (`emit_halo2_reference_proof: Option<PathBuf>` with `requires = "stub_input"`; `stub_input: Option<PathBuf>`). New helper `emit_halo2_reference_proof_sidecar(...)`. Post-run hook in `run_run_job` invokes the helper when the flag is set and emits one new stdout line (`proof_artifact_path=<path>`). New `mod stage_14_2_sidecar_proof` with 10 hermetic tests inside the existing `tests` module. |
| [`docs/operator-runbook.md`](operator-runbook.md) | New Stage 14.2 sub-section under "Phase 5 — real proof generation". |
| This doc | Engineering doc. |

**Zero changes** to: `crates/omni-zkml`, `crates/omni-sumchain`, `crates/omni-contributor`, `crates/omni-proofs-halo2-reference`, `crates/omni-node/Cargo.toml`, `.github/workflows/ci.yml`.

## Test inventory (10 hermetic tests; gated `feature = "halo2-reference-prove"`)

In [`contributor_cli.rs::tests::stage_14_2_sidecar_proof`](../crates/omni-node/src/contributor_cli.rs):

1. `run_job_with_emit_flag_writes_sidecar_artifact_under_canonical_spec_job` — happy path; sidecar parses; `proof_system == Stage11bHalo2Reference`; `testnet_or_dev_only == Some(true)`.
2. `emit_sidecar_does_not_mutate_contributor_result_bytes_on_disk` — **D6 pin**; result file is byte-untouched after successful sidecar emission.
3. `run_job_refuses_emit_flag_when_job_model_hash_is_not_canonical_spec` — **D5 + canonical-spec lock**; non-canonical `model_hash` refused via anyhow string match; no sidecar file written.
4. `run_job_refuses_emit_flag_when_stub_input_hash_mismatches_job_input_hash` — hash-binding refusal; substituted bytes hash to wrong value → refusal, no sidecar.
5. `clap_refuses_emit_flag_without_stub_input` — **Q1 + D3 pin**; clap surfaces a usage error mentioning `--stub-input` before any work runs.
6. `emit_sidecar_refuses_external_runner` — **D4 pin**; `RunnerChoice::External` is refused; no sidecar.
7. `halo2_reference_verifier_accepts_contributor_emitted_sidecar` — end-to-end stitch; emit → verify roundtrip on the canonical fixture.
8. `sidecar_artifact_carries_testnet_or_dev_only_true_and_is_mainnet_refused` — mainnet posture; `check_mainnet_eligible(meta).is_err()` for the sidecar's metadata.
9. `sidecar_public_inputs_carry_contributor_job_id` — **D2 pin**; `public_inputs["contributor_job_id"] == result.job_id`.
10. `verifier_tolerates_extra_contributor_job_id_key_in_public_inputs` — **D2 regression pin**; plants a second unknown key alongside `contributor_job_id` and confirms verification still succeeds.

Default-build regression (no new tests needed): the existing Stage 14.1 `halo2-reference-prove-build-test` CI job already runs `cargo test -p omni-node --features halo2-reference-prove --bins`. The default-build CI gates remain green; `cargo tree -p omni-node` (default) continues to exclude `halo2_proofs`, `pasta_curves`, `omni-proofs-halo2-reference`, and `rand_chacha`.

## Mainnet posture

Identical to Stage 14.1. Sidecar artifact metadata:
- `proof_system = Stage11bHalo2Reference` → refused at `check_mainnet_eligible` layer 3 (BoundedReference class) and layer 6 (empty allowlist)
- `testnet_or_dev_only = Some(true)` → refused at layer 1

Three independent refusal layers fire on the same artifact. `--allow-mainnet-submit` cannot override.

## Future outlook

- **Stage 14.3** — ExternalCommandRunner sidecar proof emission. Requires solving "how does the CLI capture runner-produced response bytes for proving" cleanly (e.g. via a stdout pipe + buffer, or a runner-side `--prove-output-to` flag).
- **Stage 14.4** — ezkl-backed sidecar emission for a single-op ONNX model. Adds a second `ProofSystem` variant (likely `Stage11bEzklReference`) and a parallel CLI flag (`--emit-ezkl-reference-proof`); design beat is whether to generalise the CLI to `--emit-proof <system>:<path>` or keep one flag per backend (D1 spirit). Out of scope for 14.2.
- **Out of scope for the Stage 14.x track**: production-MLP prover wiring, staking/slashing/reward distribution, chain-side proof verification, `Evidence` enum bump (the Stage 12.0 schema-migration roadmap covers this as `schema_version: 2`).

Stage 14.x will continue until the dual-prover surface (halo2 reference + ezkl reference) is operator-reachable across both `operator generate-reference-proof` (standalone) and `contributor run-job --emit-…-proof` (sidecar). After that, Phase 5 moves to chain-side tokenomics — chain-team dependent.
