# Stage 14.7 — proof-generation acceptance hardening

**Status:** delivered as acceptance + docs only. Zero production code changes. The Stage 14.x track approaches its natural close: both reference (Stages 14.1 / 14.2 / 14.3) and production (Stages 14.5 / 14.6) prover paths are now fully operator-reachable; Stage 14.7 stitches them with cross-family acceptance tests and a single canonical comparison table in the operator runbook.

## Scope

Stage 14.7 is **not a new capability stage.** Goal: pin the cross-family invariants that per-stage tests don't cover (each per-stage suite only knows its own family), and document the reference-vs-production contract diff in one canonical place.

### Constraints honored (re-stated from the approved plan)

- ❌ No new proof systems.
- ❌ No new `ProofSystem` / `ModelFormat` variants.
- ❌ No `ProofArtifactBody` schema changes.
- ❌ No `ContributorResult` / `Evidence` schema changes.
- ❌ No chain RPC changes; no `omni-sumchain` changes.
- ❌ No mainnet allowlist changes (`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` stays empty).
- ❌ No `crates/omni-zkml/src/error.rs` changes.
- ❌ No default-build prover dep leakage.
- ❌ No prover/verifier internal refactor (helpers reused as-is).
- ❌ No new CI gates.
- ❌ No housekeeping of pre-existing clippy warnings.
- ❌ No per-stage test consolidation — all 12 existing per-stage modules untouched.

## Halt-rule check

| Signal | State at delivery |
| --- | --- |
| Stages 14.1–14.6 merged on `main` | ✅ via PRs #58, #59, #60, #61, #62 |
| CI green on every merge | ✅ 11–15 checks across the track |
| Open incidents touching the 14.x surface | none |
| New chain contract changes pending | none |
| Mainnet allowlist still empty | ✅ — Stage 11d.3 separate track |
| Pre-existing clippy warnings on 14.x crates | inventoried previously; none introduced by 14.1–14.6 |

Halt rule satisfied. Stage 14.7 proceeded as planned.

## Umbrella test scope

The acceptance umbrella adds **4 tests** to [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) inside the existing `mod tests` block. Each test pins a cross-family invariant that no per-stage suite covers:

| # | Test | Feature gate | What it pins |
|---|---|---|---|
| 1 | `reference_and_production_artifacts_have_distinct_proof_systems_and_opposite_testnet_flags` | `cfg(all(halo2-reference-prove, stage11d-production-prove))` | Metadata contract diff (every field in the runbook comparison table) — including `EXPECTED_CIRCUIT_ID_HEX` and `EXPECTED_VK_HASH_HEX` by exact value |
| 2 | `reference_artifact_is_mainnet_refused_at_layers_1_3_and_6_with_distinct_reasons` | `cfg(halo2-reference-prove)` | Mutating one field at a time, isolate that layers 1 (TestnetOrDevOnly), 3 (BoundedReference), and 6 (NotInMainnetAllowlist) all fire for the reference family |
| 3 | `production_artifact_is_mainnet_refused_at_layer_6_only_with_layer_1_passing` | `cfg(stage11d-production-prove)` | Production artifact's sole refusal is layer 6; layer 1 does NOT fire because `testnet_or_dev_only=Some(false)` is load-bearing |
| 4 | `verify_proof_dispatch_routes_mock_reference_and_production_artifacts_correctly` | `cfg(all(halo2-reference-prove, stage11d-production-prove))` | Mock + reference + production all dispatch correctly under both prove features active |

### Cached-artifact strategy

Two `std::sync::OnceLock` statics inside the test module amortize prover invocations:

```rust
#[cfg(feature = "halo2-reference-prove")]
fn cached_reference_artifact() -> &'static omni_zkml::ProofArtifactBody { ... }

#[cfg(feature = "stage11d-production-prove")]
fn cached_production_artifact() -> &'static omni_zkml::ProofArtifactBody { ... }
```

Each prover runs **at most once per cargo test binary run**, regardless of how many sibling tests assert against the same artifact. Tests 1, 2, 4 share the reference artifact; tests 1, 3, 4 share the production artifact. Net prover cost vs. naive duplication:

| Approach | Reference prover calls | Production prover calls | Aggregate test runtime |
|---|---|---|---|
| Naive (per-test build) | 3 × ~10 s = 30 s | 3 × ~30 s = 90 s | ~120 s + framework |
| **Stage 14.7 OnceLock-cached** | 1 × ~10 s = 10 s | 1 × ~30 s = 30 s | **~40 s + framework** |

Measured: under `cargo test --features halo2-reference-prove --features stage11d-production-prove` the 4 umbrella tests complete in **~35 seconds**, dominated by the two cached prover initialisations.

## CI coverage

| Test | Runs in `halo2-reference-prove-build-test` | Runs in `stage11d-production-prove-build-test` | Local-only |
|---|---|---|---|
| 1 | — | — | ✅ (gated on both features) |
| 2 | ✅ | — | — |
| 3 | — | ✅ | — |
| 4 | — | — | ✅ (gated on both features) |

Tests 1 and 4 are gated on `cfg(all(halo2-reference-prove, stage11d-production-prove))` — **local-only validation; not in CI today.** This matches the existing Stage 14.6 test 7 situation (the clap-conflict pin). No new "both features" CI job is added by Stage 14.7. Per the user lock _"Existing feature jobs should cover the new tests if possible; avoid adding CI gates unless necessary"_, the cross-family tests are explicitly documented as local-only.

If a future PR wants CI coverage of these tests, the cheapest path is **one** new CI job activating both features, which would also exercise the existing Stage 14.6 test 7. That's out of scope for Stage 14.7.

## Surface map

| File | Stage 14.7 change |
| --- | --- |
| [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) | New helpers `cached_reference_artifact` + `cached_production_artifact` + 4 acceptance tests, all inside the existing `mod tests` block. ~300 lines added; no production-code path touched. |
| [`docs/operator-runbook.md`](operator-runbook.md) | New "Stage 14.x — proof family comparison" section: comparison table + when-to-choose prose + performance caveats. |
| This doc | Engineering doc. |

**Zero changes** to: `crates/omni-zkml`, `crates/omni-sumchain`, `crates/omni-contributor`, `crates/omni-proofs-halo2-reference`, `crates/omni-proofs-halo2-production-mlp`, any `Cargo.toml`, `.github/workflows/`.

## Stage 14.x close

Stage 14.7 closes the proof-generation track that opened with Stage 14.1. End-state coverage:

| Surface | Reference | Production |
|---|---|---|
| Operator generate-proof | ✅ Stage 14.1 | ✅ Stage 14.5 |
| Operator verify-proof dispatch | ✅ Stage 11b.1.b | ✅ Stage 11d.2 |
| Contributor StubRunner sidecar | ✅ Stage 14.2 | ✅ Stage 14.6 |
| Contributor ExternalCommandRunner sidecar | ✅ Stage 14.3 | ✅ Stage 14.6 |
| Verifier roundtrip integration tests | ✅ Stage 14.1 (4 tests) | ✅ Stage 14.5 (4 tests) |
| CI tree gates | ✅ Stage 14.1 | ✅ Stage 14.5 |
| Cross-family acceptance pin | ✅ **Stage 14.7** | ✅ **Stage 14.7** |
| Runbook documentation | ✅ Stages 14.1–14.6 per-stage | ✅ **Stage 14.7 canonical comparison** |

The remaining items in the Phase 5 zkML scope are:
- **Stage 11d.3** — chain-team-reviewed mainnet allowlist entry for `Stage11dProductionFixedPointMlp`. Separate track, chain-team-dependent.
- **Chain-side proof verification** — requires chain-team contract specs; explicitly deferred.
- **Staking / slashing / reward distribution** — requires Stage 11d.3 + chain-side verification first.
- **Real model proving beyond the canonical reference/production MLPs** — out of scope for Phase 5; would require EZKL (license-blocked per Stage 14.4) or a separately-vetted Rust prover.

The Phase 5 zkML track will not close at Stage 14.7 — it closes when chain-side mainnet eligibility (Stage 11d.3) lands and the off-chain prover generation can be lifted to chain-anchored attestations. That work depends on chain-team contract sign-off and is tracked separately.

## Future outlook

If the user wants to revisit:

- **Stage 14.8 (candidate)** — Combined-features CI job. Would activate `--features halo2-reference-prove --features stage11d-production-prove` in CI; cover Stage 14.6 test 7 + Stage 14.7 tests 1 + 4. Small CI cost (~1 new job, prover time amortized by OnceLock caches). Deferred as housekeeping; not in 14.7 scope.
- **Stage 14.x housekeeping** — Pre-existing clippy warnings on `omni-proofs-halo2-production-mlp::circuit.rs` (manual `Range::contains`, `loop variable used to index`, etc.). All pre-Stage-14 and out of scope.
- **Stage 11d.3** — Chain-team-reviewed mainnet allowlist entry; lifts the production family's layer-6 refusal. Separate chain-side track.

Stage 14.7 is the recommended last "feature-adjacent" stage in the 14.x range. After this, Phase 5 work shifts to chain-side dependencies (Stage 11d.3) and Phase 5 release packaging (Stage 10b).
