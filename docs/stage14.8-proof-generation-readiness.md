# Stage 14.8 — proof-generation track closure / readiness packet

**Status: docs + a single docs-readiness smoke test only.** Zero behavior
change. No new proof systems, no new prover/verifier features, no chain RPC
changes, no eligibility activation, no Proof Eligibility Registry
consumption, no schema / enum / artifact / `ContributorResult` / `Evidence`
changes, no default-build dependency changes, no `crates/omni-zkml/src/error.rs`
changes, no EZKL revisit beyond restating the existing Stage 14.4 rejection.

Stage 14.8 closes the Stage 14.x proof-generation track. Stages 14.1 → 14.7
delivered both reference and production halo2 prove + verify paths reachable
from the operator binary, both contributor sidecar runners (Stub +
ExternalCommandRunner), and a cross-family acceptance umbrella. Stage 11d.3A
+ 11d.3B then aligned the Proof Eligibility Registry evidence and
terminology surface. After 14.8, work moves to Stage 15 — see §7 below for
the conditional handoff.

`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` and
`MAINNET_APPROVED_PROOF_SYSTEMS = &[]` — empty by design, end-state.
`ProofSystem::Stage11dProductionFixedPointMlp` remains dormant /
mainnet-refused. Reference proof family remains dev/testnet only in
perpetuity per Stage 11d.0 §3 non-goals.

---

## 1. Stage 14 completion inventory

| Slice | Capability | Operator surface | Contributor surface | Mainnet posture | Feature gate | Acceptance tests |
| --- | --- | --- | --- | --- | --- | --- |
| **14.1** | Halo2 reference operator prover | `omni-node operator generate-reference-proof` | — | refused at layers **1 + 3 + 6** (`testnet_or_dev_only=Some(true)`, `BoundedReference`, empty registry) | `halo2-reference-prove` (superset of `halo2-reference-verify`) | 6 CLI tests in [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) + 4 roundtrip tests in [`crates/omni-proofs-halo2-reference/tests/halo2_reference_prove_verify_roundtrip.rs`](../crates/omni-proofs-halo2-reference/tests/halo2_reference_prove_verify_roundtrip.rs) |
| **14.2** | Reference StubRunner sidecar | — | `operator contributor run-job --emit-halo2-reference-proof <PATH> --runner stub --stub-input <PATH>` | inherits 14.1's 1+3+6 | `halo2-reference-prove` | 10 hermetic tests in `contributor_cli::tests::stage_14_2_halo2_reference_sidecar_proof` |
| **14.3** | Reference ExternalCommandRunner sidecar | — | `operator contributor run-job --emit-halo2-reference-proof <PATH> --runner external` (bytes captured at `InferenceRunner::run` trait boundary by `ByteCapturingRunner`) | inherits 14.1's 1+3+6 | `halo2-reference-prove` | 11 hermetic tests in `contributor_cli::tests::stage_14_3_external_command_runner_halo2_reference_sidecar_proof` (incl. `#[cfg(unix)]` real subprocess) |
| **14.4** | EZKL feasibility + **rejection** (license / supply-chain) | n/a | n/a | n/a | n/a | docs only — [`docs/stage14.4-…`](stage14.4-ezkl-discovery-rejection.md) if present; no upstream license file at `zkonduit/ezkl` as of 2026-02-20 |
| **14.5** | Halo2 production-MLP operator prover | `omni-node operator generate-production-mlp-proof` | — | refused at **layer 6 only** (production-shape: `testnet_or_dev_only=Some(false)`, required `circuit_id_hex` + `verification_key_hex`) | `stage11d-production-prove` (superset of `stage11d-production-verify`); CI sets `RUST_MIN_STACK=67108864` | 6 CLI tests in [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) + 4 roundtrip tests in [`crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs`](../crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs) |
| **14.6** | Production-MLP contributor sidecar (Stub + External in one slice) | — | `operator contributor run-job --emit-production-mlp-proof <PATH> --runner {stub,external}` | inherits 14.5's layer 6 only | `stage11d-production-prove`; clap layer enforces mutual exclusion with `--emit-halo2-reference-proof` via `cfg_attr`-gated `conflicts_with` declarations on both fields | 11 hermetic tests in `contributor_cli::tests::stage_14_6_production_mlp_sidecar_proof` |
| **14.7** | Cross-family acceptance hardening (umbrella) | `OnceLock`-cached artifact strategy in operator tests | — | both families pinned by exact value | `cfg(all(halo2-reference-prove, stage11d-production-prove))` for the two cross-family tests (local-only); the single-feature tests run inside the existing per-feature CI jobs | 4 acceptance tests in `operator::tests` + the single canonical comparison table in the runbook |

Cross-cutting closure facts:

- **`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]`** and **`MAINNET_APPROVED_PROOF_SYSTEMS = &[]`** — empty by design.
- **Default-build dep tree** pulls zero halo2 / pasta / `omni-proofs-halo2-*` / `rand_chacha`. Pinned by CI tree-isolation gates.
- **Both provers are byte-deterministic** via distinct pinned `PROVER_RNG_SEED` constants and `ChaCha20Rng`.
- **Verifier drift detection** at construction time: `Halo2ProductionMlpVerifier::from_embedded_fixtures` re-derives `circuit_id_hex` + `verification_key_hash_hex` and refuses on drift; the artifact-side check then enforces the same constants on `ProofArtifactBody.metadata`.
- **`halo2_proofs = 0.3.2`** pinned. A version bump shifts proof bytes; fixture regen lives in the workspace-excluded `tools/halo2_reference_regen/` and `tools/halo2_production_mlp_regen/` packages.
- **EZKL** remains rejected per Stage 14.4 (no LICENSE file at `zkonduit/ezkl` as of 2026-02-20); no revisit unless upstream license changes.
- **Stage 11d.3A** evidence bundle and **Stage 11d.3B** terminology alignment merged (PR #64). Any chain-side `CandidateRefused` record is the upstream blocker for Stage 11d.3C consumption — see §4.

---

## 2. Operator readiness checklist

The checklist below is the source of truth for which feature flags activate
which prove / verify surfaces, and which CLI flags the runbook examples
depend on. The Stage 14.8 smoke test (§5) parse-checks the documented flag
long names against the clap-tree to prevent silent drift; if a future PR
renames a CLI flag, that test fails before this doc goes stale.

### Feature-flag matrix

| Build | Cargo features | Pulls |
| --- | --- | --- |
| default (read-only) | none | zero halo2 / pasta / `omni-proofs-halo2-*` / `rand_chacha` |
| verify-only, reference | `--features halo2-reference-verify` | `halo2_proofs` + `omni-proofs-halo2-reference` |
| verify-only, production | `--features stage11d-production-verify` | `halo2_proofs` + `omni-proofs-halo2-production-mlp` |
| prove (reference) | `--features halo2-reference-prove` | superset of `halo2-reference-verify` + `omni-proofs-halo2-reference/prove` + `rand_chacha` |
| prove (production) | `--features stage11d-production-prove` | superset of `stage11d-production-verify` + `omni-proofs-halo2-production-mlp/prove` + `rand_chacha` |
| both prove families | `--features halo2-reference-prove --features stage11d-production-prove` | both supersets; production prover CI also exports `RUST_MIN_STACK=67108864` |

### Reference proof — operator

```text
# Generate a reference proof artifact (4 × i16 input).
# Output is a ProofArtifactBody JSON.
cargo run -p omni-node --features halo2-reference-prove -- \
    operator generate-reference-proof \
        --input-i16 "-5,10,20,-100" \
        --output-path /tmp/ref-proof.json

# Verify the artifact under the matching verify build.
cargo run -p omni-node --features halo2-reference-verify -- \
    operator verify-proof --proof-artifact /tmp/ref-proof.json
```

Expected verify output (read from stdout / `event="proof_verification"`
tracing line):

```
backend_id=halo2-reference-mlp-v1
proof_system=Stage11bHalo2Reference
verified=true
mainnet_eligible=false
mainnet_refusal_reason=TestnetOrDevOnly { … }    # layer 1 wins; layers 3 + 6 also fire
```

### Reference proof — contributor sidecar

```text
# StubRunner — operator supplies stub bytes, contributor sidecar binds them.
cargo run -p omni-node --features halo2-reference-prove -- \
    operator contributor run-job \
        --job <PATH-to-ContributorJob.json> \
        --runner stub \
        --stub-input <PATH-to-input-bytes> \
        --stub-response <PATH-to-response-bytes> \
        --seed-file <PATH-to-contributor-seed> \
        --emit-halo2-reference-proof /tmp/ref-sidecar.json

# ExternalCommandRunner — bytes captured at the trait boundary.
cargo run -p omni-node --features halo2-reference-prove -- \
    operator contributor run-job \
        --job <PATH-to-ContributorJob.json> \
        --runner external \
        --seed-file <PATH-to-contributor-seed> \
        --emit-halo2-reference-proof /tmp/ref-sidecar.json
        # --stub-input is NOT required for --runner external

# Verify (same command as the operator-generated artifact).
cargo run -p omni-node --features halo2-reference-verify -- \
    operator verify-proof --proof-artifact /tmp/ref-sidecar.json
```

### Production proof — operator

```text
# Generate a production-MLP proof artifact (16 × i16 input).
cargo run -p omni-node --features stage11d-production-prove -- \
    operator generate-production-mlp-proof \
        --input-i16 "-5,10,20,-100,7,-3,14,25,-8,1,11,-22,4,-1,17,9" \
        --output-path /tmp/prod-proof.json

# Verify under the matching verify build.
cargo run -p omni-node --features stage11d-production-verify -- \
    operator verify-proof --proof-artifact /tmp/prod-proof.json
```

Expected verify output:

```
backend_id=production-fixedpoint-mlp-v1
proof_system=Stage11dProductionFixedPointMlp
verified=true
mainnet_eligible=false
mainnet_refusal_reason=NotInMainnetAllowlist { … }    # layer 6 sole gate
```

### Production proof — contributor sidecar

```text
# StubRunner (production-shape stub_input: 32 raw bytes = 16 × i16 LE).
cargo run -p omni-node --features stage11d-production-prove -- \
    operator contributor run-job \
        --job <PATH-to-ContributorJob.json> \
        --runner stub \
        --stub-input <PATH-to-32-byte-input> \
        --stub-response <PATH-to-16-byte-response> \
        --seed-file <PATH-to-contributor-seed> \
        --emit-production-mlp-proof /tmp/prod-sidecar.json

# ExternalCommandRunner.
cargo run -p omni-node --features stage11d-production-prove -- \
    operator contributor run-job \
        --job <PATH-to-ContributorJob.json> \
        --runner external \
        --seed-file <PATH-to-contributor-seed> \
        --emit-production-mlp-proof /tmp/prod-sidecar.json
```

`--emit-halo2-reference-proof` and `--emit-production-mlp-proof` are
mutually exclusive at the clap layer when both prove features are enabled.

### Expected mainnet behavior

| Artifact | Refusal layers fired | Lift requires |
| --- | --- | --- |
| Stage 14.1 / 14.2 / 14.3 reference | 1 + 3 + 6 (defense in depth) | **never** — bounded reference is testnet/dev-only in perpetuity (criteria §1.6 hard rule H1) |
| Stage 14.5 / 14.6 production | 6 only (production-shape passes layers 1 + 3) | **Stage 11d.3C+** — chain-side `CandidateRefused` → `CandidateApproved` → `Active` record landing in the Proof Eligibility Registry, then OmniNode-side consumption |

Both proof paths verify **off-chain only**. Chain treats proof artifacts
opaquely today; see Stage 11d.3A §8 for the register-only vs. chain-side
verify architectural fork.

### Performance caveats

- Reference circuit: `4 → 8 → 4` int16 fixed-point MLP, `HALO2_K = 10`. Proof ~10 s CPU host.
- Production circuit: `16 → 32 → 16 → 8` int16 fixed-point MLP, `HALO2_K = 11`. Proof ~30 s CPU host. Production CI requires `RUST_MIN_STACK=67108864` (64 MB).
- Both provers byte-deterministic via fixed pinned `PROVER_RNG_SEED` and `ChaCha20Rng::from_seed`. Two invocations on the same input produce byte-identical proof bytes; pinned per crate by `prove_canonical_is_byte_deterministic`.
- A `halo2_proofs` version bump may shift proof bytes; regen via the workspace-excluded `tools/halo2_reference_regen/` / `tools/halo2_production_mlp_regen/`.
- Production-MLP detailed benchmark record: [`docs/stage11d.2-benchmark-record.md`](stage11d.2-benchmark-record.md).

---

## 3. Dormant / blocked items (intentionally inactive)

| Item | Why blocked | Where tracked |
| --- | --- | --- |
| Production proof mainnet eligibility | `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` empty; layer-6 refusal is sole gate | Stage 11d.3A evidence bundle §§6–9; chain-team Q1–Q9 |
| Chain Proof Eligibility Registry **consumption** | Chain-side `CandidateRefused` record does not yet exist | Stage 11d.3C (deferred) |
| `Active` registry record / activation height | Chain-team governance owns this | `sum-chain#21` + Stage 11d.3A §8 |
| Chain-side SNARK verification | Distinct architectural fork; not chosen for v1 (register-only recommended) | Stage 11d.3A §8 alternative path |
| EZKL | License rejected — no LICENSE file at `zkonduit/ezkl` as of 2026-02-20 | Stage 14.4 doc; no revisit unless upstream license changes |
| Arbitrary model proving / ONNX runtime proving | Out of Phase 5 scope; would require separately-vetted prover with clean license | n/a — future research track |
| Staking / slashing / rewards | Depends on chain-side eligibility being live | Phase 5 tokenomics, post-Stage 11d.3C |

---

## 4. Validation strategy

Stage 14.8 introduces **no duplicated long-running prover tests**. The
existing per-feature CI matrix re-runs untouched:

- `default (read-only) — build + test` (no halo2 / prove features)
- `--features halo2-reference-verify — verifier-only build + test`
- `--features halo2-reference-prove — prover-and-verifier build + test`
- `--features stage11d-production-verify — verifier-only build + test`
- `--features stage11d-production-prove — prover-and-verifier build + test`
- `--features submit — build + test`
- Tree-isolation gates (default, halo2-reference-verify, halo2-reference-prove,
  stage11d-production-verify, stage11d-production-prove,
  default-must-NOT-contain-sumchain, submit-MUST-contain-sumchain)
- `Stage 11a proof pipeline fixture — byte-stable`
- `Stage 6 chain-wire + fixture — byte-stable`
- `omni-contributor — default-features build + test`

**One new test** is added to provide drift safety for the Stage 14.8
readiness-checklist command examples — a clap-tree parse-check that
asserts the documented CLI flag long names exist on the actual subcommand
definitions. It does **not** invoke any prover. Gated
`cfg(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))`
so it runs inside the existing per-feature CI jobs without a new gate.

If a future PR renames a documented flag (`--proof-artifact`, `--input-i16`,
`--output-path`, `--emit-halo2-reference-proof`, `--emit-production-mlp-proof`,
`--stub-input`, `--runner`) without updating this doc and the runbook, the
smoke test fails.

---

## 5. Surface map

| File | Stage 14.8 change |
| --- | --- |
| This doc | New. The readiness packet. |
| [`docs/operator-runbook.md`](operator-runbook.md) | Append a **Stage 14 closure** subsection inside the "Phase 5 — real proof generation" area (after the Stage 14.x proof family comparison, before the Stage 11d.3A pointer). One paragraph + a "what runs where" matrix + a pointer to this doc. No mutation of existing Stage 14.x or Stage 11d.3 sections. |
| [`docs/stage14.7-proof-generation-acceptance-hardening.md`](stage14.7-proof-generation-acceptance-hardening.md) | One-line forward pointer to this doc in the existing "Future outlook" section. |
| [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) | One new test in `mod tests` (clap-tree parse-check, no prover invocation). |

**Zero changes** to: `crates/omni-zkml`, `crates/omni-sumchain`,
`crates/omni-contributor`, `crates/omni-proofs-halo2-reference`,
`crates/omni-proofs-halo2-production-mlp`, any `Cargo.toml`,
`.github/workflows/`, `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`, error
surface, or any per-stage Stage 14.1–14.6 doc.

---

## 6. Stage 14.x close

After Stage 14.8 the Stage 14.x proof-generation track is **closed**.
End-state coverage:

| Surface | Reference | Production |
| --- | --- | --- |
| Operator generate-proof | ✅ Stage 14.1 | ✅ Stage 14.5 |
| Operator verify-proof dispatch | ✅ Stage 11b.1.b | ✅ Stage 11d.2 |
| Contributor StubRunner sidecar | ✅ Stage 14.2 | ✅ Stage 14.6 |
| Contributor ExternalCommandRunner sidecar | ✅ Stage 14.3 | ✅ Stage 14.6 |
| Verifier roundtrip integration tests | ✅ Stage 14.1 (4 tests) | ✅ Stage 14.5 (4 tests) |
| CI tree gates | ✅ Stage 14.1 | ✅ Stage 14.5 |
| Cross-family acceptance pin | ✅ Stage 14.7 | ✅ Stage 14.7 |
| Runbook canonical comparison + closure | ✅ Stage 14.7 + **14.8** | ✅ Stage 14.7 + **14.8** |
| Operator readiness checklist | ✅ **Stage 14.8** | ✅ **Stage 14.8** |
| CLI drift safety | ✅ **Stage 14.8** | ✅ **Stage 14.8** |
| Mainnet eligibility | ❌ never — bounded reference, perpetual testnet/dev | ❌ today — pending Stage 11d.3C+ |

---

## 7. Stage 15 handoff (conditional)

The starting point for Stage 15 depends on chain-team state at that moment.

### Branch A — chain-side `CandidateRefused` record exists for `Stage11dProductionFixedPointMlp`

**Stage 15.0 = Stage 11d.3C-style consumption.** Mirror or live-read the
chain registry into OmniNode's local mainnet policy. Specifically:

- Introduce a chain-read code path that resolves
  `(proof_system, circuit_id_hex, model_hash)` against the registry's records.
- Gate layer 6 of `check_mainnet_eligible` on the mirrored result (still
  empty at the close of Stage 15.0 if no record is `Active`).
- Hermetic-test path uses synthetic registry inputs; no live-chain tests.
- Activation height handling: respect `proof_eligibility_enabled_from_height`
  per the chain-side spec.

### Branch B — chain-side record has **not** landed

**Stage 15.0 = discovery for economics + production operator packaging.**
Three sub-tracks to scope independently:

- **Economics / proof policy discovery** — what proof-rate policy do
  contributors face? How does eligibility-registry membership interact with
  reward / slash semantics? Chain-team tokenomics design is the upstream.
- **Production operator packaging** — systemd unit improvements,
  observability for proof-generation latency, `RUST_MIN_STACK` tuning per
  host, release-bundle hardening (Stage 10b precedent).
- **Phase 5 release-readiness audit** — extend
  [`docs/phase5-rc-audit.md`](phase5-rc-audit.md) to cover the closed Stage
  14 surface end-to-end.

Stage 14.8 captures both branches so the next plan is pickable without
re-litigating which fork is live.

---

## 8. Out of scope

Explicitly excluded from Stage 14.8:

- Implementation of any Proof Eligibility Registry consumption (Stage 11d.3C+ territory).
- Activation, any `Active` record, any `proof_eligibility_enabled_from_height` ≠ `None` change.
- New proof systems or `ProofSystem` enum variants.
- New `ModelFormat` enum variants.
- Chain-side SNARK verifier.
- Staking / slashing / reward distribution.
- Release packaging beyond the documented future-work pointer.
- EZKL revisit beyond restating the existing Stage 14.4 rejection.
- Any `crates/omni-zkml/src/error.rs` change.
- Any `Cargo.toml` dependency edit.
- Any new CI workflow gate.
- Renaming the grandfathered identifiers (`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`, etc.).
- Touching the unrelated `env_allowlist` subsystem in `crates/omni-contributor/src/runner.rs`.
- Mutation of any per-stage Stage 14.1–14.6 engineering doc, Stage 11d.3A/B docs, or the README.

---

## 9. Future outlook

- **Stage 11d.3C** — OmniNode-side consumption of chain registry state.
  Blocked on the chain-side `CandidateRefused` record existing first. See
  Branch A above.
- **Stage 15** — economics / proof policy discovery, production operator
  packaging, or Stage 11d.3C consumption (per §7).
- **Stage 10b** — release-bundle / packaging hardening.
- **Stage 11e (speculative)** — research track for any future GGUF /
  transformer-class proof strategy, if chain team commits to one. None is
  on the current roadmap.

Stage 14.8 is the final "feature-adjacent" stage in the 14.x range. After
this, the Phase 5 zkML track moves to chain-side dependencies (Stage 11d.3C
when the chain record exists) and Phase 5 release packaging (Stage 10b).
