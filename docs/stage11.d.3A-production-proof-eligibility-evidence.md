# Stage 11d.3A — Production proof eligibility evidence bundle

**Status: docs / evidence only.** No engineering activation. No mutation of
`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`. No code changes to eligibility logic,
no chain RPC changes, no proof artifact / verifier / prover / contributor /
`ProofSystem` / `ModelFormat` schema changes, no default-build dependency
changes. The production proof family
(`ProofSystem::Stage11dProductionFixedPointMlp`) **remains dormant / not
mainnet-eligible** at the end of this bundle.

Bundle scope: surface the evidence chain-team asked for after landing the
dormant **Proof Eligibility Registry** design package on the chain side
(`sum-chain#21`):

- `docs/SUBPROTOCOLS/PROOF-ELIGIBILITY-REGISTRY.md`
- `docs/SUBPROTOCOLS/PROOF-ELIGIBILITY-REGISTRY-ACTIVATION.md`

Chain confirmed `Stage11dProductionFixedPointMlp` stays rejected/dormant until
OmniNode supplies this bundle and the architectural fork (register-only vs.
chain-side proof verification) is settled.

The reference proof family (`Stage11bHalo2Reference`) remains dev/testnet only
in perpetuity per Stage 11d.0 criteria §3 non-goals. EZKL remains rejected per
Stage 14.4 (no license file at `zkonduit/ezkl` as of 2026-02-20). Neither is
revisited here.

---

## 1. Status / recommendation

- **This bundle is evidence for chain review, not activation.** No eligibility registry
  mutation, no registry record write, no eligibility flip.
- **Current eligibility:** `Stage11dProductionFixedPointMlp` is **dormant /
  rejected** by OmniNode local mainnet policy. `check_mainnet_eligible` returns
  `Err(NotInMainnetAllowlist { … })` for every production artifact because
  `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` is empty by design (Stage 11d.1
  schema, Stage 11d.2 verifier shipped without populating it).
- **Recommended v1 assumption:** **register-only chain registry.** Chain
  records eligibility metadata and activation state; chain does **not** verify
  SNARK proofs in v1. OmniNode's off-chain verifier
  (`Halo2ProductionMlpVerifier`) remains the enforcement point. Activation
  flips a chain-side record, not a SNARK verifier.
- **Why register-only:**
  - The chain currently treats proof artifacts opaquely (Stage 13.x evidence
    anchors carry a digest of the artifact, not the artifact). Adding a SNARK
    verifier to consensus is a separate design.
  - OmniNode already verifies off-chain via a fully-pinned verifier with
    drift-detection on construction; lifting eligibility does not require
    chain-side cryptography to gain.
  - Validator-side SNARK verification is a distinct threat-model / cost /
    upgrade decision the chain team owns. We do not want this bundle to
    pre-commit either side to it.
  - Register-only leaves a clean upgrade path: chain-side verification can be
    layered on later without re-deriving the identity tuple.

This bundle does not prescribe chain-side verification. Section 8 lists what
would change if chain chose it.

---

## 2. Candidate identity tuple

Exact values for the future `AllowlistEntry` (Stage 11d.1 schema in
[`crates/omni-zkml/src/proof.rs`](../crates/omni-zkml/src/proof.rs)) and/or the
first chain-side registry record. All values are pinned constants today; the
local-policy keys are the triple `(proof_system, circuit_id_hex, model_hash)`.

| Field | Value | Source / pin |
| --- | --- | --- |
| `proof_system` | `ProofSystem::Stage11dProductionFixedPointMlp` | [`crates/omni-zkml/src/proof.rs`](../crates/omni-zkml/src/proof.rs) enum variant |
| `backend_id` | `production-fixedpoint-mlp-v1` | [`crates/omni-proofs-halo2-production-mlp/src/shared.rs:18`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L18) `BACKEND_ID` |
| `model_format` | `ModelFormat::ProductionFixedPointMlp` | [`crates/omni-zkml/src/proof.rs`](../crates/omni-zkml/src/proof.rs) enum variant |
| `model_framework` | `ModelFramework::FrameworkAgnostic` | Set by the prover backend on artifact metadata; canonical spec is framework-neutral |
| `circuit_id_hex` | `593d027df3778bc582f9ec40bf453e757a1be6a9b6961243f2dfdf38fb4ea95d` | [`shared.rs:64-65`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L64-L65) `EXPECTED_CIRCUIT_ID_HEX` |
| `model_hash` | `1c95eea59ab7fe811f1a3c668798221577225c917846888a803b939f9cbda741` | `EXPECTED_PRODUCTION_SPEC_HASH` — BLAKE3 of [`assets/canonical_spec.json`](../crates/omni-proofs-halo2-production-mlp/assets/canonical_spec.json); injected by [`build.rs`](../crates/omni-proofs-halo2-production-mlp/build.rs) at compile time |
| `verification_key_hash_hex` | `2ec18faed223a28a23155492459c507a2672b9ff495c1df566103a19638655a9` | [`shared.rs:73-74`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L73-L74) `EXPECTED_VK_HASH_HEX` |
| `chain_team_review_ref` | **TBD** — fill from chain-team / governance sign-off output (e.g. governance proposal ID, sum-chain PR number, signed review document hash) | Not derivable from OmniNode artifacts; required field on `AllowlistEntry` |
| Artifact-side `testnet_or_dev_only` | `Some(false)` | Stage 14.5 prover writes this on every production artifact; pinned by Stage 14.5 / 14.6 tests |
| Current mainnet eligibility | **Refused — `NotInMainnetAllowlist`** | Layer 6 of `check_mainnet_eligible`; `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES.is_empty()` |

The matching key the layered policy actually consults is the triple
`(proof_system, circuit_id_hex, model_hash)`; `verification_key_hash_hex` is an
audit-only field on the entry (the verifier separately enforces it against
artifact metadata). `backend_id`, `model_format`, `model_framework`, and
`chain_team_review_ref` are also audit-only on the entry.

---

## 3. Hash derivations / provenance

### `circuit_id_hex`

- **Value:** `593d027df3778bc582f9ec40bf453e757a1be6a9b6961243f2dfdf38fb4ea95d`
- **Constant:** `EXPECTED_CIRCUIT_ID_HEX` —
  [`shared.rs:64-65`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L64-L65)
- **Derivation:** bare `BLAKE3(vk_canonical_bytes(&vk))` where `vk` is the
  halo2 `VerifyingKey<EqAffine>` for the production circuit at
  `HALO2_K = 11`. The live derivation lives in
  [`verifier.rs`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs)
  as `live_circuit_id_hex(&vk)`.
- **Drift detection:** `Halo2ProductionMlpVerifier::from_embedded_fixtures`
  re-derives the live circuit_id at construction and refuses to construct if
  it differs from `EXPECTED_CIRCUIT_ID_HEX`
  ([`verifier.rs:169-176`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs#L169-L176)).
  A `halo2_proofs` version bump, unintended circuit edit, or `HALO2_K` change
  fails verifier construction loudly. The artifact-side check then enforces
  the same string against `metadata.circuit_id_hex`
  ([`verifier.rs:299-311`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs#L299-L311)).

### `verification_key_hash_hex`

- **Value:** `2ec18faed223a28a23155492459c507a2672b9ff495c1df566103a19638655a9`
- **Constant:** `EXPECTED_VK_HASH_HEX` —
  [`shared.rs:73-74`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L73-L74)
- **Derivation:** `mainnet_vk_hash(vk_canonical_bytes(&vk))` —
  `BLAKE3("OMNINODE-VK:v1:" || vk_canonical_bytes(&vk))` per Stage 11d.0
  criteria §1.7 (domain-separated, distinct from raw VK BLAKE3). Helper lives
  at [`crates/omni-zkml/src/proof.rs:361`](../crates/omni-zkml/src/proof.rs#L361)
  `mainnet_vk_hash`; the live derivation in the production crate is
  `live_vk_hash_hex(&vk)` in
  [`verifier.rs`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs).
- **Drift detection:** same construction-time identity check
  ([`verifier.rs:177-184`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs#L177-L184))
  and same artifact-metadata enforcement
  ([`verifier.rs:321-331`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs#L321-L331)).
- **Naming note:** the eligibility registry field is `verification_key_hash_hex`; the
  artifact metadata field is `verification_key_hex`. Both carry the same hash
  string; the names differ for historical schema reasons.

### `model_hash` (= canonical spec hash)

- **Value:** `1c95eea59ab7fe811f1a3c668798221577225c917846888a803b939f9cbda741`
- **Constant:** `EXPECTED_PRODUCTION_SPEC_HASH: [u8; 32]` — generated by
  [`build.rs`](../crates/omni-proofs-halo2-production-mlp/build.rs) at compile
  time and `include!`ed at
  [`shared.rs:76-80`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L76-L80).
- **Source file:**
  [`crates/omni-proofs-halo2-production-mlp/assets/canonical_spec.json`](../crates/omni-proofs-halo2-production-mlp/assets/canonical_spec.json)
  — 9320 bytes, BLAKE3 confirmed via `b3sum` 2026-06-23.
- **Derivation:** bare `BLAKE3(read_to_end(canonical_spec.json))`. `build.rs`
  emits `cargo:rerun-if-changed` on the spec path, so any edit to the spec
  produces a different constant and a recompile.
- **Distinguishability:** distinct constant name (`EXPECTED_PRODUCTION_SPEC_HASH`)
  from the reference crate's `EXPECTED_SPEC_HASH` per criteria §1.6 hard rule
  H3 — guarantees no cross-import collision at the type level.

### `params.bin` (halo2 universal parameters)

- **Path:**
  [`crates/omni-proofs-halo2-production-mlp/fixtures/halo2/params.bin`](../crates/omni-proofs-halo2-production-mlp/fixtures/halo2/params.bin)
- **Size:** 131,140 bytes.
- **Embedded into:** the verifier via `include_bytes!` at
  [`verifier.rs:133-134`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs#L133-L134);
  `from_embedded_fixtures` is the construction surface. Operator + contributor
  CLIs reach this through the production verifier path only — no on-disk
  loading at runtime.
- **Provenance:** generated by the workspace-excluded regen tool
  `tools/halo2_production_mlp_regen/` (not in the default build tree). The
  prover re-derives from the same params; the
  `prove_canonical_is_byte_deterministic` test in
  [`crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs`](../crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs)
  pins that two runs from the same params + same seed produce byte-identical
  proof bytes.
- **Companion fixtures:** `proof.bin` (7,744 bytes) and `proof_artifact.json`
  (16,582 bytes) in the same directory, used by hermetic verifier tests.

### Halo2 dependency and seed

- **Crate:** `halo2_proofs = "0.3.2"` from crates.io, declared
  `default-features = false, optional = true` in
  [`crates/omni-proofs-halo2-production-mlp/Cargo.toml:62`](../crates/omni-proofs-halo2-production-mlp/Cargo.toml#L62).
  The `prove` feature pulls it; default builds do not.
- **`HALO2_K`:** `11` —
  [`shared.rs:51`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L51).
  Preauthorized ceiling per Stage 11d.2 plan §13 OQ9 is `k ≤ 13`;
  halt-and-report at `k ≥ 14`.
- **Prover RNG seed:** `PROVER_RNG_SEED: [u8; 32] = *b"OmniNode/Stage11d.2/prover-rngv1"` —
  [`prover.rs:51-53`](../crates/omni-proofs-halo2-production-mlp/src/prover.rs#L51-L53).
  Pure ASCII for grep-ability. Used to construct a `ChaCha20Rng`
  ([`prover.rs:89`](../crates/omni-proofs-halo2-production-mlp/src/prover.rs#L89));
  byte-determinism is pinned by the roundtrip test above.

A `halo2_proofs` version bump, a `HALO2_K` change, a circuit edit, a seed
change, or a `canonical_spec.json` edit are each detected: the first three
break VK identity (live `circuit_id_hex` / `vk_hash_hex` drift from pinned
constants → verifier refuses to construct); the seed change shifts proof
bytes (caught by the byte-determinism test); the spec edit changes
`EXPECTED_PRODUCTION_SPEC_HASH` and forces a recompile via
`cargo:rerun-if-changed`.

---

## 4. Implementation evidence

| Surface | Path | Notes |
| --- | --- | --- |
| Production circuit definition | [`crates/omni-proofs-halo2-production-mlp/src/circuit.rs`](../crates/omni-proofs-halo2-production-mlp/src/circuit.rs) | `16 → 32 → 16 → 8` int16 fixed-point MLP. Always-compiled crate; halo2 deps are feature-gated. |
| Canonical evaluator (pure-Rust reference) | [`crates/omni-proofs-halo2-production-mlp/src/canonical.rs`](../crates/omni-proofs-halo2-production-mlp/src/canonical.rs) | Framework-neutral; pinned via `canonical_invariant_holds` cross-framework corpus tests. |
| Production prover backend | [`crates/omni-proofs-halo2-production-mlp/src/prover.rs`](../crates/omni-proofs-halo2-production-mlp/src/prover.rs) | `Halo2ProductionMlpProofBackend`; gated `feature = "stage11d-production-prove"`. Deterministic via `PROVER_RNG_SEED`. |
| Production verifier (with drift detection) | [`crates/omni-proofs-halo2-production-mlp/src/verifier.rs`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs) | `Halo2ProductionMlpVerifier::from_embedded_fixtures` re-derives VK identity at construction and refuses if drifted from pinned `EXPECTED_CIRCUIT_ID_HEX` / `EXPECTED_VK_HASH_HEX`. |
| Operator generate-proof CLI | [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) `generate-production-mlp-proof` subcommand | Stage 14.5; gated `feature = "stage11d-production-prove"`. |
| Operator verify-proof dispatch | [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) `verify-proof` arm | Stage 11d.2 added the production dispatch arm; routes to `Halo2ProductionMlpVerifier`. |
| Contributor StubRunner sidecar | [`crates/omni-node/src/contributor_cli.rs`](../crates/omni-node/src/contributor_cli.rs) `emit_production_mlp_proof_sidecar` | Stage 14.6. Requires `--stub-input` (32 bytes for production shape). |
| Contributor ExternalCommandRunner sidecar | [`crates/omni-node/src/contributor_cli.rs`](../crates/omni-node/src/contributor_cli.rs) `emit_production_mlp_proof_sidecar_from_bytes` | Stage 14.6. Uses the shared Stage 14.3 `ByteCapturingRunner` trait wrapper. |
| Metadata contract enforcement | [`crates/omni-proofs-halo2-production-mlp/src/verifier.rs:299-331`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs#L299-L331) | Verifier rejects on missing/drifted `circuit_id_hex` or `verification_key_hex`. |
| Local mainnet policy (layered refusal) | [`crates/omni-zkml/src/proof.rs:923`](../crates/omni-zkml/src/proof.rs#L923) `check_mainnet_eligible` | Six refusal layers; production artifacts hit layer 6 only today. |
| Allowlist schema | [`crates/omni-zkml/src/proof.rs:317`](../crates/omni-zkml/src/proof.rs#L317) `AllowlistEntry` | Stage 11d.1. |
| Allowlist storage (empty by design) | [`crates/omni-zkml/src/proof.rs:374`](../crates/omni-zkml/src/proof.rs#L374) `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES: &[AllowlistEntry] = &[]` | Empty. Stage 11d.3A does not change this. |

No changes to `crates/omni-zkml`, `crates/omni-sumchain`, `crates/omni-contributor`,
`crates/omni-proofs-halo2-reference`, `crates/omni-proofs-halo2-production-mlp`,
or any `Cargo.toml` are made by this bundle.

---

## 5. Metadata contract

A valid production artifact (`ProofArtifactBody`) must carry the following
metadata fields with these exact values. The verifier rejects any drift; the
acceptance suite (Stage 14.5 / 14.6 / 14.7) pins each field.

| Field | Required value |
| --- | --- |
| `proof_system` | `ProofSystem::Stage11dProductionFixedPointMlp` |
| `model_format` | `ModelFormat::ProductionFixedPointMlp` |
| `backend_id` | `"production-fixedpoint-mlp-v1"` |
| `circuit_id_hex` | `Some("593d027df…fb4ea95d")` (must equal `EXPECTED_CIRCUIT_ID_HEX`) — **required, not optional** |
| `verification_key_hex` | `Some("2ec18fae…638655a9")` (must equal `EXPECTED_VK_HASH_HEX`) — **required, not optional** |
| `model_hash` | `EXPECTED_PRODUCTION_SPEC_HASH` hex (`1c95eea5…9cbda741`) |
| `testnet_or_dev_only` | `Some(false)` — load-bearing for layer 1 of `check_mainnet_eligible` |
| Public inputs shape | input = 16 × i16 LE (32 bytes); output = 8 × i16 LE (16 bytes) |
| `contributor_job_id` extra key | tolerated — the verifier accepts a `contributor_job_id` key in public inputs and ignores it (D2 regression-pinned by Stage 14.6 test 10) |

Reference (`Stage11bHalo2Reference`) contract deliberately diverges:
`testnet_or_dev_only = Some(true)`, `circuit_id_hex` optional,
`verification_key_hex = None`, input/output arity 4 × i16 / 4 × i16. The
reference family is **never** mainnet-eligible; this bundle covers production
only.

---

## 6. Test / acceptance matrix

All tests are hermetic; no live-chain dependencies. Listed by stage.

### Stage 11d.2 — production verifier

Runs under `cargo test -p omni-proofs-halo2-production-mlp --features verify`.

- [`crates/omni-proofs-halo2-production-mlp/src/verifier.rs`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs) `mod tests` — VK identity / drift refusal, metadata-field refusal (missing `circuit_id_hex`, drifted `verification_key_hex`), `contributor_job_id` tolerance.
- [`crates/omni-proofs-halo2-production-mlp/tests/halo2_proof_verifies.rs`](../crates/omni-proofs-halo2-production-mlp/tests/halo2_proof_verifies.rs) — embedded-fixture roundtrip.
- [`crates/omni-proofs-halo2-production-mlp/tests/cross_framework_corpus.rs`](../crates/omni-proofs-halo2-production-mlp/tests/cross_framework_corpus.rs) — canonical evaluator parity.
- [`crates/omni-proofs-halo2-production-mlp/tests/chain_digest_roundtrip.rs`](../crates/omni-proofs-halo2-production-mlp/tests/chain_digest_roundtrip.rs) — artifact digest stability.

### Stage 14.5 — production prover roundtrip

Runs under `cargo test -p omni-proofs-halo2-production-mlp --features stage11d-production-prove`
and `cargo test -p omni-node --features stage11d-production-prove --bins` with
`RUST_MIN_STACK=67108864`.

- [`crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs`](../crates/omni-proofs-halo2-production-mlp/tests/halo2_production_mlp_prove_verify_roundtrip.rs) — 4 tests: canonical roundtrip, byte-determinism, tamper rejection, production-shape contract pin.
- [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) Stage 14.5 CLI tests (6 tests) — `generate-production-mlp-proof` happy path, mainnet refusal posture (layer 6 only), metadata contract, dispatch into `verify-proof`.

### Stage 14.6 — contributor production sidecar

Runs under `cargo test -p omni-node --features stage11d-production-prove --bins`.

[`crates/omni-node/src/contributor_cli.rs::tests::stage_14_6_production_mlp_sidecar_proof`](../crates/omni-node/src/contributor_cli.rs) — 11 hermetic tests including:

1. StubRunner happy path (sidecar parses with `proof_system=Stage11dProductionFixedPointMlp`, `testnet_or_dev_only=Some(false)`)
2. D6 — `ContributorResult` on-disk bytes unmutated by sidecar emission
3. Non-canonical model hash refusal
4. Stub-input hash mismatch refusal
5. Production-shape arity refusal (8-byte reference-shape input rejected)
6. End-to-end sidecar → verifier stitch
7. Clap mutual exclusion between reference and production emit flags (gated `cfg(all(halo2-reference-prove, stage11d-production-prove))`)
8. Mainnet posture pin — `testnet_or_dev_only=Some(false)` passes layer 1; `check_mainnet_eligible` fails at layer 6 only
9. `contributor_job_id` carried in public inputs (D2 presence)
10. Verifier tolerates extra `contributor_job_id` key (D2 regression pin)
11. `#[cfg(unix)]` real `ExternalCommandRunner` subprocess + shell-script fixture → `ByteCapturingRunner` → `emit_production_mlp_proof_sidecar_from_bytes` → verifier accepts

### Stage 14.7 — cross-family acceptance hardening

Runs under combinations of `--features halo2-reference-prove` and `--features stage11d-production-prove`.

[`crates/omni-node/src/operator.rs::tests`](../crates/omni-node/src/operator.rs) — 4 acceptance tests using `OnceLock`-cached prover artifacts:

1. `reference_and_production_artifacts_have_distinct_proof_systems_and_opposite_testnet_flags` — metadata contract diff pin, including `EXPECTED_CIRCUIT_ID_HEX` and `EXPECTED_VK_HASH_HEX` by exact value (gated on both features)
2. `reference_artifact_is_mainnet_refused_at_layers_1_3_and_6_with_distinct_reasons` — reference defense-in-depth (gated on `halo2-reference-prove`)
3. `production_artifact_is_mainnet_refused_at_layer_6_only_with_layer_1_passing` — production sole-gate posture (gated on `stage11d-production-prove`)
4. `verify_proof_dispatch_routes_mock_reference_and_production_artifacts_correctly` — dispatch correctness with both features active (gated on both)

### CI gates

- `halo2-reference-prove-build-test` exercises Stage 14.7 test 2.
- `stage11d-production-prove-build-test` exercises Stage 14.5, 14.6 (all 11 tests), and Stage 14.7 test 3.
- Tests 1 and 4 of Stage 14.7 are local-only (gated on both features simultaneously); the same gate covers the Stage 14.6 clap-conflict test 7.
- Default-build tree-isolation gates (Stage 11d.2 + Stage 14.5) assert
  `cargo build -p omni-node` pulls **zero** halo2 / pasta /
  `omni-proofs-halo2-production-mlp` / `rand_chacha` deps.

---

## 7. Current mainnet-refusal behavior

Layered refusal model in
[`check_mainnet_eligible`](../crates/omni-zkml/src/proof.rs#L923):

| Layer | Test on artifact metadata | Production artifact today | Reference artifact today |
| --- | --- | --- | --- |
| 1 | `testnet_or_dev_only` is `Some(true)` or absent → refuse | **Passes** — production sets `Some(false)` | **Refuses** — reference sets `Some(true)` |
| 2 | `MockBackend` → refuse | Passes — not Mock | Passes — not Mock |
| 3 | `BoundedReference` (reference family) → refuse | Passes — not reference | **Refuses** — reference family hits this layer |
| 4 | `model_format == Gguf` → refuse | Passes | Passes |
| 5 | `model_format == UnknownModelFormat` → refuse | Passes | Passes |
| 6 | `(proof_system, circuit_id_hex, model_hash)` triple ∉ `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` AND `proof_system` ∉ legacy `MAINNET_APPROVED_PROOF_SYSTEMS` | **Refuses** — empty eligibility registry (`NotInMainnetAllowlist` variant — name grandfathered) | **Refuses** — empty eligibility registry |

**Production:** **sole refusal is layer 6.** Layer 1 explicitly does not fire
because `testnet_or_dev_only = Some(false)` is load-bearing. Lifting eligibility
requires populating the local eligibility registry (or, in the chain-side-verify fork, the chain
registry record reaching `Active`).

**Reference:** layers 1 + 3 + 6 all fire (defense in depth). Reference is
intentionally never mainnet-eligible per Stage 11d.0 §3.

**No chain-side activation exists yet in OmniNode.** No chain RPC reads the
local eligibility registry; no local logic reads a chain registry. The Stage 11d.0
authoritative docs and Stage 11d.1 / 11d.2 plans all describe activation as a
future track gated on chain-team review — that is what this bundle exists to
unblock.

---

## 8. Architectural fork: register-only vs chain-side verify

OmniNode's recommended v1: **register-only governance/audit registry.**

### Recommended (register-only)

- Chain ships a registry contract / pallet (already designed dormant in
  `sum-chain#21`) that records eligibility metadata and an activation state
  (`CandidateRefused`, `CandidateApproved`, `Active`, `Superseded`, …).
- A first candidate record is written carrying the identity tuple from §2
  plus the chain-team `chain_team_review_ref`.
- Activation flips a record from `CandidateApproved` → `Active` at a
  predetermined height. Until then, OmniNode local policy continues to refuse.
- Once `Active`, OmniNode reads (or mirrors) the chain registry and treats it
  as the source of truth — the local `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`
  is either populated to match or replaced by chain-driven entry resolution.
- **Chain does not verify SNARK proofs in v1.** OmniNode's
  `Halo2ProductionMlpVerifier` remains the cryptographic enforcement point;
  the chain enforces only "is this `(proof_system, circuit_id_hex, model_hash)`
  registered and `Active`."
- Validators run the existing OmniNode binary as the off-chain verifier;
  consensus accepts the layered refusal output.

**Why preferred:** smallest chain-side surface; matches today's evidence-anchor
model (chain stores digests, not artifact bytes); leaves chain-side
verification as a clean optional upgrade; does not commit validators to halo2
verification cost; preserves the verifier's drift-detection guarantees.

### Alternative (chain-side verify)

If chain chooses chain-side SNARK verification, the following additional work
is required before any eligibility registry activation:

- **Third-party cryptographic audit** of the halo2 circuit + canonical spec +
  VK derivation pipeline (Section 9 Q2). The OmniNode evidence in this bundle
  is *internal* — internal correctness is necessary but not sufficient for
  validator-executed cryptography.
- **Validator execution cost analysis:** measure per-block cost of verifying
  a Stage 14.5 production proof on validator hardware; settle on a gas /
  weight model. The CPU-host reference is ~30 s per proof; validator costs
  differ.
- **Proof format compatibility constraints:** chain consensus must commit to
  `halo2_proofs = "0.3.2"` exactly (or an audited equivalent); a chain
  runtime upgrade is required for any prover-side `halo2_proofs` bump
  (today, OmniNode handles this with a fixture-regen + a fresh eligibility registry
  entry — chain-side verification removes that flexibility).
- **New chain-side verifier code and tests:** a separate Rust crate (or
  equivalent) implementing `Halo2ProductionMlpVerifier` semantics with no
  `omni-zkml` dependency, plus byzantine test coverage.
- **Different activation risk:** activation flips a cryptographic verifier
  on consensus, not a registry record. Rollback semantics differ; a
  chain-side bug becomes a consensus bug.

This alternative is out of scope for Stage 11d.3A. If chain selects it, a
separate stage (Stage 11d.4 or similar) opens to track it.

---

## 9. Risk / audit questions

Outstanding governance / chain-team decisions blocking any Stage 11d.3
implementation PR. None of these are engineering questions; engineering can
react to any answer, but the answers must come from chain-team / governance.

| # | Question | Why it blocks |
| --- | --- | --- |
| Q1 | Is internal review sufficient for a register-only first record, or is third-party crypto audit required up front? | Determines whether Stage 11d.3 (register-only) can land with the existing internal evidence in §3–§6, or must wait for an external review. |
| Q2 | If chain-side verification is selected, is third-party crypto audit a hard precondition before any `CandidateApproved` record? | Activation risk is materially different; see §8. |
| Q3 | Is the params / VK / circuit regeneration policy acceptable? Specifically: regen via workspace-excluded `tools/halo2_production_mlp_regen/`, deterministic via `PROVER_RNG_SEED`, drift detection via `Halo2ProductionMlpVerifier::from_embedded_fixtures`. | Determines whether the candidate identity tuple in §2 is the one chain-team will sign. |
| Q4 | What is the rollback procedure if an `Active` record needs to be revoked? Append a superseding record? Add a `Revoked` state? Both? | Determines whether OmniNode local policy needs a `Superseded`/`Revoked` honoring path. |
| Q5 | What is the activation height? Is a dry-run `CandidateRefused` interval required between `CandidateApproved` and `Active`? | Affects whether OmniNode needs a "shadow eligibility" mode to publish refused-only artifacts for monitoring before live activation. |
| Q6 | What exact registry record fields does chain want in the first candidate? Just the identity tuple from §2, or also `backend_id`, `model_format`, `model_framework`, `chain_team_review_ref`, evidence-bundle hash, fixture-set hash? | Determines the on-chain schema commit. |
| Q7 | What is the value of `chain_team_review_ref` for the first record? (A governance proposal ID, a signed document hash, a sum-chain PR number, …) | Required field on `AllowlistEntry`; this bundle cannot mint it. |
| Q8 | Does chain want OmniNode to mirror chain registry state locally (and accept temporary divergence), or read it on every `check_mainnet_eligible` call? | Determines the local-policy refactor scope after activation. |
| Q9 | If a future `halo2_proofs` bump is needed (security patch, performance), is the chain registry expected to carry a new record (new VK / circuit_id), or does the chain absorb a runtime-side upgrade? Section 10 assumes the former. | Determines upgrade ergonomics. |

A Stage 11d.3 code PR should not open until Q1, Q3, Q6, Q7, and Q5 have
written answers at minimum.

---

## 10. Rollback / regeneration assumptions

Recommended invariants for the chain-side registry contract, assuming
register-only. These are OmniNode's working assumptions and are surfaced for
chain-team confirmation:

- **Append-only.** The registry should accumulate records; no in-place
  mutation. Deactivation is a new superseding record (state transition
  `Active` → `Superseded`), not a delete or field-overwrite. This matches the
  Stage 13.x evidence-anchor invariant on the OmniNode side.
- **Regeneration triggers a new record.** Any of the following force a new
  evidence bundle (this doc, refreshed) and a new registry record:
  - `halo2_proofs` version bump (changes VK derivation → new `circuit_id_hex`
    and `verification_key_hash_hex`).
  - Circuit edit (`crates/omni-proofs-halo2-production-mlp/src/circuit.rs`)
    → new `circuit_id_hex`.
  - `HALO2_K` change → new `circuit_id_hex`.
  - `assets/canonical_spec.json` edit → new `model_hash` /
    `EXPECTED_PRODUCTION_SPEC_HASH`.
  - `PROVER_RNG_SEED` change → no VK change, but proof bytes shift; treat as a
    new record because the prover identity changed.
- **Off-chain verifiability is preserved.** Existing artifacts produced under
  a now-superseded record remain verifiable off-chain (the verifier's pinned
  constants are tied to its compiled binary, not the chain registry). But
  superseded artifacts should not be treated as **eligible** under the new
  record — eligibility tracks the live `Active` record's identity tuple.
- **Reactivation is by new record.** A previously-superseded
  `(proof_system, circuit_id_hex, model_hash)` triple can re-enter `Active`
  only through a fresh `CandidateApproved` record with chain-team review;
  flipping a `Superseded` record back to `Active` is disallowed.
- **One Active record per `proof_system`.** The recommended invariant: at
  most one `Active` record per `ProofSystem` enum variant. This keeps the
  layer-6 match deterministic. Chain-team may choose differently (e.g.
  multiple concurrent `Active` records for graceful migration); OmniNode can
  honor either, but the local layer-6 lookup needs the chosen invariant
  written down before code.

---

## 11. Out of scope

Explicitly **not** part of this bundle:

- Code activation of any production proof system. `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`
  stays empty.
- Chain-side verifier implementation. See §8 alternative.
- Staking / slashing / reward distribution. Phase 5 tokenomics depends on
  chain-side eligibility being live; that's downstream.
- EZKL. Rejected per Stage 14.4 (no license file).
- Arbitrary models / ONNX. The canonical
  `production-fixedpoint-mlp-v1` is the only candidate; a real-model proof
  class requires a separately-vetted prover (license-blocked or new design).
- Reference proof (`Stage11bHalo2Reference`) mainnet eligibility. Reference
  stays dev/testnet only in perpetuity per Stage 11d.0 §3 non-goals.
- Any chain RPC, contributor schema, or `ProofArtifactBody` schema change.
- Any rewrite of the layered refusal model in `check_mainnet_eligible`.

---

## Verification

- No source code changes; this bundle is docs-only.
- `git diff` shows only:
  - new file: `docs/stage11.d.3A-production-proof-eligibility-evidence.md` (this doc)
  - additive pointer section in `docs/operator-runbook.md`
- All hashes and constants in §2–§5 cross-checked against pinned source today:
  - `EXPECTED_CIRCUIT_ID_HEX` = `593d027df3778bc582f9ec40bf453e757a1be6a9b6961243f2dfdf38fb4ea95d`
    ([`shared.rs:64-65`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L64-L65))
  - `EXPECTED_VK_HASH_HEX` = `2ec18faed223a28a23155492459c507a2672b9ff495c1df566103a19638655a9`
    ([`shared.rs:73-74`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs#L73-L74))
  - `EXPECTED_PRODUCTION_SPEC_HASH` = BLAKE3 of canonical_spec.json =
    `1c95eea59ab7fe811f1a3c668798221577225c917846888a803b939f9cbda741`
    (`b3sum` confirmed 2026-06-23)
  - `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` = `&[]`
    ([`proof.rs:374`](../crates/omni-zkml/src/proof.rs#L374))

**Awaiting chain-team / governance review before any Stage 11d.3 implementation
or activation work.**
