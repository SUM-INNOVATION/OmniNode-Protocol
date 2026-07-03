# Halo2 Production-MLP Proof — Operator & Integrator Guide

Feature-focused user guide for the Halo2 **production** fixed-point
MLP proof class (`production-fixedpoint-mlp-v1`). Covers what the
feature is, how to build and use it, the numeric contract, pinned
constants, dependency posture, and current mainnet-eligibility
posture (**dormant** — no chain-side activation).

**Audience:** operators running `omni-node`, contributors emitting
sidecar proofs alongside jobs, integrators verifying artifacts.

**Not** the chain-team review packet — for that, see
[`docs/stage11d.2-review-packet-production-fixedpoint-mlp.md`](stage11d.2-review-packet-production-fixedpoint-mlp.md).

**Not** the per-stage engineering docs — for those, see:

- [`docs/stage14.5-halo2-production-mlp-prove.md`](stage14.5-halo2-production-mlp-prove.md) — operator prover engineering.
- [`docs/stage14.6-contributor-production-mlp-proof.md`](stage14.6-contributor-production-mlp-proof.md) — contributor sidecar engineering.
- [`docs/stage14.8-proof-generation-readiness.md`](stage14.8-proof-generation-readiness.md) — proof-generation track closure / readiness.

**Feature distinction (important).** The Halo2 **production** MLP
proof is distinct from the Halo2 **reference** MLP proof and the two
are never interchangeable:

| Aspect | Reference (`halo2-mlp-v1`) | **Production (`production-fixedpoint-mlp-v1`)** |
| --- | --- | --- |
| Architecture | 4 → 8 → 4 int16 MLP | **16 → 32 → 16 → 8 int16 MLP** |
| `HALO2_K` | 10 | **11** |
| `ProofSystem` variant | `Stage11bHalo2Reference` | `Stage11dProductionFixedPointMlp` |
| `ModelFormat` variant | `Halo2ReferenceMlp` | `ProductionFixedPointMlp` |
| Verify feature flag | `halo2-reference-verify` | `stage11d-production-verify` |
| Prove feature flag | `halo2-reference-prove` | `stage11d-production-prove` |
| `testnet_or_dev_only` | `Some(true)` — bounded testnet/dev only | `Some(false)` — production-shape |
| Mainnet refusal layers fired | 1 + 3 + 6 (defense in depth) | **6 only** (sole gate; empty registry) |
| Cleared for mainnet today? | **No** — bounded reference, testnet/dev in perpetuity per Stage 11d.0 §1.6 hard rule H1 | **No** — pending Stage 11d.3C+ (chain-side dependency) |

Both paths verify off-chain only. The chain treats proof artifacts
opaquely today; validators do not run halo2 verification.

---

## 1. What the feature is

`ProofSystem::Stage11dProductionFixedPointMlp` proves that a
**16 → 32 → 16 → 8** fixed-point (`i16`) multilayer perceptron
executed a specific deterministic inference on a specific input,
against a canonical spec pinned by hash in-source. The proof is:

- **halo2 IPA / Pasta curves** — no trusted setup; PLONK-style.
- **Byte-deterministic** — same input + spec produces byte-identical
  proof bytes across hosts (fixed `ChaCha20Rng` seed).
- **Off-chain** — verifiers accept the proof from a
  `ProofArtifactBody` JSON; nothing is submitted to a chain today.
- **Independently regeneratable** — the workspace-excluded
  [`tools/halo2_production_mlp_regen/`](../tools/halo2_production_mlp_regen/)
  crate can re-derive `params.bin`, `proof.bin`, and
  `proof_artifact.json` from the source tree.
- **Drift-detected** — the verifier re-derives the VK's identity
  hashes at construction time and refuses to load if they don't
  match the pinned `EXPECTED_CIRCUIT_ID_HEX` /
  `EXPECTED_VK_HASH_HEX` — a `halo2_proofs` version bump, an
  unintended circuit edit, or a `HALO2_K` change all fail loudly.

The feature ships in the standalone
[`omni-proofs-halo2-production-mlp`](../crates/omni-proofs-halo2-production-mlp/)
crate; the operator binary pulls it only when a
`stage11d-production-{verify,prove}` cargo feature is active.

---

## 2. How to build

Two orthogonal cargo features on the `omni-node` binary:

| Feature | Pulls | Adds |
| --- | --- | --- |
| `stage11d-production-verify` | `halo2_proofs`, `omni-proofs-halo2-production-mlp` | `operator verify-proof` dispatch arm for production artifacts |
| `stage11d-production-prove` (superset) | above + `omni-proofs-halo2-production-mlp/prove`, `rand_chacha` | `operator generate-production-mlp-proof` + contributor `--emit-production-mlp-proof` |

Defaults (no features) pull **zero** halo2 / pasta / `rand_chacha`
/ `omni-proofs-halo2-*` — CI enforces this via the
`default tree — must NOT contain …` gate.

### Common build commands

```bash
# Verify-only (small; can accept and verify production artifacts):
cargo build -p omni-node --features stage11d-production-verify

# Prover + verifier (larger; can also generate production artifacts):
cargo build -p omni-node --features stage11d-production-prove

# For prover invocation (both build- and run-time), raise stack:
RUST_MIN_STACK=67108864 cargo test -p omni-node \
    --features stage11d-production-prove
```

`RUST_MIN_STACK=67108864` (64 MB) is required at **runtime** on the
`generate-production-mlp-proof` code path — the constraint-system
walker for the wider circuit needs headroom. If you invoke
generation without it, the process may abort with a stack overflow.

Pinned dep versions:

- [`halo2_proofs = "0.3.2"`](../crates/omni-proofs-halo2-production-mlp/Cargo.toml)
  (default features off; optional; pulled only by the verify/prove
  features).
- `rand_chacha` — pulled only under `stage11d-production-prove`.
- `blake3` — always compiled; provides the canonical hash chain.

---

## 3. Operator commands

Requires `--features stage11d-production-prove` (prove) or
`stage11d-production-verify` (verify only).

### Generate a production-MLP proof

```bash
RUST_MIN_STACK=67108864 \
cargo run -p omni-node --features stage11d-production-prove -- \
    operator generate-production-mlp-proof \
        --input-i16 "-5,10,20,-100,7,-3,14,25,-8,1,11,-22,4,-1,17,9" \
        --output-path /tmp/prod-proof.json
```

- `RUST_MIN_STACK=67108864` (64 MB) is set at prover invocation time
  — the constraint-system walker for the wider circuit needs
  headroom. Without it, the process may abort with a stack overflow.
- `--input-i16` takes 16 comma-separated `i16` values (the
  production MLP's input arity).
- `--output-path` receives a `ProofArtifactBody` JSON. The 8-element
  output vector is derived deterministically by the canonical
  evaluator; you do not supply it.

The command runs the prover on the operator host (~30 s CPU host on
dev hardware; see [`docs/stage11d.2-benchmark-record.md`](stage11d.2-benchmark-record.md)).
The artifact declares
`testnet_or_dev_only = Some(false)` (production-shape).

### Verify a production-MLP proof

```bash
cargo run -p omni-node --features stage11d-production-verify -- \
    operator verify-proof --proof-artifact /tmp/prod-proof.json
```

Expected verify output (bare stdout — 5 or 6 lines):

```
backend_id=production-fixedpoint-mlp-v1
proof_system=Stage11dProductionFixedPointMlp
model_format=ProductionFixedPointMlp
verified=true
mainnet_eligible=false
mainnet_refusal=NotInMainnetAllowlist { … }    # layer 6, sole gate
```

The 6th line (`mainnet_refusal`) appears only when the artifact is
refused — which is every production artifact today (§7 posture).
`mainnet_eligible=false` is by design.

---

## 4. Contributor sidecar

Requires `--features stage11d-production-prove`.

Alongside a produced `ContributorResult` (which is itself signed by
the contributor's seed), the contributor CLI can emit a verifiable
`ProofArtifactBody` sidecar that binds the tuple
`(canonical spec, input bytes, response bytes)` to a production-MLP
proof. The sidecar is written as plain JSON — its cryptographic
verifiability comes from the halo2 SNARK inside, not from a
signature over the file bytes themselves:

### StubRunner (operator supplies exact bytes)

```bash
RUST_MIN_STACK=67108864 \
cargo run -p omni-node --features stage11d-production-prove -- \
    operator contributor run-job \
        --job <PATH-to-ContributorJob.json> \
        --runner stub \
        --stub-input <PATH-to-32-byte-input> \
        --stub-response <PATH-to-16-byte-response> \
        --seed-file <PATH-to-contributor-seed> \
        --emit-production-mlp-proof /tmp/prod-sidecar.json
```

`--stub-input` for the production case is **32 raw bytes** (16 × i16
little-endian). `--stub-response` is **16 raw bytes** (8 × i16 LE).
Both bind into the emitted sidecar's `input_hash` /
`response_hash`.

### ExternalCommandRunner (bytes captured at the trait boundary)

```bash
RUST_MIN_STACK=67108864 \
cargo run -p omni-node --features stage11d-production-prove -- \
    operator contributor run-job \
        --job <PATH-to-ContributorJob.json> \
        --runner external \
        --seed-file <PATH-to-contributor-seed> \
        --emit-production-mlp-proof /tmp/prod-sidecar.json
```

For `--runner external`, the input and response bytes are captured
at the `InferenceRunner::run` trait boundary by the
`ByteCapturingRunner` (Stage 14.3 pattern); no `--stub-input` is
required.

**Mutual exclusion:** `--emit-halo2-reference-proof` and
`--emit-production-mlp-proof` are clap-layer mutually exclusive
when both prove features are enabled — a single `run-job` invocation
emits at most one sidecar family.

Verifying a sidecar is the same command as a standalone artifact:

```bash
cargo run -p omni-node --features stage11d-production-verify -- \
    operator verify-proof --proof-artifact /tmp/prod-sidecar.json
```

---

## 5. Math / numeric contract

The production MLP is a **framework-neutral canonical spec** pinned
by hash at
[`crates/omni-proofs-halo2-production-mlp/assets/canonical_spec.json`](../crates/omni-proofs-halo2-production-mlp/assets/canonical_spec.json).

| Property | Value |
| --- | --- |
| Architecture | `16 → 32 → 16 → 8` |
| Data type | `i16` (int16 fixed-point) |
| Nonlinearity | ReLU |
| Accumulator width | `i64` |
| Rounding | Round-half-away-from-zero (RHAZ) |
| Saturation policy | Three-branch (`b_lo` / `b_in` / `b_hi`) with bias-before-saturation |
| `HALO2_K` | `11` (2048 rows) |
| Circuit gadgets | Reused from Stage 11c (RHAZ + saturation + ReLU + range checks) |
| Canonical spec hash | BLAKE3 over the committed JSON — pinned as `EXPECTED_PRODUCTION_SPEC_HASH` in a build-time `include!` |
| Prover RNG seed | `PROVER_RNG_SEED = *b"OmniNode/Stage11d.2/prover-rngv1"` (ASCII 32 bytes) |
| Prover determinism | Same input + spec ⇒ byte-identical proof bytes across hosts (`ChaCha20Rng` from the pinned seed) |

The full 16-element canonical input is defined in
[`shared.rs`](../crates/omni-proofs-halo2-production-mlp/src/shared.rs)
as `CANONICAL_INPUT`; the corresponding canonical output pins in the
same file at `CANONICAL_OUTPUT`. A `canonical_invariant_holds` test
pins `canonical_evaluate(CANONICAL_INPUT) == CANONICAL_OUTPUT`.

### Pinned identity hashes (drift-detected)

| Constant | Value |
| --- | --- |
| `EXPECTED_CIRCUIT_ID_HEX` | `593d027df3778bc582f9ec40bf453e757a1be6a9b6961243f2dfdf38fb4ea95d` |
| `EXPECTED_VK_HASH_HEX` | `2ec18faed223a28a23155492459c507a2672b9ff495c1df566103a19638655a9` |

Any halo2 version bump, circuit edit, or `HALO2_K` change breaks the
verifier's construction-time identity check
([`verifier.rs`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs))
loudly and requires a coordinated fixture regen via
[`tools/halo2_production_mlp_regen/`](../tools/halo2_production_mlp_regen/).

### Cross-framework corpora — Stage 11d.2 status

The production case ships **five committed corpus JSONs** under
[`crates/omni-proofs-halo2-production-mlp/tests/fixtures/`](../crates/omni-proofs-halo2-production-mlp/tests/fixtures/)
(`rumus_corpus.json`, `pytorch_corpus.json`, `tensorflow_corpus.json`,
`caffe_corpus.json`, `framework_agnostic_corpus.json`).

**Important:** at Stage 11d.2, all five files are populated by the
**Rust canonical evaluator** — no framework runtime is invoked at
fixture-regen time. This is a deliberate deferral, documented
directly in the regen tool's source at
[`tools/halo2_production_mlp_regen/src/main.rs`](../tools/halo2_production_mlp_regen/src/main.rs)
lines 400 / 414 / 428 / 443 with "Replace this file when the … exporter lands."

**Per-framework production-MLP exporters do NOT yet exist.** The
reference case ships
[`tools/{rumus,pytorch,tensorflow,caffe}_export/`](../tools/) as
standalone packages that invoke the actual framework runtime; the
production case does **not** yet ship
`tools/halo2_production_mlp_{rumus,pytorch,tensorflow,caffe}_export/`
equivalents. This is tracked as a follow-up; the corpora are
placeholders until those exporters land.

---

## 6. Verifier drift-detection contract

`Halo2ProductionMlpVerifier::from_embedded_fixtures()` (defined in
[`verifier.rs`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs))
re-derives the live VK's identity hashes and refuses to construct if
they don't match `EXPECTED_CIRCUIT_ID_HEX` / `EXPECTED_VK_HASH_HEX`.

A production artifact **must** additionally carry:

- `metadata.circuit_id_hex = Some("593d027d…fb4ea95d")` — required,
  must equal `EXPECTED_CIRCUIT_ID_HEX`.
- `metadata.verification_key_hex = Some("2ec18fae…638655a9")` —
  required, must equal `EXPECTED_VK_HASH_HEX`.
- `metadata.testnet_or_dev_only = Some(false)` — the production
  shape. This is the *only* difference between production and
  reference at layer 1 of `check_mainnet_eligible`.

Any drift is rejected before SNARK verification. Sidecar emission
(§4) writes all three fields; standalone `generate-production-mlp-proof`
(§3) writes all three.

---

## 7. Mainnet-eligibility posture — **dormant**

The production proof family is **not** mainnet-eligible today. This
is by design at every level; there is no near-term path to changing
it without external chain-team + governance action.

- `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` (empty).
  Verified at
  [`crates/omni-zkml/src/proof.rs`](../crates/omni-zkml/src/proof.rs).
- `MAINNET_APPROVED_PROOF_SYSTEMS = &[]` (legacy alias; also empty).
- `check_mainnet_eligible` returns
  `Err(MainnetRefusalReason::NotInMainnetAllowlist { … })` for every
  production artifact — refusal at **layer 6 only** (production
  artifacts pass layers 1 – 5 by their production-shape metadata).
- No `Active` record on any Proof Eligibility Registry — the
  chain-side registry (`sum-chain#21` dormant subprotocol) has no
  `CandidateRefused` record for
  `Stage11dProductionFixedPointMlp` yet either.
- **Register-only v1** architecture (see
  [`docs/stage11.d.3B-proof-eligibility-registry-alignment.md`](stage11.d.3B-proof-eligibility-registry-alignment.md)):
  when eligibility does light up, it happens via a chain-side
  registry record flip — OmniNode's off-chain
  `Halo2ProductionMlpVerifier` remains the cryptographic enforcement
  point.
- **No chain-side SNARK verification** in v1. The chain treats
  proof artifacts opaquely.

**Do not overclaim.** Any operator or integrator running
`generate-production-mlp-proof` today is producing an artifact that
is byte-deterministic and off-chain-verifiable — but **refused on
`chain_id == 1`** by design. It is safe for staging / testnet /
evidence-gathering; it is not safe as a mainnet
attestation-eligibility signal.

The evidence bundle chain-team needs before flipping this posture
lives at
[`docs/stage11.d.3A-production-proof-eligibility-evidence.md`](stage11.d.3A-production-proof-eligibility-evidence.md).

---

## 8. Dependencies and default-build posture

Feature-gated posture, verified at time of writing on `main`:

| Cargo feature | halo2 pulled? | rand_chacha (direct prover)? | `omni-proofs-halo2-production-mlp` pulled? |
| --- | --- | --- | --- |
| default (no features) | ❌ | ❌ | ❌ |
| `submit` | ❌ | ❌ | ❌ |
| `halo2-reference-verify` | ✅ (reference only) | ❌ | ❌ |
| `halo2-reference-prove` | ✅ | ✅ | ❌ |
| `stage11d-production-verify` | ✅ | ❌ | ✅ (verify only) |
| `stage11d-production-prove` | ✅ | ✅ | ✅ (verify + prove) |

Enforced by CI tree-check gates:

- `default tree — must NOT contain sumchain-(crypto|primitives)`
  (indirectly rules out any halo2 dep on default).
- `stage11d-production-verify tree — must contain halo2_proofs + production-mlp`.
- `stage11d-production-prove tree — must contain halo2_proofs + rand_chacha + production-mlp`.

Any PR that leaks halo2 into a default build fails the tree gate.

---

## 9. Cross-references

- Chain-team review packet: [`docs/stage11d.2-review-packet-production-fixedpoint-mlp.md`](stage11d.2-review-packet-production-fixedpoint-mlp.md).
- Chain-team evidence bundle for eligibility: [`docs/stage11.d.3A-production-proof-eligibility-evidence.md`](stage11.d.3A-production-proof-eligibility-evidence.md).
- Registry-terminology alignment: [`docs/stage11.d.3B-proof-eligibility-registry-alignment.md`](stage11.d.3B-proof-eligibility-registry-alignment.md).
- Benchmark record (verify p95 ≈ 19.6 ms; prover ≈ 2.3 s release; `params.bin` 131 KB): [`docs/stage11d.2-benchmark-record.md`](stage11d.2-benchmark-record.md).
- Stage 14.5 operator prover engineering: [`docs/stage14.5-halo2-production-mlp-prove.md`](stage14.5-halo2-production-mlp-prove.md).
- Stage 14.6 contributor sidecar engineering: [`docs/stage14.6-contributor-production-mlp-proof.md`](stage14.6-contributor-production-mlp-proof.md).
- Stage 14.7 cross-family acceptance hardening: [`docs/stage14.7-proof-generation-acceptance-hardening.md`](stage14.7-proof-generation-acceptance-hardening.md).
- Stage 14.8 proof-generation track closure / operator readiness: [`docs/stage14.8-proof-generation-readiness.md`](stage14.8-proof-generation-readiness.md).
- Mainnet-eligibility criteria (canonical spec of what a candidate must satisfy): [`docs/mainnet-eligibility-criteria.md`](mainnet-eligibility-criteria.md).
- Operator runbook (Phase 5 — real proof generation section): [`docs/operator-runbook.md`](operator-runbook.md).
