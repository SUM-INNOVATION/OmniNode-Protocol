# Mainnet Eligibility Criteria (Stage 11d.0)

**Status**: this is the authoritative criteria document for adding a proof system to `MAINNET_APPROVED_PROOF_SYSTEMS`. Adopted by Stage 11d.0; the schema refactor it implies lands in Stage 11d.1.

**Wording guardrail**: Stage 11d **defines and prepares** the eligibility path. A later Stage 11d.3 entry may add the first eligible proof system after written sign-off. Stage 11d itself does not register anything for mainnet eligibility.

---

## 0. Why this document exists

After Stage 11c, the bounded `halo2-mlp-v1 / spec_version: 2` reference circuit is sound for arbitrary i16 inputs. That fact alone does **not** make it mainnet-eligible — a bounded reference circuit (architectural validation fixture for a tiny 4→8→4 MLP) and a production proof system (a real OmniNode inference workload) are different equivalence classes.

This document defines, with no ambiguity, what a proof system must satisfy before its eligibility registry entry may be added. It is consulted by:
- the SUM Chain review board (placeholder reviewer name until a board roster is recorded),
- the OmniNode core team when proposing a new proof class,
- the operator team when explaining why mainnet refusals fire,
- future stage planners considering chain-side verification.

---

## 1. Required properties of an eligible proof system

Numbered for cross-reference from the [Stage 11d.3 review packet](stage11d-review-packet.md).

### 1.1. Soundness

**Claim S1**: the proof scheme has a published soundness analysis with security parameter ≥ 128 bits against a polynomial-time adversary.

**Claim S2**: a written soundness argument specific to the OmniNode circuit (not just to the underlying scheme) has been authored by the proposing engineer and reviewed by **at least one external cryptographer** outside the OmniNode and SUM Chain teams.

**Field**: `chain_team_review_ref.external_cryptographer_signoff` — `TBD` at Stage 11d.0. No 11d.3 PR may proceed until this field is filled with the reviewer's name + date + sign-off-document path.

### 1.2. Completeness

**Claim C1**: for every input in the proof system's declared input domain, an honestly-generated proof verifies.

**Claim C2**: tested by a committed corpus of **≥ 16 inputs** spanning declared-domain boundaries. The corpus must include small / large / zero / known edge-case / known historically-problematic values for the proof system's gadgets.

### 1.3. Determinism

**Claim D1**: proof byte-stability under a pinned RNG seed. The same `(circuit, inputs, witnesses, seed)` must produce byte-identical proofs across hosts running the same library versions.

**Claim D2**: a `verify-only` mode of the proof system's regen tool must succeed against the committed fixture without re-running the prover. Drift detected by this check is a hard release blocker.

### 1.4. Verifier availability

**Claim V1**: a `ProofVerifier` impl is registered in `omni-node operator verify-proof`'s dispatch map.

**Claim V2**: the verifier is reachable either from the default `omni-node` build OR from an explicitly-documented opt-in cargo feature (the Stage 11b.1.b `halo2-reference-verify` precedent).

**Claim V3**: the verifier returns typed `omni_zkml::ProofVerifierError` variants — never panics, never returns a generic `Result<bool, String>`.

**Claim V4 (Stage 11d.0 documentation-only)**: verification on commodity laptop hardware (no GPU) completes in under one second per artifact. Benchmarked by the proof system's regen tool; recorded in the review packet.

### 1.5. Required `ProofArtifactBody` metadata

Beyond what Stage 11b.0 already requires, an artifact whose `proof_system` claims a mainnet-eligible variant **must** carry:

| Field | Stage 11d.0+ requirement |
|---|---|
| `proof_system` | `Some(<eligible-production-variant>)` — never `None`, never `Mock`, never `Stage11bOnnxReference`, never `Stage11bHalo2Reference`. |
| `model_format` | `Some(<non-`Other`>)`. |
| `model_framework` | `Some(<one of Rumus / PyTorch / TensorFlow / Caffe / FrameworkAgnostic>)`, recorded per the chain-team-approved framework binding. |
| `testnet_or_dev_only` | `Some(false)` — explicit. The bare `None` is treated as `testnet_or_dev_only=true` for safety and refuses on mainnet regardless of eligibility registry match. |
| `circuit_id_hex` | **REQUIRED** (not optional). 64-char lowercase hex of the verifier's compiled circuit identity. Must match the eligibility registry entry's `circuit_id_hex` exactly. |
| `verification_key_hex` | **REQUIRED on the artifact side.** Existing schema field — historically ambiguously named (could be raw VK bytes hex, could be a hash). Stage 11d.1 does **not** rename this field on `ProofMetadata`; renaming is a wider schema migration deferred to a future stage. The **eligibility registry** side uses the unambiguously-named `verification_key_hash_hex` (see §1.7 below). Cross-validation between the artifact's `verification_key_hex` and the registered `verification_key_hash_hex` is per-verifier code that lands in **Stage 11d.2** alongside the first production verifier — Stage 11d.1's layer 6 does NOT perform this cross-check. |
| `model_hash` | 64-char lowercase hex; must match the eligibility registry entry's `model_hash` exactly. Pins the specific (model, weights, biases) combination. |
| `input_hash` | BLAKE3 of the canonical input bytes (Stage 11b.0 contract). |
| `response_hash` | BLAKE3 of the canonical output bytes. |
| `public_inputs` | Backend-specific JSON. Mainnet-eligible verifiers must consume this via `verify_artifact`, never via `verify(&[u8], &PublicInputs)` alone — see Stage 11b.1.b `ProofVerifierError::RequiresArtifactDispatch`. |

### 1.6. Distinguishability from bounded reference

**Hard rule H1**: the production proof system MUST use a **different `ProofSystem` enum variant** from any bounded reference (`Stage11bOnnxReference`, `Stage11bHalo2Reference`, and any future bounded reference).

**Hard rule H2**: the production proof system MUST have a `circuit_id_hex` distinct from any committed bounded-reference circuit's hash.

**Hard rule H3**: the production proof system MUST have a `model_hash` distinct from any committed bounded-reference spec's hash (the `EXPECTED_SPEC_HASH` for the halo2 reference, etc.).

**Defense in depth (Stage 11d.1)**:
1. Layer 3 of `check_mainnet_eligible` fires **before** the layer-6 eligibility registry lookup for `Stage11bOnnxReference` and `Stage11bHalo2Reference`, so a bounded reference artifact is refused regardless of eligibility registry contents. Pinned by the `bounded_reference_refused_before_eligibility_registry_lookup` test.
2. The `stage11b_halo2_reference_never_in_eligibility_registry` test iterates **both** `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` and the legacy `MAINNET_APPROVED_PROOF_SYSTEMS` and asserts no entry has `proof_system ∈ {Stage11bOnnxReference, Stage11bHalo2Reference}`. The slice is empty today; the test is a forward-looking guard against future misconfiguration.
3. The `every_allowlist_entry_has_required_metadata` test additionally rejects bounded-reference `proof_system` values, `Mock`, `ModelFormat::Gguf`, and `ModelFormat::Other(_)` in any eligibility registry entry (test name grandfathered; also forward-looking).

### 1.7. VK hash scheme

Stage 11d.1 pins the canonical mainnet-eligibility VK hash as:

```text
verification_key_hash = BLAKE3(MAINNET_VK_HASH_DOMAIN_SEPARATOR || canonical_vk_bytes)
```

where:

- `MAINNET_VK_HASH_DOMAIN_SEPARATOR = b"OMNINODE-VK:v1:"` — 15 ASCII bytes, no null terminator, no length prefix. The trailing `v1` allows a future migration to a new scheme without ambiguity over which hash an eligibility registry entry's `verification_key_hash_hex` was computed under.
- `canonical_vk_bytes` is the **per-verifier** canonical serialization of its `VerifyingKey`. Each production verifier MUST document its `canonical_vk_bytes` scheme (e.g., halo2's `VerifyingKey::write` byte stream, or a backend-specific canonical encoding) and pin a stable test vector for the resulting hash.

Stage 11d.1 ships the helper `omni_zkml::mainnet_vk_hash(canonical_vk_bytes: &[u8]) -> [u8; 32]` plus the `MAINNET_VK_HASH_DOMAIN_SEPARATOR` constant. The concrete `canonical_vk_bytes` extraction lives in each production verifier (Stage 11d.2+).

**Scope of use in Stage 11d.1**:
- `AllowlistEntry.verification_key_hash_hex` records the hex of this hash (entry-side, compile-time pinning).
- The helper is available for per-verifier code to compute and compare.
- Layer 6 of `check_mainnet_eligible` does **NOT** perform cross-validation of an artifact's `metadata.verification_key_hex` against the eligibility registry's `verification_key_hash_hex` — that cross-check requires per-verifier knowledge of `canonical_vk_bytes` extraction and lands in Stage 11d.2 with the first production verifier.

The `vk_hash_helper_is_byte_stable` and `vk_hash_helper_uses_documented_domain_separator` tests pin the formula bit-for-bit.

### 1.8. Deterministic fixture byte-stability

**Hard requirement** (carrying forward the Stage 11b.1.b / 11c regen-tool pattern): a candidate proof system is **not eligible** if it cannot reproduce its committed fixture bytes with pinned RNG seed and pinned dependency versions on a fresh dev host.

Specifically:
- Two consecutive `regen` runs of the proof system's developer-host regen tool must produce byte-identical `params.bin` / `proof.bin` / `proof_artifact.json` files.
- A `verify-only` mode of the regen tool must succeed against the committed fixtures without re-running the prover.
- Drift detected by `verify-only` on a fresh dev host is a **hard release blocker**.
- The Stage 11d.3 review packet R3 (fixture byte-stability proof) records the byte-stability evidence.
- If a backend's prover is non-deterministic by design (e.g., uses true OS randomness), it is not eligible. The fix is either (a) seed all randomness via a pinned RNG, or (b) propose an alternative proof class that admits deterministic byte-stable proving.

---

## 2. Required chain-team review packet artifacts

Numbered for cross-reference from the [Stage 11d.3 review packet template](stage11d-review-packet.md).

| # | Artifact | Format | Sign-off |
|---|---|---|---|
| R1 | Written soundness argument | Structured Markdown with numbered claims / assumptions / invariants. LaTeX may be attached. | External cryptographer (Field 1.1.S2 = `TBD` at 11d.0) |
| R2 | Circuit constraint listing | One section per gadget, with annotated semantics; cross-linked to the source file. | Proposing engineer + one non-author OmniNode engineer |
| R3 | Fixture byte-stability proof | Output of the regen tool's `verify-only` mode; output of two consecutive `regen` runs producing byte-identical files. | Proposing engineer |
| R4 | Independent reproduction attempt | A non-author engineer reproduces the canonical proof on a fresh dev host using only the committed `tools/<proof-system>_regen/` package and documented dependencies. Records resulting file hashes. | Reproducing engineer |
| R5 | License + dependency audit | `cargo tree -p omni-node`, `cargo tree -p omni-node --features <opt-in>`, and `cargo info <each new dep>` outputs. Confirms every new dependency is on crates.io with `MIT OR Apache-2.0` or an equivalent compatible license. | Proposing engineer |
| R6 | Operator-facing failure-mode walkthrough | A list of every typed `ProofVerifierError` / `OperatorError` an operator can see when this proof system fails, with remediation guidance. | Proposing engineer + operator team representative |
| R7 | Test-corpus + adversarial-input results | Committed corpus + the negative-test results (forged proof rejected, wrong circuit_id_hex rejected, wrong model_hash rejected, testnet flag still refused, framework mismatch caught). | Proposing engineer |
| R8 | Mainnet impact statement | Specific claim of what this proof system enables on mainnet (e.g., "verifies inference outputs from model class X with input domain Y"), explicitly bounded so the chain team can evaluate scope. | Proposing engineer + SUM Chain review board |

---

## 3. Allowlist mechanics summary

Stage 11d.1 rewires the layer-6 refusal check to consult a structured eligibility registry:

```rust
pub struct AllowlistEntry {
    pub proof_system: ProofSystem,
    pub backend_id: &'static str,
    pub circuit_id_hex: &'static str,
    pub model_hash: &'static str,
    pub model_format: ModelFormat,
    pub verification_key_hash_hex: &'static str,
    pub chain_team_review_ref: &'static str,
}

pub const MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES: &[AllowlistEntry] = &[];
// Legacy back-compat alias — also empty by design.
pub const MAINNET_APPROVED_PROOF_SYSTEMS: &[ProofSystem] = &[];
```

**Allowlist key**: `(proof_system, circuit_id_hex, model_hash)`. Layer 6 accepts an artifact only if at least one entry matches all three fields exactly. `backend_id`, `model_format`, `verification_key_hash_hex`, and `chain_team_review_ref` are recorded for audit but are not part of the matching key.

**Why a triple key**:
- `proof_system`-alone matching is dangerous: a bounded reference + a future production circuit could share a hypothetical variant and accidentally co-register.
- `(proof_system, circuit_id_hex)` is better but allows the same circuit to be re-used with different weights.
- `(proof_system, circuit_id_hex, model_hash)` pins the exact (model + circuit + scheme) combination the chain team audited.

### 3.1. Where chain-team-approved framework bindings live

The "chain-team-approved framework binding" for an eligibility registry entry is the `(model_framework, model_format)` pair the chain team explicitly reviewed for that specific entry. The binding is recorded in two places:

- **Per-entry review packet** — Stage 11d.3 review-packet §8.1 (mainnet impact statement) lists "Approved framework bindings (chain-team-recorded)" enumerating the `(model_framework, model_format)` combinations covered by the entry. The committed `chain_team_review_ref` document path points at this record.
- **Per-entry eligibility registry record** — the `AllowlistEntry.model_format` field pins the format component. The framework component is **not** in the eligibility registry key (the chain team accepts that two frameworks can produce byte-identical proofs for the same `(proof_system, circuit_id_hex, model_hash)` triple — the four-equal-primaries posture from Stage 11b.1.a/11c carries forward). The framework binding remains a chain-team review concern recorded in the linked document.

There is **no separate `framework_bindings.json` artifact** in Stage 11d.1. If a future stage needs runtime enforcement of framework-binding mismatches (rather than runtime advisory), it requires a separate chain-team plan.

---

## 4. Operational model

- **Local proving primary.** Production proofs must be reproducible on a developer host with documented dependencies, mirroring the Stage 11b.1.b regen-tool pattern.
- **No centralized prover dependency.** A future operator may offload proving to a remote service, but the integration point is the existing `omni-zkml::ProofBackend` trait — multiple implementations coexist behind feature flags. No protocol-level remote-prover requirement is introduced.
- **Verification stays CPU/read-only.** `operator verify-proof` works without GPU acceleration. Verifiers MUST verify under one second per artifact on commodity laptop hardware.
- **Verifier reachability**: default `omni-node` build OR an explicit opt-in cargo feature (the Stage 11b.1.b `halo2-reference-verify` precedent). The default build's compile graph MUST NOT pull a production prover; verifier-only feature builds may pull verifier code.

---

## 5. Chain compatibility

**No Stage 11d substage touches Stage 6 chain wire, Stage 7b tx/signing, or SUM Chain RPCs.** Proof-system metadata stays in SNIP V2 off-chain artifacts. The on-chain `InferenceAttestationDigest` is unchanged.

**Future on-chain verifier path (Stage 12+, out of scope for 11d)**: a chain-side verifier could route by reading `attestation_digest` from the chain, fetching the SNIP V2 proof artifact off-chain, looking up the chosen `proof_system` in a chain-side mirror of `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`, and dispatching to the on-chain verifier contract. None of this is built in Stage 11d. The Stage 11d schema is *intentionally* shaped so a future on-chain verifier is a refactor rather than a schema change.

---

## 6. Hard non-goals

These do not change in Stage 11d:

- No Stage 6 chain-wire changes.
- No Stage 7b tx/signing changes.
- No new SUM Chain RPCs.
- No chain payload changes.
- No tokenomics, staking, slashing, or chain-side verification implementation.
- No GGUF/LLM inference proof claim. (A future GGUF strategy would require its own chain-team plan; no candidate is on the Stage 11d table.)
- Default `omni-node` build pulls zero framework runtimes (RUMUS / PyTorch / TensorFlow / Caffe stay equal-status compatibility targets — none of them lives in the operator's compile graph).
- No edits to immutable audit docs (`docs/mainnet-smoke-audit.md`, `docs/phase5-rc-audit.md`).
- No edits to `canonical_spec.json`, the canonical evaluator, framework manifests, or `EXPECTED_SPEC_HASH` (Stage 11c source-of-truth invariant carries forward).
- **No promotion of `Stage11bHalo2Reference` to production.** It stays a testnet/dev-only architectural-validation fixture in perpetuity.
- **No eligibility registry entry lands in Stage 11d.0, 11d.1, or 11d.2.** Only Stage 11d.3 — and only with written chain-team sign-off — modifies `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` to a non-empty list.
- **`ModelFormat::Gguf` is invalid as a Stage 11d.3 eligibility registry entry's `model_format`** unless a separately-reviewed GGUF strategy exists. Pinned by the `every_allowlist_entry_has_required_metadata` test (name grandfathered).
- **`ModelFormat::Other(_)` is invalid as a Stage 11d.3 eligibility registry entry's `model_format`.** The `Other(_)` escape hatch is for forward compatibility with formats that have not been chain-team-reviewed; if a real format needs to land, promote it to a first-class `ModelFormat` enum variant via a separate plan. Pinned by the same test.
- **`ProofSystem::Stage11bOnnxReference` and `ProofSystem::Stage11bHalo2Reference` are invalid as Stage 11d.3 eligibility registry entries' `proof_system`** — bounded references are testnet/dev-only by hard rule H1. Pinned by `stage11b_halo2_reference_never_in_eligibility_registry` and `every_allowlist_entry_has_required_metadata`.

---

## 7. Open fields (to be filled before downstream stages may proceed)

| Field | Required by | Default at Stage 11d.0 | Filled in |
|---|---|---|---|
| SUM Chain review board roster | 11d.3 sign-off | placeholder `"SUM Chain review board"` | TBD; record names in this document when chosen |
| External cryptographer for soundness review (Claim 1.1.S2) | 11d.3 sign-off | `TBD` | Must be filled before 11d.3 PR opens |
| First production proof class | 11d.2 implementation | open; small fixed-point production MLP is a **recommended candidate**, not a decision | Chain-team review during 11d.0/11d.1 cycle |
| External cryptographer sign-off path | 11d.3 sign-off | `TBD` (no document yet) | Created by the cryptographer; referenced by the AllowlistEntry's `chain_team_review_ref` |
| CODEOWNERS pattern for `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` | 11d.1 enforcement | not configured (repo has no CODEOWNERS file yet); plan documented here | Add when CODEOWNERS file is introduced for any other purpose, or as part of 11d.3 |
| Production proof class soundness-argument doc | 11d.2 implementation | not authored | Authored during 11d.2 review packet |

---

## 8. Sequencing

| Substage | Status (at the time of writing) | Allowlist state |
|---|---|---|
| **11d.0** (this document + review packet template + FAQ) | **Active PR** | `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` (does not yet exist) |
| **11d.1** (eligibility registry schema refactor, no entries) | After 11d.0 merge + chain-team review | `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` (new schema, still empty) |
| **11d.2** (first production proof class — no eligibility registry entry yet) | After 11d.1 merge + chain-team selection of production class | `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES = &[]` (new ProofSystem variant exists, but not in the eligibility registry) |
| **11d.3** (first eligibility registry entry) | **Blocked on** external-cryptographer sign-off (Claim 1.1.S2), chain-team review packet completion (R1–R8), and CODEOWNERS-style PR controls | `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` adds exactly one entry |

If any required field above is `TBD` at the time of a downstream PR, the downstream PR does not land. Stage 11d follows a **halt-and-report posture**: if mid-implementation the production proof class exposes a soundness or spec gap, the implementing engineer stops and surfaces the issue to the SUM Chain review board rather than silently patching.

---

## 9. Cross-references

- [Stage 11d Review Packet template](stage11d-review-packet.md) — the structured artifact a 11d.3 PR fills in.
- [Stage 11d Mainnet Eligibility FAQ](stage11d-mainnet-eligibility-FAQ.md) — operator-facing Q&A.
- `crates/omni-zkml/src/proof.rs` — the six-layer `check_mainnet_eligible` refusal helper; layer 6 is the one Stage 11d.1 rewires.
- [Operator Runbook §11a](operator-runbook.md) — typed-error taxonomy operators see; `ProofSystemNotMainnetApproved` is the user-visible signal of a layer-6 refusal.
