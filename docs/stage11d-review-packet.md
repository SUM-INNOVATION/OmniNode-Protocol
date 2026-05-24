# Stage 11d Review Packet (Template)

**Status**: template introduced by Stage 11d.0. To be populated by the proposing engineer when a candidate production proof class is selected (Stage 11d.2) and submitted for chain-team review prior to Stage 11d.3.

**This template is the deliverable handed to the SUM Chain review board.** It cross-references the eight criteria sections in [`mainnet-eligibility-criteria.md`](mainnet-eligibility-criteria.md). Sections marked `TBD` cannot be left as `TBD` when the 11d.3 PR opens; the criteria document's §7 enumerates which fields block which substage.

**Wording guardrail**: Stage 11d **defines and prepares** the eligibility path. A later Stage 11d.3 entry may add the first eligible proof system after written sign-off. This packet does not itself allowlist anything; it is the audit input that precedes the allowlist PR.

---

## Packet metadata

| Field | Value |
|---|---|
| Packet author | `TBD` (proposing engineer) |
| Packet version | `0.0.0` (incremented per revision; major bump on substantive content change) |
| Proposed proof system (ProofSystem enum variant) | `TBD` (MUST NOT be `Mock`, `Stage11bOnnxReference`, or `Stage11bHalo2Reference` — bounded references are testnet/dev-only) |
| Proposed `backend_id` | `TBD` |
| Proposed `circuit_id_hex` | `TBD` (64-char lowercase hex) |
| Proposed `model_hash` | `TBD` (64-char lowercase hex) |
| Proposed `verification_key_hash_hex` | `TBD` (64-char lowercase hex; computed via `mainnet_vk_hash(canonical_vk_bytes)` per criteria §1.7) |
| Proposed `model_format` | `TBD` (one of `Onnx` / `Halo2ReferenceMlp` / future first-class variants. **`Gguf` is invalid** unless a separately-reviewed GGUF strategy exists. **`Other(_)` is invalid**.) |
| Proposed `model_framework` | `TBD` (one of Rumus / PyTorch / TensorFlow / Caffe / FrameworkAgnostic) |
| `review_board_roster_ref` | `TBD` (path to a committed roster doc or external sign-off record naming the SUM Chain review board members who signed off; field added per Stage 11d.1 review tightening) |
| `benchmark_record_ref` | `TBD` (path to the committed benchmark record document; see §3.5 below for required fields) |
| Target Stage 11d.2 implementation PR | `TBD` |
| Target Stage 11d.3 allowlist PR | `TBD` |
| Date submitted to SUM Chain review board | `TBD` |
| Date of external cryptographer sign-off | `TBD` |
| Chain-team approval status | `Not yet submitted` |

---

## 1. Soundness (criteria 1.1 — Claims S1 + S2)

### 1.1.1. Underlying scheme

- Scheme name: `TBD`
- Published soundness analysis: `TBD` (citation)
- Security parameter (claimed): `TBD` bits
- Adversary model: `TBD`

### 1.1.2. OmniNode-specific soundness argument

Structured Markdown, numbered claims / assumptions / invariants. LaTeX may be attached separately if useful for derivations.

- **Assumption A1**: `TBD`
- **Assumption A2**: `TBD`
- **Invariant I1**: `TBD` (e.g., "for every public input, exactly one consistent witness assignment exists")
- **Claim C1**: `TBD` (the production statement the proof certifies)
- **Proof sketch for C1**: `TBD`

### 1.1.3. External cryptographer sign-off

- External cryptographer name: `TBD`
- Affiliation: `TBD`
- Sign-off document path: `TBD`
- Sign-off date: `TBD`

**Hard block**: this sub-section may not be left `TBD` when the 11d.3 PR opens. The chain team will reject any 11d.3 PR whose `chain_team_review_ref` doesn't resolve to a real sign-off document.

---

## 2. Circuit constraint listing (criteria R2)

A section per gadget, with annotated semantics, cross-linked to source.

### 2.1. Gadget inventory

| Gadget | Source path | Constraint count | Notes |
|---|---|---|---|
| `TBD` | `crates/omni-proofs-.../src/...` | `TBD` | `TBD` |

### 2.2. Per-gadget detail

#### 2.x.1 `<gadget name>`

- Witnesses: `TBD`
- Constraints (numbered): `TBD`
- Soundness contribution: `TBD`
- Range checks: `TBD` (widths + decomposition scheme)

(Repeat per gadget.)

### 2.3. Cross-gadget composition

How the gadgets compose into the production statement. Reference the §1.1.2 invariants that emerge from composition.

---

## 3. Fixture byte-stability proof (criteria R3)

### 3.1. Regen `verify-only` output

```
$ cd tools/<proof-system>_regen
$ cargo run --release verify-only
<paste output>
```

### 3.2. Two consecutive `regen` runs

```
$ cargo run --release regen   # run 1
$ shasum -a 256 fixtures/...
<paste hashes>

$ cargo run --release regen   # run 2
$ shasum -a 256 fixtures/...
<paste hashes>  # must match run 1
```

### 3.3. Pinned dependency versions

| Dep | Version | License |
|---|---|---|
| `halo2_proofs` | `TBD` | `TBD` |
| `pasta_curves` | `TBD` | `TBD` |
| `TBD` | `TBD` | `TBD` |

### 3.4. Hard byte-stability requirement (criteria §1.8)

**This proof system is not eligible** if any of the following fails:

- [ ] Two consecutive `regen` runs of the developer-host regen tool produce byte-identical `params.bin` / `proof.bin` / `proof_artifact.json` files on the same host.
- [ ] A `verify-only` mode of the regen tool succeeds against the committed fixtures without re-running the prover.
- [ ] A fresh dev host with pinned dependency versions reproduces the committed fixture bytes (criteria §1.8).

If the prover requires true OS randomness, the proof system is **not eligible**. Either seed all randomness via a pinned RNG (Stage 11b.1.b / 11c precedent: `rand_chacha::ChaCha20Rng::from_seed(PINNED_SEED)`) or propose an alternative proof class that admits deterministic byte-stable proving.

### 3.5. Benchmark record (criteria §1.4 V4)

Required structured record. Path recorded in packet metadata as `benchmark_record_ref`.

| Field | Value |
|---|---|
| Exact invocation command | `TBD` (e.g., `cargo bench -p <crate> --features verify -- --bench verify_artifact`) |
| Build profile | `release` (NOT `dev`) |
| Run count | `TBD` (≥ 100 iterations required) |
| Hardware: CPU | `TBD` (e.g., `Apple M3 Max`, `Intel Xeon E5-2680 v4`); include core count |
| Hardware: RAM | `TBD` (e.g., `64 GB`) |
| Hardware: OS + version | `TBD` (e.g., `Darwin 24.6.0`, `Ubuntu 22.04.4 LTS`) |
| Rust toolchain version | `TBD` (`rustc --version`) |
| `cargo --version` output | `TBD` |
| Reported metric: p50 (ms) | `TBD` |
| Reported metric: p95 (ms) | `TBD` |
| Reported metric: max (ms) | `TBD` |
| Threshold met? | `TBD` (p95 ≤ 1000 ms per criteria §1.4 V4) |

If p95 > 1000 ms on commodity laptop hardware, the proof system is **not eligible** in its current form. Either reduce circuit cost, increase the threshold via a separately-reviewed chain-team plan, or propose an alternative proof class.

---

## 4. Independent reproduction attempt (criteria R4)

| Field | Value |
|---|---|
| Reproducing engineer | `TBD` (must be non-author) |
| Dev-host platform | `TBD` (OS + arch + rust version) |
| Reproduction date | `TBD` |
| Steps followed | only `tools/<proof-system>_regen/README.md` (no out-of-band help) |
| Resulting `params.bin` BLAKE3 | `TBD` |
| Resulting `proof.bin` BLAKE3 | `TBD` |
| Match against committed fixture | `TBD` (PASS / FAIL) |

If FAIL, the proposing engineer halts the 11d.3 PR and surfaces the byte-stability discrepancy to the SUM Chain review board.

---

## 5. License + dependency audit (criteria R5)

### 5.1. `cargo tree -p omni-node` (default build)

```
$ cargo tree -p omni-node
<paste output>
$ cargo tree -p omni-node | grep -iE 'halo2|pasta|rumus|torch|tensorflow|caffe|<production-deps>'
<paste output — should not include the production proof crate>
```

### 5.2. `cargo tree -p omni-node --features <opt-in>`

```
$ cargo tree -p omni-node --features <opt-in>
<paste output>
$ cargo tree -p omni-node --features <opt-in> | grep -iE 'halo2|pasta|<production-deps>'
<paste output — should include the opt-in's transitive deps>
```

### 5.3. Per-dependency license attestation

| Dep | Version | License source | License text or URL |
|---|---|---|---|
| `TBD` | `TBD` | crates.io | `TBD` |

Every dependency MUST be on crates.io with `MIT OR Apache-2.0` (or equivalent compatible license). Git dependencies are not permitted (Stage 9c rule, carries forward).

---

## 6. Operator-facing failure-mode walkthrough (criteria R6)

For every typed error an operator can see when this proof system fails verification — **including every `MainnetRefusalReason` layer**:

| Typed error | When it fires | Operator remediation | Escalation path |
|---|---|---|---|
| `ProofVerifierError::VerifierInternal` | `TBD` | `TBD` | `TBD` |
| `ProofVerifierError::RequiresArtifactDispatch` | `TBD` | `TBD` | `TBD` |
| `OperatorError::ProofArtifactRead` | `TBD` | `TBD` | `TBD` |
| `OperatorError::ProofArtifactParse` | `TBD` | `TBD` | `TBD` |
| `OperatorError::NoVerifierForProofSystem` | `TBD` | `TBD` | `TBD` |
| `MainnetRefusalReason::TestnetOrDevOnly` (layer 1) | `TBD` (typically: artifact carries `testnet_or_dev_only: Some(true)`; UNREACHABLE for a correctly-produced production artifact for this proof system) | `TBD` | `TBD` |
| `MainnetRefusalReason::MockBackend` (layer 2) | `TBD` (UNREACHABLE for this proof system — `proof_system` is not `Mock`) | `TBD` | `TBD` |
| `MainnetRefusalReason::BoundedReference` (layer 3) | `TBD` (UNREACHABLE for this proof system — `proof_system` is not a bounded reference variant) | `TBD` | `TBD` |
| `MainnetRefusalReason::GgufClaim` (layer 4) | `TBD` (UNREACHABLE if `model_format ≠ Gguf`; otherwise: not eligible by criteria) | `TBD` | `TBD` |
| `MainnetRefusalReason::UnknownModelFormat` (layer 5) | `TBD` (UNREACHABLE if `model_format` is a first-class non-Gguf variant; otherwise: not eligible) | `TBD` | `TBD` |
| `MainnetRefusalReason::NotInMainnetAllowlist` (layer 6) | `TBD` (artifact's `(proof_system, circuit_id_hex, model_hash)` triple does not match this entry; cause is metadata drift) | `TBD` | `TBD` |

Each row's "When it fires" entry must EITHER describe the concrete scenario for this proof system OR state explicitly why the layer is unreachable for this proof system. "TBD" is not an acceptable final value for any row.

Reviewed by an operator team representative; sign-off recorded in §11.

---

## 7. Test-corpus + adversarial-input results (criteria R7)

### 7.1. Positive corpus

Committed at `crates/omni-proofs-.../tests/fixtures/<production>_corpus.json`. Minimum 16 entries spanning declared-domain boundaries. Each entry round-trips through `prove → verify_artifact → Ok(true)`.

### 7.2. Negative tests

| Negative case | Expected outcome | Test path |
|---|---|---|
| Forged proof bytes (bit-flipped) | `verify_artifact → Err(VerifierInternal)` OR `Ok(false)` | `TBD` |
| Wrong `circuit_id_hex` in metadata (mainnet path) | `check_mainnet_eligible → Err(NotInMainnetAllowlist)` (layer 6) | `TBD` |
| Wrong `model_hash` in metadata (mainnet path) | `check_mainnet_eligible → Err(NotInMainnetAllowlist)` (layer 6) | `TBD` |
| Wrong `circuit_id_hex` in metadata (verifier path) | `verify_artifact → Err(VerifierInternal)` mentioning circuit_id | `TBD` |
| Wrong `model_hash` in metadata (verifier path) | `verify_artifact → Err(VerifierInternal)` mentioning model_hash | `TBD` |
| `testnet_or_dev_only: Some(true)` on otherwise-allowlisted artifact | `check_mainnet_eligible → Err(TestnetOrDevOnly)` (layer 1) | `TBD` |
| Framework mismatch (e.g., RUMUS claim with PyTorch metadata) | `verify_artifact → Err(VerifierInternal)` mentioning framework | `TBD` |
| Bounded-reference artifact (`Stage11bHalo2Reference`) with metadata copied from allowlist | `check_mainnet_eligible → Err(BoundedReference)` (layer 3) | `TBD` |
| `model_format = Gguf` on otherwise-allowlisted metadata | `check_mainnet_eligible → Err(GgufClaim)` (layer 4) | `TBD` |
| `model_format = Other("…")` on otherwise-allowlisted metadata | `check_mainnet_eligible → Err(UnknownModelFormat)` (layer 5) | `TBD` |
| Production proof bytes against bounded-reference verifier | `verify_artifact → Err(VerifierInternal)` (vk mismatch) | `TBD` |

---

## 8. Mainnet impact statement (criteria R8)

### 8.1. What this proof system enables on mainnet

`TBD` — a specific, bounded claim of the inference workload(s) this proof system can certify. Example skeleton:

> "Verifies inference outputs from model class X, for inputs in domain Y, with input dimension ≤ N, against the model identified by `model_hash = ...`. Does not verify any other model, any other input domain, or any inference outside the committed circuit."

**Approved framework bindings (chain-team-recorded)** — per criteria §3.1, enumerate the `(model_framework, model_format)` combinations the chain team explicitly reviewed for this entry:

| Framework | Format | Reviewed? |
|---|---|---|
| `TBD` (e.g., FrameworkAgnostic) | `TBD` (e.g., Halo2ReferenceMlp) | `TBD` |
| `TBD` | `TBD` | `TBD` |

The framework component is not part of the allowlist matching key (the four-equal-primaries posture from Stage 11b.1.a allows multiple frameworks to produce byte-identical proofs for the same triple), but the chain team records the reviewed bindings here for audit.

### 8.2. What this proof system explicitly does NOT cover

`TBD` — every model class / framework / inference domain that an operator might incorrectly believe is covered. Examples:

- Does not cover GGUF inference (no GGUF proof strategy exists in this packet).
- Does not cover model variants with different weights / biases (model_hash pins one specific binding).
- Does not cover production LLM inference workloads (out of scope for the first production proof class).

### 8.3. Reversibility

Describe how the entry can be removed from the allowlist if a soundness regression is later discovered. The expected path: a PR that removes the entry from `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` and bumps the snapshot fixture. Operator-facing communication path: `TBD`.

---

## 9. Open questions for the SUM Chain review board

(Populated by the proposing engineer; the review board may add more.)

- `TBD`

---

## 10. Halt-and-report log

If at any point during preparation the proposing engineer discovers a soundness gap, a spec mismatch, or any other issue that would require silently patching the canonical spec / evaluator to make the proof system valid:

- **Stop.**
- Append an entry to this log with date + finding + the spec/evaluator behaviour the proof system would have required.
- Surface to the SUM Chain review board.
- Do not modify `canonical_spec.json`, `EXPECTED_SPEC_HASH`, framework manifests, or the canonical evaluator without explicit chain-team approval.

| Date | Finding | Resolution |
|---|---|---|
| `TBD` | `TBD` | `TBD` |

---

## 11. Sign-off

| Role | Name | Date | Status |
|---|---|---|---|
| Proposing engineer | `TBD` | `TBD` | Not yet signed |
| External cryptographer (Claim 1.1.S2) | `TBD` | `TBD` | Not yet signed |
| Non-author OmniNode engineer (R2 review) | `TBD` | `TBD` | Not yet signed |
| Independent reproducer (R4) | `TBD` | `TBD` | Not yet signed |
| Operator team representative (R6 review) | `TBD` | `TBD` | Not yet signed |
| SUM Chain review board | placeholder until roster recorded | `TBD` | Not yet signed |
| Review board roster reference | `TBD` (path; matches packet-metadata `review_board_roster_ref`) | — | — |

**No 11d.3 PR may open until every sign-off above is recorded and the chain-team approval status is `Approved`.**
