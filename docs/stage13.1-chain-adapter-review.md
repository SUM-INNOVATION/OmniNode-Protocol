# Stage 13.1 — Chain-Team Review Packet (Integrity-Evidence Anchor)

**Status:** review packet only. **Zero adapter code.** Stage 13.0 ships the wire (frozen); Stage 13.1 hands the chain team the spec + deterministic byte fixtures they need to confirm before a real `omni-sumchain` adapter can be built. The follow-on adapter stage is gated on the **per-blocker decision form** (Section 7) being returned with all five rows filled.

**Co-references:**
- [`docs/stage13-evidence-anchor-spec.md`](stage13-evidence-anchor-spec.md) — frozen Stage 13.0 wire spec.
- [`docs/operator-runbook.md`](operator-runbook.md) §Stage 13.0 — current operator workflow (stub client only).
- [`crates/omni-zkml/tests/evidence_anchor_wire_vectors.rs`](../crates/omni-zkml/tests/evidence_anchor_wire_vectors.rs) — vector regeneration + verify mode.
- [`crates/omni-zkml/tests/fixtures/evidence_anchor_wire_vectors.json`](../crates/omni-zkml/tests/fixtures/evidence_anchor_wire_vectors.json) — committed deterministic fixtures (Section 6 references the inputs/outputs).

## Scope (what Stage 13.1 covers, and does not cover)

**In scope (Stage 13.1):**

- The five blocker definitions and decision form (Sections 1-5, 7).
- Three deterministic wire vectors the chain team can rebuild and assert byte-parity against (Section 6).
- A forward-link from the frozen Stage 13.0 wire spec to this review packet.

**Out of scope (Stage 13.1):**

- ❌ No `omni-sumchain` changes. The Stage 13.0 stub remains the only `EvidenceAnchorChainClient` impl until chain-team review concludes.
- ❌ No new `omni-zkml::evidence_anchor` library functions.
- ❌ No CLI changes (no `--rpc-url` / `--expect-chain-id` / `--allow-submit` flags on any anchor subcommand).
- ❌ No new reason-tag entries (the future adapter stage will reserve `mainnet_policy_unresolved` per below).
- ❌ No Stage 12 or Stage 13.0 wire / domain / schema / reason-tag / same-key-submitter changes.

## Reframed semantics (for the future adapter stage)

These are commitments Stage 13.1 makes to the chain team and to a future implementer. They are not code today.

- **Submit:** `submit-integrity-evidence-anchor` is a **chain-write** path, gated behind `--features submit`, with operator double-gates (`--allow-submit`, `--allow-mainnet-submit`).
- **Query and reconcile:** **chain-read-only, local-registry-mutating.** Both flows make only read-only chain calls (`sum_getIntegrityEvidenceAnchorStatus`), never send a tx, and mutate **only** the local `--anchor-registry-dir` records to apply chain-returned status transitions. This mirrors the Stage 5 [`query_attestation_workflow`](../crates/omni-zkml/src/registry.rs) contract verbatim: chain-read-only, local-registry-mutating, observation-only on chain `Unknown`.
- **Verify:** `verify-integrity-evidence-anchor` and `verify-integrity-evidence-anchor-file` stay **chain-untouched, local-read-only.** Verifying local bytes against a recorded anchor under the same-key binding does not require chain interaction in any stage.
- **Mainnet policy:** unresolved. Until chain team confirms (Blocker D), the future adapter refuses mainnet submission unconditionally with the reserved closed-set reason tag `mainnet_policy_unresolved` — regardless of `--allow-mainnet-submit`. No mainnet-permitted code path lands until the policy is explicit.
- **Minimal RPC dependency:** only `sum_getIntegrityEvidenceAnchorStatus(tx_hash)` is required by the future adapter. Stage 13.0's `tx_index.json` already maps `tx_id → artifact_hash_hex` locally, so reconcile drives by `tx_id`. A `sum_getIntegrityEvidenceAnchorByHash(artifact_hash_hex)` lookup is **explicitly deferred** as a future chain feature and is **not** part of Stage 13.x implementation dependency.

## Section 1 — `TxPayload` variant proposal (Blocker A)

Propose: append a new variant to [`sumchain-primitives`'s `TxPayload`](https://crates.io/crates/sumchain-primitives) enum after the current `Education` variant.

Proposed Rust:

```rust
// crates/primitives/src/transaction.rs (chain side)
pub enum TxPayload {
    // … existing variants verbatim, append-only …
    /// Education-LMS suite (SRC-817/818) — last existing.
    Education(crate::education::EducationTxData),

    /// OmniNode Stage 13.0 integrity-evidence anchor handoff.
    /// Wire payload byte-equal to `omni_zkml::IntegrityEvidenceAnchorTxData`
    /// frozen at `INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION = 1`
    /// under `EVIDENCE_ANCHOR_DOMAIN = "OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:"`.
    /// **Append-only**: never reorder above this. See
    /// `crates/primitives/src/integrity_evidence_anchor.rs` and the wire
    /// fixtures in OmniNode's `crates/omni-zkml/tests/fixtures/evidence_anchor_wire_vectors.json`.
    IntegrityEvidenceAnchor(crate::integrity_evidence_anchor::IntegrityEvidenceAnchorTxData),
}
```

Inner module shape (proposed):

```rust
// crates/primitives/src/integrity_evidence_anchor.rs (chain side)
pub const ANCHOR_DOMAIN: &[u8] = b"OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchoredArtifactKind {
    SignedIntegrityEvidenceChainReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntegrityEvidenceAnchorDigest {
    pub anchor_schema_version: u32,           // = 1
    pub artifact_kind: AnchoredArtifactKind,  // closed enum, one variant
    pub artifact_schema_version: u32,
    pub artifact_hash: [u8; 32],              // BLAKE3 over raw on-disk wrapper bytes
    pub signer_pubkey: [u8; 32],              // Ed25519 pubkey (same-key submitter rule)
    pub signed_at_utc_unix: i64,
}

pub struct IntegrityEvidenceAnchorTxData {
    pub digest: IntegrityEvidenceAnchorDigest,
    // 64-byte Ed25519 over ANCHOR_DOMAIN || bincode-1(digest),
    // verified under digest.signer_pubkey (same-key submitter rule).
    pub submitter_signature: [u8; 64],
}
```

Variant discriminant: chain-team to assign. `TxType` discriminant on the chain side currently runs 0..22 (`Transfer` through `Education`); we expect `IntegrityEvidenceAnchor` to land at discriminant **23**, bincode-1 variant ordinal **24** (zero-indexed → 23). Confirm in the decision form.

**Asks chain team to confirm:**
- A1: variant placement — append-only after `Education`, no reordering above.
- A2: variant discriminant — proposed `23`. Confirm or revise.
- A3: bincode-1 byte parity — chain-side `IntegrityEvidenceAnchorDigest` bincode-1 layout matches OmniNode's frozen 84-byte layout (Section 6 fixtures pin the bytes).
- A4: `AnchoredArtifactKind` closed enum, one variant for v1, new variants require `anchor_schema_version` bump on the OmniNode side. Confirm chain side will accept the closed-enum posture.

## Section 2 — Status RPC shape (Blocker B, minimal-only)

**Required for Stage 13.x implementation:**

```text
JSON-RPC method: sum_getIntegrityEvidenceAnchorStatus
Params:          [tx_hash: String]    // hex-encoded chain tx hash, with or without "0x"
Result shape:    mirror sum_getInferenceAttestationStatus
                 — chain's existing five-state envelope:
                 { "status": "Submitted" | "Included" | "Finalized" | "Unknown",
                   "block_height": Option<u64>,
                   "tx_hash": String,
                   ... }
                 plus the standard "Failed" form:
                 { "status": "Failed", "reason": String, ... }
```

OmniNode-side mapper will translate to the existing closed `omni_zkml::AnchorStatus` enum (which mirrors `omni_zkml::AttestationStatus` verbatim). No new local status types.

**Asks chain team to confirm:**
- B1: method name — `sum_getIntegrityEvidenceAnchorStatus`. Confirm or revise.
- B2: result envelope shape — mirror `sum_getInferenceAttestationStatus`. Confirm.
- B3: tx-hash param form — confirm `0x`-prefix is accepted (Stage 13.0 stores the chain's `0x`-prefixed hash verbatim in `tx_index.json`).

**Explicitly deferred (not required for Stage 13.x):**

- B4: `sum_getIntegrityEvidenceAnchorByHash(artifact_hash_hex)` lookup. Stage 13.0's local `tx_index.json` carries the `tx_id → artifact_hash_hex` mapping, so reconcile drives by `tx_id` without needing a by-hash chain RPC. If the chain team wants to index anchor by-hash for future relay flows / aggregator UIs, that is a separate chain deliverable; OmniNode does not block on it.

## Section 3 — Activation gate (Blocker C)

Stage 7b's `submit_attestation` gates behind two activation flags fetched once via `sum_getChainParams`:

- `omninode_is_active()` — OmniNode subprotocol live on chain.
- `v2_is_active()` — V2 envelope accepted.

Stage 13.x's `submit_anchor` would inherit both. The open question for chain team:

- **Option C-inherit:** anchor submissions are gated by the existing two flags only. OmniNode pre-flight runs both, refuses on either inactive.
- **Option C-new-flag:** a third chain-side flag `evidence_anchor_is_active()` (mirroring `omninode_is_active`) gates `TxPayload::IntegrityEvidenceAnchor` specifically. OmniNode adds a third pre-flight call.

No OmniNode preference. The chain team's call — record in the decision form.

## Section 4 — Mainnet policy (Blocker D, unresolved)

Stage 11b's mainnet allowlists (`MAINNET_APPROVED_PROOF_SYSTEMS`, etc.) are empty by design; chain-team approval is required for any mainnet write path. Stage 13.x anchors commit only `(artifact_hash, signer_pubkey, signed_at_unix)` — **no** proof-system semantics, **no** model claims, **no** allowlist surface. Forensic-record bytes only.

**Stage 13.1 makes no assumption.** Until the chain team decides, the future adapter stage:

- Refuses mainnet submission unconditionally (closed reason tag `mainnet_policy_unresolved` — reserved by Stage 13.1 for the future adapter; not yet wired into code).
- Operator double gate (`--allow-mainnet-submit`) does NOT override this until policy is explicit.
- The reason tag is intentionally explicit and explains the actual blocker without implying a permanent refusal.

**Asks chain team to decide:**
- D1: are anchor submissions permitted on mainnet (chain_id 1) once Blockers A-C are resolved? Yes / No / Yes-with-conditions.
- D2: if yes-with-conditions, list the conditions (allowlist? per-signer authorization? rate limit?). OmniNode will encode them as separate closed reason tags in the future adapter stage.

## Section 5 — Fee schedule (Blocker E)

Stage 7b reads `params.min_fee` from `sum_getChainParams` and uses it verbatim as the tx fee (`fee = params.min_fee as u128`, widening `u64` → `Balance`). Anchor tx body is comparable in size to `InferenceAttestation` (84-byte digest + 64-byte signature + outer envelope), so proposing the same posture:

- Use `params.min_fee` for anchor txs. No separate schedule, no anchor-specific gas estimate.

**Asks chain team to confirm:**
- E1: `params.min_fee` applies, or specify an anchor-specific schedule.

## Section 6 — Wire fixtures

Three deterministic vectors live at [`crates/omni-zkml/tests/fixtures/evidence_anchor_wire_vectors.json`](../crates/omni-zkml/tests/fixtures/evidence_anchor_wire_vectors.json). The chain team is expected to:

1. Read each vector's inputs (`artifact_hash_hex`, `signed_at_utc_unix`, `submitter_seed_hex`).
2. Derive `signer_pubkey` from `submitter_seed_hex` via Ed25519 (same-key submitter rule). Compare against the recorded `signer_pubkey_hex`.
3. Build the chain-side `IntegrityEvidenceAnchorDigest` with the same field values.
4. Serialize via **bincode 1.3** and compare against the recorded `canonical_digest_bytes_hex` (84 bytes total).
5. Prepend `ANCHOR_DOMAIN` ASCII to canonical bytes and compare against `signing_input_bytes_hex` (122 bytes total).
6. Sign with Ed25519 over `signing_input_bytes` using the seed; compare against `submitter_signature_hex` (64 bytes).
7. Verify the signature under `signer_pubkey`.

A failure on any of these steps is a wire-incompatibility report back to OmniNode. Successful round-trip on all three vectors is the byte-parity contract.

### Vector 1 — `stage13.1-vec-1-normal` (realistic submit shape)

```text
artifact_schema_version : 1
artifact_hash_hex       : 1739d0e5e3e3b2bc63b67ef58ca7a99a4b50a5f3cf08af3c5e9d5d5d28d1c4b3
submitter_seed_hex      : 0101010101010101010101010101010101010101010101010101010101010101
signed_at_utc_unix      : 1750000000
signer_pubkey_hex       : 8a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d94121bf3748801b40f6f5c
canonical_digest_bytes  : 0100000000000000010000001739d0e5e3e3b2bc63b67ef58ca7a99a4b50a5f3c
                          f08af3c5e9d5d5d28d1c4b38a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d
                          94121bf3748801b40f6f5c80e14e6800000000
signing_input_bytes     : 4f4d4e494e4f44452d494e544547524954592d45564944454e43452d414e4348
                          4f523a76313a (= ANCHOR_DOMAIN ASCII)
                          || canonical_digest_bytes
submitter_signature_hex : 377995da1ec1c151f392658dd371745f0dadfd5b2990fc8ecd975439adcc1564
                          ecf2af6140102db918ff4d640a3d568b1b113235fb07ae90f52fed86d9596c01
```

### Vector 2 — `stage13.1-vec-2-minimal` (all-zero / epoch)

```text
artifact_schema_version : 1
artifact_hash_hex       : 0000000000000000000000000000000000000000000000000000000000000000
submitter_seed_hex      : 0000000000000000000000000000000000000000000000000000000000000000
signed_at_utc_unix      : 0
signer_pubkey_hex       : 3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29
canonical_digest_bytes  : 0100000000000000010000000000000000000000000000000000000000000000
                          00000000000000000000000000003b6a27bcceb6a42d62a3a8d02a6f0d736532
                          15771de243a63ac048a18b59da290000000000000000
submitter_signature_hex : 4da58709b4d8efc15a74aa46a84baace798b447e23f8579339ce2cfcc4f71ffe
                          d3a36d6beb6585b02241922150b05501aff00b1114a9e4daf2b35df6a4c4b10b
```

### Vector 3 — `stage13.1-vec-3-high-entropy` (`0xFF...` artifact + seed, 2100-01-01)

```text
artifact_schema_version : 1
artifact_hash_hex       : ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
submitter_seed_hex      : ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
signed_at_utc_unix      : 4102444800   (= 2100-01-01T00:00:00Z)
signer_pubkey_hex       : 76a1592044a6e4f511265bca73a604d90b0529d1df602be30a19a9257660d1f5
canonical_digest_bytes  : 010000000000000001000000ffffffffffffffffffffffffffffffffffffffff
                          ffffffffffffffffffffffff76a1592044a6e4f511265bca73a604d90b0529d1
                          df602be30a19a9257660d1f5005786f400000000
submitter_signature_hex : c0385b44dd6ef33c7af87cbdfa69bf2683aac2473183af8b0413fabcae4d1914
                          bc1bb4cf1ac3c4f75a58a6df5e56bb690602de8005becc791e9ba35ddc1eb20a
```

### Bincode-1.3 layout reference

The 84-byte canonical bytes break down as:

```text
offset  size  field                              encoding
------  ----  ---------------------------------  ----------------------
0       4     anchor_schema_version (u32)        little-endian
4       4     artifact_kind discriminant (u32)   little-endian (bincode-1 enum tag, = 0 for the single variant)
8       4     artifact_schema_version (u32)      little-endian
12      32    artifact_hash                      raw, no length prefix
44      32    signer_pubkey                      raw, no length prefix
76      8     signed_at_utc_unix (i64)           little-endian
------  ----
        84    total
```

The signing input prepends the 38-byte ASCII domain tag:

```text
offset  size  field
------  ----  -----------------------------------------------
0       38    "OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:" (ASCII)
38      84    canonical_digest_bytes (per above)
------  ----
        122   total
```

Ed25519 signs the 122-byte signing input. SHA-512 is internal to the Ed25519 algorithm and is not applied separately at the OmniNode boundary.

## Section 7 — Per-blocker decision form

Chain team fills in one column per row and returns. The follow-on adapter stage is gated on all five rows being filled. **OmniNode side assumes nothing about answers** — wording in the form is intentionally neutral.

| # | Blocker | Decision (✅ confirm / ↻ revise / ❌ refuse) | Notes |
| - | ------- | ------------------------------------------- | ----- |
| A | `TxPayload::IntegrityEvidenceAnchor` variant — append after `Education`, discriminant `23`, bincode-1 byte parity with Section 6 fixtures | | |
| B | Status RPC — `sum_getIntegrityEvidenceAnchorStatus(tx_hash)` mirrors `sum_getInferenceAttestationStatus` envelope; `0x`-prefix accepted | | |
| C | Activation gate — Option C-inherit (OmniNode + V2 only) or Option C-new-flag (`evidence_anchor_is_active`) | | |
| D | Mainnet policy — Yes / No / Yes-with-conditions; if conditions, list them | | |
| E | Fee schedule — `params.min_fee` applies, or specify anchor-specific schedule | | |

**Send completed form to:** OmniNode engineering (this repository's PR thread or the chain-team's preferred handoff channel).

## What happens after the form returns

If all five rows are ✅ or ↻ (with revisions OmniNode can implement), OmniNode opens the **follow-on adapter stage** (Stage 13.2 candidate) to:

1. Pull a `sumchain-primitives` release with the new `TxPayload` variant.
2. Add `impl EvidenceAnchorChainClient for SumChainClient<T>` in `omni-sumchain`.
3. Wire `--rpc-url` / `--expect-chain-id` / `--allow-submit` / `--allow-mainnet-submit` flags into `submit-integrity-evidence-anchor`.
4. Add an optional chain-mode to `query-integrity-evidence-anchor` (chain-read-only, local-registry-mutating).
5. Add `reconcile-integrity-evidence-anchor` (also chain-read-only, local-registry-mutating).
6. Reserve `mainnet_policy_unresolved` reason tag in `EvidenceAnchorError`. If Blocker D = "No" or "Yes-with-conditions-deferred", the future adapter ships with mainnet refused unconditionally under this tag and revisits when policy lands.

Stage 13.1 ships **none** of the above; this packet defines the contract Stage 13.2 implements against.
