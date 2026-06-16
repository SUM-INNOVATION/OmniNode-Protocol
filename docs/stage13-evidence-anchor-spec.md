# Stage 13.0 — Chain Anchoring for Integrity Evidence (Wire Spec)

**Status:** Stage 13.0 frozen — spec + stub implementation. The chain-team review packet that gates the real SUM Chain submission path lives at [`docs/stage13.1-chain-adapter-review.md`](stage13.1-chain-adapter-review.md); a follow-on adapter stage builds against the answers returned in that packet's decision form. No real RPC submission in Stage 13.0; the local stub client persists records to `--anchor-registry-dir` and emits deterministic stub `tx_id`s.

**Scope of this document:** the on-chain wire payload, canonical signing bytes, schema constants, refusal taxonomy, and forward-compatibility constraints that the future SUM Chain adapter must implement against.

**Co-references:**
- [`docs/operator-runbook.md`](operator-runbook.md) §Stage 13.0 — operator workflow.
- [`docs/stage12-contributor-protocol.md`](stage12-contributor-protocol.md) §Forward link — what Stage 12 ships and what Stage 13 layers on top.
- [`docs/stage13.1-chain-adapter-review.md`](stage13.1-chain-adapter-review.md) — chain-team review packet (Stage 13.1).
- `crates/omni-zkml/src/evidence_anchor/` — reference implementation.
- `crates/omni-zkml/tests/fixtures/evidence_anchor_wire_vectors.json` — three deterministic wire vectors the chain team rebuilds for byte-parity assertion.

## Goal

Let an operator prove that a specific local Stage 12.25 `SignedIntegrityEvidenceChainReport` artifact existed on disk and was submitted by its signer at a known time, without pushing the full JSON wrapper to chain. The chain stores a 32-byte BLAKE3 hash + ~70 bytes of metadata under a 64-byte Ed25519 signature; the full forensic record stays local.

## Wire payload

The frozen v1 chain wire payload — `IntegrityEvidenceAnchorTxData`:

```rust
pub const INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION: u32 = 1;

pub const EVIDENCE_ANCHOR_DOMAIN: &[u8] =
    b"OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchoredArtifactKind {
    SignedIntegrityEvidenceChainReport,
}

pub struct IntegrityEvidenceAnchorDigest {
    pub anchor_schema_version: u32,           // = 1, explicit wire field
    pub artifact_kind: AnchoredArtifactKind,  // closed enum
    pub artifact_schema_version: u32,         // from Stage 12.25 wrapper.schema_version
    pub artifact_hash: [u8; 32],              // BLAKE3 over RAW on-disk wrapper bytes
    pub signer_pubkey: [u8; 32],              // Stage 12.25 wrapper's signer pubkey
    pub signed_at_utc_unix: i64,              // RFC 3339 → Unix seconds
}

pub struct IntegrityEvidenceAnchorTxData {
    pub digest: IntegrityEvidenceAnchorDigest,
    pub submitter_signature: [u8; 64],        // Ed25519 over canonical_bytes
                                              // verified under digest.signer_pubkey
}
```

**Field order is frozen.** The on-wire bincode order matches the Rust declaration order; new fields require an `anchor_schema_version` bump (forward-incompatible).

**`anchor_schema_version` is an explicit wire field**, not domain-implied. The versioned domain tag binds canonical bytes to the schema (defense in depth), but the wire payload is the source of truth — a chain adapter / off-chain reader holding only the bytes can read the schema directly.

**`AnchoredArtifactKind` is a closed enum** with one variant in Stage 13.0. New variants require an `anchor_schema_version` bump.

**No separate `submitter_pubkey` field.** Stage 13.0 enforces the same-key-submitter rule: the anchor submitter MUST equal the Stage 12.25 wrapper signer. `digest.signer_pubkey` doubles as the submitter pubkey; the chain adapter verifies `submitter_signature` under that same key. Relay / separate-submitter flows are deferred to a future stage and would require an `anchor_schema_version` bump.

## Canonical bytes + signature

```text
canonical_bytes = EVIDENCE_ANCHOR_DOMAIN || bincode1::serialize(&digest)
submitter_signature = Ed25519 signature over canonical_bytes
                      (verified under digest.signer_pubkey)
```

- Encoder: **bincode 1.3** via the crate-local `bincode1` alias. Matches the Stage 6 chain-wire encoder posture exactly.
- Wire size: domain tag (38 bytes) + bincode-1 of the digest (84 bytes) = 122 bytes signed.
- Domain tag includes the `v1:` suffix; a future schema bump increments to `v2:` (and a new constant) so canonical bytes of two schema versions are byte-distinct even if other fields happen to collide.

Frozen 84-byte bincode-1 layout of the digest:

```text
[u32 LE  anchor_schema_version]      // 4 bytes
[u32 LE  artifact_kind discriminant] // 4 bytes (bincode-1 enum tag)
[u32 LE  artifact_schema_version]    // 4 bytes
[artifact_hash]                      // 32 bytes (no per-field length prefix)
[signer_pubkey]                      // 32 bytes
[i64 LE  signed_at_utc_unix]         // 8 bytes
```

Total: 84 bytes; deterministic and frozen for v1.

## Pre-submit gates (operator-side, enforced by the CLI)

In order, before any chain interaction:

1. Read the **raw on-disk bytes** of the Stage 12.25 wrapper file. These bytes are exactly what gets hashed into `artifact_hash`.
2. Parse the wrapper JSON to extract `schema_version`, `signer_pubkey_hex`, `signed_at_utc` (metadata only — the bytes themselves are what go into the hash).
3. **Verify the Stage 12.25 wrapper signature** under its embedded `signer_pubkey_hex` (`omni_contributor::verify_signed_integrity_evidence_chain_report`). Refuse with `wrapper_signature_invalid` on failure — we do not anchor an unverifiable artifact.
4. Read the 32-byte `--submitter-seed` file. Derive its pubkey. If the derived pubkey ≠ `signer_pubkey_hex` (parsed to 32 bytes), refuse with `submitter_pubkey_mismatch`. Same-key submitter rule.
5. Compute `artifact_hash = blake3(raw_bytes)`. Binds the chain record to the exact byte sequence the operator holds — any reformatting, key-ordering, or whitespace change in the on-disk file produces a different anchor.
6. Build the digest, sign canonical bytes with the submitter seed, submit through the chain client (Stage 13.0: stub; Stage 13.1: real adapter), persist the record locally.

## Verification

Stage 13.0 ships two verify commands:

**Registry-backed (`verify-integrity-evidence-anchor`)** — proves the on-disk artifact corresponds to a recorded anchor authored by the artifact signer. Lookup is by recomputed `blake3(raw_bytes)` by default; `--tx-id` overrides for specific-record lookup, with the recorded hash still required to match.

**Standalone JSON (`verify-integrity-evidence-anchor-file`)** — checks an anchor JSON against the local artifact bytes WITHOUT consulting the registry. Useful for offline forensic review; does NOT prove on-chain inclusion.

Both commands run the same locked pipeline at the CLI boundary:

1. `let raw_bytes = fs::read(--signed-chain-report)?` — read the exact on-disk bytes ONCE.
2. `let wrapper = serde_json::from_slice(&raw_bytes)?` — parse the wrapper from that **same** byte buffer (no second `fs::read` of the path; verify and submit hash exactly what was parsed).
3. `verify_signed_integrity_evidence_chain_report(&wrapper, &wrapper.signer_pubkey_hex)` — refuse with `wrapper_signature_invalid` if the wrapper's own signature does not verify.
4. `let expected_signer_pubkey = parse_anchor_hex_32(&wrapper.signer_pubkey_hex)?` — lift the wrapper signer pubkey for the **same-key verify binding** below.
5. Recompute `artifact_hash = blake3(raw_bytes)` and look up the record (registry-backed) or parse the standalone anchor JSON (file mode).
6. Bind: refuse with `artifact_hash_mismatch` if the recorded / anchor `digest.artifact_hash` ≠ the recomputed hash; refuse with `anchored_signer_pubkey_mismatch` if `digest.signer_pubkey` ≠ `expected_signer_pubkey`. The same-key submitter rule applies at verify time too — a hand-edited registry record or a tampered standalone anchor that reuses the artifact hash but ships a valid signature under a *different* key is refused.
7. Verify `submitter_signature` under `digest.signer_pubkey` (closes the loop on the standard Ed25519 check).

The chain adapter that ships in Stage 13.1 SHOULD additionally expose a registry-vs-chain reconciliation pass (operator confirms that the local record matches what the chain has stored under the same `tx_id`).

## Trust model

- **Attested by chain inclusion**: existence of `(artifact_hash, signer_pubkey, signed_at, artifact_kind, anchor_schema_version, artifact_schema_version)`.
- **Not attested by chain inclusion**: any semantic correctness of the underlying integrity evidence. Stage 12.20-12.25's gates remain the source of truth for whether the wrapper describes a successful chain verification — chain inclusion only proves the operator submitted a commitment to these bytes at this time, under this key.
- **Refusal of unverifiable wrappers**: the CLI's pre-submit gate refuses to anchor a wrapper whose own signature does not verify. This is policy at the submit boundary; the chain itself does not re-verify wrapper signatures and would accept any pair of `(digest, submitter_signature)` that match same-key crypto.

## Refusal taxonomy (closed-set reason tags)

Surfaced as `reason=<tag>` on every `event=integrity_evidence_anchor_..._failed` line. Single mapper `evidence_anchor_reason_tag(&EvidenceAnchorError)` in `omni-zkml::evidence_anchor`.

| Reason tag | When it fires |
| --- | --- |
| `wrapper_signature_invalid` | Stage 12.25 wrapper signature did not verify under its embedded `signer_pubkey_hex` |
| `submitter_pubkey_mismatch` | `--submitter-seed` derived pubkey ≠ wrapper `signer_pubkey_hex` |
| `submitter_signature_invalid` | Anchor `submitter_signature` did not verify under `digest.signer_pubkey` |
| `artifact_hash_mismatch` | Recomputed `blake3(raw_bytes)` ≠ recorded / anchor `digest.artifact_hash` |
| `anchored_signer_pubkey_mismatch` | Verify-time same-key binding: stored / supplied anchor's `digest.signer_pubkey` ≠ parsed Stage 12.25 wrapper's `signer_pubkey_hex` |
| `anchor_not_found` | Registry lookup miss (no record for the supplied selector) |
| `unsupported_anchor_schema_version` | Wire payload `anchor_schema_version` ≠ 1 |
| `unsupported_artifact_schema_version` | Wrapper `schema_version` outside the supported set |
| `unsupported_artifact_kind` | Wire payload `artifact_kind` not in the closed enum |
| `malformed_seed_file` | `--submitter-seed` file did not parse / was not exactly 32 bytes |
| `malformed_json` | Wrapper JSON or anchor JSON parse failure |
| `malformed_signed_at_utc` | Wrapper's `signed_at_utc` did not parse as RFC 3339 |
| `canonical_serialization` | Bincode-1 encoding of canonical bytes failed (closed struct; should be impossible in practice) |
| `signing` | Ed25519 primitive error (pubkey decode, signature decode) |
| `chain_client` | Chain-client error (stub in Stage 13.0; real adapter in Stage 13.1) |
| `io` | FS / registry IO failure |

The tag set is **closed** — adding a new variant to `EvidenceAnchorError` requires updating the mapper, which is enforced by an exhaustive-match test in `omni-zkml::evidence_anchor::reason_tag_tests`.

## Local registry layout

`--anchor-registry-dir <DIR>` holds:

- One JSON file per anchor record at `<DIR>/<artifact_hash_hex>.json`. Atomic temp+rename writes per the project-wide Stage 12 pattern.
- A `<DIR>/tx_index.json` mapping `tx_id → artifact_hash_hex` for `--tx-id` lookups.
- The directory is created on first submit if missing.

The registry is **dedicated** to Stage 13.0 anchors — distinct from the Stage 12.7 contributor `--state-dir` and from the Stage 5 attestation `--registry-path`. The flag name + directory convention (e.g. `/var/omni-anchors/`) make the boundary unambiguous.

## Stub `tx_id` format

Stage 13.0 stub: `anchor-{counter:08x}-{artifact_hash_hex[..12]}` (deterministic per submit-order per process, with a per-process counter starting at 0). Stage 13.1 replaces with the real chain tx hash; off-chain readers MUST treat `tx_id` as opaque.

## Forward compatibility

Stage 13.1 will:
- Replace the stub `EvidenceAnchorChainClient` with a real SUM Chain adapter.
- Wire `query-integrity-evidence-anchor` to real chain state instead of stub-registry status.
- Add a registry-vs-chain reconciliation helper.

Stage 13.1 MUST NOT:
- Change the wire payload's bincode-1 layout, domain tag, or schema constants.
- Change canonical-bytes shape, `submitter_signature` verification key, or the closed `AnchoredArtifactKind` set.
- Add fields to `IntegrityEvidenceAnchorDigest` without bumping `INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION` to 2 and incrementing the domain tag's `v1:` suffix to `v2:`.

Future stages (Stage 13.2+) MAY introduce:
- Separate-submitter flows (relay-style submission where submitter ≠ artifact signer). Requires a new `anchor_schema_version` and adding `submitter_pubkey: [u8; 32]` to the digest.
- New `AnchoredArtifactKind` variants for other integrity-evidence artifact families. Each new variant requires an `anchor_schema_version` bump.
- Aggregation forms (anchor-bundles, anchor-chains).

## Test coverage map

| Concern | Test |
| --- | --- |
| Canonical bytes determinism | `canonical_anchor_bytes_is_deterministic`, `evidence_anchor_digest_roundtrip` |
| Frozen 84-byte bincode-1 layout | `canonical_anchor_bytes_has_frozen_84_byte_layout` |
| Field-by-field hash sensitivity | `canonical_anchor_bytes_changes_when_any_field_changes` |
| Domain tag prefix on signing input | `anchor_signing_input_starts_with_domain_tag`, `evidence_anchor_signing_input_starts_with_domain_tag` |
| Ed25519 signature roundtrip + verification under signer pubkey | `verify_anchor_tx_data_roundtrips_under_signer_pubkey`, `evidence_anchor_signed_wire_roundtrip` |
| Tampered signature refused with `submitter_signature_invalid` | `verify_anchor_tx_data_refuses_tampered_signature`, `evidence_anchor_rejects_bad_submitter_signature` |
| Tampered `anchor_schema_version` refused with `unsupported_anchor_schema_version` | `verify_anchor_tx_data_refuses_unsupported_anchor_schema_version`, `evidence_anchor_rejects_unsupported_anchor_schema_version` |
| Same-key submitter rule (`submitter_pubkey_mismatch`) | `submit_workflow_refuses_mismatched_submitter_seed`, `evidence_anchor_rejects_seed_file_pubkey_mismatch` |
| Raw-byte hash sensitivity (1-byte mutation → different hash) | `evidence_anchor_artifact_hash_binds_raw_bytes` |
| **Raw-byte hash binds formatting** (pretty vs compact JSON of same parsed wrapper → different anchors) | `evidence_anchor_artifact_hash_binds_formatting` |
| Wrapper signature pre-submit gate | `evidence_anchor_rejects_unverified_wrapper` |
| `AnchoredArtifactKind` closed enum / serde refusal of unknown variants | `anchored_artifact_kind_serializes_as_snake_case`, `evidence_anchor_rejects_unsupported_artifact_kind`, `evidence_anchor_artifact_kind_is_closed_and_single_variant_for_stage_13_0` |
| Malformed wrapper / anchor JSON | `evidence_anchor_rejects_malformed_wrapper_json`, `evidence_anchor_rejects_malformed_anchor_json` |
| Malformed seed file reason-tag mapping | `evidence_anchor_rejects_malformed_seed_file` |
| Stub registry insert + load round-trip | `insert_persists_record_and_loads_by_hash_and_tx_id` |
| Stub registry idempotency on same artifact_hash | `insert_is_idempotent_for_same_artifact_hash` |
| Atomic writes leave no `.tmp` files | `atomic_writes_leave_no_tmp_files_on_success` |
| Submit → query status transitions | `query_workflow_transitions_to_finalized_on_chain_finalized`, `evidence_anchor_submit_then_query_status_transitions` |
| Chain `Unknown` observation-only | `query_workflow_leaves_record_unchanged_on_chain_unknown` |
| Registry-backed verify (default hash lookup) | `verify_anchor_against_registry_succeeds_on_matching_bytes`, `evidence_anchor_verify_default_hash_lookup_ok` |
| Registry-backed verify (`--tx-id` lookup) | `evidence_anchor_verify_tx_id_lookup_ok` |
| Registry miss → `anchor_not_found` | `verify_anchor_against_registry_refuses_mutated_bytes`, `evidence_anchor_verify_anchor_not_found` |
| Standalone-JSON verify refuses tampered signature | `verify_anchor_file_against_artifact_bytes_refuses_tampered_signature` |
| **Verify-time same-key binding (registry-backed)**: same artifact hash, valid signature, *different* signer ≠ wrapper signer | `verify_anchor_against_registry_refuses_record_authored_by_other_key`, `evidence_anchor_verify_registry_refuses_wrong_signer_with_same_hash` |
| **Verify-time same-key binding (standalone JSON)**: same artifact hash, valid signature, *different* signer ≠ wrapper signer | `verify_anchor_file_refuses_anchor_authored_by_other_key`, `evidence_anchor_verify_file_refuses_wrong_signer_with_same_hash` |
| Closed-set `reason=<tag>` mapper covers every variant | `every_variant_has_a_stable_tag` |
| End-to-end CLI-shaped flow (parse wrapper → verify wrapper → extract metadata → submit → query → verify → verify-file) | `evidence_anchor_end_to_end_cli_shaped_flow` |
| Schema-version constant locked at 1 | `evidence_anchor_schema_version_is_locked_at_1` |

## Out of scope (Stage 13.0)

- Real SUM Chain RPC submission path. Stage 13.1 deliverable.
- Block-height confirmation polling beyond the stub's "always Submitted unless overridden" model. Stage 13.1.
- Separate-submitter (relay) flows. Schema bump deferred.
- Aggregation forms (anchor-bundles, anchor-chains). Stage 13.2+.
- Cross-chain anchoring. Out of OmniNode scope.
- Revocation / expiry of anchor records. Anchors are append-only by design — chain state is the source of truth for what was submitted; off-chain state is local cache only.
- Replay protection beyond what the same-key submitter rule provides. The chain itself can enforce nonce / tx-id uniqueness; Stage 13.0's stub registry is idempotent on `artifact_hash`.
