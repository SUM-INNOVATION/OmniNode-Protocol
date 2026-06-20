//! Phase 5 Stage 13.0 — chain-anchor orchestration helpers.
//!
//! Stitches the wire / client / registry layers into the two
//! operator-facing flows:
//!
//! - [`submit_evidence_anchor_workflow`] — sign canonical bytes,
//!   submit through the chain client, persist the record in the
//!   stub registry.
//! - [`query_evidence_anchor_workflow`] — look up a record by
//!   `artifact_hash_hex`, query the chain by stored `tx_id`,
//!   apply the chain-returned status to the local record.
//!
//! Mirrors the Stage 5 orchestration shape (see
//! [`crate::registry::submit_attestation_workflow`] +
//! [`crate::registry::query_attestation_workflow`]).

use crate::error::{ChainClientError, EvidenceAnchorError, EvidenceAnchorResult};
use crate::evidence_anchor::client::{
    AnchorStatus, EvidenceAnchorChainClient, ANCHOR_STATUS_BATCH_MAX,
};
use crate::evidence_anchor::registry::{
    AnchorRecord, LocalAnchorStatus, LocalEvidenceAnchorRegistry, local_status_from_chain,
};
use crate::evidence_anchor::wire::{
    AnchoredArtifactKind, INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION, IntegrityEvidenceAnchorDigest,
    IntegrityEvidenceAnchorTxData, anchor_hex_lower, anchor_signer_pubkey_bytes,
    anchor_signing_input_bytes, sign_anchor_digest,
};

/// Pre-parsed metadata extracted by the CLI from a Stage 12.25
/// `SignedIntegrityEvidenceChainReport`. The CLI is responsible
/// for verifying the wrapper signature **before** constructing
/// this struct — the library never sees the wrapper type itself,
/// keeping `omni-zkml` free of an `omni-contributor` dependency.
#[derive(Debug, Clone)]
pub struct VerifiedWrapperMetadata {
    /// `wrapper.schema_version` (Stage 12.25 wrapper schema).
    pub artifact_schema_version: u32,
    /// 32-byte Ed25519 public key parsed from
    /// `wrapper.signer_pubkey_hex`.
    pub signer_pubkey: [u8; 32],
    /// Unix-seconds time parsed from `wrapper.signed_at_utc`.
    pub signed_at_utc_unix: i64,
}

/// Build a Stage 13.0 anchor digest from a wrapper's metadata and
/// the raw on-disk artifact bytes. The hash is BLAKE3 over the
/// exact file bytes (binds the chain record to the operator's
/// actual on-disk artifact, not a re-serialised representation).
pub fn build_anchor_digest(
    metadata: &VerifiedWrapperMetadata,
    raw_artifact_bytes: &[u8],
) -> IntegrityEvidenceAnchorDigest {
    let mut artifact_hash = [0u8; 32];
    artifact_hash.copy_from_slice(blake3::hash(raw_artifact_bytes).as_bytes());
    IntegrityEvidenceAnchorDigest {
        anchor_schema_version: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
        artifact_kind: AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
        artifact_schema_version: metadata.artifact_schema_version,
        artifact_hash,
        signer_pubkey: metadata.signer_pubkey,
        signed_at_utc_unix: metadata.signed_at_utc_unix,
    }
}

/// Submit a fresh anchor. Steps:
///
/// 1. Same-key check — derive pubkey from `submitter_seed`; if
///    it does not match `digest.signer_pubkey`, refuse with
///    [`EvidenceAnchorError::SubmitterPubkeyMismatch`].
/// 2. Sign canonical bytes; build
///    [`IntegrityEvidenceAnchorTxData`].
/// 3. Submit through `client`; chain-client errors propagate as
///    [`EvidenceAnchorError::ChainClient`] and the registry is
///    **not** modified (Stage 5 invariant).
/// 4. Persist the record in `registry` (idempotent on
///    `artifact_hash`).
///
/// Returns the persisted [`AnchorRecord`].
pub fn submit_evidence_anchor_workflow<C: EvidenceAnchorChainClient>(
    registry: &LocalEvidenceAnchorRegistry,
    client: &C,
    digest: IntegrityEvidenceAnchorDigest,
    submitter_seed: &[u8; 32],
) -> EvidenceAnchorResult<AnchorRecord> {
    let derived_pubkey = anchor_signer_pubkey_bytes(submitter_seed)?;
    if derived_pubkey != digest.signer_pubkey {
        return Err(EvidenceAnchorError::SubmitterPubkeyMismatch {
            derived_pubkey_hex: anchor_hex_lower(&derived_pubkey),
            wrapper_pubkey_hex: anchor_hex_lower(&digest.signer_pubkey),
        });
    }

    let signature = sign_anchor_digest(submitter_seed, &digest)?;
    let tx_data = IntegrityEvidenceAnchorTxData {
        digest,
        submitter_signature: signature,
    };

    let receipt = client
        .submit_anchor(&tx_data)
        .map_err(EvidenceAnchorError::from)?;

    let record = registry
        .insert(tx_data, receipt)
        .map_err(|e| EvidenceAnchorError::Io {
            path: registry.root().to_path_buf(),
            source: e,
        })?;
    Ok(record)
}

/// Outcome of a registry-driven query. Mirrors the Stage 5
/// observation-only contract: `Unknown` leaves the local record
/// unchanged.
///
/// ## Stage 13.9 additions (backward-compatible)
///
/// `included_at_height` and `code` were added as `Option<...>`
/// fields surfacing chain-side metadata that's now exposed via
/// the [`AnchorStatusReport`](crate::AnchorStatusReport) shape.
/// Stage 13.2 callers can continue to ignore them; the existing
/// `record` / `chain_status` / `local_status_transitioned`
/// surface is unchanged.
#[derive(Debug, Clone)]
pub struct QueryAnchorOutcome {
    pub record: AnchorRecord,
    pub chain_status: AnchorStatus,
    pub local_status_transitioned: bool,
    /// Stage 13.9 — chain's `included_at_height` field. Populated
    /// for `Included` / `Finalized` chain responses. `None` for
    /// `Submitted` / `Unknown` / `Failed` (or whenever the chain
    /// returns `null`).
    pub included_at_height: Option<u64>,
    /// Stage 13.9 — chain's `code` field. Stable on `Failed` per
    /// chain contract; `None` otherwise.
    pub code: Option<u32>,
}

/// Look up the record in `registry` and ask the chain for its
/// status. Applies the chain-returned status to the local record
/// (no-op on chain `Unknown`).
pub fn query_evidence_anchor_workflow<C: EvidenceAnchorChainClient>(
    registry: &LocalEvidenceAnchorRegistry,
    client: &C,
    selector: AnchorSelector<'_>,
) -> EvidenceAnchorResult<QueryAnchorOutcome> {
    let record = load_for_selector(registry, &selector)?;
    let tx_id = record.receipt.tx_id.clone();
    // Stage 13.9 — use the richer `query_anchor_status_report`
    // so `included_at_height` / `code` surface on the
    // `QueryAnchorOutcome`. The default trait impl wraps the
    // existing `query_anchor_status` with `None` fields, so
    // stubs that don't override are unaffected.
    let report = client
        .query_anchor_status_report(&tx_id)
        .map_err(EvidenceAnchorError::from)?;

    let prior = record.status.clone();
    let new_local = local_status_from_chain(&report.status);
    let (record, transitioned) = match new_local {
        Some(target) if target != prior => {
            let updated = registry
                .update_status(&record.artifact_hash_hex, target)
                .map_err(|e| EvidenceAnchorError::Io {
                    path: registry.root().to_path_buf(),
                    source: e,
                })?;
            (updated, true)
        }
        _ => (record, false),
    };

    Ok(QueryAnchorOutcome {
        record,
        chain_status: report.status,
        local_status_transitioned: transitioned,
        included_at_height: report.included_at_height,
        code: report.code,
    })
}

/// Lookup selector for registry-backed verify / query.
#[derive(Debug, Clone)]
pub enum AnchorSelector<'a> {
    ArtifactHashHex(&'a str),
    TxId(&'a str),
}

/// Stage 13.2 + Stage 13.9 — sweep the registry and reconcile
/// every local record's status against the chain.
///
/// ## Stage 13.9 rewrite — batched chain reads
///
/// Now uses [`EvidenceAnchorChainClient::query_anchor_status_batch`]
/// internally, chunked at [`ANCHOR_STATUS_BATCH_MAX`]. The trait
/// method's default fallback (per-call `query_anchor_status` loop,
/// fail-fast on the first error) keeps stub clients working without
/// any change to their impl; real chain clients (`omni-sumchain`)
/// override with the actual `sum_getIntegrityEvidenceAnchorStatusBatch`
/// RPC for throughput.
///
/// ## Closed transition table (Stage 13.9 lock)
///
/// | Chain says    | Local was          | Local becomes        | Why                             |
/// | ------------- | ------------------ | -------------------- | ------------------------------- |
/// | `Unknown`     | any                | unchanged            | observation-only (Stage 13.0)   |
/// | `Submitted`   | any                | unchanged            | no reorg downgrade (13.9 lock)  |
/// | `Included`    | `Submitted`        | `Included`           | normal forward transition       |
/// | `Included`    | else               | unchanged            | no reorg downgrade              |
/// | `Finalized`   | `Submitted/Included`| `Finalized`         | normal forward transition       |
/// | `Finalized`   | else               | unchanged            | no reorg downgrade              |
/// | `Failed`      | `Submitted/Included`| `Failed{reason}`    | normal forward transition       |
/// | `Failed`      | else               | unchanged            | no overwrite                    |
///
/// ## Chunk-level error fan-out
///
/// A whole-chunk failure (`Err(ChainClientError)` from the batch
/// call) fans out to ONE per-record `EvidenceAnchorResult::Err`
/// entry for EVERY tx_id in the chunk. The error text is mapped
/// via the existing `EvidenceAnchorError::from(ChainClientError)`
/// path; CLI layer maps via a read-path mapper (Stage 13.9
/// `read_rpc_error_to_evidence_anchor_error`) so JSON-RPC errors
/// route through `chain_rpc` and never through
/// `chain_submit_refused`. The sweep continues with the next
/// chunk.
///
/// ## Backward-compat
///
/// - Public signature unchanged.
/// - Records iterated in deterministic `artifact_hash_hex`
///   ascending order (unchanged from Stage 13.2).
/// - Per-record `Err` containment unchanged (per-item errors
///   from a real batch RPC AND chunk-level failures both surface
///   as `Err` entries in the result vec).
pub fn reconcile_evidence_anchors_workflow<C: EvidenceAnchorChainClient>(
    registry: &LocalEvidenceAnchorRegistry,
    client: &C,
) -> EvidenceAnchorResult<Vec<(String, EvidenceAnchorResult<QueryAnchorOutcome>)>> {
    let mut out = Vec::new();
    let records = registry.list().map_err(|e| EvidenceAnchorError::Io {
        path: registry.root().to_path_buf(),
        source: e,
    })?;

    // Collect queryable records (Submitted | Included) in
    // deterministic order.
    let queryable: Vec<&crate::evidence_anchor::registry::AnchorRecord> = records
        .iter()
        .filter(|r| {
            matches!(
                r.status,
                LocalAnchorStatus::Submitted | LocalAnchorStatus::Included
            )
        })
        .collect();

    // Stage 13.9 — chunk at 100 (chain contract max). Empty
    // collections short-circuit to a single empty batch call
    // (or zero calls), keeping behavior trivial.
    for chunk in queryable.chunks(ANCHOR_STATUS_BATCH_MAX) {
        let tx_ids: Vec<String> = chunk.iter().map(|r| r.receipt.tx_id.clone()).collect();
        match client.query_anchor_status_batch(&tx_ids) {
            Err(chain_err) => {
                // Whole-chunk failure — fan out per-record (Stage
                // 13.9 Finding 5 lock).
                for record in chunk {
                    out.push((
                        record.artifact_hash_hex.clone(),
                        Err(EvidenceAnchorError::from(chain_err.clone())),
                    ));
                }
            }
            Ok(items) => {
                if items.len() != chunk.len() {
                    // Chain returned length-mismatched response.
                    // Surface as malformed for every tx_id in the
                    // chunk. (The omni-sumchain client also
                    // refuses this internally with
                    // ChainErrorCategory::Malformed; this branch
                    // covers stubs that don't.)
                    let synthetic = ChainClientError::Other(format!(
                        "batch response length mismatch: requested {}, got {}",
                        chunk.len(),
                        items.len()
                    ));
                    for record in chunk {
                        out.push((
                            record.artifact_hash_hex.clone(),
                            Err(EvidenceAnchorError::from(synthetic.clone())),
                        ));
                    }
                    continue;
                }
                for (record, item) in chunk.iter().zip(items) {
                    let artifact_hash_hex = record.artifact_hash_hex.clone();
                    let outcome = process_batch_item(registry, record, item);
                    out.push((artifact_hash_hex, outcome));
                }
            }
        }
    }
    Ok(out)
}

/// Stage 13.9 — process one `BatchStatusItem` against a local
/// record. Applies the closed transition table (Submitted /
/// Unknown observation-only; Included / Finalized / Failed
/// forward-transitions on Submitted/Included; no downgrades).
fn process_batch_item(
    registry: &LocalEvidenceAnchorRegistry,
    record: &crate::evidence_anchor::registry::AnchorRecord,
    item: crate::evidence_anchor::client::BatchStatusItem,
) -> EvidenceAnchorResult<QueryAnchorOutcome> {
    // Per-item error from the batch RPC.
    if let Some(err_text) = item.error {
        return Err(EvidenceAnchorError::ChainRpc(err_text));
    }

    let Some(report) = item.result else {
        return Err(EvidenceAnchorError::ChainResponseMalformed(
            "batch item carried neither result nor error".to_string(),
        ));
    };

    let prior = record.status.clone();
    let chain_status = report.status.clone();

    // Stage 13.9 closed transition table (locked in doc above).
    let target: Option<LocalAnchorStatus> = match &chain_status {
        // Observation-only — no change to local.
        AnchorStatus::Unknown => None,
        AnchorStatus::Submitted => None,
        // Forward-only — only transition `Submitted → Included`.
        AnchorStatus::Included => match prior {
            LocalAnchorStatus::Submitted => Some(LocalAnchorStatus::Included),
            _ => None,
        },
        // Forward-only — `Submitted/Included → Finalized`.
        AnchorStatus::Finalized => match prior {
            LocalAnchorStatus::Submitted | LocalAnchorStatus::Included => {
                Some(LocalAnchorStatus::Finalized)
            }
            _ => None,
        },
        // Forward-only — `Submitted/Included → Failed`.
        AnchorStatus::Failed { reason } => match prior {
            LocalAnchorStatus::Submitted | LocalAnchorStatus::Included => {
                Some(LocalAnchorStatus::Failed {
                    reason: reason.clone(),
                })
            }
            _ => None,
        },
    };

    let (updated_record, transitioned) = match target {
        Some(t) => {
            let updated = registry
                .update_status(&record.artifact_hash_hex, t)
                .map_err(|e| EvidenceAnchorError::Io {
                    path: registry.root().to_path_buf(),
                    source: e,
                })?;
            (updated, true)
        }
        None => (record.clone(), false),
    };

    Ok(QueryAnchorOutcome {
        record: updated_record,
        chain_status,
        local_status_transitioned: transitioned,
        included_at_height: report.included_at_height,
        code: report.code,
    })
}

// ── Stage 13.9 — by-tuple lookup workflow ────────────────────────────────────

/// Outcome of a by-tuple chain lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TupleLookupOutcome {
    /// The chain knows about an anchor matching the 5-tuple.
    Found {
        /// Canonical `0x`-prefixed lowercase 32-byte hex of the
        /// chain's authoritative tx hash.
        canonical_tx_hash: String,
        included_at_height: u64,
        /// The local record's `receipt.tx_id` at lookup time —
        /// for cross-comparison with `canonical_tx_hash`. The
        /// CLI surfaces both so operators can detect drift
        /// (e.g. duplicate-anchor scenarios where multiple
        /// submits with the same 5-tuple race; chain first-wins).
        local_record_tx_id: String,
    },
    /// The chain's `result` was `null` — no chain anchor for
    /// this tuple. Surfaces as an informational event line at
    /// the CLI layer (NO `reason=` key per Stage 13.9
    /// REJECT-fix Finding 5).
    NotFound,
}

/// Stage 13.9 — look up the canonical chain anchor for the
/// 5-tuple derived from a local record. **Read-only** — does
/// NOT mutate the registry (Stage 13.9 Q3 lock).
///
/// The local record selector picks which record's digest fields
/// to use as the tuple; raw tuple flags are never accepted (CLI
/// footgun). Tuple fields:
/// `(anchor_schema_version, artifact_kind, artifact_schema_version,
/// artifact_hash, signer_pubkey)`.
pub fn lookup_anchor_by_tuple_workflow<C: EvidenceAnchorChainClient>(
    registry: &LocalEvidenceAnchorRegistry,
    client: &C,
    selector: AnchorSelector<'_>,
) -> EvidenceAnchorResult<TupleLookupOutcome> {
    let record = load_for_selector(registry, &selector)?;
    let local_record_tx_id = record.receipt.tx_id.clone();
    let digest = &record.tx_data.digest;
    let chain_result = client
        .lookup_anchor_by_tuple(
            digest.anchor_schema_version,
            digest.artifact_kind,
            digest.artifact_schema_version,
            &digest.artifact_hash,
            &digest.signer_pubkey,
        )
        .map_err(EvidenceAnchorError::from)?;
    match chain_result {
        Some(found) => Ok(TupleLookupOutcome::Found {
            canonical_tx_hash: found.tx_hash,
            included_at_height: found.included_at_height,
            local_record_tx_id,
        }),
        None => Ok(TupleLookupOutcome::NotFound),
    }
}

fn load_for_selector(
    registry: &LocalEvidenceAnchorRegistry,
    selector: &AnchorSelector<'_>,
) -> EvidenceAnchorResult<AnchorRecord> {
    let lookup = match selector {
        AnchorSelector::ArtifactHashHex(h) => {
            registry
                .load_by_artifact_hash(h)
                .map_err(|e| EvidenceAnchorError::Io {
                    path: registry.root().to_path_buf(),
                    source: e,
                })?
        }
        AnchorSelector::TxId(id) => {
            registry
                .load_by_tx_id(id)
                .map_err(|e| EvidenceAnchorError::Io {
                    path: registry.root().to_path_buf(),
                    source: e,
                })?
        }
    };
    lookup.ok_or_else(|| EvidenceAnchorError::AnchorNotFound {
        selector: match selector {
            AnchorSelector::ArtifactHashHex(h) => format!("artifact_hash={h}"),
            AnchorSelector::TxId(id) => format!("tx_id={id}"),
        },
    })
}

/// Registry-backed verify. Used by
/// `verify-integrity-evidence-anchor`:
///
/// 1. Caller passes the raw bytes of the on-disk Stage 12.25
///    wrapper, the 32-byte wrapper signer pubkey lifted from
///    the parsed-and-verified wrapper, and an optional `tx_id`
///    selector. Wrapper-signature verification + metadata
///    extraction has already happened in the CLI before this
///    call.
/// 2. Recompute `artifact_hash = blake3(raw_bytes)`.
/// 3. Look up the record (by `tx_id` if supplied, else by the
///    recomputed hash).
/// 4. Confirm the recorded `digest.artifact_hash` matches the
///    recomputed hash.
/// 5. **Bind to the wrapper signer.** Confirm
///    `record.digest.signer_pubkey == expected_wrapper_signer_pubkey`.
///    The chain anchor MUST be authored by the same key that
///    signed the artifact — defends against a hand-edited
///    registry record that swaps in a different signer pubkey
///    while reusing the artifact hash (and ships a valid
///    signature by that other key).
/// 6. Verify `record.submitter_signature` under
///    `record.digest.signer_pubkey` (same-key-submitter rule).
/// 7. Return the matched record.
pub fn verify_anchor_against_registry(
    registry: &LocalEvidenceAnchorRegistry,
    raw_artifact_bytes: &[u8],
    expected_wrapper_signer_pubkey: &[u8; 32],
    tx_id: Option<&str>,
) -> EvidenceAnchorResult<AnchorRecord> {
    let mut recomputed = [0u8; 32];
    recomputed.copy_from_slice(blake3::hash(raw_artifact_bytes).as_bytes());
    let recomputed_hex = anchor_hex_lower(&recomputed);

    let selector = match tx_id {
        Some(id) => AnchorSelector::TxId(id),
        None => AnchorSelector::ArtifactHashHex(&recomputed_hex),
    };
    let record = load_for_selector(registry, &selector)?;

    if record.tx_data.digest.artifact_hash != recomputed {
        return Err(EvidenceAnchorError::ArtifactHashMismatch {
            recomputed_hex,
            anchored_hex: anchor_hex_lower(&record.tx_data.digest.artifact_hash),
        });
    }
    if &record.tx_data.digest.signer_pubkey != expected_wrapper_signer_pubkey {
        return Err(EvidenceAnchorError::AnchoredSignerPubkeyMismatch {
            wrapper_pubkey_hex: anchor_hex_lower(expected_wrapper_signer_pubkey),
            anchored_pubkey_hex: anchor_hex_lower(&record.tx_data.digest.signer_pubkey),
        });
    }
    // verify_anchor_tx_data already covers schema version +
    // signature checks per same-key-submitter rule.
    crate::evidence_anchor::wire::verify_anchor_tx_data(&record.tx_data)?;
    Ok(record)
}

/// Standalone-JSON verify. Used by
/// `verify-integrity-evidence-anchor-file`: does **not** prove
/// submission / inclusion (no registry lookup), only that the
/// anchor JSON is internally consistent with the local artifact
/// bytes AND is signed by the same key that signed the wrapper.
///
/// Same same-key-submitter binding as the registry-backed
/// verify: callers MUST pass the 32-byte signer pubkey lifted
/// from the parsed-and-verified Stage 12.25 wrapper. A tampered
/// standalone anchor that reuses the artifact hash but ships a
/// valid signature by a *different* key is refused with
/// [`EvidenceAnchorError::AnchoredSignerPubkeyMismatch`].
pub fn verify_anchor_file_against_artifact_bytes(
    tx_data: &IntegrityEvidenceAnchorTxData,
    raw_artifact_bytes: &[u8],
    expected_wrapper_signer_pubkey: &[u8; 32],
) -> EvidenceAnchorResult<()> {
    let mut recomputed = [0u8; 32];
    recomputed.copy_from_slice(blake3::hash(raw_artifact_bytes).as_bytes());
    if tx_data.digest.artifact_hash != recomputed {
        return Err(EvidenceAnchorError::ArtifactHashMismatch {
            recomputed_hex: anchor_hex_lower(&recomputed),
            anchored_hex: anchor_hex_lower(&tx_data.digest.artifact_hash),
        });
    }
    if &tx_data.digest.signer_pubkey != expected_wrapper_signer_pubkey {
        return Err(EvidenceAnchorError::AnchoredSignerPubkeyMismatch {
            wrapper_pubkey_hex: anchor_hex_lower(expected_wrapper_signer_pubkey),
            anchored_pubkey_hex: anchor_hex_lower(&tx_data.digest.signer_pubkey),
        });
    }
    crate::evidence_anchor::wire::verify_anchor_tx_data(tx_data)?;
    Ok(())
}

/// Helper for the CLI: serialize canonical signing-input bytes
/// of a digest for diagnostic / pretty-output purposes. Kept
/// public so tests can pin the exact byte layout.
pub fn anchor_signing_input_for_digest(
    digest: &IntegrityEvidenceAnchorDigest,
) -> EvidenceAnchorResult<Vec<u8>> {
    anchor_signing_input_bytes(digest)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_anchor::client::{AnchorStatus, StubEvidenceAnchorChainClient};
    use crate::evidence_anchor::registry::LocalAnchorStatus;

    fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry) {
        let dir = tempfile::tempdir().unwrap();
        let reg = LocalEvidenceAnchorRegistry::open(dir.path().join("anchors")).unwrap();
        (dir, reg)
    }

    fn raw_artifact_bytes() -> Vec<u8> {
        // Synthetic raw artifact bytes. The Stage 13.0 library
        // is agnostic about JSON shape — it only hashes raw
        // bytes.
        b"{\"schema_version\":1,\"some\":\"artifact\"}".to_vec()
    }

    fn happy_metadata(seed: &[u8; 32]) -> VerifiedWrapperMetadata {
        VerifiedWrapperMetadata {
            artifact_schema_version: 1,
            signer_pubkey: anchor_signer_pubkey_bytes(seed).unwrap(),
            signed_at_utc_unix: 1_700_000_000,
        }
    }

    #[test]
    fn submit_workflow_persists_record_and_returns_receipt() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let seed = [7u8; 32];
        let raw = raw_artifact_bytes();
        let digest = build_anchor_digest(&happy_metadata(&seed), &raw);
        let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
        assert_eq!(record.status, LocalAnchorStatus::Submitted);
        assert!(record.receipt.tx_id.starts_with("anchor-"));
        // Persisted.
        let reloaded = reg
            .load_by_artifact_hash(&record.artifact_hash_hex)
            .unwrap()
            .unwrap();
        assert_eq!(reloaded, record);
    }

    #[test]
    fn submit_workflow_refuses_mismatched_submitter_seed() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let wrapper_seed = [7u8; 32];
        let bad_submitter_seed = [9u8; 32];
        let raw = raw_artifact_bytes();
        let digest = build_anchor_digest(&happy_metadata(&wrapper_seed), &raw);
        let err = submit_evidence_anchor_workflow(&reg, &client, digest, &bad_submitter_seed)
            .unwrap_err();
        assert!(matches!(
            err,
            EvidenceAnchorError::SubmitterPubkeyMismatch { .. }
        ));
    }

    #[test]
    fn query_workflow_transitions_to_finalized_on_chain_finalized() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let seed = [7u8; 32];
        let raw = raw_artifact_bytes();
        let digest = build_anchor_digest(&happy_metadata(&seed), &raw);
        let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
        client.set_status_for(&record.receipt.tx_id, AnchorStatus::Finalized);

        let outcome = query_evidence_anchor_workflow(
            &reg,
            &client,
            AnchorSelector::ArtifactHashHex(&record.artifact_hash_hex),
        )
        .unwrap();
        assert_eq!(outcome.record.status, LocalAnchorStatus::Finalized);
        assert!(outcome.local_status_transitioned);
        assert!(matches!(outcome.chain_status, AnchorStatus::Finalized));
    }

    #[test]
    fn query_workflow_leaves_record_unchanged_on_chain_unknown() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let seed = [7u8; 32];
        let raw = raw_artifact_bytes();
        let digest = build_anchor_digest(&happy_metadata(&seed), &raw);
        let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
        client.set_status_for(&record.receipt.tx_id, AnchorStatus::Unknown);
        let outcome = query_evidence_anchor_workflow(
            &reg,
            &client,
            AnchorSelector::TxId(&record.receipt.tx_id),
        )
        .unwrap();
        assert_eq!(outcome.record.status, LocalAnchorStatus::Submitted);
        assert!(!outcome.local_status_transitioned);
        assert!(matches!(outcome.chain_status, AnchorStatus::Unknown));
    }

    #[test]
    fn verify_anchor_against_registry_succeeds_on_matching_bytes() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let seed = [7u8; 32];
        let raw = raw_artifact_bytes();
        let signer = anchor_signer_pubkey_bytes(&seed).unwrap();
        let digest = build_anchor_digest(&happy_metadata(&seed), &raw);
        submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
        let _record = verify_anchor_against_registry(&reg, &raw, &signer, None).unwrap();
    }

    #[test]
    fn verify_anchor_against_registry_refuses_mutated_bytes() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let seed = [7u8; 32];
        let raw = raw_artifact_bytes();
        let signer = anchor_signer_pubkey_bytes(&seed).unwrap();
        let digest = build_anchor_digest(&happy_metadata(&seed), &raw);
        submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
        let mutated = b"{\"schema_version\":1,\"some\":\"other-artifact\"}".to_vec();
        let err = verify_anchor_against_registry(&reg, &mutated, &signer, None).unwrap_err();
        // No record exists for the mutated bytes' hash.
        assert!(matches!(err, EvidenceAnchorError::AnchorNotFound { .. }));
    }

    #[test]
    fn verify_anchor_against_registry_refuses_when_tx_id_record_hash_diverges() {
        // Synthesize a scenario where the operator passed a tx_id
        // but the on-disk artifact bytes hash differently from
        // what's stored under that tx_id. This is the
        // ArtifactHashMismatch path.
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let seed = [7u8; 32];
        let raw = raw_artifact_bytes();
        let signer = anchor_signer_pubkey_bytes(&seed).unwrap();
        let digest = build_anchor_digest(&happy_metadata(&seed), &raw);
        let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
        let mutated = b"different bytes".to_vec();
        let err = verify_anchor_against_registry(
            &reg,
            &mutated,
            &signer,
            Some(&record.receipt.tx_id),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvidenceAnchorError::ArtifactHashMismatch { .. }
        ));
    }

    /// Same-key submitter rule, verified at registry-lookup
    /// time: a hand-edited record that reuses the artifact
    /// hash but swaps in a different signer pubkey (with a
    /// valid signature by that other key) is refused.
    #[test]
    fn verify_anchor_against_registry_refuses_record_authored_by_other_key() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let attacker_seed = [9u8; 32];
        let wrapper_seed = [7u8; 32];
        let raw = raw_artifact_bytes();

        // Anchor authored by the ATTACKER key (not the wrapper signer).
        let digest = build_anchor_digest(&happy_metadata(&attacker_seed), &raw);
        let record =
            submit_evidence_anchor_workflow(&reg, &client, digest, &attacker_seed).unwrap();

        // verify is called with the *wrapper* signer (the only
        // identity that should be allowed to anchor this artifact).
        let wrapper_pubkey = anchor_signer_pubkey_bytes(&wrapper_seed).unwrap();
        let err = verify_anchor_against_registry(
            &reg,
            &raw,
            &wrapper_pubkey,
            Some(&record.receipt.tx_id),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EvidenceAnchorError::AnchoredSignerPubkeyMismatch { .. }
        ));
    }

    #[test]
    fn verify_anchor_file_against_artifact_bytes_refuses_tampered_signature() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let seed = [7u8; 32];
        let raw = raw_artifact_bytes();
        let signer = anchor_signer_pubkey_bytes(&seed).unwrap();
        let digest = build_anchor_digest(&happy_metadata(&seed), &raw);
        let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
        let mut tx_data = record.tx_data.clone();
        tx_data.submitter_signature[0] ^= 0x01;
        let err = verify_anchor_file_against_artifact_bytes(&tx_data, &raw, &signer)
            .unwrap_err();
        assert!(matches!(
            err,
            EvidenceAnchorError::SubmitterSignatureInvalid
        ));
    }

    /// Same-key submitter rule for the standalone-JSON path:
    /// a free-floating anchor authored by a different valid
    /// Ed25519 key but committing the same artifact hash is
    /// refused.
    #[test]
    fn verify_anchor_file_refuses_anchor_authored_by_other_key() {
        let (_dir, reg) = fresh_registry();
        let client = StubEvidenceAnchorChainClient::new();
        let attacker_seed = [9u8; 32];
        let wrapper_seed = [7u8; 32];
        let raw = raw_artifact_bytes();

        // Build a free-floating anchor under the attacker's key
        // (registered via submit so we have a valid tx_data shape).
        let digest = build_anchor_digest(&happy_metadata(&attacker_seed), &raw);
        let record =
            submit_evidence_anchor_workflow(&reg, &client, digest, &attacker_seed).unwrap();
        let tx_data = record.tx_data.clone();

        let wrapper_pubkey = anchor_signer_pubkey_bytes(&wrapper_seed).unwrap();
        let err = verify_anchor_file_against_artifact_bytes(&tx_data, &raw, &wrapper_pubkey)
            .unwrap_err();
        assert!(matches!(
            err,
            EvidenceAnchorError::AnchoredSignerPubkeyMismatch { .. }
        ));
    }
}
