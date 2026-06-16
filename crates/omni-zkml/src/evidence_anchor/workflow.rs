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

use crate::error::{EvidenceAnchorError, EvidenceAnchorResult};
use crate::evidence_anchor::client::{AnchorStatus, EvidenceAnchorChainClient};
use crate::evidence_anchor::registry::{
    AnchorRecord, LocalEvidenceAnchorRegistry, local_status_from_chain,
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
#[derive(Debug, Clone)]
pub struct QueryAnchorOutcome {
    pub record: AnchorRecord,
    pub chain_status: AnchorStatus,
    pub local_status_transitioned: bool,
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
    let chain_status = client
        .query_anchor_status(&tx_id)
        .map_err(EvidenceAnchorError::from)?;

    let prior = record.status.clone();
    let new_local = local_status_from_chain(&chain_status);
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
        chain_status,
        local_status_transitioned: transitioned,
    })
}

/// Lookup selector for registry-backed verify / query.
#[derive(Debug, Clone)]
pub enum AnchorSelector<'a> {
    ArtifactHashHex(&'a str),
    TxId(&'a str),
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
