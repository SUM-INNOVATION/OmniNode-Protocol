//! Phase 5 Stage 5 — offline attestation registry + chain workflow.
//!
//! On-disk store of `InferenceAttestation` records, keyed by a 32-byte
//! `AttestationId` derived deterministically from `(session_id,
//! verifier_address)` (the de-duplication key the chain proposal expects).
//! Records are persisted as one JSON file per record at
//! `<root>/<hex_id>.json`, written via the standard `.tmp` + rename
//! pattern so partial writes can never corrupt a record.
//!
//! A small state machine (`LocalAttestationStatus`) tracks lifecycle
//! progress; a `ChainClient` (see `crate::chain`) is the seam a future
//! real SUM Chain adapter will implement to drive the workflow functions
//! `submit_attestation_workflow` and `query_attestation_workflow`.
//!
//! Chain-client RPC failures propagate as `RegistryError::ChainClient(_)`
//! and **leave the local record's status unchanged**. Among chain-returned
//! statuses, only `Failed { reason }` transitions the local record into a
//! terminal local state. SUM Chain v1's `Unknown` (unrecognized tx hash —
//! could be mempool eviction, never-seen tx, or chain lag) is
//! observation-only: the record is left unchanged and a `tracing::warn!`
//! fires. The local `Dropped` state is a **client-side synthetic**
//! variant; it is never set from a chain-returned status. Stage 5.2's
//! local staleness/timeout detection will be the only writer that
//! transitions `Submitted → Dropped`.

use std::fmt;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use omni_types::phase5::InferenceAttestation;

use crate::attestation::{compute_digest, CommitmentDigest};
use crate::chain::{AttestationStatus, ChainClient, SubmissionReceipt};
use crate::error::{RegistryError, RegistryResult};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Domain tag for [`AttestationId`] derivation. Bumping the trailing `vN`
/// is the contract for any breaking change to the key derivation.
pub const ATTESTATION_ID_DOMAIN: &str = "omninode.attestation_record.v1";

// ── AttestationId ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct AttestationIdPayload {
    domain: String,
    session_id: String,
    verifier_address: String,
}

/// 32-byte deterministic identifier for a registry record.
///
/// Derived from `BLAKE3(bincode(AttestationIdPayload { domain,
/// session_id, verifier_address }))`. **Signature and commitment body are
/// deliberately not part of the key** — this matches the chain proposal's
/// `(session_id, verifier_address)` de-duplication rule. The Stage-4
/// `CommitmentDigest` is stored separately on `AttestationRecord` so
/// callers can still cross-check the commitment bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttestationId([u8; 32]);

impl AttestationId {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase 64-char hex without `0x` prefix.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in &self.0 {
            use std::fmt::Write;
            let _ = write!(&mut s, "{:02x}", b);
        }
        s
    }

    fn from_hex(s: &str) -> std::result::Result<Self, String> {
        if s.len() != 64 {
            return Err(format!(
                "AttestationId hex must be 64 chars, got {}",
                s.len()
            ));
        }
        let bytes = s.as_bytes();
        let mut out = [0u8; 32];
        for i in 0..32 {
            let hi = decode_lower_nibble(bytes[i * 2])?;
            let lo = decode_lower_nibble(bytes[i * 2 + 1])?;
            out[i] = (hi << 4) | lo;
        }
        Ok(Self(out))
    }
}

fn decode_lower_nibble(b: u8) -> std::result::Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Err(format!("uppercase hex not allowed: '{}'", b as char)),
        other => Err(format!("invalid hex digit: '{}'", other as char)),
    }
}

impl fmt::Display for AttestationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl Serialize for AttestationId {
    fn serialize<S: Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
        ser.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for AttestationId {
    fn deserialize<D: Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        AttestationId::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// Compute the deterministic [`AttestationId`] for an attestation.
///
/// Only `commitment.session_id` and `verifier_address` flow into the
/// hash, alongside [`ATTESTATION_ID_DOMAIN`]. See the type doc on
/// [`AttestationId`] for the rationale.
pub fn compute_attestation_id(att: &InferenceAttestation) -> RegistryResult<AttestationId> {
    let payload = AttestationIdPayload {
        domain: ATTESTATION_ID_DOMAIN.to_string(),
        session_id: att.commitment.session_id.clone(),
        verifier_address: att.verifier_address.clone(),
    };
    let bytes = bincode::serde::encode_to_vec(&payload, bincode::config::standard())
        .map_err(|e| RegistryError::Serialization(e.to_string()))?;
    let mut out = [0u8; 32];
    out.copy_from_slice(blake3::hash(&bytes).as_bytes());
    Ok(AttestationId(out))
}

// ── LocalAttestationStatus ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalAttestationStatus {
    /// Local record created, not yet submitted. Never sourced from chain.
    Pending,
    /// Chain accepted the submission; awaiting inclusion or finality.
    Submitted,
    /// Chain reports the submission is included in a block but not yet
    /// finalized.
    Included,
    /// Chain reports the submission is final. **Terminal.**
    Finalized,
    /// Chain reported execution failure. **Terminal for this stage.**
    Failed { reason: String },
    /// **Client-side synthetic state.** SUM Chain v1 does **not** report a
    /// `Dropped` status — chain v1 surfaces unrecognized tx hashes as
    /// [`crate::chain::AttestationStatus::Unknown`]. OmniNode marks a
    /// record `Dropped` only via local staleness/timeout detection
    /// (Stage 5.2, not part of Stage 5.1), never as a direct
    /// translation of a chain-returned status. **Retryable** via
    /// `mark_submitted` (which transitions `Dropped → Submitted`).
    Dropped { reason: Option<String> },
}

// ── AttestationRecord ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationRecord {
    pub id: AttestationId,
    pub digest: CommitmentDigest,
    pub attestation: InferenceAttestation,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub status: LocalAttestationStatus,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<SubmissionReceipt>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

// ── AttestationRegistry ──────────────────────────────────────────────────────

/// Filesystem-backed attestation store. One JSON file per record at
/// `<root>/<hex_id>.json`; atomic-rename writes; idempotent insert keyed
/// by `(session_id, verifier_address)`.
pub struct AttestationRegistry {
    root: PathBuf,
}

impl AttestationRegistry {
    /// Open (or create) a registry rooted at `root`.
    pub fn open(root: PathBuf) -> RegistryResult<Self> {
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for(&self, id: &AttestationId) -> PathBuf {
        self.root.join(format!("{}.json", id.to_hex()))
    }

    fn tmp_path_for(&self, id: &AttestationId) -> PathBuf {
        self.root.join(format!("{}.json.tmp", id.to_hex()))
    }

    fn write_atomic(&self, record: &AttestationRecord) -> RegistryResult<()> {
        let bytes = serde_json::to_vec_pretty(record)
            .map_err(|e| RegistryError::Serialization(e.to_string()))?;
        let tmp = self.tmp_path_for(&record.id);
        let final_path = self.path_for(&record.id);

        if let Err(e) = std::fs::write(&tmp, &bytes) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
        if let Err(e) = std::fs::rename(&tmp, &final_path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
        Ok(())
    }

    /// Insert a new record.
    ///
    /// - If no record exists under `compute_attestation_id(&attestation)`,
    ///   writes a new `Pending` record and returns it.
    /// - If a record exists and stores a **byte-equal** `attestation`,
    ///   returns the existing record unchanged (idempotent — no file
    ///   rewrite, `updated_at` not bumped).
    /// - If a record exists but stores a **byte-different** `attestation`,
    ///   returns [`RegistryError::ConflictingAttestation`] with the
    ///   existing record left intact.
    pub fn insert(
        &self,
        attestation: InferenceAttestation,
    ) -> RegistryResult<AttestationRecord> {
        let id = compute_attestation_id(&attestation)?;

        if self.path_for(&id).exists() {
            let existing = self.load(&id)?;
            if existing.attestation == attestation {
                return Ok(existing);
            }
            return Err(RegistryError::ConflictingAttestation { id });
        }

        let digest = compute_digest(&attestation.commitment).map_err(|e| {
            RegistryError::Serialization(format!("digest computation failed: {e}"))
        })?;

        let now = Utc::now();
        let record = AttestationRecord {
            id,
            digest,
            attestation,
            created_at: now,
            updated_at: now,
            status: LocalAttestationStatus::Pending,
            receipt: None,
            error_message: None,
        };
        self.write_atomic(&record)?;
        tracing::info!(id = %record.id, "inserted attestation record");
        Ok(record)
    }

    pub fn load(&self, id: &AttestationId) -> RegistryResult<AttestationRecord> {
        let path = self.path_for(id);
        if !path.is_file() {
            return Err(RegistryError::RecordNotFound(*id));
        }
        let bytes = std::fs::read(&path)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| RegistryError::Serialization(e.to_string()))
    }

    /// List all records on disk, sorted by `id.to_hex()` ascending.
    /// Non-`.json` files in the root are ignored (so stray `.tmp` files
    /// from a crashed write do not block `list`). A parse failure on a
    /// `.json` file surfaces as [`RegistryError::Serialization`] — `list`
    /// does **not** silently skip malformed records.
    pub fn list(&self) -> RegistryResult<Vec<AttestationRecord>> {
        let mut records = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let bytes = std::fs::read(&path)?;
            let record: AttestationRecord = serde_json::from_slice(&bytes)
                .map_err(|e| {
                    RegistryError::Serialization(format!("{}: {e}", path.display()))
                })?;
            records.push(record);
        }
        records.sort_by(|a, b| a.id.to_hex().cmp(&b.id.to_hex()));
        Ok(records)
    }

    fn update_record<F>(
        &self,
        id: &AttestationId,
        mutate: F,
    ) -> RegistryResult<AttestationRecord>
    where
        F: FnOnce(&mut AttestationRecord) -> RegistryResult<()>,
    {
        let mut record = self.load(id)?;
        mutate(&mut record)?;
        record.updated_at = Utc::now();
        self.write_atomic(&record)?;
        Ok(record)
    }

    /// `Pending → Submitted` or `Dropped → Submitted` (retry). Sets the
    /// receipt; clears any prior `error_message`.
    pub fn mark_submitted(
        &self,
        id: &AttestationId,
        receipt: SubmissionReceipt,
    ) -> RegistryResult<AttestationRecord> {
        self.update_record(id, |r| match &r.status {
            LocalAttestationStatus::Pending | LocalAttestationStatus::Dropped { .. } => {
                r.status = LocalAttestationStatus::Submitted;
                r.receipt = Some(receipt);
                r.error_message = None;
                Ok(())
            }
            _ => Err(RegistryError::InvalidStatusTransition {
                id: *id,
                from: r.status.clone(),
                to: "submitted",
            }),
        })
    }

    /// `Submitted → Included`.
    pub fn mark_included(&self, id: &AttestationId) -> RegistryResult<AttestationRecord> {
        self.update_record(id, |r| match &r.status {
            LocalAttestationStatus::Submitted => {
                r.status = LocalAttestationStatus::Included;
                Ok(())
            }
            _ => Err(RegistryError::InvalidStatusTransition {
                id: *id,
                from: r.status.clone(),
                to: "included",
            }),
        })
    }

    /// `Submitted → Finalized` (chain skipped Included) or
    /// `Included → Finalized`. Terminal.
    pub fn mark_finalized(&self, id: &AttestationId) -> RegistryResult<AttestationRecord> {
        self.update_record(id, |r| match &r.status {
            LocalAttestationStatus::Submitted | LocalAttestationStatus::Included => {
                r.status = LocalAttestationStatus::Finalized;
                Ok(())
            }
            _ => Err(RegistryError::InvalidStatusTransition {
                id: *id,
                from: r.status.clone(),
                to: "finalized",
            }),
        })
    }

    /// `Submitted → Failed` or `Included → Failed`. Terminal this stage.
    /// Records the reason on `error_message` for surface-level
    /// inspection.
    pub fn mark_failed(
        &self,
        id: &AttestationId,
        reason: String,
    ) -> RegistryResult<AttestationRecord> {
        self.update_record(id, |r| match &r.status {
            LocalAttestationStatus::Submitted | LocalAttestationStatus::Included => {
                r.error_message = Some(reason.clone());
                r.status = LocalAttestationStatus::Failed { reason };
                Ok(())
            }
            _ => Err(RegistryError::InvalidStatusTransition {
                id: *id,
                from: r.status.clone(),
                to: "failed",
            }),
        })
    }

    /// `Submitted → Dropped`. Retryable via `mark_submitted`.
    ///
    /// Invoked by local staleness/timeout detection (Stage 5.2); never as
    /// a direct translation of a chain-returned status. SUM Chain v1
    /// surfaces unrecognized tx hashes as
    /// [`crate::chain::AttestationStatus::Unknown`], which the query
    /// workflow leaves observation-only.
    pub fn mark_dropped(
        &self,
        id: &AttestationId,
        reason: Option<String>,
    ) -> RegistryResult<AttestationRecord> {
        self.update_record(id, |r| match &r.status {
            LocalAttestationStatus::Submitted => {
                r.error_message = Some(
                    reason
                        .clone()
                        .unwrap_or_else(|| "dropped by chain".to_string()),
                );
                r.status = LocalAttestationStatus::Dropped { reason };
                Ok(())
            }
            _ => Err(RegistryError::InvalidStatusTransition {
                id: *id,
                from: r.status.clone(),
                to: "dropped",
            }),
        })
    }
}

// ── Workflows ────────────────────────────────────────────────────────────────

/// Submit an attestation to the chain client and update the registry.
///
/// - If the registry already has a record under the computed
///   `(session_id, verifier_address)` key and the attestation byte-matches,
///   the existing record is returned and the chain is **only** called when
///   the record's status is `Pending` or `Dropped`.
/// - If the existing record has a byte-different attestation, the
///   underlying `insert` returns
///   [`RegistryError::ConflictingAttestation`] and the workflow does not
///   call the chain.
/// - If the chain client returns `Err`, the workflow returns
///   `Err(RegistryError::ChainClient(_))` and **leaves the record's
///   status unchanged**. RPC failures are not terminal — the caller can
///   retry.
pub fn submit_attestation_workflow<C: ChainClient>(
    registry: &AttestationRegistry,
    client: &C,
    attestation: InferenceAttestation,
) -> RegistryResult<AttestationRecord> {
    let record = registry.insert(attestation.clone())?;
    match &record.status {
        LocalAttestationStatus::Pending | LocalAttestationStatus::Dropped { .. } => {
            // proceed to chain submission
        }
        _ => return Ok(record),
    }

    let receipt = client
        .submit_attestation(&attestation)
        .map_err(RegistryError::ChainClient)?;
    registry.mark_submitted(&record.id, receipt)
}

/// Query the chain client for the latest status of a record and update
/// the registry accordingly.
///
/// External signature stays keyed by [`AttestationId`] (the local record
/// key). Internally the workflow extracts the stored
/// [`SubmissionReceipt::tx_id`](crate::chain::SubmissionReceipt::tx_id)
/// from the record and passes it to
/// [`ChainClient::query_attestation_status`], which is keyed by `tx_id`
/// to match SUM Chain v1's `sum_getInferenceAttestationStatus(tx_hash)`.
///
/// - Only `Submitted` and `Included` records are queried; others are
///   returned unchanged with no chain call.
/// - **Receipt is required.** If a `Submitted` or `Included` record has
///   `receipt: None` (only reachable via hand-edited / corrupted JSON
///   since `mark_submitted` always sets the receipt),
///   [`RegistryError::SubmittedRecordMissingReceipt`] is returned and
///   the chain is **not** called. This makes registry-directory
///   corruption visible rather than silently masked as a no-op.
/// - Chain-client RPC failures propagate as
///   `Err(RegistryError::ChainClient(_))`; the record is **not** modified.
/// - Chain status `Submitted` produces no local transition. `Included`
///   transitions `Submitted → Included` (no-op if already `Included`).
///   `Finalized` / `Failed` apply the corresponding `mark_*` method.
/// - **`Unknown` is observation-only.** SUM Chain v1 returns `Unknown`
///   for unrecognized tx hashes (could mean mempool eviction, never-seen
///   tx, or chain lag). The workflow leaves the record unchanged and
///   emits `tracing::warn!`. It does **not** mark the record `Failed`
///   (that would be terminal and wrong) and does **not** auto-transition
///   to local `Dropped` (that's a policy decision belonging to staleness
///   detection in Stage 5.2). Callers that need timeout-based retry must
///   implement it on top — Stage 5.2 will land that surface.
pub fn query_attestation_workflow<C: ChainClient>(
    registry: &AttestationRegistry,
    client: &C,
    id: &AttestationId,
) -> RegistryResult<AttestationRecord> {
    let record = registry.load(id)?;
    match &record.status {
        LocalAttestationStatus::Submitted | LocalAttestationStatus::Included => {
            // proceed to chain query
        }
        _ => return Ok(record),
    }

    let tx_id = record
        .receipt
        .as_ref()
        .map(|r| r.tx_id.as_str())
        .ok_or(RegistryError::SubmittedRecordMissingReceipt { id: *id })?;

    let status = client
        .query_attestation_status(tx_id)
        .map_err(RegistryError::ChainClient)?;
    match status {
        AttestationStatus::Submitted => Ok(record),
        AttestationStatus::Included => {
            if matches!(record.status, LocalAttestationStatus::Submitted) {
                registry.mark_included(id)
            } else {
                Ok(record)
            }
        }
        AttestationStatus::Finalized => registry.mark_finalized(id),
        AttestationStatus::Failed { reason } => registry.mark_failed(id, reason),
        AttestationStatus::Unknown => {
            tracing::warn!(
                id = %id,
                tx_id = %tx_id,
                "chain reports Unknown for a locally-known record; leaving \
                 unchanged (staleness detection is Stage 5.2)"
            );
            Ok(record)
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::HashMap;

    use omni_types::phase5::{InferenceCommitment, SnipV2ObjectId};

    use crate::error::ChainClientError;

    // ── Fake chain client ────────────────────────────────────────────────

    /// Stage 5.1: query path is keyed by `tx_id: String` (matching the
    /// chain RPC), not by `AttestationId`. The registry workflow
    /// extracts the tx_id from the local record's stored receipt and
    /// passes it to `query_attestation_status`.
    struct FakeChainClient {
        submit_outcome: RefCell<std::result::Result<SubmissionReceipt, ChainClientError>>,
        query_outcomes: RefCell<HashMap<String, AttestationStatus>>,
        default_query: RefCell<std::result::Result<AttestationStatus, ChainClientError>>,
        submit_calls: RefCell<Vec<InferenceAttestation>>,
        query_calls: RefCell<Vec<String>>,
    }

    impl FakeChainClient {
        fn ok_submit(tx_id: &str) -> Self {
            Self {
                submit_outcome: RefCell::new(Ok(SubmissionReceipt {
                    tx_id: tx_id.to_string(),
                    note: None,
                })),
                query_outcomes: RefCell::new(HashMap::new()),
                default_query: RefCell::new(Ok(AttestationStatus::Submitted)),
                submit_calls: RefCell::new(Vec::new()),
                query_calls: RefCell::new(Vec::new()),
            }
        }

        fn submit_err(msg: &str) -> Self {
            Self {
                submit_outcome: RefCell::new(Err(ChainClientError::Other(msg.into()))),
                query_outcomes: RefCell::new(HashMap::new()),
                default_query: RefCell::new(Ok(AttestationStatus::Submitted)),
                submit_calls: RefCell::new(Vec::new()),
                query_calls: RefCell::new(Vec::new()),
            }
        }

        fn set_query_for(&self, tx_id: String, status: AttestationStatus) {
            self.query_outcomes.borrow_mut().insert(tx_id, status);
        }

        fn set_default_query_err(&self, msg: &str) {
            *self.default_query.borrow_mut() = Err(ChainClientError::Other(msg.into()));
        }

        fn set_submit_outcome(
            &self,
            outcome: std::result::Result<SubmissionReceipt, ChainClientError>,
        ) {
            *self.submit_outcome.borrow_mut() = outcome;
        }

        fn submit_call_count(&self) -> usize {
            self.submit_calls.borrow().len()
        }
    }

    impl ChainClient for FakeChainClient {
        fn submit_attestation(
            &self,
            attestation: &InferenceAttestation,
        ) -> std::result::Result<SubmissionReceipt, ChainClientError> {
            self.submit_calls.borrow_mut().push(attestation.clone());
            self.submit_outcome.borrow().clone()
        }

        fn query_attestation_status(
            &self,
            tx_id: &str,
        ) -> std::result::Result<AttestationStatus, ChainClientError> {
            self.query_calls.borrow_mut().push(tx_id.to_string());
            if let Some(s) = self.query_outcomes.borrow().get(tx_id).cloned() {
                return Ok(s);
            }
            self.default_query.borrow().clone()
        }
    }

    // ── Fixtures ─────────────────────────────────────────────────────────

    fn snip_id(byte: u8) -> SnipV2ObjectId {
        let mut b = [0u8; 32];
        b.fill(byte);
        SnipV2ObjectId::from_bytes(b)
    }

    fn make_commitment(session: &str, response_byte: u8) -> InferenceCommitment {
        let mut response = String::with_capacity(64);
        for _ in 0..64 {
            use std::fmt::Write;
            let _ = write!(&mut response, "{:x}", response_byte & 0x0F);
        }
        InferenceCommitment {
            session_id: session.into(),
            model_hash: "a".repeat(64),
            manifest_snip_root: snip_id(0x11),
            response_hash: response,
            proof_snip_root: snip_id(0x22),
        }
    }

    fn make_attestation(
        session: &str,
        address: &str,
        signature: &str,
    ) -> InferenceAttestation {
        InferenceAttestation {
            commitment: make_commitment(session, 0xB),
            verifier_address: address.into(),
            verifier_signature: signature.into(),
        }
    }

    fn open_temp_registry() -> (tempfile::TempDir, AttestationRegistry) {
        let dir = tempfile::tempdir().unwrap();
        let reg = AttestationRegistry::open(dir.path().join("attestations")).unwrap();
        (dir, reg)
    }

    // ── ID derivation (4) ────────────────────────────────────────────────

    #[test]
    fn attestation_id_is_deterministic() {
        let a = make_attestation("sess-1", "addr-1", "sig-1");
        let id1 = compute_attestation_id(&a).unwrap();
        let id2 = compute_attestation_id(&a).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn attestation_id_changes_when_session_id_changes() {
        let a = make_attestation("sess-1", "addr-1", "sig-1");
        let b = make_attestation("sess-2", "addr-1", "sig-1");
        assert_ne!(
            compute_attestation_id(&a).unwrap(),
            compute_attestation_id(&b).unwrap(),
        );
    }

    #[test]
    fn attestation_id_changes_when_verifier_address_changes() {
        let a = make_attestation("sess-1", "addr-1", "sig-1");
        let b = make_attestation("sess-1", "addr-2", "sig-1");
        assert_ne!(
            compute_attestation_id(&a).unwrap(),
            compute_attestation_id(&b).unwrap(),
        );
    }

    /// Same (session_id, verifier_address) — different signature AND
    /// different commitment body — must still produce the SAME id.
    /// Pins the keying contract.
    #[test]
    fn attestation_id_unchanged_when_attestation_body_changes_same_key() {
        let mut a = make_attestation("sess-1", "addr-1", "sig-1");
        let mut b = make_attestation("sess-1", "addr-1", "sig-2");
        // Mutate commitment.response_hash so the bodies are clearly different.
        b.commitment.response_hash = "f".repeat(64);
        // Also mutate something else just to be thorough.
        b.commitment.model_hash = "c".repeat(64);
        // Sanity: the attestations themselves must differ.
        assert_ne!(a, b);
        // …but their ids do not.
        let id_a = compute_attestation_id(&a).unwrap();
        let id_b = compute_attestation_id(&b).unwrap();
        assert_eq!(id_a, id_b);
        // Mutate `a` too just to prevent the test from passing if both
        // accidentally went through the same code path.
        a.verifier_signature = "yet-another-sig".into();
        let id_a_again = compute_attestation_id(&a).unwrap();
        assert_eq!(id_a, id_a_again);
    }

    // ── Registry storage (5) ─────────────────────────────────────────────

    #[test]
    fn insert_creates_record_and_load_round_trip() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let inserted = reg.insert(att.clone()).unwrap();
        assert_eq!(inserted.status, LocalAttestationStatus::Pending);
        assert!(inserted.receipt.is_none());
        assert!(inserted.error_message.is_none());
        assert_eq!(inserted.attestation, att);

        let loaded = reg.load(&inserted.id).unwrap();
        assert_eq!(loaded, inserted);
    }

    #[test]
    fn insert_is_idempotent_for_byte_equal_attestation() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let first = reg.insert(att.clone()).unwrap();
        let file = reg.path_for(&first.id);
        let bytes_first = std::fs::read(&file).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let second = reg.insert(att.clone()).unwrap();
        let bytes_second = std::fs::read(&file).unwrap();
        assert_eq!(first, second);
        // File contents must be byte-identical — no rewrite happened.
        assert_eq!(bytes_first, bytes_second);
        assert_eq!(first.updated_at, second.updated_at);
    }

    #[test]
    fn insert_returns_conflict_for_same_key_different_attestation() {
        let (_dir, reg) = open_temp_registry();
        let a = make_attestation("sess-1", "addr-1", "sig-1");
        let mut b = make_attestation("sess-1", "addr-1", "sig-2");
        b.commitment.response_hash = "f".repeat(64);
        // a and b share the same (session_id, verifier_address) but differ
        // in body/signature, so they will compute the same AttestationId
        // but be byte-different.
        assert_eq!(
            compute_attestation_id(&a).unwrap(),
            compute_attestation_id(&b).unwrap()
        );
        assert_ne!(a, b);

        let first = reg.insert(a.clone()).unwrap();
        let id = first.id;
        let bytes_before = std::fs::read(reg.path_for(&id)).unwrap();

        let err = reg.insert(b).unwrap_err();
        match err {
            RegistryError::ConflictingAttestation { id: e_id } => assert_eq!(e_id, id),
            other => panic!("expected ConflictingAttestation, got {other:?}"),
        }

        // Existing record on disk is untouched.
        let bytes_after = std::fs::read(reg.path_for(&id)).unwrap();
        assert_eq!(bytes_before, bytes_after);
        let reloaded = reg.load(&id).unwrap();
        assert_eq!(reloaded.attestation, a);
    }

    #[test]
    fn load_missing_returns_record_not_found() {
        let (_dir, reg) = open_temp_registry();
        let phantom = AttestationId::from_bytes([0xEE; 32]);
        let err = reg.load(&phantom).unwrap_err();
        assert!(matches!(err, RegistryError::RecordNotFound(id) if id == phantom));
    }

    #[test]
    fn list_returns_all_records_in_stable_order() {
        let (_dir, reg) = open_temp_registry();
        let atts = vec![
            make_attestation("sess-c", "addr-1", "sig"),
            make_attestation("sess-a", "addr-1", "sig"),
            make_attestation("sess-b", "addr-1", "sig"),
        ];
        for a in &atts {
            reg.insert(a.clone()).unwrap();
        }

        let first = reg.list().unwrap();
        let second = reg.list().unwrap();
        assert_eq!(first.len(), 3);
        assert_eq!(first, second);

        // Sorted by id.to_hex().
        let hexes: Vec<String> = first.iter().map(|r| r.id.to_hex()).collect();
        let mut sorted = hexes.clone();
        sorted.sort();
        assert_eq!(hexes, sorted);
    }

    // ── Positive transitions (8) ─────────────────────────────────────────

    fn fresh_pending(reg: &AttestationRegistry) -> AttestationRecord {
        let att = make_attestation("sess-tx", "addr-1", "sig-1");
        reg.insert(att).unwrap()
    }

    fn dummy_receipt(tx: &str) -> SubmissionReceipt {
        SubmissionReceipt {
            tx_id: tx.into(),
            note: None,
        }
    }

    #[test]
    fn mark_submitted_from_pending() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        let updated = reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Submitted);
        assert_eq!(updated.receipt.as_ref().unwrap().tx_id, "tx-1");
        assert!(updated.updated_at >= r.updated_at);
    }

    #[test]
    fn mark_included_from_submitted() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        let updated = reg.mark_included(&r.id).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Included);
    }

    #[test]
    fn mark_finalized_from_submitted_skipping_included() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        let updated = reg.mark_finalized(&r.id).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Finalized);
    }

    #[test]
    fn mark_finalized_from_included() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        reg.mark_included(&r.id).unwrap();
        let updated = reg.mark_finalized(&r.id).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Finalized);
    }

    #[test]
    fn mark_failed_from_submitted() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        let updated = reg.mark_failed(&r.id, "boom".into()).unwrap();
        assert!(matches!(updated.status, LocalAttestationStatus::Failed { ref reason } if reason == "boom"));
        assert_eq!(updated.error_message.as_deref(), Some("boom"));
        // Receipt is preserved.
        assert_eq!(updated.receipt.as_ref().unwrap().tx_id, "tx-1");
    }

    #[test]
    fn mark_failed_from_included() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        reg.mark_included(&r.id).unwrap();
        let updated = reg.mark_failed(&r.id, "reverted".into()).unwrap();
        assert!(matches!(updated.status, LocalAttestationStatus::Failed { ref reason } if reason == "reverted"));
    }

    #[test]
    fn mark_dropped_from_submitted() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        let updated = reg.mark_dropped(&r.id, Some("evicted".into())).unwrap();
        assert!(
            matches!(updated.status, LocalAttestationStatus::Dropped { ref reason } if reason.as_deref() == Some("evicted"))
        );
        assert_eq!(updated.error_message.as_deref(), Some("evicted"));
    }

    #[test]
    fn mark_submitted_from_dropped_retry() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-old")).unwrap();
        reg.mark_dropped(&r.id, Some("dropped".into())).unwrap();
        let retried = reg.mark_submitted(&r.id, dummy_receipt("tx-new")).unwrap();
        assert_eq!(retried.status, LocalAttestationStatus::Submitted);
        // New receipt replaces the old one; error_message cleared.
        assert_eq!(retried.receipt.as_ref().unwrap().tx_id, "tx-new");
        assert!(retried.error_message.is_none());
    }

    // ── Negative transitions (5) ─────────────────────────────────────────

    #[test]
    fn mark_submitted_from_finalized_rejected() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        reg.mark_finalized(&r.id).unwrap();
        let err = reg.mark_submitted(&r.id, dummy_receipt("tx-2")).unwrap_err();
        assert!(matches!(
            err,
            RegistryError::InvalidStatusTransition { to: "submitted", .. }
        ));
    }

    #[test]
    fn mark_finalized_from_pending_rejected() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        let err = reg.mark_finalized(&r.id).unwrap_err();
        assert!(matches!(
            err,
            RegistryError::InvalidStatusTransition { to: "finalized", .. }
        ));
    }

    #[test]
    fn mark_finalized_from_failed_rejected() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        reg.mark_failed(&r.id, "boom".into()).unwrap();
        let err = reg.mark_finalized(&r.id).unwrap_err();
        assert!(matches!(
            err,
            RegistryError::InvalidStatusTransition { to: "finalized", .. }
        ));
    }

    /// Pins the corrected design point: chain-client RPC failures cannot
    /// transition a Pending record directly to Failed. Only an explicit
    /// chain status of `Failed { reason }` from `Submitted` or `Included`
    /// is allowed.
    #[test]
    fn mark_failed_from_pending_rejected() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        let err = reg.mark_failed(&r.id, "rpc gone".into()).unwrap_err();
        assert!(matches!(
            err,
            RegistryError::InvalidStatusTransition { to: "failed", .. }
        ));
    }

    #[test]
    fn mark_dropped_from_finalized_rejected() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();
        reg.mark_finalized(&r.id).unwrap();
        let err = reg.mark_dropped(&r.id, None).unwrap_err();
        assert!(matches!(
            err,
            RegistryError::InvalidStatusTransition { to: "dropped", .. }
        ));
    }

    // ── Submit workflow (4) ──────────────────────────────────────────────

    #[test]
    fn submit_workflow_records_receipt_on_chain_success() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let client = FakeChainClient::ok_submit("tx-1");
        let record = submit_attestation_workflow(&reg, &client, att).unwrap();
        assert_eq!(record.status, LocalAttestationStatus::Submitted);
        assert_eq!(record.receipt.as_ref().unwrap().tx_id, "tx-1");
        assert_eq!(client.submit_call_count(), 1);
    }

    /// Pins the corrected behaviour: a chain-client error during submit
    /// is NOT terminalised as `Failed`. The record stays `Pending` and
    /// the caller observes a typed `ChainClient` error.
    #[test]
    fn submit_workflow_returns_chain_error_and_leaves_record_pending() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let client = FakeChainClient::submit_err("rpc gone");
        let err = submit_attestation_workflow(&reg, &client, att.clone()).unwrap_err();
        match err {
            RegistryError::ChainClient(ChainClientError::Other(msg)) => {
                assert_eq!(msg, "rpc gone");
            }
            other => panic!("expected ChainClient(Other(...)), got {other:?}"),
        }
        // Record is still Pending — chain RPC errors do not terminalise.
        let id = compute_attestation_id(&att).unwrap();
        let loaded = reg.load(&id).unwrap();
        assert_eq!(loaded.status, LocalAttestationStatus::Pending);
        assert!(loaded.error_message.is_none());
    }

    #[test]
    fn submit_workflow_resubmits_dropped_records() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        // First successful submission, then dropped.
        let client = FakeChainClient::ok_submit("tx-old");
        let first = submit_attestation_workflow(&reg, &client, att.clone()).unwrap();
        reg.mark_dropped(&first.id, Some("evicted".into())).unwrap();
        // Reconfigure client to return a new receipt.
        client.set_submit_outcome(Ok(SubmissionReceipt {
            tx_id: "tx-new".into(),
            note: None,
        }));
        let retried = submit_attestation_workflow(&reg, &client, att).unwrap();
        assert_eq!(retried.status, LocalAttestationStatus::Submitted);
        assert_eq!(retried.receipt.as_ref().unwrap().tx_id, "tx-new");
        assert_eq!(client.submit_call_count(), 2);
    }

    #[test]
    fn submit_workflow_is_noop_for_already_submitted_records() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let client = FakeChainClient::ok_submit("tx-1");
        let first = submit_attestation_workflow(&reg, &client, att.clone()).unwrap();
        // Second call: record is Submitted, should not re-submit.
        let second = submit_attestation_workflow(&reg, &client, att).unwrap();
        assert_eq!(first.id, second.id);
        assert_eq!(second.status, LocalAttestationStatus::Submitted);
        assert_eq!(client.submit_call_count(), 1);
    }

    // ── Query workflow (8) — Stage 5.1: keyed by receipt.tx_id ──────────

    fn setup_submitted(reg: &AttestationRegistry) -> AttestationRecord {
        let r = fresh_pending(reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-x")).unwrap()
    }

    /// Helper: pull the tx_id out of a record that was just `mark_submitted`.
    /// Panics if the record has no receipt (the helper is only used in the
    /// happy-path tests).
    fn receipt_tx_id(r: &AttestationRecord) -> String {
        r.receipt.as_ref().expect("setup record must have a receipt").tx_id.clone()
    }

    #[test]
    fn query_workflow_marks_included_on_chain_included() {
        let (_dir, reg) = open_temp_registry();
        let r = setup_submitted(&reg);
        let tx_id = receipt_tx_id(&r);
        let client = FakeChainClient::ok_submit("unused");
        client.set_query_for(tx_id.clone(), AttestationStatus::Included);
        let updated = query_attestation_workflow(&reg, &client, &r.id).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Included);
        // Stage 5.1: chain is queried by tx_id, not by AttestationId.
        assert_eq!(client.query_calls.borrow().as_slice(), &[tx_id]);
    }

    #[test]
    fn query_workflow_marks_finalized_on_chain_finalized() {
        let (_dir, reg) = open_temp_registry();
        let r = setup_submitted(&reg);
        let tx_id = receipt_tx_id(&r);
        let client = FakeChainClient::ok_submit("unused");
        client.set_query_for(tx_id.clone(), AttestationStatus::Finalized);
        let updated = query_attestation_workflow(&reg, &client, &r.id).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Finalized);
        assert_eq!(client.query_calls.borrow().as_slice(), &[tx_id]);
    }

    #[test]
    fn query_workflow_marks_failed_on_chain_failed_with_reason() {
        let (_dir, reg) = open_temp_registry();
        let r = setup_submitted(&reg);
        let tx_id = receipt_tx_id(&r);
        let client = FakeChainClient::ok_submit("unused");
        client.set_query_for(
            tx_id.clone(),
            AttestationStatus::Failed {
                reason: "execution reverted".into(),
            },
        );
        let updated = query_attestation_workflow(&reg, &client, &r.id).unwrap();
        assert!(matches!(updated.status, LocalAttestationStatus::Failed { ref reason } if reason == "execution reverted"));
        assert_eq!(updated.error_message.as_deref(), Some("execution reverted"));
        assert_eq!(client.query_calls.borrow().as_slice(), &[tx_id]);
    }

    /// Pins the Stage 5 behaviour: a chain-client error during query does
    /// NOT alter the local record. Caller observes a typed `ChainClient`
    /// error. Stage 5.1 also pins that the workflow reached the chain call
    /// after extracting tx_id (so the failure is on the chain side, not in
    /// the receipt-lookup defense).
    #[test]
    fn query_workflow_returns_chain_error_and_leaves_record_unchanged() {
        let (_dir, reg) = open_temp_registry();
        let r = setup_submitted(&reg);
        let tx_id = receipt_tx_id(&r);
        let client = FakeChainClient::ok_submit("unused");
        client.set_default_query_err("rpc gone");
        let err = query_attestation_workflow(&reg, &client, &r.id).unwrap_err();
        match err {
            RegistryError::ChainClient(ChainClientError::Other(msg)) => {
                assert_eq!(msg, "rpc gone");
            }
            other => panic!("expected ChainClient(Other(...)), got {other:?}"),
        }
        let reloaded = reg.load(&r.id).unwrap();
        assert_eq!(reloaded.status, LocalAttestationStatus::Submitted);
        // The chain WAS reached — proves the receipt was extracted before
        // the RPC failure surfaced.
        assert_eq!(client.query_calls.borrow().as_slice(), &[tx_id]);
    }

    // ── Stage 5.1 additions: Unknown + tx_id-keying + missing-receipt ────

    /// Chain returns `Unknown` for a `Submitted` record. Stage 5.1
    /// contract: leave the record unchanged. Do not mark `Failed`. Do not
    /// auto-mark `Dropped` (staleness detection is Stage 5.2). Also pins
    /// that the chain was queried by the receipt's tx_id.
    #[test]
    fn query_workflow_leaves_submitted_record_unchanged_on_chain_unknown() {
        let (_dir, reg) = open_temp_registry();
        let r = setup_submitted(&reg);
        let tx_id = receipt_tx_id(&r);
        let client = FakeChainClient::ok_submit("unused");
        client.set_query_for(tx_id.clone(), AttestationStatus::Unknown);

        let returned = query_attestation_workflow(&reg, &client, &r.id).unwrap();
        assert_eq!(returned.status, LocalAttestationStatus::Submitted);
        assert!(returned.error_message.is_none());

        let reloaded = reg.load(&r.id).unwrap();
        assert_eq!(reloaded.status, LocalAttestationStatus::Submitted);
        assert_eq!(reloaded.receipt.as_ref().unwrap().tx_id, tx_id);
        assert_eq!(client.query_calls.borrow().as_slice(), &[tx_id]);
    }

    /// Same contract from the `Included` source state — Stage 5.1 covers
    /// the rare chain reorg case (chain forgot a previously-Included tx)
    /// by leaving the record alone and surfacing via tracing.
    #[test]
    fn query_workflow_leaves_included_record_unchanged_on_chain_unknown() {
        let (_dir, reg) = open_temp_registry();
        let r = setup_submitted(&reg);
        let tx_id = receipt_tx_id(&r);
        reg.mark_included(&r.id).unwrap();
        let client = FakeChainClient::ok_submit("unused");
        client.set_query_for(tx_id.clone(), AttestationStatus::Unknown);

        let returned = query_attestation_workflow(&reg, &client, &r.id).unwrap();
        assert_eq!(returned.status, LocalAttestationStatus::Included);

        let reloaded = reg.load(&r.id).unwrap();
        assert_eq!(reloaded.status, LocalAttestationStatus::Included);
        assert_eq!(client.query_calls.borrow().as_slice(), &[tx_id]);
    }

    /// Pins the Stage 5.1 contract: the chain is queried by the
    /// receipt's tx_id, not by `AttestationId` or any encoding of it.
    /// Failing this test indicates the workflow regressed to keying the
    /// chain by registry-local identifiers.
    #[test]
    fn query_workflow_queries_by_receipt_tx_id() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-x")).unwrap();

        let client = FakeChainClient::ok_submit("unused");
        client.set_query_for("tx-x".to_string(), AttestationStatus::Finalized);

        let updated = query_attestation_workflow(&reg, &client, &r.id).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Finalized);
        assert_eq!(client.query_calls.borrow().as_slice(), &["tx-x".to_string()]);
        // Negative cross-check: the hex of AttestationId is NOT what was
        // sent to the chain.
        assert_ne!(client.query_calls.borrow()[0], r.id.to_hex());
    }

    /// Defensive integrity test: a queryable record (`Submitted` or
    /// `Included`) with `receipt: None` is corrupt — `mark_submitted`
    /// always sets the receipt, so this state can only arise from
    /// hand-edited / corrupted JSON in the registry directory. The
    /// workflow surfaces a typed
    /// `RegistryError::SubmittedRecordMissingReceipt` rather than silently
    /// returning the record unchanged. The chain is NEVER reached.
    #[test]
    fn query_workflow_errors_when_submitted_record_has_no_receipt() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-orig")).unwrap();

        // Hand-corrupt the on-disk record: status stays `Submitted` but
        // receipt is cleared. Bypass the public API (which always sets
        // the receipt) by re-serializing a mutated copy directly.
        let mut record = reg.load(&r.id).unwrap();
        record.receipt = None;
        let bytes = serde_json::to_vec_pretty(&record).unwrap();
        std::fs::write(reg.path_for(&r.id), bytes).unwrap();

        let client = FakeChainClient::ok_submit("unused");
        let err = query_attestation_workflow(&reg, &client, &r.id).unwrap_err();
        match err {
            RegistryError::SubmittedRecordMissingReceipt { id } => {
                assert_eq!(id, r.id);
            }
            other => panic!("expected SubmittedRecordMissingReceipt, got {other:?}"),
        }
        // Defense fires BEFORE the chain is reached.
        assert!(client.query_calls.borrow().is_empty());
    }

    // ── Atomic writes (1) ────────────────────────────────────────────────

    #[test]
    fn atomic_write_leaves_no_tmp_on_success() {
        let (_dir, reg) = open_temp_registry();
        let r = fresh_pending(&reg);
        reg.mark_submitted(&r.id, dummy_receipt("tx-1")).unwrap();

        let mut json_count = 0;
        let mut tmp_count = 0;
        for entry in std::fs::read_dir(reg.root()).unwrap() {
            let path = entry.unwrap().path();
            match path.extension().and_then(|s| s.to_str()) {
                Some("json") => json_count += 1,
                Some("tmp") => tmp_count += 1,
                _ => {}
            }
        }
        assert_eq!(json_count, 1);
        assert_eq!(tmp_count, 0);
    }
}
