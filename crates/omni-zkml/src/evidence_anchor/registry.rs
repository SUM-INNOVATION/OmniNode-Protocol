//! Phase 5 Stage 13.0 — local anchor registry.
//!
//! Filesystem-backed store of submitted anchor records keyed by
//! the artifact's BLAKE3 hash. One JSON file per record at
//! `<root>/<artifact_hash_hex>.json` plus a `<root>/tx_index.json`
//! mapping `tx_id → artifact_hash_hex` for `--tx-id` lookups.
//!
//! The registry is **distinct** from the Stage 12.7 contributor
//! `--state-dir` and from the Stage 5
//! [`crate::registry::AttestationRegistry`] root — the CLI's
//! `--anchor-registry-dir` flag is the only way to address it,
//! and the directory name makes it unambiguous.
//!
//! Atomic temp+rename writes per the project-wide Stage 12
//! pattern. The directory is created on first submit if missing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::evidence_anchor::client::{AnchorStatus, AnchorSubmissionReceipt};
use crate::evidence_anchor::wire::{IntegrityEvidenceAnchorTxData, anchor_hex_lower};

// ── Status model (local) ──────────────────────────────────────────────────────

/// Local-side anchor lifecycle. Mirrors the Stage 5
/// [`crate::registry::LocalAttestationStatus`] shape but drops
/// `Pending` (Stage 13.0 only persists records that have already
/// reached chain submission) and `Dropped` (Stage 13.0 has no
/// staleness sweep).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalAnchorStatus {
    Submitted,
    Included,
    Finalized,
    Failed { reason: String },
}

impl LocalAnchorStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            LocalAnchorStatus::Submitted => "submitted",
            LocalAnchorStatus::Included => "included",
            LocalAnchorStatus::Finalized => "finalized",
            LocalAnchorStatus::Failed { .. } => "failed",
        }
    }
}

/// Lifted from [`AnchorStatus`] when the orchestration layer
/// applies a chain-returned status to a local record.
pub fn local_status_from_chain(status: &AnchorStatus) -> Option<LocalAnchorStatus> {
    match status {
        AnchorStatus::Submitted => Some(LocalAnchorStatus::Submitted),
        AnchorStatus::Included => Some(LocalAnchorStatus::Included),
        AnchorStatus::Finalized => Some(LocalAnchorStatus::Finalized),
        AnchorStatus::Failed { reason } => Some(LocalAnchorStatus::Failed {
            reason: reason.clone(),
        }),
        // Stage 13.0 leaves the record unchanged on chain
        // `Unknown`. See `query_anchor_workflow` for the
        // observation-only treatment.
        AnchorStatus::Unknown => None,
    }
}

// ── Record + tx index ─────────────────────────────────────────────────────────

/// Persisted anchor record. One per `artifact_hash_hex`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorRecord {
    /// Lowercase 64-char hex of `tx_data.digest.artifact_hash`.
    /// Doubles as the filename stem under the registry root.
    pub artifact_hash_hex: String,

    /// Lowercase 64-char hex of `tx_data.digest.signer_pubkey`.
    /// Convenience denormalisation for `event=...` and pretty
    /// output; `tx_data` carries the canonical bytes.
    pub signer_pubkey_hex: String,

    /// Full signed anchor wire payload, persisted so the verify
    /// command can re-check `submitter_signature` against the
    /// stored canonical bytes without trusting registry-side
    /// metadata.
    pub tx_data: IntegrityEvidenceAnchorTxData,

    pub receipt: AnchorSubmissionReceipt,
    pub status: LocalAnchorStatus,
    pub submitted_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// `tx_id → artifact_hash_hex` index, persisted as
/// `<root>/tx_index.json`. Used by the registry-backed verify /
/// query commands when the operator supplies `--tx-id` instead
/// of `--artifact-hash-hex`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TxIndex {
    by_tx_id: BTreeMap<String, String>,
}

// ── Registry ──────────────────────────────────────────────────────────────────

const TX_INDEX_FILENAME: &str = "tx_index.json";
const RECORD_EXTENSION: &str = "json";

/// Local registry of submitted anchor records. Atomic
/// temp+rename writes for both per-record JSON files and the
/// `tx_index.json` index.
pub struct LocalEvidenceAnchorRegistry {
    root: PathBuf,
}

impl LocalEvidenceAnchorRegistry {
    /// Open or create a registry rooted at `root`. The
    /// directory is created if it does not exist.
    pub fn open(root: PathBuf) -> Result<Self, std::io::Error> {
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn record_path(&self, artifact_hash_hex: &str) -> PathBuf {
        self.root
            .join(format!("{artifact_hash_hex}.{RECORD_EXTENSION}"))
    }

    fn tx_index_path(&self) -> PathBuf {
        self.root.join(TX_INDEX_FILENAME)
    }

    fn load_tx_index(&self) -> Result<TxIndex, std::io::Error> {
        let path = self.tx_index_path();
        if !path.is_file() {
            return Ok(TxIndex::default());
        }
        let bytes = std::fs::read(&path)?;
        serde_json::from_slice(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("malformed tx_index.json at {}: {e}", path.display()),
            )
        })
    }

    fn write_tx_index(&self, index: &TxIndex) -> Result<(), std::io::Error> {
        let path = self.tx_index_path();
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(index)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        if let Err(e) = std::fs::write(&tmp, &bytes) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }

    fn write_record_atomic(&self, record: &AnchorRecord) -> Result<(), std::io::Error> {
        let path = self.record_path(&record.artifact_hash_hex);
        let tmp = path.with_extension(format!("{RECORD_EXTENSION}.tmp"));
        let bytes = serde_json::to_vec_pretty(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        if let Err(e) = std::fs::write(&tmp, &bytes) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }

    /// Persist a new anchor record. If a record already exists
    /// under the same `artifact_hash_hex`, returns it unchanged
    /// (idempotent — no file rewrite, no tx-index churn).
    /// Otherwise writes the record + updates the tx-index.
    pub fn insert(
        &self,
        tx_data: IntegrityEvidenceAnchorTxData,
        receipt: AnchorSubmissionReceipt,
    ) -> Result<AnchorRecord, std::io::Error> {
        let artifact_hash_hex = anchor_hex_lower(&tx_data.digest.artifact_hash);
        if let Some(existing) = self.load_by_artifact_hash(&artifact_hash_hex)? {
            return Ok(existing);
        }
        let now = Utc::now();
        let record = AnchorRecord {
            artifact_hash_hex: artifact_hash_hex.clone(),
            signer_pubkey_hex: anchor_hex_lower(&tx_data.digest.signer_pubkey),
            tx_data,
            receipt: receipt.clone(),
            status: LocalAnchorStatus::Submitted,
            submitted_at: now,
            updated_at: now,
        };
        self.write_record_atomic(&record)?;
        let mut index = self.load_tx_index()?;
        index
            .by_tx_id
            .insert(receipt.tx_id.clone(), artifact_hash_hex);
        self.write_tx_index(&index)?;
        Ok(record)
    }

    /// Look up a record by lowercase-hex `artifact_hash_hex`.
    /// Returns `Ok(None)` on registry miss; bubbles FS / parse
    /// errors as `Err`.
    pub fn load_by_artifact_hash(
        &self,
        artifact_hash_hex: &str,
    ) -> Result<Option<AnchorRecord>, std::io::Error> {
        let path = self.record_path(artifact_hash_hex);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)?;
        let record: AnchorRecord = serde_json::from_slice(&bytes).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("malformed record at {}: {e}", path.display()),
            )
        })?;
        Ok(Some(record))
    }

    /// Look up a record by stored `tx_id`. Returns `Ok(None)`
    /// when no `tx_index.json` entry maps the `tx_id`; bubbles
    /// FS / parse errors as `Err`.
    pub fn load_by_tx_id(&self, tx_id: &str) -> Result<Option<AnchorRecord>, std::io::Error> {
        let index = self.load_tx_index()?;
        let Some(artifact_hash_hex) = index.by_tx_id.get(tx_id) else {
            return Ok(None);
        };
        self.load_by_artifact_hash(artifact_hash_hex)
    }

    /// Stage 13.2 — iterate every persisted record. Non-record
    /// entries (`.tmp`, `tx_index.json`, anything that isn't a
    /// 64-hex `*.json`) are skipped silently. A parse failure
    /// on a record file surfaces as
    /// `io::Error(InvalidData, …)`. Records are returned sorted
    /// by `artifact_hash_hex` ascending for deterministic
    /// reconcile-sweep ordering.
    ///
    /// Consumers (the reconcile workflow) iterate this and
    /// query the chain per record; per-record RPC failures do
    /// not abort the sweep.
    pub fn list(&self) -> Result<Vec<AnchorRecord>, std::io::Error> {
        let mut records = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            let Some(stem) = path
                .file_stem()
                .and_then(|s| s.to_str())
            else {
                continue;
            };
            // Only consider `<64-hex>.json` files; ignore
            // `tx_index.json` (stem length 8) and any stray
            // `.tmp` / other extensions.
            if path.extension().and_then(|s| s.to_str()) != Some(RECORD_EXTENSION) {
                continue;
            }
            if stem.len() != 64 {
                continue;
            }
            let bytes = std::fs::read(&path)?;
            let record: AnchorRecord = serde_json::from_slice(&bytes).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("malformed record at {}: {e}", path.display()),
                )
            })?;
            records.push(record);
        }
        records.sort_by(|a, b| a.artifact_hash_hex.cmp(&b.artifact_hash_hex));
        Ok(records)
    }

    /// Apply a chain-returned local status transition; clears
    /// `error_message` semantics is unnecessary because
    /// `LocalAnchorStatus::Failed` carries the reason in-band.
    pub fn update_status(
        &self,
        artifact_hash_hex: &str,
        new_status: LocalAnchorStatus,
    ) -> Result<AnchorRecord, std::io::Error> {
        let mut record = self
            .load_by_artifact_hash(artifact_hash_hex)?
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("anchor record not found for hash {artifact_hash_hex}"),
                )
            })?;
        record.status = new_status;
        record.updated_at = Utc::now();
        self.write_record_atomic(&record)?;
        Ok(record)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_anchor::wire::{
        AnchoredArtifactKind, INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
        IntegrityEvidenceAnchorDigest, anchor_signer_pubkey_bytes, sign_anchor_digest,
    };

    fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry) {
        let dir = tempfile::tempdir().unwrap();
        let reg = LocalEvidenceAnchorRegistry::open(dir.path().join("anchors")).unwrap();
        (dir, reg)
    }

    fn build_tx(seed: [u8; 32], hash_byte: u8) -> IntegrityEvidenceAnchorTxData {
        let pubkey = anchor_signer_pubkey_bytes(&seed).unwrap();
        let digest = IntegrityEvidenceAnchorDigest {
            anchor_schema_version: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
            artifact_kind: AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
            artifact_schema_version: 1,
            artifact_hash: [hash_byte; 32],
            signer_pubkey: pubkey,
            signed_at_utc_unix: 1_700_000_000,
        };
        let sig = sign_anchor_digest(&seed, &digest).unwrap();
        IntegrityEvidenceAnchorTxData {
            digest,
            submitter_signature: sig,
        }
    }

    fn dummy_receipt(tx_id: &str) -> AnchorSubmissionReceipt {
        AnchorSubmissionReceipt {
            tx_id: tx_id.to_string(),
            note: None,
        }
    }

    #[test]
    fn insert_persists_record_and_loads_by_hash_and_tx_id() {
        let (_dir, reg) = fresh_registry();
        let tx = build_tx([7u8; 32], 0x11);
        let receipt = dummy_receipt("anchor-1");
        let inserted = reg.insert(tx.clone(), receipt.clone()).unwrap();
        assert_eq!(inserted.artifact_hash_hex, "11".repeat(32));
        assert_eq!(inserted.status, LocalAnchorStatus::Submitted);

        let by_hash = reg
            .load_by_artifact_hash(&inserted.artifact_hash_hex)
            .unwrap()
            .expect("record present");
        assert_eq!(by_hash, inserted);

        let by_tx = reg.load_by_tx_id("anchor-1").unwrap().expect("indexed");
        assert_eq!(by_tx, inserted);
    }

    #[test]
    fn insert_is_idempotent_for_same_artifact_hash() {
        let (_dir, reg) = fresh_registry();
        let tx = build_tx([7u8; 32], 0x11);
        let r1 = reg.insert(tx.clone(), dummy_receipt("anchor-a")).unwrap();
        let r2 = reg.insert(tx, dummy_receipt("anchor-b")).unwrap();
        assert_eq!(
            r1, r2,
            "second insert must return the existing record unchanged"
        );
        // tx_index should not have anchor-b → conflict; only
        // anchor-a is recorded.
        assert_eq!(reg.load_by_tx_id("anchor-a").unwrap().unwrap(), r1);
        assert!(reg.load_by_tx_id("anchor-b").unwrap().is_none());
    }

    #[test]
    fn load_by_artifact_hash_misses_return_none() {
        let (_dir, reg) = fresh_registry();
        assert!(
            reg.load_by_artifact_hash(&"00".repeat(32))
                .unwrap()
                .is_none()
        );
        assert!(reg.load_by_tx_id("anchor-nonexistent").unwrap().is_none());
    }

    #[test]
    fn update_status_persists_transition() {
        let (_dir, reg) = fresh_registry();
        let tx = build_tx([7u8; 32], 0x11);
        let inserted = reg.insert(tx, dummy_receipt("anchor-x")).unwrap();
        let updated = reg
            .update_status(&inserted.artifact_hash_hex, LocalAnchorStatus::Finalized)
            .unwrap();
        assert_eq!(updated.status, LocalAnchorStatus::Finalized);
        let reloaded = reg
            .load_by_artifact_hash(&inserted.artifact_hash_hex)
            .unwrap()
            .unwrap();
        assert_eq!(reloaded.status, LocalAnchorStatus::Finalized);
        assert!(reloaded.updated_at >= inserted.updated_at);
    }

    #[test]
    fn atomic_writes_leave_no_tmp_files_on_success() {
        let (_dir, reg) = fresh_registry();
        let tx = build_tx([7u8; 32], 0x11);
        reg.insert(tx, dummy_receipt("anchor-1")).unwrap();
        for entry in std::fs::read_dir(reg.root()).unwrap() {
            let path = entry.unwrap().path();
            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            assert_ne!(
                ext, "tmp",
                "no .tmp files must remain after a successful write"
            );
        }
    }
}
