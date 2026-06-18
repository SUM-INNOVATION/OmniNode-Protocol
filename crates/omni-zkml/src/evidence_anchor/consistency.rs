//! Phase 5 Stage 13.8 — local integrity-evidence-anchor
//! consistency report.
//!
//! Strictly local, strictly read-only. Inspects the hot anchor
//! registry plus optional Stage 13.5 exports and Stage 13.7
//! archives in a single sweep and returns a typed report.
//!
//! ## Locked Stage 13.8 invariants
//!
//! - **Local-only.** No SUM Chain RPCs, no `omni-sumchain`
//!   types, no private chain repo dep.
//! - **Read-only.** Library uses only `std::fs::read` +
//!   `std::fs::read_dir`. No mutation helper from Stage 13.4 /
//!   13.6 / 13.7 is called.
//! - **No new `reason=` tags.** Findings are typed report data
//!   ON THE REPORT, not refusal taxonomy. The error.rs file is
//!   not modified.
//! - **Severity principle (Stage 13.8 v2 lock):**
//!   - Optional path cannot be opened OR no manifest found at
//!     the expected location → **`warning`**.
//!   - Optional path is readable AND claims to be an archive /
//!     export but fails integrity / schema checks → **`error`**.
//! - **Cross-surface byte-equal duplicates are summary counters,
//!   not per-item findings.** A 10k-record export overlapping
//!   with the hot registry must not emit 10k info findings.
//!   Overlap counts use unique `(surface, artifact_hash_hex)`
//!   set semantics so duplicate manifest entries don't inflate
//!   summary counters.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{EvidenceAnchorError, EvidenceAnchorResult};
use crate::evidence_anchor::archive::{
    AnchorArchiveManifest, ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION,
};
use crate::evidence_anchor::export::{verify_anchor_export, AnchorExportVerifyOptions};
use crate::evidence_anchor::registry::AnchorRecord;
use crate::evidence_anchor::wire::verify_anchor_tx_data;

// ── Schema constant ───────────────────────────────────────────────────────────

pub const ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION: u32 = 1;

const TX_INDEX_FILENAME: &str = "tx_index.json";
const ARCHIVE_MANIFEST_FILENAME: &str = "archive_manifest.json";

// ── Closed-set enums ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorConsistencySeverity {
    Info,
    Warning,
    Error,
}

impl AnchorConsistencySeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            AnchorConsistencySeverity::Info => "info",
            AnchorConsistencySeverity::Warning => "warning",
            AnchorConsistencySeverity::Error => "error",
        }
    }
}

/// Closed taxonomy of Stage 13.8 finding kinds. 24 variants
/// after v2 review: HotAndExportDuplicate +
/// ArchiveAndExportDuplicate dropped (those overlaps live in
/// summary counters); ArchiveDirNoManifest added to separate
/// "no manifest found" from "directory unreadable" (Finding 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorConsistencyFindingKind {
    // Hot registry (9)
    HotRecordMalformed,
    HotRecordSignatureInvalid,
    HotRecordSchemaUnsupported,
    HotFilenameHashMismatch,
    HotTxIndexOrphan,
    HotTxIndexMismatch,
    HotTxIdDuplicate,
    HotStaleOpenRecord,
    HotTmpOrphan,
    // Archive (13)
    ArchiveDirUnreadable,
    ArchiveDirNoManifest,
    ArchiveManifestMalformed,
    ArchiveManifestSchemaUnsupported,
    ArchiveEntryInvalidPath,
    ArchiveEntryMissingFile,
    ArchiveEntryBlake3Mismatch,
    ArchiveEntryRecordMalformed,
    ArchiveEntrySignatureInvalid,
    ArchiveEntryMetadataMismatch,
    ArchiveHotCollisionSameBytes,
    ArchiveHotCollisionDifferentBytes,
    ArchiveTxIdCollision,
    // Export (2)
    ExportDirUnreadable,
    ExportVerifyFailed,
}

impl AnchorConsistencyFindingKind {
    pub fn as_str(self) -> &'static str {
        use AnchorConsistencyFindingKind::*;
        match self {
            HotRecordMalformed => "hot_record_malformed",
            HotRecordSignatureInvalid => "hot_record_signature_invalid",
            HotRecordSchemaUnsupported => "hot_record_schema_unsupported",
            HotFilenameHashMismatch => "hot_filename_hash_mismatch",
            HotTxIndexOrphan => "hot_tx_index_orphan",
            HotTxIndexMismatch => "hot_tx_index_mismatch",
            HotTxIdDuplicate => "hot_tx_id_duplicate",
            HotStaleOpenRecord => "hot_stale_open_record",
            HotTmpOrphan => "hot_tmp_orphan",
            ArchiveDirUnreadable => "archive_dir_unreadable",
            ArchiveDirNoManifest => "archive_dir_no_manifest",
            ArchiveManifestMalformed => "archive_manifest_malformed",
            ArchiveManifestSchemaUnsupported => "archive_manifest_schema_unsupported",
            ArchiveEntryInvalidPath => "archive_entry_invalid_path",
            ArchiveEntryMissingFile => "archive_entry_missing_file",
            ArchiveEntryBlake3Mismatch => "archive_entry_blake3_mismatch",
            ArchiveEntryRecordMalformed => "archive_entry_record_malformed",
            ArchiveEntrySignatureInvalid => "archive_entry_signature_invalid",
            ArchiveEntryMetadataMismatch => "archive_entry_metadata_mismatch",
            ArchiveHotCollisionSameBytes => "archive_hot_collision_same_bytes",
            ArchiveHotCollisionDifferentBytes => "archive_hot_collision_different_bytes",
            ArchiveTxIdCollision => "archive_tx_id_collision",
            ExportDirUnreadable => "export_dir_unreadable",
            ExportVerifyFailed => "export_verify_failed",
        }
    }
}

// ── Report / summary / finding ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorConsistencyFinding {
    pub severity: AnchorConsistencySeverity,
    pub kind: AnchorConsistencyFindingKind,
    pub location: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_hash_hex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tx_id: Option<String>,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_action: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorConsistencySummary {
    pub hot_total: u64,
    pub hot_submitted: u64,
    pub hot_included: u64,
    pub hot_finalized: u64,
    pub hot_failed: u64,
    pub hot_malformed_records: u64,
    pub tx_index_entries: u64,
    pub tx_index_orphans: u64,
    pub archive_manifests_scanned: u64,
    pub archive_entries_scanned: u64,
    pub export_manifests_scanned: u64,
    pub export_entries_scanned: u64,
    /// Stage 13.8 v2 lock — unique `(hot, artifact_hash_hex)` ∩
    /// `(export, artifact_hash_hex)` overlap count. Duplicate
    /// manifest entries within a single export do NOT inflate
    /// this counter (set semantics).
    pub hot_export_overlaps: u64,
    /// Stage 13.8 v2 lock — unique `(archive, artifact_hash_hex)`
    /// ∩ `(export, artifact_hash_hex)` overlap count. Set
    /// semantics; aggregated across all archive surfaces.
    pub archive_export_overlaps: u64,
    pub findings_by_severity_info: u64,
    pub findings_by_severity_warning: u64,
    pub findings_by_severity_error: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorConsistencyReport {
    pub schema_version: u32,
    pub created_at_utc: String,
    /// Forensic record — local; not portable.
    pub anchor_registry_dir: String,
    pub archive_dirs: Vec<String>,
    pub export_dirs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stale_threshold_secs: Option<u64>,
    pub summary: AnchorConsistencySummary,
    pub findings: Vec<AnchorConsistencyFinding>,
}

// ── Options ──────────────────────────────────────────────────────────────────

pub struct AnchorConsistencyOptions<'a> {
    pub anchor_registry_dir: &'a Path,
    pub archive_dirs: &'a [PathBuf],
    pub export_dirs: &'a [PathBuf],
    pub stale_threshold_secs: Option<u64>,
    /// RFC 3339; tests inject for determinism.
    pub now_utc: &'a str,
}

// ── Build entry point ────────────────────────────────────────────────────────

/// Read-only multi-surface scan.
///
/// Returns `Ok(...)` whenever the REQUIRED `anchor_registry_dir`
/// is openable. Optional archive / export paths that can't be
/// opened or that don't carry a manifest become `warning`-level
/// findings; the report continues with the rest.
///
/// Only the case where `anchor_registry_dir` itself is unreadable
/// bubbles up as [`EvidenceAnchorError::Io`] (existing tag).
pub fn build_anchor_consistency_report(
    opts: &AnchorConsistencyOptions<'_>,
) -> EvidenceAnchorResult<AnchorConsistencyReport> {
    // ── Required: open the hot registry directory. ─────────
    if !opts.anchor_registry_dir.is_dir() {
        return Err(EvidenceAnchorError::Io {
            path: opts.anchor_registry_dir.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "--anchor-registry-dir {} is not a directory",
                    opts.anchor_registry_dir.display()
                ),
            ),
        });
    }

    let mut findings: Vec<AnchorConsistencyFinding> = Vec::new();
    let mut summary = AnchorConsistencySummary::zero();

    // Hot scan + collection of unique hashes/tx_ids for cross-
    // surface set semantics.
    let hot_scan = scan_hot_registry(opts.anchor_registry_dir, &mut findings, &mut summary)?;

    // Stale detection (only when threshold supplied).
    if let Some(threshold) = opts.stale_threshold_secs {
        let now = parse_rfc3339(opts.now_utc)?;
        for record in &hot_scan.well_formed_records {
            let stale = matches!(
                record.status,
                crate::evidence_anchor::registry::LocalAnchorStatus::Submitted
                    | crate::evidence_anchor::registry::LocalAnchorStatus::Included
            );
            if !stale {
                continue;
            }
            // Skip future-dated records (clock skew).
            if record.submitted_at > now {
                continue;
            }
            let age_secs = (now - record.submitted_at).num_seconds().max(0) as u64;
            if age_secs >= threshold {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Warning,
                    kind: AnchorConsistencyFindingKind::HotStaleOpenRecord,
                    location: format!("hot/{}.json", record.artifact_hash_hex),
                    artifact_hash_hex: Some(record.artifact_hash_hex.clone()),
                    tx_id: Some(record.receipt.tx_id.clone()),
                    detail: format!(
                        "{} record age {}s exceeds threshold {}s",
                        record.status.as_str(),
                        age_secs,
                        threshold
                    ),
                    suggested_action: Some(
                        "investigate via Stage 13.3 reconcile or Stage 13.4 cleanup"
                            .to_string(),
                    ),
                });
            }
        }
    }

    // Archive scans.
    let mut archive_hash_union: BTreeSet<String> = BTreeSet::new();
    for archive_dir in opts.archive_dirs {
        scan_archive_input(
            archive_dir,
            &hot_scan,
            &mut findings,
            &mut summary,
            &mut archive_hash_union,
        );
    }

    // Export scans.
    let mut export_hash_union: BTreeSet<String> = BTreeSet::new();
    for export_dir in opts.export_dirs {
        scan_export_input(
            export_dir,
            &mut findings,
            &mut summary,
            &mut export_hash_union,
        );
    }

    // Cross-surface overlap counters (set semantics — duplicate
    // manifest entries don't inflate).
    summary.hot_export_overlaps = hot_scan
        .hot_hash_union
        .intersection(&export_hash_union)
        .count() as u64;
    summary.archive_export_overlaps = archive_hash_union
        .intersection(&export_hash_union)
        .count() as u64;

    // findings_by_severity rollup.
    for finding in &findings {
        match finding.severity {
            AnchorConsistencySeverity::Info => summary.findings_by_severity_info += 1,
            AnchorConsistencySeverity::Warning => {
                summary.findings_by_severity_warning += 1
            }
            AnchorConsistencySeverity::Error => summary.findings_by_severity_error += 1,
        }
    }

    Ok(AnchorConsistencyReport {
        schema_version: ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION,
        created_at_utc: opts.now_utc.to_string(),
        anchor_registry_dir: forward_slash_path(opts.anchor_registry_dir),
        archive_dirs: opts
            .archive_dirs
            .iter()
            .map(|p| forward_slash_path(p))
            .collect(),
        export_dirs: opts
            .export_dirs
            .iter()
            .map(|p| forward_slash_path(p))
            .collect(),
        stale_threshold_secs: opts.stale_threshold_secs,
        summary,
        findings,
    })
}

impl AnchorConsistencySummary {
    fn zero() -> Self {
        Self {
            hot_total: 0,
            hot_submitted: 0,
            hot_included: 0,
            hot_finalized: 0,
            hot_failed: 0,
            hot_malformed_records: 0,
            tx_index_entries: 0,
            tx_index_orphans: 0,
            archive_manifests_scanned: 0,
            archive_entries_scanned: 0,
            export_manifests_scanned: 0,
            export_entries_scanned: 0,
            hot_export_overlaps: 0,
            archive_export_overlaps: 0,
            findings_by_severity_info: 0,
            findings_by_severity_warning: 0,
            findings_by_severity_error: 0,
        }
    }
}

// ── Hot registry scan ────────────────────────────────────────────────────────

struct HotScan {
    /// Unique artifact_hashes present as well-formed records in
    /// the hot registry — used for cross-surface set semantics.
    hot_hash_union: BTreeSet<String>,
    /// `artifact_hash_hex -> blake3_hex` of the on-disk file
    /// bytes. Used by archive collision detection.
    hot_blake3_by_hash: BTreeMap<String, String>,
    /// `tx_id -> artifact_hash_hex` from well-formed records.
    /// Used to detect cross-surface tx_id collisions.
    hot_tx_id_to_hash: BTreeMap<String, String>,
    /// Successfully-parsed records, retained for stale detection.
    well_formed_records: Vec<AnchorRecord>,
}

fn scan_hot_registry(
    root: &Path,
    findings: &mut Vec<AnchorConsistencyFinding>,
    summary: &mut AnchorConsistencySummary,
) -> EvidenceAnchorResult<HotScan> {
    let mut hot_hash_union: BTreeSet<String> = BTreeSet::new();
    let mut hot_blake3_by_hash: BTreeMap<String, String> = BTreeMap::new();
    let mut hot_tx_id_to_hash: BTreeMap<String, String> = BTreeMap::new();
    let mut well_formed_records: Vec<AnchorRecord> = Vec::new();
    let mut tx_id_first_seen: BTreeMap<String, String> = BTreeMap::new();

    // First pass — record files + .tmp orphans.
    let read_dir = std::fs::read_dir(root).map_err(|e| EvidenceAnchorError::Io {
        path: root.to_path_buf(),
        source: e,
    })?;
    for entry in read_dir {
        let entry = entry.map_err(|e| EvidenceAnchorError::Io {
            path: root.to_path_buf(),
            source: e,
        })?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();

        if name == TX_INDEX_FILENAME {
            continue;
        }
        if ext == "tmp" {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Warning,
                kind: AnchorConsistencyFindingKind::HotTmpOrphan,
                location: format!("hot/{}", name),
                artifact_hash_hex: None,
                tx_id: None,
                detail: format!("orphan .tmp file at hot/{}", name),
                suggested_action: Some("run Stage 13.4 cleanup to remove".to_string()),
            });
            continue;
        }
        if ext != "json" || stem.len() != 64 {
            continue;
        }

        // Try to parse as AnchorRecord.
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Error,
                    kind: AnchorConsistencyFindingKind::HotRecordMalformed,
                    location: format!("hot/{}", name),
                    artifact_hash_hex: Some(stem.clone()),
                    tx_id: None,
                    detail: "failed to read record file".to_string(),
                    suggested_action: Some(
                        "run Stage 13.4 cleanup to quarantine".to_string(),
                    ),
                });
                summary.hot_malformed_records += 1;
                continue;
            }
        };
        // Populate file-presence-based blake3 by filename stem
        // for EVERY `<64-hex>.json` regardless of parse status.
        // This is what archive-hot collision detection compares
        // against — a malformed/tampered hot file at a hash
        // should still trip ArchiveHotCollisionDifferentBytes.
        let file_blake3 = blake3::hash(&bytes).to_hex().to_string();
        hot_blake3_by_hash.insert(stem.clone(), file_blake3.clone());

        let record: AnchorRecord = match serde_json::from_slice(&bytes) {
            Ok(r) => r,
            Err(e) => {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Error,
                    kind: AnchorConsistencyFindingKind::HotRecordMalformed,
                    location: format!("hot/{}", name),
                    artifact_hash_hex: Some(stem.clone()),
                    tx_id: None,
                    detail: format!("malformed JSON: {e}"),
                    suggested_action: Some(
                        "run Stage 13.4 cleanup to quarantine".to_string(),
                    ),
                });
                summary.hot_malformed_records += 1;
                continue;
            }
        };

        // Filename-hash check.
        if stem != record.artifact_hash_hex {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind: AnchorConsistencyFindingKind::HotFilenameHashMismatch,
                location: format!("hot/{}", name),
                artifact_hash_hex: Some(record.artifact_hash_hex.clone()),
                tx_id: Some(record.receipt.tx_id.clone()),
                detail: format!(
                    "filename stem {} != record.artifact_hash_hex {}",
                    stem, record.artifact_hash_hex
                ),
                suggested_action: Some(
                    "investigate manually — record was hand-edited".to_string(),
                ),
            });
            // Do NOT advance the counters / unions for a hand-
            // edited record.
            continue;
        }

        // Signature defense-in-depth.
        if let Err(err) = verify_anchor_tx_data(&record.tx_data) {
            let kind = match err {
                EvidenceAnchorError::UnsupportedAnchorSchemaVersion { .. } => {
                    AnchorConsistencyFindingKind::HotRecordSchemaUnsupported
                }
                EvidenceAnchorError::SubmitterSignatureInvalid => {
                    AnchorConsistencyFindingKind::HotRecordSignatureInvalid
                }
                _ => AnchorConsistencyFindingKind::HotRecordMalformed,
            };
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind,
                location: format!("hot/{}", name),
                artifact_hash_hex: Some(record.artifact_hash_hex.clone()),
                tx_id: Some(record.receipt.tx_id.clone()),
                detail: format!("verify_anchor_tx_data: {err}"),
                suggested_action: Some(
                    "investigate manually — record was hand-edited or schema drifted"
                        .to_string(),
                ),
            });
            // Continue counting it as a record — it parses; only
            // sig/schema is bad. But do NOT include in cross-
            // surface unions (we treat tampered records as not
            // truly part of the registry's authoritative state).
            continue;
        }

        // Well-formed record — count + collect.
        summary.hot_total += 1;
        match &record.status {
            crate::evidence_anchor::registry::LocalAnchorStatus::Submitted => {
                summary.hot_submitted += 1
            }
            crate::evidence_anchor::registry::LocalAnchorStatus::Included => {
                summary.hot_included += 1
            }
            crate::evidence_anchor::registry::LocalAnchorStatus::Finalized => {
                summary.hot_finalized += 1
            }
            crate::evidence_anchor::registry::LocalAnchorStatus::Failed { .. } => {
                summary.hot_failed += 1
            }
        }
        hot_hash_union.insert(record.artifact_hash_hex.clone());
        // hot_blake3_by_hash already populated above by filename
        // stem (before parse). The record's artifact_hash_hex
        // equals the filename stem here (filename check
        // passed), so the entry is correct.

        // tx_id duplicate check.
        if let Some(prev_hash) = tx_id_first_seen.get(&record.receipt.tx_id) {
            if prev_hash != &record.artifact_hash_hex {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Error,
                    kind: AnchorConsistencyFindingKind::HotTxIdDuplicate,
                    location: format!("hot/{}", name),
                    artifact_hash_hex: Some(record.artifact_hash_hex.clone()),
                    tx_id: Some(record.receipt.tx_id.clone()),
                    detail: format!(
                        "tx_id {} is also claimed by record {}.json",
                        record.receipt.tx_id, prev_hash
                    ),
                    suggested_action: Some(
                        "investigate manually — records were hand-edited".to_string(),
                    ),
                });
            }
        } else {
            tx_id_first_seen.insert(
                record.receipt.tx_id.clone(),
                record.artifact_hash_hex.clone(),
            );
        }

        well_formed_records.push(record);
    }

    // Build TWO maps for HotTxIndexMismatch detection:
    // - record_tx_id_to_hash: `receipt.tx_id` → `artifact_hash_hex`.
    //   Used to catch the case where a record claiming the
    //   tx_index entry's tx_id exists under a DIFFERENT hash
    //   than the index points at.
    // - hash_to_record_tx_id: `artifact_hash_hex` →
    //   `receipt.tx_id`. Used to catch the reviewer-flagged
    //   case where the tx_index entry points at a record file
    //   that exists, but that record's `receipt.tx_id` is
    //   different from what the index claims.
    let mut record_tx_id_to_hash: BTreeMap<String, String> = BTreeMap::new();
    let mut hash_to_record_tx_id: BTreeMap<String, String> = BTreeMap::new();
    for record in &well_formed_records {
        record_tx_id_to_hash.insert(
            record.receipt.tx_id.clone(),
            record.artifact_hash_hex.clone(),
        );
        hash_to_record_tx_id.insert(
            record.artifact_hash_hex.clone(),
            record.receipt.tx_id.clone(),
        );
    }

    // Second pass — tx_index.json. The tx_index is the
    // CANONICAL source of truth for the (tx_id → artifact_hash)
    // mapping. Cross-surface tx_id collision detection in the
    // archive scan reads `hot_tx_id_to_hash`, which is
    // populated from tx_index.json (regardless of whether the
    // hot record file exists).
    let tx_index_path = root.join(TX_INDEX_FILENAME);
    if tx_index_path.is_file() {
        let bytes = std::fs::read(&tx_index_path).map_err(|e| EvidenceAnchorError::Io {
            path: tx_index_path.clone(),
            source: e,
        })?;
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(map) = value.get("by_tx_id").and_then(|v| v.as_object()) {
                for (tx_id, hash_value) in map {
                    let hash_hex = match hash_value.as_str() {
                        Some(s) => s,
                        None => continue,
                    };
                    summary.tx_index_entries += 1;
                    // Always populate hot_tx_id_to_hash from
                    // tx_index.json — canonical.
                    hot_tx_id_to_hash.insert(tx_id.clone(), hash_hex.to_string());
                    // Orphan: tx_index entry → absent record.
                    if !hot_blake3_by_hash.contains_key(hash_hex) {
                        findings.push(AnchorConsistencyFinding {
                            severity: AnchorConsistencySeverity::Warning,
                            kind: AnchorConsistencyFindingKind::HotTxIndexOrphan,
                            location: "hot/tx_index.json".to_string(),
                            artifact_hash_hex: Some(hash_hex.to_string()),
                            tx_id: Some(tx_id.clone()),
                            detail: format!(
                                "tx_index maps {} → {} but no hot record file exists",
                                tx_id, hash_hex
                            ),
                            suggested_action: Some(
                                "run Stage 13.4 cleanup to remove the orphan entry"
                                    .to_string(),
                            ),
                        });
                        summary.tx_index_orphans += 1;
                        continue;
                    }
                    // HotTxIndexMismatch — two failure modes;
                    // at most one finding per tx_index entry.
                    //
                    // (A) FORWARD: tx_index maps tx_id → hash,
                    //     but the record at hash.json claims a
                    //     DIFFERENT receipt.tx_id. Lookup by
                    //     tx_id would route to a record that
                    //     doesn't claim it (reviewer's flagged
                    //     case).
                    //
                    // (B) REVERSE: tx_index maps tx_id → hash,
                    //     but a record with that tx_id exists
                    //     under a DIFFERENT hash. The pre-
                    //     existing case.
                    //
                    // Forward wins if both apply — it's the
                    // more direct statement of the problem.
                    let mismatch_detail: Option<String> =
                        hash_to_record_tx_id.get(hash_hex).and_then(|record_tx_id| {
                            if record_tx_id != tx_id {
                                Some(format!(
                                    "tx_index maps {} → {}; record at {}.json claims \
                                     receipt.tx_id = {}",
                                    tx_id, hash_hex, hash_hex, record_tx_id
                                ))
                            } else {
                                None
                            }
                        });
                    let mismatch_detail = mismatch_detail.or_else(|| {
                        record_tx_id_to_hash.get(tx_id).and_then(|record_hash| {
                            if record_hash != hash_hex {
                                Some(format!(
                                    "tx_index maps {} → {}; record claiming \
                                     receipt.tx_id = {} is {}.json",
                                    tx_id, hash_hex, tx_id, record_hash
                                ))
                            } else {
                                None
                            }
                        })
                    });
                    if let Some(detail) = mismatch_detail {
                        findings.push(AnchorConsistencyFinding {
                            severity: AnchorConsistencySeverity::Error,
                            kind: AnchorConsistencyFindingKind::HotTxIndexMismatch,
                            location: "hot/tx_index.json".to_string(),
                            artifact_hash_hex: Some(hash_hex.to_string()),
                            tx_id: Some(tx_id.clone()),
                            detail,
                            suggested_action: Some(
                                "investigate manually; do not run reconcile or import \
                                 until resolved"
                                    .to_string(),
                            ),
                        });
                    }
                }
            }
        }
    }

    Ok(HotScan {
        hot_hash_union,
        hot_blake3_by_hash,
        hot_tx_id_to_hash,
        well_formed_records,
    })
}

// ── Archive scans ────────────────────────────────────────────────────────────

fn scan_archive_input(
    archive_dir: &Path,
    hot: &HotScan,
    findings: &mut Vec<AnchorConsistencyFinding>,
    summary: &mut AnchorConsistencySummary,
    archive_hash_union: &mut BTreeSet<String>,
) {
    // Stage 13.8 Q1 + Finding 5 lock: detect concrete vs root.
    // 1. <dir>/archive_manifest.json present → concrete plan_id dir.
    // 2. Otherwise scan immediate children for <child>/archive_manifest.json.
    // 3. If neither, emit ArchiveDirNoManifest (warning).
    //
    // Actual read_dir failures → ArchiveDirUnreadable (warning).
    let dir_basename = display_basename(archive_dir);

    if archive_dir.join(ARCHIVE_MANIFEST_FILENAME).is_file() {
        scan_archive_concrete(archive_dir, &dir_basename, hot, findings, summary, archive_hash_union);
        return;
    }

    let read_dir = match std::fs::read_dir(archive_dir) {
        Ok(r) => r,
        Err(e) => {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Warning,
                kind: AnchorConsistencyFindingKind::ArchiveDirUnreadable,
                location: format!("archive/{}", dir_basename),
                artifact_hash_hex: None,
                tx_id: None,
                detail: format!("read_dir failed: {e}"),
                suggested_action: Some(
                    "verify --archive-dir path and permissions".to_string(),
                ),
            });
            return;
        }
    };

    let mut found_any_child_manifest = false;
    for entry in read_dir.flatten() {
        let child = entry.path();
        if !child.is_dir() {
            continue;
        }
        if child.join(ARCHIVE_MANIFEST_FILENAME).is_file() {
            found_any_child_manifest = true;
            scan_archive_concrete(&child, &dir_basename, hot, findings, summary, archive_hash_union);
        }
    }
    if !found_any_child_manifest {
        findings.push(AnchorConsistencyFinding {
            severity: AnchorConsistencySeverity::Warning,
            kind: AnchorConsistencyFindingKind::ArchiveDirNoManifest,
            location: format!("archive/{}", dir_basename),
            artifact_hash_hex: None,
            tx_id: None,
            detail: format!(
                "--archive-dir {} is readable but contains no archive_manifest.json \
                 and no immediate child directory with one",
                archive_dir.display()
            ),
            suggested_action: Some(
                "verify --archive-dir points at a Stage 13.7 archive root or concrete plan_id dir"
                    .to_string(),
            ),
        });
    }
}

fn scan_archive_concrete(
    archive_plan_dir: &Path,
    input_dir_basename: &str,
    hot: &HotScan,
    findings: &mut Vec<AnchorConsistencyFinding>,
    summary: &mut AnchorConsistencySummary,
    archive_hash_union: &mut BTreeSet<String>,
) {
    let plan_id = display_basename(archive_plan_dir);
    let manifest_path = archive_plan_dir.join(ARCHIVE_MANIFEST_FILENAME);
    let location_prefix = format!("archive/{}/{}", input_dir_basename, plan_id);

    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(b) => b,
        Err(e) => {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind: AnchorConsistencyFindingKind::ArchiveManifestMalformed,
                location: format!("{}/archive_manifest.json", location_prefix),
                artifact_hash_hex: None,
                tx_id: None,
                detail: format!("read manifest failed: {e}"),
                suggested_action: None,
            });
            return;
        }
    };
    let manifest: AnchorArchiveManifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(m) => m,
        Err(e) => {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind: AnchorConsistencyFindingKind::ArchiveManifestMalformed,
                location: format!("{}/archive_manifest.json", location_prefix),
                artifact_hash_hex: None,
                tx_id: None,
                detail: format!("parse failed: {e}"),
                suggested_action: None,
            });
            return;
        }
    };
    if manifest.schema_version != ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION {
        findings.push(AnchorConsistencyFinding {
            severity: AnchorConsistencySeverity::Error,
            kind: AnchorConsistencyFindingKind::ArchiveManifestSchemaUnsupported,
            location: format!("{}/archive_manifest.json", location_prefix),
            artifact_hash_hex: None,
            tx_id: None,
            detail: format!(
                "manifest schema_version {} != expected {}",
                manifest.schema_version, ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION
            ),
            suggested_action: None,
        });
        return;
    }
    summary.archive_manifests_scanned += 1;

    for entry in &manifest.entries {
        summary.archive_entries_scanned += 1;

        // Path shape — anchors/<64-lower-hex>.json under the plan dir.
        if let Err(reason) = validate_archive_entry_relative(&entry.archive_relative) {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind: AnchorConsistencyFindingKind::ArchiveEntryInvalidPath,
                location: format!("{}/{}", location_prefix, entry.archive_relative),
                artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                tx_id: Some(entry.tx_id.clone()),
                detail: format!("invalid archive_relative: {reason}"),
                suggested_action: None,
            });
            continue;
        }

        let entry_path = archive_plan_dir.join(&entry.archive_relative);
        let bytes = match std::fs::read(&entry_path) {
            Ok(b) => b,
            Err(_) => {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Error,
                    kind: AnchorConsistencyFindingKind::ArchiveEntryMissingFile,
                    location: format!("{}/{}", location_prefix, entry.archive_relative),
                    artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                    tx_id: Some(entry.tx_id.clone()),
                    detail: format!(
                        "archived file missing at {}",
                        entry_path.display()
                    ),
                    suggested_action: None,
                });
                continue;
            }
        };
        let computed_blake3 = blake3::hash(&bytes).to_hex().to_string();
        if computed_blake3 != entry.blake3_hex || bytes.len() as u64 != entry.bytes {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind: AnchorConsistencyFindingKind::ArchiveEntryBlake3Mismatch,
                location: format!("{}/{}", location_prefix, entry.archive_relative),
                artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                tx_id: Some(entry.tx_id.clone()),
                detail: format!(
                    "computed BLAKE3 {} differs from manifest declared {}",
                    computed_blake3, entry.blake3_hex
                ),
                suggested_action: None,
            });
            continue;
        }
        let record: AnchorRecord = match serde_json::from_slice(&bytes) {
            Ok(r) => r,
            Err(e) => {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Error,
                    kind: AnchorConsistencyFindingKind::ArchiveEntryRecordMalformed,
                    location: format!("{}/{}", location_prefix, entry.archive_relative),
                    artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                    tx_id: Some(entry.tx_id.clone()),
                    detail: format!("parse archived record failed: {e}"),
                    suggested_action: None,
                });
                continue;
            }
        };
        if let Err(err) = verify_anchor_tx_data(&record.tx_data) {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind: AnchorConsistencyFindingKind::ArchiveEntrySignatureInvalid,
                location: format!("{}/{}", location_prefix, entry.archive_relative),
                artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                tx_id: Some(entry.tx_id.clone()),
                detail: format!("verify_anchor_tx_data: {err}"),
                suggested_action: None,
            });
            continue;
        }
        // Metadata cross-check.
        if record.artifact_hash_hex != entry.artifact_hash_hex
            || record.receipt.tx_id != entry.tx_id
            || record.status.as_str() != entry.status
        {
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind: AnchorConsistencyFindingKind::ArchiveEntryMetadataMismatch,
                location: format!("{}/{}", location_prefix, entry.archive_relative),
                artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                tx_id: Some(entry.tx_id.clone()),
                detail: "manifest entry fields don't match record fields".to_string(),
                suggested_action: None,
            });
            continue;
        }

        // Set membership for cross-surface overlap counters.
        archive_hash_union.insert(entry.artifact_hash_hex.clone());

        // Cross-surface vs hot.
        if let Some(hot_blake3) = hot.hot_blake3_by_hash.get(&entry.artifact_hash_hex) {
            if hot_blake3 == &entry.blake3_hex {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Warning,
                    kind: AnchorConsistencyFindingKind::ArchiveHotCollisionSameBytes,
                    location: format!("{}/{}", location_prefix, entry.archive_relative),
                    artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                    tx_id: Some(entry.tx_id.clone()),
                    detail: "archive entry duplicates hot record byte-for-byte; \
                             possible partial Phase-2 archive or out-of-band restore"
                        .to_string(),
                    suggested_action: Some(
                        "investigate via Stage 13.7 archive logs; resolve by completing \
                         the archive or accepting the duplication"
                            .to_string(),
                    ),
                });
            } else {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Error,
                    kind: AnchorConsistencyFindingKind::ArchiveHotCollisionDifferentBytes,
                    location: format!("{}/{}", location_prefix, entry.archive_relative),
                    artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                    tx_id: Some(entry.tx_id.clone()),
                    detail: format!(
                        "archive BLAKE3 {} != hot BLAKE3 {}",
                        entry.blake3_hex, hot_blake3
                    ),
                    suggested_action: Some(
                        "investigate manually; do not run Stage 13.7 restore until \
                         resolved"
                            .to_string(),
                    ),
                });
            }
        }
        // tx_id collision — hot maps the same tx_id to a different hash.
        if let Some(hot_hash_for_tx) = hot.hot_tx_id_to_hash.get(&entry.tx_id) {
            if hot_hash_for_tx != &entry.artifact_hash_hex {
                findings.push(AnchorConsistencyFinding {
                    severity: AnchorConsistencySeverity::Error,
                    kind: AnchorConsistencyFindingKind::ArchiveTxIdCollision,
                    location: format!("{}/{}", location_prefix, entry.archive_relative),
                    artifact_hash_hex: Some(entry.artifact_hash_hex.clone()),
                    tx_id: Some(entry.tx_id.clone()),
                    detail: format!(
                        "hot tx_index maps {} → {}; archive entry claims {}",
                        entry.tx_id, hot_hash_for_tx, entry.artifact_hash_hex
                    ),
                    suggested_action: Some(
                        "investigate manually; do not run Stage 13.7 restore until \
                         resolved"
                            .to_string(),
                    ),
                });
            }
        }
    }
}

// ── Export scan (coarse — Q2 lock) ───────────────────────────────────────────

fn scan_export_input(
    export_dir: &Path,
    findings: &mut Vec<AnchorConsistencyFinding>,
    summary: &mut AnchorConsistencySummary,
    export_hash_union: &mut BTreeSet<String>,
) {
    let dir_basename = display_basename(export_dir);
    let location = format!("export/{}", dir_basename);

    // Optional-path open check.
    if !export_dir.is_dir() {
        findings.push(AnchorConsistencyFinding {
            severity: AnchorConsistencySeverity::Warning,
            kind: AnchorConsistencyFindingKind::ExportDirUnreadable,
            location: location.clone(),
            artifact_hash_hex: None,
            tx_id: None,
            detail: format!(
                "--export-dir {} is not a directory",
                export_dir.display()
            ),
            suggested_action: Some(
                "verify --export-dir path and permissions".to_string(),
            ),
        });
        return;
    }

    // Coarse delegation to Stage 13.5.
    let verify_result = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir,
        strict: false,
    });
    let manifest = match verify_result {
        Err(err) => {
            // Convert Stage 13.5 typed error → coarse finding +
            // detail carrying the reason tag.
            let reason_tag =
                crate::evidence_anchor::evidence_anchor_reason_tag(&err);
            findings.push(AnchorConsistencyFinding {
                severity: AnchorConsistencySeverity::Error,
                kind: AnchorConsistencyFindingKind::ExportVerifyFailed,
                location: location.clone(),
                artifact_hash_hex: None,
                tx_id: None,
                detail: format!("Stage 13.5 verify failed (reason_tag={reason_tag}): {err}"),
                suggested_action: Some(
                    "follow Stage 13.5 verify-integrity-evidence-anchor-export to diagnose"
                        .to_string(),
                ),
            });
            return;
        }
        Ok(_report) => {
            // Re-read the manifest for overlap counting.
            let manifest_path = export_dir
                .join(crate::evidence_anchor::export::EXPORT_MANIFEST_FILENAME);
            let bytes = match std::fs::read(&manifest_path) {
                Ok(b) => b,
                Err(_) => return,
            };
            match serde_json::from_slice::<
                crate::evidence_anchor::export::AnchorExportManifest,
            >(&bytes)
            {
                Ok(m) => m,
                Err(_) => return,
            }
        }
    };

    summary.export_manifests_scanned += 1;
    for entry in &manifest.entries {
        summary.export_entries_scanned += 1;
        // Set semantics — only anchor_record entries contribute
        // to the cross-surface overlap counter (artifact_bytes /
        // signed_chain_report aren't anchors).
        if entry.kind
            == crate::evidence_anchor::export::AnchorExportEntryKind::AnchorRecord
                .as_str()
        {
            if let Some(hash) = entry.artifact_hash_hex.as_ref() {
                export_hash_union.insert(hash.clone());
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn display_basename(p: &Path) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

fn forward_slash_path(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

fn parse_rfc3339(s: &str) -> EvidenceAnchorResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| EvidenceAnchorError::MalformedSignedAtUtc {
            raw: s.to_string(),
            reason: e.to_string(),
        })
}

fn validate_archive_entry_relative(p: &str) -> Result<(), &'static str> {
    if p.is_empty() {
        return Err("empty");
    }
    if p.contains('\\') || p.contains('\0') {
        return Err("backslash or null byte");
    }
    if std::path::Path::new(p).is_absolute() {
        return Err("absolute path");
    }
    if p.starts_with('/') {
        return Err("leading slash");
    }
    if p.split('/').any(|c| c == "..") {
        return Err("parent traversal");
    }
    let suffix = p.strip_prefix("anchors/").ok_or("must live under anchors/")?;
    if suffix.contains('/') {
        return Err("nested subdir under anchors/");
    }
    let stem = suffix.strip_suffix(".json").ok_or("missing .json suffix")?;
    if stem.len() != 64 {
        return Err("stem not 64 hex chars");
    }
    if !stem.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err("stem not lowercase hex");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_version_constant_is_one() {
        assert_eq!(ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION, 1);
    }

    #[test]
    fn severity_str_is_stable() {
        assert_eq!(AnchorConsistencySeverity::Info.as_str(), "info");
        assert_eq!(AnchorConsistencySeverity::Warning.as_str(), "warning");
        assert_eq!(AnchorConsistencySeverity::Error.as_str(), "error");
    }

    #[test]
    fn finding_kind_count_is_24() {
        // V2 lock: 9 hot + 13 archive + 2 export = 24.
        // HotAndExportDuplicate / ArchiveAndExportDuplicate
        // dropped per Finding 4.
        use AnchorConsistencyFindingKind::*;
        let kinds = [
            HotRecordMalformed,
            HotRecordSignatureInvalid,
            HotRecordSchemaUnsupported,
            HotFilenameHashMismatch,
            HotTxIndexOrphan,
            HotTxIndexMismatch,
            HotTxIdDuplicate,
            HotStaleOpenRecord,
            HotTmpOrphan,
            ArchiveDirUnreadable,
            ArchiveDirNoManifest,
            ArchiveManifestMalformed,
            ArchiveManifestSchemaUnsupported,
            ArchiveEntryInvalidPath,
            ArchiveEntryMissingFile,
            ArchiveEntryBlake3Mismatch,
            ArchiveEntryRecordMalformed,
            ArchiveEntrySignatureInvalid,
            ArchiveEntryMetadataMismatch,
            ArchiveHotCollisionSameBytes,
            ArchiveHotCollisionDifferentBytes,
            ArchiveTxIdCollision,
            ExportDirUnreadable,
            ExportVerifyFailed,
        ];
        assert_eq!(kinds.len(), 24);
    }
}
