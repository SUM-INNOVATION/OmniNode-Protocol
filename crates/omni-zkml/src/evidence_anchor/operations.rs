//! Phase 5 Stage 13.3 — operator-facing hardening for the
//! integrity-evidence-anchor lifecycle.
//!
//! Three pure read-only helpers consumed by Stage 13.3's CLI
//! subcommands (`summary-integrity-evidence-anchors`,
//! `watch-integrity-evidence-anchors`):
//!
//! - [`list_evidence_anchors_by_status`] — counts records per
//!   local status. Returns a serde-friendly summary struct.
//! - [`check_evidence_anchor_registry_health`] — read-only
//!   directory scan: counts records, malformed records, orphan
//!   `tx_index.json` entries, and orphan `.tmp` files. **Does
//!   not delete or quarantine** — the operator decides what to
//!   do.
//! - [`list_stale_submitted_or_included`] — time-based stale
//!   detection using the existing
//!   [`crate::evidence_anchor::AnchorRecord::submitted_at`]
//!   field. **No `AnchorRecord` shape change.** Block-based
//!   staleness is deferred (Stage 13.3 scope locked at
//!   time-based only).
//!
//! All three are pure reads — no chain interaction, no
//! registry mutation. Locked Stage 13.3 invariants:
//! `summary-integrity-evidence-anchors` is fully local
//! (no `--rpc-url` flag); `watch-integrity-evidence-anchors`
//! reuses Stage 13.2's
//! [`crate::evidence_anchor::reconcile_evidence_anchors_workflow`]
//! verbatim for chain-side reconciliation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{EvidenceAnchorError, EvidenceAnchorResult};
use crate::evidence_anchor::registry::{
    LocalAnchorStatus, LocalEvidenceAnchorRegistry,
};

// ── Registry summary ──────────────────────────────────────────────────────────

/// Stage 13.3 — counts-by-local-status snapshot of an anchor
/// registry. Produced by [`list_evidence_anchors_by_status`].
///
/// Serde-friendly so the CLI can emit it as both a structured
/// event line and (optionally) as JSON. Closed shape: the field
/// set matches the closed [`LocalAnchorStatus`] enum and bumps
/// whenever that enum bumps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAnchorRegistrySummary {
    pub total: u64,
    pub submitted: u64,
    pub included: u64,
    pub finalized: u64,
    pub failed: u64,
}

/// Iterate the registry and count records by local status.
///
/// Pure read — no chain interaction, no registry mutation.
/// Records the registry's `list()` helper would reject as
/// malformed bubble as `EvidenceAnchorError::Io { source }`;
/// see [`check_evidence_anchor_registry_health`] for a
/// failure-tolerant directory scan.
pub fn list_evidence_anchors_by_status(
    registry: &LocalEvidenceAnchorRegistry,
) -> EvidenceAnchorResult<EvidenceAnchorRegistrySummary> {
    let records = registry.list().map_err(|e| EvidenceAnchorError::Io {
        path: registry.root().to_path_buf(),
        source: e,
    })?;
    let mut out = EvidenceAnchorRegistrySummary {
        total: 0,
        submitted: 0,
        included: 0,
        finalized: 0,
        failed: 0,
    };
    for record in &records {
        out.total += 1;
        match &record.status {
            LocalAnchorStatus::Submitted => out.submitted += 1,
            LocalAnchorStatus::Included => out.included += 1,
            LocalAnchorStatus::Finalized => out.finalized += 1,
            LocalAnchorStatus::Failed { .. } => out.failed += 1,
        }
    }
    Ok(out)
}

// ── Registry health ───────────────────────────────────────────────────────────

/// Stage 13.3 — registry-health diagnostic. Produced by
/// [`check_evidence_anchor_registry_health`].
///
/// Read-only diagnostic. Reports counts; the operator decides
/// whether to clean up. Closed shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAnchorRegistryHealth {
    /// Successfully-parsed records under the registry root.
    pub records: u64,
    /// Files matching the `<64-hex>.json` shape that failed to
    /// parse as `AnchorRecord`. Likely operator-introduced
    /// corruption.
    pub malformed_records: u64,
    /// `tx_index.json` entries whose mapped
    /// `artifact_hash_hex` does NOT correspond to a record
    /// file on disk. Indicates either:
    /// - a record file was manually deleted (orphan index
    ///   entry), OR
    /// - the index was hand-edited.
    pub orphan_tx_index_entries: u64,
    /// `.tmp` files under the registry root. Atomic-write
    /// scratch left behind by a crashed `mark_submitted` /
    /// `update_status` call. Safe to delete after operator
    /// review.
    pub orphan_tmp_files: u64,
}

/// Read-only registry-directory scan.
///
/// Iterates the registry root once. Files matching
/// `<64-hex>.json` are parse-attempted; success increments
/// `records`, parse failure increments `malformed_records`.
/// Files with a `.tmp` extension increment `orphan_tmp_files`.
/// The `tx_index.json` is parsed (parse failure increments
/// `malformed_records`) and each entry whose mapped record
/// file is absent increments `orphan_tx_index_entries`.
///
/// **Does not delete, rewrite, or quarantine anything.** The
/// caller (CLI) decides what to do with the counts.
pub fn check_evidence_anchor_registry_health(
    registry: &LocalEvidenceAnchorRegistry,
) -> EvidenceAnchorResult<EvidenceAnchorRegistryHealth> {
    let root = registry.root();
    let mut out = EvidenceAnchorRegistryHealth {
        records: 0,
        malformed_records: 0,
        orphan_tx_index_entries: 0,
        orphan_tmp_files: 0,
    };
    let entries =
        std::fs::read_dir(root).map_err(|e| EvidenceAnchorError::Io {
            path: root.to_path_buf(),
            source: e,
        })?;
    for entry in entries {
        let entry = entry.map_err(|e| EvidenceAnchorError::Io {
            path: root.to_path_buf(),
            source: e,
        })?;
        let path = entry.path();
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
        // `.tmp` files anywhere under the registry root are
        // orphan atomic-write scratch.
        if ext == "tmp" {
            out.orphan_tmp_files += 1;
            continue;
        }
        // `<64-hex>.json` is a record file; anything else
        // (notably `tx_index.json`) is skipped here and
        // handled separately below.
        if ext != "json" {
            continue;
        }
        if stem.len() != 64 {
            // `tx_index.json` and anything else with non-64-char
            // stem is not a record file.
            continue;
        }
        // Try to parse as AnchorRecord. We can't go through
        // registry.load_by_artifact_hash because that returns
        // an io::Error wrapped with a specific shape; we want
        // a tolerant read that counts rather than aborts.
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                out.malformed_records += 1;
                continue;
            }
        };
        if serde_json::from_slice::<
            crate::evidence_anchor::registry::AnchorRecord,
        >(&bytes)
        .is_ok()
        {
            out.records += 1;
        } else {
            out.malformed_records += 1;
        }
    }
    // Inspect the tx_index for orphan entries.
    let tx_index_path = root.join("tx_index.json");
    if tx_index_path.is_file() {
        let bytes = match std::fs::read(&tx_index_path) {
            Ok(b) => b,
            Err(_) => {
                out.malformed_records += 1;
                return Ok(out);
            }
        };
        let index: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => {
                out.malformed_records += 1;
                return Ok(out);
            }
        };
        // Shape: { "by_tx_id": { "<tx_id>": "<artifact_hash_hex>", ... } }
        if let Some(map) = index.get("by_tx_id").and_then(|v| v.as_object()) {
            for hash_value in map.values() {
                let Some(hash_hex) = hash_value.as_str() else {
                    out.malformed_records += 1;
                    continue;
                };
                let record_path = root.join(format!("{hash_hex}.json"));
                if !record_path.is_file() {
                    out.orphan_tx_index_entries += 1;
                }
            }
        }
    }
    Ok(out)
}

// ── Stale detection (time-based) ──────────────────────────────────────────────

/// Stage 13.3 — one stale-anchor report row. Produced by
/// [`list_stale_submitted_or_included`] and emitted as
/// `event=integrity_evidence_anchor_stale` per record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleAnchorInfo {
    pub artifact_hash_hex: String,
    pub tx_id: String,
    pub status: LocalAnchorStatus,
    pub submitted_at: DateTime<Utc>,
    pub age_secs: u64,
}

/// List `Submitted` / `Included` records whose age in seconds
/// since [`crate::evidence_anchor::registry::AnchorRecord::submitted_at`]
/// is at least `threshold_secs`.
///
/// **Time-based only** (Stage 13.3 locked scope). `now_utc`
/// is a caller parameter (not `Utc::now()` direct) so tests
/// inject synthetic timestamps deterministically.
///
/// - `Finalized` and `Failed` records are skipped — terminal
///   states never report stale.
/// - Records whose `submitted_at` is in the future relative to
///   `now_utc` (clock skew) are skipped — no underflow, no
///   spurious "stale" rows.
/// - Returned in deterministic `artifact_hash_hex` ascending
///   order (matches the registry's `list()` ordering).
pub fn list_stale_submitted_or_included(
    registry: &LocalEvidenceAnchorRegistry,
    now_utc: DateTime<Utc>,
    threshold_secs: u64,
) -> EvidenceAnchorResult<Vec<StaleAnchorInfo>> {
    let records = registry.list().map_err(|e| EvidenceAnchorError::Io {
        path: registry.root().to_path_buf(),
        source: e,
    })?;
    let mut out = Vec::new();
    for record in records {
        let is_open = matches!(
            record.status,
            LocalAnchorStatus::Submitted | LocalAnchorStatus::Included
        );
        if !is_open {
            continue;
        }
        let elapsed = now_utc.signed_duration_since(record.submitted_at);
        let secs = elapsed.num_seconds();
        if secs < 0 {
            // Future-dated `submitted_at` (clock skew) —
            // never stale.
            continue;
        }
        let age_secs = secs as u64;
        if age_secs < threshold_secs {
            continue;
        }
        out.push(StaleAnchorInfo {
            artifact_hash_hex: record.artifact_hash_hex,
            tx_id: record.receipt.tx_id,
            status: record.status,
            submitted_at: record.submitted_at,
            age_secs,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry) {
        let dir = tempfile::tempdir().unwrap();
        let reg = LocalEvidenceAnchorRegistry::open(dir.path().join("anchors")).unwrap();
        (dir, reg)
    }

    #[test]
    fn summary_empty_registry_is_all_zero() {
        let (_dir, reg) = fresh_registry();
        let s = list_evidence_anchors_by_status(&reg).unwrap();
        assert_eq!(
            s,
            EvidenceAnchorRegistrySummary {
                total: 0,
                submitted: 0,
                included: 0,
                finalized: 0,
                failed: 0,
            }
        );
    }

    #[test]
    fn health_empty_registry_is_all_zero() {
        let (_dir, reg) = fresh_registry();
        let h = check_evidence_anchor_registry_health(&reg).unwrap();
        assert_eq!(
            h,
            EvidenceAnchorRegistryHealth {
                records: 0,
                malformed_records: 0,
                orphan_tx_index_entries: 0,
                orphan_tmp_files: 0,
            }
        );
    }

    #[test]
    fn stale_detection_empty_registry_returns_empty() {
        let (_dir, reg) = fresh_registry();
        let now = Utc::now();
        let stale = list_stale_submitted_or_included(&reg, now, 60).unwrap();
        assert!(stale.is_empty());
    }
}
