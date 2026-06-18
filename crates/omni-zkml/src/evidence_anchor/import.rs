//! Phase 5 Stage 13.6 — local-only export-import / registry
//! restore for Stage 13.5 integrity-evidence-anchor exports.
//!
//! Stage 13.6 ships an operator-facing import path that restores
//! selected `anchor_record` entries from a previously-verified
//! Stage 13.5 export into a target local anchor registry. Fully
//! local — no SUM Chain RPCs, no `omni-sumchain` types, no
//! private-chain repo deps. The Stage 13.0 wire / domain /
//! canonical-bytes / signing surface is **read only**; not a byte
//! is re-signed.
//!
//! ## Locked Stage 13.6 invariants
//!
//! - **Verify-first.** Both [`plan_anchor_export_import`] and
//!   [`apply_anchor_export_import`] call
//!   [`crate::evidence_anchor::verify_anchor_export`] before
//!   touching the target registry. Apply re-runs verify as a
//!   durability fence — drift between plan and apply is normal
//!   and caught.
//! - **Default dry-run.** Mutations only when the caller sets
//!   [`AnchorImportOptions::dry_run = false`] explicitly. The
//!   CLI gate is `--apply`.
//! - **Preflight-all-before-mutate.** Apply classifies EVERY
//!   selected action against the target state and refuses on
//!   ANY `import_target_exists` BEFORE writing any record file.
//!   No partial-write state if action #5 of 5 trips a tx_id
//!   collision.
//! - **Byte-preserve.** Imported record files are copied
//!   verbatim from `<export-dir>/anchors/<hash>.json` to
//!   `<registry>/<hash>.json` — NO serde round-trip. Imported
//!   records carry historical `submitted_at` / `updated_at`
//!   timestamps from the source registry. Stage 13.3 stale-age
//!   views may report historical ages; that's a documented,
//!   intentional consequence of the forensic-fidelity contract
//!   (locked decision D5).
//! - **tx_index.json is merged, not rebuilt.** Load existing
//!   tx_index → add the imported records' `(tx_id →
//!   artifact_hash_hex)` mappings → atomic temp+rename. Preserves
//!   every unrelated entry.
//! - **One new closed reason tag** (`import_target_exists`) with
//!   a `field=artifact_hash | tx_id` discriminator. The Stage
//!   13.5 export-side and Stage 13.0 verifier-side tag sets are
//!   reused verbatim.
//!
//! ## Conflict matrix (D1 locked)
//!
//! | row | target file present? | target BLAKE3 vs manifest | tx_index has tx_id? | tx_index → | outcome (apply) | outcome (dry-run) | refuse |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 1   | no  | n/a   | no  | n/a   | `imported`                   | `would_import`                  | — |
//! | 2   | yes | equal | yes | same  | `skipped_already_imported`   | `skipped_already_imported`      | — |
//! | 3   | yes | equal | no  | n/a   | `re_added_tx_index_entry`    | `would_re_add_tx_index_entry`   | — |
//! | 4   | yes | diff  | any | any   | refuse                       | refuse                          | `import_target_exists` (field=artifact_hash) |
//! | 5   | any | n/a   | yes | diff  | refuse                       | refuse                          | `import_target_exists` (field=tx_id) |
//!
//! Row precedence: 5 → 4 → 1/2/3. A row-5 tx_id collision wins
//! over a row-1 fresh-import even when the artifact-hash file is
//! absent — the tx_id is the harder failure mode, so it short-
//! circuits first.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{EvidenceAnchorError, EvidenceAnchorResult};
use crate::evidence_anchor::export::{
    verify_anchor_export, AnchorExportEntry, AnchorExportEntryKind,
    AnchorExportManifest, AnchorExportVerifyOptions, EXPORT_MANIFEST_FILENAME,
};
use crate::evidence_anchor::registry::{
    AnchorRecord, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
};
use crate::evidence_anchor::wire::verify_anchor_tx_data;

// ── Closed outcome strings ───────────────────────────────────────────────────

/// Closed-set apply outcomes. Operator log scrapers depend on
/// these stable strings.
const OUTCOME_IMPORTED: &str = "imported";
const OUTCOME_WOULD_IMPORT: &str = "would_import";
const OUTCOME_SKIPPED_ALREADY_IMPORTED: &str = "skipped_already_imported";
const OUTCOME_RE_ADDED_TX_INDEX_ENTRY: &str = "re_added_tx_index_entry";
const OUTCOME_WOULD_RE_ADD_TX_INDEX_ENTRY: &str = "would_re_add_tx_index_entry";

const TX_INDEX_FILENAME: &str = "tx_index.json";

// ── Selection ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnchorImportSelection {
    /// Filter by `LocalAnchorStatus` kind (OR within this kind).
    /// Empty = no status constraint.
    pub statuses: Vec<LocalAnchorStatus>,
    /// Filter by `receipt.tx_id` (OR within this kind). Empty =
    /// no tx_id constraint. A non-matching value refuses with
    /// `anchor_not_found`.
    pub tx_ids: Vec<String>,
    /// Filter by `artifact_hash_hex` (OR within this kind). Empty
    /// = no hash constraint. A non-matching value refuses with
    /// `anchor_not_found`.
    pub artifact_hashes: Vec<String>,
}

impl AnchorImportSelection {
    pub fn is_empty(&self) -> bool {
        self.statuses.is_empty()
            && self.tx_ids.is_empty()
            && self.artifact_hashes.is_empty()
    }

    /// Match an `anchor_record` manifest entry against the
    /// selection — AND across kinds, OR within a kind. Empty
    /// selection (D6 — Stage 13.5 export asymmetry) matches
    /// EVERY entry.
    fn matches_entry(&self, entry: &AnchorExportEntry) -> bool {
        if !self.statuses.is_empty() {
            let entry_status = entry.status.as_deref().unwrap_or("");
            if !self
                .statuses
                .iter()
                .any(|s| s.as_str() == entry_status)
            {
                return false;
            }
        }
        if !self.tx_ids.is_empty() {
            let entry_tx_id = entry.tx_id.as_deref().unwrap_or("");
            if !self.tx_ids.iter().any(|t| t == entry_tx_id) {
                return false;
            }
        }
        if !self.artifact_hashes.is_empty() {
            let entry_hash = entry.artifact_hash_hex.as_deref().unwrap_or("");
            if !self.artifact_hashes.iter().any(|h| h == entry_hash) {
                return false;
            }
        }
        true
    }
}

// ── Options ──────────────────────────────────────────────────────────────────

pub struct AnchorImportOptions<'a> {
    /// When true (default at the CLI layer), classification runs
    /// but no FS mutations land on the target registry. Outcomes
    /// carry `would_*` prefixes for the cases that would mutate.
    pub dry_run: bool,
    /// Passed through to
    /// [`verify_anchor_export`](crate::evidence_anchor::verify_anchor_export).
    /// `--strict` validates the WHOLE export tree (D4 — every
    /// `anchor_record` in the manifest must have a paired
    /// `artifact_bytes`), NOT just the records being imported.
    pub strict: bool,
    /// Filter manifest `anchor_record` entries. Empty selection
    /// = import all (intentional asymmetry vs Stage 13.5
    /// export's required-selector, locked as D6).
    pub selection: &'a AnchorImportSelection,
    /// RFC 3339 UTC. Stamped into the import event line. Tests
    /// inject for determinism.
    pub now_utc: &'a str,
}

// ── Plan + report ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorImportPlan {
    pub export_id: String,
    pub created_at_utc: String,
    /// Sorted by `artifact_hash_hex` ascending — deterministic.
    pub actions: Vec<PlannedImportAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedImportAction {
    pub artifact_hash_hex: String,
    pub tx_id: String,
    /// Closed-set: `submitted | included | finalized | failed`.
    pub status: String,
    /// `anchors/<artifact_hash_hex>.json` — verified to match
    /// the Stage 13.5 per-kind shape rule before this point.
    pub relative_path: String,
    pub bytes: u64,
    /// Closed-set classification (mirrors the conflict matrix in
    /// the module-level doc). Apply re-classifies against
    /// CURRENT target state.
    pub pre_action_outcome: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorImportActionOutcome {
    pub artifact_hash_hex: String,
    pub tx_id: String,
    pub status: String,
    /// Closed set: `imported | would_import |
    /// skipped_already_imported | re_added_tx_index_entry |
    /// would_re_add_tx_index_entry`.
    pub outcome: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorImportReport {
    pub export_id: String,
    /// `apply` | `dry_run`. Closed set.
    pub mode: String,
    pub actions_imported: u32,
    pub actions_would_import: u32,
    pub actions_skipped_already_imported: u32,
    pub actions_re_added_tx_index_entry: u32,
    pub actions_would_re_add_tx_index_entry: u32,
    pub outcomes: Vec<AnchorImportActionOutcome>,
}

// ── Plan ──────────────────────────────────────────────────────────────────────

/// Build a typed import plan without touching the target.
///
/// Sequence:
/// 1. `verify_anchor_export(export_dir, strict=opts.strict)` —
///    propagate any Stage 13.5 refusal verbatim with its
///    existing closed tag.
/// 2. Parse the manifest. Filter `anchor_record` entries via the
///    selection (AND across kinds, OR within a kind; empty
///    selection matches all per D6).
/// 3. Selector misses refuse with
///    [`EvidenceAnchorError::AnchorNotFound`] (D8).
/// 4. For each selected entry, validate metadata against the
///    embedded record (reuses Stage 13.5
///    `export_entry_metadata_mismatch` per D7).
/// 5. Classify against the CURRENT target registry state per
///    the §6 matrix.
/// 6. Return the typed plan.
pub fn plan_anchor_export_import(
    export_dir: &Path,
    target_registry: &LocalEvidenceAnchorRegistry,
    opts: &AnchorImportOptions<'_>,
) -> EvidenceAnchorResult<AnchorImportPlan> {
    // Step 1: verify export tree (whole-export per D4).
    verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir,
        strict: opts.strict,
    })?;

    // Step 2: load + parse manifest.
    let manifest = read_manifest(export_dir)?;

    // Step 3: filter manifest entries to the selection.
    let mut selected: Vec<&AnchorExportEntry> = manifest
        .entries
        .iter()
        .filter(|e| e.kind == AnchorExportEntryKind::AnchorRecord.as_str())
        .filter(|e| opts.selection.matches_entry(e))
        .collect();
    selected.sort_by(|a, b| {
        a.artifact_hash_hex
            .as_deref()
            .unwrap_or("")
            .cmp(b.artifact_hash_hex.as_deref().unwrap_or(""))
    });

    // Step 3.5: selector-miss refusal (D8). For each
    // explicitly-supplied tx_id / artifact_hash selector, refuse
    // if NO manifest entry matches.
    refuse_on_selector_miss(&manifest, opts.selection)?;

    // Step 4 + 5: per-entry metadata cross-check + target-state
    // classification.
    let mut actions: Vec<PlannedImportAction> = Vec::with_capacity(selected.len());
    for entry in &selected {
        let (artifact_hash_hex, tx_id, status) = required_anchor_record_fields(entry)?;
        let record = read_and_parse_record(export_dir, &entry.relative_path)?;
        cross_check_metadata(entry, &record)?;
        let outcome = classify_target_state(target_registry, entry, &record, true)?;
        actions.push(PlannedImportAction {
            artifact_hash_hex: artifact_hash_hex.to_string(),
            tx_id: tx_id.to_string(),
            status: status.to_string(),
            relative_path: entry.relative_path.clone(),
            bytes: entry.bytes,
            pre_action_outcome: outcome,
        });
    }

    Ok(AnchorImportPlan {
        export_id: manifest.export_id,
        created_at_utc: opts.now_utc.to_string(),
        actions,
    })
}

// ── Apply ─────────────────────────────────────────────────────────────────────

/// Execute (or dry-run) the import.
///
/// Apply ordering (the two implementation-notes lock):
///
/// 1. Re-run [`verify_anchor_export`] as a durability fence.
/// 2. Parse manifest, apply selection + selector-miss refusal,
///    and CLASSIFY EVERY selected action against the CURRENT
///    target state BEFORE any FS mutation. (Implementation note
///    1 — preflight-all-before-mutate.)
/// 3. If ANY classification routes to `import_target_exists`,
///    refuse with no writes.
/// 4. Pass 1 — write each row-1 (`imported`) action's bytes
///    atomically into `<registry>/<hash>.json`. Row-3
///    (`re_added_tx_index_entry`) actions DO NOT touch the
///    record file (implementation note 2 — file already present,
///    only tx_index changes). Row-2 (`skipped_already_imported`)
///    actions touch nothing.
/// 5. Pass 1.5 (durability fence) — merge `tx_index.json`:
///    load existing → add accumulated (tx_id, hash) entries →
///    atomic `.tmp + rename`. Preserves unrelated entries (D3
///    lock).
pub fn apply_anchor_export_import(
    export_dir: &Path,
    target_registry: &LocalEvidenceAnchorRegistry,
    opts: &AnchorImportOptions<'_>,
) -> EvidenceAnchorResult<AnchorImportReport> {
    // Step 1: durability-fence verify.
    verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir,
        strict: opts.strict,
    })?;

    // Step 2: manifest parse + selection.
    let manifest = read_manifest(export_dir)?;
    let mut selected: Vec<&AnchorExportEntry> = manifest
        .entries
        .iter()
        .filter(|e| e.kind == AnchorExportEntryKind::AnchorRecord.as_str())
        .filter(|e| opts.selection.matches_entry(e))
        .collect();
    selected.sort_by(|a, b| {
        a.artifact_hash_hex
            .as_deref()
            .unwrap_or("")
            .cmp(b.artifact_hash_hex.as_deref().unwrap_or(""))
    });

    // Step 2.5: selector miss.
    refuse_on_selector_miss(&manifest, opts.selection)?;

    // Step 3: PREFLIGHT — classify EVERY selected action BEFORE
    // any FS mutation. This is the implementation-note 1 lock:
    // a tx_id collision on action #5 of 5 must refuse BEFORE any
    // of actions #1-#4 land bytes on disk. Per-entry cross-check
    // runs too, since metadata mismatch refuses cleanly via the
    // reused Stage 13.5 tag.
    struct ClassifiedAction<'a> {
        entry: &'a AnchorExportEntry,
        outcome: String,
    }
    let mut classified: Vec<ClassifiedAction<'_>> = Vec::with_capacity(selected.len());
    for entry in &selected {
        let _ = required_anchor_record_fields(entry)?;
        let record = read_and_parse_record(export_dir, &entry.relative_path)?;
        cross_check_metadata(entry, &record)?;
        let outcome = classify_target_state(target_registry, entry, &record, opts.dry_run)?;
        classified.push(ClassifiedAction { entry, outcome });
    }

    let mut actions_imported: u32 = 0;
    let mut actions_would_import: u32 = 0;
    let mut actions_skipped_already_imported: u32 = 0;
    let mut actions_re_added_tx_index_entry: u32 = 0;
    let mut actions_would_re_add_tx_index_entry: u32 = 0;
    let mut outcomes: Vec<AnchorImportActionOutcome> = Vec::with_capacity(classified.len());
    let mut tx_index_updates: BTreeMap<String, String> = BTreeMap::new();

    // Step 4: Pass 1 — write record bytes for `imported` only.
    // `re_added_tx_index_entry` doesn't touch the record file
    // (implementation note 2 — the byte-equal file is already
    // present). `skipped_already_imported` does nothing.
    for action in &classified {
        let artifact_hash_hex = action
            .entry
            .artifact_hash_hex
            .as_deref()
            .expect("validated in classify_target_state");
        let tx_id = action
            .entry
            .tx_id
            .as_deref()
            .expect("validated in classify_target_state");
        let status = action
            .entry
            .status
            .as_deref()
            .expect("validated in classify_target_state");

        match action.outcome.as_str() {
            s if s == OUTCOME_IMPORTED => {
                let src_path = export_dir.join(&action.entry.relative_path);
                let dst_path = target_registry
                    .root()
                    .join(format!("{artifact_hash_hex}.json"));
                // Byte-preserve (D5): copy the export's exact
                // bytes into the registry. No serde round-trip.
                let src_bytes =
                    std::fs::read(&src_path).map_err(|e| EvidenceAnchorError::Io {
                        path: src_path.clone(),
                        source: e,
                    })?;
                if let Some(parent) = dst_path.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| EvidenceAnchorError::Io {
                        path: parent.to_path_buf(),
                        source: e,
                    })?;
                }
                let tmp = dst_path.with_extension("json.tmp");
                std::fs::write(&tmp, &src_bytes).map_err(|e| EvidenceAnchorError::Io {
                    path: tmp.clone(),
                    source: e,
                })?;
                std::fs::rename(&tmp, &dst_path).map_err(|e| EvidenceAnchorError::Io {
                    path: dst_path.clone(),
                    source: e,
                })?;
                tx_index_updates.insert(tx_id.to_string(), artifact_hash_hex.to_string());
                actions_imported += 1;
            }
            s if s == OUTCOME_WOULD_IMPORT => {
                actions_would_import += 1;
            }
            s if s == OUTCOME_SKIPPED_ALREADY_IMPORTED => {
                actions_skipped_already_imported += 1;
            }
            s if s == OUTCOME_RE_ADDED_TX_INDEX_ENTRY => {
                // Implementation note 2: do NOT rewrite the
                // record file — the byte-equal file is already
                // present. Only the tx_index entry needs to
                // appear, which happens in Pass 1.5.
                tx_index_updates.insert(tx_id.to_string(), artifact_hash_hex.to_string());
                actions_re_added_tx_index_entry += 1;
            }
            s if s == OUTCOME_WOULD_RE_ADD_TX_INDEX_ENTRY => {
                actions_would_re_add_tx_index_entry += 1;
            }
            other => {
                // Should never happen — classify_target_state
                // returns only the closed-set strings above.
                return Err(EvidenceAnchorError::ExportEntryMetadataMismatch {
                    relative_path: action.entry.relative_path.clone(),
                    field: "outcome",
                    computed: other.to_string(),
                    manifest: "closed-set classification".to_string(),
                });
            }
        }
        outcomes.push(AnchorImportActionOutcome {
            artifact_hash_hex: artifact_hash_hex.to_string(),
            tx_id: tx_id.to_string(),
            status: status.to_string(),
            outcome: action.outcome.clone(),
        });
    }

    // Step 5: Pass 1.5 — merge tx_index.json (D3 lock). Skip on
    // dry-run AND when no updates accumulated.
    if !opts.dry_run && !tx_index_updates.is_empty() {
        merge_tx_index(target_registry.root(), &tx_index_updates)?;
    }

    let mode = if opts.dry_run { "dry_run" } else { "apply" };
    Ok(AnchorImportReport {
        export_id: manifest.export_id,
        mode: mode.to_string(),
        actions_imported,
        actions_would_import,
        actions_skipped_already_imported,
        actions_re_added_tx_index_entry,
        actions_would_re_add_tx_index_entry,
        outcomes,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn read_manifest(export_dir: &Path) -> EvidenceAnchorResult<AnchorExportManifest> {
    let path = export_dir.join(EXPORT_MANIFEST_FILENAME);
    let bytes = std::fs::read(&path).map_err(|e| EvidenceAnchorError::Io {
        path: path.clone(),
        source: e,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| EvidenceAnchorError::MalformedJson {
        path,
        source: e,
    })
}

fn required_anchor_record_fields(
    entry: &AnchorExportEntry,
) -> EvidenceAnchorResult<(&str, &str, &str)> {
    let h = entry.artifact_hash_hex.as_deref().ok_or_else(|| {
        EvidenceAnchorError::ExportEntryMetadataMismatch {
            relative_path: entry.relative_path.clone(),
            field: "artifact_hash_hex",
            computed: String::new(),
            manifest: "<absent>".to_string(),
        }
    })?;
    let t = entry.tx_id.as_deref().ok_or_else(|| {
        EvidenceAnchorError::ExportEntryMetadataMismatch {
            relative_path: entry.relative_path.clone(),
            field: "tx_id",
            computed: String::new(),
            manifest: "<absent>".to_string(),
        }
    })?;
    let s = entry.status.as_deref().ok_or_else(|| {
        EvidenceAnchorError::ExportEntryMetadataMismatch {
            relative_path: entry.relative_path.clone(),
            field: "status",
            computed: String::new(),
            manifest: "<absent>".to_string(),
        }
    })?;
    Ok((h, t, s))
}

fn read_and_parse_record(
    export_dir: &Path,
    relative_path: &str,
) -> EvidenceAnchorResult<AnchorRecord> {
    let path = export_dir.join(relative_path);
    let bytes = std::fs::read(&path).map_err(|e| EvidenceAnchorError::Io {
        path: path.clone(),
        source: e,
    })?;
    let record: AnchorRecord =
        serde_json::from_slice(&bytes).map_err(|e| EvidenceAnchorError::MalformedJson {
            path: path.clone(),
            source: e,
        })?;
    // Defense-in-depth: re-run Stage 13.0 signature check on the
    // record we parsed. verify_anchor_export already did this for
    // every anchor_record entry on the way in; we re-run here so
    // future refactors of either side can't drift apart.
    verify_anchor_tx_data(&record.tx_data)?;
    Ok(record)
}

fn cross_check_metadata(
    entry: &AnchorExportEntry,
    record: &AnchorRecord,
) -> EvidenceAnchorResult<()> {
    if let Some(declared) = &entry.artifact_hash_hex {
        if declared != &record.artifact_hash_hex {
            return Err(EvidenceAnchorError::ExportEntryMetadataMismatch {
                relative_path: entry.relative_path.clone(),
                field: "artifact_hash_hex",
                computed: record.artifact_hash_hex.clone(),
                manifest: declared.clone(),
            });
        }
    }
    if let Some(declared) = &entry.tx_id {
        if declared != &record.receipt.tx_id {
            return Err(EvidenceAnchorError::ExportEntryMetadataMismatch {
                relative_path: entry.relative_path.clone(),
                field: "tx_id",
                computed: record.receipt.tx_id.clone(),
                manifest: declared.clone(),
            });
        }
    }
    if let Some(declared) = &entry.status {
        if declared != record.status.as_str() {
            return Err(EvidenceAnchorError::ExportEntryMetadataMismatch {
                relative_path: entry.relative_path.clone(),
                field: "status",
                computed: record.status.as_str().to_string(),
                manifest: declared.clone(),
            });
        }
    }
    Ok(())
}

fn classify_target_state(
    target_registry: &LocalEvidenceAnchorRegistry,
    entry: &AnchorExportEntry,
    _record: &AnchorRecord,
    dry_run: bool,
) -> EvidenceAnchorResult<String> {
    let artifact_hash_hex = entry
        .artifact_hash_hex
        .as_deref()
        .expect("validated by required_anchor_record_fields");
    let tx_id = entry
        .tx_id
        .as_deref()
        .expect("validated by required_anchor_record_fields");
    let expected_blake3 = entry.blake3_hex.as_str();
    let registry_root = target_registry.root();

    let target_record_path = registry_root.join(format!("{artifact_hash_hex}.json"));
    let target_file_present = target_record_path.is_file();

    // Read the on-disk record file's BLAKE3 if present.
    let target_file_blake3 = if target_file_present {
        let bytes =
            std::fs::read(&target_record_path).map_err(|e| EvidenceAnchorError::Io {
                path: target_record_path.clone(),
                source: e,
            })?;
        Some(blake3::hash(&bytes).to_hex().to_string())
    } else {
        None
    };

    // Read the on-disk tx_index.json (if present) directly. We
    // do this via raw JSON to avoid depending on the registry
    // module's private TxIndex type.
    let tx_index_path = registry_root.join(TX_INDEX_FILENAME);
    let existing_mapping_for_tx_id = if tx_index_path.is_file() {
        let bytes = std::fs::read(&tx_index_path).map_err(|e| EvidenceAnchorError::Io {
            path: tx_index_path.clone(),
            source: e,
        })?;
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| EvidenceAnchorError::MalformedJson {
                path: tx_index_path.clone(),
                source: e,
            })?;
        value
            .get("by_tx_id")
            .and_then(|v| v.get(tx_id))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    // Row 5: tx_id collision (HIGHEST precedence). If the
    // existing tx_index entry maps to a DIFFERENT artifact_hash,
    // refuse regardless of file presence.
    if let Some(existing_hash) = &existing_mapping_for_tx_id {
        if existing_hash != artifact_hash_hex {
            return Err(EvidenceAnchorError::ImportTargetExists {
                field: "tx_id",
                artifact_hash_hex: artifact_hash_hex.to_string(),
                tx_id: tx_id.to_string(),
            });
        }
    }

    // Row 4: artifact-hash collision with byte-different record.
    if target_file_present {
        let actual = target_file_blake3.as_deref().unwrap_or("");
        if actual != expected_blake3 {
            return Err(EvidenceAnchorError::ImportTargetExists {
                field: "artifact_hash",
                artifact_hash_hex: artifact_hash_hex.to_string(),
                tx_id: tx_id.to_string(),
            });
        }
        // File present + byte-equal. Distinguish row 2 vs row 3
        // by whether the tx_index already maps the tx_id.
        if existing_mapping_for_tx_id.is_some() {
            // Row 2: fully idempotent. No mutation in apply OR
            // dry-run — same outcome string in both modes.
            return Ok(OUTCOME_SKIPPED_ALREADY_IMPORTED.to_string());
        }
        // Row 3: tx_index re-add only. Record file untouched.
        return Ok(if dry_run {
            OUTCOME_WOULD_RE_ADD_TX_INDEX_ENTRY.to_string()
        } else {
            OUTCOME_RE_ADDED_TX_INDEX_ENTRY.to_string()
        });
    }

    // Row 1: fresh import.
    Ok(if dry_run {
        OUTCOME_WOULD_IMPORT.to_string()
    } else {
        OUTCOME_IMPORTED.to_string()
    })
}

/// Selector-miss refusal (D8 — reuse `anchor_not_found`).
///
/// For each operator-supplied selector value (tx_id /
/// artifact_hash_hex), refuse if NO manifest anchor_record entry
/// matches that value. Status selectors are NOT included — they
/// describe "filter to these statuses", not "find a record with
/// this status"; an empty match-set on `--status submitted` is
/// just zero work.
fn refuse_on_selector_miss(
    manifest: &AnchorExportManifest,
    selection: &AnchorImportSelection,
) -> EvidenceAnchorResult<()> {
    let mut missing: Vec<String> = Vec::new();
    let anchor_record_str = AnchorExportEntryKind::AnchorRecord.as_str();
    let anchor_entries: Vec<&AnchorExportEntry> = manifest
        .entries
        .iter()
        .filter(|e| e.kind == anchor_record_str)
        .collect();
    for hash in &selection.artifact_hashes {
        if !anchor_entries
            .iter()
            .any(|e| e.artifact_hash_hex.as_deref() == Some(hash.as_str()))
        {
            missing.push(format!("artifact_hash_hex={hash}"));
        }
    }
    for tx_id in &selection.tx_ids {
        if !anchor_entries
            .iter()
            .any(|e| e.tx_id.as_deref() == Some(tx_id.as_str()))
        {
            missing.push(format!("tx_id={tx_id}"));
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(EvidenceAnchorError::AnchorNotFound {
            selector: missing.join(", "),
        })
    }
}

/// Merge new `(tx_id → artifact_hash_hex)` entries on top of the
/// existing `tx_index.json`. Atomic temp+rename. Preserves
/// unrelated entries (D3 lock).
fn merge_tx_index(
    registry_root: &Path,
    updates: &BTreeMap<String, String>,
) -> EvidenceAnchorResult<()> {
    if updates.is_empty() {
        return Ok(());
    }
    let path = registry_root.join(TX_INDEX_FILENAME);
    let mut value: serde_json::Value = if path.is_file() {
        let bytes = std::fs::read(&path).map_err(|e| EvidenceAnchorError::Io {
            path: path.clone(),
            source: e,
        })?;
        serde_json::from_slice(&bytes).map_err(|e| EvidenceAnchorError::MalformedJson {
            path: path.clone(),
            source: e,
        })?
    } else {
        serde_json::json!({ "by_tx_id": {} })
    };
    let map = value
        .get_mut("by_tx_id")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| EvidenceAnchorError::MalformedJson {
            path: path.clone(),
            source: serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tx_index.json missing by_tx_id field",
            )),
        })?;
    for (tx_id, hash_hex) in updates {
        map.insert(tx_id.clone(), serde_json::Value::String(hash_hex.clone()));
    }
    let new_bytes = serde_json::to_vec_pretty(&value).map_err(|e| {
        EvidenceAnchorError::MalformedJson {
            path: path.clone(),
            source: e,
        }
    })?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &new_bytes).map_err(|e| EvidenceAnchorError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, &path).map_err(|e| EvidenceAnchorError::Io {
        path: path.clone(),
        source: e,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_selection_matches_every_entry() {
        let sel = AnchorImportSelection::default();
        let e = AnchorExportEntry {
            kind: "anchor_record".to_string(),
            relative_path: "anchors/aa.json".to_string(),
            blake3_hex: "0".repeat(64),
            bytes: 0,
            artifact_hash_hex: Some("aa".repeat(32)),
            tx_id: Some("anchor-1".to_string()),
            status: Some("submitted".to_string()),
            source_basename: None,
        };
        assert!(sel.matches_entry(&e));
        assert!(sel.is_empty());
    }

    #[test]
    fn outcome_strings_are_closed_set() {
        assert_eq!(OUTCOME_IMPORTED, "imported");
        assert_eq!(OUTCOME_WOULD_IMPORT, "would_import");
        assert_eq!(OUTCOME_SKIPPED_ALREADY_IMPORTED, "skipped_already_imported");
        assert_eq!(OUTCOME_RE_ADDED_TX_INDEX_ENTRY, "re_added_tx_index_entry");
        assert_eq!(
            OUTCOME_WOULD_RE_ADD_TX_INDEX_ENTRY,
            "would_re_add_tx_index_entry"
        );
    }
}
