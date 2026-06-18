//! Phase 5 Stage 13.5 — local-only export / import verification
//! for integrity-evidence anchor records.
//!
//! Stage 13.5 ships a portable handoff form: a JSON manifest plus
//! a small directory tree of copied bytes. Verify re-reads the
//! manifest, hashes every copied file, parses every anchor
//! record, and confirms internal consistency.
//!
//! ## Locked Stage 13.5 invariants
//!
//! - **Fully local.** No chain RPCs. The Stage 13.0 wire / domain /
//!   canonical-bytes / signing surface is **read only**; not a
//!   byte is re-signed.
//! - **Default read-only.** Both export (writes to operator-
//!   supplied `<export-out>`) and verify (pure read) leave the
//!   anchor registry untouched.
//! - **No `anchor_registry_dir` in the manifest.** A portable
//!   handoff artifact does not leak host-local path layout
//!   (Stage 13.5 REJECT-fix Finding 2). Operator provenance lives
//!   in `label` / `notes`.
//! - **At-least-one-selector required** at the CLI layer. No
//!   accidental "export everything" (Q4).
//! - **Selector misses route through `anchor_not_found`** — no
//!   new "record_not_found" tag (Stage 13.5 REJECT-fix Finding 3).
//! - **What we prove**: anchor records' submitter signatures
//!   (via `verify_anchor_tx_data` verbatim) and the artifact-hash
//!   binding `blake3(artifact_bytes) == record.tx_data.digest.artifact_hash`
//!   for paired entries.
//! - **What we do NOT prove**: the Stage 12.25 wrapper signer.
//!   That binding lives in a signed-chain-report (Stage 12.25's
//!   own signature verification is out of scope here, Q7). The
//!   `verify_anchor_record_with_artifact_bytes` helper is private
//!   to this module specifically so it does NOT call
//!   `verify_anchor_file_against_artifact_bytes` — passing
//!   `digest.signer_pubkey` back as the "expected wrapper signer"
//!   would make the signer-binding check tautological (Stage 13.5
//!   REJECT-fix Finding 4).
//!
//! ## Manifest schema (v1)
//!
//! ```text
//! {
//!   "schema_version": 1,
//!   "export_id": "<16-lower-hex>",
//!   "created_at_utc": "2026-06-17T22:00:00Z",
//!   "label": "...",
//!   "notes": "...",
//!   "entries": [
//!     {
//!       "kind": "anchor_record" | "artifact_bytes" | "signed_chain_report",
//!       "relative_path": "anchors/<64hex>.json",
//!       "blake3_hex": "<64hex>",
//!       "bytes": 1234,
//!       "artifact_hash_hex": "<64hex>?",
//!       "tx_id": "<string>?",
//!       "status": "submitted | included | finalized | failed",
//!       "source_basename": "<string>?"
//!     }
//!   ],
//!   "export_manifest_hash": "<64-lower-hex>"
//! }
//! ```
//!
//! `export_id = lower_hex(BLAKE3(record_set_hash || "||" || created_at_utc))[..16]`
//! (timestamp included so back-to-back exports of the same
//! selection produce distinct ids; tests inject `created_at_utc`).
//! `export_manifest_hash = BLAKE3(canonical JSON of manifest with
//! this field blanked)`.
//!
//! ## Verify preflight ordering (fixed)
//!
//! 1. schema-version → `unsupported_export_manifest_schema_version`
//! 2. manifest-hash → `export_manifest_hash_mismatch`
//! 3. per-entry `relative_path` shape → `export_invalid_path`
//! 4. per-entry file presence → `io`
//! 5. per-entry BLAKE3 + length → `export_blake3_mismatch`
//! 6. anchor_record parse → `malformed_json`, then
//!    `verify_anchor_tx_data` → `unsupported_anchor_schema_version`
//!    / `submitter_signature_invalid`
//! 7. anchor_record metadata cross-check → `export_entry_metadata_mismatch`
//! 8. artifact_bytes — blake3 already proved == declared
//!    artifact_hash by step 5; plus cross-pair check against
//!    record digest → `export_entry_metadata_mismatch`
//! 9. paired artifact-hash binding → `export_entry_metadata_mismatch`
//! 10. signed_chain_report — BLAKE3 only (step 5); Stage 12.25
//!     own-signature verification out of scope
//! 11. `--strict` mode → `export_strict_mode_artifact_bytes_missing`

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{EvidenceAnchorError, EvidenceAnchorResult};
use crate::evidence_anchor::registry::{
    AnchorRecord, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
};
use crate::evidence_anchor::wire::{anchor_hex_lower, verify_anchor_tx_data};

// ── Schema constants ──────────────────────────────────────────────────────────

/// Stage 13.5 export-manifest JSON schema version. Persisted
/// manifests declare this. Verify refuses on mismatch.
pub const EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Manifest filename, written at `<export-dir>/<EXPORT_MANIFEST_FILENAME>`.
pub const EXPORT_MANIFEST_FILENAME: &str = "evidence_anchor_export_manifest.json";

const ANCHORS_SUBDIR: &str = "anchors";
const ARTIFACTS_SUBDIR: &str = "artifacts";
const SIGNED_CHAIN_REPORTS_SUBDIR: &str = "signed_chain_reports";

const EXPORT_MANIFEST_HASH_BLANK: &str = "";

// ── Closed entry-kind enum ────────────────────────────────────────────────────

/// Closed taxonomy of export-manifest entry kinds. Mirrors the
/// Stage 13.4 `AnchorCleanupActionKind` shape — a single enum
/// drives both serde and the per-kind path validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorExportEntryKind {
    /// Verbatim copy of `<registry>/<artifact_hash_hex>.json`.
    AnchorRecord,
    /// Operator-supplied raw artifact bytes, named after the
    /// claimed artifact hash. Optional.
    ArtifactBytes,
    /// Operator-supplied Stage 12.25 signed-chain-report file.
    /// Optional. Stage 13.5 verifies BLAKE3 only — Stage 12.25
    /// own-signature verification is out of scope.
    SignedChainReport,
}

impl AnchorExportEntryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            AnchorExportEntryKind::AnchorRecord => "anchor_record",
            AnchorExportEntryKind::ArtifactBytes => "artifact_bytes",
            AnchorExportEntryKind::SignedChainReport => "signed_chain_report",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "anchor_record" => Some(AnchorExportEntryKind::AnchorRecord),
            "artifact_bytes" => Some(AnchorExportEntryKind::ArtifactBytes),
            "signed_chain_report" => Some(AnchorExportEntryKind::SignedChainReport),
            _ => None,
        }
    }

    fn subdir(self) -> &'static str {
        match self {
            AnchorExportEntryKind::AnchorRecord => ANCHORS_SUBDIR,
            AnchorExportEntryKind::ArtifactBytes => ARTIFACTS_SUBDIR,
            AnchorExportEntryKind::SignedChainReport => SIGNED_CHAIN_REPORTS_SUBDIR,
        }
    }
}

// ── Manifest structs ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorExportManifest {
    pub schema_version: u32,
    pub export_id: String,
    pub created_at_utc: String,
    pub label: String,
    pub notes: String,
    pub entries: Vec<AnchorExportEntry>,
    /// BLAKE3 over canonical JSON of this manifest with
    /// `export_manifest_hash` blanked. Verify recomputes and
    /// refuses with [`EvidenceAnchorError::ExportManifestHashMismatch`].
    pub export_manifest_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorExportEntry {
    /// Closed-set string. Backed by [`AnchorExportEntryKind`].
    pub kind: String,
    /// Forward-slash, root-relative path under `<export-dir>`.
    /// Validated per kind.
    pub relative_path: String,
    pub blake3_hex: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_hash_hex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tx_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_basename: Option<String>,
}

// ── Plan / apply / verify options ────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnchorExportSelection {
    /// Filter by `LocalAnchorStatus` (OR within this kind). Empty
    /// means no status constraint.
    pub statuses: Vec<LocalAnchorStatus>,
    /// Filter by `receipt.tx_id` (OR within this kind). Empty
    /// means no tx_id constraint.
    pub tx_ids: Vec<String>,
    /// Filter by `artifact_hash_hex` (OR within this kind).
    /// Empty means no hash constraint.
    pub artifact_hashes: Vec<String>,
}

impl AnchorExportSelection {
    /// True iff this selection has any selector kind populated.
    /// The CLI refuses an empty selection at the clap layer; the
    /// library only checks emptiness for assertion convenience.
    pub fn is_empty(&self) -> bool {
        self.statuses.is_empty()
            && self.tx_ids.is_empty()
            && self.artifact_hashes.is_empty()
    }

    fn matches(&self, record: &AnchorRecord) -> bool {
        if !self.statuses.is_empty()
            && !self.statuses.iter().any(|s| status_kind_eq(s, &record.status))
        {
            return false;
        }
        if !self.tx_ids.is_empty()
            && !self.tx_ids.iter().any(|t| t == &record.receipt.tx_id)
        {
            return false;
        }
        if !self.artifact_hashes.is_empty()
            && !self
                .artifact_hashes
                .iter()
                .any(|h| h == &record.artifact_hash_hex)
        {
            return false;
        }
        true
    }
}

fn status_kind_eq(a: &LocalAnchorStatus, b: &LocalAnchorStatus) -> bool {
    a.as_str() == b.as_str()
}

/// Operator-supplied claim that `path` carries raw artifact
/// bytes whose `blake3(bytes) == artifact_hash_hex`. Validated at
/// export time (Stage 13.5 REJECT-fix Q3 — pair form is explicit).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactBytesInclusion {
    pub path: PathBuf,
    pub artifact_hash_hex: String,
}

pub struct AnchorExportOptions<'a> {
    /// Destination directory. Must be non-existent or empty
    /// (no-clobber, Q8). Apply will `mkdir -p` it.
    pub export_out: &'a Path,
    /// Record selection. At least one selector populated
    /// (the CLI enforces this at the clap layer).
    pub selection: &'a AnchorExportSelection,
    /// Optional operator-supplied artifact-bytes claims. Each
    /// is copied to `artifacts/<artifact_hash_hex>` after the
    /// claimed BLAKE3 is verified.
    pub artifact_bytes_inclusions: &'a [ArtifactBytesInclusion],
    /// Optional Stage 12.25 signed-chain-report files. Each is
    /// copied verbatim to `signed_chain_reports/<basename>`.
    /// Stage 13.5 verifies BLAKE3 only — Stage 12.25
    /// own-signature verification is out of scope.
    pub signed_chain_report_paths: &'a [PathBuf],
    /// Operator-supplied free-form provenance. Default `""`.
    pub label: &'a str,
    /// Operator-supplied free-form provenance. Default `""`.
    pub notes: &'a str,
    /// RFC 3339 UTC. Stamped into `created_at_utc` and folded
    /// into `export_id`. Caller injects so tests get
    /// deterministic ids.
    pub now_utc: &'a str,
}

pub struct AnchorExportVerifyOptions<'a> {
    /// Directory holding `evidence_anchor_export_manifest.json`
    /// + the copied bytes subtree.
    pub export_dir: &'a Path,
    /// When true, every `anchor_record` entry must have a paired
    /// `artifact_bytes` entry for the same `artifact_hash_hex`.
    /// Missing pairings refuse with
    /// [`EvidenceAnchorError::ExportStrictModeArtifactBytesMissing`].
    pub strict: bool,
}

// ── Plan + reports ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AnchorExportPlan {
    pub export_id: String,
    pub created_at_utc: String,
    pub entries: Vec<PlannedEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlannedEntry {
    pub kind: String,
    pub relative_path: String,
    pub bytes: u64,
    pub source_absolute_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_hash_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tx_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_basename: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AnchorExportReport {
    pub export_id: String,
    pub anchors_written: u32,
    pub artifact_bytes_written: u32,
    pub signed_chain_reports_written: u32,
    pub manifest_relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AnchorExportVerifyReport {
    pub export_id: String,
    pub entries_verified: u32,
    pub anchor_records_verified: u32,
    pub artifact_bytes_verified: u32,
    pub signed_chain_reports_verified: u32,
    pub pairings_artifact_hash_bound: u32,
    pub strict: bool,
}

// ── Plan ──────────────────────────────────────────────────────────────────────

/// Compute the export plan without touching the filesystem.
///
/// Errors:
/// - [`EvidenceAnchorError::AnchorNotFound`] when a `--tx-id` or
///   `--artifact-hash-hex` selector points at a record that does
///   not exist (Stage 13.5 REJECT-fix Finding 3 — correct tag).
/// - [`EvidenceAnchorError::ExportEntryMetadataMismatch`] when a
///   `--include-artifact-bytes <path>:<hash>` pair's file BLAKE3
///   does not equal the operator-claimed hash.
/// - [`EvidenceAnchorError::Io`] on read failures.
pub fn plan_anchor_export(
    registry: &LocalEvidenceAnchorRegistry,
    opts: &AnchorExportOptions<'_>,
) -> EvidenceAnchorResult<AnchorExportPlan> {
    // Step 1: refuse on selector misses for tx_id / artifact_hash.
    let mut missing: Vec<String> = Vec::new();
    for hash in &opts.selection.artifact_hashes {
        match registry.load_by_artifact_hash(hash) {
            Ok(Some(_)) => {}
            Ok(None) => missing.push(format!("artifact_hash_hex={hash}")),
            Err(e) => {
                return Err(EvidenceAnchorError::Io {
                    path: registry.root().to_path_buf(),
                    source: e,
                });
            }
        }
    }
    for tx_id in &opts.selection.tx_ids {
        match registry.load_by_tx_id(tx_id) {
            Ok(Some(_)) => {}
            Ok(None) => missing.push(format!("tx_id={tx_id}")),
            Err(e) => {
                return Err(EvidenceAnchorError::Io {
                    path: registry.root().to_path_buf(),
                    source: e,
                });
            }
        }
    }
    if !missing.is_empty() {
        return Err(EvidenceAnchorError::AnchorNotFound {
            selector: missing.join(", "),
        });
    }

    // Step 2: filter records.
    let all_records = registry.list().map_err(|e| EvidenceAnchorError::Io {
        path: registry.root().to_path_buf(),
        source: e,
    })?;
    let mut selected: Vec<AnchorRecord> = all_records
        .into_iter()
        .filter(|r| opts.selection.matches(r))
        .collect();
    selected.sort_by(|a, b| a.artifact_hash_hex.cmp(&b.artifact_hash_hex));

    // Step 3: validate operator-supplied artifact_bytes inclusions
    // (BLAKE3-of-file must equal claimed hash) and plan entries.
    let mut planned: Vec<PlannedEntry> = Vec::new();

    // anchors/...
    for record in &selected {
        let registry_record_path =
            registry.root().join(format!("{}.json", record.artifact_hash_hex));
        let bytes_len = std::fs::metadata(&registry_record_path)
            .map_err(|e| EvidenceAnchorError::Io {
                path: registry_record_path.clone(),
                source: e,
            })?
            .len();
        planned.push(PlannedEntry {
            kind: AnchorExportEntryKind::AnchorRecord.as_str().to_string(),
            relative_path: format!(
                "{ANCHORS_SUBDIR}/{}.json",
                record.artifact_hash_hex
            ),
            bytes: bytes_len,
            source_absolute_path: registry_record_path.to_string_lossy().into_owned(),
            artifact_hash_hex: Some(record.artifact_hash_hex.clone()),
            tx_id: Some(record.receipt.tx_id.clone()),
            status: Some(record.status.as_str().to_string()),
            source_basename: None,
        });
    }

    // artifacts/...
    for inc in opts.artifact_bytes_inclusions {
        let bytes = std::fs::read(&inc.path).map_err(|e| EvidenceAnchorError::Io {
            path: inc.path.clone(),
            source: e,
        })?;
        let computed = blake3::hash(&bytes).to_hex().to_string();
        if computed != inc.artifact_hash_hex {
            return Err(EvidenceAnchorError::ExportEntryMetadataMismatch {
                relative_path: format!(
                    "{ARTIFACTS_SUBDIR}/{}",
                    inc.artifact_hash_hex
                ),
                field: "artifact_hash_hex",
                computed,
                manifest: inc.artifact_hash_hex.clone(),
            });
        }
        planned.push(PlannedEntry {
            kind: AnchorExportEntryKind::ArtifactBytes.as_str().to_string(),
            relative_path: format!("{ARTIFACTS_SUBDIR}/{}", inc.artifact_hash_hex),
            bytes: bytes.len() as u64,
            source_absolute_path: inc.path.to_string_lossy().into_owned(),
            artifact_hash_hex: Some(inc.artifact_hash_hex.clone()),
            tx_id: None,
            status: None,
            source_basename: None,
        });
    }

    // signed_chain_reports/...
    for path in opts.signed_chain_report_paths {
        let basename = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| EvidenceAnchorError::ExportInvalidPath {
                entry_kind: AnchorExportEntryKind::SignedChainReport.as_str(),
                relative_path: path.to_string_lossy().into_owned(),
                reason: "signed-chain-report path has no valid basename",
            })?
            .to_string();
        validate_safe_basename(&basename, AnchorExportEntryKind::SignedChainReport)?;
        let bytes_len = std::fs::metadata(path)
            .map_err(|e| EvidenceAnchorError::Io {
                path: path.clone(),
                source: e,
            })?
            .len();
        planned.push(PlannedEntry {
            kind: AnchorExportEntryKind::SignedChainReport.as_str().to_string(),
            relative_path: format!("{SIGNED_CHAIN_REPORTS_SUBDIR}/{basename}"),
            bytes: bytes_len,
            source_absolute_path: path.to_string_lossy().into_owned(),
            artifact_hash_hex: None,
            tx_id: None,
            status: None,
            source_basename: Some(basename),
        });
    }

    // Sort by relative_path ascending (deterministic).
    planned.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    // Duplicate-`relative_path` defense. Two source files sharing
    // a destination path would have the apply step overwrite the
    // earlier file with the later one, yet both entries would
    // land in the manifest with their own `blake3_hex` — the
    // export would fail its own verifier with
    // `export_blake3_mismatch`. Common cause: two
    // `--include-signed-chain-report` paths with the same
    // basename. Refuse at plan time, before any FS write, via
    // the closed `export_invalid_path` tag.
    for pair in planned.windows(2) {
        if pair[0].relative_path == pair[1].relative_path {
            let kind_str: &'static str = match AnchorExportEntryKind::parse(&pair[0].kind) {
                Some(AnchorExportEntryKind::AnchorRecord) => "anchor_record",
                Some(AnchorExportEntryKind::ArtifactBytes) => "artifact_bytes",
                Some(AnchorExportEntryKind::SignedChainReport) => "signed_chain_report",
                None => "unknown",
            };
            return Err(EvidenceAnchorError::ExportInvalidPath {
                entry_kind: kind_str,
                relative_path: pair[0].relative_path.clone(),
                reason: "duplicate relative_path",
            });
        }
    }

    // Compute record_set_hash and export_id.
    let record_set_hash = compute_record_set_hash(&selected);
    let export_id = compute_export_id(&record_set_hash, opts.now_utc);

    Ok(AnchorExportPlan {
        export_id,
        created_at_utc: opts.now_utc.to_string(),
        entries: planned,
    })
}

// ── Apply (export to disk) ────────────────────────────────────────────────────

/// Write the export tree to `opts.export_out`. The directory must
/// be non-existent or empty (no-clobber, Q8). Returns a report
/// listing per-kind written counts and the manifest's
/// relative-path within the export dir.
///
/// On-disk layout:
///
/// ```text
/// <export-out>/
/// ├── evidence_anchor_export_manifest.json   # written LAST (durability fence)
/// ├── anchors/<artifact_hash_hex>.json
/// ├── artifacts/<artifact_hash_hex>          # only when included
/// └── signed_chain_reports/<basename>        # only when included
/// ```
///
/// Atomic temp+rename per file. Manifest written last so a crash
/// before manifest landing leaves the operator with an incomplete
/// tree; verify will not run because the manifest is missing.
pub fn apply_anchor_export(
    registry: &LocalEvidenceAnchorRegistry,
    opts: &AnchorExportOptions<'_>,
) -> EvidenceAnchorResult<AnchorExportReport> {
    // No-clobber: refuse if export_out exists AND is non-empty.
    if opts.export_out.exists() {
        let mut iter = std::fs::read_dir(opts.export_out)
            .map_err(|e| EvidenceAnchorError::Io {
                path: opts.export_out.to_path_buf(),
                source: e,
            })?;
        if iter.next().is_some() {
            return Err(EvidenceAnchorError::Io {
                path: opts.export_out.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "export_out exists and is non-empty: {}",
                        opts.export_out.display()
                    ),
                ),
            });
        }
    }
    std::fs::create_dir_all(opts.export_out).map_err(|e| EvidenceAnchorError::Io {
        path: opts.export_out.to_path_buf(),
        source: e,
    })?;

    let plan = plan_anchor_export(registry, opts)?;

    let mut entries: Vec<AnchorExportEntry> = Vec::with_capacity(plan.entries.len());
    let mut anchors_written: u32 = 0;
    let mut artifact_bytes_written: u32 = 0;
    let mut signed_chain_reports_written: u32 = 0;

    for planned in &plan.entries {
        let kind = AnchorExportEntryKind::parse(&planned.kind).ok_or_else(|| {
            EvidenceAnchorError::ExportInvalidPath {
                entry_kind: "unknown",
                relative_path: planned.relative_path.clone(),
                reason: "unknown planned entry kind",
            }
        })?;
        let src = PathBuf::from(&planned.source_absolute_path);
        let bytes = std::fs::read(&src).map_err(|e| EvidenceAnchorError::Io {
            path: src.clone(),
            source: e,
        })?;
        let blake3_hex = blake3::hash(&bytes).to_hex().to_string();
        let dst = opts.export_out.join(&planned.relative_path);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|e| EvidenceAnchorError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let tmp = with_tmp_suffix(&dst);
        std::fs::write(&tmp, &bytes).map_err(|e| EvidenceAnchorError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp, &dst).map_err(|e| EvidenceAnchorError::Io {
            path: dst.clone(),
            source: e,
        })?;

        match kind {
            AnchorExportEntryKind::AnchorRecord => anchors_written += 1,
            AnchorExportEntryKind::ArtifactBytes => artifact_bytes_written += 1,
            AnchorExportEntryKind::SignedChainReport => {
                signed_chain_reports_written += 1
            }
        }

        entries.push(AnchorExportEntry {
            kind: planned.kind.clone(),
            relative_path: planned.relative_path.clone(),
            blake3_hex,
            bytes: planned.bytes,
            artifact_hash_hex: planned.artifact_hash_hex.clone(),
            tx_id: planned.tx_id.clone(),
            status: planned.status.clone(),
            source_basename: planned.source_basename.clone(),
        });
    }

    // entries already sorted by plan ordering (relative_path asc).
    let mut manifest = AnchorExportManifest {
        schema_version: EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION,
        export_id: plan.export_id.clone(),
        created_at_utc: plan.created_at_utc.clone(),
        label: opts.label.to_string(),
        notes: opts.notes.to_string(),
        entries,
        export_manifest_hash: EXPORT_MANIFEST_HASH_BLANK.to_string(),
    };
    manifest.export_manifest_hash = compute_export_manifest_hash(&manifest)?;

    let manifest_path = opts.export_out.join(EXPORT_MANIFEST_FILENAME);
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| {
        EvidenceAnchorError::MalformedJson {
            path: manifest_path.clone(),
            source: e,
        }
    })?;
    let tmp = with_tmp_suffix(&manifest_path);
    std::fs::write(&tmp, &manifest_bytes).map_err(|e| EvidenceAnchorError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, &manifest_path).map_err(|e| EvidenceAnchorError::Io {
        path: manifest_path.clone(),
        source: e,
    })?;

    Ok(AnchorExportReport {
        export_id: plan.export_id,
        anchors_written,
        artifact_bytes_written,
        signed_chain_reports_written,
        manifest_relative_path: EXPORT_MANIFEST_FILENAME.to_string(),
    })
}

// ── Verify ────────────────────────────────────────────────────────────────────

/// Verify an export directory.
///
/// Preflight order (fixed; each step refuses with the closed
/// taxonomy and stops):
///
/// 1. parse manifest → `malformed_json` on bad JSON.
/// 2. schema-version → `unsupported_export_manifest_schema_version`.
/// 3. manifest-hash → `export_manifest_hash_mismatch`.
/// 4. per-entry path validation → `export_invalid_path`.
/// 5. per-entry file presence → `io`.
/// 6. per-entry BLAKE3 + length → `export_blake3_mismatch`.
/// 7. anchor_record entries — parse → `malformed_json`;
///    `verify_anchor_tx_data` → `unsupported_anchor_schema_version`
///    / `submitter_signature_invalid`.
/// 8. anchor_record metadata cross-check (`artifact_hash_hex` /
///    `tx_id` / `status` vs record's own fields) →
///    `export_entry_metadata_mismatch`.
/// 9. artifact_bytes entries — `entry.artifact_hash_hex` must
///    equal `entry.blake3_hex` (the file's BLAKE3 IS its claimed
///    hash) → `export_entry_metadata_mismatch`.
/// 10. paired (anchor_record + artifact_bytes for same
///     `artifact_hash_hex`) — artifact-hash binding:
///     `hex_lower(record.tx_data.digest.artifact_hash) ==
///     entry_artifact_bytes.artifact_hash_hex`. Honest check
///     only — wrapper-signer binding is out of scope. →
///     `export_entry_metadata_mismatch`.
/// 11. signed_chain_report — BLAKE3 only (covered by step 6).
///     Stage 12.25 own-signature verification is out of scope.
/// 12. `--strict` — every anchor_record must have a paired
///     artifact_bytes for the same `artifact_hash_hex` →
///     `export_strict_mode_artifact_bytes_missing`.
pub fn verify_anchor_export(
    opts: &AnchorExportVerifyOptions<'_>,
) -> EvidenceAnchorResult<AnchorExportVerifyReport> {
    let manifest_path = opts.export_dir.join(EXPORT_MANIFEST_FILENAME);
    let bytes = std::fs::read(&manifest_path).map_err(|e| EvidenceAnchorError::Io {
        path: manifest_path.clone(),
        source: e,
    })?;
    let manifest: AnchorExportManifest =
        serde_json::from_slice(&bytes).map_err(|e| EvidenceAnchorError::MalformedJson {
            path: manifest_path.clone(),
            source: e,
        })?;

    // Step 2: schema-version.
    if manifest.schema_version != EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION {
        return Err(EvidenceAnchorError::ExportManifestSchemaUnsupported {
            got: manifest.schema_version,
            expected: EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION,
        });
    }

    // Step 3: manifest-hash.
    let computed = compute_export_manifest_hash(&manifest)?;
    if computed != manifest.export_manifest_hash {
        return Err(EvidenceAnchorError::ExportManifestHashMismatch {
            computed,
            expected: manifest.export_manifest_hash.clone(),
        });
    }

    let mut anchor_records_by_hash: BTreeMap<String, &AnchorExportEntry> = BTreeMap::new();
    let mut artifact_bytes_by_hash: BTreeMap<String, &AnchorExportEntry> = BTreeMap::new();
    let mut anchor_records_verified: u32 = 0;
    let mut artifact_bytes_verified: u32 = 0;
    let mut signed_chain_reports_verified: u32 = 0;
    let mut pairings_artifact_hash_bound: u32 = 0;

    // Step 4: per-entry path validation (BEFORE any FS read).
    for entry in &manifest.entries {
        let kind = AnchorExportEntryKind::parse(&entry.kind).ok_or_else(|| {
            EvidenceAnchorError::ExportInvalidPath {
                entry_kind: "unknown",
                relative_path: entry.relative_path.clone(),
                reason: "unknown manifest entry kind",
            }
        })?;
        validate_relative_path_for_kind(&entry.relative_path, kind)?;
    }

    // Step 4b: duplicate-`relative_path` defense. A hand-edited
    // manifest with two entries pointing at the same destination
    // would lead to one read but two BLAKE3 expectations — the
    // refusal is deterministic to the FIRST duplicate seen in
    // sorted order. Pinned by the export-side dedup, but verify
    // re-checks as defense-in-depth against operator-supplied
    // manifests.
    {
        let mut by_path: BTreeMap<&str, &AnchorExportEntry> = BTreeMap::new();
        for entry in &manifest.entries {
            if let Some(prev) = by_path.insert(entry.relative_path.as_str(), entry) {
                let kind_str: &'static str = match AnchorExportEntryKind::parse(&prev.kind) {
                    Some(AnchorExportEntryKind::AnchorRecord) => "anchor_record",
                    Some(AnchorExportEntryKind::ArtifactBytes) => "artifact_bytes",
                    Some(AnchorExportEntryKind::SignedChainReport) => "signed_chain_report",
                    None => "unknown",
                };
                return Err(EvidenceAnchorError::ExportInvalidPath {
                    entry_kind: kind_str,
                    relative_path: entry.relative_path.clone(),
                    reason: "duplicate relative_path",
                });
            }
        }
    }

    // Step 5-10: per-entry FS read + hashing + per-kind checks.
    for entry in &manifest.entries {
        let kind = AnchorExportEntryKind::parse(&entry.kind).expect("validated above");
        let entry_path = opts.export_dir.join(&entry.relative_path);

        let file_bytes = std::fs::read(&entry_path).map_err(|e| EvidenceAnchorError::Io {
            path: entry_path.clone(),
            source: e,
        })?;

        // Step 5: BLAKE3 + length.
        let computed_blake3 = blake3::hash(&file_bytes).to_hex().to_string();
        if computed_blake3 != entry.blake3_hex
            || file_bytes.len() as u64 != entry.bytes
        {
            return Err(EvidenceAnchorError::ExportBlake3Mismatch {
                relative_path: entry.relative_path.clone(),
                computed: computed_blake3,
                expected: entry.blake3_hex.clone(),
            });
        }

        match kind {
            AnchorExportEntryKind::AnchorRecord => {
                // Step 6 (a): parse.
                let record: AnchorRecord =
                    serde_json::from_slice(&file_bytes).map_err(|e| {
                        EvidenceAnchorError::MalformedJson {
                            path: entry_path.clone(),
                            source: e,
                        }
                    })?;
                // Step 6 (b): verify_anchor_tx_data (Stage 13.0
                // verifier — schema + submitter signature).
                verify_anchor_tx_data(&record.tx_data)?;
                // Step 7: metadata cross-check.
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
                anchor_records_by_hash.insert(record.artifact_hash_hex.clone(), entry);
                anchor_records_verified += 1;
            }
            AnchorExportEntryKind::ArtifactBytes => {
                // Step 8: blake3 == declared artifact_hash_hex.
                if let Some(declared) = &entry.artifact_hash_hex {
                    if &computed_blake3 != declared {
                        return Err(EvidenceAnchorError::ExportEntryMetadataMismatch {
                            relative_path: entry.relative_path.clone(),
                            field: "artifact_hash_hex",
                            computed: computed_blake3.clone(),
                            manifest: declared.clone(),
                        });
                    }
                    artifact_bytes_by_hash.insert(declared.clone(), entry);
                } else {
                    // Missing artifact_hash_hex on an
                    // artifact_bytes entry is a manifest shape
                    // violation; refuse via the metadata-
                    // mismatch tag with a clear field.
                    return Err(EvidenceAnchorError::ExportEntryMetadataMismatch {
                        relative_path: entry.relative_path.clone(),
                        field: "artifact_hash_hex",
                        computed: computed_blake3.clone(),
                        manifest: "<absent>".to_string(),
                    });
                }
                artifact_bytes_verified += 1;
            }
            AnchorExportEntryKind::SignedChainReport => {
                // Step 11: BLAKE3 only (covered by step 5).
                signed_chain_reports_verified += 1;
            }
        }
    }

    // Step 10: paired artifact-hash binding for matching pairs.
    let paired_hashes: Vec<String> = anchor_records_by_hash
        .keys()
        .filter(|h| artifact_bytes_by_hash.contains_key(*h))
        .cloned()
        .collect();
    for hash in &paired_hashes {
        let anchor_entry = anchor_records_by_hash
            .get(hash)
            .expect("anchor entry present");
        let bytes_entry = artifact_bytes_by_hash
            .get(hash)
            .expect("bytes entry present");
        // Re-read both to perform the honest artifact-hash
        // binding check using the private helper that does NOT
        // include the tautological signer-self check (Stage 13.5
        // REJECT-fix Finding 4).
        let record_path = opts.export_dir.join(&anchor_entry.relative_path);
        let bytes_path = opts.export_dir.join(&bytes_entry.relative_path);
        let record_bytes =
            std::fs::read(&record_path).map_err(|e| EvidenceAnchorError::Io {
                path: record_path.clone(),
                source: e,
            })?;
        let record: AnchorRecord =
            serde_json::from_slice(&record_bytes).map_err(|e| {
                EvidenceAnchorError::MalformedJson {
                    path: record_path.clone(),
                    source: e,
                }
            })?;
        let artifact_bytes =
            std::fs::read(&bytes_path).map_err(|e| EvidenceAnchorError::Io {
                path: bytes_path.clone(),
                source: e,
            })?;
        verify_anchor_record_with_artifact_bytes(&record, &artifact_bytes)?;
        pairings_artifact_hash_bound += 1;
    }

    // Step 12: --strict — every anchor_record must have a paired
    // artifact_bytes.
    if opts.strict {
        for (hash, entry) in &anchor_records_by_hash {
            if !artifact_bytes_by_hash.contains_key(hash) {
                return Err(EvidenceAnchorError::ExportStrictModeArtifactBytesMissing {
                    anchor_record_relative_path: entry.relative_path.clone(),
                    artifact_hash_hex: hash.clone(),
                });
            }
        }
    }

    Ok(AnchorExportVerifyReport {
        export_id: manifest.export_id,
        entries_verified: manifest.entries.len() as u32,
        anchor_records_verified,
        artifact_bytes_verified,
        signed_chain_reports_verified,
        pairings_artifact_hash_bound,
        strict: opts.strict,
    })
}

// ── Private artifact-hash binding helper ──────────────────────────────────────

/// Stage 13.5 private artifact-hash binding helper.
///
/// Does the two genuine cross-pair checks for an
/// `(anchor_record, artifact_bytes)` pair, and only those:
///
/// 1. `blake3(artifact_bytes) == record.tx_data.digest.artifact_hash` — the
///    artifact-hash binding.
/// 2. `verify_anchor_tx_data(&record.tx_data)` — re-runs the Stage
///    13.0 schema + submitter signature check (defense-in-depth;
///    the verify pass already ran this once on this record).
///
/// **Does NOT** check
/// `record.tx_data.digest.signer_pubkey == expected_wrapper_signer_pubkey`
/// because Stage 13.5 has no wrapper-signer context. Passing
/// `record.tx_data.digest.signer_pubkey` back as the expected
/// value would make that check tautological — a lie about what
/// Stage 13.5 verify proves. Wrapper-signer binding requires
/// including + verifying a Stage 12.25 signed-chain-report,
/// which is out of scope here (Q7).
fn verify_anchor_record_with_artifact_bytes(
    record: &AnchorRecord,
    artifact_bytes: &[u8],
) -> EvidenceAnchorResult<()> {
    let mut recomputed = [0u8; 32];
    recomputed.copy_from_slice(blake3::hash(artifact_bytes).as_bytes());
    if record.tx_data.digest.artifact_hash != recomputed {
        let anchor_record_relative_path = format!(
            "{ANCHORS_SUBDIR}/{}.json",
            anchor_hex_lower(&record.tx_data.digest.artifact_hash)
        );
        return Err(EvidenceAnchorError::ExportEntryMetadataMismatch {
            relative_path: anchor_record_relative_path,
            field: "artifact_hash_binding",
            computed: anchor_hex_lower(&recomputed),
            manifest: anchor_hex_lower(&record.tx_data.digest.artifact_hash),
        });
    }
    verify_anchor_tx_data(&record.tx_data)?;
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Stage 13.5 path-validation helper. Refuses any
/// `relative_path` that:
/// - is empty,
/// - is absolute,
/// - starts with `/`, contains `\`, or contains `//`,
/// - contains a `..` path component,
/// - or violates the per-kind shape rule:
///   - `anchor_record`: `anchors/<64-lower-hex>.json`,
///   - `artifact_bytes`: `artifacts/<64-lower-hex>` (no
///     extension),
///   - `signed_chain_report`: `signed_chain_reports/<safe-basename>`
///     (safe-basename: no `/`, no `..`, no `\`, no null byte;
///     extension free-form).
fn validate_relative_path_for_kind(
    relative_path: &str,
    kind: AnchorExportEntryKind,
) -> EvidenceAnchorResult<()> {
    let reject = |reason: &'static str| EvidenceAnchorError::ExportInvalidPath {
        entry_kind: kind.as_str(),
        relative_path: relative_path.to_string(),
        reason,
    };
    if relative_path.is_empty() {
        return Err(reject("empty relative_path"));
    }
    if relative_path.contains('\\') {
        return Err(reject("backslash not allowed in relative_path"));
    }
    if relative_path.contains('\0') {
        return Err(reject("null byte not allowed"));
    }
    let path = std::path::Path::new(relative_path);
    if path.is_absolute() {
        return Err(reject("absolute paths not allowed"));
    }
    if relative_path.starts_with('/') {
        return Err(reject("leading slash not allowed"));
    }
    if relative_path.contains("//") {
        return Err(reject("duplicate slash not allowed"));
    }
    if relative_path.split('/').any(|c| c == "..") {
        return Err(reject("parent traversal forbidden"));
    }
    if relative_path.split('/').any(|c| c == ".") {
        return Err(reject("self traversal forbidden"));
    }
    // Per-kind shape.
    let expected_subdir = kind.subdir();
    let suffix = match relative_path.strip_prefix(&format!("{expected_subdir}/")) {
        Some(rest) => rest,
        None => {
            return Err(reject("wrong top-level directory for entry kind"));
        }
    };
    if suffix.contains('/') {
        return Err(reject("nested subdirectories not allowed under top-level"));
    }
    match kind {
        AnchorExportEntryKind::AnchorRecord => {
            let stem = match suffix.strip_suffix(".json") {
                Some(s) => s,
                None => {
                    return Err(reject(
                        "anchor_record requires .json suffix",
                    ));
                }
            };
            if stem.len() != 64 {
                return Err(reject(
                    "anchor_record stem must be exactly 64 hex chars",
                ));
            }
            if !stem
                .bytes()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
            {
                return Err(reject(
                    "anchor_record stem must be lowercase hex only",
                ));
            }
        }
        AnchorExportEntryKind::ArtifactBytes => {
            if suffix.len() != 64 {
                return Err(reject(
                    "artifact_bytes filename must be exactly 64 hex chars (no extension)",
                ));
            }
            if !suffix
                .bytes()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
            {
                return Err(reject(
                    "artifact_bytes filename must be lowercase hex only",
                ));
            }
        }
        AnchorExportEntryKind::SignedChainReport => {
            validate_safe_basename(suffix, kind)?;
        }
    }
    Ok(())
}

fn validate_safe_basename(
    basename: &str,
    kind: AnchorExportEntryKind,
) -> EvidenceAnchorResult<()> {
    let reject = |reason: &'static str| EvidenceAnchorError::ExportInvalidPath {
        entry_kind: kind.as_str(),
        relative_path: basename.to_string(),
        reason,
    };
    if basename.is_empty() {
        return Err(reject("empty basename"));
    }
    if basename.contains('/') || basename.contains('\\') {
        return Err(reject("basename must not contain a path separator"));
    }
    if basename == "." || basename == ".." {
        return Err(reject("basename must not be '.' or '..'"));
    }
    if basename.contains('\0') {
        return Err(reject("null byte not allowed in basename"));
    }
    Ok(())
}

fn with_tmp_suffix(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    name.push_str(".tmp");
    path.with_file_name(name)
}

fn compute_record_set_hash(records: &[AnchorRecord]) -> String {
    let mut sorted = records.to_vec();
    sorted.sort_by(|a, b| a.artifact_hash_hex.cmp(&b.artifact_hash_hex));
    let canonical = serde_json::json!(
        sorted
            .iter()
            .map(|r| serde_json::json!({
                "artifact_hash_hex": r.artifact_hash_hex,
                "tx_id": r.receipt.tx_id,
                "status": r.status.as_str(),
            }))
            .collect::<Vec<_>>()
    );
    let bytes = serde_json::to_vec(&canonical).expect("canonical JSON of record-set");
    blake3::hash(&bytes).to_hex().to_string()
}

fn compute_export_id(record_set_hash: &str, now_utc: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(record_set_hash.as_bytes());
    hasher.update(b"||");
    hasher.update(now_utc.as_bytes());
    let hex = hasher.finalize().to_hex();
    hex[..16].to_string()
}

fn compute_export_manifest_hash(
    manifest: &AnchorExportManifest,
) -> EvidenceAnchorResult<String> {
    let blanked = AnchorExportManifest {
        export_manifest_hash: EXPORT_MANIFEST_HASH_BLANK.to_string(),
        ..manifest.clone()
    };
    let bytes = serde_json::to_vec(&blanked).map_err(|e| {
        EvidenceAnchorError::CanonicalSerialization(e.to_string())
    })?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_kind_as_str_round_trips() {
        for kind in [
            AnchorExportEntryKind::AnchorRecord,
            AnchorExportEntryKind::ArtifactBytes,
            AnchorExportEntryKind::SignedChainReport,
        ] {
            assert_eq!(AnchorExportEntryKind::parse(kind.as_str()), Some(kind));
        }
    }

    #[test]
    fn manifest_hash_blanks_self_field_in_canonical_bytes() {
        let mut m = AnchorExportManifest {
            schema_version: 1,
            export_id: "abc".repeat(16),
            created_at_utc: "2026-06-17T00:00:00Z".to_string(),
            label: "".to_string(),
            notes: "".to_string(),
            entries: vec![],
            export_manifest_hash: "old-junk".to_string(),
        };
        let with_a = compute_export_manifest_hash(&m).unwrap();
        m.export_manifest_hash = "totally-different".to_string();
        let with_b = compute_export_manifest_hash(&m).unwrap();
        assert_eq!(
            with_a, with_b,
            "compute_export_manifest_hash must blank the field"
        );
    }
}
