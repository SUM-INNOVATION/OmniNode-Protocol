//! Phase 5 Stage 13.4 — anchor-registry cleanup with quarantine.
//!
//! Mirrors the Stage 12.17/12.18 [`crate::evidence_anchor`-equivalent
//! `omni_contributor::cleanup`] surface (plan → apply → restore)
//! narrowed to the integrity-evidence anchor registry. Detection
//! is reused verbatim from Stage 13.3
//! [`crate::evidence_anchor::check_evidence_anchor_registry_health`]
//! and [`crate::evidence_anchor::list_stale_submitted_or_included`];
//! this module turns those signals into a typed plan, applies it
//! with dry-run-first posture, and provides symmetric restore.
//!
//! ## Locked Stage 13.4 invariants
//!
//! - **Fully local.** No chain RPCs, no `omni-sumchain` types.
//! - **Dry-run is the default.** Caller must set
//!   [`AnchorApplyOptions::dry_run = false`] explicitly.
//! - **Quarantine, don't delete** for Tier B actions.
//! - **Stale-cleanup needs two opt-ins**: a non-`None`
//!   [`AnchorPlanOptions::stale_threshold_secs`] AND
//!   [`AnchorApplyOptions::allow_stale_quarantine`].
//! - **One new closed reason tag** (`cleanup_drift`); four
//!   reused tag strings as new [`EvidenceAnchorError`] variants
//!   (`cleanup_plan_hash_mismatch`, `gate_required`,
//!   `quarantine_blake3_mismatch`, `restore_target_exists`).
//!
//! ## Drift & plan-hash recipes (Q3 locked)
//!
//! - `registry_state_hash = BLAKE3(canonical JSON of {
//!     records: sorted_by(artifact_hash_hex) [{ artifact_hash_hex,
//!     status, submitted_at_unix }],
//!     tx_index_entries: sorted_by(tx_id) [{ tx_id,
//!     artifact_hash_hex }] })`. **Includes `status`** so a
//!   chain-driven transition between plan and apply correctly
//!   trips drift. **Excludes `updated_at`** so no-op
//!   `update_status` calls don't trip drift.
//! - `cleanup_plan_hash = BLAKE3(canonical JSON of plan with
//!   this field blanked)`.
//! - `plan_id = lower_hex(BLAKE3(anchor_registry_dir ||
//!   registry_state_hash || created_at_utc))[..16]`.
//!
//! ## Quarantine layout (Q4 locked)
//!
//! `<quarantine-dir>/<plan_id>/<source_relative>` — source paths
//! mirrored verbatim. Manifest lives at
//! `<quarantine-dir>/<plan_id>/quarantine_manifest.json`; restore
//! derives the quarantine root from `manifest_path.parent()`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{EvidenceAnchorError, EvidenceAnchorResult};
use crate::evidence_anchor::registry::{
    AnchorRecord, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
};

// ── Schema constants ──────────────────────────────────────────────────────────

/// Stage 13.4 cleanup-plan JSON schema version. Persisted plans
/// declare this. Apply refuses on mismatch.
pub const ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION: u32 = 1;

/// Stage 13.4 quarantine-manifest JSON schema version. Persisted
/// manifests declare this. Restore refuses on mismatch.
pub const ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION: u32 = 1;

const PLAN_HASH_BLANK: &str = "";
const TX_INDEX_FILENAME: &str = "tx_index.json";
const QUARANTINE_MANIFEST_FILENAME: &str = "quarantine_manifest.json";

// ── Action taxonomy (closed) ──────────────────────────────────────────────────

/// Closed taxonomy of operator-applyable cleanup actions on the
/// anchor registry. Three Tier A + one Tier B variant cover
/// every Stage 13.3 health/stale finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AnchorCleanupActionKind {
    /// Tier A — orphan `.tmp` file in the registry root.
    /// Delete in place under `--apply`.
    RemoveOrphanTmpFile,
    /// Tier A — `tx_index.json` entry whose mapped record file
    /// is absent. Atomic-rewrite `tx_index.json` minus the
    /// entry. No quarantine (Q5 locked — re-adding would
    /// re-create the orphan).
    RemoveOrphanTxIndexEntry,
    /// Tier B — `<64-hex>.json` that doesn't parse as
    /// `AnchorRecord`. Copy to quarantine, append manifest
    /// entry, then delete source.
    QuarantineMalformedRecord,
    /// Tier B (gated) — `Submitted` / `Included` record past
    /// the `stale_threshold_secs` set at plan time. Requires
    /// [`AnchorApplyOptions::allow_stale_quarantine`] at apply
    /// time. Copies record to quarantine, appends manifest
    /// entry, then deletes record and removes its
    /// `tx_index.json` entry.
    QuarantineStaleOpenRecord,
}

impl AnchorCleanupActionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            AnchorCleanupActionKind::RemoveOrphanTmpFile => "remove_orphan_tmp_file",
            AnchorCleanupActionKind::RemoveOrphanTxIndexEntry => {
                "remove_orphan_tx_index_entry"
            }
            AnchorCleanupActionKind::QuarantineMalformedRecord => {
                "quarantine_malformed_record"
            }
            AnchorCleanupActionKind::QuarantineStaleOpenRecord => {
                "quarantine_stale_open_record"
            }
        }
    }

    pub fn is_tier_b(self) -> bool {
        matches!(
            self,
            AnchorCleanupActionKind::QuarantineMalformedRecord
                | AnchorCleanupActionKind::QuarantineStaleOpenRecord
        )
    }

    /// Apply-time gate flag, or `None` for non-gated actions.
    pub fn gate_flag(self) -> Option<&'static str> {
        match self {
            AnchorCleanupActionKind::QuarantineStaleOpenRecord => {
                Some("--allow-stale-quarantine")
            }
            _ => None,
        }
    }
}

// ── Plan + action structs ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorCleanupAction {
    pub kind: AnchorCleanupActionKind,
    /// Path relative to `anchor_registry_dir`, forward-slash
    /// normalized. For `RemoveOrphanTxIndexEntry` this is
    /// `tx_index.json` (with the targeted entry identified by
    /// `tx_id`); for record-level actions it's `<hash>.json`.
    pub source_relative: String,
    /// Only populated for actions that touch a `tx_id` (i.e.
    /// `RemoveOrphanTxIndexEntry` and `QuarantineStaleOpenRecord`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tx_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorCleanupPlan {
    pub schema_version: u32,
    pub plan_id: String,
    pub created_at_utc: String,
    pub anchor_registry_dir: String,
    pub registry_state_hash: String,
    pub omni_zkml_version: String,
    pub actions: Vec<AnchorCleanupAction>,
    /// BLAKE3 over the canonical JSON of this plan with
    /// `cleanup_plan_hash` blanked. Apply recomputes and
    /// refuses with [`EvidenceAnchorError::CleanupPlanHashMismatch`]
    /// on tampered plans.
    pub cleanup_plan_hash: String,
}

// ── Plan / apply / restore option structs ─────────────────────────────────────

pub struct AnchorPlanOptions<'a> {
    /// RFC 3339 UTC. Stamped into `created_at_utc` and used to
    /// derive `plan_id`. Caller injects (e.g. `Utc::now()`) so
    /// tests get deterministic plan ids.
    pub now_utc: &'a str,
    /// Threshold for `QuarantineStaleOpenRecord` actions.
    /// `None` (default) → no stale-cleanup actions emitted, per
    /// Q1's two-opt-in lock.
    pub stale_threshold_secs: Option<u64>,
}

pub struct AnchorApplyOptions<'a> {
    /// Root under which the `<plan_id>/...` quarantine subtree
    /// and `quarantine_manifest.json` are written. The caller is
    /// responsible for ensuring the directory exists and is
    /// writable. Used for both Tier B writes and the manifest
    /// landing site.
    pub quarantine_dir: &'a Path,
    /// When true, no FS writes / removes happen. Outcomes
    /// record `would_apply` and the report's `mode` is
    /// `dry_run`. Drift / plan-hash / gate checks still run —
    /// dry-run is a real preflight.
    pub dry_run: bool,
    /// Required for any `QuarantineStaleOpenRecord` action.
    /// Without it, apply refuses with
    /// [`EvidenceAnchorError::CleanupGateRequired`].
    pub allow_stale_quarantine: bool,
    /// RFC 3339 UTC. Stamped into the quarantine manifest's
    /// `created_at_utc`.
    pub now_utc: &'a str,
}

pub struct AnchorRestoreOptions<'a> {
    /// `<quarantine-dir>/<plan_id>/`. CLI derives this from
    /// `manifest_path.parent()` (Finding 4 fix); the library
    /// takes it as an explicit argument.
    pub quarantine_dir: &'a Path,
    /// Where to put the restored files. Restore refuses with
    /// [`EvidenceAnchorError::RestoreTargetExists`] on collisions.
    pub anchor_registry_dir: &'a Path,
    /// When true, restore validates everything but writes
    /// nothing.
    pub dry_run: bool,
}

// ── Outcome / report / manifest ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorCleanupActionOutcome {
    pub action_index: u32,
    pub kind: String,
    pub source_relative: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tx_id: Option<String>,
    /// Closed set: `"applied" | "would_apply" | "skipped_missing"`.
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorCleanupReport {
    pub plan_id: String,
    /// `"apply"` / `"dry_run"`. Closed set.
    pub mode: String,
    pub actions_applied: u32,
    pub actions_dry_run: u32,
    pub actions_skipped: u32,
    pub quarantine_dir: String,
    /// `<plan_id>/quarantine_manifest.json` when a manifest was
    /// written; `None` for dry-runs and plans with no Tier B
    /// actions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quarantine_manifest_relative: Option<String>,
    pub outcomes: Vec<AnchorCleanupActionOutcome>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorQuarantineEntry {
    /// Path under the source `anchor_registry_dir`, forward-slash
    /// normalized.
    pub source_relative: String,
    /// Path under `<quarantine-dir>/<plan_id>/`, forward-slash
    /// normalized. Mirrors `source_relative` verbatim per Q4.
    pub quarantine_relative: String,
    /// BLAKE3 of the file bytes as quarantined. Restore
    /// recomputes and refuses on drift.
    pub blake3_hex: String,
    pub bytes: u64,
    pub action_kind: String,
    /// For `QuarantineStaleOpenRecord`: the `tx_id` whose
    /// `tx_index.json` entry was removed at apply time. Restore
    /// uses this to re-add the entry symmetrically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tx_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorQuarantineManifest {
    pub schema_version: u32,
    pub plan_id: String,
    pub created_at_utc: String,
    pub entries: Vec<AnchorQuarantineEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorQuarantineRestoreOutcome {
    pub source_relative: String,
    /// Closed set: `"restored" | "would_restore" | "skipped_already_restored"`.
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorQuarantineRestoreReport {
    pub plan_id: String,
    pub mode: String,
    pub restored: u32,
    pub skipped: u32,
    pub outcomes: Vec<AnchorQuarantineRestoreOutcome>,
}

// ── Plan ──────────────────────────────────────────────────────────────────────

/// Scan the registry and produce a typed cleanup plan.
///
/// Pure read — no FS mutations. The returned plan carries a
/// `registry_state_hash` (Q3 recipe) and a `cleanup_plan_hash`
/// (canonical JSON with the field blanked) so apply can refuse
/// on drift or tampering.
pub fn plan_anchor_cleanup(
    registry: &LocalEvidenceAnchorRegistry,
    opts: &AnchorPlanOptions<'_>,
) -> EvidenceAnchorResult<AnchorCleanupPlan> {
    let root = registry.root();
    let anchor_registry_dir = forward_slash_path(root);
    let registry_state_hash = compute_registry_state_hash(root)?;

    let mut actions: Vec<AnchorCleanupAction> = Vec::new();

    // ── Health scan ──────────────────────────────────────────
    // Orphan .tmp files + malformed records, in a single pass.
    let mut tx_index_entries: BTreeMap<String, String> = BTreeMap::new();
    for entry in std::fs::read_dir(root).map_err(|e| EvidenceAnchorError::Io {
        path: root.to_path_buf(),
        source: e,
    })? {
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
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();

        if ext == "tmp" {
            actions.push(AnchorCleanupAction {
                kind: AnchorCleanupActionKind::RemoveOrphanTmpFile,
                source_relative: name,
                tx_id: None,
            });
            continue;
        }
        if name == TX_INDEX_FILENAME {
            // Defer; we read this after the record sweep.
            continue;
        }
        if ext != "json" || stem.len() != 64 {
            continue;
        }
        // Try to parse as AnchorRecord.
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                actions.push(AnchorCleanupAction {
                    kind: AnchorCleanupActionKind::QuarantineMalformedRecord,
                    source_relative: name,
                    tx_id: None,
                });
                continue;
            }
        };
        if serde_json::from_slice::<AnchorRecord>(&bytes).is_err() {
            actions.push(AnchorCleanupAction {
                kind: AnchorCleanupActionKind::QuarantineMalformedRecord,
                source_relative: name,
                tx_id: None,
            });
        }
    }

    // ── tx_index orphan scan ─────────────────────────────────
    let tx_index_path = root.join(TX_INDEX_FILENAME);
    if tx_index_path.is_file() {
        let bytes = std::fs::read(&tx_index_path).map_err(|e| EvidenceAnchorError::Io {
            path: tx_index_path.clone(),
            source: e,
        })?;
        if let Ok(index) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(map) = index.get("by_tx_id").and_then(|v| v.as_object()) {
                for (tx_id, hash_value) in map {
                    if let Some(hash_hex) = hash_value.as_str() {
                        tx_index_entries.insert(tx_id.clone(), hash_hex.to_string());
                        let record_path = root.join(format!("{hash_hex}.json"));
                        if !record_path.is_file() {
                            actions.push(AnchorCleanupAction {
                                kind:
                                    AnchorCleanupActionKind::RemoveOrphanTxIndexEntry,
                                source_relative: TX_INDEX_FILENAME.to_string(),
                                tx_id: Some(tx_id.clone()),
                            });
                        }
                    }
                }
            }
        }
    }

    // ── Stale-cleanup scan ───────────────────────────────────
    if let Some(threshold_secs) = opts.stale_threshold_secs {
        let now_utc = parse_rfc3339(opts.now_utc)?;
        // Reuse Stage 13.3 stale-detection helper.
        let stale = super::operations::list_stale_submitted_or_included(
            registry,
            now_utc,
            threshold_secs,
        )?;
        for row in stale {
            actions.push(AnchorCleanupAction {
                kind: AnchorCleanupActionKind::QuarantineStaleOpenRecord,
                source_relative: format!("{}.json", row.artifact_hash_hex),
                tx_id: Some(row.tx_id),
            });
        }
    }

    // Sort actions deterministically.
    actions.sort_by(|a, b| {
        a.kind
            .as_str()
            .cmp(b.kind.as_str())
            .then_with(|| a.source_relative.cmp(&b.source_relative))
            .then_with(|| a.tx_id.cmp(&b.tx_id))
    });

    let plan_id = compute_plan_id(
        &anchor_registry_dir,
        &registry_state_hash,
        opts.now_utc,
    );

    let mut plan = AnchorCleanupPlan {
        schema_version: ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION,
        plan_id,
        created_at_utc: opts.now_utc.to_string(),
        anchor_registry_dir,
        registry_state_hash,
        omni_zkml_version: env!("CARGO_PKG_VERSION").to_string(),
        actions,
        cleanup_plan_hash: PLAN_HASH_BLANK.to_string(),
    };
    plan.cleanup_plan_hash = compute_plan_hash(&plan)?;
    Ok(plan)
}

// ── Apply ─────────────────────────────────────────────────────────────────────

/// Apply a previously-generated [`AnchorCleanupPlan`].
///
/// Preflights (run in dry-run AND real-run modes):
/// 1. **Schema-version check** — refuse with
///    [`EvidenceAnchorError::CleanupPlanSchemaUnsupported`] on
///    `plan.schema_version != ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION`.
///    Runs **before** the hash check so a future-schema plan
///    with a recomputed hash can't sneak through as v1.
/// 2. **Plan-hash check** —
///    [`EvidenceAnchorError::CleanupPlanHashMismatch`] on
///    tampered plans.
/// 3. **Drift check** —
///    [`EvidenceAnchorError::CleanupDrift`] when the registry
///    state has changed since the plan was generated.
/// 4. **Gate check** —
///    [`EvidenceAnchorError::CleanupGateRequired`] for missing
///    `--allow-stale-quarantine`.
/// 5. **Per-action path validation** —
///    [`EvidenceAnchorError::CleanupInvalidPath`] on absolute
///    paths, `..` traversal, separator misuse, or per-kind shape
///    violations (see [`validate_source_relative_for_kind`]).
///
/// Real-run ordering (durability-first):
/// 1. Tier B copies — write all quarantine files; collect
///    manifest entries. **No source deletions yet.**
/// 2. Write the quarantine manifest atomically. If this step
///    fails, every source is still on disk.
/// 3. Tier B + Tier A deletions — remove sources only after
///    the manifest is durably on disk. Tier-A `tx_index.json`
///    rewrite happens at the end.
///
/// Dry-run mode: all preflights run; no FS mutations.
/// Idempotent misses (source already gone) record
/// `skipped_missing`.
pub fn apply_anchor_cleanup(
    plan: &AnchorCleanupPlan,
    opts: &AnchorApplyOptions<'_>,
) -> EvidenceAnchorResult<AnchorCleanupReport> {
    // ── Preflight 1: schema-version check (FIRST — even before hash) ──
    if plan.schema_version != ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION {
        return Err(EvidenceAnchorError::CleanupPlanSchemaUnsupported {
            got: plan.schema_version,
            expected: ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION,
        });
    }

    // ── Preflight 2: plan-hash check ─────────────────────────
    let computed_plan_hash = compute_plan_hash(plan)?;
    if computed_plan_hash != plan.cleanup_plan_hash {
        return Err(EvidenceAnchorError::CleanupPlanHashMismatch {
            computed: computed_plan_hash,
            expected: plan.cleanup_plan_hash.clone(),
        });
    }

    // ── Preflight 3: drift check ─────────────────────────────
    let registry_root = PathBuf::from(&plan.anchor_registry_dir);
    let computed_state_hash = compute_registry_state_hash(&registry_root)?;
    if computed_state_hash != plan.registry_state_hash {
        return Err(EvidenceAnchorError::CleanupDrift {
            computed: computed_state_hash,
            expected: plan.registry_state_hash.clone(),
        });
    }

    // ── Preflight 4: gate check ──────────────────────────────
    for action in &plan.actions {
        if let Some(flag) = action.kind.gate_flag() {
            if !opts.allow_stale_quarantine
                && matches!(
                    action.kind,
                    AnchorCleanupActionKind::QuarantineStaleOpenRecord
                )
            {
                return Err(EvidenceAnchorError::CleanupGateRequired {
                    action_kind: action.kind.as_str(),
                    gate_flag: flag,
                });
            }
        }
    }

    // ── Preflight 5: per-action path validation ──────────────
    for action in &plan.actions {
        validate_source_relative_for_kind(&action.source_relative, action.kind)?;
    }

    let mode = if opts.dry_run { "dry_run" } else { "apply" };
    let quarantine_dir_plan = opts.quarantine_dir.join(&plan.plan_id);
    let has_tier_b = plan.actions.iter().any(|a| a.kind.is_tier_b());

    let mut outcomes: Vec<AnchorCleanupActionOutcome> = Vec::with_capacity(plan.actions.len());
    let mut actions_applied: u32 = 0;
    let mut actions_dry_run: u32 = 0;
    let mut actions_skipped: u32 = 0;
    let mut quarantine_entries: Vec<AnchorQuarantineEntry> = Vec::new();
    // Deferred destructive ops (Pass 2 — runs only after the
    // manifest is durably on disk).
    let mut deferred_source_removals: Vec<(u32, PathBuf)> = Vec::new();
    let mut tx_ids_to_remove: Vec<String> = Vec::new();
    // Track action indices that produced quarantine entries —
    // their `applied` increment is deferred to Pass 2.
    let mut deferred_quarantine_action_indices: Vec<u32> = Vec::new();

    // ── Pass 1 — non-destructive: classify per action, copy
    //    Tier B sources to quarantine, accumulate manifest
    //    entries. No source files are removed yet. ─────────
    for (idx, action) in plan.actions.iter().enumerate() {
        let action_index = idx as u32;
        let kind_str = action.kind.as_str().to_string();
        let source_relative = action.source_relative.clone();
        let tx_id = action.tx_id.clone();
        let source_path = registry_root.join(&source_relative);

        let status: &'static str = match action.kind {
            AnchorCleanupActionKind::RemoveOrphanTmpFile => {
                if !source_path.exists() {
                    actions_skipped += 1;
                    "skipped_missing"
                } else if opts.dry_run {
                    actions_dry_run += 1;
                    "would_apply"
                } else {
                    deferred_source_removals.push((action_index, source_path.clone()));
                    "applied"
                }
            }
            AnchorCleanupActionKind::RemoveOrphanTxIndexEntry => {
                if opts.dry_run {
                    actions_dry_run += 1;
                    "would_apply"
                } else {
                    if let Some(ref t) = tx_id {
                        tx_ids_to_remove.push(t.clone());
                    }
                    "applied"
                }
            }
            AnchorCleanupActionKind::QuarantineMalformedRecord
            | AnchorCleanupActionKind::QuarantineStaleOpenRecord => {
                if !source_path.exists() {
                    actions_skipped += 1;
                    "skipped_missing"
                } else if opts.dry_run {
                    actions_dry_run += 1;
                    "would_apply"
                } else {
                    let bytes =
                        std::fs::read(&source_path).map_err(|e| EvidenceAnchorError::Io {
                            path: source_path.clone(),
                            source: e,
                        })?;
                    let blake3_hex = blake3::hash(&bytes).to_hex().to_string();
                    let bytes_len = bytes.len() as u64;
                    // Pass-1: COPY to quarantine; do NOT delete source yet.
                    let quarantine_target = quarantine_dir_plan.join(&source_relative);
                    if let Some(parent) = quarantine_target.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| EvidenceAnchorError::Io {
                            path: parent.to_path_buf(),
                            source: e,
                        })?;
                    }
                    let tmp = quarantine_target.with_extension("json.tmp");
                    std::fs::write(&tmp, &bytes).map_err(|e| EvidenceAnchorError::Io {
                        path: tmp.clone(),
                        source: e,
                    })?;
                    std::fs::rename(&tmp, &quarantine_target).map_err(|e| {
                        EvidenceAnchorError::Io {
                            path: quarantine_target.clone(),
                            source: e,
                        }
                    })?;
                    quarantine_entries.push(AnchorQuarantineEntry {
                        source_relative: source_relative.clone(),
                        quarantine_relative: source_relative.clone(),
                        blake3_hex,
                        bytes: bytes_len,
                        action_kind: kind_str.clone(),
                        tx_id: tx_id.clone(),
                    });
                    // Defer the source removal to Pass 2.
                    deferred_source_removals
                        .push((action_index, source_path.clone()));
                    deferred_quarantine_action_indices.push(action_index);
                    if let AnchorCleanupActionKind::QuarantineStaleOpenRecord =
                        action.kind
                    {
                        if let Some(ref t) = tx_id {
                            tx_ids_to_remove.push(t.clone());
                        }
                    }
                    "applied"
                }
            }
        };

        outcomes.push(AnchorCleanupActionOutcome {
            action_index,
            kind: kind_str,
            source_relative,
            tx_id,
            status: status.to_string(),
        });
    }

    // ── Pass 1.5 — Manifest write (durability fence) ─────────
    // We commit the manifest to disk BEFORE any source is
    // removed, so that if anything below fails the operator
    // has a complete restore path.
    let mut manifest_relative: Option<String> = None;
    if !opts.dry_run && !quarantine_entries.is_empty() {
        let manifest = AnchorQuarantineManifest {
            schema_version: ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION,
            plan_id: plan.plan_id.clone(),
            created_at_utc: opts.now_utc.to_string(),
            entries: quarantine_entries,
        };
        let manifest_path = quarantine_dir_plan.join(QUARANTINE_MANIFEST_FILENAME);
        if let Some(parent) = manifest_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| EvidenceAnchorError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| {
            EvidenceAnchorError::MalformedJson {
                path: manifest_path.clone(),
                source: e,
            }
        })?;
        let tmp = manifest_path.with_extension("json.tmp");
        std::fs::write(&tmp, &manifest_bytes).map_err(|e| EvidenceAnchorError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp, &manifest_path).map_err(|e| EvidenceAnchorError::Io {
            path: manifest_path.clone(),
            source: e,
        })?;
        manifest_relative = Some(format!(
            "{}/{}",
            plan.plan_id, QUARANTINE_MANIFEST_FILENAME
        ));
    }

    // ── Pass 2 — destructive: now the manifest is durably on
    //    disk (or wasn't needed). Delete sources and rewrite
    //    tx_index.json. A failure in this pass leaves the
    //    operator with a valid restore path. ─────────────────
    if !opts.dry_run {
        for (action_index, source_path) in &deferred_source_removals {
            std::fs::remove_file(source_path).map_err(|e| EvidenceAnchorError::Io {
                path: source_path.clone(),
                source: e,
            })?;
            // RemoveOrphanTmpFile is the only Tier A in
            // `deferred_source_removals`; it doesn't go through
            // `deferred_quarantine_action_indices`. Increment
            // its `actions_applied` here. Quarantine actions
            // get their `actions_applied` increment below.
            if !deferred_quarantine_action_indices.contains(action_index) {
                actions_applied += 1;
            }
        }
        // Quarantine actions have completed their copy + manifest
        // + delete — count them as applied now.
        actions_applied += deferred_quarantine_action_indices.len() as u32;
        // RemoveOrphanTxIndexEntry doesn't go through
        // `deferred_source_removals`; count those as applied here.
        for outcome in &outcomes {
            if outcome.kind == AnchorCleanupActionKind::RemoveOrphanTxIndexEntry.as_str()
                && outcome.status == "applied"
            {
                actions_applied += 1;
            }
        }

        // tx_index.json rewrite (single atomic write).
        if !tx_ids_to_remove.is_empty() {
            let index_path = registry_root.join(TX_INDEX_FILENAME);
            if index_path.is_file() {
                let bytes = std::fs::read(&index_path).map_err(|e| EvidenceAnchorError::Io {
                    path: index_path.clone(),
                    source: e,
                })?;
                let mut value: serde_json::Value = serde_json::from_slice(&bytes).map_err(
                    |e| EvidenceAnchorError::MalformedJson {
                        path: index_path.clone(),
                        source: e,
                    },
                )?;
                if let Some(map) =
                    value.get_mut("by_tx_id").and_then(|v| v.as_object_mut())
                {
                    for tx_id in &tx_ids_to_remove {
                        map.remove(tx_id);
                    }
                }
                let new_bytes = serde_json::to_vec_pretty(&value).map_err(|e| {
                    EvidenceAnchorError::MalformedJson {
                        path: index_path.clone(),
                        source: e,
                    }
                })?;
                let tmp = index_path.with_extension("json.tmp");
                std::fs::write(&tmp, &new_bytes).map_err(|e| EvidenceAnchorError::Io {
                    path: tmp.clone(),
                    source: e,
                })?;
                std::fs::rename(&tmp, &index_path).map_err(|e| EvidenceAnchorError::Io {
                    path: index_path.clone(),
                    source: e,
                })?;
            }
        }
    }
    // For dry-run mode, declare the future manifest path even
    // though no file was written, so the operator can preview
    // where it will land.
    if opts.dry_run && has_tier_b {
        manifest_relative = Some(format!(
            "{}/{}",
            plan.plan_id, QUARANTINE_MANIFEST_FILENAME
        ));
    }

    Ok(AnchorCleanupReport {
        plan_id: plan.plan_id.clone(),
        mode: mode.to_string(),
        actions_applied,
        actions_dry_run,
        actions_skipped,
        quarantine_dir: forward_slash_path(&quarantine_dir_plan),
        quarantine_manifest_relative: manifest_relative,
        outcomes,
    })
}

// ── Restore ───────────────────────────────────────────────────────────────────

/// Restore quarantined bytes back into the anchor registry.
///
/// Per-entry preflights:
/// - BLAKE3 of the quarantined file must match the manifest's
///   recorded hash. Refuses with
///   [`EvidenceAnchorError::QuarantineBlake3Mismatch`].
/// - Target path must not exist. Refuses with
///   [`EvidenceAnchorError::RestoreTargetExists`]. Exception:
///   an already-restored entry (target exists AND BLAKE3
///   matches the manifest's recorded hash) records
///   `skipped_already_restored` rather than refusing.
///
/// On restore-success, the file is atomic-copied back and (for
/// `QuarantineStaleOpenRecord` entries) the `tx_id` is re-added
/// to `tx_index.json` symmetrically.
pub fn restore_anchor_cleanup_quarantine(
    manifest: &AnchorQuarantineManifest,
    opts: &AnchorRestoreOptions<'_>,
) -> EvidenceAnchorResult<AnchorQuarantineRestoreReport> {
    if manifest.schema_version != ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION {
        return Err(EvidenceAnchorError::MalformedJson {
            path: opts.quarantine_dir.join(QUARANTINE_MANIFEST_FILENAME),
            source: serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported manifest schema_version: got {}, expected {}",
                    manifest.schema_version, ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION
                ),
            )),
        });
    }

    // Per-entry path validation BEFORE any FS mutation. Defends
    // against absolute paths, `..` traversal, separator misuse,
    // and per-kind shape violations in operator-supplied JSON.
    for entry in &manifest.entries {
        let action_kind =
            parse_action_kind_for_manifest_entry(&entry.action_kind)?;
        validate_source_relative_for_kind(&entry.source_relative, action_kind)?;
        validate_source_relative_for_kind(&entry.quarantine_relative, action_kind)?;
        // Q4 lock: quarantine layout mirrors source paths verbatim.
        if entry.source_relative != entry.quarantine_relative {
            return Err(EvidenceAnchorError::CleanupInvalidPath {
                action_kind: action_kind.as_str(),
                source_relative: entry.source_relative.clone(),
                reason:
                    "quarantine_relative must mirror source_relative verbatim (Q4 layout)",
            });
        }
    }

    let mode = if opts.dry_run { "dry_run" } else { "apply" };
    let mut outcomes: Vec<AnchorQuarantineRestoreOutcome> =
        Vec::with_capacity(manifest.entries.len());
    let mut restored: u32 = 0;
    let mut skipped: u32 = 0;
    let mut tx_ids_to_restore: Vec<(String, String)> = Vec::new();

    for entry in &manifest.entries {
        let quarantine_path = opts.quarantine_dir.join(&entry.quarantine_relative);
        let target_path = opts.anchor_registry_dir.join(&entry.source_relative);

        // Read quarantine bytes; recompute BLAKE3.
        let bytes = std::fs::read(&quarantine_path).map_err(|e| EvidenceAnchorError::Io {
            path: quarantine_path.clone(),
            source: e,
        })?;
        let computed = blake3::hash(&bytes).to_hex().to_string();
        if computed != entry.blake3_hex {
            return Err(EvidenceAnchorError::QuarantineBlake3Mismatch {
                source_relative: entry.source_relative.clone(),
                computed,
                expected: entry.blake3_hex.clone(),
            });
        }

        // For stale-record entries, queue the tx_index re-add
        // regardless of which branch the file-restore path
        // takes below. This is the idempotency fix: a previous
        // restore that copied the file but failed before the
        // tx_index rewrite would otherwise leave the record
        // file present but permanently missing from
        // tx_index.json.
        let queue_tx_index_restore_if_stale = |dst: &mut Vec<(String, String)>| {
            if entry.action_kind
                == AnchorCleanupActionKind::QuarantineStaleOpenRecord.as_str()
            {
                if let Some(ref t) = entry.tx_id {
                    let hash_hex = entry
                        .source_relative
                        .strip_suffix(".json")
                        .unwrap_or(&entry.source_relative)
                        .to_string();
                    dst.push((t.clone(), hash_hex));
                }
            }
        };

        // Existing-target handling.
        if target_path.exists() {
            // Idempotency: target exists AND its bytes match
            // the manifest → operator already restored the file.
            // Still queue the tx_index re-add (idempotent at the
            // index-rewrite step — overwrite is a no-op when the
            // entry is already present).
            let existing_bytes = std::fs::read(&target_path).map_err(|e| EvidenceAnchorError::Io {
                path: target_path.clone(),
                source: e,
            })?;
            let existing_hash = blake3::hash(&existing_bytes).to_hex().to_string();
            if existing_hash == entry.blake3_hex {
                skipped += 1;
                queue_tx_index_restore_if_stale(&mut tx_ids_to_restore);
                outcomes.push(AnchorQuarantineRestoreOutcome {
                    source_relative: entry.source_relative.clone(),
                    status: "skipped_already_restored".to_string(),
                });
                continue;
            }
            return Err(EvidenceAnchorError::RestoreTargetExists {
                target_path: target_path.clone(),
            });
        }

        if opts.dry_run {
            outcomes.push(AnchorQuarantineRestoreOutcome {
                source_relative: entry.source_relative.clone(),
                status: "would_restore".to_string(),
            });
            continue;
        }

        // Atomic copy: write to .tmp + rename.
        if let Some(parent) = target_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| EvidenceAnchorError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let tmp = target_path.with_extension("json.tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| EvidenceAnchorError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp, &target_path).map_err(|e| EvidenceAnchorError::Io {
            path: target_path.clone(),
            source: e,
        })?;
        restored += 1;
        queue_tx_index_restore_if_stale(&mut tx_ids_to_restore);
        outcomes.push(AnchorQuarantineRestoreOutcome {
            source_relative: entry.source_relative.clone(),
            status: "restored".to_string(),
        });
    }

    if !opts.dry_run && !tx_ids_to_restore.is_empty() {
        let index_path = opts.anchor_registry_dir.join(TX_INDEX_FILENAME);
        let mut value: serde_json::Value = if index_path.is_file() {
            let bytes = std::fs::read(&index_path).map_err(|e| EvidenceAnchorError::Io {
                path: index_path.clone(),
                source: e,
            })?;
            serde_json::from_slice(&bytes).map_err(|e| EvidenceAnchorError::MalformedJson {
                path: index_path.clone(),
                source: e,
            })?
        } else {
            serde_json::json!({ "by_tx_id": {} })
        };
        let map = value
            .get_mut("by_tx_id")
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| EvidenceAnchorError::MalformedJson {
                path: index_path.clone(),
                source: serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "tx_index.json missing by_tx_id field",
                )),
            })?;
        for (tx_id, hash_hex) in tx_ids_to_restore {
            map.insert(tx_id, serde_json::Value::String(hash_hex));
        }
        let new_bytes = serde_json::to_vec_pretty(&value).map_err(|e| {
            EvidenceAnchorError::MalformedJson {
                path: index_path.clone(),
                source: e,
            }
        })?;
        let tmp = index_path.with_extension("json.tmp");
        std::fs::write(&tmp, &new_bytes).map_err(|e| EvidenceAnchorError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp, &index_path).map_err(|e| EvidenceAnchorError::Io {
            path: index_path.clone(),
            source: e,
        })?;
    }

    Ok(AnchorQuarantineRestoreReport {
        plan_id: manifest.plan_id.clone(),
        mode: mode.to_string(),
        restored,
        skipped,
        outcomes,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Stage 13.4 path-validation helper. Refuses any
/// `source_relative` / `quarantine_relative` that:
/// - is empty,
/// - is absolute,
/// - contains a path separator (`/` or `\`) — the anchor
///   registry is flat by construction; subdirectory escapes
///   are out of scope,
/// - contains a `..` component (defense in depth),
/// - or violates the per-action-kind shape rule:
///   - `RemoveOrphanTmpFile`: must end in `.tmp`.
///   - `RemoveOrphanTxIndexEntry`: must equal `tx_index.json`.
///   - `QuarantineMalformedRecord` / `QuarantineStaleOpenRecord`:
///     must match `<64-lower-hex>.json`.
///
/// Called by both `apply_anchor_cleanup` (on each
/// plan action) and `restore_anchor_cleanup_quarantine` (on
/// each manifest entry's `source_relative` AND
/// `quarantine_relative`) BEFORE any FS mutation.
fn validate_source_relative_for_kind(
    source_relative: &str,
    kind: AnchorCleanupActionKind,
) -> EvidenceAnchorResult<()> {
    let reject = |reason: &'static str| EvidenceAnchorError::CleanupInvalidPath {
        action_kind: kind.as_str(),
        source_relative: source_relative.to_string(),
        reason,
    };
    if source_relative.is_empty() {
        return Err(reject("empty source_relative"));
    }
    if source_relative.contains('/') || source_relative.contains('\\') {
        return Err(reject("path separator not allowed (registry is flat)"));
    }
    let path = std::path::Path::new(source_relative);
    if path.is_absolute() {
        return Err(reject("absolute paths not allowed"));
    }
    // `..` traversal defense-in-depth — also catches embedded
    // patterns like `..abc` only when they appear as a full
    // path component, which on the registry's flat layout
    // means the whole string equals `..` or `.`.
    if source_relative == ".." || source_relative == "." {
        return Err(reject("parent / self traversal forbidden"));
    }
    if source_relative.contains("/..") || source_relative.contains("../") {
        return Err(reject("parent traversal forbidden"));
    }
    // Per-kind shape.
    match kind {
        AnchorCleanupActionKind::RemoveOrphanTmpFile => {
            if !source_relative.ends_with(".tmp") {
                return Err(reject(
                    "RemoveOrphanTmpFile requires a .tmp suffix",
                ));
            }
        }
        AnchorCleanupActionKind::RemoveOrphanTxIndexEntry => {
            if source_relative != TX_INDEX_FILENAME {
                return Err(reject(
                    "RemoveOrphanTxIndexEntry requires source_relative == tx_index.json",
                ));
            }
        }
        AnchorCleanupActionKind::QuarantineMalformedRecord
        | AnchorCleanupActionKind::QuarantineStaleOpenRecord => {
            let stem = match source_relative.strip_suffix(".json") {
                Some(s) => s,
                None => {
                    return Err(reject("quarantine record requires .json suffix"));
                }
            };
            if stem.len() != 64 {
                return Err(reject(
                    "quarantine record stem must be exactly 64 hex chars",
                ));
            }
            if !stem.bytes().all(|b| {
                matches!(b, b'0'..=b'9' | b'a'..=b'f')
            }) {
                return Err(reject(
                    "quarantine record stem must be lowercase hex only",
                ));
            }
        }
    }
    Ok(())
}

/// Re-parse the manifest's serde `action_kind` string back into
/// the closed enum so the restore path can dispatch the
/// per-kind validation rule. Unknown strings refuse with
/// `cleanup_invalid_path`.
fn parse_action_kind_for_manifest_entry(
    action_kind: &str,
) -> EvidenceAnchorResult<AnchorCleanupActionKind> {
    match action_kind {
        "remove_orphan_tmp_file" => Ok(AnchorCleanupActionKind::RemoveOrphanTmpFile),
        "remove_orphan_tx_index_entry" => {
            Ok(AnchorCleanupActionKind::RemoveOrphanTxIndexEntry)
        }
        "quarantine_malformed_record" => {
            Ok(AnchorCleanupActionKind::QuarantineMalformedRecord)
        }
        "quarantine_stale_open_record" => {
            Ok(AnchorCleanupActionKind::QuarantineStaleOpenRecord)
        }
        other => Err(EvidenceAnchorError::CleanupInvalidPath {
            action_kind: "<unknown>",
            source_relative: other.to_string(),
            reason: "unknown manifest action_kind",
        }),
    }
}

fn forward_slash_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn parse_rfc3339(s: &str) -> EvidenceAnchorResult<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| EvidenceAnchorError::MalformedSignedAtUtc {
            raw: s.to_string(),
            reason: e.to_string(),
        })
}

/// Q3-locked recipe: BLAKE3 over canonical JSON of
/// `{ records: sorted_by(artifact_hash_hex) [{ artifact_hash_hex,
/// status, submitted_at_unix }], tx_index_entries: sorted_by(tx_id)
/// [{ tx_id, artifact_hash_hex }] }`. Includes `status`; excludes
/// `updated_at`.
fn compute_registry_state_hash(root: &Path) -> EvidenceAnchorResult<String> {
    let mut records: Vec<(String, String, i64)> = Vec::new();
    for entry in std::fs::read_dir(root).map_err(|e| EvidenceAnchorError::Io {
        path: root.to_path_buf(),
        source: e,
    })? {
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
        if ext != "json" || stem.len() != 64 {
            continue;
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue, // malformed records don't drift the state
        };
        let Ok(record) = serde_json::from_slice::<AnchorRecord>(&bytes) else {
            continue;
        };
        let status_tag = status_tag(&record.status);
        records.push((
            record.artifact_hash_hex,
            status_tag.to_string(),
            record.submitted_at.timestamp(),
        ));
    }
    records.sort_by(|a, b| a.0.cmp(&b.0));

    let mut tx_index_entries: Vec<(String, String)> = Vec::new();
    let tx_index_path = root.join(TX_INDEX_FILENAME);
    if tx_index_path.is_file() {
        let bytes = std::fs::read(&tx_index_path).map_err(|e| EvidenceAnchorError::Io {
            path: tx_index_path.clone(),
            source: e,
        })?;
        if let Ok(index) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(map) = index.get("by_tx_id").and_then(|v| v.as_object()) {
                for (tx_id, hash_value) in map {
                    if let Some(hash_hex) = hash_value.as_str() {
                        tx_index_entries.push((tx_id.clone(), hash_hex.to_string()));
                    }
                }
            }
        }
    }
    tx_index_entries.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical = serde_json::json!({
        "records": records
            .iter()
            .map(|(h, s, u)| serde_json::json!({
                "artifact_hash_hex": h,
                "status": s,
                "submitted_at_unix": u,
            }))
            .collect::<Vec<_>>(),
        "tx_index_entries": tx_index_entries
            .iter()
            .map(|(tx, h)| serde_json::json!({
                "tx_id": tx,
                "artifact_hash_hex": h,
            }))
            .collect::<Vec<_>>(),
    });
    let canonical_bytes = serde_json::to_vec(&canonical).map_err(|e| {
        EvidenceAnchorError::CanonicalSerialization(e.to_string())
    })?;
    Ok(blake3::hash(&canonical_bytes).to_hex().to_string())
}

fn status_tag(status: &LocalAnchorStatus) -> &'static str {
    match status {
        LocalAnchorStatus::Submitted => "submitted",
        LocalAnchorStatus::Included => "included",
        LocalAnchorStatus::Finalized => "finalized",
        LocalAnchorStatus::Failed { .. } => "failed",
    }
}

fn compute_plan_id(
    anchor_registry_dir: &str,
    registry_state_hash: &str,
    now_utc: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(anchor_registry_dir.as_bytes());
    hasher.update(b"||");
    hasher.update(registry_state_hash.as_bytes());
    hasher.update(b"||");
    hasher.update(now_utc.as_bytes());
    let hex = hasher.finalize().to_hex();
    hex[..16].to_string()
}

fn compute_plan_hash(plan: &AnchorCleanupPlan) -> EvidenceAnchorResult<String> {
    let blanked = AnchorCleanupPlan {
        cleanup_plan_hash: PLAN_HASH_BLANK.to_string(),
        ..plan.clone()
    };
    let bytes = serde_json::to_vec(&blanked).map_err(|e| {
        EvidenceAnchorError::CanonicalSerialization(e.to_string())
    })?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
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
    fn action_kind_as_str_round_trips() {
        for kind in [
            AnchorCleanupActionKind::RemoveOrphanTmpFile,
            AnchorCleanupActionKind::RemoveOrphanTxIndexEntry,
            AnchorCleanupActionKind::QuarantineMalformedRecord,
            AnchorCleanupActionKind::QuarantineStaleOpenRecord,
        ] {
            assert!(!kind.as_str().is_empty());
        }
    }

    #[test]
    fn stale_quarantine_is_tier_b_and_gated() {
        let k = AnchorCleanupActionKind::QuarantineStaleOpenRecord;
        assert!(k.is_tier_b());
        assert_eq!(k.gate_flag(), Some("--allow-stale-quarantine"));
    }

    #[test]
    fn remove_orphan_tmp_is_not_gated() {
        let k = AnchorCleanupActionKind::RemoveOrphanTmpFile;
        assert!(!k.is_tier_b());
        assert!(k.gate_flag().is_none());
    }

    #[test]
    fn plan_empty_registry_yields_empty_plan() {
        let (_dir, reg) = fresh_registry();
        let plan = plan_anchor_cleanup(
            &reg,
            &AnchorPlanOptions {
                now_utc: "2026-06-17T00:00:00Z",
                stale_threshold_secs: None,
            },
        )
        .unwrap();
        assert_eq!(plan.actions.len(), 0);
        assert_eq!(plan.schema_version, ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION);
        assert_eq!(plan.plan_id.len(), 16);
        assert!(!plan.cleanup_plan_hash.is_empty());
    }
}
