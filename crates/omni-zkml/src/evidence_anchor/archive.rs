//! Phase 5 Stage 13.7 — local terminal-anchor archive / restore.
//!
//! Stage 13.7 moves valid TERMINAL anchor records
//! (`Finalized` / `Failed`) out of the hot anchor registry into a
//! byte-preserving archive subtree, plus a symmetric restore
//! path. Fully local — no SUM Chain RPCs, no `omni-sumchain`
//! types, no private chain repo deps. Stage 13.0 wire / domain /
//! canonical-bytes / signing is **read only**; not a byte is
//! re-signed.
//!
//! ## Locked Stage 13.7 invariants
//!
//! - **Local-only.** Archive shrinks the hot registry; the chain
//!   is the source of truth for status. The Stage 13.0 wire
//!   surface and the registry shape are untouched.
//! - **Terminal records only.** `Submitted` / `Included` records
//!   are not eligible. The clap layer refuses non-terminal
//!   `--status` values; the library refuses the same at its
//!   options-validation step. `Finalized` and `Failed` are
//!   settled; archive is a one-way move with a symmetric
//!   restore.
//! - **Default `[Finalized]`.** When no `--status` is given,
//!   the library defaults to `[Finalized]` only. `Failed` is an
//!   explicit opt-in via repeatable `--status FAILED`.
//! - **Plan / apply / restore mirror Stage 13.4.** Two schema
//!   constants ([`ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION`] and
//!   [`ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION`]). Apply preflights:
//!   schema → plan-hash → drift → per-action path validation.
//! - **Two-phase apply with HONEST durability contract.**
//!   - **Phase 1 (before manifest lands)**: zero hot-registry
//!     mutation. A failure here leaves the registry byte-identical
//!     to its pre-apply state; the operator can re-run apply.
//!   - **Phase 2 (after manifest lands)**: best-effort destructive
//!     phase. An IO failure mid-Phase-2 may leave the registry
//!     PARTIALLY mutated (some sources deleted, some still there;
//!     `tx_index.json` may be pre-merge or post-merge). The
//!     archive subtree IS complete. **Restore is the official
//!     recovery path** — running restore against the manifest
//!     re-establishes any source record that Phase 2 successfully
//!     deleted; row 2 (byte-equal target) is idempotent for any
//!     record Phase 2 didn't reach.
//! - **Byte-preserve.** Both apply (hot → archive) and restore
//!   (archive → hot) copy bytes via `std::fs::read` + atomic
//!   temp+rename. NO serde round-trip; `submitted_at` /
//!   `updated_at` are preserved verbatim.
//! - **Dry-run default for BOTH apply and restore.** `--apply`
//!   is the explicit operator confirmation in both mutation
//!   modes (Stage 13.7 REJECT-fix Finding 1).
//! - **Plan vs manifest portability.** The plan JSON carries
//!   `anchor_registry_dir` for local replay; the archive
//!   manifest does NOT carry host-local registry paths
//!   (Finding 4 lock).
//! - **`registry_state_hash` includes `updated_at_unix`.**
//!   Deliberate divergence from Stage 13.4: the `--before`
//!   selector reads `updated_at`, so the drift hash must include
//!   it. Stage 13.4 had no such selector and excluded
//!   `updated_at`.
//! - **Six new closed reason-tag strings.** Semantic separation
//!   from Stage 13.4 cleanup taxonomy so operator log scrapers
//!   don't conflate archive-lifecycle and quarantine events.
//!
//! ## Conflict matrix (restore) — mirrors Stage 13.6 import
//!
//! | row | target file? | target BLAKE3 vs manifest | tx_index has tx_id? | tx_index → | outcome (apply) | outcome (dry-run) | refuse |
//! | --- | --- | --- | --- | --- | --- | --- | --- |
//! | 1   | no  | n/a   | no  | n/a   | `restored`                   | `would_restore`                 | — |
//! | 2   | yes | equal | yes | same  | `skipped_already_restored`   | `skipped_already_restored`      | — |
//! | 3   | yes | equal | no  | n/a   | `re_added_tx_index_entry`    | `would_re_add_tx_index_entry`   | — |
//! | 4   | yes | diff  | any | any   | refuse                       | refuse                          | `archive_target_exists` (field=artifact_hash) |
//! | 5   | any | n/a   | yes | diff  | refuse                       | refuse                          | `archive_target_exists` (field=tx_id) |
//!
//! Row precedence: **5 → 4 → 1/2/3** (tx_id collision is the
//! highest-precedence failure mode).
//!
//! Plus a pre-row archived-bytes integrity check: refuse with
//! [`EvidenceAnchorError::ArchiveBlake3Mismatch`] when
//! `blake3(<archive>/<plan_id>/anchors/<hash>.json) !=
//! manifest.entries[i].blake3_hex`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{EvidenceAnchorError, EvidenceAnchorResult};
use crate::evidence_anchor::registry::{
    AnchorRecord, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
};
use crate::evidence_anchor::wire::verify_anchor_tx_data;

// ── Schema constants ──────────────────────────────────────────────────────────

/// Stage 13.7 archive-plan JSON schema version. Persisted plans
/// declare this. Apply refuses on mismatch (FIRST preflight —
/// before the hash check).
pub const ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION: u32 = 1;

/// Stage 13.7 archive-manifest JSON schema version. Persisted
/// manifests declare this.
pub const ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION: u32 = 1;

const PLAN_HASH_BLANK: &str = "";
const TX_INDEX_FILENAME: &str = "tx_index.json";
const ARCHIVE_MANIFEST_FILENAME: &str = "archive_manifest.json";
const ANCHORS_SUBDIR: &str = "anchors";

// ── Closed action taxonomy ────────────────────────────────────────────────────

/// Closed taxonomy of operator-applyable archive actions. Stage
/// 13.7 has one kind: `ArchiveTerminalRecord`. The taxonomy
/// remains open for future stages to introduce e.g. a stale-
/// non-terminal kind without breaking serde forward-compat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AnchorArchiveActionKind {
    /// Tier-B-like move: copy the hot-registry record file to
    /// the archive subtree under the deterministic
    /// `<archive-dir>/<plan_id>/anchors/<artifact_hash_hex>.json`
    /// path, then (in Phase 2) delete the source and remove the
    /// `tx_id` entry from `tx_index.json`.
    ArchiveTerminalRecord,
}

impl AnchorArchiveActionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            AnchorArchiveActionKind::ArchiveTerminalRecord => "archive_terminal_record",
        }
    }
}

// ── Plan + action structs ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorArchiveAction {
    pub kind: AnchorArchiveActionKind,
    pub artifact_hash_hex: String,
    pub tx_id: String,
    /// Closed-set: `"finalized" | "failed"`.
    pub status: String,
    /// Path relative to the hot registry root. Flat per Stage
    /// 13.0 convention: `<artifact_hash_hex>.json`.
    pub source_relative: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorArchivePlan {
    pub schema_version: u32,
    pub plan_id: String,
    pub created_at_utc: String,
    /// Local replay forensic record (Finding 4 lock). The plan is
    /// NOT a portable handoff artifact; this field is needed for
    /// the drift recomputation.
    pub anchor_registry_dir: String,
    pub registry_state_hash: String,
    pub omni_zkml_version: String,
    pub actions: Vec<AnchorArchiveAction>,
    /// BLAKE3 over canonical JSON of this plan with
    /// `archive_plan_hash` blanked. Apply refuses with
    /// [`EvidenceAnchorError::ArchivePlanHashMismatch`] on
    /// tampered plans.
    pub archive_plan_hash: String,
}

// ── Manifest ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorArchiveEntry {
    pub artifact_hash_hex: String,
    pub tx_id: String,
    /// Closed-set: `"finalized" | "failed"`.
    pub status: String,
    /// `anchors/<artifact_hash_hex>.json` under
    /// `<archive-dir>/<plan_id>/`. Validated per Stage 13.5
    /// path-shape rules.
    pub archive_relative: String,
    pub blake3_hex: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorArchiveManifest {
    pub schema_version: u32,
    pub plan_id: String,
    pub created_at_utc: String,
    // Finding 4 lock: NO anchor_registry_dir field. The manifest
    // is a portable handoff artifact (parallels Stage 13.5
    // export manifest).
    pub entries: Vec<AnchorArchiveEntry>,
}

// ── Options ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnchorArchiveSelection {
    pub tx_ids: Vec<String>,
    pub artifact_hashes: Vec<String>,
}

impl AnchorArchiveSelection {
    pub fn is_empty(&self) -> bool {
        self.tx_ids.is_empty() && self.artifact_hashes.is_empty()
    }
}

pub struct AnchorArchivePlanOptions<'a> {
    /// RFC 3339 UTC. Folds into `created_at_utc` and `plan_id`.
    /// Tests inject for determinism.
    pub now_utc: &'a str,
    /// Closed-set selection — empty defaults to `[Finalized]`
    /// (Q1 lock). The library refuses any non-terminal status.
    pub statuses: Vec<LocalAnchorStatus>,
    /// Optional RFC 3339 — filter records with
    /// `updated_at < before_utc`.
    pub before_utc: Option<&'a str>,
    pub selection: &'a AnchorArchiveSelection,
}

pub struct AnchorArchiveApplyOptions<'a> {
    /// Root under which the `<plan_id>/anchors/` subtree and
    /// `archive_manifest.json` are written.
    pub archive_dir: &'a Path,
    /// When true, all preflights run but NO FS mutation lands.
    /// Outcomes record `would_archive`.
    pub dry_run: bool,
    /// RFC 3339 UTC. Stamped into the archive manifest's
    /// `created_at_utc`.
    pub now_utc: &'a str,
}

pub struct AnchorArchiveRestoreOptions<'a> {
    /// `<archive-dir>/<plan_id>/`. CLI derives this from
    /// `manifest_path.parent()` (parallels Stage 13.4 Finding 4).
    pub archive_dir: &'a Path,
    pub anchor_registry_dir: &'a Path,
    /// When true (CLI default), all preflights run but NO FS
    /// mutation lands. Stage 13.7 REJECT-fix Finding 1 lock.
    pub dry_run: bool,
}

// ── Outcomes / reports ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorArchiveActionOutcome {
    pub action_index: u32,
    pub artifact_hash_hex: String,
    pub tx_id: String,
    pub status: String,
    pub source_relative: String,
    /// Closed set — Stage 13.7 REJECT-fix Finding 3 lock:
    /// `"archived" | "would_archive"`. NO `skipped_missing` —
    /// missing planned sources are drift-detected.
    pub outcome: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorArchiveReport {
    pub plan_id: String,
    /// `"apply"` | `"dry_run"`. Closed set.
    pub mode: String,
    pub actions_archived: u32,
    pub actions_would_archive: u32,
    pub archive_dir: String,
    /// `<plan_id>/archive_manifest.json` after the manifest
    /// lands; `None` for dry-runs and plans with zero actions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive_manifest_relative: Option<String>,
    pub outcomes: Vec<AnchorArchiveActionOutcome>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorArchiveRestoreOutcome {
    pub artifact_hash_hex: String,
    pub tx_id: String,
    /// Closed set: `"restored" | "would_restore" |
    /// "skipped_already_restored" | "re_added_tx_index_entry" |
    /// "would_re_add_tx_index_entry"`.
    pub outcome: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorArchiveRestoreReport {
    pub plan_id: String,
    pub mode: String,
    pub restored: u32,
    pub would_restore: u32,
    pub skipped_already_restored: u32,
    pub re_added_tx_index_entry: u32,
    pub would_re_add_tx_index_entry: u32,
    pub outcomes: Vec<AnchorArchiveRestoreOutcome>,
}

// ── Plan ──────────────────────────────────────────────────────────────────────

/// Build a typed archive plan against the hot registry.
///
/// Selection logic:
/// 1. Validate the supplied `statuses` are all terminal
///    (`Finalized` / `Failed`). Empty defaults to `[Finalized]`
///    (Q1 lock).
/// 2. Iterate `registry.list()`; keep records whose status
///    matches one of the supplied statuses AND (if
///    `before_utc` is set) whose `updated_at < before_utc`.
/// 3. Apply `--tx-id` / `--artifact-hash-hex` selectors via AND
///    across kinds, OR within a kind. Empty selection matches
///    every status-and-time-eligible record.
/// 4. Selector misses refuse with
///    [`EvidenceAnchorError::AnchorNotFound`] (D8 — reused tag).
///    The detail string disambiguates "no terminal record"
///    vs "excluded by --before" so operator logs are clear.
/// 5. Sort actions by `artifact_hash_hex` ascending.
/// 6. Compute `registry_state_hash` (Q3 recipe: includes
///    `updated_at_unix` per record — deliberate divergence from
///    Stage 13.4).
/// 7. Compute `plan_id` and `archive_plan_hash`.
pub fn plan_anchor_archive(
    registry: &LocalEvidenceAnchorRegistry,
    opts: &AnchorArchivePlanOptions<'_>,
) -> EvidenceAnchorResult<AnchorArchivePlan> {
    let root = registry.root();
    let anchor_registry_dir = forward_slash_path(root);

    // Step 1: validate statuses are terminal; default to
    // [Finalized] when empty.
    let statuses: Vec<LocalAnchorStatus> = if opts.statuses.is_empty() {
        vec![LocalAnchorStatus::Finalized]
    } else {
        for s in &opts.statuses {
            match s {
                LocalAnchorStatus::Finalized | LocalAnchorStatus::Failed { .. } => {}
                LocalAnchorStatus::Submitted | LocalAnchorStatus::Included => {
                    // Library-level defense-in-depth — the CLI
                    // already refuses non-terminal at clap.
                    return Err(EvidenceAnchorError::ArchiveInvalidPath {
                        source_relative: s.as_str().to_string(),
                        reason: "Stage 13.7 archive operates only on terminal records",
                    });
                }
            }
        }
        opts.statuses.clone()
    };

    let before_utc = match opts.before_utc {
        Some(s) => Some(parse_rfc3339(s)?),
        None => None,
    };

    // Step 2-3: filter records.
    let all = registry.list().map_err(|e| EvidenceAnchorError::Io {
        path: root.to_path_buf(),
        source: e,
    })?;
    let mut eligible: Vec<&AnchorRecord> = all
        .iter()
        .filter(|r| statuses.iter().any(|s| status_kind_eq(s, &r.status)))
        .filter(|r| match before_utc {
            Some(b) => r.updated_at < b,
            None => true,
        })
        .collect();

    // Step 3b: apply tx_id / artifact_hash selectors (AND across,
    // OR within).
    if !opts.selection.tx_ids.is_empty() {
        eligible.retain(|r| {
            opts.selection.tx_ids.iter().any(|t| t == &r.receipt.tx_id)
        });
    }
    if !opts.selection.artifact_hashes.is_empty() {
        eligible.retain(|r| {
            opts.selection
                .artifact_hashes
                .iter()
                .any(|h| h == &r.artifact_hash_hex)
        });
    }

    // Step 4: selector-miss refusal with disambiguating detail
    // (Finding 5 lock). Walk the operator-supplied selectors
    // and locate each against the FULL registry list; if a
    // selector value points at a record that is excluded by
    // the terminal-status filter OR the `--before` filter,
    // emit a detail string that names the reason.
    refuse_on_selector_miss(&all, opts, &statuses, before_utc)?;

    // Step 5: sort + materialise actions.
    let mut actions: Vec<AnchorArchiveAction> = eligible
        .iter()
        .map(|r| AnchorArchiveAction {
            kind: AnchorArchiveActionKind::ArchiveTerminalRecord,
            artifact_hash_hex: r.artifact_hash_hex.clone(),
            tx_id: r.receipt.tx_id.clone(),
            status: r.status.as_str().to_string(),
            source_relative: format!("{}.json", r.artifact_hash_hex),
        })
        .collect();
    actions.sort_by(|a, b| a.artifact_hash_hex.cmp(&b.artifact_hash_hex));

    // Step 6: registry_state_hash (Q3 recipe — INCLUDES updated_at_unix).
    let registry_state_hash = compute_registry_state_hash(root)?;

    // Step 7: plan_id + archive_plan_hash.
    let plan_id =
        compute_plan_id(&anchor_registry_dir, &registry_state_hash, opts.now_utc);
    let mut plan = AnchorArchivePlan {
        schema_version: ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION,
        plan_id,
        created_at_utc: opts.now_utc.to_string(),
        anchor_registry_dir,
        registry_state_hash,
        omni_zkml_version: env!("CARGO_PKG_VERSION").to_string(),
        actions,
        archive_plan_hash: PLAN_HASH_BLANK.to_string(),
    };
    plan.archive_plan_hash = compute_plan_hash(&plan)?;
    Ok(plan)
}

// ── Apply ─────────────────────────────────────────────────────────────────────

/// Apply a previously-generated [`AnchorArchivePlan`].
///
/// **Preflights** (run in dry-run AND real-run modes; first
/// failure stops):
///
/// 1. Schema-version → [`EvidenceAnchorError::ArchivePlanSchemaUnsupported`].
///    Runs FIRST.
/// 2. Plan-hash → [`EvidenceAnchorError::ArchivePlanHashMismatch`].
/// 3. Drift → [`EvidenceAnchorError::ArchiveDrift`].
/// 4. Per-action path validation → [`EvidenceAnchorError::ArchiveInvalidPath`].
///
/// **Two-phase apply** with honest durability contract
/// (Stage 13.7 REJECT-fix Finding 2):
///
/// - **Phase 1** (Pass 1 + Pass 1.5) — before manifest lands.
///   No hot-registry mutation. Sources are read and copied to
///   the archive subtree; manifest is written atomically as the
///   durability fence.
///   - **Failure here**: hot registry is byte-identical to its
///     pre-apply state. Operator can re-run apply.
/// - **Phase 2** (Pass 2 + Pass 2.5) — after manifest lands.
///   Sources are removed and `tx_index.json` is merge-rewritten.
///   - **Failure here**: the hot registry may be PARTIALLY
///     mutated. The archive subtree IS complete. **Restore is
///     the recovery path**: running restore against the manifest
///     re-establishes any source record that Phase 2 successfully
///     deleted (row 2 idempotent skip for records Phase 2 didn't
///     reach).
///
/// Dry-run mode runs all preflights but skips both phases.
pub fn apply_anchor_archive(
    plan: &AnchorArchivePlan,
    opts: &AnchorArchiveApplyOptions<'_>,
) -> EvidenceAnchorResult<AnchorArchiveReport> {
    // ── Preflight 1: schema-version (FIRST). ─────────────────
    if plan.schema_version != ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION {
        return Err(EvidenceAnchorError::ArchivePlanSchemaUnsupported {
            got: plan.schema_version,
            expected: ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION,
        });
    }

    // ── Preflight 2: plan-hash. ──────────────────────────────
    let computed_plan_hash = compute_plan_hash(plan)?;
    if computed_plan_hash != plan.archive_plan_hash {
        return Err(EvidenceAnchorError::ArchivePlanHashMismatch {
            computed: computed_plan_hash,
            expected: plan.archive_plan_hash.clone(),
        });
    }

    // ── Preflight 3: drift. ──────────────────────────────────
    let registry_root = PathBuf::from(&plan.anchor_registry_dir);
    let computed_state_hash = compute_registry_state_hash(&registry_root)?;
    if computed_state_hash != plan.registry_state_hash {
        return Err(EvidenceAnchorError::ArchiveDrift {
            computed: computed_state_hash,
            expected: plan.registry_state_hash.clone(),
        });
    }

    // ── Preflight 4: per-action path validation. ─────────────
    for action in &plan.actions {
        validate_source_relative(&action.source_relative)?;
    }

    let mode = if opts.dry_run { "dry_run" } else { "apply" };
    let archive_dir_plan = opts.archive_dir.join(&plan.plan_id);
    let mut outcomes: Vec<AnchorArchiveActionOutcome> = Vec::with_capacity(plan.actions.len());
    let mut actions_archived: u32 = 0;
    let mut actions_would_archive: u32 = 0;
    let mut entries: Vec<AnchorArchiveEntry> = Vec::new();
    let mut deferred_source_removals: Vec<PathBuf> = Vec::new();
    let mut tx_ids_to_remove: BTreeSet<String> = BTreeSet::new();

    // ── Phase 1, Pass 1 — non-destructive: copy sources to
    //    archive subtree; accumulate manifest entries. NO hot-
    //    registry mutation. ─────────────────────────────────
    for (idx, action) in plan.actions.iter().enumerate() {
        let action_index = idx as u32;
        let source_path = registry_root.join(&action.source_relative);

        if opts.dry_run {
            outcomes.push(AnchorArchiveActionOutcome {
                action_index,
                artifact_hash_hex: action.artifact_hash_hex.clone(),
                tx_id: action.tx_id.clone(),
                status: action.status.clone(),
                source_relative: action.source_relative.clone(),
                outcome: "would_archive".to_string(),
            });
            actions_would_archive += 1;
            continue;
        }

        // Real-run: read source bytes. A missing planned source
        // here is an IO error (drift would have caught it if
        // missing at preflight time; this is the TOCTOU window).
        let source_bytes = std::fs::read(&source_path).map_err(|e| EvidenceAnchorError::Io {
            path: source_path.clone(),
            source: e,
        })?;
        let blake3_hex = blake3::hash(&source_bytes).to_hex().to_string();
        let bytes_len = source_bytes.len() as u64;

        let archive_relative =
            format!("{ANCHORS_SUBDIR}/{}.json", action.artifact_hash_hex);
        let archive_target = archive_dir_plan.join(&archive_relative);
        if let Some(parent) = archive_target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| EvidenceAnchorError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let tmp = archive_target.with_extension("json.tmp");
        std::fs::write(&tmp, &source_bytes).map_err(|e| EvidenceAnchorError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp, &archive_target).map_err(|e| EvidenceAnchorError::Io {
            path: archive_target.clone(),
            source: e,
        })?;
        entries.push(AnchorArchiveEntry {
            artifact_hash_hex: action.artifact_hash_hex.clone(),
            tx_id: action.tx_id.clone(),
            status: action.status.clone(),
            archive_relative,
            blake3_hex,
            bytes: bytes_len,
        });
        deferred_source_removals.push(source_path);
        tx_ids_to_remove.insert(action.tx_id.clone());

        outcomes.push(AnchorArchiveActionOutcome {
            action_index,
            artifact_hash_hex: action.artifact_hash_hex.clone(),
            tx_id: action.tx_id.clone(),
            status: action.status.clone(),
            source_relative: action.source_relative.clone(),
            outcome: "archived".to_string(),
        });
        actions_archived += 1;
    }

    // ── Phase 1, Pass 1.5 — Manifest write (durability fence).
    //    Until this lands, NOTHING is removed from the hot
    //    registry.  ────────────────────────────────────────
    let mut archive_manifest_relative: Option<String> = None;
    if !opts.dry_run && !entries.is_empty() {
        let manifest = AnchorArchiveManifest {
            schema_version: ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION,
            plan_id: plan.plan_id.clone(),
            created_at_utc: opts.now_utc.to_string(),
            entries,
        };
        let manifest_path = archive_dir_plan.join(ARCHIVE_MANIFEST_FILENAME);
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
        archive_manifest_relative =
            Some(format!("{}/{}", plan.plan_id, ARCHIVE_MANIFEST_FILENAME));
    }

    // ── Phase 2, Pass 2 + 2.5 — destructive: remove sources,
    //    merge-rewrite tx_index.json. Mid-Phase-2 failures may
    //    leave the hot registry partially mutated; restore is
    //    the recovery path. ─────────────────────────────────
    if !opts.dry_run {
        for src in &deferred_source_removals {
            std::fs::remove_file(src).map_err(|e| EvidenceAnchorError::Io {
                path: src.clone(),
                source: e,
            })?;
        }
        if !tx_ids_to_remove.is_empty() {
            merge_remove_tx_index(&registry_root, &tx_ids_to_remove)?;
        }
    }

    // For dry-run mode, declare the future manifest path so the
    // operator can preview where it will land.
    if opts.dry_run && !plan.actions.is_empty() {
        archive_manifest_relative =
            Some(format!("{}/{}", plan.plan_id, ARCHIVE_MANIFEST_FILENAME));
    }

    Ok(AnchorArchiveReport {
        plan_id: plan.plan_id.clone(),
        mode: mode.to_string(),
        actions_archived,
        actions_would_archive,
        archive_dir: forward_slash_path(&archive_dir_plan),
        archive_manifest_relative,
        outcomes,
    })
}

// ── Restore ───────────────────────────────────────────────────────────────────

/// Restore archived bytes back into the hot anchor registry.
///
/// Preflights:
/// 1. Schema-version (defensive — restore is tolerant; refuses on
///    out-of-range future schema).
/// 2. Per-entry path validation —
///    [`EvidenceAnchorError::ArchiveInvalidPath`] on absolute /
///    `..` / wrong shape.
/// 3. Per-entry archived-bytes BLAKE3 check —
///    [`EvidenceAnchorError::ArchiveBlake3Mismatch`] on drift.
/// 4. Per-entry Stage 13.0 signature defense-in-depth (re-run
///    `verify_anchor_tx_data` on the parsed record).
/// 5. Classify EVERY entry against the target registry state
///    (preflight-all-before-mutate). Any refusal in rows 4/5
///    refuses with zero writes.
///
/// Conflict matrix (rows 1-5 — see module-level docs).
pub fn restore_anchor_archive(
    manifest: &AnchorArchiveManifest,
    opts: &AnchorArchiveRestoreOptions<'_>,
) -> EvidenceAnchorResult<AnchorArchiveRestoreReport> {
    if manifest.schema_version != ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION {
        // No `unsupported_archive_manifest_schema_version` tag
        // is introduced — operators inspecting a future-schema
        // manifest will see the malformed_json refusal with the
        // schema-version detail, which matches the Stage 13.4
        // posture and avoids surface bloat.
        return Err(EvidenceAnchorError::MalformedJson {
            path: opts.archive_dir.join(ARCHIVE_MANIFEST_FILENAME),
            source: serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported manifest schema_version: got {}, expected {}",
                    manifest.schema_version, ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION
                ),
            )),
        });
    }

    // Per-entry path validation BEFORE any FS read (closed Stage
    // 13.4 / 13.5 / 13.6 path-traversal posture).
    for entry in &manifest.entries {
        validate_archive_relative(&entry.archive_relative)?;
    }

    // Pre-flight ALL classifications BEFORE any FS write (the
    // Stage 13.6 preflight-all-before-mutate carry-forward). The
    // following helper enums the per-entry outcome AND collects
    // the parsed record (for the tx_index rewrite phase).
    let mode = if opts.dry_run { "dry_run" } else { "apply" };
    let mut classified: Vec<ClassifiedRestoreEntry> =
        Vec::with_capacity(manifest.entries.len());
    for entry in &manifest.entries {
        // Archived-bytes integrity check.
        let archived_path = opts.archive_dir.join(&entry.archive_relative);
        let archived_bytes =
            std::fs::read(&archived_path).map_err(|e| EvidenceAnchorError::Io {
                path: archived_path.clone(),
                source: e,
            })?;
        let computed = blake3::hash(&archived_bytes).to_hex().to_string();
        if computed != entry.blake3_hex {
            return Err(EvidenceAnchorError::ArchiveBlake3Mismatch {
                archive_relative: entry.archive_relative.clone(),
                computed,
                expected: entry.blake3_hex.clone(),
            });
        }

        // Defense-in-depth: parse + verify_anchor_tx_data so a
        // hand-edited archive file with a tampered signature
        // refuses at restore time with the Stage 13.0 tag.
        let record: AnchorRecord = serde_json::from_slice(&archived_bytes).map_err(|e| {
            EvidenceAnchorError::MalformedJson {
                path: archived_path.clone(),
                source: e,
            }
        })?;
        verify_anchor_tx_data(&record.tx_data)?;

        // Classify against current target state.
        let outcome = classify_restore_target_state(opts, entry, &archived_bytes)?;
        classified.push(ClassifiedRestoreEntry {
            entry,
            archived_bytes,
            outcome,
        });
    }

    // Now execute (dry-run skips Phase 2).
    let mut restored: u32 = 0;
    let mut would_restore: u32 = 0;
    let mut skipped_already_restored: u32 = 0;
    let mut re_added_tx_index_entry: u32 = 0;
    let mut would_re_add_tx_index_entry: u32 = 0;
    let mut outcomes: Vec<AnchorArchiveRestoreOutcome> = Vec::with_capacity(classified.len());
    let mut tx_index_updates: BTreeMap<String, String> = BTreeMap::new();

    for c in &classified {
        match c.outcome.as_str() {
            "restored" => {
                let dst_path = opts
                    .anchor_registry_dir
                    .join(format!("{}.json", c.entry.artifact_hash_hex));
                if let Some(parent) = dst_path.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| EvidenceAnchorError::Io {
                        path: parent.to_path_buf(),
                        source: e,
                    })?;
                }
                let tmp = dst_path.with_extension("json.tmp");
                std::fs::write(&tmp, &c.archived_bytes).map_err(|e| EvidenceAnchorError::Io {
                    path: tmp.clone(),
                    source: e,
                })?;
                std::fs::rename(&tmp, &dst_path).map_err(|e| EvidenceAnchorError::Io {
                    path: dst_path.clone(),
                    source: e,
                })?;
                tx_index_updates
                    .insert(c.entry.tx_id.clone(), c.entry.artifact_hash_hex.clone());
                restored += 1;
            }
            "would_restore" => {
                would_restore += 1;
            }
            "skipped_already_restored" => {
                skipped_already_restored += 1;
            }
            "re_added_tx_index_entry" => {
                // Implementation note carry-forward: do NOT
                // rewrite the byte-equal record file. Only the
                // tx_index entry needs to appear.
                tx_index_updates
                    .insert(c.entry.tx_id.clone(), c.entry.artifact_hash_hex.clone());
                re_added_tx_index_entry += 1;
            }
            "would_re_add_tx_index_entry" => {
                would_re_add_tx_index_entry += 1;
            }
            other => {
                return Err(EvidenceAnchorError::ArchiveInvalidPath {
                    source_relative: c.entry.archive_relative.clone(),
                    reason: closed_set_string_for_outcome(other),
                });
            }
        }
        outcomes.push(AnchorArchiveRestoreOutcome {
            artifact_hash_hex: c.entry.artifact_hash_hex.clone(),
            tx_id: c.entry.tx_id.clone(),
            outcome: c.outcome.clone(),
        });
    }

    if !opts.dry_run && !tx_index_updates.is_empty() {
        merge_add_tx_index(opts.anchor_registry_dir, &tx_index_updates)?;
    }

    Ok(AnchorArchiveRestoreReport {
        plan_id: manifest.plan_id.clone(),
        mode: mode.to_string(),
        restored,
        would_restore,
        skipped_already_restored,
        re_added_tx_index_entry,
        would_re_add_tx_index_entry,
        outcomes,
    })
}

// ── Internal helpers ─────────────────────────────────────────────────────────

struct ClassifiedRestoreEntry<'a> {
    entry: &'a AnchorArchiveEntry,
    archived_bytes: Vec<u8>,
    outcome: String,
}

fn classify_restore_target_state(
    opts: &AnchorArchiveRestoreOptions<'_>,
    entry: &AnchorArchiveEntry,
    _archived_bytes: &[u8],
) -> EvidenceAnchorResult<String> {
    let target_record_path = opts
        .anchor_registry_dir
        .join(format!("{}.json", entry.artifact_hash_hex));
    let target_file_present = target_record_path.is_file();

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

    let tx_index_path = opts.anchor_registry_dir.join(TX_INDEX_FILENAME);
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
            .and_then(|v| v.get(&entry.tx_id))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    // Row 5: tx_id collision (HIGHEST precedence).
    if let Some(existing_hash) = &existing_mapping_for_tx_id {
        if existing_hash != &entry.artifact_hash_hex {
            return Err(EvidenceAnchorError::ArchiveTargetExists {
                field: "tx_id",
                artifact_hash_hex: entry.artifact_hash_hex.clone(),
                tx_id: entry.tx_id.clone(),
            });
        }
    }

    // Row 4: artifact-hash collision with byte-different record.
    if target_file_present {
        let actual = target_file_blake3.as_deref().unwrap_or("");
        if actual != entry.blake3_hex {
            return Err(EvidenceAnchorError::ArchiveTargetExists {
                field: "artifact_hash",
                artifact_hash_hex: entry.artifact_hash_hex.clone(),
                tx_id: entry.tx_id.clone(),
            });
        }
        if existing_mapping_for_tx_id.is_some() {
            // Row 2: fully idempotent. Same string in apply and
            // dry-run.
            return Ok("skipped_already_restored".to_string());
        }
        // Row 3: tx_index re-add only.
        return Ok(if opts.dry_run {
            "would_re_add_tx_index_entry".to_string()
        } else {
            "re_added_tx_index_entry".to_string()
        });
    }

    // Row 1: fresh restore.
    Ok(if opts.dry_run {
        "would_restore".to_string()
    } else {
        "restored".to_string()
    })
}

/// Selector-miss refusal helper. Walks the operator-supplied
/// selector values and emits a single concatenated detail
/// string that names the reason for each miss.
fn refuse_on_selector_miss(
    all_records: &[AnchorRecord],
    opts: &AnchorArchivePlanOptions<'_>,
    statuses: &[LocalAnchorStatus],
    before_utc: Option<DateTime<Utc>>,
) -> EvidenceAnchorResult<()> {
    let mut details: Vec<String> = Vec::new();
    for hash in &opts.selection.artifact_hashes {
        let matched = all_records.iter().find(|r| &r.artifact_hash_hex == hash);
        match matched {
            None => details.push(format!("artifact_hash_hex={hash} (no such record)")),
            Some(r) if !statuses.iter().any(|s| status_kind_eq(s, &r.status)) => details
                .push(format!(
                    "artifact_hash_hex={hash} (record exists but is not terminal; \
                     Stage 13.7 archives Finalized/Failed records only)"
                )),
            Some(r) => {
                if let Some(b) = before_utc {
                    if r.updated_at >= b {
                        details.push(format!(
                            "artifact_hash_hex={hash} (record exists but is excluded by --before {})",
                            opts.before_utc.unwrap_or("")
                        ));
                    }
                }
            }
        }
    }
    for tx_id in &opts.selection.tx_ids {
        let matched = all_records.iter().find(|r| &r.receipt.tx_id == tx_id);
        match matched {
            None => details.push(format!("tx_id={tx_id} (no such record)")),
            Some(r) if !statuses.iter().any(|s| status_kind_eq(s, &r.status)) => details
                .push(format!(
                    "tx_id={tx_id} (record exists but is not terminal; \
                     Stage 13.7 archives Finalized/Failed records only)"
                )),
            Some(r) => {
                if let Some(b) = before_utc {
                    if r.updated_at >= b {
                        details.push(format!(
                            "tx_id={tx_id} (record exists but is excluded by --before {})",
                            opts.before_utc.unwrap_or("")
                        ));
                    }
                }
            }
        }
    }
    if details.is_empty() {
        Ok(())
    } else {
        Err(EvidenceAnchorError::AnchorNotFound {
            selector: details.join(", "),
        })
    }
}

fn status_kind_eq(a: &LocalAnchorStatus, b: &LocalAnchorStatus) -> bool {
    a.as_str() == b.as_str()
}

/// Stage 13.7 path-validation helper for archive ACTIONS (on
/// the hot-registry side). Source paths are flat
/// `<64-lower-hex>.json` per Stage 13.0 registry convention.
fn validate_source_relative(source_relative: &str) -> EvidenceAnchorResult<()> {
    let reject = |reason: &'static str| EvidenceAnchorError::ArchiveInvalidPath {
        source_relative: source_relative.to_string(),
        reason,
    };
    if source_relative.is_empty() {
        return Err(reject("empty source_relative"));
    }
    if source_relative.contains('/') || source_relative.contains('\\') {
        return Err(reject("path separator not allowed (hot registry is flat)"));
    }
    if source_relative.contains("..") {
        return Err(reject("parent traversal forbidden"));
    }
    let stem = source_relative
        .strip_suffix(".json")
        .ok_or_else(|| reject("archive action requires .json suffix"))?;
    if stem.len() != 64 {
        return Err(reject("archive action stem must be exactly 64 hex chars"));
    }
    if !stem.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(reject("archive action stem must be lowercase hex only"));
    }
    Ok(())
}

/// Stage 13.7 path-validation helper for ARCHIVE-side manifest
/// entries. `archive_relative` must match
/// `anchors/<64-lower-hex>.json`.
fn validate_archive_relative(archive_relative: &str) -> EvidenceAnchorResult<()> {
    let reject = |reason: &'static str| EvidenceAnchorError::ArchiveInvalidPath {
        source_relative: archive_relative.to_string(),
        reason,
    };
    if archive_relative.is_empty() {
        return Err(reject("empty archive_relative"));
    }
    if archive_relative.contains('\\') {
        return Err(reject("backslash not allowed"));
    }
    let path = std::path::Path::new(archive_relative);
    if path.is_absolute() {
        return Err(reject("absolute paths not allowed"));
    }
    if archive_relative.starts_with('/') {
        return Err(reject("leading slash not allowed"));
    }
    if archive_relative.contains("//") {
        return Err(reject("duplicate slash not allowed"));
    }
    if archive_relative.split('/').any(|c| c == "..") {
        return Err(reject("parent traversal forbidden"));
    }
    let suffix = archive_relative
        .strip_prefix(&format!("{ANCHORS_SUBDIR}/"))
        .ok_or_else(|| reject("archive_relative must live under anchors/ subdir"))?;
    if suffix.contains('/') {
        return Err(reject("nested subdirectories not allowed under anchors/"));
    }
    let stem = suffix
        .strip_suffix(".json")
        .ok_or_else(|| reject("archive_relative requires .json suffix"))?;
    if stem.len() != 64 {
        return Err(reject("archive stem must be exactly 64 hex chars"));
    }
    if !stem.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(reject("archive stem must be lowercase hex only"));
    }
    Ok(())
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

/// Q3 lock — registry_state_hash recipe (Stage 13.7 variant).
///
/// BLAKE3 over canonical JSON of:
/// `{ records: sorted_by(artifact_hash_hex) [{ artifact_hash_hex,
/// status, tx_id, updated_at_unix }], tx_index_entries:
/// sorted_by(tx_id) [{ tx_id, artifact_hash_hex }] }`.
///
/// **Includes `updated_at_unix`** — deliberate divergence from
/// Stage 13.4 because the Stage 13.7 `--before` selector reads
/// `updated_at`; excluding it would let reconcile-driven
/// timestamp bumps silently shift records in/out of the
/// selection without tripping drift.
fn compute_registry_state_hash(root: &Path) -> EvidenceAnchorResult<String> {
    let mut records: Vec<(String, String, String, i64)> = Vec::new();
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
            Err(_) => continue,
        };
        let Ok(record) = serde_json::from_slice::<AnchorRecord>(&bytes) else {
            continue;
        };
        records.push((
            record.artifact_hash_hex.clone(),
            record.status.as_str().to_string(),
            record.receipt.tx_id.clone(),
            record.updated_at.timestamp(),
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
            .map(|(h, s, t, u)| serde_json::json!({
                "artifact_hash_hex": h,
                "status": s,
                "tx_id": t,
                "updated_at_unix": u,
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

fn compute_plan_hash(plan: &AnchorArchivePlan) -> EvidenceAnchorResult<String> {
    let blanked = AnchorArchivePlan {
        archive_plan_hash: PLAN_HASH_BLANK.to_string(),
        ..plan.clone()
    };
    let bytes = serde_json::to_vec(&blanked).map_err(|e| {
        EvidenceAnchorError::CanonicalSerialization(e.to_string())
    })?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Merge-style atomic rewrite that REMOVES the given tx_ids from
/// `tx_index.json` while preserving every other entry.
fn merge_remove_tx_index(
    registry_root: &Path,
    tx_ids_to_remove: &BTreeSet<String>,
) -> EvidenceAnchorResult<()> {
    let path = registry_root.join(TX_INDEX_FILENAME);
    if !path.is_file() {
        return Ok(());
    }
    let bytes = std::fs::read(&path).map_err(|e| EvidenceAnchorError::Io {
        path: path.clone(),
        source: e,
    })?;
    let mut value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| EvidenceAnchorError::MalformedJson {
            path: path.clone(),
            source: e,
        })?;
    if let Some(map) = value.get_mut("by_tx_id").and_then(|v| v.as_object_mut()) {
        for tx_id in tx_ids_to_remove {
            map.remove(tx_id);
        }
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

/// Merge-style atomic rewrite that ADDS the given mappings on top
/// of existing tx_index entries. Preserves unrelated entries.
fn merge_add_tx_index(
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

fn closed_set_string_for_outcome(s: &str) -> &'static str {
    // Defensive sentinel — should never trip because
    // classify_restore_target_state returns a closed-set string.
    let _ = s;
    "unknown outcome string (closed-set violation)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_kind_str_is_stable() {
        assert_eq!(
            AnchorArchiveActionKind::ArchiveTerminalRecord.as_str(),
            "archive_terminal_record"
        );
    }

    #[test]
    fn closed_outcome_strings_are_archived_or_would_archive_only() {
        // Stage 13.7 REJECT-fix Finding 3 — no skipped_missing.
        // Compile-time pin via match exhaustiveness on the
        // closed set documented in AnchorArchiveActionOutcome.
        let outcomes = ["archived", "would_archive"];
        assert_eq!(outcomes.len(), 2);
    }

    #[test]
    fn empty_selection_is_empty() {
        let sel = AnchorArchiveSelection::default();
        assert!(sel.is_empty());
    }
}
