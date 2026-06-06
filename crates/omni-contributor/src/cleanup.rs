//! Stage 12.17 — local state-dir cleanup planner / applier.
//!
//! Two-step operator-driven flow that composes the Stage 12.16
//! `scan_state_integrity` findings into a deterministic JSON
//! plan + a typed applier that walks the plan one action at a
//! time. Mirrors the Stage 12.10 / 12.11 `SessionRepairPlan`
//! posture: plan is operator-reviewable JSON, apply
//! re-projects state for drift detection before mutating.
//!
//! **Scope and safety**:
//!
//! - **Tier A** actions (`RemoveSeenMarker`, `WriteSeenMarker`,
//!   `RemoveSeenFile`) are reversible idempotency-hint edits
//!   under `seen/...` — no recoverable payload is lost.
//! - **Tier B** actions (`QuarantineVerifiedFile`,
//!   `QuarantineAndUnmark*`) copy the verified body bytes into
//!   `<quarantine-dir>/<plan_id>/<source_relative>` (BLAKE3
//!   verified) BEFORE removing the source. The applier writes
//!   a `quarantine-manifest.json` LAST mirroring Stage 12.14's
//!   manifest-written-last invariant; an aborted apply leaves
//!   the manifest visibly missing without source corruption.
//! - **Gated** actions (`QuarantineAndUnmarkPartial`,
//!   `QuarantineAndUnmarkOrphanAssignment`) plan freely but
//!   refuse at apply time unless the operator passes
//!   `--allow-invalid-partial-cleanup` /
//!   `--allow-orphan-assignments` respectively.
//! - **Out of scope (v1)**: `InvalidSession` and
//!   `InvalidAggregate` cleanup. Removing a session.json
//!   cascades the whole session out and is operator-routed
//!   through Stage 12.14 `archive-session --move`; removing an
//!   aggregate is a material status change and stays manual.
//! - **No protocol surface**: no envelope, no canonical-byte
//!   changes, no `STATE_VERSION` bump, no SNIP / mesh / chain
//!   / payment / proof / marketplace surface.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::CleanupError;
use crate::integrity::{
    scan_state_integrity_with_audit_orphans, FindingKind, IntegrityFinding,
    ScanOptions, StateIntegrityReport,
};
use crate::state::{ContributorStateStore, StateNamespace, STATE_VERSION};
use crate::status::build_session_status_report;
use crate::resume::{compute_audit_health, AuditCoherence};

/// Stage 12.17 — plan / report / quarantine-manifest schema
/// version. Bumping this is a forward-incompatible change.
pub const CLEANUP_PLAN_SCHEMA_VERSION: u32 = 1;

/// Quarantine manifest schema version (kept distinct from the
/// plan version so a future-stage planner can co-exist with the
/// v1 quarantine layout).
pub const QUARANTINE_MANIFEST_SCHEMA_VERSION: u32 = 1;

// ── CleanupActionKind ─────────────────────────────────────────

/// Closed taxonomy of operator-applyable cleanup actions. Each
/// variant maps 1:1 to a [`FindingKind`] family the Stage 12.17
/// planner knows how to translate. Renaming a variant is a
/// contract break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CleanupActionKind {
    /// Tier A — delete a well-formed `seen/<ns>/<key>` marker
    /// whose verified body is missing.
    RemoveSeenMarker,
    /// Tier A — write a missing `seen/<ns>/<key>` marker for a
    /// verified body that's on disk and trusted.
    WriteSeenMarker,
    /// Tier A — delete a stray file under `seen/...` (unknown
    /// namespace dir OR shape-malformed key).
    RemoveSeenFile,
    /// Tier B — quarantine + remove a `verified/sessions/<id>/...`
    /// file whose relative path doesn't match the documented
    /// layout (typically operator-left `.tmp` / `.bak`).
    QuarantineVerifiedFile,
    /// Tier B — quarantine the join body + remove the matching
    /// `seen/joins/<sid>--<pubkey>` marker.
    QuarantineAndUnmarkJoin,
    /// Tier B — same shape for assignments.
    QuarantineAndUnmarkAssignment,
    /// Tier B, **gated**. Requires
    /// `--allow-invalid-partial-cleanup`. Quarantine the
    /// partial + remove the matching seen marker.
    QuarantineAndUnmarkPartial,
    /// Tier B — quarantine the supersession + remove the
    /// matching `seen/assignment-supersessions/<sid>--<sup>`
    /// marker. Replacement / superseded assignments are left in
    /// place.
    QuarantineAndUnmarkSupersession,
    /// Tier B, **gated**. Requires `--allow-orphan-assignments`.
    /// Quarantine an orphan replacement assignment (Phase-B
    /// partial-apply leftover) + remove the matching seen
    /// marker. Apply re-runs the audit projection per session
    /// and refuses if the orphan id set has changed.
    QuarantineAndUnmarkOrphanAssignment,
}

impl CleanupActionKind {
    /// Stable kebab/snake-case wire tag. Mirrors the
    /// `FindingKind::as_str` convention.
    pub fn as_str(self) -> &'static str {
        match self {
            CleanupActionKind::RemoveSeenMarker => "remove_seen_marker",
            CleanupActionKind::WriteSeenMarker => "write_seen_marker",
            CleanupActionKind::RemoveSeenFile => "remove_seen_file",
            CleanupActionKind::QuarantineVerifiedFile => "quarantine_verified_file",
            CleanupActionKind::QuarantineAndUnmarkJoin => "quarantine_and_unmark_join",
            CleanupActionKind::QuarantineAndUnmarkAssignment => {
                "quarantine_and_unmark_assignment"
            }
            CleanupActionKind::QuarantineAndUnmarkPartial => {
                "quarantine_and_unmark_partial"
            }
            CleanupActionKind::QuarantineAndUnmarkSupersession => {
                "quarantine_and_unmark_supersession"
            }
            CleanupActionKind::QuarantineAndUnmarkOrphanAssignment => {
                "quarantine_and_unmark_orphan_assignment"
            }
        }
    }

    /// Tier B actions require a quarantine destination + write
    /// the body bytes there BEFORE deleting the source.
    pub fn is_tier_b(self) -> bool {
        matches!(
            self,
            CleanupActionKind::QuarantineVerifiedFile
                | CleanupActionKind::QuarantineAndUnmarkJoin
                | CleanupActionKind::QuarantineAndUnmarkAssignment
                | CleanupActionKind::QuarantineAndUnmarkPartial
                | CleanupActionKind::QuarantineAndUnmarkSupersession
                | CleanupActionKind::QuarantineAndUnmarkOrphanAssignment
        )
    }

    /// Gate flag this action requires at apply time, or `None`
    /// if it's freely applyable. The CLI flag name is the human
    /// label; the typed [`CleanupError::GateRequired`] carries
    /// it verbatim.
    pub fn gate_flag(self) -> Option<&'static str> {
        match self {
            CleanupActionKind::QuarantineAndUnmarkPartial => {
                Some("--allow-invalid-partial-cleanup")
            }
            CleanupActionKind::QuarantineAndUnmarkOrphanAssignment => {
                Some("--allow-orphan-assignments")
            }
            _ => None,
        }
    }
}

// ── CleanupAction ─────────────────────────────────────────────

/// One unit of work in a [`StateCleanupPlan`]. Per-field
/// semantics depend on `kind`:
///
/// - `Remove*`: `path` is the file to remove.
/// - `WriteSeenMarker`: `path` is the verified body whose
///   absence-of-marker the planner detected; the applier
///   derives the seen-side path from it.
/// - `Quarantine*` / `QuarantineAndUnmark*`: `path` is the
///   verified body to quarantine, and (for `QuarantineAndUnmark*`)
///   `seen_marker_path` is the matching `seen/...` marker.
///
/// `source_finding_kind` and `source_reason_tag` mirror the
/// originating Stage 12.16 finding for forensics; they have
/// no apply-time effect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupAction {
    pub kind: CleanupActionKind,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_id: Option<String>,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub seen_marker_path: Option<String>,
    pub source_finding_kind: String,
    pub source_reason_tag: String,
}

// ── StateCleanupPlan ──────────────────────────────────────────

/// Operator-reviewable JSON plan. Built from a
/// [`StateIntegrityReport`] + the Stage 12.17 orphan side-channel.
/// `source_integrity_hash` is re-projected at apply time;
/// `cleanup_plan_hash` proves the plan body wasn't hand-edited.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateCleanupPlan {
    pub schema_version: u32,
    /// 16-char lowercase hex BLAKE3 prefix of
    /// `(state_dir || source_integrity_hash || created_at_utc)`.
    /// Scopes the plan + its quarantine subtree.
    pub plan_id: String,
    pub created_at_utc: String,
    /// Forward-slash normalized state-dir path the plan was
    /// built against.
    pub state_dir: String,
    /// BLAKE3 of the canonical projection of the
    /// `StateIntegrityReport` with `generated_at_utc` and
    /// `state_dir` blanked. Apply-time recompute → typed drift
    /// error.
    pub source_integrity_hash: String,
    pub omni_contributor_version: String,
    pub actions: Vec<CleanupAction>,
    /// BLAKE3 over the canonical JSON of this plan with
    /// `cleanup_plan_hash` cleared. Same recipe as
    /// `repair_plan_hash_hex` (Stage 12.11).
    pub cleanup_plan_hash: String,
}

// ── Plan options ──────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct PlanOptions<'a> {
    /// RFC 3339 UTC. Stamped into `created_at_utc` and used to
    /// derive `plan_id`.
    pub now_utc: &'a str,
    /// `None` = plan over every session touched by the report.
    /// `Some(hex)` = filter the plan's session-scoped actions
    /// to one session id (cross-session strays still planned).
    pub session_id_filter: Option<&'a str>,
}

// ── Apply options ─────────────────────────────────────────────

#[derive(Debug)]
pub struct ApplyOptions<'a> {
    /// Root under which `<plan_id>/...` quarantine subtree is
    /// written. Required even for tier-A-only plans so the
    /// quarantine-manifest goes somewhere consistent. The
    /// caller is responsible for ensuring the directory exists
    /// and is writable.
    pub quarantine_dir: &'a Path,
    /// When true, no FS writes / removes happen; outcomes
    /// record `would_apply` and the report's `mode` is
    /// `dry_run`. The drift / hash / gate checks still run so
    /// dry-run is a real preflight.
    pub dry_run: bool,
    /// Required for any `QuarantineAndUnmarkPartial` action.
    pub allow_invalid_partial_cleanup: bool,
    /// Required for any `QuarantineAndUnmarkOrphanAssignment`
    /// action.
    pub allow_orphan_assignments: bool,
    /// When true, `QuarantineVerifiedFile` and `RemoveSeenFile`
    /// actions skip the quarantine copy and just delete the
    /// source. Tier-B integrity-bearing actions
    /// (`QuarantineAndUnmark*`) still quarantine — the flag
    /// only relaxes the stray-file pathways.
    pub purge_stray: bool,
    /// RFC 3339 UTC. Used to re-run `scan_state_integrity` for
    /// drift detection.
    pub now_utc: &'a str,
}

// ── Action outcome / report ───────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupActionOutcome {
    pub action_index: u32,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_id: Option<String>,
    pub path: String,
    /// Closed set:
    /// - `"applied"` — action mutated the FS successfully.
    /// - `"would_apply"` — dry-run mode.
    /// - `"skipped_missing"` — idempotent miss (file already gone).
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupReport {
    pub plan_id: String,
    /// `"apply"` / `"dry_run"`. Closed set.
    pub mode: String,
    pub actions_applied: u32,
    pub actions_dry_run: u32,
    pub actions_skipped: u32,
    /// Forward-slash normalized path to `<quarantine-dir>/<plan_id>/`.
    pub quarantine_dir: String,
    /// Relative path (under `quarantine_dir`) to the manifest,
    /// or `None` for dry-run / tier-A-only plans where no
    /// manifest was written.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub quarantine_manifest_relative: Option<String>,
    pub outcomes: Vec<CleanupActionOutcome>,
}

// ── Quarantine manifest ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuarantineEntry {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_id: Option<String>,
    /// Path under the source state-dir.
    pub source_relative: String,
    /// Path under `<quarantine-dir>/<plan_id>/` — Stage 12.14
    /// shape `<source_relative>` mirrored verbatim.
    pub quarantine_relative: String,
    pub blake3_hex: String,
    pub bytes: u64,
    pub source_finding_kind: String,
    pub source_reason_tag: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuarantineManifest {
    pub schema_version: u32,
    pub plan_id: String,
    pub created_at_utc: String,
    pub state_dir: String,
    pub omni_contributor_version: String,
    pub source_state_version: u32,
    pub files: Vec<QuarantineEntry>,
}

// ── Public hashes ─────────────────────────────────────────────

/// BLAKE3 of the canonical projection of `report` with
/// `generated_at_utc` and `state_dir` blanked. Findings are
/// already sorted deterministically by the scanner, so the
/// projection is stable across runs on the same content.
pub fn source_integrity_hash_hex(report: &StateIntegrityReport) -> String {
    let mut projection = report.clone();
    projection.generated_at_utc = String::new();
    projection.state_dir = String::new();
    let bytes = serde_json::to_vec(&projection).expect("serialize integrity projection");
    let h = blake3::hash(&bytes);
    let mut s = String::with_capacity(64);
    for b in h.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// BLAKE3 of the canonical JSON of `plan` with
/// `cleanup_plan_hash` cleared. Same recipe as Stage 12.11
/// `repair_plan_hash_hex`.
pub fn cleanup_plan_hash_hex(plan: &StateCleanupPlan) -> String {
    let mut cloned = plan.clone();
    cloned.cleanup_plan_hash = String::new();
    let bytes = serde_json::to_vec(&cloned).expect("serialize cleanup plan");
    let h = blake3::hash(&bytes);
    let mut s = String::with_capacity(64);
    for b in h.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn compute_plan_id(state_dir: &str, source_hash: &str, created_at: &str) -> String {
    let mut h = blake3::Hasher::new();
    h.update(state_dir.as_bytes());
    h.update(b"\0");
    h.update(source_hash.as_bytes());
    h.update(b"\0");
    h.update(created_at.as_bytes());
    let bytes = h.finalize();
    let mut s = String::with_capacity(16);
    for b in &bytes.as_bytes()[..8] {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ── Public planner ────────────────────────────────────────────

/// Build a deterministic [`StateCleanupPlan`] from the Stage
/// 12.16 report + the Stage 12.17 orphan side-channel.
///
/// `audit_orphans` maps `session_id → orphan_assignment_ids` as
/// returned by [`scan_state_integrity_with_audit_orphans`]. The
/// planner consumes one `QuarantineAndUnmarkOrphanAssignment`
/// per orphan id; apply-time re-projection verifies the orphan
/// set didn't change.
///
/// Findings outside the v1 cleanup tier (e.g. `InvalidSession`,
/// `InvalidAggregate`, `PartialApplySupersession`, archive
/// findings) are silently dropped — the cleanup planner has no
/// safe v1 action for them.
pub fn plan_state_cleanup(
    report: &StateIntegrityReport,
    audit_orphans: &HashMap<String, Vec<String>>,
    opts: &PlanOptions<'_>,
) -> Result<StateCleanupPlan, CleanupError> {
    let mut actions: Vec<CleanupAction> = Vec::new();
    for f in &report.findings {
        if let Some(filter) = opts.session_id_filter {
            // Drop session-scoped findings outside the filter.
            // Findings without a session_id (cross-session
            // strays) still plan — same convention as Stage
            // 12.16's filter behavior.
            if let Some(sid) = f.session_id.as_deref() {
                if sid != filter {
                    continue;
                }
            }
        }
        match action_for_finding(f) {
            Some(action) => actions.push(action),
            None => continue,
        }
    }
    // Expand orphan side-channel into one action per id.
    let mut orphan_session_ids: Vec<&String> = audit_orphans.keys().collect();
    orphan_session_ids.sort();
    for sid in orphan_session_ids {
        if let Some(filter) = opts.session_id_filter {
            if sid != filter {
                continue;
            }
        }
        let ids = audit_orphans.get(sid).expect("present");
        let mut sorted_ids: Vec<&String> = ids.iter().collect();
        sorted_ids.sort();
        for asn_id in sorted_ids {
            let body = format!("verified/sessions/{sid}/assignments/{asn_id}.json");
            let seen = format!("seen/assignments/{sid}--{asn_id}");
            actions.push(CleanupAction {
                kind: CleanupActionKind::QuarantineAndUnmarkOrphanAssignment,
                session_id: Some(sid.clone()),
                path: body,
                seen_marker_path: Some(seen),
                source_finding_kind: "orphan_replacement_assignments".to_string(),
                source_reason_tag: format!("orphan_assignment_id={asn_id}"),
            });
        }
    }

    actions.sort_by(|a, b| {
        (
            a.session_id.as_deref().unwrap_or(""),
            a.kind.as_str(),
            a.path.as_str(),
        )
            .cmp(&(
                b.session_id.as_deref().unwrap_or(""),
                b.kind.as_str(),
                b.path.as_str(),
            ))
    });

    let source_integrity_hash = source_integrity_hash_hex(report);
    let plan_id = compute_plan_id(&report.state_dir, &source_integrity_hash, opts.now_utc);
    let mut plan = StateCleanupPlan {
        schema_version: CLEANUP_PLAN_SCHEMA_VERSION,
        plan_id,
        created_at_utc: opts.now_utc.to_string(),
        state_dir: report.state_dir.clone(),
        source_integrity_hash,
        omni_contributor_version: env!("CARGO_PKG_VERSION").to_string(),
        actions,
        cleanup_plan_hash: String::new(),
    };
    plan.cleanup_plan_hash = cleanup_plan_hash_hex(&plan);
    Ok(plan)
}

/// Map one [`IntegrityFinding`] to the closed cleanup action it
/// should produce, or `None` for findings outside the v1 scope.
fn action_for_finding(f: &IntegrityFinding) -> Option<CleanupAction> {
    let path = f.path.as_deref()?;
    let session_id = f.session_id.clone();
    let source_finding_kind = f.kind.as_str().to_string();
    let source_reason_tag = f.reason_tag.clone();

    let (kind, seen_marker_path) = match f.kind {
        FindingKind::StaleSeenMarker => (CleanupActionKind::RemoveSeenMarker, None),
        FindingKind::MissingSeenMarker => (CleanupActionKind::WriteSeenMarker, None),
        FindingKind::StraySeenFile => (CleanupActionKind::RemoveSeenFile, None),
        FindingKind::StrayVerifiedFile => {
            (CleanupActionKind::QuarantineVerifiedFile, None)
        }
        FindingKind::InvalidJoin => {
            let seen = seen_for_verified(path)?;
            (CleanupActionKind::QuarantineAndUnmarkJoin, Some(seen))
        }
        FindingKind::InvalidAssignment => {
            let seen = seen_for_verified(path)?;
            (CleanupActionKind::QuarantineAndUnmarkAssignment, Some(seen))
        }
        FindingKind::InvalidPartial => {
            let seen = seen_for_verified(path)?;
            (CleanupActionKind::QuarantineAndUnmarkPartial, Some(seen))
        }
        FindingKind::InvalidSupersession => {
            let seen = seen_for_verified(path)?;
            (
                CleanupActionKind::QuarantineAndUnmarkSupersession,
                Some(seen),
            )
        }
        // Out of v1 scope.
        FindingKind::InvalidSession
        | FindingKind::InvalidAggregate
        | FindingKind::OrphanReplacementAssignments
        | FindingKind::PartialApplySupersession
        | FindingKind::ReassignTriagable
        | FindingKind::NotReassignTriagable
        | FindingKind::ArchiveManifestMalformed
        | FindingKind::ArchiveBlakeMismatch
        | FindingKind::ArchiveCoveredSession => return None,
    };
    Some(CleanupAction {
        kind,
        session_id,
        path: path.to_string(),
        seen_marker_path,
        source_finding_kind,
        source_reason_tag,
    })
}

/// Map a `verified/sessions/<sid>/...` body relative path to
/// the corresponding `seen/<ns>/<key>` marker path. Returns
/// `None` for paths that don't match the documented layout —
/// those findings get dropped by the planner.
fn seen_for_verified(verified_relative: &str) -> Option<String> {
    let rest = verified_relative.strip_prefix("verified/sessions/")?;
    let mut parts = rest.splitn(2, '/');
    let sid = parts.next()?;
    let tail = parts.next()?;
    if tail == "session.json" {
        return Some(format!("seen/sessions/{sid}"));
    }
    if tail == "aggregated.json" {
        return Some(format!("seen/aggregates/{sid}"));
    }
    // Subtree forms: `<leaf>/<key>.json`.
    let mut tparts = tail.splitn(2, '/');
    let leaf = tparts.next()?;
    let leaf_key = tparts.next()?;
    let key_stem = leaf_key.strip_suffix(".json")?;
    let seen_dir = match leaf {
        "joins" => "joins",
        "assignments" => "assignments",
        "partials" => "partials",
        "peer-adverts" => "peer-adverts",
        "supersessions" => "assignment-supersessions",
        _ => return None,
    };
    Some(format!("seen/{seen_dir}/{sid}--{key_stem}"))
}

/// Inverse of [`seen_for_verified`]: parse a well-formed
/// `seen/<ns>/<key>` relative path into the typed
/// [`StateNamespace`] + key. Returns `None` for paths whose
/// namespace dir is outside the closed set OR whose key shape
/// is invalid (callers fall back to direct `fs::remove_file`
/// for stray paths).
fn parse_seen_relative(rel: &str) -> Option<(StateNamespace, String)> {
    let rest = rel.strip_prefix("seen/")?;
    let mut parts = rest.splitn(2, '/');
    let ns_dir = parts.next()?;
    let key = parts.next()?;
    let ns = match ns_dir {
        "sessions" => StateNamespace::Sessions,
        "aggregates" => StateNamespace::Aggregates,
        "joins" => StateNamespace::Joins,
        "assignments" => StateNamespace::Assignments,
        "partials" => StateNamespace::Partials,
        "peer-adverts" => StateNamespace::PeerAdverts,
        "assignment-supersessions" => StateNamespace::AssignmentSupersessions,
        _ => return None,
    };
    Some((ns, key.to_string()))
}

// ── Public applier ────────────────────────────────────────────

/// Run the cleanup plan. The applier:
///
/// 1. Verifies `plan.schema_version == 1` and
///    `plan.cleanup_plan_hash` against a recompute.
/// 2. Re-runs [`scan_state_integrity_with_audit_orphans`]; the
///    resulting `source_integrity_hash` must match
///    `plan.source_integrity_hash`. Mismatch → typed drift error.
/// 3. For every action gated by a flag the operator didn't
///    pass, returns [`CleanupError::GateRequired`] BEFORE any
///    mutation. (Plan-time emits gated actions whenever the
///    finding warrants them.)
/// 4. For every `QuarantineAndUnmarkOrphanAssignment`, the
///    apply-time `compute_audit_health` projection's orphan id
///    set must equal the planner's. Mismatch → typed audit
///    drift error.
/// 5. Stage 12.17 review fix — validates every action's `path`
///    and `seen_marker_path` against the per-kind whitelist
///    (`verified/sessions/...` for tier B + `WriteSeenMarker`;
///    `seen/...` for tier A; no `..`, no absolute, no
///    backslash). Returns [`CleanupError::UnsafePlanPath`]
///    BEFORE any IO when a tampered/hand-edited plan with a
///    recomputed `cleanup_plan_hash` carries a malicious path.
/// 6. Refuses pre-existing `<quarantine-dir>/<plan_id>/` with
///    [`CleanupError::QuarantineCollision`].
/// 7. Stage 12.17 review fix — walks three explicit phases in
///    order; if **Phase A** or **Phase B** fails, **Phase C**
///    never runs, so the state-dir is byte-identical to
///    pre-apply:
///    - **Phase A — Quarantine.** For each tier-B action: read
///      source bytes → compute BLAKE3 → write to
///      `<quarantine-dir>/<plan_id>/<source_relative>` →
///      re-read + BLAKE3 verify the copy. Build
///      `QuarantineEntry` records in memory. **No source
///      removal here.**
///    - **Phase B — Manifest.** Write
///      `quarantine-manifest.json` atomically (tempfile +
///      rename) under `<quarantine-dir>/<plan_id>/`. By the
///      time Phase C runs, the manifest is durable on disk.
///    - **Phase C — State-dir mutation.** Walk actions in
///      plan order: `RemoveSeenMarker` → `unmark_seen`;
///      `RemoveSeenFile` → direct `fs::remove_file`;
///      `WriteSeenMarker` → `mark_seen`; tier-B →
///      `remove_verified_relative` then `unmark_seen` for the
///      matching `seen_marker_path`. Tier-B actions whose
///      Phase A saw the source missing record
///      `skipped_missing`. Tier-A skips Phase A entirely.
pub fn apply_state_cleanup(
    store: &ContributorStateStore,
    plan: &StateCleanupPlan,
    opts: &ApplyOptions<'_>,
) -> Result<CleanupReport, CleanupError> {
    // 1. Plan hash.
    if plan.schema_version != CLEANUP_PLAN_SCHEMA_VERSION {
        return Err(CleanupError::UnsupportedPlanVersion {
            got: plan.schema_version,
            expected: CLEANUP_PLAN_SCHEMA_VERSION,
        });
    }
    let recomputed_plan_hash = cleanup_plan_hash_hex(plan);
    if recomputed_plan_hash != plan.cleanup_plan_hash {
        return Err(CleanupError::PlanHashMismatch {
            stored: plan.cleanup_plan_hash.clone(),
            recomputed: recomputed_plan_hash,
        });
    }

    // 2. Integrity re-projection + drift check.
    let scan_opts = ScanOptions {
        session_id_filter: None,
        archive_dir: None,
        now_utc: opts.now_utc,
    };
    let (current_report, current_orphans) =
        scan_state_integrity_with_audit_orphans(store, &scan_opts)?;
    let current_hash = source_integrity_hash_hex(&current_report);
    if current_hash != plan.source_integrity_hash {
        return Err(CleanupError::SourceIntegrityDrift {
            expected: plan.source_integrity_hash.clone(),
            got: current_hash,
        });
    }

    // 3. Gate pre-checks (sweep the plan once before mutating).
    for action in &plan.actions {
        if let Some(flag) = action.kind.gate_flag() {
            let allowed = match action.kind {
                CleanupActionKind::QuarantineAndUnmarkPartial => {
                    opts.allow_invalid_partial_cleanup
                }
                CleanupActionKind::QuarantineAndUnmarkOrphanAssignment => {
                    opts.allow_orphan_assignments
                }
                _ => true,
            };
            if !allowed {
                return Err(CleanupError::GateRequired {
                    kind: action.kind.as_str().to_string(),
                    flag: flag.to_string(),
                });
            }
        }
    }

    // 4. Orphan audit re-check per session that has an orphan
    //    action in the plan. Apply-time builds the status
    //    report + audit projection and compares the orphan id
    //    set.
    let plan_orphans = collect_plan_orphans(plan);
    for (sid, plan_ids) in &plan_orphans {
        let status = build_session_status_report(
            store,
            sid,
            opts.now_utc,
            /* include_expired = */ false,
        )?;
        let audit = compute_audit_health(&status);
        let current_ids: HashSet<String> = match audit.coherence {
            AuditCoherence::OrphanReplacementAssignments { ref assignment_ids } => {
                assignment_ids.iter().cloned().collect()
            }
            _ => HashSet::new(),
        };
        let plan_set: HashSet<String> = plan_ids.iter().cloned().collect();
        if current_ids != plan_set {
            return Err(CleanupError::OrphanAuditDrift {
                session_id: sid.clone(),
                plan_count: plan_set.len() as u32,
                current_count: current_ids.len() as u32,
            });
        }
        let _ = current_orphans.get(sid); // current_orphans informational
    }

    // 5. Stage 12.17 review fix — pre-mutation path-safety
    //    sweep. A self-consistent `cleanup_plan_hash` does NOT
    //    vouch for paths: a hand-edited plan can rebuild its
    //    hash trivially. We validate every action's path +
    //    seen_marker_path against the per-kind whitelist
    //    BEFORE any FS interaction so no `..` / absolute /
    //    backslash payload ever reaches `std::fs::read` /
    //    `std::fs::write` / `remove_verified_relative` /
    //    `unmark_seen`.
    for action in &plan.actions {
        validate_cleanup_action_paths(action)?;
    }

    let plan_quarantine_root = opts.quarantine_dir.join(&plan.plan_id);
    let mut outcomes: Vec<CleanupActionOutcome> = Vec::with_capacity(plan.actions.len());
    let mut actions_applied: u32 = 0;
    let mut actions_dry_run: u32 = 0;
    let mut actions_skipped: u32 = 0;

    // 6. Refuse pre-existing quarantine plan-id dir.
    if !opts.dry_run && plan_quarantine_root.exists() {
        return Err(CleanupError::QuarantineCollision {
            path: plan_quarantine_root,
        });
    }

    // 7. Dry-run short-circuit: emit `would_apply` per action
    //    after every preflight has cleared. No FS touch.
    if opts.dry_run {
        for (idx, action) in plan.actions.iter().enumerate() {
            outcomes.push(CleanupActionOutcome {
                action_index: idx as u32,
                kind: action.kind.as_str().to_string(),
                session_id: action.session_id.clone(),
                path: action.path.clone(),
                status: "would_apply".to_string(),
            });
            actions_dry_run += 1;
        }
        return Ok(CleanupReport {
            plan_id: plan.plan_id.clone(),
            mode: "dry_run".to_string(),
            actions_applied,
            actions_dry_run,
            actions_skipped,
            quarantine_dir: plan_quarantine_root.to_string_lossy().replace('\\', "/"),
            quarantine_manifest_relative: None,
            outcomes,
        });
    }

    // 8. Stage 12.17 review fix — quarantine-first ordering.
    //    Phase A: walk tier-B actions, copy bytes to
    //              `<quarantine-dir>/<plan_id>/<source_relative>`,
    //              BLAKE3-verify the copy. NO source removal.
    //    Phase B: write the quarantine manifest LAST atomically
    //              (tempfile + rename).
    //    Phase C: walk every action and remove sources +
    //              unmark seen.
    //
    //    The Phase A→B→C order is the safety invariant: if
    //    Phase A or B fails (IO error, BLAKE3 mismatch, rename
    //    failure), Phase C never runs, so the state-dir is
    //    byte-identical to pre-apply. The orphan quarantine
    //    subtree under `<plan_id>/` is operator-disposable —
    //    no destructive state-dir mutation has happened.
    //
    //    Action-index → manifest-entry-index lookup so Phase C
    //    can emit the correct outcome string per action.
    let mut quarantine_entries: Vec<QuarantineEntry> = Vec::new();
    let mut action_quarantine_state: Vec<TierBPlan> =
        Vec::with_capacity(plan.actions.len());

    // ── Phase A ─────────────────────────────────────────────
    for action in &plan.actions {
        if !action.kind.is_tier_b() {
            action_quarantine_state.push(TierBPlan::NotTierB);
            continue;
        }
        let skip_quarantine = matches!(
            action.kind,
            CleanupActionKind::QuarantineVerifiedFile
        ) && opts.purge_stray;
        let source = store.root().join(&action.path);
        if !source.is_file() {
            // Idempotent miss; nothing to quarantine.
            action_quarantine_state.push(TierBPlan::SourceMissing);
            continue;
        }
        if skip_quarantine {
            action_quarantine_state.push(TierBPlan::SkipQuarantine);
            continue;
        }
        let bytes = std::fs::read(&source).map_err(|e| CleanupError::Io {
            path: source.clone(),
            source: e,
        })?;
        let blake3_hex = blake3_hex_of(&bytes);
        let bytes_len = bytes.len() as u64;

        let dest = plan_quarantine_root.join(&action.path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CleanupError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        if dest.exists() {
            return Err(CleanupError::QuarantineCollision { path: dest });
        }
        std::fs::write(&dest, &bytes).map_err(|e| CleanupError::Io {
            path: dest.clone(),
            source: e,
        })?;
        let verified = std::fs::read(&dest).map_err(|e| CleanupError::Io {
            path: dest.clone(),
            source: e,
        })?;
        if blake3_hex_of(&verified) != blake3_hex {
            return Err(CleanupError::Io {
                path: dest,
                source: std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "quarantine BLAKE3 mismatch after copy",
                ),
            });
        }
        quarantine_entries.push(QuarantineEntry {
            session_id: action.session_id.clone(),
            source_relative: action.path.clone(),
            quarantine_relative: action.path.clone(),
            blake3_hex,
            bytes: bytes_len,
            source_finding_kind: action.source_finding_kind.clone(),
            source_reason_tag: action.source_reason_tag.clone(),
        });
        action_quarantine_state.push(TierBPlan::Quarantined);
    }

    // ── Phase B ─────────────────────────────────────────────
    let mut manifest_relative: Option<String> = None;
    if !quarantine_entries.is_empty() {
        let manifest = QuarantineManifest {
            schema_version: QUARANTINE_MANIFEST_SCHEMA_VERSION,
            plan_id: plan.plan_id.clone(),
            created_at_utc: opts.now_utc.to_string(),
            state_dir: plan.state_dir.clone(),
            omni_contributor_version: env!("CARGO_PKG_VERSION").to_string(),
            source_state_version: STATE_VERSION,
            files: quarantine_entries,
        };
        let manifest_path = plan_quarantine_root.join("quarantine-manifest.json");
        let tmp_path =
            plan_quarantine_root.join("quarantine-manifest.json.tmp");
        let bytes = serde_json::to_vec_pretty(&manifest)
            .expect("serialize quarantine manifest");
        std::fs::write(&tmp_path, &bytes).map_err(|e| CleanupError::Io {
            path: tmp_path.clone(),
            source: e,
        })?;
        std::fs::rename(&tmp_path, &manifest_path).map_err(|e| {
            CleanupError::Io {
                path: manifest_path.clone(),
                source: e,
            }
        })?;
        manifest_relative = Some("quarantine-manifest.json".to_string());
    }

    // ── Phase C ─────────────────────────────────────────────
    for (idx, action) in plan.actions.iter().enumerate() {
        let outcome_kind = action.kind.as_str().to_string();
        let outcome_sid = action.session_id.clone();
        let outcome_path = action.path.clone();
        let applied = match action.kind {
            CleanupActionKind::RemoveSeenMarker => {
                apply_remove_seen_marker(store, &action.path)?
            }
            CleanupActionKind::RemoveSeenFile => {
                apply_remove_seen_file(store, &action.path)?
            }
            CleanupActionKind::WriteSeenMarker => {
                apply_write_seen_marker(store, &action.path)?
            }
            CleanupActionKind::QuarantineVerifiedFile
            | CleanupActionKind::QuarantineAndUnmarkJoin
            | CleanupActionKind::QuarantineAndUnmarkAssignment
            | CleanupActionKind::QuarantineAndUnmarkPartial
            | CleanupActionKind::QuarantineAndUnmarkSupersession
            | CleanupActionKind::QuarantineAndUnmarkOrphanAssignment => {
                let state = action_quarantine_state[idx];
                apply_tier_b_finalize(store, action, state)?
            }
        };
        if applied {
            actions_applied += 1;
            outcomes.push(CleanupActionOutcome {
                action_index: idx as u32,
                kind: outcome_kind,
                session_id: outcome_sid,
                path: outcome_path,
                status: "applied".to_string(),
            });
        } else {
            actions_skipped += 1;
            outcomes.push(CleanupActionOutcome {
                action_index: idx as u32,
                kind: outcome_kind,
                session_id: outcome_sid,
                path: outcome_path,
                status: "skipped_missing".to_string(),
            });
        }
    }

    Ok(CleanupReport {
        plan_id: plan.plan_id.clone(),
        mode: if opts.dry_run { "dry_run" } else { "apply" }.to_string(),
        actions_applied,
        actions_dry_run,
        actions_skipped,
        quarantine_dir: plan_quarantine_root.to_string_lossy().replace('\\', "/"),
        quarantine_manifest_relative: manifest_relative,
        outcomes,
    })
}

fn collect_plan_orphans(plan: &StateCleanupPlan) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for action in &plan.actions {
        if action.kind != CleanupActionKind::QuarantineAndUnmarkOrphanAssignment {
            continue;
        }
        let sid = match &action.session_id {
            Some(s) => s.clone(),
            None => continue,
        };
        let asn_id = action
            .source_reason_tag
            .strip_prefix("orphan_assignment_id=")
            .unwrap_or("")
            .to_string();
        if !asn_id.is_empty() {
            out.entry(sid).or_default().push(asn_id);
        }
    }
    out
}

fn apply_remove_seen_marker(
    store: &ContributorStateStore,
    rel: &str,
) -> Result<bool, CleanupError> {
    if let Some((ns, key)) = parse_seen_relative(rel) {
        Ok(store.unmark_seen(ns, &key)?)
    } else {
        apply_remove_seen_file(store, rel)
    }
}

fn apply_remove_seen_file(
    store: &ContributorStateStore,
    rel: &str,
) -> Result<bool, CleanupError> {
    let dest = store.root().join(rel);
    match std::fs::remove_file(&dest) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(CleanupError::Io { path: dest, source: e }),
    }
}

fn apply_write_seen_marker(
    store: &ContributorStateStore,
    verified_relative: &str,
) -> Result<bool, CleanupError> {
    let seen_rel = seen_for_verified(verified_relative).ok_or_else(|| {
        CleanupError::Io {
            path: PathBuf::from(verified_relative),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "verified path does not map to a seen marker",
            ),
        }
    })?;
    let (ns, key) = parse_seen_relative(&seen_rel).ok_or_else(|| CleanupError::Io {
        path: PathBuf::from(seen_rel.clone()),
        source: std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "derived seen path is malformed",
        ),
    })?;
    store.mark_seen(ns, &key)?;
    Ok(true)
}

/// Per-action quarantine-phase outcome the Phase C finalizer
/// reads. Captured at Phase A time so Phase C can emit the
/// correct outcome string per action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TierBPlan {
    /// Phase A successfully quarantined the body bytes.
    Quarantined,
    /// `purge_stray` was set and the action is
    /// `QuarantineVerifiedFile`; no quarantine copy was
    /// written. Phase C still removes the source.
    SkipQuarantine,
    /// Phase A saw the source missing on disk; Phase C records
    /// this as `skipped_missing`.
    SourceMissing,
    /// Non-tier-B variants get this value; never read.
    NotTierB,
}

/// Phase C finalizer. Removes the source (idempotent NotFound)
/// and unmarks any matching seen marker. Quarantine writes are
/// already durable by the time Phase C runs — every
/// destructive operation here is recoverable from the
/// quarantine manifest if needed.
fn apply_tier_b_finalize(
    store: &ContributorStateStore,
    action: &CleanupAction,
    state: TierBPlan,
) -> Result<bool, CleanupError> {
    let applied = match state {
        TierBPlan::SourceMissing => false,
        TierBPlan::Quarantined | TierBPlan::SkipQuarantine => {
            store.remove_verified_relative(&action.path)?
        }
        TierBPlan::NotTierB => unreachable!("non-tier-B in tier-B finalizer"),
    };
    if let Some(seen_rel) = &action.seen_marker_path {
        if let Some((ns, key)) = parse_seen_relative(seen_rel) {
            store.unmark_seen(ns, &key)?;
        }
    }
    Ok(applied)
}

// ── Path-safety validators ────────────────────────────────────
//
// Self-consistent `cleanup_plan_hash` is not a security gate
// (anyone who hand-edits a plan can rebuild the hash). Every
// path the applier consumes is validated against the per-kind
// whitelist BEFORE any FS interaction.

/// Per-action whitelist: `path` must match the action's
/// expected prefix shape; `seen_marker_path` (when present)
/// must point under `seen/...`. No `..`, no absolute prefix,
/// no backslash, no empty segments — for either field.
fn validate_cleanup_action_paths(action: &CleanupAction) -> Result<(), CleanupError> {
    match action.kind {
        CleanupActionKind::RemoveSeenMarker
        | CleanupActionKind::RemoveSeenFile => check_seen_relative(&action.path)?,
        CleanupActionKind::WriteSeenMarker => {
            check_verified_relative(&action.path)?
        }
        CleanupActionKind::QuarantineVerifiedFile
        | CleanupActionKind::QuarantineAndUnmarkJoin
        | CleanupActionKind::QuarantineAndUnmarkAssignment
        | CleanupActionKind::QuarantineAndUnmarkPartial
        | CleanupActionKind::QuarantineAndUnmarkSupersession
        | CleanupActionKind::QuarantineAndUnmarkOrphanAssignment => {
            check_verified_relative(&action.path)?
        }
    }
    if let Some(seen) = &action.seen_marker_path {
        check_seen_relative(seen)?;
    }
    Ok(())
}

fn check_relative_safety(rel: &str) -> Result<(), CleanupError> {
    if rel.is_empty() {
        return Err(CleanupError::UnsafePlanPath {
            path: rel.to_string(),
            reason: "empty path",
        });
    }
    if rel.starts_with('/') {
        return Err(CleanupError::UnsafePlanPath {
            path: rel.to_string(),
            reason: "absolute path",
        });
    }
    if rel.contains('\\') {
        return Err(CleanupError::UnsafePlanPath {
            path: rel.to_string(),
            reason: "backslash separator",
        });
    }
    if rel.starts_with("./") || rel.contains("/./") || rel.ends_with("/.") {
        return Err(CleanupError::UnsafePlanPath {
            path: rel.to_string(),
            reason: "dot path segment",
        });
    }
    for segment in rel.split('/') {
        if segment.is_empty() {
            return Err(CleanupError::UnsafePlanPath {
                path: rel.to_string(),
                reason: "empty path segment",
            });
        }
        if segment == ".." {
            return Err(CleanupError::UnsafePlanPath {
                path: rel.to_string(),
                reason: "parent-directory traversal",
            });
        }
    }
    Ok(())
}

fn check_seen_relative(rel: &str) -> Result<(), CleanupError> {
    check_relative_safety(rel)?;
    if !rel.starts_with("seen/") {
        return Err(CleanupError::UnsafePlanPath {
            path: rel.to_string(),
            reason: "must start with seen/",
        });
    }
    Ok(())
}

fn check_verified_relative(rel: &str) -> Result<(), CleanupError> {
    check_relative_safety(rel)?;
    if !rel.starts_with("verified/sessions/") {
        return Err(CleanupError::UnsafePlanPath {
            path: rel.to_string(),
            reason: "must start with verified/sessions/",
        });
    }
    Ok(())
}

fn blake3_hex_of(bytes: &[u8]) -> String {
    let h = blake3::hash(bytes);
    let mut s = String::with_capacity(64);
    for b in h.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
