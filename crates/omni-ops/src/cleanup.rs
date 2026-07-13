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
use omni_contributor::state::{
    ContributorStateStore, StateNamespace, STATE_VERSION,
};
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

// ── Stage 12.18 — cleanup-quarantine restore ──────────────────
//
// Consumes the v1 `QuarantineManifest` Stage 12.17 wrote in
// Phase B and undoes the cleanup that produced it. Tier-A
// actions had no payload to quarantine and are deliberately
// not recoverable (documented gap). Tier-B `--purge-stray`
// entries are also unrecoverable for the same reason.

use crate::error::QuarantineRestoreError;

/// Closed source for [`restore_state_cleanup_quarantine`]. The
/// CLI accepts `--quarantine-plan-dir` xor
/// (`--quarantine-dir + --plan-id`); both shapes resolve to a
/// concrete `<plan_id>` directory on disk + an authoritative
/// expected `plan_id` for manifest pinning.
#[derive(Debug, Clone, Copy)]
pub enum QuarantineRestoreSource<'a> {
    /// Direct path to `<quarantine-dir>/<plan_id>/`. The
    /// directory's basename is the authoritative `plan_id`
    /// expectation.
    PlanDir(&'a Path),
    /// Paired form: `<quarantine-dir>` + `<plan_id>`.
    QuarantineRoot {
        quarantine_dir: &'a Path,
        plan_id: &'a str,
    },
}

impl<'a> QuarantineRestoreSource<'a> {
    pub fn plan_dir(&self) -> PathBuf {
        match self {
            Self::PlanDir(p) => p.to_path_buf(),
            Self::QuarantineRoot {
                quarantine_dir,
                plan_id,
            } => quarantine_dir.join(plan_id),
        }
    }

    pub fn expected_plan_id(&self) -> Option<String> {
        match self {
            Self::PlanDir(p) => p
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string()),
            Self::QuarantineRoot { plan_id, .. } => Some((*plan_id).to_string()),
        }
    }
}

/// Stage 12.18 restore options. Mirrors Stage 12.15
/// `RestoreOptions` shape (closed mode matrix; `verify_only`
/// wins when both `dry_run` and `verify_only` are set).
#[derive(Debug, Clone)]
pub struct QuarantineRestoreOptions<'a> {
    pub source: QuarantineRestoreSource<'a>,
    pub dry_run: bool,
    pub verify_only: bool,
    pub overwrite_existing: bool,
    /// Default `true` — restore seen markers for entries
    /// whose `source_finding_kind` proves a marker was
    /// unmarked.
    pub restore_seen_markers: bool,
    /// Gate for `orphan_replacement_assignments` entries.
    pub allow_restore_orphan_assignments: bool,
    pub now_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuarantineRestoreOutcome {
    pub entry_index: u32,
    pub source_relative: String,
    pub source_finding_kind: String,
    /// `"applied"` / `"would_apply"` / `"verify_only"`. Closed.
    pub status: String,
    pub seen_marker_written: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuarantineRestoreReport {
    pub plan_id: String,
    pub mode: String,
    pub manifest_schema_version: u32,
    pub source_state_version: u32,
    pub files_restored: u32,
    pub seen_markers_restored: u32,
    pub bytes_restored: u64,
    pub quarantine_dir: String,
    pub outcomes: Vec<QuarantineRestoreOutcome>,
}

#[derive(Debug, Clone, Copy)]
struct RestoreClassification {
    restores_seen_marker: bool,
    gate: Option<&'static str>,
}

/// Closed set of `CleanupAction::source_finding_kind` tags the
/// Stage 12.17 planner emits.
fn classify_finding_kind(tag: &str) -> Option<RestoreClassification> {
    match tag {
        "stray_verified_file" => Some(RestoreClassification {
            restores_seen_marker: false,
            gate: None,
        }),
        "invalid_join" | "invalid_assignment" | "invalid_partial"
        | "invalid_supersession" => Some(RestoreClassification {
            restores_seen_marker: true,
            gate: None,
        }),
        "orphan_replacement_assignments" => Some(RestoreClassification {
            restores_seen_marker: true,
            gate: Some("--allow-restore-orphan-assignments"),
        }),
        _ => None,
    }
}

/// Parse + validate `quarantine-manifest.json` only. Used as a
/// stand-alone helper AND as the first step of
/// [`restore_state_cleanup_quarantine`].
pub fn verify_quarantine_manifest(
    source: &QuarantineRestoreSource<'_>,
) -> Result<QuarantineManifest, QuarantineRestoreError> {
    let plan_dir = source.plan_dir();
    if !plan_dir.exists() {
        return Err(QuarantineRestoreError::QuarantineDirNotFound { path: plan_dir });
    }
    let manifest_path = plan_dir.join("quarantine-manifest.json");
    if !manifest_path.is_file() {
        return Err(QuarantineRestoreError::ManifestMissing { path: manifest_path });
    }
    let bytes = std::fs::read(&manifest_path).map_err(|e| {
        QuarantineRestoreError::Io {
            path: manifest_path.clone(),
            source: e,
        }
    })?;
    let manifest: QuarantineManifest = serde_json::from_slice(&bytes).map_err(|e| {
        QuarantineRestoreError::MalformedManifest {
            path: manifest_path.clone(),
            source: e,
        }
    })?;
    if manifest.schema_version != QUARANTINE_MANIFEST_SCHEMA_VERSION {
        return Err(QuarantineRestoreError::UnsupportedManifestVersion {
            got: manifest.schema_version,
            expected: QUARANTINE_MANIFEST_SCHEMA_VERSION,
        });
    }
    if manifest.source_state_version != STATE_VERSION {
        return Err(QuarantineRestoreError::IncompatibleSourceStateVersion {
            manifest: manifest.source_state_version,
            current: STATE_VERSION,
        });
    }
    if let Some(expected) = source.expected_plan_id() {
        if manifest.plan_id != expected {
            return Err(QuarantineRestoreError::PlanIdMismatch {
                manifest_plan_id: manifest.plan_id.clone(),
                supplied_plan_id: expected,
            });
        }
    }
    Ok(manifest)
}

/// Run the quarantine restore. Five explicit phases — A, B,
/// then C split into C1/C2/C3:
///
/// - **Phase A** — parse + validate the manifest, path-check
///   every entry, classify each `source_finding_kind` against
///   the closed set, refuse any unknown tag with
///   [`QuarantineRestoreError::UnknownFindingKind`], refuse any
///   gated entry whose flag wasn't passed with
///   [`QuarantineRestoreError::GatedRestoreRequired`]. NO file
///   reads, NO writes.
/// - **Phase B** — when `verify_only || !dry_run`, read each
///   quarantine file and BLAKE3-verify it against the manifest.
/// - **Phase C1** — body destination preflight. All-or-nothing:
///   any pre-existing body destination refuses the whole
///   restore with [`QuarantineRestoreError::DestinationExists`]
///   unless `overwrite_existing == true`.
/// - **Phase C2** — Stage 12.18 review fix — seen-marker
///   destination preflight. For every entry whose
///   `source_finding_kind` would restore a marker, the
///   `seen/<ns>/<key>` path is checked: any non-file occupant
///   (typically a directory) refuses the whole restore with
///   [`QuarantineRestoreError::SeenMarkerPathBlocked`] BEFORE
///   any body is written. `--overwrite-existing` does NOT cover
///   marker hazards — the preflight is unconditional whenever
///   `restore_seen_markers == true`. This preserves the
///   all-or-nothing invariant even when the operator-visible
///   marker side of the restore is at risk.
/// - **Phase C3** — writes. Each body lands via
///   `store.write_archived_bytes` (which independently
///   re-validates the path against the Stage 12.14 archive
///   whitelist) and (when `restore_seen_markers`) the matching
///   `seen/<ns>/<key>` marker lands via `store.mark_seen`.
///
/// Per-action gating: an entry tagged
/// `orphan_replacement_assignments` requires
/// `allow_restore_orphan_assignments = true` at Phase A.
/// Partial entries (`invalid_partial`) have no extra gate in
/// v1.
pub fn restore_state_cleanup_quarantine(
    store: &ContributorStateStore,
    opts: &QuarantineRestoreOptions<'_>,
) -> Result<QuarantineRestoreReport, QuarantineRestoreError> {
    let manifest = verify_quarantine_manifest(&opts.source)?;
    let plan_dir = opts.source.plan_dir();

    let mode = if opts.verify_only {
        "verify_only"
    } else if opts.dry_run {
        "dry_run"
    } else {
        "restore"
    };

    // ── Phase A — per-entry path + finding-kind validation ──
    for entry in &manifest.files {
        check_quarantine_relative_path(&entry.source_relative)?;
        check_quarantine_relative_path(&entry.quarantine_relative)?;
        let classification = classify_finding_kind(&entry.source_finding_kind)
            .ok_or_else(|| QuarantineRestoreError::UnknownFindingKind {
                kind: entry.source_finding_kind.clone(),
            })?;
        if let Some(flag) = classification.gate {
            let allowed = match entry.source_finding_kind.as_str() {
                "orphan_replacement_assignments" => {
                    opts.allow_restore_orphan_assignments
                }
                _ => false,
            };
            if !allowed {
                return Err(QuarantineRestoreError::GatedRestoreRequired {
                    kind: "orphan_replacement_assignments",
                    flag,
                });
            }
        }
    }

    // ── Phase B — BLAKE3 verify (verify-only or real restore) ──
    let mut prepared: Vec<(usize, Vec<u8>)> = Vec::new();
    if opts.verify_only || !opts.dry_run {
        for (idx, entry) in manifest.files.iter().enumerate() {
            let quarantine_path = plan_dir.join(&entry.quarantine_relative);
            if !quarantine_path.is_file() {
                return Err(QuarantineRestoreError::ManifestFileMissing {
                    path: quarantine_path,
                });
            }
            let bytes = std::fs::read(&quarantine_path).map_err(|e| {
                QuarantineRestoreError::Io {
                    path: quarantine_path.clone(),
                    source: e,
                }
            })?;
            let got = blake3_hex_of(&bytes);
            if got != entry.blake3_hex {
                return Err(QuarantineRestoreError::BlakeMismatch {
                    path: quarantine_path,
                    expected: entry.blake3_hex.clone(),
                    got,
                });
            }
            if !opts.verify_only {
                prepared.push((idx, bytes));
            }
        }
    }

    // ── Phase C — destination preflight + write ─────────────
    let mut files_restored = 0u32;
    let mut bytes_restored = 0u64;
    let mut seen_markers_restored = 0u32;
    let mut outcomes: Vec<QuarantineRestoreOutcome> =
        Vec::with_capacity(manifest.files.len());

    if matches!(mode, "dry_run" | "verify_only") {
        for (idx, entry) in manifest.files.iter().enumerate() {
            let status = if opts.verify_only {
                "verify_only"
            } else {
                "would_apply"
            };
            outcomes.push(QuarantineRestoreOutcome {
                entry_index: idx as u32,
                source_relative: entry.source_relative.clone(),
                source_finding_kind: entry.source_finding_kind.clone(),
                status: status.to_string(),
                seen_marker_written: false,
            });
        }
        return Ok(QuarantineRestoreReport {
            plan_id: manifest.plan_id,
            mode: mode.to_string(),
            manifest_schema_version: manifest.schema_version,
            source_state_version: manifest.source_state_version,
            files_restored,
            seen_markers_restored,
            bytes_restored,
            quarantine_dir: plan_dir.to_string_lossy().replace('\\', "/"),
            outcomes,
        });
    }

    // ── Phase C1 — body destination preflight ─────────────
    // All-or-nothing: any pre-existing destination refuses
    // the whole restore (unless `overwrite_existing`).
    if !opts.overwrite_existing {
        for entry in &manifest.files {
            let dest = store.root().join(&entry.source_relative);
            if dest.exists() {
                return Err(QuarantineRestoreError::DestinationExists { path: dest });
            }
        }
    }

    // ── Phase C2 — seen-marker destination preflight ──────
    //
    // Stage 12.18 review fix: marker writes used to happen
    // after each body write, so a marker hazard (typically a
    // directory at the seen path) could fail AFTER one or
    // more bodies had already landed in the state-dir,
    // breaking the all-or-nothing rollback story. The
    // preflight here runs BEFORE any body write and refuses
    // with `SeenMarkerPathBlocked` if any marker destination
    // is occupied by a non-file. An existing regular file at
    // the marker path is FINE — `mark_seen` is idempotent
    // (treats `AlreadyExists` as success).
    if opts.restore_seen_markers {
        for entry in &manifest.files {
            let classification = classify_finding_kind(&entry.source_finding_kind)
                .expect("validated in Phase A");
            if !classification.restores_seen_marker {
                continue;
            }
            let seen_rel = match seen_for_verified(&entry.source_relative) {
                Some(s) => s,
                None => continue,
            };
            let marker_path = store.root().join(&seen_rel);
            if marker_path.exists() && !marker_path.is_file() {
                return Err(QuarantineRestoreError::SeenMarkerPathBlocked {
                    path: marker_path,
                    reason: "destination is not a regular file",
                });
            }
        }
    }

    // ── Phase C3 — writes (bodies, then matching markers) ──
    for (idx, bytes) in prepared {
        let entry = &manifest.files[idx];
        store.write_archived_bytes(
            &entry.source_relative,
            &bytes,
            opts.overwrite_existing,
        )?;
        files_restored += 1;
        bytes_restored += entry.bytes;

        let classification = classify_finding_kind(&entry.source_finding_kind)
            .expect("validated in Phase A");
        let mut seen_marker_written = false;
        if opts.restore_seen_markers && classification.restores_seen_marker {
            if let Some(seen_rel) = seen_for_verified(&entry.source_relative) {
                if let Some((ns, key)) = parse_seen_relative(&seen_rel) {
                    store.mark_seen(ns, &key)?;
                    seen_marker_written = true;
                    seen_markers_restored += 1;
                }
            }
        }

        outcomes.push(QuarantineRestoreOutcome {
            entry_index: idx as u32,
            source_relative: entry.source_relative.clone(),
            source_finding_kind: entry.source_finding_kind.clone(),
            status: "applied".to_string(),
            seen_marker_written,
        });
    }

    Ok(QuarantineRestoreReport {
        plan_id: manifest.plan_id,
        mode: mode.to_string(),
        manifest_schema_version: manifest.schema_version,
        source_state_version: manifest.source_state_version,
        files_restored,
        seen_markers_restored,
        bytes_restored,
        quarantine_dir: plan_dir.to_string_lossy().replace('\\', "/"),
        outcomes,
    })
}

/// Per-path whitelist: every `source_relative` and
/// `quarantine_relative` must shape-match
/// `verified/sessions/<64hex>/...` — the only target subtree
/// Stage 12.17 cleanup ever quarantines.
fn check_quarantine_relative_path(rel: &str) -> Result<(), QuarantineRestoreError> {
    if rel.is_empty() {
        return Err(QuarantineRestoreError::UnsafeRelativePath {
            path: rel.to_string(),
            reason: "empty path",
        });
    }
    if rel.starts_with('/') {
        return Err(QuarantineRestoreError::UnsafeRelativePath {
            path: rel.to_string(),
            reason: "absolute path",
        });
    }
    if rel.contains('\\') {
        return Err(QuarantineRestoreError::UnsafeRelativePath {
            path: rel.to_string(),
            reason: "backslash separator",
        });
    }
    if rel.starts_with("./") || rel.contains("/./") || rel.ends_with("/.") {
        return Err(QuarantineRestoreError::UnsafeRelativePath {
            path: rel.to_string(),
            reason: "dot path segment",
        });
    }
    for segment in rel.split('/') {
        if segment.is_empty() {
            return Err(QuarantineRestoreError::UnsafeRelativePath {
                path: rel.to_string(),
                reason: "empty path segment",
            });
        }
        if segment == ".." {
            return Err(QuarantineRestoreError::UnsafeRelativePath {
                path: rel.to_string(),
                reason: "parent-directory traversal",
            });
        }
    }
    let parts: Vec<&str> = rel.split('/').collect();
    let ok = matches!(
        parts.as_slice(),
        ["verified", "sessions", sid, rest @ ..]
            if is_64_hex_lower(sid)
                && !rest.is_empty()
                && rest.iter().all(|s| !s.is_empty())
    );
    if !ok {
        return Err(QuarantineRestoreError::UnsafeRelativePath {
            path: rel.to_string(),
            reason: "path outside verified/sessions/<64hex>/... whitelist",
        });
    }
    Ok(())
}

fn is_64_hex_lower(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}
