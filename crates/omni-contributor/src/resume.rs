//! Stage 12.13 — restart / resume helpers + audit ergonomics.
//!
//! Two concerns live in this module:
//!
//! 1. **Restart snapshot loader** — `load_verified_restart_snapshot`
//!    walks the [`ContributorStateStore`] in dependency order
//!    (sessions → joins → assignments → supersessions), re-runs
//!    each Stage 12.3 / 12.11 verifier on the way, and returns a
//!    typed in-memory snapshot the watcher can warm its caches
//!    from. The Stage 12.7 trust boundary stays in place:
//!    `list_verified_*` are parse-only and this is the **caller**
//!    re-running the verifier. Corrupted local entries surface as
//!    structured rejection notes rather than panicking or silently
//!    polluting the cache.
//!
//! 2. **Audit ergonomics** — `compute_audit_health` derives a
//!    coordinator-readable health summary from an existing
//!    [`SessionStatusReport`]. The audit fields are local-only and
//!    are NOT added to the report struct (the v3 JSON schema is
//!    frozen at Stage 12.12); they live in renderers + this helper.
//!    Reuses Stage 12.12's
//!    [`check_invalid_partial_plan_eligible`] to decide whether the
//!    session is reassign-triagable.
//!
//! No new envelope. No new canonical bytes. No `schema_version`
//! bump. No `omni-net` topic change. The renderer changes in
//! `omni-node` consume these helpers; the helpers themselves
//! produce a closed `AuditCoherence` enum + a static-string
//! recommended action so scripts can pattern-match without
//! parsing free-form text.

use std::collections::HashMap;

use crate::error::StatusError;
use crate::repair::check_invalid_partial_plan_eligible;
use crate::session::{ContributorJoin, ExecutionSession, WorkAssignment};
use crate::session_verify::{
    verify_contributor_join, verify_execution_session, verify_work_assignment,
};
use crate::supersession_verify::verify_assignment_supersession;
use crate::state::ContributorStateStore;
use crate::status::{
    AssignmentStatus, InvalidArtifactStatus, SessionOverallStatus, SessionStatusReport,
};
use crate::supersession::WorkAssignmentSupersession;

// ── RestartSnapshot / RestartReport ─────────────────────────────

/// Typed in-memory snapshot the `watch-sessions` restart preload
/// walks the state-dir to populate. Caller (the watcher) uses
/// the maps directly to seed its caches.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RestartSnapshot {
    pub sessions: HashMap<String, ExecutionSession>,
    /// `session_id → verified joins` (in load order, which is
    /// directory-order on disk; the watcher does not depend on
    /// any particular ordering).
    pub joins_by_session: HashMap<String, Vec<ContributorJoin>>,
    /// `session_id → verified assignments`.
    pub assignments_by_session: HashMap<String, Vec<WorkAssignment>>,
    /// `session_id → verified supersessions`. Cross-supersession
    /// double-supersedes is checked at status-build time (the
    /// Stage 12.11 status reporter) — this preload does NOT
    /// dedupe; it just filters out individually-malformed bodies.
    pub supersessions_by_session: HashMap<String, Vec<WorkAssignmentSupersession>>,
}

/// Per-namespace accept/reject tally + structured rejection notes
/// so the watcher can emit one `event=state_store_restart_loaded`
/// line with the counts and warn separately on each rejected
/// entry. Rejection notes are stable-shape strings (`"kind=...
/// id=... reason_tag=..."`) so log scrapers can pattern-match
/// without picking apart human prose.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RestartReport {
    pub sessions_accepted: u32,
    pub sessions_rejected: u32,
    pub joins_accepted: u32,
    pub joins_rejected: u32,
    pub assignments_accepted: u32,
    pub assignments_rejected: u32,
    pub supersessions_accepted: u32,
    pub supersessions_rejected: u32,
    /// One line per rejected entry. Format:
    ///   `kind=<session|join|assignment|supersession>
    ///    session_id=<hex> id=<hex|->
    ///    reason_tag=<SessionVerifyOutcome::reason_tag>`.
    pub rejection_notes: Vec<String>,
}

/// Walk the state-dir in dependency order, re-running each Stage
/// 12.3 / 12.11 verifier, and return a typed snapshot the
/// watcher can warm its caches from.
///
/// Per-namespace policy:
/// - **session**: drops the entire `<session_id>` subtree on
///   verifier failure — without a verified session there is no
///   chain root for the joins / assignments / supersessions
///   below it.
/// - **join**: drops the individual join; keeps the session.
/// - **assignment**: drops the individual assignment (the
///   contributor must be in the verified-join set; otherwise the
///   assignment is dropped); keeps the session.
/// - **supersession**: drops the individual supersession on any
///   `verify_assignment_supersession` failure (incl. reference
///   resolution against the verified-assignment slice computed
///   above). Keeps the session.
///
/// Aggregate verification is **not** part of the preload — Stage
/// 12.13 decision 3. The first `session-status` build re-runs
/// `verify_aggregated_result_with_supersessions` on its own.
///
/// Partials are not loaded at all by this preload — the watcher
/// handles them lazily per-announcement (Stage 12.13 decision
/// 5).
pub fn load_verified_restart_snapshot(
    store: &ContributorStateStore,
) -> Result<(RestartSnapshot, RestartReport), StatusError> {
    let mut snapshot = RestartSnapshot::default();
    let mut report = RestartReport::default();

    let raw_sessions = store.list_verified_sessions()?;
    for (session_id_from_dir, session) in raw_sessions {
        let outcome = verify_execution_session(&session);
        if !outcome.is_ok() {
            report.sessions_rejected += 1;
            report.rejection_notes.push(format!(
                "kind=session session_id={session_id_from_dir} id={} reason_tag={}",
                session.session_id,
                outcome.reason_tag(),
            ));
            continue;
        }
        // The state-dir keys sessions by the directory name, but
        // the in-memory cache keys by `session.session_id` (the
        // verifier already pinned the two are equal via
        // `SessionIdMismatch`).
        let session_id = session.session_id.clone();

        // ── joins ──────────────────────────────────────────
        let raw_joins = store.list_verified_joins_for(&session_id)?;
        let mut joins: Vec<ContributorJoin> = Vec::new();
        for j in raw_joins {
            let outcome = verify_contributor_join(&session, &j);
            if outcome.is_ok() {
                joins.push(j);
            } else {
                report.joins_rejected += 1;
                report.rejection_notes.push(format!(
                    "kind=join session_id={session_id} id={} reason_tag={}",
                    j.contributor_pubkey_hex,
                    outcome.reason_tag(),
                ));
            }
        }
        report.joins_accepted += joins.len() as u32;
        let joined_pubkeys: std::collections::HashSet<String> =
            joins.iter().map(|j| j.contributor_pubkey_hex.clone()).collect();

        // ── assignments ────────────────────────────────────
        let raw_assignments = store.list_verified_assignments_for(&session_id)?;
        let mut assignments: Vec<WorkAssignment> = Vec::new();
        for a in raw_assignments {
            let outcome = verify_work_assignment(&session, &joined_pubkeys, &a);
            if outcome.is_ok() {
                assignments.push(a);
            } else {
                report.assignments_rejected += 1;
                report.rejection_notes.push(format!(
                    "kind=assignment session_id={session_id} id={} reason_tag={}",
                    a.assignment_id,
                    outcome.reason_tag(),
                ));
            }
        }
        report.assignments_accepted += assignments.len() as u32;

        // ── supersessions ──────────────────────────────────
        let raw_supersessions = store.list_verified_supersessions_for(&session_id)?;
        let mut supersessions: Vec<WorkAssignmentSupersession> = Vec::new();
        for s in raw_supersessions {
            let outcome =
                verify_assignment_supersession(&session, &assignments, &s);
            if outcome.is_ok() {
                supersessions.push(s);
            } else {
                report.supersessions_rejected += 1;
                report.rejection_notes.push(format!(
                    "kind=supersession session_id={session_id} id={} reason_tag={}",
                    s.supersession_id,
                    outcome.reason_tag(),
                ));
            }
        }
        report.supersessions_accepted += supersessions.len() as u32;

        snapshot.sessions.insert(session_id.clone(), session);
        if !joins.is_empty() {
            snapshot.joins_by_session.insert(session_id.clone(), joins);
        }
        if !assignments.is_empty() {
            snapshot
                .assignments_by_session
                .insert(session_id.clone(), assignments);
        }
        if !supersessions.is_empty() {
            snapshot
                .supersessions_by_session
                .insert(session_id.clone(), supersessions);
        }
        report.sessions_accepted += 1;
    }

    Ok((snapshot, report))
}

// ── compute_audit_health ────────────────────────────────────────

/// Stage 12.13 audit-coherence summary. Closed enum so scripts
/// can pattern-match on the discriminator without parsing free-
/// form notes. The field names are stable across Stage 12.13's
/// scripting surface (`event=audit_health` lines).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditCoherence {
    /// The status report shape matches a coherent, fully-applied
    /// chain. No partial-apply suspected; no `InvalidState`.
    Coherent,
    /// One or more supersessions reference `assignment_id`s that
    /// are not in the verified-assignment set. Suspected
    /// out-of-order arrival (handler accepted in best-effort
    /// mode) or a Phase-B partial apply where the supersession
    /// was published before its replacements were written.
    PartialApplySupersession {
        supersession_id: String,
        unresolved_count: u32,
    },
    /// Verified active assignments exist whose covering
    /// supersession is **missing or invalid**. Two structural
    /// signals union into this set:
    ///
    ///   - **Invalid-supersession reference**: an active
    ///     assignment is named in some supersession's
    ///     `replacement_assignment_ids` but that supersession is
    ///     `valid == false`. Typical cause: Phase B2 wrote a
    ///     corrupted supersession body, or the file was
    ///     hand-tampered post-write.
    ///   - **Duplicate active stage**: more than one
    ///     non-superseded assignment shares the same `stage_index`.
    ///     The v1 planner emits exactly one assignment per
    ///     stage, and the Stage 12.11 reassign planner builds
    ///     replacements that match the original `stage_index` —
    ///     so >1 active assignment at a single stage is a
    ///     coordination invariant violation. The most common
    ///     cause is a **Phase B1 partial apply**: B1 wrote the
    ///     replacement assignment file to the state-dir, but the
    ///     B2 supersession write never landed (FS op failed or
    ///     the operator aborted between B1 and B2). No
    ///     supersession file exists on disk, so the
    ///     invalid-supersession signal cannot catch it — this
    ///     duplicate-stage signal can.
    OrphanReplacementAssignments {
        assignment_ids: Vec<String>,
    },
    /// `overall_status == InvalidState` and at least one entry in
    /// `invalid_artifacts` is not an `InvalidPartial` —
    /// triage via `--reason invalid-partial` would refuse.
    NotReassignTriagable,
    /// `overall_status == InvalidState` and every entry IS an
    /// `InvalidPartial`. Triage via `--reason invalid-partial` is
    /// available.
    ReassignTriagable,
}

/// Operator-visible health summary derived from an existing
/// [`SessionStatusReport`]. Stage 12.13 keeps this **out of**
/// the v3 JSON schema (the report struct is frozen) — it lives
/// in renderers and library callers only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditHealth {
    pub coherence: AuditCoherence,
    /// True iff the v3 status is `InvalidState` AND every
    /// `invalid_artifacts` entry is `InvalidPartial` whose
    /// `assignment_id` is in the active-and-unsuperseded set.
    /// Cheap pre-check that mirrors the Stage 12.12 plan-time
    /// helper without building a synthetic plan.
    pub triagable_by_reassign: bool,
    /// Closed/static set so scripts can branch deterministically.
    /// Values:
    /// - `"none"` — no operator action needed.
    /// - `"run plan-session-reassign --reason missing-partial"`
    /// - `"run plan-session-reassign --reason invalid-partial"`
    /// - `"clean state-dir orphan replacements before retry"`
    /// - `"operator triage required"`
    /// - `"run archive-session --require-status aggregated"` (Stage 12.14)
    /// - `"run archive-session --require-status expired-incomplete"` (Stage 12.14)
    pub recommended_action: &'static str,
}

/// Derive the audit summary from a v3 status report. Does NOT
/// touch the state-dir, does NOT run any verifier — purely a
/// projection over fields the Stage 12.12 reporter already
/// populated. The result is stable across renderings on the same
/// input report.
pub fn compute_audit_health(report: &SessionStatusReport) -> AuditHealth {
    use std::collections::HashSet;

    // Reassign-triagable iff the Stage 12.12 plan-time precheck
    // says so. The precheck accepts InProgress and forwards
    // every other non-InvalidState arm to `check_repair_eligible`
    // — i.e. only `InProgress` returns `Ok` for non-InvalidState.
    // For audit purposes, "triagable_by_reassign" specifically
    // means "InvalidState that is reassign-fixable", not "the
    // session is healthy". So we compute it on InvalidState
    // explicitly.
    let triagable_by_reassign = report.overall_status
        == SessionOverallStatus::InvalidState
        && check_invalid_partial_plan_eligible(report).is_ok();

    // Build lookup sets for coherence detection.
    let active_assignment_ids: HashSet<&str> = report
        .assignments
        .iter()
        .filter(|a| !a.superseded)
        .map(|a| a.assignment_id.as_str())
        .collect();
    let verified_assignment_ids: HashSet<&str> = report
        .assignments
        .iter()
        .map(|a| a.assignment_id.as_str())
        .collect();

    // (1) Partial-apply suspected: a supersession's
    // `superseded_assignment_ids` ∪ `replacement_assignment_ids`
    // contains an id that is NOT in the verified-assignment set.
    // The state-dir lost track of an assignment the supersession
    // names — either it was never written (out-of-order arrival
    // accepted in best-effort mode) OR a Phase-B partial apply
    // wrote the supersession before its replacement assignment.
    for s in &report.supersessions {
        let mut unresolved = 0u32;
        for id in s.superseded_assignment_ids.iter().chain(s.replacement_assignment_ids.iter()) {
            if !verified_assignment_ids.contains(id.as_str()) {
                unresolved += 1;
            }
        }
        if unresolved > 0 {
            return AuditHealth {
                coherence: AuditCoherence::PartialApplySupersession {
                    supersession_id: s.supersession_id.clone(),
                    unresolved_count: unresolved,
                },
                triagable_by_reassign,
                recommended_action: "operator triage required",
            };
        }
    }

    // (2) Orphan replacement assignments. See the docstring on
    // `AuditCoherence::OrphanReplacementAssignments` for the full
    // taxonomy; here we union two structural signals:
    //
    //   (2a) An active assignment is named in some supersession's
    //        replacement_assignment_ids BUT that supersession is
    //        `valid == false`. Phase B2 wrote a corrupt body or
    //        a hand-tamper occurred post-write.
    //
    //   (2b) Multiple active (non-superseded) assignments share
    //        the same stage_index. The v1 planner emits exactly
    //        one assignment per stage, and the Stage 12.11
    //        reassign planner builds replacements that match the
    //        original stage_index — so >1 active assignment at
    //        a single stage is a coordination invariant
    //        violation. Common cause: the B1-only partial-apply
    //        case where the supersession file never landed, so
    //        (2a) can't see it. (2b) catches it via the
    //        structural duplicate.
    let referenced_replacement_ids: HashSet<&str> = report
        .supersessions
        .iter()
        .filter(|s| !s.valid)
        .flat_map(|s| s.replacement_assignment_ids.iter().map(|s| s.as_str()))
        .collect();
    let mut orphans: HashSet<String> = active_assignment_ids
        .iter()
        .filter(|id| referenced_replacement_ids.contains(*id))
        .map(|s| s.to_string())
        .collect();

    // (2b) Duplicate-active-stage detection. Group all active
    // assignments by stage_index; any stage with > 1 entry
    // contributes every one of its assignment_ids to the orphan
    // set. The audit's `recommended_action` tells the operator
    // to clean state-dir orphans; the body files carry
    // `assigned_at_utc` so the operator can identify which is
    // the original vs the replacement during cleanup.
    let mut by_stage: HashMap<u32, Vec<&AssignmentStatus>> = HashMap::new();
    for a in &report.assignments {
        if !a.superseded {
            by_stage.entry(a.stage_index).or_default().push(a);
        }
    }
    for (_stage, rows) in by_stage {
        if rows.len() > 1 {
            for row in rows {
                orphans.insert(row.assignment_id.clone());
            }
        }
    }

    if !orphans.is_empty() {
        let mut sorted: Vec<String> = orphans.into_iter().collect();
        sorted.sort();
        return AuditHealth {
            coherence: AuditCoherence::OrphanReplacementAssignments {
                assignment_ids: sorted,
            },
            triagable_by_reassign,
            recommended_action: "clean state-dir orphan replacements before retry",
        };
    }

    // (3) InvalidState arms.
    if report.overall_status == SessionOverallStatus::InvalidState {
        if triagable_by_reassign {
            return AuditHealth {
                coherence: AuditCoherence::ReassignTriagable,
                triagable_by_reassign,
                recommended_action:
                    "run plan-session-reassign --reason invalid-partial",
            };
        } else {
            return AuditHealth {
                coherence: AuditCoherence::NotReassignTriagable,
                triagable_by_reassign,
                recommended_action: "operator triage required",
            };
        }
    }

    // (4) Default-healthy. Stage 12.14 — recommend archival for
    // coherent terminal states (Aggregated, ExpiredIncomplete);
    // otherwise fall through to the Stage 12.13 missing-partial
    // recommendation for InProgress with an active-missing entry,
    // or "none" for everything else. `CompletePartials` is
    // intentionally NOT recommended for archival — the operator
    // may still want the aggregate to land.
    let any_active_missing = report
        .assignments
        .iter()
        .any(|a: &AssignmentStatus| !a.superseded && !a.partial_present);
    let recommended_action = match report.overall_status {
        SessionOverallStatus::Aggregated => {
            "run archive-session --require-status aggregated"
        }
        SessionOverallStatus::ExpiredIncomplete => {
            "run archive-session --require-status expired-incomplete"
        }
        SessionOverallStatus::InProgress if any_active_missing => {
            "run plan-session-reassign --reason missing-partial"
        }
        _ => "none",
    };
    AuditHealth {
        coherence: AuditCoherence::Coherent,
        triagable_by_reassign,
        recommended_action,
    }
}

// Suppress unused-import lint when the InvalidArtifactStatus path
// is referenced only through `report.invalid_artifacts` indexing
// in `check_invalid_partial_plan_eligible`. (Kept for clarity.)
#[allow(dead_code)]
fn _ensure_invalid_artifact_status_in_scope(_x: &InvalidArtifactStatus) {}
