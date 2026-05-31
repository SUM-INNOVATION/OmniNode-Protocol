//! Stage 12.10 — local pooled-session repair planner.
//!
//! Given a Stage 12.9 [`SessionStatusReport`], produce a
//! deterministic [`SessionRepairPlan`] of operator follow-up
//! actions. The plan is a **local review artifact** — never signed,
//! never SNIP-published, never network-visible. The signed trust
//! artifact remains the Stage 12.3 `WorkAssignment` at apply time.
//!
//! ## v1 scope (deliberately narrow)
//!
//! Stage 12.10 ships ONE strategy
//! ([`RepairStrategy::ReannounceMissing`]) and ONE action variant
//! ([`RepairAction::ReannounceAssignment`]). That single combination
//! is sufficient to nudge a late-arriving contributor's libp2p
//! subscription without changing any protocol state:
//!
//! 1. Pick every assignment whose `partial_present == false`.
//! 2. Sort actions by `(stage_index, assignment_id)`.
//! 3. Emit an `AssignmentToReannounce` carrying enough context for
//!    the applier to refetch the assignment from the state-dir,
//!    re-verify it, republish its bytes to SNIP (content-addressed
//!    → same root), and broadcast a fresh
//!    `NetworkWorkAssignedAnnouncement`.
//!
//! ### Why no `ReassignMissing` in v1
//!
//! [`crate::session_verify::verify_aggregated_result`] requires
//! every assignment in the supplied slice to be referenced exactly
//! once in the aggregate's `partial_refs`. Adding a replacement
//! assignment for a missing partial would leave the old assignment
//! in the state-dir's `verified/sessions/<id>/assignments/`,
//! tripping `AggregateMissingPartialFor` at aggregate time. Without
//! a canonical supersession/cancellation envelope (a Stage 12.3
//! schema-bump migration), replacement assignments cannot be
//! safely added. Reassignment ships in Stage 12.11+ after an
//! explicit supersession design.
//!
//! ### Why no `AbandonLocal` in v1
//!
//! Local abandonment markers would invent state semantics that
//! Stage 12.9's status reporter and future retry loops would have
//! to learn. The state-dir is documented as the protocol-shape
//! cache; we don't grow it with local control bits in 12.10.
//!
//! ## Trust boundary
//!
//! The planner does NOT load anything from disk. It consumes an
//! already-built `SessionStatusReport` (which itself re-runs every
//! Stage 12.3 verifier — Stage 12.9 trust boundary). The applier
//! re-verifies each assignment fetched from the state-dir against
//! the freshly-fetched session before publishing. The plan is a
//! **suggestion**, not a trust anchor.

use serde::{Deserialize, Serialize};

use crate::error::RepairError;
use crate::status::{SessionOverallStatus, SessionStatusReport};

/// Pinned planner schema version. A future stage that adds
/// `RepairAction::ReassignMissing` (with a supersession field)
/// bumps this and migrates.
pub const REPAIR_PLAN_SCHEMA_VERSION: u32 = 1;

// ── RepairStrategy ──────────────────────────────────────────────

/// Closed enum. Single variant in v1.
///
/// Adding `ReassignMissing` is a `schema_version: 2` migration
/// because it requires a Stage 12.3-level supersession envelope
/// that does not exist in 12.10. See module-level doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub enum RepairStrategy {
    ReannounceMissing,
}

// ── RepairAction ────────────────────────────────────────────────

/// Closed enum. Single variant in v1.
///
/// `serde`'s externally-tagged format keeps the JSON forward-compat
/// when a future stage adds a second variant: existing readers will
/// fail on the unknown tag (good — they don't know how to apply
/// it), and the introduction of a new variant is observable from
/// `schema_version` alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum RepairAction {
    /// Republish the existing assignment's bytes to SNIP
    /// (content-addressed → same root) and broadcast a fresh
    /// `NetworkWorkAssignedAnnouncement`. The on-disk assignment
    /// body is unchanged; only the gossip announcement is new.
    ReannounceAssignment {
        /// 64-char lowercase hex. Maps to
        /// `verified/sessions/<id>/assignments/<assignment_id>.json`
        /// in the state-dir.
        assignment_id: String,
        /// Copied from the status report's `AssignmentStatus` so
        /// the applier can emit a useful `event=would_reannounce`
        /// without re-loading the assignment body for the dry-run
        /// path.
        stage_index: u32,
        /// Copied from the status report. Verified against the
        /// re-fetched assignment body at apply time.
        contributor_pubkey_hex: String,
    },
}

// ── SessionRepairPlan ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRepairPlan {
    pub schema_version: u32,
    pub session_id: String,
    /// BLAKE3 of the `(session_id, sorted [assignment_id,
    /// partial_present] pairs)` projection of the status report
    /// used to build this plan. Apply-time recomputes from the
    /// current state-dir; drift → typed error.
    pub source_status_hash: String,
    pub strategy: RepairStrategy,
    pub created_at_utc: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub coordinator_pubkey_hex: Option<String>,
    pub actions: Vec<RepairAction>,
    /// BLAKE3 over the canonical JSON body with `repair_plan_hash`
    /// itself cleared. Apply-time recompute → typed error on drift.
    pub repair_plan_hash: String,
}

// ── source_status_hash projection ───────────────────────────────

/// Compute the projection hash the apply step uses to detect drift.
///
/// The projection is **operator-meaningful**: `(session_id, sorted
/// [(assignment_id, partial_present)] pairs)`. Two status reports
/// with different `generated_at_utc` or different `notes` but the
/// same shape produce the same projection — so a plan built from a
/// long-running watcher's snapshot stays valid until a partial
/// actually arrives.
pub fn source_status_hash_hex(status: &SessionStatusReport) -> String {
    #[derive(serde::Serialize)]
    struct Projection<'a> {
        session_id: &'a str,
        assignments: Vec<(String, bool)>,
    }
    let mut assignments: Vec<(String, bool)> = status
        .assignments
        .iter()
        .map(|a| (a.assignment_id.clone(), a.partial_present))
        .collect();
    assignments.sort_by(|a, b| a.0.cmp(&b.0));
    let p = Projection {
        session_id: &status.session_id,
        assignments,
    };
    let bytes = serde_json::to_vec(&p).expect("serialize projection");
    let h = blake3::hash(&bytes);
    let mut s = String::with_capacity(64);
    for b in h.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ── repair_plan_hash ───────────────────────────────────────────

/// BLAKE3 over the canonical JSON of the plan with
/// `repair_plan_hash` cleared. Deterministic across serde
/// round-trips.
pub fn repair_plan_hash_hex(plan: &SessionRepairPlan) -> String {
    let mut cloned = plan.clone();
    cloned.repair_plan_hash = String::new();
    let bytes = serde_json::to_vec(&cloned).expect("serialize plan");
    let h = blake3::hash(&bytes);
    let mut s = String::with_capacity(64);
    for b in h.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ── build_session_repair_plan ──────────────────────────────────

/// Map a `SessionStatusReport.overall_status` onto the planner /
/// applier eligibility posture. Returns `Ok(())` when status is
/// `InProgress` (the only repair-eligible state at v1) and a typed
/// `RepairError` for every refused status.
///
/// Used by [`build_session_repair_plan`] at plan time AND by
/// `apply-session-repair` at apply time — the applier rebuilds the
/// current `SessionStatusReport` and re-checks before any SNIP /
/// mesh work, because [`source_status_hash_hex`] is a projection
/// over `(session_id, sorted [(assignment_id, partial_present)]
/// pairs)` ONLY: an `ExpiredIncomplete` (`--no-prune-state-on-start`)
/// or an `InvalidState` (e.g. invalid aggregate body) with the same
/// partial-present shape would otherwise slip past the projection
/// drift check.
pub fn check_repair_eligible(
    status: &SessionStatusReport,
) -> Result<(), RepairError> {
    match status.overall_status {
        SessionOverallStatus::NoSession => Err(RepairError::SessionNotPresent {
            session_id: status.session_id.clone(),
        }),
        SessionOverallStatus::InvalidState => Err(RepairError::InvalidState),
        SessionOverallStatus::ExpiredIncomplete => Err(RepairError::SessionExpired),
        SessionOverallStatus::NoAssignments
        | SessionOverallStatus::CompletePartials
        | SessionOverallStatus::Aggregated => Err(RepairError::NothingToRepair {
            status: format!("{:?}", status.overall_status),
        }),
        SessionOverallStatus::InProgress => Ok(()),
    }
}

/// Build a deterministic repair plan from a verified
/// `SessionStatusReport`.
///
/// The status report is the trust boundary: by construction it has
/// already re-verified every Stage 12.3 envelope it counts (Stage
/// 12.9 contract). This planner does **no I/O**, so unit tests can
/// drive it with synthetic reports.
///
/// Refused statuses (typed errors):
/// - `NoSession` → [`RepairError::SessionNotPresent`]
/// - `NoAssignments` / `CompletePartials` / `Aggregated` →
///   [`RepairError::NothingToRepair`] (the latter two because there
///   is nothing missing to nudge)
/// - `InvalidState` → [`RepairError::InvalidState`] (operator must
///   clean tampered artifacts first)
/// - `ExpiredIncomplete` → [`RepairError::SessionExpired`] (extend
///   the session via `open-session` first if you really mean it)
///
/// Only `InProgress` produces a non-empty plan. `coordinator_pubkey_hex`
/// is an operator hint copied into the plan for dry-run review; it
/// is NOT a trust check — the applier re-verifies the coordinator
/// seed against the freshly-fetched session.
pub fn build_session_repair_plan(
    status: &SessionStatusReport,
    strategy: RepairStrategy,
    now_utc: &str,
    coordinator_pubkey_hex: Option<&str>,
) -> Result<SessionRepairPlan, RepairError> {
    if status.schema_version != crate::status::STATUS_SCHEMA_VERSION {
        return Err(RepairError::UnsupportedSchemaVersion {
            got: status.schema_version,
            expected: crate::status::STATUS_SCHEMA_VERSION,
        });
    }
    check_repair_eligible(status)?;

    let mut actions: Vec<RepairAction> = status
        .assignments
        .iter()
        .filter(|a| !a.partial_present)
        .map(|a| RepairAction::ReannounceAssignment {
            assignment_id: a.assignment_id.clone(),
            stage_index: a.stage_index,
            contributor_pubkey_hex: a.contributor_pubkey_hex.clone(),
        })
        .collect();

    // Determinism: sort by (stage_index ASC, assignment_id ASC).
    actions.sort_by(|x, y| {
        let (xs, xi) = match x {
            RepairAction::ReannounceAssignment {
                stage_index,
                assignment_id,
                ..
            } => (*stage_index, assignment_id.clone()),
        };
        let (ys, yi) = match y {
            RepairAction::ReannounceAssignment {
                stage_index,
                assignment_id,
                ..
            } => (*stage_index, assignment_id.clone()),
        };
        xs.cmp(&ys).then_with(|| xi.cmp(&yi))
    });

    // Defense-in-depth: even though `InProgress` implies at least
    // one missing partial (Stage 12.9 decision tree), guard the
    // empty-action case explicitly. If someone introduces a future
    // status variant that maps here, we'd rather a typed error than
    // a silently-empty plan.
    if actions.is_empty() {
        return Err(RepairError::NothingToRepair {
            status: "InProgress with zero missing partials".into(),
        });
    }

    let source_status_hash = source_status_hash_hex(status);
    let mut plan = SessionRepairPlan {
        schema_version: REPAIR_PLAN_SCHEMA_VERSION,
        session_id: status.session_id.clone(),
        source_status_hash,
        strategy,
        created_at_utc: now_utc.to_string(),
        coordinator_pubkey_hex: coordinator_pubkey_hex.map(str::to_string),
        actions,
        repair_plan_hash: String::new(),
    };
    plan.repair_plan_hash = repair_plan_hash_hex(&plan);
    Ok(plan)
}
