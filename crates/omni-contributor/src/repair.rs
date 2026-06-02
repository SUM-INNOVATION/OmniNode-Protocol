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
/// Stage 12.11 bump: v1 → v2. v2 adds
/// `RepairStrategy::ReassignMissing` + `RepairAction::ReassignAssignment`
/// alongside the existing reannounce-only variants. Stage 12.10 v1
/// readers refuse to parse v2 plans (deny_unknown_fields on the
/// new enum tag), forcing an explicit upgrade signal.
pub const REPAIR_PLAN_SCHEMA_VERSION: u32 = 2;

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
    /// Stage 12.11 — build replacement `WorkAssignment`s for every
    /// missing active assignment, then publish a single signed
    /// `WorkAssignmentSupersession` covering the set.
    ReassignMissing,
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
    /// Stage 12.11 — supersede a missing assignment and replace it
    /// with a newly-signed `WorkAssignment` targeting a different
    /// contributor (or the same, with a different `assigned_at_utc`
    /// to produce a fresh `assignment_id`). The applier publishes
    /// the replacement assignment first, then publishes a single
    /// signed `WorkAssignmentSupersession` referencing all
    /// superseded + replacement IDs.
    ReassignAssignment {
        /// `assignment_id` of the missing original.
        superseded_assignment_id: String,
        /// Copied from the status report for dry-run context.
        original_stage_index: u32,
        /// `contributor_pubkey_hex` of the new contributor.
        replacement_contributor_pubkey_hex: String,
        /// Stage index for the replacement (typically matches the
        /// original — the planner enforces compatibility).
        replacement_stage_index: u32,
        /// `WorkKind` for the replacement assignment.
        replacement_work_kind: crate::session::WorkKind,
        replacement_expected_work_units: u64,
        replacement_expected_work_unit_kind: crate::result::WorkUnitKind,
        /// `SupersessionReason` that will be embedded in the
        /// signed `WorkAssignmentSupersession`.
        reason: crate::supersession::SupersessionReason,
    },
}

// ── SessionRepairPlan ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRepairPlan {
    pub schema_version: u32,
    pub session_id: String,
    /// BLAKE3 of the `(session_id, sorted [(assignment_id,
    /// partial_present, superseded, superseded_by_supersession_id)]
    /// pairs)` projection of the status report used to build this
    /// plan (Stage 12.11 v2 projection; see
    /// [`source_status_hash_hex`] for the v1 → v2 lift rationale).
    /// Apply-time recomputes from the current state-dir; drift →
    /// typed error.
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
/// [(assignment_id, partial_present, superseded,
/// superseded_by_supersession_id)] pairs)`. Two status reports
/// with different `generated_at_utc` or different `notes` but the
/// same shape produce the same projection — so a plan built from a
/// long-running watcher's snapshot stays valid until a partial
/// actually arrives OR a supersession changes the active-assignment
/// shape.
///
/// **Stage 12.11 lift**: prior to this stage the projection only
/// covered `(assignment_id, partial_present)`, which let a
/// supersession that arrived between plan and apply slip past the
/// drift check (the `partial_present` flag wouldn't flip if the
/// retired assignment never had a partial in the first place). The
/// applier would then reannounce/reassign an already-retired
/// assignment. Including the per-assignment `superseded` flag plus
/// the retiring `supersession_id` makes any change to the
/// active-assignment cover observable through this projection.
pub fn source_status_hash_hex(status: &SessionStatusReport) -> String {
    #[derive(serde::Serialize)]
    struct Projection<'a> {
        session_id: &'a str,
        assignments: Vec<(String, bool, bool, String)>,
    }
    let mut assignments: Vec<(String, bool, bool, String)> = status
        .assignments
        .iter()
        .map(|a| {
            (
                a.assignment_id.clone(),
                a.partial_present,
                a.superseded,
                a.superseded_by_supersession_id
                    .clone()
                    .unwrap_or_default(),
            )
        })
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
/// over `(session_id, sorted [(assignment_id, partial_present,
/// superseded, superseded_by_supersession_id)] pairs)` ONLY (Stage
/// 12.11 lift; see [`source_status_hash_hex`] for the v1 → v2
/// rationale): an `ExpiredIncomplete` (`--no-prune-state-on-start`)
/// or an `InvalidState` (e.g. invalid aggregate body) with the same
/// active-assignment cover would otherwise slip past the projection
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

/// Stage 12.12 — relaxed apply-time eligibility check for the
/// `--reason invalid-partial` reassign path. **Only** this helper
/// accepts `InvalidState`, and **only** when every entry in
/// `status.invalid_artifacts` is an `InvalidPartial` whose
/// `assignment_id` is in the plan's superseded set. Every other
/// status (`NoSession`, `NoAssignments`, `CompletePartials`,
/// `Aggregated`, `ExpiredIncomplete`) and every other invalid-
/// artifact kind (`InvalidSession`, `InvalidJoin`,
/// `InvalidAssignment`, `InvalidSupersession`, `InvalidAggregate`)
/// is refused via a typed [`RepairError::InvalidStateNotTriagable`]
/// with a stable `kind` tag, OR via [`check_repair_eligible`] for
/// the non-`InvalidState` overall-status arms.
///
/// `ReannounceMissing` and the `MissingPartial` / `OperatorRebalance`
/// reassign variants do NOT use this helper — they keep calling
/// `check_repair_eligible` directly and continue to refuse
/// `InvalidState` outright. The Stage 12.12 split is **purely**
/// `--reason invalid-partial` plumbing; nothing else relaxes.
///
/// Operator-meaningful invariant: triage is bounded by what the
/// plan retires. An operator cannot use one tampered partial to
/// shortcut around unrelated chain damage — every InvalidState
/// cause must be covered by the supersession this plan produces.
pub fn check_reassign_eligible_allowing_invalid_partials(
    status: &SessionStatusReport,
    plan: &SessionRepairPlan,
) -> Result<(), RepairError> {
    use crate::status::InvalidArtifactStatus;

    // Forward to the standard eligibility check for every overall
    // status arm except InvalidState. This preserves the typed
    // error surface for NoSession / NoAssignments / CompletePartials
    // / Aggregated / ExpiredIncomplete and keeps the InProgress
    // happy path.
    if status.overall_status != SessionOverallStatus::InvalidState {
        return check_repair_eligible(status);
    }

    // InvalidState path: only ReassignMissing plans are even
    // considered. ReannounceMissing never accepts InvalidState.
    if !matches!(plan.strategy, RepairStrategy::ReassignMissing) {
        return Err(RepairError::InvalidState);
    }

    // Collect the planned superseded set so per-entry checks can
    // ask "is this InvalidPartial covered by the plan?".
    let planned_targets: std::collections::HashSet<&str> = plan
        .actions
        .iter()
        .filter_map(|a| match a {
            RepairAction::ReassignAssignment {
                superseded_assignment_id,
                ..
            } => Some(superseded_assignment_id.as_str()),
            _ => None,
        })
        .collect();

    for entry in &status.invalid_artifacts {
        match entry {
            InvalidArtifactStatus::InvalidPartial {
                assignment_id,
                reason_tag: _,
            } => {
                if !planned_targets.contains(assignment_id.as_str()) {
                    return Err(RepairError::InvalidStateNotTriagable {
                        kind: "invalid_partial_not_in_plan",
                        context: format!(" assignment_id={assignment_id}"),
                    });
                }
            }
            InvalidArtifactStatus::InvalidSession { .. } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_session",
                    context: String::new(),
                });
            }
            InvalidArtifactStatus::InvalidJoin {
                contributor_pubkey_hex,
                ..
            } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_join",
                    context: format!(
                        " contributor_pubkey_hex={contributor_pubkey_hex}"
                    ),
                });
            }
            InvalidArtifactStatus::InvalidAssignment {
                assignment_id,
                ..
            } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_assignment",
                    context: format!(" assignment_id={assignment_id}"),
                });
            }
            InvalidArtifactStatus::InvalidSupersession {
                supersession_id,
                ..
            } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_supersession",
                    context: format!(" supersession_id={supersession_id}"),
                });
            }
            InvalidArtifactStatus::InvalidAggregate { .. } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_aggregate",
                    context: String::new(),
                });
            }
        }
    }

    // Defense-in-depth: if `overall_status == InvalidState` but
    // `invalid_artifacts` is empty, the v3 reporter invariant has
    // been violated by a hand-edited report. Refuse rather than
    // silently proceed.
    if status.invalid_artifacts.is_empty() {
        return Err(RepairError::InvalidStateNotTriagable {
            kind: "invalid_state_without_diagnostics",
            context: String::new(),
        });
    }

    Ok(())
}

/// Stage 12.12 review fix — **plan-time** triage precheck. The
/// apply-side
/// [`check_reassign_eligible_allowing_invalid_partials`] enforces
/// that every `invalid_artifacts` entry is an `InvalidPartial`
/// targeted by the plan. Without a matching gate at plan time,
/// `plan-session-reassign --reason invalid-partial` could emit a
/// plan that apply will reliably reject — e.g. when the status
/// has `InvalidPartial(A) + InvalidJoin(...)`, the planner used
/// to skim out only A, return a plan, and let apply reject
/// later. The contract is that the planner only accepts
/// `InvalidState` when **every** invalid artifact is a planned
/// invalid partial, so the planner must refuse the same set of
/// inputs the apply helper would refuse.
///
/// This helper runs that policy against the **candidate target
/// set the planner would emit**, which is:
/// `{ ip.assignment_id  for ip in InvalidPartial  s.t.
///     status.assignments contains row(ip.assignment_id)
///     && !row.superseded
///     && reason_tag != "unmatched" }`.
///
/// Decision matrix (mirrors the apply helper):
///
/// - `overall_status != InvalidState` → `Ok(())`. The caller
///   (the planner) then runs the usual `check_repair_eligible`.
/// - `overall_status == InvalidState` and
///   `invalid_artifacts.is_empty()` →
///   `InvalidStateNotTriagable { kind: "invalid_state_without_diagnostics" }`
///   (defense-in-depth — Stage 12.12 reporter invariant violation).
/// - Any non-`InvalidPartial` entry →
///   `InvalidStateNotTriagable { kind: <invalid_session | invalid_join |
///   invalid_assignment | invalid_supersession | invalid_aggregate>,
///   context }`.
/// - Any `InvalidPartial` that the planner would NOT emit an
///   action for (orphan via `reason_tag == "unmatched"`,
///   `assignment_id` absent from `status.assignments`, or
///   `row.superseded`) →
///   `InvalidStateNotTriagable { kind: "invalid_partial_not_in_plan",
///   context: " assignment_id=..." }` matching the apply helper's
///   error surface so scripts can write one rule for both
///   layers.
///
/// **Why a separate helper instead of reusing the apply
/// helper:** the apply helper needs a built plan to compute
/// `planned_targets`; at plan time we have the status only. This
/// function expresses the same rule against the implicit
/// candidate set the planner is about to emit.
pub fn check_invalid_partial_plan_eligible(
    status: &SessionStatusReport,
) -> Result<(), RepairError> {
    use crate::status::InvalidArtifactStatus;

    if status.overall_status != SessionOverallStatus::InvalidState {
        return Ok(());
    }
    if status.invalid_artifacts.is_empty() {
        return Err(RepairError::InvalidStateNotTriagable {
            kind: "invalid_state_without_diagnostics",
            context: String::new(),
        });
    }
    let assignments_by_id: std::collections::HashMap<
        &str,
        &crate::status::AssignmentStatus,
    > = status
        .assignments
        .iter()
        .map(|a| (a.assignment_id.as_str(), a))
        .collect();
    for entry in &status.invalid_artifacts {
        match entry {
            InvalidArtifactStatus::InvalidPartial {
                assignment_id,
                reason_tag,
            } => {
                let would_be_planned = reason_tag != "unmatched"
                    && assignments_by_id
                        .get(assignment_id.as_str())
                        .map(|row| !row.superseded)
                        .unwrap_or(false);
                if !would_be_planned {
                    return Err(RepairError::InvalidStateNotTriagable {
                        kind: "invalid_partial_not_in_plan",
                        context: format!(" assignment_id={assignment_id}"),
                    });
                }
            }
            InvalidArtifactStatus::InvalidSession { .. } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_session",
                    context: String::new(),
                });
            }
            InvalidArtifactStatus::InvalidJoin {
                contributor_pubkey_hex,
                ..
            } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_join",
                    context: format!(
                        " contributor_pubkey_hex={contributor_pubkey_hex}"
                    ),
                });
            }
            InvalidArtifactStatus::InvalidAssignment {
                assignment_id,
                ..
            } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_assignment",
                    context: format!(" assignment_id={assignment_id}"),
                });
            }
            InvalidArtifactStatus::InvalidSupersession {
                supersession_id,
                ..
            } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_supersession",
                    context: format!(" supersession_id={supersession_id}"),
                });
            }
            InvalidArtifactStatus::InvalidAggregate { .. } => {
                return Err(RepairError::InvalidStateNotTriagable {
                    kind: "invalid_aggregate",
                    context: String::new(),
                });
            }
        }
    }
    Ok(())
}

/// Stage 12.11 — apply-time per-action enforcement for
/// `ReassignMissing` plans. The `source_status_hash` projection
/// re-check binds the session-level *shape*, and `check_repair_eligible`
/// binds the overall status, but neither verifies that each
/// individual `ReassignAssignment` action targets a row that is
/// safe to retire.
///
/// `SessionRepairPlan` is **unsigned** and stored as local JSON, so
/// `repair_plan_hash_hex` is reachable by hand-editing the JSON
/// then recomputing the hash. Without this enforcement, an edited
/// plan can target an assignment that already has a valid partial
/// or is already retired by a prior verified supersession —
/// wasting valid work or producing a no-op publish — while the
/// session itself stays `InProgress` because some OTHER assignment
/// is still missing.
///
/// For each `ReassignAssignment` action this helper verifies the
/// targeted `superseded_assignment_id` exists in the current
/// status report, is `partial_present == false` AND
/// `superseded == false`, and has a `stage_index` matching the
/// plan's `original_stage_index`. Returns
/// [`RepairError::ReassignTargetNotActiveMissing`] with a
/// stable `reason` tag on the first failure.
///
/// No-op when `plan.actions` is empty or when every action is a
/// non-`ReassignAssignment` variant.
pub fn check_reassign_targets_active_missing(
    plan: &SessionRepairPlan,
    current_status: &SessionStatusReport,
) -> Result<(), RepairError> {
    let status_by_id: std::collections::HashMap<&str, &crate::status::AssignmentStatus> =
        current_status
            .assignments
            .iter()
            .map(|a| (a.assignment_id.as_str(), a))
            .collect();
    for action in &plan.actions {
        let RepairAction::ReassignAssignment {
            superseded_assignment_id,
            original_stage_index,
            ..
        } = action
        else {
            continue;
        };
        let row = match status_by_id.get(superseded_assignment_id.as_str()) {
            Some(r) => *r,
            None => {
                return Err(RepairError::ReassignTargetNotActiveMissing {
                    session_id: plan.session_id.clone(),
                    assignment_id: superseded_assignment_id.clone(),
                    reason: "not_in_status",
                });
            }
        };
        if row.superseded {
            return Err(RepairError::ReassignTargetNotActiveMissing {
                session_id: plan.session_id.clone(),
                assignment_id: superseded_assignment_id.clone(),
                reason: "already_superseded",
            });
        }
        if row.partial_present {
            return Err(RepairError::ReassignTargetNotActiveMissing {
                session_id: plan.session_id.clone(),
                assignment_id: superseded_assignment_id.clone(),
                reason: "already_completed",
            });
        }
        if row.stage_index != *original_stage_index {
            return Err(RepairError::ReassignTargetNotActiveMissing {
                session_id: plan.session_id.clone(),
                assignment_id: superseded_assignment_id.clone(),
                reason: "stage_index_mismatch",
            });
        }
    }
    Ok(())
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
    build_session_repair_plan_with_reason(
        status,
        strategy,
        crate::supersession::SupersessionReason::MissingPartial,
        now_utc,
        coordinator_pubkey_hex,
    )
}

/// Stage 12.11 — same as [`build_session_repair_plan`] but lets the
/// caller specify the `SupersessionReason` that gets embedded in
/// every `ReassignAssignment` action's downstream supersession.
/// `ReannounceMissing` ignores the reason (reannounce doesn't carry
/// one).
pub fn build_session_repair_plan_with_reason(
    status: &SessionStatusReport,
    strategy: RepairStrategy,
    reassign_reason: crate::supersession::SupersessionReason,
    now_utc: &str,
    coordinator_pubkey_hex: Option<&str>,
) -> Result<SessionRepairPlan, RepairError> {
    use crate::status::InvalidArtifactStatus;
    use crate::supersession::SupersessionReason;

    if status.schema_version != crate::status::STATUS_SCHEMA_VERSION {
        return Err(RepairError::UnsupportedSchemaVersion {
            got: status.schema_version,
            expected: crate::status::STATUS_SCHEMA_VERSION,
        });
    }

    // Stage 12.12 — `--reason invalid-partial` + `ReassignMissing`
    // is the only path that accepts `InvalidState` (and only when
    // every InvalidState cause is an InvalidPartial). Every other
    // reason / strategy combination falls through to the standard
    // `check_repair_eligible` gate — preserving the Stage 12.10 /
    // 12.11 contract that those paths hard-refuse `InvalidState`.
    let invalid_partial_path = matches!(
        (&strategy, &reassign_reason),
        (RepairStrategy::ReassignMissing, SupersessionReason::InvalidPartial)
    );

    if invalid_partial_path {
        // Stage 12.12 review fix — mirror the apply helper at
        // plan time. Without this, a status with
        // `InvalidPartial(A) + InvalidJoin(...)` would still
        // produce a plan covering A; the operator would then
        // hit a refusal at `apply-session-reassign`, having
        // already paid the cost of plan creation, review, and
        // hash-pinning. Refuse upfront with the same typed
        // error surface so the planner contract (`emits a plan
        // <=> apply will accept it`) holds end-to-end.
        check_invalid_partial_plan_eligible(status)?;
    } else {
        check_repair_eligible(status)?;
    }

    // Action selection branches on the path. Both branches still
    // operate on ACTIVE assignments (a superseded entry is already
    // retired; the aggregate verifier ignores it).
    let assignments_by_id: std::collections::HashMap<
        &str,
        &crate::status::AssignmentStatus,
    > = status
        .assignments
        .iter()
        .map(|a| (a.assignment_id.as_str(), a))
        .collect();

    let mut actions: Vec<RepairAction> = if invalid_partial_path {
        // Stage 12.12 — pick ACTIVE assignments whose status row
        // has an `InvalidPartial` diagnostic in
        // `status.invalid_artifacts`. By contract,
        // `check_invalid_partial_plan_eligible` already ran above
        // and refused any orphan (`reason_tag == "unmatched"`),
        // any superseded target, any unknown `assignment_id`, AND
        // any extra non-`InvalidPartial` entry. So the
        // `if let Some(row)` + `!row.superseded` filters here are
        // **defense-in-depth invariants** that hold by
        // construction; they are NOT the policy gate. The policy
        // gate is the precheck. Future edits that loosen the
        // precheck MUST NOT rely on these filters as the new
        // gate — extend the precheck instead, so the planner
        // contract (`emits a plan <=> apply will accept it`)
        // stays end-to-end.
        let mut targets: Vec<&crate::status::AssignmentStatus> = Vec::new();
        let mut seen: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        for entry in &status.invalid_artifacts {
            if let InvalidArtifactStatus::InvalidPartial { assignment_id, .. } = entry {
                if seen.contains(assignment_id.as_str()) {
                    continue;
                }
                if let Some(row) = assignments_by_id.get(assignment_id.as_str()) {
                    if row.superseded {
                        continue;
                    }
                    targets.push(*row);
                    seen.insert(assignment_id.as_str());
                }
            }
        }
        targets
            .into_iter()
            .map(|a| RepairAction::ReassignAssignment {
                superseded_assignment_id: a.assignment_id.clone(),
                original_stage_index: a.stage_index,
                replacement_contributor_pubkey_hex: a.contributor_pubkey_hex.clone(),
                replacement_stage_index: a.stage_index,
                replacement_work_kind: a.work_kind.clone(),
                replacement_expected_work_units: a.expected_work_units,
                replacement_expected_work_unit_kind: a.expected_work_unit_kind,
                reason: reassign_reason.clone(),
            })
            .collect()
    } else {
        // Stage 12.11 — MissingPartial / OperatorRebalance /
        // Custom (and ReannounceMissing): retire assignments whose
        // partial is genuinely missing.
        let actionable: Vec<&crate::status::AssignmentStatus> = status
            .assignments
            .iter()
            .filter(|a| !a.partial_present && !a.superseded)
            .collect();
        match strategy {
            RepairStrategy::ReannounceMissing => actionable
                .iter()
                .map(|a| RepairAction::ReannounceAssignment {
                    assignment_id: a.assignment_id.clone(),
                    stage_index: a.stage_index,
                    contributor_pubkey_hex: a.contributor_pubkey_hex.clone(),
                })
                .collect(),
            RepairStrategy::ReassignMissing => actionable
                .iter()
                .map(|a| RepairAction::ReassignAssignment {
                    superseded_assignment_id: a.assignment_id.clone(),
                    original_stage_index: a.stage_index,
                    replacement_contributor_pubkey_hex: a
                        .contributor_pubkey_hex
                        .clone(),
                    replacement_stage_index: a.stage_index,
                    replacement_work_kind: a.work_kind.clone(),
                    replacement_expected_work_units: a.expected_work_units,
                    replacement_expected_work_unit_kind: a.expected_work_unit_kind,
                    reason: reassign_reason.clone(),
                })
                .collect(),
        }
    };

    // Determinism: sort by (stage_index ASC, then assignment_id /
    // superseded_assignment_id ASC).
    actions.sort_by(|x, y| {
        let (xs, xi) = match x {
            RepairAction::ReannounceAssignment {
                stage_index,
                assignment_id,
                ..
            } => (*stage_index, assignment_id.clone()),
            RepairAction::ReassignAssignment {
                original_stage_index,
                superseded_assignment_id,
                ..
            } => (*original_stage_index, superseded_assignment_id.clone()),
        };
        let (ys, yi) = match y {
            RepairAction::ReannounceAssignment {
                stage_index,
                assignment_id,
                ..
            } => (*stage_index, assignment_id.clone()),
            RepairAction::ReassignAssignment {
                original_stage_index,
                superseded_assignment_id,
                ..
            } => (*original_stage_index, superseded_assignment_id.clone()),
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
