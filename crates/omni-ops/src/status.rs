//! Stage 12.9 — local pooled-session progress monitor.
//!
//! Read-only observability layer over the Stage 12.7
//! `ContributorStateStore`. Given a `session_id`, load every body
//! in `verified/sessions/<id>/...`, re-run the relevant Stage 12.3
//! verifiers, and produce a deterministic [`SessionStatusReport`].
//!
//! ## What this module is NOT
//!
//! - **NOT a coordination enforcer.** The report doesn't fail a
//!   session, refuse work, retry, or change anything on disk
//!   beyond an optional operator-supplied `--json-out`. It
//!   reports.
//! - **NOT a network protocol.** `SessionStatusReport` is local-
//!   only and **unsigned**. It is never SNIP-published, never
//!   gossiped, never canonical bytes. Two operators looking at the
//!   same session from different machines may see different
//!   reports — and that's correct, because the report describes
//!   *what their local state-dir contains*, not a global truth.
//! - **NOT a chain authority.** No transaction, no signature, no
//!   on-chain anchor.
//! - **NOT a peer-advert-driven liveness probe.** Peer adverts
//!   inform `peer_advert_present` per assignment but don't drive
//!   completion. A session can be `CompletePartials` or
//!   `Aggregated` with no advert in the state-dir at all.
//!
//! ## Trust boundary
//!
//! State-dir loaders are parse-only (Stage 12.7 review). This
//! module re-runs:
//!
//! - [`verify_execution_session`] on `session.json`,
//! - [`verify_contributor_join`] per join,
//! - [`verify_work_assignment`] per assignment (against the
//!   verified-joins pubkey set),
//! - [`verify_partial_result`] per partial (matched by
//!   `assignment_id`),
//! - [`verify_peer_advertisement_body`] per advert (against the
//!   verified joins),
//! - [`verify_aggregated_result`] full-chain when an aggregate is
//!   present.
//!
//! Any failing verifier drops the artifact from the **valid**
//! counts and surfaces in `notes`. The presence of any invalid
//! body sets `overall_status = InvalidState` so operators see the
//! signal at a glance.
//!
//! ## Determinism
//!
//! The same state-dir + the same `now_utc` yields the same report.
//! `assignments` are sorted by `(stage_index, assignment_id)` and
//! `notes` are appended in iteration order.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::StatusError;
use omni_contributor::peer_advert::ContributorPeerAdvertisement;
use omni_contributor::peer_routing::{
    verify_peer_advertisement_body, PeerAdvertisementOutcome,
};
use omni_contributor::result::WorkUnitKind;
use omni_contributor::session::{
    AggregatedContributorResult, ContributorJoin, ExecutionSession,
    PartialContributorResult, WorkAssignment, WorkKind,
};
use omni_contributor::session_verify::{
    check_not_expired, verify_aggregated_result_with_supersessions,
    verify_contributor_join, verify_execution_session, verify_partial_result,
    verify_work_assignment, SessionVerifyOutcome,
};
use omni_contributor::state::{ContributorStateStore, StateObjectKind};
use omni_contributor::supersession::WorkAssignmentSupersession;
use omni_contributor::supersession_verify::verify_assignment_supersession;

/// Stage 12.12 bump: v2 → v3. v3 adds **structured** chain-failure
/// diagnostics via `invalid_artifacts: Vec<InvalidArtifactStatus>`
/// so automation can decide whether an `InvalidState` is triagable
/// (e.g. via a Stage 12.11 `--reason invalid-partial` reassignment)
/// without parsing free-form `notes` strings. Every site that
/// currently flips `any_chain_invalid` also pushes a structured
/// entry. `notes` text is preserved (operator dashboards keep
/// rendering it); automation **must not** parse `notes` — the
/// stable contract is `invalid_artifacts` + each entry's
/// `reason_tag` from [`crate::SessionVerifyOutcome::reason_tag`].
///
/// Stage 12.11 bump: v1 → v2 added supersession-aware fields
/// (`supersessions`, `superseded` / `superseded_by_supersession_id`,
/// `active_assignment_count`, `superseded_assignment_count`).
/// Old v1/v2 readers refuse v3 (`deny_unknown_fields`) and the
/// schema_version mismatch surfaces clearly.
pub const STATUS_SCHEMA_VERSION: u32 = 3;

// ── SessionOverallStatus ────────────────────────────────────────

/// Closed enum describing the operator-visible state of a pooled
/// session at report time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionOverallStatus {
    /// No `session.json` for the requested `session_id` in the
    /// state-dir (or session.json failed its individual verifier
    /// — without a verified session we have no chain root to
    /// report against).
    NoSession,
    /// Session verified; zero valid assignments on disk.
    NoAssignments,
    /// At least one valid assignment lacks a valid partial.
    InProgress,
    /// Every valid assignment has exactly one valid partial; no
    /// aggregate yet.
    CompletePartials,
    /// Aggregate present AND `verify_aggregated_result` passed.
    Aggregated,
    /// `now_utc >= session.expires_at_utc` AND not aggregated.
    ExpiredIncomplete,
    /// Some loaded body failed individual re-verification (e.g.
    /// tampered signature, drift mismatch). Counts still report
    /// valid-only artifacts; see `notes` for per-artifact reasons.
    InvalidState,
}

// ── AssignmentStatus ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssignmentStatus {
    pub assignment_id: String,
    pub stage_index: u32,
    pub contributor_pubkey_hex: String,
    pub work_kind: WorkKind,
    pub expected_work_units: u64,
    pub expected_work_unit_kind: WorkUnitKind,
    /// True when at least one valid `ContributorJoin` exists for
    /// `contributor_pubkey_hex` (this is implied by the
    /// assignment having passed `verify_work_assignment`, but we
    /// surface it so dashboards can spot the case where someone
    /// dropped the join file but left the assignment).
    pub join_present: bool,
    /// True when a non-expired (or any, with `--include-expired`)
    /// verified peer advert exists for the contributor.
    pub peer_advert_present: bool,
    pub partial_present: bool,
    pub partial_valid: bool,
    pub partial_snip_root: Option<String>,
    /// Stage 12.11: true iff this assignment_id appears in some
    /// verified supersession's `superseded_assignment_ids`.
    pub superseded: bool,
    /// Stage 12.11: when `superseded == true`, the supersession_id
    /// of the verified supersession that supersedes this
    /// assignment. (At most one — double-supersession is verifier-
    /// level rejected.)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub superseded_by_supersession_id: Option<String>,
    pub notes: Vec<String>,
}

/// Stage 12.11 — per-supersession status entry on the report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupersessionStatus {
    pub supersession_id: String,
    pub superseded_assignment_ids: Vec<String>,
    pub replacement_assignment_ids: Vec<String>,
    pub reason: omni_contributor::supersession::SupersessionReason,
    /// `true` iff `verify_assignment_supersession` returned `Ok`
    /// against the loaded session + assignments.
    pub valid: bool,
    pub notes: Vec<String>,
}

// ── SessionStatusReport ─────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionStatusReport {
    pub schema_version: u32,
    pub session_id: String,
    /// `None` only when the session is missing entirely.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub posted_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model_hash: Option<String>,
    pub generated_at_utc: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_expires_at_utc: Option<String>,
    pub session_expired: bool,
    /// VALID-ONLY counts. Tampered artifacts surface as `notes`
    /// + `InvalidState` overall, not as inflated counts.
    pub join_count: u32,
    pub peer_advert_count: u32,
    pub assignment_count: u32,
    pub partial_count: u32,
    /// Stage 12.11 — Active = assignments NOT superseded by a
    /// verified `WorkAssignmentSupersession`. Derived counts.
    pub active_assignment_count: u32,
    pub superseded_assignment_count: u32,
    pub supersession_count: u32,
    /// Stage 12.11 — `missing_assignment_ids` is **active-only**:
    /// superseded assignments missing partials do NOT appear here
    /// because the chain doesn't need their partials anymore.
    pub missing_assignment_ids: Vec<String>,
    /// State layout writes
    /// `verified/sessions/<id>/partials/<assignment_id>.json`, so
    /// a duplicate per assignment_id is filesystem-impossible.
    /// Always empty in v1+v2; retained for forward compatibility.
    pub duplicate_partial_assignment_ids: Vec<String>,
    pub aggregate_present: bool,
    pub aggregate_valid: bool,
    pub overall_status: SessionOverallStatus,
    pub assignments: Vec<AssignmentStatus>,
    /// Stage 12.11 — verified supersessions for this session.
    pub supersessions: Vec<SupersessionStatus>,
    /// Stage 12.12 — **structured** per-artifact failure diagnostics
    /// suitable for machine policy. Populated alongside (not instead
    /// of) `notes` so existing operator dashboards keep rendering
    /// the free-form text. Invariant:
    /// `invalid_artifacts.is_empty() <==> overall_status != InvalidState`.
    /// Stable order: session → joins → assignments → partials →
    /// supersessions → aggregate (mirrors the loader walk in
    /// `build_session_status_report`). Automation that drives a
    /// triage policy (e.g. `--reason invalid-partial` reassignment)
    /// MUST read this field and never parse `notes`.
    #[serde(default)]
    pub invalid_artifacts: Vec<InvalidArtifactStatus>,
    /// Free-form per-artifact failure descriptions. Stable order:
    /// session → joins → assignments → partials → adverts →
    /// supersessions → aggregate. Renderers print these as
    /// `event=note context=... message=...` lines. Stage 12.12 keeps
    /// this populated alongside `invalid_artifacts` so dashboard
    /// output does not go dark — but automation must read the
    /// structured field instead.
    pub notes: Vec<String>,
}

// ── InvalidArtifactStatus (Stage 12.12) ─────────────────────────

/// Stage 12.12 — closed, externally-tagged enum naming **which**
/// chain artifact failed verification and **why**. One variant per
/// failure mode; the `reason_tag` field within each variant is the
/// stable string returned by
/// [`crate::SessionVerifyOutcome::reason_tag`]. Renaming a
/// `SessionVerifyOutcome` variant must NOT silently change the
/// reason tag — `reason_tag` is part of the v3 contract and is
/// pinned by `outcome_reason_tag_strings_are_stable` in
/// `session_verify` tests.
///
/// The `kind` discriminator (`invalid_session`, `invalid_join`,
/// `invalid_assignment`, `invalid_partial`, `invalid_supersession`,
/// `invalid_aggregate`) plus the `reason_tag` is what Stage 12.12
/// `check_reassign_eligible_allowing_invalid_partials` consults to
/// decide whether an `InvalidState` is triagable.
///
/// Special tags:
/// - `InvalidPartial { reason_tag: "unmatched" }` covers the "partial
///   body has no matching verified assignment" case. There is no
///   separate `OrphanPartial` variant — the apply-time
///   `check_reassign_targets_active_missing` check already refuses
///   `not_in_status` for any plan targeting an unmatched id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InvalidArtifactStatus {
    InvalidSession {
        reason_tag: String,
    },
    InvalidJoin {
        contributor_pubkey_hex: String,
        reason_tag: String,
    },
    InvalidAssignment {
        assignment_id: String,
        reason_tag: String,
    },
    InvalidPartial {
        assignment_id: String,
        reason_tag: String,
    },
    InvalidSupersession {
        supersession_id: String,
        reason_tag: String,
    },
    InvalidAggregate {
        reason_tag: String,
    },
}

// ── public entry point ─────────────────────────────────────────

/// Load + re-verify + report. The CLI's `session-status`
/// subcommand calls this; tests can also call it against a
/// `tempdir`-backed `ContributorStateStore` directly.
///
/// `now_utc` is RFC 3339 with `Z` suffix. Used for session expiry
/// and peer-advert expiry checks.
///
/// `include_expired` controls peer-advert filtering ONLY. Session
/// expiry is always evaluated and reported.
pub fn build_session_status_report(
    store: &ContributorStateStore,
    session_id: &str,
    now_utc: &str,
    include_expired: bool,
) -> Result<SessionStatusReport, StatusError> {
    let mut notes: Vec<String> = Vec::new();
    // Stage 12.12 — every chain-link failure ALSO pushes a typed
    // entry here. Invariant on return:
    //   invalid_artifacts.is_empty() <==> overall_status != InvalidState
    let mut invalid_artifacts: Vec<InvalidArtifactStatus> = Vec::new();
    // Chain-link failures (session/joins/assignments/partials/
    // aggregate) flip overall_status to InvalidState. Advert
    // failures produce notes but do NOT — adverts are routing
    // helpers, documented as not required for completion. We track
    // the two separately.
    let mut any_chain_invalid = false;

    // ── 1. Load session.json ─────────────────────────────────
    let session: Option<ExecutionSession> =
        store.read_verified_json(StateObjectKind::Session, session_id)?;
    let Some(session) = session else {
        return Ok(empty_no_session_report(session_id, now_utc));
    };
    if session.session_id != session_id {
        notes.push(format!(
            "session.json carries session_id={} but state-dir directory says {}; \
             treating as missing",
            session.session_id, session_id
        ));
        return Ok(empty_no_session_report(session_id, now_utc));
    }
    // Fail-closed: session signature must verify or we have NO
    // anchor for the rest of the chain. Continuing here would feed
    // an untrusted ExecutionSession into `verify_contributor_join`
    // / `verify_work_assignment` / etc., and the report would
    // carry session-derived fields (posted_id, model_hash,
    // expires_at_utc) lifted from a body whose coordinator
    // signature failed. Instead, return an InvalidState report
    // carrying NO session-derived fields and a single note
    // pointing at the failure.
    let session_outcome = verify_execution_session(&session);
    if !session_outcome.is_ok() {
        let mut report = empty_no_session_report(session_id, now_utc);
        report.overall_status = SessionOverallStatus::InvalidState;
        report.notes.push(format!(
            "session.json failed verify_execution_session for session_id={}",
            session_id
        ));
        report.invalid_artifacts.push(InvalidArtifactStatus::InvalidSession {
            reason_tag: session_outcome.reason_tag().to_string(),
        });
        return Ok(report);
    }

    // ── 2. Load + re-verify joins ──────────────────────────────
    let raw_joins = store.list_verified_joins_for(session_id)?;
    let mut joins: Vec<ContributorJoin> = Vec::new();
    for j in raw_joins {
        let outcome = verify_contributor_join(&session, &j);
        if outcome.is_ok() {
            joins.push(j);
        } else {
            notes.push(format!(
                "join contributor_pubkey={} failed verify_contributor_join",
                j.contributor_pubkey_hex
            ));
            invalid_artifacts.push(InvalidArtifactStatus::InvalidJoin {
                contributor_pubkey_hex: j.contributor_pubkey_hex.clone(),
                reason_tag: outcome.reason_tag().to_string(),
            });
            any_chain_invalid = true;
        }
    }
    let joined_pubkeys: HashSet<String> = joins
        .iter()
        .map(|j| j.contributor_pubkey_hex.clone())
        .collect();

    // ── 3. Load + re-verify assignments ────────────────────────
    let raw_assignments = store.list_verified_assignments_for(session_id)?;
    let mut assignments: Vec<WorkAssignment> = Vec::new();
    for a in raw_assignments {
        match verify_work_assignment(&session, &joined_pubkeys, &a) {
            SessionVerifyOutcome::Ok => assignments.push(a),
            other => {
                notes.push(format!(
                    "assignment_id={} failed verify_work_assignment: {other:?}",
                    a.assignment_id
                ));
                invalid_artifacts.push(InvalidArtifactStatus::InvalidAssignment {
                    assignment_id: a.assignment_id.clone(),
                    reason_tag: other.reason_tag().to_string(),
                });
                any_chain_invalid = true;
            }
        }
    }
    // Deterministic order: stage_index ASC, then assignment_id ASC.
    assignments.sort_by(|a, b| {
        a.stage_index
            .cmp(&b.stage_index)
            .then_with(|| a.assignment_id.cmp(&b.assignment_id))
    });

    // ── 4. Load + re-verify supersessions (Stage 12.11) ──────
    //
    // Stage 12.11 review fix (post-merge): supersessions MUST be
    // loaded and validated BEFORE partials. The aggregate verifier
    // skips partials whose assignment is superseded (the entire
    // point of `SupersessionReason::InvalidPartial`); the status
    // reporter has to match that posture or a coordinator-signed
    // InvalidPartial supersession would still leave the session
    // stuck at `InvalidState` because the tampered partial body
    // failed `verify_partial_result` before the supersession
    // told us it didn't need to be valid.
    let raw_supersessions = store.list_verified_supersessions_for(session_id)?;
    let mut supersession_statuses: Vec<SupersessionStatus> = Vec::new();
    let mut verified_supersessions: Vec<WorkAssignmentSupersession> =
        Vec::new();
    // Map: assignment_id → the supersession_id that supersedes it
    // (for per-assignment status reporting).
    let mut superseded_by: HashMap<String, String> = HashMap::new();
    let mut seen_superseded: HashSet<String> = HashSet::new();
    for s in raw_supersessions {
        let outcome = verify_assignment_supersession(
            &session,
            &assignments,
            &s,
        );
        let mut s_notes: Vec<String> = Vec::new();
        let mut s_valid = false;
        if outcome.is_ok() {
            // Check cross-supersession duplicate-supersedes here
            // too — the standalone verifier only checks one
            // supersession at a time.
            let mut any_duplicate = false;
            for id in &s.superseded_assignment_ids {
                if !seen_superseded.insert(id.clone()) {
                    s_notes.push(format!(
                        "double-supersession of assignment_id={id}"
                    ));
                    invalid_artifacts.push(InvalidArtifactStatus::InvalidSupersession {
                        supersession_id: s.supersession_id.clone(),
                        reason_tag: SessionVerifyOutcome::SupersessionDuplicateSupersedes {
                            assignment_id: id.clone(),
                        }
                        .reason_tag()
                        .to_string(),
                    });
                    any_chain_invalid = true;
                    any_duplicate = true;
                }
            }
            if !any_duplicate {
                s_valid = true;
                for id in &s.superseded_assignment_ids {
                    superseded_by.insert(id.clone(), s.supersession_id.clone());
                }
                verified_supersessions.push(s.clone());
            }
        } else {
            s_notes.push(format!(
                "supersession_id={} failed verify_assignment_supersession: {outcome:?}",
                s.supersession_id
            ));
            invalid_artifacts.push(InvalidArtifactStatus::InvalidSupersession {
                supersession_id: s.supersession_id.clone(),
                reason_tag: outcome.reason_tag().to_string(),
            });
            any_chain_invalid = true;
        }
        supersession_statuses.push(SupersessionStatus {
            supersession_id: s.supersession_id.clone(),
            superseded_assignment_ids: s.superseded_assignment_ids.clone(),
            replacement_assignment_ids: s.replacement_assignment_ids.clone(),
            reason: s.reason.clone(),
            valid: s_valid,
            notes: s_notes,
        });
    }
    let superseded_set: HashSet<String> = superseded_by.keys().cloned().collect();

    // ── 4'. Load + re-verify partials (matched by assignment) ───
    //
    // Partials whose `assignment_id` is in `superseded_set` are
    // SKIPPED — they're declared not-needed by a coordinator-signed
    // supersession. A tampered bad partial body for such an
    // assignment is exactly the `SupersessionReason::InvalidPartial`
    // case; failing the verifier on it would defeat the supersession.
    let raw_partials = store.list_verified_partials_for(session_id)?;
    let mut valid_partials_by_assignment: HashMap<String, PartialContributorResult> =
        HashMap::new();
    let assignments_by_id: HashMap<String, &WorkAssignment> = assignments
        .iter()
        .map(|a| (a.assignment_id.clone(), a))
        .collect();
    for p in raw_partials {
        if superseded_set.contains(&p.assignment_id) {
            // Skipped — supersession says we don't need this one.
            continue;
        }
        let Some(asn) = assignments_by_id.get(&p.assignment_id) else {
            notes.push(format!(
                "partial assignment_id={} has no matching verified assignment",
                p.assignment_id
            ));
            // Stage 12.12 — orphan partial. Approved plan uses
            // a uniform InvalidPartial variant with the stable
            // tag "unmatched" rather than a separate
            // OrphanPartial variant. The apply-time
            // check_reassign_targets_active_missing already
            // refuses any plan whose superseded_assignment_id
            // is unknown, so unmatched partials cannot be
            // triaged via reassignment.
            invalid_artifacts.push(InvalidArtifactStatus::InvalidPartial {
                assignment_id: p.assignment_id.clone(),
                reason_tag: "unmatched".to_string(),
            });
            any_chain_invalid = true;
            continue;
        };
        match verify_partial_result(asn, &p) {
            SessionVerifyOutcome::Ok => {
                valid_partials_by_assignment.insert(p.assignment_id.clone(), p);
            }
            other => {
                notes.push(format!(
                    "partial assignment_id={} failed verify_partial_result: {other:?}",
                    p.assignment_id
                ));
                invalid_artifacts.push(InvalidArtifactStatus::InvalidPartial {
                    assignment_id: p.assignment_id.clone(),
                    reason_tag: other.reason_tag().to_string(),
                });
                any_chain_invalid = true;
            }
        }
    }

    // ── 5. Load + re-verify peer adverts ───────────────────────
    let raw_adverts = store.list_verified_peer_adverts_for(session_id)?;
    let mut peer_adverts: Vec<ContributorPeerAdvertisement> = Vec::new();
    for a in raw_adverts {
        // Adverts re-verify against the verified joins set.
        // `Some(now_utc)` enforces expiry; `--include-expired`
        // overrides expiry filtering AFTER verification by adding
        // adverts that only failed on the expiry leg.
        let outcome = verify_peer_advertisement_body(&a, &joins, Some(now_utc));
        match outcome {
            PeerAdvertisementOutcome::Verified { body } => peer_adverts.push(*body),
            PeerAdvertisementOutcome::Expired { .. } if include_expired => {
                peer_adverts.push(a);
            }
            other => {
                notes.push(format!(
                    "peer advert contributor_pubkey={} failed verify_peer_advertisement_body: {}",
                    a.contributor_pubkey_hex,
                    stringify_advert_outcome(&other)
                ));
            }
        }
    }
    let peer_advert_pubkeys: HashSet<String> = peer_adverts
        .iter()
        .map(|a| a.contributor_pubkey_hex.clone())
        .collect();

    // ── 6. Aggregate (optional) ────────────────────────────────
    let raw_aggregate: Option<AggregatedContributorResult> =
        store.read_verified_aggregate_for(session_id)?;
    let (aggregate_present, aggregate_valid) = match raw_aggregate.as_ref() {
        Some(agg) => {
            // The state-dir keeps every body the full-chain
            // verifier needs in-memory — no SNIP fetch required.
            // Stage 12.11: we pass the verified supersessions
            // through to the supersession-aware verifier.
            let partials_vec: Vec<PartialContributorResult> = valid_partials_by_assignment
                .values()
                .cloned()
                .collect();
            let outcome =
                verify_aggregated_result_with_supersessions(
                    &session,
                    &joins,
                    &assignments,
                    &verified_supersessions,
                    &partials_vec,
                    agg,
                );
            if outcome.is_ok() {
                (true, true)
            } else {
                notes.push(format!(
                    "aggregated.json failed verify_aggregated_result_with_supersessions: {outcome:?}"
                ));
                invalid_artifacts.push(InvalidArtifactStatus::InvalidAggregate {
                    reason_tag: outcome.reason_tag().to_string(),
                });
                any_chain_invalid = true;
                (true, false)
            }
        }
        None => (false, false),
    };

    // ── 7. Compose per-assignment statuses + missing list ──────
    let mut missing_assignment_ids: Vec<String> = Vec::new();
    let mut assignment_statuses: Vec<AssignmentStatus> =
        Vec::with_capacity(assignments.len());
    for a in &assignments {
        let join_present = joined_pubkeys.contains(&a.contributor_pubkey_hex);
        let peer_advert_present =
            peer_advert_pubkeys.contains(&a.contributor_pubkey_hex);
        let partial = valid_partials_by_assignment.get(&a.assignment_id);
        let partial_present = partial.is_some();
        let partial_valid = partial.is_some();
        let partial_snip_root = partial
            .map(|p| p.partial_artifact_snip_root.clone());
        let superseded = superseded_set.contains(&a.assignment_id);
        let superseded_by_supersession_id =
            superseded_by.get(&a.assignment_id).cloned();
        let mut a_notes = Vec::new();
        if superseded {
            a_notes.push("superseded".to_string());
        }
        // Stage 12.11 — `missing_assignment_ids` is active-only.
        if !partial_present && !superseded {
            a_notes.push("missing partial".to_string());
            missing_assignment_ids.push(a.assignment_id.clone());
        }
        assignment_statuses.push(AssignmentStatus {
            assignment_id: a.assignment_id.clone(),
            stage_index: a.stage_index,
            contributor_pubkey_hex: a.contributor_pubkey_hex.clone(),
            work_kind: a.work_kind.clone(),
            expected_work_units: a.expected_work_units,
            expected_work_unit_kind: a.expected_work_unit_kind,
            join_present,
            peer_advert_present,
            partial_present,
            partial_valid,
            partial_snip_root,
            superseded,
            superseded_by_supersession_id,
            notes: a_notes,
        });
    }

    // ── 8. Expiry + overall status ─────────────────────────────
    let session_expired = matches!(
        check_not_expired(now_utc, &session.expires_at_utc),
        SessionVerifyOutcome::ExpiredAtCheck { .. }
    );
    let active_assignment_count =
        (assignments.len() as u32).saturating_sub(superseded_set.len() as u32);
    let overall_status = decide_overall_status(
        any_chain_invalid,
        aggregate_present,
        aggregate_valid,
        session_expired,
        &assignments,
        &superseded_set,
        &valid_partials_by_assignment,
    );

    Ok(SessionStatusReport {
        schema_version: STATUS_SCHEMA_VERSION,
        session_id: session_id.to_string(),
        posted_id: Some(session.posted_id.clone()),
        model_hash: Some(session.model_hash.clone()),
        generated_at_utc: now_utc.to_string(),
        session_expires_at_utc: Some(session.expires_at_utc.clone()),
        session_expired,
        join_count: joins.len() as u32,
        peer_advert_count: peer_adverts.len() as u32,
        assignment_count: assignments.len() as u32,
        partial_count: valid_partials_by_assignment.len() as u32,
        active_assignment_count,
        superseded_assignment_count: superseded_set.len() as u32,
        supersession_count: supersession_statuses.len() as u32,
        missing_assignment_ids,
        duplicate_partial_assignment_ids: Vec::new(),
        aggregate_present,
        aggregate_valid,
        overall_status,
        assignments: assignment_statuses,
        supersessions: supersession_statuses,
        invalid_artifacts,
        notes,
    })
}

// ── decide_overall_status ──────────────────────────────────────

fn decide_overall_status(
    any_chain_invalid: bool,
    aggregate_present: bool,
    aggregate_valid: bool,
    session_expired: bool,
    assignments: &[WorkAssignment],
    superseded_set: &HashSet<String>,
    valid_partials_by_assignment: &HashMap<String, PartialContributorResult>,
) -> SessionOverallStatus {
    // The priority order is documented at module level + Stage
    // 12.9 doc. Stage 12.11 update: `CompletePartials` is computed
    // over ACTIVE assignments (`assignments - superseded`) rather
    // than all assignments. `NoAssignments` likewise requires zero
    // ACTIVE assignments.
    if any_chain_invalid {
        return SessionOverallStatus::InvalidState;
    }
    if aggregate_present && aggregate_valid {
        return SessionOverallStatus::Aggregated;
    }
    if session_expired && !aggregate_valid {
        return SessionOverallStatus::ExpiredIncomplete;
    }
    let active: Vec<&WorkAssignment> = assignments
        .iter()
        .filter(|a| !superseded_set.contains(&a.assignment_id))
        .collect();
    if active.is_empty() {
        return SessionOverallStatus::NoAssignments;
    }
    let all_have_partials = active
        .iter()
        .all(|a| valid_partials_by_assignment.contains_key(&a.assignment_id));
    if all_have_partials {
        return SessionOverallStatus::CompletePartials;
    }
    SessionOverallStatus::InProgress
}

fn empty_no_session_report(session_id: &str, now_utc: &str) -> SessionStatusReport {
    SessionStatusReport {
        schema_version: STATUS_SCHEMA_VERSION,
        session_id: session_id.to_string(),
        posted_id: None,
        model_hash: None,
        generated_at_utc: now_utc.to_string(),
        session_expires_at_utc: None,
        session_expired: false,
        join_count: 0,
        peer_advert_count: 0,
        assignment_count: 0,
        partial_count: 0,
        active_assignment_count: 0,
        superseded_assignment_count: 0,
        supersession_count: 0,
        missing_assignment_ids: Vec::new(),
        duplicate_partial_assignment_ids: Vec::new(),
        aggregate_present: false,
        aggregate_valid: false,
        overall_status: SessionOverallStatus::NoSession,
        assignments: Vec::new(),
        supersessions: Vec::new(),
        invalid_artifacts: Vec::new(),
        notes: Vec::new(),
    }
}

fn stringify_advert_outcome(o: &PeerAdvertisementOutcome) -> String {
    use omni_contributor::peer_routing::PeerAdvertisementOutcome as O;
    match o {
        O::Verified { .. } => "verified".into(),
        O::AnnouncementSchemaMalformed(s) => format!("ann_schema_malformed:{s}"),
        O::AnnouncerSignatureFailed => "announcer_signature_fail".into(),
        O::SnipFetchFailed(s) => format!("snip_fetch_failed:{s}"),
        O::BodyParseFailed(s) => format!("body_parse_failed:{s}"),
        O::BodySchemaInvalid(s) => format!("body_schema_invalid:{s}"),
        O::AdvertisementIdMismatch { stored, derived } => {
            format!("advertisement_id_mismatch:stored={stored}:derived={derived}")
        }
        O::ContributorSignatureFailed => "contributor_signature_fail".into(),
        O::DriftMismatch { field } => format!("drift:{field}"),
        O::NoMatchingJoin => "no_matching_join".into(),
        O::Expired { expires_at, now } => {
            format!("expired:expires_at={expires_at}:now={now}")
        }
    }
}
