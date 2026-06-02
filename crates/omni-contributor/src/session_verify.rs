//! Stage 12.3 — local verifier helpers for session-shaped artifacts.
//!
//! Pure helpers: bytes in, typed outcome out. Each helper is the
//! moral equivalent of one of the per-step checks the CLI's
//! `watch-sessions` emitter routes to bare-stdout events.
//!
//! What we verify:
//!   - Coordinator signature on the session.
//!   - Contributor signature on each join, bound to the session.
//!   - Coordinator signature on each assignment, verified against
//!     the *session's* `coordinator_pubkey_hex` (assignments do not
//!     themselves carry one).
//!   - Contributor signature on each partial, bound to its assignment.
//!   - Topology: assignments target joined contributors; partials
//!     match assignments; aggregate covers every assignment.
//!   - Time bounds (now < session.expires_at_utc) where applicable.
//!   - Coordinator signature on the aggregate, equal to the session
//!     coordinator.
//!
//! What we do NOT verify:
//!   - Semantic correctness of any partial's output bytes.
//!   - That contributors actually have the RAM they advertised.
//!   - That A's partial output is the right input for B's stage.
//!   - That the model_hash names a "good" model.
//!
//! Stage 12.3 accounting rule (verifier-side):
//!   - Final `ContributorResult.measured_accounting.total_base_units`
//!     is whatever the existing 12.0 `verify_result` says it should
//!     be (i.e. `input + output`).
//!   - This module does NOT sum partial totals into the final.
//!   - It DOES check each partial's accounting is structurally
//!     valid (one stage_contribution, matching pubkey).

use std::collections::HashSet;

use omni_store::SnipV2Adapter;
use omni_types::phase5::SnipV2ObjectId;

use crate::canonical::{
    aggregated_result_signing_input, assignment_id_hex, contributor_join_signing_input,
    execution_session_signing_input, net_aggregated_signing_input, net_assign_signing_input,
    net_join_signing_input, net_partial_signing_input, net_session_opened_signing_input,
    partial_result_signing_input, session_id_hex, work_assignment_signing_input,
};
use crate::net::{
    NetworkAggregatedResultAnnouncement, NetworkContributorJoinedAnnouncement,
    NetworkPartialResultAnnouncement, NetworkSessionOpenedAnnouncement,
    NetworkWorkAssignedAnnouncement,
};
use crate::session::{
    AggregatedContributorResult, ContributorJoin, ExecutionSession, PartialContributorResult,
    WorkAssignment,
};
use crate::signing::verify_signature_hex;

/// Per-helper outcome. Each variant maps to one CLI bare-stdout event
/// and one typed test assertion (mirrors 12.2's pattern).
///
/// `#[non_exhaustive]` since Stage 12.11 — future Stage 12.x verifier
/// extensions may add new variants without breaking downstream
/// `matches!` / `.is_ok()` patterns. Exhaustive `match` blocks need a
/// wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SessionVerifyOutcome {
    Ok,
    SchemaMalformed(String),
    SessionIdMismatch { stored: String, derived: String },
    AssignmentIdMismatch { stored: String, derived: String },
    CoordinatorSignatureFailed,
    ContributorSignatureFailed,
    BindingMismatch { field: &'static str },
    ExpiredAtCheck { now: String, expires_at: String },
    AggregateMissingPartialFor { assignment_id: String },
    AggregateExtraPartialFor { assignment_id: String },
    AggregateDuplicatePartialFor { assignment_id: String },
    AggregatePartialRefDrift { field: &'static str, assignment_id: String },
    AggregateCoordinatorMismatch,
    InternalError(String),

    // ── Stage 12.11 — assignment supersession verifier outcomes ──

    /// Supersession body failed its standalone schema check.
    SupersessionSchemaMalformed(String),
    /// `supersession.session_id != session.session_id`.
    SupersessionSessionMismatch,
    /// `supersession.coordinator_pubkey_hex != session.coordinator_pubkey_hex`.
    SupersessionCoordinatorMismatch,
    /// Stored `supersession_id` does not equal the recomputed
    /// canonical hash.
    SupersessionIdMismatch { stored: String, derived: String },
    /// `coordinator_signature_hex` over canonical signing input is
    /// invalid against `session.coordinator_pubkey_hex`.
    SupersessionCoordinatorSignatureFailed,
    /// A `superseded_assignment_ids` or `replacement_assignment_ids`
    /// entry references an `assignment_id` not present in the
    /// supplied assignments slice.
    SupersessionReferenceUnknown {
        kind: &'static str,
        assignment_id: String,
    },
    /// Two supersessions claim the same `assignment_id` in their
    /// `superseded_assignment_ids` lists. No double-supersession.
    SupersessionDuplicateSupersedes { assignment_id: String },
    /// `aggregate.partial_refs` references an `assignment_id` that
    /// is in the superseded set computed from the supersessions
    /// slice.
    AggregatePartialRefSuperseded { assignment_id: String },
}

impl SessionVerifyOutcome {
    pub fn is_ok(&self) -> bool {
        matches!(self, SessionVerifyOutcome::Ok)
    }

    /// Stage 12.12 — stable, machine-readable label naming the
    /// verifier-outcome variant. Used by the Stage 12.9 status
    /// reporter v3 to populate
    /// `InvalidArtifactStatus::*.reason_tag` so automation can
    /// decide whether an `InvalidState` is triagable (e.g. an
    /// `InvalidPartial` with `reason_tag = "ContributorSignatureFailed"`
    /// is reassignable; an `InvalidJoin` with `reason_tag =
    /// "ContributorSignatureFailed"` is NOT).
    ///
    /// Tags are part of the local `SessionStatusReport` v3 contract:
    /// they may be added (alongside new variants) but the existing
    /// tag → variant mapping is frozen. **Never derive from
    /// `format!("{self:?}")`** — `Debug` is not a stability surface
    /// and the tags must not silently drift if a variant is renamed.
    pub fn reason_tag(&self) -> &'static str {
        match self {
            SessionVerifyOutcome::Ok => "Ok",
            SessionVerifyOutcome::SchemaMalformed(_) => "SchemaMalformed",
            SessionVerifyOutcome::SessionIdMismatch { .. } => "SessionIdMismatch",
            SessionVerifyOutcome::AssignmentIdMismatch { .. } => {
                "AssignmentIdMismatch"
            }
            SessionVerifyOutcome::CoordinatorSignatureFailed => {
                "CoordinatorSignatureFailed"
            }
            SessionVerifyOutcome::ContributorSignatureFailed => {
                "ContributorSignatureFailed"
            }
            SessionVerifyOutcome::BindingMismatch { .. } => "BindingMismatch",
            SessionVerifyOutcome::ExpiredAtCheck { .. } => "ExpiredAtCheck",
            SessionVerifyOutcome::AggregateMissingPartialFor { .. } => {
                "AggregateMissingPartialFor"
            }
            SessionVerifyOutcome::AggregateExtraPartialFor { .. } => {
                "AggregateExtraPartialFor"
            }
            SessionVerifyOutcome::AggregateDuplicatePartialFor { .. } => {
                "AggregateDuplicatePartialFor"
            }
            SessionVerifyOutcome::AggregatePartialRefDrift { .. } => {
                "AggregatePartialRefDrift"
            }
            SessionVerifyOutcome::AggregateCoordinatorMismatch => {
                "AggregateCoordinatorMismatch"
            }
            SessionVerifyOutcome::InternalError(_) => "InternalError",
            SessionVerifyOutcome::SupersessionSchemaMalformed(_) => {
                "SupersessionSchemaMalformed"
            }
            SessionVerifyOutcome::SupersessionSessionMismatch => {
                "SupersessionSessionMismatch"
            }
            SessionVerifyOutcome::SupersessionCoordinatorMismatch => {
                "SupersessionCoordinatorMismatch"
            }
            SessionVerifyOutcome::SupersessionIdMismatch { .. } => {
                "SupersessionIdMismatch"
            }
            SessionVerifyOutcome::SupersessionCoordinatorSignatureFailed => {
                "SupersessionCoordinatorSignatureFailed"
            }
            SessionVerifyOutcome::SupersessionReferenceUnknown { .. } => {
                "SupersessionReferenceUnknown"
            }
            SessionVerifyOutcome::SupersessionDuplicateSupersedes { .. } => {
                "SupersessionDuplicateSupersedes"
            }
            SessionVerifyOutcome::AggregatePartialRefSuperseded { .. } => {
                "AggregatePartialRefSuperseded"
            }
        }
    }
}

/// Verify an `ExecutionSession` standalone: schema OK, `session_id`
/// equals the canonical hash, coordinator signature verifies.
pub fn verify_execution_session(session: &ExecutionSession) -> SessionVerifyOutcome {
    if let Err(e) = session.validate_schema() {
        return SessionVerifyOutcome::SchemaMalformed(e.to_string());
    }
    let derived = match session_id_hex(session) {
        Ok(s) => s,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    if derived != session.session_id {
        return SessionVerifyOutcome::SessionIdMismatch {
            stored: session.session_id.clone(),
            derived,
        };
    }
    let signing_input = match execution_session_signing_input(session) {
        Ok(b) => b,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    let ok = verify_signature_hex(
        &session.coordinator_pubkey_hex,
        &signing_input,
        &session.coordinator_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return SessionVerifyOutcome::CoordinatorSignatureFailed;
    }
    SessionVerifyOutcome::Ok
}

/// Verify a `ContributorJoin` bound to a previously-verified session.
pub fn verify_contributor_join(
    session: &ExecutionSession,
    join: &ContributorJoin,
) -> SessionVerifyOutcome {
    if let Err(e) = join.validate_schema() {
        return SessionVerifyOutcome::SchemaMalformed(e.to_string());
    }
    if join.session_id != session.session_id {
        return SessionVerifyOutcome::BindingMismatch {
            field: "join.session_id",
        };
    }
    let signing_input = match contributor_join_signing_input(join) {
        Ok(b) => b,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    let ok = verify_signature_hex(
        &join.contributor_pubkey_hex,
        &signing_input,
        &join.contributor_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return SessionVerifyOutcome::ContributorSignatureFailed;
    }
    SessionVerifyOutcome::Ok
}

/// Verify a `WorkAssignment`. The assignment carries no
/// `coordinator_pubkey_hex` field; its signature is verified against
/// the session's `coordinator_pubkey_hex`. The contributor must be
/// in the supplied set of joined pubkeys.
pub fn verify_work_assignment(
    session: &ExecutionSession,
    joined_pubkeys: &HashSet<String>,
    assignment: &WorkAssignment,
) -> SessionVerifyOutcome {
    if let Err(e) = assignment.validate_schema() {
        return SessionVerifyOutcome::SchemaMalformed(e.to_string());
    }
    if assignment.session_id != session.session_id {
        return SessionVerifyOutcome::BindingMismatch {
            field: "assignment.session_id",
        };
    }
    let derived = match assignment_id_hex(assignment) {
        Ok(s) => s,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    if derived != assignment.assignment_id {
        return SessionVerifyOutcome::AssignmentIdMismatch {
            stored: assignment.assignment_id.clone(),
            derived,
        };
    }
    if !joined_pubkeys.contains(&assignment.contributor_pubkey_hex) {
        return SessionVerifyOutcome::BindingMismatch {
            field: "assignment.contributor_pubkey_hex_not_joined",
        };
    }
    let signing_input = match work_assignment_signing_input(assignment) {
        Ok(b) => b,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    let ok = verify_signature_hex(
        &session.coordinator_pubkey_hex,
        &signing_input,
        &assignment.coordinator_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return SessionVerifyOutcome::CoordinatorSignatureFailed;
    }
    SessionVerifyOutcome::Ok
}

/// Verify a `PartialContributorResult` against its referenced
/// assignment.
pub fn verify_partial_result(
    assignment: &WorkAssignment,
    partial: &PartialContributorResult,
) -> SessionVerifyOutcome {
    if let Err(e) = partial.validate_schema() {
        return SessionVerifyOutcome::SchemaMalformed(e.to_string());
    }
    if partial.session_id != assignment.session_id {
        return SessionVerifyOutcome::BindingMismatch {
            field: "partial.session_id",
        };
    }
    if partial.assignment_id != assignment.assignment_id {
        return SessionVerifyOutcome::BindingMismatch {
            field: "partial.assignment_id",
        };
    }
    if partial.contributor_pubkey_hex != assignment.contributor_pubkey_hex {
        return SessionVerifyOutcome::BindingMismatch {
            field: "partial.contributor_pubkey_hex",
        };
    }
    let signing_input = match partial_result_signing_input(partial) {
        Ok(b) => b,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    let ok = verify_signature_hex(
        &partial.contributor_pubkey_hex,
        &signing_input,
        &partial.contributor_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return SessionVerifyOutcome::ContributorSignatureFailed;
    }
    SessionVerifyOutcome::Ok
}

/// Full-chain aggregate verifier. Verifies (in order):
///   1. The `ExecutionSession` itself (schema, id, coordinator sig).
///   2. Every supplied `ContributorJoin` (schema, binding to
///      session, contributor signature) — builds the joined-pubkey
///      set used to validate assignments.
///   3. Every supplied `WorkAssignment` (schema, binding to session,
///      assignment_id derivation, contributor in joined set,
///      coordinator signature against `session.coordinator_pubkey_hex`).
///   4. Every supplied `PartialContributorResult` (schema, binding
///      to its referenced assignment, contributor signature).
///   5. The aggregate envelope itself (schema, posted_id/session_id
///      drift against session, coordinator_pubkey equals session's).
///   6. Coverage: every assignment has exactly one ref; refs
///      drift-match supplied partials by stage_index +
///      contributor_pubkey + canonical_hash.
///   7. Coordinator signature on the aggregate body.
///
/// Returns the first failing outcome encountered (short-circuits).
/// On success, every link in the chain has been verified.
///
/// The joins argument is **load-bearing**: without it the verifier
/// cannot prove an assignment targets a joined contributor, so an
/// attacker-supplied aggregate could reference assignments for
/// contributors that never joined. The CLI's `aggregate-session`
/// fetches join SNIP roots and passes them here.
pub fn verify_aggregated_result(
    session: &ExecutionSession,
    joins: &[ContributorJoin],
    assignments: &[WorkAssignment],
    partials: &[PartialContributorResult],
    aggregate: &AggregatedContributorResult,
) -> SessionVerifyOutcome {
    // Stage 12.11 — wrapper for backwards compatibility. The Stage
    // 12.3 contract ("every assignment referenced exactly once") is
    // exactly the supersession-aware contract with an empty
    // supersessions slice (active == all assignments).
    verify_aggregated_result_with_supersessions(
        session,
        joins,
        assignments,
        &[],
        partials,
        aggregate,
    )
}

/// Stage 12.11 — full-chain aggregate verifier with supersession
/// support. Same posture as [`verify_aggregated_result`] plus:
///
/// - Each supplied `WorkAssignmentSupersession` is verified
///   individually via [`verify_assignment_supersession`].
/// - No `assignment_id` may appear in `superseded_assignment_ids`
///   of more than one supersession (no double-supersession). Chains
///   via sequential supersessions (`A→B`, `B→C`) ARE permitted.
/// - Coverage is computed over `active = assignments \ superseded`
///   rather than `assignments`.
/// - Partials whose `assignment_id` is in the superseded set are
///   **skipped** from the per-partial binding loop (a tampered
///   partial for an `InvalidPartial`-superseded assignment is
///   precisely the reason the supersession exists — re-failing the
///   verifier on the same bytes the coordinator already declared
///   invalid would defeat the purpose).
/// - `aggregate.partial_refs` referencing a superseded assignment
///   is rejected with `AggregatePartialRefSuperseded`.
pub fn verify_aggregated_result_with_supersessions(
    session: &ExecutionSession,
    joins: &[ContributorJoin],
    assignments: &[WorkAssignment],
    supersessions: &[crate::supersession::WorkAssignmentSupersession],
    partials: &[PartialContributorResult],
    aggregate: &AggregatedContributorResult,
) -> SessionVerifyOutcome {
    // 1. Session.
    let s_out = verify_execution_session(session);
    if !s_out.is_ok() {
        return s_out;
    }
    // 2. Joins → joined-pubkey set.
    let mut joined_pubkeys: HashSet<String> = HashSet::new();
    for j in joins {
        let j_out = verify_contributor_join(session, j);
        if !j_out.is_ok() {
            return j_out;
        }
        joined_pubkeys.insert(j.contributor_pubkey_hex.clone());
    }
    // 3. Assignments.
    for a in assignments {
        let a_out = verify_work_assignment(session, &joined_pubkeys, a);
        if !a_out.is_ok() {
            return a_out;
        }
    }

    // 3'. Supersessions (Stage 12.11). Verify each individually
    // before computing the union, then enforce no double-
    // supersession. Replacement assignments are normal
    // WorkAssignments and were verified in step 3.
    let mut superseded: HashSet<String> = HashSet::new();
    for s in supersessions {
        let s_out = crate::supersession_verify::verify_assignment_supersession(
            session,
            assignments,
            s,
        );
        if !s_out.is_ok() {
            return s_out;
        }
        for id in &s.superseded_assignment_ids {
            if !superseded.insert(id.clone()) {
                return SessionVerifyOutcome::SupersessionDuplicateSupersedes {
                    assignment_id: id.clone(),
                };
            }
        }
    }

    // 4. Partials, each bound to its assignment. Skip partials
    // whose assignment is in the superseded set.
    for p in partials {
        if superseded.contains(&p.assignment_id) {
            continue;
        }
        let assignment = match assignments
            .iter()
            .find(|a| a.assignment_id == p.assignment_id)
        {
            Some(a) => a,
            None => {
                return SessionVerifyOutcome::BindingMismatch {
                    field: "partial.assignment_id_not_in_supplied_assignments",
                };
            }
        };
        let p_out = verify_partial_result(assignment, p);
        if !p_out.is_ok() {
            return p_out;
        }
    }
    // 5. Aggregate self-checks.
    if let Err(e) = aggregate.validate_schema() {
        return SessionVerifyOutcome::SchemaMalformed(e.to_string());
    }
    if aggregate.session_id != session.session_id {
        return SessionVerifyOutcome::BindingMismatch {
            field: "aggregate.session_id",
        };
    }
    if aggregate.posted_id != session.posted_id {
        return SessionVerifyOutcome::BindingMismatch {
            field: "aggregate.posted_id",
        };
    }
    if aggregate.coordinator_pubkey_hex != session.coordinator_pubkey_hex {
        return SessionVerifyOutcome::AggregateCoordinatorMismatch;
    }

    // Every ACTIVE assignment must have exactly one ref; every ref
    // must resolve to a partial we have; drift fields must match.
    // Refs pointing at superseded assignments are rejected.
    let mut seen_for: HashSet<&str> = HashSet::new();
    for r in &aggregate.partial_refs {
        if superseded.contains(&r.assignment_id) {
            return SessionVerifyOutcome::AggregatePartialRefSuperseded {
                assignment_id: r.assignment_id.clone(),
            };
        }
        if !seen_for.insert(r.assignment_id.as_str()) {
            return SessionVerifyOutcome::AggregateDuplicatePartialFor {
                assignment_id: r.assignment_id.clone(),
            };
        }
        let assignment = match assignments
            .iter()
            .find(|a| a.assignment_id == r.assignment_id)
        {
            Some(a) => a,
            None => {
                return SessionVerifyOutcome::AggregateExtraPartialFor {
                    assignment_id: r.assignment_id.clone(),
                };
            }
        };
        let partial = match partials
            .iter()
            .find(|p| p.assignment_id == r.assignment_id)
        {
            Some(p) => p,
            None => {
                return SessionVerifyOutcome::AggregateExtraPartialFor {
                    assignment_id: r.assignment_id.clone(),
                };
            }
        };
        if r.stage_index != assignment.stage_index {
            return SessionVerifyOutcome::AggregatePartialRefDrift {
                field: "stage_index",
                assignment_id: r.assignment_id.clone(),
            };
        }
        if r.contributor_pubkey_hex != partial.contributor_pubkey_hex {
            return SessionVerifyOutcome::AggregatePartialRefDrift {
                field: "contributor_pubkey_hex",
                assignment_id: r.assignment_id.clone(),
            };
        }
        // partial_canonical_hash check against the partial we have.
        let bytes = match crate::canonical::canonical_partial_result_bytes(partial) {
            Ok(b) => b,
            Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
        };
        let recomputed = crate::canonical::hex_lower(blake3::hash(&bytes).as_bytes());
        if recomputed != r.partial_canonical_hash {
            return SessionVerifyOutcome::AggregatePartialRefDrift {
                field: "partial_canonical_hash",
                assignment_id: r.assignment_id.clone(),
            };
        }
    }
    // Coverage: every ACTIVE assignment must be referenced exactly
    // once. Superseded assignments are excluded.
    for a in assignments {
        if superseded.contains(&a.assignment_id) {
            continue;
        }
        if !seen_for.contains(a.assignment_id.as_str()) {
            return SessionVerifyOutcome::AggregateMissingPartialFor {
                assignment_id: a.assignment_id.clone(),
            };
        }
    }
    // Coordinator signature.
    let signing_input = match aggregated_result_signing_input(aggregate) {
        Ok(b) => b,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    let ok = verify_signature_hex(
        &aggregate.coordinator_pubkey_hex,
        &signing_input,
        &aggregate.coordinator_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return SessionVerifyOutcome::CoordinatorSignatureFailed;
    }
    SessionVerifyOutcome::Ok
}

/// Convenience: check `now_utc < expires_at_utc`. Returns
/// [`SessionVerifyOutcome::ExpiredAtCheck`] on failure, or `Ok`
/// otherwise. RFC 3339 / ISO 8601 lex-comparison is sufficient given
/// every timestamp ends with `Z` and uses the same precision.
pub fn check_not_expired(
    now_utc: &str,
    expires_at_utc: &str,
) -> SessionVerifyOutcome {
    if now_utc >= expires_at_utc {
        return SessionVerifyOutcome::ExpiredAtCheck {
            now: now_utc.to_string(),
            expires_at: expires_at_utc.to_string(),
        };
    }
    SessionVerifyOutcome::Ok
}

// ── Stage 12.3 — per-announcement processing helpers ──────────────────────
//
// One helper per session topic. Each helper performs, in order:
//   1. Announcement schema validate.
//   2. Announcer signature verify against the canonical net body.
//   3. SNIP fetch by `*_snip_root`.
//   4. Parse + schema-validate the inner envelope.
//   5. Verify the inner self-signature WHERE POSSIBLE (i.e. for
//      every envelope whose signer's pubkey is recoverable from
//      either the inner body alone or the supplied session
//      context).
//   6. Drift-check every shared field between announcement and body.
//
// Returns `AnnouncementOutcome<T>` for the inner envelope `T`. The
// `watch-sessions` CLI handler routes outcomes to bare-stdout events
// and writes the inner body to disk only on `Verified`.

/// Generic per-announcement processing outcome. `T` is the inner
/// envelope type (e.g. `ExecutionSession`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnouncementOutcome<T> {
    /// All checks passed; `body` is the parsed inner envelope.
    Verified { body: T },
    /// The announcement envelope itself failed schema validation.
    AnnouncementSchemaMalformed(String),
    /// The announcer signature did not verify against
    /// `announcer_pubkey_hex`.
    AnnouncerSignatureFailed,
    /// SNIP fetch failed (bad root, missing object, transport error).
    SnipFetchFailed(String),
    /// The fetched bytes did not parse as the expected inner envelope.
    BodyParseFailed(String),
    /// The fetched inner envelope failed schema validation.
    BodySchemaInvalid(String),
    /// The inner envelope's own signature did not verify (where the
    /// processor could check it without external context).
    BodySignatureFailed,
    /// A drift-guard field on the announcement disagreed with the
    /// matching field on the fetched body.
    DriftMismatch { field: &'static str },
}

impl<T> AnnouncementOutcome<T> {
    pub fn is_verified(&self) -> bool {
        matches!(self, AnnouncementOutcome::Verified { .. })
    }
}

fn snip_fetch_body<A: SnipV2Adapter + ?Sized, T: serde::de::DeserializeOwned>(
    adapter: &A,
    snip_root_hex: &str,
) -> Result<(T, Vec<u8>), (&'static str, String)> {
    let root = SnipV2ObjectId::from_hex(snip_root_hex)
        .map_err(|e| ("snip-root-parse", format!("{e:?}")))?;
    let bytes = crate::snip::fetch_bytes(adapter, &root)
        .map_err(|e| ("snip-fetch", e.to_string()))?;
    let body: T =
        serde_json::from_slice(&bytes).map_err(|e| ("body-parse", e.to_string()))?;
    Ok((body, bytes))
}

/// Process a `NetworkSessionOpenedAnnouncement`: validates the
/// announcement schema + announcer signature, fetches the
/// `ExecutionSession` from SNIP, validates + verifies its
/// coordinator signature, then drift-checks every shared field.
pub fn process_session_opened_announcement<A: SnipV2Adapter + ?Sized>(
    ann: &NetworkSessionOpenedAnnouncement,
    adapter: &A,
) -> AnnouncementOutcome<ExecutionSession> {
    if let Err(e) = ann.validate_schema() {
        return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string());
    }
    let signing_input = match net_session_opened_signing_input(ann) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string()),
    };
    let ok = verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &signing_input,
        &ann.announcer_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return AnnouncementOutcome::AnnouncerSignatureFailed;
    }
    let (session, _bytes): (ExecutionSession, _) =
        match snip_fetch_body(adapter, &ann.execution_session_snip_root) {
            Ok(x) => x,
            Err((tag, msg)) => match tag {
                "body-parse" => return AnnouncementOutcome::BodyParseFailed(msg),
                _ => return AnnouncementOutcome::SnipFetchFailed(format!("{tag}: {msg}")),
            },
        };
    // Body self-verify (schema + coordinator sig + session_id derivation).
    let s_out = verify_execution_session(&session);
    match s_out {
        SessionVerifyOutcome::Ok => {}
        SessionVerifyOutcome::SchemaMalformed(s) => {
            return AnnouncementOutcome::BodySchemaInvalid(s);
        }
        SessionVerifyOutcome::CoordinatorSignatureFailed => {
            return AnnouncementOutcome::BodySignatureFailed;
        }
        other => {
            return AnnouncementOutcome::BodySchemaInvalid(format!("{other:?}"));
        }
    }
    // Drift checks.
    if ann.session_id != session.session_id {
        return AnnouncementOutcome::DriftMismatch { field: "session_id" };
    }
    if ann.posted_id != session.posted_id {
        return AnnouncementOutcome::DriftMismatch { field: "posted_id" };
    }
    AnnouncementOutcome::Verified { body: session }
}

/// Process a `NetworkContributorJoinedAnnouncement`.
pub fn process_contributor_joined_announcement<A: SnipV2Adapter + ?Sized>(
    ann: &NetworkContributorJoinedAnnouncement,
    adapter: &A,
) -> AnnouncementOutcome<ContributorJoin> {
    if let Err(e) = ann.validate_schema() {
        return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string());
    }
    let signing_input = match net_join_signing_input(ann) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string()),
    };
    let ok = verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &signing_input,
        &ann.announcer_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return AnnouncementOutcome::AnnouncerSignatureFailed;
    }
    let (join, _bytes): (ContributorJoin, _) =
        match snip_fetch_body(adapter, &ann.contributor_join_snip_root) {
            Ok(x) => x,
            Err((tag, msg)) => match tag {
                "body-parse" => return AnnouncementOutcome::BodyParseFailed(msg),
                _ => return AnnouncementOutcome::SnipFetchFailed(format!("{tag}: {msg}")),
            },
        };
    if let Err(e) = join.validate_schema() {
        return AnnouncementOutcome::BodySchemaInvalid(e.to_string());
    }
    // Body sig: contributor signs the canonical join. No external
    // context needed.
    let body_sig_input = match contributor_join_signing_input(&join) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::BodySchemaInvalid(e.to_string()),
    };
    let body_sig_ok = verify_signature_hex(
        &join.contributor_pubkey_hex,
        &body_sig_input,
        &join.contributor_signature_hex,
    )
    .unwrap_or(false);
    if !body_sig_ok {
        return AnnouncementOutcome::BodySignatureFailed;
    }
    if ann.session_id != join.session_id {
        return AnnouncementOutcome::DriftMismatch { field: "session_id" };
    }
    if ann.contributor_pubkey_hex != join.contributor_pubkey_hex {
        return AnnouncementOutcome::DriftMismatch {
            field: "contributor_pubkey_hex",
        };
    }
    AnnouncementOutcome::Verified { body: join }
}

/// Process a `NetworkWorkAssignedAnnouncement`. If
/// `session_coord_pubkey_hex` is `Some`, the processor also
/// verifies the assignment's coordinator signature against it
/// (assignments do not carry their own coordinator pubkey — the
/// session is the source of truth). If `None`, the body-sig check
/// is skipped (watcher hasn't seen the session yet).
pub fn process_work_assigned_announcement<A: SnipV2Adapter + ?Sized>(
    ann: &NetworkWorkAssignedAnnouncement,
    adapter: &A,
    session_coord_pubkey_hex: Option<&str>,
) -> AnnouncementOutcome<WorkAssignment> {
    if let Err(e) = ann.validate_schema() {
        return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string());
    }
    let signing_input = match net_assign_signing_input(ann) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string()),
    };
    let ok = verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &signing_input,
        &ann.announcer_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return AnnouncementOutcome::AnnouncerSignatureFailed;
    }
    let (asn, _bytes): (WorkAssignment, _) =
        match snip_fetch_body(adapter, &ann.work_assignment_snip_root) {
            Ok(x) => x,
            Err((tag, msg)) => match tag {
                "body-parse" => return AnnouncementOutcome::BodyParseFailed(msg),
                _ => return AnnouncementOutcome::SnipFetchFailed(format!("{tag}: {msg}")),
            },
        };
    if let Err(e) = asn.validate_schema() {
        return AnnouncementOutcome::BodySchemaInvalid(e.to_string());
    }
    // assignment_id must equal the canonical hash of the body.
    let derived = match assignment_id_hex(&asn) {
        Ok(s) => s,
        Err(e) => return AnnouncementOutcome::BodySchemaInvalid(e.to_string()),
    };
    if derived != asn.assignment_id {
        return AnnouncementOutcome::BodySchemaInvalid(format!(
            "assignment_id mismatch: stored={}, derived={derived}",
            asn.assignment_id
        ));
    }
    // Body sig: assignment is signed by the *session's* coordinator
    // (the body itself doesn't carry the coordinator pubkey). If
    // the caller supplied it, verify; else skip and note the
    // caller is on the hook to chain-verify later.
    if let Some(coord_pubkey) = session_coord_pubkey_hex {
        let body_sig_input = match work_assignment_signing_input(&asn) {
            Ok(b) => b,
            Err(e) => return AnnouncementOutcome::BodySchemaInvalid(e.to_string()),
        };
        let body_sig_ok = verify_signature_hex(
            coord_pubkey,
            &body_sig_input,
            &asn.coordinator_signature_hex,
        )
        .unwrap_or(false);
        if !body_sig_ok {
            return AnnouncementOutcome::BodySignatureFailed;
        }
    }
    if ann.session_id != asn.session_id {
        return AnnouncementOutcome::DriftMismatch { field: "session_id" };
    }
    if ann.assignment_id != asn.assignment_id {
        return AnnouncementOutcome::DriftMismatch { field: "assignment_id" };
    }
    if ann.contributor_pubkey_hex != asn.contributor_pubkey_hex {
        return AnnouncementOutcome::DriftMismatch {
            field: "contributor_pubkey_hex",
        };
    }
    AnnouncementOutcome::Verified { body: asn }
}

/// Process a `NetworkPartialResultAnnouncement`.
pub fn process_partial_result_announcement<A: SnipV2Adapter + ?Sized>(
    ann: &NetworkPartialResultAnnouncement,
    adapter: &A,
) -> AnnouncementOutcome<PartialContributorResult> {
    if let Err(e) = ann.validate_schema() {
        return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string());
    }
    let signing_input = match net_partial_signing_input(ann) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string()),
    };
    let ok = verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &signing_input,
        &ann.announcer_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return AnnouncementOutcome::AnnouncerSignatureFailed;
    }
    let (par, _bytes): (PartialContributorResult, _) =
        match snip_fetch_body(adapter, &ann.partial_result_snip_root) {
            Ok(x) => x,
            Err((tag, msg)) => match tag {
                "body-parse" => return AnnouncementOutcome::BodyParseFailed(msg),
                _ => return AnnouncementOutcome::SnipFetchFailed(format!("{tag}: {msg}")),
            },
        };
    if let Err(e) = par.validate_schema() {
        return AnnouncementOutcome::BodySchemaInvalid(e.to_string());
    }
    let body_sig_input = match partial_result_signing_input(&par) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::BodySchemaInvalid(e.to_string()),
    };
    let body_sig_ok = verify_signature_hex(
        &par.contributor_pubkey_hex,
        &body_sig_input,
        &par.contributor_signature_hex,
    )
    .unwrap_or(false);
    if !body_sig_ok {
        return AnnouncementOutcome::BodySignatureFailed;
    }
    if ann.session_id != par.session_id {
        return AnnouncementOutcome::DriftMismatch { field: "session_id" };
    }
    if ann.assignment_id != par.assignment_id {
        return AnnouncementOutcome::DriftMismatch { field: "assignment_id" };
    }
    if ann.contributor_pubkey_hex != par.contributor_pubkey_hex {
        return AnnouncementOutcome::DriftMismatch {
            field: "contributor_pubkey_hex",
        };
    }
    AnnouncementOutcome::Verified { body: par }
}

/// Process a `NetworkAggregatedResultAnnouncement`. Verifies the
/// announcer signature, fetches the aggregate, validates its
/// schema, and verifies the coordinator's signature on the body
/// (the aggregate carries the coordinator pubkey itself).
pub fn process_aggregated_result_announcement<A: SnipV2Adapter + ?Sized>(
    ann: &NetworkAggregatedResultAnnouncement,
    adapter: &A,
) -> AnnouncementOutcome<AggregatedContributorResult> {
    if let Err(e) = ann.validate_schema() {
        return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string());
    }
    let signing_input = match net_aggregated_signing_input(ann) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string()),
    };
    let ok = verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &signing_input,
        &ann.announcer_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return AnnouncementOutcome::AnnouncerSignatureFailed;
    }
    let (agg, _bytes): (AggregatedContributorResult, _) =
        match snip_fetch_body(adapter, &ann.aggregated_result_snip_root) {
            Ok(x) => x,
            Err((tag, msg)) => match tag {
                "body-parse" => return AnnouncementOutcome::BodyParseFailed(msg),
                _ => return AnnouncementOutcome::SnipFetchFailed(format!("{tag}: {msg}")),
            },
        };
    if let Err(e) = agg.validate_schema() {
        return AnnouncementOutcome::BodySchemaInvalid(e.to_string());
    }
    let body_sig_input = match aggregated_result_signing_input(&agg) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::BodySchemaInvalid(e.to_string()),
    };
    let body_sig_ok = verify_signature_hex(
        &agg.coordinator_pubkey_hex,
        &body_sig_input,
        &agg.coordinator_signature_hex,
    )
    .unwrap_or(false);
    if !body_sig_ok {
        return AnnouncementOutcome::BodySignatureFailed;
    }
    if ann.session_id != agg.session_id {
        return AnnouncementOutcome::DriftMismatch { field: "session_id" };
    }
    if ann.posted_id != agg.posted_id {
        return AnnouncementOutcome::DriftMismatch { field: "posted_id" };
    }
    AnnouncementOutcome::Verified { body: agg }
}

/// Stage 12.11 — process a
/// `NetworkWorkAssignmentSupersessionAnnouncement`.
///
/// Validates the announcement schema + announcer signature, fetches
/// the `WorkAssignmentSupersession` body from SNIP, validates the
/// body schema, re-derives `supersession_id`, verifies the body's
/// coordinator signature against the body's own
/// `coordinator_pubkey_hex`, and drift-checks every shared field.
///
/// If the caller knows the parent `ExecutionSession`, it can pass
/// it as `session`: the helper will then additionally pin the body
/// `coordinator_pubkey_hex` to `session.coordinator_pubkey_hex`. If
/// the caller also passes the list of verified `WorkAssignment`s
/// for the session, the helper will run the full
/// `verify_assignment_supersession` reference-resolution leg too
/// — including the v1 non-empty replacement and disjointness
/// invariants. (Both contexts default to `None` so a watcher that
/// has not yet seen the session can still accept the announcement
/// for later coord-pinning at aggregate time.)
pub fn process_assignment_supersession_announcement<A: SnipV2Adapter + ?Sized>(
    ann: &crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
    adapter: &A,
    session: Option<&ExecutionSession>,
    assignments: Option<&[WorkAssignment]>,
) -> AnnouncementOutcome<crate::supersession::WorkAssignmentSupersession> {
    use crate::canonical::{
        net_supersession_signing_input, supersession_id_hex,
        work_assignment_supersession_signing_input,
    };
    use crate::supersession::WorkAssignmentSupersession;

    if let Err(e) = ann.validate_schema() {
        return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string());
    }
    let signing_input = match net_supersession_signing_input(ann) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::AnnouncementSchemaMalformed(e.to_string()),
    };
    let ok = verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &signing_input,
        &ann.announcer_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return AnnouncementOutcome::AnnouncerSignatureFailed;
    }
    let (sup, _bytes): (WorkAssignmentSupersession, _) =
        match snip_fetch_body(adapter, &ann.work_assignment_supersession_snip_root) {
            Ok(x) => x,
            Err((tag, msg)) => match tag {
                "body-parse" => return AnnouncementOutcome::BodyParseFailed(msg),
                _ => return AnnouncementOutcome::SnipFetchFailed(format!("{tag}: {msg}")),
            },
        };
    if let Err(e) = sup.validate_schema() {
        return AnnouncementOutcome::BodySchemaInvalid(e.to_string());
    }
    // Body coordinator signature verifies against the body's
    // self-declared `coordinator_pubkey_hex`. The session-pubkey
    // pin is performed below as a drift check if a session was
    // supplied.
    let derived = match supersession_id_hex(&sup) {
        Ok(s) => s,
        Err(e) => return AnnouncementOutcome::BodySchemaInvalid(e.to_string()),
    };
    if derived != sup.supersession_id {
        return AnnouncementOutcome::BodySchemaInvalid(format!(
            "supersession_id mismatch: stored={}, derived={derived}",
            sup.supersession_id
        ));
    }
    let body_sig_input = match work_assignment_supersession_signing_input(&sup) {
        Ok(b) => b,
        Err(e) => return AnnouncementOutcome::BodySchemaInvalid(e.to_string()),
    };
    let body_sig_ok = verify_signature_hex(
        &sup.coordinator_pubkey_hex,
        &body_sig_input,
        &sup.coordinator_signature_hex,
    )
    .unwrap_or(false);
    if !body_sig_ok {
        return AnnouncementOutcome::BodySignatureFailed;
    }
    if ann.session_id != sup.session_id {
        return AnnouncementOutcome::DriftMismatch { field: "session_id" };
    }
    if ann.supersession_id != sup.supersession_id {
        return AnnouncementOutcome::DriftMismatch { field: "supersession_id" };
    }
    // Optional session pinning: when a session is supplied, the
    // body must name the same coordinator. This is the
    // moral-equivalent of `process_work_assigned_announcement`'s
    // `session_coord_pubkey_hex` path.
    if let Some(s) = session {
        if sup.session_id != s.session_id {
            return AnnouncementOutcome::DriftMismatch { field: "session_id" };
        }
        if sup.coordinator_pubkey_hex != s.coordinator_pubkey_hex {
            return AnnouncementOutcome::DriftMismatch {
                field: "coordinator_pubkey_hex",
            };
        }
        if let Some(asns) = assignments {
            // Full reference resolution + re-runs the same checks
            // above. We've already gated on schema + sig +
            // supersession_id derivation; this leg adds the
            // reference-set invariants from
            // `verify_assignment_supersession`.
            match crate::supersession_verify::verify_assignment_supersession(s, asns, &sup) {
                crate::session_verify::SessionVerifyOutcome::Ok => {}
                crate::session_verify::SessionVerifyOutcome::SupersessionSchemaMalformed(m) => {
                    return AnnouncementOutcome::BodySchemaInvalid(m);
                }
                crate::session_verify::SessionVerifyOutcome::SupersessionCoordinatorSignatureFailed => {
                    return AnnouncementOutcome::BodySignatureFailed;
                }
                other => {
                    return AnnouncementOutcome::BodySchemaInvalid(format!("{other:?}"));
                }
            }
        }
    }
    AnnouncementOutcome::Verified { body: sup }
}
