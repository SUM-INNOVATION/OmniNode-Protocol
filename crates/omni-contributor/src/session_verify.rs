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
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

impl SessionVerifyOutcome {
    pub fn is_ok(&self) -> bool {
        matches!(self, SessionVerifyOutcome::Ok)
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
    // 4. Partials, each bound to its assignment.
    for p in partials {
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

    // Every assignment must have exactly one ref; every ref must
    // resolve to a partial we have; drift fields must match.
    let mut seen_for: HashSet<&str> = HashSet::new();
    for r in &aggregate.partial_refs {
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
    // Now check coverage: every assignment must be referenced exactly once.
    for a in assignments {
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
