//! Stage 12.11 — `WorkAssignmentSupersession` verifier.
//!
//! Pure helper: bytes in, typed `SessionVerifyOutcome` out. The
//! aggregate-level verifier in `session_verify.rs` calls this once
//! per supersession before computing the union of superseded
//! assignment IDs.

use crate::canonical::{
    supersession_id_hex, work_assignment_supersession_signing_input,
};
use crate::session::{ExecutionSession, WorkAssignment};
use crate::session_verify::SessionVerifyOutcome;
use crate::signing::verify_signature_hex;
use crate::supersession::WorkAssignmentSupersession;

/// Standalone verifier for a `WorkAssignmentSupersession`. Verifies:
///
/// 1. Body schema (incl. v1's non-empty replacement-only invariant
///    and disjointness).
/// 2. `supersession.session_id == session.session_id`.
/// 3. `supersession.coordinator_pubkey_hex == session.coordinator_pubkey_hex`.
/// 4. Stored `supersession_id` equals the recomputed canonical hash.
/// 5. `coordinator_signature_hex` verifies over the canonical
///    signing input.
/// 6. Every entry in `superseded_assignment_ids` resolves to an
///    `assignment_id` in the supplied `assignments` slice.
/// 7. Every entry in `replacement_assignment_ids` resolves likewise.
///
/// Cross-supersession invariants (no double-supersession, chain
/// permitted) are checked at aggregate-verifier time in
/// `verify_aggregated_result_with_supersessions`.
pub fn verify_assignment_supersession(
    session: &ExecutionSession,
    assignments: &[WorkAssignment],
    supersession: &WorkAssignmentSupersession,
) -> SessionVerifyOutcome {
    // 1. Schema.
    if let Err(e) = supersession.validate_schema() {
        return SessionVerifyOutcome::SupersessionSchemaMalformed(e.to_string());
    }
    // 2. Session binding.
    if supersession.session_id != session.session_id {
        return SessionVerifyOutcome::SupersessionSessionMismatch;
    }
    // 3. Coordinator pubkey binding.
    if supersession.coordinator_pubkey_hex != session.coordinator_pubkey_hex {
        return SessionVerifyOutcome::SupersessionCoordinatorMismatch;
    }
    // 4. supersession_id derivation.
    let derived = match supersession_id_hex(supersession) {
        Ok(h) => h,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    if derived != supersession.supersession_id {
        return SessionVerifyOutcome::SupersessionIdMismatch {
            stored: supersession.supersession_id.clone(),
            derived,
        };
    }
    // 5. Coordinator signature.
    let signing_input = match work_assignment_supersession_signing_input(supersession) {
        Ok(b) => b,
        Err(e) => return SessionVerifyOutcome::InternalError(e.to_string()),
    };
    let sig_ok = verify_signature_hex(
        &supersession.coordinator_pubkey_hex,
        &signing_input,
        &supersession.coordinator_signature_hex,
    )
    .unwrap_or(false);
    if !sig_ok {
        return SessionVerifyOutcome::SupersessionCoordinatorSignatureFailed;
    }
    // 6 + 7. Reference resolution. Build a HashSet of valid
    // assignment_ids once for O(1) lookup.
    let known: std::collections::HashSet<&str> = assignments
        .iter()
        .map(|a| a.assignment_id.as_str())
        .collect();
    for id in &supersession.superseded_assignment_ids {
        if !known.contains(id.as_str()) {
            return SessionVerifyOutcome::SupersessionReferenceUnknown {
                kind: "superseded",
                assignment_id: id.clone(),
            };
        }
    }
    for id in &supersession.replacement_assignment_ids {
        if !known.contains(id.as_str()) {
            return SessionVerifyOutcome::SupersessionReferenceUnknown {
                kind: "replacement",
                assignment_id: id.clone(),
            };
        }
    }
    SessionVerifyOutcome::Ok
}
