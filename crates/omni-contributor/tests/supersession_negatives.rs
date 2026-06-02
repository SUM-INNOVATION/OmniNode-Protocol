//! Stage 12.11 — `verify_assignment_supersession` negative cases.
//!
//! Each test drives one specific rejection path of the supersession
//! verifier (schema, session binding, coordinator binding, ID
//! derivation, signature, reference resolution) so a future
//! refactor that quietly weakens one of those checks trips
//! immediately. Includes the v1-replacement-only enforcement
//! mandated by the Stage 12.11 review (empty replacement list →
//! schema-malformed; no "operator abandons stage" happy path).

use omni_contributor::{
    canonical::{
        assignment_id_hex, execution_session_signing_input, hex_lower, session_id_hex,
        supersession_id_hex, work_assignment_signing_input,
    },
    result::WorkUnitKind,
    session::{WorkAssignment, WorkKind},
    supersession::{SupersessionReason, WorkAssignmentSupersession},
    verify_assignment_supersession, CoordinatorSigner, ExecutionSession,
    SessionVerifyOutcome, SESSION_SCHEMA_VERSION, SUPERSESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.11-supersession-coord-32";
const COORD_WRONG_SEED: [u8; 32] = *b"stage12.11-supersession-wrong-32";

fn signed_session(coord: &CoordinatorSigner) -> ExecutionSession {
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-06-01T00:00:00Z".into(),
        expires_at_utc: "2026-06-02T00:00:00Z".into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn signed_assignment(
    session: &ExecutionSession,
    coord: &CoordinatorSigner,
    contributor_pubkey_hex: &str,
    stage_index: u32,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index,
        contributor_pubkey_hex: contributor_pubkey_hex.to_string(),
        work_kind: WorkKind::Layers {
            start: stage_index * 8,
            end: stage_index * 8 + 8,
        },
        expected_work_units: 8,
        expected_work_unit_kind: WorkUnitKind::Layers,
        assigned_at_utc: "2026-06-01T00:00:02Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

/// Build a fresh, fully-valid supersession over `superseded`
/// IDs and `replacement` IDs (both sorted ascending). Returns it
/// signed.
fn signed_supersession(
    session: &ExecutionSession,
    coord: &CoordinatorSigner,
    superseded: Vec<String>,
    replacement: Vec<String>,
    reason: SupersessionReason,
) -> WorkAssignmentSupersession {
    let mut superseded = superseded;
    superseded.sort();
    let mut replacement = replacement;
    replacement.sort();
    let mut s = WorkAssignmentSupersession {
        schema_version: SUPERSESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        supersession_id: String::new(),
        superseded_assignment_ids: superseded,
        replacement_assignment_ids: replacement,
        reason,
        created_at_utc: "2026-06-01T00:30:00Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    s.supersession_id = supersession_id_hex(&s).unwrap();
    let si =
        omni_contributor::canonical::work_assignment_supersession_signing_input(&s)
            .unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

// ── Happy path (positive control) ─────────────────────────────

#[test]
fn happy_path_supersession_passes() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_old = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let asn_new = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let s = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id.clone()],
        vec![asn_new.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    let outcome = verify_assignment_supersession(&session, &[asn_old, asn_new], &s);
    assert!(matches!(outcome, SessionVerifyOutcome::Ok), "{outcome:?}");
}

// ── v1 replacement-only scope (Stage 12.11 review hardening) ──

/// Empty `replacement_assignment_ids` must fail schema. v1 is
/// reassignment-only — abandonment is deferred to a later stage
/// that ships explicit partial-aggregate semantics.
#[test]
fn empty_replacement_assignment_ids_fails_schema() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_old = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id.clone()],
        vec![asn_old.assignment_id.clone()],   // build first with a placeholder
        SupersessionReason::MissingPartial,
    );
    // Force-empty the replacement list AFTER signing so we can drive
    // the verifier's schema check independently of signature.
    s.replacement_assignment_ids.clear();
    let outcome = verify_assignment_supersession(&session, &[asn_old], &s);
    assert!(
        matches!(
            outcome,
            SessionVerifyOutcome::SupersessionSchemaMalformed(ref msg)
                if msg.contains("replacement_assignment_ids")
        ),
        "{outcome:?}"
    );
}

#[test]
fn empty_superseded_assignment_ids_fails_schema() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn.assignment_id.clone()],
        vec![asn.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    s.superseded_assignment_ids.clear();
    let outcome = verify_assignment_supersession(&session, &[asn], &s);
    assert!(
        matches!(
            outcome,
            SessionVerifyOutcome::SupersessionSchemaMalformed(ref msg)
                if msg.contains("superseded_assignment_ids")
        ),
        "{outcome:?}"
    );
}

// ── Schema invariants ──────────────────────────────────────────

#[test]
fn duplicate_id_in_superseded_list_fails_schema() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_a = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let asn_b = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_b.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    s.superseded_assignment_ids =
        vec![asn_a.assignment_id.clone(), asn_a.assignment_id.clone()];
    let outcome = verify_assignment_supersession(&session, &[asn_a, asn_b], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionSchemaMalformed(_)
    ));
}

#[test]
fn id_appears_in_both_lists_fails_schema() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn.assignment_id.clone()],
        vec![asn.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    // Both lists explicitly contain the same id (disjoint check).
    s.superseded_assignment_ids = vec![asn.assignment_id.clone()];
    s.replacement_assignment_ids = vec![asn.assignment_id.clone()];
    let outcome = verify_assignment_supersession(&session, &[asn], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionSchemaMalformed(_)
    ));
}

#[test]
fn ids_not_sorted_ascending_fails_schema() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_a = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let asn_b = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let asn_c = signed_assignment(&session, &coord, &"03".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone(), asn_b.assignment_id.clone()],
        vec![asn_c.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    // Sort the two superseded entries in DESCENDING order.
    let mut entries = s.superseded_assignment_ids.clone();
    entries.sort_by(|x, y| y.cmp(x));
    s.superseded_assignment_ids = entries;
    let outcome =
        verify_assignment_supersession(&session, &[asn_a, asn_b, asn_c], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionSchemaMalformed(_)
    ));
}

#[test]
fn custom_reason_with_empty_label_fails_schema() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_a = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let asn_b = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let s = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_b.assignment_id.clone()],
        SupersessionReason::Custom {
            label: String::new(),
        },
    );
    let outcome = verify_assignment_supersession(&session, &[asn_a, asn_b], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionSchemaMalformed(_)
    ));
}

// ── Binding checks ────────────────────────────────────────────

#[test]
fn session_id_mismatch_is_rejected() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_a = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let asn_b = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_b.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    s.session_id = "ff".repeat(32);
    let outcome = verify_assignment_supersession(&session, &[asn_a, asn_b], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionSessionMismatch
    ));
}

#[test]
fn wrong_coordinator_pubkey_is_rejected() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let wrong = CoordinatorSigner::from_seed_bytes(&COORD_WRONG_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_a = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let asn_b = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_b.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    s.coordinator_pubkey_hex = wrong.pubkey_hex();
    let outcome = verify_assignment_supersession(&session, &[asn_a, asn_b], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionCoordinatorMismatch
    ));
}

#[test]
fn tampered_supersession_id_is_rejected() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_a = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let asn_b = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_b.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    s.supersession_id = "ff".repeat(32);
    let outcome = verify_assignment_supersession(&session, &[asn_a, asn_b], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionIdMismatch { .. }
    ));
}

#[test]
fn tampered_coordinator_signature_is_rejected() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_a = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let asn_b = signed_assignment(&session, &coord, &"02".repeat(32), 0);
    let mut s = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_b.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    // Flip the last hex digit of the sig.
    let mut sig: Vec<char> = s.coordinator_signature_hex.chars().collect();
    let last = sig.len() - 1;
    sig[last] = if sig[last] == '0' { '1' } else { '0' };
    s.coordinator_signature_hex = sig.into_iter().collect();
    let outcome = verify_assignment_supersession(&session, &[asn_a, asn_b], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionCoordinatorSignatureFailed
    ));
}

// ── Reference resolution ───────────────────────────────────────

#[test]
fn unknown_superseded_id_is_rejected() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let phantom = hex_lower(blake3::hash(b"phantom").as_bytes());
    let s = signed_supersession(
        &session,
        &coord,
        vec![phantom.clone()],
        vec![asn.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    let outcome = verify_assignment_supersession(&session, &[asn], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionReferenceUnknown {
            kind: "superseded", ..
        }
    ));
}

#[test]
fn unknown_replacement_id_is_rejected() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord);
    let asn = signed_assignment(&session, &coord, &"01".repeat(32), 0);
    let phantom = hex_lower(blake3::hash(b"phantom").as_bytes());
    let s = signed_supersession(
        &session,
        &coord,
        vec![asn.assignment_id.clone()],
        vec![phantom.clone()],
        SupersessionReason::MissingPartial,
    );
    let outcome = verify_assignment_supersession(&session, &[asn], &s);
    assert!(matches!(
        outcome,
        SessionVerifyOutcome::SupersessionReferenceUnknown {
            kind: "replacement", ..
        }
    ));
}
