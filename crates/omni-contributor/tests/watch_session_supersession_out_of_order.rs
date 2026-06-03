//! Stage 12.13 — out-of-order supersession handling.
//!
//! The watcher's `handle_assignment_supersession` (Bucket 2) uses
//! a two-pass strategy:
//!
//!   1. `process_assignment_supersession_announcement` with
//!      `Some(session)`, `None` for assignments. This runs
//!      announcer-sig + body schema + body sig + drift + session
//!      pinning. The body is returned even though reference
//!      resolution is deferred.
//!   2. If the in-memory assignments cache covers every referenced
//!      id, the watcher runs `verify_assignment_supersession`
//!      directly against the slice to complete reference resolution.
//!      If the cache is missing entries, the supersession is
//!      accepted on the first-pass checks and the operator sees
//!      `event=supersession_partial_verify`.
//!
//! These tests exercise both legs end-to-end against the public
//! library helpers the watcher calls, so a future refactor that
//! breaks one of the legs (or misorders the two passes) trips
//! immediately. No mesh / no SNIP — the helpers are the unit
//! boundary.

mod common;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{
        assignment_id_hex, execution_session_signing_input, net_supersession_signing_input,
        session_id_hex, supersession_id_hex, work_assignment_signing_input,
        work_assignment_supersession_signing_input,
    },
    process_assignment_supersession_announcement,
    result::WorkUnitKind,
    session::{ExecutionSession, WorkAssignment, WorkKind},
    supersession::{SupersessionReason, WorkAssignmentSupersession},
    verify_assignment_supersession, AnnouncementOutcome, ContributorSigner,
    CoordinatorSigner, NetworkWorkAssignmentSupersessionAnnouncement,
    SessionVerifyOutcome, NET_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
    SUPERSESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.13-watch-coord-seed-32!!";
const CONTRIB_SEED: [u8; 32] = *b"stage12.13-watch-contrib-32-byte";

fn signed_session(coord: &CoordinatorSigner) -> ExecutionSession {
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-06-02T00:00:00Z".into(),
        expires_at_utc: "2026-06-03T00:00:00Z".into(),
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
        contributor_pubkey_hex: contributor_pubkey_hex.into(),
        work_kind: WorkKind::Layers {
            start: stage_index * 8,
            end: stage_index * 8 + 8,
        },
        expected_work_units: 8,
        expected_work_unit_kind: WorkUnitKind::Layers,
        assigned_at_utc: "2026-06-02T00:00:05Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

fn signed_supersession(
    session: &ExecutionSession,
    coord: &CoordinatorSigner,
    superseded: Vec<String>,
    replacement: Vec<String>,
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
        reason: SupersessionReason::MissingPartial,
        created_at_utc: "2026-06-02T00:00:10Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    s.supersession_id = supersession_id_hex(&s).unwrap();
    let si = work_assignment_supersession_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn signed_announcement(
    snip: &MockSnipStore,
    coord: &CoordinatorSigner,
    sup: &WorkAssignmentSupersession,
) -> NetworkWorkAssignmentSupersessionAnnouncement {
    let root = snip.insert_bytes(&serde_json::to_vec(sup).unwrap());
    let mut ann = NetworkWorkAssignmentSupersessionAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        work_assignment_supersession_snip_root: root.to_hex(),
        session_id: sup.session_id.clone(),
        supersession_id: sup.supersession_id.clone(),
        announced_at_utc: "2026-06-02T00:00:11Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_supersession_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    ann
}

// ── Best-effort accept: cache misses one reference ─────────────

#[test]
fn supersession_with_missing_reference_accepts_via_best_effort_first_pass() {
    // Out-of-order arrival: the supersession references both an
    // old (cached) assignment AND a replacement that the watcher
    // does not yet have locally. The first pass with
    // `assignments = None` must still accept the supersession.
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_old = signed_assignment(&session, &coord, &contrib.pubkey_hex(), 0);
    let asn_new = signed_assignment(&session, &coord, &contrib.pubkey_hex(), 1);
    let sup = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id.clone()],
        vec![asn_new.assignment_id.clone()],
    );
    let ann = signed_announcement(&snip, &coord, &sup);
    // Best-effort first pass: assignments slice is None.
    let outcome = process_assignment_supersession_announcement(
        &ann,
        &snip,
        Some(&session),
        /* assignments = */ None,
    );
    assert!(
        outcome.is_verified(),
        "best-effort first pass must accept on announcer-sig + \
         body-sig + drift + session pinning; got {outcome:?}"
    );
    // Watcher then attempts a second-pass full reference
    // resolution against the in-memory cache. With only asn_old
    // cached, `verify_assignment_supersession` would refuse
    // because asn_new isn't in the slice — the watcher detects
    // this missing-reference case BEFORE the second pass and
    // emits `event=supersession_partial_verify`. Pinned by the
    // standalone verifier returning the right outcome:
    let outcome = verify_assignment_supersession(&session, &[asn_old], &sup);
    match outcome {
        SessionVerifyOutcome::SupersessionReferenceUnknown {
            kind,
            assignment_id,
        } => {
            assert_eq!(kind, "replacement");
            assert_eq!(assignment_id, sup.replacement_assignment_ids[0]);
        }
        other => panic!("expected SupersessionReferenceUnknown, got {other:?}"),
    }
}

// ── Full verification path: cache covers all references ────────

#[test]
fn supersession_with_complete_assignment_slice_runs_full_verification() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_old = signed_assignment(&session, &coord, &contrib.pubkey_hex(), 0);
    let asn_new = signed_assignment(&session, &coord, &contrib.pubkey_hex(), 1);
    let sup = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id.clone()],
        vec![asn_new.assignment_id.clone()],
    );
    let ann = signed_announcement(&snip, &coord, &sup);

    // The watcher's second-pass full verification: pass the
    // complete in-memory assignment slice. The processor's
    // built-in reference-resolution leg (the
    // `Some(session) + Some(assignments)` arm) MUST succeed.
    let outcome = process_assignment_supersession_announcement(
        &ann,
        &snip,
        Some(&session),
        Some(&[asn_old.clone(), asn_new.clone()]),
    );
    assert!(
        outcome.is_verified(),
        "full verification with complete slice must accept; got {outcome:?}"
    );
    // Equivalent direct call to `verify_assignment_supersession`
    // returns Ok — pinning the watcher's second-pass contract.
    assert!(matches!(
        verify_assignment_supersession(&session, &[asn_old, asn_new], &sup),
        SessionVerifyOutcome::Ok
    ));
}

// ── Best-effort accept: no session cached ──────────────────────

#[test]
fn supersession_with_no_session_pinning_accepts_via_best_effort() {
    // The watcher restart preload could miss a session (e.g. a
    // brand-new session whose `session_opened` announcement has
    // not been received yet). The processor still accepts on
    // announcer sig + body schema + body sig + drift, but skips
    // both session pinning and reference resolution. Status-time
    // re-verify will catch any mismatch.
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_old = signed_assignment(&session, &coord, &contrib.pubkey_hex(), 0);
    let asn_new = signed_assignment(&session, &coord, &contrib.pubkey_hex(), 1);
    let sup = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id],
        vec![asn_new.assignment_id],
    );
    let ann = signed_announcement(&snip, &coord, &sup);
    let outcome = process_assignment_supersession_announcement(
        &ann, &snip, /* session = */ None, None,
    );
    assert!(
        outcome.is_verified(),
        "no-session best-effort must still accept; got {outcome:?}"
    );
}

// ── Forged supersession rejected at body-sig leg ───────────────

#[test]
fn supersession_with_forged_body_signature_rejected_even_in_best_effort_pass() {
    // The best-effort first pass MUST NOT loosen the body
    // signature check. A supersession whose body signature is
    // from a rogue key (but whose `coordinator_pubkey_hex` field
    // advertises the real coord) is rejected, not accepted with
    // partial-verify.
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(&[0xCD; 32]).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_old = signed_assignment(&session, &coord, &contrib.pubkey_hex(), 0);
    let asn_new = signed_assignment(&session, &coord, &contrib.pubkey_hex(), 1);
    // Build a valid supersession by the real coord, then
    // replace the signature with the rogue's. The body's
    // `coordinator_pubkey_hex` is still the real coord, so the
    // body-sig leg fails.
    let mut sup = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id],
        vec![asn_new.assignment_id],
    );
    let si = work_assignment_supersession_signing_input(&sup).unwrap();
    sup.coordinator_signature_hex = rogue.sign_hex(&si);
    let ann = signed_announcement(&snip, &coord, &sup);
    let outcome = process_assignment_supersession_announcement(
        &ann, &snip, Some(&session), None,
    );
    assert!(
        matches!(outcome, AnnouncementOutcome::BodySignatureFailed),
        "best-effort path must NOT relax body-sig; got {outcome:?}"
    );
}
