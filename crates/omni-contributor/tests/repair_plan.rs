//! Stage 12.10 — unit + integration tests for the local
//! pooled-session repair planner / applier.
//!
//! The planner consumes a Stage 12.9 `SessionStatusReport` and
//! produces a `SessionRepairPlan`. These tests:
//!
//!   1. Drive the planner directly with synthetic `SessionStatusReport`
//!      values to pin every "refusal" path (NoSession, NoAssignments,
//!      CompletePartials, Aggregated, InvalidState, ExpiredIncomplete)
//!      and the single accepted path (InProgress).
//!   2. Drive the planner end-to-end against a real
//!      `ContributorStateStore` populated via the Stage 12.7 writers,
//!      ensuring `build_session_status_report` + planner compose
//!      correctly and that `source_status_hash` binds to the
//!      operator-meaningful projection.
//!   3. Exercise the applier-side guarantees the CLI depends on:
//!      deterministic action ordering, plan-hash stability across
//!      serde round-trips, and the source-status drift signal.

use omni_contributor::{
    build_session_repair_plan, build_session_status_report,
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, partial_result_signing_input,
        session_id_hex, work_assignment_signing_input,
    },
    repair_plan_hash_hex,
    repair::{RepairAction, RepairStrategy, SessionRepairPlan, REPAIR_PLAN_SCHEMA_VERSION},
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    session::{
        AggregatedContributorResult, AggregatedPartialRef, PartialContributorResult,
        WorkAssignment, WorkKind,
    },
    source_status_hash_hex,
    status::{AssignmentStatus, SessionOverallStatus, SessionStatusReport, STATUS_SCHEMA_VERSION},
    ContributorJoin, ContributorSigner, ContributorStateStore, CoordinatorSigner,
    ExecutionSession, RepairError, StateObjectKind, SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.10-repair-coord-seed-32!";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.10-repair-contrib-a-32-b";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.10-repair-contrib-b-32-b";
const NOW_UTC: &str = "2026-06-01T00:30:00Z";
const FAR_FUTURE: &str = "2026-06-02T00:00:00Z";

// ── Synthetic-report helpers ──────────────────────────────────────

fn empty_report(
    overall: SessionOverallStatus,
    session_id: &str,
) -> SessionStatusReport {
    SessionStatusReport {
        schema_version: STATUS_SCHEMA_VERSION,
        session_id: session_id.to_string(),
        posted_id: None,
        model_hash: None,
        generated_at_utc: NOW_UTC.into(),
        session_expires_at_utc: None,
        session_expired: matches!(overall, SessionOverallStatus::ExpiredIncomplete),
        join_count: 0,
        peer_advert_count: 0,
        assignment_count: 0,
        partial_count: 0,
        active_assignment_count: 0,
        superseded_assignment_count: 0,
        supersession_count: 0,
        missing_assignment_ids: Vec::new(),
        duplicate_partial_assignment_ids: Vec::new(),
        aggregate_present: matches!(overall, SessionOverallStatus::Aggregated),
        aggregate_valid: matches!(overall, SessionOverallStatus::Aggregated),
        overall_status: overall,
        assignments: Vec::new(),
        supersessions: Vec::new(),
        notes: Vec::new(),
    }
}

fn in_progress_with(assignments: Vec<AssignmentStatus>) -> SessionStatusReport {
    let missing: Vec<String> = assignments
        .iter()
        .filter(|a| !a.partial_present)
        .map(|a| a.assignment_id.clone())
        .collect();
    let count = assignments.len() as u32;
    SessionStatusReport {
        schema_version: STATUS_SCHEMA_VERSION,
        session_id: "11".repeat(32),
        posted_id: Some("22".repeat(32)),
        model_hash: Some("33".repeat(32)),
        generated_at_utc: NOW_UTC.into(),
        session_expires_at_utc: Some(FAR_FUTURE.into()),
        session_expired: false,
        join_count: count,
        peer_advert_count: 0,
        assignment_count: count,
        partial_count: assignments.iter().filter(|a| a.partial_present).count() as u32,
        active_assignment_count: count,
        superseded_assignment_count: 0,
        supersession_count: 0,
        missing_assignment_ids: missing,
        duplicate_partial_assignment_ids: Vec::new(),
        aggregate_present: false,
        aggregate_valid: false,
        overall_status: SessionOverallStatus::InProgress,
        assignments,
        supersessions: Vec::new(),
        notes: Vec::new(),
    }
}

fn assignment_status(
    assignment_id: &str,
    stage_index: u32,
    contributor: &str,
    partial_present: bool,
) -> AssignmentStatus {
    AssignmentStatus {
        assignment_id: assignment_id.to_string(),
        stage_index,
        contributor_pubkey_hex: contributor.to_string(),
        work_kind: WorkKind::Layers { start: 0, end: 8 },
        expected_work_units: 8,
        expected_work_unit_kind: WorkUnitKind::Layers,
        join_present: true,
        peer_advert_present: false,
        partial_present,
        partial_valid: partial_present,
        partial_snip_root: None,
        superseded: false,
        superseded_by_supersession_id: None,
        notes: Vec::new(),
    }
}

// ── 1. Refusal paths ──────────────────────────────────────────────

#[test]
fn planner_refuses_no_session() {
    let r = empty_report(SessionOverallStatus::NoSession, &"de".repeat(32));
    let err = build_session_repair_plan(
        &r,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RepairError::SessionNotPresent { .. }));
}

#[test]
fn planner_refuses_no_assignments() {
    let r = empty_report(SessionOverallStatus::NoAssignments, &"de".repeat(32));
    let err = build_session_repair_plan(
        &r,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RepairError::NothingToRepair { .. }));
}

#[test]
fn planner_refuses_complete_partials() {
    // CompletePartials with at least one assignment that has its
    // partial.
    let r = SessionStatusReport {
        overall_status: SessionOverallStatus::CompletePartials,
        ..in_progress_with(vec![assignment_status(
            &"aa".repeat(32),
            0,
            &"01".repeat(32),
            true,
        )])
    };
    let err = build_session_repair_plan(
        &r,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RepairError::NothingToRepair { .. }));
}

#[test]
fn planner_refuses_aggregated() {
    let r = empty_report(SessionOverallStatus::Aggregated, &"de".repeat(32));
    let err = build_session_repair_plan(
        &r,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RepairError::NothingToRepair { .. }));
}

#[test]
fn planner_refuses_invalid_state() {
    let r = empty_report(SessionOverallStatus::InvalidState, &"de".repeat(32));
    let err = build_session_repair_plan(
        &r,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RepairError::InvalidState));
}

#[test]
fn planner_refuses_expired_incomplete() {
    let r = empty_report(SessionOverallStatus::ExpiredIncomplete, &"de".repeat(32));
    let err = build_session_repair_plan(
        &r,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, RepairError::SessionExpired));
}

// ── 2. Accepted path ──────────────────────────────────────────────

#[test]
fn planner_emits_one_reannounce_per_missing_partial() {
    let r = in_progress_with(vec![
        assignment_status(&"aa".repeat(32), 0, &"01".repeat(32), false),
        assignment_status(&"bb".repeat(32), 1, &"02".repeat(32), true),
        assignment_status(&"cc".repeat(32), 2, &"03".repeat(32), false),
    ]);
    let plan = build_session_repair_plan(
        &r,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 2);
    // Stage 0 first, then stage 2 — the partial-present stage 1 is
    // skipped.
    match &plan.actions[0] {
        RepairAction::ReannounceAssignment {
            assignment_id,
            stage_index,
            ..
        } => {
            assert_eq!(assignment_id, &"aa".repeat(32));
            assert_eq!(*stage_index, 0);
        }
        other => panic!("expected ReannounceAssignment, got {other:?}"),
    }
    match &plan.actions[1] {
        RepairAction::ReannounceAssignment {
            assignment_id,
            stage_index,
            ..
        } => {
            assert_eq!(assignment_id, &"cc".repeat(32));
            assert_eq!(*stage_index, 2);
        }
        other => panic!("expected ReannounceAssignment, got {other:?}"),
    }
    assert_eq!(plan.schema_version, REPAIR_PLAN_SCHEMA_VERSION);
    assert_eq!(plan.strategy, RepairStrategy::ReannounceMissing);
}

// ── 3. Determinism ───────────────────────────────────────────────

#[test]
fn planner_deterministic_under_input_shuffle() {
    let a = assignment_status(&"cc".repeat(32), 2, &"03".repeat(32), false);
    let b = assignment_status(&"aa".repeat(32), 0, &"01".repeat(32), false);
    let c = assignment_status(&"bb".repeat(32), 1, &"02".repeat(32), false);
    let r1 = in_progress_with(vec![a.clone(), b.clone(), c.clone()]);
    let r2 = in_progress_with(vec![c, b, a]);
    let p1 = build_session_repair_plan(
        &r1,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap();
    let p2 = build_session_repair_plan(
        &r2,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap();
    assert_eq!(p1.actions, p2.actions);
    assert_eq!(p1.repair_plan_hash, p2.repair_plan_hash);
    assert_eq!(p1.source_status_hash, p2.source_status_hash);
}

// ── 4. repair_plan_hash serde round-trip ────────────────────────

#[test]
fn repair_plan_hash_round_trips_through_json() {
    let r = in_progress_with(vec![assignment_status(
        &"aa".repeat(32),
        0,
        &"01".repeat(32),
        false,
    )]);
    let plan = build_session_repair_plan(
        &r,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        Some(&"05".repeat(32)),
    )
    .unwrap();
    let bytes = serde_json::to_vec_pretty(&plan).unwrap();
    let back: SessionRepairPlan = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back, plan);
    assert_eq!(back.repair_plan_hash, repair_plan_hash_hex(&back));
}

// ── 5. source_status_hash projection: stable across noise ──────

/// A status report regenerated with a different `generated_at_utc`
/// but the same `(session_id, [assignment_id, partial_present])`
/// shape produces the same `source_status_hash`.
#[test]
fn source_status_hash_ignores_generated_at_utc_and_notes() {
    let assignments = vec![
        assignment_status(&"aa".repeat(32), 0, &"01".repeat(32), false),
        assignment_status(&"bb".repeat(32), 1, &"02".repeat(32), true),
    ];
    let r1 = SessionStatusReport {
        generated_at_utc: "2026-06-01T00:00:00Z".into(),
        notes: vec!["something innocent".into()],
        ..in_progress_with(assignments.clone())
    };
    let r2 = SessionStatusReport {
        generated_at_utc: "2026-06-01T12:00:00Z".into(),
        notes: vec![],
        ..in_progress_with(assignments)
    };
    assert_eq!(source_status_hash_hex(&r1), source_status_hash_hex(&r2));
}

/// But it CHANGES when a partial_present flag flips.
#[test]
fn source_status_hash_changes_when_partial_present_flips() {
    let r1 = in_progress_with(vec![assignment_status(
        &"aa".repeat(32),
        0,
        &"01".repeat(32),
        false,
    )]);
    let mut r2 = r1.clone();
    r2.assignments[0].partial_present = true;
    r2.assignments[0].partial_valid = true;
    assert_ne!(source_status_hash_hex(&r1), source_status_hash_hex(&r2));
}

// ── 6. End-to-end: state-dir → status report → plan ────────────

fn fresh_dir() -> tempfile::TempDir {
    tempfile::TempDir::new().expect("tempdir")
}

fn fresh_store() -> (tempfile::TempDir, ContributorStateStore) {
    let d = fresh_dir();
    let (s, _) = ContributorStateStore::open(d.path(), false, NOW_UTC).unwrap();
    (d, s)
}

fn signed_session(coord: &CoordinatorSigner, expires_at_utc: &str) -> ExecutionSession {
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-06-01T00:00:00Z".into(),
        expires_at_utc: expires_at_utc.into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn signed_join(session: &ExecutionSession, contrib: &ContributorSigner) -> ContributorJoin {
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 1 << 30,
        max_input_tokens: 1024,
        max_output_tokens: 1024,
        supported_work_unit_kinds: vec![WorkUnitKind::Layers],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-06-01T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
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

fn signed_partial(
    assignment: &WorkAssignment,
    contrib: &ContributorSigner,
) -> PartialContributorResult {
    let mut p = PartialContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: assignment.session_id.clone(),
        assignment_id: assignment.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        partial_artifact_snip_root: format!("0x{}", "aa".repeat(32)),
        partial_artifact_hash: hex_lower(blake3::hash(b"partial-bytes").as_bytes()),
        measured_accounting: MeasuredAccounting {
            tokenizer_hash: "44".repeat(32),
            input_token_count: 100,
            output_token_count: 0,
            total_base_units: 100,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: contrib.pubkey_hex(),
                stage_label: "stub".into(),
                work_unit_kind: assignment.expected_work_unit_kind,
                work_units: assignment.expected_work_units,
            }],
        },
        produced_at_utc: format!("2026-06-01T00:00:1{}Z", assignment.stage_index),
        contributor_signature_hex: String::new(),
    };
    let si = partial_result_signing_input(&p).unwrap();
    p.contributor_signature_hex = contrib.sign_hex(&si);
    p
}

#[test]
fn end_to_end_state_dir_status_then_plan() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    for j in [signed_join(&session, &contrib_a), signed_join(&session, &contrib_b)] {
        store
            .write_verified_json(
                StateObjectKind::Join {
                    session_id: session.session_id.clone(),
                },
                &j.contributor_pubkey_hex,
                &j,
            )
            .unwrap();
    }
    let asn_0 = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0);
    let asn_1 = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 1);
    for a in &[asn_0.clone(), asn_1.clone()] {
        store
            .write_verified_json(
                StateObjectKind::Assignment {
                    session_id: session.session_id.clone(),
                },
                &a.assignment_id,
                a,
            )
            .unwrap();
    }
    // Only stage 0's partial arrives → InProgress with one missing.
    let p_0 = signed_partial(&asn_0, &contrib_a);
    store
        .write_verified_json(
            StateObjectKind::Partial {
                session_id: session.session_id.clone(),
            },
            &p_0.assignment_id,
            &p_0,
        )
        .unwrap();

    let status =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(status.overall_status, SessionOverallStatus::InProgress);
    let plan = build_session_repair_plan(
        &status,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        Some(&coord.pubkey_hex()),
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 1);
    match &plan.actions[0] {
        RepairAction::ReannounceAssignment {
            assignment_id,
            stage_index,
            contributor_pubkey_hex,
        } => {
            assert_eq!(assignment_id, &asn_1.assignment_id);
            assert_eq!(*stage_index, 1);
            assert_eq!(contributor_pubkey_hex, &contrib_b.pubkey_hex());
        }
        other => panic!("expected ReannounceAssignment, got {other:?}"),
    }
    // source_status_hash binds to current state-dir shape.
    assert_eq!(plan.source_status_hash, source_status_hash_hex(&status));
}

// ── 7. Drift detection ──────────────────────────────────────────

#[test]
fn source_status_hash_drifts_when_partial_arrives_after_planning() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    for j in [signed_join(&session, &contrib_a), signed_join(&session, &contrib_b)] {
        store
            .write_verified_json(
                StateObjectKind::Join {
                    session_id: session.session_id.clone(),
                },
                &j.contributor_pubkey_hex,
                &j,
            )
            .unwrap();
    }
    let asn_0 = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0);
    let asn_1 = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 1);
    for a in &[asn_0.clone(), asn_1.clone()] {
        store
            .write_verified_json(
                StateObjectKind::Assignment {
                    session_id: session.session_id.clone(),
                },
                &a.assignment_id,
                a,
            )
            .unwrap();
    }
    let status_before =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    let plan = build_session_repair_plan(
        &status_before,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        Some(&coord.pubkey_hex()),
    )
    .unwrap();

    // Simulate: a partial for asn_1 arrives between plan and apply.
    let p_1 = signed_partial(&asn_1, &contrib_b);
    store
        .write_verified_json(
            StateObjectKind::Partial {
                session_id: session.session_id.clone(),
            },
            &p_1.assignment_id,
            &p_1,
        )
        .unwrap();
    let status_after =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    // The applier's drift check: recompute projection from current
    // status and compare against the plan's stored hash.
    let recomputed = source_status_hash_hex(&status_after);
    assert_ne!(
        recomputed, plan.source_status_hash,
        "drift must be detected when a partial arrives"
    );
}

/// Stage 12.11 — same drift posture, but the change between plan
/// and apply is a **supersession** arriving (not a partial). Prior
/// to the projection lift the hash only covered
/// `(assignment_id, partial_present)`, so a supersession that
/// retires a missing assignment without a partial flipping would
/// slip past the drift check, and the applier could reannounce or
/// reassign an already-retired assignment. The v2 projection
/// includes per-assignment `(superseded, superseded_by_supersession_id)`,
/// so this test fails closed.
#[test]
fn source_status_hash_drifts_when_supersession_arrives_after_planning() {
    let assignments = vec![
        assignment_status(&"aa".repeat(32), 0, &"01".repeat(32), false),
        assignment_status(&"bb".repeat(32), 1, &"02".repeat(32), false),
    ];
    let status_before = in_progress_with(assignments.clone());
    // The pre-supersession projection is the plan's baseline.
    let baseline = source_status_hash_hex(&status_before);

    // Simulate: between plan and apply, a verified supersession
    // arrived that retires assignment "aa..." with replacement
    // "cc...". `partial_present` did NOT flip — the retired
    // assignment never had a partial. The status reporter's v2
    // semantics flag the superseded assignment AND drop it from
    // active accounting.
    let supersession_id = "ee".repeat(32);
    let mut status_after = status_before.clone();
    status_after.assignments[0].superseded = true;
    status_after.assignments[0].superseded_by_supersession_id =
        Some(supersession_id.clone());
    // Active accounting drops the retired assignment + bumps the
    // supersession counters. (Not strictly needed for the
    // projection assertion, but mirrors what the reporter would
    // produce so the test reads like the real call site.)
    status_after.active_assignment_count = 1;
    status_after.superseded_assignment_count = 1;
    status_after.supersession_count = 1;

    let recomputed = source_status_hash_hex(&status_after);
    assert_ne!(
        recomputed, baseline,
        "drift must be detected when a verified supersession arrives \
         between plan and apply (Stage 12.11 projection lift)"
    );
}

// ── 8. Apply-time eligibility re-check ──────────────────────────

/// `source_status_hash_hex`'s projection (as of Stage 12.11:
/// `(session_id, sorted [(assignment_id, partial_present,
/// superseded, superseded_by_supersession_id)] pairs)`) is
/// intentionally narrow — it covers the active-assignment cover
/// but NOT session-level health. So a status that flips from
/// `InProgress` to `ExpiredIncomplete` (e.g. clock advances past
/// `session.expires_at_utc`) leaves the projection IDENTICAL.
/// The applier must therefore re-check eligibility against the
/// rebuilt current status BEFORE any SNIP or mesh work, using
/// the same matrix as the planner. This test pins that contract
/// via the library-level `check_repair_eligible` helper that the
/// apply CLI calls.
#[test]
fn apply_eligibility_check_catches_expired_when_projection_is_unchanged() {
    use omni_contributor::check_repair_eligible;

    // Build two status reports with the SAME assignments/partial
    // shape — only the `overall_status` differs. The projection
    // must be identical AND the apply-time check must catch the
    // shift.
    let assignments = vec![assignment_status(
        &"aa".repeat(32),
        0,
        &"01".repeat(32),
        false,
    )];
    let mut original = in_progress_with(assignments.clone());
    original.session_expires_at_utc = Some(FAR_FUTURE.into());
    let plan = build_session_repair_plan(
        &original,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap();
    // Simulate: clock has advanced past expiry; the rebuilt
    // status now says ExpiredIncomplete, but the assignment list
    // is unchanged.
    let now_expired = SessionStatusReport {
        overall_status: SessionOverallStatus::ExpiredIncomplete,
        session_expired: true,
        ..in_progress_with(assignments)
    };
    // Projection identical → the existing source_status_hash drift
    // check would miss this.
    assert_eq!(
        source_status_hash_hex(&now_expired),
        plan.source_status_hash
    );
    // But the eligibility re-check catches it.
    let err = check_repair_eligible(&now_expired).unwrap_err();
    assert!(matches!(err, RepairError::SessionExpired));
}

/// Same posture as the expiry case, but for `InvalidState` — e.g.
/// an aggregate body was added between plan and apply and it
/// failed the verifier. The partial-present projection didn't
/// change, but the chain is now invalid. Apply must refuse.
#[test]
fn apply_eligibility_check_catches_invalid_state_when_projection_is_unchanged() {
    use omni_contributor::check_repair_eligible;

    let assignments = vec![assignment_status(
        &"aa".repeat(32),
        0,
        &"01".repeat(32),
        false,
    )];
    let original = in_progress_with(assignments.clone());
    let plan = build_session_repair_plan(
        &original,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        None,
    )
    .unwrap();
    let now_invalid = SessionStatusReport {
        overall_status: SessionOverallStatus::InvalidState,
        notes: vec!["aggregated.json failed verify_aggregated_result".into()],
        ..in_progress_with(assignments)
    };
    assert_eq!(
        source_status_hash_hex(&now_invalid),
        plan.source_status_hash
    );
    let err = check_repair_eligible(&now_invalid).unwrap_err();
    assert!(matches!(err, RepairError::InvalidState));
}

/// Stage 12.11 — apply-time per-action enforcement. The plan is
/// unsigned + local so an operator can hand-edit the JSON and
/// recompute `repair_plan_hash`. Without this check the applier
/// would happily retire an assignment that already has a valid
/// partial (`partial_present == true`) or is already superseded
/// by a prior verified supersession — wasting valid work — while
/// the session stayed `InProgress` due to some OTHER missing
/// assignment.
///
/// Status: A missing (`partial_present == false`), B completed
/// (`partial_present == true`). A hand-edited `ReassignMissing`
/// plan targets B instead of A; the apply-time enforcement must
/// reject with `reason = "already_completed"`. This test also
/// exercises the other rejection reasons:
/// `already_superseded`, `not_in_status`, `stage_index_mismatch`.
#[test]
fn apply_per_action_check_rejects_hand_edited_plan_targeting_completed_assignment() {
    use omni_contributor::{
        check_reassign_targets_active_missing, supersession::SupersessionReason,
        SupersessionStatus,
    };

    let asn_a_id = "aa".repeat(32);
    let asn_b_id = "bb".repeat(32);
    let asn_a = AssignmentStatus {
        partial_present: false,
        partial_valid: false,
        ..assignment_status(&asn_a_id, 0, &"01".repeat(32), false)
    };
    let asn_b = AssignmentStatus {
        partial_present: true,
        partial_valid: true,
        ..assignment_status(&asn_b_id, 1, &"02".repeat(32), true)
    };
    let status = in_progress_with(vec![asn_a.clone(), asn_b.clone()]);

    // Hand-build a reassignment plan whose ONLY action targets
    // assignment B (the COMPLETED one). The `source_status_hash`
    // and `repair_plan_hash` are recomputed so the integrity
    // checks would pass — only the per-action active-missing
    // check stands between the edited plan and a wasted publish.
    let mut hand_edited = SessionRepairPlan {
        schema_version: REPAIR_PLAN_SCHEMA_VERSION,
        session_id: status.session_id.clone(),
        source_status_hash: source_status_hash_hex(&status),
        created_at_utc: NOW_UTC.into(),
        strategy: RepairStrategy::ReassignMissing,
        actions: vec![RepairAction::ReassignAssignment {
            superseded_assignment_id: asn_b_id.clone(),
            original_stage_index: asn_b.stage_index,
            replacement_contributor_pubkey_hex: "ff".repeat(32),
            replacement_stage_index: asn_b.stage_index,
            replacement_work_kind: WorkKind::Layers { start: 0, end: 8 },
            replacement_expected_work_units: 8,
            replacement_expected_work_unit_kind: WorkUnitKind::Layers,
            reason: SupersessionReason::MissingPartial,
        }],
        coordinator_pubkey_hex: None,
        repair_plan_hash: String::new(),
    };
    hand_edited.repair_plan_hash = repair_plan_hash_hex(&hand_edited);
    let err =
        check_reassign_targets_active_missing(&hand_edited, &status).unwrap_err();
    match err {
        RepairError::ReassignTargetNotActiveMissing {
            ref assignment_id,
            reason,
            ..
        } => {
            assert_eq!(assignment_id, &asn_b_id);
            assert_eq!(reason, "already_completed");
        }
        other => panic!("expected ReassignTargetNotActiveMissing, got {other:?}"),
    }

    // `not_in_status`: same plan shape, but pointing at an
    // assignment_id the current status doesn't know.
    let unknown_id = "cc".repeat(32);
    let mut p2 = hand_edited.clone();
    if let RepairAction::ReassignAssignment {
        ref mut superseded_assignment_id,
        ..
    } = p2.actions[0]
    {
        *superseded_assignment_id = unknown_id.clone();
    }
    p2.repair_plan_hash = repair_plan_hash_hex(&p2);
    let err = check_reassign_targets_active_missing(&p2, &status).unwrap_err();
    match err {
        RepairError::ReassignTargetNotActiveMissing { reason, .. } => {
            assert_eq!(reason, "not_in_status");
        }
        other => panic!("expected ReassignTargetNotActiveMissing, got {other:?}"),
    }

    // `already_superseded`: status now shows A retired by a
    // prior verified supersession (per Stage 12.11 status v2
    // semantics).
    let supersession_id = "ee".repeat(32);
    let mut status_a_superseded = status.clone();
    status_a_superseded.assignments[0].superseded = true;
    status_a_superseded.assignments[0].superseded_by_supersession_id =
        Some(supersession_id.clone());
    status_a_superseded.supersessions.push(SupersessionStatus {
        supersession_id: supersession_id.clone(),
        superseded_assignment_ids: vec![asn_a_id.clone()],
        replacement_assignment_ids: vec!["dd".repeat(32)],
        reason: SupersessionReason::MissingPartial,
        valid: true,
        notes: vec![],
    });
    let mut p3 = hand_edited.clone();
    if let RepairAction::ReassignAssignment {
        ref mut superseded_assignment_id,
        ref mut original_stage_index,
        ..
    } = p3.actions[0]
    {
        *superseded_assignment_id = asn_a_id.clone();
        *original_stage_index = asn_a.stage_index;
    }
    p3.repair_plan_hash = repair_plan_hash_hex(&p3);
    let err =
        check_reassign_targets_active_missing(&p3, &status_a_superseded).unwrap_err();
    match err {
        RepairError::ReassignTargetNotActiveMissing { reason, .. } => {
            assert_eq!(reason, "already_superseded");
        }
        other => panic!("expected ReassignTargetNotActiveMissing, got {other:?}"),
    }

    // `stage_index_mismatch`: target A (active-missing) but the
    // plan's `original_stage_index` doesn't match the status row.
    let mut p4 = hand_edited.clone();
    if let RepairAction::ReassignAssignment {
        ref mut superseded_assignment_id,
        ref mut original_stage_index,
        ..
    } = p4.actions[0]
    {
        *superseded_assignment_id = asn_a_id.clone();
        *original_stage_index = asn_a.stage_index + 7;
    }
    p4.repair_plan_hash = repair_plan_hash_hex(&p4);
    let err = check_reassign_targets_active_missing(&p4, &status).unwrap_err();
    match err {
        RepairError::ReassignTargetNotActiveMissing { reason, .. } => {
            assert_eq!(reason, "stage_index_mismatch");
        }
        other => panic!("expected ReassignTargetNotActiveMissing, got {other:?}"),
    }

    // Positive control: a well-formed plan targeting A (the
    // active-missing one) passes.
    let mut p5 = hand_edited.clone();
    if let RepairAction::ReassignAssignment {
        ref mut superseded_assignment_id,
        ref mut original_stage_index,
        ..
    } = p5.actions[0]
    {
        *superseded_assignment_id = asn_a_id.clone();
        *original_stage_index = asn_a.stage_index;
    }
    p5.repair_plan_hash = repair_plan_hash_hex(&p5);
    assert!(check_reassign_targets_active_missing(&p5, &status).is_ok());
}

// ── 9. Apply-time library-contract tests ─────────────────────

/// `apply-session-repair` recomputes `repair_plan_hash_hex(&plan)`
/// on read and refuses on drift. This test pins the underlying
/// contract: a hand-edited plan (operator added an action that
/// wasn't in the original) trips the check at the library level.
#[test]
fn apply_path_detects_plan_hash_drift() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    for j in [signed_join(&session, &contrib_a), signed_join(&session, &contrib_b)] {
        store
            .write_verified_json(
                StateObjectKind::Join {
                    session_id: session.session_id.clone(),
                },
                &j.contributor_pubkey_hex,
                &j,
            )
            .unwrap();
    }
    let asn_0 = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0);
    let asn_1 = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 1);
    for a in &[asn_0.clone(), asn_1.clone()] {
        store
            .write_verified_json(
                StateObjectKind::Assignment {
                    session_id: session.session_id.clone(),
                },
                &a.assignment_id,
                a,
            )
            .unwrap();
    }
    let status =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    let mut plan = build_session_repair_plan(
        &status,
        RepairStrategy::ReannounceMissing,
        NOW_UTC,
        Some(&coord.pubkey_hex()),
    )
    .unwrap();

    // Tamper: drop the second action without re-stamping the hash.
    // The applier's recomputed hash will then differ from the
    // stored one.
    let stored_hash = plan.repair_plan_hash.clone();
    plan.actions.truncate(1);
    let recomputed = repair_plan_hash_hex(&plan);
    assert_ne!(recomputed, stored_hash);
}

/// `apply-session-repair` loads each referenced assignment via
/// `store.read_verified_json(StateObjectKind::Assignment {...},
/// &assignment_id)` and raises `RepairError::AssignmentNotPresent`
/// on `None`. This test pins the underlying store contract: a plan
/// whose action references an `assignment_id` not on disk returns
/// `None` (so the CLI's typed-error branch fires).
#[test]
fn apply_path_detects_missing_assignment_in_state() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session.session_id.clone(),
            },
            &contrib_a.pubkey_hex(),
            &signed_join(&session, &contrib_a),
        )
        .unwrap();
    // NOTE: NOT writing any assignment file to disk.
    let fictional_assignment_id = "be".repeat(32);
    let result: Option<WorkAssignment> = store
        .read_verified_json(
            StateObjectKind::Assignment {
                session_id: session.session_id.clone(),
            },
            &fictional_assignment_id,
        )
        .unwrap();
    assert!(
        result.is_none(),
        "missing-assignment must surface as None so the CLI's \
         RepairError::AssignmentNotPresent branch fires"
    );
}

/// `apply-session-repair` re-verifies each loaded assignment via
/// `verify_work_assignment(&session, &joined_pubkeys, &asn)`. A
/// tampered on-disk assignment.json (e.g. flipped sig hex) must
/// fail that re-verify. This test pins the contract the CLI
/// relies on.
#[test]
fn apply_path_detects_tampered_assignment_in_state() {
    use std::collections::HashSet;
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session.session_id.clone(),
            },
            &contrib_a.pubkey_hex(),
            &signed_join(&session, &contrib_a),
        )
        .unwrap();
    let mut bad_asn = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0);
    // Flip the LAST hex digit of the coordinator signature.
    let mut sig: Vec<char> = bad_asn.coordinator_signature_hex.chars().collect();
    let last = sig.len() - 1;
    sig[last] = if sig[last] == '0' { '1' } else { '0' };
    bad_asn.coordinator_signature_hex = sig.into_iter().collect();
    store
        .write_verified_json(
            StateObjectKind::Assignment {
                session_id: session.session_id.clone(),
            },
            &bad_asn.assignment_id,
            &bad_asn,
        )
        .unwrap();

    // Now exercise the applier's re-verify step exactly as the CLI
    // does: load joined-pubkey set, then run verify_work_assignment.
    let raw_joins = store.list_verified_joins_for(&session.session_id).unwrap();
    let joins: Vec<ContributorJoin> = raw_joins
        .into_iter()
        .filter(|j| {
            omni_contributor::verify_contributor_join(&session, j).is_ok()
        })
        .collect();
    let joined_pubkeys: HashSet<String> =
        joins.iter().map(|j| j.contributor_pubkey_hex.clone()).collect();
    let on_disk: WorkAssignment = store
        .read_verified_json(
            StateObjectKind::Assignment {
                session_id: session.session_id.clone(),
            },
            &bad_asn.assignment_id,
        )
        .unwrap()
        .unwrap();
    let outcome = omni_contributor::verify_work_assignment(
        &session,
        &joined_pubkeys,
        &on_disk,
    );
    assert!(
        !outcome.is_ok(),
        "tampered assignment in state must trip verify_work_assignment; \
         got {outcome:?}"
    );
}

// ── 10. Aggregate verifier sanity (the halt-finding rationale) ──

/// Pin the verifier behavior that drove the Stage 12.10 halt
/// finding: every assignment in the supplied slice must appear in
/// `aggregate.partial_refs`, so adding a replacement assignment for
/// a missing partial would break aggregate verification without a
/// supersession model. This test exists so a future Stage 12.11
/// design discussion has an explicit fixture documenting the
/// constraint.
#[test]
fn aggregate_verifier_rejects_extra_assignment_without_partial() {
    use omni_contributor::verify_aggregated_result;

    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let join_b = signed_join(&session, &contrib_b);
    let asn_0 = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0);
    // Hypothetical "replacement" assignment for the same stage but
    // a different contributor. No partial exists for this one.
    let asn_0_replacement = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0);
    let p_0 = signed_partial(&asn_0, &contrib_a);

    // Build an aggregate that only references asn_0's partial.
    let bytes = canonical_partial_result_bytes(&p_0).unwrap();
    let mut agg = AggregatedContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        final_result_snip_root: format!("0x{}", "bb".repeat(32)),
        final_result_canonical_hash: hex_lower(blake3::hash(b"final").as_bytes()),
        partial_refs: vec![AggregatedPartialRef {
            assignment_id: asn_0.assignment_id.clone(),
            stage_index: 0,
            contributor_pubkey_hex: p_0.contributor_pubkey_hex.clone(),
            partial_snip_root: format!("0x{}", "cc".repeat(32)),
            partial_canonical_hash: hex_lower(blake3::hash(&bytes).as_bytes()),
        }],
        aggregated_at_utc: "2026-06-01T01:00:00Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let si = aggregated_result_signing_input(&agg).unwrap();
    agg.coordinator_signature_hex = coord.sign_hex(&si);

    // Pass BOTH assignments to the verifier. The replacement
    // assignment has no partial → `AggregateMissingPartialFor`.
    let outcome = verify_aggregated_result(
        &session,
        &[join_a, join_b],
        &[asn_0, asn_0_replacement.clone()],
        &[p_0],
        &agg,
    );
    let matched = matches!(
        &outcome,
        omni_contributor::SessionVerifyOutcome::AggregateMissingPartialFor { assignment_id }
            if assignment_id == &asn_0_replacement.assignment_id
    );
    assert!(
        matched,
        "extra assignment without partial must be rejected; got {outcome:?}"
    );
}
