//! Stage 12.12 — end-to-end policy tests for
//! `check_reassign_eligible_allowing_invalid_partials`.
//!
//! The library-level helper is the only path that accepts an
//! `InvalidState` status, and only when every `invalid_artifacts`
//! entry is an `InvalidPartial` whose `assignment_id` is targeted
//! by the reassignment plan. These tests synthesize the precise
//! `(SessionStatusReport, SessionRepairPlan)` shapes the apply
//! path will encounter and pin the accept/refuse decision matrix.
//!
//! Hard rules preserved: no chain wire, no payment, no proof
//! mode, no marketplace, no new gossipsub topic, no SNIP wire
//! change.

use omni_contributor::{
    check_reassign_eligible_allowing_invalid_partials, repair_plan_hash_hex,
    source_status_hash_hex,
    status::{
        AssignmentStatus, SessionOverallStatus, SessionStatusReport,
        STATUS_SCHEMA_VERSION,
    },
    supersession::SupersessionReason,
    InvalidArtifactStatus, RepairAction, RepairError, RepairStrategy,
    SessionRepairPlan, REPAIR_PLAN_SCHEMA_VERSION,
};
use omni_contributor::result::WorkUnitKind;
use omni_contributor::session::WorkKind;
use omni_contributor::SessionVerifyOutcome;

const NOW_UTC: &str = "2026-06-02T01:00:00Z";
const FAR_FUTURE: &str = "2026-06-03T00:00:00Z";

fn synth_assignment_status(
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

fn synth_invalid_state_status(
    assignments: Vec<AssignmentStatus>,
    invalid_artifacts: Vec<InvalidArtifactStatus>,
) -> SessionStatusReport {
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
        missing_assignment_ids: assignments
            .iter()
            .filter(|a| !a.partial_present)
            .map(|a| a.assignment_id.clone())
            .collect(),
        duplicate_partial_assignment_ids: Vec::new(),
        aggregate_present: false,
        aggregate_valid: false,
        overall_status: SessionOverallStatus::InvalidState,
        assignments,
        supersessions: Vec::new(),
        invalid_artifacts,
        notes: vec!["operator-visible note kept alongside".into()],
    }
}

fn synth_plan(status: &SessionStatusReport, targets: &[&AssignmentStatus]) -> SessionRepairPlan {
    let actions: Vec<RepairAction> = targets
        .iter()
        .map(|a| RepairAction::ReassignAssignment {
            superseded_assignment_id: a.assignment_id.clone(),
            original_stage_index: a.stage_index,
            replacement_contributor_pubkey_hex: a.contributor_pubkey_hex.clone(),
            replacement_stage_index: a.stage_index,
            replacement_work_kind: a.work_kind.clone(),
            replacement_expected_work_units: a.expected_work_units,
            replacement_expected_work_unit_kind: a.expected_work_unit_kind,
            reason: SupersessionReason::InvalidPartial,
        })
        .collect();
    let mut plan = SessionRepairPlan {
        schema_version: REPAIR_PLAN_SCHEMA_VERSION,
        session_id: status.session_id.clone(),
        source_status_hash: source_status_hash_hex(status),
        strategy: RepairStrategy::ReassignMissing,
        created_at_utc: NOW_UTC.into(),
        coordinator_pubkey_hex: None,
        actions,
        repair_plan_hash: String::new(),
    };
    plan.repair_plan_hash = repair_plan_hash_hex(&plan);
    plan
}

// ── SessionVerifyOutcome::reason_tag stability ──────────────────
//
// Stage 12.12 made `reason_tag` part of the local
// `SessionStatusReport` v3 contract. The tag strings are
// **frozen** per variant — renaming a variant in source must not
// silently change the wire surface read by automation.

#[test]
fn session_verify_outcome_reason_tags_are_stable() {
    // Construct each variant with placeholder fields and pin the
    // returned tag. Any variant rename or accidental
    // `format!("{:?}")` substitution will fail this test.
    let cases: Vec<(SessionVerifyOutcome, &'static str)> = vec![
        (SessionVerifyOutcome::Ok, "Ok"),
        (
            SessionVerifyOutcome::SchemaMalformed("x".into()),
            "SchemaMalformed",
        ),
        (
            SessionVerifyOutcome::SessionIdMismatch {
                stored: "a".into(),
                derived: "b".into(),
            },
            "SessionIdMismatch",
        ),
        (
            SessionVerifyOutcome::AssignmentIdMismatch {
                stored: "a".into(),
                derived: "b".into(),
            },
            "AssignmentIdMismatch",
        ),
        (
            SessionVerifyOutcome::CoordinatorSignatureFailed,
            "CoordinatorSignatureFailed",
        ),
        (
            SessionVerifyOutcome::ContributorSignatureFailed,
            "ContributorSignatureFailed",
        ),
        (
            SessionVerifyOutcome::BindingMismatch { field: "x" },
            "BindingMismatch",
        ),
        (
            SessionVerifyOutcome::ExpiredAtCheck {
                now: "n".into(),
                expires_at: "e".into(),
            },
            "ExpiredAtCheck",
        ),
        (
            SessionVerifyOutcome::AggregateMissingPartialFor {
                assignment_id: "a".into(),
            },
            "AggregateMissingPartialFor",
        ),
        (
            SessionVerifyOutcome::AggregateExtraPartialFor {
                assignment_id: "a".into(),
            },
            "AggregateExtraPartialFor",
        ),
        (
            SessionVerifyOutcome::AggregateDuplicatePartialFor {
                assignment_id: "a".into(),
            },
            "AggregateDuplicatePartialFor",
        ),
        (
            SessionVerifyOutcome::AggregatePartialRefDrift {
                field: "x",
                assignment_id: "a".into(),
            },
            "AggregatePartialRefDrift",
        ),
        (
            SessionVerifyOutcome::AggregateCoordinatorMismatch,
            "AggregateCoordinatorMismatch",
        ),
        (
            SessionVerifyOutcome::InternalError("x".into()),
            "InternalError",
        ),
        (
            SessionVerifyOutcome::SupersessionSchemaMalformed("x".into()),
            "SupersessionSchemaMalformed",
        ),
        (
            SessionVerifyOutcome::SupersessionSessionMismatch,
            "SupersessionSessionMismatch",
        ),
        (
            SessionVerifyOutcome::SupersessionCoordinatorMismatch,
            "SupersessionCoordinatorMismatch",
        ),
        (
            SessionVerifyOutcome::SupersessionIdMismatch {
                stored: "a".into(),
                derived: "b".into(),
            },
            "SupersessionIdMismatch",
        ),
        (
            SessionVerifyOutcome::SupersessionCoordinatorSignatureFailed,
            "SupersessionCoordinatorSignatureFailed",
        ),
        (
            SessionVerifyOutcome::SupersessionReferenceUnknown {
                kind: "superseded",
                assignment_id: "a".into(),
            },
            "SupersessionReferenceUnknown",
        ),
        (
            SessionVerifyOutcome::SupersessionDuplicateSupersedes {
                assignment_id: "a".into(),
            },
            "SupersessionDuplicateSupersedes",
        ),
        (
            SessionVerifyOutcome::AggregatePartialRefSuperseded {
                assignment_id: "a".into(),
            },
            "AggregatePartialRefSuperseded",
        ),
    ];
    for (outcome, expected) in &cases {
        assert_eq!(
            outcome.reason_tag(),
            *expected,
            "reason_tag drift on {outcome:?}; v3 contract requires \
             stable tags, not Debug derivations"
        );
    }
}

// ── Accepted: every InvalidPartial entry is covered ─────────────

#[test]
fn accepts_invalid_state_when_all_invalid_artifacts_are_planned_invalid_partials() {
    let asn_a_id = "aa".repeat(32);
    let asn_b_id = "bb".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let asn_b = synth_assignment_status(&asn_b_id, 1, &"02".repeat(32), false);
    let invalid_artifacts = vec![
        InvalidArtifactStatus::InvalidPartial {
            assignment_id: asn_a_id.clone(),
            reason_tag: "ContributorSignatureFailed".into(),
        },
        InvalidArtifactStatus::InvalidPartial {
            assignment_id: asn_b_id.clone(),
            reason_tag: "BindingMismatch".into(),
        },
    ];
    let status = synth_invalid_state_status(
        vec![asn_a.clone(), asn_b.clone()],
        invalid_artifacts,
    );
    let plan = synth_plan(&status, &[&asn_a, &asn_b]);
    check_reassign_eligible_allowing_invalid_partials(&status, &plan)
        .expect("must accept when every InvalidPartial is in the plan");
}

// ── Refused: extra invalid kinds ────────────────────────────────

#[test]
fn refuses_when_invalid_state_also_has_invalid_join() {
    let asn_a_id = "aa".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let invalid_artifacts = vec![
        InvalidArtifactStatus::InvalidPartial {
            assignment_id: asn_a_id.clone(),
            reason_tag: "ContributorSignatureFailed".into(),
        },
        InvalidArtifactStatus::InvalidJoin {
            contributor_pubkey_hex: "01".repeat(32),
            reason_tag: "ContributorSignatureFailed".into(),
        },
    ];
    let status =
        synth_invalid_state_status(vec![asn_a.clone()], invalid_artifacts);
    let plan = synth_plan(&status, &[&asn_a]);
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    match err {
        RepairError::InvalidStateNotTriagable { kind, .. } => {
            assert_eq!(kind, "invalid_join");
        }
        other => panic!("expected InvalidStateNotTriagable invalid_join, got {other:?}"),
    }
}

#[test]
fn refuses_when_invalid_state_also_has_invalid_assignment() {
    let asn_a_id = "aa".repeat(32);
    let asn_b_id = "bb".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let invalid_artifacts = vec![
        InvalidArtifactStatus::InvalidPartial {
            assignment_id: asn_a_id.clone(),
            reason_tag: "ContributorSignatureFailed".into(),
        },
        InvalidArtifactStatus::InvalidAssignment {
            assignment_id: asn_b_id.clone(),
            reason_tag: "CoordinatorSignatureFailed".into(),
        },
    ];
    let status =
        synth_invalid_state_status(vec![asn_a.clone()], invalid_artifacts);
    let plan = synth_plan(&status, &[&asn_a]);
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    match err {
        RepairError::InvalidStateNotTriagable { kind, context } => {
            assert_eq!(kind, "invalid_assignment");
            assert!(context.contains(&asn_b_id));
        }
        other => panic!("expected invalid_assignment, got {other:?}"),
    }
}

#[test]
fn refuses_when_invalid_state_has_invalid_aggregate() {
    let asn_a_id = "aa".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let invalid_artifacts = vec![InvalidArtifactStatus::InvalidAggregate {
        reason_tag: "AggregateCoordinatorMismatch".into(),
    }];
    let status =
        synth_invalid_state_status(vec![asn_a.clone()], invalid_artifacts);
    let plan = synth_plan(&status, &[&asn_a]);
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    match err {
        RepairError::InvalidStateNotTriagable { kind, .. } => {
            assert_eq!(kind, "invalid_aggregate");
        }
        other => panic!("expected invalid_aggregate, got {other:?}"),
    }
}

#[test]
fn refuses_when_invalid_state_has_invalid_supersession() {
    let asn_a_id = "aa".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let sup_id = "ee".repeat(32);
    let invalid_artifacts = vec![InvalidArtifactStatus::InvalidSupersession {
        supersession_id: sup_id.clone(),
        reason_tag: "SupersessionCoordinatorSignatureFailed".into(),
    }];
    let status =
        synth_invalid_state_status(vec![asn_a.clone()], invalid_artifacts);
    let plan = synth_plan(&status, &[&asn_a]);
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    match err {
        RepairError::InvalidStateNotTriagable { kind, context } => {
            assert_eq!(kind, "invalid_supersession");
            assert!(context.contains(&sup_id));
        }
        other => panic!("expected invalid_supersession, got {other:?}"),
    }
}

#[test]
fn refuses_when_invalid_state_has_invalid_session() {
    // Session-failure path: the reporter returns early with NO
    // assignments populated. Apply still must refuse.
    let status = SessionStatusReport {
        schema_version: STATUS_SCHEMA_VERSION,
        session_id: "11".repeat(32),
        posted_id: None,
        model_hash: None,
        generated_at_utc: NOW_UTC.into(),
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
        overall_status: SessionOverallStatus::InvalidState,
        assignments: Vec::new(),
        supersessions: Vec::new(),
        invalid_artifacts: vec![InvalidArtifactStatus::InvalidSession {
            reason_tag: "CoordinatorSignatureFailed".into(),
        }],
        notes: vec!["session.json failed verify_execution_session".into()],
    };
    // Plan is empty (no assignments to target).
    let mut plan = SessionRepairPlan {
        schema_version: REPAIR_PLAN_SCHEMA_VERSION,
        session_id: status.session_id.clone(),
        source_status_hash: source_status_hash_hex(&status),
        strategy: RepairStrategy::ReassignMissing,
        created_at_utc: NOW_UTC.into(),
        coordinator_pubkey_hex: None,
        actions: vec![],
        repair_plan_hash: String::new(),
    };
    plan.repair_plan_hash = repair_plan_hash_hex(&plan);
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    match err {
        RepairError::InvalidStateNotTriagable { kind, .. } => {
            assert_eq!(kind, "invalid_session");
        }
        other => panic!("expected invalid_session, got {other:?}"),
    }
}

// ── Refused: InvalidPartial not in plan ─────────────────────────

#[test]
fn refuses_when_invalid_partial_assignment_is_not_in_plan() {
    let asn_a_id = "aa".repeat(32);
    let asn_b_id = "bb".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let asn_b = synth_assignment_status(&asn_b_id, 1, &"02".repeat(32), false);
    let invalid_artifacts = vec![
        InvalidArtifactStatus::InvalidPartial {
            assignment_id: asn_a_id.clone(),
            reason_tag: "ContributorSignatureFailed".into(),
        },
        InvalidArtifactStatus::InvalidPartial {
            assignment_id: asn_b_id.clone(),
            reason_tag: "ContributorSignatureFailed".into(),
        },
    ];
    let status = synth_invalid_state_status(
        vec![asn_a.clone(), asn_b.clone()],
        invalid_artifacts,
    );
    // Plan targets ONLY A; B is left out.
    let plan = synth_plan(&status, &[&asn_a]);
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    match err {
        RepairError::InvalidStateNotTriagable { kind, context } => {
            assert_eq!(kind, "invalid_partial_not_in_plan");
            assert!(context.contains(&asn_b_id));
        }
        other => panic!("expected invalid_partial_not_in_plan, got {other:?}"),
    }
}

// ── Refused: ReannounceMissing never accepts InvalidState ───────

#[test]
fn refuses_reannounce_missing_strategy_under_invalid_state() {
    // Even when every invalid_artifact is an InvalidPartial,
    // ReannounceMissing must refuse — reannounce is not a
    // supersession path.
    let asn_a_id = "aa".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let invalid_artifacts = vec![InvalidArtifactStatus::InvalidPartial {
        assignment_id: asn_a_id.clone(),
        reason_tag: "ContributorSignatureFailed".into(),
    }];
    let status =
        synth_invalid_state_status(vec![asn_a.clone()], invalid_artifacts);
    let mut plan = synth_plan(&status, &[&asn_a]);
    plan.strategy = RepairStrategy::ReannounceMissing;
    plan.repair_plan_hash = repair_plan_hash_hex(&plan);
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    assert!(matches!(err, RepairError::InvalidState));
}

// ── Defense-in-depth: InvalidState without diagnostics ──────────

#[test]
fn refuses_invalid_state_when_diagnostics_array_is_empty() {
    // Hand-crafted hostile status: overall_status InvalidState
    // but invalid_artifacts is empty. The Stage 12.12 reporter
    // never produces this, but a tampered status report could.
    // The helper refuses defensively.
    let asn_a_id = "aa".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let status =
        synth_invalid_state_status(vec![asn_a.clone()], Vec::new());
    let plan = synth_plan(&status, &[&asn_a]);
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    match err {
        RepairError::InvalidStateNotTriagable { kind, .. } => {
            assert_eq!(kind, "invalid_state_without_diagnostics");
        }
        other => panic!(
            "expected invalid_state_without_diagnostics defensive refusal, \
             got {other:?}"
        ),
    }
}

// ── Pass-through to check_repair_eligible for non-InvalidState ──

#[test]
fn forwards_to_check_repair_eligible_for_non_invalid_state() {
    // InProgress passes; the helper takes the standard path.
    let asn_a_id = "aa".repeat(32);
    let asn_a = synth_assignment_status(&asn_a_id, 0, &"01".repeat(32), false);
    let mut status =
        synth_invalid_state_status(vec![asn_a.clone()], Vec::new());
    status.overall_status = SessionOverallStatus::InProgress;
    let plan = synth_plan(&status, &[&asn_a]);
    check_reassign_eligible_allowing_invalid_partials(&status, &plan)
        .expect("InProgress falls through to check_repair_eligible (Ok)");

    // Aggregated still refused via NothingToRepair.
    status.overall_status = SessionOverallStatus::Aggregated;
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    assert!(matches!(err, RepairError::NothingToRepair { .. }));

    // ExpiredIncomplete refused via SessionExpired.
    status.overall_status = SessionOverallStatus::ExpiredIncomplete;
    let err =
        check_reassign_eligible_allowing_invalid_partials(&status, &plan).unwrap_err();
    assert!(matches!(err, RepairError::SessionExpired));
}
