//! Stage 12.13 — Phase B partial-apply audit tests.
//!
//! `apply-session-reassign` Phase B (state-dir mutation + mesh
//! broadcast) was reordered into B1..B4 so the worst-case mid-loop
//! failure window is narrowed:
//!
//!   - B1 writes every replacement assignment to the state-dir.
//!   - B2 writes the supersession to the state-dir.
//!   - B3 broadcasts each replacement on the mesh.
//!   - B4 broadcasts the supersession on the mesh ONLY when every
//!     B3 broadcast succeeded.
//!
//! Two failure modes still produce visible diagnostic state:
//!
//!   1. **B1 OK + B2 fails** → state-dir holds replacement
//!      assignments without the retiring supersession. The
//!      Stage 12.13 `compute_audit_health` projection over the
//!      v3 status report MUST tag this as
//!      `OrphanReplacementAssignments` so operators see it.
//!   2. **B1 OK + B2 OK** (the happy path) → audit is `Coherent`.
//!
//! These tests synthesize the precise `SessionStatusReport`
//! shapes that follow each phase boundary and pin the audit
//! decisions. They do NOT drive the CLI — the CLI's Bucket 3
//! reorder is a refactor whose end-to-end coverage is the
//! existing apply-session-reassign suite.

use omni_contributor::{
    compute_audit_health, source_status_hash_hex,
    status::{
        AssignmentStatus, SessionOverallStatus, SessionStatusReport, SupersessionStatus,
        STATUS_SCHEMA_VERSION,
    },
    supersession::SupersessionReason,
    AuditCoherence,
};
use omni_contributor::result::WorkUnitKind;
use omni_contributor::session::WorkKind;

const NOW_UTC: &str = "2026-06-02T22:00:00Z";
const FAR_FUTURE: &str = "2026-12-31T23:59:59Z";

fn synth_assignment(
    id: &str,
    stage: u32,
    pubkey: &str,
    partial_present: bool,
    superseded: bool,
    superseded_by: Option<&str>,
) -> AssignmentStatus {
    AssignmentStatus {
        assignment_id: id.to_string(),
        stage_index: stage,
        contributor_pubkey_hex: pubkey.to_string(),
        work_kind: WorkKind::Layers { start: 0, end: 8 },
        expected_work_units: 8,
        expected_work_unit_kind: WorkUnitKind::Layers,
        join_present: true,
        peer_advert_present: false,
        partial_present,
        partial_valid: partial_present,
        partial_snip_root: None,
        superseded,
        superseded_by_supersession_id: superseded_by.map(String::from),
        notes: Vec::new(),
    }
}

fn synth_report(
    assignments: Vec<AssignmentStatus>,
    supersessions: Vec<SupersessionStatus>,
    overall_status: SessionOverallStatus,
) -> SessionStatusReport {
    let count = assignments.len() as u32;
    let superseded_count = assignments.iter().filter(|a| a.superseded).count() as u32;
    let active_count = count.saturating_sub(superseded_count);
    let missing: Vec<String> = assignments
        .iter()
        .filter(|a| !a.superseded && !a.partial_present)
        .map(|a| a.assignment_id.clone())
        .collect();
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
        active_assignment_count: active_count,
        superseded_assignment_count: superseded_count,
        supersession_count: supersessions.len() as u32,
        missing_assignment_ids: missing,
        duplicate_partial_assignment_ids: Vec::new(),
        aggregate_present: false,
        aggregate_valid: false,
        overall_status,
        assignments,
        supersessions,
        invalid_artifacts: Vec::new(),
        notes: Vec::new(),
    }
}

// ── Happy path: Coherent ────────────────────────────────────────

#[test]
fn complete_apply_audits_coherent() {
    // B1 + B2 + B3 + B4 all succeeded. State has the original
    // assignment marked superseded, the replacement assignment
    // active with no partial yet, and the supersession is valid.
    let asn_old_id = "aa".repeat(32);
    let asn_new_id = "bb".repeat(32);
    let sup_id = "ee".repeat(32);
    let old = synth_assignment(
        &asn_old_id,
        0,
        &"01".repeat(32),
        /* partial_present = */ false,
        /* superseded = */ true,
        Some(&sup_id),
    );
    let new = synth_assignment(
        &asn_new_id,
        0,
        &"02".repeat(32),
        false,
        false,
        None,
    );
    let sup = SupersessionStatus {
        supersession_id: sup_id.clone(),
        superseded_assignment_ids: vec![asn_old_id.clone()],
        replacement_assignment_ids: vec![asn_new_id.clone()],
        reason: SupersessionReason::MissingPartial,
        valid: true,
        notes: vec![],
    };
    let report = synth_report(
        vec![old, new],
        vec![sup],
        SessionOverallStatus::InProgress,
    );
    let audit = compute_audit_health(&report);
    assert!(matches!(audit.coherence, AuditCoherence::Coherent));
    assert!(!audit.triagable_by_reassign);
    // InProgress with an active-missing partial → recommended
    // missing-partial reannounce.
    assert_eq!(
        audit.recommended_action,
        "run plan-session-reassign --reason missing-partial"
    );
}

// ── Orphan replacement after Phase B2 failure ───────────────────

#[test]
fn audit_detects_orphan_replacement_assignment() {
    // B1 OK: replacement assignment is in the state-dir as an
    // active assignment.
    // B2 failed: the supersession is in the state-dir but its
    // `verify_assignment_supersession` re-run fails (the
    // reporter loaded it but its reference resolution against
    // the FULL chain — including the still-superseded old
    // assignment — fails, so `valid = false`).
    //
    // Operator-visible signal: the active assignment B is named
    // in some supersession's replacement_assignment_ids but the
    // covering supersession is invalid. `compute_audit_health`
    // must flag this as `OrphanReplacementAssignments`.
    let asn_old_id = "aa".repeat(32);
    let asn_new_id = "bb".repeat(32);
    let sup_id = "ee".repeat(32);
    let old = synth_assignment(
        &asn_old_id,
        0,
        &"01".repeat(32),
        false,
        // The supersession is invalid so it does NOT supersede
        // the old assignment — the chain still sees both as
        // active.
        false,
        None,
    );
    // Put the replacement at a DIFFERENT stage_index so this
    // test isolates the (2a) invalid-supersession signal from
    // the (2b) duplicate-active-stage signal added in the
    // Stage 12.13 review fix. The Stage 12.11 reassign planner
    // normally matches `stage_index`, but a hand-crafted /
    // operator-rebalance plan can ship a replacement at a
    // different stage — that's the case this test pins.
    let new = synth_assignment(
        &asn_new_id,
        7,
        &"02".repeat(32),
        false,
        false,
        None,
    );
    let sup = SupersessionStatus {
        supersession_id: sup_id.clone(),
        superseded_assignment_ids: vec![asn_old_id.clone()],
        replacement_assignment_ids: vec![asn_new_id.clone()],
        reason: SupersessionReason::OperatorRebalance,
        valid: false,
        notes: vec!["invalid supersession (forged sig)".into()],
    };
    let report = synth_report(
        vec![old, new],
        vec![sup],
        // The status reporter would normally flip InvalidState
        // here because the supersession failed verify. Synthesize
        // the same shape so the audit projection runs on a
        // realistic input.
        SessionOverallStatus::InvalidState,
    );
    let audit = compute_audit_health(&report);
    match audit.coherence {
        AuditCoherence::OrphanReplacementAssignments { ref assignment_ids } => {
            assert_eq!(assignment_ids.len(), 1);
            assert_eq!(assignment_ids[0], asn_new_id);
        }
        other => panic!("expected OrphanReplacementAssignments, got {other:?}"),
    }
    assert_eq!(
        audit.recommended_action,
        "clean state-dir orphan replacements before retry"
    );
}

// ── B1-only orphan: state-dir has no supersession file at all ───

#[test]
fn audit_detects_b1_only_orphan_via_duplicate_active_stage_index() {
    // Phase B1 succeeded: state-dir wrote the replacement
    // assignment at stage 0. Phase B2 never landed: there is NO
    // supersession file on disk, so the status report has zero
    // supersessions. The chain now sees TWO active assignments
    // at stage 0 — the original (untouched) and the
    // freshly-written replacement. The invalid-supersession
    // signal can't catch this (no supersession exists to
    // examine), but the duplicate-active-stage signal does.
    let asn_old_id = "aa".repeat(32);
    let asn_new_id = "bb".repeat(32);
    let old = synth_assignment(
        &asn_old_id,
        0,
        &"01".repeat(32),
        /* partial_present = */ false,
        /* superseded = */ false,
        None,
    );
    let new = synth_assignment(
        &asn_new_id,
        // Same stage_index as `old` — the planner's contract is
        // one active assignment per stage, so duplicates are
        // structurally a B1-only partial apply.
        0,
        &"02".repeat(32),
        false,
        false,
        None,
    );
    // Zero supersessions — Phase B2 never landed.
    let report = synth_report(
        vec![old, new],
        vec![],
        SessionOverallStatus::InProgress,
    );
    let audit = compute_audit_health(&report);
    match audit.coherence {
        AuditCoherence::OrphanReplacementAssignments { ref assignment_ids } => {
            // Both active assignments at the conflicted stage
            // are reported so the operator can inspect each
            // body's `assigned_at_utc` to decide which is the
            // replacement vs the original.
            assert_eq!(assignment_ids.len(), 2);
            assert!(assignment_ids.contains(&asn_old_id));
            assert!(assignment_ids.contains(&asn_new_id));
        }
        other => panic!(
            "expected OrphanReplacementAssignments for B1-only \
             duplicate-active-stage, got {other:?}"
        ),
    }
    assert_eq!(
        audit.recommended_action,
        "clean state-dir orphan replacements before retry"
    );
}

#[test]
fn audit_does_not_flag_duplicate_when_one_assignment_is_superseded() {
    // Negative control: a clean post-apply state has the old
    // assignment marked superseded by a valid supersession AND
    // the replacement active at the same stage_index. The
    // duplicate-active-stage detector must NOT fire because the
    // old assignment is NOT active.
    let asn_old_id = "aa".repeat(32);
    let asn_new_id = "bb".repeat(32);
    let sup_id = "ee".repeat(32);
    let old = synth_assignment(
        &asn_old_id,
        0,
        &"01".repeat(32),
        false,
        /* superseded = */ true,
        Some(&sup_id),
    );
    let new = synth_assignment(&asn_new_id, 0, &"02".repeat(32), false, false, None);
    let sup = SupersessionStatus {
        supersession_id: sup_id.clone(),
        superseded_assignment_ids: vec![asn_old_id.clone()],
        replacement_assignment_ids: vec![asn_new_id.clone()],
        reason: SupersessionReason::MissingPartial,
        valid: true,
        notes: vec![],
    };
    let report = synth_report(
        vec![old, new],
        vec![sup],
        SessionOverallStatus::InProgress,
    );
    let audit = compute_audit_health(&report);
    // Coherent — exactly one ACTIVE assignment per stage.
    assert!(matches!(audit.coherence, AuditCoherence::Coherent));
}

// ── Partial-apply via missing reference (best-effort accept) ────

#[test]
fn audit_detects_partial_apply_supersession_with_unresolved_reference() {
    // The watcher's Stage 12.13 best-effort path accepted a
    // supersession whose referenced replacement assignment is
    // NOT yet in the state-dir (out-of-order arrival). The
    // status reporter's projection: the supersession sits in
    // `supersessions` with valid=true (announcer+body verified),
    // but its `superseded_assignment_ids` ∪
    // `replacement_assignment_ids` mentions an id that has no
    // matching `AssignmentStatus` row. Audit must flag
    // `PartialApplySupersession`.
    let asn_old_id = "aa".repeat(32);
    let asn_new_id = "bb".repeat(32);  // present
    let asn_missing_id = "cc".repeat(32);  // referenced but not in state
    let sup_id = "ee".repeat(32);
    let old = synth_assignment(
        &asn_old_id,
        0,
        &"01".repeat(32),
        false,
        true,
        Some(&sup_id),
    );
    let new = synth_assignment(
        &asn_new_id,
        0,
        &"02".repeat(32),
        false,
        false,
        None,
    );
    let sup = SupersessionStatus {
        supersession_id: sup_id.clone(),
        superseded_assignment_ids: vec![asn_old_id.clone()],
        // References both the present `bb...` AND the missing
        // `cc...`. The audit projection counts only those NOT in
        // the assignment set.
        replacement_assignment_ids: vec![asn_new_id.clone(), asn_missing_id.clone()],
        reason: SupersessionReason::MissingPartial,
        valid: true,
        notes: vec![],
    };
    let report = synth_report(
        vec![old, new],
        vec![sup],
        SessionOverallStatus::InProgress,
    );
    let audit = compute_audit_health(&report);
    match audit.coherence {
        AuditCoherence::PartialApplySupersession {
            ref supersession_id,
            unresolved_count,
        } => {
            assert_eq!(supersession_id, &sup_id);
            assert_eq!(unresolved_count, 1);
        }
        other => panic!("expected PartialApplySupersession, got {other:?}"),
    }
    assert_eq!(audit.recommended_action, "operator triage required");
}

// ── source_status_hash drift across Phase B partial-apply ───────

#[test]
fn phase_b_partial_apply_state_drifts_source_status_hash() {
    // Plan-time: the status report shows the original assignment
    // active and no replacement / supersession.
    let asn_old_id = "aa".repeat(32);
    let original_old = synth_assignment(
        &asn_old_id,
        0,
        &"01".repeat(32),
        false,
        false,
        None,
    );
    let report_pre = synth_report(
        vec![original_old.clone()],
        vec![],
        SessionOverallStatus::InProgress,
    );
    let plan_hash = source_status_hash_hex(&report_pre);

    // Apply-time (after Phase B1 succeeds but B2 fails): state
    // now has a new active replacement assignment AND the
    // original — but no supersession to mark the original
    // retired. The active-cover shape changed: the projection
    // includes per-assignment `(partial_present, superseded,
    // superseded_by_supersession_id)`, and the replacement is a
    // new row.
    let asn_new_id = "bb".repeat(32);
    let new = synth_assignment(
        &asn_new_id,
        0,
        &"02".repeat(32),
        false,
        false,
        None,
    );
    let report_after_b1 = synth_report(
        vec![original_old.clone(), new],
        vec![],
        SessionOverallStatus::InProgress,
    );
    let drifted_hash = source_status_hash_hex(&report_after_b1);
    assert_ne!(
        plan_hash, drifted_hash,
        "source_status_hash MUST drift across a Phase B1-only \
         partial apply so the apply retry refuses with \
         SourceStatusDrift"
    );
}

// ── Stage 12.12 v3 InvalidState arms still reachable through audit ──

#[test]
fn audit_reports_reassign_triagable_when_only_invalid_partial_entries() {
    use omni_contributor::InvalidArtifactStatus;
    let asn_id = "aa".repeat(32);
    let asn = synth_assignment(&asn_id, 0, &"01".repeat(32), false, false, None);
    let mut report = synth_report(
        vec![asn],
        vec![],
        SessionOverallStatus::InvalidState,
    );
    report.invalid_artifacts = vec![InvalidArtifactStatus::InvalidPartial {
        assignment_id: asn_id.clone(),
        reason_tag: "ContributorSignatureFailed".into(),
    }];
    let audit = compute_audit_health(&report);
    assert!(matches!(audit.coherence, AuditCoherence::ReassignTriagable));
    assert!(audit.triagable_by_reassign);
    assert_eq!(
        audit.recommended_action,
        "run plan-session-reassign --reason invalid-partial"
    );
}

#[test]
fn audit_reports_not_reassign_triagable_when_invalid_join_present() {
    use omni_contributor::InvalidArtifactStatus;
    let asn_id = "aa".repeat(32);
    let asn = synth_assignment(&asn_id, 0, &"01".repeat(32), false, false, None);
    let mut report = synth_report(
        vec![asn],
        vec![],
        SessionOverallStatus::InvalidState,
    );
    report.invalid_artifacts = vec![
        InvalidArtifactStatus::InvalidPartial {
            assignment_id: asn_id.clone(),
            reason_tag: "ContributorSignatureFailed".into(),
        },
        InvalidArtifactStatus::InvalidJoin {
            contributor_pubkey_hex: "01".repeat(32),
            reason_tag: "ContributorSignatureFailed".into(),
        },
    ];
    let audit = compute_audit_health(&report);
    assert!(matches!(audit.coherence, AuditCoherence::NotReassignTriagable));
    assert!(!audit.triagable_by_reassign);
    assert_eq!(audit.recommended_action, "operator triage required");
}

// ── Stage 12.14 — archive recommendations ──────────────────────

#[test]
fn audit_recommends_archive_for_coherent_aggregated() {
    let asn_id = "aa".repeat(32);
    let asn = synth_assignment(
        &asn_id,
        0,
        &"01".repeat(32),
        /* partial_present = */ true,
        false,
        None,
    );
    let report = synth_report(
        vec![asn],
        vec![],
        SessionOverallStatus::Aggregated,
    );
    let audit = compute_audit_health(&report);
    assert!(matches!(audit.coherence, AuditCoherence::Coherent));
    assert_eq!(
        audit.recommended_action,
        "run archive-session --require-status aggregated"
    );
}

#[test]
fn audit_recommends_archive_for_coherent_expired_incomplete() {
    let asn_id = "aa".repeat(32);
    let asn = synth_assignment(
        &asn_id,
        0,
        &"01".repeat(32),
        false,
        false,
        None,
    );
    let mut report = synth_report(
        vec![asn],
        vec![],
        SessionOverallStatus::ExpiredIncomplete,
    );
    report.session_expired = true;
    let audit = compute_audit_health(&report);
    assert!(matches!(audit.coherence, AuditCoherence::Coherent));
    assert_eq!(
        audit.recommended_action,
        "run archive-session --require-status expired-incomplete"
    );
}

#[test]
fn audit_does_not_recommend_archive_for_complete_partials() {
    // Stage 12.14 decision 3: archive is recommended only for
    // Aggregated + ExpiredIncomplete. CompletePartials is left
    // alone because the operator may still want the aggregate to
    // land.
    let asn_id = "aa".repeat(32);
    let asn = synth_assignment(
        &asn_id,
        0,
        &"01".repeat(32),
        /* partial_present = */ true,
        false,
        None,
    );
    let report = synth_report(
        vec![asn],
        vec![],
        SessionOverallStatus::CompletePartials,
    );
    let audit = compute_audit_health(&report);
    assert!(matches!(audit.coherence, AuditCoherence::Coherent));
    assert_eq!(audit.recommended_action, "none");
}
