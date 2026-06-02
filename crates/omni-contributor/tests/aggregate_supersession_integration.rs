//! Stage 12.11 — full-chain `verify_aggregated_result_with_supersessions`
//! integration tests.
//!
//! Covers the v1 supersession-aware contract end to end: the happy
//! path with a single supersession + replacement, sequential
//! supersession chains, partials whose assignment is superseded
//! being skipped from re-verification, aggregate refs pointing at
//! superseded assignments being rejected, double-supersession
//! refusal, and the InvalidPartial reason path.

use omni_contributor::{
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, partial_result_signing_input,
        session_id_hex, supersession_id_hex, work_assignment_signing_input,
        work_assignment_supersession_signing_input,
    },
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    session::{
        AggregatedContributorResult, AggregatedPartialRef, PartialContributorResult,
        WorkAssignment, WorkKind,
    },
    supersession::{SupersessionReason, WorkAssignmentSupersession},
    verify_aggregated_result_with_supersessions, ContributorJoin, ContributorSigner,
    CoordinatorSigner, ExecutionSession, SessionVerifyOutcome, SESSION_SCHEMA_VERSION,
    SUPERSESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.11-aggregate-coord-32-by";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.11-aggregate-contrib-a-3";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.11-aggregate-contrib-b-3";
const CONTRIB_C_SEED: [u8; 32] = *b"stage12.11-aggregate-contrib-c-3";

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
    assigned_at_seconds: u32,
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
        // Different `assigned_at_utc` distinguishes the original
        // and replacement assignments so they hash to different
        // assignment_ids even for the same stage_index.
        assigned_at_utc: format!("2026-06-01T00:00:{:02}Z", assigned_at_seconds),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

fn signed_partial(
    assignment: &WrkAssignmentForBuilder,
    contrib: &ContributorSigner,
) -> PartialContributorResult {
    let WrkAssignmentForBuilder(assignment) = assignment;
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

/// Newtype just to keep `signed_partial`'s API shape readable.
struct WrkAssignmentForBuilder<'a>(&'a WorkAssignment);

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
    let si = work_assignment_supersession_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn signed_aggregate(
    session: &ExecutionSession,
    refs_for: &[(&WorkAssignment, &PartialContributorResult)],
    coord: &CoordinatorSigner,
) -> AggregatedContributorResult {
    let mut partial_refs: Vec<AggregatedPartialRef> = refs_for
        .iter()
        .map(|(a, p)| {
            let bytes = canonical_partial_result_bytes(p).unwrap();
            AggregatedPartialRef {
                assignment_id: a.assignment_id.clone(),
                stage_index: a.stage_index,
                contributor_pubkey_hex: p.contributor_pubkey_hex.clone(),
                partial_snip_root: format!("0x{}", "cc".repeat(32)),
                partial_canonical_hash: hex_lower(blake3::hash(&bytes).as_bytes()),
            }
        })
        .collect();
    partial_refs.sort_by_key(|r| r.stage_index);
    let mut g = AggregatedContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        final_result_snip_root: format!("0x{}", "bb".repeat(32)),
        final_result_canonical_hash: hex_lower(blake3::hash(b"final-canonical").as_bytes()),
        partial_refs,
        aggregated_at_utc: "2026-06-01T01:00:00Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let si = aggregated_result_signing_input(&g).unwrap();
    g.coordinator_signature_hex = coord.sign_hex(&si);
    g
}

// ── 1. Happy path: single supersession with valid replacement ──

/// Stage 12.10's halt-finding fixture pinned that an extra
/// assignment without a partial → `AggregateMissingPartialFor`.
/// Stage 12.11's positive counterpart: an extra assignment WITH a
/// signed supersession AND a valid replacement partial → `Ok`.
#[test]
fn aggregate_verifier_accepts_extra_assignment_with_supersession() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    let join_a = signed_join(&session, &contrib_a);
    let join_b = signed_join(&session, &contrib_b);
    // Stage 0 — original contributor A, never delivered.
    let asn_0_old = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0, 2);
    // Stage 0 — replacement contributor B (different
    // assigned_at_utc → distinct assignment_id).
    let asn_0_new = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0, 30);
    // Stage 1 — never reassigned.
    let asn_1 = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 1, 2);

    // Only the replacement + stage 1 have partials.
    let p_0_new = signed_partial(&WrkAssignmentForBuilder(&asn_0_new), &contrib_b);
    let p_1 = signed_partial(&WrkAssignmentForBuilder(&asn_1), &contrib_b);

    let supersession = signed_supersession(
        &session,
        &coord,
        vec![asn_0_old.assignment_id.clone()],
        vec![asn_0_new.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );

    let aggregate = signed_aggregate(
        &session,
        &[(&asn_0_new, &p_0_new), (&asn_1, &p_1)],
        &coord,
    );

    let outcome = verify_aggregated_result_with_supersessions(
        &session,
        &[join_a, join_b],
        &[asn_0_old, asn_0_new, asn_1],
        &[supersession],
        &[p_0_new, p_1],
        &aggregate,
    );
    assert!(matches!(outcome, SessionVerifyOutcome::Ok), "{outcome:?}");
}

// ── 2. Chain: sequential supersessions A→B and B→C ─────────────

#[test]
fn aggregate_verifier_accepts_sequential_supersession_chain() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let contrib_c = ContributorSigner::from_seed_bytes(&CONTRIB_C_SEED).unwrap();
    let session = signed_session(&coord);
    let joins = vec![
        signed_join(&session, &contrib_a),
        signed_join(&session, &contrib_b),
        signed_join(&session, &contrib_c),
    ];
    let asn_a = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0, 2);
    let asn_b = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0, 20);
    let asn_c = signed_assignment(&session, &coord, &contrib_c.pubkey_hex(), 0, 40);
    let p_c = signed_partial(&WrkAssignmentForBuilder(&asn_c), &contrib_c);

    let supersession_1 = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_b.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    let supersession_2 = signed_supersession(
        &session,
        &coord,
        vec![asn_b.assignment_id.clone()],
        vec![asn_c.assignment_id.clone()],
        SupersessionReason::OperatorRebalance,
    );

    let aggregate = signed_aggregate(&session, &[(&asn_c, &p_c)], &coord);

    let outcome = verify_aggregated_result_with_supersessions(
        &session,
        &joins,
        &[asn_a, asn_b, asn_c],
        &[supersession_1, supersession_2],
        &[p_c],
        &aggregate,
    );
    assert!(matches!(outcome, SessionVerifyOutcome::Ok), "{outcome:?}");
}

// ── 3. Double supersession of the same assignment ──────────────

#[test]
fn double_supersession_of_same_assignment_is_rejected() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let contrib_c = ContributorSigner::from_seed_bytes(&CONTRIB_C_SEED).unwrap();
    let session = signed_session(&coord);
    let joins = vec![
        signed_join(&session, &contrib_a),
        signed_join(&session, &contrib_b),
        signed_join(&session, &contrib_c),
    ];
    let asn_a = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0, 2);
    let asn_b = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0, 20);
    let asn_c = signed_assignment(&session, &coord, &contrib_c.pubkey_hex(), 0, 40);
    let p_b = signed_partial(&WrkAssignmentForBuilder(&asn_b), &contrib_b);

    // Two supersessions both name asn_a as superseded — explicit
    // double-supersession. Must be rejected.
    let supersession_1 = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_b.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    let supersession_2 = signed_supersession(
        &session,
        &coord,
        vec![asn_a.assignment_id.clone()],
        vec![asn_c.assignment_id.clone()],
        SupersessionReason::OperatorRebalance,
    );

    let aggregate = signed_aggregate(&session, &[(&asn_b, &p_b)], &coord);

    let outcome = verify_aggregated_result_with_supersessions(
        &session,
        &joins,
        &[asn_a.clone(), asn_b, asn_c],
        &[supersession_1, supersession_2],
        &[p_b],
        &aggregate,
    );
    assert!(
        matches!(
            outcome,
            SessionVerifyOutcome::SupersessionDuplicateSupersedes { ref assignment_id }
                if assignment_id == &asn_a.assignment_id
        ),
        "{outcome:?}"
    );
}

// ── 4. Aggregate ref points at a superseded assignment ─────────

#[test]
fn aggregate_partial_ref_pointing_at_superseded_is_rejected() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    let joins =
        vec![signed_join(&session, &contrib_a), signed_join(&session, &contrib_b)];
    let asn_a_old = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0, 2);
    let asn_a_new = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0, 30);
    let p_a_old = signed_partial(&WrkAssignmentForBuilder(&asn_a_old), &contrib_a);

    let supersession = signed_supersession(
        &session,
        &coord,
        vec![asn_a_old.assignment_id.clone()],
        vec![asn_a_new.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );

    // Aggregate names asn_a_old's partial — that's the superseded
    // assignment. Must be rejected.
    let aggregate = signed_aggregate(&session, &[(&asn_a_old, &p_a_old)], &coord);

    let outcome = verify_aggregated_result_with_supersessions(
        &session,
        &joins,
        &[asn_a_old.clone(), asn_a_new],
        &[supersession],
        &[p_a_old],
        &aggregate,
    );
    assert!(
        matches!(
            outcome,
            SessionVerifyOutcome::AggregatePartialRefSuperseded { ref assignment_id }
                if assignment_id == &asn_a_old.assignment_id
        ),
        "{outcome:?}"
    );
}

// ── 5. Replacement assignment without its partial fails ────────

#[test]
fn replacement_assignment_must_have_a_partial() {
    // The aggregate schema requires `partial_refs` non-empty, so
    // we set up a session with an UNRELATED stage 1 that DOES have
    // a partial — and the replacement (stage 0) that does NOT. The
    // aggregate references stage 1's partial only; the active set
    // = {replacement, stage_1}, and the replacement is missing.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    let joins =
        vec![signed_join(&session, &contrib_a), signed_join(&session, &contrib_b)];
    let asn_old = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0, 2);
    let asn_new = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0, 30);
    let asn_1 = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 1, 2);
    let p_1 = signed_partial(&WrkAssignmentForBuilder(&asn_1), &contrib_b);

    let supersession = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id.clone()],
        vec![asn_new.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );

    // Aggregate covers stage 1 but not the replacement at stage 0.
    let aggregate = signed_aggregate(&session, &[(&asn_1, &p_1)], &coord);

    let outcome = verify_aggregated_result_with_supersessions(
        &session,
        &joins,
        &[asn_old, asn_new.clone(), asn_1],
        &[supersession],
        &[p_1],
        &aggregate,
    );
    assert!(
        matches!(
            outcome,
            SessionVerifyOutcome::AggregateMissingPartialFor { ref assignment_id }
                if assignment_id == &asn_new.assignment_id
        ),
        "{outcome:?}"
    );
}

// ── 6. Tampered partial for a superseded assignment is skipped ─

/// `InvalidPartial` reason exists precisely so the coordinator can
/// declare a tampered partial superseded. The aggregate verifier
/// must skip that partial from re-verification — otherwise the
/// supersession can't repair the chain.
#[test]
fn tampered_partial_whose_assignment_is_superseded_is_skipped() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    let joins =
        vec![signed_join(&session, &contrib_a), signed_join(&session, &contrib_b)];
    let asn_old = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0, 2);
    let asn_new = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0, 30);

    // Build a tampered partial for the OLD assignment (flip a
    // hex digit in the contributor signature).
    let mut bad_partial =
        signed_partial(&WrkAssignmentForBuilder(&asn_old), &contrib_a);
    let mut sig: Vec<char> = bad_partial.contributor_signature_hex.chars().collect();
    let last = sig.len() - 1;
    sig[last] = if sig[last] == '0' { '1' } else { '0' };
    bad_partial.contributor_signature_hex = sig.into_iter().collect();

    let good_partial = signed_partial(&WrkAssignmentForBuilder(&asn_new), &contrib_b);

    let supersession = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id.clone()],
        vec![asn_new.assignment_id.clone()],
        SupersessionReason::InvalidPartial,
    );

    let aggregate = signed_aggregate(&session, &[(&asn_new, &good_partial)], &coord);

    let outcome = verify_aggregated_result_with_supersessions(
        &session,
        &joins,
        &[asn_old, asn_new],
        &[supersession],
        // The tampered partial is in the slice — but its assignment
        // is superseded, so the verifier must skip it.
        &[bad_partial, good_partial],
        &aggregate,
    );
    assert!(matches!(outcome, SessionVerifyOutcome::Ok), "{outcome:?}");
}

// ── 7. Empty supersessions slice preserves Stage 12.3 contract ─

#[test]
fn empty_supersessions_slice_preserves_stage_12_3_contract() {
    // This is the same posture as
    // `aggregate_verifier_rejects_extra_assignment_without_partial`
    // from Stage 12.10 — except we go through the new
    // supersession-aware entry point with an empty slice. Both
    // entry points must reject the extra assignment identically.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    let joins =
        vec![signed_join(&session, &contrib_a), signed_join(&session, &contrib_b)];
    let asn_a = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0, 2);
    let asn_b_extra = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0, 30);
    let p_a = signed_partial(&WrkAssignmentForBuilder(&asn_a), &contrib_a);
    let aggregate = signed_aggregate(&session, &[(&asn_a, &p_a)], &coord);

    let outcome = verify_aggregated_result_with_supersessions(
        &session,
        &joins,
        &[asn_a, asn_b_extra.clone()],
        &[], // empty supersessions → Stage 12.3 strict coverage
        &[p_a],
        &aggregate,
    );
    assert!(
        matches!(
            outcome,
            SessionVerifyOutcome::AggregateMissingPartialFor { ref assignment_id }
                if assignment_id == &asn_b_extra.assignment_id
        ),
        "{outcome:?}"
    );
}

// ── 8. Sanity: all active assignments still bind to joined pubkeys

/// Even with supersession, replacement assignments must target
/// joined contributors. A replacement whose pubkey is not in the
/// joined set should fail `verify_work_assignment` upstream, never
/// reaching the supersession check.
#[test]
fn replacement_assignment_pubkey_must_be_in_joined_set() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    // Only A is joined.
    let joins = vec![signed_join(&session, &contrib_a)];
    let asn_old = signed_assignment(&session, &coord, &contrib_a.pubkey_hex(), 0, 2);
    // Replacement contributor B did NOT join.
    let asn_new = signed_assignment(&session, &coord, &contrib_b.pubkey_hex(), 0, 30);
    let p_new = signed_partial(&WrkAssignmentForBuilder(&asn_new), &contrib_b);
    let supersession = signed_supersession(
        &session,
        &coord,
        vec![asn_old.assignment_id.clone()],
        vec![asn_new.assignment_id.clone()],
        SupersessionReason::MissingPartial,
    );
    let aggregate = signed_aggregate(&session, &[(&asn_new, &p_new)], &coord);
    let outcome = verify_aggregated_result_with_supersessions(
        &session,
        &joins,
        &[asn_old, asn_new],
        &[supersession],
        &[p_new],
        &aggregate,
    );
    // The replacement assignment fails `verify_work_assignment`
    // because B is not in the joined set — BEFORE any supersession
    // logic runs.
    assert!(
        matches!(
            outcome,
            SessionVerifyOutcome::BindingMismatch {
                field: "assignment.contributor_pubkey_hex_not_joined"
            }
        ),
        "{outcome:?}"
    );
}
