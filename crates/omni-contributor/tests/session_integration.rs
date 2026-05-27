//! Stage 12.3 — end-to-end session integration tests using
//! `InMemoryRelay` + `MockSnipStore`. Exercises the
//! open → join → assign → partial → aggregate → broadcast → verify
//! pipeline without any real networking.
//!
//! All paths run synchronously. No tokio runtime needed.

mod common;

use std::collections::HashSet;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{
        aggregated_result_signing_input, assignment_id_hex, canonical_partial_result_bytes,
        contributor_join_signing_input, execution_session_signing_input, hex_lower,
        net_aggregated_signing_input, net_assign_signing_input, net_join_signing_input,
        net_partial_signing_input, net_session_opened_signing_input,
        partial_result_signing_input, session_id_hex, work_assignment_signing_input,
    },
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    verify_aggregated_result, verify_contributor_join, verify_execution_session,
    verify_partial_result, verify_work_assignment, AggregatedContributorResult,
    AggregatedPartialRef, ContributorJoin, ContributorRelay, ContributorSigner,
    CoordinatorSigner, ExecutionSession, InMemoryRelay,
    NetworkAggregatedResultAnnouncement, NetworkContributorJoinedAnnouncement,
    NetworkPartialResultAnnouncement, NetworkSessionOpenedAnnouncement,
    NetworkWorkAssignedAnnouncement, PartialContributorResult, SessionVerifyOutcome,
    WorkAssignment, WorkKind, NET_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.3-int-coord-seed-32byte!";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.3-int-contrib-A-32bytes!";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.3-int-contrib-B-32bytes!";
const CONTRIB_C_SEED: [u8; 32] = *b"stage12.3-int-contrib-C-32bytes!";

fn signed_session(coord: &CoordinatorSigner) -> ExecutionSession {
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-05-27T00:00:00Z".into(),
        expires_at_utc: "2026-05-27T01:00:00Z".into(),
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
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 200_000,
        max_output_tokens: 1_000_000,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens, WorkUnitKind::Layers],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-27T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

fn signed_assignment(
    session: &ExecutionSession,
    contrib_pub: &str,
    stage_index: u32,
    work_kind: WorkKind,
    units: u64,
    coord: &CoordinatorSigner,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index,
        contributor_pubkey_hex: contrib_pub.to_string(),
        work_kind,
        expected_work_units: units,
        expected_work_unit_kind: WorkUnitKind::Layers,
        assigned_at_utc: format!("2026-05-27T00:00:0{stage_index}Z"),
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
    artifact_label: &str,
) -> PartialContributorResult {
    let mut p = PartialContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: assignment.session_id.clone(),
        assignment_id: assignment.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        partial_artifact_snip_root: format!("0x{}", "aa".repeat(32)),
        partial_artifact_hash: hex_lower(blake3::hash(artifact_label.as_bytes()).as_bytes()),
        measured_accounting: MeasuredAccounting {
            tokenizer_hash: "44".repeat(32),
            input_token_count: 100,
            output_token_count: 0,
            total_base_units: 100,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: contrib.pubkey_hex(),
                stage_label: artifact_label.into(),
                work_unit_kind: assignment.expected_work_unit_kind,
                work_units: assignment.expected_work_units,
            }],
        },
        produced_at_utc: format!("2026-05-27T00:00:1{}Z", assignment.stage_index),
        contributor_signature_hex: String::new(),
    };
    let si = partial_result_signing_input(&p).unwrap();
    p.contributor_signature_hex = contrib.sign_hex(&si);
    p
}

fn signed_aggregate(
    session: &ExecutionSession,
    assignments: &[WorkAssignment],
    partials: &[PartialContributorResult],
    coord: &CoordinatorSigner,
) -> AggregatedContributorResult {
    let mut partial_refs: Vec<AggregatedPartialRef> = assignments
        .iter()
        .zip(partials.iter())
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
        aggregated_at_utc: "2026-05-27T00:00:30Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let si = aggregated_result_signing_input(&g).unwrap();
    g.coordinator_signature_hex = coord.sign_hex(&si);
    g
}

// ── Happy path: 1-of-1 session ───────────────────────────────────────────

#[test]
fn session_one_of_one_e2e_publish_via_inmemory_relay() {
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();

    // 1. Coordinator opens the session.
    let session = signed_session(&coord);
    let session_root = snip.insert_bytes(&serde_json::to_vec(&session).unwrap());
    let opened = {
        let mut a = NetworkSessionOpenedAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            execution_session_snip_root: session_root.to_hex(),
            session_id: session.session_id.clone(),
            posted_id: session.posted_id.clone(),
            announced_at_utc: "2026-05-27T00:00:00Z".into(),
            announcer_pubkey_hex: coord.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let si = net_session_opened_signing_input(&a).unwrap();
        a.announcer_signature_hex = coord.sign_hex(&si);
        a
    };
    relay.publish_session_opened(&opened).unwrap();

    // 2. Contributor joins.
    let join = signed_join(&session, &contrib);
    let join_root = snip.insert_bytes(&serde_json::to_vec(&join).unwrap());
    let joined_ann = {
        let mut a = NetworkContributorJoinedAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            contributor_join_snip_root: join_root.to_hex(),
            session_id: session.session_id.clone(),
            contributor_pubkey_hex: contrib.pubkey_hex(),
            announced_at_utc: "2026-05-27T00:00:01Z".into(),
            announcer_pubkey_hex: contrib.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let si = net_join_signing_input(&a).unwrap();
        a.announcer_signature_hex = contrib.sign_hex(&si);
        a
    };
    relay.publish_contributor_joined(&joined_ann).unwrap();

    // 3. Coordinator assigns.
    let assignment = signed_assignment(
        &session,
        &contrib.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 32 },
        32,
        &coord,
    );
    let asn_root = snip.insert_bytes(&serde_json::to_vec(&assignment).unwrap());
    let asn_ann = {
        let mut a = NetworkWorkAssignedAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            work_assignment_snip_root: asn_root.to_hex(),
            session_id: session.session_id.clone(),
            assignment_id: assignment.assignment_id.clone(),
            contributor_pubkey_hex: contrib.pubkey_hex(),
            announced_at_utc: "2026-05-27T00:00:02Z".into(),
            announcer_pubkey_hex: coord.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let si = net_assign_signing_input(&a).unwrap();
        a.announcer_signature_hex = coord.sign_hex(&si);
        a
    };
    relay.publish_work_assigned(&asn_ann).unwrap();

    // 4. Contributor publishes partial.
    let partial = signed_partial(&assignment, &contrib, "single-stage");
    let par_root = snip.insert_bytes(&serde_json::to_vec(&partial).unwrap());
    let par_ann = {
        let mut a = NetworkPartialResultAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            partial_result_snip_root: par_root.to_hex(),
            session_id: session.session_id.clone(),
            assignment_id: assignment.assignment_id.clone(),
            contributor_pubkey_hex: contrib.pubkey_hex(),
            announced_at_utc: "2026-05-27T00:00:11Z".into(),
            announcer_pubkey_hex: contrib.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let si = net_partial_signing_input(&a).unwrap();
        a.announcer_signature_hex = contrib.sign_hex(&si);
        a
    };
    relay.publish_partial_result(&par_ann).unwrap();

    // 5. Coordinator aggregates.
    let aggregate = signed_aggregate(&session, &[assignment.clone()], &[partial.clone()], &coord);
    let agg_root = snip.insert_bytes(&serde_json::to_vec(&aggregate).unwrap());
    let agg_ann = {
        let mut a = NetworkAggregatedResultAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            aggregated_result_snip_root: agg_root.to_hex(),
            session_id: session.session_id.clone(),
            posted_id: session.posted_id.clone(),
            announced_at_utc: "2026-05-27T00:00:30Z".into(),
            announcer_pubkey_hex: coord.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let si = net_aggregated_signing_input(&a).unwrap();
        a.announcer_signature_hex = coord.sign_hex(&si);
        a
    };
    relay.publish_aggregated_result(&agg_ann).unwrap();

    // 6. Drain + verify each topic.
    assert_eq!(relay.poll_sessions_opened().unwrap().len(), 1);
    assert_eq!(relay.poll_contributors_joined().unwrap().len(), 1);
    assert_eq!(relay.poll_work_assigned().unwrap().len(), 1);
    assert_eq!(relay.poll_partial_results().unwrap().len(), 1);
    assert_eq!(relay.poll_aggregated_results().unwrap().len(), 1);

    // 7. Run the structural verifier chain.
    assert!(verify_execution_session(&session).is_ok());
    assert!(verify_contributor_join(&session, &join).is_ok());
    let mut joined_set = HashSet::new();
    joined_set.insert(contrib.pubkey_hex());
    assert!(verify_work_assignment(&session, &joined_set, &assignment).is_ok());
    assert!(verify_partial_result(&assignment, &partial).is_ok());
    let out =
        verify_aggregated_result(&session, &[join], &[assignment], &[partial], &aggregate);
    assert!(out.is_ok(), "{out:?}");
}

// ── Happy path: 3-stage pipeline (the RAM-pooling shape) ─────────────────

#[test]
fn session_three_stage_pipeline_aggregates_in_stage_order() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let b_signer = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let c_signer = ContributorSigner::from_seed_bytes(&CONTRIB_C_SEED).unwrap();
    let session = signed_session(&coord);
    let joins = [
        signed_join(&session, &a_signer),
        signed_join(&session, &b_signer),
        signed_join(&session, &c_signer),
    ];
    let assignments = [
        signed_assignment(
            &session,
            &a_signer.pubkey_hex(),
            0,
            WorkKind::Layers { start: 0, end: 16 },
            16,
            &coord,
        ),
        signed_assignment(
            &session,
            &b_signer.pubkey_hex(),
            1,
            WorkKind::Layers { start: 16, end: 24 },
            8,
            &coord,
        ),
        signed_assignment(
            &session,
            &c_signer.pubkey_hex(),
            2,
            WorkKind::Layers { start: 24, end: 32 },
            8,
            &coord,
        ),
    ];
    let partials = [
        signed_partial(&assignments[0], &a_signer, "stage-0"),
        signed_partial(&assignments[1], &b_signer, "stage-1"),
        signed_partial(&assignments[2], &c_signer, "stage-2"),
    ];

    // Verifier chain.
    assert!(verify_execution_session(&session).is_ok());
    let mut joined_pubkeys = HashSet::new();
    for j in &joins {
        assert!(verify_contributor_join(&session, j).is_ok());
        joined_pubkeys.insert(j.contributor_pubkey_hex.clone());
    }
    for a in &assignments {
        assert!(verify_work_assignment(&session, &joined_pubkeys, a).is_ok());
    }
    for (a, p) in assignments.iter().zip(partials.iter()) {
        assert!(verify_partial_result(a, p).is_ok());
    }

    let aggregate = signed_aggregate(&session, &assignments, &partials, &coord);
    // Aggregate's partial_refs must be sorted by stage_index.
    for (i, r) in aggregate.partial_refs.iter().enumerate() {
        assert_eq!(r.stage_index, i as u32);
    }
    let out =
        verify_aggregated_result(&session, &joins, &assignments, &partials, &aggregate);
    assert!(out.is_ok(), "{out:?}");
}

// ── Negative: assignment to non-joined contributor ───────────────────────

#[test]
fn assignment_to_non_joined_contributor_refused_in_verifier() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let b_signer = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    let join_a = signed_join(&session, &a_signer);
    // assignment targets B but only A joined.
    let asn = signed_assignment(
        &session,
        &b_signer.pubkey_hex(),
        0,
        WorkKind::Prefill,
        100,
        &coord,
    );
    let mut joined = HashSet::new();
    joined.insert(join_a.contributor_pubkey_hex.clone());
    let out = verify_work_assignment(&session, &joined, &asn);
    assert!(
        matches!(out, SessionVerifyOutcome::BindingMismatch { .. }),
        "{out:?}"
    );
}

// ── Negative: partial signed by wrong contributor ─────────────────────────

#[test]
fn partial_signed_by_non_assignee_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let b_signer = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    let asn = signed_assignment(
        &session,
        &a_signer.pubkey_hex(),
        0,
        WorkKind::Prefill,
        100,
        &coord,
    );
    // B signs a partial against A's assignment.
    let bad = signed_partial(&asn, &b_signer, "wrong-signer");
    let out = verify_partial_result(&asn, &bad);
    assert!(
        matches!(out, SessionVerifyOutcome::BindingMismatch { .. }),
        "{out:?}"
    );
}

// ── Negative: aggregate missing partial for one assignment ───────────────

#[test]
fn aggregate_missing_one_partial_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let b_signer = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord);
    let asn_a = signed_assignment(
        &session,
        &a_signer.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 16 },
        16,
        &coord,
    );
    let asn_b = signed_assignment(
        &session,
        &b_signer.pubkey_hex(),
        1,
        WorkKind::Layers { start: 16, end: 32 },
        16,
        &coord,
    );
    // Only one partial.
    let join_a = signed_join(&session, &a_signer);
    let join_b = signed_join(&session, &b_signer);
    let par_a = signed_partial(&asn_a, &a_signer, "stage-0");
    let aggregate = signed_aggregate(&session, &[asn_a.clone()], &[par_a.clone()], &coord);
    let out = verify_aggregated_result(
        &session,
        &[join_a, join_b],
        &[asn_a, asn_b],
        &[par_a],
        &aggregate,
    );
    assert!(
        matches!(out, SessionVerifyOutcome::AggregateMissingPartialFor { .. }),
        "{out:?}"
    );
}

// ── Negative: expired session rejected at every downstream check ─────────

#[test]
fn expired_session_rejected_by_check_not_expired() {
    let out = omni_contributor::check_not_expired(
        "2026-05-27T02:00:00Z",
        "2026-05-27T01:00:00Z",
    );
    assert!(
        matches!(out, SessionVerifyOutcome::ExpiredAtCheck { .. }),
        "{out:?}"
    );
}

// ── Negative: coordinator key swap mid-session ───────────────────────────

#[test]
fn assignment_signed_by_different_coordinator_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(b"rogue-coord-seed-32-bytes-key!!!").unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord);
    // Assignment signed by rogue, not session's coordinator.
    let asn = signed_assignment(
        &session,
        &a_signer.pubkey_hex(),
        0,
        WorkKind::Prefill,
        100,
        &rogue,
    );
    let mut joined = HashSet::new();
    joined.insert(a_signer.pubkey_hex());
    let out = verify_work_assignment(&session, &joined, &asn);
    assert!(
        matches!(out, SessionVerifyOutcome::CoordinatorSignatureFailed),
        "{out:?}"
    );
}

// ── Negative: aggregate with wrong coordinator pubkey ────────────────────

#[test]
fn aggregate_coordinator_pubkey_mismatch_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(b"rogue-coord-seed-32-bytes-key!!!").unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord);
    let asn = signed_assignment(
        &session,
        &a_signer.pubkey_hex(),
        0,
        WorkKind::Prefill,
        100,
        &coord,
    );
    let par = signed_partial(&asn, &a_signer, "x");
    let join_a = signed_join(&session, &a_signer);
    let aggregate = signed_aggregate(&session, &[asn.clone()], &[par.clone()], &rogue);
    let out = verify_aggregated_result(&session, &[join_a], &[asn], &[par], &aggregate);
    assert!(
        matches!(out, SessionVerifyOutcome::AggregateCoordinatorMismatch),
        "{out:?}"
    );
}

// ── Posture: WorkKind::Custom round-trips through canonical bytes ────────

#[test]
fn workkind_custom_round_trips_through_canonical_bytes() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord);
    let asn = signed_assignment(
        &session,
        &a_signer.pubkey_hex(),
        0,
        WorkKind::Custom {
            label: "kv-cache-shard".into(),
        },
        100,
        &coord,
    );
    let mut joined = HashSet::new();
    joined.insert(a_signer.pubkey_hex());
    let out = verify_work_assignment(&session, &joined, &asn);
    assert!(out.is_ok(), "{out:?}");
}
