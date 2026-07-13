//! Stage 12.9 — unit + integration tests for the local
//! session-status reporter.
//!
//! Each test writes a controlled set of artifacts into a tempdir
//! `ContributorStateStore`, runs `build_session_status_report`,
//! and asserts on the resulting `SessionStatusReport`. The
//! reporter is read-only, so these tests double as a fixture for
//! the documented overall-status decision tree.

use std::fs;

use omni_contributor::{
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower,
        partial_result_signing_input, peer_advertisement_signing_input,
        session_id_hex, work_assignment_signing_input,
    },
    handoff::TensorDtype,
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    session::{
        AggregatedContributorResult, AggregatedPartialRef,
        PartialContributorResult, WorkAssignment, WorkKind,
    },
    ContributorJoin, ContributorPeerAdvertisement, ContributorSigner,
    ContributorStateStore, CoordinatorSigner, ExecutionSession,
    PeerCapabilities, StateNamespace, StateObjectKind,
    PEER_ADVERTISEMENT_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
};
use omni_ops::{
    build_session_status_report, InvalidArtifactStatus, SessionOverallStatus,
    SessionStatusReport, STATUS_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.9-status-coord-seed-32b!";
const COORD_WRONG_SEED: [u8; 32] = *b"stage12.9-status-wrong-coord-32!";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.9-status-contrib-a-seed!";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.9-status-contrib-b-seed!";
const NOW_UTC: &str = "2026-05-31T00:30:00Z";
const FAR_FUTURE: &str = "2026-06-01T00:00:00Z";

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
        created_at_utc: "2026-05-31T00:00:00Z".into(),
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
        joined_at_utc: "2026-05-31T00:00:01Z".into(),
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
    work_kind: WorkKind,
    expected_work_units: u64,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index,
        contributor_pubkey_hex: contributor_pubkey_hex.to_string(),
        work_kind,
        expected_work_units,
        expected_work_unit_kind: WorkUnitKind::Layers,
        assigned_at_utc: "2026-05-31T00:00:02Z".into(),
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
        produced_at_utc: format!("2026-05-31T00:00:1{}Z", assignment.stage_index),
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
        aggregated_at_utc: "2026-05-31T00:00:30Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let si = aggregated_result_signing_input(&g).unwrap();
    g.coordinator_signature_hex = coord.sign_hex(&si);
    g
}

fn signed_peer_advert(
    session: &ExecutionSession,
    contrib: &ContributorSigner,
    advertised_at_utc: &str,
    expires_at_utc: &str,
) -> ContributorPeerAdvertisement {
    let mut a = ContributorPeerAdvertisement {
        schema_version: PEER_ADVERTISEMENT_SCHEMA_VERSION,
        advertisement_id: String::new(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        libp2p_peer_id: "12D3KooWGjMyD3v8UYjK3bV7Sqg3WcWzkF7QbTtwYNRsBwCb6KpC".into(),
        listen_multiaddrs: vec![],
        capabilities: PeerCapabilities {
            supports_live_handoff: true,
            max_handoff_chunk_bytes: 32 * 1024 * 1024,
            supported_dtypes: vec![TensorDtype::F16],
        },
        advertised_at_utc: advertised_at_utc.into(),
        expires_at_utc: expires_at_utc.into(),
        contributor_signature_hex: String::new(),
    };
    a.advertisement_id = omni_contributor::canonical::advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    a
}

fn write_session_chain(
    store: &ContributorStateStore,
    session: &ExecutionSession,
    joins: &[ContributorJoin],
    assignments: &[WorkAssignment],
    partials: &[PartialContributorResult],
    aggregate: Option<&AggregatedContributorResult>,
    peer_adverts: &[ContributorPeerAdvertisement],
) {
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, session)
        .unwrap();
    for j in joins {
        store
            .write_verified_json(
                StateObjectKind::Join {
                    session_id: session.session_id.clone(),
                },
                &j.contributor_pubkey_hex,
                j,
            )
            .unwrap();
    }
    for a in assignments {
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
    for p in partials {
        store
            .write_verified_json(
                StateObjectKind::Partial {
                    session_id: session.session_id.clone(),
                },
                &p.assignment_id,
                p,
            )
            .unwrap();
    }
    if let Some(agg) = aggregate {
        store
            .write_verified_json(
                StateObjectKind::Aggregate,
                &session.session_id,
                agg,
            )
            .unwrap();
    }
    for advert in peer_adverts {
        store
            .write_verified_json(
                StateObjectKind::PeerAdvert {
                    session_id: session.session_id.clone(),
                },
                &advert.contributor_pubkey_hex,
                advert,
            )
            .unwrap();
    }
}

// ── 1. NoSession ─────────────────────────────────────────────────

#[test]
fn report_no_session_when_missing() {
    let (_d, store) = fresh_store();
    let report =
        build_session_status_report(&store, &"de".repeat(32), NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::NoSession);
    assert_eq!(report.session_id, "de".repeat(32));
    assert!(report.posted_id.is_none());
    assert!(report.notes.is_empty());
    assert_eq!(report.schema_version, STATUS_SCHEMA_VERSION);
}

// ── 2. NoAssignments ─────────────────────────────────────────────

#[test]
fn report_no_assignments_for_valid_session() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join = signed_join(&session, &contrib);
    write_session_chain(&store, &session, &[join], &[], &[], None, &[]);
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::NoAssignments
    );
    assert_eq!(report.join_count, 1);
    assert_eq!(report.assignment_count, 0);
    assert!(report.notes.is_empty());
}

// ── 3. InProgress ───────────────────────────────────────────────

#[test]
fn report_in_progress_with_missing_partials() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let join_b = signed_join(&session, &contrib_b);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 16 },
        16,
    );
    let asn_1 = signed_assignment(
        &session,
        &coord,
        &contrib_b.pubkey_hex(),
        1,
        WorkKind::Layers {
            start: 16,
            end: 32,
        },
        16,
    );
    let p_0 = signed_partial(&asn_0, &contrib_a, "stage0");
    write_session_chain(
        &store,
        &session,
        &[join_a, join_b],
        &[asn_0.clone(), asn_1.clone()],
        &[p_0],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::InProgress);
    assert_eq!(report.assignment_count, 2);
    assert_eq!(report.partial_count, 1);
    assert_eq!(report.missing_assignment_ids, vec![asn_1.assignment_id]);
    assert!(report.duplicate_partial_assignment_ids.is_empty());
}

// ── 4. CompletePartials ──────────────────────────────────────────

#[test]
fn report_complete_partials_when_all_assignments_have_valid_partials() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let join_b = signed_join(&session, &contrib_b);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 16 },
        16,
    );
    let asn_1 = signed_assignment(
        &session,
        &coord,
        &contrib_b.pubkey_hex(),
        1,
        WorkKind::Layers {
            start: 16,
            end: 32,
        },
        16,
    );
    let p_0 = signed_partial(&asn_0, &contrib_a, "stage0");
    let p_1 = signed_partial(&asn_1, &contrib_b, "stage1");
    write_session_chain(
        &store,
        &session,
        &[join_a, join_b],
        &[asn_0, asn_1],
        &[p_0, p_1],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::CompletePartials
    );
    assert_eq!(report.assignment_count, 2);
    assert_eq!(report.partial_count, 2);
    assert!(report.missing_assignment_ids.is_empty());
    assert!(!report.aggregate_present);
    for a in &report.assignments {
        assert!(a.partial_present);
        assert!(a.partial_valid);
        assert!(a.partial_snip_root.is_some());
    }
}

// ── 5. Aggregated ───────────────────────────────────────────────

#[test]
fn report_aggregated_when_valid_aggregate_present() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let join_b = signed_join(&session, &contrib_b);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 16 },
        16,
    );
    let asn_1 = signed_assignment(
        &session,
        &coord,
        &contrib_b.pubkey_hex(),
        1,
        WorkKind::Layers {
            start: 16,
            end: 32,
        },
        16,
    );
    let p_0 = signed_partial(&asn_0, &contrib_a, "stage0");
    let p_1 = signed_partial(&asn_1, &contrib_b, "stage1");
    let aggregate = signed_aggregate(
        &session,
        &[asn_0.clone(), asn_1.clone()],
        &[p_0.clone(), p_1.clone()],
        &coord,
    );
    write_session_chain(
        &store,
        &session,
        &[join_a, join_b],
        &[asn_0, asn_1],
        &[p_0, p_1],
        Some(&aggregate),
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::Aggregated);
    assert!(report.aggregate_present);
    assert!(report.aggregate_valid);
    assert!(report.notes.is_empty());
}

// ── 6. InvalidState — tampered assignment coord signature ───────

#[test]
fn report_invalid_state_when_assignment_signed_by_wrong_coordinator() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let wrong_coord = CoordinatorSigner::from_seed_bytes(&COORD_WRONG_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    // Build an assignment signed by the WRONG coordinator. The
    // session's coordinator_pubkey_hex is the real coord, so
    // verify_work_assignment must fail.
    let bad_assignment = signed_assignment(
        &session,
        &wrong_coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    write_session_chain(
        &store,
        &session,
        &[join_a],
        &[bad_assignment.clone()],
        &[],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::InvalidState
    );
    // The tampered assignment is dropped from valid counts.
    assert_eq!(report.assignment_count, 0);
    // A note explains why.
    assert!(report
        .notes
        .iter()
        .any(|n| n.contains(&bad_assignment.assignment_id)));
}

// ── 7. InvalidState — tampered partial contributor signature ────

#[test]
fn report_invalid_state_when_partial_signed_by_wrong_contributor() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    // Partial signed by the WRONG contributor.
    let bad_partial = signed_partial(&asn_0, &contrib_b, "stage0-forged");
    write_session_chain(
        &store,
        &session,
        &[join_a],
        &[asn_0.clone()],
        &[bad_partial.clone()],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::InvalidState
    );
    assert_eq!(report.partial_count, 0);
    // The dropped partial surfaces as missing.
    assert_eq!(report.missing_assignment_ids, vec![asn_0.assignment_id]);
    assert!(report
        .notes
        .iter()
        .any(|n| n.contains("verify_partial_result")));
}

// ── 8. ExpiredIncomplete ────────────────────────────────────────

#[test]
fn report_expired_incomplete() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    // Session expires BEFORE NOW_UTC.
    let session = signed_session(&coord, "2026-05-31T00:00:00Z");
    let join_a = signed_join(&session, &contrib_a);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    write_session_chain(
        &store,
        &session,
        &[join_a],
        &[asn_0],
        &[],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::ExpiredIncomplete
    );
    assert!(report.session_expired);
}

// ── 9. Re-verify forged join from state-dir ─────────────────────

#[test]
fn report_reverifies_forged_join_from_state_dir() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let mut bad_join = signed_join(&session, &contrib_a);
    // Flip a hex digit in the contributor signature.
    let mut sig: Vec<char> = bad_join.contributor_signature_hex.chars().collect();
    let last = sig.len() - 1;
    sig[last] = if sig[last] == '0' { '1' } else { '0' };
    bad_join.contributor_signature_hex = sig.into_iter().collect();
    write_session_chain(
        &store,
        &session,
        &[bad_join],
        &[],
        &[],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report.join_count, 0);
    // Forged join surfaces in notes and trips InvalidState.
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::InvalidState
    );
    assert!(report
        .notes
        .iter()
        .any(|n| n.contains("verify_contributor_join")));
}

// ── 10. JSON round-trip ─────────────────────────────────────────

#[test]
fn report_json_roundtrip() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    write_session_chain(&store, &session, &[join_a], &[], &[], None, &[]);
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    let json = serde_json::to_vec_pretty(&report).unwrap();
    let back: SessionStatusReport = serde_json::from_slice(&json).unwrap();
    assert_eq!(back, report);
    assert_eq!(back.schema_version, STATUS_SCHEMA_VERSION);
}

// ── 11. Peer-advert filtering: expired vs --include-expired ────

#[test]
fn report_peer_advert_count_filters_expired_unless_include_expired() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let join_b = signed_join(&session, &contrib_b);
    // Fresh: advertised at NOW_UTC, expires 12h later (well
    // within the 24h schema cap). NOW_UTC = 2026-05-31T00:30:00Z,
    // so a 12h window covers all of report time.
    let fresh_advert = signed_peer_advert(
        &session,
        &contrib_a,
        "2026-05-31T00:00:00Z",
        "2026-05-31T12:00:00Z",
    );
    // Expired-but-schema-valid: advertised 24h before NOW,
    // expired before NOW. Inside the 24h max-lifetime cap.
    let expired_advert = signed_peer_advert(
        &session,
        &contrib_b,
        "2026-05-30T00:00:00Z",
        "2026-05-30T12:00:00Z",
    );
    write_session_chain(
        &store,
        &session,
        &[join_a, join_b],
        &[],
        &[],
        None,
        &[fresh_advert, expired_advert],
    );
    let report_default =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report_default.peer_advert_count, 1);
    // The expired advert produces a note but does NOT make the
    // overall status InvalidState — adverts are routing helpers,
    // not chain links.
    assert_eq!(
        report_default.overall_status,
        SessionOverallStatus::NoAssignments,
        "notes={:?}",
        report_default.notes
    );

    let report_include =
        build_session_status_report(&store, &session.session_id, NOW_UTC, true).unwrap();
    assert_eq!(report_include.peer_advert_count, 2);
}

// ── 12. Aggregate body present but invalid ─────────────────────

#[test]
fn report_aggregate_present_but_invalid_yields_invalid_state() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    let p_0 = signed_partial(&asn_0, &contrib_a, "stage0");
    let mut bad_aggregate =
        signed_aggregate(&session, &[asn_0.clone()], &[p_0.clone()], &coord);
    // Flip the coordinator signature so the full-chain verifier
    // rejects.
    let mut sig: Vec<char> = bad_aggregate.coordinator_signature_hex.chars().collect();
    let last = sig.len() - 1;
    sig[last] = if sig[last] == '0' { '1' } else { '0' };
    bad_aggregate.coordinator_signature_hex = sig.into_iter().collect();
    write_session_chain(
        &store,
        &session,
        &[join_a],
        &[asn_0],
        &[p_0],
        Some(&bad_aggregate),
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert!(report.aggregate_present);
    assert!(!report.aggregate_valid);
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::InvalidState
    );
    assert!(report
        .notes
        .iter()
        .any(|n| n.contains("verify_aggregated_result")));
}

// ── 13. Tampered session signature → fail-closed InvalidState ──

/// A tampered session.json must NOT be used as verifier context for
/// downstream joins/assignments/partials. The report must carry NO
/// session-derived trust fields (posted_id, model_hash,
/// session_expires_at_utc), report counts of zero, and surface a
/// single note pointing at the failed verifier.
#[test]
fn report_fails_closed_when_session_signature_invalid() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let mut session = signed_session(&coord, FAR_FUTURE);
    // Flip a hex digit in the coordinator signature.
    let mut sig: Vec<char> = session.coordinator_signature_hex.chars().collect();
    let last = sig.len() - 1;
    sig[last] = if sig[last] == '0' { '1' } else { '0' };
    session.coordinator_signature_hex = sig.into_iter().collect();
    // Even though we'll lay down joins/assignments/partials below,
    // they MUST NOT be re-verified using the tampered session as
    // context — the report should fail-closed before getting that
    // far.
    let join_a = signed_join(&session, &contrib_a);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    let p_0 = signed_partial(&asn_0, &contrib_a, "stage0");
    write_session_chain(
        &store,
        &session,
        &[join_a],
        &[asn_0],
        &[p_0],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false)
            .unwrap();
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::InvalidState,
        "tampered session must produce InvalidState"
    );
    // No session-derived trust fields.
    assert!(
        report.posted_id.is_none(),
        "tampered session must NOT leak posted_id into the report"
    );
    assert!(
        report.model_hash.is_none(),
        "tampered session must NOT leak model_hash"
    );
    assert!(
        report.session_expires_at_utc.is_none(),
        "tampered session must NOT leak expires_at_utc"
    );
    // No counts from downstream artifacts — we fail-closed before
    // verifying them against an untrusted session.
    assert_eq!(report.join_count, 0);
    assert_eq!(report.assignment_count, 0);
    assert_eq!(report.partial_count, 0);
    assert!(report.assignments.is_empty());
    // Single note explaining why.
    assert_eq!(report.notes.len(), 1);
    assert!(report
        .notes
        .iter()
        .any(|n| n.contains("verify_execution_session")));
}

// ── 14. session.json present but session_id directory mismatches ─

/// Defense-in-depth: the state-dir writer keys by session_id and
/// the loader trusts the inner body's session_id. If they disagree
/// (corruption or hand-edit) the reporter treats the session as
/// missing rather than reporting a confused chain.
#[test]
fn report_treats_session_id_mismatch_as_missing() {
    let (d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    // Write the session under a DIFFERENT directory key (simulate a
    // stale file). The verifier path then sees an inner session_id
    // that disagrees with the directory key.
    let fake_id = "ee".repeat(32);
    let path = d
        .path()
        .join("verified")
        .join("sessions")
        .join(&fake_id);
    fs::create_dir_all(&path).unwrap();
    fs::write(
        path.join("session.json"),
        serde_json::to_vec_pretty(&session).unwrap(),
    )
    .unwrap();
    let _ = StateNamespace::Sessions; // silence unused-import linter
    let report = build_session_status_report(&store, &fake_id, NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::NoSession);
}

// ── Stage 12.11 regression — tampered partial under InvalidPartial
// supersession must NOT flip overall_status to InvalidState. ───────

/// The Stage 12.11 review caught this directly: the status reporter
/// was loading partials BEFORE supersessions, so a tampered partial
/// for an assignment later superseded with `InvalidPartial` would
/// set `any_chain_invalid = true` and stick the session at
/// `InvalidState` forever, defeating the whole point of the
/// supersession. The fix loads supersessions FIRST and skips
/// partials whose assignment is superseded. This regression test
/// pins that fix.
#[test]
fn tampered_partial_under_invalid_partial_supersession_is_not_invalid_state() {
    use omni_contributor::{
        canonical::{supersession_id_hex, work_assignment_supersession_signing_input},
        supersession::{SupersessionReason, WorkAssignmentSupersession},
        SUPERSESSION_SCHEMA_VERSION,
    };

    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let join_b = signed_join(&session, &contrib_b);

    // Original stage 0 — assigned to A. A delivered a tampered
    // partial. The operator superseded with `InvalidPartial` and
    // reassigned to B.
    let asn_old = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    // Use a different `assigned_at_utc` so the replacement hashes
    // to a different `assignment_id` even though stage_index +
    // work_kind match.
    let asn_new = {
        let mut a = WorkAssignment {
            schema_version: SESSION_SCHEMA_VERSION,
            session_id: session.session_id.clone(),
            assignment_id: String::new(),
            stage_index: 0,
            contributor_pubkey_hex: contrib_b.pubkey_hex(),
            work_kind: WorkKind::Layers { start: 0, end: 8 },
            expected_work_units: 8,
            expected_work_unit_kind: WorkUnitKind::Layers,
            assigned_at_utc: "2026-05-31T00:00:30Z".into(),
            coordinator_signature_hex: String::new(),
        };
        a.assignment_id = assignment_id_hex(&a).unwrap();
        let si = work_assignment_signing_input(&a).unwrap();
        a.coordinator_signature_hex = coord.sign_hex(&si);
        a
    };

    // Tampered partial for the OLD assignment.
    let mut bad_partial = signed_partial(&asn_old, &contrib_a, "stage0-forged");
    let mut sig: Vec<char> = bad_partial.contributor_signature_hex.chars().collect();
    let last = sig.len() - 1;
    sig[last] = if sig[last] == '0' { '1' } else { '0' };
    bad_partial.contributor_signature_hex = sig.into_iter().collect();

    // Valid partial for the REPLACEMENT.
    let good_partial = signed_partial(&asn_new, &contrib_b, "stage0-replacement");

    // Coordinator-signed supersession: old → new, reason InvalidPartial.
    let mut superseded_ids = vec![asn_old.assignment_id.clone()];
    superseded_ids.sort();
    let mut replacement_ids = vec![asn_new.assignment_id.clone()];
    replacement_ids.sort();
    let mut s = WorkAssignmentSupersession {
        schema_version: SUPERSESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        supersession_id: String::new(),
        superseded_assignment_ids: superseded_ids,
        replacement_assignment_ids: replacement_ids,
        reason: SupersessionReason::InvalidPartial,
        created_at_utc: "2026-05-31T00:30:00Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    s.supersession_id = supersession_id_hex(&s).unwrap();
    let si = work_assignment_supersession_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);

    // Write everything to the state-dir.
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    for j in [&join_a, &join_b] {
        store
            .write_verified_json(
                StateObjectKind::Join {
                    session_id: session.session_id.clone(),
                },
                &j.contributor_pubkey_hex,
                j,
            )
            .unwrap();
    }
    for a in [&asn_old, &asn_new] {
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
    store
        .write_verified_json(
            StateObjectKind::Partial {
                session_id: session.session_id.clone(),
            },
            &bad_partial.assignment_id,
            &bad_partial,
        )
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Partial {
                session_id: session.session_id.clone(),
            },
            &good_partial.assignment_id,
            &good_partial,
        )
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::AssignmentSupersession {
                session_id: session.session_id.clone(),
            },
            &s.supersession_id,
            &s,
        )
        .unwrap();

    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();

    // The CORE assertion: the tampered partial for the SUPERSEDED
    // assignment must NOT flip the overall status to InvalidState.
    assert_ne!(
        report.overall_status,
        SessionOverallStatus::InvalidState,
        "tampered partial whose assignment is superseded with \
         InvalidPartial must NOT make the session InvalidState; \
         notes={:?}",
        report.notes
    );
    // Reported counts:
    //   - assignment_count = 2 (old + replacement both verify on
    //     their own bytes)
    //   - superseded_assignment_count = 1 (old)
    //   - active_assignment_count = 1 (replacement)
    //   - partial_count = 1 (the good one; the bad one is skipped)
    //   - missing_assignment_ids is empty (active replacement has
    //     its partial)
    assert_eq!(report.assignment_count, 2);
    assert_eq!(report.superseded_assignment_count, 1);
    assert_eq!(report.active_assignment_count, 1);
    assert_eq!(report.partial_count, 1);
    assert!(report.missing_assignment_ids.is_empty());
    // No aggregate yet → CompletePartials.
    assert_eq!(report.overall_status, SessionOverallStatus::CompletePartials);
    // Supersession is in the report and valid.
    assert_eq!(report.supersession_count, 1);
    assert!(report.supersessions[0].valid);
}

// ── Stage 12.12 — structured `invalid_artifacts` diagnostics ─────
//
// The reporter must publish a typed entry for every chain-link
// failure so automation can decide whether an `InvalidState` is
// triagable via `--reason invalid-partial`. The `notes` free-form
// strings stay populated alongside — dashboards keep working —
// but automation reads `invalid_artifacts` only. Each test pins
// the variant + the stable `reason_tag` so a verifier-side rename
// trips immediately.

#[test]
fn invalid_artifacts_emit_invalid_partial_for_tampered_partial() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    // Forge the partial by signing with the WRONG contributor.
    let bad_partial = signed_partial(&asn_0, &contrib_b, "stage0-forged");
    write_session_chain(
        &store,
        &session,
        &[join_a],
        &[asn_0.clone()],
        &[bad_partial],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::InvalidState);
    let entry = report
        .invalid_artifacts
        .iter()
        .find(|e| matches!(e, InvalidArtifactStatus::InvalidPartial { .. }))
        .expect("expected at least one InvalidPartial entry");
    match entry {
        InvalidArtifactStatus::InvalidPartial {
            assignment_id,
            reason_tag,
        } => {
            assert_eq!(assignment_id, &asn_0.assignment_id);
            // The signed_partial fixture sets the partial's
            // contributor_pubkey_hex to the signer (contrib_b),
            // which does NOT match the assignment's
            // contributor_pubkey_hex (contrib_a) —
            // `verify_partial_result` rejects on the binding
            // leg before reaching the signature leg.
            assert_eq!(reason_tag, "BindingMismatch");
        }
        other => panic!("expected InvalidPartial, got {other:?}"),
    }
    // notes still populated alongside.
    assert!(!report.notes.is_empty());
    // Invariant holds.
    assert!(!report.invalid_artifacts.is_empty());
}

#[test]
fn invalid_artifacts_emit_invalid_join_for_forged_join() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    // Sign the join with the WRONG contributor key; its
    // `contributor_pubkey_hex` advertises contrib_a but the
    // signature is from contrib_b. `verify_contributor_join` fails.
    let mut forged_join = signed_join(&session, &contrib_a);
    let si = contributor_join_signing_input(&forged_join).unwrap();
    forged_join.contributor_signature_hex = contrib_b.sign_hex(&si);
    write_session_chain(
        &store,
        &session,
        &[forged_join.clone()],
        &[],
        &[],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::InvalidState);
    let entry = report
        .invalid_artifacts
        .iter()
        .find(|e| matches!(e, InvalidArtifactStatus::InvalidJoin { .. }))
        .expect("expected an InvalidJoin entry");
    match entry {
        InvalidArtifactStatus::InvalidJoin {
            contributor_pubkey_hex,
            reason_tag,
        } => {
            assert_eq!(contributor_pubkey_hex, &contrib_a.pubkey_hex());
            assert_eq!(reason_tag, "ContributorSignatureFailed");
        }
        other => panic!("expected InvalidJoin, got {other:?}"),
    }
}

#[test]
fn invalid_artifacts_emit_invalid_assignment_for_tampered_assignment() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let wrong_coord = CoordinatorSigner::from_seed_bytes(&COORD_WRONG_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    // Assignment signed by the WRONG coordinator key — session's
    // coordinator_pubkey_hex is the real coord, so
    // verify_work_assignment fails on the coord-signature leg.
    let bad_assignment = signed_assignment(
        &session,
        &wrong_coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    write_session_chain(
        &store,
        &session,
        &[join_a],
        &[bad_assignment.clone()],
        &[],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::InvalidState);
    let entry = report
        .invalid_artifacts
        .iter()
        .find(|e| matches!(e, InvalidArtifactStatus::InvalidAssignment { .. }))
        .expect("expected an InvalidAssignment entry");
    match entry {
        InvalidArtifactStatus::InvalidAssignment {
            assignment_id,
            reason_tag,
        } => {
            assert_eq!(assignment_id, &bad_assignment.assignment_id);
            assert_eq!(reason_tag, "CoordinatorSignatureFailed");
        }
        other => panic!("expected InvalidAssignment, got {other:?}"),
    }
}

#[test]
fn invalid_artifacts_emit_invalid_aggregate_for_aggregate_signed_by_wrong_coord() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let wrong_coord = CoordinatorSigner::from_seed_bytes(&COORD_WRONG_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    let p_0 = signed_partial(&asn_0, &contrib_a, "stage0");
    // Build a valid aggregate then resign with the wrong coord
    // (so coordinator_pubkey_hex still advertises the real coord
    // but the signature is from the wrong one).
    let mut bad_agg = signed_aggregate(&session, &[asn_0.clone()], &[p_0.clone()], &coord);
    let si = aggregated_result_signing_input(&bad_agg).unwrap();
    bad_agg.coordinator_signature_hex = wrong_coord.sign_hex(&si);
    write_session_chain(
        &store,
        &session,
        &[join_a],
        &[asn_0],
        &[p_0],
        Some(&bad_agg),
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::InvalidState);
    let entry = report
        .invalid_artifacts
        .iter()
        .find(|e| matches!(e, InvalidArtifactStatus::InvalidAggregate { .. }))
        .expect("expected an InvalidAggregate entry");
    if let InvalidArtifactStatus::InvalidAggregate { reason_tag } = entry {
        // `verify_aggregated_result_with_supersessions` returns
        // `CoordinatorSignatureFailed` for a bad agg sig.
        assert_eq!(reason_tag, "CoordinatorSignatureFailed");
    } else {
        panic!("expected InvalidAggregate");
    }
}

#[test]
fn invalid_artifacts_emit_invalid_session_returned_early() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let wrong_coord = CoordinatorSigner::from_seed_bytes(&COORD_WRONG_SEED).unwrap();
    // Build a session with the coord pubkey but resign with the
    // wrong key — verify_execution_session fails on the
    // coord-signature leg and the reporter returns early with an
    // InvalidSession diagnostic.
    let mut bad_session = signed_session(&coord, FAR_FUTURE);
    let si = execution_session_signing_input(&bad_session).unwrap();
    bad_session.coordinator_signature_hex = wrong_coord.sign_hex(&si);
    // Insert directly so the reporter's read picks it up.
    store
        .write_verified_json(
            StateObjectKind::Session,
            &bad_session.session_id,
            &bad_session,
        )
        .unwrap();
    let report =
        build_session_status_report(&store, &bad_session.session_id, NOW_UTC, false)
            .unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::InvalidState);
    // Early-return path: invalid_artifacts holds exactly one entry,
    // the InvalidSession. Joins / assignments / etc. are NOT
    // populated because we have no anchored session.
    assert_eq!(report.invalid_artifacts.len(), 1);
    match &report.invalid_artifacts[0] {
        InvalidArtifactStatus::InvalidSession { reason_tag } => {
            assert_eq!(reason_tag, "CoordinatorSignatureFailed");
        }
        other => panic!("expected InvalidSession, got {other:?}"),
    }
    assert!(report.assignments.is_empty());
    assert!(report.supersessions.is_empty());
}

#[test]
fn invalid_artifacts_emit_invalid_partial_unmatched_for_orphan_partial() {
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    // Build an assignment but do NOT write it. Build a partial
    // for it and write the partial — the reporter then sees a
    // partial with no matching verified assignment.
    let orphan_asn = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    let orphan_partial = signed_partial(&orphan_asn, &contrib_a, "stage0");
    write_session_chain(
        &store,
        &session,
        &[join_a],
        /* assignments = */ &[],
        &[orphan_partial],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(report.overall_status, SessionOverallStatus::InvalidState);
    let entry = report
        .invalid_artifacts
        .iter()
        .find(|e| matches!(e, InvalidArtifactStatus::InvalidPartial { .. }))
        .expect("expected an InvalidPartial entry for the orphan");
    match entry {
        InvalidArtifactStatus::InvalidPartial {
            assignment_id,
            reason_tag,
        } => {
            assert_eq!(assignment_id, &orphan_asn.assignment_id);
            // Approved plan: orphan partials use the uniform
            // InvalidPartial variant with the stable
            // "unmatched" tag (NOT a separate OrphanPartial).
            assert_eq!(reason_tag, "unmatched");
        }
        other => panic!("expected InvalidPartial, got {other:?}"),
    }
}

#[test]
fn invalid_artifacts_is_empty_when_overall_status_is_not_invalid_state() {
    // Stage 12.12 reporter invariant: the structured diagnostics
    // are populated <==> overall_status == InvalidState. The
    // existing complete-partials test gives us a happy-path
    // status; just re-run it and assert the new field stays
    // empty.
    let (_d, store) = fresh_store();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = signed_session(&coord, FAR_FUTURE);
    let join_a = signed_join(&session, &contrib_a);
    let join_b = signed_join(&session, &contrib_b);
    let asn_0 = signed_assignment(
        &session,
        &coord,
        &contrib_a.pubkey_hex(),
        0,
        WorkKind::Layers { start: 0, end: 8 },
        8,
    );
    let asn_1 = signed_assignment(
        &session,
        &coord,
        &contrib_b.pubkey_hex(),
        1,
        WorkKind::Layers { start: 8, end: 16 },
        8,
    );
    let p_0 = signed_partial(&asn_0, &contrib_a, "stage0");
    let p_1 = signed_partial(&asn_1, &contrib_b, "stage1");
    write_session_chain(
        &store,
        &session,
        &[join_a, join_b],
        &[asn_0, asn_1],
        &[p_0, p_1],
        None,
        &[],
    );
    let report =
        build_session_status_report(&store, &session.session_id, NOW_UTC, false).unwrap();
    assert_eq!(
        report.overall_status,
        SessionOverallStatus::CompletePartials
    );
    assert!(
        report.invalid_artifacts.is_empty(),
        "Stage 12.12 invariant: invalid_artifacts must be empty when \
         overall_status != InvalidState; got {:?}",
        report.invalid_artifacts
    );
}
