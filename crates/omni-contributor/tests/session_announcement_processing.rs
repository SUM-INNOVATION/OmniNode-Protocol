//! Stage 12.3 — tests for the 5 `process_*_announcement` helpers.
//! Covers the happy path + the failure modes a watch-sessions
//! deployment must defend against: tampered announcer signatures,
//! drift between announcement and body, malformed body, bad SNIP
//! root, and (for assignments) a missing or wrong session coord
//! pubkey.

mod common;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{
        aggregated_result_signing_input, assignment_id_hex, canonical_partial_result_bytes,
        contributor_join_signing_input, execution_session_signing_input, hex_lower,
        net_aggregated_signing_input, net_assign_signing_input, net_join_signing_input,
        net_partial_signing_input, net_session_opened_signing_input,
        partial_result_signing_input, session_id_hex, work_assignment_signing_input,
    },
    process_aggregated_result_announcement, process_contributor_joined_announcement,
    process_partial_result_announcement, process_session_opened_announcement,
    process_work_assigned_announcement,
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    AggregatedContributorResult, AggregatedPartialRef, AnnouncementOutcome,
    ContributorJoin, ContributorSigner, CoordinatorSigner, ExecutionSession,
    NetworkAggregatedResultAnnouncement, NetworkContributorJoinedAnnouncement,
    NetworkPartialResultAnnouncement, NetworkSessionOpenedAnnouncement,
    NetworkWorkAssignedAnnouncement, PartialContributorResult, WorkAssignment, WorkKind,
    NET_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.3-proc-coord-32-bytes!!!";
const CONTRIB_SEED: [u8; 32] = *b"stage12.3-proc-contrib-32-byte!!";
const ROGUE_SEED: [u8; 32] = *b"stage12.3-proc-rogue-seed-32!!!!";

fn make_session(coord: &CoordinatorSigner) -> ExecutionSession {
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

fn make_join(session: &ExecutionSession, contrib: &ContributorSigner) -> ContributorJoin {
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 100,
        max_output_tokens: 100,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-27T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

fn make_assignment(
    session: &ExecutionSession,
    contrib_pub: &str,
    coord: &CoordinatorSigner,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index: 0,
        contributor_pubkey_hex: contrib_pub.to_string(),
        work_kind: WorkKind::Prefill,
        expected_work_units: 100,
        expected_work_unit_kind: WorkUnitKind::PrefillTokens,
        assigned_at_utc: "2026-05-27T00:00:02Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

fn make_partial(
    assignment: &WorkAssignment,
    contrib: &ContributorSigner,
) -> PartialContributorResult {
    let mut p = PartialContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: assignment.session_id.clone(),
        assignment_id: assignment.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        partial_artifact_snip_root: format!("0x{}", "aa".repeat(32)),
        partial_artifact_hash: hex_lower(blake3::hash(b"artifact").as_bytes()),
        measured_accounting: MeasuredAccounting {
            tokenizer_hash: "44".repeat(32),
            input_token_count: 100,
            output_token_count: 0,
            total_base_units: 100,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: contrib.pubkey_hex(),
                stage_label: "prefill".into(),
                work_unit_kind: WorkUnitKind::PrefillTokens,
                work_units: 100,
            }],
        },
        produced_at_utc: "2026-05-27T00:00:10Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = partial_result_signing_input(&p).unwrap();
    p.contributor_signature_hex = contrib.sign_hex(&si);
    p
}

fn make_aggregate(
    session: &ExecutionSession,
    asn: &WorkAssignment,
    par: &PartialContributorResult,
    coord: &CoordinatorSigner,
) -> AggregatedContributorResult {
    let par_hash =
        hex_lower(blake3::hash(&canonical_partial_result_bytes(par).unwrap()).as_bytes());
    let mut g = AggregatedContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        final_result_snip_root: format!("0x{}", "bb".repeat(32)),
        final_result_canonical_hash: hex_lower(blake3::hash(b"final").as_bytes()),
        partial_refs: vec![AggregatedPartialRef {
            assignment_id: asn.assignment_id.clone(),
            stage_index: asn.stage_index,
            contributor_pubkey_hex: par.contributor_pubkey_hex.clone(),
            partial_snip_root: format!("0x{}", "cc".repeat(32)),
            partial_canonical_hash: par_hash,
        }],
        aggregated_at_utc: "2026-05-27T00:00:30Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let si = aggregated_result_signing_input(&g).unwrap();
    g.coordinator_signature_hex = coord.sign_hex(&si);
    g
}

// ── Session-opened ──────────────────────────────────────────────────

#[test]
fn process_session_opened_happy_path() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = make_session(&coord);
    let session_root = snip.insert_bytes(&serde_json::to_vec(&session).unwrap());

    let mut ann = NetworkSessionOpenedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        execution_session_snip_root: session_root.to_hex(),
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        announced_at_utc: "2026-05-27T00:00:00Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_session_opened_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);

    let out = process_session_opened_announcement(&ann, &snip);
    assert!(out.is_verified(), "{out:?}");
}

#[test]
fn process_session_opened_tampered_announcer_sig_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = make_session(&coord);
    let session_root = snip.insert_bytes(&serde_json::to_vec(&session).unwrap());
    let mut ann = NetworkSessionOpenedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        execution_session_snip_root: session_root.to_hex(),
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        announced_at_utc: "2026-05-27T00:00:00Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_session_opened_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    // Flip one nibble.
    let mut sig = ann.announcer_signature_hex.into_bytes();
    sig[0] = if sig[0] == b'a' { b'b' } else { b'a' };
    ann.announcer_signature_hex = String::from_utf8(sig).unwrap();
    let out = process_session_opened_announcement(&ann, &snip);
    assert!(matches!(out, AnnouncementOutcome::AnnouncerSignatureFailed), "{out:?}");
}

#[test]
fn process_session_opened_drift_session_id_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = make_session(&coord);
    let session_root = snip.insert_bytes(&serde_json::to_vec(&session).unwrap());
    let mut ann = NetworkSessionOpenedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        execution_session_snip_root: session_root.to_hex(),
        // Liar: claim a different session_id than the inner body.
        session_id: "ff".repeat(32),
        posted_id: session.posted_id.clone(),
        announced_at_utc: "2026-05-27T00:00:00Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_session_opened_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    let out = process_session_opened_announcement(&ann, &snip);
    assert!(matches!(out, AnnouncementOutcome::DriftMismatch { field: "session_id" }), "{out:?}");
}

#[test]
fn process_session_opened_missing_snip_object_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = make_session(&coord);
    // Announce a SNIP root that doesn't exist in the store.
    let mut ann = NetworkSessionOpenedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        execution_session_snip_root: format!("0x{}", "ee".repeat(32)),
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        announced_at_utc: "2026-05-27T00:00:00Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_session_opened_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    let out = process_session_opened_announcement(&ann, &snip);
    assert!(matches!(out, AnnouncementOutcome::SnipFetchFailed(_)), "{out:?}");
}

#[test]
fn process_session_opened_body_sig_failed_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    // Forge a session signed by a different key than its declared
    // coord pubkey. The body sig verify must catch this even though
    // the announcer sig (from a real coord here) is valid.
    let mut session = make_session(&coord);
    let si = execution_session_signing_input(&session).unwrap();
    session.coordinator_signature_hex = rogue.sign_hex(&si);
    let session_root = snip.insert_bytes(&serde_json::to_vec(&session).unwrap());
    let mut ann = NetworkSessionOpenedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        execution_session_snip_root: session_root.to_hex(),
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        announced_at_utc: "2026-05-27T00:00:00Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let ann_sig = net_session_opened_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&ann_sig);
    let out = process_session_opened_announcement(&ann, &snip);
    assert!(matches!(out, AnnouncementOutcome::BodySignatureFailed), "{out:?}");
}

// ── Join ────────────────────────────────────────────────────────────

#[test]
fn process_join_happy_path() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let join = make_join(&session, &contrib);
    let join_root = snip.insert_bytes(&serde_json::to_vec(&join).unwrap());
    let mut ann = NetworkContributorJoinedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        contributor_join_snip_root: join_root.to_hex(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:01Z".into(),
        announcer_pubkey_hex: contrib.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_join_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = contrib.sign_hex(&si);
    let out = process_contributor_joined_announcement(&ann, &snip);
    assert!(out.is_verified(), "{out:?}");
}

#[test]
fn process_join_drift_contributor_pubkey_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let rogue = ContributorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let session = make_session(&coord);
    let join = make_join(&session, &contrib);
    let join_root = snip.insert_bytes(&serde_json::to_vec(&join).unwrap());
    // Liar: announce a different contributor_pubkey than the body's.
    let mut ann = NetworkContributorJoinedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        contributor_join_snip_root: join_root.to_hex(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: rogue.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:01Z".into(),
        announcer_pubkey_hex: rogue.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_join_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = rogue.sign_hex(&si);
    let out = process_contributor_joined_announcement(&ann, &snip);
    assert!(
        matches!(out, AnnouncementOutcome::DriftMismatch { field: "contributor_pubkey_hex" }),
        "{out:?}"
    );
}

// ── Assignment ──────────────────────────────────────────────────────

#[test]
fn process_assignment_happy_path_with_session_coord() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let asn = make_assignment(&session, &contrib.pubkey_hex(), &coord);
    let asn_root = snip.insert_bytes(&serde_json::to_vec(&asn).unwrap());
    let mut ann = NetworkWorkAssignedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        work_assignment_snip_root: asn_root.to_hex(),
        session_id: session.session_id.clone(),
        assignment_id: asn.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:02Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_assign_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    let out = process_work_assigned_announcement(
        &ann,
        &snip,
        Some(&session.coordinator_pubkey_hex),
    );
    assert!(out.is_verified(), "{out:?}");
}

#[test]
fn process_assignment_with_wrong_session_coord_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let asn = make_assignment(&session, &contrib.pubkey_hex(), &coord);
    let asn_root = snip.insert_bytes(&serde_json::to_vec(&asn).unwrap());
    let mut ann = NetworkWorkAssignedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        work_assignment_snip_root: asn_root.to_hex(),
        session_id: session.session_id.clone(),
        assignment_id: asn.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:02Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_assign_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    // Pass the rogue's pubkey — body sig was signed by the real
    // coord, so verification against the rogue's pubkey must fail.
    let out = process_work_assigned_announcement(&ann, &snip, Some(&rogue.pubkey_hex()));
    assert!(matches!(out, AnnouncementOutcome::BodySignatureFailed), "{out:?}");
}

#[test]
fn process_assignment_no_session_skips_body_sig_check() {
    // When the watcher hasn't seen the session yet, the processor
    // still does announcer + schema + drift, but skips the body's
    // coord sig check. We exercise that path by passing None.
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let asn = make_assignment(&session, &contrib.pubkey_hex(), &coord);
    let asn_root = snip.insert_bytes(&serde_json::to_vec(&asn).unwrap());
    let mut ann = NetworkWorkAssignedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        work_assignment_snip_root: asn_root.to_hex(),
        session_id: session.session_id.clone(),
        assignment_id: asn.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:02Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_assign_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    let out = process_work_assigned_announcement(&ann, &snip, None);
    assert!(out.is_verified(), "{out:?}");
}

// ── Partial ─────────────────────────────────────────────────────────

#[test]
fn process_partial_tampered_announcer_sig_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let asn = make_assignment(&session, &contrib.pubkey_hex(), &coord);
    let par = make_partial(&asn, &contrib);
    let par_root = snip.insert_bytes(&serde_json::to_vec(&par).unwrap());
    let mut ann = NetworkPartialResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        partial_result_snip_root: par_root.to_hex(),
        session_id: session.session_id.clone(),
        assignment_id: asn.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:11Z".into(),
        announcer_pubkey_hex: contrib.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_partial_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = contrib.sign_hex(&si);
    // Tamper the announcer sig.
    let mut sigb = ann.announcer_signature_hex.into_bytes();
    sigb[10] = if sigb[10] == b'a' { b'b' } else { b'a' };
    ann.announcer_signature_hex = String::from_utf8(sigb).unwrap();
    let out = process_partial_result_announcement(&ann, &snip);
    assert!(matches!(out, AnnouncementOutcome::AnnouncerSignatureFailed), "{out:?}");
}

#[test]
fn process_partial_tampered_body_sig_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let asn = make_assignment(&session, &contrib.pubkey_hex(), &coord);
    let mut par = make_partial(&asn, &contrib);
    // Tamper the body sig after signing.
    let mut sigb = par.contributor_signature_hex.into_bytes();
    sigb[5] = if sigb[5] == b'1' { b'2' } else { b'1' };
    par.contributor_signature_hex = String::from_utf8(sigb).unwrap();
    let par_root = snip.insert_bytes(&serde_json::to_vec(&par).unwrap());
    let mut ann = NetworkPartialResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        partial_result_snip_root: par_root.to_hex(),
        session_id: session.session_id.clone(),
        assignment_id: asn.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:11Z".into(),
        announcer_pubkey_hex: contrib.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_partial_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = contrib.sign_hex(&si);
    let out = process_partial_result_announcement(&ann, &snip);
    assert!(matches!(out, AnnouncementOutcome::BodySignatureFailed), "{out:?}");
}

// ── Aggregated ──────────────────────────────────────────────────────

#[test]
fn process_aggregated_happy_path() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let asn = make_assignment(&session, &contrib.pubkey_hex(), &coord);
    let par = make_partial(&asn, &contrib);
    let agg = make_aggregate(&session, &asn, &par, &coord);
    let agg_root = snip.insert_bytes(&serde_json::to_vec(&agg).unwrap());
    let mut ann = NetworkAggregatedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        aggregated_result_snip_root: agg_root.to_hex(),
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        announced_at_utc: "2026-05-27T00:00:30Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_aggregated_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    let out = process_aggregated_result_announcement(&ann, &snip);
    assert!(out.is_verified(), "{out:?}");
}

#[test]
fn process_aggregated_drift_posted_id_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let asn = make_assignment(&session, &contrib.pubkey_hex(), &coord);
    let par = make_partial(&asn, &contrib);
    let agg = make_aggregate(&session, &asn, &par, &coord);
    let agg_root = snip.insert_bytes(&serde_json::to_vec(&agg).unwrap());
    let mut ann = NetworkAggregatedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        aggregated_result_snip_root: agg_root.to_hex(),
        session_id: session.session_id.clone(),
        posted_id: "dd".repeat(32),
        announced_at_utc: "2026-05-27T00:00:30Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_aggregated_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&si);
    let out = process_aggregated_result_announcement(&ann, &snip);
    assert!(matches!(out, AnnouncementOutcome::DriftMismatch { field: "posted_id" }), "{out:?}");
}

#[test]
fn process_aggregated_body_sig_failed_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = make_session(&coord);
    let asn = make_assignment(&session, &contrib.pubkey_hex(), &coord);
    let par = make_partial(&asn, &contrib);
    // Aggregate's coord_pubkey says coord, but body is signed by rogue.
    let mut agg = make_aggregate(&session, &asn, &par, &coord);
    let bsi = aggregated_result_signing_input(&agg).unwrap();
    agg.coordinator_signature_hex = rogue.sign_hex(&bsi);
    let agg_root = snip.insert_bytes(&serde_json::to_vec(&agg).unwrap());
    let mut ann = NetworkAggregatedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        aggregated_result_snip_root: agg_root.to_hex(),
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        announced_at_utc: "2026-05-27T00:00:30Z".into(),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let ann_sig = net_aggregated_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = coord.sign_hex(&ann_sig);
    let out = process_aggregated_result_announcement(&ann, &snip);
    assert!(matches!(out, AnnouncementOutcome::BodySignatureFailed), "{out:?}");
}

// ── Stage 12.11 — assignment supersession ───────────────────────

mod supersession_processor {
    use super::*;
    use omni_contributor::{
        canonical::{net_supersession_signing_input, supersession_id_hex},
        process_assignment_supersession_announcement,
        supersession::{SupersessionReason, WorkAssignmentSupersession},
        NetworkWorkAssignmentSupersessionAnnouncement, SUPERSESSION_SCHEMA_VERSION,
    };

    /// Local helper: like `make_assignment` but lets the caller
    /// pick the stage_index so two assignments in the same
    /// session hash to distinct `assignment_id`s. Without this
    /// the supersession's superseded vs replacement IDs would
    /// collide and trip the disjointness schema check.
    fn make_assignment_stage(
        session: &ExecutionSession,
        contrib_pub: &str,
        coord: &CoordinatorSigner,
        stage_index: u32,
    ) -> WorkAssignment {
        use omni_contributor::canonical::{assignment_id_hex, work_assignment_signing_input};
        let mut a = WorkAssignment {
            schema_version: SESSION_SCHEMA_VERSION,
            session_id: session.session_id.clone(),
            assignment_id: String::new(),
            stage_index,
            contributor_pubkey_hex: contrib_pub.to_string(),
            work_kind: WorkKind::Prefill,
            expected_work_units: 100,
            expected_work_unit_kind: WorkUnitKind::PrefillTokens,
            assigned_at_utc: "2026-05-27T00:00:02Z".into(),
            coordinator_signature_hex: String::new(),
        };
        a.assignment_id = assignment_id_hex(&a).unwrap();
        let si = work_assignment_signing_input(&a).unwrap();
        a.coordinator_signature_hex = coord.sign_hex(&si);
        a
    }

    fn make_supersession(
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
            created_at_utc: "2026-05-27T00:00:20Z".into(),
            coordinator_pubkey_hex: coord.pubkey_hex(),
            coordinator_signature_hex: String::new(),
        };
        s.supersession_id = supersession_id_hex(&s).unwrap();
        let si = omni_contributor::canonical::work_assignment_supersession_signing_input(
            &s,
        )
        .unwrap();
        s.coordinator_signature_hex = coord.sign_hex(&si);
        s
    }

    fn make_ann(
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
            announced_at_utc: "2026-05-27T00:00:25Z".into(),
            announcer_pubkey_hex: coord.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let si = net_supersession_signing_input(&ann).unwrap();
        ann.announcer_signature_hex = coord.sign_hex(&si);
        ann
    }

    #[test]
    fn happy_path_no_session_context() {
        let snip = MockSnipStore::new();
        let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
        let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
        let session = make_session(&coord);
        // The processor's reference-resolution leg is gated on
        // `session.is_some() && assignments.is_some()`; passing
        // None for both must still pass when announcer sig + body
        // sig + drift are all clean.
        let asn_old = make_assignment_stage(&session, &contrib.pubkey_hex(), &coord, 0);
        let asn_new = make_assignment_stage(&session, &contrib.pubkey_hex(), &coord, 1);
        let sup = make_supersession(
            &session,
            &coord,
            vec![asn_old.assignment_id.clone()],
            vec![asn_new.assignment_id.clone()],
        );
        let ann = make_ann(&snip, &coord, &sup);
        let out = process_assignment_supersession_announcement(&ann, &snip, None, None);
        assert!(out.is_verified(), "{out:?}");
    }

    #[test]
    fn happy_path_with_session_and_assignments() {
        let snip = MockSnipStore::new();
        let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
        let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
        let session = make_session(&coord);
        let asn_old = make_assignment_stage(&session, &contrib.pubkey_hex(), &coord, 0);
        let asn_new = make_assignment_stage(&session, &contrib.pubkey_hex(), &coord, 1);
        let sup = make_supersession(
            &session,
            &coord,
            vec![asn_old.assignment_id.clone()],
            vec![asn_new.assignment_id.clone()],
        );
        let ann = make_ann(&snip, &coord, &sup);
        let assignments = vec![asn_old, asn_new];
        let out = process_assignment_supersession_announcement(
            &ann,
            &snip,
            Some(&session),
            Some(&assignments),
        );
        assert!(out.is_verified(), "{out:?}");
    }

    #[test]
    fn drift_session_id_refused() {
        let snip = MockSnipStore::new();
        let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
        let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
        let session = make_session(&coord);
        let asn_old = make_assignment_stage(&session, &contrib.pubkey_hex(), &coord, 0);
        let asn_new = make_assignment_stage(&session, &contrib.pubkey_hex(), &coord, 1);
        let sup = make_supersession(
            &session,
            &coord,
            vec![asn_old.assignment_id.clone()],
            vec![asn_new.assignment_id.clone()],
        );
        // The body is correctly bound to `sup.session_id`; the
        // announcement lies about it. Announcer-sig recomputes
        // over the lying announcement, so re-sign it after the
        // tamper.
        let root = snip.insert_bytes(&serde_json::to_vec(&sup).unwrap());
        let mut ann = NetworkWorkAssignmentSupersessionAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            work_assignment_supersession_snip_root: root.to_hex(),
            session_id: "ff".repeat(32),
            supersession_id: sup.supersession_id.clone(),
            announced_at_utc: "2026-05-27T00:00:25Z".into(),
            announcer_pubkey_hex: coord.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let si = net_supersession_signing_input(&ann).unwrap();
        ann.announcer_signature_hex = coord.sign_hex(&si);
        let out = process_assignment_supersession_announcement(&ann, &snip, None, None);
        assert!(
            matches!(out, AnnouncementOutcome::DriftMismatch { field: "session_id" }),
            "{out:?}"
        );
    }

    #[test]
    fn body_coord_pubkey_drift_under_session_refused() {
        // Body is signed by a rogue coord; the body's
        // self-declared `coordinator_pubkey_hex` is the rogue's,
        // so the body sig itself verifies. The session-pinning
        // drift check must catch the mismatch against the
        // session's coord pubkey.
        let snip = MockSnipStore::new();
        let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
        let rogue = CoordinatorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
        let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
        let session = make_session(&coord);
        let asn_old = make_assignment_stage(&session, &contrib.pubkey_hex(), &coord, 0);
        let asn_new = make_assignment_stage(&session, &contrib.pubkey_hex(), &coord, 1);
        let sup = make_supersession(
            &session,
            &rogue,
            vec![asn_old.assignment_id.clone()],
            vec![asn_new.assignment_id.clone()],
        );
        // Announce with the rogue's pubkey too so the announcer
        // sig matches but the session pinning fails.
        let root = snip.insert_bytes(&serde_json::to_vec(&sup).unwrap());
        let mut ann = NetworkWorkAssignmentSupersessionAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            work_assignment_supersession_snip_root: root.to_hex(),
            session_id: sup.session_id.clone(),
            supersession_id: sup.supersession_id.clone(),
            announced_at_utc: "2026-05-27T00:00:25Z".into(),
            announcer_pubkey_hex: rogue.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let si = net_supersession_signing_input(&ann).unwrap();
        ann.announcer_signature_hex = rogue.sign_hex(&si);
        let out = process_assignment_supersession_announcement(
            &ann,
            &snip,
            Some(&session),
            None,
        );
        assert!(
            matches!(
                out,
                AnnouncementOutcome::DriftMismatch {
                    field: "coordinator_pubkey_hex",
                }
            ),
            "{out:?}"
        );
    }
}
