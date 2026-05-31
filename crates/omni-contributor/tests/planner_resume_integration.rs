//! Stage 12.8 — integration between the planner and the Stage 12.7
//! `ContributorStateStore`.
//!
//! These pin the contract that the planner is a library entry whose
//! caller is responsible for loading + re-verifying envelopes from
//! the state-dir. The planner's defense-in-depth re-verification
//! catches tampered loads but the canonical trust boundary lives in
//! the caller (CLI). These tests exercise both layers — load from
//! state-dir, re-verify, plan — to make sure the documented
//! workflow round-trips.

use std::fs;

use omni_contributor::{
    canonical::{
        contributor_join_signing_input, execution_session_signing_input,
        peer_advertisement_signing_input, session_id_hex,
    },
    handoff::TensorDtype,
    plan_assignments,
    planner::{PlannerInputs, PlannerStrategy},
    result::WorkUnitKind,
    AssignmentPlan, ContributorJoin, ContributorPeerAdvertisement, ContributorSigner,
    ContributorStateStore, CoordinatorSigner, ExecutionSession, PeerCapabilities,
    PlannerError, StateObjectKind, PEER_ADVERTISEMENT_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.8-resume-coord-seed-32b!";
const CONTRIB_SEED_A: [u8; 32] = *b"stage12.8-resume-contrib-a-32-by";
const CONTRIB_SEED_B: [u8; 32] = *b"stage12.8-resume-contrib-b-32-by";

fn fresh_dir() -> tempfile::TempDir {
    tempfile::TempDir::new().expect("tempdir")
}

fn now_inside_window() -> String {
    "2026-05-31T00:00:00Z".to_string()
}

fn build_session() -> ExecutionSession {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-05-31T00:00:00Z".into(),
        expires_at_utc: "2026-06-01T00:00:00Z".into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn build_join(session: &ExecutionSession, seed: [u8; 32]) -> ContributorJoin {
    let contrib = ContributorSigner::from_seed_bytes(&seed).unwrap();
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

fn build_advert(
    session: &ExecutionSession,
    contrib_seed: [u8; 32],
    expires_at_utc: &str,
) -> ContributorPeerAdvertisement {
    let contrib = ContributorSigner::from_seed_bytes(&contrib_seed).unwrap();
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
        advertised_at_utc: "2026-05-31T00:00:00Z".into(),
        expires_at_utc: expires_at_utc.into(),
        contributor_signature_hex: String::new(),
    };
    a.advertisement_id =
        omni_contributor::canonical::advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    a
}

// ── State-dir → planner happy path ─────────────────────────────

/// Write session + 2 joins to a fresh state-dir, load them back, run
/// the planner with `sequential-layers` + `--layer-count 24`, and
/// assert the published plan round-trips through serde.
#[test]
fn state_dir_to_planner_to_plan_roundtrips() {
    let d = fresh_dir();
    let session = build_session();
    let join_a = build_join(&session, CONTRIB_SEED_A);
    let join_b = build_join(&session, CONTRIB_SEED_B);

    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
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

    // Caller-side load + re-verify (mirrors what
    // run_plan_session_assignments does on the CLI).
    let loaded_session: ExecutionSession = store
        .read_verified_json(StateObjectKind::Session, &session.session_id)
        .unwrap()
        .unwrap();
    assert!(omni_contributor::verify_execution_session(&loaded_session).is_ok());

    let raw_joins = store
        .list_verified_joins_for(&session.session_id)
        .unwrap();
    let joins: Vec<ContributorJoin> = raw_joins
        .into_iter()
        .filter(|j| {
            omni_contributor::verify_contributor_join(&loaded_session, j).is_ok()
        })
        .collect();
    assert_eq!(joins.len(), 2);

    let plan = plan_assignments(
        PlannerInputs {
            session: &loaded_session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(24),
        },
        PlannerStrategy::SequentialLayers,
        None,
        &now_inside_window(),
        &now_inside_window(),
    )
    .unwrap();
    assert_eq!(plan.assignments.len(), 2);
    assert_eq!(plan.session_id, session.session_id);

    // Round-trip the plan through JSON; plan_hash must remain valid.
    let bytes = serde_json::to_vec_pretty(&plan).unwrap();
    let back: AssignmentPlan = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back, plan);
    assert_eq!(back.plan_hash, omni_contributor::plan_hash_hex(&back));
}

// ── Tampered-load defense ──────────────────────────────────────

/// If a forged `join.json` is dropped into the state-dir, the
/// caller-side filter (Stage 12.7 trust boundary) drops it. The
/// planner is fed only the verified subset.
#[test]
fn caller_side_filter_drops_forged_join_from_state_dir() {
    let d = fresh_dir();
    let session = build_session();
    let good_join = build_join(&session, CONTRIB_SEED_A);
    let bad_join = {
        let mut j = build_join(&session, CONTRIB_SEED_B);
        // Flip a hex digit in the signature.
        let mut s: Vec<char> = j.contributor_signature_hex.chars().collect();
        let last = s.len() - 1;
        s[last] = if s[last] == '0' { '1' } else { '0' };
        j.contributor_signature_hex = s.into_iter().collect();
        j
    };

    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session.session_id.clone(),
            },
            &good_join.contributor_pubkey_hex,
            &good_join,
        )
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session.session_id.clone(),
            },
            &bad_join.contributor_pubkey_hex,
            &bad_join,
        )
        .unwrap();

    let loaded_session: ExecutionSession = store
        .read_verified_json(StateObjectKind::Session, &session.session_id)
        .unwrap()
        .unwrap();
    let raw_joins = store
        .list_verified_joins_for(&session.session_id)
        .unwrap();
    assert_eq!(raw_joins.len(), 2);
    let joins: Vec<ContributorJoin> = raw_joins
        .into_iter()
        .filter(|j| {
            omni_contributor::verify_contributor_join(&loaded_session, j).is_ok()
        })
        .collect();
    assert_eq!(
        joins.len(),
        1,
        "forged join must NOT survive caller-side re-verification"
    );
    assert_eq!(joins[0].contributor_pubkey_hex, good_join.contributor_pubkey_hex);
}

/// The planner library entry runs its own defense-in-depth
/// re-verification of every supplied join. Even if a caller skips
/// the explicit filter, a forged join is silently dropped before
/// strategy dispatch.
#[test]
fn planner_internal_reverify_drops_tampered_join() {
    let session = build_session();
    let good_join = build_join(&session, CONTRIB_SEED_A);
    let bad_join = {
        let mut j = build_join(&session, CONTRIB_SEED_B);
        let mut s: Vec<char> = j.contributor_signature_hex.chars().collect();
        let last = s.len() - 1;
        s[last] = if s[last] == '0' { '1' } else { '0' };
        j.contributor_signature_hex = s.into_iter().collect();
        j
    };

    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &[good_join.clone(), bad_join],
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        &now_inside_window(),
        &now_inside_window(),
    )
    .unwrap();
    assert_eq!(plan.assignments.len(), 1);
    assert_eq!(
        plan.assignments[0].contributor_pubkey_hex,
        good_join.contributor_pubkey_hex
    );
}

// ── Live routing posture via state-dir ─────────────────────────

#[test]
fn state_dir_load_filters_expired_advert_when_live_routing_required() {
    let d = fresh_dir();
    let session = build_session();
    let join_a = build_join(&session, CONTRIB_SEED_A);
    let join_b = build_join(&session, CONTRIB_SEED_B);
    // Two adverts (one per contributor): A is fresh, B is expired.
    let fresh = build_advert(&session, CONTRIB_SEED_A, "2026-06-01T00:00:00Z");
    let expired = build_advert(&session, CONTRIB_SEED_B, "2026-05-30T00:00:00Z");

    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
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
    store
        .write_verified_json(
            StateObjectKind::PeerAdvert {
                session_id: session.session_id.clone(),
            },
            &fresh.contributor_pubkey_hex,
            &fresh,
        )
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::PeerAdvert {
                session_id: session.session_id.clone(),
            },
            &expired.contributor_pubkey_hex,
            &expired,
        )
        .unwrap();

    let loaded_session: ExecutionSession = store
        .read_verified_json(StateObjectKind::Session, &session.session_id)
        .unwrap()
        .unwrap();
    let raw_joins = store
        .list_verified_joins_for(&session.session_id)
        .unwrap();
    let joins: Vec<ContributorJoin> = raw_joins
        .into_iter()
        .filter(|j| {
            omni_contributor::verify_contributor_join(&loaded_session, j).is_ok()
        })
        .collect();
    assert_eq!(joins.len(), 2);

    let raw_adverts = store
        .list_verified_peer_adverts_for(&session.session_id)
        .unwrap();
    // Mirror what the CLI does: re-verify via
    // verify_peer_advertisement_body before planning.
    let peer_adverts: Vec<ContributorPeerAdvertisement> = raw_adverts
        .into_iter()
        .filter(|a| {
            matches!(
                omni_contributor::verify_peer_advertisement_body(
                    a,
                    &joins,
                    Some(&now_inside_window()),
                ),
                omni_contributor::PeerAdvertisementOutcome::Verified { .. }
            )
        })
        .collect();
    // The fresh advert survives; the expired one does not.
    assert_eq!(peer_adverts.len(), 1);
    assert_eq!(
        peer_adverts[0].contributor_pubkey_hex,
        fresh.contributor_pubkey_hex
    );

    let plan = plan_assignments(
        PlannerInputs {
            session: &loaded_session,
            joins: &joins,
            peer_adverts: &peer_adverts,
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: true,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        &now_inside_window(),
        &now_inside_window(),
    )
    .unwrap();
    // Only the fresh-advert contributor is eligible.
    assert_eq!(plan.assignments.len(), 1);
    assert_eq!(
        plan.assignments[0].contributor_pubkey_hex,
        fresh.contributor_pubkey_hex
    );
}

// ── Empty inputs are a typed error ────────────────────────────

#[test]
fn state_dir_with_no_joins_yields_typed_empty_error() {
    let d = fresh_dir();
    let session = build_session();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();

    let loaded_session: ExecutionSession = store
        .read_verified_json(StateObjectKind::Session, &session.session_id)
        .unwrap()
        .unwrap();
    let joins: Vec<ContributorJoin> = Vec::new();
    let err = plan_assignments(
        PlannerInputs {
            session: &loaded_session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        &now_inside_window(),
        &now_inside_window(),
    )
    .unwrap_err();
    assert!(matches!(err, PlannerError::NoEligibleContributors { .. }));
}

/// A hand-edited plan with `expected_work_units = 0` survives
/// `plan_hash_hex` (because the operator recomputed it after the
/// edit) but the resulting `WorkAssignment` fails
/// `validate_schema`. Stage 12.8 dry-run runs `validate_schema` on
/// every planned body before exit so this is caught without
/// touching SNIP. This test pins the underlying contract.
#[test]
fn hand_edited_zero_work_units_fails_work_assignment_schema() {
    use omni_contributor::canonical::{assignment_id_hex, work_assignment_signing_input};

    let session = build_session();
    let join_a = build_join(&session, CONTRIB_SEED_A);
    let mut plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &[join_a],
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        &now_inside_window(),
        &now_inside_window(),
    )
    .unwrap();
    // Hand-edit the planned entry and re-stamp plan_hash so the
    // CLI's hash check passes.
    plan.assignments[0].expected_work_units = 0;
    plan.plan_hash = String::new();
    plan.plan_hash = omni_contributor::plan_hash_hex(&plan);
    assert_eq!(plan.plan_hash, omni_contributor::plan_hash_hex(&plan));

    // Mirror what `run_assign_session_plan --dry-run` does after
    // the review fix: build + sign + validate_schema each planned
    // body. The zero `expected_work_units` must trip the schema
    // check.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut a = plan.assignments[0]
        .to_unsigned_work_assignment(&plan.session_id, &now_inside_window());
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    assert!(
        a.validate_schema().is_err(),
        "WorkAssignment::validate_schema must reject expected_work_units = 0"
    );
}

/// Each `PlannedAssignment` is shaped so it can be turned into a
/// real signed `WorkAssignment` that passes the Stage 12.3 verifier.
/// This pins the contract on which `assign-session-plan` depends:
/// "plan → unsigned envelope → sign → publish" is a valid path.
#[test]
fn planned_assignments_pass_stage12_3_verifier_after_signing() {
    use omni_contributor::canonical::{assignment_id_hex, work_assignment_signing_input};
    use omni_contributor::session_verify::SessionVerifyOutcome;
    use std::collections::HashSet;

    let session = build_session();
    let join_a = build_join(&session, CONTRIB_SEED_A);
    let join_b = build_join(&session, CONTRIB_SEED_B);
    let joins = vec![join_a.clone(), join_b.clone()];
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(24),
        },
        PlannerStrategy::SequentialLayers,
        None,
        &now_inside_window(),
        &now_inside_window(),
    )
    .unwrap();

    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let joined_pubkeys: HashSet<String> = joins
        .iter()
        .map(|j| j.contributor_pubkey_hex.clone())
        .collect();

    for pa in &plan.assignments {
        let mut a = pa.to_unsigned_work_assignment(
            &session.session_id,
            "2026-05-31T01:00:00Z",
        );
        a.assignment_id = assignment_id_hex(&a).unwrap();
        let sig_input = work_assignment_signing_input(&a).unwrap();
        a.coordinator_signature_hex = coord.sign_hex(&sig_input);
        let outcome = omni_contributor::verify_work_assignment(
            &session,
            &joined_pubkeys,
            &a,
        );
        assert!(
            matches!(outcome, SessionVerifyOutcome::Ok),
            "signed assignment must pass Stage 12.3 verifier; got {outcome:?}"
        );
    }
}

/// Sanity: the plan written to disk parses back into the same
/// AssignmentPlan and the recomputed plan_hash matches. Mirrors what
/// `assign-session-plan` does on the publish side.
#[test]
fn plan_dry_run_serialization_roundtrip_preserves_plan_hash() {
    let session = build_session();
    let join_a = build_join(&session, CONTRIB_SEED_A);
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &[join_a],
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        &now_inside_window(),
        &now_inside_window(),
    )
    .unwrap();
    let d = fresh_dir();
    let out = d.path().join("plan.json");
    fs::write(&out, serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
    let back: AssignmentPlan = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(back, plan);
    assert_eq!(back.plan_hash, omni_contributor::plan_hash_hex(&back));
}
