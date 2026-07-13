//! Stage 12.7 — integration tests for restart resume via
//! `ContributorStateStore`.
//!
//! These exercise the four contracts a restarted `omni-node`
//! contributor process depends on:
//!
//!   1. The `verified/sessions/<id>/...` subtree under the state-dir
//!      is **bit-identical** to the existing Stage 12.3
//!      `watch-sessions --out-dir` layout, so an operator can point
//!      either tool at the same directory.
//!   2. Peer advertisements written through the state store survive
//!      a `ContributorStateStore::open` reopen and round-trip
//!      cleanly back into `ContributorPeerAdvertisement`.
//!   3. A `PeerRoutingCache` can be rebuilt **from the state dir
//!      alone** (no in-memory pre-warming, no separate peer-advert
//!      dir flag) and `resolve` returns the cached route.
//!   4. Cross-restart deduplication: a `posted-jobs` marker laid
//!      down before restart still reports `is_seen == true` after
//!      a second `open`, so the watch loop can skip already-handled
//!      announcements without re-fetching from SNIP.
//!
//! Stage 12.7 hard limits: no Stage 12.0–12.6 schema or canonical
//! byte changes, no omni-net changes, no chain wire, no payment /
//! reward / staking / slashing, no marketplace / auction / bid
//! logic, no proof-mode plumbing, SNIP terminology only.

use std::fs;

use omni_contributor::{
    canonical::{
        advertisement_id_hex, contributor_join_signing_input,
        execution_session_signing_input, peer_advertisement_signing_input,
        session_id_hex,
    },
    handoff::TensorDtype,
    peer_routing::{PeerRoutingCache, RouteResolution},
    result::WorkUnitKind,
    ContributorJoin, ContributorPeerAdvertisement, ContributorSigner,
    ContributorStateStore, CoordinatorSigner, ExecutionSession, PeerCapabilities,
    StateNamespace, StateObjectKind, PEER_ADVERTISEMENT_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.7-resume-coord-seed-32b!";
const CONTRIB_SEED: [u8; 32] = *b"stage12.7-resume-contrib-seed32!";
const PEER_ID_FIXTURE: &str = "12D3KooWGjMyD3v8UYjK3bV7Sqg3WcWzkF7QbTtwYNRsBwCb6KpC";

fn fresh_dir() -> tempfile::TempDir {
    tempfile::TempDir::new().expect("tempdir")
}

fn now_inside_window() -> String {
    "2026-05-30T00:00:00Z".to_string()
}

fn build_session(expires_at_utc: &str) -> ExecutionSession {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-05-30T00:00:00Z".into(),
        expires_at_utc: expires_at_utc.into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn build_join(session: &ExecutionSession) -> ContributorJoin {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 100,
        max_output_tokens: 100,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-30T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

fn build_advert(
    session: &ExecutionSession,
    expires_at_utc: &str,
) -> ContributorPeerAdvertisement {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let mut a = ContributorPeerAdvertisement {
        schema_version: PEER_ADVERTISEMENT_SCHEMA_VERSION,
        advertisement_id: String::new(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        libp2p_peer_id: PEER_ID_FIXTURE.into(),
        listen_multiaddrs: vec!["/ip4/127.0.0.1/tcp/4001".into()],
        capabilities: PeerCapabilities {
            supports_live_handoff: true,
            max_handoff_chunk_bytes: 32 * 1024 * 1024,
            supported_dtypes: vec![TensorDtype::F16],
        },
        advertised_at_utc: "2026-05-30T00:00:00Z".into(),
        expires_at_utc: expires_at_utc.into(),
        contributor_signature_hex: String::new(),
    };
    a.advertisement_id = advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    a
}

// ── 1. State-dir layout matches watch-sessions --out-dir layout ─────

/// The state-dir's `verified/sessions/<id>/...` subtree is the same
/// shape as the existing Stage 12.3 `watch-sessions --out-dir` tree.
/// We verify by writing into the state store, then reading the
/// raw files at the exact paths the existing CLI handlers expect.
#[test]
fn watch_sessions_layout_dual_writes_to_state_dir() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    let session = build_session("2026-05-30T01:00:00Z");
    let join = build_join(&session);
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session.session_id.clone(),
            },
            &join.contributor_pubkey_hex,
            &join,
        )
        .unwrap();

    // The state-dir layout the existing CLI handler can consume:
    //   <state-dir>/verified/sessions/<id>/session.json
    //   <state-dir>/verified/sessions/<id>/joins/<pubkey>.json
    let sessions_root = d.path().join("verified").join("sessions");
    let session_path = sessions_root
        .join(&session.session_id)
        .join("session.json");
    let join_path = sessions_root
        .join(&session.session_id)
        .join("joins")
        .join(format!("{}.json", join.contributor_pubkey_hex));
    assert!(session_path.is_file(), "session.json missing at expected layout");
    assert!(join_path.is_file(), "joins/<pubkey>.json missing at expected layout");

    // And the bytes deserialize back into Stage 12.3 envelopes that
    // round-trip through the regular verifiers.
    let s: ExecutionSession =
        serde_json::from_slice(&fs::read(&session_path).unwrap()).unwrap();
    let j: ContributorJoin =
        serde_json::from_slice(&fs::read(&join_path).unwrap()).unwrap();
    assert!(omni_contributor::verify_execution_session(&s).is_ok());
    assert!(omni_contributor::verify_contributor_join(&s, &j).is_ok());
}

// ── 2. Peer adverts survive reopen ──────────────────────────────────

#[test]
fn peer_advert_dual_writes_to_state_dir() {
    let d = fresh_dir();
    let session = build_session("2026-05-30T01:00:00Z");
    let advert = build_advert(&session, "2026-05-30T01:00:00Z");
    {
        let (store, _) =
            ContributorStateStore::open(d.path(), false, &now_inside_window())
                .unwrap();
        store
            .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
            .unwrap();
        store
            .write_verified_json(
                StateObjectKind::PeerAdvert {
                    session_id: session.session_id.clone(),
                },
                &advert.contributor_pubkey_hex,
                &advert,
            )
            .unwrap();
        store
            .mark_seen(
                StateNamespace::PeerAdverts,
                &format!(
                    "{}--{}",
                    session.session_id, advert.contributor_pubkey_hex
                ),
            )
            .unwrap();
    }
    // Simulate restart by reopening (auto_prune disabled so we
    // don't lose anything to the now-inside-window timestamp).
    let (store, report) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    assert_eq!(report.removed_sessions, 0);
    assert_eq!(report.removed_peer_adverts, 0);
    let adverts = store
        .list_verified_peer_adverts_for(&session.session_id)
        .unwrap();
    assert_eq!(adverts.len(), 1);
    assert_eq!(adverts[0].advertisement_id, advert.advertisement_id);
    assert_eq!(adverts[0].libp2p_peer_id, advert.libp2p_peer_id);
    // The seen marker survived too — caller will dedup against it.
    assert!(store
        .is_seen(
            StateNamespace::PeerAdverts,
            &format!("{}--{}", session.session_id, advert.contributor_pubkey_hex)
        )
        .unwrap());
}

// ── 3. PeerRoutingCache rebuilt from state dir alone ────────────────

#[test]
fn routing_cache_rebuilt_from_state_only() {
    let d = fresh_dir();
    let session = build_session("2026-05-30T01:00:00Z");
    let advert = build_advert(&session, "2026-05-30T01:00:00Z");
    {
        let (store, _) =
            ContributorStateStore::open(d.path(), false, &now_inside_window())
                .unwrap();
        store
            .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
            .unwrap();
        store
            .write_verified_json(
                StateObjectKind::PeerAdvert {
                    session_id: session.session_id.clone(),
                },
                &advert.contributor_pubkey_hex,
                &advert,
            )
            .unwrap();
    }
    // Restart path: rebuild the routing cache from the state-dir
    // ONLY. No separate --peer-advert-dir / --joins-dir flags.
    let (store, _report) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    let mut cache = PeerRoutingCache::new();
    for (session_id, _session) in store.list_verified_sessions().unwrap() {
        for advert in store.list_verified_peer_adverts_for(&session_id).unwrap() {
            cache.insert_verified(advert);
        }
    }
    assert_eq!(cache.len(), 1);

    let resolution = cache.resolve(
        &session.session_id,
        &advert.contributor_pubkey_hex,
        TensorDtype::F16,
        64 * 1024 * 1024,
        &now_inside_window(),
    );
    match resolution {
        RouteResolution::Found(route) => {
            assert_eq!(route.peer_id, PEER_ID_FIXTURE);
            assert_eq!(route.negotiated_dtype, TensorDtype::F16);
            assert_eq!(route.max_handoff_chunk_bytes, 32 * 1024 * 1024);
        }
        other => panic!("expected Found, got {other:?}"),
    }
}

// ── 4. Cross-restart dedup of posted-jobs ───────────────────────────

#[test]
fn restart_skips_already_seen_posted_job_via_state_store() {
    let d = fresh_dir();
    let posted_id = "deadbeef".repeat(8);
    {
        let (store, _) =
            ContributorStateStore::open(d.path(), false, &now_inside_window())
                .unwrap();
        store
            .mark_seen(StateNamespace::PostedJobs, &posted_id)
            .unwrap();
    }
    // Simulate process restart.
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    assert!(
        store
            .is_seen(StateNamespace::PostedJobs, &posted_id)
            .unwrap(),
        "posted-jobs marker must survive across restart"
    );
    // A second mark_seen for the same id is idempotent.
    store
        .mark_seen(StateNamespace::PostedJobs, &posted_id)
        .unwrap();
    assert!(store
        .is_seen(StateNamespace::PostedJobs, &posted_id)
        .unwrap());
}

// ── 5. State-dir loaders return raw bytes; callers MUST re-verify ────

/// `list_verified_joins_for` and friends do NOT perform cryptographic
/// verification — they only parse JSON. A consumer that uses the
/// returned joins as a trust source (e.g. the matching-join gate in
/// `verify_peer_advertisement_body`) MUST run
/// `verify_execution_session` + `verify_contributor_join` first.
/// This test pins that behavior so the trust boundary stays
/// explicit; if a future refactor moves verification INTO the
/// loader, this test should be updated, not silently passed.
#[test]
fn state_dir_loaders_do_not_run_signature_verification() {
    let d = fresh_dir();
    let session = build_session("2026-05-30T01:00:00Z");
    let mut tampered_join = build_join(&session);
    // Flip a hex digit in the signature — a legitimate read
    // should still return this join, but `verify_contributor_join`
    // will reject it. Mutate the LAST char so the input remains
    // even-length valid hex.
    let mut sig: Vec<char> = tampered_join.contributor_signature_hex.chars().collect();
    let last = sig.len() - 1;
    sig[last] = if sig[last] == '0' { '1' } else { '0' };
    tampered_join.contributor_signature_hex = sig.into_iter().collect();

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
            &tampered_join.contributor_pubkey_hex,
            &tampered_join,
        )
        .unwrap();
    // The loader returns the tampered join verbatim — it's a cache
    // reader, not a trust gate.
    let joins = store
        .list_verified_joins_for(&session.session_id)
        .unwrap();
    assert_eq!(joins.len(), 1);
    // Sanity: the loaded join must fail signature verification
    // when run through the real verifier — proving the trust
    // boundary lives in the caller.
    assert!(
        !omni_contributor::verify_contributor_join(&session, &joins[0]).is_ok(),
        "tampered join must fail verify_contributor_join — \
         callers must NOT trust list_verified_joins_for output \
         as already-verified"
    );
}

// ── Stage 12.13 — restart preload hardening ─────────────────────
//
// `load_verified_restart_snapshot` walks the state-dir in
// dependency order (sessions → joins → assignments →
// supersessions), re-runs each Stage 12.3 / 12.11 verifier, and
// returns the typed snapshot the `watch-sessions` restart preload
// uses to warm its in-memory caches. These tests pin:
//   - happy-path: every artifact re-verifies and is included.
//   - rejection: a forged supersession is dropped with a structured
//     note carrying the SessionVerifyOutcome reason_tag.
//   - cache shape: the assignments map is populated per-session so
//     the watcher's out-of-order supersession handler can resolve
//     references from memory.

use omni_contributor::{
    canonical::{
        assignment_id_hex, supersession_id_hex, work_assignment_signing_input,
        work_assignment_supersession_signing_input,
    },
    session::{WorkAssignment, WorkKind},
    supersession::{SupersessionReason, WorkAssignmentSupersession},
    SESSION_SCHEMA_VERSION as SESSION_V,
};

fn build_assignment(
    session: &ExecutionSession,
    contrib: &ContributorSigner,
    coord: &CoordinatorSigner,
    stage_index: u32,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_V,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index,
        contributor_pubkey_hex: contrib.pubkey_hex(),
        work_kind: WorkKind::Layers {
            start: stage_index * 8,
            end: stage_index * 8 + 8,
        },
        expected_work_units: 8,
        expected_work_unit_kind: WorkUnitKind::Layers,
        assigned_at_utc: "2026-05-30T00:00:05Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

fn build_supersession(
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
        schema_version: omni_contributor::SUPERSESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        supersession_id: String::new(),
        superseded_assignment_ids: superseded,
        replacement_assignment_ids: replacement,
        reason: SupersessionReason::MissingPartial,
        created_at_utc: "2026-05-30T00:00:10Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    s.supersession_id = supersession_id_hex(&s).unwrap();
    let si = work_assignment_supersession_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn write_session_chain_into_state(
    store: &ContributorStateStore,
    session: &ExecutionSession,
    joins: &[ContributorJoin],
    assignments: &[WorkAssignment],
    supersessions: &[WorkAssignmentSupersession],
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
    for s in supersessions {
        store
            .write_verified_json(
                StateObjectKind::AssignmentSupersession {
                    session_id: session.session_id.clone(),
                },
                &s.supersession_id,
                s,
            )
            .unwrap();
    }
}

const FAR_FUTURE_V13: &str = "2026-12-31T23:59:59Z";

#[test]
fn restart_preload_loads_assignments_and_caches_them() {
    let dir = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(dir.path(), false, &now_inside_window()).unwrap();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = build_session(FAR_FUTURE_V13);
    let join = build_join(&session);
    let asn_0 = build_assignment(&session, &contrib, &coord, 0);
    let asn_1 = build_assignment(&session, &contrib, &coord, 1);
    let asn_2 = build_assignment(&session, &contrib, &coord, 2);
    write_session_chain_into_state(
        &store,
        &session,
        &[join],
        &[asn_0.clone(), asn_1.clone(), asn_2.clone()],
        &[],
    );

    // Reopen and run the Stage 12.13 preload.
    let (store, _) =
        ContributorStateStore::open(dir.path(), false, &now_inside_window()).unwrap();
    let (snapshot, report) =
        omni_ops::load_verified_restart_snapshot(&store).unwrap();
    assert_eq!(report.sessions_accepted, 1);
    assert_eq!(report.assignments_accepted, 3);
    assert_eq!(report.supersessions_accepted, 0);
    assert!(report.rejection_notes.is_empty());
    let cached = snapshot
        .assignments_by_session
        .get(&session.session_id)
        .expect("session cached with assignments");
    assert_eq!(cached.len(), 3);
    let ids: std::collections::HashSet<String> =
        cached.iter().map(|a| a.assignment_id.clone()).collect();
    assert!(ids.contains(&asn_0.assignment_id));
    assert!(ids.contains(&asn_1.assignment_id));
    assert!(ids.contains(&asn_2.assignment_id));
}

#[test]
fn restart_after_supersession_yields_same_status_report() {
    let dir = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(dir.path(), false, &now_inside_window()).unwrap();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = build_session(FAR_FUTURE_V13);
    let join = build_join(&session);
    let asn_0 = build_assignment(&session, &contrib, &coord, 0);
    let asn_1 = build_assignment(&session, &contrib, &coord, 1);
    let sup = build_supersession(
        &session,
        &coord,
        vec![asn_0.assignment_id.clone()],
        vec![asn_1.assignment_id.clone()],
    );
    write_session_chain_into_state(
        &store,
        &session,
        &[join],
        &[asn_0.clone(), asn_1.clone()],
        &[sup.clone()],
    );

    // Build status report from the original store.
    let report_a = omni_ops::build_session_status_report(
        &store,
        &session.session_id,
        &now_inside_window(),
        false,
    )
    .unwrap();

    // Reopen the same state-dir; build the status report again.
    let (store_reopen, _) =
        ContributorStateStore::open(dir.path(), false, &now_inside_window()).unwrap();
    let report_b = omni_ops::build_session_status_report(
        &store_reopen,
        &session.session_id,
        &now_inside_window(),
        false,
    )
    .unwrap();

    // Strip the timestamp-y fields that legitimately differ; the
    // chain shape must be bit-equal.
    let strip = |r: omni_ops::SessionStatusReport| {
        let mut r = r;
        r.generated_at_utc.clear();
        r.notes.clear();
        r
    };
    assert_eq!(strip(report_a), strip(report_b));
}

#[test]
fn restart_preload_drops_corrupted_supersession_with_note() {
    let dir = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(dir.path(), false, &now_inside_window()).unwrap();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let wrong = CoordinatorSigner::from_seed_bytes(&[0xAB; 32]).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = build_session(FAR_FUTURE_V13);
    let join = build_join(&session);
    let asn = build_assignment(&session, &contrib, &coord, 0);

    // Forge a supersession signed by the wrong coordinator key.
    // The body advertises the real coord, so
    // `verify_assignment_supersession` fails on the
    // SupersessionCoordinatorSignatureFailed leg. Use a distinct
    // 64-hex replacement id so the schema's disjointness check
    // passes (the verifier reaches the signature leg).
    let mut sup = build_supersession(
        &session,
        &coord,
        vec![asn.assignment_id.clone()],
        vec!["cd".repeat(32)],
    );
    // After tampering with the sig, the body still re-derives a
    // stable supersession_id from its canonical body — we keep
    // the body as-is and just replace the signature with the
    // rogue's.
    let si = work_assignment_supersession_signing_input(&sup).unwrap();
    sup.coordinator_signature_hex = wrong.sign_hex(&si);
    write_session_chain_into_state(
        &store,
        &session,
        &[join],
        &[asn.clone()],
        &[sup.clone()],
    );

    let (store, _) =
        ContributorStateStore::open(dir.path(), false, &now_inside_window()).unwrap();
    let (snapshot, report) =
        omni_ops::load_verified_restart_snapshot(&store).unwrap();
    assert_eq!(report.sessions_accepted, 1);
    assert_eq!(report.supersessions_accepted, 0);
    assert_eq!(report.supersessions_rejected, 1);
    let note = report
        .rejection_notes
        .iter()
        .find(|n| n.contains("kind=supersession"))
        .expect("a rejection note for the forged supersession");
    assert!(
        note.contains("reason_tag=SupersessionCoordinatorSignatureFailed"),
        "reason_tag must come from SessionVerifyOutcome::reason_tag(); got {note}"
    );
    assert!(snapshot
        .supersessions_by_session
        .get(&session.session_id)
        .is_none());
}
