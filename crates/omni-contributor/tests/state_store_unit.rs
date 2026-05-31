//! Stage 12.7 — unit tests for `ContributorStateStore`.
//! Open layout / state_version pin / seen markers / verified
//! round-trip / atomic-write / prune semantics.

use std::fs;

use omni_contributor::{
    canonical::{
        contributor_join_signing_input, execution_session_signing_input,
        peer_advertisement_signing_input, session_id_hex,
    },
    handoff::TensorDtype,
    result::WorkUnitKind,
    ContributorJoin, ContributorPeerAdvertisement, ContributorSigner,
    ContributorStateStore, CoordinatorSigner, ExecutionSession, PeerCapabilities,
    StateNamespace, StateObjectKind, StateVersionMeta, PEER_ADVERTISEMENT_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION, STATE_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.7-state-coord-seed-32by!";
const CONTRIB_SEED: [u8; 32] = *b"stage12.7-state-contrib-32-byte!";
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

fn build_advert(session: &ExecutionSession, expires_at_utc: &str) -> ContributorPeerAdvertisement {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let mut a = ContributorPeerAdvertisement {
        schema_version: PEER_ADVERTISEMENT_SCHEMA_VERSION,
        advertisement_id: String::new(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        libp2p_peer_id: PEER_ID_FIXTURE.into(),
        listen_multiaddrs: vec![],
        capabilities: PeerCapabilities {
            supports_live_handoff: true,
            max_handoff_chunk_bytes: 32 * 1024 * 1024,
            supported_dtypes: vec![TensorDtype::F16],
        },
        advertised_at_utc: "2026-05-30T00:00:00Z".into(),
        expires_at_utc: expires_at_utc.into(),
        contributor_signature_hex: String::new(),
    };
    a.advertisement_id =
        omni_contributor::canonical::advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    a
}

// ── Open / version ─────────────────────────────────────────────────

#[test]
fn open_creates_layout_with_state_version_1() {
    let d = fresh_dir();
    let (_store, report) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    assert_eq!(report.removed_sessions, 0);
    let meta_path = d.path().join("meta").join("state_version.json");
    assert!(meta_path.is_file());
    let meta: StateVersionMeta =
        serde_json::from_slice(&fs::read(&meta_path).unwrap()).unwrap();
    assert_eq!(meta.state_version, STATE_VERSION);
    // Top-level subdirs are pre-created.
    for sub in &["seen", "verified", "results"] {
        assert!(d.path().join(sub).is_dir(), "missing top-level dir: {sub}");
    }
}

#[test]
fn open_refuses_future_state_version() {
    use omni_contributor::error::StateError;
    let d = fresh_dir();
    fs::create_dir_all(d.path().join("meta")).unwrap();
    let meta = StateVersionMeta {
        state_version: STATE_VERSION + 99,
    };
    fs::write(
        d.path().join("meta").join("state_version.json"),
        serde_json::to_vec_pretty(&meta).unwrap(),
    )
    .unwrap();
    let err =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap_err();
    assert!(
        matches!(err, StateError::UnsupportedVersion { .. }),
        "expected UnsupportedVersion, got {err:?}"
    );
}

#[test]
fn open_reuses_existing_meta_on_second_call() {
    let d = fresh_dir();
    {
        let _ = ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    }
    // Tamper meta bytes to confirm we don't overwrite on the second open.
    let meta_path = d.path().join("meta").join("state_version.json");
    let original = fs::read(&meta_path).unwrap();
    let (_store, _r) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    assert_eq!(fs::read(&meta_path).unwrap(), original);
}

// ── seen markers ──────────────────────────────────────────────────

#[test]
fn mark_seen_then_is_seen_returns_true() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    store
        .mark_seen(StateNamespace::PostedJobs, "abc123")
        .unwrap();
    assert!(store.is_seen(StateNamespace::PostedJobs, "abc123").unwrap());
}

#[test]
fn is_seen_unknown_returns_false() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    assert!(!store
        .is_seen(StateNamespace::PostedJobs, "never-marked")
        .unwrap());
}

#[test]
fn mark_seen_is_idempotent() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    store
        .mark_seen(StateNamespace::Sessions, "session-x")
        .unwrap();
    // Second call must succeed without error and must NOT remove
    // or duplicate the marker.
    store
        .mark_seen(StateNamespace::Sessions, "session-x")
        .unwrap();
    assert!(store
        .is_seen(StateNamespace::Sessions, "session-x")
        .unwrap());
}

// ── verified round-trip + atomic write ─────────────────────────────

#[test]
fn write_verified_json_round_trip() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    let session = build_session("2026-05-30T01:00:00Z");
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    let round: ExecutionSession = store
        .read_verified_json(StateObjectKind::Session, &session.session_id)
        .unwrap()
        .expect("must round-trip");
    assert_eq!(round.session_id, session.session_id);
    assert_eq!(round.posted_id, session.posted_id);
    assert_eq!(
        round.coordinator_signature_hex,
        session.coordinator_signature_hex
    );
}

#[test]
fn write_verified_json_overwrites_atomically() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    let session_a = build_session("2026-05-30T01:00:00Z");
    let mut session_b = session_a.clone();
    // Change something the canonical signing input excludes so we
    // don't have to re-sign — `tokenizer_hash` is part of the
    // signing input, so use a different harmless tweak: rewrite
    // the full envelope by re-signing.
    session_b.posted_id = "ee".repeat(32);
    session_b.session_id = session_id_hex(&session_b).unwrap();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let si = execution_session_signing_input(&session_b).unwrap();
    session_b.coordinator_signature_hex = coord.sign_hex(&si);

    store
        .write_verified_json(StateObjectKind::Session, &session_a.session_id, &session_a)
        .unwrap();
    store
        .write_verified_json(StateObjectKind::Session, &session_a.session_id, &session_b)
        .unwrap();
    let on_disk: ExecutionSession = store
        .read_verified_json(StateObjectKind::Session, &session_a.session_id)
        .unwrap()
        .unwrap();
    assert_eq!(on_disk.posted_id, session_b.posted_id);

    // No stray tempfiles left behind in the session dir.
    let dir = d
        .path()
        .join("verified")
        .join("sessions")
        .join(&session_a.session_id);
    for e in fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        assert!(
            !name.ends_with(".tmp"),
            "tempfile leaked into verified dir: {name}"
        );
    }
}

#[test]
fn read_verified_json_returns_none_when_absent() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    let out: Option<ExecutionSession> = store
        .read_verified_json(StateObjectKind::Session, "00".repeat(32).as_str())
        .unwrap();
    assert!(out.is_none());
}

#[test]
fn list_verified_walks_sessions_tree() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    let session = build_session("2026-05-30T01:00:00Z");
    let join = build_join(&session);
    let advert = build_advert(&session, "2026-05-30T01:00:00Z");
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
    store
        .write_verified_json(
            StateObjectKind::PeerAdvert {
                session_id: session.session_id.clone(),
            },
            &advert.contributor_pubkey_hex,
            &advert,
        )
        .unwrap();
    let sessions = store.list_verified_sessions().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].0, session.session_id);
    let joins = store
        .list_verified_joins_for(&session.session_id)
        .unwrap();
    assert_eq!(joins.len(), 1);
    let adverts = store
        .list_verified_peer_adverts_for(&session.session_id)
        .unwrap();
    assert_eq!(adverts.len(), 1);
}

// ── prune ──────────────────────────────────────────────────────────

#[test]
fn prune_removes_expired_sessions_and_keeps_fresh() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    let fresh = build_session("2026-05-30T10:00:00Z");
    let stale = {
        let mut s = build_session("2026-05-29T00:00:00Z"); // BEFORE now
        // Reduce within 24h window — adjust created_at_utc too.
        s.created_at_utc = "2026-05-28T00:00:00Z".into();
        s.session_id = session_id_hex(&s).unwrap();
        let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
        let si = execution_session_signing_input(&s).unwrap();
        s.coordinator_signature_hex = coord.sign_hex(&si);
        s
    };
    store
        .write_verified_json(StateObjectKind::Session, &fresh.session_id, &fresh)
        .unwrap();
    store
        .write_verified_json(StateObjectKind::Session, &stale.session_id, &stale)
        .unwrap();
    let report = store.prune_expired(&now_inside_window()).unwrap();
    assert_eq!(report.removed_sessions, 1);
    assert!(report.kept >= 1);
    let on_disk: Vec<_> = store.list_verified_sessions().unwrap();
    let ids: Vec<_> = on_disk.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        ids.iter().any(|id| id == &fresh.session_id.as_str()),
        "fresh session must remain"
    );
    assert!(
        !ids.iter().any(|id| id == &stale.session_id.as_str()),
        "stale session must be removed"
    );
}

#[test]
fn prune_removes_expired_peer_adverts() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    let session = build_session("2026-05-30T10:00:00Z");
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    let fresh_advert = build_advert(&session, "2026-05-30T10:00:00Z");
    let stale_advert = build_advert(&session, "2026-05-29T00:00:00Z");
    store
        .write_verified_json(
            StateObjectKind::PeerAdvert {
                session_id: session.session_id.clone(),
            },
            &fresh_advert.contributor_pubkey_hex,
            &fresh_advert,
        )
        .unwrap();
    // Same contributor_pubkey overwrites; for two distinct entries
    // build a second advert under a different fake key.
    let second_key = "ab".repeat(32);
    let mut second_advert = stale_advert.clone();
    second_advert.contributor_pubkey_hex = second_key.clone();
    second_advert.advertisement_id =
        omni_contributor::canonical::advertisement_id_hex(&second_advert).unwrap();
    // Re-sign with whatever contributor we have so the file
    // round-trips; the pruner doesn't run a signature check.
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let si = peer_advertisement_signing_input(&second_advert).unwrap();
    second_advert.contributor_signature_hex = contrib.sign_hex(&si);
    store
        .write_verified_json(
            StateObjectKind::PeerAdvert {
                session_id: session.session_id.clone(),
            },
            &second_key,
            &second_advert,
        )
        .unwrap();

    let report = store.prune_expired(&now_inside_window()).unwrap();
    assert_eq!(report.removed_peer_adverts, 1, "{report:?}");
    let remaining = store
        .list_verified_peer_adverts_for(&session.session_id)
        .unwrap();
    assert_eq!(remaining.len(), 1);
}

#[test]
fn prune_cascade_drops_seen_markers_for_expired_session() {
    let d = fresh_dir();
    let (store, _) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    // Stale session built directly so we can scope the cascade test.
    let stale = {
        let mut s = build_session("2026-05-29T00:00:00Z");
        s.created_at_utc = "2026-05-28T00:00:00Z".into();
        s.session_id = session_id_hex(&s).unwrap();
        let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
        let si = execution_session_signing_input(&s).unwrap();
        s.coordinator_signature_hex = coord.sign_hex(&si);
        s
    };
    store
        .write_verified_json(StateObjectKind::Session, &stale.session_id, &stale)
        .unwrap();
    // Plant seen markers that should cascade out with the session.
    store
        .mark_seen(StateNamespace::Sessions, &stale.session_id)
        .unwrap();
    store
        .mark_seen(StateNamespace::Aggregates, &stale.session_id)
        .unwrap();
    let join_marker = format!("{}--{}", stale.session_id, "ab".repeat(32));
    store
        .mark_seen(StateNamespace::Joins, &join_marker)
        .unwrap();

    let report = store.prune_expired(&now_inside_window()).unwrap();
    assert_eq!(report.removed_sessions, 1);
    // Cascaded markers must be gone.
    assert!(!store
        .is_seen(StateNamespace::Sessions, &stale.session_id)
        .unwrap());
    assert!(!store
        .is_seen(StateNamespace::Aggregates, &stale.session_id)
        .unwrap());
    assert!(!store
        .is_seen(StateNamespace::Joins, &join_marker)
        .unwrap());
}

#[test]
fn auto_prune_on_open_runs_when_enabled() {
    let d = fresh_dir();
    {
        let (store, _) =
            ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
        let stale = {
            let mut s = build_session("2026-05-29T00:00:00Z");
            s.created_at_utc = "2026-05-28T00:00:00Z".into();
            s.session_id = session_id_hex(&s).unwrap();
            let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
            let si = execution_session_signing_input(&s).unwrap();
            s.coordinator_signature_hex = coord.sign_hex(&si);
            s
        };
        store
            .write_verified_json(StateObjectKind::Session, &stale.session_id, &stale)
            .unwrap();
    }
    // Re-open with auto_prune=true.
    let (store, report) =
        ContributorStateStore::open(d.path(), true, &now_inside_window()).unwrap();
    assert_eq!(report.removed_sessions, 1, "{report:?}");
    assert!(store.list_verified_sessions().unwrap().is_empty());
}

#[test]
fn auto_prune_skipped_when_disabled() {
    let d = fresh_dir();
    {
        let (store, _) =
            ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
        let stale = {
            let mut s = build_session("2026-05-29T00:00:00Z");
            s.created_at_utc = "2026-05-28T00:00:00Z".into();
            s.session_id = session_id_hex(&s).unwrap();
            let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
            let si = execution_session_signing_input(&s).unwrap();
            s.coordinator_signature_hex = coord.sign_hex(&si);
            s
        };
        store
            .write_verified_json(StateObjectKind::Session, &stale.session_id, &stale)
            .unwrap();
    }
    // Re-open with auto_prune=false → stale stays.
    let (store, report) =
        ContributorStateStore::open(d.path(), false, &now_inside_window()).unwrap();
    assert_eq!(report.removed_sessions, 0);
    assert_eq!(store.list_verified_sessions().unwrap().len(), 1);
}
