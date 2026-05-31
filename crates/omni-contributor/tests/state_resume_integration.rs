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
