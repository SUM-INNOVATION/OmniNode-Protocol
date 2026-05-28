//! Stage 12.5 — `PeerRoutingCache` insert/resolve semantics:
//! newest non-expired wins, dtype mismatch refused, chunk cap
//! negotiated as `min(local, advertised)`, expiry strict in
//! `resolve`.

use omni_contributor::{
    canonical::{
        advertisement_id_hex, execution_session_signing_input,
        peer_advertisement_signing_input, session_id_hex,
    },
    handoff::TensorDtype,
    ContributorPeerAdvertisement, ContributorSigner, CoordinatorSigner,
    ExecutionSession, PeerCapabilities, PeerRoutingCache, ResolvedPeerRoute, RouteResolution,
    PEER_ADVERTISEMENT_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.5-cache-coord-32-bytes!!";
const CONTRIB_SEED: [u8; 32] = *b"stage12.5-cache-contrib-32-byte!";
const PEER_ID_FIXTURE: &str = "12D3KooWGjMyD3v8UYjK3bV7Sqg3WcWzkF7QbTtwYNRsBwCb6KpC";

fn session() -> ExecutionSession {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-05-28T00:00:00Z".into(),
        expires_at_utc: "2026-05-29T00:00:00Z".into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn signed_advert(
    s: &ExecutionSession,
    contrib: &ContributorSigner,
    advertised_at_utc: &str,
    expires_at_utc: &str,
    cap: u64,
    dtypes: Vec<TensorDtype>,
    supports_live_handoff: bool,
) -> ContributorPeerAdvertisement {
    let mut a = ContributorPeerAdvertisement {
        schema_version: PEER_ADVERTISEMENT_SCHEMA_VERSION,
        advertisement_id: String::new(),
        session_id: s.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        libp2p_peer_id: PEER_ID_FIXTURE.into(),
        listen_multiaddrs: vec!["/ip4/127.0.0.1/udp/4001/quic-v1".into()],
        capabilities: PeerCapabilities {
            supports_live_handoff,
            max_handoff_chunk_bytes: cap,
            supported_dtypes: dtypes,
        },
        advertised_at_utc: advertised_at_utc.into(),
        expires_at_utc: expires_at_utc.into(),
        contributor_signature_hex: String::new(),
    };
    a.advertisement_id = advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    a
}

#[test]
fn resolve_unknown_session_returns_no_advertisement() {
    let cache = PeerRoutingCache::new();
    let out = cache.resolve(
        &"aa".repeat(32),
        &"bb".repeat(32),
        TensorDtype::F16,
        16,
        "2026-05-28T00:00:30Z",
    );
    assert!(matches!(out, RouteResolution::NoAdvertisement), "{out:?}");
}

#[test]
fn resolve_chunk_cap_is_min_of_local_and_advertised() {
    let s = session();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let a = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:10Z",
        "2026-05-28T01:00:00Z",
        /* advertised cap = */ 32 * 1024 * 1024,
        vec![TensorDtype::F16],
        true,
    );
    let mut cache = PeerRoutingCache::new();
    cache.insert_verified(a.clone());
    // Local cap below advertised → resolution returns local.
    let r = cache.resolve(
        &s.session_id,
        &contrib.pubkey_hex(),
        TensorDtype::F16,
        4 * 1024 * 1024,
        "2026-05-28T00:00:30Z",
    );
    let ResolvedPeerRoute {
        peer_id,
        max_handoff_chunk_bytes,
        ..
    } = match r {
        RouteResolution::Found(rr) => rr,
        other => panic!("expected Found, got {other:?}"),
    };
    assert_eq!(peer_id, PEER_ID_FIXTURE);
    assert_eq!(max_handoff_chunk_bytes, 4 * 1024 * 1024);

    // Local cap above advertised → resolution returns advertised.
    let r = cache.resolve(
        &s.session_id,
        &contrib.pubkey_hex(),
        TensorDtype::F16,
        128 * 1024 * 1024,
        "2026-05-28T00:00:30Z",
    );
    let ResolvedPeerRoute {
        max_handoff_chunk_bytes,
        ..
    } = match r {
        RouteResolution::Found(rr) => rr,
        other => panic!("expected Found, got {other:?}"),
    };
    assert_eq!(max_handoff_chunk_bytes, 32 * 1024 * 1024);
}

#[test]
fn resolve_dtype_not_supported_refused() {
    let s = session();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let a = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:10Z",
        "2026-05-28T01:00:00Z",
        32 * 1024 * 1024,
        vec![TensorDtype::F16],
        true,
    );
    let mut cache = PeerRoutingCache::new();
    cache.insert_verified(a);
    let r = cache.resolve(
        &s.session_id,
        &contrib.pubkey_hex(),
        TensorDtype::Bf16,
        4 * 1024 * 1024,
        "2026-05-28T00:00:30Z",
    );
    assert!(matches!(r, RouteResolution::DtypeNotSupported { .. }), "{r:?}");
}

#[test]
fn resolve_live_handoff_not_supported_refused() {
    let s = session();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let a = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:10Z",
        "2026-05-28T01:00:00Z",
        32 * 1024 * 1024,
        vec![TensorDtype::F16],
        false,
    );
    let mut cache = PeerRoutingCache::new();
    cache.insert_verified(a);
    let r = cache.resolve(
        &s.session_id,
        &contrib.pubkey_hex(),
        TensorDtype::F16,
        4 * 1024 * 1024,
        "2026-05-28T00:00:30Z",
    );
    assert!(matches!(r, RouteResolution::LiveHandoffNotSupported), "{r:?}");
}

#[test]
fn resolve_expired_refused() {
    let s = session();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let a = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:10Z",
        "2026-05-28T00:30:00Z",
        32 * 1024 * 1024,
        vec![TensorDtype::F16],
        true,
    );
    let mut cache = PeerRoutingCache::new();
    cache.insert_verified(a);
    // now > expires_at_utc.
    let r = cache.resolve(
        &s.session_id,
        &contrib.pubkey_hex(),
        TensorDtype::F16,
        4 * 1024 * 1024,
        "2026-05-28T01:00:00Z",
    );
    assert!(matches!(r, RouteResolution::AllExpired { .. }), "{r:?}");
}

#[test]
fn insert_newer_advertised_at_wins() {
    let s = session();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let older = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:10Z",
        "2026-05-28T01:00:00Z",
        16 * 1024 * 1024,
        vec![TensorDtype::F16],
        true,
    );
    let newer = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:20Z",
        "2026-05-28T01:00:00Z",
        64 * 1024 * 1024,
        vec![TensorDtype::F16, TensorDtype::Bf16],
        true,
    );
    let mut cache = PeerRoutingCache::new();
    cache.insert_verified(older);
    cache.insert_verified(newer);
    // The newer cap (64 MiB) is what gets cached. We resolve with
    // a 128 MiB local cap so the returned value is the advertised
    // cap, which proves the newer entry won.
    let r = cache.resolve(
        &s.session_id,
        &contrib.pubkey_hex(),
        TensorDtype::Bf16,
        128 * 1024 * 1024,
        "2026-05-28T00:00:30Z",
    );
    let ResolvedPeerRoute {
        max_handoff_chunk_bytes,
        ..
    } = match r {
        RouteResolution::Found(rr) => rr,
        other => panic!("expected Found, got {other:?}"),
    };
    assert_eq!(max_handoff_chunk_bytes, 64 * 1024 * 1024);
    assert_eq!(cache.len(), 1);
}

#[test]
fn resolve_expiry_uses_chrono_not_string_compare() {
    // Regression: prior to the chrono-based comparison fix, a
    // `now_utc` of `2026-05-28T00:30:00Z` and an `expires_at_utc`
    // of `2026-05-28T00:30:00.5Z` (later by 500 ms) would
    // string-compare as `now > expires` because `.5Z` sorts
    // after `Z` lexicographically — wrong. With chrono-based
    // parsing the resolver correctly sees the fractional variant
    // as *later* and returns `Found`.
    let s = session();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let mut a = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:10Z",
        // Half-second fractional expiry. Schema accepts; resolver
        // must parse, not string-compare.
        "2026-05-28T00:30:00.500Z",
        32 * 1024 * 1024,
        vec![TensorDtype::F16],
        true,
    );
    // Re-derive advertisement_id + sign over the fractional
    // expiry timestamp (we mutated it after the helper's signing).
    a.advertisement_id =
        omni_contributor::canonical::advertisement_id_hex(&a).unwrap();
    let si = omni_contributor::canonical::peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    let mut cache = PeerRoutingCache::new();
    cache.insert_verified(a);
    // now = 30:00 sharp. Expires at 30:00.5. Resolver must say "found".
    let r = cache.resolve(
        &s.session_id,
        &contrib.pubkey_hex(),
        TensorDtype::F16,
        4 * 1024 * 1024,
        "2026-05-28T00:30:00Z",
    );
    match r {
        RouteResolution::Found(_) => {}
        other => panic!(
            "fractional-second expiry must be honored via chrono parse, \
             not string compare; got {other:?}"
        ),
    }
}

#[test]
fn insert_older_advertised_at_does_not_evict() {
    let s = session();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let newer = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:20Z",
        "2026-05-28T01:00:00Z",
        64 * 1024 * 1024,
        vec![TensorDtype::F16, TensorDtype::Bf16],
        true,
    );
    let older = signed_advert(
        &s,
        &contrib,
        "2026-05-28T00:00:10Z",
        "2026-05-28T01:00:00Z",
        16 * 1024 * 1024,
        vec![TensorDtype::F16],
        true,
    );
    let mut cache = PeerRoutingCache::new();
    cache.insert_verified(newer);
    cache.insert_verified(older);
    // The newer entry must still win: cap returned must be 64 MiB.
    let r = cache.resolve(
        &s.session_id,
        &contrib.pubkey_hex(),
        TensorDtype::F16,
        128 * 1024 * 1024,
        "2026-05-28T00:00:30Z",
    );
    let ResolvedPeerRoute {
        max_handoff_chunk_bytes,
        ..
    } = match r {
        RouteResolution::Found(rr) => rr,
        other => panic!("expected Found, got {other:?}"),
    };
    assert_eq!(max_handoff_chunk_bytes, 64 * 1024 * 1024);
    assert_eq!(cache.len(), 1);
}
