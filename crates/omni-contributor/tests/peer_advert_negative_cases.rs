//! Stage 12.5 — schema / signature / drift / binding / expiry
//! negative tests for `ContributorPeerAdvertisement`,
//! `NetworkPeerAdvertisementAnnouncement`, and the
//! `process_peer_advertisement_announcement` pipeline.

mod common;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{
        advertisement_id_hex, assignment_id_hex, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, net_peer_advert_signing_input,
        peer_advertisement_signing_input, session_id_hex, work_assignment_signing_input,
    },
    error::SchemaError,
    handoff::{TensorDtype, HANDOFF_CHUNK_MAX_BYTES},
    process_peer_advertisement_announcement,
    result::WorkUnitKind,
    ContributorJoin, ContributorPeerAdvertisement, ContributorSigner, CoordinatorSigner,
    ExecutionSession, NetworkPeerAdvertisementAnnouncement, PeerAdvertisementOutcome,
    PeerCapabilities, WorkAssignment, WorkKind, NET_SCHEMA_VERSION,
    PEER_ADVERTISEMENT_MAX_LIFETIME_SECS, PEER_ADVERTISEMENT_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.5-neg-coord-32-byte-key!";
const CONTRIB_SEED: [u8; 32] = *b"stage12.5-neg-contrib-seed-32by!";
const ROGUE_SEED: [u8; 32] = *b"stage12.5-neg-rogue-seed-32-byte";
const ANNOUNCER_SEED: [u8; 32] = *b"stage12.5-neg-announcer-32-byte!";
const PEER_ID_FIXTURE: &str = "12D3KooWGjMyD3v8UYjK3bV7Sqg3WcWzkF7QbTtwYNRsBwCb6KpC";
const OTHER_PEER_ID: &str = "12D3KooWNJSE2Db8aFmgTGKbFvT4yfRbN1XPwbHTFLDqEgCFQpZx";

fn signed_session(coord: &CoordinatorSigner) -> ExecutionSession {
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

fn signed_join(session: &ExecutionSession, contrib: &ContributorSigner) -> ContributorJoin {
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 100,
        max_output_tokens: 100,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-28T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

fn signed_assignment(
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
        assigned_at_utc: "2026-05-28T00:00:02Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

fn signed_advert(
    session: &ExecutionSession,
    contrib: &ContributorSigner,
) -> ContributorPeerAdvertisement {
    let mut a = ContributorPeerAdvertisement {
        schema_version: PEER_ADVERTISEMENT_SCHEMA_VERSION,
        advertisement_id: String::new(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        libp2p_peer_id: PEER_ID_FIXTURE.into(),
        listen_multiaddrs: vec!["/ip4/127.0.0.1/udp/4001/quic-v1".into()],
        capabilities: PeerCapabilities {
            supports_live_handoff: true,
            max_handoff_chunk_bytes: 32 * 1024 * 1024,
            supported_dtypes: vec![TensorDtype::F16],
        },
        advertised_at_utc: "2026-05-28T00:00:10Z".into(),
        expires_at_utc: "2026-05-28T01:00:00Z".into(),
        contributor_signature_hex: String::new(),
    };
    a.advertisement_id = advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    a
}

fn signed_announcement_for(
    advert: &ContributorPeerAdvertisement,
    snip_root_hex: &str,
    announcer: &CoordinatorSigner,
) -> NetworkPeerAdvertisementAnnouncement {
    let mut ann = NetworkPeerAdvertisementAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        peer_advertisement_snip_root: snip_root_hex.to_string(),
        advertisement_id: advert.advertisement_id.clone(),
        session_id: advert.session_id.clone(),
        contributor_pubkey_hex: advert.contributor_pubkey_hex.clone(),
        announced_at_utc: "2026-05-28T00:00:11Z".into(),
        announcer_pubkey_hex: announcer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = net_peer_advert_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&si);
    ann
}

// ── Schema-level negatives ─────────────────────────────────────────────

#[test]
fn schema_malformed_peer_id_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.libp2p_peer_id = "not-a-real-peer-id".into();
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::PeerAdvertLibp2pPeerIdMalformed { .. })
    ));
}

#[test]
fn schema_malformed_multiaddr_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.listen_multiaddrs = vec!["totally not a multiaddr".into()];
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::PeerAdvertMultiaddrMalformed { .. })
    ));
}

#[test]
fn schema_multiaddr_p2p_mismatch_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    // multiaddr's /p2p/... is OTHER_PEER_ID; libp2p_peer_id is
    // PEER_ID_FIXTURE.
    a.listen_multiaddrs = vec![format!(
        "/ip4/127.0.0.1/udp/4001/quic-v1/p2p/{OTHER_PEER_ID}"
    )];
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::PeerAdvertMultiaddrP2pMismatch { .. })
    ));
}

#[test]
fn schema_missing_z_timestamp_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.advertised_at_utc = "2026-05-28T00:00:10".into();
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::MalformedTimestamp { .. })
    ));
}

#[test]
fn schema_expiry_beyond_24h_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    // advertised_at + 24h = 2026-05-29T00:00:10Z. Expiry one second
    // beyond that.
    a.expires_at_utc = "2026-05-29T00:00:11Z".into();
    let err = a.validate_schema();
    assert!(matches!(err, Err(SchemaError::PeerAdvertExpiryTooFar { .. })), "{err:?}");
}

#[test]
fn schema_expiry_at_24h_accepted() {
    // Boundary: exactly +24h is allowed.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.expires_at_utc = "2026-05-29T00:00:10Z".into();
    a.advertisement_id = advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    let _ = PEER_ADVERTISEMENT_MAX_LIFETIME_SECS; // documentation anchor
    assert!(a.validate_schema().is_ok());
}

#[test]
fn schema_expiry_not_after_advertised_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.expires_at_utc = a.advertised_at_utc.clone();
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::PeerAdvertExpiryNotAfterAdvertised { .. })
    ));
}

#[test]
fn schema_unsupported_version_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.schema_version = 99;
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::UnsupportedVersion { got: 99 })
    ));
}

#[test]
fn schema_empty_dtype_list_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.capabilities.supported_dtypes.clear();
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::PeerAdvertSupportedDtypesEmpty)
    ));
}

#[test]
fn schema_zero_chunk_cap_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.capabilities.max_handoff_chunk_bytes = 0;
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::PeerAdvertChunkCapZero)
    ));
}

#[test]
fn schema_chunk_cap_too_large_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut a = signed_advert(&s, &contrib);
    a.capabilities.max_handoff_chunk_bytes = HANDOFF_CHUNK_MAX_BYTES + 1;
    assert!(matches!(
        a.validate_schema(),
        Err(SchemaError::PeerAdvertChunkCapTooLarge { .. })
    ));
}

// ── Processor / signature / drift negatives ────────────────────────────

fn put_advert_on_snip(snip: &MockSnipStore, a: &ContributorPeerAdvertisement) -> String {
    let bytes = serde_json::to_vec(a).unwrap();
    snip.insert_bytes(&bytes).to_hex()
}

#[test]
fn process_happy_path_verifies() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let _a = signed_assignment(&s, &contrib.pubkey_hex(), &coord);
    let advert = signed_advert(&s, &contrib);
    let root = put_advert_on_snip(&snip, &advert);
    let ann = signed_announcement_for(&advert, &root, &announcer);
    let out = process_peer_advertisement_announcement(
        &ann,
        &snip,
        &[j],
        Some("2026-05-28T00:00:20Z"),
    );
    assert!(out.is_verified(), "{out:?}");
}

#[test]
fn process_tampered_announcer_signature_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let advert = signed_advert(&s, &contrib);
    let root = put_advert_on_snip(&snip, &advert);
    let mut ann = signed_announcement_for(&advert, &root, &announcer);
    let mut sig = ann.announcer_signature_hex.into_bytes();
    sig[0] = if sig[0] == b'a' { b'b' } else { b'a' };
    ann.announcer_signature_hex = String::from_utf8(sig).unwrap();
    let out = process_peer_advertisement_announcement(
        &ann,
        &snip,
        &[j],
        Some("2026-05-28T00:00:20Z"),
    );
    assert!(matches!(out, PeerAdvertisementOutcome::AnnouncerSignatureFailed), "{out:?}");
}

#[test]
fn process_tampered_contributor_signature_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let mut advert = signed_advert(&s, &contrib);
    // Tamper after signing → the contributor signature in the
    // on-SNIP body no longer matches.
    let mut sig = advert.contributor_signature_hex.into_bytes();
    sig[10] = if sig[10] == b'a' { b'b' } else { b'a' };
    advert.contributor_signature_hex = String::from_utf8(sig).unwrap();
    let root = put_advert_on_snip(&snip, &advert);
    let ann = signed_announcement_for(&advert, &root, &announcer);
    let out = process_peer_advertisement_announcement(
        &ann,
        &snip,
        &[j],
        Some("2026-05-28T00:00:20Z"),
    );
    assert!(matches!(out, PeerAdvertisementOutcome::ContributorSignatureFailed), "{out:?}");
}

#[test]
fn process_announcement_body_drift_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let rogue_contrib = ContributorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let advert = signed_advert(&s, &contrib);
    let root = put_advert_on_snip(&snip, &advert);
    let mut ann = signed_announcement_for(&advert, &root, &announcer);
    // Liar: announce someone else's pubkey for this SNIP root.
    ann.contributor_pubkey_hex = rogue_contrib.pubkey_hex();
    let si = net_peer_advert_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&si);
    let out = process_peer_advertisement_announcement(
        &ann,
        &snip,
        &[j],
        Some("2026-05-28T00:00:20Z"),
    );
    assert!(
        matches!(
            out,
            PeerAdvertisementOutcome::DriftMismatch {
                field: "contributor_pubkey_hex"
            }
        ),
        "{out:?}"
    );
}

#[test]
fn process_no_matching_join_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let s = signed_session(&coord);
    // No join supplied for this contributor.
    let advert = signed_advert(&s, &contrib);
    let root = put_advert_on_snip(&snip, &advert);
    let ann = signed_announcement_for(&advert, &root, &announcer);
    let out = process_peer_advertisement_announcement(
        &ann,
        &snip,
        &[],
        Some("2026-05-28T00:00:20Z"),
    );
    assert!(matches!(out, PeerAdvertisementOutcome::NoMatchingJoin), "{out:?}");
}

#[test]
fn process_expired_refused() {
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let advert = signed_advert(&s, &contrib);
    let root = put_advert_on_snip(&snip, &advert);
    let ann = signed_announcement_for(&advert, &root, &announcer);
    let out = process_peer_advertisement_announcement(
        &ann,
        &snip,
        &[j],
        // now > expires_at_utc.
        Some("2026-05-29T00:00:00Z"),
    );
    assert!(matches!(out, PeerAdvertisementOutcome::Expired { .. }), "{out:?}");
}

#[test]
fn process_tampered_advertisement_id_refused() {
    // Critical regression: the contributor signature EXCLUDES
    // `advertisement_id` from the canonical body, so a relayer
    // could otherwise swap the id field and still pass signature
    // verification. The processor must recompute the id and
    // reject on mismatch.
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let mut advert = signed_advert(&s, &contrib);
    // Tamper the advertisement_id after signing — body sig still
    // verifies because the canonical body excludes this field.
    advert.advertisement_id = "ff".repeat(32);
    let root = put_advert_on_snip(&snip, &advert);
    let mut ann = signed_announcement_for(&advert, &root, &announcer);
    // The announcement carries the (tampered) id, so the
    // announcement/body drift check on `advertisement_id` does NOT
    // fire. The new advertisement_id-recompute check must catch
    // this on the body side.
    ann.advertisement_id = advert.advertisement_id.clone();
    let si = net_peer_advert_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&si);
    let out = process_peer_advertisement_announcement(
        &ann,
        &snip,
        &[j],
        Some("2026-05-28T00:00:20Z"),
    );
    assert!(
        matches!(out, PeerAdvertisementOutcome::AdvertisementIdMismatch { .. }),
        "{out:?}"
    );
}

#[test]
fn verify_peer_advertisement_body_helper_happy_path() {
    // The body-only helper is what the CLI's
    // `--resolve-downstream-peer-from-session` path runs over
    // every on-disk advertisement file before cache insert.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let advert = signed_advert(&s, &contrib);
    let out = omni_contributor::verify_peer_advertisement_body(
        &advert,
        &[j],
        Some("2026-05-28T00:00:20Z"),
    );
    assert!(out.is_verified(), "{out:?}");
}

#[test]
fn verify_peer_advertisement_body_helper_catches_tampered_id() {
    // The CLI's local-file path is the second high-severity hole
    // the review surfaced. The body-only verifier must refuse a
    // tampered advertisement_id even when the file deserializes
    // and signature-verifies. This test mirrors the on-disk
    // forging scenario.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let mut advert = signed_advert(&s, &contrib);
    advert.advertisement_id = "ff".repeat(32);
    let out = omni_contributor::verify_peer_advertisement_body(
        &advert,
        &[j],
        Some("2026-05-28T00:00:20Z"),
    );
    assert!(
        matches!(out, PeerAdvertisementOutcome::AdvertisementIdMismatch { .. }),
        "{out:?}"
    );
}

#[test]
fn process_forensic_mode_accepts_expired() {
    // `now_utc = None` skips expiry check (forensic re-runs).
    let snip = MockSnipStore::new();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let advert = signed_advert(&s, &contrib);
    let root = put_advert_on_snip(&snip, &advert);
    let ann = signed_announcement_for(&advert, &root, &announcer);
    let out = process_peer_advertisement_announcement(&ann, &snip, &[j], None);
    assert!(out.is_verified(), "{out:?}");
    // The Stage-12.5 doc anchor: routine watchers should NOT use
    // None — this is a forensic escape hatch.
    let _ = hex_lower(b"forensic-doc-anchor");
}
