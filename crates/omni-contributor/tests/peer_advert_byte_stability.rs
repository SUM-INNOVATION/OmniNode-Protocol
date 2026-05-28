//! Stage 12.5 — byte-stability tests for the
//! `ContributorPeerAdvertisement` and
//! `NetworkPeerAdvertisementAnnouncement` canonical signing bodies.
//! Pinned BLAKE3 catches drift on field order / domain separator /
//! sub-struct layout.

use omni_contributor::{
    canonical::{
        advertisement_id_hex, canonical_net_peer_advert_bytes,
        canonical_peer_advertisement_bytes, hex_lower,
        peer_advertisement_signing_input,
    },
    handoff::TensorDtype,
    ContributorPeerAdvertisement, ContributorSigner, CoordinatorSigner,
    NetworkPeerAdvertisementAnnouncement, PeerCapabilities,
    PEER_ADVERTISEMENT_SCHEMA_VERSION,
};

const CONTRIB_SEED: [u8; 32] = *b"stage12.5-byte-contrib-seed-32b!";
const ANNOUNCER_SEED: [u8; 32] = *b"stage12.5-byte-announcer-32byte!";

fn fixture_advert() -> ContributorPeerAdvertisement {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let mut a = ContributorPeerAdvertisement {
        schema_version: PEER_ADVERTISEMENT_SCHEMA_VERSION,
        advertisement_id: String::new(),
        session_id: "11".repeat(32),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        // Fixture: a well-known base58 PeerId. (The actual peer-id
        // string is the v1 wire commitment; flipping it changes the
        // canonical bytes hash.)
        libp2p_peer_id: "12D3KooWGjMyD3v8UYjK3bV7Sqg3WcWzkF7QbTtwYNRsBwCb6KpC".into(),
        listen_multiaddrs: vec!["/ip4/127.0.0.1/udp/4001/quic-v1".into()],
        capabilities: PeerCapabilities {
            supports_live_handoff: true,
            max_handoff_chunk_bytes: 32 * 1024 * 1024,
            supported_dtypes: vec![TensorDtype::F16, TensorDtype::Bf16],
        },
        advertised_at_utc: "2026-05-28T00:00:00Z".into(),
        expires_at_utc: "2026-05-28T01:00:00Z".into(),
        contributor_signature_hex: String::new(),
    };
    a.advertisement_id = advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    a
}

#[test]
fn peer_advertisement_canonical_bytes_blake3_is_pinned() {
    let a = fixture_advert();
    let bytes = canonical_peer_advertisement_bytes(&a).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "2ce9243ed040eb35a447518d7d9f2e6e272365b0a826f9c6f60e2e8be5b0795d",
        "drift in canonical_peer_advertisement_bytes — recompute and re-pin"
    );
}

#[test]
fn net_peer_advert_canonical_bytes_blake3_is_pinned() {
    let advert = fixture_advert();
    let announcer = CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let mut ann = NetworkPeerAdvertisementAnnouncement {
        schema_version: omni_contributor::NET_SCHEMA_VERSION,
        peer_advertisement_snip_root: format!("0x{}", "ab".repeat(32)),
        advertisement_id: advert.advertisement_id.clone(),
        session_id: advert.session_id.clone(),
        contributor_pubkey_hex: advert.contributor_pubkey_hex.clone(),
        announced_at_utc: "2026-05-28T00:00:01Z".into(),
        announcer_pubkey_hex: announcer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si =
        omni_contributor::canonical::net_peer_advert_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&si);
    let bytes = canonical_net_peer_advert_bytes(&ann).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "ba87d3fcd4ed80a7f53d0fa2405807fe3762480c3bf2f34a42ecc5d6c9086565",
        "drift in canonical_net_peer_advert_bytes — recompute and re-pin"
    );
}

#[test]
fn advertisement_id_is_derived_from_canonical_bytes() {
    let a = fixture_advert();
    let derived = advertisement_id_hex(&a).unwrap();
    assert_eq!(a.advertisement_id, derived);
}
