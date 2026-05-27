//! Stage 12.3 — byte-stability tests for the 5 network session
//! announcement envelopes. Pinned BLAKE3 hashes catch any drift in
//! field order, domain separators, or canonical body shape.

use omni_contributor::{
    canonical::{
        canonical_net_aggregated_bytes, canonical_net_assign_bytes,
        canonical_net_join_bytes, canonical_net_partial_bytes,
        canonical_net_session_opened_bytes, hex_lower,
    },
    ContributorSigner, CoordinatorSigner, NetworkAggregatedResultAnnouncement,
    NetworkContributorJoinedAnnouncement, NetworkPartialResultAnnouncement,
    NetworkSessionOpenedAnnouncement, NetworkWorkAssignedAnnouncement, NET_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.3-coord-seed-32-byte-key";
const CONTRIB_SEED: [u8; 32] = *b"stage12.3-contrib-seed-32-bytes!";
const ANNOUNCER_SEED: [u8; 32] = *b"stage12.3-announcer-seed-32byte!";

fn announcer() -> CoordinatorSigner {
    CoordinatorSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap()
}

fn session_opened() -> NetworkSessionOpenedAnnouncement {
    let ann = announcer();
    let mut a = NetworkSessionOpenedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        execution_session_snip_root: format!("0x{}", "11".repeat(32)),
        session_id: "22".repeat(32),
        posted_id: "33".repeat(32),
        announced_at_utc: "2026-05-27T00:00:00Z".into(),
        announcer_pubkey_hex: ann.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si =
        omni_contributor::canonical::net_session_opened_signing_input(&a).unwrap();
    a.announcer_signature_hex = ann.sign_hex(&si);
    a
}

fn contributor_joined() -> NetworkContributorJoinedAnnouncement {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let ann = announcer();
    let mut a = NetworkContributorJoinedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        contributor_join_snip_root: format!("0x{}", "44".repeat(32)),
        session_id: "22".repeat(32),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:01Z".into(),
        announcer_pubkey_hex: ann.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = omni_contributor::canonical::net_join_signing_input(&a).unwrap();
    a.announcer_signature_hex = ann.sign_hex(&si);
    a
}

fn work_assigned() -> NetworkWorkAssignedAnnouncement {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let ann = announcer();
    let mut a = NetworkWorkAssignedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        work_assignment_snip_root: format!("0x{}", "55".repeat(32)),
        session_id: "22".repeat(32),
        assignment_id: "66".repeat(32),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:02Z".into(),
        announcer_pubkey_hex: ann.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = omni_contributor::canonical::net_assign_signing_input(&a).unwrap();
    a.announcer_signature_hex = ann.sign_hex(&si);
    a
}

fn partial_result() -> NetworkPartialResultAnnouncement {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let ann = announcer();
    let mut a = NetworkPartialResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        partial_result_snip_root: format!("0x{}", "77".repeat(32)),
        session_id: "22".repeat(32),
        assignment_id: "66".repeat(32),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: "2026-05-27T00:00:10Z".into(),
        announcer_pubkey_hex: ann.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si = omni_contributor::canonical::net_partial_signing_input(&a).unwrap();
    a.announcer_signature_hex = ann.sign_hex(&si);
    a
}

fn aggregated_result() -> NetworkAggregatedResultAnnouncement {
    let ann = announcer();
    let mut a = NetworkAggregatedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        aggregated_result_snip_root: format!("0x{}", "88".repeat(32)),
        session_id: "22".repeat(32),
        posted_id: "33".repeat(32),
        announced_at_utc: "2026-05-27T00:00:30Z".into(),
        announcer_pubkey_hex: ann.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let si =
        omni_contributor::canonical::net_aggregated_signing_input(&a).unwrap();
    a.announcer_signature_hex = ann.sign_hex(&si);
    a
}

#[test]
fn net_session_opened_canonical_bytes_blake3_is_pinned() {
    // Unused: COORD_SEED kept as a doc anchor; tests use the
    // announcer key directly to avoid touching the inner session.
    let _ = COORD_SEED;
    let a = session_opened();
    let bytes = canonical_net_session_opened_bytes(&a).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "1d8767a1074acf44b80e4ad9a19e79e0efd22eb025cf1985f853cc4b0f5b7843",
        "drift in canonical_net_session_opened_bytes — recompute and re-pin"
    );
}

#[test]
fn net_contributor_joined_canonical_bytes_blake3_is_pinned() {
    let a = contributor_joined();
    let bytes = canonical_net_join_bytes(&a).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "60f5a2052261aa352dca884218b38621c2b92594be4766d7458fb9b9dd0f9bb4",
        "drift in canonical_net_join_bytes — recompute and re-pin"
    );
}

#[test]
fn net_work_assigned_canonical_bytes_blake3_is_pinned() {
    let a = work_assigned();
    let bytes = canonical_net_assign_bytes(&a).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "bfc626607144cac915ec04ed286194f57acad236fa3a39eae1894c82f5ce9a4f",
        "drift in canonical_net_assign_bytes — recompute and re-pin"
    );
}

#[test]
fn net_partial_result_canonical_bytes_blake3_is_pinned() {
    let a = partial_result();
    let bytes = canonical_net_partial_bytes(&a).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "75df3c5a47762e9467477f80f9c228ebffa897ce1e758331f38603bf311fe000",
        "drift in canonical_net_partial_bytes — recompute and re-pin"
    );
}

#[test]
fn net_aggregated_result_canonical_bytes_blake3_is_pinned() {
    let a = aggregated_result();
    let bytes = canonical_net_aggregated_bytes(&a).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "fe9aa50570b34e2be3c792067f243fe8d54411fcc44c270b054100db6a61743f",
        "drift in canonical_net_aggregated_bytes — recompute and re-pin"
    );
}
