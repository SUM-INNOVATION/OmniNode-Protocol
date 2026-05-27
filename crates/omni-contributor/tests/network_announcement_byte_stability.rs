//! Stage 12.2 — byte-stability + signature roundtrip + JSON
//! round-trip for the two new network announcement envelopes.

use omni_contributor::{
    canonical::{
        canonical_network_job_announcement_bytes,
        canonical_network_result_announcement_bytes,
        hex_lower,
        network_job_announcement_signing_input,
        network_result_announcement_signing_input,
    },
    signing::{verify_signature_hex, DispatcherSigner},
    NetworkPostedJobAnnouncement, NetworkPostedResultAnnouncement, NET_SCHEMA_VERSION,
};

const ANNOUNCER_SEED: [u8; 32] = *b"net-byte-stab-announcer-seed-32!";

fn minimal_job_announcement() -> NetworkPostedJobAnnouncement {
    let signer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let mut a = NetworkPostedJobAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_job_snip_root: format!("0x{}", "11".repeat(32)),
        posted_id: "22".repeat(32),
        job_hash: "33".repeat(32),
        model_hash: "44".repeat(32),
        tokenizer_hash: None,
        announced_at_utc: "2026-05-26T00:00:00Z".into(),
        announcer_pubkey_hex: signer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_job_announcement_signing_input(&a).unwrap();
    a.announcer_signature_hex = signer.sign_hex(&signing_input);
    a
}

fn minimal_result_announcement() -> NetworkPostedResultAnnouncement {
    let signer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let mut a = NetworkPostedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_id: "55".repeat(32),
        posted_result_link_snip_root: format!("0x{}", "66".repeat(32)),
        result_canonical_hash: "77".repeat(32),
        contributor_pubkey_hex: "88".repeat(32),
        announced_at_utc: "2026-05-26T00:00:01Z".into(),
        announcer_pubkey_hex: signer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_result_announcement_signing_input(&a).unwrap();
    a.announcer_signature_hex = signer.sign_hex(&signing_input);
    a
}

#[test]
fn network_job_announcement_canonical_bytes_blake3_is_pinned() {
    let a = minimal_job_announcement();
    let bytes = canonical_network_job_announcement_bytes(&a).unwrap();
    let h = hex_lower(blake3::hash(&bytes).as_bytes());
    let expected = "3efeb84f35d69a7d5e6535c60b8ca0b12d815d65317bea0a0a37d9df3047961b";
    if std::env::var("OMNI_CONTRIBUTOR_REGEN").is_ok() {
        eprintln!("REGEN: network_job_announcement canonical_bytes BLAKE3 = {h}");
    } else {
        assert_eq!(
            h, expected,
            "Stage 12.2 NetworkPostedJobAnnouncement canonical body BLAKE3 drift. \
             Rerun with OMNI_CONTRIBUTOR_REGEN=1 to print the new hex."
        );
    }
}

#[test]
fn network_result_announcement_canonical_bytes_blake3_is_pinned() {
    let a = minimal_result_announcement();
    let bytes = canonical_network_result_announcement_bytes(&a).unwrap();
    let h = hex_lower(blake3::hash(&bytes).as_bytes());
    let expected = "b564e134097939adc86a274c45bb14b8827406be81da4b5c6ecdd0347ed20720";
    if std::env::var("OMNI_CONTRIBUTOR_REGEN").is_ok() {
        eprintln!("REGEN: network_result_announcement canonical_bytes BLAKE3 = {h}");
    } else {
        assert_eq!(h, expected, "drift; rerun with OMNI_CONTRIBUTOR_REGEN=1");
    }
}

#[test]
fn announcer_signature_roundtrips_for_job_announcement() {
    let a = minimal_job_announcement();
    let signing_input = network_job_announcement_signing_input(&a).unwrap();
    let ok =
        verify_signature_hex(&a.announcer_pubkey_hex, &signing_input, &a.announcer_signature_hex)
            .unwrap();
    assert!(ok, "announcer signature should verify");
}

#[test]
fn announcer_signature_roundtrips_for_result_announcement() {
    let a = minimal_result_announcement();
    let signing_input = network_result_announcement_signing_input(&a).unwrap();
    let ok =
        verify_signature_hex(&a.announcer_pubkey_hex, &signing_input, &a.announcer_signature_hex)
            .unwrap();
    assert!(ok, "announcer signature should verify");
}

#[test]
fn job_announcement_json_round_trips() {
    let a = minimal_job_announcement();
    let s = serde_json::to_string(&a).unwrap();
    let back: NetworkPostedJobAnnouncement = serde_json::from_str(&s).unwrap();
    assert_eq!(a, back);
}

#[test]
fn result_announcement_json_round_trips() {
    let a = minimal_result_announcement();
    let s = serde_json::to_string(&a).unwrap();
    let back: NetworkPostedResultAnnouncement = serde_json::from_str(&s).unwrap();
    assert_eq!(a, back);
}
