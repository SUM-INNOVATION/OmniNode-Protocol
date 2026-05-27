//! Stage 12.2 — schema-validation + signature negatives for the two
//! new network announcement envelopes.

use omni_contributor::{
    canonical::{
        canonical_network_job_announcement_bytes, network_job_announcement_signing_input,
    },
    error::SchemaError,
    signing::{verify_signature_hex, DispatcherSigner},
    NetworkPostedJobAnnouncement, NetworkPostedResultAnnouncement, NET_SCHEMA_VERSION,
};

const ANNOUNCER_SEED: [u8; 32] = *b"net-neg-announcer-seed-32-bytes!";

fn happy_job() -> NetworkPostedJobAnnouncement {
    let signer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let mut a = NetworkPostedJobAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_job_snip_root: format!("0x{}", "11".repeat(32)),
        posted_id: "22".repeat(32),
        job_hash: "33".repeat(32),
        model_hash: "44".repeat(32),
        tokenizer_hash: Some("55".repeat(32)),
        announced_at_utc: "2026-05-26T00:00:00Z".into(),
        announcer_pubkey_hex: signer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_job_announcement_signing_input(&a).unwrap();
    a.announcer_signature_hex = signer.sign_hex(&signing_input);
    a
}

#[test]
fn happy_job_validates() {
    happy_job().validate_schema().unwrap();
}

#[test]
fn unsupported_schema_version_rejected() {
    let mut a = happy_job();
    a.schema_version = 999;
    let err = a.validate_schema().unwrap_err();
    assert!(matches!(err, SchemaError::UnsupportedVersion { .. }), "{err:?}");
}

#[test]
fn malformed_posted_id_rejected() {
    let mut a = happy_job();
    a.posted_id = "not-64-hex".into();
    let err = a.validate_schema().unwrap_err();
    assert!(matches!(err, SchemaError::MalformedHash { field: "posted_id", .. }), "{err:?}");
}

#[test]
fn malformed_snip_root_rejected() {
    let mut a = happy_job();
    a.posted_job_snip_root = "no-prefix-here".into();
    let err = a.validate_schema().unwrap_err();
    assert!(
        matches!(err, SchemaError::MalformedHash { field: "posted_job_snip_root", .. }),
        "{err:?}"
    );
}

#[test]
fn timestamp_without_z_suffix_rejected() {
    let mut a = happy_job();
    a.announced_at_utc = "2026-05-26T00:00:00+00:00".into();
    let err = a.validate_schema().unwrap_err();
    assert!(matches!(err, SchemaError::MalformedTimestamp { .. }), "{err:?}");
}

#[test]
fn unknown_field_rejected_by_serde() {
    let bad = r#"{
        "schema_version": 1,
        "posted_job_snip_root": "0x00",
        "posted_id": "00",
        "job_hash": "00",
        "model_hash": "00",
        "announced_at_utc": "2026-05-26T00:00:00Z",
        "announcer_pubkey_hex": "00",
        "announcer_signature_hex": "00",
        "future_field": "not allowed"
    }"#;
    let r: Result<NetworkPostedJobAnnouncement, _> = serde_json::from_str(bad);
    assert!(r.is_err(), "deny_unknown_fields must reject");
}

#[test]
fn tampered_signature_does_not_verify() {
    let mut a = happy_job();
    // Flip one nibble.
    let mut bytes = a.announcer_signature_hex.into_bytes();
    bytes[0] = if bytes[0] == b'a' { b'b' } else { b'a' };
    a.announcer_signature_hex = String::from_utf8(bytes).unwrap();
    let signing_input = network_job_announcement_signing_input(&a).unwrap();
    let ok =
        verify_signature_hex(&a.announcer_pubkey_hex, &signing_input, &a.announcer_signature_hex)
            .unwrap();
    assert!(!ok, "tampered signature must NOT verify");
}

#[test]
fn signature_does_not_verify_against_unrelated_pubkey() {
    let a = happy_job();
    // Wrong pubkey (zeroed).
    let other_pubkey = "0".repeat(64);
    let signing_input = network_job_announcement_signing_input(&a).unwrap();
    let ok =
        verify_signature_hex(&other_pubkey, &signing_input, &a.announcer_signature_hex).unwrap();
    assert!(!ok, "signature must not verify against unrelated pubkey");
}

#[test]
fn signature_does_not_verify_when_body_mutated_after_signing() {
    let mut a = happy_job();
    // Mutate the announcement body AFTER signing — signature must
    // no longer verify against the new signing input.
    a.posted_id = "99".repeat(32);
    let signing_input = network_job_announcement_signing_input(&a).unwrap();
    let ok =
        verify_signature_hex(&a.announcer_pubkey_hex, &signing_input, &a.announcer_signature_hex)
            .unwrap();
    assert!(!ok, "signature must not verify on a mutated body");
}

#[test]
fn malformed_pubkey_hex_rejected_by_schema() {
    let mut a = happy_job();
    a.announcer_pubkey_hex = "NOT-HEX".into();
    let err = a.validate_schema().unwrap_err();
    assert!(
        matches!(err, SchemaError::MalformedPubkey { field: "announcer_pubkey_hex", .. }),
        "{err:?}"
    );
}

#[test]
fn malformed_signature_hex_rejected_by_schema() {
    let mut a = happy_job();
    a.announcer_signature_hex = "BAD".into();
    let err = a.validate_schema().unwrap_err();
    assert!(
        matches!(err, SchemaError::MalformedSignature { field: "announcer_signature_hex", .. }),
        "{err:?}"
    );
}

#[test]
fn result_announcement_schema_validation_smoke() {
    // Sanity: build, sign, validate.
    use omni_contributor::canonical::network_result_announcement_signing_input;
    let signer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let mut a = NetworkPostedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_id: "11".repeat(32),
        posted_result_link_snip_root: format!("0x{}", "22".repeat(32)),
        result_canonical_hash: "33".repeat(32),
        contributor_pubkey_hex: "44".repeat(32),
        announced_at_utc: "2026-05-26T00:00:00Z".into(),
        announcer_pubkey_hex: signer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_result_announcement_signing_input(&a).unwrap();
    a.announcer_signature_hex = signer.sign_hex(&signing_input);
    a.validate_schema().unwrap();
    // The schema validator does NOT verify the signature; that's the
    // caller's responsibility — confirm by verifying separately.
    let ok = verify_signature_hex(
        &a.announcer_pubkey_hex,
        &signing_input,
        &a.announcer_signature_hex,
    )
    .unwrap();
    assert!(ok);
}

#[test]
fn canonical_bytes_change_when_pubkey_changes_proves_pubkey_in_signing_input() {
    // Sanity: changing the announcer_pubkey_hex changes the canonical
    // body bytes (the signing input). Otherwise a signature would
    // bind to bytes that don't include the pubkey — an attacker
    // could swap pubkeys without invalidating the signature.
    let mut a = happy_job();
    let bytes_a = canonical_network_job_announcement_bytes(&a).unwrap();
    a.announcer_pubkey_hex = "99".repeat(32);
    let bytes_b = canonical_network_job_announcement_bytes(&a).unwrap();
    assert_ne!(bytes_a, bytes_b);
}
