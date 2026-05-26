//! Stage 12.1 — schema-validation negatives for the posted
//! envelopes. Each test mutates exactly one field on a happy-path
//! envelope and asserts the validator (or the FilesystemSource's
//! posted_id-recompute step) refuses.

use omni_contributor::{
    canonical::{canonical_posted_job_bytes, posted_id_hex},
    error::{DiscoverError, SchemaError},
    posted::POSTED_SCHEMA_VERSION,
    DispatcherSigner, FilesystemSource, JobSource, PostedJob, PostedResultLink,
    ContributorSigner,
};

const POSTER_SEED: [u8; 32] = *b"posted-neg-seed-32-bytes-test!!!";
const CONTRIB_SEED: [u8; 32] = *b"contrib-neg-seed-32-bytes-test!!";

fn happy_posted_job() -> PostedJob {
    let mut p = PostedJob {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: String::new(),
        job_snip_root: format!("0x{}", "11".repeat(32)),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        posted_at_utc: "2026-05-26T00:00:00Z".into(),
        expires_at_utc: None,
        poster_pubkey_hex: None,
        poster_signature_hex: None,
        notes: None,
    };
    p.posted_id = posted_id_hex(&p).unwrap();
    p
}

#[test]
fn posted_job_unsupported_schema_version_rejected() {
    let mut p = happy_posted_job();
    p.schema_version = 999;
    let err = p.validate_schema().unwrap_err();
    assert!(matches!(err, SchemaError::UnsupportedVersion { .. }), "{err:?}");
}

#[test]
fn posted_job_malformed_posted_id_rejected() {
    let mut p = happy_posted_job();
    p.posted_id = "not-64-hex".into();
    let err = p.validate_schema().unwrap_err();
    assert!(matches!(err, SchemaError::MalformedHash { field: "posted_id", .. }), "{err:?}");
}

#[test]
fn posted_job_malformed_snip_root_rejected() {
    let mut p = happy_posted_job();
    p.job_snip_root = "no-0x-prefix".into();
    let err = p.validate_schema().unwrap_err();
    assert!(matches!(err, SchemaError::MalformedHash { field: "job_snip_root", .. }), "{err:?}");
}

#[test]
fn posted_job_timestamp_without_z_suffix_rejected() {
    let mut p = happy_posted_job();
    p.posted_at_utc = "2026-05-26T00:00:00+00:00".into();
    let err = p.validate_schema().unwrap_err();
    assert!(matches!(err, SchemaError::MalformedTimestamp { field: "posted_at_utc", .. }), "{err:?}");
}

#[test]
fn posted_job_half_set_poster_identity_rejected() {
    let signer = DispatcherSigner::from_seed_bytes(&POSTER_SEED).unwrap();
    let mut p = happy_posted_job();
    p.poster_pubkey_hex = Some(signer.pubkey_hex());
    // poster_signature_hex stays None → inconsistent.
    let err = p.validate_schema().unwrap_err();
    assert!(
        matches!(err, SchemaError::InconsistentDispatcherIdentity { .. }),
        "{err:?}"
    );

    let mut p2 = happy_posted_job();
    p2.poster_signature_hex = Some("f".repeat(128));
    let err2 = p2.validate_schema().unwrap_err();
    assert!(
        matches!(err2, SchemaError::InconsistentDispatcherIdentity { .. }),
        "{err2:?}"
    );
}

#[test]
fn filesystem_source_rejects_posted_id_drift() {
    // Write a posted-job file whose `posted_id` doesn't match its
    // canonical-bytes BLAKE3.
    let tmp = tempfile::tempdir().unwrap();
    let mut p = happy_posted_job();
    p.posted_id = "0".repeat(64); // wrong on purpose
    let path = tmp.path().join("bad.json");
    std::fs::write(&path, serde_json::to_string_pretty(&p).unwrap()).unwrap();

    let mut source = FilesystemSource::new(tmp.path().to_path_buf());
    let entries = source.poll().unwrap();
    assert_eq!(entries.len(), 1);
    let result = entries.into_iter().next().unwrap().result;
    let err = result.unwrap_err();
    assert!(matches!(err, DiscoverError::PostedIdMismatch { .. }), "{err:?}");
}

#[test]
fn filesystem_source_rejects_malformed_json_with_typed_error() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("bad.json"), "{not valid").unwrap();
    let mut source = FilesystemSource::new(tmp.path().to_path_buf());
    let entries = source.poll().unwrap();
    assert_eq!(entries.len(), 1);
    let err = entries.into_iter().next().unwrap().result.unwrap_err();
    assert!(matches!(err, DiscoverError::Parse { .. }), "{err:?}");
}

#[test]
fn filesystem_source_skips_non_json_files() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("README.md"), "ignore me").unwrap();
    std::fs::write(tmp.path().join("notes.txt"), "ignore me too").unwrap();
    let mut source = FilesystemSource::new(tmp.path().to_path_buf());
    let entries = source.poll().unwrap();
    assert_eq!(entries.len(), 0);
}

#[test]
fn posted_job_unknown_field_rejected_by_serde() {
    let bad = r#"{
        "schema_version": 1,
        "posted_id": "00",
        "job_snip_root": "0x00",
        "job_hash": "00",
        "model_hash": "00",
        "posted_at_utc": "2026-05-26T00:00:00Z",
        "rogue_field": "not allowed"
    }"#;
    let r: Result<PostedJob, _> = serde_json::from_str(bad);
    assert!(r.is_err(), "deny_unknown_fields must reject");
}

#[test]
fn posted_job_poster_signature_round_trip() {
    let signer = DispatcherSigner::from_seed_bytes(&POSTER_SEED).unwrap();
    let mut p = happy_posted_job();
    p.poster_pubkey_hex = Some(signer.pubkey_hex());
    p.posted_id = posted_id_hex(&p).unwrap();
    let signing_input = canonical_posted_job_bytes(&p).unwrap();
    p.poster_signature_hex = Some(signer.sign_hex(&signing_input));
    p.validate_schema().unwrap();
    let s = serde_json::to_string(&p).unwrap();
    let back: PostedJob = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}

#[test]
fn posted_result_link_malformed_pubkey_rejected() {
    let signer = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let mut link = PostedResultLink {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: "44".repeat(32),
        result_snip_root: format!("0x{}", "55".repeat(32)),
        result_canonical_hash: "66".repeat(32),
        contributor_pubkey_hex: "BAD-NOT-HEX".into(), // wrong
        contributor_signature_hex: "f".repeat(128),
        published_at_utc: "2026-05-26T00:00:00Z".into(),
    };
    let err = link.validate_schema().unwrap_err();
    assert!(matches!(err, SchemaError::MalformedPubkey { .. }), "{err:?}");
    // Repair and assert acceptance.
    link.contributor_pubkey_hex = signer.pubkey_hex();
    link.validate_schema().unwrap();
}
