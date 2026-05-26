//! Stage 12.1 — byte-stability + JSON round-trip for the two new
//! posted envelopes. Pin the canonical-bytes BLAKE3 hashes so a
//! schema reorder / domain-separator change / bincode bump fails
//! loudly.

use omni_contributor::{
    canonical::{
        canonical_posted_job_bytes, canonical_posted_result_link_bytes, hex_lower,
        posted_id_hex, posted_result_link_signing_input,
    },
    ContributorSigner, DispatcherSigner, PostedJob, PostedResultLink, POSTED_SCHEMA_VERSION,
};

const POSTER_SEED: [u8; 32] = *b"posted-byte-stability-seed-32b!!";
const CONTRIB_SEED: [u8; 32] = *b"contrib-byte-stability-seed-32b!";

fn minimal_posted_job() -> PostedJob {
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
fn posted_job_canonical_bytes_blake3_is_pinned() {
    let p = minimal_posted_job();
    let bytes = canonical_posted_job_bytes(&p).unwrap();
    let h = hex_lower(blake3::hash(&bytes).as_bytes());
    let expected = "ba384c237502d3cac387f1ec606bd683c72d38f1bfe4a8a57534a9711355962c";
    if std::env::var("OMNI_CONTRIBUTOR_REGEN").is_ok() {
        eprintln!("REGEN: posted_job_canonical_bytes BLAKE3 = {h}");
    } else {
        assert_eq!(
            h, expected,
            "Stage 12.1 PostedJob canonical body BLAKE3 drift. \
             If schema changed intentionally, rerun with \
             OMNI_CONTRIBUTOR_REGEN=1 and update the pinned hex."
        );
    }
}

#[test]
fn posted_job_with_poster_signature_canonical_bytes_is_pinned() {
    let signer = DispatcherSigner::from_seed_bytes(&POSTER_SEED).unwrap();
    let mut p = minimal_posted_job();
    p.poster_pubkey_hex = Some(signer.pubkey_hex());
    p.posted_id = posted_id_hex(&p).unwrap();
    let signing_input = canonical_posted_job_bytes(&p).unwrap();
    p.poster_signature_hex = Some(signer.sign_hex(&signing_input));

    let bytes = canonical_posted_job_bytes(&p).unwrap();
    let h = hex_lower(blake3::hash(&bytes).as_bytes());
    let expected = "9e9a05d8da9a9cb89af63ef74ef00ef8c48d333f2b0950086e46876a6efc4d4a";
    if std::env::var("OMNI_CONTRIBUTOR_REGEN").is_ok() {
        eprintln!("REGEN: posted_job_with_signature canonical_bytes BLAKE3 = {h}");
    } else {
        assert_eq!(h, expected, "drift; rerun with OMNI_CONTRIBUTOR_REGEN=1");
    }
}

#[test]
fn posted_job_id_equals_canonical_blake3() {
    let p = minimal_posted_job();
    let bytes = canonical_posted_job_bytes(&p).unwrap();
    let expected_id = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(p.posted_id, expected_id);
}

#[test]
fn posted_job_json_round_trips() {
    let p = minimal_posted_job();
    let s = serde_json::to_string(&p).unwrap();
    let back: PostedJob = serde_json::from_str(&s).unwrap();
    assert_eq!(p, back);
}

fn build_link() -> PostedResultLink {
    let signer = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let mut link = PostedResultLink {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: "44".repeat(32),
        result_snip_root: format!("0x{}", "55".repeat(32)),
        result_canonical_hash: "66".repeat(32),
        contributor_pubkey_hex: signer.pubkey_hex(),
        contributor_signature_hex: String::new(),
        published_at_utc: "2026-05-26T00:00:01Z".into(),
    };
    let signing_input = posted_result_link_signing_input(&link).unwrap();
    link.contributor_signature_hex = signer.sign_hex(&signing_input);
    link
}

#[test]
fn posted_result_link_canonical_bytes_blake3_is_pinned() {
    let link = build_link();
    let bytes = canonical_posted_result_link_bytes(&link).unwrap();
    let h = hex_lower(blake3::hash(&bytes).as_bytes());
    let expected = "2bfa6a0e040dc52457a92df738a0e2543b2bf20dcd23f50f592c4e456cfe4ff6";
    if std::env::var("OMNI_CONTRIBUTOR_REGEN").is_ok() {
        eprintln!("REGEN: posted_result_link canonical_bytes BLAKE3 = {h}");
    } else {
        assert_eq!(h, expected, "drift; rerun with OMNI_CONTRIBUTOR_REGEN=1");
    }
}

#[test]
fn posted_result_link_json_round_trips() {
    let link = build_link();
    let s = serde_json::to_string(&link).unwrap();
    let back: PostedResultLink = serde_json::from_str(&s).unwrap();
    assert_eq!(link, back);
}
