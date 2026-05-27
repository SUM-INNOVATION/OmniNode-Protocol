//! Stage 12.2 — process-result-announcement integration tests
//! exercising the helper that the `watch-network-results` CLI
//! subcommand calls once per polled announcement.

mod common;

use std::collections::HashSet;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{
        hex_lower, network_result_announcement_signing_input, posted_result_link_signing_input,
    },
    posted::POSTED_SCHEMA_VERSION,
    process_result_announcement, ContributorSigner, DispatcherSigner,
    NetworkPostedResultAnnouncement, PostedResultLink, ResultAnnouncementOutcome,
    NET_SCHEMA_VERSION,
};

const CONTRIB_SEED: [u8; 32] = *b"net-results-contrib-seed-32-byt!";
const ANNOUNCER_SEED: [u8; 32] = *b"net-results-announcer-seed-32by!";

/// Build + sign a `PostedResultLink`, publish it to SNIP, build + sign
/// the corresponding `NetworkPostedResultAnnouncement` whose
/// `posted_result_link_snip_root` points at the SNIP-published bytes.
fn build_publish_and_announce(
    snip: &MockSnipStore,
    contrib: &ContributorSigner,
    announcer: &DispatcherSigner,
    posted_id: &str,
) -> NetworkPostedResultAnnouncement {
    let mut link = PostedResultLink {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: posted_id.to_string(),
        result_snip_root: format!("0x{}", "aa".repeat(32)),
        result_canonical_hash: hex_lower(blake3::hash(b"result-canonical-bytes").as_bytes()),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        contributor_signature_hex: String::new(),
        published_at_utc: "2026-05-26T00:00:00Z".into(),
    };
    let signing_input = posted_result_link_signing_input(&link).unwrap();
    link.contributor_signature_hex = contrib.sign_hex(&signing_input);
    let link_json = serde_json::to_vec(&link).unwrap();
    let link_root = snip.insert_bytes(&link_json);

    let mut ann = NetworkPostedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_id: link.posted_id.clone(),
        posted_result_link_snip_root: link_root.to_hex(),
        result_canonical_hash: link.result_canonical_hash.clone(),
        contributor_pubkey_hex: link.contributor_pubkey_hex.clone(),
        announced_at_utc: "2026-05-26T00:00:01Z".into(),
        announcer_pubkey_hex: announcer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_result_announcement_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&signing_input);
    ann
}

#[test]
fn process_writes_link_file_on_happy_path() {
    let snip = MockSnipStore::new();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let posted_id = "11".repeat(32);
    let ann = build_publish_and_announce(&snip, &contrib, &announcer, &posted_id);
    let out = tempfile::tempdir().unwrap();
    let outcome =
        process_result_announcement(&ann, &snip, &HashSet::new(), out.path());
    match outcome {
        ResultAnnouncementOutcome::LinkWritten { posted_id: p, link_path } => {
            assert_eq!(p, posted_id);
            assert!(link_path.is_file());
            // The bytes on disk must equal what was on SNIP.
            let on_disk = std::fs::read(&link_path).unwrap();
            // The MockSnipStore stores bytes by BLAKE3; verify the
            // on-disk bytes' BLAKE3 matches the SNIP root we
            // announced.
            let expected_root_hex =
                hex_lower(blake3::hash(&on_disk).as_bytes());
            assert_eq!(
                ann.posted_result_link_snip_root,
                format!("0x{expected_root_hex}"),
                "fetched + on-disk bytes must round-trip to the announced SNIP root"
            );
        }
        other => panic!("expected LinkWritten, got {other:?}"),
    }
}

#[test]
fn process_skips_when_posted_id_not_in_filter() {
    let snip = MockSnipStore::new();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let posted_id = "22".repeat(32);
    let ann = build_publish_and_announce(&snip, &contrib, &announcer, &posted_id);
    // Filter contains a different posted_id.
    let mut filter = HashSet::new();
    filter.insert("ff".repeat(32));
    let out = tempfile::tempdir().unwrap();
    let outcome = process_result_announcement(&ann, &snip, &filter, out.path());
    assert!(
        matches!(outcome, ResultAnnouncementOutcome::FilteredOut { .. }),
        "{outcome:?}"
    );
    // No file should have been written.
    let path = out.path().join(format!("{posted_id}.link.json"));
    assert!(!path.is_file());
}

#[test]
fn process_rejects_tampered_announcer_signature() {
    let snip = MockSnipStore::new();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let posted_id = "33".repeat(32);
    let mut ann = build_publish_and_announce(&snip, &contrib, &announcer, &posted_id);
    let mut sig = ann.announcer_signature_hex.into_bytes();
    sig[0] = if sig[0] == b'a' { b'b' } else { b'a' };
    ann.announcer_signature_hex = String::from_utf8(sig).unwrap();
    let out = tempfile::tempdir().unwrap();
    let outcome =
        process_result_announcement(&ann, &snip, &HashSet::new(), out.path());
    assert!(
        matches!(outcome, ResultAnnouncementOutcome::AnnouncerSignatureFailed { .. }),
        "{outcome:?}"
    );
}

#[test]
fn process_rejects_drift_between_announcement_and_link() {
    let snip = MockSnipStore::new();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let posted_id = "44".repeat(32);
    let mut ann = build_publish_and_announce(&snip, &contrib, &announcer, &posted_id);
    // Mutate result_canonical_hash AFTER signing → re-sign so the
    // announcer signature still verifies; the drift then surfaces
    // because the on-SNIP link has the original hash.
    ann.result_canonical_hash = "ff".repeat(32);
    let signing_input = network_result_announcement_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&signing_input);
    let out = tempfile::tempdir().unwrap();
    let outcome =
        process_result_announcement(&ann, &snip, &HashSet::new(), out.path());
    assert!(
        matches!(
            outcome,
            ResultAnnouncementOutcome::LinkDrift {
                field: "result_canonical_hash",
                ..
            }
        ),
        "{outcome:?}"
    );
}

#[test]
fn process_rejects_malformed_announcement_schema() {
    let snip = MockSnipStore::new();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let posted_id = "55".repeat(32);
    let mut ann = build_publish_and_announce(&snip, &contrib, &announcer, &posted_id);
    // Bad hex width on contributor_pubkey_hex → schema invalid.
    ann.contributor_pubkey_hex = "short".into();
    let out = tempfile::tempdir().unwrap();
    let outcome =
        process_result_announcement(&ann, &snip, &HashSet::new(), out.path());
    assert!(
        matches!(outcome, ResultAnnouncementOutcome::SchemaMalformed { .. }),
        "{outcome:?}"
    );
}

#[test]
fn process_rejects_tampered_contributor_signature_on_fetched_link() {
    // An honest announcer can still announce a link whose
    // contributor_signature_hex is forged. Without explicit
    // verification of the link's contributor signature, the watcher
    // would accept a forged link as long as the announcer's
    // signature checks out. Stage 12.2 fix: verify the contributor
    // signature on the fetched link before trusting its fields.
    let snip = MockSnipStore::new();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let posted_id = "77".repeat(32);

    // Hand-build a link with a forged contributor signature.
    let mut link = PostedResultLink {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: posted_id.clone(),
        result_snip_root: format!("0x{}", "aa".repeat(32)),
        result_canonical_hash: hex_lower(blake3::hash(b"result-canonical-bytes").as_bytes()),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        contributor_signature_hex: String::new(),
        published_at_utc: "2026-05-26T00:00:00Z".into(),
    };
    let signing_input = posted_result_link_signing_input(&link).unwrap();
    link.contributor_signature_hex = contrib.sign_hex(&signing_input);
    // Tamper with the signature AFTER signing — flip one nibble in
    // the middle so the hex still parses but verification fails.
    let mut sig_bytes = link.contributor_signature_hex.into_bytes();
    let mid = sig_bytes.len() / 2;
    sig_bytes[mid] = if sig_bytes[mid] == b'a' { b'b' } else { b'a' };
    link.contributor_signature_hex = String::from_utf8(sig_bytes).unwrap();
    let link_json = serde_json::to_vec(&link).unwrap();
    let link_root = snip.insert_bytes(&link_json);

    // Build a valid announcer signature over an announcement that
    // points at the tampered link.
    let mut ann = NetworkPostedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_id: link.posted_id.clone(),
        posted_result_link_snip_root: link_root.to_hex(),
        result_canonical_hash: link.result_canonical_hash.clone(),
        contributor_pubkey_hex: link.contributor_pubkey_hex.clone(),
        announced_at_utc: "2026-05-26T00:00:01Z".into(),
        announcer_pubkey_hex: announcer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let ann_signing_input = network_result_announcement_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&ann_signing_input);

    let out = tempfile::tempdir().unwrap();
    let outcome =
        process_result_announcement(&ann, &snip, &HashSet::new(), out.path());
    assert!(
        matches!(
            outcome,
            ResultAnnouncementOutcome::LinkContributorSignatureFailed { .. }
        ),
        "{outcome:?}"
    );
    // No file should have been written.
    let path = out.path().join(format!("{posted_id}.link.json"));
    assert!(!path.is_file());
}

#[test]
fn process_rejects_when_snip_root_does_not_resolve() {
    let snip = MockSnipStore::new();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let posted_id = "66".repeat(32);
    let mut ann = build_publish_and_announce(&snip, &contrib, &announcer, &posted_id);
    // Point at a SNIP root the mock doesn't have.
    ann.posted_result_link_snip_root = format!("0x{}", "ee".repeat(32));
    let signing_input = network_result_announcement_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&signing_input);
    let out = tempfile::tempdir().unwrap();
    let outcome =
        process_result_announcement(&ann, &snip, &HashSet::new(), out.path());
    assert!(
        matches!(outcome, ResultAnnouncementOutcome::SnipFetchFailed { .. }),
        "{outcome:?}"
    );
}
