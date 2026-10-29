//! Issue #149 — every `snip::publish_bytes` caller (posted jobs, result
//! JSON, result links, `announce-result`'s re-publication of a link, the
//! session artifacts) publishes idempotently: the canonical, unsalted root;
//! the same bytes twice succeed; and a foreign Pending, an Abandoned or an
//! incompatible record is refused clearly, with nothing registered.

mod common;

use common::{MOCK_ACCOUNT, MockSnipStore};
use omni_contributor::ContributorSigner;
use omni_contributor::posted::PostedJob;
use omni_contributor::snip::{publish_bytes, publish_bytes_idempotent, publish_bytes_with_hash};
use omni_contributor::watch::{EventEmitter, WatchEvent, publish_result_link_for};
use omni_store::snip_v2_publish::{PublicationAction, SnipV2FileInfo, public_object_root};

#[test]
fn the_same_bytes_twice_register_once_under_their_canonical_root() {
    let snip = MockSnipStore::new();
    let bytes = br#"{"link":"bytes"}"#;
    let a = publish_bytes(&snip, bytes, "a").unwrap();
    let b = publish_bytes(&snip, bytes, "b").unwrap();
    assert_eq!(a, b);
    // Canonical and unsalted: for one chunk, the BLAKE3 of the bytes.
    assert_eq!(a, public_object_root(bytes));
    assert_eq!(*a.as_bytes(), *blake3::hash(bytes).as_bytes());
    assert_eq!(snip.calls().ingests, 1);

    let (root, hash) = publish_bytes_with_hash(&snip, bytes, "c").unwrap();
    assert_eq!((root, hash), (a, *blake3::hash(bytes).as_bytes()));
    let p = publish_bytes_idempotent(&snip, bytes, "d").unwrap();
    assert_eq!(p.action, PublicationAction::ReusedActive);
    assert_eq!(snip.calls().ingests, 1);
}

#[test]
fn announce_result_republishing_a_published_link_reuses_it() {
    // `publish-result-link` publishes the link bytes; `announce-result`
    // publishes the same file's bytes again to learn their root.
    let snip = MockSnipStore::new();
    let link = br#"{"schema_version":1,"posted_id":"p"}"#;
    let published = publish_bytes(&snip, link, "result-link").unwrap();
    let announced = publish_bytes(&snip, link, "announce-result-link").unwrap();
    assert_eq!(announced, published);
    assert_eq!(snip.calls().ingests, 1);
}

#[test]
fn foreign_pending_abandoned_and_incompatible_records_are_refused() {
    type Case = (&'static str, u8, &'static str, fn(&mut SnipV2FileInfo));
    let cases: [Case; 3] = [
        (
            "SomeoneElse111111111111111111111",
            0,
            "SomeoneElse111111111111111111111",
            |_| {},
        ),
        (MOCK_ACCOUNT, 2, "Abandoned", |_| {}),
        (MOCK_ACCOUNT, 1, "stored_size_bytes", |r| {
            r.stored_size_bytes += 1
        }),
    ];
    for (owner, lifecycle, expect, edit) in cases {
        let snip = MockSnipStore::new();
        let bytes = b"artifact bytes";
        snip.put_record(bytes, owner, lifecycle, edit);
        let err = publish_bytes(&snip, bytes, "x").unwrap_err().to_string();
        assert!(err.contains(expect), "{expect}: {err}");
        assert_eq!(snip.calls().ingests, 0, "{expect}");
        assert_eq!(snip.calls().resumes, 0, "{expect}");
    }
}

#[test]
fn publishing_a_result_link_twice_does_not_fail_on_the_result_json() {
    // `publish-result-link` run again for the same result: the result JSON
    // is identical (reused); the link carries a new timestamp (new root).
    let snip = MockSnipStore::new();
    let signer = ContributorSigner::from_seed_bytes(&[3u8; 32]).unwrap();
    let posted = PostedJob {
        schema_version: 1,
        posted_id: "a".repeat(64),
        job_snip_root: format!("0x{}", "11".repeat(32)),
        job_hash: "22".repeat(32),
        model_hash: "44".repeat(32),
        poster_pubkey_hex: None,
        poster_signature_hex: None,
        posted_at_utc: "2026-10-28T00:00:00Z".into(),
        expires_at_utc: None,
        notes: None,
    };
    let result =
        serde_json::from_value::<omni_contributor::ContributorResult>(sample_result(&signer))
            .unwrap();
    let json = serde_json::to_string_pretty(&result).unwrap();
    struct Quiet;
    impl EventEmitter for Quiet {
        fn emit(&mut self, _: WatchEvent) {}
    }
    let mut emit = Quiet;
    let first =
        publish_result_link_for(&snip, &posted, &json, &result, &signer, &mut emit).unwrap();
    let second =
        publish_result_link_for(&snip, &posted, &json, &result, &signer, &mut emit).unwrap();
    assert_eq!(first.result_snip_root, second.result_snip_root);
    assert_eq!(snip.record(&first.result_snip_root).unwrap().lifecycle, 1);
}

fn sample_result(signer: &ContributorSigner) -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "job_id": "33".repeat(32),
        "job_hash": "33".repeat(32),
        "model_hash": "44".repeat(32),
        "input_hash": "55".repeat(32),
        "response_snip_root": format!("0x{}", "66".repeat(32)),
        "response_hash": "77".repeat(32),
        "evidence": "attestation_only",
        "measured_accounting": {
            "tokenizer_hash": "88".repeat(32),
            "input_token_count": 1,
            "output_token_count": 1,
            "total_base_units": 2,
            "stage_contributions": [{
                "contributor_pubkey_hex": signer.pubkey_hex(),
                "stage_label": "s",
                "work_unit_kind": "tokens",
                "work_units": 2
            }]
        },
        "produced_at_utc": "2026-10-28T00:00:01Z",
        "contributor_pubkey_hex": signer.pubkey_hex(),
        "contributor_signature_hex": "99".repeat(64)
    })
}
