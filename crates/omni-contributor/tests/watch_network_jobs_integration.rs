//! Stage 12.2 — end-to-end watch-network-jobs integration using
//! `InMemoryRelay` + `MockSnipStore`. All tests run synchronously
//! with no tokio runtime and no real networking.

mod common;

use std::time::Duration;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{
        hex_lower, job_hash_hex, network_job_announcement_signing_input,
        network_result_announcement_signing_input, posted_id_hex,
    },
    posted::POSTED_SCHEMA_VERSION,
    run_watch_loop,
    signing::verify_signature_hex,
    AcceptFilters, BaseUnitRewardPolicy, ContributorJob, ContributorRelay, ContributorSigner,
    CostCaps, DispatcherSigner, EventEmitter, InMemoryRelay, JobAccounting,
    NetworkPostedJobAnnouncement, NetworkPostedResultAnnouncement, NetworkSource, PostedJob,
    PublishedResultLink, ResultBroadcaster, SkipReason, StubRunner, VerificationRequirement,
    WatchEvent, WatchOptions, NET_SCHEMA_VERSION,
};

const CONTRIBUTOR_SEED: [u8; 32] = *b"net-watch-contrib-seed-32-bytes!";
const ANNOUNCER_SEED: [u8; 32] = *b"net-watch-announcer-seed-32b!!!!";

struct CollectEmitter {
    events: Vec<WatchEvent>,
}
impl CollectEmitter {
    fn new() -> Self {
        Self { events: Vec::new() }
    }
}
impl EventEmitter for CollectEmitter {
    fn emit(&mut self, e: WatchEvent) {
        self.events.push(e);
    }
}

fn build_minimal_job(snip: &MockSnipStore) -> ContributorJob {
    let manifest_bytes = b"net-test-manifest".to_vec();
    let input_bytes = b"net-test-input".to_vec();
    let manifest_root = snip.insert_bytes(&manifest_bytes);
    let input_root = snip.insert_bytes(&input_bytes);
    let input_hash = hex_lower(blake3::hash(&input_bytes).as_bytes());
    let model_hash = hex_lower(blake3::hash(&manifest_bytes).as_bytes());
    let tokenizer_hash = hex_lower(blake3::hash(b"net-test-tok").as_bytes());

    let mut job = ContributorJob {
        schema_version: 1,
        job_id: String::new(),
        model_hash,
        manifest_snip_root: manifest_root.to_hex(),
        input_snip_root: input_root.to_hex(),
        input_hash,
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: JobAccounting {
            tokenizer_hash,
            tokenizer_id: "net/tok".into(),
            input_token_count: 9,
            max_output_token_count: 75,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-05-26T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: None,
    };
    job.job_id = job_hash_hex(&job).unwrap();
    job
}

fn publish_posted_and_announce(
    snip: &MockSnipStore,
    relay: &mut InMemoryRelay,
    job: &ContributorJob,
    announcer: &DispatcherSigner,
) -> NetworkPostedJobAnnouncement {
    // Publish the ContributorJob to SNIP.
    let job_json = serde_json::to_string_pretty(job).unwrap();
    let _job_root_inner_for_inner_storage = snip.insert_bytes(job_json.as_bytes());

    // Build the PostedJob envelope and publish it too.
    let mut posted = PostedJob {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: String::new(),
        job_snip_root: _job_root_inner_for_inner_storage.to_hex(),
        job_hash: job_hash_hex(job).unwrap(),
        model_hash: job.model_hash.clone(),
        posted_at_utc: "2026-05-26T00:00:01Z".into(),
        expires_at_utc: None,
        poster_pubkey_hex: None,
        poster_signature_hex: None,
        notes: None,
    };
    posted.posted_id = posted_id_hex(&posted).unwrap();
    let posted_json = serde_json::to_string_pretty(&posted).unwrap();
    let posted_root = snip.insert_bytes(posted_json.as_bytes());

    // Build + sign the network announcement pointing at the PostedJob.
    let mut ann = NetworkPostedJobAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_job_snip_root: posted_root.to_hex(),
        posted_id: posted.posted_id.clone(),
        job_hash: posted.job_hash.clone(),
        model_hash: posted.model_hash.clone(),
        tokenizer_hash: None,
        announced_at_utc: "2026-05-26T00:00:02Z".into(),
        announcer_pubkey_hex: announcer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_job_announcement_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&signing_input);
    relay.publish_job(&ann).unwrap();
    ann
}

fn liberal_caps() -> CostCaps {
    CostCaps {
        max_input_tokens: 1_000_000,
        max_output_tokens: 1_000_000,
        max_total_base_units: 2_000_000,
    }
}
fn no_filters() -> AcceptFilters {
    AcceptFilters::default()
}

fn run_once_through_network(
    snip: &MockSnipStore,
    relay: &mut InMemoryRelay,
    out_dir: std::path::PathBuf,
    caps: CostCaps,
    filters: AcceptFilters,
) -> Vec<WatchEvent> {
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"net-resp".to_vec(), 9, 11);
    let mut source = NetworkSource::new(relay, snip);
    let mut emitter = CollectEmitter::new();
    let opts = WatchOptions {
        poll_interval: Duration::ZERO,
        max_jobs: None,
        max_polls: Some(1),
        filters,
        caps,
        runner: &runner,
        signer: &signer,
        result_out_dir: out_dir,
        publish_link: false,
        emit: &mut emitter,
        result_broadcaster: None,
    };
    run_watch_loop(snip, &mut source, opts).unwrap();
    emitter.events
}

#[test]
fn watch_network_jobs_picks_up_signed_announcement_and_writes_result() {
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let job = build_minimal_job(&snip);
    let _ann = publish_posted_and_announce(&snip, &mut relay, &job, &announcer);

    let events = run_once_through_network(
        &snip,
        &mut relay,
        tmp_out.path().to_path_buf(),
        liberal_caps(),
        no_filters(),
    );
    let verify_ok = events.iter().any(|e| {
        matches!(e, WatchEvent::VerifyOk { job_id, .. } if job_id == &job.job_id)
    });
    assert!(verify_ok, "expected verify_ok in {events:?}");
    let path = tmp_out.path().join(format!("{}.json", job.job_id));
    assert!(path.is_file(), "accepted result file missing");
}

#[test]
fn watch_network_jobs_skips_bad_announcer_signature_before_running() {
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let job = build_minimal_job(&snip);
    let mut ann = publish_posted_and_announce(&snip, &mut relay, &job, &announcer);
    // Tamper the announcer signature.
    let mut sig = ann.announcer_signature_hex.into_bytes();
    sig[0] = if sig[0] == b'a' { b'b' } else { b'a' };
    ann.announcer_signature_hex = String::from_utf8(sig).unwrap();
    // Re-publish through the relay (overwriting nothing — the queue
    // is append-only, so we instead drain + re-push):
    relay.poll_jobs().unwrap();
    relay.publish_job(&ann).unwrap();

    let events = run_once_through_network(
        &snip,
        &mut relay,
        tmp_out.path().to_path_buf(),
        liberal_caps(),
        no_filters(),
    );
    // The bad-signature announcement is dropped by NetworkSource
    // before reaching the watch loop's pipeline; the watch loop sees
    // a poll with 1 entry, then emits an `error` (per-entry result is
    // Err). No pickup, no result file.
    let path = tmp_out.path().join(format!("{}.json", job.job_id));
    assert!(!path.is_file(), "must not write a result on bad-sig announcement");
    let saw_error = events.iter().any(|e| matches!(e, WatchEvent::Error { .. }));
    assert!(saw_error, "expected an Error event in {events:?}");
}

#[test]
fn watch_network_jobs_skips_drift_between_announcement_and_fetched_posted_job() {
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let job = build_minimal_job(&snip);
    // Build a normal announcement first, then mutate the
    // announcement's `model_hash` AFTER the signing input is
    // computed — but to keep the signature valid, re-sign over the
    // mutated body. The SNIP-fetched PostedJob still has the
    // original model_hash; the drift guard catches the mismatch.
    let mut ann = publish_posted_and_announce(&snip, &mut relay, &job, &announcer);
    // Drain the original announcement.
    relay.poll_jobs().unwrap();
    // Mutate + re-sign.
    ann.model_hash = "ee".repeat(32);
    let signing_input = network_job_announcement_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&signing_input);
    relay.publish_job(&ann).unwrap();

    let events = run_once_through_network(
        &snip,
        &mut relay,
        tmp_out.path().to_path_buf(),
        liberal_caps(),
        no_filters(),
    );
    // The signature verifies (we re-signed the mutated body), but
    // the SNIP-fetched PostedJob's model_hash doesn't match the
    // announcement's claim → AnnouncementDrift surfaces as an Error
    // event from NetworkSource.
    let saw_error = events.iter().any(|e| matches!(e, WatchEvent::Error { .. }));
    assert!(saw_error, "expected drift Error event in {events:?}");
    let path = tmp_out.path().join(format!("{}.json", job.job_id));
    assert!(!path.is_file(), "must not write a result on drift");
}

#[test]
fn watch_network_jobs_dedups_by_posted_id() {
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let job = build_minimal_job(&snip);
    let ann = publish_posted_and_announce(&snip, &mut relay, &job, &announcer);
    // Push a second copy of the same announcement.
    relay.publish_job(&ann).unwrap();
    let events = run_once_through_network(
        &snip,
        &mut relay,
        tmp_out.path().to_path_buf(),
        liberal_caps(),
        no_filters(),
    );
    let saw_already_seen = events.iter().any(|e| {
        matches!(e, WatchEvent::Skip { reason: SkipReason::AlreadySeen, .. })
    });
    assert!(saw_already_seen, "expected AlreadySeen skip in {events:?}");
}

#[test]
fn watch_network_jobs_enforces_cost_caps_before_runner() {
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();

    // Build a job whose declared input_token_count exceeds the cap
    // we'll configure.
    let mut job = build_minimal_job(&snip);
    job.accounting.input_token_count = 1_000_000;
    job.job_id = job_hash_hex(&job).unwrap();
    let _ = publish_posted_and_announce(&snip, &mut relay, &job, &announcer);

    let caps = CostCaps {
        max_input_tokens: 100,
        max_output_tokens: 1_000_000,
        max_total_base_units: 2_000_000,
    };
    let events = run_once_through_network(
        &snip,
        &mut relay,
        tmp_out.path().to_path_buf(),
        caps,
        no_filters(),
    );
    let cap_skip = events.iter().any(|e| {
        matches!(
            e,
            WatchEvent::Skip {
                reason: SkipReason::CostCapExceeded { field: "max_input_tokens" },
                ..
            }
        )
    });
    assert!(cap_skip, "expected cap-exceeded skip in {events:?}");
    let path = tmp_out.path().join(format!("{}.json", job.job_id));
    assert!(!path.is_file(), "must not write a result on cost-cap refusal");
}

#[test]
fn watch_network_jobs_skips_when_model_hash_not_in_accept_set() {
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let job = build_minimal_job(&snip);
    let _ = publish_posted_and_announce(&snip, &mut relay, &job, &announcer);

    let filters = AcceptFilters {
        model_hash_allow: vec!["00".repeat(32)],
        tokenizer_hash_allow: vec![],
    };
    let events = run_once_through_network(
        &snip,
        &mut relay,
        tmp_out.path().to_path_buf(),
        liberal_caps(),
        filters,
    );
    let saw = events.iter().any(|e| {
        matches!(e, WatchEvent::Skip { reason: SkipReason::ModelHashNotInAcceptSet, .. })
    });
    assert!(saw, "expected ModelHashNotInAcceptSet skip in {events:?}");
}

#[test]
fn watch_network_jobs_skips_malformed_posted_job_on_snip() {
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let job = build_minimal_job(&snip);

    // Insert a garbage byte sequence under a SNIP root, then build
    // an announcement pointing at that root (we won't be able to
    // honestly compute posted_id since the bytes aren't a valid
    // PostedJob, so we craft a *signed* announcement with random
    // posted_id / job_hash / model_hash — the schema-level
    // validation passes; the SNIP-fetch + parse step in
    // NetworkSource is where it dies).
    let garbage = b"this is not a valid PostedJob JSON";
    let bad_root = snip.insert_bytes(garbage);

    let mut ann = NetworkPostedJobAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_job_snip_root: bad_root.to_hex(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: None,
        announced_at_utc: "2026-05-26T00:00:00Z".into(),
        announcer_pubkey_hex: announcer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_job_announcement_signing_input(&ann).unwrap();
    ann.announcer_signature_hex = announcer.sign_hex(&signing_input);
    relay.publish_job(&ann).unwrap();

    let events = run_once_through_network(
        &snip,
        &mut relay,
        tmp_out.path().to_path_buf(),
        liberal_caps(),
        no_filters(),
    );
    let saw_error = events.iter().any(|e| matches!(e, WatchEvent::Error { .. }));
    assert!(saw_error, "expected parse Error event in {events:?}");
    let path = tmp_out.path().join(format!("{}.json", job.job_id));
    assert!(!path.is_file(), "must not write a result on parse failure");
}

// ── ResultBroadcaster wiring (Stage 12.2 Fix #3 regression) ────────────────
//
// The CLI's `watch-network-jobs --publish-result-link` wires a custom
// `ResultBroadcaster` into `WatchOptions` so that every verified
// result link is also announced on the contributor-result topic.
// This test stands in for that wiring with a recording broadcaster:
// after a happy-path job → result → link publish, the broadcaster
// must (a) be invoked exactly once with the published link, and (b)
// produce a `NetworkPostedResultAnnouncement` that schema-validates
// and whose announcer signature verifies against the canonical
// signing input.

struct RecordingBroadcaster {
    signer: ContributorSigner,
    captured: Vec<NetworkPostedResultAnnouncement>,
}

impl ResultBroadcaster for RecordingBroadcaster {
    fn broadcast(&mut self, published: &PublishedResultLink) -> Result<(), String> {
        let announced_at_utc = "2026-05-26T00:00:42Z".to_string();
        let mut ann = NetworkPostedResultAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            posted_id: published.link.posted_id.clone(),
            posted_result_link_snip_root: format!(
                "0x{}",
                hex_lower(published.link_snip_root.as_bytes())
            ),
            result_canonical_hash: published.link.result_canonical_hash.clone(),
            contributor_pubkey_hex: published.link.contributor_pubkey_hex.clone(),
            announced_at_utc,
            announcer_pubkey_hex: self.signer.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let signing_input = network_result_announcement_signing_input(&ann)
            .map_err(|e| format!("canonical: {e}"))?;
        ann.announcer_signature_hex = self.signer.sign_hex(&signing_input);
        ann.validate_schema().map_err(|e| format!("schema: {e}"))?;
        self.captured.push(ann);
        Ok(())
    }
}

#[test]
fn watch_loop_invokes_result_broadcaster_after_verified_result_link() {
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let job = build_minimal_job(&snip);
    let _ann = publish_posted_and_announce(&snip, &mut relay, &job, &announcer);

    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"net-resp".to_vec(), 9, 11);
    let mut source = NetworkSource::new(&mut relay, &snip);
    let mut emitter = CollectEmitter::new();
    let mut broadcaster = RecordingBroadcaster {
        signer: ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap(),
        captured: Vec::new(),
    };
    let opts = WatchOptions {
        poll_interval: Duration::ZERO,
        max_jobs: None,
        max_polls: Some(1),
        filters: no_filters(),
        caps: liberal_caps(),
        runner: &runner,
        signer: &signer,
        result_out_dir: tmp_out.path().to_path_buf(),
        publish_link: true,
        emit: &mut emitter,
        result_broadcaster: Some(&mut broadcaster),
    };
    run_watch_loop(&snip, &mut source, opts).unwrap();

    // (a) Exactly one capture.
    assert_eq!(
        broadcaster.captured.len(),
        1,
        "broadcaster should be invoked once per published link; got {:?}\nemitted events: {:?}",
        broadcaster.captured,
        emitter.events,
    );
    let captured = &broadcaster.captured[0];

    // (b) Captured announcement's posted_id matches the watch loop's
    // input announcement, and its announcer signature verifies.
    assert_eq!(captured.posted_id, _ann.posted_id);
    let signing_input = network_result_announcement_signing_input(captured).unwrap();
    let sig_ok = verify_signature_hex(
        &captured.announcer_pubkey_hex,
        &signing_input,
        &captured.announcer_signature_hex,
    )
    .unwrap();
    assert!(sig_ok, "broadcaster-emitted announcer signature must verify");

    // Sanity: the result-link-published event was emitted before the
    // broadcaster ran (the watch loop publishes to SNIP first, then
    // hands off to the broadcaster).
    let saw_link_published = emitter
        .events
        .iter()
        .any(|e| matches!(e, WatchEvent::ResultLinkPublished { .. }));
    assert!(
        saw_link_published,
        "expected ResultLinkPublished event before broadcaster runs"
    );
}

#[test]
fn watch_loop_does_not_invoke_broadcaster_when_publish_link_disabled() {
    // When --publish-result-link is false, the watch loop must NOT
    // call the broadcaster: there's no link to broadcast. (Regression
    // guard: if someone "simplifies" the conditional and always
    // calls the broadcaster, this catches it.)
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut relay = InMemoryRelay::new();
    let announcer = DispatcherSigner::from_seed_bytes(&ANNOUNCER_SEED).unwrap();
    let job = build_minimal_job(&snip);
    let _ann = publish_posted_and_announce(&snip, &mut relay, &job, &announcer);

    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"net-resp".to_vec(), 9, 11);
    let mut source = NetworkSource::new(&mut relay, &snip);
    let mut emitter = CollectEmitter::new();
    let mut broadcaster = RecordingBroadcaster {
        signer: ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap(),
        captured: Vec::new(),
    };
    let opts = WatchOptions {
        poll_interval: Duration::ZERO,
        max_jobs: None,
        max_polls: Some(1),
        filters: no_filters(),
        caps: liberal_caps(),
        runner: &runner,
        signer: &signer,
        result_out_dir: tmp_out.path().to_path_buf(),
        publish_link: false,
        emit: &mut emitter,
        result_broadcaster: Some(&mut broadcaster),
    };
    run_watch_loop(&snip, &mut source, opts).unwrap();
    assert!(
        broadcaster.captured.is_empty(),
        "broadcaster must not be invoked when publish_link is false"
    );
}
