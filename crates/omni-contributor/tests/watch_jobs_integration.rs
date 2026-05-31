//! Stage 12.1 — end-to-end watch-jobs integration.
//!
//! All tests use the in-memory `MockSnipStore` from `common/mod.rs`
//! plus a tempdir-based filesystem source. Each test builds a job,
//! publishes it to the mock SNIP, writes a `PostedJob` envelope into
//! the watched dir, and drives `run_watch_loop` with `max_jobs:
//! Some(1)` so the loop exits 0 after a single pickup.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{
        canonical_posted_job_bytes, hex_lower, job_hash_hex, posted_id_hex,
    },
    posted::POSTED_SCHEMA_VERSION,
    run_watch_loop, AcceptFilters, BaseUnitRewardPolicy, ContributorJob, ContributorSigner,
    CostCaps, DispatcherSigner, EventEmitter, FilesystemSource, JobAccounting, PostedJob,
    SkipReason, StubRunner, VerificationRequirement, WatchEvent, WatchOptions,
};

const CONTRIBUTOR_SEED: [u8; 32] = *b"watch-contributor-seed-bytes-32!";
const POSTER_SEED: [u8; 32] = *b"watch-poster-seed-bytes-32-byte!";

/// Test-side event collector. Stage 12.1's `EventEmitter` trait lets
/// tests inspect the full event sequence.
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
    let manifest_bytes = b"watch-test-manifest".to_vec();
    let input_bytes = b"watch-test-input".to_vec();
    let manifest_root = snip.insert_bytes(&manifest_bytes);
    let input_root = snip.insert_bytes(&input_bytes);
    let input_hash = hex_lower(blake3::hash(&input_bytes).as_bytes());
    let model_hash = hex_lower(blake3::hash(&manifest_bytes).as_bytes());
    let tokenizer_hash = hex_lower(blake3::hash(b"watch-test-tok").as_bytes());

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
            tokenizer_id: "test/tok".into(),
            input_token_count: 7,
            max_output_token_count: 50,
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

fn publish_job_and_post(
    snip: &MockSnipStore,
    job: &ContributorJob,
    jobs_dir: &std::path::Path,
    poster: Option<&DispatcherSigner>,
) -> PostedJob {
    let job_json = serde_json::to_string_pretty(job).unwrap();
    let job_root = snip.insert_bytes(job_json.as_bytes());
    let mut posted = PostedJob {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: String::new(),
        job_snip_root: job_root.to_hex(),
        job_hash: job_hash_hex(job).unwrap(),
        model_hash: job.model_hash.clone(),
        posted_at_utc: "2026-05-26T00:00:01Z".into(),
        expires_at_utc: None,
        poster_pubkey_hex: poster.map(|p| p.pubkey_hex()),
        poster_signature_hex: None,
        notes: None,
    };
    posted.posted_id = posted_id_hex(&posted).unwrap();
    if let Some(p) = poster {
        let signing_input = canonical_posted_job_bytes(&posted).unwrap();
        posted.poster_signature_hex = Some(p.sign_hex(&signing_input));
    }
    let path = jobs_dir.join(format!("{}.json", posted.posted_id));
    std::fs::write(&path, serde_json::to_string_pretty(&posted).unwrap()).unwrap();
    posted
}

fn run_once(
    snip: &MockSnipStore,
    jobs_dir: PathBuf,
    out_dir: PathBuf,
    publish_link: bool,
    caps: CostCaps,
    filters: AcceptFilters,
) -> Vec<WatchEvent> {
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"watch-resp".to_vec(), 7, 13);
    let mut source = FilesystemSource::new(jobs_dir);
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
        publish_link,
        emit: &mut emitter,
        result_broadcaster: None,
        state_store: None,
    };
    run_watch_loop(snip, &mut source, opts).unwrap();
    emitter.events
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

#[test]
fn watch_loop_picks_up_unsigned_posted_job_and_writes_accepted_result() {
    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let job = build_minimal_job(&snip);
    let _ = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);

    let events = run_once(
        &snip,
        tmp_jobs.path().to_path_buf(),
        tmp_out.path().to_path_buf(),
        false,
        liberal_caps(),
        no_filters(),
    );
    // Expect a verify_ok event for our job.
    let verify_ok = events.iter().any(|e| {
        matches!(e, WatchEvent::VerifyOk { job_id, .. } if job_id == &job.job_id)
    });
    assert!(verify_ok, "expected verify_ok event in {events:?}");

    // Expect the accepted JSON to exist; no `.rejected.json`.
    let accepted = tmp_out.path().join(format!("{}.json", job.job_id));
    assert!(accepted.is_file(), "accepted result missing");
    let rejected = tmp_out.path().join(format!("{}.rejected.json", job.job_id));
    assert!(!rejected.is_file(), "rejected file should not exist");
}

#[test]
fn watch_loop_publishes_result_link_when_flag_set() {
    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let job = build_minimal_job(&snip);
    let _ = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);
    let baseline = snip.object_count();
    let events = run_once(
        &snip,
        tmp_jobs.path().to_path_buf(),
        tmp_out.path().to_path_buf(),
        true,
        liberal_caps(),
        no_filters(),
    );
    // Expect a result_link_published event.
    let link_event = events
        .iter()
        .find(|e| matches!(e, WatchEvent::ResultLinkPublished { .. }));
    assert!(link_event.is_some(), "expected ResultLinkPublished in {events:?}");
    // The SNIP store should have grown by at least 2 objects (response
    // bytes via run_job + link envelope; the result JSON itself is
    // published once by publish_result_link_for).
    assert!(
        snip.object_count() > baseline,
        "SNIP store should have grown after publish_link"
    );
}

#[test]
fn watch_loop_skips_poster_signature_failure() {
    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let job = build_minimal_job(&snip);
    let poster = DispatcherSigner::from_seed_bytes(&POSTER_SEED).unwrap();
    let mut posted = publish_job_and_post(&snip, &job, tmp_jobs.path(), Some(&poster));
    // Tamper with the signature.
    let mut sig = posted.poster_signature_hex.take().unwrap().into_bytes();
    sig[0] = if sig[0] == b'a' { b'b' } else { b'a' };
    posted.poster_signature_hex = Some(String::from_utf8(sig).unwrap());
    let path = tmp_jobs.path().join(format!("{}.json", posted.posted_id));
    std::fs::write(&path, serde_json::to_string_pretty(&posted).unwrap()).unwrap();

    // The tampered file's `posted_id` no longer matches its
    // canonical bytes (the body is unchanged, but FilesystemSource
    // recomputes posted_id at load time and posted_id is derived
    // from the body which EXCLUDES poster_signature_hex). So the
    // tampered signature is independent of posted_id and the file
    // still passes posted_id-mismatch; then poster signature
    // verification catches the tamper.
    let mut emitter = CollectEmitter::new();
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"r".to_vec(), 7, 5);
    let mut source = FilesystemSource::new(tmp_jobs.path().to_path_buf());
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
        result_broadcaster: None,
        state_store: None,
    };
    // max_jobs(1) won't be reached because the bad-signature entry
    // is skipped, not picked. Force the loop to exit by also using
    // max_jobs(0). But max_jobs(0) means "exit immediately after
    // the first decision"; safer to drive it differently: poll
    // once, then assert the events contain a poster_signature_fail
    // skip, then break out of the loop manually via Ctrl-C-style
    // (not available in tests). Workaround: set max_jobs(1) and
    // post a second valid file that picks up after the bad one is
    // skipped. Two files, one bad, one good — observe both events.
    let job2 = build_minimal_job(&snip);
    let _ = publish_job_and_post(&snip, &job2, tmp_jobs.path(), None);
    run_watch_loop(&snip, &mut source, opts).unwrap();
    let events = emitter.events;
    let saw_skip = events.iter().any(|e| {
        matches!(e, WatchEvent::Skip { reason: SkipReason::PosterSignatureFail, .. })
    });
    assert!(saw_skip, "expected PosterSignatureFail skip in {events:?}");
}

#[test]
fn watch_loop_skips_when_cost_cap_max_input_tokens_exceeded() {
    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut job = build_minimal_job(&snip);
    job.accounting.input_token_count = 1_000_000;
    job.job_id = job_hash_hex(&job).unwrap();
    let _ = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);
    // Cap below the job's declared count.
    let caps = CostCaps {
        max_input_tokens: 100,
        max_output_tokens: 1_000_000,
        max_total_base_units: 2_000_000,
    };
    // Post a fallback acceptable job so max_jobs(1) can fire.
    let job2 = build_minimal_job(&snip);
    let _ = publish_job_and_post(&snip, &job2, tmp_jobs.path(), None);
    let events = run_once(
        &snip,
        tmp_jobs.path().to_path_buf(),
        tmp_out.path().to_path_buf(),
        false,
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
    assert!(cap_skip, "expected max_input_tokens cap-exceeded in {events:?}");
}

#[test]
fn watch_loop_skips_when_cost_cap_max_output_tokens_exceeded() {
    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut job = build_minimal_job(&snip);
    job.accounting.max_output_token_count = 1_000_000;
    job.job_id = job_hash_hex(&job).unwrap();
    let _ = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);
    let caps = CostCaps {
        max_input_tokens: 1_000_000,
        max_output_tokens: 100,
        max_total_base_units: 2_000_000,
    };
    let job2 = build_minimal_job(&snip);
    let _ = publish_job_and_post(&snip, &job2, tmp_jobs.path(), None);
    let events = run_once(
        &snip,
        tmp_jobs.path().to_path_buf(),
        tmp_out.path().to_path_buf(),
        false,
        caps,
        no_filters(),
    );
    let cap_skip = events.iter().any(|e| {
        matches!(
            e,
            WatchEvent::Skip {
                reason: SkipReason::CostCapExceeded { field: "max_output_tokens" },
                ..
            }
        )
    });
    assert!(cap_skip, "expected max_output_tokens cap-exceeded in {events:?}");
}

#[test]
fn watch_loop_skips_when_cost_cap_max_total_base_units_exceeded() {
    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let mut job = build_minimal_job(&snip);
    job.accounting.input_token_count = 100;
    job.accounting.max_output_token_count = 100;
    job.job_id = job_hash_hex(&job).unwrap();
    let _ = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);
    let caps = CostCaps {
        max_input_tokens: 1_000_000,
        max_output_tokens: 1_000_000,
        max_total_base_units: 150, // input(100) + max_output(100) = 200 > 150
    };
    let job2 = build_minimal_job(&snip);
    let _ = publish_job_and_post(&snip, &job2, tmp_jobs.path(), None);
    let events = run_once(
        &snip,
        tmp_jobs.path().to_path_buf(),
        tmp_out.path().to_path_buf(),
        false,
        caps,
        no_filters(),
    );
    let cap_skip = events.iter().any(|e| {
        matches!(
            e,
            WatchEvent::Skip {
                reason: SkipReason::CostCapExceeded { field: "max_total_base_units" },
                ..
            }
        )
    });
    assert!(cap_skip, "expected max_total_base_units cap-exceeded in {events:?}");
}

#[test]
fn watch_loop_skips_when_model_hash_not_in_accept_set() {
    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let job = build_minimal_job(&snip);
    let _ = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);
    // Accept only a different model_hash.
    let filters = AcceptFilters {
        model_hash_allow: vec!["00".repeat(32)],
        tokenizer_hash_allow: vec![],
    };
    let job2 = build_minimal_job(&snip);
    let _ = publish_job_and_post(&snip, &job2, tmp_jobs.path(), None);
    let events = run_once(
        &snip,
        tmp_jobs.path().to_path_buf(),
        tmp_out.path().to_path_buf(),
        false,
        liberal_caps(),
        filters,
    );
    let saw = events.iter().any(|e| {
        matches!(e, WatchEvent::Skip { reason: SkipReason::ModelHashNotInAcceptSet, .. })
    });
    assert!(saw, "expected ModelHashNotInAcceptSet skip in {events:?}");
}

#[test]
fn standalone_publish_result_link_writes_exact_published_bytes() {
    // Build a job + result; publish via the helper; assert the bytes
    // returned in PublishedResultLink.link_json match what the
    // MockSnipStore now holds at the link_snip_root.
    use omni_contributor::canonical::{job_hash_hex, posted_id_hex};
    use omni_contributor::posted::POSTED_SCHEMA_VERSION;
    use omni_contributor::watch::publish_result_link_for;
    use omni_contributor::{
        run_job, ContributorSigner, PostedJob, RunJobOptions, StubRunner,
    };

    let snip = MockSnipStore::new();
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let job = build_minimal_job(&snip);

    // Materialise the inner ContributorJob → SNIP so run_job can fetch.
    let job_json = serde_json::to_string_pretty(&job).unwrap();
    let job_root = snip.insert_bytes(job_json.as_bytes());

    // Construct a PostedJob handle the helper needs (we don't go
    // through the watch loop for this test; just exercise the
    // standalone-CLI path's invariant).
    let mut posted = PostedJob {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: String::new(),
        job_snip_root: job_root.to_hex(),
        job_hash: job_hash_hex(&job).unwrap(),
        model_hash: job.model_hash.clone(),
        posted_at_utc: "2026-05-26T00:00:00Z".into(),
        expires_at_utc: None,
        poster_pubkey_hex: None,
        poster_signature_hex: None,
        notes: None,
    };
    posted.posted_id = posted_id_hex(&posted).unwrap();

    // Run a real job through the orchestrator to produce a result.
    let runner = StubRunner::new(signer.pubkey_hex(), b"link-test".to_vec(), 7, 4);
    let result = run_job(
        &job,
        &snip,
        &runner,
        RunJobOptions {
            produced_at_utc: "2026-05-26T00:00:02Z".into(),
            signer: &signer,
            notes: None,
            job_snip_root: None,
        },
    )
    .unwrap();
    let result_json = serde_json::to_string_pretty(&result).unwrap();

    let mut emitter = CollectEmitter::new();
    let published = publish_result_link_for(
        &snip,
        &posted,
        &result_json,
        &result,
        &signer,
        &mut emitter,
    )
    .expect("publish_result_link_for");

    // Invariant: bytes returned in published.link_json must equal
    // the bytes the MockSnipStore now stores at link_snip_root.
    // (MockSnipStore is content-addressed by BLAKE3, so the round-trip
    // is byte-exact iff the helper actually published `link_json`.)
    let fetched = {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        use omni_store::SnipV2Adapter;
        snip.download_public(&published.link_snip_root, tmp.path())
            .unwrap();
        std::fs::read(tmp.path()).unwrap()
    };
    assert_eq!(
        fetched.as_slice(),
        published.link_json.as_bytes(),
        "PublishedResultLink.link_json must equal the bytes stored on SNIP at link_snip_root"
    );

    // Also assert the link round-trips through serde — i.e. the
    // operator can deserialize the local file (written from
    // link_json) back into a valid PostedResultLink.
    let parsed: omni_contributor::PostedResultLink =
        serde_json::from_str(&published.link_json).unwrap();
    assert_eq!(parsed, published.link);
}

/// Stage 12.7 — when a `ContributorStateStore` is supplied via
/// `WatchOptions.state_store`, the accepted result JSON must be
/// dual-written into `<state>/results/contributor-results/<job_id>.json`
/// AND the `posted-jobs` seen marker must survive a reopen so a
/// restart-shaped second `run_watch_loop` skips the same envelope
/// without re-fetching from SNIP.
#[test]
fn watch_loop_state_store_dual_writes_accepted_result_and_dedups_across_restart() {
    use omni_contributor::{ContributorStateStore, StateNamespace, StateObjectKind};

    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let tmp_state = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let job = build_minimal_job(&snip);
    let posted = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);
    let now_utc = "2026-05-30T00:00:00Z";

    // First "process" run: open the state store, run the loop, see
    // the result dual-written + seen marker laid down.
    {
        let (store, _) =
            ContributorStateStore::open(tmp_state.path(), false, now_utc).unwrap();
        let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
        let runner =
            StubRunner::new(signer.pubkey_hex(), b"watch-resp".to_vec(), 7, 13);
        let mut source = FilesystemSource::new(tmp_jobs.path().to_path_buf());
        let mut emitter = CollectEmitter::new();
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
            result_broadcaster: None,
            state_store: Some(&store),
        };
        run_watch_loop(&snip, &mut source, opts).unwrap();
        assert!(emitter.events.iter().any(|e| matches!(
            e,
            WatchEvent::VerifyOk { job_id, .. } if job_id == &job.job_id
        )));
        // Dual-write must have landed.
        let on_disk: Option<omni_contributor::ContributorResult> = store
            .read_verified_json(StateObjectKind::ContributorResult, &job.job_id)
            .unwrap();
        assert!(
            on_disk.is_some(),
            "accepted result must dual-write into the state store"
        );
        assert_eq!(on_disk.unwrap().job_id, job.job_id);
        // Seen marker must be down.
        assert!(store
            .is_seen(StateNamespace::PostedJobs, &posted.posted_id)
            .unwrap());
    }
    // Second "process" run: same state-dir, brand-new in-memory
    // dedup set, but the seen marker on disk must cause the loop
    // to skip the same posted_id with AlreadySeen.
    {
        let (store, _) =
            ContributorStateStore::open(tmp_state.path(), false, now_utc).unwrap();
        let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
        let runner =
            StubRunner::new(signer.pubkey_hex(), b"watch-resp".to_vec(), 7, 13);
        let mut source = FilesystemSource::new(tmp_jobs.path().to_path_buf());
        let mut emitter = CollectEmitter::new();
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
            result_broadcaster: None,
            state_store: Some(&store),
        };
        run_watch_loop(&snip, &mut source, opts).unwrap();
        let saw_dedup = emitter.events.iter().any(|e| matches!(
            e,
            WatchEvent::Skip { posted_id, reason: SkipReason::AlreadySeen }
                if posted_id == &posted.posted_id
        ));
        let saw_verify = emitter.events.iter().any(|e| matches!(
            e, WatchEvent::VerifyOk { .. }
        ));
        assert!(saw_dedup, "cross-restart skip expected; events={:?}", emitter.events);
        assert!(!saw_verify, "re-pickup should not happen on restart");
    }
}

/// Stage 12.7 — when both `publish_link` and `state_store` are set,
/// the watch loop must dual-write the `PostedResultLink` into the
/// state-dir's `results/result-links/<posted_id>.link.json` after
/// publishing it to SNIP. The CLI flag doc + Stage 12.7 protocol
/// doc both promise this; this test pins the contract.
#[test]
fn watch_loop_state_store_dual_writes_result_link_when_publish_link_enabled() {
    use omni_contributor::{ContributorStateStore, StateObjectKind};

    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let tmp_state = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let job = build_minimal_job(&snip);
    let posted = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);
    let now_utc = "2026-05-30T00:00:00Z";

    let (store, _) =
        ContributorStateStore::open(tmp_state.path(), false, now_utc).unwrap();
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"watch-resp".to_vec(), 7, 13);
    let mut source = FilesystemSource::new(tmp_jobs.path().to_path_buf());
    let mut emitter = CollectEmitter::new();
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
        result_broadcaster: None,
        state_store: Some(&store),
    };
    run_watch_loop(&snip, &mut source, opts).unwrap();

    let saw_publish = emitter
        .events
        .iter()
        .any(|e| matches!(e, WatchEvent::ResultLinkPublished { .. }));
    assert!(saw_publish, "expected ResultLinkPublished in {:?}", emitter.events);

    let link: Option<omni_contributor::PostedResultLink> = store
        .read_verified_json(StateObjectKind::PostedResultLink, &posted.posted_id)
        .unwrap();
    let link = link.expect(
        "PostedResultLink must dual-write into <state>/results/result-links/...",
    );
    assert_eq!(link.posted_id, posted.posted_id);
    assert_eq!(link.contributor_pubkey_hex, signer.pubkey_hex());
}

#[test]
fn watch_loop_skips_already_seen_posted_id() {
    // Two posted-job files for the same posted_id: only the first
    // should be picked; the second emits a skip(AlreadySeen).
    let tmp_jobs = tempfile::tempdir().unwrap();
    let tmp_out = tempfile::tempdir().unwrap();
    let snip = MockSnipStore::new();
    let job = build_minimal_job(&snip);
    let posted = publish_job_and_post(&snip, &job, tmp_jobs.path(), None);
    // Duplicate file under a different filename.
    std::fs::write(
        tmp_jobs.path().join("duplicate.json"),
        serde_json::to_string_pretty(&posted).unwrap(),
    )
    .unwrap();
    let events = run_once(
        &snip,
        tmp_jobs.path().to_path_buf(),
        tmp_out.path().to_path_buf(),
        false,
        liberal_caps(),
        no_filters(),
    );
    let saw = events.iter().any(|e| {
        matches!(e, WatchEvent::Skip { reason: SkipReason::AlreadySeen, .. })
    });
    assert!(saw, "expected AlreadySeen skip in {events:?}");
}
