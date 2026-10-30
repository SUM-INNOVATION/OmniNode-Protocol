//! Issue #149 — a restarted run resumes from its checkpoint.
//!
//! The runner here returns DIFFERENT bytes on every call, the way a
//! nondeterministic model may; once a checkpoint exists it must never be
//! called again. A "restart" is a fresh `RunCheckpointStore` over the same
//! directory and a fresh call, with no state carried in memory. Each
//! interruption point is covered: before registration, after
//! registration (root Pending), after activation before the result write,
//! during the result write, and after completion; plus repeated retries.
//! Throughout: one runner call, at most one registration, one result
//! (byte-identical on every attempt) and one accounting per job.

mod common;

use std::cell::Cell;
use std::path::{Path, PathBuf};

use common::MockSnipStore;
use omni_contributor::checkpoint::{RunCheckpointStore, RunPhase};
use omni_contributor::{
    BaseUnitRewardPolicy, CheckpointedRun, ContributorError, ContributorJob, ContributorSigner,
    InferenceRunner, JobAccounting, RunJobOptions, RunOutput, RunnerError, StageContribution,
    VerificationRequirement, WorkUnitKind, canonical::hex_lower, run_job_checkpointed,
    verify_result,
};
use omni_store::snip_v2_publish::public_object_root;

const SEED: [u8; 32] = *b"restart-test-contributor-seed-32";

/// Returns `16 bytes tagged with the call number`: different every call.
struct ChangingRunner {
    pubkey: String,
    calls: Cell<u32>,
}

impl ChangingRunner {
    fn new() -> Self {
        Self {
            pubkey: ContributorSigner::from_seed_bytes(&SEED)
                .unwrap()
                .pubkey_hex(),
            calls: Cell::new(0),
        }
    }

    fn bytes_of_call(n: u32) -> Vec<u8> {
        (0..16u8).map(|i| i ^ (n as u8).wrapping_mul(37)).collect()
    }
}

impl InferenceRunner for ChangingRunner {
    fn run(&self, _manifest: &Path, _input: &[u8]) -> Result<RunOutput, RunnerError> {
        let n = self.calls.get() + 1;
        self.calls.set(n);
        Ok(RunOutput {
            response_bytes: Self::bytes_of_call(n),
            measured_input_tokens: 16,
            // A different accounting on every call, too.
            measured_output_tokens: u64::from(n % 8),
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: self.pubkey.clone(),
                stage_label: format!("call-{n}"),
                work_unit_kind: WorkUnitKind::Tokens,
                work_units: 16 + u64::from(n % 8),
            }],
        })
    }
}

fn job(snip: &MockSnipStore) -> ContributorJob {
    let manifest = b"manifest (stand-in)".to_vec();
    let input = b"input".to_vec();
    let mut job = ContributorJob {
        schema_version: 1,
        job_id: String::new(),
        model_hash: hex_lower(blake3::hash(&manifest).as_bytes()),
        manifest_snip_root: snip.insert_bytes(&manifest).to_hex(),
        input_snip_root: snip.insert_bytes(&input).to_hex(),
        input_hash: hex_lower(blake3::hash(&input).as_bytes()),
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: JobAccounting {
            tokenizer_hash: hex_lower(blake3::hash(b"tok").as_bytes()),
            tokenizer_id: "t".into(),
            input_token_count: 16,
            max_output_token_count: 8,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-10-28T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: None,
    };
    job.job_id = omni_contributor::canonical::job_hash_hex(&job).unwrap();
    job
}

struct Rig {
    _dir: tempfile::TempDir,
    cp_dir: PathBuf,
    out: PathBuf,
    snip: MockSnipStore,
    job: ContributorJob,
    runner: ChangingRunner,
    signer: ContributorSigner,
}

impl Rig {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let snip = MockSnipStore::new();
        let job = job(&snip);
        Rig {
            cp_dir: dir.path().join(".omni-run-job"),
            out: dir.path().join("result.json"),
            _dir: dir,
            snip,
            job,
            runner: ChangingRunner::new(),
            signer: ContributorSigner::from_seed_bytes(&SEED).unwrap(),
        }
    }

    /// One attempt as a fresh process would make it, stopping after the
    /// signed result is built (the result write is separate).
    fn attempt(&self, produced_at: &str) -> Result<CheckpointedRun, ContributorError> {
        let store = RunCheckpointStore::new(&self.cp_dir);
        run_job_checkpointed(
            &self.job,
            &self.snip,
            &self.runner,
            RunJobOptions {
                produced_at_utc: produced_at.into(),
                signer: &self.signer,
                notes: Some("restart test".into()),
                job_snip_root: None,
            },
            &store,
        )
    }

    fn write(&self, run: &mut CheckpointedRun) {
        RunCheckpointStore::new(&self.cp_dir)
            .write_result(&mut run.checkpoint, &self.out, run.result_json.as_bytes())
            .unwrap();
    }

    fn phase(&self) -> Option<RunPhase> {
        RunCheckpointStore::new(&self.cp_dir)
            .load(&self.job.job_id, &self.signer.pubkey_hex())
            .unwrap()
            .map(|c| c.checkpoint.phase)
    }

    fn first_root(&self) -> omni_types::phase5::SnipV2ObjectId {
        public_object_root(&ChangingRunner::bytes_of_call(1))
    }

    /// The invariants every scenario ends with.
    fn assert_one_of_everything(&self, run: &CheckpointedRun) {
        assert_eq!(self.runner.calls.get(), 1, "the runner ran exactly once");
        assert!(self.snip.calls().ingests <= 1, "at most one registration");
        // The result is the first call's response and accounting.
        assert_eq!(run.result.response_snip_root, self.first_root().to_hex());
        assert_eq!(run.result.measured_accounting.output_token_count, 1);
        assert_eq!(
            run.result.measured_accounting.stage_contributions[0].stage_label,
            "call-1"
        );
        // No other response was ever registered.
        let second = public_object_root(&ChangingRunner::bytes_of_call(2));
        assert!(self.snip.record(&second).is_none());
        let outcome = verify_result(&self.job, &run.result, &self.snip).unwrap();
        assert!(outcome.overall_ok, "{outcome:?}");
        // One checkpoint directory for the job.
        let jobs: Vec<_> = std::fs::read_dir(&self.cp_dir).unwrap().collect();
        assert_eq!(jobs.len(), 1);
    }
}

#[test]
fn uninterrupted_run_completes_in_one_attempt() {
    let r = Rig::new();
    let mut run = r.attempt("2026-10-28T01:00:00Z").unwrap();
    assert_eq!(run.resumed_from, None);
    assert_eq!(run.publication.action, "ingested");
    r.write(&mut run);
    assert_eq!(r.phase(), Some(RunPhase::ResultWritten));
    assert_eq!(std::fs::read(&r.out).unwrap(), run.result_json.as_bytes());
    r.assert_one_of_everything(&run);
}

#[test]
fn restart_before_registration_publishes_the_original_bytes() {
    let r = Rig::new();
    // The chain RPC is down when publication starts: the checkpoint is
    // durable, nothing is registered.
    r.snip.set_rpc_down(true);
    assert!(r.attempt("2026-10-28T01:00:00Z").is_err());
    assert_eq!(r.phase(), Some(RunPhase::Publishing));
    assert_eq!(r.snip.calls().ingests, 0);
    assert!(!r.out.exists());

    r.snip.set_rpc_down(false);
    let mut run = r.attempt("2026-10-28T02:00:00Z").unwrap();
    assert_eq!(run.resumed_from, Some(RunPhase::Publishing));
    assert_eq!(run.publication.action, "ingested");
    // The recorded production time, not this attempt's.
    assert_eq!(run.result.produced_at_utc, "2026-10-28T01:00:00Z");
    r.write(&mut run);
    r.assert_one_of_everything(&run);
}

#[test]
fn restart_after_registration_resumes_the_pending_root() {
    let r = Rig::new();
    r.snip.lose_rpc_after_next_registration();
    assert!(r.attempt("2026-10-28T01:00:00Z").is_err());
    assert_eq!(
        r.snip.record(&r.first_root()).unwrap().lifecycle,
        0,
        "Pending"
    );
    assert_eq!(r.phase(), Some(RunPhase::Publishing));
    assert!(!r.out.exists());

    r.snip.set_rpc_down(false);
    let mut run = r.attempt("2026-10-28T02:00:00Z").unwrap();
    assert_eq!(run.publication.action, "resumed");
    assert_eq!(r.snip.calls().ingests, 1);
    assert_eq!(r.snip.calls().resumes, 1);
    r.write(&mut run);
    r.assert_one_of_everything(&run);
}

#[test]
fn restart_after_activation_before_the_result_write_only_signs() {
    let r = Rig::new();
    let first = r.attempt("2026-10-28T01:00:00Z").unwrap();
    // Crash: the result is never written.
    drop(first);
    assert_eq!(r.phase(), Some(RunPhase::Active));
    let before = r.snip.calls();

    let mut run = r.attempt("2026-10-28T02:00:00Z").unwrap();
    assert_eq!(run.resumed_from, Some(RunPhase::Active));
    assert_eq!(r.snip.calls(), before, "no SNIP or chain call at all");
    r.write(&mut run);
    r.assert_one_of_everything(&run);
}

#[test]
fn restart_during_the_result_write_finishes_it_identically() {
    let r = Rig::new();
    let first = r.attempt("2026-10-28T01:00:00Z").unwrap();
    // Crash between the result file's rename and the checkpoint update:
    // the file holds the result, the checkpoint still reads `active`;
    // a temporary file of an interrupted write is left beside it.
    std::fs::write(&r.out, first.result_json.as_bytes()).unwrap();
    std::fs::write(r.out.with_file_name(".result.json.999.1.tmp"), b"{partial").unwrap();
    assert_eq!(r.phase(), Some(RunPhase::Active));

    let mut run = r.attempt("2026-10-28T02:00:00Z").unwrap();
    assert_eq!(run.result_json, first.result_json, "byte-identical result");
    r.write(&mut run);
    assert_eq!(r.phase(), Some(RunPhase::ResultWritten));
    assert_eq!(std::fs::read(&r.out).unwrap(), first.result_json.as_bytes());
    r.assert_one_of_everything(&run);
}

#[test]
fn a_completed_checkpoint_is_not_reprocessed() {
    let r = Rig::new();
    let mut first = r.attempt("2026-10-28T01:00:00Z").unwrap();
    r.write(&mut first);
    let before = r.snip.calls();

    let mut again = r.attempt("2026-10-28T03:00:00Z").unwrap();
    assert!(again.completed());
    assert_eq!(r.snip.calls(), before, "no SNIP or chain call");
    assert_eq!(again.result_json, first.result_json);
    r.write(&mut again);

    // The result file removed: rewritten byte for byte, nothing else.
    std::fs::remove_file(&r.out).unwrap();
    let mut again = r.attempt("2026-10-28T04:00:00Z").unwrap();
    r.write(&mut again);
    assert_eq!(std::fs::read(&r.out).unwrap(), first.result_json.as_bytes());

    // A different file at the result path is never overwritten.
    std::fs::write(&r.out, b"{\"not\":\"this result\"}").unwrap();
    let mut again = r.attempt("2026-10-28T05:00:00Z").unwrap();
    let err = RunCheckpointStore::new(&r.cp_dir)
        .write_result(&mut again.checkpoint, &r.out, again.result_json.as_bytes())
        .unwrap_err();
    assert!(err.to_string().contains("refusing to overwrite"), "{err}");
    r.assert_one_of_everything(&first);
}

#[test]
fn repeated_failed_retries_never_rerun_or_reregister() {
    let r = Rig::new();
    r.snip.set_rpc_down(true);
    for i in 0..4 {
        assert!(r.attempt(&format!("2026-10-28T0{i}:00:00Z")).is_err());
        assert!(!r.out.exists());
    }
    r.snip.lose_rpc_after_next_registration();
    r.snip.set_rpc_down(false);
    assert!(r.attempt("2026-10-28T05:00:00Z").is_err());
    for i in 6..8 {
        assert!(r.attempt(&format!("2026-10-28T0{i}:00:00Z")).is_err());
    }
    r.snip.set_rpc_down(false);
    let mut run = r.attempt("2026-10-28T09:00:00Z").unwrap();
    assert_eq!(run.publication.action, "resumed");
    assert_eq!(run.result.produced_at_utc, "2026-10-28T00:00:00Z");
    r.write(&mut run);
    assert_eq!(r.snip.calls().ingests, 1);
    r.assert_one_of_everything(&run);
}

#[test]
fn a_checkpoint_with_other_result_options_is_refused_untouched() {
    let r = Rig::new();
    r.snip.set_rpc_down(true);
    assert!(r.attempt("2026-10-28T01:00:00Z").is_err());
    r.snip.set_rpc_down(false);
    let before = r.snip.calls();
    let err = run_job_checkpointed(
        &r.job,
        &r.snip,
        &r.runner,
        RunJobOptions {
            produced_at_utc: "x".into(),
            signer: &r.signer,
            notes: Some("other notes".into()),
            job_snip_root: None,
        },
        &RunCheckpointStore::new(&r.cp_dir),
    )
    .unwrap_err();
    assert!(err.to_string().contains("notes"), "{err}");
    assert_eq!(r.snip.calls(), before);
    assert_eq!(r.runner.calls.get(), 1);
}

#[test]
fn a_damaged_stored_response_is_refused_before_any_publication() {
    let r = Rig::new();
    r.snip.set_rpc_down(true);
    assert!(r.attempt("2026-10-28T01:00:00Z").is_err());
    r.snip.set_rpc_down(false);
    let cp = RunCheckpointStore::new(&r.cp_dir)
        .load(&r.job.job_id, &r.signer.pubkey_hex())
        .unwrap()
        .unwrap();
    std::fs::write(cp.response_path(), ChangingRunner::bytes_of_call(2)).unwrap();
    let before = r.snip.calls();
    let err = r.attempt("2026-10-28T02:00:00Z").unwrap_err();
    assert!(err.to_string().contains("damaged"), "{err}");
    assert_eq!(r.snip.calls().ingests, before.ingests);
    assert_eq!(
        r.runner.calls.get(),
        1,
        "a damaged checkpoint is not a reason to rerun"
    );
}

#[test]
fn the_checkpoint_records_no_key_material() {
    let r = Rig::new();
    let run = r.attempt("2026-10-28T01:00:00Z").unwrap();
    let text = std::fs::read_to_string(run.checkpoint.path()).unwrap();
    assert!(!text.contains(&hex_lower(&SEED)), "no seed");
    assert!(!text.contains("signature"), "no signature field");
    for dir_entry in std::fs::read_dir(&run.checkpoint.dir).unwrap() {
        let bytes = std::fs::read(dir_entry.unwrap().path()).unwrap();
        assert!(
            !bytes.windows(32).any(|w| w == SEED),
            "no raw seed in any file"
        );
    }
}
