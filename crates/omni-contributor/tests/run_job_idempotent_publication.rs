//! Issue #149 — `run_job` publishes its response bytes idempotently.
//!
//! Against [`MockSnipStore`], which keeps a chain record per root and, like
//! the chain, refuses to register a root it already holds: running a job
//! again, two jobs with identical response bytes, and a retry after an
//! interruption all finish, with exactly one registration of the root and a
//! result that verifies; every state that is not a matching Active record
//! ends in an error and no result.

mod common;

use common::{MOCK_ACCOUNT, MockSnipStore};
use omni_contributor::{
    BaseUnitRewardPolicy, ContributorError, ContributorJob, ContributorResult, ContributorSigner,
    JobAccounting, RunJobOptions, StubRunner, VerificationRequirement, canonical::hex_lower,
    run_job, run_job_with_publication, verify_result,
};
use omni_store::snip_v2_publish::{PublicationAction, public_object_root};

const CONTRIBUTOR_SEED: [u8; 32] = *b"contributor-test-seed-bytes-32!!";
/// The canonical model maps many inputs to the same 16 output bytes.
const RESPONSE: &[u8] = b"\x01\x00\xff\xff\x02\x00\xfe\xff\x03\x00\xfd\xff\x04\x00\xfc\xff";

fn job_with_input(snip: &MockSnipStore, input: &[u8]) -> ContributorJob {
    let manifest = b"manifest: production-fixedpoint-mlp-v1 (stand-in)".to_vec();
    let manifest_root = snip.insert_bytes(&manifest);
    let input_root = snip.insert_bytes(input);
    let mut job = ContributorJob {
        schema_version: 1,
        job_id: String::new(),
        model_hash: hex_lower(blake3::hash(&manifest).as_bytes()),
        manifest_snip_root: manifest_root.to_hex(),
        input_snip_root: input_root.to_hex(),
        input_hash: hex_lower(blake3::hash(input).as_bytes()),
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: JobAccounting {
            tokenizer_hash: hex_lower(blake3::hash(b"tok").as_bytes()),
            tokenizer_id: "demo/tokenizer".into(),
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

fn run(
    snip: &MockSnipStore,
    job: &ContributorJob,
) -> Result<(ContributorResult, PublicationAction), ContributorError> {
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), RESPONSE.to_vec(), 16, 8);
    run_job_with_publication(
        job,
        snip,
        &runner,
        RunJobOptions {
            produced_at_utc: "2026-10-28T00:00:01Z".into(),
            signer: &signer,
            notes: None,
            job_snip_root: None,
        },
    )
    .map(|(r, p)| (r, p.action))
}

fn assert_verifies(snip: &MockSnipStore, job: &ContributorJob, result: &ContributorResult) {
    let outcome = verify_result(job, result, snip).expect("verify_result");
    assert!(outcome.overall_ok, "{outcome:?}");
    assert!(outcome.response_hash_ok);
}

fn response_root_hex() -> String {
    public_object_root(RESPONSE).to_hex()
}

#[test]
fn the_same_job_run_twice_registers_its_response_once() {
    let snip = MockSnipStore::new();
    let job = job_with_input(&snip, b"input-a");

    let (first, a1) = run(&snip, &job).expect("first run");
    let (second, a2) = run(&snip, &job).expect("second run");

    assert_eq!(a1, PublicationAction::Ingested);
    assert_eq!(a2, PublicationAction::ReusedActive);
    assert_eq!(first.response_snip_root, response_root_hex());
    assert_eq!(second.response_snip_root, first.response_snip_root);
    assert_eq!(second.response_hash, first.response_hash);
    assert_eq!(snip.calls().ingests, 1, "one registration");
    assert_eq!(snip.calls().resumes, 0);
    assert_verifies(&snip, &job, &first);
    assert_verifies(&snip, &job, &second);
}

#[test]
fn different_jobs_with_identical_response_bytes_share_one_registration() {
    let snip = MockSnipStore::new();
    let job_a = job_with_input(&snip, b"input-a");
    let job_b = job_with_input(&snip, b"input-b");
    assert_ne!(job_a.job_id, job_b.job_id);

    let (ra, aa) = run(&snip, &job_a).unwrap();
    let (rb, ab) = run(&snip, &job_b).unwrap();

    assert_eq!(aa, PublicationAction::Ingested);
    assert_eq!(ab, PublicationAction::ReusedActive);
    assert_eq!(ra.response_snip_root, rb.response_snip_root);
    assert_ne!(ra.job_id, rb.job_id);
    assert_eq!(snip.calls().ingests, 1);
    assert_verifies(&snip, &job_a, &ra);
    assert_verifies(&snip, &job_b, &rb);
}

#[test]
fn an_interruption_after_registration_is_resumed_not_registered_again() {
    let snip = MockSnipStore::new();
    let job = job_with_input(&snip, b"input-a");

    // Attempt 1: the registration lands, then the run loses its RPC.
    snip.lose_rpc_after_next_registration();
    let err = run(&snip, &job).expect_err("attempt 1 must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("error sending request"),
        "SNIP's reason kept: {msg}"
    );
    assert!(
        msg.contains("Connection refused"),
        "the re-read's failure kept: {msg}"
    );
    let root = public_object_root(RESPONSE);
    assert_eq!(snip.record(&root).unwrap().lifecycle, 0, "left Pending");

    // Attempt 2: the RPC is back; the run resumes its own Pending root.
    snip.set_rpc_down(false);
    let (result, action) = run(&snip, &job).expect("retry");
    assert_eq!(action, PublicationAction::Resumed);
    assert_eq!(snip.record(&root).unwrap().lifecycle, 1);
    assert_eq!(snip.calls().ingests, 1, "never registered again");
    assert_eq!(snip.calls().resumes, 1);
    assert_verifies(&snip, &job, &result);
}

#[test]
fn a_retry_after_activation_but_before_the_result_reuses_the_active_root() {
    // The earlier run got as far as Active and died before its result was
    // written: all that is left of it is the Active record.
    let snip = MockSnipStore::new();
    let job = job_with_input(&snip, b"input-a");
    snip.put_record(RESPONSE, MOCK_ACCOUNT, 1, |_| {});
    snip.insert_bytes(RESPONSE);

    let (result, action) = run(&snip, &job).expect("retry");
    assert_eq!(action, PublicationAction::ReusedActive);
    assert_eq!(snip.calls().ingests, 0);
    assert_eq!(snip.calls().resumes, 0);
    assert_verifies(&snip, &job, &result);
}

#[test]
fn an_unreachable_rpc_fails_before_registering_and_produces_no_result() {
    let snip = MockSnipStore::new();
    let job = job_with_input(&snip, b"input-a");
    snip.set_rpc_down(true);
    let err = run(&snip, &job).expect_err("must fail");
    assert!(err.to_string().contains("storage_getFileInfoV2"), "{err}");
    assert_eq!(snip.calls().ingests, 0, "nothing registered blindly");
    assert!(snip.record(&public_object_root(RESPONSE)).is_none());
}

#[test]
fn a_pending_root_of_another_account_is_refused() {
    let snip = MockSnipStore::new();
    let job = job_with_input(&snip, b"input-a");
    snip.put_record(RESPONSE, "AnotherAccount11111111111111111", 0, |_| {});
    let err = run(&snip, &job).expect_err("must fail");
    let msg = err.to_string();
    assert!(msg.contains("AnotherAccount11111111111111111"), "{msg}");
    assert!(msg.contains(MOCK_ACCOUNT), "{msg}");
    assert_eq!(snip.calls().ingests, 0);
    assert_eq!(snip.calls().resumes, 0);
}

#[test]
fn conflicting_chain_metadata_is_refused() {
    for lifecycle in [0u8, 1] {
        let snip = MockSnipStore::new();
        let job = job_with_input(&snip, b"input-a");
        snip.put_record(RESPONSE, MOCK_ACCOUNT, lifecycle, |r| {
            r.plaintext_size_bytes += 1;
            r.stored_size_bytes += 1;
        });
        let err = run(&snip, &job).expect_err("must fail");
        assert!(err.to_string().contains("plaintext_size_bytes"), "{err}");
        assert_eq!(snip.calls().ingests, 0);
        assert_eq!(snip.calls().resumes, 0);
    }
}

#[test]
fn an_abandoned_root_is_refused() {
    let snip = MockSnipStore::new();
    let job = job_with_input(&snip, b"input-a");
    snip.put_record(RESPONSE, MOCK_ACCOUNT, 2, |_| {});
    let err = run(&snip, &job).expect_err("must fail");
    assert!(err.to_string().contains("Abandoned"), "{err}");
    assert_eq!(snip.calls().ingests, 0);
    assert_eq!(snip.calls().resumes, 0);
}

#[test]
fn run_job_returns_the_same_result_as_run_job_with_publication() {
    let snip = MockSnipStore::new();
    let job = job_with_input(&snip, b"input-a");
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), RESPONSE.to_vec(), 16, 8);
    let opts = || RunJobOptions {
        produced_at_utc: "2026-10-28T00:00:01Z".into(),
        signer: &signer,
        notes: None,
        job_snip_root: None,
    };
    let plain = run_job(&job, &snip, &runner, opts()).unwrap();
    let (with, published) = run_job_with_publication(&job, &snip, &runner, opts()).unwrap();
    assert_eq!(plain, with);
    assert_eq!(published.root.to_hex(), plain.response_snip_root);
    assert_eq!(hex_lower(&published.hash), plain.response_hash);
}
