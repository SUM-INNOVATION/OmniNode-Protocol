//! Stage 12.0 — negative-case verification tests. Each test starts
//! from a happy-path `(job, result)` and mutates exactly one field
//! to assert the verifier catches the drift.

mod common;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{hex_lower, job_hash_hex},
    error::{SchemaError, VerifyError},
    run_job, verify_result, BaseUnitRewardPolicy, ContributorJob, ContributorResult,
    ContributorSigner, DispatcherSigner, JobAccounting, RunJobOptions, StubRunner,
    VerificationRequirement,
};

const CONTRIBUTOR_SEED: [u8; 32] = *b"contributor-test-seed-bytes-32!!";
const DISPATCHER_SEED: [u8; 32] = *b"dispatcher-test-seed-bytes-32!!!";

fn happy_path() -> (MockSnipStore, ContributorJob, ContributorResult) {
    let snip = MockSnipStore::new();
    let manifest_bytes = b"manifest-bytes".to_vec();
    let input_bytes = b"input-bytes".to_vec();
    let manifest_root = snip.insert_bytes(&manifest_bytes);
    let input_root = snip.insert_bytes(&input_bytes);
    let input_hash = hex_lower(blake3::hash(&input_bytes).as_bytes());
    let model_hash = hex_lower(blake3::hash(&manifest_bytes).as_bytes());
    let tokenizer_hash = hex_lower(blake3::hash(b"tok").as_bytes());

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
            tokenizer_id: "demo/tokenizer".into(),
            input_token_count: 5,
            max_output_token_count: 50,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-05-25T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: None,
    };
    job.job_id = job_hash_hex(&job).unwrap();

    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"resp".to_vec(), 5, 10);
    let result = run_job(
        &job,
        &snip,
        &runner,
        RunJobOptions {
            produced_at_utc: "2026-05-25T00:00:01Z".into(),
            signer: &signer,
            notes: None,
            job_snip_root: None,
        },
    )
    .unwrap();
    (snip, job, result)
}

#[test]
fn happy_path_actually_passes() {
    let (snip, job, result) = happy_path();
    let outcome = verify_result(&job, &result, &snip).unwrap();
    assert!(outcome.overall_ok, "sanity-check baseline must pass");
}

#[test]
fn job_hash_mismatch_in_result_is_flagged() {
    let (snip, job, mut result) = happy_path();
    // Flip one nibble in result.job_hash so it no longer matches the
    // recomputed value.
    let mut bytes = result.job_hash.clone().into_bytes();
    bytes[0] = if bytes[0] == b'a' { b'b' } else { b'a' };
    result.job_hash = String::from_utf8(bytes).unwrap();
    let outcome = verify_result(&job, &result, &snip).unwrap();
    assert!(!outcome.overall_ok);
    assert!(!outcome.job_hash_ok);
}

#[test]
fn model_hash_drift_between_job_and_result_rejected() {
    let (snip, job, mut result) = happy_path();
    result.model_hash = "0".repeat(64);
    let err = verify_result(&job, &result, &snip).unwrap_err();
    assert!(matches!(err, VerifyError::ModelHashMismatch { .. }), "{err:?}");
}

#[test]
fn input_hash_drift_between_job_and_result_rejected() {
    let (snip, job, mut result) = happy_path();
    result.input_hash = "1".repeat(64);
    let err = verify_result(&job, &result, &snip).unwrap_err();
    assert!(matches!(err, VerifyError::InputHashMismatch { .. }), "{err:?}");
}

#[test]
fn response_hash_drift_is_flagged() {
    let (snip, job, mut result) = happy_path();
    result.response_hash = "2".repeat(64);
    let outcome = verify_result(&job, &result, &snip).unwrap();
    assert!(!outcome.overall_ok);
    assert!(!outcome.response_hash_ok);
}

#[test]
fn tampered_contributor_signature_is_flagged() {
    let (snip, job, mut result) = happy_path();
    // Flip the last nibble.
    let mut bytes = result.contributor_signature_hex.clone().into_bytes();
    let last = bytes.len() - 1;
    bytes[last] = if bytes[last] == b'a' { b'b' } else { b'a' };
    result.contributor_signature_hex = String::from_utf8(bytes).unwrap();
    let outcome = verify_result(&job, &result, &snip).unwrap();
    assert!(!outcome.overall_ok);
    assert!(!outcome.contributor_signature_ok);
}

#[test]
fn tampered_dispatcher_signature_is_flagged() {
    let snip = MockSnipStore::new();
    let dispatcher = DispatcherSigner::from_seed_bytes(&DISPATCHER_SEED).unwrap();
    let manifest_bytes = b"mb".to_vec();
    let input_bytes = b"ib".to_vec();
    let manifest_root = snip.insert_bytes(&manifest_bytes);
    let input_root = snip.insert_bytes(&input_bytes);
    let input_hash = hex_lower(blake3::hash(&input_bytes).as_bytes());
    let model_hash = hex_lower(blake3::hash(&manifest_bytes).as_bytes());
    let tokenizer_hash = hex_lower(blake3::hash(b"tok").as_bytes());
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
            tokenizer_id: "x/y".into(),
            input_token_count: 1,
            max_output_token_count: 10,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-05-25T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: Some(dispatcher.pubkey_hex()),
        dispatcher_signature_hex: None,
        notes: None,
    };
    job.job_id = job_hash_hex(&job).unwrap();
    let signing_input =
        omni_contributor::canonical::canonical_job_bytes(&job).unwrap();
    job.dispatcher_signature_hex = Some(dispatcher.sign_hex(&signing_input));
    // Tamper.
    let mut bytes = job.dispatcher_signature_hex.clone().unwrap().into_bytes();
    bytes[0] = if bytes[0] == b'a' { b'b' } else { b'a' };
    job.dispatcher_signature_hex = Some(String::from_utf8(bytes).unwrap());

    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"r".to_vec(), 1, 3);
    // run_job would reject the bad dispatcher signature before
    // reaching the runner. Verify that.
    let err = run_job(
        &job,
        &snip,
        &runner,
        RunJobOptions {
            produced_at_utc: "2026-05-25T00:00:01Z".into(),
            signer: &signer,
            notes: None,
            job_snip_root: None,
        },
    )
    .unwrap_err();
    // The orchestrator surfaces dispatcher-signature failure via
    // VerifyError::DispatcherSignatureFailed wrapped in ContributorError.
    assert!(format!("{err}").contains("dispatcher"), "{err}");
}

#[test]
fn dispatcher_half_identity_pubkey_only_rejected() {
    let (snip, mut job, result) = happy_path();
    job.dispatcher_pubkey_hex = Some("a".repeat(64));
    job.dispatcher_signature_hex = None;
    let err = verify_result(&job, &result, &snip).unwrap_err();
    assert!(
        matches!(
            err,
            VerifyError::Schema(SchemaError::InconsistentDispatcherIdentity { .. })
        ),
        "{err:?}"
    );
}

#[test]
fn dispatcher_half_identity_signature_only_rejected() {
    let (snip, mut job, result) = happy_path();
    job.dispatcher_pubkey_hex = None;
    job.dispatcher_signature_hex = Some("b".repeat(128));
    let err = verify_result(&job, &result, &snip).unwrap_err();
    assert!(
        matches!(
            err,
            VerifyError::Schema(SchemaError::InconsistentDispatcherIdentity { .. })
        ),
        "{err:?}"
    );
}

#[test]
fn expired_job_rejected() {
    let (snip, mut job, mut result) = happy_path();
    // Real-world flow: dispatcher sets expiry on the body BEFORE
    // computing job_id. Mirror that by recomputing job_id after the
    // mutation; the verifier then succeeds at job_hash check, reaches
    // the expiry check, and refuses.
    job.expires_at_utc = Some("2000-01-01T00:00:00Z".into());
    job.job_id = job_hash_hex(&job).unwrap();
    result.job_id = job.job_id.clone();
    result.job_hash = job.job_id.clone();
    // Re-sign the result so contributor_signature_hex matches the new
    // job_id / job_hash. Build a signer from the same seed the
    // happy_path used.
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    result.contributor_signature_hex = String::new();
    let signing_input =
        omni_contributor::canonical::contributor_signing_input(&result).unwrap();
    result.contributor_signature_hex = signer.sign_hex(&signing_input);

    let err = verify_result(&job, &result, &snip).unwrap_err();
    assert!(matches!(err, VerifyError::JobExpired { .. }), "{err:?}");
}

#[test]
fn accounting_input_mismatch_flagged() {
    let (snip, mut job, result) = happy_path();
    job.accounting.input_token_count = 999;
    // Recompute job_id since we changed the body.
    job.job_id = job_hash_hex(&job).unwrap();
    // result.job_hash still points to the old value, so step 2 will
    // also fail. But accounting check happens regardless; the outcome
    // booleans tell us both.
    let outcome = verify_result(&job, &result, &snip).unwrap();
    assert!(!outcome.overall_ok);
    assert!(!outcome.accounting_input_match);
}

#[test]
fn accounting_output_over_cap_flagged() {
    let (snip, mut job, mut result) = happy_path();
    // Lower the cap below what the result reports.
    job.accounting.max_output_token_count = 5;
    job.job_id = job_hash_hex(&job).unwrap();
    // Result's output_token_count is 10; ensure consistency in the
    // result itself so the totals check still passes.
    result.measured_accounting.output_token_count = 10;
    result.measured_accounting.total_base_units =
        result.measured_accounting.input_token_count + 10;
    let outcome = verify_result(&job, &result, &snip).unwrap();
    assert!(!outcome.overall_ok);
    assert!(!outcome.accounting_output_under_cap);
}

#[test]
fn accounting_total_inconsistent_flagged() {
    let (snip, job, mut result) = happy_path();
    // Total no longer matches input + output.
    result.measured_accounting.total_base_units = 999;
    let outcome = verify_result(&job, &result, &snip).unwrap();
    assert!(!outcome.overall_ok);
    assert!(!outcome.accounting_total_consistent);
}

#[test]
fn tokenizer_hash_mismatch_flagged() {
    let (snip, job, mut result) = happy_path();
    result.measured_accounting.tokenizer_hash = "f".repeat(64);
    let outcome = verify_result(&job, &result, &snip).unwrap();
    assert!(!outcome.overall_ok);
    assert!(!outcome.tokenizer_hash_ok);
}

#[test]
fn empty_stage_contributions_rejected() {
    let (snip, job, mut result) = happy_path();
    result.measured_accounting.stage_contributions.clear();
    let err = verify_result(&job, &result, &snip).unwrap_err();
    assert!(
        matches!(err, VerifyError::Schema(SchemaError::EmptyStageContributions)),
        "{err:?}"
    );
}

#[test]
fn unknown_work_unit_kind_in_json_rejected_at_parse() {
    // JSON deserialization of a result with an unknown work_unit_kind
    // variant must fail (closed enum).
    let bad = r#"{"contributor_pubkey_hex":"00","stage_label":"x","work_unit_kind":"flux_capacitor","work_units":1}"#;
    let result: Result<omni_contributor::StageContribution, _> = serde_json::from_str(bad);
    assert!(result.is_err(), "unknown work_unit_kind must reject");
}

// ── run-job pre-signing refusals (orchestrator hardening) ────────────────
// These tests prove run_job refuses BEFORE producing a signed
// ContributorResult, instead of emitting a result the verifier would
// later reject.

fn run_job_with_runner_counts(
    snip: &MockSnipStore,
    job: &ContributorJob,
    measured_input: u64,
    measured_output: u64,
) -> Result<omni_contributor::ContributorResult, omni_contributor::error::ContributorError> {
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(
        signer.pubkey_hex(),
        b"r".to_vec(),
        measured_input,
        measured_output,
    );
    omni_contributor::run_job(
        job,
        snip,
        &runner,
        omni_contributor::RunJobOptions {
            produced_at_utc: "2026-05-25T00:00:01Z".into(),
            signer: &signer,
            notes: None,
            job_snip_root: None,
        },
    )
}

#[test]
fn run_job_refuses_job_with_drifted_job_id_before_signing() {
    let (snip, mut job, _result) = happy_path();
    // Mutate job_id without recomputing — the orchestrator must catch
    // the drift and refuse BEFORE the SNIP fetch / runner step.
    job.job_id = "0".repeat(64);
    let err = run_job_with_runner_counts(
        &snip,
        &job,
        job.accounting.input_token_count,
        10,
    )
    .unwrap_err();
    let s = format!("{err}");
    assert!(
        s.contains("job_id") || s.contains("JobIdMismatch"),
        "expected job-id-mismatch refusal, got {s}"
    );
}

#[test]
fn run_job_refuses_expired_job_before_signing() {
    let (snip, mut job, _result) = happy_path();
    job.expires_at_utc = Some("2000-01-01T00:00:00Z".into());
    job.job_id = job_hash_hex(&job).unwrap();
    let err = run_job_with_runner_counts(
        &snip,
        &job,
        job.accounting.input_token_count,
        10,
    )
    .unwrap_err();
    let s = format!("{err}");
    assert!(
        s.contains("expired") || s.contains("Expired") || s.contains("JobExpired"),
        "expected expiry refusal, got {s}"
    );
}

#[test]
fn run_job_refuses_runner_with_wrong_input_count() {
    let (snip, job, _result) = happy_path();
    // job declares 5 input tokens; runner reports 7.
    let err = run_job_with_runner_counts(&snip, &job, 7, 10).unwrap_err();
    let s = format!("{err}");
    assert!(
        s.contains("input_token") || s.contains("AccountingInput"),
        "expected accounting input mismatch refusal, got {s}"
    );
}

#[test]
fn run_job_refuses_runner_exceeding_output_cap() {
    let (snip, job, _result) = happy_path();
    // job's max_output_token_count is 50; runner reports 999.
    let err =
        run_job_with_runner_counts(&snip, &job, job.accounting.input_token_count, 999)
            .unwrap_err();
    let s = format!("{err}");
    assert!(
        s.contains("output_token") || s.contains("AccountingOutput"),
        "expected output-over-cap refusal, got {s}"
    );
}

#[test]
fn run_job_refuses_runner_with_overflowing_total() {
    // Build a job whose max_output_token_count is large enough to
    // *admit* a count near u64::MAX, then run the runner with values
    // that overflow input+output. The pre-signing check catches the
    // overflow as a typed error instead of panicking.
    let snip = MockSnipStore::new();
    let manifest_bytes = b"m".to_vec();
    let input_bytes = b"i".to_vec();
    let manifest_root = snip.insert_bytes(&manifest_bytes);
    let input_root = snip.insert_bytes(&input_bytes);
    let input_hash = hex_lower(blake3::hash(&input_bytes).as_bytes());
    let model_hash = hex_lower(blake3::hash(&manifest_bytes).as_bytes());
    let tokenizer_hash = hex_lower(blake3::hash(b"t").as_bytes());

    // input_token_count must match the runner's value (the orchestrator
    // enforces strict equality). Pick u64::MAX so input+anything > 0
    // overflows.
    let huge = u64::MAX;
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
            tokenizer_id: "demo/tokenizer".into(),
            input_token_count: huge,
            max_output_token_count: huge,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-05-25T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: None,
    };
    job.job_id = job_hash_hex(&job).unwrap();

    // Runner reports huge for both → input + output overflows u64.
    let err = run_job_with_runner_counts(&snip, &job, huge, huge).unwrap_err();
    let s = format!("{err}");
    assert!(
        s.contains("overflow") || s.contains("Overflow"),
        "expected overflow refusal (no panic!), got {s}"
    );
}

#[test]
fn run_job_does_not_publish_response_when_runner_output_invalid() {
    // Orphan-prevention guarantee: if `run_job` refuses on runner
    // accounting (or stage_contributions) validity, SNIP must not
    // have been touched. The MockSnipStore tracks total object
    // count; we snapshot it after happy_path setup, attempt a
    // run_job with a bad runner, and assert the count is unchanged.
    let (snip, job, _result) = happy_path();
    let baseline = snip.object_count();
    // Wrong input count → AccountingInputMismatch before publish.
    let err =
        run_job_with_runner_counts(&snip, &job, job.accounting.input_token_count + 1, 5)
            .unwrap_err();
    let after = snip.object_count();
    assert_eq!(
        baseline, after,
        "run_job must not publish a response to SNIP when runner output is rejected; \
         err was: {err}"
    );
    // Output-over-cap path.
    let err2 = run_job_with_runner_counts(
        &snip,
        &job,
        job.accounting.input_token_count,
        job.accounting.max_output_token_count + 1,
    )
    .unwrap_err();
    let after2 = snip.object_count();
    assert_eq!(
        baseline, after2,
        "run_job must not publish on output-over-cap refusal; err: {err2}"
    );
    // Overflow path: construct a job that admits u64::MAX for both
    // counts so we can drive the runner into overflow without first
    // tripping the input-count check. Use a fresh store so we can
    // re-snapshot the baseline.
    let snip3 = MockSnipStore::new();
    let manifest_bytes = b"m3".to_vec();
    let input_bytes = b"i3".to_vec();
    let manifest_root = snip3.insert_bytes(&manifest_bytes);
    let input_root = snip3.insert_bytes(&input_bytes);
    let input_hash = hex_lower(blake3::hash(&input_bytes).as_bytes());
    let model_hash = hex_lower(blake3::hash(&manifest_bytes).as_bytes());
    let tokenizer_hash = hex_lower(blake3::hash(b"t3").as_bytes());
    let huge = u64::MAX;
    let mut overflow_job = ContributorJob {
        schema_version: 1,
        job_id: String::new(),
        model_hash,
        manifest_snip_root: manifest_root.to_hex(),
        input_snip_root: input_root.to_hex(),
        input_hash,
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: JobAccounting {
            tokenizer_hash,
            tokenizer_id: "x/y".into(),
            input_token_count: huge,
            max_output_token_count: huge,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-05-25T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: None,
    };
    overflow_job.job_id = job_hash_hex(&overflow_job).unwrap();
    let baseline3 = snip3.object_count();
    let err3 = run_job_with_runner_counts(&snip3, &overflow_job, huge, huge).unwrap_err();
    let after3 = snip3.object_count();
    assert_eq!(
        baseline3, after3,
        "run_job must not publish on overflow refusal; err: {err3}"
    );
}

#[test]
fn timestamp_without_z_suffix_rejected() {
    // Same offset semantically (+00:00 == Z) but the protocol pins
    // the literal `Z` suffix. The validator must refuse.
    let (snip, mut job, result) = happy_path();
    job.dispatched_at_utc = "2026-05-25T00:00:00+00:00".into();
    job.job_id = job_hash_hex(&job).unwrap();
    let err = verify_result(&job, &result, &snip).unwrap_err();
    let s = format!("{err}");
    assert!(
        s.contains("MalformedTimestamp") || s.contains("dispatched_at_utc"),
        "expected malformed-timestamp refusal, got {s}"
    );
    // Also the result-side timestamps.
    let (snip2, job2, mut result2) = happy_path();
    result2.produced_at_utc = "2026-05-25T00:00:01-00:00".into();
    let err2 = verify_result(&job2, &result2, &snip2).unwrap_err();
    let s2 = format!("{err2}");
    assert!(
        s2.contains("MalformedTimestamp") || s2.contains("produced_at_utc"),
        "expected malformed-timestamp refusal, got {s2}"
    );
}

#[test]
fn external_runner_envelope_rejects_unknown_fields() {
    // The ExternalRunnerEnvelope uses deny_unknown_fields. Adding any
    // unrecognised top-level field must fail parsing.
    let bad = r#"{
        "response_b64": "aGVsbG8=",
        "measured_input_tokens": 1,
        "measured_output_tokens": 2,
        "stage_contributions": [],
        "future_field": "not allowed"
    }"#;
    let parsed: Result<omni_contributor::runner::ExternalRunnerEnvelope, _> =
        serde_json::from_str(bad);
    assert!(parsed.is_err(), "unknown field must reject");
}
