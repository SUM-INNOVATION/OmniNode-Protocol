//! #131 item 1(a) — the in-process fixed-point MLP runner through the real
//! run path: SNIP fetch, the runner, the pre-publish accounting check, the
//! signed result, and `verify_result`.
//!
//! The manifest is `canonical_spec.json` itself (the model is its exact
//! bytes), and the input is the frozen canonical input.

mod common;

use common::MockSnipStore;
use omni_contributor::{
    ContributorError, ContributorJob, ContributorSigner, FixedPointMlpRunner, RunJobOptions,
    RunnerError, VerificationRequirement, WorkUnitKind, canonical::hex_lower, run_job,
    verify_result,
};

const CONTRIBUTOR_SEED: [u8; 32] = *b"contributor-test-seed-bytes-32!!";

const CANONICAL_INPUT: [i16; 16] = [
    -5, 10, 20, -100, 7, -3, 14, 25, -8, 1, 11, -22, 4, -1, 17, 9,
];
const CANONICAL_OUTPUT: [i16; 8] = [-32, -12, 6, 25, -21, -3, 16, -29];

fn le(values: &[i16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn canonical_spec() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../omni-proofs-halo2-production-mlp/assets/canonical_spec.json");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// A job for the canonical model whose manifest object holds
/// `manifest_bytes` (the canonical spec, unless a test substitutes others).
fn mlp_job(
    snip: &MockSnipStore,
    runner: &FixedPointMlpRunner,
    manifest_bytes: &[u8],
) -> ContributorJob {
    let input_bytes = le(&CANONICAL_INPUT);
    let manifest_root = snip.insert_bytes(manifest_bytes);
    let input_root = snip.insert_bytes(&input_bytes);
    let mut job = ContributorJob {
        schema_version: 1,
        job_id: String::new(),
        model_hash: runner.model_hash_hex().to_string(),
        manifest_snip_root: manifest_root.to_hex(),
        input_snip_root: input_root.to_hex(),
        input_hash: hex_lower(blake3::hash(&input_bytes).as_bytes()),
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: runner.job_accounting(),
        dispatched_at_utc: "2026-11-06T00:00:00Z".to_string(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: None,
    };
    job.job_id = omni_contributor::canonical::job_hash_hex(&job).expect("job_hash_hex");
    job
}

fn opts(signer: &ContributorSigner) -> RunJobOptions<'_> {
    RunJobOptions {
        produced_at_utc: "2026-11-06T00:00:01Z".to_string(),
        signer,
        notes: None,
        job_snip_root: None,
    }
}

#[test]
fn a_canonical_mlp_job_runs_and_verifies_with_c1a_accounting() {
    let snip = MockSnipStore::new();
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = FixedPointMlpRunner::canonical(signer.pubkey_hex());
    let job = mlp_job(&snip, &runner, &canonical_spec());
    runner.check_job(&job).expect("check_job");

    let result = run_job(&job, &snip, &runner, opts(&signer)).expect("run_job");

    let a = &result.measured_accounting;
    assert_eq!((a.input_token_count, a.output_token_count), (16, 8));
    assert_eq!(a.total_base_units, 24);
    assert_eq!(a.tokenizer_hash, job.model_hash);
    assert_eq!(a.stage_contributions.len(), 1);
    let sc = &a.stage_contributions[0];
    assert_eq!(sc.work_unit_kind, WorkUnitKind::FlopsEstimate);
    assert_eq!(sc.work_units, 2_304);
    assert_eq!(sc.contributor_pubkey_hex, signer.pubkey_hex());
    assert_eq!(
        result.response_hash,
        hex_lower(blake3::hash(&le(&CANONICAL_OUTPUT)).as_bytes())
    );

    let outcome = verify_result(&job, &result, &snip).expect("verify_result");
    assert!(outcome.overall_ok, "{outcome:?}");
    assert!(outcome.response_hash_ok);
    assert!(outcome.tokenizer_hash_ok);
    assert!(outcome.accounting_input_match);
    assert!(outcome.accounting_output_under_cap);
    assert_eq!(outcome.total_base_units, 24);
}

/// A job that claims the canonical model but whose manifest object holds
/// other bytes is refused by the runner, before anything is published.
#[test]
fn a_job_whose_manifest_is_not_the_model_is_refused_before_publication() {
    let snip = MockSnipStore::new();
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = FixedPointMlpRunner::canonical(signer.pubkey_hex());
    let mut other = canonical_spec();
    other[200] ^= 1;
    let job = mlp_job(&snip, &runner, &other);
    let objects_before = snip.object_count();

    match run_job(&job, &snip, &runner, opts(&signer)) {
        Err(ContributorError::Runner(RunnerError::ModelManifestMismatch { expected, got })) => {
            assert_eq!(expected, runner.model_hash_hex());
            assert_eq!(got, blake3::hash(&other).to_hex().as_str());
        }
        other => panic!("expected a manifest mismatch, got {other:?}"),
    }
    assert_eq!(snip.object_count(), objects_before, "nothing published");
}

/// The orchestrator's own check still applies: a job declaring another input
/// count is refused after the run, before publication.
#[test]
fn a_job_with_another_input_count_is_refused_by_the_orchestrator() {
    let snip = MockSnipStore::new();
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = FixedPointMlpRunner::canonical(signer.pubkey_hex());
    let mut job = mlp_job(&snip, &runner, &canonical_spec());
    job.accounting.input_token_count = 32;
    job.job_id = omni_contributor::canonical::job_hash_hex(&job).expect("job_hash_hex");
    assert!(runner.check_job(&job).is_err());
    let objects_before = snip.object_count();
    assert!(matches!(
        run_job(&job, &snip, &runner, opts(&signer)),
        Err(ContributorError::Verify(_))
    ));
    assert_eq!(snip.object_count(), objects_before, "nothing published");
}
