//! Stage 12.0 — end-to-end AttestationOnly roundtrip.

mod common;

use common::MockSnipStore;
use omni_contributor::{
    canonical::hex_lower, run_job, verify_result, BaseUnitRewardPolicy, ContributorJob,
    ContributorSigner, DispatcherSigner, Evidence, JobAccounting, RunJobOptions, StubRunner,
    VerificationRequirement,
};

const CONTRIBUTOR_SEED: [u8; 32] = *b"contributor-test-seed-bytes-32!!";
const DISPATCHER_SEED: [u8; 32] = *b"dispatcher-test-seed-bytes-32!!!";

fn build_canonical_inputs() -> (Vec<u8>, Vec<u8>) {
    // Manifest is just a stand-in opaque payload.
    let manifest_bytes = b"manifest-v1: model=demo, weights=[...], tokenizer=ref".to_vec();
    let input_bytes = b"prompt: tell me a joke".to_vec();
    (manifest_bytes, input_bytes)
}

fn build_job(
    snip: &MockSnipStore,
    dispatcher: Option<&DispatcherSigner>,
) -> ContributorJob {
    let (manifest_bytes, input_bytes) = build_canonical_inputs();
    let manifest_root = snip.insert_bytes(&manifest_bytes);
    let input_root = snip.insert_bytes(&input_bytes);
    let input_hash = hex_lower(blake3::hash(&input_bytes).as_bytes());
    let model_hash = hex_lower(blake3::hash(&manifest_bytes).as_bytes());
    let tokenizer_hash = hex_lower(blake3::hash(b"demo-tokenizer-bytes").as_bytes());

    let mut job = ContributorJob {
        schema_version: 1,
        job_id: String::new(), // filled in below
        model_hash,
        manifest_snip_root: manifest_root.to_hex(),
        input_snip_root: input_root.to_hex(),
        input_hash,
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: JobAccounting {
            tokenizer_hash,
            tokenizer_id: "demo/tokenizer".to_string(),
            input_token_count: 7,
            max_output_token_count: 100,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-05-25T00:00:00Z".to_string(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: dispatcher.map(|d| d.pubkey_hex()),
        dispatcher_signature_hex: None, // filled below if signer present
        notes: None,
    };

    // Fill job_id from canonical job_hash.
    job.job_id =
        omni_contributor::canonical::job_hash_hex(&job).expect("job_hash_hex");
    if let Some(d) = dispatcher {
        let signing_input =
            omni_contributor::canonical::canonical_job_bytes(&job).expect("canonical_job_bytes");
        job.dispatcher_signature_hex = Some(d.sign_hex(&signing_input));
    }
    job
}

#[test]
fn attestation_only_roundtrip_unsigned_dispatcher() {
    let snip = MockSnipStore::new();
    let job = build_job(&snip, None);
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();

    let runner = StubRunner::new(
        signer.pubkey_hex(),
        b"42, said the model.".to_vec(),
        7,
        13,
    );

    let result = run_job(
        &job,
        &snip,
        &runner,
        RunJobOptions {
            produced_at_utc: "2026-05-25T00:00:01Z".to_string(),
            signer: &signer,
            notes: Some("unsigned dispatcher".into()),
            job_snip_root: None,
        },
    )
    .expect("run_job");

    // Verify it.
    let outcome = verify_result(&job, &result, &snip).expect("verify_result");
    assert!(outcome.overall_ok, "outcome should be ok: {outcome:?}");
    assert!(outcome.job_hash_ok);
    assert!(outcome.input_hash_ok);
    assert!(outcome.response_hash_ok);
    assert!(outcome.tokenizer_hash_ok);
    assert!(outcome.accounting_input_match);
    assert!(outcome.accounting_output_under_cap);
    assert!(outcome.accounting_total_consistent);
    assert_eq!(outcome.total_base_units, 20);
    assert!(outcome.contributor_signature_ok);
    assert_eq!(outcome.evidence_mode, "attestation_only");
    assert!(outcome.evidence_ok);
    assert!(outcome.requirement_satisfied);
    assert!(matches!(
        outcome.dispatcher_signature,
        omni_contributor::verify::DispatcherSignatureOutcome::NotSigned
    ));
    assert!(matches!(result.evidence, Evidence::AttestationOnly));
}

#[test]
fn attestation_only_roundtrip_signed_dispatcher() {
    let snip = MockSnipStore::new();
    let dispatcher = DispatcherSigner::from_seed_bytes(&DISPATCHER_SEED).unwrap();
    let job = build_job(&snip, Some(&dispatcher));
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let runner = StubRunner::new(signer.pubkey_hex(), b"signed-response".to_vec(), 7, 4);

    let result = run_job(
        &job,
        &snip,
        &runner,
        RunJobOptions {
            produced_at_utc: "2026-05-25T00:00:02Z".to_string(),
            signer: &signer,
            notes: None,
            job_snip_root: None,
        },
    )
    .expect("run_job");

    let outcome = verify_result(&job, &result, &snip).expect("verify_result");
    assert!(outcome.overall_ok, "outcome={outcome:?}");
    assert!(matches!(
        outcome.dispatcher_signature,
        omni_contributor::verify::DispatcherSignatureOutcome::Ok
    ));
}

#[test]
fn dispatcher_seed_distinct_from_contributor_seed() {
    let dispatcher = DispatcherSigner::from_seed_bytes(&DISPATCHER_SEED).unwrap();
    let signer = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    assert_ne!(
        dispatcher.pubkey_hex(),
        signer.pubkey_hex(),
        "seeds must derive distinct pubkeys (sanity check on test setup)"
    );
}

#[test]
fn job_hash_is_deterministic_across_clones() {
    let snip = MockSnipStore::new();
    let job_a = build_job(&snip, None);
    let job_b = job_a.clone();
    let h_a = omni_contributor::canonical::job_hash_hex(&job_a).unwrap();
    let h_b = omni_contributor::canonical::job_hash_hex(&job_b).unwrap();
    assert_eq!(h_a, h_b);
    assert_eq!(job_a.job_id, h_a);
}
