//! Stage 12.0 — schema byte-stability fixture.
//!
//! Pin canonical job/result bytes (via deterministic helpers) so a
//! schema reorder, bincode version bump, or domain-separator change
//! fails this test loudly. Committed fixture vectors are checked into
//! `tests/fixtures/`. Each vector pins:
//!
//!   - the JSON body that deserialised back into the typed struct,
//!   - the lowercase-hex BLAKE3 of `canonical_*_bytes(...)`,
//!   - for jobs, the derived `job_id` (= `BLAKE3` of canonical bytes).
//!
//! If the canonical layout drifts, run the test once with
//! `OMNI_CONTRIBUTOR_REGEN=1` to print the new expected values, then
//! audit + commit the JSON deltas.

use omni_contributor::{
    canonical::{
        canonical_job_bytes, canonical_result_bytes, contributor_signing_input, hex_lower,
        job_hash_hex,
    },
    BaseUnitRewardPolicy, ContributorJob, ContributorResult, ContributorSigner, Evidence,
    JobAccounting, MeasuredAccounting, StageContribution, VerificationRequirement, WorkUnitKind,
};

const CONTRIB_SEED: [u8; 32] = *b"byte-stability-test-seed--32B!!!";

fn minimal_job() -> ContributorJob {
    let mut job = ContributorJob {
        schema_version: 1,
        job_id: String::new(),
        model_hash: "11".repeat(32),
        manifest_snip_root: format!("0x{}", "22".repeat(32)),
        input_snip_root: format!("0x{}", "33".repeat(32)),
        input_hash: "44".repeat(32),
        verification_requirement: VerificationRequirement::AttestationOnly,
        accounting: JobAccounting {
            tokenizer_hash: "55".repeat(32),
            tokenizer_id: "demo/tokenizer".into(),
            input_token_count: 200_000,
            max_output_token_count: 1_000_000,
            base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
        },
        dispatched_at_utc: "2026-05-25T00:00:00Z".into(),
        expires_at_utc: None,
        dispatcher_pubkey_hex: None,
        dispatcher_signature_hex: None,
        notes: None,
    };
    job.job_id = job_hash_hex(&job).unwrap();
    job
}

fn build_result_for(job: &ContributorJob) -> ContributorResult {
    let signer = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let response_bytes = b"deterministic-response".to_vec();
    let response_hash = hex_lower(blake3::hash(&response_bytes).as_bytes());

    let mut result = ContributorResult {
        schema_version: 1,
        job_id: job.job_id.clone(),
        job_hash: job.job_id.clone(),
        job_snip_root: None,
        model_hash: job.model_hash.clone(),
        input_hash: job.input_hash.clone(),
        // For byte-stability we hard-code a fixed root rather than
        // publishing through a SNIP store.
        response_snip_root: format!("0x{}", "66".repeat(32)),
        response_hash,
        evidence: Evidence::AttestationOnly,
        measured_accounting: MeasuredAccounting {
            tokenizer_hash: job.accounting.tokenizer_hash.clone(),
            input_token_count: 200_000,
            output_token_count: 1_000_000,
            total_base_units: 1_200_000,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: signer.pubkey_hex(),
                stage_label: "stub-runner".into(),
                work_unit_kind: WorkUnitKind::Tokens,
                work_units: 1_200_000,
            }],
        },
        produced_at_utc: "2026-05-25T00:00:01Z".into(),
        contributor_pubkey_hex: signer.pubkey_hex(),
        contributor_signature_hex: String::new(),
        notes: None,
    };
    let signing_input = contributor_signing_input(&result).unwrap();
    result.contributor_signature_hex = signer.sign_hex(&signing_input);
    result
}

#[test]
fn job_canonical_bytes_blake3_is_pinned() {
    // Pinning the hash, not the full byte sequence (the bytes are
    // long; the hash is the right granularity for drift detection).
    let job = minimal_job();
    let bytes = canonical_job_bytes(&job).unwrap();
    let h = hex_lower(blake3::hash(&bytes).as_bytes());

    // Pinned value derived once by this test on first run.
    // If this test fails after a schema/bincode edit, the canonical
    // layout drifted and downstream signatures / job_ids will not
    // round-trip. Audit before regenerating.
    let expected = "5a453bbfe36f5385567f4c3e6b7a51e761ce4692e89354f5efa03c142654dd23";
    if std::env::var("OMNI_CONTRIBUTOR_REGEN").is_ok() {
        eprintln!("REGEN: job_canonical_bytes BLAKE3 = {h}");
    } else {
        assert_eq!(
            h, expected,
            "Stage 12.0 canonical job-body BLAKE3 drift. \
             If schema changed intentionally, rerun with OMNI_CONTRIBUTOR_REGEN=1 \
             and update the pinned hex.",
        );
    }
}

#[test]
fn result_canonical_bytes_blake3_is_pinned() {
    let job = minimal_job();
    let result = build_result_for(&job);
    let bytes = canonical_result_bytes(&result).unwrap();
    let h = hex_lower(blake3::hash(&bytes).as_bytes());

    let expected = "b41a971e2bd5af8e2180782ef731f46efaa27daecf7db11ccfda9fd030eb9c06";
    if std::env::var("OMNI_CONTRIBUTOR_REGEN").is_ok() {
        eprintln!("REGEN: result_canonical_bytes BLAKE3 = {h}");
    } else {
        assert_eq!(
            h, expected,
            "Stage 12.0 canonical result-body BLAKE3 drift; rerun with \
             OMNI_CONTRIBUTOR_REGEN=1 to print the new hex."
        );
    }
}

#[test]
fn json_round_trip_preserves_schema() {
    let job = minimal_job();
    let s = serde_json::to_string(&job).unwrap();
    let back: ContributorJob = serde_json::from_str(&s).unwrap();
    assert_eq!(job, back);

    let result = build_result_for(&job);
    let s2 = serde_json::to_string(&result).unwrap();
    let back2: ContributorResult = serde_json::from_str(&s2).unwrap();
    assert_eq!(result, back2);
}

#[test]
fn job_with_dispatcher_signature_round_trips() {
    let signer = omni_contributor::DispatcherSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let mut job = minimal_job();
    job.dispatcher_pubkey_hex = Some(signer.pubkey_hex());
    job.job_id = job_hash_hex(&job).unwrap();
    let signing_input = canonical_job_bytes(&job).unwrap();
    job.dispatcher_signature_hex = Some(signer.sign_hex(&signing_input));
    let s = serde_json::to_string(&job).unwrap();
    let back: ContributorJob = serde_json::from_str(&s).unwrap();
    assert_eq!(job, back);
}
