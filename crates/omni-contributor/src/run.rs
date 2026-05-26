//! Stage 12.0 — `run_job` orchestrator.
//!
//! Glues schema validation + SNIP fetch + runner invocation + result
//! construction + Ed25519 signing + SNIP publish of the response
//! bytes into one entry point. No chain wire involvement.

use std::io::Write;

use omni_store::SnipV2Adapter;
use omni_types::phase5::SnipV2ObjectId;

use crate::canonical::{
    canonical_job_bytes, contributor_signing_input, hex_lower, job_hash_hex,
};
use crate::error::ContributorError;
use crate::job::{ContributorJob, VerificationRequirement, SCHEMA_VERSION};
use crate::result::{
    ContributorResult, Evidence, MeasuredAccounting, StageContribution,
};
use crate::runner::InferenceRunner;
use crate::signing::ContributorSigner;
use crate::snip;

/// Options for [`run_job`]. Kept explicit so the CLI can fill them
/// from flags without surprising defaults.
pub struct RunJobOptions<'a> {
    /// Free-form `produced_at_utc` value (RFC 3339). Caller usually
    /// passes `chrono::Utc::now().to_rfc3339()`.
    pub produced_at_utc: String,

    /// Pre-loaded contributor signer.
    pub signer: &'a ContributorSigner,

    /// Optional `notes` field on the produced `ContributorResult`.
    pub notes: Option<String>,

    /// Whether to record the job's SNIP root on the result.
    /// Convenience for `verify-result` consumers; the verifier still
    /// requires `--job` explicitly.
    pub job_snip_root: Option<String>,
}

/// Full pickup-to-result orchestration.
pub fn run_job<A: SnipV2Adapter, R: InferenceRunner>(
    job: &ContributorJob,
    adapter: &A,
    runner: &R,
    opts: RunJobOptions<'_>,
) -> Result<ContributorResult, ContributorError> {
    // 1. Schema + identity + expiry checks. Refuses BEFORE we incur
    //    the SNIP fetch + runner cost on a stale or malformed job.
    //    Reuses the same VerifyError variants the verifier uses, so
    //    a run-job refusal here matches what verify-result would
    //    have produced.
    crate::verify::validate_job_consistency(job)?;

    // 2. Refuse production-proof requirements at parse time — Stage
    //    12.0 ships only AttestationOnly. The closed-enum match below
    //    is exhaustive over v1 variants; if a future
    //    schema_version: 2 variant lands without a corresponding
    //    producer path, this match is the compile-time forcing
    //    function.
    match job.verification_requirement {
        VerificationRequirement::AttestationOnly => {}
    }

    // 3. Verify dispatcher signature if present.
    if let (Some(pk), Some(sig)) =
        (&job.dispatcher_pubkey_hex, &job.dispatcher_signature_hex)
    {
        let signing_input = canonical_job_bytes(job)?;
        let ok = crate::signing::verify_signature_hex(pk, &signing_input, sig)?;
        if !ok {
            return Err(crate::error::VerifyError::DispatcherSignatureFailed.into());
        }
    }

    // 4. Fetch input bytes from SNIP and integrity-check.
    let input_snip_root = parse_snip_root_hex(&job.input_snip_root)?;
    let input_bytes = snip::fetch_bytes_with_integrity_check(
        adapter,
        &input_snip_root,
        &job.input_hash,
        "input",
    )?;

    // 5. Fetch manifest into a tempdir. The manifest is a SNIP object;
    //    we download to a single file. Multi-file manifests are out of
    //    scope for Stage 12.0 (the manifest tree retrieval would
    //    require additional `omni-store` primitives the contributor
    //    crate is not touching). For now the manifest is treated as a
    //    single blob and passed to the runner via its path.
    let manifest_snip_root = parse_snip_root_hex(&job.manifest_snip_root)?;
    let manifest_bytes = snip::fetch_bytes(adapter, &manifest_snip_root)?;
    let manifest_tmp = tempfile::Builder::new()
        .prefix("omni-contributor-manifest-")
        .tempfile()?;
    {
        let mut handle = std::fs::File::create(manifest_tmp.path())?;
        handle.write_all(&manifest_bytes)?;
        handle.flush()?;
    }

    // 6. Invoke the runner.
    let run_output = runner.run(manifest_tmp.path(), &input_bytes)?;

    // 7. Pre-publish validation. Refuses runner output that violates
    //    the job's declared input_token_count, exceeds
    //    max_output_token_count, overflows u64 on the total, or
    //    fails the structural stage_contributions check — BEFORE
    //    publishing the response to SNIP. Otherwise a buggy or
    //    malicious runner would orphan response bytes on SNIP for
    //    every refused result.
    let total_base_units = crate::verify::validate_runner_output_against_job(
        job,
        run_output.measured_input_tokens,
        run_output.measured_output_tokens,
    )?;
    if run_output.stage_contributions.is_empty() {
        return Err(crate::error::SchemaError::EmptyStageContributions.into());
    }
    // Validate each stage's pubkey hex shape via the same helper the
    // result-schema check uses. Catches an external runner that
    // emits malformed contributor_pubkey_hex before any SNIP write.
    for (i, sc) in run_output.stage_contributions.iter().enumerate() {
        sc.validate_schema(i).map_err(crate::error::ContributorError::Schema)?;
    }

    // 8. Publish response bytes to SNIP. Only reached AFTER all
    //    runner-output validation has passed.
    let (response_snip_root, response_hash_bytes) =
        snip::publish_bytes_with_hash(adapter, &run_output.response_bytes, "response")?;

    // 9. Build the result envelope (unsigned).

    let measured = MeasuredAccounting {
        tokenizer_hash: job.accounting.tokenizer_hash.clone(),
        input_token_count: run_output.measured_input_tokens,
        output_token_count: run_output.measured_output_tokens,
        total_base_units,
        stage_contributions: run_output.stage_contributions,
    };

    let response_snip_root_hex = format!("0x{}", hex_lower(response_snip_root.as_bytes()));

    let job_hash_value = job_hash_hex(job)?;
    let mut unsigned = ContributorResult {
        schema_version: SCHEMA_VERSION,
        job_id: job.job_id.clone(),
        job_hash: job_hash_value,
        job_snip_root: opts.job_snip_root,
        model_hash: job.model_hash.clone(),
        input_hash: job.input_hash.clone(),
        response_snip_root: response_snip_root_hex,
        response_hash: hex_lower(&response_hash_bytes),
        evidence: Evidence::AttestationOnly,
        measured_accounting: measured,
        produced_at_utc: opts.produced_at_utc,
        contributor_pubkey_hex: opts.signer.pubkey_hex(),
        // Placeholder filled in below — the signing input includes
        // every field EXCEPT this one.
        contributor_signature_hex: String::new(),
        notes: opts.notes,
    };

    // 10. Sign the canonical signing input (constructed from a copy
    //     where contributor_signature_hex is excluded — see
    //     canonical::ResultCanonicalBody).
    let signing_input = contributor_signing_input(&unsigned)?;
    let signature_hex = opts.signer.sign_hex(&signing_input);
    unsigned.contributor_signature_hex = signature_hex;

    // 11. Re-validate (catches programmer errors that desync field widths).
    unsigned.validate_schema()?;

    // Confirm that the just-built result's stage contributions
    // attribute work to the same pubkey the contributor signed with.
    // This is a soft-but-loud check — schema validates pubkey hex
    // shape per-stage; here we additionally require at least one
    // stage to match the top-level signer (multi-stage splits may
    // include other pubkeys too).
    let top = &unsigned.contributor_pubkey_hex;
    if !unsigned
        .measured_accounting
        .stage_contributions
        .iter()
        .any(|s| &s.contributor_pubkey_hex == top)
    {
        // Not a schema-level failure; warn via tracing and continue.
        // The verifier does not enforce this; it's a contributor-side
        // sanity heuristic.
        tracing::warn!(
            contributor_pubkey_hex = %top,
            "no stage_contributions entry matches contributor_pubkey_hex; \
             multi-contributor split or runner misconfiguration?"
        );
    }
    // (No-op, but keeps the variable used.)
    let _ = StageContribution::default_check();

    Ok(unsigned)
}

impl StageContribution {
    /// Tiny no-op kept as a compile-time anchor so future code that
    /// edits this struct will be reminded to update orchestrator
    /// assumptions. Returns `()`.
    fn default_check() {}
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn parse_snip_root_hex(s: &str) -> Result<SnipV2ObjectId, ContributorError> {
    SnipV2ObjectId::from_hex(s).map_err(|e| {
        crate::error::SchemaError::MalformedHash {
            field: "snip_root",
            got: format!("{s} ({e:?})"),
        }
        .into()
    })
}
