//! Stage 12.0 — `run_job` orchestrator.
//!
//! Glues schema validation + SNIP fetch + runner invocation + result
//! construction + Ed25519 signing + SNIP publish of the response
//! bytes into one entry point. No chain wire involvement.
//!
//! Issue #149: the response is published idempotently
//! ([`crate::snip::publish_file_idempotent`]) and the signed result is
//! built only once the chain records it Active and it was fetched back.
//! [`run_job_checkpointed`] adds a durable checkpoint
//! ([`crate::checkpoint`]) written before the first registration, so a
//! restarted run resumes from the original response instead of running
//! the job again.

use std::io::Write;
use std::path::Path;

use omni_store::SnipV2Adapter;
use omni_types::phase5::SnipV2ObjectId;

use crate::checkpoint::{
    CheckpointError, CheckpointPublication, LoadedCheckpoint, Preparation, RunCheckpointStore,
    RunPhase,
};

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
pub fn run_job<A: SnipV2Adapter + ?Sized, R: InferenceRunner + ?Sized>(
    job: &ContributorJob,
    adapter: &A,
    runner: &R,
    opts: RunJobOptions<'_>,
) -> Result<ContributorResult, ContributorError> {
    run_job_with_publication(job, adapter, runner, opts).map(|(result, _)| result)
}

/// [`run_job`], also returning how the response bytes were published
/// (issue #149): ingested now, already Active (an earlier run, or another
/// job with identical response bytes), or this account's Pending
/// registration resumed. No checkpoint: an interrupted run starts again
/// from the runner; use [`run_job_checkpointed`] to resume instead.
pub fn run_job_with_publication<A: SnipV2Adapter + ?Sized, R: InferenceRunner + ?Sized>(
    job: &ContributorJob,
    adapter: &A,
    runner: &R,
    opts: RunJobOptions<'_>,
) -> Result<(ContributorResult, snip::PublishedBytes), ContributorError> {
    check_job(job)?;
    let (response, measured) = compute(job, adapter, runner)?;

    // 8. Publish response bytes to SNIP. Only reached AFTER all
    //    runner-output validation has passed. Idempotent and retry-safe;
    //    returns only once the chain records the response Active and it
    //    was fetched back. Nothing below exists unless it did.
    let published = snip::publish_bytes_idempotent(adapter, &response, "response")?;
    log_publication(&published);

    let result = sign_result(
        job,
        &published.root,
        &published.hash,
        measured,
        opts.produced_at_utc,
        opts.notes,
        opts.job_snip_root,
        opts.signer,
    )?;
    Ok((result, published))
}

/// What [`run_job_checkpointed`] produced.
#[derive(Debug)]
pub struct CheckpointedRun {
    /// The signed result. Identical on every attempt for the same
    /// checkpoint.
    pub result: ContributorResult,
    /// The exact bytes to write as the result file
    /// (`serde_json::to_string_pretty(&result)`).
    pub result_json: String,
    /// Phase the checkpoint was in when this attempt started; `None` when
    /// this attempt created it (and ran the job).
    pub resumed_from: Option<RunPhase>,
    /// How the response became Active: this attempt's publication, or the
    /// one recorded by an earlier attempt.
    pub publication: CheckpointPublication,
    /// The checkpoint, at phase `active` or `result_written`. Hand it to
    /// [`RunCheckpointStore::write_result`].
    pub checkpoint: LoadedCheckpoint,
}

impl CheckpointedRun {
    /// The result file was already written by an earlier attempt.
    pub fn completed(&self) -> bool {
        self.resumed_from == Some(RunPhase::ResultWritten)
    }
}

/// [`run_job`] with a durable checkpoint in `store` (issue #149).
///
/// With no checkpoint for (job, contributor): validate, fetch, run the
/// runner, validate its output, and durably record the response and the
/// result inputs **before anything is registered**; then publish. With a
/// checkpoint: never run the job again — publish (or confirm) from the
/// stored original response, or, from phase `active` on, only sign. The
/// caller writes the result with [`RunCheckpointStore::write_result`],
/// which completes the checkpoint.
///
/// `opts.notes` and `opts.job_snip_root` must equal what the checkpoint
/// recorded; `opts.produced_at_utc` is ignored for an existing checkpoint
/// (the recorded one keeps the result identical across attempts).
pub fn run_job_checkpointed<A: SnipV2Adapter + ?Sized, R: InferenceRunner + ?Sized>(
    job: &ContributorJob,
    adapter: &A,
    runner: &R,
    opts: RunJobOptions<'_>,
    store: &RunCheckpointStore,
) -> Result<CheckpointedRun, ContributorError> {
    check_job(job)?;
    let job_hash = job_hash_hex(job)?;
    let contributor = opts.signer.pubkey_hex();

    let (mut cp, resumed_from) = match store.load(&job.job_id, &contributor)? {
        Some(cp) => {
            let recorded = |what: &str, want: Option<&str>, got: Option<&str>| {
                if want == got {
                    Ok(())
                } else {
                    Err(CheckpointError::Mismatch {
                        path: cp.path(),
                        detail: format!(
                            "it records {what} {got:?}, this run asks for {want:?}; \
                             rerun with the recorded value or remove the checkpoint"
                        ),
                    })
                }
            };
            if cp.checkpoint.job_hash != job_hash {
                return Err(CheckpointError::Mismatch {
                    path: cp.path(),
                    detail: format!("job_hash {} != {job_hash}", cp.checkpoint.job_hash),
                }
                .into());
            }
            recorded("notes", opts.notes.as_deref(), cp.checkpoint.notes.as_deref())?;
            recorded(
                "job_snip_root",
                opts.job_snip_root.as_deref(),
                cp.checkpoint.job_snip_root.as_deref(),
            )?;
            // The stored accounting must still satisfy the job.
            let m = &cp.checkpoint.measured_accounting;
            let total = crate::verify::validate_runner_output_against_job(
                job,
                m.input_token_count,
                m.output_token_count,
            )?;
            if total != m.total_base_units {
                return Err(CheckpointError::Mismatch {
                    path: cp.path(),
                    detail: format!("total_base_units {} != {total}", m.total_base_units),
                }
                .into());
            }
            let phase = cp.checkpoint.phase;
            tracing::info!(
                job_id = %job.job_id,
                %phase,
                checkpoint = %cp.dir.display(),
                "resuming from the run checkpoint; the job is not run again"
            );
            (cp, Some(phase))
        }
        None => {
            let (response, measured) = compute(job, adapter, runner)?;
            let cp = store.prepare(&Preparation {
                job_id: &job.job_id,
                job_hash: &job_hash,
                contributor_pubkey_hex: &contributor,
                response: &response,
                measured_accounting: &measured,
                produced_at_utc: &opts.produced_at_utc,
                notes: opts.notes.as_deref(),
                job_snip_root: opts.job_snip_root.as_deref(),
            })?;
            (cp, None)
        }
    };
    // Every attempt checks the stored original against its recorded
    // identity before using it.
    let response = cp.verified_response()?;

    let publication = if cp.checkpoint.phase < RunPhase::Active {
        store.advance(&mut cp, RunPhase::Publishing, None, None)?;
        let published = snip::publish_file_idempotent(adapter, &cp.response_path())?;
        if published.root.to_hex() != cp.checkpoint.response_snip_root {
            return Err(CheckpointError::ResponseDamaged {
                path: cp.path(),
                detail: format!(
                    "published root {} != recorded {}",
                    published.root, cp.checkpoint.response_snip_root
                ),
            }
            .into());
        }
        log_publication(&published);
        let publication = CheckpointPublication {
            action: published.action.as_str().into(),
            owner: published.owner.clone(),
        };
        store.advance(&mut cp, RunPhase::Active, Some(publication.clone()), None)?;
        publication
    } else {
        cp.checkpoint
            .publication
            .clone()
            .ok_or_else(|| CheckpointError::Malformed {
                path: cp.path(),
                detail: format!("phase {} without a publication record", cp.checkpoint.phase),
            })?
    };

    let root = parse_snip_root_hex(&cp.checkpoint.response_snip_root)?;
    let c = &cp.checkpoint;
    let result = sign_result(
        job,
        &root,
        blake3::hash(&response).as_bytes(),
        c.measured_accounting.clone(),
        c.produced_at_utc.clone(),
        c.notes.clone(),
        c.job_snip_root.clone(),
        opts.signer,
    )?;
    let result_json = serde_json::to_string_pretty(&result)?;
    if let Some(recorded) = &c.result_hash {
        let now = hex_lower(blake3::hash(result_json.as_bytes()).as_bytes());
        if *recorded != now {
            return Err(CheckpointError::Mismatch {
                path: cp.path(),
                detail: format!("the recorded result hash {recorded} != the re-signed {now}"),
            }
            .into());
        }
    }
    Ok(CheckpointedRun {
        result,
        result_json,
        resumed_from,
        publication,
        checkpoint: cp,
    })
}

/// [`run_job_checkpointed`] with the default store beside `out`, then the
/// durable, idempotent result write to `out`. What `run-job` does.
pub fn run_job_to_file<A: SnipV2Adapter + ?Sized, R: InferenceRunner + ?Sized>(
    job: &ContributorJob,
    adapter: &A,
    runner: &R,
    opts: RunJobOptions<'_>,
    store: &RunCheckpointStore,
    out: &Path,
) -> Result<CheckpointedRun, ContributorError> {
    let mut run = run_job_checkpointed(job, adapter, runner, opts, store)?;
    store.write_result(&mut run.checkpoint, out, run.result_json.as_bytes())?;
    Ok(run)
}

fn log_publication(published: &snip::PublishedBytes) {
    tracing::info!(
        response_snip_root = %published.root,
        publication = %published.action,
        owner = %published.owner,
        "response bytes are Active on SNIP and fetched back"
    );
}

/// Steps 1–3: schema, identity, expiry, requirement, dispatcher signature.
fn check_job(job: &ContributorJob) -> Result<(), ContributorError> {
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
    Ok(())
}

/// Steps 4–7: fetch the input and the manifest, run the runner, and
/// validate its output. Returns the response bytes and the accounting.
fn compute<A: SnipV2Adapter + ?Sized, R: InferenceRunner + ?Sized>(
    job: &ContributorJob,
    adapter: &A,
    runner: &R,
) -> Result<(Vec<u8>, MeasuredAccounting), ContributorError> {
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

    let measured = MeasuredAccounting {
        tokenizer_hash: job.accounting.tokenizer_hash.clone(),
        input_token_count: run_output.measured_input_tokens,
        output_token_count: run_output.measured_output_tokens,
        total_base_units,
        stage_contributions: run_output.stage_contributions,
    };
    Ok((run_output.response_bytes, measured))
}

/// Steps 9–11: build, sign and re-validate the result for a response that
/// is Active on SNIP.
#[allow(clippy::too_many_arguments)]
fn sign_result(
    job: &ContributorJob,
    response_snip_root: &SnipV2ObjectId,
    response_hash: &[u8; 32],
    measured: MeasuredAccounting,
    produced_at_utc: String,
    notes: Option<String>,
    job_snip_root: Option<String>,
    signer: &ContributorSigner,
) -> Result<ContributorResult, ContributorError> {
    // 9. Build the result envelope (unsigned).
    let response_snip_root_hex = format!("0x{}", hex_lower(response_snip_root.as_bytes()));

    let job_hash_value = job_hash_hex(job)?;
    let mut unsigned = ContributorResult {
        schema_version: SCHEMA_VERSION,
        job_id: job.job_id.clone(),
        job_hash: job_hash_value,
        job_snip_root,
        model_hash: job.model_hash.clone(),
        input_hash: job.input_hash.clone(),
        response_snip_root: response_snip_root_hex,
        response_hash: hex_lower(response_hash),
        evidence: Evidence::AttestationOnly,
        measured_accounting: measured,
        produced_at_utc,
        contributor_pubkey_hex: signer.pubkey_hex(),
        // Placeholder filled in below — the signing input includes
        // every field EXCEPT this one.
        contributor_signature_hex: String::new(),
        notes,
    };

    // 10. Sign the canonical signing input (constructed from a copy
    //     where contributor_signature_hex is excluded — see
    //     canonical::ResultCanonicalBody).
    let signing_input = contributor_signing_input(&unsigned)?;
    let signature_hex = signer.sign_hex(&signing_input);
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
    StageContribution::default_check();

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
