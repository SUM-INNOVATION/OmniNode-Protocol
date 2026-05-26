//! Stage 12.0 — `verify_result` off-chain verification pipeline.
//!
//! Implements the 10-step pipeline described in the Stage 12.0 plan:
//! schema validation → job_hash recompute → dispatcher signature
//! check → binding-field equality → input integrity → response
//! integrity → accounting checks → contributor signature check →
//! evidence-mode check → requirement satisfaction.
//!
//! No chain authority is consulted. Stage 11d.3 reframe posture
//! preserved: this is a local verifier policy gate that an operator
//! runs against a fetched (job, result) pair.

use omni_store::SnipV2Adapter;
use omni_types::phase5::SnipV2ObjectId;

use crate::canonical::{
    canonical_job_bytes, contributor_signing_input, hex_lower, job_hash_hex,
};
use crate::error::{SchemaError, VerifyError};
use crate::job::ContributorJob;
use crate::result::{ContributorResult, Evidence};
use crate::signing::verify_signature_hex;
use crate::snip;

/// Structured outcome of `verify_result`. Each field maps to one
/// bare-stdout line the CLI emits. `overall_ok = true` iff every
/// boolean below is `true` (treating `dispatcher_signature` =
/// `NotSigned` as a passing condition).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyOutcome {
    pub job_id: String,
    pub job_hash_ok: bool,
    pub dispatcher_signature: DispatcherSignatureOutcome,
    pub input_hash_ok: bool,
    pub response_hash_ok: bool,
    pub tokenizer_hash_ok: bool,
    pub accounting_input_match: bool,
    pub accounting_output_under_cap: bool,
    pub accounting_total_consistent: bool,
    pub total_base_units: u64,
    pub contributor_signature_ok: bool,
    pub evidence_mode: &'static str,
    pub evidence_ok: bool,
    pub requirement_satisfied: bool,
    pub overall_ok: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatcherSignatureOutcome {
    NotSigned,
    Ok,
    Fail,
}

/// Full verification pipeline. Returns `Ok(VerifyOutcome)` even when
/// individual checks fail (the outcome's booleans report which
/// fired); returns `Err(VerifyError)` only on hard structural errors
/// (schema malformed, SNIP fetch failure, signing-input encode
/// failure, etc.) — those genuinely prevent forming a verdict.
pub fn verify_result<A: SnipV2Adapter>(
    job: &ContributorJob,
    result: &ContributorResult,
    adapter: &A,
) -> Result<VerifyOutcome, VerifyError> {
    // ── Step 1: schema validation on both sides. ──
    job.validate_schema()?;
    result.validate_schema()?;

    // Sanity: job_id agreement.
    let derived_job_id = job_hash_hex(job).map_err(|e| {
        VerifyError::Schema(SchemaError::JobIdMismatch {
            job_id: job.job_id.clone(),
            derived: format!("<encode failure: {e}>"),
        })
    })?;
    if derived_job_id != job.job_id {
        return Err(VerifyError::Schema(SchemaError::JobIdMismatch {
            job_id: job.job_id.clone(),
            derived: derived_job_id,
        }));
    }

    // Expiry check.
    if let Some(ref exp) = job.expires_at_utc {
        let exp_dt = chrono::DateTime::parse_from_rfc3339(exp).map_err(|_| {
            VerifyError::Schema(SchemaError::MalformedTimestamp {
                field: "expires_at_utc",
                got: exp.clone(),
            })
        })?;
        let now = chrono::Utc::now();
        if exp_dt.with_timezone(&chrono::Utc) < now {
            return Err(VerifyError::JobExpired {
                expires_at: exp.clone(),
                now: now.to_rfc3339(),
            });
        }
    }

    // ── Step 2: recompute job_hash from the fetched job; assert
    // ──         result.job_hash == recomputed.
    let recomputed_job_hash = derived_job_id.clone();
    let job_hash_ok = result.job_hash == recomputed_job_hash;

    // ── Step 3: dispatcher signature (if present). ──
    let dispatcher_signature = match (
        job.dispatcher_pubkey_hex.as_deref(),
        job.dispatcher_signature_hex.as_deref(),
    ) {
        (Some(pk), Some(sig)) => {
            let signing_input = canonical_job_bytes(job).map_err(|e| {
                VerifyError::Schema(SchemaError::MalformedHash {
                    field: "canonical_job_bytes",
                    got: e.to_string(),
                })
            })?;
            let ok = verify_signature_hex(pk, &signing_input, sig).map_err(|e| {
                VerifyError::Schema(SchemaError::MalformedSignature {
                    field: "dispatcher_signature_hex",
                    got: e.to_string(),
                })
            })?;
            if ok {
                DispatcherSignatureOutcome::Ok
            } else {
                DispatcherSignatureOutcome::Fail
            }
        }
        (None, None) => DispatcherSignatureOutcome::NotSigned,
        _ => {
            return Err(VerifyError::Schema(
                SchemaError::InconsistentDispatcherIdentity {
                    pubkey_set: if job.dispatcher_pubkey_hex.is_some() { "Some" } else { "None" },
                    signature_set: if job.dispatcher_signature_hex.is_some() { "Some" } else { "None" },
                },
            ));
        }
    };

    // ── Step 4: binding-field equality between job and result. ──
    if result.model_hash != job.model_hash {
        return Err(VerifyError::ModelHashMismatch {
            job_model_hash: job.model_hash.clone(),
            result_model_hash: result.model_hash.clone(),
        });
    }
    if result.input_hash != job.input_hash {
        return Err(VerifyError::InputHashMismatch { field: "job_vs_result" });
    }

    // ── Step 5: input integrity. Fetch input bytes from SNIP and
    // ──         BLAKE3-check against job.input_hash. ──
    let input_root = parse_snip_root(&job.input_snip_root, "input_snip_root")?;
    let input_bytes = snip::fetch_bytes(adapter, &input_root).map_err(|e| {
        VerifyError::Schema(SchemaError::MalformedHash {
            field: "input_snip_root",
            got: format!("snip fetch failed: {e}"),
        })
    })?;
    let input_hash_actual = hex_lower(blake3::hash(&input_bytes).as_bytes());
    let input_hash_ok = input_hash_actual == job.input_hash;

    // ── Step 6: response integrity. ──
    let response_root = parse_snip_root(&result.response_snip_root, "response_snip_root")?;
    let response_bytes = snip::fetch_bytes(adapter, &response_root).map_err(|e| {
        VerifyError::Schema(SchemaError::MalformedHash {
            field: "response_snip_root",
            got: format!("snip fetch failed: {e}"),
        })
    })?;
    let response_hash_actual = hex_lower(blake3::hash(&response_bytes).as_bytes());
    let response_hash_ok = response_hash_actual == result.response_hash;

    // ── Step 7: accounting. ──
    let tokenizer_hash_ok =
        result.measured_accounting.tokenizer_hash == job.accounting.tokenizer_hash;
    let accounting_input_match = result.measured_accounting.input_token_count
        == job.accounting.input_token_count;
    let accounting_output_under_cap = result.measured_accounting.output_token_count
        <= job.accounting.max_output_token_count;
    let expected_total = result
        .measured_accounting
        .input_token_count
        .checked_add(result.measured_accounting.output_token_count)
        .ok_or(VerifyError::AccountingTotalInconsistent {
            total: result.measured_accounting.total_base_units,
            input: result.measured_accounting.input_token_count,
            output: result.measured_accounting.output_token_count,
        })?;
    let accounting_total_consistent =
        result.measured_accounting.total_base_units == expected_total;

    // ── Step 8: contributor signature. ──
    let signing_input = contributor_signing_input(result).map_err(|e| {
        VerifyError::Schema(SchemaError::MalformedHash {
            field: "contributor_signing_input",
            got: e.to_string(),
        })
    })?;
    let contributor_signature_ok = verify_signature_hex(
        &result.contributor_pubkey_hex,
        &signing_input,
        &result.contributor_signature_hex,
    )
    .map_err(|e| {
        VerifyError::Schema(SchemaError::MalformedSignature {
            field: "contributor_signature_hex",
            got: e.to_string(),
        })
    })?;

    // ── Step 9: evidence-mode-specific check. ──
    let (evidence_mode, evidence_ok) = match result.evidence {
        Evidence::AttestationOnly => {
            // Signature already verified at step 8; evidence_ok
            // mirrors contributor_signature_ok.
            ("attestation_only", contributor_signature_ok)
        }
    };

    // ── Step 10: requirement satisfaction. ──
    let requirement_satisfied = matches!(
        (
            &job.verification_requirement,
            &result.evidence,
        ),
        (
            crate::job::VerificationRequirement::AttestationOnly,
            Evidence::AttestationOnly,
        )
    ) && evidence_ok;

    // ── Overall ──
    let dispatcher_ok = !matches!(dispatcher_signature, DispatcherSignatureOutcome::Fail);
    let overall_ok = job_hash_ok
        && dispatcher_ok
        && input_hash_ok
        && response_hash_ok
        && tokenizer_hash_ok
        && accounting_input_match
        && accounting_output_under_cap
        && accounting_total_consistent
        && contributor_signature_ok
        && evidence_ok
        && requirement_satisfied;

    Ok(VerifyOutcome {
        job_id: job.job_id.clone(),
        job_hash_ok,
        dispatcher_signature,
        input_hash_ok,
        response_hash_ok,
        tokenizer_hash_ok,
        accounting_input_match,
        accounting_output_under_cap,
        accounting_total_consistent,
        total_base_units: result.measured_accounting.total_base_units,
        contributor_signature_ok,
        evidence_mode,
        evidence_ok,
        requirement_satisfied,
        overall_ok,
    })
}

fn parse_snip_root(s: &str, field: &'static str) -> Result<SnipV2ObjectId, VerifyError> {
    SnipV2ObjectId::from_hex(s).map_err(|e| {
        VerifyError::Schema(SchemaError::MalformedHash {
            field,
            got: format!("{s} ({e:?})"),
        })
    })
}

// ── Pre-signing checks (also reused by run_job) ───────────────────────────
//
// The orchestrator (`run::run_job`) must apply these checks BEFORE
// signing a result the verifier will later reject. Implementing them
// here keeps the orchestrator and the verifier consistent — both use
// the same VerifyError variants — and lets tests assert refusal
// happens at the right point in the pipeline.

/// Schema-level + identity + expiry checks on a `ContributorJob`.
/// Does NOT touch the SNIP store or the result envelope. Suitable
/// for either run-job (pre-signing) or verify-result (mid-pipeline)
/// reuse.
pub fn validate_job_consistency(job: &ContributorJob) -> Result<(), VerifyError> {
    job.validate_schema()?;
    let derived = job_hash_hex(job).map_err(|e| {
        VerifyError::Schema(SchemaError::JobIdMismatch {
            job_id: job.job_id.clone(),
            derived: format!("<encode failure: {e}>"),
        })
    })?;
    if derived != job.job_id {
        return Err(VerifyError::Schema(SchemaError::JobIdMismatch {
            job_id: job.job_id.clone(),
            derived,
        }));
    }
    if let Some(ref exp) = job.expires_at_utc {
        let exp_dt = chrono::DateTime::parse_from_rfc3339(exp).map_err(|_| {
            VerifyError::Schema(SchemaError::MalformedTimestamp {
                field: "expires_at_utc",
                got: exp.clone(),
            })
        })?;
        let now = chrono::Utc::now();
        if exp_dt.with_timezone(&chrono::Utc) < now {
            return Err(VerifyError::JobExpired {
                expires_at: exp.clone(),
                now: now.to_rfc3339(),
            });
        }
    }
    Ok(())
}

/// Validates a runner's measured token counts against the job's
/// accounting bounds + checks for u64 overflow on the total. Returns
/// the safely-computed `total_base_units` on success. Run-job uses
/// this BEFORE signing so a runner that violates the job's accounting
/// is refused with a typed error, not by producing a result the
/// verifier later rejects.
pub fn validate_runner_output_against_job(
    job: &ContributorJob,
    measured_input_tokens: u64,
    measured_output_tokens: u64,
) -> Result<u64, VerifyError> {
    if measured_input_tokens != job.accounting.input_token_count {
        return Err(VerifyError::AccountingInputMismatch {
            job_count: job.accounting.input_token_count,
            result_count: measured_input_tokens,
        });
    }
    if measured_output_tokens > job.accounting.max_output_token_count {
        return Err(VerifyError::AccountingOutputOverCap {
            output: measured_output_tokens,
            max: job.accounting.max_output_token_count,
        });
    }
    measured_input_tokens
        .checked_add(measured_output_tokens)
        .ok_or(VerifyError::AccountingTotalOverflow {
            input: measured_input_tokens,
            output: measured_output_tokens,
        })
}
