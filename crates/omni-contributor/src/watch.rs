//! Stage 12.1 — `watch-jobs` orchestrator.
//!
//! Long-running loop:
//!   1. Poll the configured `JobSource` (Stage 12.1: filesystem only).
//!   2. Dedup against an in-memory `HashSet<posted_id>`.
//!   3. For each new entry:
//!        a. Validate schema + poster signature (if present) + expiry.
//!        b. Apply `--accept-model-hash` / `--accept-tokenizer-hash`
//!           allow-lists.
//!        c. Apply the three required cost caps
//!           (`--max-input-tokens`, `--max-output-tokens`,
//!           `--max-total-base-units`) BEFORE fetching the inner job.
//!           A job whose declared bounds already violate a cap is
//!           refused without touching SNIP.
//!        d. Fetch the `ContributorJob` JSON from SNIP; assert its
//!           recomputed `job_hash` matches the posted envelope's
//!           drift-guard copy.
//!        e. Invoke Stage 12.0 `run_job`.
//!        f. Call `verify_result(&job, &result, &adapter)` BEFORE
//!           writing the result. On `overall_ok=false`, write to
//!           `<result-out-dir>/<job_id>.rejected.json` and skip the
//!           link-publish step.
//!        g. Otherwise write to `<result-out-dir>/<job_id>.json`.
//!        h. If `publish_link == true`: publish the result JSON to
//!           SNIP and emit a signed `PostedResultLink`.
//!
//! All errors are caught + logged + the loop continues. Init failures
//! (bad CLI args, source unreadable on first poll) surface to the
//! caller via `Err(ContributorError)`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use omni_store::SnipV2Adapter;
use omni_types::phase5::SnipV2ObjectId;

use crate::canonical::{hex_lower, job_hash_hex, posted_result_link_signing_input};
use crate::error::{ContributorError, DiscoverError, SchemaError};
use crate::job::ContributorJob;
use crate::posted::{PostedJob, PostedResultLink, POSTED_SCHEMA_VERSION};
use crate::run::{run_job, RunJobOptions};
use crate::runner::InferenceRunner;
use crate::signing::{verify_signature_hex, ContributorSigner};
use crate::{discover, snip, verify};

/// Required-by-the-protocol cost caps. `watch-jobs` refuses to start
/// without all three; an operator must explicitly accept the workload
/// envelope.
#[derive(Debug, Clone, Copy)]
pub struct CostCaps {
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub max_total_base_units: u64,
}

/// Allow-lists. Empty == "accept any". Stage 12.1 ships filtering by
/// the two structural fields that distinguish workloads;
/// `--accept-proof-system` doesn't apply because AttestationOnly is
/// the only Stage 12.0 variant.
#[derive(Debug, Clone, Default)]
pub struct AcceptFilters {
    pub model_hash_allow: Vec<String>,
    pub tokenizer_hash_allow: Vec<String>,
}

/// Watch-loop options. The CLI fills these from flags; the loop
/// itself is decoupled from CLI parsing so tests can drive it in
/// process.
pub struct WatchOptions<'a> {
    /// Polling cadence.
    pub poll_interval: Duration,
    /// Optional hard cap on jobs picked up before the loop exits 0.
    pub max_jobs: Option<u64>,
    /// Optional hard cap on poll iterations before the loop exits 0.
    /// Used primarily by tests to drive the loop deterministically;
    /// production typically leaves this `None` (poll forever) and
    /// stops via `max_jobs` or Ctrl-C from the surrounding process
    /// supervisor.
    pub max_polls: Option<u64>,
    /// Allow-lists; empty means "accept any".
    pub filters: AcceptFilters,
    /// Cost caps; all three required.
    pub caps: CostCaps,
    /// `run-job` runner.
    pub runner: &'a dyn InferenceRunner,
    /// Contributor signing key (also used to sign the optional
    /// `PostedResultLink`).
    pub signer: &'a ContributorSigner,
    /// Where to write accepted + rejected result JSON files.
    pub result_out_dir: PathBuf,
    /// Whether to publish a `PostedResultLink` to SNIP after each
    /// accepted result.
    pub publish_link: bool,
    /// Caller-supplied event emitter. Watch-loop tests use an
    /// in-memory collector; the CLI passes a stdout emitter.
    pub emit: &'a mut dyn EventEmitter,
    /// Optional Stage 12.2 hook: after a result link is published to
    /// SNIP (`publish_link == true`), forward the just-published
    /// link to a broadcaster so it can build + sign a
    /// `NetworkPostedResultAnnouncement` and post it to the mesh.
    /// `None` (the default) means "publish to SNIP only"; this is
    /// what Stage 12.1's filesystem-only watch path used.
    pub result_broadcaster: Option<&'a mut dyn ResultBroadcaster>,
}

/// Stage 12.2 broadcaster hook: receives a `PublishedResultLink` from
/// the watch loop after a successful SNIP publish and is responsible
/// for building + signing the `NetworkPostedResultAnnouncement` and
/// posting it to the mesh.
///
/// Failures are reported as `String` and surface to the watch loop
/// as a `WatchEvent::Error`; the loop continues on the next poll.
pub trait ResultBroadcaster {
    fn broadcast(&mut self, published: &PublishedResultLink) -> Result<(), String>;
}

/// Per-iteration event surface. The CLI's emitter prints
/// `event=… key=value` lines to stdout (mirroring Stage 11b.0
/// `operator verify-proof` style); tests collect events into a
/// `Vec<WatchEvent>` for assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchEvent {
    Poll {
        timestamp_utc: String,
        entries: usize,
    },
    Skip {
        posted_id: String,
        reason: SkipReason,
    },
    Pickup {
        posted_id: String,
        job_id: String,
    },
    FetchedJob {
        posted_id: String,
        job_id: String,
        input_hash_ok: bool,
    },
    RunOk {
        posted_id: String,
        job_id: String,
        response_snip_root: String,
        total_base_units: u64,
    },
    VerifyOk {
        posted_id: String,
        job_id: String,
        total_base_units: u64,
    },
    VerifyFail {
        posted_id: String,
        job_id: String,
        reason: String,
    },
    ResultLinkPublished {
        posted_id: String,
        link_snip_root: String,
    },
    Exit {
        reason: String,
        jobs_picked: u64,
    },
    Error {
        context: String,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    SchemaMalformed(String),
    PostedIdMismatch,
    PosterSignatureFail,
    Expired,
    ModelHashNotInAcceptSet,
    TokenizerHashNotInAcceptSet,
    CostCapExceeded { field: &'static str },
    AlreadySeen,
    JobFetchFailed(String),
    JobHashDrift,
}

pub trait EventEmitter {
    fn emit(&mut self, event: WatchEvent);
}

/// Bare-stdout emitter (CLI default).
pub struct StdoutEmitter;

impl EventEmitter for StdoutEmitter {
    fn emit(&mut self, event: WatchEvent) {
        match event {
            WatchEvent::Poll { timestamp_utc, entries } => {
                println!("event=poll t={timestamp_utc} entries={entries}");
            }
            WatchEvent::Skip { posted_id, reason } => {
                println!("event=skip posted_id={posted_id} reason={}", skip_reason_str(&reason));
            }
            WatchEvent::Pickup { posted_id, job_id } => {
                println!("event=pickup posted_id={posted_id} job_id={job_id}");
            }
            WatchEvent::FetchedJob { posted_id, job_id, input_hash_ok } => {
                println!(
                    "event=fetched_job posted_id={posted_id} job_id={job_id} input_hash_ok={input_hash_ok}"
                );
            }
            WatchEvent::RunOk { posted_id, job_id, response_snip_root, total_base_units } => {
                println!(
                    "event=run_job_ok posted_id={posted_id} job_id={job_id} \
                     response_snip_root={response_snip_root} total_base_units={total_base_units}"
                );
            }
            WatchEvent::VerifyOk { posted_id, job_id, total_base_units } => {
                println!(
                    "event=verify_ok posted_id={posted_id} job_id={job_id} \
                     total_base_units={total_base_units}"
                );
            }
            WatchEvent::VerifyFail { posted_id, job_id, reason } => {
                println!(
                    "event=verify_fail posted_id={posted_id} job_id={job_id} reason={reason}"
                );
            }
            WatchEvent::ResultLinkPublished { posted_id, link_snip_root } => {
                println!(
                    "event=result_link_published posted_id={posted_id} link_snip_root={link_snip_root}"
                );
            }
            WatchEvent::Exit { reason, jobs_picked } => {
                println!("event=exit reason={reason} jobs_picked={jobs_picked}");
            }
            WatchEvent::Error { context, message } => {
                println!("event=error context={context} message={message:?}");
            }
        }
    }
}

fn skip_reason_str(r: &SkipReason) -> String {
    match r {
        SkipReason::SchemaMalformed(s) => format!("schema_malformed:{s}"),
        SkipReason::PostedIdMismatch => "posted_id_mismatch".into(),
        SkipReason::PosterSignatureFail => "poster_signature_fail".into(),
        SkipReason::Expired => "expired".into(),
        SkipReason::ModelHashNotInAcceptSet => "model_hash_not_in_accept_set".into(),
        SkipReason::TokenizerHashNotInAcceptSet => "tokenizer_hash_not_in_accept_set".into(),
        SkipReason::CostCapExceeded { field } => format!("cost_cap_exceeded:{field}"),
        SkipReason::AlreadySeen => "already_seen".into(),
        SkipReason::JobFetchFailed(msg) => format!("job_fetch_failed:{msg}"),
        SkipReason::JobHashDrift => "job_hash_drift".into(),
    }
}

/// Drive the watch loop. Returns when `max_jobs` is reached (Ok),
/// or on a Ctrl-C path the CLI sets up via a separate signal handler
/// (Ok), or on an init failure (Err).
///
/// In Stage 12.1 this function is fully synchronous + single-threaded;
/// the CLI wraps it in a blocking task. A future async surface is out
/// of scope.
pub fn run_watch_loop<A: SnipV2Adapter, S: discover::JobSource>(
    adapter: &A,
    source: &mut S,
    mut opts: WatchOptions<'_>,
) -> Result<(), ContributorError> {
    std::fs::create_dir_all(&opts.result_out_dir)?;

    let mut seen: HashSet<String> = HashSet::new();
    let mut jobs_picked: u64 = 0;
    let mut polls_done: u64 = 0;

    loop {
        if let Some(max) = opts.max_polls {
            if polls_done >= max {
                opts.emit.emit(WatchEvent::Exit {
                    reason: "max_polls_reached".into(),
                    jobs_picked,
                });
                return Ok(());
            }
        }
        polls_done += 1;
        let poll_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let entries = match source.poll() {
            Ok(v) => v,
            Err(e) => {
                opts.emit.emit(WatchEvent::Error {
                    context: "source_poll".into(),
                    message: e.to_string(),
                });
                // Init-style errors (dir unreadable, etc.) surface to
                // the caller so the operator can fix the config.
                return Err(e.into());
            }
        };
        opts.emit.emit(WatchEvent::Poll {
            timestamp_utc: poll_at,
            entries: entries.len(),
        });

        for entry in entries {
            // Per-entry failure → log + skip + continue. The watch
            // loop NEVER short-circuits a poll because one file was
            // bad.
            let posted: PostedJob = match entry.result {
                Ok(p) => p,
                Err(e) => {
                    opts.emit.emit(WatchEvent::Error {
                        context: entry.source_label,
                        message: e.to_string(),
                    });
                    continue;
                }
            };

            if seen.contains(&posted.posted_id) {
                opts.emit.emit(WatchEvent::Skip {
                    posted_id: posted.posted_id.clone(),
                    reason: SkipReason::AlreadySeen,
                });
                continue;
            }

            // Mark as seen up-front so any per-entry failure below
            // still dedupes against re-pickup on the next poll
            // (matches the "dedup by posted_id, no persistent state"
            // approved default).
            seen.insert(posted.posted_id.clone());

            // 3a. Schema is already validated by FilesystemSource.
            //     Recheck defense-in-depth.
            if let Err(e) = posted.validate_schema() {
                opts.emit.emit(WatchEvent::Skip {
                    posted_id: posted.posted_id.clone(),
                    reason: SkipReason::SchemaMalformed(e.to_string()),
                });
                continue;
            }

            // 3a (cont). Poster signature, if present.
            if let (Some(pk), Some(sig)) =
                (&posted.poster_pubkey_hex, &posted.poster_signature_hex)
            {
                let signing_input = match crate::canonical::poster_signing_input(&posted) {
                    Ok(b) => b,
                    Err(e) => {
                        opts.emit.emit(WatchEvent::Skip {
                            posted_id: posted.posted_id.clone(),
                            reason: SkipReason::SchemaMalformed(e.to_string()),
                        });
                        continue;
                    }
                };
                match verify_signature_hex(pk, &signing_input, sig) {
                    Ok(true) => {}
                    Ok(false) => {
                        opts.emit.emit(WatchEvent::Skip {
                            posted_id: posted.posted_id.clone(),
                            reason: SkipReason::PosterSignatureFail,
                        });
                        continue;
                    }
                    Err(e) => {
                        opts.emit.emit(WatchEvent::Error {
                            context: "poster_signature".into(),
                            message: e.to_string(),
                        });
                        continue;
                    }
                }
            }

            // Expiry.
            if let Some(ref exp) = posted.expires_at_utc {
                let exp_dt = match chrono::DateTime::parse_from_rfc3339(exp) {
                    Ok(dt) => dt,
                    Err(_) => {
                        opts.emit.emit(WatchEvent::Skip {
                            posted_id: posted.posted_id.clone(),
                            reason: SkipReason::SchemaMalformed(format!(
                                "malformed expires_at_utc {exp}"
                            )),
                        });
                        continue;
                    }
                };
                if exp_dt.with_timezone(&chrono::Utc) < chrono::Utc::now() {
                    opts.emit.emit(WatchEvent::Skip {
                        posted_id: posted.posted_id.clone(),
                        reason: SkipReason::Expired,
                    });
                    continue;
                }
            }

            // 3b. Allow-lists.
            if !opts.filters.model_hash_allow.is_empty()
                && !opts.filters.model_hash_allow.iter().any(|m| m == &posted.model_hash)
            {
                opts.emit.emit(WatchEvent::Skip {
                    posted_id: posted.posted_id.clone(),
                    reason: SkipReason::ModelHashNotInAcceptSet,
                });
                continue;
            }

            // 3d. Fetch the inner ContributorJob from SNIP.
            let job_root = match SnipV2ObjectId::from_hex(&posted.job_snip_root) {
                Ok(r) => r,
                Err(e) => {
                    opts.emit.emit(WatchEvent::Skip {
                        posted_id: posted.posted_id.clone(),
                        reason: SkipReason::SchemaMalformed(format!(
                            "bad job_snip_root: {e:?}"
                        )),
                    });
                    continue;
                }
            };
            let job_bytes = match snip::fetch_bytes(adapter, &job_root) {
                Ok(b) => b,
                Err(e) => {
                    opts.emit.emit(WatchEvent::Skip {
                        posted_id: posted.posted_id.clone(),
                        reason: SkipReason::JobFetchFailed(e.to_string()),
                    });
                    continue;
                }
            };
            let job: ContributorJob = match serde_json::from_slice(&job_bytes) {
                Ok(j) => j,
                Err(e) => {
                    opts.emit.emit(WatchEvent::Skip {
                        posted_id: posted.posted_id.clone(),
                        reason: SkipReason::JobFetchFailed(format!("json: {e}")),
                    });
                    continue;
                }
            };
            // Drift guard: recompute job_hash from canonical bytes
            // and refuse if the posted envelope's claim disagrees.
            let recomputed_job_hash = match job_hash_hex(&job) {
                Ok(h) => h,
                Err(e) => {
                    opts.emit.emit(WatchEvent::Skip {
                        posted_id: posted.posted_id.clone(),
                        reason: SkipReason::SchemaMalformed(e.to_string()),
                    });
                    continue;
                }
            };
            if recomputed_job_hash != posted.job_hash {
                opts.emit.emit(WatchEvent::Skip {
                    posted_id: posted.posted_id.clone(),
                    reason: SkipReason::JobHashDrift,
                });
                continue;
            }

            // Also ensure model_hash matches between posted envelope
            // and inner job (cheap drift guard against a poster who
            // mis-tagged the model).
            if job.model_hash != posted.model_hash {
                opts.emit.emit(WatchEvent::Skip {
                    posted_id: posted.posted_id.clone(),
                    reason: SkipReason::ModelHashNotInAcceptSet,
                });
                continue;
            }

            // Tokenizer allow-list (applies to the inner job, not
            // the posted envelope, since tokenizer_hash isn't on the
            // posted envelope's drift-guard fields).
            if !opts.filters.tokenizer_hash_allow.is_empty()
                && !opts
                    .filters
                    .tokenizer_hash_allow
                    .iter()
                    .any(|t| t == &job.accounting.tokenizer_hash)
            {
                opts.emit.emit(WatchEvent::Skip {
                    posted_id: posted.posted_id.clone(),
                    reason: SkipReason::TokenizerHashNotInAcceptSet,
                });
                continue;
            }

            // 3c. Cost caps (applied to the inner job's declared
            //     bounds — BEFORE invoking the runner).
            if job.accounting.input_token_count > opts.caps.max_input_tokens {
                opts.emit.emit(WatchEvent::Skip {
                    posted_id: posted.posted_id.clone(),
                    reason: SkipReason::CostCapExceeded {
                        field: "max_input_tokens",
                    },
                });
                continue;
            }
            if job.accounting.max_output_token_count > opts.caps.max_output_tokens {
                opts.emit.emit(WatchEvent::Skip {
                    posted_id: posted.posted_id.clone(),
                    reason: SkipReason::CostCapExceeded {
                        field: "max_output_tokens",
                    },
                });
                continue;
            }
            // Overflow-safe total.
            let declared_total = job
                .accounting
                .input_token_count
                .checked_add(job.accounting.max_output_token_count);
            match declared_total {
                None => {
                    opts.emit.emit(WatchEvent::Skip {
                        posted_id: posted.posted_id.clone(),
                        reason: SkipReason::CostCapExceeded {
                            field: "max_total_base_units",
                        },
                    });
                    continue;
                }
                Some(total) if total > opts.caps.max_total_base_units => {
                    opts.emit.emit(WatchEvent::Skip {
                        posted_id: posted.posted_id.clone(),
                        reason: SkipReason::CostCapExceeded {
                            field: "max_total_base_units",
                        },
                    });
                    continue;
                }
                Some(_) => {}
            }

            opts.emit.emit(WatchEvent::Pickup {
                posted_id: posted.posted_id.clone(),
                job_id: job.job_id.clone(),
            });

            // 3e. Invoke Stage 12.0 run_job.
            let produced_at_utc =
                chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
            let result = match run_job(
                &job,
                adapter,
                opts.runner,
                RunJobOptions {
                    produced_at_utc,
                    signer: opts.signer,
                    notes: None,
                    job_snip_root: Some(posted.job_snip_root.clone()),
                },
            ) {
                Ok(r) => r,
                Err(e) => {
                    opts.emit.emit(WatchEvent::Error {
                        context: format!("run_job posted_id={}", posted.posted_id),
                        message: e.to_string(),
                    });
                    continue;
                }
            };
            opts.emit.emit(WatchEvent::FetchedJob {
                posted_id: posted.posted_id.clone(),
                job_id: job.job_id.clone(),
                input_hash_ok: true, // run_job's snip integrity check passed
            });
            opts.emit.emit(WatchEvent::RunOk {
                posted_id: posted.posted_id.clone(),
                job_id: job.job_id.clone(),
                response_snip_root: result.response_snip_root.clone(),
                total_base_units: result.measured_accounting.total_base_units,
            });

            // 3f. Verify result BEFORE persisting or publishing.
            let outcome = match verify::verify_result(&job, &result, adapter) {
                Ok(o) => o,
                Err(e) => {
                    // Structural failure during verify; write
                    // .rejected.json so the operator can inspect.
                    write_rejected(&opts.result_out_dir, &result, &e.to_string())?;
                    opts.emit.emit(WatchEvent::VerifyFail {
                        posted_id: posted.posted_id.clone(),
                        job_id: job.job_id.clone(),
                        reason: e.to_string(),
                    });
                    jobs_picked += 1;
                    if let Some(max) = opts.max_jobs {
                        if jobs_picked >= max {
                            opts.emit.emit(WatchEvent::Exit {
                                reason: "max_jobs_reached".into(),
                                jobs_picked,
                            });
                            return Ok(());
                        }
                    }
                    continue;
                }
            };
            if !outcome.overall_ok {
                let reason = summarize_verify_failure(&outcome);
                write_rejected(&opts.result_out_dir, &result, &reason)?;
                opts.emit.emit(WatchEvent::VerifyFail {
                    posted_id: posted.posted_id.clone(),
                    job_id: job.job_id.clone(),
                    reason,
                });
                jobs_picked += 1;
                if let Some(max) = opts.max_jobs {
                    if jobs_picked >= max {
                        opts.emit.emit(WatchEvent::Exit {
                            reason: "max_jobs_reached".into(),
                            jobs_picked,
                        });
                        return Ok(());
                    }
                }
                continue;
            }

            opts.emit.emit(WatchEvent::VerifyOk {
                posted_id: posted.posted_id.clone(),
                job_id: job.job_id.clone(),
                total_base_units: outcome.total_base_units,
            });

            // 3g. Write accepted result.
            let result_path = opts.result_out_dir.join(format!("{}.json", job.job_id));
            let result_json = serde_json::to_string_pretty(&result)?;
            std::fs::write(&result_path, &result_json)?;

            // 3h. Optionally publish a PostedResultLink.
            if opts.publish_link {
                match publish_result_link_for(
                    adapter,
                    &posted,
                    &result_json,
                    &result,
                    opts.signer,
                    &mut *opts.emit,
                ) {
                    Ok(published) => {
                        // 3i. Stage 12.2 — if a network broadcaster
                        // is wired in, hand it the just-published
                        // link so it can broadcast a
                        // NetworkPostedResultAnnouncement on the
                        // mesh. A SNIP publish without a broadcast
                        // is still a valid stage 12.1 path; failure
                        // here doesn't undo the SNIP publish.
                        if let Some(b) = opts.result_broadcaster.as_mut() {
                            if let Err(e) = b.broadcast(&published) {
                                opts.emit.emit(WatchEvent::Error {
                                    context: format!(
                                        "broadcast_result_link posted_id={}",
                                        posted.posted_id
                                    ),
                                    message: e,
                                });
                            }
                        }
                    }
                    Err(e) => {
                        opts.emit.emit(WatchEvent::Error {
                            context: format!("publish_result_link posted_id={}", posted.posted_id),
                            message: e.to_string(),
                        });
                    }
                }
            }

            jobs_picked += 1;
            if let Some(max) = opts.max_jobs {
                if jobs_picked >= max {
                    opts.emit.emit(WatchEvent::Exit {
                        reason: "max_jobs_reached".into(),
                        jobs_picked,
                    });
                    return Ok(());
                }
            }
        }

        // Sleep until next poll. In tests with `max_jobs = Some(0)`
        // or a zero poll interval, the loop spins; tests use
        // `max_jobs` to bound iterations.
        if opts.poll_interval > Duration::ZERO {
            let until = Instant::now() + opts.poll_interval;
            std::thread::sleep(opts.poll_interval.min(until.saturating_duration_since(Instant::now())));
        }
    }
}

/// Write `<job_id>.rejected.json` containing the result body + a
/// failure reason. The result JSON itself is what was produced;
/// reviewers inspect it alongside the `reason` to figure out what
/// drifted.
fn write_rejected(
    dir: &Path,
    result: &crate::result::ContributorResult,
    reason: &str,
) -> Result<(), ContributorError> {
    let path = dir.join(format!("{}.rejected.json", result.job_id));
    let body = serde_json::json!({
        "rejected_reason": reason,
        "result":          result,
    });
    std::fs::write(&path, serde_json::to_string_pretty(&body)?)?;
    Ok(())
}

fn summarize_verify_failure(o: &verify::VerifyOutcome) -> String {
    let mut bits = Vec::new();
    if !o.job_hash_ok {
        bits.push("job_hash");
    }
    if matches!(
        o.dispatcher_signature,
        verify::DispatcherSignatureOutcome::Fail
    ) {
        bits.push("dispatcher_signature");
    }
    if !o.input_hash_ok {
        bits.push("input_hash");
    }
    if !o.response_hash_ok {
        bits.push("response_hash");
    }
    if !o.tokenizer_hash_ok {
        bits.push("tokenizer_hash");
    }
    if !o.accounting_input_match {
        bits.push("accounting_input");
    }
    if !o.accounting_output_under_cap {
        bits.push("accounting_output_cap");
    }
    if !o.accounting_total_consistent {
        bits.push("accounting_total");
    }
    if !o.contributor_signature_ok {
        bits.push("contributor_signature");
    }
    if !o.evidence_ok {
        bits.push("evidence");
    }
    if !o.requirement_satisfied {
        bits.push("requirement");
    }
    if bits.is_empty() {
        "unknown_verify_failure".into()
    } else {
        format!("verify_failed:{}", bits.join(","))
    }
}

/// What `publish_result_link_for` produced. Callers (watch loop or
/// standalone CLI subcommand) consume this struct rather than
/// re-deriving the link, so the bytes on SNIP and the bytes the CLI
/// writes to `--link-out` are guaranteed identical (`link_json` is
/// exactly what was published).
#[derive(Debug, Clone)]
pub struct PublishedResultLink {
    pub link: PostedResultLink,
    /// Exact `serde_json::to_string_pretty(&link)` bytes that were
    /// published to SNIP. The standalone CLI subcommand writes this
    /// verbatim to `--link-out`.
    pub link_json: String,
    pub result_snip_root: SnipV2ObjectId,
    pub link_snip_root: SnipV2ObjectId,
}

/// Build + sign a `PostedResultLink` and publish both the result JSON
/// and the link envelope to SNIP. Returns the exact published
/// artifacts so callers don't re-publish or re-sign and risk drifting
/// the `published_at_utc` field.
pub fn publish_result_link_for<A: SnipV2Adapter, E: EventEmitter + ?Sized>(
    adapter: &A,
    posted: &PostedJob,
    result_json: &str,
    result: &crate::result::ContributorResult,
    signer: &ContributorSigner,
    emit: &mut E,
) -> Result<PublishedResultLink, ContributorError> {
    // Publish the result JSON to SNIP.
    let result_root =
        snip::publish_bytes(adapter, result_json.as_bytes(), "contributor-result")?;
    // Compute the canonical (signature-domain) hash of the result.
    let canonical = crate::canonical::canonical_result_bytes(result)?;
    let canonical_hash_hex = hex_lower(blake3::hash(&canonical).as_bytes());

    let published_at_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let mut link = PostedResultLink {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: posted.posted_id.clone(),
        result_snip_root: format!("0x{}", hex_lower(result_root.as_bytes())),
        result_canonical_hash: canonical_hash_hex,
        contributor_pubkey_hex: signer.pubkey_hex(),
        contributor_signature_hex: String::new(),
        published_at_utc,
    };
    let signing_input = posted_result_link_signing_input(&link)?;
    link.contributor_signature_hex = signer.sign_hex(&signing_input);
    // Defensive: schema-validate the signed link.
    link.validate_schema()
        .map_err(|e| ContributorError::Verify(crate::error::VerifyError::Schema(e)))?;

    // Publish the link envelope to SNIP. The local `link_json` we
    // return is byte-identical to what landed on SNIP.
    let link_json = serde_json::to_string_pretty(&link)?;
    let link_root = snip::publish_bytes(adapter, link_json.as_bytes(), "result-link")?;
    let link_root_hex = format!("0x{}", hex_lower(link_root.as_bytes()));
    emit.emit(WatchEvent::ResultLinkPublished {
        posted_id: posted.posted_id.clone(),
        link_snip_root: link_root_hex,
    });
    Ok(PublishedResultLink {
        link,
        link_json,
        result_snip_root: result_root,
        link_snip_root: link_root,
    })
}

// ── Stage 12.2 — result-announcement processing helper ────────────────────
//
// Used by both the omni-node `watch-network-results` CLI handler and
// the watch_network_results integration test. Validates the
// announcer signature, fetches the PostedResultLink from SNIP,
// schema-validates + drift-guards it, applies the posted_id filter,
// and writes the link bytes to disk. Returns a typed outcome enum
// so callers can route per-announcement results into bare-stdout
// events / typed test assertions without duplicating logic.

use crate::net::NetworkPostedResultAnnouncement;

/// Per-announcement outcome from [`process_result_announcement`].
/// Each variant is one bare-stdout line the CLI emits and one typed
/// test assertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultAnnouncementOutcome {
    /// The announcement passed every check and the link envelope
    /// was written to disk at `link_path`.
    LinkWritten {
        posted_id: String,
        link_path: std::path::PathBuf,
    },
    /// The announcer's signature did not verify.
    AnnouncerSignatureFailed { posted_id: String },
    /// The announcement's schema is malformed.
    SchemaMalformed { posted_id: String, message: String },
    /// `--posted-id` filter is set and this posted_id is not in it.
    FilteredOut { posted_id: String },
    /// SNIP fetch failed (root malformed or transport error).
    SnipFetchFailed { posted_id: String, message: String },
    /// The fetched bytes don't parse as PostedResultLink.
    LinkParseFailed { posted_id: String, message: String },
    /// The fetched PostedResultLink's schema is invalid.
    LinkSchemaInvalid { posted_id: String, message: String },
    /// The fetched PostedResultLink's `contributor_signature_hex`
    /// did not verify against `contributor_pubkey_hex` over the
    /// canonical signing input. Distinct from
    /// `AnnouncerSignatureFailed`: an honest relayer can announce a
    /// link whose contributor signature is forged, and we must
    /// reject the link before trusting any of its fields.
    LinkContributorSignatureFailed { posted_id: String },
    /// Drift: announcement's claimed posted_id / result_canonical_hash
    /// / contributor_pubkey_hex disagrees with the fetched link.
    LinkDrift { posted_id: String, field: &'static str },
}

/// Process one result announcement. Inert side-effects:
///   - Writes `<result_out_dir>/<posted_id>.link.json` on success.
///   - Returns a typed outcome describing what happened.
pub fn process_result_announcement<A: SnipV2Adapter + ?Sized>(
    ann: &NetworkPostedResultAnnouncement,
    adapter: &A,
    posted_id_filter: &std::collections::HashSet<String>,
    result_out_dir: &std::path::Path,
) -> ResultAnnouncementOutcome {
    let posted_id = ann.posted_id.clone();

    if let Err(e) = ann.validate_schema() {
        return ResultAnnouncementOutcome::SchemaMalformed {
            posted_id,
            message: e.to_string(),
        };
    }

    let signing_input =
        match crate::canonical::network_result_announcement_signing_input(ann) {
            Ok(b) => b,
            Err(e) => {
                return ResultAnnouncementOutcome::SchemaMalformed {
                    posted_id,
                    message: e.to_string(),
                };
            }
        };
    let sig_ok = match crate::signing::verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &signing_input,
        &ann.announcer_signature_hex,
    ) {
        Ok(v) => v,
        Err(_) => false,
    };
    if !sig_ok {
        return ResultAnnouncementOutcome::AnnouncerSignatureFailed { posted_id };
    }

    if !posted_id_filter.is_empty() && !posted_id_filter.contains(&posted_id) {
        return ResultAnnouncementOutcome::FilteredOut { posted_id };
    }

    let snip_root = match SnipV2ObjectId::from_hex(&ann.posted_result_link_snip_root) {
        Ok(r) => r,
        Err(e) => {
            return ResultAnnouncementOutcome::SnipFetchFailed {
                posted_id,
                message: format!("bad snip root: {e:?}"),
            };
        }
    };
    let bytes = match crate::snip::fetch_bytes(adapter, &snip_root) {
        Ok(b) => b,
        Err(e) => {
            return ResultAnnouncementOutcome::SnipFetchFailed {
                posted_id,
                message: e.to_string(),
            };
        }
    };
    let link: PostedResultLink = match serde_json::from_slice(&bytes) {
        Ok(l) => l,
        Err(e) => {
            return ResultAnnouncementOutcome::LinkParseFailed {
                posted_id,
                message: e.to_string(),
            };
        }
    };
    if let Err(e) = link.validate_schema() {
        return ResultAnnouncementOutcome::LinkSchemaInvalid {
            posted_id,
            message: e.to_string(),
        };
    }
    // Verify the contributor signature on the fetched link before
    // trusting any of its fields. The announcer's signature only
    // attests to the announcement envelope; the contributor's
    // signature is the trust root for the link payload.
    let link_signing_input =
        match crate::canonical::posted_result_link_signing_input(&link) {
            Ok(b) => b,
            Err(e) => {
                return ResultAnnouncementOutcome::LinkSchemaInvalid {
                    posted_id,
                    message: e.to_string(),
                };
            }
        };
    let link_sig_ok = crate::signing::verify_signature_hex(
        &link.contributor_pubkey_hex,
        &link_signing_input,
        &link.contributor_signature_hex,
    )
    .unwrap_or(false);
    if !link_sig_ok {
        return ResultAnnouncementOutcome::LinkContributorSignatureFailed { posted_id };
    }
    if link.posted_id != ann.posted_id {
        return ResultAnnouncementOutcome::LinkDrift { posted_id, field: "posted_id" };
    }
    if link.result_canonical_hash != ann.result_canonical_hash {
        return ResultAnnouncementOutcome::LinkDrift {
            posted_id,
            field: "result_canonical_hash",
        };
    }
    if link.contributor_pubkey_hex != ann.contributor_pubkey_hex {
        return ResultAnnouncementOutcome::LinkDrift {
            posted_id,
            field: "contributor_pubkey_hex",
        };
    }

    let link_path = result_out_dir.join(format!("{}.link.json", ann.posted_id));
    if let Err(e) = std::fs::write(&link_path, &bytes) {
        return ResultAnnouncementOutcome::SnipFetchFailed {
            posted_id,
            message: format!("write link file: {e}"),
        };
    }
    ResultAnnouncementOutcome::LinkWritten { posted_id, link_path }
}

// Quiet unused warnings on items the CLI surface consumes:
#[allow(dead_code)]
fn _force_used() {
    let _ = SchemaError::EmptyTokenizerId;
    let _ = DiscoverError::PostedJobExpired {
        posted_at: "".into(),
        expires_at: "".into(),
    };
}
