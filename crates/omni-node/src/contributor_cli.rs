//! Stage 12.0 — `omni-node operator contributor` CLI subcommand tree.
//!
//! Three subcommands:
//!   - `validate-job`   — read-only schema + dispatcher-signature check.
//!   - `run-job`        — full pickup-to-result orchestration.
//!   - `verify-result`  — off-chain verification pipeline.
//!
//! Output is bare-stdout key=value lines (same convention as
//! `operator verify-proof`). No new chain-authority key names are
//! introduced — the Stage 11d.3 reframe posture is preserved.

use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};

use omni_contributor::{
    canonical::job_hash_hex, run_job, verify::DispatcherSignatureOutcome, verify_result,
    ContributorJob, ContributorResult, ContributorSigner, ExternalCommandRunner, InferenceRunner,
    RunJobOptions, StubRunner, WorkUnitKind,
};
use omni_store::{SnipV2Cli, SnipV2CliConfig};

#[derive(Args)]
pub struct ContributorArgs {
    #[command(subcommand)]
    cmd: ContributorCmd,
}

#[derive(Subcommand)]
enum ContributorCmd {
    /// Read-only: parse + schema-check a `ContributorJob`, verify the
    /// dispatcher signature if present, and probe SNIP-root
    /// resolvability.
    ValidateJob(ValidateJobArgs),

    /// Full pickup-to-result orchestration: validate → fetch from SNIP
    /// → run inference via the chosen runner → sign + publish result.
    RunJob(RunJobArgs),

    /// Off-chain verification of a `(job, result)` pair. The verifier
    /// requires both `--job` and `--result` explicitly; the optional
    /// `job_snip_root` field in the result is convenience-only and
    /// NOT implicitly trusted.
    VerifyResult(VerifyResultArgs),

    /// Stage 12.1 — publish a `ContributorJob` JSON to SNIP and write
    /// a `PostedJob` JSON file that a contributor's `watch-jobs`
    /// loop will pick up. Dispatcher-side.
    PostJob(PostJobArgs),

    /// Stage 12.1 — long-running: watch a directory for `PostedJob`
    /// envelopes, fetch each job from SNIP, apply filters + cost caps,
    /// run inference, verify the result, write to `--result-out-dir`,
    /// and optionally publish a `PostedResultLink` to SNIP.
    WatchJobs(WatchJobsArgs),

    /// Stage 12.1 — publish an existing `ContributorResult` JSON to
    /// SNIP and emit a signed `PostedResultLink` envelope.
    PublishResultLink(PublishResultLinkArgs),
}

// ── validate-job ──────────────────────────────────────────────────────────

#[derive(Args)]
struct ValidateJobArgs {
    /// Path to a `ContributorJob` JSON file.
    #[arg(long)]
    job: PathBuf,

    /// Optional `sum-node` binary path (defaults to PATH-resolved
    /// `sum-node`). Used to probe SNIP-root resolvability; if the
    /// binary is missing the validator reports the SNIP probes as
    /// unavailable but still completes the schema check.
    #[arg(long)]
    snip_binary: Option<PathBuf>,

    /// Optional SNIP seed file path passed through to `sum-node`.
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── run-job ───────────────────────────────────────────────────────────────

#[derive(Args)]
struct RunJobArgs {
    #[arg(long)]
    job: PathBuf,

    /// Output path for the produced `ContributorResult` JSON.
    #[arg(long)]
    out: PathBuf,

    /// Which inference runner to use.
    #[arg(long, value_enum, default_value_t = RunnerChoice::Stub)]
    runner: RunnerChoice,

    /// `external` runner: path to the external command binary.
    #[arg(long, required_if_eq("runner", "external"))]
    external_command: Option<PathBuf>,

    /// `external` runner: extra args prepended to `--manifest` / `--input`.
    #[arg(long = "external-arg")]
    external_args: Vec<String>,

    /// `external` runner: env vars to forward from the surrounding
    /// shell (otherwise the subprocess sees an empty environment).
    #[arg(long = "external-env-allow")]
    external_env_allow: Vec<String>,

    /// `stub` runner: path to the fixed response bytes file.
    #[arg(long, required_if_eq("runner", "stub"))]
    stub_response: Option<PathBuf>,

    /// `stub` runner: measured input token count to report.
    #[arg(long, default_value_t = 0)]
    stub_input_tokens: u64,

    /// `stub` runner: measured output token count to report.
    #[arg(long, default_value_t = 0)]
    stub_output_tokens: u64,

    /// Path to a 32-byte raw contributor seed file. Each contributor
    /// must hold a seed distinct from any chain-attestation seed.
    #[arg(long)]
    seed_file: PathBuf,

    /// Optional free-form notes recorded on the produced result.
    #[arg(long)]
    notes: Option<String>,

    /// Optional SNIP-root convenience field on the produced result.
    #[arg(long)]
    job_snip_root: Option<String>,

    /// `sum-node` binary path.
    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    /// Optional SNIP seed file.
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum RunnerChoice {
    External,
    Stub,
}

// ── verify-result ─────────────────────────────────────────────────────────

#[derive(Args)]
struct VerifyResultArgs {
    /// Path to the `ContributorJob` JSON.
    #[arg(long)]
    job: PathBuf,

    /// Path to the `ContributorResult` JSON.
    #[arg(long)]
    result: PathBuf,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Stage 12.1: post-job ──────────────────────────────────────────────────

#[derive(Args)]
struct PostJobArgs {
    /// Path to the `ContributorJob` JSON to publish.
    #[arg(long)]
    job: PathBuf,

    /// Output path for the produced `PostedJob` JSON envelope.
    #[arg(long)]
    posted_out: PathBuf,

    /// Optional 32-byte raw seed file for the **poster**. When set,
    /// `PostedJob.poster_pubkey_hex` + `poster_signature_hex` are
    /// populated; when absent, both stay None (unsigned posting).
    #[arg(long)]
    seed_file: Option<PathBuf>,

    /// Optional RFC 3339 UTC expiry (`Z` suffix). When set,
    /// `watch-jobs` skips the posting after this instant.
    #[arg(long)]
    expires_at_utc: Option<String>,

    /// Optional free-form audit string recorded on the envelope.
    #[arg(long)]
    notes: Option<String>,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Stage 12.1: watch-jobs ────────────────────────────────────────────────

#[derive(Args)]
struct WatchJobsArgs {
    /// Stage 12.1: only `fs` is implemented. SNIP-index polling is
    /// deferred to Stage 12.2+ because SNIP roots are immutable.
    #[arg(long, value_enum, default_value_t = WatchSource::Fs)]
    source: WatchSource,

    /// Directory the dispatcher writes `PostedJob` envelopes into.
    /// Required when `--source fs`.
    #[arg(long)]
    jobs_dir: PathBuf,

    /// Polling interval in seconds.
    #[arg(long, default_value_t = 30)]
    poll_interval_secs: u64,

    /// Optional hard cap on jobs picked up before the loop exits 0.
    #[arg(long)]
    max_jobs: Option<u64>,

    /// Optional hard cap on poll iterations before the loop exits 0.
    /// Production typically omits this; useful for smoke runs.
    #[arg(long)]
    max_polls: Option<u64>,

    /// Allow-list of `model_hash` values to accept. Repeated. Empty
    /// = accept any.
    #[arg(long = "accept-model-hash")]
    accept_model_hash: Vec<String>,

    /// Allow-list of `tokenizer_hash` values to accept. Repeated.
    /// Empty = accept any.
    #[arg(long = "accept-tokenizer-hash")]
    accept_tokenizer_hash: Vec<String>,

    // Cost caps — REQUIRED. No defaults. An operator must explicitly
    // decide the workload envelope; a default that fits a dev box is
    // dangerous for production, and vice-versa.
    #[arg(long)]
    max_input_tokens: u64,
    #[arg(long)]
    max_output_tokens: u64,
    #[arg(long)]
    max_total_base_units: u64,

    /// Inference runner.
    #[arg(long, value_enum, default_value_t = RunnerChoice::Stub)]
    runner: RunnerChoice,

    #[arg(long, required_if_eq("runner", "external"))]
    external_command: Option<PathBuf>,

    #[arg(long = "external-arg")]
    external_args: Vec<String>,

    #[arg(long = "external-env-allow")]
    external_env_allow: Vec<String>,

    #[arg(long, required_if_eq("runner", "stub"))]
    stub_response: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    stub_input_tokens: u64,

    #[arg(long, default_value_t = 0)]
    stub_output_tokens: u64,

    /// 32-byte raw contributor seed.
    #[arg(long)]
    seed_file: PathBuf,

    /// Directory to write accepted `<job_id>.json` + rejected
    /// `<job_id>.rejected.json` result files into. Created if absent.
    #[arg(long)]
    result_out_dir: PathBuf,

    /// If set, publishes a signed `PostedResultLink` to SNIP after
    /// each accepted result.
    #[arg(long, default_value_t = false)]
    publish_result_link: bool,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum WatchSource {
    Fs,
}

// ── Stage 12.1: publish-result-link ───────────────────────────────────────

#[derive(Args)]
struct PublishResultLinkArgs {
    /// Path to the `ContributorResult` JSON to publish.
    #[arg(long)]
    result: PathBuf,

    /// Path to the `PostedJob` JSON the result answers. The link's
    /// `posted_id` is copied from this envelope.
    #[arg(long)]
    posted_job: PathBuf,

    /// Output path for the produced `PostedResultLink` JSON.
    #[arg(long)]
    link_out: PathBuf,

    /// 32-byte raw contributor seed.
    #[arg(long)]
    seed_file: PathBuf,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Entry point ───────────────────────────────────────────────────────────

pub async fn dispatch(args: ContributorArgs) -> Result<()> {
    match args.cmd {
        ContributorCmd::ValidateJob(a) => run_validate_job(a),
        ContributorCmd::RunJob(a) => run_run_job(a),
        ContributorCmd::VerifyResult(a) => run_verify_result(a),
        ContributorCmd::PostJob(a) => run_post_job(a),
        ContributorCmd::WatchJobs(a) => run_watch_jobs(a),
        ContributorCmd::PublishResultLink(a) => run_publish_result_link(a),
    }
}

fn build_snip_adapter(
    binary: PathBuf,
    seed: Option<PathBuf>,
) -> SnipV2Cli {
    SnipV2Cli::new(SnipV2CliConfig {
        binary_path: binary,
        seed_path: seed,
        extra_args: Vec::new(),
        allow_non_active: false,
    })
}

fn read_job(path: &std::path::Path) -> Result<ContributorJob> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read job json: {}", path.display()))?;
    serde_json::from_slice::<ContributorJob>(&bytes)
        .with_context(|| format!("parse job json: {}", path.display()))
}

fn read_result(path: &std::path::Path) -> Result<ContributorResult> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read result json: {}", path.display()))?;
    serde_json::from_slice::<ContributorResult>(&bytes)
        .with_context(|| format!("parse result json: {}", path.display()))
}

fn run_validate_job(args: ValidateJobArgs) -> Result<()> {
    let job = read_job(&args.job)?;
    let schema_ok = job.validate_schema().is_ok();
    let recomputed_job_id = job_hash_hex(&job).ok();
    let job_hash_ok = recomputed_job_id.as_deref() == Some(job.job_id.as_str());
    let dispatcher_signature = match (
        job.dispatcher_pubkey_hex.as_deref(),
        job.dispatcher_signature_hex.as_deref(),
    ) {
        (Some(pk), Some(sig)) => {
            match omni_contributor::canonical::canonical_job_bytes(&job) {
                Ok(input) => match omni_contributor::signing::verify_signature_hex(pk, &input, sig)
                {
                    Ok(true) => "ok",
                    Ok(false) => "fail",
                    Err(_) => "fail",
                },
                Err(_) => "fail",
            }
        }
        (None, None) => "not_signed",
        _ => "fail",
    };
    let expired = job
        .expires_at_utc
        .as_deref()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc) < chrono::Utc::now())
        .unwrap_or(false);

    println!("job_id={}", job.job_id);
    println!("schema_ok={}", schema_ok);
    println!("job_hash_ok={}", job_hash_ok);
    println!("dispatcher_signature={}", dispatcher_signature);
    println!("expired={}", expired);

    // SNIP-root resolvability probe is optional and best-effort. The
    // current `omni-store` API does not expose a cheap "object exists"
    // primitive; a real probe would download. Skip for Stage 12.0 and
    // emit `unknown`; downstream operators run `run-job` which
    // exercises the full fetch path.
    let _ = (args.snip_binary, args.snip_seed);
    println!("manifest_snip_resolvable=unknown");
    println!("input_snip_resolvable=unknown");

    let overall = schema_ok
        && job_hash_ok
        && !matches!(dispatcher_signature, "fail")
        && !expired;
    println!("overall={}", if overall { "ok" } else { "fail" });
    Ok(())
}

fn run_run_job(args: RunJobArgs) -> Result<()> {
    let job = read_job(&args.job)?;
    let adapter = build_snip_adapter(args.snip_binary, args.snip_seed);
    let signer = ContributorSigner::from_seed_file(&args.seed_file)
        .with_context(|| format!("load seed: {}", args.seed_file.display()))?;
    let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let opts = RunJobOptions {
        produced_at_utc: now,
        signer: &signer,
        notes: args.notes,
        job_snip_root: args.job_snip_root,
    };

    let result = match args.runner {
        RunnerChoice::Stub => {
            let response_bytes = std::fs::read(
                args.stub_response.as_ref().ok_or_else(|| anyhow!("--stub-response required"))?,
            )?;
            let runner = StubRunner::new(
                signer.pubkey_hex(),
                response_bytes,
                args.stub_input_tokens,
                args.stub_output_tokens,
            );
            run_job(&job, &adapter, &runner, opts)?
        }
        RunnerChoice::External => {
            let mut runner = ExternalCommandRunner::new(
                args.external_command
                    .ok_or_else(|| anyhow!("--external-command required for --runner external"))?,
            );
            runner.extra_args = args.external_args;
            runner.env_allowlist = args.external_env_allow;
            run_external_with_runner(&job, &adapter, &runner, opts)?
        }
    };

    let json = serde_json::to_string_pretty(&result)
        .context("serialize ContributorResult to JSON")?;
    std::fs::write(&args.out, json)
        .with_context(|| format!("write result: {}", args.out.display()))?;
    println!("result_path={}", args.out.display());
    println!("job_id={}", result.job_id);
    println!("response_snip_root={}", result.response_snip_root);
    println!("total_base_units={}", result.measured_accounting.total_base_units);
    println!("overall=ok");
    Ok(())
}

// Helper so we can keep `runner: &dyn InferenceRunner` style without
// borrowing through ExternalCommandRunner's owned fields.
fn run_external_with_runner<R: InferenceRunner>(
    job: &ContributorJob,
    adapter: &SnipV2Cli,
    runner: &R,
    opts: RunJobOptions<'_>,
) -> Result<ContributorResult> {
    Ok(run_job(job, adapter, runner, opts)?)
}

fn run_verify_result(args: VerifyResultArgs) -> Result<()> {
    let job = read_job(&args.job)?;
    let result = read_result(&args.result)?;
    let adapter = build_snip_adapter(args.snip_binary, args.snip_seed);
    let outcome = match verify_result(&job, &result, &adapter) {
        Ok(o) => o,
        Err(e) => {
            // Structural failure (schema error / expired job / etc.).
            // Emit a minimal bare-stdout error block + non-zero exit.
            println!("job_id={}", job.job_id);
            println!("overall=fail");
            println!("error={e}");
            bail!("verify-result structural failure: {e}");
        }
    };

    println!("job_id={}", outcome.job_id);
    println!("job_hash_ok={}", outcome.job_hash_ok);
    println!(
        "dispatcher_signature={}",
        match outcome.dispatcher_signature {
            DispatcherSignatureOutcome::Ok => "ok",
            DispatcherSignatureOutcome::Fail => "fail",
            DispatcherSignatureOutcome::NotSigned => "not_signed",
        }
    );
    println!("input_hash_ok={}", outcome.input_hash_ok);
    println!("response_hash_ok={}", outcome.response_hash_ok);
    println!("tokenizer_hash_ok={}", outcome.tokenizer_hash_ok);
    println!(
        "accounting_input_match={}",
        outcome.accounting_input_match
    );
    println!(
        "accounting_output_under_cap={}",
        outcome.accounting_output_under_cap
    );
    println!(
        "accounting_total_consistent={}",
        outcome.accounting_total_consistent
    );
    println!("total_base_units={}", outcome.total_base_units);
    println!(
        "contributor_signature_ok={}",
        outcome.contributor_signature_ok
    );
    println!("evidence_mode={}", outcome.evidence_mode);
    println!("evidence_ok={}", outcome.evidence_ok);
    println!("requirement_satisfied={}", outcome.requirement_satisfied);
    println!("overall={}", if outcome.overall_ok { "ok" } else { "fail" });

    // Catch unused-import warning for WorkUnitKind by suppressing
    // here — the type is part of the public re-export the CLI uses
    // transitively via ContributorResult.
    let _ = std::marker::PhantomData::<WorkUnitKind>;
    Ok(())
}

// ── Stage 12.1: post-job handler ──────────────────────────────────────────

fn run_post_job(args: PostJobArgs) -> Result<()> {
    use omni_contributor::canonical::{canonical_posted_job_bytes, hex_lower, posted_id_hex};
    use omni_contributor::posted::{PostedJob, POSTED_SCHEMA_VERSION};
    use omni_contributor::signing::DispatcherSigner;

    let job = read_job(&args.job)?;
    let adapter = build_snip_adapter(args.snip_binary, args.snip_seed);

    // Publish the ContributorJob JSON to SNIP.
    let job_json = serde_json::to_string_pretty(&job)?;
    let job_root = omni_contributor::snip::publish_bytes(
        &adapter,
        job_json.as_bytes(),
        "contributor-job",
    )?;
    let job_snip_root_hex = format!("0x{}", hex_lower(job_root.as_bytes()));

    // Build the PostedJob envelope (unsigned; sign below if seed given).
    let posted_at_utc = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let signer = match &args.seed_file {
        Some(p) => Some(DispatcherSigner::from_seed_file(p)?),
        None => None,
    };

    let mut posted = PostedJob {
        schema_version: POSTED_SCHEMA_VERSION,
        posted_id: String::new(),
        job_snip_root: job_snip_root_hex.clone(),
        job_hash: omni_contributor::canonical::job_hash_hex(&job)?,
        model_hash: job.model_hash.clone(),
        posted_at_utc,
        expires_at_utc: args.expires_at_utc,
        poster_pubkey_hex: signer.as_ref().map(|s| s.pubkey_hex()),
        poster_signature_hex: None,
        notes: args.notes,
    };
    posted.posted_id = posted_id_hex(&posted)?;
    if let Some(ref s) = signer {
        let signing_input = canonical_posted_job_bytes(&posted)?;
        posted.poster_signature_hex = Some(s.sign_hex(&signing_input));
    }
    posted
        .validate_schema()
        .map_err(|e| anyhow!("invalid PostedJob after build: {e}"))?;

    let posted_json = serde_json::to_string_pretty(&posted)?;
    std::fs::write(&args.posted_out, posted_json)
        .with_context(|| format!("write posted-job: {}", args.posted_out.display()))?;

    println!("job_snip_root={}", job_snip_root_hex);
    println!("posted_id={}", posted.posted_id);
    println!("posted_path={}", args.posted_out.display());
    Ok(())
}

// ── Stage 12.1: watch-jobs handler ────────────────────────────────────────

fn run_watch_jobs(args: WatchJobsArgs) -> Result<()> {
    use omni_contributor::runner::{ExternalCommandRunner, StubRunner};
    use omni_contributor::{
        AcceptFilters, ContributorSigner, CostCaps, FilesystemSource, InferenceRunner,
        StdoutEmitter, WatchOptions,
    };
    use std::time::Duration;

    match args.source {
        WatchSource::Fs => {}
    }

    let adapter = build_snip_adapter(args.snip_binary, args.snip_seed);
    let mut source = FilesystemSource::new(args.jobs_dir);
    let signer = ContributorSigner::from_seed_file(&args.seed_file)
        .with_context(|| format!("load seed: {}", args.seed_file.display()))?;

    // Build the runner. Same pattern as run-job.
    enum AnyRunner {
        Stub(StubRunner),
        External(ExternalCommandRunner),
    }
    impl InferenceRunner for AnyRunner {
        fn run(
            &self,
            manifest_path: &std::path::Path,
            input_bytes: &[u8],
        ) -> std::result::Result<
            omni_contributor::RunOutput,
            omni_contributor::RunnerError,
        > {
            match self {
                AnyRunner::Stub(r) => r.run(manifest_path, input_bytes),
                AnyRunner::External(r) => r.run(manifest_path, input_bytes),
            }
        }
    }
    let runner = match args.runner {
        RunnerChoice::Stub => {
            let response_bytes = std::fs::read(
                args.stub_response
                    .as_ref()
                    .ok_or_else(|| anyhow!("--stub-response required"))?,
            )?;
            AnyRunner::Stub(StubRunner::new(
                signer.pubkey_hex(),
                response_bytes,
                args.stub_input_tokens,
                args.stub_output_tokens,
            ))
        }
        RunnerChoice::External => {
            let mut r = ExternalCommandRunner::new(
                args.external_command
                    .ok_or_else(|| anyhow!("--external-command required for --runner external"))?,
            );
            r.extra_args = args.external_args;
            r.env_allowlist = args.external_env_allow;
            AnyRunner::External(r)
        }
    };

    let mut emitter = StdoutEmitter;
    let opts = WatchOptions {
        poll_interval: Duration::from_secs(args.poll_interval_secs),
        max_jobs: args.max_jobs,
        max_polls: args.max_polls,
        filters: AcceptFilters {
            model_hash_allow: args.accept_model_hash,
            tokenizer_hash_allow: args.accept_tokenizer_hash,
        },
        caps: CostCaps {
            max_input_tokens: args.max_input_tokens,
            max_output_tokens: args.max_output_tokens,
            max_total_base_units: args.max_total_base_units,
        },
        runner: &runner,
        signer: &signer,
        result_out_dir: args.result_out_dir,
        publish_link: args.publish_result_link,
        emit: &mut emitter,
    };

    omni_contributor::run_watch_loop(&adapter, &mut source, opts)
        .map_err(|e| anyhow!("watch-jobs error: {e}"))?;
    Ok(())
}

// ── Stage 12.1: publish-result-link handler ───────────────────────────────

fn run_publish_result_link(args: PublishResultLinkArgs) -> Result<()> {
    use omni_contributor::canonical::hex_lower;
    use omni_contributor::posted::PostedJob;
    use omni_contributor::watch::{publish_result_link_for, StdoutEmitter};
    use omni_contributor::ContributorSigner;

    let result = read_result(&args.result)?;
    let posted_bytes = std::fs::read(&args.posted_job)?;
    let posted: PostedJob = serde_json::from_slice(&posted_bytes)
        .with_context(|| format!("parse posted-job: {}", args.posted_job.display()))?;
    posted
        .validate_schema()
        .map_err(|e| anyhow!("invalid PostedJob: {e}"))?;

    let adapter = build_snip_adapter(args.snip_binary, args.snip_seed);
    let signer = ContributorSigner::from_seed_file(&args.seed_file)?;
    let result_json = serde_json::to_string_pretty(&result)?;
    let mut emitter = StdoutEmitter;

    let published = publish_result_link_for(
        &adapter,
        &posted,
        &result_json,
        &result,
        &signer,
        &mut emitter,
    )
    .map_err(|e| anyhow!("publish result link failed: {e}"))?;

    // Write the EXACT bytes that were published to SNIP. No
    // re-signing, no fresh timestamp — that would diverge the local
    // file from the SNIP-published artifact.
    std::fs::write(&args.link_out, &published.link_json)
        .with_context(|| format!("write link: {}", args.link_out.display()))?;

    println!("result_snip_root={}", published.link.result_snip_root);
    println!(
        "link_snip_root=0x{}",
        hex_lower(published.link_snip_root.as_bytes())
    );
    println!("posted_id={}", published.link.posted_id);
    println!("link_path={}", args.link_out.display());
    Ok(())
}
