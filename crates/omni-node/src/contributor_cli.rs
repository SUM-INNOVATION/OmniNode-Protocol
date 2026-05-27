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

    /// Stage 12.2 — broadcast a signed `NetworkPostedJobAnnouncement`
    /// for a `PostedJob` over the contributor gossip mesh.
    AnnounceJob(AnnounceJobArgs),

    /// Stage 12.2 — long-running: subscribe to the contributor job
    /// gossip topic, validate + fetch each announced `PostedJob`
    /// from SNIP, apply filters + cost caps, run inference, verify
    /// the result, write output, and optionally publish + broadcast
    /// a `PostedResultLink`.
    WatchNetworkJobs(WatchNetworkJobsArgs),

    /// Stage 12.2 — broadcast a signed
    /// `NetworkPostedResultAnnouncement` for an existing
    /// `PostedResultLink`.
    AnnounceResult(AnnounceResultArgs),

    /// Stage 12.2 — long-running: subscribe to the contributor
    /// result gossip topic, fetch + validate each announced
    /// `PostedResultLink` from SNIP, optionally filter by
    /// `posted_id`, write the link envelopes to disk.
    WatchNetworkResults(WatchNetworkResultsArgs),

    /// Stage 12.3 — coordinator opens an `ExecutionSession` for a
    /// posted job, publishes the session envelope to SNIP, and
    /// broadcasts a `NetworkSessionOpenedAnnouncement`.
    OpenSession(OpenSessionArgs),

    /// Stage 12.3 — contributor advertises capability + RAM hints
    /// for an open session, publishes a signed `ContributorJoin`
    /// to SNIP, and broadcasts a
    /// `NetworkContributorJoinedAnnouncement`.
    JoinSession(JoinSessionArgs),

    /// Stage 12.3 — coordinator builds + signs `WorkAssignment`s
    /// from a spec file, publishes each to SNIP, and broadcasts a
    /// `NetworkWorkAssignedAnnouncement` per assignment.
    AssignWork(AssignWorkArgs),

    /// Stage 12.3 — contributor reads a WorkAssignment, runs local
    /// inference, publishes the partial artifact + signed
    /// `PartialContributorResult` to SNIP, and broadcasts a
    /// `NetworkPartialResultAnnouncement`.
    RunAssignment(RunAssignmentArgs),

    /// Stage 12.3 — coordinator validates the chain of partials,
    /// builds + signs an `AggregatedContributorResult`, publishes
    /// to SNIP, and broadcasts a
    /// `NetworkAggregatedResultAnnouncement`.
    AggregateSession(AggregateSessionArgs),

    /// Stage 12.3 — long-running passive observer: subscribes to
    /// every session topic, fetches each announced body from SNIP,
    /// runs the local session verifier, and writes a per-session
    /// tree (`session.json`, `joins/`, `assignments/`, `partials/`,
    /// `aggregated.json`) under `--out-dir`.
    WatchSessions(WatchSessionsArgs),
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

// ── Stage 12.2: announce-job args ─────────────────────────────────────────

#[derive(Args)]
struct AnnounceJobArgs {
    /// Path to a `PostedJob` JSON (produced by `post-job`).
    #[arg(long)]
    posted_job: PathBuf,

    /// 32-byte raw announcer seed.
    #[arg(long)]
    seed_file: PathBuf,

    /// Optional: also fetch the inner `ContributorJob` from SNIP and
    /// record its `tokenizer_hash` in the announcement (advisory
    /// field). On any fetch / parse failure, leave the field None
    /// and emit the announcement without it.
    #[arg(long, default_value_t = false)]
    include_tokenizer_hash: bool,

    /// libp2p listen port. Mapped to `NetConfig.listen_port`.
    #[arg(long, default_value_t = 0)]
    listen_port: u16,

    /// Bootstrap peer multiaddr (repeatable). Mapped to
    /// `NetConfig.bootstrap_peers`.
    #[arg(long = "peer")]
    peer: Vec<String>,

    /// Brief wait (milliseconds) after publishing before the
    /// subcommand exits, to give gossipsub a chance to propagate.
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,

    /// Bounded wait (seconds) for the first peer to appear on the
    /// mesh BEFORE publishing. Mirrors `omni-node send`'s 30s
    /// default. Set to 0 to skip the wait (only useful in tests
    /// against an already-connected node).
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,

    /// Brief wait (milliseconds) AFTER the first peer is observed
    /// and BEFORE publishing, so gossipsub has time to form a topic
    /// mesh. Same heuristic the `omni-node send` path uses.
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Stage 12.2: watch-network-jobs args ───────────────────────────────────

#[derive(Args)]
struct WatchNetworkJobsArgs {
    // Cost caps — REQUIRED. Same policy as Stage 12.1's watch-jobs.
    #[arg(long)]
    max_input_tokens: u64,
    #[arg(long)]
    max_output_tokens: u64,
    #[arg(long)]
    max_total_base_units: u64,

    #[arg(long = "accept-model-hash")]
    accept_model_hash: Vec<String>,
    #[arg(long = "accept-tokenizer-hash")]
    accept_tokenizer_hash: Vec<String>,

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

    #[arg(long)]
    seed_file: PathBuf,

    #[arg(long)]
    result_out_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    publish_result_link: bool,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,

    #[arg(long = "peer")]
    peer: Vec<String>,

    #[arg(long, default_value_t = 5)]
    poll_interval_secs: u64,

    #[arg(long)]
    max_jobs: Option<u64>,

    #[arg(long)]
    max_polls: Option<u64>,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Stage 12.2: announce-result args ──────────────────────────────────────

#[derive(Args)]
struct AnnounceResultArgs {
    /// Path to a `PostedResultLink` JSON (produced by `publish-result-link`).
    #[arg(long)]
    posted_result_link: PathBuf,

    /// Announcer seed (may be the same key as the contributor that
    /// signed the link, or a different relayer).
    #[arg(long)]
    seed_file: PathBuf,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,

    #[arg(long = "peer")]
    peer: Vec<String>,

    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,

    /// Bounded wait (seconds) for the first peer to appear on the
    /// mesh BEFORE publishing. Mirrors `omni-node send`'s 30s
    /// default. Set to 0 to skip.
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,

    /// Brief wait (milliseconds) AFTER first peer + BEFORE publish
    /// so gossipsub forms a topic mesh.
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Stage 12.2: watch-network-results args ────────────────────────────────

#[derive(Args)]
struct WatchNetworkResultsArgs {
    /// Optional filter: only fetch links whose `posted_id` matches
    /// one of these values. Empty = accept any.
    #[arg(long = "posted-id")]
    posted_id: Vec<String>,

    /// Directory to write fetched `PostedResultLink` JSON files.
    #[arg(long)]
    result_out_dir: PathBuf,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,

    #[arg(long = "peer")]
    peer: Vec<String>,

    #[arg(long, default_value_t = 5)]
    poll_interval_secs: u64,

    #[arg(long)]
    max_results: Option<u64>,

    #[arg(long)]
    max_polls: Option<u64>,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,

    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Stage 12.3: session args ───────────────────────────────────────────────

#[derive(Args)]
struct OpenSessionArgs {
    /// Path to a `PostedJob` JSON file on disk.
    #[arg(long)]
    posted_job: PathBuf,

    /// 32-byte raw coordinator seed file (role-distinct from
    /// contributor/dispatcher seeds — see CoordinatorSigner).
    #[arg(long)]
    coordinator_seed: PathBuf,

    /// RFC 3339 UTC (`Z` suffix). Sessions are bounded; required.
    #[arg(long)]
    expires_at_utc: String,

    /// libp2p listen port + bootstrap peers (same shape as 12.2
    /// announce-*). Defaults: ephemeral port, no bootstrap peers.
    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

#[derive(Args)]
struct JoinSessionArgs {
    /// SNIP V2 root of the open `ExecutionSession` (`0x`-prefixed
    /// lowercase hex, 66 chars). The CLI fetches and verifies the
    /// session before publishing the join.
    #[arg(long)]
    execution_session_snip_root: String,

    /// 32-byte raw contributor seed file.
    #[arg(long)]
    contributor_seed: PathBuf,

    /// Advertised free RAM at join time. Hint; the protocol does
    /// not verify it.
    #[arg(long)]
    available_ram_bytes: u64,

    #[arg(long)]
    max_input_tokens: u64,
    #[arg(long)]
    max_output_tokens: u64,

    /// Repeatable. At least one must be supplied.
    #[arg(long = "supported-work-unit-kind", value_enum)]
    supported_work_unit_kinds: Vec<WorkUnitKindArg>,

    /// Free-form runner kind (`stub`, `external`, `vllm-shim`, …).
    #[arg(long)]
    runner_kind: String,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

/// CLI mirror of `omni_contributor::WorkUnitKind`. Kept here (rather
/// than re-exporting the contributor enum with clap derives) so we
/// don't pollute the contributor crate with clap.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum WorkUnitKindArg {
    Tokens,
    PrefillTokens,
    DecodeTokens,
    Layers,
    FlopsEstimate,
}

impl From<WorkUnitKindArg> for omni_contributor::result::WorkUnitKind {
    fn from(w: WorkUnitKindArg) -> Self {
        use omni_contributor::result::WorkUnitKind as W;
        match w {
            WorkUnitKindArg::Tokens => W::Tokens,
            WorkUnitKindArg::PrefillTokens => W::PrefillTokens,
            WorkUnitKindArg::DecodeTokens => W::DecodeTokens,
            WorkUnitKindArg::Layers => W::Layers,
            WorkUnitKindArg::FlopsEstimate => W::FlopsEstimate,
        }
    }
}

#[derive(Args)]
struct AssignWorkArgs {
    /// SNIP V2 root of the open `ExecutionSession`.
    #[arg(long)]
    execution_session_snip_root: String,

    /// Path to a JSON file describing one or more assignments. See
    /// docs/stage12-contributor-protocol.md for the schema; in short,
    /// an array of `{ contributor_pubkey_hex, stage_index, work_kind,
    /// expected_work_units, expected_work_unit_kind, assigned_at_utc }`.
    #[arg(long)]
    assignments_file: PathBuf,

    #[arg(long)]
    coordinator_seed: PathBuf,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

#[derive(Args)]
struct RunAssignmentArgs {
    /// SNIP V2 root of the `WorkAssignment`.
    #[arg(long)]
    assignment_snip_root: String,

    /// SNIP V2 root of the parent `ExecutionSession`. Required so
    /// the contributor can drift-guard against it.
    #[arg(long)]
    execution_session_snip_root: String,

    /// 32-byte raw contributor seed file. Must correspond to the
    /// assignment's `contributor_pubkey_hex`.
    #[arg(long)]
    contributor_seed: PathBuf,

    /// Runner selection — mirrors `watch-jobs`'s flags. Stage 12.3
    /// keeps the runner choice opaque; the partial artifact bytes
    /// are whatever the runner stdout-envelope's `response_b64`
    /// decodes to.
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

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

#[derive(Args)]
struct AggregateSessionArgs {
    #[arg(long)]
    execution_session_snip_root: String,

    #[arg(long)]
    coordinator_seed: PathBuf,

    /// SNIP V2 root of the standalone (Stage 12.0 v1) final
    /// `ContributorResult` JSON produced by the last stage's
    /// contributor.
    #[arg(long)]
    final_result_snip_root: String,

    /// SNIP V2 roots of every `ContributorJoin` that participated
    /// in the session. Repeatable; **required**. The aggregate
    /// verifier needs the joined-pubkey set to prove each
    /// assignment targets a joined contributor.
    #[arg(long = "join-snip-root")]
    join_snip_roots: Vec<String>,

    /// SNIP V2 roots of every `WorkAssignment` in the session.
    /// Repeatable; required (every assignment must be covered by
    /// a partial — Stage 12.3 has no `--allow-incomplete`).
    #[arg(long = "assignment-snip-root")]
    assignment_snip_roots: Vec<String>,

    /// SNIP V2 roots of every `PartialContributorResult`.
    /// Repeatable; required.
    #[arg(long = "partial-snip-root")]
    partial_snip_roots: Vec<String>,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

#[derive(Args)]
struct WatchSessionsArgs {
    /// Root directory under which per-session subdirectories are
    /// written (`<out-dir>/<session_id>/...`).
    #[arg(long)]
    out_dir: PathBuf,

    /// Optional posted_id filter (repeatable). Empty = accept any.
    #[arg(long = "posted-id")]
    posted_id: Vec<String>,

    /// Optional session_id filter (repeatable). Empty = accept any.
    #[arg(long = "session-id")]
    session_id: Vec<String>,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,
    #[arg(long, default_value_t = 5)]
    poll_interval_secs: u64,
    #[arg(long)]
    max_polls: Option<u64>,

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
        ContributorCmd::AnnounceJob(a) => run_announce_job(a).await,
        ContributorCmd::WatchNetworkJobs(a) => run_watch_network_jobs(a).await,
        ContributorCmd::AnnounceResult(a) => run_announce_result(a).await,
        ContributorCmd::WatchNetworkResults(a) => run_watch_network_results(a).await,
        ContributorCmd::OpenSession(a) => run_open_session(a).await,
        ContributorCmd::JoinSession(a) => run_join_session(a).await,
        ContributorCmd::AssignWork(a) => run_assign_work(a).await,
        ContributorCmd::RunAssignment(a) => run_assignment(a).await,
        ContributorCmd::AggregateSession(a) => run_aggregate_session(a).await,
        ContributorCmd::WatchSessions(a) => run_watch_sessions(a).await,
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

/// Wait until at least one peer is reachable on the mesh, with a
/// bounded timeout. Mirrors the `omni-node send` discovery pattern:
/// a bare `OmniNet::new()` is not yet connected to any peer, so
/// `publish()` on an empty mesh is a silent drop. Accept either an
/// mDNS-style `PeerDiscovered` event (LAN) or a `PeerConnected`
/// event (explicit `--peer` bootstrap multiaddr) — both are
/// sufficient evidence of a reachable transport.
///
/// After the first peer event the function also waits a brief
/// `mesh_stabilize_ms` so gossipsub has a chance to form a topic
/// mesh before the caller publishes.
async fn wait_for_first_peer(
    net: &mut omni_net::OmniNet,
    timeout_secs: u64,
    mesh_stabilize_ms: u64,
) -> Result<()> {
    use omni_net::OmniNetEvent;
    let wait_fut = async {
        while let Some(event) = net.next_event().await {
            if matches!(
                event,
                OmniNetEvent::PeerDiscovered { .. } | OmniNetEvent::PeerConnected { .. }
            ) {
                return Ok::<(), anyhow::Error>(());
            }
        }
        Err(anyhow!("OmniNet event stream closed before any peer appeared"))
    };
    tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), wait_fut)
        .await
        .map_err(|_| {
            anyhow!(
                "no peer reachable within {timeout_secs}s — check --peer bootstrap and LAN reachability"
            )
        })??;
    if mesh_stabilize_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(mesh_stabilize_ms)).await;
    }
    Ok(())
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
        result_broadcaster: None,
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

// ── Stage 12.2 handlers ───────────────────────────────────────────────────

async fn run_announce_job(args: AnnounceJobArgs) -> Result<()> {
    use omni_contributor::canonical::{
        network_job_announcement_signing_input,
    };
    use omni_contributor::posted::PostedJob;
    use omni_contributor::signing::DispatcherSigner;
    use omni_contributor::{ContributorRelay, NetworkPostedJobAnnouncement, OmniNetRelay, NET_SCHEMA_VERSION};
    use omni_net::OmniNet;
    use omni_types::config::NetConfig;

    let posted_bytes = std::fs::read(&args.posted_job)
        .with_context(|| format!("read posted-job: {}", args.posted_job.display()))?;
    let posted: PostedJob = serde_json::from_slice(&posted_bytes)
        .with_context(|| format!("parse posted-job: {}", args.posted_job.display()))?;
    posted
        .validate_schema()
        .map_err(|e| anyhow!("invalid PostedJob: {e}"))?;

    // Optional: fetch the inner ContributorJob to record tokenizer_hash.
    let tokenizer_hash = if args.include_tokenizer_hash {
        let snip_adapter = build_snip_adapter(args.snip_binary.clone(), args.snip_seed.clone());
        let snip_root = omni_types::phase5::SnipV2ObjectId::from_hex(&posted.job_snip_root)
            .map_err(|e| anyhow!("bad job_snip_root: {e:?}"))?;
        match omni_contributor::snip::fetch_bytes(&snip_adapter, &snip_root) {
            Ok(bytes) => serde_json::from_slice::<omni_contributor::ContributorJob>(&bytes)
                .ok()
                .map(|j| j.accounting.tokenizer_hash),
            Err(_) => None,
        }
    } else {
        None
    };

    let signer = DispatcherSigner::from_seed_file(&args.seed_file)?;
    let announced_at_utc = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // Publish the PostedJob envelope bytes themselves to SNIP. The
    // resulting SNIP root is what the network announcement carries
    // as `posted_job_snip_root`. (The inner job's
    // `posted.job_snip_root` is a separate SNIP root for the
    // ContributorJob JSON; do NOT confuse the two — NetworkSource
    // fetches `posted_job_snip_root` expecting PostedJob bytes.)
    let snip_adapter_for_publish = build_snip_adapter(
        args.snip_binary.clone(),
        args.snip_seed.clone(),
    );
    let posted_job_snip_root_obj = omni_contributor::snip::publish_bytes(
        &snip_adapter_for_publish,
        &posted_bytes,
        "announce-job-posted",
    )
    .map_err(|e| anyhow!("publish PostedJob to SNIP: {e}"))?;
    let posted_job_snip_root_hex = format!(
        "0x{}",
        omni_contributor::canonical::hex_lower(posted_job_snip_root_obj.as_bytes())
    );

    let mut ann = NetworkPostedJobAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_job_snip_root: posted_job_snip_root_hex.clone(),
        posted_id: posted.posted_id.clone(),
        job_hash: posted.job_hash.clone(),
        model_hash: posted.model_hash.clone(),
        tokenizer_hash,
        announced_at_utc,
        announcer_pubkey_hex: signer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_job_announcement_signing_input(&ann)
        .map_err(|e| anyhow!("canonical encode: {e}"))?;
    ann.announcer_signature_hex = signer.sign_hex(&signing_input);
    ann.validate_schema()
        .map_err(|e| anyhow!("invalid NetworkPostedJobAnnouncement: {e}"))?;

    // Open OmniNet and publish.
    let net_config = NetConfig {
        listen_port: args.listen_port,
        bootstrap_peers: args.peer,
        ..NetConfig::default()
    };
    let mut net = OmniNet::new(net_config)
        .await
        .map_err(|e| anyhow!("OmniNet::new: {e}"))?;
    // Bounded peer-wait BEFORE publish so the announcement isn't a
    // silent drop on an empty mesh. Mirrors `omni-node send`'s
    // PeerDiscovered-wait pattern at main.rs:309.
    if args.peer_wait_secs > 0 {
        wait_for_first_peer(&mut net, args.peer_wait_secs, args.mesh_stabilize_ms).await?;
    }
    let net = std::sync::Arc::new(tokio::sync::Mutex::new(net));
    let handle = tokio::runtime::Handle::current();
    let mut relay = OmniNetRelay::new(net.clone(), handle);
    relay
        .publish_job(&ann)
        .map_err(|e| anyhow!("publish: {e}"))?;

    // Brief propagation wait.
    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }

    println!("posted_id={}", ann.posted_id);
    println!("posted_job_snip_root={}", ann.posted_job_snip_root);
    println!("announced=true");
    Ok(())
}

async fn run_watch_network_jobs(args: WatchNetworkJobsArgs) -> Result<()> {
    use omni_contributor::canonical::network_result_announcement_signing_input;
    use omni_contributor::runner::{ExternalCommandRunner, StubRunner};
    use omni_contributor::{
        AcceptFilters, ContributorRelay, ContributorSigner, CostCaps, InferenceRunner,
        NetworkPostedResultAnnouncement, NetworkSource, OmniNetRelay, PublishedResultLink,
        ResultBroadcaster, StdoutEmitter, WatchOptions, NET_SCHEMA_VERSION,
    };
    use omni_net::OmniNet;
    use omni_types::config::NetConfig;

    let signer = ContributorSigner::from_seed_file(&args.seed_file)?;

    // Construct the runner (same shape as Stage 12.1's watch-jobs).
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
            let bytes = std::fs::read(
                args.stub_response
                    .as_ref()
                    .ok_or_else(|| anyhow!("--stub-response required"))?,
            )?;
            AnyRunner::Stub(StubRunner::new(
                signer.pubkey_hex(),
                bytes,
                args.stub_input_tokens,
                args.stub_output_tokens,
            ))
        }
        RunnerChoice::External => {
            let mut r = ExternalCommandRunner::new(
                args.external_command
                    .ok_or_else(|| anyhow!("--external-command required"))?,
            );
            r.extra_args = args.external_args;
            r.env_allowlist = args.external_env_allow;
            AnyRunner::External(r)
        }
    };

    let snip_adapter = build_snip_adapter(args.snip_binary, args.snip_seed);
    let net_config = NetConfig {
        listen_port: args.listen_port,
        bootstrap_peers: args.peer,
        ..NetConfig::default()
    };
    let net = OmniNet::new(net_config)
        .await
        .map_err(|e| anyhow!("OmniNet::new: {e}"))?;
    let net = std::sync::Arc::new(tokio::sync::Mutex::new(net));
    let handle = tokio::runtime::Handle::current();

    // Build a result broadcaster that piggybacks on the same
    // OmniNet via a second `OmniNetRelay` clone. When the watch
    // loop publishes a result link to SNIP, this broadcaster
    // builds + signs a `NetworkPostedResultAnnouncement` (the
    // contributor signer doubles as the announcer) and posts it on
    // the contributor-result topic, so peers running
    // `watch-network-results` learn the link's SNIP root.
    let broadcaster_relay = OmniNetRelay::new(net.clone(), handle.clone());
    let broadcaster_signer = ContributorSigner::from_seed_file(&args.seed_file)?;
    let publish_result_link = args.publish_result_link;

    let run_result = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut relay = OmniNetRelay::new(net.clone(), handle);
        let mut source = NetworkSource::new(&mut relay, &snip_adapter);
        let mut emitter = StdoutEmitter;

        struct NetResultBroadcaster {
            relay: OmniNetRelay,
            signer: ContributorSigner,
        }
        impl ResultBroadcaster for NetResultBroadcaster {
            fn broadcast(
                &mut self,
                published: &PublishedResultLink,
            ) -> std::result::Result<(), String> {
                let announced_at_utc = chrono::Utc::now()
                    .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
                let mut ann = NetworkPostedResultAnnouncement {
                    schema_version: NET_SCHEMA_VERSION,
                    posted_id: published.link.posted_id.clone(),
                    posted_result_link_snip_root: format!(
                        "0x{}",
                        omni_contributor::canonical::hex_lower(
                            published.link_snip_root.as_bytes()
                        )
                    ),
                    result_canonical_hash: published.link.result_canonical_hash.clone(),
                    contributor_pubkey_hex: published.link.contributor_pubkey_hex.clone(),
                    announced_at_utc,
                    announcer_pubkey_hex: self.signer.pubkey_hex(),
                    announcer_signature_hex: String::new(),
                };
                let signing_input = network_result_announcement_signing_input(&ann)
                    .map_err(|e| format!("canonical: {e}"))?;
                ann.announcer_signature_hex = self.signer.sign_hex(&signing_input);
                ann.validate_schema()
                    .map_err(|e| format!("schema: {e}"))?;
                self.relay
                    .publish_result(&ann)
                    .map_err(|e| format!("publish: {e}"))?;
                println!(
                    "event=result_announcement_broadcast posted_id={} link_snip_root={}",
                    ann.posted_id, ann.posted_result_link_snip_root
                );
                Ok(())
            }
        }
        let mut broadcaster = NetResultBroadcaster {
            relay: broadcaster_relay,
            signer: broadcaster_signer,
        };

        let opts = WatchOptions {
            poll_interval: std::time::Duration::from_secs(args.poll_interval_secs),
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
            publish_link: publish_result_link,
            emit: &mut emitter,
            result_broadcaster: if publish_result_link {
                Some(&mut broadcaster)
            } else {
                None
            },
        };
        omni_contributor::run_watch_loop(&snip_adapter, &mut source, opts)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| anyhow!("watch-network-jobs join: {e}"))?;
    run_result
}

async fn run_announce_result(args: AnnounceResultArgs) -> Result<()> {
    use omni_contributor::canonical::network_result_announcement_signing_input;
    use omni_contributor::posted::PostedResultLink;
    use omni_contributor::signing::ContributorSigner;
    use omni_contributor::{ContributorRelay, NetworkPostedResultAnnouncement, OmniNetRelay, NET_SCHEMA_VERSION};
    use omni_net::OmniNet;
    use omni_types::config::NetConfig;

    let bytes = std::fs::read(&args.posted_result_link)
        .with_context(|| format!("read link: {}", args.posted_result_link.display()))?;
    let link: PostedResultLink = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse link: {}", args.posted_result_link.display()))?;
    link.validate_schema()
        .map_err(|e| anyhow!("invalid PostedResultLink: {e}"))?;

    let signer = ContributorSigner::from_seed_file(&args.seed_file)?;
    let announced_at_utc = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // The link envelope on disk doesn't itself record its own SNIP
    // root (the standalone publish-result-link CLI prints it, but
    // the on-disk JSON only carries result_snip_root, not its own
    // posted_result_link_snip_root). To produce a faithful network
    // announcement we re-publish the link bytes to SNIP and use
    // that root.
    let snip_adapter = build_snip_adapter(
        args.snip_binary.clone(),
        args.snip_seed.clone(),
    );
    let link_root =
        omni_contributor::snip::publish_bytes(&snip_adapter, &bytes, "announce-result-link")
            .map_err(|e| anyhow!("snip publish: {e}"))?;
    let link_root_hex = format!(
        "0x{}",
        omni_contributor::canonical::hex_lower(link_root.as_bytes())
    );

    let mut ann = NetworkPostedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        posted_id: link.posted_id.clone(),
        posted_result_link_snip_root: link_root_hex.clone(),
        result_canonical_hash: link.result_canonical_hash.clone(),
        contributor_pubkey_hex: link.contributor_pubkey_hex.clone(),
        announced_at_utc,
        announcer_pubkey_hex: signer.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let signing_input = network_result_announcement_signing_input(&ann)
        .map_err(|e| anyhow!("canonical encode: {e}"))?;
    ann.announcer_signature_hex = signer.sign_hex(&signing_input);
    ann.validate_schema()
        .map_err(|e| anyhow!("invalid NetworkPostedResultAnnouncement: {e}"))?;

    let net_config = NetConfig {
        listen_port: args.listen_port,
        bootstrap_peers: args.peer,
        ..NetConfig::default()
    };
    let mut net = OmniNet::new(net_config)
        .await
        .map_err(|e| anyhow!("OmniNet::new: {e}"))?;
    // Same bounded peer-wait as announce-job: without it, a
    // freshly-opened OmniNet has zero peers and `publish` silently
    // drops the announcement.
    if args.peer_wait_secs > 0 {
        wait_for_first_peer(&mut net, args.peer_wait_secs, args.mesh_stabilize_ms).await?;
    }
    let net = std::sync::Arc::new(tokio::sync::Mutex::new(net));
    let handle = tokio::runtime::Handle::current();
    let mut relay = OmniNetRelay::new(net.clone(), handle);
    relay
        .publish_result(&ann)
        .map_err(|e| anyhow!("publish: {e}"))?;
    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }

    println!("posted_id={}", ann.posted_id);
    println!("posted_result_link_snip_root={}", ann.posted_result_link_snip_root);
    println!("announced=true");
    Ok(())
}

async fn run_watch_network_results(args: WatchNetworkResultsArgs) -> Result<()> {
    use omni_contributor::{
        process_result_announcement, ContributorRelay, OmniNetRelay,
        ResultAnnouncementOutcome,
    };
    use omni_net::OmniNet;
    use omni_types::config::NetConfig;
    use std::collections::HashSet;
    use std::time::Duration;

    let snip_adapter = build_snip_adapter(args.snip_binary, args.snip_seed);
    let net_config = NetConfig {
        listen_port: args.listen_port,
        bootstrap_peers: args.peer,
        ..NetConfig::default()
    };
    let net = OmniNet::new(net_config)
        .await
        .map_err(|e| anyhow!("OmniNet::new: {e}"))?;
    let net = std::sync::Arc::new(tokio::sync::Mutex::new(net));
    let handle = tokio::runtime::Handle::current();
    std::fs::create_dir_all(&args.result_out_dir)?;

    let filter: HashSet<String> = args.posted_id.into_iter().collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut polls_done: u64 = 0;
    let mut results_written: u64 = 0;

    let run_result = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut relay = OmniNetRelay::new(net.clone(), handle);
        loop {
            if let Some(max) = args.max_polls {
                if polls_done >= max {
                    println!("event=exit reason=max_polls_reached results_written={results_written}");
                    return Ok(());
                }
            }
            polls_done += 1;
            let anns = relay
                .poll_results()
                .map_err(|e| anyhow!("poll: {e}"))?;
            for ann in anns {
                if seen.contains(&ann.posted_result_link_snip_root) {
                    println!(
                        "event=skip posted_id={} reason=already_seen",
                        ann.posted_id
                    );
                    continue;
                }
                seen.insert(ann.posted_result_link_snip_root.clone());

                let outcome = process_result_announcement(
                    &ann,
                    &snip_adapter,
                    &filter,
                    &args.result_out_dir,
                );
                match outcome {
                    ResultAnnouncementOutcome::LinkWritten { posted_id, link_path } => {
                        println!(
                            "event=link_written posted_id={posted_id} path={}",
                            link_path.display()
                        );
                        results_written += 1;
                        if let Some(max) = args.max_results {
                            if results_written >= max {
                                println!(
                                    "event=exit reason=max_results_reached results_written={results_written}"
                                );
                                return Ok(());
                            }
                        }
                    }
                    ResultAnnouncementOutcome::AnnouncerSignatureFailed { posted_id } => {
                        println!(
                            "event=skip posted_id={posted_id} reason=announcer_signature_fail"
                        );
                    }
                    ResultAnnouncementOutcome::SchemaMalformed { posted_id, message } => {
                        println!(
                            "event=skip posted_id={posted_id} reason=schema_malformed:{message}"
                        );
                    }
                    ResultAnnouncementOutcome::FilteredOut { posted_id } => {
                        println!(
                            "event=skip posted_id={posted_id} reason=posted_id_not_in_accept_set"
                        );
                    }
                    ResultAnnouncementOutcome::SnipFetchFailed { posted_id, message } => {
                        println!(
                            "event=skip posted_id={posted_id} reason=snip_fetch_failed:{message}"
                        );
                    }
                    ResultAnnouncementOutcome::LinkParseFailed { posted_id, message } => {
                        println!(
                            "event=skip posted_id={posted_id} reason=link_parse_failed:{message}"
                        );
                    }
                    ResultAnnouncementOutcome::LinkSchemaInvalid { posted_id, message } => {
                        println!(
                            "event=skip posted_id={posted_id} reason=link_schema:{message}"
                        );
                    }
                    ResultAnnouncementOutcome::LinkContributorSignatureFailed { posted_id } => {
                        println!(
                            "event=skip posted_id={posted_id} reason=link_contributor_signature_fail"
                        );
                    }
                    ResultAnnouncementOutcome::LinkDrift { posted_id, field } => {
                        println!(
                            "event=skip posted_id={posted_id} reason=link_drift:{field}"
                        );
                    }
                }
            }
            std::thread::sleep(Duration::from_secs(args.poll_interval_secs));
        }
    })
    .await
    .map_err(|e| anyhow!("watch-network-results join: {e}"))?;
    run_result
}

// ── Stage 12.3 — session subcommand handlers ─────────────────────────────

/// Shared helper: open an OmniNet, wait for first peer, return an
/// Arc-wrapped handle ready for an OmniNetRelay clone. Used by every
/// 12.3 publish-and-broadcast subcommand below.
async fn open_omninet_with_peer_wait(
    listen_port: u16,
    peer: Vec<String>,
    peer_wait_secs: u64,
    mesh_stabilize_ms: u64,
) -> Result<(std::sync::Arc<tokio::sync::Mutex<omni_net::OmniNet>>, tokio::runtime::Handle)>
{
    use omni_net::OmniNet;
    use omni_types::config::NetConfig;
    let net_config = NetConfig {
        listen_port,
        bootstrap_peers: peer,
        ..NetConfig::default()
    };
    let mut net = OmniNet::new(net_config)
        .await
        .map_err(|e| anyhow!("OmniNet::new: {e}"))?;
    if peer_wait_secs > 0 {
        wait_for_first_peer(&mut net, peer_wait_secs, mesh_stabilize_ms).await?;
    }
    let net = std::sync::Arc::new(tokio::sync::Mutex::new(net));
    let handle = tokio::runtime::Handle::current();
    Ok((net, handle))
}

async fn run_open_session(args: OpenSessionArgs) -> Result<()> {
    use omni_contributor::canonical::{
        execution_session_signing_input, hex_lower, net_session_opened_signing_input,
        session_id_hex,
    };
    use omni_contributor::posted::PostedJob;
    use omni_contributor::{
        ContributorRelay, CoordinatorSigner, ExecutionSession,
        NetworkSessionOpenedAnnouncement, OmniNetRelay, NET_SCHEMA_VERSION,
        SESSION_SCHEMA_VERSION,
    };

    // 1. Load + validate the PostedJob.
    let posted_bytes = std::fs::read(&args.posted_job)
        .with_context(|| format!("read posted-job: {}", args.posted_job.display()))?;
    let posted: PostedJob = serde_json::from_slice(&posted_bytes)
        .with_context(|| format!("parse posted-job: {}", args.posted_job.display()))?;
    posted
        .validate_schema()
        .map_err(|e| anyhow!("invalid PostedJob: {e}"))?;

    // 2. Build + sign the ExecutionSession.
    let coord = CoordinatorSigner::from_seed_file(&args.coordinator_seed)?;
    let created_at_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let mut session = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: posted.posted_id.clone(),
        job_hash: posted.job_hash.clone(),
        model_hash: posted.model_hash.clone(),
        tokenizer_hash: None,
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc,
        expires_at_utc: args.expires_at_utc,
        coordinator_signature_hex: String::new(),
    };
    session.session_id = session_id_hex(&session)?;
    let sig_input = execution_session_signing_input(&session)?;
    session.coordinator_signature_hex = coord.sign_hex(&sig_input);
    session
        .validate_schema()
        .map_err(|e| anyhow!("invalid ExecutionSession: {e}"))?;

    // 3. Publish session JSON to SNIP.
    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session_json = serde_json::to_vec_pretty(&session)?;
    let session_root = omni_contributor::snip::publish_bytes(
        &snip,
        &session_json,
        "session",
    )
    .map_err(|e| anyhow!("snip publish session: {e}"))?;
    let session_root_hex = format!("0x{}", hex_lower(session_root.as_bytes()));

    // 4. Build + sign the network announcement.
    let mut ann = NetworkSessionOpenedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        execution_session_snip_root: session_root_hex.clone(),
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        announced_at_utc: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let ann_sig = net_session_opened_signing_input(&ann)?;
    ann.announcer_signature_hex = coord.sign_hex(&ann_sig);

    // 5. Open OmniNet + peer-wait + broadcast.
    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        args.peer_wait_secs,
        args.mesh_stabilize_ms,
    )
    .await?;
    let mut relay = OmniNetRelay::new(net.clone(), handle);
    relay
        .publish_session_opened(&ann)
        .map_err(|e| anyhow!("publish: {e}"))?;
    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }

    println!("session_id={}", session.session_id);
    println!("execution_session_snip_root={}", session_root_hex);
    println!("opened=true");
    Ok(())
}

/// Fetch + verify the `ExecutionSession` at the given SNIP root.
/// Returns the parsed session.
fn fetch_and_verify_session<A: omni_store::SnipV2Adapter + ?Sized>(
    adapter: &A,
    snip_root_hex: &str,
) -> Result<omni_contributor::ExecutionSession> {
    use omni_contributor::{verify_execution_session, ExecutionSession};
    use omni_types::phase5::SnipV2ObjectId;

    let root = SnipV2ObjectId::from_hex(snip_root_hex)
        .map_err(|e| anyhow!("bad session snip root: {e:?}"))?;
    let bytes = omni_contributor::snip::fetch_bytes(adapter, &root)
        .map_err(|e| anyhow!("snip fetch session: {e}"))?;
    let session: ExecutionSession = serde_json::from_slice(&bytes)
        .map_err(|e| anyhow!("parse session: {e}"))?;
    let out = verify_execution_session(&session);
    if !out.is_ok() {
        return Err(anyhow!("session verify failed: {out:?}"));
    }
    Ok(session)
}

async fn run_join_session(args: JoinSessionArgs) -> Result<()> {
    use omni_contributor::canonical::{
        contributor_join_signing_input, hex_lower, net_join_signing_input,
    };
    use omni_contributor::{
        ContributorJoin, ContributorRelay, ContributorSigner,
        NetworkContributorJoinedAnnouncement, OmniNetRelay, NET_SCHEMA_VERSION,
        SESSION_SCHEMA_VERSION,
    };

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.execution_session_snip_root)?;

    let contrib = ContributorSigner::from_seed_file(&args.contributor_seed)?;
    let joined_at_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let mut join = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: args.available_ram_bytes,
        max_input_tokens: args.max_input_tokens,
        max_output_tokens: args.max_output_tokens,
        supported_work_unit_kinds: args
            .supported_work_unit_kinds
            .into_iter()
            .map(Into::into)
            .collect(),
        runner_kind: args.runner_kind,
        joined_at_utc,
        contributor_signature_hex: String::new(),
    };
    let sig_input = contributor_join_signing_input(&join)?;
    join.contributor_signature_hex = contrib.sign_hex(&sig_input);
    join.validate_schema()
        .map_err(|e| anyhow!("invalid ContributorJoin: {e}"))?;

    let join_json = serde_json::to_vec_pretty(&join)?;
    let join_root = omni_contributor::snip::publish_bytes(&snip, &join_json, "join")
        .map_err(|e| anyhow!("snip publish join: {e}"))?;
    let join_root_hex = format!("0x{}", hex_lower(join_root.as_bytes()));

    let mut ann = NetworkContributorJoinedAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        contributor_join_snip_root: join_root_hex.clone(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        announcer_pubkey_hex: contrib.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let ann_sig = net_join_signing_input(&ann)?;
    ann.announcer_signature_hex = contrib.sign_hex(&ann_sig);

    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        args.peer_wait_secs,
        args.mesh_stabilize_ms,
    )
    .await?;
    let mut relay = OmniNetRelay::new(net.clone(), handle);
    relay
        .publish_contributor_joined(&ann)
        .map_err(|e| anyhow!("publish: {e}"))?;
    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }

    println!("session_id={}", session.session_id);
    println!("contributor_join_snip_root={}", join_root_hex);
    println!("joined=true");
    Ok(())
}

#[derive(serde::Deserialize)]
struct AssignmentSpec {
    contributor_pubkey_hex: String,
    stage_index: u32,
    work_kind: omni_contributor::WorkKind,
    expected_work_units: u64,
    expected_work_unit_kind: omni_contributor::result::WorkUnitKind,
    /// Optional override; if absent, the CLI fills in `now()`.
    #[serde(default)]
    assigned_at_utc: Option<String>,
}

async fn run_assign_work(args: AssignWorkArgs) -> Result<()> {
    use omni_contributor::canonical::{
        assignment_id_hex, hex_lower, net_assign_signing_input, work_assignment_signing_input,
    };
    use omni_contributor::{
        ContributorRelay, CoordinatorSigner, NetworkWorkAssignedAnnouncement, OmniNetRelay,
        WorkAssignment, NET_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
    };

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.execution_session_snip_root)?;

    let coord = CoordinatorSigner::from_seed_file(&args.coordinator_seed)?;
    if coord.pubkey_hex() != session.coordinator_pubkey_hex {
        return Err(anyhow!(
            "coordinator_seed pubkey does not match session.coordinator_pubkey_hex"
        ));
    }

    let spec_bytes = std::fs::read(&args.assignments_file)
        .with_context(|| format!("read assignments-file: {}", args.assignments_file.display()))?;
    let specs: Vec<AssignmentSpec> = serde_json::from_slice(&spec_bytes)
        .with_context(|| format!("parse assignments-file: {}", args.assignments_file.display()))?;
    if specs.is_empty() {
        return Err(anyhow!("assignments-file must contain at least one entry"));
    }

    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        args.peer_wait_secs,
        args.mesh_stabilize_ms,
    )
    .await?;
    let mut relay = OmniNetRelay::new(net.clone(), handle);

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    for spec in specs {
        let mut a = WorkAssignment {
            schema_version: SESSION_SCHEMA_VERSION,
            session_id: session.session_id.clone(),
            assignment_id: String::new(),
            stage_index: spec.stage_index,
            contributor_pubkey_hex: spec.contributor_pubkey_hex,
            work_kind: spec.work_kind,
            expected_work_units: spec.expected_work_units,
            expected_work_unit_kind: spec.expected_work_unit_kind,
            assigned_at_utc: spec.assigned_at_utc.unwrap_or_else(|| now_utc.clone()),
            coordinator_signature_hex: String::new(),
        };
        a.assignment_id = assignment_id_hex(&a)?;
        let sig_input = work_assignment_signing_input(&a)?;
        a.coordinator_signature_hex = coord.sign_hex(&sig_input);
        a.validate_schema()
            .map_err(|e| anyhow!("invalid WorkAssignment: {e}"))?;

        let json = serde_json::to_vec_pretty(&a)?;
        let root = omni_contributor::snip::publish_bytes(&snip, &json, "assignment")
            .map_err(|e| anyhow!("snip publish assignment: {e}"))?;
        let root_hex = format!("0x{}", hex_lower(root.as_bytes()));

        let mut ann = NetworkWorkAssignedAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            work_assignment_snip_root: root_hex.clone(),
            session_id: session.session_id.clone(),
            assignment_id: a.assignment_id.clone(),
            contributor_pubkey_hex: a.contributor_pubkey_hex.clone(),
            announced_at_utc: chrono::Utc::now()
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            announcer_pubkey_hex: coord.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let ann_sig = net_assign_signing_input(&ann)?;
        ann.announcer_signature_hex = coord.sign_hex(&ann_sig);

        relay
            .publish_work_assigned(&ann)
            .map_err(|e| anyhow!("publish: {e}"))?;
        println!(
            "assignment_id={} work_assignment_snip_root={}",
            a.assignment_id, root_hex
        );
    }

    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }
    println!("assigned=true");
    Ok(())
}

async fn run_assignment(args: RunAssignmentArgs) -> Result<()> {
    use omni_contributor::canonical::{
        canonical_partial_result_bytes, hex_lower, net_partial_signing_input,
        partial_result_signing_input,
    };
    use omni_contributor::runner::{ExternalCommandRunner, StubRunner};
    use omni_contributor::{
        ContributorRelay, ContributorSigner, InferenceRunner,
        NetworkPartialResultAnnouncement, OmniNetRelay, PartialContributorResult,
        WorkAssignment, NET_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
    };
    use omni_contributor::result::{MeasuredAccounting, StageContribution};
    use omni_types::phase5::SnipV2ObjectId;

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.execution_session_snip_root)?;

    // Fetch + parse + structurally-validate the assignment.
    let asn_root = SnipV2ObjectId::from_hex(&args.assignment_snip_root)
        .map_err(|e| anyhow!("bad assignment snip root: {e:?}"))?;
    let asn_bytes = omni_contributor::snip::fetch_bytes(&snip, &asn_root)
        .map_err(|e| anyhow!("snip fetch assignment: {e}"))?;
    let assignment: WorkAssignment = serde_json::from_slice(&asn_bytes)
        .map_err(|e| anyhow!("parse assignment: {e}"))?;
    assignment
        .validate_schema()
        .map_err(|e| anyhow!("invalid WorkAssignment: {e}"))?;
    if assignment.session_id != session.session_id {
        return Err(anyhow!(
            "assignment.session_id != session.session_id (drift)"
        ));
    }

    let contrib = ContributorSigner::from_seed_file(&args.contributor_seed)?;
    if contrib.pubkey_hex() != assignment.contributor_pubkey_hex {
        return Err(anyhow!(
            "contributor_seed pubkey does not match assignment.contributor_pubkey_hex"
        ));
    }

    // Build the runner. The session's job_hash is the manifest's role
    // for the existing runner trait; Stage 12.3 keeps runner I/O
    // opaque — the partial artifact is whatever the runner returns.
    // For 12.3 a runner is invoked with an empty manifest path +
    // empty input bytes (operators wire real inputs through
    // `--external-arg`s on the external runner if needed).
    let empty_manifest = std::path::PathBuf::from("/dev/null");
    let runner_output = match args.runner {
        RunnerChoice::Stub => {
            let bytes = std::fs::read(
                args.stub_response
                    .as_ref()
                    .ok_or_else(|| anyhow!("--stub-response required for stub runner"))?,
            )?;
            let r = StubRunner::new(
                contrib.pubkey_hex(),
                bytes,
                args.stub_input_tokens,
                args.stub_output_tokens,
            );
            r.run(&empty_manifest, b"")
                .map_err(|e| anyhow!("runner: {e}"))?
        }
        RunnerChoice::External => {
            let mut r = ExternalCommandRunner::new(
                args.external_command
                    .ok_or_else(|| anyhow!("--external-command required for external runner"))?,
            );
            r.extra_args = args.external_args;
            r.env_allowlist = args.external_env_allow;
            r.run(&empty_manifest, b"")
                .map_err(|e| anyhow!("runner: {e}"))?
        }
    };

    // Publish the runner's response bytes as the partial artifact.
    let artifact_root = omni_contributor::snip::publish_bytes(
        &snip,
        &runner_output.response_bytes,
        "partial-artifact",
    )
    .map_err(|e| anyhow!("snip publish artifact: {e}"))?;
    let artifact_root_hex = format!("0x{}", hex_lower(artifact_root.as_bytes()));
    let artifact_hash =
        hex_lower(blake3::hash(&runner_output.response_bytes).as_bytes());

    // Stage accounting: structural-only at 12.3. Use the
    // assignment's pre-declared work-unit kind.
    let measured = MeasuredAccounting {
        tokenizer_hash: session
            .tokenizer_hash
            .clone()
            .unwrap_or_else(|| "00".repeat(32)),
        input_token_count: runner_output.measured_input_tokens,
        output_token_count: runner_output.measured_output_tokens,
        total_base_units: runner_output
            .measured_input_tokens
            .saturating_add(runner_output.measured_output_tokens),
        stage_contributions: vec![StageContribution {
            contributor_pubkey_hex: contrib.pubkey_hex(),
            stage_label: format!("stage-{}", assignment.stage_index),
            work_unit_kind: assignment.expected_work_unit_kind,
            work_units: assignment.expected_work_units,
        }],
    };

    let mut partial = PartialContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: assignment.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        partial_artifact_snip_root: artifact_root_hex.clone(),
        partial_artifact_hash: artifact_hash,
        measured_accounting: measured,
        produced_at_utc: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        contributor_signature_hex: String::new(),
    };
    let sig_input = partial_result_signing_input(&partial)?;
    partial.contributor_signature_hex = contrib.sign_hex(&sig_input);
    partial
        .validate_schema()
        .map_err(|e| anyhow!("invalid PartialContributorResult: {e}"))?;

    let partial_json = serde_json::to_vec_pretty(&partial)?;
    let partial_root =
        omni_contributor::snip::publish_bytes(&snip, &partial_json, "partial")
            .map_err(|e| anyhow!("snip publish partial: {e}"))?;
    let partial_root_hex = format!("0x{}", hex_lower(partial_root.as_bytes()));
    let partial_canonical_hash = {
        let bytes = canonical_partial_result_bytes(&partial)?;
        hex_lower(blake3::hash(&bytes).as_bytes())
    };

    let mut ann = NetworkPartialResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        partial_result_snip_root: partial_root_hex.clone(),
        session_id: session.session_id.clone(),
        assignment_id: assignment.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        announced_at_utc: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        announcer_pubkey_hex: contrib.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let ann_sig = net_partial_signing_input(&ann)?;
    ann.announcer_signature_hex = contrib.sign_hex(&ann_sig);

    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        args.peer_wait_secs,
        args.mesh_stabilize_ms,
    )
    .await?;
    let mut relay = OmniNetRelay::new(net.clone(), handle);
    relay
        .publish_partial_result(&ann)
        .map_err(|e| anyhow!("publish: {e}"))?;
    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }

    println!("assignment_id={}", assignment.assignment_id);
    println!("partial_result_snip_root={}", partial_root_hex);
    println!("partial_artifact_snip_root={}", artifact_root_hex);
    println!("partial_canonical_hash={}", partial_canonical_hash);
    println!("partial=published");
    Ok(())
}

async fn run_aggregate_session(args: AggregateSessionArgs) -> Result<()> {
    use omni_contributor::canonical::{
        aggregated_result_signing_input, canonical_partial_result_bytes, hex_lower,
        net_aggregated_signing_input,
    };
    use omni_contributor::{
        verify_aggregated_result, AggregatedContributorResult, AggregatedPartialRef,
        ContributorRelay, CoordinatorSigner, NetworkAggregatedResultAnnouncement,
        OmniNetRelay, NET_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
    };
    use omni_types::phase5::SnipV2ObjectId;

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.execution_session_snip_root)?;

    let coord = CoordinatorSigner::from_seed_file(&args.coordinator_seed)?;
    if coord.pubkey_hex() != session.coordinator_pubkey_hex {
        return Err(anyhow!(
            "coordinator_seed pubkey does not match session.coordinator_pubkey_hex"
        ));
    }
    if args.join_snip_roots.is_empty() {
        return Err(anyhow!(
            "must supply at least one --join-snip-root; the aggregate verifier needs \
             the joined-pubkey set to prove assignments target joined contributors"
        ));
    }
    if args.assignment_snip_roots.len() != args.partial_snip_roots.len() {
        return Err(anyhow!(
            "must have same number of --assignment-snip-root and --partial-snip-root flags"
        ));
    }

    // Fetch every ContributorJoin. The verifier checks each join's
    // schema + binding + signature, but we already need them in
    // hand to build the joined-pubkey set the chain check runs over.
    let mut joins = Vec::with_capacity(args.join_snip_roots.len());
    for jhex in &args.join_snip_roots {
        let jroot = SnipV2ObjectId::from_hex(jhex)
            .map_err(|e| anyhow!("bad join snip root: {e:?}"))?;
        let jbytes = omni_contributor::snip::fetch_bytes(&snip, &jroot)
            .map_err(|e| anyhow!("snip fetch join: {e}"))?;
        let j: omni_contributor::ContributorJoin =
            serde_json::from_slice(&jbytes).map_err(|e| anyhow!("parse join: {e}"))?;
        joins.push(j);
    }

    // Fetch + validate all assignments and partials. We pair them
    // by index (caller's responsibility to keep order consistent).
    let mut assignments = Vec::with_capacity(args.assignment_snip_roots.len());
    let mut partials = Vec::with_capacity(args.partial_snip_roots.len());
    let mut partial_refs = Vec::with_capacity(args.partial_snip_roots.len());
    for (asn_hex, par_hex) in args
        .assignment_snip_roots
        .iter()
        .zip(args.partial_snip_roots.iter())
    {
        let asn_root = SnipV2ObjectId::from_hex(asn_hex)
            .map_err(|e| anyhow!("bad assignment snip root: {e:?}"))?;
        let asn_bytes = omni_contributor::snip::fetch_bytes(&snip, &asn_root)
            .map_err(|e| anyhow!("snip fetch assignment: {e}"))?;
        let asn: omni_contributor::WorkAssignment = serde_json::from_slice(&asn_bytes)
            .map_err(|e| anyhow!("parse assignment: {e}"))?;
        let par_root = SnipV2ObjectId::from_hex(par_hex)
            .map_err(|e| anyhow!("bad partial snip root: {e:?}"))?;
        let par_bytes = omni_contributor::snip::fetch_bytes(&snip, &par_root)
            .map_err(|e| anyhow!("snip fetch partial: {e}"))?;
        let par: omni_contributor::PartialContributorResult =
            serde_json::from_slice(&par_bytes).map_err(|e| anyhow!("parse partial: {e}"))?;
        let par_canonical_hash =
            hex_lower(blake3::hash(&canonical_partial_result_bytes(&par)?).as_bytes());
        partial_refs.push(AggregatedPartialRef {
            assignment_id: asn.assignment_id.clone(),
            stage_index: asn.stage_index,
            contributor_pubkey_hex: par.contributor_pubkey_hex.clone(),
            partial_snip_root: par_hex.clone(),
            partial_canonical_hash: par_canonical_hash,
        });
        assignments.push(asn);
        partials.push(par);
    }
    // Order partial_refs by stage_index for deterministic canonical bytes.
    partial_refs.sort_by_key(|r| r.stage_index);

    // Fetch the final ContributorResult to compute its canonical hash.
    let final_root = SnipV2ObjectId::from_hex(&args.final_result_snip_root)
        .map_err(|e| anyhow!("bad final-result snip root: {e:?}"))?;
    let final_bytes = omni_contributor::snip::fetch_bytes(&snip, &final_root)
        .map_err(|e| anyhow!("snip fetch final result: {e}"))?;
    let final_result: omni_contributor::ContributorResult =
        serde_json::from_slice(&final_bytes).map_err(|e| anyhow!("parse final result: {e}"))?;
    let final_canonical_hash = hex_lower(
        blake3::hash(&omni_contributor::canonical::canonical_result_bytes(&final_result)?)
            .as_bytes(),
    );

    let mut aggregate = AggregatedContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        final_result_snip_root: args.final_result_snip_root.clone(),
        final_result_canonical_hash: final_canonical_hash,
        partial_refs,
        aggregated_at_utc: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let sig_input = aggregated_result_signing_input(&aggregate)?;
    aggregate.coordinator_signature_hex = coord.sign_hex(&sig_input);

    // Full-chain verify: session → joins → assignments → partials →
    // aggregate. No --allow-incomplete in v1.
    let out = verify_aggregated_result(&session, &joins, &assignments, &partials, &aggregate);
    if !out.is_ok() {
        return Err(anyhow!("aggregate verify failed: {out:?}"));
    }

    let agg_json = serde_json::to_vec_pretty(&aggregate)?;
    let agg_root = omni_contributor::snip::publish_bytes(&snip, &agg_json, "aggregated")
        .map_err(|e| anyhow!("snip publish aggregated: {e}"))?;
    let agg_root_hex = format!("0x{}", hex_lower(agg_root.as_bytes()));

    let mut ann = NetworkAggregatedResultAnnouncement {
        schema_version: NET_SCHEMA_VERSION,
        aggregated_result_snip_root: agg_root_hex.clone(),
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        announced_at_utc: chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        announcer_pubkey_hex: coord.pubkey_hex(),
        announcer_signature_hex: String::new(),
    };
    let ann_sig = net_aggregated_signing_input(&ann)?;
    ann.announcer_signature_hex = coord.sign_hex(&ann_sig);

    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        args.peer_wait_secs,
        args.mesh_stabilize_ms,
    )
    .await?;
    let mut relay = OmniNetRelay::new(net.clone(), handle);
    relay
        .publish_aggregated_result(&ann)
        .map_err(|e| anyhow!("publish: {e}"))?;
    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }

    println!("session_id={}", session.session_id);
    println!("aggregated_result_snip_root={}", agg_root_hex);
    println!("aggregated=true");
    Ok(())
}

async fn run_watch_sessions(args: WatchSessionsArgs) -> Result<()> {
    use omni_contributor::{ContributorRelay, ExecutionSession, OmniNetRelay};
    use std::collections::{HashMap, HashSet};
    use std::time::Duration;

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        /* peer_wait_secs = */ 0,
        /* mesh_stabilize_ms = */ 0,
    )
    .await?;
    std::fs::create_dir_all(&args.out_dir)?;
    let posted_id_filter: HashSet<String> = args.posted_id.into_iter().collect();
    let session_id_filter: HashSet<String> = args.session_id.into_iter().collect();
    let mut polls_done: u64 = 0;

    let run_result = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut relay = OmniNetRelay::new(net, handle);
        // Session cache: needed so the assignment processor can
        // verify each assignment's coordinator signature against
        // the session's coordinator_pubkey_hex (assignments don't
        // carry their own coordinator pubkey).
        let mut sessions: HashMap<String, ExecutionSession> = HashMap::new();
        loop {
            if let Some(max) = args.max_polls {
                if polls_done >= max {
                    println!("event=exit reason=max_polls_reached");
                    return Ok(());
                }
            }
            polls_done += 1;
            for ann in relay
                .poll_sessions_opened()
                .map_err(|e| anyhow!("poll session-opened: {e}"))?
            {
                handle_session_opened(
                    &snip,
                    &args.out_dir,
                    &posted_id_filter,
                    &session_id_filter,
                    &ann,
                    &mut sessions,
                );
            }
            for ann in relay
                .poll_contributors_joined()
                .map_err(|e| anyhow!("poll joined: {e}"))?
            {
                handle_join(&snip, &args.out_dir, &session_id_filter, &ann);
            }
            for ann in relay
                .poll_work_assigned()
                .map_err(|e| anyhow!("poll assigned: {e}"))?
            {
                handle_assignment(
                    &snip,
                    &args.out_dir,
                    &session_id_filter,
                    &ann,
                    &sessions,
                );
            }
            for ann in relay
                .poll_partial_results()
                .map_err(|e| anyhow!("poll partial: {e}"))?
            {
                handle_partial(&snip, &args.out_dir, &session_id_filter, &ann);
            }
            for ann in relay
                .poll_aggregated_results()
                .map_err(|e| anyhow!("poll aggregated: {e}"))?
            {
                handle_aggregated(
                    &snip,
                    &args.out_dir,
                    &posted_id_filter,
                    &session_id_filter,
                    &ann,
                );
            }
            std::thread::sleep(Duration::from_secs(args.poll_interval_secs));
        }
    })
    .await
    .map_err(|e| anyhow!("watch-sessions join: {e}"))?;
    run_result
}

// ── watch-sessions per-topic handlers (in-process; print bare-stdout events).
//
// Each handler delegates to the corresponding `process_*_announcement`
// helper, which verifies the announcer signature, fetches the inner
// body from SNIP, schema-validates it, verifies the inner
// self-signature where possible, and drift-checks every shared field
// — only THEN does the handler write the body to disk.

fn write_session_artifact(
    out_dir: &std::path::Path,
    session_id: &str,
    leaf_dir: Option<&str>,
    filename: &str,
    bytes: &[u8],
) -> std::io::Result<std::path::PathBuf> {
    let mut dir = out_dir.join(session_id);
    if let Some(leaf) = leaf_dir {
        dir = dir.join(leaf);
    }
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(filename);
    std::fs::write(&path, bytes)?;
    Ok(path)
}

/// Pretty-print an [`omni_contributor::AnnouncementOutcome`] failure
/// to stdout. Returns true when the outcome was Verified (caller can
/// proceed to write the body) and false otherwise.
fn log_announcement_failure<T: std::fmt::Debug>(
    kind: &'static str,
    session_id: &str,
    outcome: &omni_contributor::AnnouncementOutcome<T>,
) -> bool {
    use omni_contributor::AnnouncementOutcome as O;
    match outcome {
        O::Verified { .. } => true,
        O::AnnouncementSchemaMalformed(s) => {
            println!(
                "event=skip kind={kind} session_id={session_id} reason=announcement_schema_malformed:{s}"
            );
            false
        }
        O::AnnouncerSignatureFailed => {
            println!(
                "event=skip kind={kind} session_id={session_id} reason=announcer_signature_fail"
            );
            false
        }
        O::SnipFetchFailed(s) => {
            println!(
                "event=skip kind={kind} session_id={session_id} reason=snip_fetch_failed:{s}"
            );
            false
        }
        O::BodyParseFailed(s) => {
            println!(
                "event=skip kind={kind} session_id={session_id} reason=body_parse_failed:{s}"
            );
            false
        }
        O::BodySchemaInvalid(s) => {
            println!(
                "event=skip kind={kind} session_id={session_id} reason=body_schema_invalid:{s}"
            );
            false
        }
        O::BodySignatureFailed => {
            println!(
                "event=skip kind={kind} session_id={session_id} reason=body_signature_fail"
            );
            false
        }
        O::DriftMismatch { field } => {
            println!(
                "event=skip kind={kind} session_id={session_id} reason=drift:{field}"
            );
            false
        }
    }
}

fn handle_session_opened<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    posted_id_filter: &std::collections::HashSet<String>,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkSessionOpenedAnnouncement,
    sessions: &mut std::collections::HashMap<String, omni_contributor::ExecutionSession>,
) {
    if !posted_id_filter.is_empty() && !posted_id_filter.contains(&ann.posted_id) {
        return;
    }
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    let outcome = omni_contributor::process_session_opened_announcement(ann, snip);
    if !log_announcement_failure("session_opened", &ann.session_id, &outcome) {
        return;
    }
    let session = match outcome {
        omni_contributor::AnnouncementOutcome::Verified { body } => body,
        _ => return,
    };
    let bytes = serde_json::to_vec_pretty(&session).unwrap_or_default();
    match write_session_artifact(out_dir, &session.session_id, None, "session.json", &bytes) {
        Ok(p) => {
            println!(
                "event=session_opened session_id={} path={}",
                session.session_id,
                p.display()
            );
            sessions.insert(session.session_id.clone(), session);
        }
        Err(e) => println!("event=error context=write_session message={e}"),
    }
}

fn handle_join<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkContributorJoinedAnnouncement,
) {
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    let outcome = omni_contributor::process_contributor_joined_announcement(ann, snip);
    if !log_announcement_failure("join", &ann.session_id, &outcome) {
        return;
    }
    let join = match outcome {
        omni_contributor::AnnouncementOutcome::Verified { body } => body,
        _ => return,
    };
    let bytes = serde_json::to_vec_pretty(&join).unwrap_or_default();
    let filename = format!("{}.json", join.contributor_pubkey_hex);
    match write_session_artifact(out_dir, &join.session_id, Some("joins"), &filename, &bytes) {
        Ok(p) => println!(
            "event=join session_id={} contributor_pubkey={} path={}",
            join.session_id,
            join.contributor_pubkey_hex,
            p.display()
        ),
        Err(e) => println!("event=error context=write_join message={e}"),
    }
}

fn handle_assignment<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkWorkAssignedAnnouncement,
    sessions: &std::collections::HashMap<String, omni_contributor::ExecutionSession>,
) {
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    // If we've seen the session, pass its coord pubkey so the
    // processor verifies the assignment's coord signature. Else
    // pass None (announcer sig + body schema + drift are still
    // checked; coord sig is checked when the chain is later
    // verified at aggregate time).
    let session_coord = sessions
        .get(&ann.session_id)
        .map(|s| s.coordinator_pubkey_hex.as_str());
    let outcome = omni_contributor::process_work_assigned_announcement(ann, snip, session_coord);
    if !log_announcement_failure("assignment", &ann.session_id, &outcome) {
        return;
    }
    let asn = match outcome {
        omni_contributor::AnnouncementOutcome::Verified { body } => body,
        _ => return,
    };
    let bytes = serde_json::to_vec_pretty(&asn).unwrap_or_default();
    let filename = format!("{}.json", asn.assignment_id);
    match write_session_artifact(out_dir, &asn.session_id, Some("assignments"), &filename, &bytes)
    {
        Ok(p) => println!(
            "event=assignment session_id={} assignment_id={} coord_sig_verified={} path={}",
            asn.session_id,
            asn.assignment_id,
            session_coord.is_some(),
            p.display()
        ),
        Err(e) => println!("event=error context=write_assignment message={e}"),
    }
}

fn handle_partial<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkPartialResultAnnouncement,
) {
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    let outcome = omni_contributor::process_partial_result_announcement(ann, snip);
    if !log_announcement_failure("partial", &ann.session_id, &outcome) {
        return;
    }
    let par = match outcome {
        omni_contributor::AnnouncementOutcome::Verified { body } => body,
        _ => return,
    };
    let bytes = serde_json::to_vec_pretty(&par).unwrap_or_default();
    let filename = format!("{}.json", par.assignment_id);
    match write_session_artifact(out_dir, &par.session_id, Some("partials"), &filename, &bytes) {
        Ok(p) => println!(
            "event=partial session_id={} assignment_id={} path={}",
            par.session_id,
            par.assignment_id,
            p.display()
        ),
        Err(e) => println!("event=error context=write_partial message={e}"),
    }
}

fn handle_aggregated<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    posted_id_filter: &std::collections::HashSet<String>,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkAggregatedResultAnnouncement,
) {
    if !posted_id_filter.is_empty() && !posted_id_filter.contains(&ann.posted_id) {
        return;
    }
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    let outcome = omni_contributor::process_aggregated_result_announcement(ann, snip);
    if !log_announcement_failure("aggregated", &ann.session_id, &outcome) {
        return;
    }
    let agg = match outcome {
        omni_contributor::AnnouncementOutcome::Verified { body } => body,
        _ => return,
    };
    let bytes = serde_json::to_vec_pretty(&agg).unwrap_or_default();
    match write_session_artifact(out_dir, &agg.session_id, None, "aggregated.json", &bytes) {
        Ok(p) => println!(
            "event=aggregated session_id={} path={}",
            agg.session_id,
            p.display()
        ),
        Err(e) => println!("event=error context=write_aggregated message={e}"),
    }
}
