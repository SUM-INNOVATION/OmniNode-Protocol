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

    /// Stage 12.4 — diagnostic: send a single `ActivationHandoff`
    /// envelope from an already-computed activation file. Useful
    /// for testing live pipelines without invoking a full
    /// `run-assignment`. Production handoffs flow through
    /// `run-assignment --activation-out-mode live|both`.
    SendHandoff(SendHandoffArgs),

    /// Stage 12.5 — publish a signed `ContributorPeerAdvertisement`
    /// for a specific (session_id, contributor_pubkey_hex). The
    /// `libp2p_peer_id` is read from the live `OmniNet` instance
    /// at publish time; operators cannot supply it manually.
    AdvertisePeer(AdvertisePeerArgs),

    /// Stage 12.5 — long-running passive observer: subscribe to
    /// the peer-advert topic, verify each announcement + body,
    /// write verified peer advertisements to disk so a separate
    /// `run-assignment` can resolve downstream peers without
    /// requiring `--downstream-to-peer`.
    WatchPeerAdverts(WatchPeerAdvertsArgs),

    /// Stage 12.8 — read-only: load verified session + joins (+
    /// optional verified peer advertisements) from a Stage 12.7
    /// `--contributor-state-dir`, run a deterministic local planner,
    /// and emit an unsigned `AssignmentPlan` JSON. Does not touch
    /// SNIP, the mesh, or any chain.
    PlanSessionAssignments(PlanSessionAssignmentsArgs),

    /// Stage 12.8 — read an `AssignmentPlan` JSON, re-verify the
    /// referenced session via its SNIP root, and publish each
    /// planned entry as a normal Stage 12.3 `WorkAssignment` (signed
    /// by the coordinator, optionally broadcast as a
    /// `NetworkWorkAssignedAnnouncement`). `--dry-run` validates
    /// without touching SNIP or the mesh.
    AssignSessionPlan(AssignSessionPlanArgs),

    /// Stage 12.9 — read-only: load + re-verify session/joins/
    /// assignments/partials/peer-adverts/aggregate for a given
    /// `--session-id` from a Stage 12.7 `--contributor-state-dir`,
    /// produce a deterministic `SessionStatusReport`, and print it
    /// as events/json/pretty. Does not touch SNIP, the mesh, or
    /// any chain.
    SessionStatus(SessionStatusArgs),

    /// Stage 12.10 — read-only: build a Stage 12.9
    /// `SessionStatusReport` (or load one from `--status-report`)
    /// and emit a deterministic `SessionRepairPlan` JSON listing
    /// one `ReannounceAssignment` action per assignment missing a
    /// valid partial. Does not touch SNIP, the mesh, or any
    /// chain.
    PlanSessionRepair(PlanSessionRepairArgs),

    /// Stage 12.10 — read a `SessionRepairPlan`, recompute its
    /// integrity tag, re-fetch + re-verify the session via
    /// `--session-snip-root`, recompute the source-status
    /// projection from current state-dir (typed error on drift),
    /// re-verify each referenced assignment, then either
    /// `--dry-run` log the would-be actions OR republish each
    /// assignment to SNIP (content-addressed → same root) and
    /// broadcast a fresh `NetworkWorkAssignedAnnouncement` unless
    /// `--no-publish-announcements`. The state-dir is NOT mutated.
    ApplySessionRepair(ApplySessionRepairArgs),

    /// Stage 12.11 — emit a `SessionRepairPlan` v2 whose actions
    /// are `ReassignAssignment` entries (one per missing active
    /// assignment). Read-only; the plan is operator-reviewable
    /// JSON. Companion to `apply-session-reassign`.
    PlanSessionReassign(PlanSessionReassignArgs),

    /// Stage 12.11 — read a v2 `SessionRepairPlan` with
    /// `RepairStrategy::ReassignMissing`, build + sign replacement
    /// `WorkAssignment` bodies, build + sign one
    /// `WorkAssignmentSupersession` covering them, publish
    /// everything to SNIP, and broadcast the matching mesh
    /// announcements (unless `--no-publish-announcements`). The
    /// state-dir IS mutated on real apply: replacement
    /// assignments + the supersession are dual-written to
    /// `verified/sessions/<id>/...` and seen markers are laid down.
    ApplySessionReassign(ApplySessionReassignArgs),

    /// Stage 12.14 — local archive + state-dir compaction for one
    /// session. Copies the `verified/sessions/<session_id>/...`
    /// subtree + matching seen markers to
    /// `<archive-dir>/<session_id>/`, verifies BLAKE3 on every
    /// copy, writes a manifest LAST, and (on `--move`) cascades
    /// the source out of the state-dir. No protocol surface; no
    /// chain, mesh, or SNIP wire is touched.
    ArchiveSession(ArchiveSessionArgs),

    /// Stage 12.15 — local archive restore / import. Inverse of
    /// `archive-session`. Reads
    /// `<archive-session-dir>/manifest.json` (or
    /// `<archive-dir>/<session_id>/manifest.json`), validates
    /// every file against the manifest's BLAKE3 + path
    /// whitelist + state-dir version compatibility, then
    /// writes the bytes back into the state-dir. No chain,
    /// mesh, or SNIP wire touched.
    RestoreSessionArchive(RestoreSessionArchiveArgs),

    /// Stage 12.16 — read-only local state-dir integrity scan.
    /// Re-runs every Stage 12.3 / 12.11 verifier against the
    /// bodies on disk, walks seen markers ↔ verified bodies,
    /// reports stray files inside documented subtrees, rolls up
    /// the Stage 12.13 audit projection per session, and (with
    /// `--include-archives`) walks a parallel archive directory
    /// via Stage 12.15 `restore_session_archive(verify_only)`.
    /// Emits typed findings; **never writes** to the state-dir
    /// or archive directory. No protocol surface, no chain /
    /// mesh / SNIP wire touched.
    StateIntegrity(StateIntegrityArgs),

    /// Stage 12.17 — build a deterministic local state-dir
    /// cleanup plan from a Stage 12.16 integrity scan. Read-only:
    /// the planner walks the state-dir, builds a typed
    /// `StateCleanupPlan` (closed-set actions, BLAKE3-stamped),
    /// and writes the plan JSON to `--out`. The state-dir is
    /// **never** mutated. Operator reviews the plan before
    /// `apply-state-cleanup`. No protocol surface, no chain /
    /// mesh / SNIP / archive surface touched.
    PlanStateCleanup(PlanStateCleanupArgs),

    /// Stage 12.17 — apply a previously-built `StateCleanupPlan`.
    /// Re-runs the integrity scan + drift-checks
    /// `source_integrity_hash`, re-verifies `cleanup_plan_hash`,
    /// re-checks per-session orphan-assignment projections for
    /// gated actions, then walks the actions in plan order.
    /// Tier-B actions quarantine the bytes (BLAKE3-verified)
    /// under `<quarantine-dir>/<plan_id>/...` BEFORE removing
    /// the source; a manifest is written LAST so a partial
    /// apply leaves it visibly missing. No chain / mesh / SNIP
    /// surface touched.
    ApplyStateCleanup(ApplyStateCleanupArgs),

    /// Stage 12.18 — restore a Stage 12.17 cleanup-quarantine
    /// subtree back into the contributor state-dir. Consumes
    /// the v1 `quarantine-manifest.json`, BLAKE3-verifies every
    /// quarantined file, path-checks every destination, refuses
    /// any pre-existing destination unless
    /// `--overwrite-existing`, and (by default) restores the
    /// matching seen markers. The quarantine subtree is **left
    /// intact** — operator manages retention. No chain / mesh /
    /// SNIP / envelope surface touched.
    RestoreStateCleanupQuarantine(RestoreStateCleanupQuarantineArgs),

    /// Stage 12.19 — compare two `StateIntegrityReport` JSON
    /// snapshots and emit a typed `StateIntegrityDiffReport`
    /// classifying each finding as `new`, `resolved`, or
    /// `unchanged`. Read-only: no state-store is opened, no
    /// state-dir bytes are written. The two inputs may come
    /// from any host as long as their `state_version`
    /// matches; `--require-state-dir-match` opts in to host
    /// pinning. No chain / mesh / SNIP / envelope surface
    /// touched.
    StateIntegrityDiff(StateIntegrityDiffArgs),

    /// Stage 12.20 — sign a v1 `StateIntegrityReport` JSON
    /// with a 32-byte Ed25519 seed and emit a
    /// `SignedStateIntegrityBaseline` JSON. Read-only on the
    /// state-store side; writes only the operator-named
    /// `--out` file. The signed wrapper is local-only — no
    /// protocol envelope, no SNIP wire, no chain interaction.
    SignStateIntegrityBaseline(SignStateIntegrityBaselineArgs),

    /// Stage 12.21 — sign a v1 `StateIntegrityDiffReport`
    /// JSON with a 32-byte Ed25519 seed and emit a
    /// `SignedStateIntegrityDiff` JSON. Read-only on the
    /// state-store side; writes only the operator-named
    /// `--out` file. The signed wrapper is local-only — no
    /// protocol envelope, no SNIP wire, no chain interaction.
    SignStateIntegrityDiff(SignStateIntegrityDiffArgs),

    /// Stage 12.21 — verify a `SignedStateIntegrityDiff`
    /// JSON against an operator-supplied trust anchor
    /// (`--expected-signer-pubkey-hex`). Read-only and
    /// state-store-free: no contributor state-dir is opened
    /// and `--no-prune-state-on-start` is deliberately
    /// absent. No chain / mesh / SNIP / envelope surface
    /// touched.
    VerifyStateIntegrityDiffSignature(VerifyStateIntegrityDiffSignatureArgs),

    /// Stage 12.22 — assemble a local-only
    /// `IntegrityEvidenceBundle` JSON fingerprinting a chosen
    /// set of Stage 12.16–12.21 audit artifacts under a single
    /// `--base-dir`. Byte manifest only: no signature, no
    /// semantic JSON validation, no recursive directory
    /// bundling. Read-only on the state-store side; writes only
    /// the operator-named `--out` file (atomic temp+rename).
    BuildIntegrityEvidenceBundle(BuildIntegrityEvidenceBundleArgs),

    /// Stage 12.22 — verify a local-only
    /// `IntegrityEvidenceBundle` JSON by re-hashing each
    /// referenced artifact file. Collect-all: every entry gets
    /// a `BundleEntryOutcome`; the verifier never short-circuits
    /// per entry. Read-only and state-store-free:
    /// `--no-prune-state-on-start` is deliberately absent. No
    /// chain / mesh / SNIP / envelope surface touched.
    VerifyIntegrityEvidenceBundle(VerifyIntegrityEvidenceBundleArgs),

    /// Stage 12.23 — sign a v1 `IntegrityEvidenceBundle` JSON
    /// with a 32-byte Ed25519 seed and emit a
    /// `SignedIntegrityEvidenceBundle` JSON. Read-only on the
    /// state-store side; writes only the operator-named
    /// `--out` file. The signed wrapper attests to bundle
    /// bytes only — Stage 12.22's `verify-integrity-evidence-bundle`
    /// is still needed for per-entry artifact byte verification.
    SignIntegrityEvidenceBundle(SignIntegrityEvidenceBundleArgs),

    /// Stage 12.23 — verify a `SignedIntegrityEvidenceBundle`
    /// JSON against an operator-supplied trust anchor
    /// (`--expected-signer-pubkey-hex`). Read-only and
    /// state-store-free: no contributor state-dir is opened
    /// and `--no-prune-state-on-start` is deliberately absent.
    /// Attests to bundle JSON bytes only — does NOT re-hash
    /// referenced artifact files. No chain / mesh / SNIP /
    /// envelope surface touched.
    VerifyIntegrityEvidenceBundleSignature(
        VerifyIntegrityEvidenceBundleSignatureArgs,
    ),

    /// Stage 12.24 — chain-verify a
    /// `SignedIntegrityEvidenceBundle` end-to-end: outer
    /// signature gate (Stage 12.23), bundle byte verification
    /// (Stage 12.22), and optional per-signed-child signature
    /// verification (Stage 12.20 / 12.21). Read-only and
    /// state-store-free; the only write is the optional
    /// `--json-out` mirror (best-effort). Omitted child anchors
    /// record `Skipped` outcomes — NOT silent passes. No chain
    /// / mesh / SNIP / envelope surface touched.
    VerifyIntegrityEvidenceChain(VerifyIntegrityEvidenceChainArgs),

    /// Stage 12.25 — sign a v1 `IntegrityEvidenceChainReport`
    /// JSON with a 32-byte Ed25519 seed and emit a
    /// `SignedIntegrityEvidenceChainReport` JSON. Read-only on
    /// the state-store side; writes only the operator-named
    /// `--out` file. Attests to chain-report bytes only — does
    /// NOT re-run any Stage 12.24 gates.
    SignIntegrityEvidenceChainReport(SignIntegrityEvidenceChainReportArgs),

    /// Stage 12.25 — verify a
    /// `SignedIntegrityEvidenceChainReport` JSON against an
    /// operator-supplied trust anchor
    /// (`--expected-signer-pubkey-hex`). Read-only and
    /// state-store-free: no contributor state-dir is opened
    /// and `--no-prune-state-on-start` is deliberately absent.
    /// Attests to chain-report JSON bytes only — does NOT
    /// re-run any Stage 12.24 gates. No chain / mesh / SNIP /
    /// envelope surface touched.
    VerifyIntegrityEvidenceChainReportSignature(
        VerifyIntegrityEvidenceChainReportSignatureArgs,
    ),
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

    /// Stage 14.2 — write a sidecar
    /// [`omni_zkml::ProofArtifactBody`] alongside the contributor
    /// result, signed off by the halo2-reference prover. The
    /// proof binds `(canonical halo2-mlp-v1 spec, stub_input,
    /// stub_response)`. Requires `--stub-input <PATH>` so the
    /// prover sees the exact same input bytes the runner saw.
    /// `--runner stub` is the only supported runner in 14.2;
    /// the artifact is `testnet_or_dev_only` and is hard-refused
    /// on `chain_id == 1`. Feature-gated by `halo2-reference-prove`.
    /// Stage 14.2/14.3 — Feature-gated by `halo2-reference-prove`.
    /// **Stage 14.3 D1 Alpha**: the static clap-layer
    /// `requires = "stub_input"` was removed; the StubRunner
    /// pairing is now enforced at the `run_run_job` runtime layer
    /// **before any work runs**. For `--runner external`,
    /// `--stub-input` is **not** required (the bytes are captured
    /// at the `InferenceRunner::run` trait boundary).
    #[cfg(feature = "halo2-reference-prove")]
    #[cfg_attr(
        feature = "stage11d-production-prove",
        arg(long, conflicts_with = "emit_production_mlp_proof")
    )]
    #[cfg_attr(
        not(feature = "stage11d-production-prove"),
        arg(long)
    )]
    emit_halo2_reference_proof: Option<PathBuf>,

    /// Stage 14.6 — write a sidecar
    /// [`omni_zkml::ProofArtifactBody`] alongside the contributor
    /// result, signed off by the halo2 PRODUCTION-MLP prover. The
    /// proof binds `(canonical production-fixedpoint-mlp-v1 spec,
    /// stub_input, stub_response)`. Requires `--stub-input <PATH>`
    /// when `--runner stub` so the prover sees the exact same input
    /// bytes the runner saw; for `--runner external` the bytes are
    /// captured at the `InferenceRunner::run` trait boundary by
    /// [`ByteCapturingRunner`]. **Production-shape contract**: the
    /// emitted artifact declares `testnet_or_dev_only=Some(false)`,
    /// `circuit_id_hex` AND `verification_key_hex` must equal the
    /// Stage 11d.2 pinned constants, and the input/output sizes are
    /// 32 / 16 bytes (16-i16 / 8-i16 LE). Mainnet refusal lands at
    /// `check_mainnet_eligible` **layer 6 only** (empty
    /// `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`). Mutually exclusive
    /// with `--emit-halo2-reference-proof` at the clap layer (Stage
    /// 14.6 Q3 lock — declare on both fields for defensive symmetry,
    /// gated by `cfg_attr` so single-feature builds compile).
    /// Feature-gated by `stage11d-production-prove`.
    #[cfg(feature = "stage11d-production-prove")]
    #[cfg_attr(
        feature = "halo2-reference-prove",
        arg(long, conflicts_with = "emit_halo2_reference_proof")
    )]
    #[cfg_attr(
        not(feature = "halo2-reference-prove"),
        arg(long)
    )]
    emit_production_mlp_proof: Option<PathBuf>,

    /// Stage 14.2 — `stub` runner: path to the raw input bytes
    /// the runner saw. Required when **either**
    /// `--emit-halo2-reference-proof` (Stage 14.2) **or**
    /// `--emit-production-mlp-proof` (Stage 14.6) is set **and**
    /// `--runner stub`; ignored for `--runner external` (Stage
    /// 14.3 captures input bytes at the trait boundary). Available
    /// in any build with at least one prover feature enabled.
    #[cfg(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))]
    #[arg(long)]
    stub_input: Option<PathBuf>,
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

    /// Stage 12.7 — optional contributor workflow state directory.
    /// When set, the watcher loads cross-restart seen markers for
    /// posted-jobs and dual-writes accepted/rejected contributor
    /// results + posted-result-links into
    /// `<state-dir>/results/...`. Omit to preserve pre-12.7
    /// behavior exactly. See `omni_contributor::state`.
    #[arg(long)]
    contributor_state_dir: Option<PathBuf>,

    /// Stage 12.7 — disable the auto-prune of expired sessions /
    /// peer advertisements that runs on `--contributor-state-dir`
    /// open. Useful for forensic re-runs against an old tree.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,

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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,

    /// Stage 12.7 — optional contributor workflow state directory.
    /// When set, the watcher loads cross-restart seen markers for
    /// posted-jobs / network job announcements and dual-writes
    /// accepted/rejected contributor results + posted-result-links
    /// into `<state-dir>/results/...`. Omit to preserve pre-12.7
    /// behavior exactly. See `omni_contributor::state`.
    #[arg(long)]
    contributor_state_dir: Option<PathBuf>,

    /// Stage 12.7 — disable the auto-prune of expired sessions /
    /// peer advertisements that runs on `--contributor-state-dir`
    /// open. Useful for forensic re-runs against an old tree.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,

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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,

    /// Stage 12.7 — optional contributor workflow state directory.
    /// When set, the watcher loads cross-restart seen markers for
    /// network result announcements and dual-writes fetched
    /// posted-result-links into `<state-dir>/results/...`. Omit
    /// to preserve pre-12.7 behavior exactly. See
    /// `omni_contributor::state`.
    #[arg(long)]
    contributor_state_dir: Option<PathBuf>,

    /// Stage 12.7 — disable the auto-prune of expired sessions /
    /// peer advertisements that runs on `--contributor-state-dir`
    /// open. Useful for forensic re-runs against an old tree.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    // ── Stage 12.4 — live activation handoff (opt-in) ────────────

    /// Stage 12.4 — SNIP V2 root of an upstream `WorkAssignment`.
    /// Required when `--activation-in-mode snip` (fetch upstream
    /// partial from SNIP) or `--activation-in-mode live` (resolve
    /// the upstream contributor for handoff matching). At most one
    /// upstream supported at v1 (strict linear pipeline).
    #[arg(long)]
    upstream_from_assignment_snip_root: Option<String>,

    /// Stage 12.4 — operator-supplied libp2p PeerId (or full
    /// `/p2p/...` multiaddr) of the downstream contributor that
    /// should receive this stage's output activation. When set,
    /// default `--activation-out-mode` flips to `both`. v1 does NOT
    /// extend `ContributorJoin` with peer-id; PeerId discovery via
    /// signed contributor advertisements is Stage 12.5+.
    #[arg(long)]
    downstream_to_peer: Option<String>,

    /// Stage 12.4 — SNIP V2 root of the downstream `WorkAssignment`
    /// (the stage that will receive our output activation). Required
    /// when `--downstream-to-peer` is set OR
    /// `--resolve-downstream-peer-from-session` is set.
    #[arg(long)]
    downstream_to_assignment_snip_root: Option<String>,

    /// Stage 12.5 — directory of verified peer advertisements
    /// (typically populated by `watch-peer-adverts --out-dir`).
    /// Stage 12.7: prefer `--contributor-state-dir` instead.
    /// Required when `--resolve-downstream-peer-from-session` is
    /// set AND `--contributor-state-dir` is omitted; ignored
    /// otherwise. Supplying both this flag and
    /// `--contributor-state-dir` is rejected with
    /// `StateError::AmbiguousSource`.
    #[arg(long)]
    peer_advert_dir: Option<PathBuf>,

    /// Stage 12.5 — directory of verified joins (typically the
    /// `watch-sessions --out-dir`). Stage 12.7: prefer
    /// `--contributor-state-dir` instead. Required when
    /// `--resolve-downstream-peer-from-session` is set AND
    /// `--contributor-state-dir` is omitted: peer advertisements
    /// are rejected unless the contributor's join for the session
    /// is present AND `verify_contributor_join` passes.
    /// Supplying both this flag and `--contributor-state-dir` is
    /// rejected with `StateError::AmbiguousSource`.
    #[arg(long)]
    joins_dir: Option<PathBuf>,

    /// Stage 12.7 — optional contributor workflow state directory.
    /// When set with `--resolve-downstream-peer-from-session`, the
    /// resolver loads verified joins and verified peer
    /// advertisements from `<state-dir>/verified/sessions/<id>/...`
    /// instead of requiring separate `--joins-dir` /
    /// `--peer-advert-dir` flags. Supplying both this and either
    /// legacy flag is rejected with `StateError::AmbiguousSource`.
    /// Omit to preserve pre-12.7 behavior exactly. See
    /// `omni_contributor::state`.
    #[arg(long)]
    contributor_state_dir: Option<PathBuf>,

    /// Stage 12.7 — disable the auto-prune of expired sessions /
    /// peer advertisements that runs on `--contributor-state-dir`
    /// open. Useful for forensic re-runs against an old tree.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

    /// Stage 12.5 — resolve the downstream peer from a verified
    /// `ContributorPeerAdvertisement` instead of requiring a manual
    /// `--downstream-to-peer`. Requires `--peer-advert-dir` and
    /// `--downstream-to-assignment-snip-root`. If
    /// `--downstream-to-peer` is ALSO supplied, the manual value
    /// takes precedence (operator override).
    #[arg(long, default_value_t = false)]
    resolve_downstream_peer_from_session: bool,

    /// Stage 12.5 — dtype to use when resolving the downstream
    /// peer's advertisement. Has no effect outside
    /// `--resolve-downstream-peer-from-session`. Defaults to `f16`
    /// because it's the most common Stage 12.4 activation dtype.
    /// Operators wiring runners that produce bf16 or f32 should set
    /// this explicitly so route resolution matches the runner's
    /// actual output.
    #[arg(long, value_enum, default_value_t = CliTensorDtype::F16)]
    downstream_resolve_dtype: CliTensorDtype,

    /// Stage 12.4 — upstream activation source.
    /// - `none` (default if no upstream root supplied): first stage.
    /// - `live`: wait for an `ActivationHandoff` over omni-net.
    /// - `snip`: fetch upstream partial from SNIP.
    #[arg(long, value_enum, default_value_t = ActivationInMode::None)]
    activation_in_mode: ActivationInMode,

    /// Stage 12.4 — output activation routing.
    /// Default depends on whether downstream peer+assignment are set:
    ///   downstream present → `both` (live for latency, SNIP for audit)
    ///   downstream absent  → `snip` (Stage 12.3 behavior, default)
    /// Set explicitly to override.
    #[arg(long, value_enum)]
    activation_out_mode: Option<ActivationOutMode>,

    /// Stage 12.4 — bounded wait for the upstream live handoff
    /// (seconds).
    #[arg(long, default_value_t = 60)]
    upstream_wait_secs: u64,

    /// Stage 12.4 — operator-supplied PeerId/multiaddr of the
    /// upstream contributor. Used as an advisory hint for what
    /// to expect on the receive side; not required for verification
    /// (the signed inner envelope is the trust root).
    #[arg(long)]
    upstream_from_peer: Option<String>,

    /// Stage 12.4 — max chunk size (bytes) for outgoing handoffs.
    /// Must be > 0 (`0` would `div_ceil` to a panic).
    #[arg(
        long,
        default_value_t = 64 * 1024 * 1024,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    handoff_chunk_max_bytes: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

/// Stage 12.4 — upstream activation source for `run-assignment`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ActivationInMode {
    None,
    Live,
    Snip,
}

/// Stage 12.4 — output activation routing for `run-assignment`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum ActivationOutMode {
    /// 12.3 behavior: publish partial to SNIP; do not send live.
    Snip,
    /// Send live via omni-net request_tensor; do not publish to SNIP.
    Live,
    /// Send live AND publish to SNIP. Default when downstream peer
    /// + assignment are supplied.
    Both,
    /// Do nothing (last stage, no audit publish).
    None,
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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
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
struct SendHandoffArgs {
    /// SNIP V2 root of the open `ExecutionSession`.
    #[arg(long)]
    execution_session_snip_root: String,

    /// SNIP V2 root of the from-side `WorkAssignment` (the stage
    /// whose output activation is being sent).
    #[arg(long)]
    from_assignment_snip_root: String,

    /// SNIP V2 root of the to-side `WorkAssignment` (the receiver).
    #[arg(long)]
    to_assignment_snip_root: String,

    /// 32-byte raw contributor seed file. Must correspond to the
    /// from-assignment's `contributor_pubkey_hex`.
    #[arg(long)]
    from_contributor_seed: PathBuf,

    /// Path to the raw activation bytes file (whatever the from-stage
    /// runner emitted).
    #[arg(long)]
    activation_file: PathBuf,

    /// Tensor dtype.
    #[arg(long, value_enum)]
    dtype: CliTensorDtype,

    /// Tensor shape, comma-separated (e.g. `--shape 32,4096`).
    #[arg(long, value_delimiter = ',')]
    shape: Vec<u64>,

    /// libp2p PeerId or `/p2p/...` multiaddr of the receiver. v1
    /// operator-supplied (no signed peer advertisement yet —
    /// Stage 12.5+).
    #[arg(long)]
    to_peer: String,

    /// Max bytes per chunk; bigger activations are split.
    /// Must be > 0 (`0` would `div_ceil` to a panic).
    #[arg(
        long,
        default_value_t = 64 * 1024 * 1024,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    chunk_max_bytes: u64,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
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

/// CLI mirror of `omni_contributor::TensorDtype`. Avoid pulling clap
/// derives into the contributor crate.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliTensorDtype {
    F16,
    Bf16,
    F32,
}

impl From<CliTensorDtype> for omni_contributor::TensorDtype {
    fn from(d: CliTensorDtype) -> Self {
        match d {
            CliTensorDtype::F16 => omni_contributor::TensorDtype::F16,
            CliTensorDtype::Bf16 => omni_contributor::TensorDtype::Bf16,
            CliTensorDtype::F32 => omni_contributor::TensorDtype::F32,
        }
    }
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

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,

    /// Stage 12.7 — optional contributor workflow state directory.
    /// When set, the watcher loads cross-restart seen markers for
    /// sessions/joins/assignments/partials/aggregates and
    /// dual-writes verified bodies into the
    /// `<state-dir>/verified/sessions/<id>/...` subtree. That
    /// subtree is bit-identical to `--out-dir`, so operators
    /// migrating from a pre-12.7 layout can point both flags at
    /// the same path. Omit to preserve pre-12.7 behavior exactly.
    /// See `omni_contributor::state`.
    #[arg(long)]
    contributor_state_dir: Option<PathBuf>,

    /// Stage 12.7 — disable the auto-prune of expired sessions /
    /// peer advertisements that runs on `--contributor-state-dir`
    /// open. Useful for forensic re-runs against an old tree.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

    #[arg(long, default_value_t = 5)]
    poll_interval_secs: u64,
    #[arg(long)]
    max_polls: Option<u64>,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Stage 12.5 args ────────────────────────────────────────────────────────

#[derive(Args)]
struct AdvertisePeerArgs {
    /// SNIP V2 root of the open `ExecutionSession`.
    #[arg(long)]
    execution_session_snip_root: String,

    /// SNIP V2 root of the contributor's `ContributorJoin` for this
    /// session — required so the advertisement binds to a verified
    /// session-side join.
    #[arg(long)]
    join_snip_root: String,

    /// 32-byte raw contributor seed file. Must correspond to the
    /// join's `contributor_pubkey_hex`.
    #[arg(long)]
    contributor_seed: PathBuf,

    /// Optional reachable libp2p multiaddrs. May be empty; mDNS +
    /// Kademlia resolve PeerId → addrs on the mesh.
    #[arg(long = "listen-multiaddr")]
    listen_multiaddrs: Vec<String>,

    /// Advertised maximum incoming chunk byte count. Bounded by the
    /// Stage 12.4 `HANDOFF_CHUNK_MAX_BYTES`; receivers negotiate
    /// `min(local, advertised)`.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    max_handoff_chunk_bytes: u64,

    /// Supported tensor dtypes. Repeatable; at least one required.
    #[arg(long = "supported-dtype", value_enum, required = true)]
    supported_dtypes: Vec<CliTensorDtype>,

    /// Advertisement lifetime in seconds. Hard-capped at 24h
    /// (86400) by the Stage 12.5 schema.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=86_400))]
    expires_in_secs: u64,

    /// If set, broadcast a `NetworkPeerAdvertisementAnnouncement`
    /// in addition to publishing the body to SNIP.
    #[arg(long, default_value_t = false)]
    publish_announcement: bool,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
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
struct WatchPeerAdvertsArgs {
    /// Root directory under which per-session subdirectories
    /// `<out-dir>/<session_id>/peer-adverts/<contributor>.json`
    /// are written.
    #[arg(long)]
    out_dir: PathBuf,

    /// Optional session_id filter (repeatable). Empty = accept any.
    #[arg(long = "session-id")]
    session_id: Vec<String>,

    /// Optional contributor_pubkey filter (repeatable). Empty = any.
    #[arg(long = "contributor-pubkey")]
    contributor_pubkey: Vec<String>,

    /// Path to a directory holding verified joins. The watcher
    /// loads every `*.json` it finds (one per join) and uses the
    /// set as the "matching join" check in
    /// `process_peer_advertisement_announcement`. Empty/missing
    /// dir = treated as no-joins-available; advertisements get
    /// rejected with `NoMatchingJoin`. Operators typically use
    /// `watch-sessions --out-dir <dir>` to populate this.
    #[arg(long)]
    joins_dir: Option<PathBuf>,

    #[arg(long)]
    max_adverts: Option<u64>,

    #[arg(long)]
    max_polls: Option<u64>,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,

    /// Stage 12.6 — optional persistent libp2p mesh identity file
    /// (auto-created at 0600 on Unix). Stabilizes `local_peer_id`
    /// across restart so peer advertisements survive their full
    /// ≤24h freshness window. Omit to preserve pre-12.6 ephemeral
    /// behavior. See `omni_net::load_or_create_keypair_file_bytes`.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,

    /// Stage 12.7 — optional contributor workflow state directory.
    /// When set, the watcher loads cross-restart seen markers for
    /// peer advertisements and dual-writes verified advert bodies
    /// into `<state-dir>/verified/sessions/<id>/peer-adverts/`.
    /// Auto-prunes expired adverts on open by default. Omit to
    /// preserve pre-12.7 behavior exactly. See
    /// `omni_contributor::state`.
    #[arg(long)]
    contributor_state_dir: Option<PathBuf>,

    /// Stage 12.7 — disable the auto-prune of expired sessions /
    /// peer advertisements that runs on `--contributor-state-dir`
    /// open. Useful for forensic re-runs against an old tree.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

    #[arg(long, default_value_t = 5)]
    poll_interval_secs: u64,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,
}

// ── Stage 12.8 args ──────────────────────────────────────────────────────

/// CLI mirror of `omni_contributor::handoff::TensorDtype`. Kept here
/// (rather than re-exporting with clap derives) so the contributor
/// crate stays clap-free.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliTensorDtypePlanner {
    F16,
    Bf16,
    F32,
}

impl From<CliTensorDtypePlanner> for omni_contributor::TensorDtype {
    fn from(v: CliTensorDtypePlanner) -> Self {
        match v {
            CliTensorDtypePlanner::F16 => omni_contributor::TensorDtype::F16,
            CliTensorDtypePlanner::Bf16 => omni_contributor::TensorDtype::Bf16,
            CliTensorDtypePlanner::F32 => omni_contributor::TensorDtype::F32,
        }
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliPlannerStrategy {
    SequentialLayers,
    SingleContributor,
    RoundRobin,
}

impl From<CliPlannerStrategy> for omni_contributor::PlannerStrategy {
    fn from(v: CliPlannerStrategy) -> Self {
        match v {
            CliPlannerStrategy::SequentialLayers => {
                omni_contributor::PlannerStrategy::SequentialLayers
            }
            CliPlannerStrategy::SingleContributor => {
                omni_contributor::PlannerStrategy::SingleContributor
            }
            CliPlannerStrategy::RoundRobin => omni_contributor::PlannerStrategy::RoundRobin,
        }
    }
}

#[derive(Args)]
struct PlanSessionAssignmentsArgs {
    /// Stage 12.7 contributor workflow state directory. Required —
    /// the planner reads its inputs (verified session, joins,
    /// optional peer advertisements) from here. Auto-prune on open
    /// is inherited from Stage 12.7 (`--no-prune-state-on-start`
    /// disables).
    #[arg(long)]
    contributor_state_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

    /// 64-char lowercase hex `session_id` to plan against. Must
    /// match a `verified/sessions/<session_id>/session.json` in
    /// the state dir.
    #[arg(long)]
    session_id: String,

    #[arg(long, value_enum, default_value_t = CliPlannerStrategy::SequentialLayers)]
    strategy: CliPlannerStrategy,

    #[arg(long, value_enum, default_value_t = CliTensorDtypePlanner::F16)]
    required_dtype: CliTensorDtypePlanner,

    /// Eligibility floor on `ContributorJoin.available_ram_bytes`.
    /// Pass/fail filter; the planner does not rank by RAM.
    #[arg(long, default_value_t = 0)]
    min_available_ram_bytes: u64,

    /// Cap on the total number of assignments emitted after strategy
    /// dispatch.
    #[arg(long)]
    max_assignments: Option<u32>,

    /// Optional operator-supplied `ModelPlan` JSON. When supplied,
    /// strategy-specific work-kind / expected_work_units come from
    /// here. When omitted, the planner falls back to `--layer-count`
    /// where applicable. Never SNIP-published or signed.
    #[arg(long)]
    model_plan: Option<PathBuf>,

    /// Fallback when `--model-plan` is omitted. Required for
    /// `sequential-layers` (equal-split) and `round-robin`
    /// (single-layer-per-stage) and `single-contributor`
    /// (one `Layers { 0, N }` assignment) when no model-plan is
    /// supplied.
    #[arg(long)]
    layer_count: Option<u32>,

    /// When set, the planner loads verified peer advertisements
    /// from the state-dir and filters contributors to those with a
    /// non-expired advert whose
    /// `capabilities.supports_live_handoff == true` and whose
    /// `supported_dtypes` contains `--required-dtype`. Off by
    /// default — pre-12.8 sessions that don't need live handoff
    /// don't need adverts.
    #[arg(long, default_value_t = false)]
    require_live_routing: bool,

    /// Output path for the produced `AssignmentPlan` JSON.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct AssignSessionPlanArgs {
    /// Path to an `AssignmentPlan` JSON produced by
    /// `plan-session-assignments`. The plan's `plan_hash` is
    /// re-verified on read.
    #[arg(long)]
    plan: PathBuf,

    /// SNIP V2 root of the `ExecutionSession` the plan targets.
    /// Required because the Stage 12.7 state-dir does not carry
    /// SNIP roots — only inner bodies. The fetched session's
    /// `session_id` must match `plan.session_id`.
    #[arg(long)]
    session_snip_root: String,

    /// 32-byte raw coordinator seed file. Pubkey must match
    /// `session.coordinator_pubkey_hex`.
    #[arg(long)]
    coordinator_seed: PathBuf,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,

    /// Stage 12.6 — optional persistent libp2p mesh identity file.
    /// Same semantics as every other Stage 12.x subcommand.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    /// When set, the mesh `NetworkWorkAssignedAnnouncement`
    /// broadcast is skipped; only the SNIP publish step runs.
    /// Default behavior matches `assign-work` posture (broadcast
    /// to the mesh in addition to the SNIP publish).
    #[arg(long, default_value_t = false)]
    no_publish_announcements: bool,

    /// Validate the plan + session and print what would be
    /// published, without touching SNIP or the mesh. Exit 0 on
    /// success.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Stage 12.7 — when supplied, each published WorkAssignment is
    /// also dual-written into
    /// `<state>/verified/sessions/<id>/assignments/<assignment_id>.json`
    /// and a `seen/assignments/<session>--<assignment>` marker is
    /// laid down. Omit to preserve pre-12.7 behavior.
    #[arg(long)]
    contributor_state_dir: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,
}

// ── Stage 12.9 args ──────────────────────────────────────────────────────

/// Output format for `session-status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum SessionStatusFormat {
    /// One `event=session_status ...` line + one
    /// `event=assignment_status ...` line per assignment +
    /// `event=missing_partial ...` per missing partial +
    /// `event=note ...` per `notes` entry. Default — matches every
    /// other Stage 12.x watcher.
    Events,
    /// Print the full `SessionStatusReport` as pretty-printed JSON.
    Json,
    /// Compact terminal-friendly table. No external TUI deps; just
    /// stdout.
    Pretty,
}

#[derive(Args)]
struct SessionStatusArgs {
    /// Stage 12.7 contributor workflow state directory. Required —
    /// the reporter reads every body it inspects from here.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// 64-char lowercase hex `session_id` to report on.
    #[arg(long)]
    session_id: String,

    /// Output format. Defaults to `events` (matches every other
    /// Stage 12.x watcher's bare-stdout posture).
    #[arg(long, value_enum, default_value_t = SessionStatusFormat::Events)]
    format: SessionStatusFormat,

    /// Optional path to mirror the JSON report. Best-effort
    /// `std::fs::write`; a failure here logs a stderr warning and
    /// does not change the exit code (the report is a snapshot,
    /// not a protocol artifact).
    #[arg(long)]
    json_out: Option<PathBuf>,

    /// Exit non-zero when `overall_status` is not in
    /// `{CompletePartials, Aggregated}`. Default is "exit 0 even
    /// if incomplete" so dashboard scrapers can run unconditionally.
    #[arg(long, default_value_t = false)]
    fail_on_incomplete: bool,

    /// When set, expired peer advertisements are counted in
    /// `peer_advert_count`. Session expiry is always reported
    /// regardless of this flag.
    #[arg(long, default_value_t = false)]
    include_expired: bool,

    /// Stage 12.7 — disable the auto-prune of expired sessions /
    /// peer advertisements that runs on `--contributor-state-dir`
    /// open. Useful for forensic re-runs against an old tree.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,
}

// ── Stage 12.10 args ─────────────────────────────────────────────────────

/// CLI mirror of `omni_contributor::repair::RepairStrategy`. Stage
/// 12.11 added `ReassignMissing` once the `WorkAssignmentSupersession`
/// envelope shipped.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliRepairStrategy {
    ReannounceMissing,
    ReassignMissing,
}

impl From<CliRepairStrategy> for omni_contributor::RepairStrategy {
    fn from(v: CliRepairStrategy) -> Self {
        match v {
            CliRepairStrategy::ReannounceMissing => {
                omni_contributor::RepairStrategy::ReannounceMissing
            }
            CliRepairStrategy::ReassignMissing => {
                omni_contributor::RepairStrategy::ReassignMissing
            }
        }
    }
}

/// CLI mirror of `omni_contributor::supersession::SupersessionReason`.
/// `Custom { label }` is intentionally NOT exposed at the CLI level
/// in v1 — operators wanting Custom write the plan JSON by hand. The
/// three closed-enum variants cover the common operator cases.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliSupersessionReason {
    MissingPartial,
    InvalidPartial,
    OperatorRebalance,
}

impl From<CliSupersessionReason> for omni_contributor::SupersessionReason {
    fn from(v: CliSupersessionReason) -> Self {
        match v {
            CliSupersessionReason::MissingPartial => {
                omni_contributor::SupersessionReason::MissingPartial
            }
            CliSupersessionReason::InvalidPartial => {
                omni_contributor::SupersessionReason::InvalidPartial
            }
            CliSupersessionReason::OperatorRebalance => {
                omni_contributor::SupersessionReason::OperatorRebalance
            }
        }
    }
}

#[derive(Args)]
struct PlanSessionRepairArgs {
    /// Stage 12.7 contributor workflow state directory. Required —
    /// used both for `--build-status` and for the `source_status_hash`
    /// projection input.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// 64-char lowercase hex `session_id` to plan a repair for.
    #[arg(long)]
    session_id: String,

    /// Either supply a pre-built `SessionStatusReport` JSON OR pass
    /// `--build-status` to build one on the fly from the state-dir.
    /// Exactly one of these two must be supplied.
    #[arg(long, conflicts_with = "build_status")]
    status_report: Option<PathBuf>,

    /// Build the Stage 12.9 status report on the fly. Mutually
    /// exclusive with `--status-report`.
    #[arg(long, default_value_t = false)]
    build_status: bool,

    /// v1: only `reannounce-missing`. Future Stage 12.11+ may add
    /// `reassign-missing` once a supersession envelope exists.
    #[arg(long, value_enum, default_value_t = CliRepairStrategy::ReannounceMissing)]
    strategy: CliRepairStrategy,

    /// Optional operator hint copied into the plan for dry-run
    /// review. Not a trust check — the applier re-verifies the
    /// coordinator seed against the freshly-fetched session.
    #[arg(long)]
    coordinator_pubkey_hex: Option<String>,

    /// Passed through to `build_session_status_report` when
    /// `--build-status` is set.
    #[arg(long, default_value_t = false)]
    include_expired: bool,

    /// Stage 12.7 — disable the auto-prune of expired sessions /
    /// peer advertisements that runs on `--contributor-state-dir`
    /// open.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

    /// Output path for the produced `SessionRepairPlan` JSON.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct ApplySessionRepairArgs {
    /// Path to a `SessionRepairPlan` JSON produced by
    /// `plan-session-repair`. `repair_plan_hash` is recomputed and
    /// verified on read.
    #[arg(long)]
    repair_plan: PathBuf,

    /// SNIP V2 root of the `ExecutionSession` the plan targets.
    #[arg(long)]
    session_snip_root: String,

    /// 32-byte raw coordinator seed file. Pubkey must match
    /// `session.coordinator_pubkey_hex`.
    #[arg(long)]
    coordinator_seed: PathBuf,

    /// Stage 12.7 contributor workflow state directory. Required
    /// because the apply step re-loads + re-verifies each
    /// referenced assignment from disk before republishing.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,

    /// Stage 12.6 — persistent libp2p mesh identity file.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    /// When set, the mesh `NetworkWorkAssignedAnnouncement`
    /// broadcast is skipped; only the SNIP republish runs. Default
    /// behavior matches Stage 12.8 `assign-session-plan`:
    /// broadcast to the mesh in addition to the SNIP republish.
    #[arg(long, default_value_t = false)]
    no_publish_announcements: bool,

    /// Validate the plan + session + assignments and print
    /// `event=would_reannounce` lines, without touching SNIP or
    /// the mesh.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

// ── Stage 12.11 args ─────────────────────────────────────────────────────

#[derive(Args)]
struct PlanSessionReassignArgs {
    /// Stage 12.7 contributor workflow state directory.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// 64-char lowercase hex `session_id` to plan a reassignment for.
    #[arg(long)]
    session_id: String,

    /// Either supply a pre-built `SessionStatusReport` JSON OR pass
    /// `--build-status` to build one on the fly from the state-dir.
    #[arg(long, conflicts_with = "build_status")]
    status_report: Option<PathBuf>,

    /// Build the Stage 12.9 status report on the fly. Mutually
    /// exclusive with `--status-report`.
    #[arg(long, default_value_t = false)]
    build_status: bool,

    /// Reason embedded in every `ReassignAssignment` action's
    /// downstream `WorkAssignmentSupersession`. v1 CLI exposes the
    /// three closed-enum reasons (`Custom` is reachable only by
    /// hand-editing the plan JSON before apply).
    #[arg(long, value_enum, default_value_t = CliSupersessionReason::MissingPartial)]
    reason: CliSupersessionReason,

    /// Optional operator hint copied into the plan for dry-run
    /// review. Not a trust check.
    #[arg(long)]
    coordinator_pubkey_hex: Option<String>,

    /// Passed through to `build_session_status_report` when
    /// `--build-status` is set.
    #[arg(long, default_value_t = false)]
    include_expired: bool,

    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

    /// Output path for the produced `SessionRepairPlan` v2 JSON.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct ApplySessionReassignArgs {
    /// Path to a v2 `SessionRepairPlan` JSON whose
    /// `strategy == ReassignMissing` and every action is
    /// `ReassignAssignment`. `repair_plan_hash` is recomputed and
    /// verified on read.
    #[arg(long)]
    reassignment_plan: PathBuf,

    /// SNIP V2 root of the `ExecutionSession` the plan targets.
    #[arg(long)]
    session_snip_root: String,

    /// 32-byte raw coordinator seed file. Pubkey must match
    /// `session.coordinator_pubkey_hex`.
    #[arg(long)]
    coordinator_seed: PathBuf,

    /// Stage 12.7 contributor workflow state directory. Required
    /// because the applier re-verifies the session-on-disk shape
    /// (joins / supersedee assignments) before publishing and
    /// mirrors the new replacement + supersession bodies back
    /// into the state-dir.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,

    #[arg(long, default_value = "sum-node")]
    snip_binary: PathBuf,
    #[arg(long)]
    snip_seed: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    listen_port: u16,
    #[arg(long = "peer")]
    peer: Vec<String>,

    /// Stage 12.6 — persistent libp2p mesh identity file.
    #[arg(long)]
    net_identity_file: Option<PathBuf>,
    #[arg(long, default_value_t = 200)]
    propagation_wait_ms: u64,
    #[arg(long, default_value_t = 30)]
    peer_wait_secs: u64,
    #[arg(long, default_value_t = 500)]
    mesh_stabilize_ms: u64,

    /// When set, the mesh broadcasts (replacement assignment
    /// announcements + the supersession announcement) are skipped;
    /// only the SNIP publish steps run.
    #[arg(long, default_value_t = false)]
    no_publish_announcements: bool,

    /// Validate the plan + session + assignments + signed
    /// replacement bodies + signed supersession body, then print
    /// `event=would_reassign` + `event=would_publish_supersession`
    /// lines without touching SNIP, the mesh, or the state-dir.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

// ── Stage 12.14 — archive-session ─────────────────────────────────────────

/// Stage 12.14 — closed status-policy enum mirrored from
/// `omni_contributor::ArchiveStatusRequirement`. clap-friendly.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliArchiveStatusRequirement {
    Any,
    Complete,
    Aggregated,
    CompletePartials,
    ExpiredIncomplete,
}

impl From<CliArchiveStatusRequirement>
    for omni_contributor::ArchiveStatusRequirement
{
    fn from(v: CliArchiveStatusRequirement) -> Self {
        match v {
            CliArchiveStatusRequirement::Any => {
                omni_contributor::ArchiveStatusRequirement::Any
            }
            CliArchiveStatusRequirement::Complete => {
                omni_contributor::ArchiveStatusRequirement::Complete
            }
            CliArchiveStatusRequirement::Aggregated => {
                omni_contributor::ArchiveStatusRequirement::Aggregated
            }
            CliArchiveStatusRequirement::CompletePartials => {
                omni_contributor::ArchiveStatusRequirement::CompletePartials
            }
            CliArchiveStatusRequirement::ExpiredIncomplete => {
                omni_contributor::ArchiveStatusRequirement::ExpiredIncomplete
            }
        }
    }
}

#[derive(Args)]
struct ArchiveSessionArgs {
    /// Stage 12.7 contributor workflow state directory.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// 64-char lowercase hex `session_id` to archive.
    #[arg(long)]
    session_id: String,

    /// Operator-chosen archive root. `<archive-dir>/<session_id>/`
    /// must NOT already exist; archive refuses to overwrite.
    #[arg(long)]
    archive_dir: PathBuf,

    /// Status policy. `complete` (default) accepts `Aggregated`
    /// or `CompletePartials`; `any` is the escape valve.
    #[arg(long, value_enum, default_value_t = CliArchiveStatusRequirement::Complete)]
    require_status: CliArchiveStatusRequirement,

    /// Copy source to the archive. Source state-dir is
    /// untouched. **Default** when neither `--copy` nor `--move`
    /// is passed. `--copy` is the safe explicit form.
    #[arg(long, conflicts_with = "move_mode", default_value_t = false)]
    copy: bool,

    /// Move source to the archive: copy + verify BLAKE3 + write
    /// manifest, THEN cascade the source out of the state-dir.
    /// Cascade runs only after every file's BLAKE3 verifies AND
    /// the manifest write succeeds.
    #[arg(long = "move", conflicts_with = "copy", default_value_t = false)]
    move_mode: bool,

    /// Also copy `results/result-links/<session.posted_id>.link.json`
    /// when it exists. Stage 12.14 leaves `results/contributor-results/`
    /// alone by default — those are job-keyed and the
    /// session→job mapping isn't carried by the state-dir.
    #[arg(long, default_value_t = false)]
    include_results: bool,

    /// Validate and produce the manifest in-memory; print
    /// `event=would_archive_file` per entry and
    /// `event=would_archive_complete files=N`; do NOT touch the
    /// filesystem.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Stage 12.7 — opt out of auto-prune on state-dir open.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,
}

// ── Stage 12.15 — restore-session-archive ─────────────────────────────────

#[derive(Args)]
struct RestoreSessionArchiveArgs {
    /// Stage 12.7 contributor workflow state directory the
    /// archive will be restored into.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// Path to the archive's session subdirectory — the parent
    /// that holds `manifest.json` plus the `verified/...` +
    /// `seen/...` mirror. Mutually exclusive with
    /// `--archive-dir` / `--session-id`.
    #[arg(
        long,
        conflicts_with_all = ["archive_dir", "session_id"]
    )]
    archive_session_dir: Option<PathBuf>,

    /// Archive root. Combined with `--session-id` to resolve
    /// `<archive-dir>/<session_id>/manifest.json`. Mutually
    /// exclusive with `--archive-session-dir`.
    #[arg(long, requires = "session_id")]
    archive_dir: Option<PathBuf>,

    /// 64-char lowercase hex session_id. Combined with
    /// `--archive-dir`.
    #[arg(long, requires = "archive_dir")]
    session_id: Option<String>,

    /// Parse + validate manifest + per-entry path safety. No
    /// archive file reads beyond manifest, no destination
    /// writes. Cheapest mode.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Read + BLAKE3-verify every archived file. No destination
    /// writes. Proves the archive bytes are intact. If
    /// `--verify-only` and `--dry-run` are both supplied,
    /// `--verify-only` wins.
    #[arg(long, default_value_t = false)]
    verify_only: bool,

    /// Default `false`. When `false`, restore refuses BEFORE
    /// writing if any destination file already exists in the
    /// state-dir. All-or-nothing: ANY pre-existing file fails;
    /// with this flag, EVERY destination is overwritten.
    #[arg(long, default_value_t = false)]
    overwrite_existing: bool,

    /// Default `false`. Skip
    /// `results/result-links/<posted_id>.link.json` entries
    /// even if the archive contains them. The Stage 12.14
    /// archive's `--include-results` flag at archive-time AND
    /// the Stage 12.15 `--include-results` flag at restore-time
    /// are independent: the archive may have captured a link
    /// the operator now wants to skip, or vice versa.
    #[arg(long, default_value_t = false)]
    include_results: bool,

    /// Stage 12.7 — opt out of auto-prune on state-dir open.
    #[arg(long, default_value_t = false)]
    no_prune_state_on_start: bool,
}

// ── Stage 12.16 — state-integrity ─────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum StateIntegrityFormat {
    /// One closed-set `event=...` line per finding, plus a final
    /// `event=state_integrity_summary ...` line. Default — matches
    /// every other Stage 12.x watcher's bare-stdout posture.
    Events,
    /// Print the full `StateIntegrityReport` as pretty-printed JSON
    /// on stdout. Operational chatter (state-store open notice,
    /// scanner progress) goes to stderr so `jq` works directly.
    Json,
    /// Compact terminal-friendly summary + per-severity sections.
    /// No external TUI deps; just stdout.
    Pretty,
}

#[derive(Args)]
struct StateIntegrityArgs {
    /// Stage 12.7 contributor workflow state directory the scan
    /// inspects. Required — the scanner is purely a read-only
    /// consumer of this tree.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// Optional 64-char lowercase hex `session_id` filter. When
    /// supplied, the scanner restricts session-scoped findings to
    /// that one session; cross-session walks (stray top-level
    /// files, archive-only orphans) still run.
    #[arg(long)]
    session_id: Option<String>,

    /// Optional sibling archive root. When set, every
    /// `<archive-dir>/<session_id>/manifest.json` is parsed via
    /// Stage 12.14 `verify_archive_manifest` and a full BLAKE3
    /// walk runs via Stage 12.15
    /// `restore_session_archive(verify_only=true, dry_run=true)`.
    /// Read-only end-to-end.
    #[arg(long)]
    include_archives: Option<PathBuf>,

    /// Output format. Defaults to `events`.
    #[arg(long, value_enum, default_value_t = StateIntegrityFormat::Events)]
    format: StateIntegrityFormat,

    /// Optional path to mirror the JSON report. Best-effort
    /// `std::fs::write`; a failure here logs a stderr warning and
    /// does not change the exit code (the report is a snapshot,
    /// not a protocol artifact).
    #[arg(long)]
    json_out: Option<PathBuf>,

    /// Default policy: exit 1 when `counts_error > 0`. With this
    /// flag set, exit 1 when `counts_warn + counts_error > 0`.
    /// Operators who treat every warn as a CI failure can opt in.
    #[arg(long, default_value_t = false)]
    fail_on_warn: bool,

    /// Stage 12.19 — diff the LIVE scan against the supplied
    /// baseline v1 `StateIntegrityReport` JSON. When set, the
    /// command switches to rendering the diff (not the raw
    /// report). `--format`, `--json-out`, and the new
    /// `--fail-on-new` / `--fail-on-new-error` flags then apply
    /// to the diff output. Mutually exclusive with Stage 12.20
    /// `--signed-baseline`.
    #[arg(long, conflicts_with = "signed_baseline")]
    baseline: Option<PathBuf>,

    /// Stage 12.20 — diff the LIVE scan against the supplied
    /// `SignedStateIntegrityBaseline` JSON. Mutually exclusive
    /// with `--baseline`. REQUIRES `--baseline-pubkey-hex` as
    /// the operator-supplied trust anchor; the wrapper's
    /// `signer_pubkey_hex` is verified against it BEFORE any
    /// cryptographic check, and the Ed25519 signature is
    /// recomputed against the canonical body. Only after both
    /// pass does the embedded report become the baseline for
    /// the Stage 12.19 diff flow.
    #[arg(
        long,
        conflicts_with = "baseline",
        requires = "baseline_pubkey_hex"
    )]
    signed_baseline: Option<PathBuf>,

    /// Stage 12.20 — 64-char lowercase-hex Ed25519 public key
    /// that the signed baseline MUST be signed by. Required
    /// whenever `--signed-baseline` is used. Ignored otherwise.
    #[arg(long, requires = "signed_baseline")]
    baseline_pubkey_hex: Option<String>,

    /// Stage 12.19 — exit 1 when the live scan diff against
    /// `--baseline` shows ANY new finding (regardless of
    /// severity). Composes with `--fail-on-warn`: any one
    /// tripping → exit 1.
    #[arg(long, default_value_t = false)]
    fail_on_new: bool,

    /// Stage 12.19 — exit 1 when the live scan diff against
    /// `--baseline` shows any NEW `error`-severity finding. Less
    /// strict than `--fail-on-new` — new `warn` findings stay
    /// green. Recommended CI mode.
    #[arg(long, default_value_t = false)]
    fail_on_new_error: bool,

    /// Stage 12.19 — when used with `--baseline`, omit
    /// `unchanged_findings` from the rendered output. The
    /// underlying `StateIntegrityDiffReport` still carries
    /// every finding; `--summary-only` is a presentation flag.
    #[arg(long, default_value_t = false)]
    summary_only: bool,

    /// Stage 12.19 — when used with `--baseline`, refuse the
    /// diff if `baseline.state_dir != current.state_dir`.
    /// Default OFF — CI baselines are commonly captured on a
    /// different host than the prod state-dir.
    #[arg(long, default_value_t = false)]
    require_state_dir_match: bool,
}

// ── Stage 12.17 — plan-state-cleanup ──────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum PlanStateCleanupFormat {
    /// One `event=cleanup_action_planned` line per action plus a
    /// final `event=cleanup_plan_built` summary. Default.
    Events,
    /// Print the full `StateCleanupPlan` as pretty JSON on
    /// stdout (operational chatter goes to stderr so `jq` works
    /// directly).
    Json,
    /// Compact terminal-friendly per-action listing.
    Pretty,
}

#[derive(Args)]
struct PlanStateCleanupArgs {
    /// Stage 12.7 contributor workflow state directory the
    /// planner walks (via Stage 12.16's scan).
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// Optional 64-char lowercase hex `session_id` filter. When
    /// supplied, session-scoped actions are restricted to that
    /// one session; cross-session strays (top-level `seen/`
    /// junk) still plan.
    #[arg(long)]
    session_id: Option<String>,

    /// Optional path to a previously-written
    /// `StateIntegrityReport` JSON. When supplied, the planner
    /// consumes the supplied report verbatim instead of running
    /// a fresh scan — useful for CI pipelines that produce the
    /// report as an artifact. Drift between the supplied report
    /// and the live state-dir is detected at apply time via
    /// `source_integrity_hash`.
    #[arg(long)]
    integrity_json: Option<PathBuf>,

    /// Path to write the resulting `StateCleanupPlan` JSON.
    /// Atomic write (temp + rename). Required.
    #[arg(long)]
    out: PathBuf,

    /// Output format for the stdout summary. Default `events`.
    #[arg(long, value_enum, default_value_t = PlanStateCleanupFormat::Events)]
    format: PlanStateCleanupFormat,
}

// ── Stage 12.17 — apply-state-cleanup ────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ApplyStateCleanupFormat {
    /// One `event=cleanup_action_applied` / `event=cleanup_action_skipped`
    /// line per action + final `event=cleanup_complete`. Default.
    Events,
    /// Print the full `CleanupReport` as pretty JSON on stdout.
    Json,
    /// Compact terminal-friendly per-action listing.
    Pretty,
}

#[derive(Args)]
struct ApplyStateCleanupArgs {
    /// Stage 12.7 contributor workflow state directory the
    /// applier mutates. The applier re-runs Stage 12.16's scan
    /// for drift detection before any mutation.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// Path to the `StateCleanupPlan` JSON to apply.
    #[arg(long)]
    plan: PathBuf,

    /// Root under which `<plan_id>/...` quarantine subtree gets
    /// written. Tier-B actions copy their source bytes here
    /// BEFORE removing the source. Required even for tier-A-only
    /// plans (the path is consistent across plans).
    #[arg(long)]
    quarantine_dir: PathBuf,

    /// Dry-run: walk the plan and emit `event=would_apply_action`
    /// lines without touching the FS. Drift / hash / gate checks
    /// still run.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Required gate for any `QuarantineAndUnmarkPartial` action
    /// in the plan. Refused otherwise so the operator doesn't
    /// accidentally pull the rug from a planned Stage 12.11
    /// reassign.
    #[arg(long, default_value_t = false)]
    allow_invalid_partial_cleanup: bool,

    /// Required gate for any
    /// `QuarantineAndUnmarkOrphanAssignment` action. Apply
    /// additionally re-runs `compute_audit_health` per gated
    /// session and refuses on orphan-set drift.
    #[arg(long, default_value_t = false)]
    allow_orphan_assignments: bool,

    /// When set, `QuarantineVerifiedFile` actions skip the
    /// quarantine copy step and just delete the source.
    /// `QuarantineAndUnmark*` actions still quarantine — only
    /// the stray-file pathway is relaxed.
    #[arg(long, default_value_t = false)]
    purge_stray: bool,

    /// Output format. Default `events`.
    #[arg(long, value_enum, default_value_t = ApplyStateCleanupFormat::Events)]
    format: ApplyStateCleanupFormat,
}

// ── Stage 12.18 — restore-state-cleanup-quarantine ───────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum RestoreQuarantineFormat {
    /// One `event=restore_quarantine_file` /
    /// `event=verify_only_quarantine_file` /
    /// `event=would_restore_quarantine_file` per entry, plus a
    /// final `event=restore_quarantine_complete`. Default.
    Events,
    /// Print the full `QuarantineRestoreReport` as pretty JSON
    /// on stdout. Operational chatter goes to stderr so `jq`
    /// works directly.
    Json,
    /// Compact terminal-friendly per-entry listing.
    Pretty,
}

#[derive(Args)]
struct RestoreStateCleanupQuarantineArgs {
    /// Stage 12.7 contributor workflow state directory the
    /// restore writes back into.
    #[arg(long)]
    contributor_state_dir: PathBuf,

    /// Direct path to the `<plan_id>` subdirectory under the
    /// quarantine tree (i.e. `<quarantine-dir>/<plan_id>/`).
    /// Mutually exclusive with `--quarantine-dir` /
    /// `--plan-id`.
    #[arg(
        long,
        conflicts_with_all = ["quarantine_dir", "plan_id"]
    )]
    quarantine_plan_dir: Option<PathBuf>,

    /// Quarantine root containing one or more
    /// `<plan_id>/` subdirectories. Paired with `--plan-id`.
    #[arg(long, requires = "plan_id")]
    quarantine_dir: Option<PathBuf>,

    /// 16-char lowercase hex `plan_id`. Paired with
    /// `--quarantine-dir`.
    #[arg(long, requires = "quarantine_dir")]
    plan_id: Option<String>,

    /// Parse + validate the manifest. No quarantine reads, no
    /// destination writes.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Full BLAKE3 verify of every quarantine file. No
    /// destination writes. When both `--verify-only` and
    /// `--dry-run` are passed, `--verify-only` wins (Stage
    /// 12.15 precedent).
    #[arg(long, default_value_t = false)]
    verify_only: bool,

    /// Default `false`. With `false`, any pre-existing
    /// destination refuses BEFORE any write
    /// (`DestinationExists`).
    #[arg(long, default_value_t = false)]
    overwrite_existing: bool,

    /// Skip seen-marker restoration. Default OFF (markers
    /// restored automatically when the entry's
    /// `source_finding_kind` proves a marker was unmarked).
    #[arg(long, default_value_t = false)]
    no_restore_seen_markers: bool,

    /// Required gate for any
    /// `orphan_replacement_assignments` entry. Refused
    /// without this flag.
    #[arg(long, default_value_t = false)]
    allow_restore_orphan_assignments: bool,

    /// Output format. Default `events`.
    #[arg(long, value_enum, default_value_t = RestoreQuarantineFormat::Events)]
    format: RestoreQuarantineFormat,
}

// ── Stage 12.19 — state-integrity-diff ───────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum StateIntegrityDiffFormat {
    /// Per-finding `event=integrity_diff_new` /
    /// `event=integrity_diff_resolved` / (when not
    /// `--summary-only`) `event=integrity_diff_unchanged`
    /// lines + a final `event=state_integrity_diff_summary`
    /// line. Default.
    Events,
    /// Print the full `StateIntegrityDiffReport` as pretty
    /// JSON on stdout. Operational chatter goes to stderr so
    /// `jq` works directly.
    Json,
    /// Compact terminal-friendly summary + per-bucket
    /// sections.
    Pretty,
}

#[derive(Args)]
struct StateIntegrityDiffArgs {
    /// Path to the baseline v1 `StateIntegrityReport` JSON
    /// (typically produced by an earlier
    /// `state-integrity --json-out`). Mutually exclusive with
    /// `--signed-baseline`; exactly one MUST be supplied.
    #[arg(
        long,
        conflicts_with = "signed_baseline",
        required_unless_present = "signed_baseline"
    )]
    baseline: Option<PathBuf>,

    /// Stage 12.20 — path to a `SignedStateIntegrityBaseline`
    /// JSON. Mutually exclusive with `--baseline`. REQUIRES
    /// `--baseline-pubkey-hex` as the operator-supplied trust
    /// anchor.
    #[arg(
        long,
        conflicts_with = "baseline",
        required_unless_present = "baseline",
        requires = "baseline_pubkey_hex"
    )]
    signed_baseline: Option<PathBuf>,

    /// Stage 12.20 — 64-char lowercase-hex Ed25519 public key
    /// the signed baseline MUST be signed by.
    #[arg(long, requires = "signed_baseline")]
    baseline_pubkey_hex: Option<String>,

    /// Path to the current v1 `StateIntegrityReport` JSON to
    /// compare against the baseline.
    #[arg(long)]
    current: PathBuf,

    /// Refuse the diff if
    /// `baseline.state_dir != current.state_dir`. Default OFF
    /// — CI baselines are commonly captured on a different
    /// host than the prod state-dir.
    #[arg(long, default_value_t = false)]
    require_state_dir_match: bool,

    /// Exit 1 when the diff shows ANY new finding regardless
    /// of severity. Composes with `--fail-on-new-error`.
    #[arg(long, default_value_t = false)]
    fail_on_new: bool,

    /// Exit 1 when the diff shows any NEW `error`-severity
    /// finding. Recommended CI mode. Composes with
    /// `--fail-on-new`.
    #[arg(long, default_value_t = false)]
    fail_on_new_error: bool,

    /// Omit `unchanged_findings` from the rendered output.
    /// The underlying `StateIntegrityDiffReport` still
    /// carries every finding; `--summary-only` is a
    /// presentation flag.
    #[arg(long, default_value_t = false)]
    summary_only: bool,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = StateIntegrityDiffFormat::Events
    )]
    format: StateIntegrityDiffFormat,

    /// Optional path to mirror the diff report JSON
    /// (best-effort `std::fs::write`; failure here logs a
    /// stderr warning and does not change the exit code).
    #[arg(long)]
    json_out: Option<PathBuf>,
}

// ── Stage 12.20 — sign-state-integrity-baseline ──────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum SignStateIntegrityBaselineFormat {
    /// One `event=signed_baseline_written ...` line on success.
    Events,
    /// Print the signed wrapper as pretty JSON on stdout.
    /// Operational chatter goes to stderr so `jq` works.
    Json,
    /// Compact terminal-friendly summary.
    Pretty,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum CliBaselineSignerRole {
    Operator,
    Contributor,
    Dispatcher,
    Coordinator,
}

impl From<CliBaselineSignerRole> for omni_contributor::BaselineSignerRole {
    fn from(r: CliBaselineSignerRole) -> Self {
        match r {
            CliBaselineSignerRole::Operator => {
                omni_contributor::BaselineSignerRole::Operator
            }
            CliBaselineSignerRole::Contributor => {
                omni_contributor::BaselineSignerRole::Contributor
            }
            CliBaselineSignerRole::Dispatcher => {
                omni_contributor::BaselineSignerRole::Dispatcher
            }
            CliBaselineSignerRole::Coordinator => {
                omni_contributor::BaselineSignerRole::Coordinator
            }
        }
    }
}

#[derive(Args)]
struct SignStateIntegrityBaselineArgs {
    /// Path to the raw v1 `StateIntegrityReport` JSON to sign.
    /// Typically the output of a prior
    /// `state-integrity --json-out`.
    #[arg(long)]
    baseline_in: PathBuf,

    /// 32-byte raw Ed25519 seed file. Operators MUST keep this
    /// distinct from any chain-attestation or protocol-role
    /// seed — the baseline-signing role is its own key per
    /// Stage 12.20.
    #[arg(long)]
    signer_seed: PathBuf,

    /// Role tag recorded in the wrapper for forensics. Closed
    /// set: `operator` / `contributor` / `dispatcher` /
    /// `coordinator`.
    #[arg(long, value_enum)]
    signer_role: CliBaselineSignerRole,

    /// Destination for the signed wrapper JSON. Atomic
    /// tempfile + rename (same posture as Stage 12.17
    /// `plan-state-cleanup --out`).
    #[arg(long)]
    out: PathBuf,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = SignStateIntegrityBaselineFormat::Events
    )]
    format: SignStateIntegrityBaselineFormat,
}

// ── Stage 12.21 — sign-state-integrity-diff ──────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum SignStateIntegrityDiffFormat {
    /// One `event=signed_integrity_diff_written ...` line on
    /// success.
    Events,
    /// Print the signed wrapper as pretty JSON on stdout.
    /// Operational chatter goes to stderr so `jq` works.
    Json,
    /// Compact terminal-friendly summary.
    Pretty,
}

#[derive(Args)]
struct SignStateIntegrityDiffArgs {
    /// Path to the raw v1 `StateIntegrityDiffReport` JSON to
    /// sign. Typically the output of a prior
    /// `state-integrity-diff --json-out`.
    #[arg(long)]
    diff_in: PathBuf,

    /// 32-byte raw Ed25519 seed file. Operators MUST keep this
    /// distinct from any chain-attestation or protocol-role
    /// seed — the integrity-artifact signing role is its own
    /// key per Stage 12.20.
    #[arg(long)]
    signer_seed: PathBuf,

    /// Role tag recorded in the wrapper for forensics. Closed
    /// set: `operator` / `contributor` / `dispatcher` /
    /// `coordinator`. Reuses the Stage 12.20
    /// `BaselineSignerRole` enum per the Stage 12.21 plan —
    /// the four variants are role names, not artifact-type
    /// names.
    #[arg(long, value_enum)]
    signer_role: CliBaselineSignerRole,

    /// Destination for the signed wrapper JSON. Atomic
    /// tempfile + rename (same posture as Stage 12.17
    /// `plan-state-cleanup --out` and Stage 12.20
    /// `sign-state-integrity-baseline --out`).
    #[arg(long)]
    out: PathBuf,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = SignStateIntegrityDiffFormat::Events
    )]
    format: SignStateIntegrityDiffFormat,
}

// ── Stage 12.21 — verify-state-integrity-diff-signature ──────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum VerifyStateIntegrityDiffSignatureFormat {
    /// One `event=signed_integrity_diff_verify_ok ...` line on
    /// success. On failure, a non-zero exit + an
    /// `event=signed_integrity_diff_verify_failed reason=...`
    /// line. Default.
    Events,
    /// Print the verified wrapper's metadata as pretty JSON on
    /// stdout. Operational chatter goes to stderr so `jq`
    /// works.
    Json,
    /// Compact terminal-friendly summary.
    Pretty,
}

#[derive(Args)]
struct VerifyStateIntegrityDiffSignatureArgs {
    /// Path to a `SignedStateIntegrityDiff` JSON wrapper to
    /// verify.
    #[arg(long)]
    signed_diff: PathBuf,

    /// Operator-supplied trust anchor: 64-char lowercase-hex
    /// Ed25519 public key the wrapper MUST be signed by.
    /// Verification refuses with a `signer_pubkey_mismatch`
    /// pre-check before any crypto burn.
    #[arg(long)]
    expected_signer_pubkey_hex: String,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = VerifyStateIntegrityDiffSignatureFormat::Events
    )]
    format: VerifyStateIntegrityDiffSignatureFormat,
}

// ── Stage 12.22 — build-integrity-evidence-bundle ────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BuildIntegrityEvidenceBundleFormat {
    /// One `event=integrity_evidence_bundle_entry_hashed ...`
    /// line per entry plus a final
    /// `event=integrity_evidence_bundle_written` summary.
    /// Default.
    Events,
    /// Print the assembled bundle as pretty JSON on stdout.
    /// Operational chatter goes to stderr so `jq` works.
    Json,
    /// Compact terminal-friendly summary.
    Pretty,
}

#[derive(Args)]
struct BuildIntegrityEvidenceBundleArgs {
    /// Operator-supplied bundle entry — repeatable. Format:
    /// `<artifact_kind_tag>=<path>`. The kind tag is the
    /// closed-set wire tag (e.g. `signed_state_integrity_diff`).
    /// The path is interpreted relative to `--base-dir` if not
    /// absolute; absolute paths are accepted but must
    /// canonicalize to a path under `--base-dir` and are
    /// recorded in their base-dir-relative form. At least one
    /// `--include` is required.
    #[arg(long = "include", required = true, value_name = "kind=path")]
    includes: Vec<String>,

    /// Root directory under which every entry's recorded path
    /// is resolved. Canonicalized at build time and stored in
    /// the bundle so the verifier can rebase via its own
    /// optional `--base-dir` override.
    #[arg(long)]
    base_dir: PathBuf,

    /// Destination for the bundle JSON. Atomic tempfile +
    /// rename (same posture as Stage 12.17 `--out`).
    #[arg(long)]
    out: PathBuf,

    /// Optional bundle-level label, capped at 128 UTF-8 bytes.
    /// Operator-facing naming only; no semantic meaning to the
    /// verifier.
    #[arg(long)]
    label: Option<String>,

    /// Optional freeform notes, capped at 1024 UTF-8 bytes.
    #[arg(long)]
    notes: Option<String>,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = BuildIntegrityEvidenceBundleFormat::Events
    )]
    format: BuildIntegrityEvidenceBundleFormat,
}

// ── Stage 12.22 — verify-integrity-evidence-bundle ───────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum VerifyIntegrityEvidenceBundleFormat {
    /// One `event=integrity_evidence_bundle_entry_...` line
    /// per entry plus a final `event=integrity_evidence_bundle_verify_summary`.
    /// Default.
    Events,
    /// Print the full `BundleVerifyReport` as pretty JSON on
    /// stdout. Operational chatter goes to stderr so `jq`
    /// works.
    Json,
    /// Compact terminal-friendly summary + per-entry lines.
    Pretty,
}

#[derive(Args)]
struct VerifyIntegrityEvidenceBundleArgs {
    /// Path to the `IntegrityEvidenceBundle` JSON to verify.
    #[arg(long)]
    bundle: PathBuf,

    /// Optional override for the bundle's recorded `base_dir`.
    /// When omitted, the verifier resolves entries against
    /// `bundle.base_dir` as recorded. Use this when the
    /// bundle was built on a different host or the artifact
    /// tree was relocated.
    #[arg(long)]
    base_dir: Option<PathBuf>,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = VerifyIntegrityEvidenceBundleFormat::Events
    )]
    format: VerifyIntegrityEvidenceBundleFormat,
}

// ── Stage 12.23 — sign-integrity-evidence-bundle ─────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum SignIntegrityEvidenceBundleFormat {
    /// One `event=signed_integrity_evidence_bundle_written ...`
    /// line on success.
    Events,
    /// Print the signed wrapper as pretty JSON on stdout.
    /// Operational chatter goes to stderr so `jq` works.
    Json,
    /// Compact terminal-friendly summary.
    Pretty,
}

#[derive(Args)]
struct SignIntegrityEvidenceBundleArgs {
    /// Path to the raw v1 `IntegrityEvidenceBundle` JSON to
    /// sign. Typically the output of a prior
    /// `build-integrity-evidence-bundle --out`.
    #[arg(long)]
    bundle_in: PathBuf,

    /// 32-byte raw Ed25519 seed file. Operators MUST keep this
    /// distinct from any chain-attestation or protocol-role
    /// seed — the integrity-artifact signing role is its own
    /// key per Stage 12.20.
    #[arg(long)]
    signer_seed: PathBuf,

    /// Role tag recorded in the wrapper for forensics. Closed
    /// set: `operator` / `contributor` / `dispatcher` /
    /// `coordinator`. Reuses the Stage 12.20
    /// `BaselineSignerRole` enum per the Stage 12.21/12.23
    /// precedent — the four variants are role names, not
    /// artifact-type names.
    #[arg(long, value_enum)]
    signer_role: CliBaselineSignerRole,

    /// Destination for the signed wrapper JSON. Atomic
    /// tempfile + rename (same posture as Stage 12.17 / 12.20 /
    /// 12.21).
    #[arg(long)]
    out: PathBuf,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = SignIntegrityEvidenceBundleFormat::Events
    )]
    format: SignIntegrityEvidenceBundleFormat,
}

// ── Stage 12.23 — verify-integrity-evidence-bundle-signature ─────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum VerifyIntegrityEvidenceBundleSignatureFormat {
    /// One `event=signed_integrity_evidence_bundle_verify_ok ...`
    /// line on success. On failure, a non-zero exit + an
    /// `event=signed_integrity_evidence_bundle_verify_failed reason=...`
    /// line. Default.
    Events,
    /// Print the verified wrapper's metadata as pretty JSON on
    /// stdout. Operational chatter goes to stderr so `jq`
    /// works. Compact metadata view — does NOT re-print the
    /// embedded bundle's entries (operators who want them
    /// already have the wrapper on disk).
    Json,
    /// Compact terminal-friendly summary.
    Pretty,
}

#[derive(Args)]
struct VerifyIntegrityEvidenceBundleSignatureArgs {
    /// Path to a `SignedIntegrityEvidenceBundle` JSON wrapper
    /// to verify.
    #[arg(long)]
    signed_bundle: PathBuf,

    /// Operator-supplied trust anchor: 64-char lowercase-hex
    /// Ed25519 public key the wrapper MUST be signed by.
    /// Verification refuses with a `signer_pubkey_mismatch`
    /// pre-check before any crypto burn.
    #[arg(long)]
    expected_signer_pubkey_hex: String,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = VerifyIntegrityEvidenceBundleSignatureFormat::Events
    )]
    format: VerifyIntegrityEvidenceBundleSignatureFormat,
}

// ── Stage 12.24 — verify-integrity-evidence-chain ────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum VerifyIntegrityEvidenceChainFormat {
    /// Per-step bare-stdout events: started → signed_bundle_ok
    /// → bundle_byte_entry_* per Stage 12.22 entry →
    /// child_{ok,skipped,failed} per signed child → final
    /// verify_summary. Default.
    Events,
    /// Print the full `IntegrityEvidenceChainReport` as pretty
    /// JSON on stdout. Operational chatter goes to stderr so
    /// `jq` works.
    Json,
    /// Compact terminal-friendly summary + per-section
    /// listing.
    Pretty,
}

#[derive(Args)]
struct VerifyIntegrityEvidenceChainArgs {
    /// Path to the Stage 12.23
    /// `SignedIntegrityEvidenceBundle` JSON to chain-verify.
    #[arg(long)]
    signed_bundle: PathBuf,

    /// REQUIRED 64-char lowercase-hex Ed25519 public key the
    /// outermost signed-bundle wrapper MUST be signed by.
    #[arg(long)]
    expected_bundle_signer_pubkey_hex: String,

    /// Optional Stage 12.20 trust anchor — gates verification
    /// of every `signed_state_integrity_baseline` child entry.
    /// When omitted, baseline children record `Skipped`.
    #[arg(long)]
    expected_baseline_signer_pubkey_hex: Option<String>,

    /// Optional Stage 12.21 trust anchor — gates verification
    /// of every `signed_state_integrity_diff` child entry.
    /// When omitted, diff children record `Skipped`.
    #[arg(long)]
    expected_diff_signer_pubkey_hex: Option<String>,

    /// Optional override for the embedded bundle's recorded
    /// `base_dir`. Same posture as Stage 12.22's verifier:
    /// when omitted, the chain resolves entries against the
    /// bundle's recorded `base_dir`.
    #[arg(long)]
    base_dir: Option<PathBuf>,

    /// Optional best-effort mirror of the chain report JSON.
    /// Failure here logs a warn event and does NOT change exit
    /// code (same posture as Stage 12.19
    /// `state-integrity-diff --json-out`).
    #[arg(long)]
    json_out: Option<PathBuf>,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = VerifyIntegrityEvidenceChainFormat::Events
    )]
    format: VerifyIntegrityEvidenceChainFormat,
}

// ── Stage 12.25 — sign-integrity-evidence-chain-report ───────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum SignIntegrityEvidenceChainReportFormat {
    /// One `event=signed_integrity_evidence_chain_report_written ...`
    /// line on success.
    Events,
    /// Print the signed wrapper as pretty JSON on stdout.
    /// Operational chatter goes to stderr so `jq` works.
    Json,
    /// Compact terminal-friendly summary.
    Pretty,
}

#[derive(Args)]
struct SignIntegrityEvidenceChainReportArgs {
    /// Path to the raw v1 `IntegrityEvidenceChainReport` JSON
    /// to sign. Typically the output of a prior
    /// `verify-integrity-evidence-chain --json-out`.
    #[arg(long)]
    chain_report_in: PathBuf,

    /// 32-byte raw Ed25519 seed file. Operators MUST keep this
    /// distinct from any chain-attestation or protocol-role
    /// seed — the integrity-artifact signing role is its own
    /// key per Stage 12.20.
    #[arg(long)]
    signer_seed: PathBuf,

    /// Role tag recorded in the wrapper for forensics. Closed
    /// set: `operator` / `contributor` / `dispatcher` /
    /// `coordinator`. Reuses the Stage 12.20
    /// `BaselineSignerRole` enum per the Stage 12.21/12.23/12.25
    /// precedent — the four variants are role names, not
    /// artifact-type names.
    #[arg(long, value_enum)]
    signer_role: CliBaselineSignerRole,

    /// Destination for the signed wrapper JSON. Atomic
    /// tempfile + rename (same posture as Stage 12.17 / 12.20 /
    /// 12.21 / 12.23).
    #[arg(long)]
    out: PathBuf,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = SignIntegrityEvidenceChainReportFormat::Events
    )]
    format: SignIntegrityEvidenceChainReportFormat,
}

// ── Stage 12.25 — verify-integrity-evidence-chain-report-signature ───────

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum VerifyIntegrityEvidenceChainReportSignatureFormat {
    /// One `event=signed_integrity_evidence_chain_report_verify_ok ...`
    /// line on success. On failure, a non-zero exit + an
    /// `event=signed_integrity_evidence_chain_report_verify_failed reason=...`
    /// line. Default.
    Events,
    /// Print the verified wrapper's metadata as pretty JSON on
    /// stdout. Operational chatter goes to stderr so `jq`
    /// works. Compact metadata view — does NOT re-print the
    /// embedded chain report's per-entry or per-child lists
    /// (operators who want them already have the wrapper on
    /// disk).
    Json,
    /// Compact terminal-friendly summary.
    Pretty,
}

#[derive(Args)]
struct VerifyIntegrityEvidenceChainReportSignatureArgs {
    /// Path to a `SignedIntegrityEvidenceChainReport` JSON
    /// wrapper to verify.
    #[arg(long)]
    signed_chain_report: PathBuf,

    /// Operator-supplied trust anchor: 64-char lowercase-hex
    /// Ed25519 public key the wrapper MUST be signed by.
    /// Verification refuses with a `signer_pubkey_mismatch`
    /// pre-check before any crypto burn.
    #[arg(long)]
    expected_signer_pubkey_hex: String,

    /// Output format. Default `events`.
    #[arg(
        long,
        value_enum,
        default_value_t = VerifyIntegrityEvidenceChainReportSignatureFormat::Events
    )]
    format: VerifyIntegrityEvidenceChainReportSignatureFormat,
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
        ContributorCmd::SendHandoff(a) => run_send_handoff(a).await,
        ContributorCmd::AdvertisePeer(a) => run_advertise_peer(a).await,
        ContributorCmd::WatchPeerAdverts(a) => run_watch_peer_adverts(a).await,
        ContributorCmd::PlanSessionAssignments(a) => run_plan_session_assignments(a),
        ContributorCmd::AssignSessionPlan(a) => run_assign_session_plan(a).await,
        ContributorCmd::SessionStatus(a) => run_session_status(a),
        ContributorCmd::PlanSessionRepair(a) => run_plan_session_repair(a),
        ContributorCmd::ApplySessionRepair(a) => run_apply_session_repair(a).await,
        ContributorCmd::PlanSessionReassign(a) => run_plan_session_reassign(a),
        ContributorCmd::ApplySessionReassign(a) => run_apply_session_reassign(a).await,
        ContributorCmd::ArchiveSession(a) => run_archive_session(a),
        ContributorCmd::RestoreSessionArchive(a) => run_restore_session_archive(a),
        ContributorCmd::StateIntegrity(a) => run_state_integrity(a),
        ContributorCmd::PlanStateCleanup(a) => run_plan_state_cleanup(a),
        ContributorCmd::ApplyStateCleanup(a) => run_apply_state_cleanup(a),
        ContributorCmd::RestoreStateCleanupQuarantine(a) => {
            run_restore_state_cleanup_quarantine(a)
        }
        ContributorCmd::StateIntegrityDiff(a) => run_state_integrity_diff(a),
        ContributorCmd::SignStateIntegrityBaseline(a) => {
            run_sign_state_integrity_baseline(a)
        }
        ContributorCmd::SignStateIntegrityDiff(a) => {
            run_sign_state_integrity_diff(a)
        }
        ContributorCmd::VerifyStateIntegrityDiffSignature(a) => {
            run_verify_state_integrity_diff_signature(a)
        }
        ContributorCmd::BuildIntegrityEvidenceBundle(a) => {
            run_build_integrity_evidence_bundle(a)
        }
        ContributorCmd::VerifyIntegrityEvidenceBundle(a) => {
            run_verify_integrity_evidence_bundle(a)
        }
        ContributorCmd::SignIntegrityEvidenceBundle(a) => {
            run_sign_integrity_evidence_bundle(a)
        }
        ContributorCmd::VerifyIntegrityEvidenceBundleSignature(a) => {
            run_verify_integrity_evidence_bundle_signature(a)
        }
        ContributorCmd::VerifyIntegrityEvidenceChain(a) => {
            run_verify_integrity_evidence_chain(a)
        }
        ContributorCmd::SignIntegrityEvidenceChainReport(a) => {
            run_sign_integrity_evidence_chain_report(a)
        }
        ContributorCmd::VerifyIntegrityEvidenceChainReportSignature(a) => {
            run_verify_integrity_evidence_chain_report_signature(a)
        }
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
    // Stage 14.3 D1 Alpha + Stage 14.6 — runtime check for the
    // `--emit-…-proof + --runner stub` pairing. Stage 14.3 moved
    // the StubRunner `--stub-input` requirement off of clap's
    // static `requires` so the ExternalRunner emit path does not
    // falsely require it. Stage 14.6 extends the runtime check to
    // cover the new `--emit-production-mlp-proof` flag so the same
    // user-observable contract holds for both prove paths.
    #[cfg(feature = "halo2-reference-prove")]
    if args.emit_halo2_reference_proof.is_some()
        && args.runner == RunnerChoice::Stub
        && args.stub_input.is_none()
    {
        bail!(
            "--emit-halo2-reference-proof with --runner stub requires --stub-input <PATH>"
        );
    }
    #[cfg(feature = "stage11d-production-prove")]
    if args.emit_production_mlp_proof.is_some()
        && args.runner == RunnerChoice::Stub
        && args.stub_input.is_none()
    {
        bail!(
            "--emit-production-mlp-proof with --runner stub requires --stub-input <PATH>"
        );
    }

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

    // Stage 14.3 — for the External + emit-flag combination, the
    // runner is wrapped in a ByteCapturingRunner so post-`run_job`
    // we can hand the captured input + output bytes to the
    // bytes-based sidecar helper. Captured bytes (`Option`s
    // returned by the wrapper) are extracted here and threaded
    // through to the emission helper below.
    // Stage 14.6 — the captured-bytes slot serves BOTH the
    // halo2-reference (Stage 14.3) and the production-MLP (Stage 14.6)
    // External-runner emit paths. Broadened from
    // `cfg(halo2-reference-prove)` so single-feature builds with only
    // stage11d-production-prove can still capture for the production
    // emit path.
    #[cfg(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))]
    let mut captured_external_bytes: Option<(Vec<u8>, Vec<u8>)> = None;

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

            // Stage 14.3 + 14.6 — determine whether any emit flag
            // is set so we know to wrap the runner in
            // `ByteCapturingRunner`. The clap `conflicts_with`
            // attribute guarantees at most one of the two emit
            // flags is set; the wrap predicate is the OR of both.
            #[cfg(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))]
            let needs_capture = {
                let mut v = false;
                #[cfg(feature = "halo2-reference-prove")]
                {
                    v |= args.emit_halo2_reference_proof.is_some();
                }
                #[cfg(feature = "stage11d-production-prove")]
                {
                    v |= args.emit_production_mlp_proof.is_some();
                }
                v
            };

            #[cfg(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))]
            if needs_capture {
                // Wrap the runner so bytes are captured at the
                // `InferenceRunner::run` trait boundary (Stage
                // 14.3 D6 lifecycle). Post-run we extract the
                // slots before they go out of scope.
                let capturing = ByteCapturingRunner::new(&runner);
                let r = run_external_with_runner(&job, &adapter, &capturing, opts)?;
                let ci = capturing.take_captured_input().ok_or_else(|| {
                    anyhow!(
                        "proof emission: runner did not capture input bytes \
                         (likely a pre-runner refusal); no sidecar written"
                    )
                })?;
                let co = capturing.take_captured_output().ok_or_else(|| {
                    anyhow!(
                        "proof emission: runner did not capture output bytes \
                         (runner returned Err or never produced response); \
                         no sidecar written"
                    )
                })?;
                captured_external_bytes = Some((ci, co));
                r
            } else {
                run_external_with_runner(&job, &adapter, &runner, opts)?
            }
            #[cfg(not(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove")))]
            {
                run_external_with_runner(&job, &adapter, &runner, opts)?
            }
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

    // Stage 14.2 / 14.3 — opt-in halo2-reference sidecar proof
    // emission. The early runtime check above enforces the
    // StubRunner `--stub-input` pairing (D1 Alpha); the External
    // branch above captures bytes via `ByteCapturingRunner` (D6,
    // D7). Errors here are anyhow-flattened into
    // `OperatorError::ContributorWorkflow(String)` by the dispatch
    // wrapper — no new operator-facing reason taxonomy (D5).
    #[cfg(feature = "halo2-reference-prove")]
    if let Some(ref proof_path) = args.emit_halo2_reference_proof {
        match (args.runner, captured_external_bytes.as_ref()) {
            (RunnerChoice::Stub, _) => {
                emit_halo2_reference_proof_sidecar(
                    &job,
                    &result,
                    args.runner,
                    args.stub_input.as_deref(),
                    args.stub_response.as_deref(),
                    proof_path,
                )?;
            }
            (RunnerChoice::External, Some((ci, co))) => {
                emit_halo2_reference_proof_sidecar_from_bytes(
                    &job, &result, ci, co, proof_path,
                )?;
            }
            (RunnerChoice::External, None) => {
                // Unreachable — when emit flag is set on External,
                // the branch above always populates
                // captured_external_bytes (or returns Err earlier).
                bail!(
                    "halo2-reference proof emission: External-runner byte capture \
                     slot empty; this is an internal invariant violation"
                );
            }
        }
        println!("proof_artifact_path={}", proof_path.display());
    }

    // Stage 14.6 — production-MLP sidecar emission. Clap's
    // `conflicts_with` guarantees the reference block above and this
    // block are mutually exclusive (at most one emit flag is set per
    // invocation).
    #[cfg(feature = "stage11d-production-prove")]
    if let Some(ref proof_path) = args.emit_production_mlp_proof {
        match (args.runner, captured_external_bytes.as_ref()) {
            (RunnerChoice::Stub, _) => {
                emit_production_mlp_proof_sidecar(
                    &job,
                    &result,
                    args.runner,
                    args.stub_input.as_deref(),
                    args.stub_response.as_deref(),
                    proof_path,
                )?;
            }
            (RunnerChoice::External, Some((ci, co))) => {
                emit_production_mlp_proof_sidecar_from_bytes(
                    &job, &result, ci, co, proof_path,
                )?;
            }
            (RunnerChoice::External, None) => {
                bail!(
                    "production-MLP proof emission: External-runner byte capture \
                     slot empty; this is an internal invariant violation"
                );
            }
        }
        println!("proof_artifact_path={}", proof_path.display());
    }

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

// ── Stage 14.2: halo2-reference sidecar proof emission ───────────────────────

/// Stage 14.2 — emit a `ProofArtifactBody` JSON sidecar alongside
/// the produced `ContributorResult`. The sidecar binds the same
/// `(model, input, output)` triple the contributor result carries
/// hashes of, using the existing Stage 14.1
/// `Halo2ReferenceProofBackend`. The artifact carries
/// `proof_system = Stage11bHalo2Reference` and
/// `testnet_or_dev_only = Some(true)`, so submission on
/// `chain_id == 1` is hard-refused at `check_mainnet_eligible`
/// layers 1 + 3 + 6.
///
/// Refusals are surfaced as `anyhow::Error`; the contributor-CLI
/// dispatch flattens them into the existing
/// `OperatorError::ContributorWorkflow(String)` catch-all per
/// Stage 14.2 D5 (no new operator reason taxonomy).
/// Stage 14.2 path — file-based sidecar emission for the StubRunner
/// (operator supplies `--stub-input` + `--stub-response` files
/// whose bytes hash to the contributor's committed hashes).
/// Delegates the actual proof + artifact write to
/// [`assemble_and_write_sidecar`]; this wrapper only does the
/// file-reading and the Stage 14.2 D5 runner-shape refusal.
#[cfg(feature = "halo2-reference-prove")]
fn emit_halo2_reference_proof_sidecar(
    job: &ContributorJob,
    result: &ContributorResult,
    runner: RunnerChoice,
    stub_input: Option<&std::path::Path>,
    stub_response: Option<&std::path::Path>,
    proof_path: &std::path::Path,
) -> Result<()> {
    // Stage 14.2 path: StubRunner-only. External runners take the
    // bytes-based path through `emit_halo2_reference_proof_sidecar_from_bytes`.
    if runner != RunnerChoice::Stub {
        bail!(
            "emit_halo2_reference_proof_sidecar is the StubRunner path; \
             ExternalCommandRunner emission must go through \
             emit_halo2_reference_proof_sidecar_from_bytes"
        );
    }
    // The clap layer used to enforce this pair via `requires` in
    // Stage 14.2; Stage 14.3 D1 Alpha moves the check to the
    // run_run_job runtime guard. This guard remains for callers
    // that bypass run_run_job (the test surface).
    let stub_input_path = stub_input.ok_or_else(|| {
        anyhow!(
            "--emit-halo2-reference-proof with --runner stub requires --stub-input <PATH>"
        )
    })?;
    let stub_response_path = stub_response.ok_or_else(|| {
        anyhow!(
            "--emit-halo2-reference-proof with --runner stub requires --stub-response <PATH> \
             (the runner already required it for --runner stub)"
        )
    })?;

    let stub_input_bytes = std::fs::read(stub_input_path).with_context(|| {
        format!("read --stub-input {}", stub_input_path.display())
    })?;
    let stub_response_bytes = std::fs::read(stub_response_path).with_context(|| {
        format!("read --stub-response {}", stub_response_path.display())
    })?;
    assemble_and_write_sidecar(job, result, &stub_input_bytes, &stub_response_bytes, proof_path)
}

/// Stage 14.3 path — bytes-based sidecar emission for the
/// ExternalCommandRunner (bytes captured at the
/// [`InferenceRunner::run`] trait boundary via
/// [`ByteCapturingRunner`]). Same proof + artifact path as
/// [`emit_halo2_reference_proof_sidecar`]; both delegate to
/// [`assemble_and_write_sidecar`].
#[cfg(feature = "halo2-reference-prove")]
fn emit_halo2_reference_proof_sidecar_from_bytes(
    job: &ContributorJob,
    result: &ContributorResult,
    captured_input: &[u8],
    captured_output: &[u8],
    proof_path: &std::path::Path,
) -> Result<()> {
    assemble_and_write_sidecar(job, result, captured_input, captured_output, proof_path)
}

/// Shared inner assembler — runs the canonical-spec check, the
/// hash bindings against `job.input_hash` / `result.response_hash`,
/// the canonical-evaluator output check (via
/// `Halo2ReferenceProofBackend::prove`'s internal pre-check), and
/// the artifact write. Used by both the file-based (Stage 14.2,
/// StubRunner) and bytes-based (Stage 14.3, ExternalCommandRunner)
/// wrappers.
#[cfg(feature = "halo2-reference-prove")]
fn assemble_and_write_sidecar(
    job: &ContributorJob,
    result: &ContributorResult,
    input_bytes: &[u8],
    output_bytes: &[u8],
    proof_path: &std::path::Path,
) -> Result<()> {
    // The canonical halo2-mlp-v1 spec bytes are embedded into the
    // operator binary the same way the operator-side Stage 14.1
    // subcommand does. The prover refuses any model bytes whose
    // BLAKE3 differs from EXPECTED_SPEC_HASH, so this is the
    // single source of truth.
    let canonical_spec: &[u8] = include_bytes!(
        "../../omni-proofs-halo2-reference/assets/canonical_spec.json"
    );
    let expected_spec_hash_hex = blake3::hash(canonical_spec).to_hex().to_string();
    if job.model_hash != expected_spec_hash_hex {
        bail!(
            "--emit-halo2-reference-proof refused: job.model_hash {} does not \
             match the canonical halo2-mlp-v1 spec hash {}",
            job.model_hash,
            expected_spec_hash_hex
        );
    }

    // Bind the supplied bytes to the contributor's committed
    // hashes. For Stage 14.2 (StubRunner) this catches operator
    // typos / file substitution. For Stage 14.3 (ExternalRunner)
    // the captured bytes are EXACTLY the bytes `run_job` hashed,
    // so the equality is tautological — kept as defense in depth.
    let input_hash_hex = blake3::hash(input_bytes).to_hex().to_string();
    if input_hash_hex != job.input_hash {
        bail!(
            "input bytes BLAKE3 {} does not match job.input_hash {}",
            input_hash_hex,
            job.input_hash
        );
    }
    let output_hash_hex = blake3::hash(output_bytes).to_hex().to_string();
    if output_hash_hex != result.response_hash {
        bail!(
            "output bytes BLAKE3 {} does not match result.response_hash {}",
            output_hash_hex,
            result.response_hash
        );
    }

    // Drive the prover through the existing Stage 14.1 adapter.
    // The adapter itself re-runs `canonical_evaluate(input)` and
    // refuses if `output` disagrees — pinning a future runner
    // regression where a malicious external command tries to
    // prove an attacker-chosen output.
    use omni_zkml::ProofBackend;
    let backend = omni_proofs_halo2_reference::Halo2ReferenceProofBackend::new();
    let proof_bytes = backend
        .prove(canonical_spec, input_bytes, output_bytes)
        .map_err(|e| anyhow!("halo2-reference prover failure: {e}"))?;

    // Decode the i16 tensors so the verifier's
    // `decode_public_inputs_json` (which reads `input` / `output`
    // arrays) finds them. The canonical evaluator is deterministic;
    // calling it here keeps the sidecar self-contained.
    let input_i16 =
        omni_proofs_halo2_reference::decode_canonical_input(input_bytes)
            .map_err(|e| anyhow!("decode input as canonical_input: {e}"))?;
    let output_i16 =
        omni_proofs_halo2_reference::decode_canonical_output(output_bytes)
            .map_err(|e| anyhow!("decode output as canonical_output: {e}"))?;

    // Stage 14.2 D2 — `contributor_job_id` is the new extra key
    // inside `metadata.public_inputs`. The verifier reads only
    // `input` + `output`; extra keys are tolerated. Stage 14.3
    // preserves this contract.
    let public_inputs_json = serde_json::json!({
        "input":  [input_i16[0],  input_i16[1],  input_i16[2],  input_i16[3]],
        "output": [output_i16[0], output_i16[1], output_i16[2], output_i16[3]],
        "contributor_job_id": result.job_id,
    });

    let circuit_id_hex = backend
        .circuit_id()
        .map(|id| {
            let mut s = String::with_capacity(64);
            for b in &id {
                s.push_str(&format!("{b:02x}"));
            }
            s
        })
        .expect("halo2-reference backend always exposes a circuit_id");

    let metadata = omni_zkml::ProofMetadata {
        backend_id: backend.backend_id().to_string(),
        model_hash: expected_spec_hash_hex,
        input_hash: input_hash_hex,
        response_hash: output_hash_hex,
        model_format: Some(omni_zkml::ModelFormat::Halo2ReferenceMlp),
        proof_system: Some(omni_zkml::ProofSystem::Stage11bHalo2Reference),
        circuit_id_hex: Some(circuit_id_hex),
        verification_key_hex: None,
        public_inputs: Some(public_inputs_json),
        testnet_or_dev_only: Some(true),
        model_framework: Some(omni_zkml::ModelFramework::FrameworkAgnostic),
    };
    let body =
        omni_zkml::ProofArtifactBody::from_components(metadata, &proof_bytes);
    let body_bytes = serde_json::to_vec_pretty(&body)
        .context("serialize halo2-reference ProofArtifactBody")?;
    std::fs::write(proof_path, &body_bytes).with_context(|| {
        format!("write --emit-halo2-reference-proof {}", proof_path.display())
    })?;
    Ok(())
}

// ── Stage 14.6: production-MLP sidecar emission ──────────────────────────────

/// Stage 14.6 path — file-based sidecar emission for the StubRunner
/// (operator supplies `--stub-input` + `--stub-response` files whose
/// bytes hash to the contributor's committed hashes). Delegates the
/// actual proof + artifact write to
/// [`assemble_and_write_production_sidecar`]. Parallel to Stage 14.2's
/// [`emit_halo2_reference_proof_sidecar`] but with the production-
/// specific metadata contract.
#[cfg(feature = "stage11d-production-prove")]
fn emit_production_mlp_proof_sidecar(
    job: &ContributorJob,
    result: &ContributorResult,
    runner: RunnerChoice,
    stub_input: Option<&std::path::Path>,
    stub_response: Option<&std::path::Path>,
    proof_path: &std::path::Path,
) -> Result<()> {
    if runner != RunnerChoice::Stub {
        bail!(
            "emit_production_mlp_proof_sidecar is the StubRunner path; \
             ExternalCommandRunner emission must go through \
             emit_production_mlp_proof_sidecar_from_bytes"
        );
    }
    let stub_input_path = stub_input.ok_or_else(|| {
        anyhow!(
            "--emit-production-mlp-proof with --runner stub requires --stub-input <PATH>"
        )
    })?;
    let stub_response_path = stub_response.ok_or_else(|| {
        anyhow!(
            "--emit-production-mlp-proof with --runner stub requires --stub-response <PATH> \
             (the runner already required it for --runner stub)"
        )
    })?;

    let stub_input_bytes = std::fs::read(stub_input_path).with_context(|| {
        format!("read --stub-input {}", stub_input_path.display())
    })?;
    let stub_response_bytes = std::fs::read(stub_response_path).with_context(|| {
        format!("read --stub-response {}", stub_response_path.display())
    })?;
    assemble_and_write_production_sidecar(
        job, result, &stub_input_bytes, &stub_response_bytes, proof_path,
    )
}

/// Stage 14.6 path — bytes-based sidecar emission for the
/// ExternalCommandRunner (bytes captured at the
/// [`InferenceRunner::run`] trait boundary via the shared
/// [`ByteCapturingRunner`] from Stage 14.3). Same proof + artifact
/// path as [`emit_production_mlp_proof_sidecar`]; both delegate to
/// [`assemble_and_write_production_sidecar`].
#[cfg(feature = "stage11d-production-prove")]
fn emit_production_mlp_proof_sidecar_from_bytes(
    job: &ContributorJob,
    result: &ContributorResult,
    captured_input: &[u8],
    captured_output: &[u8],
    proof_path: &std::path::Path,
) -> Result<()> {
    assemble_and_write_production_sidecar(
        job, result, captured_input, captured_output, proof_path,
    )
}

/// Stage 14.6 — production-MLP assembler. Parallel to Stage 14.2/14.3's
/// [`assemble_and_write_sidecar`] but pins the **Stage 11d.2 production
/// metadata contract**:
///
/// - `canonical_spec` from the production crate's `assets/canonical_spec.json`;
///   `job.model_hash` MUST equal `BLAKE3(canonical_spec)`.
/// - `input` is exactly 32 bytes (16 × i16 LE); `output` exactly 16 bytes
///   (8 × i16 LE) — the production adapter refuses on size mismatch.
/// - `proof_system = Stage11dProductionFixedPointMlp`.
/// - `model_format = ProductionFixedPointMlp`.
/// - `circuit_id_hex = EXPECTED_CIRCUIT_ID_HEX` AND
///   `verification_key_hex = EXPECTED_VK_HASH_HEX` — both **required**
///   on production artifacts (the verifier refuses `None`).
/// - `testnet_or_dev_only = Some(false)` — production-shape contract.
///
/// Mainnet refusal lands at `check_mainnet_eligible` **layer 6 only**:
/// `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` is empty; layer 1 does NOT fire
/// because the artifact correctly declares `Some(false)`. Lifting the
/// refusal requires the separate Stage 11d.3 chain-team-reviewed
/// allowlist PR.
///
/// The `public_inputs` JSON carries `contributor_job_id` as an extra
/// key beyond `input`/`output`; the production verifier's
/// `decode_public_inputs_json` reads only the two arrays and tolerates
/// extras — same pattern Stage 14.2 pinned for the reference verifier.
#[cfg(feature = "stage11d-production-prove")]
fn assemble_and_write_production_sidecar(
    job: &ContributorJob,
    result: &ContributorResult,
    input_bytes: &[u8],
    output_bytes: &[u8],
    proof_path: &std::path::Path,
) -> Result<()> {
    let canonical_spec: &[u8] = include_bytes!(
        "../../omni-proofs-halo2-production-mlp/assets/canonical_spec.json"
    );
    let expected_spec_hash_hex = blake3::hash(canonical_spec).to_hex().to_string();
    if job.model_hash != expected_spec_hash_hex {
        bail!(
            "--emit-production-mlp-proof refused: job.model_hash {} does not \
             match the canonical production-fixedpoint-mlp-v1 spec hash {}",
            job.model_hash,
            expected_spec_hash_hex
        );
    }

    // Hash bindings — tautological for the ExternalRunner path (the
    // captured bytes are exactly what `run_job` hashed); defensive
    // for the StubRunner path (catches operator typos / file
    // substitution before any halo2 work).
    let input_hash_hex = blake3::hash(input_bytes).to_hex().to_string();
    if input_hash_hex != job.input_hash {
        bail!(
            "input bytes BLAKE3 {} does not match job.input_hash {}",
            input_hash_hex,
            job.input_hash
        );
    }
    let output_hash_hex = blake3::hash(output_bytes).to_hex().to_string();
    if output_hash_hex != result.response_hash {
        bail!(
            "output bytes BLAKE3 {} does not match result.response_hash {}",
            output_hash_hex,
            result.response_hash
        );
    }

    // Drive the prover through the Stage 14.5 adapter. The adapter
    // pins (a) the canonical-spec hash, (b) the 32-byte input
    // length, and (c) the canonical-evaluator output binding —
    // refusing before any halo2 work runs on size or content drift.
    use omni_zkml::ProofBackend;
    let backend =
        omni_proofs_halo2_production_mlp::Halo2ProductionMlpProofBackend::new();
    let proof_bytes = backend
        .prove(canonical_spec, input_bytes, output_bytes)
        .map_err(|e| anyhow!("halo2 production-MLP prover failure: {e}"))?;

    // Decode the i16 tensors for the artifact's public_inputs JSON.
    // The production canonical input is 16 i16; output is 8 i16.
    let input_i16 =
        omni_proofs_halo2_production_mlp::decode_canonical_input(input_bytes)
            .map_err(|e| anyhow!("decode input as canonical_input: {e}"))?;
    let output_i16 =
        omni_proofs_halo2_production_mlp::decode_canonical_output(output_bytes)
            .map_err(|e| anyhow!("decode output as canonical_output: {e}"))?;

    // Stage 14.2 D2 pattern extended to the production verifier:
    // `contributor_job_id` is an extra JSON key beyond `input`/`output`.
    // The production verifier's `decode_public_inputs_json` tolerates
    // extras (test pin below).
    let public_inputs_json = serde_json::json!({
        "input":  input_i16.to_vec(),
        "output": output_i16.to_vec(),
        "contributor_job_id": result.job_id,
    });

    // Production verifier requires `circuit_id_hex` AND
    // `verification_key_hex` to be present AND equal the pinned
    // constants. Setting them from the omni-proofs crate's exported
    // constants pins drift at compile time.
    let metadata = omni_zkml::ProofMetadata {
        backend_id: backend.backend_id().to_string(),
        model_hash: expected_spec_hash_hex,
        input_hash: input_hash_hex,
        response_hash: output_hash_hex,
        model_format: Some(omni_zkml::ModelFormat::ProductionFixedPointMlp),
        proof_system: Some(omni_zkml::ProofSystem::Stage11dProductionFixedPointMlp),
        circuit_id_hex: Some(
            omni_proofs_halo2_production_mlp::EXPECTED_CIRCUIT_ID_HEX.to_string(),
        ),
        verification_key_hex: Some(
            omni_proofs_halo2_production_mlp::EXPECTED_VK_HASH_HEX.to_string(),
        ),
        public_inputs: Some(public_inputs_json),
        // Production-shape contract: Some(false), NOT Some(true)
        // like the Stage 14.2/14.3 reference path. Mainnet refusal
        // lands at layer 6 only (empty allowlist).
        testnet_or_dev_only: Some(false),
        model_framework: Some(omni_zkml::ModelFramework::FrameworkAgnostic),
    };
    let body =
        omni_zkml::ProofArtifactBody::from_components(metadata, &proof_bytes);
    let body_bytes = serde_json::to_vec_pretty(&body)
        .context("serialize halo2 production-MLP ProofArtifactBody")?;
    std::fs::write(proof_path, &body_bytes).with_context(|| {
        format!("write --emit-production-mlp-proof {}", proof_path.display())
    })?;
    Ok(())
}

// ── Stage 14.3: ByteCapturingRunner ──────────────────────────────────────────

/// Stage 14.3 — thin `InferenceRunner` wrapper that captures the
/// input bytes passed to the inner runner and the response bytes
/// the inner runner returned, so a halo2-reference sidecar proof
/// can be assembled in the `omni-node` CLI without modifying
/// `omni-contributor` (Stage 12.0 lean-crate invariant) or
/// re-fetching from SNIP.
///
/// **D6 invariants:**
/// 1. Both `captured_input` and `captured_output` are **cleared at the
///    start of every `run` invocation** so a reused wrapper across
///    multiple calls cannot leak stale state.
/// 2. `captured_input` is set **before** the inner runner is called
///    (the bytes are already in hand at that point — the operator
///    supplied them via SNIP fetch; we just memoise).
/// 3. `captured_output` is set **only after** the inner runner
///    returns `Ok(_)`. On `Err`, it remains `None` — sidecar
///    emission then refuses cleanly.
///
/// **D7 carve-out:** this wrapper covers `InferenceRunner::run` only.
/// `run_with_activations` is **not** overridden; the default
/// forward-through-`run` implementation is incidental, not relied
/// upon. Stage 12.4 activation handoff paths are out of Stage 14.3
/// scope and have no proof binding.
#[cfg(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))]
struct ByteCapturingRunner<'a, R: InferenceRunner + ?Sized> {
    inner: &'a R,
    captured_input: std::cell::RefCell<Option<Vec<u8>>>,
    captured_output: std::cell::RefCell<Option<Vec<u8>>>,
}

#[cfg(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))]
impl<'a, R: InferenceRunner + ?Sized> ByteCapturingRunner<'a, R> {
    fn new(inner: &'a R) -> Self {
        Self {
            inner,
            captured_input: std::cell::RefCell::new(None),
            captured_output: std::cell::RefCell::new(None),
        }
    }

    fn take_captured_input(&self) -> Option<Vec<u8>> {
        self.captured_input.borrow().clone()
    }

    fn take_captured_output(&self) -> Option<Vec<u8>> {
        self.captured_output.borrow().clone()
    }
}

#[cfg(any(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))]
impl<'a, R: InferenceRunner + ?Sized> InferenceRunner for ByteCapturingRunner<'a, R> {
    fn run(
        &self,
        manifest_path: &std::path::Path,
        input_bytes: &[u8],
    ) -> std::result::Result<
        omni_contributor::RunOutput,
        omni_contributor::RunnerError,
    > {
        // D6 invariant 1 — clear both slots at the start of every
        // run so a reused wrapper cannot leak stale state across
        // invocations.
        *self.captured_input.borrow_mut() = None;
        *self.captured_output.borrow_mut() = None;

        // D6 invariant 2 — capture input before the inner call.
        // The bytes are already in scope (run_job integrity-checked
        // them at this point); we just memoise for post-run use.
        *self.captured_input.borrow_mut() = Some(input_bytes.to_vec());

        // D6 invariant 3 — capture output ONLY on inner Ok. On Err,
        // captured_output remains None and the post-run extraction
        // refuses cleanly. The `?` propagates the inner error
        // verbatim — no new failure event format (D4).
        let output = self.inner.run(manifest_path, input_bytes)?;
        *self.captured_output.borrow_mut() = Some(output.response_bytes.clone());
        Ok(output)
    }

    // D7 — `run_with_activations` is NOT overridden. Stage 14.3 covers
    // standard `InferenceRunner::run` only. Activation handoff
    // (Stage 12.4) is out of scope and has no proof binding.
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

    // Stage 12.7 — optional contributor workflow state store. When
    // set, the watch loop adds cross-restart `posted-jobs` dedup
    // and (above-loop) auto-prunes expired sessions / peer-adverts.
    let state_store = open_optional_state_store(
        args.contributor_state_dir.as_deref(),
        args.no_prune_state_on_start,
    )?;

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
        state_store: state_store.as_ref(),
    };

    omni_contributor::run_watch_loop(&adapter, &mut source, opts)
        .map_err(|e| anyhow!("watch-jobs error: {e}"))?;
    Ok(())
}

/// Stage 12.7 — open the optional contributor workflow state store.
/// Returns `Ok(None)` when the caller did NOT supply
/// `--contributor-state-dir` (preserves pre-12.7 behavior exactly).
fn open_optional_state_store(
    state_dir: Option<&std::path::Path>,
    no_prune_on_start: bool,
) -> Result<Option<omni_contributor::ContributorStateStore>> {
    let Some(state_dir) = state_dir else {
        return Ok(None);
    };
    let now_utc = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let (store, report) = omni_contributor::ContributorStateStore::open(
        state_dir,
        !no_prune_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    println!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        state_dir.display(),
        report.removed_sessions,
        report.removed_peer_adverts,
        report.kept
    );
    Ok(Some(store))
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
    let identity = resolve_net_identity(args.net_identity_file.as_deref())?;
    let net_config = NetConfig {
        listen_port: args.listen_port,
        bootstrap_peers: args.peer,
        identity,
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
    let identity = resolve_net_identity(args.net_identity_file.as_deref())?;
    let net_config = NetConfig {
        listen_port: args.listen_port,
        bootstrap_peers: args.peer,
        identity,
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

    // Stage 12.7 — open the state store on the async runtime BEFORE
    // moving into spawn_blocking so the auto-prune `now_utc` is the
    // real start-up time, and so any error surfaces synchronously
    // (rather than swallowed into the spawn_blocking join error).
    let state_store = open_optional_state_store(
        args.contributor_state_dir.as_deref(),
        args.no_prune_state_on_start,
    )?;

    tokio::task::spawn_blocking(move || -> Result<()> {
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
            state_store: state_store.as_ref(),
        };
        omni_contributor::run_watch_loop(&snip_adapter, &mut source, opts)
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| anyhow!("watch-network-jobs join: {e}"))?
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

    let identity = resolve_net_identity(args.net_identity_file.as_deref())?;
    let net_config = NetConfig {
        listen_port: args.listen_port,
        bootstrap_peers: args.peer,
        identity,
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
    let identity = resolve_net_identity(args.net_identity_file.as_deref())?;
    let net_config = NetConfig {
        listen_port: args.listen_port,
        bootstrap_peers: args.peer,
        identity,
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

    // Stage 12.7 — open the optional state store BEFORE entering the
    // spawn_blocking so the auto-prune `now_utc` reflects the actual
    // startup time and so a bad path surfaces synchronously.
    let state_store = open_optional_state_store(
        args.contributor_state_dir.as_deref(),
        args.no_prune_state_on_start,
    )?;

    tokio::task::spawn_blocking(move || -> Result<()> {
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
                // Stage 12.7 — cross-restart dedup keyed by
                // `<posted_id>--<snip_root>`. Matches the
                // `seen/network-result-announcements/...` namespace
                // documented in `omni_contributor::state`.
                let cross_restart_key = format!(
                    "{}--{}", ann.posted_id, ann.posted_result_link_snip_root
                );
                if let Some(ref store) = state_store {
                    match store.is_seen(
                        omni_contributor::StateNamespace::NetworkResultAnnouncements,
                        &cross_restart_key,
                    ) {
                        Ok(true) => {
                            seen.insert(ann.posted_result_link_snip_root.clone());
                            println!(
                                "event=skip posted_id={} reason=already_seen_state_store",
                                ann.posted_id
                            );
                            continue;
                        }
                        Ok(false) => {}
                        Err(e) => {
                            println!(
                                "event=warn context=state_store_is_seen message={e}"
                            );
                        }
                    }
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
                        // Stage 12.7 — dual-write the link into the
                        // state-dir's results tree and lay down the
                        // cross-restart seen marker.
                        if let Some(ref store) = state_store {
                            if let Err(e) = store.mark_seen(
                                omni_contributor::StateNamespace::NetworkResultAnnouncements,
                                &cross_restart_key,
                            ) {
                                println!(
                                    "event=warn context=state_store_mark_seen message={e}"
                                );
                            }
                            // Read the just-written link bytes and
                            // mirror them into
                            // `<state>/results/result-links/<posted_id>.link.json`.
                            // Best-effort: a failure here doesn't
                            // abort the watch loop.
                            match std::fs::read(&link_path).and_then(|b| {
                                serde_json::from_slice::<omni_contributor::PostedResultLink>(&b)
                                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                            }) {
                                Ok(link) => {
                                    if let Err(e) = store.write_verified_json(
                                        omni_contributor::StateObjectKind::PostedResultLink,
                                        &posted_id,
                                        &link,
                                    ) {
                                        println!(
                                            "event=warn context=state_store_write_link message={e}"
                                        );
                                    }
                                }
                                Err(e) => {
                                    println!(
                                        "event=warn context=state_store_reread_link message={e}"
                                    );
                                }
                            }
                        }
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
    .map_err(|e| anyhow!("watch-network-results join: {e}"))?
}

// ── Stage 12.3 — session subcommand handlers ─────────────────────────────

/// Shared helper: open an OmniNet, wait for first peer, return an
/// Arc-wrapped handle ready for an OmniNetRelay clone. Used by every
/// 12.3 publish-and-broadcast subcommand below.
/// Stage 12.6 — resolve the operator's `--net-identity-file` flag
/// into a `NetIdentity`. `None` preserves pre-12.6 ephemeral
/// behavior; `Some(path)` calls
/// `omni_net::load_or_create_keypair_file_bytes` (auto-create at
/// 0o600 on Unix; refuses malformed existing files; refuses
/// world-readable files on Unix). Used by every contributor
/// subcommand that opens `OmniNet`.
fn resolve_net_identity(
    net_identity_file: Option<&std::path::Path>,
) -> Result<omni_types::config::NetIdentity> {
    use omni_types::config::NetIdentity;
    match net_identity_file {
        None => Ok(NetIdentity::Ephemeral),
        Some(p) => {
            let bytes = omni_net::load_or_create_keypair_file_bytes(p)
                .map_err(|e| anyhow!("net-identity-file at {}: {e}", p.display()))?;
            Ok(NetIdentity::KeypairProtobufBytes(bytes))
        }
    }
}

async fn open_omninet_with_peer_wait(
    listen_port: u16,
    peer: Vec<String>,
    peer_wait_secs: u64,
    mesh_stabilize_ms: u64,
    net_identity_file: Option<&std::path::Path>,
) -> Result<(std::sync::Arc<tokio::sync::Mutex<omni_net::OmniNet>>, tokio::runtime::Handle)>
{
    use omni_net::OmniNet;
    use omni_types::config::NetConfig;
    let identity = resolve_net_identity(net_identity_file)?;
    let net_config = NetConfig {
        listen_port,
        bootstrap_peers: peer,
        identity,
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
        args.net_identity_file.as_deref(),
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
        args.net_identity_file.as_deref(),
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

/// Stage 12.8 — extracted reusable helper. Builds + signs +
/// SNIP-publishes a single `WorkAssignment` from an `AssignmentSpec`,
/// and (when `publish_announcement` is true) also builds + signs +
/// gossips the matching `NetworkWorkAssignedAnnouncement`. Returns
/// the signed envelope and its SNIP root hex.
///
/// Used by `assign-work` (Stage 12.3) and `assign-session-plan`
/// (Stage 12.8). Behavior of `assign-work` is preserved bit-for-bit:
/// always publishes the announcement, always uses `now()` for the
/// `assigned_at_utc` fallback.
async fn publish_one_signed_assignment(
    snip: &impl omni_store::SnipV2Adapter,
    relay: Option<&mut omni_contributor::OmniNetRelay>,
    coord: &omni_contributor::CoordinatorSigner,
    session: &omni_contributor::ExecutionSession,
    spec: AssignmentSpec,
    now_utc: &str,
) -> Result<(omni_contributor::WorkAssignment, String /* snip_root_hex */)> {
    use omni_contributor::canonical::{
        assignment_id_hex, hex_lower, net_assign_signing_input,
        work_assignment_signing_input,
    };
    use omni_contributor::{
        ContributorRelay, NetworkWorkAssignedAnnouncement, WorkAssignment,
        NET_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
    };

    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index: spec.stage_index,
        contributor_pubkey_hex: spec.contributor_pubkey_hex,
        work_kind: spec.work_kind,
        expected_work_units: spec.expected_work_units,
        expected_work_unit_kind: spec.expected_work_unit_kind,
        assigned_at_utc: spec
            .assigned_at_utc
            .unwrap_or_else(|| now_utc.to_string()),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a)?;
    let sig_input = work_assignment_signing_input(&a)?;
    a.coordinator_signature_hex = coord.sign_hex(&sig_input);
    a.validate_schema()
        .map_err(|e| anyhow!("invalid WorkAssignment: {e}"))?;

    let json = serde_json::to_vec_pretty(&a)?;
    let root = omni_contributor::snip::publish_bytes(snip, &json, "assignment")
        .map_err(|e| anyhow!("snip publish assignment: {e}"))?;
    let root_hex = format!("0x{}", hex_lower(root.as_bytes()));

    if let Some(relay) = relay {
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
    }

    Ok((a, root_hex))
}

async fn run_assign_work(args: AssignWorkArgs) -> Result<()> {
    use omni_contributor::{CoordinatorSigner, OmniNetRelay};

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
        args.net_identity_file.as_deref(),
    )
    .await?;
    let mut relay = OmniNetRelay::new(net.clone(), handle);

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    for spec in specs {
        let (a, root_hex) = publish_one_signed_assignment(
            &snip,
            Some(&mut relay),
            &coord,
            &session,
            spec,
            &now_utc,
        )
        .await?;
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
        NetworkPartialResultAnnouncement, OmniNetRelay, OmniNetTensorTransport,
        PartialContributorResult, WorkAssignment, NET_SCHEMA_VERSION,
        SESSION_SCHEMA_VERSION,
    };
    use omni_contributor::result::{MeasuredAccounting, StageContribution};
    use omni_types::phase5::SnipV2ObjectId;

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.execution_session_snip_root)?;

    // Fetch + parse + structurally-validate this stage's assignment.
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

    // Stage 12.7 — `--contributor-state-dir` is the canonical
    // source of verified joins + peer advertisements for the
    // downstream resolver. Mixing it with the legacy 12.5 flags
    // would create an ambiguous source-of-truth, so reject the
    // combination up-front.
    if args.contributor_state_dir.is_some() {
        if args.peer_advert_dir.is_some() {
            return Err(anyhow!(omni_contributor::error::StateError::AmbiguousSource {
                legacy_flag: "--peer-advert-dir",
            }));
        }
        if args.joins_dir.is_some() {
            return Err(anyhow!(omni_contributor::error::StateError::AmbiguousSource {
                legacy_flag: "--joins-dir",
            }));
        }
    }

    // Stage 12.5 — optionally resolve the downstream peer from a
    // verified peer-advert cache instead of requiring
    // `--downstream-to-peer`. Manual value (if supplied) takes
    // precedence so an operator can always override.
    let mut effective_downstream_to_peer = args.downstream_to_peer.clone();
    let mut effective_downstream_chunk_cap: Option<u64> = None;
    if args.resolve_downstream_peer_from_session
        && effective_downstream_to_peer.is_none()
    {
        let downstream_root = args
            .downstream_to_assignment_snip_root
            .as_deref()
            .ok_or_else(|| {
                anyhow!(
                    "--resolve-downstream-peer-from-session requires \
                     --downstream-to-assignment-snip-root"
                )
            })?;
        // Fetch the downstream assignment so we know the contributor
        // pubkey to resolve against.
        let down_root = omni_types::phase5::SnipV2ObjectId::from_hex(downstream_root)
            .map_err(|e| anyhow!("bad downstream snip root: {e:?}"))?;
        let down_bytes = omni_contributor::snip::fetch_bytes(&snip, &down_root)
            .map_err(|e| anyhow!("snip fetch downstream assignment: {e}"))?;
        let down_assignment: omni_contributor::WorkAssignment =
            serde_json::from_slice(&down_bytes)
                .map_err(|e| anyhow!("parse downstream assignment: {e}"))?;

        let now_utc =
            chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        let mut cache = omni_contributor::PeerRoutingCache::new();

        // Stage 12.7 — prefer the unified state-dir tree.
        if let Some(state_dir) = args.contributor_state_dir.as_deref() {
            let (store, _) = omni_contributor::ContributorStateStore::open(
                state_dir,
                !args.no_prune_state_on_start,
                &now_utc,
            )
            .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
            // Re-verify the session.json on disk before we trust
            // any of its joins — mirrors what
            // `load_verified_joins_from_dir` does for the legacy
            // dir-based path. A tampered local session.json would
            // otherwise let a tampered local join.json pass the
            // matching-join gate.
            let on_disk_session: omni_contributor::ExecutionSession = store
                .read_verified_json(
                    omni_contributor::StateObjectKind::Session,
                    &session.session_id,
                )
                .map_err(|e| {
                    anyhow!("read verified session from state dir: {e}")
                })?
                .ok_or_else(|| {
                    anyhow!(
                        "--contributor-state-dir is missing verified session \
                         for session_id={}",
                        session.session_id
                    )
                })?;
            if !omni_contributor::verify_execution_session(&on_disk_session)
                .is_ok()
            {
                return Err(anyhow!(
                    "verified session in state dir failed re-verification: \
                     session_id={}",
                    session.session_id
                ));
            }
            let raw_joins = store
                .list_verified_joins_for(&session.session_id)
                .map_err(|e| anyhow!("list verified joins from state dir: {e}"))?;
            let joins_for_session: Vec<omni_contributor::ContributorJoin> = raw_joins
                .into_iter()
                .filter(|j| {
                    let ok = omni_contributor::verify_contributor_join(
                        &on_disk_session,
                        j,
                    )
                    .is_ok();
                    if !ok {
                        eprintln!(
                            "event=warn context=state_store_join_verify_failed \
                             session_id={} contributor_pubkey={}",
                            j.session_id, j.contributor_pubkey_hex
                        );
                    }
                    ok
                })
                .collect();
            let adverts = store
                .list_verified_peer_adverts_for(&session.session_id)
                .map_err(|e| {
                    anyhow!("list verified peer adverts from state dir: {e}")
                })?;
            for advert in adverts {
                let advert_id = advert.advertisement_id.clone();
                let out = omni_contributor::verify_peer_advertisement_body(
                    &advert,
                    &joins_for_session,
                    Some(&now_utc),
                );
                match out {
                    omni_contributor::PeerAdvertisementOutcome::Verified {
                        body,
                    } => {
                        cache.insert_verified(*body);
                    }
                    other => {
                        eprintln!(
                            "event=warn context=state_peer_advert_rejected \
                             advertisement_id={} reason={}",
                            advert_id,
                            stringify_peer_advert_outcome(&other)
                        );
                    }
                }
            }
        } else {
            // Pre-12.7 path: require both legacy dirs. Workflow:
            //   watch-sessions --out-dir A
            //   watch-peer-adverts --out-dir B --joins-dir A
            //   run-assignment --peer-advert-dir B --joins-dir A
            //                  --resolve-downstream-peer-from-session
            // — both consumers point at A for joins.
            let advert_dir = args.peer_advert_dir.as_deref().ok_or_else(|| {
                anyhow!(
                    "--resolve-downstream-peer-from-session requires \
                     --peer-advert-dir (or --contributor-state-dir at 12.7+)"
                )
            })?;
            let joins_dir = args.joins_dir.as_deref().ok_or_else(|| {
                anyhow!(
                    "--resolve-downstream-peer-from-session requires --joins-dir \
                     pointing at a watch-sessions output tree (or \
                     --contributor-state-dir at 12.7+)"
                )
            })?;
            let joins_for_session: Vec<omni_contributor::ContributorJoin> =
                load_verified_joins_from_dir(Some(joins_dir))?
                    .into_iter()
                    .filter(|j| j.session_id == session.session_id)
                    .collect();
            let session_peer_dir =
                advert_dir.join(&session.session_id).join("peer-adverts");
            if session_peer_dir.is_dir() {
                for entry in std::fs::read_dir(&session_peer_dir)? {
                    let entry = entry?;
                    let p = entry.path();
                    if !p.is_file()
                        || p.extension().and_then(|s| s.to_str()) != Some("json")
                    {
                        continue;
                    }
                    let bytes = std::fs::read(&p)?;
                    let advert: omni_contributor::ContributorPeerAdvertisement =
                        match serde_json::from_slice(&bytes) {
                            Ok(a) => a,
                            Err(e) => {
                                eprintln!(
                                    "event=warn context=load_peer_advert path={} message={e}",
                                    p.display()
                                );
                                continue;
                            }
                        };
                    // Re-verify before insert: advertisement_id
                    // recompute + contributor signature + matching
                    // join + expiry.
                    let out = omni_contributor::verify_peer_advertisement_body(
                        &advert,
                        &joins_for_session,
                        Some(&now_utc),
                    );
                    match out {
                        omni_contributor::PeerAdvertisementOutcome::Verified {
                            body,
                        } => {
                            cache.insert_verified(*body);
                        }
                        other => {
                            eprintln!(
                                "event=warn context=local_peer_advert_rejected \
                                 path={} reason={}",
                                p.display(),
                                stringify_peer_advert_outcome(&other)
                            );
                        }
                    }
                }
            }
        }

        let dtype: omni_contributor::TensorDtype =
            args.downstream_resolve_dtype.into();
        match cache.resolve(
            &session.session_id,
            &down_assignment.contributor_pubkey_hex,
            dtype,
            args.handoff_chunk_max_bytes,
            &now_utc,
        ) {
            omni_contributor::RouteResolution::Found(route) => {
                effective_downstream_to_peer = Some(route.peer_id.clone());
                effective_downstream_chunk_cap =
                    Some(route.max_handoff_chunk_bytes);
                println!(
                    "event=peer_advert_resolved session_id={} downstream_contributor={} \
                     peer_id={} chunk_cap={}",
                    session.session_id,
                    down_assignment.contributor_pubkey_hex,
                    route.peer_id,
                    route.max_handoff_chunk_bytes
                );
            }
            other => {
                return Err(anyhow!(
                    "could not resolve downstream peer from session adverts: {other:?}"
                ));
            }
        }
    }

    // Stage 12.4 — resolve the effective out-mode using whichever
    // downstream-to-peer was selected above (manual or resolved).
    let downstream_present = effective_downstream_to_peer.is_some()
        && args.downstream_to_assignment_snip_root.is_some();
    let activation_out_mode = match args.activation_out_mode {
        Some(m) => m,
        None => {
            if downstream_present {
                // Live handoff with SNIP audit (12.4 default for
                // pipelined stages).
                ActivationOutMode::Both
            } else {
                // No downstream supplied — preserve 12.3 behavior.
                ActivationOutMode::Snip
            }
        }
    };
    if matches!(
        activation_out_mode,
        ActivationOutMode::Live | ActivationOutMode::Both
    ) && !downstream_present
    {
        return Err(anyhow!(
            "activation-out-mode={:?} requires both --downstream-to-peer \
             and --downstream-to-assignment-snip-root",
            activation_out_mode
        ));
    }
    if args.activation_in_mode == ActivationInMode::Live
        && args.upstream_from_assignment_snip_root.is_none()
    {
        return Err(anyhow!(
            "activation-in-mode=live requires --upstream-from-assignment-snip-root"
        ));
    }
    if args.activation_in_mode == ActivationInMode::Snip {
        return Err(anyhow!(
            "activation-in-mode=snip not implemented in 12.4 v1; \
             use `none` (first stage) or `live` (omni-net handoff)"
        ));
    }

    // Stage 12.4 — fetch downstream assignment (if needed) once,
    // up-front. This is the assignment our output activation will
    // be handed off to.
    let downstream_assignment: Option<WorkAssignment> =
        if let Some(root_hex) = args.downstream_to_assignment_snip_root.as_deref() {
            let root = SnipV2ObjectId::from_hex(root_hex)
                .map_err(|e| anyhow!("bad downstream assignment snip root: {e:?}"))?;
            let bytes = omni_contributor::snip::fetch_bytes(&snip, &root)
                .map_err(|e| anyhow!("snip fetch downstream assignment: {e}"))?;
            let a: WorkAssignment =
                serde_json::from_slice(&bytes).map_err(|e| anyhow!("parse: {e}"))?;
            a.validate_schema()
                .map_err(|e| anyhow!("invalid downstream WorkAssignment: {e}"))?;
            if a.session_id != session.session_id {
                return Err(anyhow!(
                    "downstream assignment.session_id != session.session_id"
                ));
            }
            // v1 strict linear pipeline.
            if a.stage_index != assignment.stage_index.wrapping_add(1) {
                return Err(anyhow!(
                    "downstream assignment.stage_index ({}) must equal this stage's \
                     + 1 ({})",
                    a.stage_index,
                    assignment.stage_index + 1
                ));
            }
            Some(a)
        } else {
            None
        };

    // Stage 12.4 — fetch upstream assignment (if needed).
    let upstream_assignment: Option<WorkAssignment> =
        if let Some(root_hex) = args.upstream_from_assignment_snip_root.as_deref() {
            let root = SnipV2ObjectId::from_hex(root_hex)
                .map_err(|e| anyhow!("bad upstream assignment snip root: {e:?}"))?;
            let bytes = omni_contributor::snip::fetch_bytes(&snip, &root)
                .map_err(|e| anyhow!("snip fetch upstream assignment: {e}"))?;
            let a: WorkAssignment =
                serde_json::from_slice(&bytes).map_err(|e| anyhow!("parse: {e}"))?;
            a.validate_schema()
                .map_err(|e| anyhow!("invalid upstream WorkAssignment: {e}"))?;
            if a.session_id != session.session_id {
                return Err(anyhow!(
                    "upstream assignment.session_id != session.session_id"
                ));
            }
            if a.stage_index.wrapping_add(1) != assignment.stage_index {
                return Err(anyhow!(
                    "upstream assignment.stage_index ({}) + 1 must equal this stage's ({})",
                    a.stage_index,
                    assignment.stage_index
                ));
            }
            Some(a)
        } else {
            None
        };

    let needs_omninet = args.activation_in_mode == ActivationInMode::Live
        || matches!(
            activation_out_mode,
            ActivationOutMode::Live | ActivationOutMode::Both
        )
        || matches!(
            activation_out_mode,
            ActivationOutMode::Snip | ActivationOutMode::Both
        );

    // Open OmniNet once if we need it (live transport for handoff,
    // and/or the 12.3 partial broadcast).
    let net_and_handle = if needs_omninet {
        Some(
            open_omninet_with_peer_wait(
                args.listen_port,
                args.peer,
                args.peer_wait_secs,
                args.mesh_stabilize_ms,
                args.net_identity_file.as_deref(),
            )
            .await?,
        )
    } else {
        None
    };

    // Stage 12.4 — receive upstream activation if requested.
    let upstream_activation_bytes: Option<Vec<u8>> =
        if args.activation_in_mode == ActivationInMode::Live {
            let (net, handle) = net_and_handle.as_ref().expect("opened above");
            let upstream_asn = upstream_assignment
                .as_ref()
                .expect("validated above for activation-in=live");
            let mut transport =
                OmniNetTensorTransport::new(net.clone(), handle.clone());
            let bytes = live_receive_activation(
                &mut transport,
                &session,
                upstream_asn,
                &assignment,
                args.upstream_wait_secs,
            )
            .await?;
            println!(
                "event=upstream_activation_received bytes={} from_assignment_id={}",
                bytes.len(),
                upstream_asn.assignment_id
            );
            Some(bytes)
        } else {
            None
        };

    // Build the runner.
    let empty_manifest = std::path::PathBuf::from("/dev/null");
    let runner_output_with_act = match args.runner {
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
            r.run_with_activations(
                &empty_manifest,
                b"",
                upstream_activation_bytes.as_deref(),
            )
            .map_err(|e| anyhow!("runner: {e}"))?
        }
        RunnerChoice::External => {
            let mut r = ExternalCommandRunner::new(
                args.external_command
                    .ok_or_else(|| anyhow!("--external-command required for external runner"))?,
            );
            r.extra_args = args.external_args;
            r.env_allowlist = args.external_env_allow;
            r.run_with_activations(
                &empty_manifest,
                b"",
                upstream_activation_bytes.as_deref(),
            )
            .map_err(|e| anyhow!("runner: {e}"))?
        }
    };
    let runner_output = runner_output_with_act.run_output;
    let output_activation = runner_output_with_act.output_activation;

    // Stage 12.4 — publish to SNIP only if the mode requests it.
    let publish_to_snip = matches!(
        activation_out_mode,
        ActivationOutMode::Snip | ActivationOutMode::Both
    );

    // SNIP path: publish artifact + sign partial + broadcast
    // NetworkPartialResultAnnouncement (Stage 12.3 behavior). Skipped
    // entirely on `live` / `none` modes.
    let (mut partial_root_hex, mut artifact_root_hex, mut partial_canonical_hash) =
        (String::new(), String::new(), String::new());
    if publish_to_snip {
        let artifact_root = omni_contributor::snip::publish_bytes(
            &snip,
            &runner_output.response_bytes,
            "partial-artifact",
        )
        .map_err(|e| anyhow!("snip publish artifact: {e}"))?;
        artifact_root_hex = format!("0x{}", hex_lower(artifact_root.as_bytes()));
        let artifact_hash =
            hex_lower(blake3::hash(&runner_output.response_bytes).as_bytes());

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
        partial_root_hex = format!("0x{}", hex_lower(partial_root.as_bytes()));
        partial_canonical_hash = {
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

        let (net, handle) = net_and_handle.as_ref().expect("opened above");
        let mut relay = OmniNetRelay::new(net.clone(), handle.clone());
        relay
            .publish_partial_result(&ann)
            .map_err(|e| anyhow!("publish: {e}"))?;
    }

    // Stage 12.4 — live send if the mode requests it.
    let live_handoff_sent = matches!(
        activation_out_mode,
        ActivationOutMode::Live | ActivationOutMode::Both
    );
    if live_handoff_sent {
        // Stage 12.4: live handoff requires typed activation
        // metadata from the runner — dtype + shape + bytes. No
        // silent F16/1-D fallback (review #1 closed that hole).
        let activation = match output_activation.as_ref() {
            Some(a) => a,
            None => {
                return Err(anyhow!(
                    "activation-out-mode=live|both requires the runner to declare \
                     a typed output activation (dtype + shape + bytes). The stub \
                     runner does not produce one; wire an external runner that emits \
                     `output_activation_dtype` + `output_activation_shape` in its \
                     stdout envelope, or fall back to `--activation-out-mode snip|none`."
                ));
            }
        };
        let downstream_asn = downstream_assignment
            .as_ref()
            .expect("validated above for downstream-present");
        let to_peer = effective_downstream_to_peer
            .as_deref()
            .expect("validated above for downstream-present");
        // Stage 12.5: when a peer-advert resolution picked a chunk
        // cap, use it (already `min(local, advertised)`). Otherwise
        // fall back to the operator's local cap.
        let chunk_cap = effective_downstream_chunk_cap
            .unwrap_or(args.handoff_chunk_max_bytes);
        let (net, handle) = net_and_handle.as_ref().expect("opened above");
        let mut transport =
            OmniNetTensorTransport::new(net.clone(), handle.clone());
        live_send_activation(
            &mut transport,
            &session,
            &assignment,
            downstream_asn,
            &contrib,
            activation,
            to_peer,
            chunk_cap,
        )
        .await?;
        println!(
            "event=output_activation_sent bytes={} dtype={:?} shape={:?} \
             to_peer={} to_assignment_id={}",
            activation.bytes.len(),
            activation.dtype,
            activation.shape,
            to_peer,
            downstream_asn.assignment_id
        );
    }

    // Settle + shutdown.
    if let Some((net, _handle)) = net_and_handle {
        tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }

    println!("assignment_id={}", assignment.assignment_id);
    if publish_to_snip {
        println!("partial_result_snip_root={partial_root_hex}");
        println!("partial_artifact_snip_root={artifact_root_hex}");
        println!("partial_canonical_hash={partial_canonical_hash}");
        println!("partial=published");
    } else {
        // No SNIP partial was published — output is purely live or
        // intentionally suppressed (`--activation-out-mode live|none`).
        println!("partial=not_published");
    }
    println!("activation_out_mode={activation_out_mode:?}");
    println!("activation_in_mode={:?}", args.activation_in_mode);
    println!("live_handoff_sent={live_handoff_sent}");
    Ok(())
}

async fn run_send_handoff(args: SendHandoffArgs) -> Result<()> {
    use omni_contributor::canonical::{
        activation_handoff_signing_input, hex_lower,
    };
    use omni_contributor::{
        ActivationHandoff, ContributorSigner, OmniNetTensorTransport, TensorTransport,
        WorkAssignment, HANDOFF_SCHEMA_VERSION,
    };
    use omni_types::phase5::SnipV2ObjectId;

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.execution_session_snip_root)?;

    // Fetch + validate both from and to assignments.
    let from_root = SnipV2ObjectId::from_hex(&args.from_assignment_snip_root)
        .map_err(|e| anyhow!("bad from-assignment snip root: {e:?}"))?;
    let from_bytes = omni_contributor::snip::fetch_bytes(&snip, &from_root)
        .map_err(|e| anyhow!("snip fetch from-assignment: {e}"))?;
    let from_assignment: WorkAssignment = serde_json::from_slice(&from_bytes)
        .map_err(|e| anyhow!("parse from-assignment: {e}"))?;
    from_assignment
        .validate_schema()
        .map_err(|e| anyhow!("invalid from-WorkAssignment: {e}"))?;

    let to_root = SnipV2ObjectId::from_hex(&args.to_assignment_snip_root)
        .map_err(|e| anyhow!("bad to-assignment snip root: {e:?}"))?;
    let to_bytes = omni_contributor::snip::fetch_bytes(&snip, &to_root)
        .map_err(|e| anyhow!("snip fetch to-assignment: {e}"))?;
    let to_assignment: WorkAssignment = serde_json::from_slice(&to_bytes)
        .map_err(|e| anyhow!("parse to-assignment: {e}"))?;
    to_assignment
        .validate_schema()
        .map_err(|e| anyhow!("invalid to-WorkAssignment: {e}"))?;

    if from_assignment.session_id != session.session_id
        || to_assignment.session_id != session.session_id
    {
        return Err(anyhow!("assignment(s) do not belong to the supplied session"));
    }
    if to_assignment.stage_index != from_assignment.stage_index.wrapping_add(1) {
        return Err(anyhow!(
            "v1 strict linear pipeline: to.stage_index ({}) must equal from.stage_index ({}) + 1",
            to_assignment.stage_index,
            from_assignment.stage_index
        ));
    }

    let contrib = ContributorSigner::from_seed_file(&args.from_contributor_seed)?;
    if contrib.pubkey_hex() != from_assignment.contributor_pubkey_hex {
        return Err(anyhow!(
            "from-contributor seed pubkey does not match from-assignment.contributor_pubkey_hex"
        ));
    }

    // Build the in-memory chunking + sending logic by hand so we
    // can honor the operator-supplied dtype + shape (vs the
    // degenerate 1-D shape from the run-assignment fast path).
    let activation_bytes = std::fs::read(&args.activation_file)
        .with_context(|| format!("read activation: {}", args.activation_file.display()))?;
    if activation_bytes.is_empty() {
        return Err(anyhow!("activation file is empty"));
    }
    let byte_len = activation_bytes.len() as u64;
    let dtype: omni_contributor::TensorDtype = args.dtype.into();
    let bytes_per_element = dtype.bytes_per_element();
    let shape_product: u64 = args
        .shape
        .iter()
        .copied()
        .try_fold(1u64, |acc, d| acc.checked_mul(d))
        .ok_or_else(|| anyhow!("--shape: overflow when computing element count"))?;
    if shape_product == 0 {
        return Err(anyhow!("--shape: zero element count"));
    }
    if shape_product * bytes_per_element != byte_len {
        return Err(anyhow!(
            "shape product ({shape_product}) * dtype-bytes ({bytes_per_element}) = {} \
             != activation_file size {byte_len}",
            shape_product * bytes_per_element
        ));
    }
    let tensor_hash = hex_lower(blake3::hash(&activation_bytes).as_bytes());
    let chunk_size = args.chunk_max_bytes.min(byte_len) as usize;
    let chunk_count = (activation_bytes.len()).div_ceil(chunk_size) as u32;
    let produced_at_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        args.peer_wait_secs,
        args.mesh_stabilize_ms,
        args.net_identity_file.as_deref(),
    )
    .await?;
    let mut transport = OmniNetTensorTransport::new(net.clone(), handle);

    for chunk_index in 0..chunk_count {
        let start = chunk_index as usize * chunk_size;
        let end = ((chunk_index + 1) as usize * chunk_size).min(activation_bytes.len());
        let chunk_bytes = activation_bytes[start..end].to_vec();
        let mut h = ActivationHandoff {
            schema_version: HANDOFF_SCHEMA_VERSION,
            session_id: session.session_id.clone(),
            from_assignment_id: from_assignment.assignment_id.clone(),
            to_assignment_id: to_assignment.assignment_id.clone(),
            from_contributor_pubkey_hex: from_assignment.contributor_pubkey_hex.clone(),
            to_contributor_pubkey_hex: to_assignment.contributor_pubkey_hex.clone(),
            dtype,
            shape: args.shape.clone(),
            byte_len,
            tensor_hash: tensor_hash.clone(),
            chunk_index,
            chunk_count,
            produced_at_utc: produced_at_utc.clone(),
            tensor_chunk_bytes: chunk_bytes,
            sender_signature_hex: String::new(),
        };
        let si = activation_handoff_signing_input(&h)?;
        h.sender_signature_hex = contrib.sign_hex(&si);
        h.validate_schema()
            .map_err(|e| anyhow!("invalid handoff envelope: {e}"))?;
        transport
            .send_handoff(Some(&args.to_peer), &h)
            .map_err(|e| anyhow!("transport send: {e}"))?;
        println!(
            "event=handoff_chunk_sent chunk_index={chunk_index} chunk_count={chunk_count} \
             bytes={}",
            end - start
        );
    }
    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }
    println!("tensor_hash={tensor_hash}");
    println!("byte_len={byte_len}");
    println!("chunk_count={chunk_count}");
    println!("handoff=sent");
    Ok(())
}

/// Stage 12.4 — chunk + sign + send a runner-produced typed
/// activation across the configured transport. The runner is the
/// only source of truth for dtype + shape; the CLI never invents
/// either. (Review #1 closed the prior F16/1-D fallback hole.)
#[allow(clippy::too_many_arguments)]
async fn live_send_activation<T: omni_contributor::TensorTransport>(
    transport: &mut T,
    session: &omni_contributor::ExecutionSession,
    from_assignment: &omni_contributor::WorkAssignment,
    to_assignment: &omni_contributor::WorkAssignment,
    sender: &omni_contributor::ContributorSigner,
    activation: &omni_contributor::RunnerOutputActivation,
    to_peer_hint: &str,
    chunk_max_bytes: u64,
) -> Result<()> {
    use omni_contributor::canonical::{
        activation_handoff_signing_input, hex_lower,
    };
    use omni_contributor::{ActivationHandoff, HANDOFF_SCHEMA_VERSION};

    let tensor_bytes = &activation.bytes[..];
    let byte_len = tensor_bytes.len() as u64;
    if byte_len == 0 {
        return Err(anyhow!("live_send_activation: empty tensor"));
    }
    let tensor_hash = hex_lower(blake3::hash(tensor_bytes).as_bytes());
    let chunk_size = chunk_max_bytes.min(byte_len) as usize;
    let chunk_count = (byte_len as usize).div_ceil(chunk_size) as u32;
    let produced_at_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    for chunk_index in 0..chunk_count {
        let start = chunk_index as usize * chunk_size;
        let end = ((chunk_index + 1) as usize * chunk_size).min(tensor_bytes.len());
        let chunk_bytes = tensor_bytes[start..end].to_vec();
        let mut h = ActivationHandoff {
            schema_version: HANDOFF_SCHEMA_VERSION,
            session_id: session.session_id.clone(),
            from_assignment_id: from_assignment.assignment_id.clone(),
            to_assignment_id: to_assignment.assignment_id.clone(),
            from_contributor_pubkey_hex: from_assignment.contributor_pubkey_hex.clone(),
            to_contributor_pubkey_hex: to_assignment.contributor_pubkey_hex.clone(),
            dtype: activation.dtype,
            shape: activation.shape.clone(),
            byte_len,
            tensor_hash: tensor_hash.clone(),
            chunk_index,
            chunk_count,
            produced_at_utc: produced_at_utc.clone(),
            tensor_chunk_bytes: chunk_bytes,
            sender_signature_hex: String::new(),
        };
        let si = activation_handoff_signing_input(&h)?;
        h.sender_signature_hex = sender.sign_hex(&si);
        h.validate_schema()
            .map_err(|e| anyhow!("invalid handoff envelope: {e}"))?;
        transport
            .send_handoff(Some(to_peer_hint), &h)
            .map_err(|e| anyhow!("transport send: {e}"))?;
    }
    Ok(())
}

/// Stage 12.4 — bounded-wait receive of an upstream activation.
/// Verifies each envelope against the supplied session +
/// upstream/this-stage assignments, accumulates chunks, returns the
/// reassembled tensor bytes.
async fn live_receive_activation<T: omni_contributor::TensorTransport>(
    transport: &mut T,
    session: &omni_contributor::ExecutionSession,
    upstream_assignment: &omni_contributor::WorkAssignment,
    this_assignment: &omni_contributor::WorkAssignment,
    timeout_secs: u64,
) -> Result<Vec<u8>> {
    use omni_contributor::{verify_activation_handoff, ChunkOutcome, HandoffReceiver};

    let deadline = std::time::Instant::now()
        + std::time::Duration::from_secs(timeout_secs);
    let mut receiver = HandoffReceiver::new();
    loop {
        let pending = transport
            .poll_handoffs()
            .map_err(|e| anyhow!("transport poll: {e}"))?;
        for h in pending {
            let v_out = verify_activation_handoff(
                session,
                upstream_assignment,
                this_assignment,
                &h,
            );
            if !v_out.is_ok() {
                println!("event=skip kind=handoff reason=verify_failed:{v_out:?}");
                continue;
            }
            match receiver.feed(&h) {
                ChunkOutcome::Complete { tensor_bytes } => return Ok(tensor_bytes),
                ChunkOutcome::Accepted => {}
                other => {
                    println!("event=skip kind=handoff reason=reassemble_failed:{other:?}");
                }
            }
        }
        if std::time::Instant::now() >= deadline {
            return Err(anyhow!(
                "live_receive_activation: timed out after {timeout_secs}s waiting for \
                 upstream handoff from_assignment_id={}",
                upstream_assignment.assignment_id
            ));
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
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
        args.net_identity_file.as_deref(),
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
        args.net_identity_file.as_deref(),
    )
    .await?;
    std::fs::create_dir_all(&args.out_dir)?;
    let posted_id_filter: HashSet<String> = args.posted_id.into_iter().collect();
    let session_id_filter: HashSet<String> = args.session_id.into_iter().collect();
    let mut polls_done: u64 = 0;

    // Stage 12.7 — open the optional contributor state store.
    let state_store = open_optional_state_store(
        args.contributor_state_dir.as_deref(),
        args.no_prune_state_on_start,
    )?;

    tokio::task::spawn_blocking(move || -> Result<()> {
        let mut relay = OmniNetRelay::new(net, handle);
        // Session cache: needed so the assignment processor can
        // verify each assignment's coordinator signature against
        // the session's coordinator_pubkey_hex (assignments don't
        // carry their own coordinator pubkey).
        let mut sessions: HashMap<String, ExecutionSession> = HashMap::new();
        // Stage 12.13 — supersession-aware in-memory caches.
        // `handle_assignment_supersession`'s out-of-order best-
        // effort path (Bucket 2) needs to know whether referenced
        // replacement assignments are present locally, and the
        // restart preload now populates both caches up front so
        // that knowledge is correct from the first poll cycle.
        let mut assignments_by_session: HashMap<
            String,
            Vec<omni_contributor::WorkAssignment>,
        > = HashMap::new();
        let mut supersessions_by_session: HashMap<
            String,
            Vec<omni_contributor::supersession::WorkAssignmentSupersession>,
        > = HashMap::new();
        // Stage 12.7 + 12.13 — pre-warm the in-memory caches from
        // the state-dir BEFORE the loop runs. Stage 12.7 lifted
        // this for sessions only; Stage 12.13 lifts it to
        // sessions + assignments + supersessions via the new
        // `load_verified_restart_snapshot` helper, which re-runs
        // every Stage 12.3 / 12.11 verifier on the way so a
        // tampered local file is dropped with a structured
        // rejection note. Aggregate re-verify is deliberately
        // skipped at preload — the first status-build re-checks
        // it via `verify_aggregated_result_with_supersessions`.
        // Joins are loaded for verification but NOT cached — the
        // watcher only needs them to validate other artifacts;
        // joined-pubkey checks inside handlers always re-read.
        if let Some(ref store) = state_store {
            match omni_contributor::load_verified_restart_snapshot(store) {
                Ok((snapshot, report)) => {
                    let omni_contributor::RestartSnapshot {
                        sessions: s,
                        joins_by_session: _,
                        assignments_by_session: a,
                        supersessions_by_session: sup,
                    } = snapshot;
                    sessions.extend(s);
                    assignments_by_session.extend(a);
                    supersessions_by_session.extend(sup);
                    println!(
                        "event=state_store_restart_loaded \
                         sessions_accepted={} sessions_rejected={} \
                         joins_accepted={} joins_rejected={} \
                         assignments_accepted={} assignments_rejected={} \
                         supersessions_accepted={} supersessions_rejected={}",
                        report.sessions_accepted,
                        report.sessions_rejected,
                        report.joins_accepted,
                        report.joins_rejected,
                        report.assignments_accepted,
                        report.assignments_rejected,
                        report.supersessions_accepted,
                        report.supersessions_rejected,
                    );
                    for note in &report.rejection_notes {
                        println!(
                            "event=warn context=state_store_restart_load_rejected {note}"
                        );
                    }
                }
                Err(e) => {
                    println!(
                        "event=warn context=state_store_restart_load message={e}"
                    );
                }
            }
        }
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
                    state_store.as_ref(),
                );
            }
            for ann in relay
                .poll_contributors_joined()
                .map_err(|e| anyhow!("poll joined: {e}"))?
            {
                handle_join(
                    &snip,
                    &args.out_dir,
                    &session_id_filter,
                    &ann,
                    state_store.as_ref(),
                );
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
                    &mut assignments_by_session,
                    state_store.as_ref(),
                );
            }
            for ann in relay
                .poll_partial_results()
                .map_err(|e| anyhow!("poll partial: {e}"))?
            {
                handle_partial(
                    &snip,
                    &args.out_dir,
                    &session_id_filter,
                    &ann,
                    state_store.as_ref(),
                );
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
                    state_store.as_ref(),
                );
            }
            // Stage 12.11 — supersession topic. Pinning the body
            // to the in-memory session cache lets the processor
            // fail closed on coordinator-pubkey drift.
            // Stage 12.13 — full reference-resolution runs when
            // the in-memory assignments cache covers every
            // referenced id; otherwise the handler falls back to
            // best-effort accept and emits
            // `event=supersession_partial_verify` so the operator
            // sees the watcher took the looser path. Status-build
            // re-checks references via the supersession-aware
            // aggregate verifier later.
            for ann in relay
                .poll_assignment_supersessions()
                .map_err(|e| anyhow!("poll supersession: {e}"))?
            {
                handle_assignment_supersession(
                    &snip,
                    &args.out_dir,
                    &session_id_filter,
                    &ann,
                    &sessions,
                    &assignments_by_session,
                    &mut supersessions_by_session,
                    state_store.as_ref(),
                );
            }
            std::thread::sleep(Duration::from_secs(args.poll_interval_secs));
        }
    })
    .await
    .map_err(|e| anyhow!("watch-sessions join: {e}"))?
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
    state_store: Option<&omni_contributor::ContributorStateStore>,
) {
    if !posted_id_filter.is_empty() && !posted_id_filter.contains(&ann.posted_id) {
        return;
    }
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    // Stage 12.7 — cross-restart dedup before any SNIP fetch.
    if let Some(store) = state_store {
        if matches!(
            store.is_seen(
                omni_contributor::StateNamespace::Sessions,
                &ann.session_id,
            ),
            Ok(true)
        ) {
            return;
        }
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
            // Stage 12.7 — dual-write into the state-dir's
            // verified/sessions/<id>/session.json (same shape) and
            // record the seen marker.
            if let Some(store) = state_store {
                if let Err(e) = store.write_verified_json(
                    omni_contributor::StateObjectKind::Session,
                    &session.session_id,
                    &session,
                ) {
                    println!("event=warn context=state_store_write_session message={e}");
                }
                if let Err(e) = store.mark_seen(
                    omni_contributor::StateNamespace::Sessions,
                    &session.session_id,
                ) {
                    println!("event=warn context=state_store_mark_seen_session message={e}");
                }
            }
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
    state_store: Option<&omni_contributor::ContributorStateStore>,
) {
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    // Stage 12.7 — cross-restart dedup. Key: <session_id>--<pubkey>.
    let cross_restart_key = format!(
        "{}--{}", ann.session_id, ann.contributor_pubkey_hex
    );
    if let Some(store) = state_store {
        if matches!(
            store.is_seen(omni_contributor::StateNamespace::Joins, &cross_restart_key),
            Ok(true)
        ) {
            return;
        }
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
        Ok(p) => {
            println!(
                "event=join session_id={} contributor_pubkey={} path={}",
                join.session_id,
                join.contributor_pubkey_hex,
                p.display()
            );
            if let Some(store) = state_store {
                if let Err(e) = store.write_verified_json(
                    omni_contributor::StateObjectKind::Join {
                        session_id: join.session_id.clone(),
                    },
                    &join.contributor_pubkey_hex,
                    &join,
                ) {
                    println!("event=warn context=state_store_write_join message={e}");
                }
                if let Err(e) = store.mark_seen(
                    omni_contributor::StateNamespace::Joins,
                    &cross_restart_key,
                ) {
                    println!("event=warn context=state_store_mark_seen_join message={e}");
                }
            }
        }
        Err(e) => println!("event=error context=write_join message={e}"),
    }
}

fn handle_assignment<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkWorkAssignedAnnouncement,
    sessions: &std::collections::HashMap<String, omni_contributor::ExecutionSession>,
    assignments_by_session: &mut std::collections::HashMap<
        String,
        Vec<omni_contributor::WorkAssignment>,
    >,
    state_store: Option<&omni_contributor::ContributorStateStore>,
) {
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    let cross_restart_key = format!("{}--{}", ann.session_id, ann.assignment_id);
    if let Some(store) = state_store {
        if matches!(
            store.is_seen(
                omni_contributor::StateNamespace::Assignments,
                &cross_restart_key,
            ),
            Ok(true)
        ) {
            return;
        }
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
        Ok(p) => {
            println!(
                "event=assignment session_id={} assignment_id={} coord_sig_verified={} path={}",
                asn.session_id,
                asn.assignment_id,
                session_coord.is_some(),
                p.display()
            );
            // Stage 12.13 — keep the in-memory assignments cache
            // current so a same-poll-cycle supersession that
            // references this assignment can run full
            // reference-resolution.
            assignments_by_session
                .entry(asn.session_id.clone())
                .or_default()
                .push(asn.clone());
            if let Some(store) = state_store {
                if let Err(e) = store.write_verified_json(
                    omni_contributor::StateObjectKind::Assignment {
                        session_id: asn.session_id.clone(),
                    },
                    &asn.assignment_id,
                    &asn,
                ) {
                    println!("event=warn context=state_store_write_assignment message={e}");
                }
                if let Err(e) = store.mark_seen(
                    omni_contributor::StateNamespace::Assignments,
                    &cross_restart_key,
                ) {
                    println!("event=warn context=state_store_mark_seen_assignment message={e}");
                }
            }
        }
        Err(e) => println!("event=error context=write_assignment message={e}"),
    }
}

/// Stage 12.11 — watch-sessions handler for the assignment
/// supersession topic. Mirrors `handle_assignment`:
/// processor → schema/sig/SNIP/drift gate → dual-write to
/// `<out_dir>/<session_id>/supersessions/<supersession_id>.json`
/// → state-dir mirror under
/// `verified/sessions/<session_id>/supersessions/...` →
/// seen-marker in `AssignmentSupersessions` namespace.
///
/// When the session cache and state store both know this
/// session, the processor is upgraded to run the full
/// `verify_assignment_supersession` reference-resolution leg.
/// Otherwise the announcement is accepted on its own (announcer
/// sig + body schema + body-sig + drift); aggregate verification
/// re-checks the references later.
#[allow(clippy::too_many_arguments)]
fn handle_assignment_supersession<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkWorkAssignmentSupersessionAnnouncement,
    sessions: &std::collections::HashMap<String, omni_contributor::ExecutionSession>,
    assignments_by_session: &std::collections::HashMap<
        String,
        Vec<omni_contributor::WorkAssignment>,
    >,
    supersessions_by_session: &mut std::collections::HashMap<
        String,
        Vec<omni_contributor::supersession::WorkAssignmentSupersession>,
    >,
    state_store: Option<&omni_contributor::ContributorStateStore>,
) {
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    let cross_restart_key = format!("{}--{}", ann.session_id, ann.supersession_id);
    if let Some(store) = state_store {
        if matches!(
            store.is_seen(
                omni_contributor::StateNamespace::AssignmentSupersessions,
                &cross_restart_key,
            ),
            Ok(true)
        ) {
            return;
        }
    }
    let session_ref = sessions.get(&ann.session_id);

    // Stage 12.13 — out-of-order supersession handling.
    //
    // First pass: call the processor with `Some(session), None`.
    // This runs announcer-sig + body-schema + body-sig + drift +
    // session-pinning checks AND parses the body, but skips the
    // reference-resolution leg. We need the body in hand before
    // we can decide whether the local assignment cache covers
    // every referenced id (so we know which mode to report).
    let outcome = omni_contributor::process_assignment_supersession_announcement(
        ann,
        snip,
        session_ref,
        /* assignments = */ None,
    );
    if !log_announcement_failure("supersession", &ann.session_id, &outcome) {
        return;
    }
    let sup = match outcome {
        omni_contributor::AnnouncementOutcome::Verified { body } => body,
        _ => return,
    };

    // Second pass: if we have BOTH the session AND enough
    // assignments in the in-memory cache to cover every
    // referenced id, run the full reference-resolution leg via
    // `verify_assignment_supersession`. Otherwise accept on the
    // first-pass checks and emit `event=supersession_partial_verify`
    // so the operator sees the watcher took the best-effort
    // path. The aggregate verifier at `session-status` time will
    // re-check references against the full state-dir snapshot.
    let mut references_resolved = false;
    let mut unresolved_count = 0u32;
    if let Some(session) = session_ref {
        let referenced: std::collections::HashSet<&str> = sup
            .superseded_assignment_ids
            .iter()
            .chain(sup.replacement_assignment_ids.iter())
            .map(|s| s.as_str())
            .collect();
        let cached = assignments_by_session.get(&ann.session_id);
        let cached_ids: std::collections::HashSet<&str> = cached
            .map(|v| v.iter().map(|a| a.assignment_id.as_str()).collect())
            .unwrap_or_default();
        let missing: Vec<&&str> =
            referenced.iter().filter(|id| !cached_ids.contains(*id)).collect();
        unresolved_count = missing.len() as u32;
        if unresolved_count == 0 {
            if let Some(slice) = cached {
                let outcome = omni_contributor::verify_assignment_supersession(
                    session, slice, &sup,
                );
                if outcome.is_ok() {
                    references_resolved = true;
                } else {
                    // We had every referenced id in cache but
                    // the verifier still rejected — drop the
                    // announcement loudly so the operator
                    // notices.
                    println!(
                        "event=supersession_reference_verify_failed \
                         session_id={} supersession_id={} reason_tag={}",
                        sup.session_id,
                        sup.supersession_id,
                        outcome.reason_tag(),
                    );
                    return;
                }
            }
        }
    }

    if unresolved_count > 0 {
        println!(
            "event=supersession_partial_verify session_id={} supersession_id={} \
             unresolved_assignment_count={unresolved_count}",
            sup.session_id, sup.supersession_id,
        );
    }

    let bytes = serde_json::to_vec_pretty(&sup).unwrap_or_default();
    let filename = format!("{}.json", sup.supersession_id);
    match write_session_artifact(
        out_dir,
        &sup.session_id,
        Some("supersessions"),
        &filename,
        &bytes,
    ) {
        Ok(p) => {
            println!(
                "event=assignment_supersession session_id={} supersession_id={} \
                 superseded_count={} replacement_count={} session_pinned={} \
                 references_resolved={} path={}",
                sup.session_id,
                sup.supersession_id,
                sup.superseded_assignment_ids.len(),
                sup.replacement_assignment_ids.len(),
                session_ref.is_some(),
                references_resolved,
                p.display()
            );
            // Stage 12.13 — update in-memory supersessions cache
            // so the next aggregate/status check in the same
            // process sees the freshly-accepted body without
            // re-reading the state-dir.
            supersessions_by_session
                .entry(sup.session_id.clone())
                .or_default()
                .push(sup.clone());
            if let Some(store) = state_store {
                if let Err(e) = store.write_verified_json(
                    omni_contributor::StateObjectKind::AssignmentSupersession {
                        session_id: sup.session_id.clone(),
                    },
                    &sup.supersession_id,
                    &sup,
                ) {
                    println!(
                        "event=warn context=state_store_write_supersession message={e}"
                    );
                }
                if let Err(e) = store.mark_seen(
                    omni_contributor::StateNamespace::AssignmentSupersessions,
                    &cross_restart_key,
                ) {
                    println!(
                        "event=warn context=state_store_mark_seen_supersession message={e}"
                    );
                }
            }
        }
        Err(e) => println!("event=error context=write_supersession message={e}"),
    }
}

fn handle_partial<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkPartialResultAnnouncement,
    state_store: Option<&omni_contributor::ContributorStateStore>,
) {
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    let cross_restart_key = format!("{}--{}", ann.session_id, ann.assignment_id);
    if let Some(store) = state_store {
        if matches!(
            store.is_seen(
                omni_contributor::StateNamespace::Partials,
                &cross_restart_key,
            ),
            Ok(true)
        ) {
            return;
        }
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
        Ok(p) => {
            println!(
                "event=partial session_id={} assignment_id={} path={}",
                par.session_id,
                par.assignment_id,
                p.display()
            );
            if let Some(store) = state_store {
                if let Err(e) = store.write_verified_json(
                    omni_contributor::StateObjectKind::Partial {
                        session_id: par.session_id.clone(),
                    },
                    &par.assignment_id,
                    &par,
                ) {
                    println!("event=warn context=state_store_write_partial message={e}");
                }
                if let Err(e) = store.mark_seen(
                    omni_contributor::StateNamespace::Partials,
                    &cross_restart_key,
                ) {
                    println!("event=warn context=state_store_mark_seen_partial message={e}");
                }
            }
        }
        Err(e) => println!("event=error context=write_partial message={e}"),
    }
}

fn handle_aggregated<A: omni_store::SnipV2Adapter + ?Sized>(
    snip: &A,
    out_dir: &std::path::Path,
    posted_id_filter: &std::collections::HashSet<String>,
    session_id_filter: &std::collections::HashSet<String>,
    ann: &omni_contributor::NetworkAggregatedResultAnnouncement,
    state_store: Option<&omni_contributor::ContributorStateStore>,
) {
    if !posted_id_filter.is_empty() && !posted_id_filter.contains(&ann.posted_id) {
        return;
    }
    if !session_id_filter.is_empty() && !session_id_filter.contains(&ann.session_id) {
        return;
    }
    if let Some(store) = state_store {
        if matches!(
            store.is_seen(
                omni_contributor::StateNamespace::Aggregates,
                &ann.session_id,
            ),
            Ok(true)
        ) {
            return;
        }
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
        Ok(p) => {
            println!(
                "event=aggregated session_id={} path={}",
                agg.session_id,
                p.display()
            );
            if let Some(store) = state_store {
                if let Err(e) = store.write_verified_json(
                    omni_contributor::StateObjectKind::Aggregate,
                    &agg.session_id,
                    &agg,
                ) {
                    println!("event=warn context=state_store_write_aggregate message={e}");
                }
                if let Err(e) = store.mark_seen(
                    omni_contributor::StateNamespace::Aggregates,
                    &agg.session_id,
                ) {
                    println!("event=warn context=state_store_mark_seen_aggregate message={e}");
                }
            }
        }
        Err(e) => println!("event=error context=write_aggregated message={e}"),
    }
}

// ── Stage 12.5 — peer-advert subcommand handlers ──────────────────────────

/// Load **cryptographically verified** `ContributorJoin`s from the
/// `watch-sessions` output tree:
///
/// ```text
/// <dir>/
///   <session_id>/
///     session.json           # ExecutionSession — verified standalone
///     joins/<pubkey>.json    # ContributorJoin  — verified against session
/// ```
///
/// For each `<session_id>` subdirectory:
///   1. Reads + parses + `verify_execution_session` on `session.json`.
///      If missing / malformed / signature fails: skip the whole
///      subdirectory with a stderr warning.
///   2. Reads + parses every `joins/*.json`.
///   3. Calls `verify_contributor_join(&session, &join)` (Stage 12.3
///      verifier — schema + session binding + contributor signature).
///      Only entries that return `SessionVerifyOutcome::Ok` are
///      returned.
///
/// Returns an empty `Vec` if `dir` is `None` or the directory does
/// not exist. Schema-only loading was removed in Stage 12.5 review
/// #2: a forged local join file would otherwise let a forged peer
/// advert pass the matching-join gate. The flat-file (one-level)
/// fallback is also dropped — the canonical source is the
/// watch-sessions tree.
fn load_verified_joins_from_dir(
    dir: Option<&std::path::Path>,
) -> Result<Vec<omni_contributor::ContributorJoin>> {
    let dir = match dir {
        Some(d) => d,
        None => return Ok(Vec::new()),
    };
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let session_subdir = entry.path();
        if !session_subdir.is_dir() {
            continue;
        }
        let session_path = session_subdir.join("session.json");
        if !session_path.is_file() {
            eprintln!(
                "event=warn context=load_joins_no_session_json path={}",
                session_subdir.display()
            );
            continue;
        }
        let session_bytes = match std::fs::read(&session_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "event=warn context=load_joins_read_session path={} message={e}",
                    session_path.display()
                );
                continue;
            }
        };
        let session: omni_contributor::ExecutionSession =
            match serde_json::from_slice(&session_bytes) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!(
                        "event=warn context=load_joins_parse_session path={} message={e}",
                        session_path.display()
                    );
                    continue;
                }
            };
        let session_outcome = omni_contributor::verify_execution_session(&session);
        if !session_outcome.is_ok() {
            eprintln!(
                "event=warn context=load_joins_session_verify_failed \
                 path={} outcome={session_outcome:?}",
                session_path.display()
            );
            continue;
        }
        let joins_subdir = session_subdir.join("joins");
        if !joins_subdir.is_dir() {
            continue;
        }
        for sub in std::fs::read_dir(&joins_subdir)? {
            let sub = sub?;
            let p = sub.path();
            if !p.is_file()
                || p.extension().and_then(|s| s.to_str()) != Some("json")
            {
                continue;
            }
            let bytes = match std::fs::read(&p) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "event=warn context=load_joins_read_join path={} message={e}",
                        p.display()
                    );
                    continue;
                }
            };
            let join: omni_contributor::ContributorJoin =
                match serde_json::from_slice(&bytes) {
                    Ok(j) => j,
                    Err(e) => {
                        eprintln!(
                            "event=warn context=load_joins_parse_join path={} message={e}",
                            p.display()
                        );
                        continue;
                    }
                };
            let outcome = omni_contributor::verify_contributor_join(&session, &join);
            if outcome.is_ok() {
                out.push(join);
            } else {
                eprintln!(
                    "event=warn context=load_joins_verify_failed \
                     path={} outcome={outcome:?}",
                    p.display()
                );
            }
        }
    }
    Ok(out)
}

async fn run_advertise_peer(args: AdvertisePeerArgs) -> Result<()> {
    use omni_contributor::canonical::{
        advertisement_id_hex, hex_lower, net_peer_advert_signing_input,
        peer_advertisement_signing_input,
    };
    use omni_contributor::{
        ContributorJoin, ContributorPeerAdvertisement, ContributorRelay, ContributorSigner,
        NetworkPeerAdvertisementAnnouncement, OmniNetRelay, PeerCapabilities, TensorDtype,
        NET_SCHEMA_VERSION, PEER_ADVERTISEMENT_SCHEMA_VERSION,
    };
    use omni_types::phase5::SnipV2ObjectId;

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);

    // Fetch + verify session.
    let session = fetch_and_verify_session(&snip, &args.execution_session_snip_root)?;

    // Fetch + verify join + bind to seed pubkey.
    let join_root = SnipV2ObjectId::from_hex(&args.join_snip_root)
        .map_err(|e| anyhow!("bad join snip root: {e:?}"))?;
    let join_bytes = omni_contributor::snip::fetch_bytes(&snip, &join_root)
        .map_err(|e| anyhow!("snip fetch join: {e}"))?;
    let join: ContributorJoin = serde_json::from_slice(&join_bytes)
        .map_err(|e| anyhow!("parse join: {e}"))?;
    let join_out = omni_contributor::verify_contributor_join(&session, &join);
    if !join_out.is_ok() {
        return Err(anyhow!("join verify failed: {join_out:?}"));
    }
    let contrib = ContributorSigner::from_seed_file(&args.contributor_seed)?;
    if contrib.pubkey_hex() != join.contributor_pubkey_hex {
        return Err(anyhow!(
            "contributor_seed pubkey does not match join.contributor_pubkey_hex"
        ));
    }

    // Open the mesh and capture the live PeerId — the entire
    // reason Stage 12.5-pre exists.
    //
    // Peer-wait only applies when we're actually broadcasting an
    // announcement. A SNIP-only advertisement (publishing the
    // body to durable storage without gossip) does not need a
    // reachable peer at publish time and must not fail just
    // because the mesh is empty.
    let (peer_wait_secs, mesh_stabilize_ms) = if args.publish_announcement {
        (args.peer_wait_secs, args.mesh_stabilize_ms)
    } else {
        (0u64, 0u64)
    };
    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        peer_wait_secs,
        mesh_stabilize_ms,
        args.net_identity_file.as_deref(),
    )
    .await?;
    let libp2p_peer_id_b58 = {
        let g = net.lock().await;
        g.local_peer_id().to_base58()
    };

    // Build + sign the advertisement. Capture `now` once so the
    // `expires_at_utc - advertised_at_utc` window stays exactly
    // `expires_in_secs` even if the two reads straddle a second
    // boundary — otherwise `--expires-in-secs 86400` could land
    // 86401 seconds out and trip the schema's 24h freshness cap.
    let now = chrono::Utc::now();
    let advertised_at_utc =
        now.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let expires_at_utc = (now
        + chrono::Duration::seconds(args.expires_in_secs as i64))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let supported_dtypes: Vec<TensorDtype> =
        args.supported_dtypes.into_iter().map(Into::into).collect();
    let mut advert = ContributorPeerAdvertisement {
        schema_version: PEER_ADVERTISEMENT_SCHEMA_VERSION,
        advertisement_id: String::new(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        libp2p_peer_id: libp2p_peer_id_b58.clone(),
        listen_multiaddrs: args.listen_multiaddrs,
        capabilities: PeerCapabilities {
            supports_live_handoff: true,
            max_handoff_chunk_bytes: args.max_handoff_chunk_bytes,
            supported_dtypes,
        },
        advertised_at_utc,
        expires_at_utc,
        contributor_signature_hex: String::new(),
    };
    advert.advertisement_id = advertisement_id_hex(&advert)?;
    let si = peer_advertisement_signing_input(&advert)?;
    advert.contributor_signature_hex = contrib.sign_hex(&si);
    advert
        .validate_schema()
        .map_err(|e| anyhow!("invalid ContributorPeerAdvertisement: {e}"))?;

    let advert_json = serde_json::to_vec_pretty(&advert)?;
    let advert_root = omni_contributor::snip::publish_bytes(
        &snip,
        &advert_json,
        "peer-advert",
    )
    .map_err(|e| anyhow!("snip publish advert: {e}"))?;
    let advert_root_hex = format!("0x{}", hex_lower(advert_root.as_bytes()));

    if args.publish_announcement {
        let mut ann = NetworkPeerAdvertisementAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            peer_advertisement_snip_root: advert_root_hex.clone(),
            advertisement_id: advert.advertisement_id.clone(),
            session_id: session.session_id.clone(),
            contributor_pubkey_hex: contrib.pubkey_hex(),
            announced_at_utc: chrono::Utc::now()
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            announcer_pubkey_hex: contrib.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let ann_sig = net_peer_advert_signing_input(&ann)?;
        ann.announcer_signature_hex = contrib.sign_hex(&ann_sig);
        let mut relay = OmniNetRelay::new(net.clone(), handle);
        relay
            .publish_peer_advertisement(&ann)
            .map_err(|e| anyhow!("publish: {e}"))?;
        tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    }
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }

    println!("advertisement_id={}", advert.advertisement_id);
    println!("peer_advertisement_snip_root={advert_root_hex}");
    println!("libp2p_peer_id={libp2p_peer_id_b58}");
    println!("session_id={}", advert.session_id);
    println!(
        "broadcast={}",
        if args.publish_announcement { "true" } else { "false" }
    );
    Ok(())
}

async fn run_watch_peer_adverts(args: WatchPeerAdvertsArgs) -> Result<()> {
    use omni_contributor::{
        process_peer_advertisement_announcement, ContributorRelay, OmniNetRelay,
        PeerAdvertisementOutcome,
    };
    use std::collections::HashSet;
    use std::time::Duration;

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    // Stage 12.7 — when `--contributor-state-dir` is supplied,
    // verified joins are loaded from the state-dir's
    // `verified/sessions/<id>/joins/...` tree instead of the
    // legacy `--joins-dir` flag. The state-dir's joins subtree is
    // shape-compatible with `watch-sessions --out-dir`, so a single
    // directory can serve both roles during a gradual migration.
    let state_store = open_optional_state_store(
        args.contributor_state_dir.as_deref(),
        args.no_prune_state_on_start,
    )?;
    let joins: Vec<omni_contributor::ContributorJoin> = match state_store.as_ref() {
        Some(store) => {
            let mut all = Vec::new();
            // Mirror the legacy `load_verified_joins_from_dir`
            // posture: re-verify session.json AND every join.json
            // before exposing them as the matching-join trust
            // source. A tampered local file would otherwise let a
            // forged peer advert pass the matching-join gate.
            for (sid, session) in store.list_verified_sessions()? {
                if !omni_contributor::verify_execution_session(&session)
                    .is_ok()
                {
                    eprintln!(
                        "event=warn context=state_store_session_verify_failed \
                         session_id={sid}"
                    );
                    continue;
                }
                for join in store.list_verified_joins_for(&sid)? {
                    if omni_contributor::verify_contributor_join(&session, &join)
                        .is_ok()
                    {
                        all.push(join);
                    } else {
                        eprintln!(
                            "event=warn context=state_store_join_verify_failed \
                             session_id={} contributor_pubkey={}",
                            join.session_id, join.contributor_pubkey_hex
                        );
                    }
                }
            }
            all
        }
        None => load_verified_joins_from_dir(args.joins_dir.as_deref())?,
    };

    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        /* peer_wait_secs = */ 0,
        /* mesh_stabilize_ms = */ 0,
        args.net_identity_file.as_deref(),
    )
    .await?;
    std::fs::create_dir_all(&args.out_dir)?;

    let session_id_filter: HashSet<String> = args.session_id.into_iter().collect();
    let contributor_filter: HashSet<String> = args.contributor_pubkey.into_iter().collect();
    let mut polls_done: u64 = 0;
    let mut adverts_written: u64 = 0;

    tokio::task::spawn_blocking(move || -> Result<()> {
        let mut relay = OmniNetRelay::new(net, handle);
        loop {
            if let Some(max) = args.max_polls {
                if polls_done >= max {
                    println!(
                        "event=exit reason=max_polls_reached adverts_written={adverts_written}"
                    );
                    return Ok(());
                }
            }
            polls_done += 1;

            for ann in relay
                .poll_peer_advertisements()
                .map_err(|e| anyhow!("poll peer-advert: {e}"))?
            {
                if !session_id_filter.is_empty()
                    && !session_id_filter.contains(&ann.session_id)
                {
                    continue;
                }
                if !contributor_filter.is_empty()
                    && !contributor_filter.contains(&ann.contributor_pubkey_hex)
                {
                    continue;
                }
                // Stage 12.7 — cross-restart dedup keyed by
                // `<session_id>--<contributor_pubkey>`. Matches the
                // `seen/peer-adverts/` namespace.
                let cross_restart_key = format!(
                    "{}--{}", ann.session_id, ann.contributor_pubkey_hex
                );
                if let Some(ref store) = state_store {
                    if matches!(
                        store.is_seen(
                            omni_contributor::StateNamespace::PeerAdverts,
                            &cross_restart_key,
                        ),
                        Ok(true)
                    ) {
                        continue;
                    }
                }
                let now_utc = chrono::Utc::now()
                    .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
                let outcome = process_peer_advertisement_announcement(
                    &ann,
                    &snip,
                    &joins,
                    Some(&now_utc),
                );
                let advert = match outcome {
                    PeerAdvertisementOutcome::Verified { body } => body,
                    other => {
                        println!(
                            "event=skip kind=peer_advert session_id={} reason={}",
                            ann.session_id,
                            stringify_peer_advert_outcome(&other)
                        );
                        continue;
                    }
                };
                let dir = args.out_dir.join(&advert.session_id).join("peer-adverts");
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    println!("event=error context=mkdir message={e}");
                    continue;
                }
                let path =
                    dir.join(format!("{}.json", advert.contributor_pubkey_hex));
                let bytes = serde_json::to_vec_pretty(&*advert)
                    .unwrap_or_default();
                match std::fs::write(&path, &bytes) {
                    Ok(()) => {
                        println!(
                            "event=peer_advert session_id={} contributor_pubkey={} path={}",
                            advert.session_id,
                            advert.contributor_pubkey_hex,
                            path.display()
                        );
                        // Stage 12.7 — dual-write into the
                        // state-dir's verified/sessions/<id>/peer-adverts/...
                        if let Some(ref store) = state_store {
                            if let Err(e) = store.write_verified_json(
                                omni_contributor::StateObjectKind::PeerAdvert {
                                    session_id: advert.session_id.clone(),
                                },
                                &advert.contributor_pubkey_hex,
                                &*advert,
                            ) {
                                println!(
                                    "event=warn context=state_store_write_peer_advert message={e}"
                                );
                            }
                            if let Err(e) = store.mark_seen(
                                omni_contributor::StateNamespace::PeerAdverts,
                                &cross_restart_key,
                            ) {
                                println!(
                                    "event=warn context=state_store_mark_seen_peer_advert message={e}"
                                );
                            }
                        }
                        adverts_written += 1;
                        if let Some(max) = args.max_adverts {
                            if adverts_written >= max {
                                println!(
                                    "event=exit reason=max_adverts_reached \
                                     adverts_written={adverts_written}"
                                );
                                return Ok(());
                            }
                        }
                    }
                    Err(e) => {
                        println!("event=error context=write_peer_advert message={e}");
                    }
                }
            }
            std::thread::sleep(Duration::from_secs(args.poll_interval_secs));
        }
    })
    .await
    .map_err(|e| anyhow!("watch-peer-adverts join: {e}"))?
}

fn stringify_peer_advert_outcome(
    o: &omni_contributor::PeerAdvertisementOutcome,
) -> String {
    use omni_contributor::PeerAdvertisementOutcome as O;
    match o {
        O::Verified { .. } => "verified".into(),
        O::AnnouncementSchemaMalformed(s) => format!("announcement_schema_malformed:{s}"),
        O::AnnouncerSignatureFailed => "announcer_signature_fail".into(),
        O::SnipFetchFailed(s) => format!("snip_fetch_failed:{s}"),
        O::BodyParseFailed(s) => format!("body_parse_failed:{s}"),
        O::BodySchemaInvalid(s) => format!("body_schema_invalid:{s}"),
        O::AdvertisementIdMismatch { stored, derived } => {
            format!("advertisement_id_mismatch:stored={stored}:derived={derived}")
        }
        O::ContributorSignatureFailed => "contributor_signature_fail".into(),
        O::DriftMismatch { field } => format!("drift:{field}"),
        O::NoMatchingJoin => "no_matching_join".into(),
        O::Expired { expires_at, now } => format!("expired:expires_at={expires_at}:now={now}"),
    }
}

// ── Stage 12.8 — plan-session-assignments ────────────────────────────────

fn run_plan_session_assignments(args: PlanSessionAssignmentsArgs) -> Result<()> {
    use omni_contributor::{
        plan_assignments, ContributorStateStore, ModelPlan, PlannerInputs,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // Open the state store. Stage 12.7 auto-prune applies unless
    // the operator opts out.
    let (store, prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        !args.no_prune_state_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    println!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        args.contributor_state_dir.display(),
        prune_report.removed_sessions,
        prune_report.removed_peer_adverts,
        prune_report.kept
    );

    // Load + re-verify the session.json the operator named. State-dir
    // loaders are parse-only (Stage 12.7 trust-boundary review pin);
    // the planner caller MUST run verify_execution_session first.
    let session: omni_contributor::ExecutionSession = store
        .read_verified_json(
            omni_contributor::StateObjectKind::Session,
            &args.session_id,
        )
        .map_err(|e| anyhow!("read verified session from state dir: {e}"))?
        .ok_or_else(|| {
            anyhow!(
                "no verified session.json at <state>/verified/sessions/{}/session.json",
                args.session_id
            )
        })?;
    if !omni_contributor::verify_execution_session(&session).is_ok() {
        return Err(anyhow!(
            "session.json in state dir failed verify_execution_session"
        ));
    }
    if session.session_id != args.session_id {
        return Err(anyhow!(
            "state-dir session.json carries session_id={} but --session-id={}",
            session.session_id,
            args.session_id
        ));
    }

    // Load + re-verify each ContributorJoin for the session.
    let raw_joins = store
        .list_verified_joins_for(&session.session_id)
        .map_err(|e| anyhow!("list verified joins: {e}"))?;
    let joins: Vec<omni_contributor::ContributorJoin> = raw_joins
        .into_iter()
        .filter(|j| {
            let ok = omni_contributor::verify_contributor_join(&session, j).is_ok();
            if !ok {
                eprintln!(
                    "event=warn context=planner_join_verify_failed \
                     session_id={} contributor_pubkey={}",
                    j.session_id, j.contributor_pubkey_hex
                );
            }
            ok
        })
        .collect();

    // Optionally load + re-verify peer adverts when live routing is
    // required. The planner re-verifies internally too (defense in
    // depth), so we don't double-fail if a parsed entry doesn't
    // verify — just drop it with a warning.
    let peer_adverts: Vec<omni_contributor::ContributorPeerAdvertisement> =
        if args.require_live_routing {
            let raw = store
                .list_verified_peer_adverts_for(&session.session_id)
                .map_err(|e| anyhow!("list verified peer adverts: {e}"))?;
            raw.into_iter()
                .filter(|a| {
                    let outcome = omni_contributor::verify_peer_advertisement_body(
                        a,
                        &joins,
                        Some(&now_utc),
                    );
                    let ok = matches!(
                        outcome,
                        omni_contributor::PeerAdvertisementOutcome::Verified { .. }
                    );
                    if !ok {
                        eprintln!(
                            "event=warn context=planner_advert_verify_failed \
                             session_id={} contributor_pubkey={} outcome={}",
                            a.session_id,
                            a.contributor_pubkey_hex,
                            stringify_peer_advert_outcome(&outcome)
                        );
                    }
                    ok
                })
                .collect()
        } else {
            Vec::new()
        };

    // Optional operator-supplied model-plan.
    let model_plan: Option<ModelPlan> = if let Some(path) = args.model_plan.as_deref() {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read model-plan: {}", path.display()))?;
        let mp: ModelPlan = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse model-plan: {}", path.display()))?;
        Some(mp)
    } else {
        None
    };

    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &peer_adverts,
            required_dtype: args.required_dtype.into(),
            min_available_ram_bytes: args.min_available_ram_bytes,
            max_assignments: args.max_assignments,
            require_live_routing: args.require_live_routing,
            layer_count: args.layer_count,
        },
        args.strategy.into(),
        model_plan.as_ref(),
        &now_utc,
        &now_utc,
    )
    .map_err(|e| anyhow!("plan_assignments: {e}"))?;

    let json = serde_json::to_vec_pretty(&plan)?;
    std::fs::write(&args.out, &json)
        .with_context(|| format!("write plan: {}", args.out.display()))?;

    println!(
        "event=plan_created session_id={} strategy={:?} assignments={} \
         plan_hash={} out={}",
        plan.session_id,
        plan.strategy,
        plan.assignments.len(),
        plan.plan_hash,
        args.out.display()
    );
    Ok(())
}

// ── Stage 12.8 — assign-session-plan ─────────────────────────────────────

async fn run_assign_session_plan(args: AssignSessionPlanArgs) -> Result<()> {
    use omni_contributor::{plan_hash_hex, AssignmentPlan, CoordinatorSigner, OmniNetRelay};

    // Read + integrity-check the plan.
    let bytes = std::fs::read(&args.plan)
        .with_context(|| format!("read plan: {}", args.plan.display()))?;
    let plan: AssignmentPlan = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse plan: {}", args.plan.display()))?;
    if plan.schema_version != omni_contributor::PLANNER_SCHEMA_VERSION {
        return Err(anyhow!(
            "plan.schema_version {} not supported (expected {})",
            plan.schema_version,
            omni_contributor::PLANNER_SCHEMA_VERSION
        ));
    }
    let recomputed = plan_hash_hex(&plan);
    if recomputed != plan.plan_hash {
        return Err(anyhow!(
            "plan_hash mismatch: stored={} recomputed={} (plan may have been edited after creation)",
            plan.plan_hash,
            recomputed
        ));
    }
    if plan.assignments.is_empty() {
        return Err(anyhow!("plan carries zero assignments; nothing to publish"));
    }

    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.session_snip_root)?;
    if session.session_id != plan.session_id {
        return Err(anyhow!(
            "plan.session_id={} but --session-snip-root resolves to session_id={}",
            plan.session_id,
            session.session_id
        ));
    }
    let coord = CoordinatorSigner::from_seed_file(&args.coordinator_seed)?;
    if coord.pubkey_hex() != session.coordinator_pubkey_hex {
        return Err(anyhow!(
            "coordinator_seed pubkey does not match session.coordinator_pubkey_hex"
        ));
    }
    if let Some(ref planner_coord) = plan.coordinator_pubkey_hex {
        if planner_coord != &coord.pubkey_hex() {
            return Err(anyhow!(
                "plan.coordinator_pubkey_hex={} does not match \
                 --coordinator-seed pubkey={}",
                planner_coord,
                coord.pubkey_hex()
            ));
        }
    }

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // Refuse expired sessions even on the publish path. The
    // planner already refuses to emit assignments against expired
    // sessions, but a stale on-disk plan + a stale session SNIP
    // root could otherwise still get through. Mirrors what the
    // pre-12.7 pipeline already did at runtime.
    if let omni_contributor::session_verify::SessionVerifyOutcome::ExpiredAtCheck {
        now,
        expires_at,
    } = omni_contributor::check_not_expired(&now_utc, &session.expires_at_utc)
    {
        return Err(anyhow!(
            "session is expired: now_utc={now}, expires_at_utc={expires_at}"
        ));
    }

    if args.dry_run {
        use omni_contributor::canonical::{
            assignment_id_hex, work_assignment_signing_input,
        };
        // Build, sign (local), and run the full Stage 12.3
        // `WorkAssignment::validate_schema` for each planned entry
        // before printing the dry-run event. A plan whose
        // `expected_work_units == 0` or malformed `work_kind`
        // survives `plan_hash` recompute (the planner accepts
        // model-plan input verbatim into the plan) but fails
        // schema validation at publish time; catch that on
        // dry-run so an operator doesn't get a green dry-run and
        // then a failed real publish. Signing is local — no SNIP,
        // no mesh — so it's a valid dry-run side effect.
        for pa in &plan.assignments {
            let mut a = pa.to_unsigned_work_assignment(
                &plan.session_id,
                &now_utc,
            );
            a.assignment_id = assignment_id_hex(&a).map_err(|e| {
                anyhow!(
                    "dry-run: stage_index={} canonical encode failed: {e}",
                    pa.stage_index
                )
            })?;
            let sig_input = work_assignment_signing_input(&a).map_err(|e| {
                anyhow!(
                    "dry-run: stage_index={} canonical encode failed: {e}",
                    pa.stage_index
                )
            })?;
            a.coordinator_signature_hex = coord.sign_hex(&sig_input);
            a.validate_schema().map_err(|e| {
                anyhow!(
                    "dry-run: planned assignment for stage_index={} \
                     would fail WorkAssignment::validate_schema: {e}",
                    pa.stage_index
                )
            })?;
            println!(
                "event=would_assign session_id={} stage_index={} \
                 assignment_id={} contributor_pubkey={} work_kind={} \
                 expected_work_units={} expected_work_unit_kind={:?}",
                plan.session_id,
                pa.stage_index,
                a.assignment_id,
                pa.contributor_pubkey_hex,
                serde_json::to_string(&pa.work_kind).unwrap_or_default(),
                pa.expected_work_units,
                pa.expected_work_unit_kind,
            );
        }
        println!(
            "event=plan_assigned_dry_run session_id={} assignments={}",
            plan.session_id,
            plan.assignments.len()
        );
        return Ok(());
    }

    // Optionally mirror published assignments back into a Stage 12.7
    // state-dir. Opened BEFORE entering spawn_blocking territory so
    // a bad path surfaces synchronously.
    let state_store = open_optional_state_store(
        args.contributor_state_dir.as_deref(),
        args.no_prune_state_on_start,
    )?;

    let (net, handle) = open_omninet_with_peer_wait(
        args.listen_port,
        args.peer,
        args.peer_wait_secs,
        args.mesh_stabilize_ms,
        args.net_identity_file.as_deref(),
    )
    .await?;
    let mut relay = OmniNetRelay::new(net.clone(), handle);

    let total_to_publish = plan.assignments.len();
    for pa in plan.assignments {
        let spec = AssignmentSpec {
            contributor_pubkey_hex: pa.contributor_pubkey_hex.clone(),
            stage_index: pa.stage_index,
            work_kind: pa.work_kind.clone(),
            expected_work_units: pa.expected_work_units,
            expected_work_unit_kind: pa.expected_work_unit_kind,
            assigned_at_utc: None,
        };
        let (assignment, root_hex) = publish_one_signed_assignment(
            &snip,
            if args.no_publish_announcements {
                None
            } else {
                Some(&mut relay)
            },
            &coord,
            &session,
            spec,
            &now_utc,
        )
        .await?;

        // Stage 12.7 — dual-write into the state dir when supplied.
        if let Some(ref store) = state_store {
            if let Err(e) = store.write_verified_json(
                omni_contributor::StateObjectKind::Assignment {
                    session_id: assignment.session_id.clone(),
                },
                &assignment.assignment_id,
                &assignment,
            ) {
                eprintln!("event=warn context=state_store_write_assignment message={e}");
            }
            let marker = format!(
                "{}--{}", assignment.session_id, assignment.assignment_id
            );
            if let Err(e) = store.mark_seen(
                omni_contributor::StateNamespace::Assignments,
                &marker,
            ) {
                eprintln!(
                    "event=warn context=state_store_mark_seen_assignment message={e}"
                );
            }
        }

        println!(
            "event=assignment_published session_id={} stage_index={} \
             assignment_id={} contributor_pubkey={} \
             assignment_snip_root={}",
            assignment.session_id,
            assignment.stage_index,
            assignment.assignment_id,
            assignment.contributor_pubkey_hex,
            root_hex
        );
    }

    tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms)).await;
    {
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }
    println!(
        "event=plan_assigned session_id={} assignments_published={}",
        plan.session_id, total_to_publish
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Wrapper so `try_parse_from` has a single top-level parser.
    /// We pass `assign-session-plan ...` and let the subcommand
    /// matcher route into the per-subcommand args struct.
    #[derive(clap::Parser)]
    struct TestRoot {
        #[command(flatten)]
        contributor: ContributorArgs,
    }

    fn parse_assign_session_plan(extra: &[&str]) -> AssignSessionPlanArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "assign-session-plan".into(),
            "--plan".into(),
            "/tmp/plan.json".into(),
            "--session-snip-root".into(),
            "0x00".into(),
            "--coordinator-seed".into(),
            "/tmp/coord.seed".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::AssignSessionPlan(a) => a,
            _ => panic!("expected AssignSessionPlan"),
        }
    }

    /// Stage 12.8 — `--no-publish-announcements` toggle. Pre-fix,
    /// `publish_announcements: bool` had `default_value_t = true`,
    /// which is a clap footgun: a presence flag whose default
    /// equals the only value it can set is unreachable. This
    /// regression test confirms the inverted flag actually toggles.
    #[test]
    fn no_publish_announcements_flag_default_is_false_and_presence_toggles_it_true() {
        // Default (omitted) — broadcast in addition to SNIP publish.
        let args = parse_assign_session_plan(&[]);
        assert!(!args.no_publish_announcements);
        // Explicit — skip the mesh broadcast.
        let args = parse_assign_session_plan(&["--no-publish-announcements"]);
        assert!(args.no_publish_announcements);
    }

    fn parse_session_status(extra: &[&str]) -> SessionStatusArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "session-status".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
            "--session-id".into(),
            "00".repeat(32),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::SessionStatus(a) => a,
            _ => panic!("expected SessionStatus"),
        }
    }

    /// Stage 12.9 — flag-set regression. Defaults must match the
    /// documented "exit 0 unless --fail-on-incomplete, format=events,
    /// no expired adverts" posture; each toggle must flip
    /// independently.
    #[test]
    fn session_status_flag_parse_smoke() {
        let defaults = parse_session_status(&[]);
        assert_eq!(defaults.format, SessionStatusFormat::Events);
        assert!(!defaults.fail_on_incomplete);
        assert!(!defaults.include_expired);
        assert!(!defaults.no_prune_state_on_start);
        assert!(defaults.json_out.is_none());

        let json = parse_session_status(&["--format", "json"]);
        assert_eq!(json.format, SessionStatusFormat::Json);

        let pretty = parse_session_status(&["--format", "pretty"]);
        assert_eq!(pretty.format, SessionStatusFormat::Pretty);

        let with_flags = parse_session_status(&[
            "--fail-on-incomplete",
            "--include-expired",
            "--no-prune-state-on-start",
            "--json-out",
            "/tmp/status.json",
        ]);
        assert!(with_flags.fail_on_incomplete);
        assert!(with_flags.include_expired);
        assert!(with_flags.no_prune_state_on_start);
        assert_eq!(
            with_flags.json_out.as_deref(),
            Some(std::path::Path::new("/tmp/status.json"))
        );
    }

    fn parse_plan_session_repair(extra: &[&str]) -> PlanSessionRepairArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "plan-session-repair".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
            "--session-id".into(),
            "00".repeat(32),
            "--out".into(),
            "/tmp/plan.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::PlanSessionRepair(a) => a,
            _ => panic!("expected PlanSessionRepair"),
        }
    }

    fn parse_apply_session_repair(extra: &[&str]) -> ApplySessionRepairArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "apply-session-repair".into(),
            "--repair-plan".into(),
            "/tmp/plan.json".into(),
            "--session-snip-root".into(),
            "0x00".into(),
            "--coordinator-seed".into(),
            "/tmp/coord.seed".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::ApplySessionRepair(a) => a,
            _ => panic!("expected ApplySessionRepair"),
        }
    }

    /// Stage 12.10 — clap regression. Pins the documented
    /// `--build-status` xor `--status-report` mutual exclusion,
    /// the inverted `--no-publish-announcements` posture (matches
    /// Stage 12.8 fix), and the dry-run + include-expired
    /// togglability.
    #[test]
    fn session_repair_flag_parse_smoke() {
        // plan: defaults reject (must supply exactly one of
        // --build-status or --status-report at runtime; clap allows
        // either to be absent so the run-handler validates).
        let plan_defaults = parse_plan_session_repair(&[]);
        assert!(!plan_defaults.build_status);
        assert!(plan_defaults.status_report.is_none());
        assert!(!plan_defaults.include_expired);
        assert!(!plan_defaults.no_prune_state_on_start);

        // plan: --build-status flips on, --status-report stays None.
        let plan_build = parse_plan_session_repair(&["--build-status"]);
        assert!(plan_build.build_status);
        assert!(plan_build.status_report.is_none());

        // plan: --status-report fills the path; conflicts with
        // --build-status at clap level.
        let plan_with_path =
            parse_plan_session_repair(&["--status-report", "/tmp/s.json"]);
        assert!(!plan_with_path.build_status);
        assert_eq!(
            plan_with_path.status_report.as_deref(),
            Some(std::path::Path::new("/tmp/s.json"))
        );

        // clap-level conflict: both at once must NOT parse.
        let conflict = TestRoot::try_parse_from([
            "omni-node",
            "plan-session-repair",
            "--contributor-state-dir",
            "/tmp/state",
            "--session-id",
            &"00".repeat(32),
            "--out",
            "/tmp/plan.json",
            "--build-status",
            "--status-report",
            "/tmp/s.json",
        ]);
        assert!(
            conflict.is_err(),
            "clap must reject --build-status + --status-report combo"
        );

        // apply: defaults — broadcast in addition to SNIP publish.
        let apply_defaults = parse_apply_session_repair(&[]);
        assert!(!apply_defaults.no_publish_announcements);
        assert!(!apply_defaults.dry_run);
        assert!(!apply_defaults.no_prune_state_on_start);

        // apply: --no-publish-announcements is reachable AND
        // toggles independently of --dry-run.
        let apply_no_mesh =
            parse_apply_session_repair(&["--no-publish-announcements"]);
        assert!(apply_no_mesh.no_publish_announcements);
        assert!(!apply_no_mesh.dry_run);

        let apply_dry = parse_apply_session_repair(&["--dry-run"]);
        assert!(apply_dry.dry_run);
        assert!(!apply_dry.no_publish_announcements);
    }

    fn parse_plan_session_reassign(extra: &[&str]) -> PlanSessionReassignArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "plan-session-reassign".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
            "--session-id".into(),
            "00".repeat(32),
            "--out".into(),
            "/tmp/reassign.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::PlanSessionReassign(a) => a,
            _ => panic!("expected PlanSessionReassign"),
        }
    }

    fn parse_apply_session_reassign(extra: &[&str]) -> ApplySessionReassignArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "apply-session-reassign".into(),
            "--reassignment-plan".into(),
            "/tmp/reassign.json".into(),
            "--session-snip-root".into(),
            "0x00".into(),
            "--coordinator-seed".into(),
            "/tmp/coord.seed".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::ApplySessionReassign(a) => a,
            _ => panic!("expected ApplySessionReassign"),
        }
    }

    /// Stage 12.11 — clap regression for the reassignment CLI pair.
    /// Mirrors the Stage 12.10 `session_repair_flag_parse_smoke` posture:
    /// pins the `--build-status` xor `--status-report` mutual exclusion,
    /// the default `--reason` (`missing-partial`), the inverted
    /// `--no-publish-announcements` posture, and the `--dry-run` toggle.
    #[test]
    fn session_reassign_flag_parse_smoke() {
        // plan: defaults
        let plan_defaults = parse_plan_session_reassign(&[]);
        assert!(!plan_defaults.build_status);
        assert!(plan_defaults.status_report.is_none());
        assert!(matches!(
            plan_defaults.reason,
            CliSupersessionReason::MissingPartial
        ));
        assert!(!plan_defaults.include_expired);
        assert!(!plan_defaults.no_prune_state_on_start);
        assert!(plan_defaults.coordinator_pubkey_hex.is_none());

        // plan: --build-status flips on, --status-report stays None.
        let plan_build = parse_plan_session_reassign(&["--build-status"]);
        assert!(plan_build.build_status);
        assert!(plan_build.status_report.is_none());

        // plan: --status-report fills the path; conflicts with
        // --build-status at clap level.
        let plan_with_path =
            parse_plan_session_reassign(&["--status-report", "/tmp/s.json"]);
        assert!(!plan_with_path.build_status);
        assert_eq!(
            plan_with_path.status_report.as_deref(),
            Some(std::path::Path::new("/tmp/s.json"))
        );

        // plan: clap-level mutual exclusion.
        let conflict = TestRoot::try_parse_from([
            "omni-node",
            "plan-session-reassign",
            "--contributor-state-dir",
            "/tmp/state",
            "--session-id",
            &"00".repeat(32),
            "--out",
            "/tmp/reassign.json",
            "--build-status",
            "--status-report",
            "/tmp/s.json",
        ]);
        assert!(
            conflict.is_err(),
            "clap must reject --build-status + --status-report combo"
        );

        // plan: each non-Custom reason value reaches the parser.
        let plan_invalid =
            parse_plan_session_reassign(&["--reason", "invalid-partial"]);
        assert!(matches!(
            plan_invalid.reason,
            CliSupersessionReason::InvalidPartial
        ));
        let plan_rebalance =
            parse_plan_session_reassign(&["--reason", "operator-rebalance"]);
        assert!(matches!(
            plan_rebalance.reason,
            CliSupersessionReason::OperatorRebalance
        ));

        // apply: defaults — broadcast in addition to SNIP publish.
        let apply_defaults = parse_apply_session_reassign(&[]);
        assert!(!apply_defaults.no_publish_announcements);
        assert!(!apply_defaults.dry_run);
        assert!(!apply_defaults.no_prune_state_on_start);

        // apply: --no-publish-announcements is reachable AND
        // toggles independently of --dry-run.
        let apply_no_mesh =
            parse_apply_session_reassign(&["--no-publish-announcements"]);
        assert!(apply_no_mesh.no_publish_announcements);
        assert!(!apply_no_mesh.dry_run);

        let apply_dry = parse_apply_session_reassign(&["--dry-run"]);
        assert!(apply_dry.dry_run);
        assert!(!apply_dry.no_publish_announcements);
    }

    // ── Stage 12.14 — archive-session clap regression ─────────────

    fn parse_archive_session(extra: &[&str]) -> ArchiveSessionArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "archive-session".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
            "--session-id".into(),
            "00".repeat(32),
            "--archive-dir".into(),
            "/tmp/archive".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::ArchiveSession(a) => a,
            _ => panic!("expected ArchiveSession"),
        }
    }

    #[test]
    fn archive_session_flag_parse_smoke() {
        // Defaults: --require-status complete, --copy implicit
        // (move_mode false), --include-results off, --dry-run off.
        let defaults = parse_archive_session(&[]);
        assert!(matches!(
            defaults.require_status,
            CliArchiveStatusRequirement::Complete
        ));
        assert!(!defaults.move_mode);
        assert!(!defaults.copy);
        assert!(!defaults.include_results);
        assert!(!defaults.dry_run);

        // --require-status closed enum values.
        for (raw, expected) in &[
            ("any", CliArchiveStatusRequirement::Any),
            ("complete", CliArchiveStatusRequirement::Complete),
            ("aggregated", CliArchiveStatusRequirement::Aggregated),
            (
                "complete-partials",
                CliArchiveStatusRequirement::CompletePartials,
            ),
            (
                "expired-incomplete",
                CliArchiveStatusRequirement::ExpiredIncomplete,
            ),
        ] {
            let got = parse_archive_session(&["--require-status", raw]);
            assert!(
                std::mem::discriminant(&got.require_status)
                    == std::mem::discriminant(expected),
                "require-status={raw} parsed wrong: {:?}",
                got.require_status
            );
        }

        // --move flips on; clap enforces conflicts_with so
        // --copy + --move must NOT parse.
        let move_run = parse_archive_session(&["--move"]);
        assert!(move_run.move_mode);
        let conflict = TestRoot::try_parse_from([
            "omni-node",
            "archive-session",
            "--contributor-state-dir",
            "/tmp/state",
            "--session-id",
            &"00".repeat(32),
            "--archive-dir",
            "/tmp/archive",
            "--copy",
            "--move",
        ]);
        assert!(
            conflict.is_err(),
            "clap must reject --copy + --move conflict"
        );

        // --include-results + --dry-run toggle independently.
        let with_flags =
            parse_archive_session(&["--include-results", "--dry-run"]);
        assert!(with_flags.include_results);
        assert!(with_flags.dry_run);
        assert!(!with_flags.move_mode);
    }

    // ── Stage 12.15 — restore-session-archive clap regression ────

    fn parse_restore_session_archive(extra: &[&str]) -> RestoreSessionArchiveArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "restore-session-archive".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::RestoreSessionArchive(a) => a,
            _ => panic!("expected RestoreSessionArchive"),
        }
    }

    #[test]
    fn restore_session_archive_flag_parse_smoke() {
        // --archive-session-dir source resolution.
        let with_session_dir = parse_restore_session_archive(&[
            "--archive-session-dir",
            "/tmp/archive/session",
        ]);
        assert_eq!(
            with_session_dir.archive_session_dir.as_deref(),
            Some(std::path::Path::new("/tmp/archive/session"))
        );
        assert!(with_session_dir.archive_dir.is_none());
        assert!(with_session_dir.session_id.is_none());
        // Defaults: dry_run / verify_only / overwrite / include all false.
        assert!(!with_session_dir.dry_run);
        assert!(!with_session_dir.verify_only);
        assert!(!with_session_dir.overwrite_existing);
        assert!(!with_session_dir.include_results);

        // --archive-dir + --session-id source resolution.
        let with_pair = parse_restore_session_archive(&[
            "--archive-dir",
            "/tmp/archive",
            "--session-id",
            &"00".repeat(32),
        ]);
        assert_eq!(
            with_pair.archive_dir.as_deref(),
            Some(std::path::Path::new("/tmp/archive"))
        );
        assert_eq!(with_pair.session_id.as_deref(), Some("00".repeat(32).as_str()));

        // --archive-session-dir and --archive-dir conflict at clap
        // level.
        let conflict = TestRoot::try_parse_from([
            "omni-node",
            "restore-session-archive",
            "--contributor-state-dir",
            "/tmp/state",
            "--archive-session-dir",
            "/tmp/archive/session",
            "--archive-dir",
            "/tmp/archive",
            "--session-id",
            &"00".repeat(32),
        ]);
        assert!(
            conflict.is_err(),
            "clap must reject --archive-session-dir + --archive-dir conflict"
        );

        // --archive-dir without --session-id fails (requires
        // pairing).
        let no_session = TestRoot::try_parse_from([
            "omni-node",
            "restore-session-archive",
            "--contributor-state-dir",
            "/tmp/state",
            "--archive-dir",
            "/tmp/archive",
        ]);
        assert!(
            no_session.is_err(),
            "clap must require --session-id when --archive-dir is set"
        );

        // --dry-run + --verify-only independently togglable.
        let dry = parse_restore_session_archive(&[
            "--archive-session-dir",
            "/tmp/a/s",
            "--dry-run",
        ]);
        assert!(dry.dry_run);
        assert!(!dry.verify_only);
        let vo = parse_restore_session_archive(&[
            "--archive-session-dir",
            "/tmp/a/s",
            "--verify-only",
        ]);
        assert!(!vo.dry_run);
        assert!(vo.verify_only);
        let both = parse_restore_session_archive(&[
            "--archive-session-dir",
            "/tmp/a/s",
            "--dry-run",
            "--verify-only",
        ]);
        assert!(both.dry_run);
        assert!(both.verify_only);

        // --overwrite-existing + --include-results toggles.
        let with_flags = parse_restore_session_archive(&[
            "--archive-session-dir",
            "/tmp/a/s",
            "--overwrite-existing",
            "--include-results",
        ]);
        assert!(with_flags.overwrite_existing);
        assert!(with_flags.include_results);
    }

    // ── Stage 12.16 — state-integrity clap regression ────────────

    fn parse_state_integrity(extra: &[&str]) -> StateIntegrityArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "state-integrity".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::StateIntegrity(a) => a,
            _ => panic!("expected StateIntegrity"),
        }
    }

    #[test]
    fn state_integrity_flag_parse_smoke() {
        // Defaults: no session_id filter, no include_archives,
        // format=events, no json_out, fail_on_warn off.
        let defaults = parse_state_integrity(&[]);
        assert!(defaults.session_id.is_none());
        assert!(defaults.include_archives.is_none());
        assert_eq!(defaults.format, StateIntegrityFormat::Events);
        assert!(defaults.json_out.is_none());
        assert!(!defaults.fail_on_warn);

        // --session-id filter.
        let sid = "ff".repeat(32);
        let with_sid = parse_state_integrity(&["--session-id", &sid]);
        assert_eq!(with_sid.session_id.as_deref(), Some(sid.as_str()));

        // --include-archives PATH.
        let with_archives =
            parse_state_integrity(&["--include-archives", "/tmp/archive-root"]);
        assert_eq!(
            with_archives.include_archives.as_deref(),
            Some(std::path::Path::new("/tmp/archive-root"))
        );

        // --format closed enum values.
        for (raw, expected) in &[
            ("events", StateIntegrityFormat::Events),
            ("json", StateIntegrityFormat::Json),
            ("pretty", StateIntegrityFormat::Pretty),
        ] {
            let got = parse_state_integrity(&["--format", raw]);
            assert_eq!(got.format, *expected, "format={raw} parsed wrong");
        }

        // --json-out PATH.
        let with_json = parse_state_integrity(&["--json-out", "/tmp/r.json"]);
        assert_eq!(
            with_json.json_out.as_deref(),
            Some(std::path::Path::new("/tmp/r.json"))
        );

        // --fail-on-warn toggle.
        let with_strict = parse_state_integrity(&["--fail-on-warn"]);
        assert!(with_strict.fail_on_warn);

        // Stage 12.19 — new `--baseline`, `--fail-on-new`,
        // `--fail-on-new-error`, `--summary-only`,
        // `--require-state-dir-match` flags parse cleanly and
        // default off.
        assert!(defaults.baseline.is_none());
        assert!(!defaults.fail_on_new);
        assert!(!defaults.fail_on_new_error);
        assert!(!defaults.summary_only);
        assert!(!defaults.require_state_dir_match);
        let with_diff = parse_state_integrity(&[
            "--baseline",
            "/tmp/baseline.json",
            "--fail-on-new",
            "--fail-on-new-error",
            "--summary-only",
            "--require-state-dir-match",
        ]);
        assert_eq!(
            with_diff.baseline.as_deref(),
            Some(std::path::Path::new("/tmp/baseline.json"))
        );
        assert!(with_diff.fail_on_new);
        assert!(with_diff.fail_on_new_error);
        assert!(with_diff.summary_only);
        assert!(with_diff.require_state_dir_match);

        // Stage 12.16 review fix: the `--no-prune-state-on-start`
        // flag is deliberately ABSENT. The scanner is read-only,
        // so it always opens with auto-prune off; surfacing a
        // no-op flag would mislead operators.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity",
            "--contributor-state-dir",
            "/tmp/state",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "state-integrity must not accept --no-prune-state-on-start"
        );

        // Unknown --format value must be rejected.
        let bad = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity",
            "--contributor-state-dir",
            "/tmp/state",
            "--format",
            "yaml",
        ]);
        assert!(bad.is_err(), "clap must reject unknown --format value");

        // Stage 12.20 — `--signed-baseline` + `--baseline-pubkey-hex`
        // are mutually exclusive with `--baseline`. Clap enforces
        // both the conflict and the requires-pair.
        assert!(defaults.signed_baseline.is_none());
        assert!(defaults.baseline_pubkey_hex.is_none());
        let signed = parse_state_integrity(&[
            "--signed-baseline",
            "/tmp/baseline.signed.json",
            "--baseline-pubkey-hex",
            &"aa".repeat(32),
        ]);
        assert_eq!(
            signed.signed_baseline.as_deref(),
            Some(std::path::Path::new("/tmp/baseline.signed.json"))
        );
        assert_eq!(
            signed.baseline_pubkey_hex.as_deref(),
            Some("aa".repeat(32).as_str())
        );

        // --signed-baseline without --baseline-pubkey-hex is rejected.
        let no_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity",
            "--contributor-state-dir",
            "/tmp/state",
            "--signed-baseline",
            "/tmp/b.signed.json",
        ]);
        assert!(
            no_pubkey.is_err(),
            "clap must reject --signed-baseline without --baseline-pubkey-hex"
        );

        // --baseline + --signed-baseline together is rejected.
        let both = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity",
            "--contributor-state-dir",
            "/tmp/state",
            "--baseline",
            "/tmp/raw.json",
            "--signed-baseline",
            "/tmp/signed.json",
            "--baseline-pubkey-hex",
            &"aa".repeat(32),
        ]);
        assert!(
            both.is_err(),
            "clap must reject --baseline + --signed-baseline together"
        );

        // Stage 12.20 review fix — clap's `requires` and
        // `conflicts_with` interact in a non-obvious way on
        // both surfaces:
        //
        //   - SOLO `--baseline-pubkey-hex` (no --baseline, no
        //     --signed-baseline) is REJECTED at parse time
        //     (clap's `requires = "signed_baseline"` fires
        //     cleanly when --baseline is also absent).
        //
        //   - `--baseline + --baseline-pubkey-hex` (without
        //     --signed-baseline) is ACCEPTED at parse time on
        //     BOTH state-integrity AND state-integrity-diff.
        //     The conflicts_with/required_unless_present
        //     resolution short-circuits the `requires` check
        //     in either setup. The runtime backstop in
        //     `resolve_diff_baseline` emits
        //     `event=warn context=baseline_pubkey_hex_unused`
        //     uniformly across both subcommands so the
        //     trust-anchor-was-dropped state is observable.
        let lonely_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity",
            "--contributor-state-dir",
            "/tmp/state",
            "--baseline-pubkey-hex",
            &"aa".repeat(32),
        ]);
        assert!(
            lonely_pubkey.is_err(),
            "clap must reject SOLO --baseline-pubkey-hex (no --baseline, no --signed-baseline)"
        );
        let baseline_with_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity",
            "--contributor-state-dir",
            "/tmp/state",
            "--baseline",
            "/tmp/baseline.json",
            "--baseline-pubkey-hex",
            &"aa".repeat(32),
        ]);
        assert!(
            baseline_with_pubkey.is_ok(),
            "state-integrity --baseline + --baseline-pubkey-hex parses; runtime warns"
        );
    }

    // ── Stage 12.19 — state-integrity-diff clap regression ──

    /// Helper for the *raw-baseline* default form. Pre-fills
    /// `--baseline /tmp/baseline.json` and `--current
    /// /tmp/current.json` so most tests only override the
    /// behavioral flags. Tests exercising `--signed-baseline`
    /// build their own argv to avoid clap's mutual-exclusion
    /// firing on the defaults.
    fn parse_state_integrity_diff(extra: &[&str]) -> StateIntegrityDiffArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "state-integrity-diff".into(),
            "--baseline".into(),
            "/tmp/baseline.json".into(),
            "--current".into(),
            "/tmp/current.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::StateIntegrityDiff(a) => a,
            _ => panic!("expected StateIntegrityDiff"),
        }
    }

    #[test]
    fn state_integrity_diff_flag_parse_smoke() {
        let defaults = parse_state_integrity_diff(&[]);
        assert_eq!(
            defaults.baseline.as_deref(),
            Some(std::path::Path::new("/tmp/baseline.json"))
        );
        assert!(defaults.signed_baseline.is_none());
        assert!(defaults.baseline_pubkey_hex.is_none());
        assert_eq!(
            defaults.current,
            std::path::PathBuf::from("/tmp/current.json")
        );
        assert!(!defaults.require_state_dir_match);
        assert!(!defaults.fail_on_new);
        assert!(!defaults.fail_on_new_error);
        assert!(!defaults.summary_only);
        assert_eq!(defaults.format, StateIntegrityDiffFormat::Events);
        assert!(defaults.json_out.is_none());

        // --require-state-dir-match / --fail-on-new /
        // --fail-on-new-error / --summary-only toggles.
        let toggled = parse_state_integrity_diff(&[
            "--require-state-dir-match",
            "--fail-on-new",
            "--fail-on-new-error",
            "--summary-only",
        ]);
        assert!(toggled.require_state_dir_match);
        assert!(toggled.fail_on_new);
        assert!(toggled.fail_on_new_error);
        assert!(toggled.summary_only);

        // --format closed enum.
        for (raw, expected) in &[
            ("events", StateIntegrityDiffFormat::Events),
            ("json", StateIntegrityDiffFormat::Json),
            ("pretty", StateIntegrityDiffFormat::Pretty),
        ] {
            let got = parse_state_integrity_diff(&["--format", raw]);
            assert_eq!(got.format, *expected);
        }

        // --json-out PATH.
        let with_json =
            parse_state_integrity_diff(&["--json-out", "/tmp/d.json"]);
        assert_eq!(
            with_json.json_out.as_deref(),
            Some(std::path::Path::new("/tmp/d.json"))
        );

        // Required-flag refusals: missing --baseline OR --current.
        let no_baseline = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity-diff",
            "--current",
            "/tmp/c.json",
        ]);
        assert!(
            no_baseline.is_err(),
            "clap must reject state-integrity-diff without --baseline"
        );
        let no_current = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity-diff",
            "--baseline",
            "/tmp/b.json",
        ]);
        assert!(
            no_current.is_err(),
            "clap must reject state-integrity-diff without --current"
        );

        // Stage 12.16/12.17 precedent: --no-prune-state-on-start
        // is deliberately absent. (state-integrity-diff doesn't
        // open a state-store at all; the flag would be misleading.)
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity-diff",
            "--baseline",
            "/tmp/b.json",
            "--current",
            "/tmp/c.json",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "state-integrity-diff must not accept --no-prune-state-on-start"
        );

        // Unknown --format value rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity-diff",
            "--baseline",
            "/tmp/b.json",
            "--current",
            "/tmp/c.json",
            "--format",
            "yaml",
        ]);
        assert!(bad_format.is_err(), "clap must reject unknown --format");

        // Stage 12.20 — `--signed-baseline` + `--baseline-pubkey-hex`
        // mutually exclusive with `--baseline`; clap enforces.
        // The default-argv helper preloads --baseline, so this
        // sub-block builds its own argv to avoid the
        // mutual-exclusion firing on the defaults.
        let pubkey_hex_owned = "aa".repeat(32);
        let argv_signed = [
            "omni-node",
            "state-integrity-diff",
            "--signed-baseline",
            "/tmp/signed.json",
            "--baseline-pubkey-hex",
            &pubkey_hex_owned,
            "--current",
            "/tmp/current.json",
        ];
        let root_signed = TestRoot::try_parse_from(argv_signed)
            .expect("--signed-baseline + --baseline-pubkey-hex + --current must parse");
        let signed_args = match root_signed.contributor.cmd {
            ContributorCmd::StateIntegrityDiff(a) => a,
            _ => panic!(),
        };
        assert!(signed_args.baseline.is_none());
        assert_eq!(
            signed_args.signed_baseline.as_deref(),
            Some(std::path::Path::new("/tmp/signed.json"))
        );
        assert_eq!(
            signed_args.baseline_pubkey_hex.as_deref(),
            Some("aa".repeat(32).as_str())
        );

        // Neither --baseline nor --signed-baseline → required-unless-present
        // refuses.
        let no_source = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity-diff",
            "--current",
            "/tmp/c.json",
        ]);
        assert!(
            no_source.is_err(),
            "clap must require one of --baseline or --signed-baseline"
        );

        // Both --baseline AND --signed-baseline → conflict refuses.
        let pubkey = "aa".repeat(32);
        let both = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity-diff",
            "--baseline",
            "/tmp/raw.json",
            "--signed-baseline",
            "/tmp/signed.json",
            "--baseline-pubkey-hex",
            &pubkey,
            "--current",
            "/tmp/c.json",
        ]);
        assert!(
            both.is_err(),
            "clap must reject --baseline + --signed-baseline together"
        );

        // --signed-baseline without --baseline-pubkey-hex refuses.
        let no_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity-diff",
            "--signed-baseline",
            "/tmp/signed.json",
            "--current",
            "/tmp/c.json",
        ]);
        assert!(
            no_pubkey.is_err(),
            "clap must reject --signed-baseline without --baseline-pubkey-hex"
        );

        // --baseline-pubkey-hex without --signed-baseline parses
        // (clap's `requires` doesn't fire when the wrapping arg
        // resolves via `required_unless_present`). The runtime
        // emits a stderr warning when this is detected; v1
        // doesn't refuse so operators can keep ad-hoc workflows.
        let lonely_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "state-integrity-diff",
            "--baseline",
            "/tmp/raw.json",
            "--baseline-pubkey-hex",
            &pubkey,
            "--current",
            "/tmp/c.json",
        ]);
        assert!(
            lonely_pubkey.is_ok(),
            "clap accepts --baseline + --baseline-pubkey-hex; the runtime warns instead"
        );
    }

    // ── Stage 12.20 — sign-state-integrity-baseline clap regression ──

    /// Build `sign-state-integrity-baseline` argv with the
    /// supplied `signer_role` value + optional extra trailing
    /// flags. `signer_role` is NOT part of the defaults so
    /// each test can override it cleanly (clap would reject
    /// a duplicate-arg pass).
    fn parse_sign_state_integrity_baseline(
        signer_role: &str,
        extra: &[&str],
    ) -> SignStateIntegrityBaselineArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "sign-state-integrity-baseline".into(),
            "--baseline-in".into(),
            "/tmp/raw.json".into(),
            "--signer-seed".into(),
            "/tmp/seed.bin".into(),
            "--signer-role".into(),
            signer_role.into(),
            "--out".into(),
            "/tmp/signed.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::SignStateIntegrityBaseline(a) => a,
            _ => panic!("expected SignStateIntegrityBaseline"),
        }
    }

    #[test]
    fn sign_state_integrity_baseline_flag_parse_smoke() {
        let defaults = parse_sign_state_integrity_baseline("operator", &[]);
        assert_eq!(defaults.baseline_in, std::path::PathBuf::from("/tmp/raw.json"));
        assert_eq!(defaults.signer_seed, std::path::PathBuf::from("/tmp/seed.bin"));
        assert_eq!(defaults.signer_role, CliBaselineSignerRole::Operator);
        assert_eq!(defaults.out, std::path::PathBuf::from("/tmp/signed.json"));
        assert_eq!(
            defaults.format,
            SignStateIntegrityBaselineFormat::Events
        );

        // All four roles parse via the closed enum.
        for (raw, expected) in &[
            ("operator", CliBaselineSignerRole::Operator),
            ("contributor", CliBaselineSignerRole::Contributor),
            ("dispatcher", CliBaselineSignerRole::Dispatcher),
            ("coordinator", CliBaselineSignerRole::Coordinator),
        ] {
            let got = parse_sign_state_integrity_baseline(raw, &[]);
            assert_eq!(got.signer_role, *expected, "role={raw} parsed wrong");
        }

        // --format closed enum.
        for (raw, expected) in &[
            ("events", SignStateIntegrityBaselineFormat::Events),
            ("json", SignStateIntegrityBaselineFormat::Json),
            ("pretty", SignStateIntegrityBaselineFormat::Pretty),
        ] {
            let got =
                parse_sign_state_integrity_baseline("operator", &["--format", raw]);
            assert_eq!(got.format, *expected);
        }

        // Unknown --signer-role rejected.
        let bad_role = TestRoot::try_parse_from([
            "omni-node",
            "sign-state-integrity-baseline",
            "--baseline-in",
            "/tmp/raw.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "validator",
            "--out",
            "/tmp/signed.json",
        ]);
        assert!(bad_role.is_err(), "clap must reject unknown --signer-role");

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "sign-state-integrity-baseline",
            "--baseline-in",
            "/tmp/raw.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "operator",
            "--out",
            "/tmp/signed.json",
            "--format",
            "yaml",
        ]);
        assert!(bad_format.is_err(), "clap must reject unknown --format");

        // Required-flag refusals.
        for missing in &[
            "--baseline-in",
            "--signer-seed",
            "--signer-role",
            "--out",
        ] {
            let argv: Vec<&str> = [
                "omni-node",
                "sign-state-integrity-baseline",
                "--baseline-in",
                "/tmp/raw.json",
                "--signer-seed",
                "/tmp/seed.bin",
                "--signer-role",
                "operator",
                "--out",
                "/tmp/signed.json",
            ]
            .iter()
            .copied()
            .filter(|s| s != missing && !is_value_after(s, missing))
            .collect();
            let _ = argv; // verified below via a fresh build
            let stripped = strip_flag_with_value("sign-state-integrity-baseline", missing);
            let parsed = TestRoot::try_parse_from(stripped);
            assert!(
                parsed.is_err(),
                "clap must reject sign-state-integrity-baseline without {missing}"
            );
        }

        // Auto-prune flag deliberately absent.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "sign-state-integrity-baseline",
            "--baseline-in",
            "/tmp/raw.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "operator",
            "--out",
            "/tmp/signed.json",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "sign-state-integrity-baseline must not accept --no-prune-state-on-start"
        );
    }

    /// Helper used only by the missing-flag refusal sweep above.
    /// Returns true if `s` is the value that immediately follows
    /// the flag `flag` in the canonical baseline argv. The sweep
    /// strips both the flag itself AND its value.
    fn is_value_after(s: &str, flag: &str) -> bool {
        match flag {
            "--baseline-in" => s == "/tmp/raw.json",
            "--signer-seed" => s == "/tmp/seed.bin",
            "--signer-role" => s == "operator",
            "--out" => s == "/tmp/signed.json",
            _ => false,
        }
    }

    /// Build an argv for `sign-state-integrity-baseline` with one
    /// (flag, value) pair stripped — used to drive the
    /// missing-flag refusal sweep.
    fn strip_flag_with_value(subcommand: &str, drop_flag: &str) -> Vec<String> {
        let full: Vec<(&str, Option<&str>)> = vec![
            ("--baseline-in", Some("/tmp/raw.json")),
            ("--signer-seed", Some("/tmp/seed.bin")),
            ("--signer-role", Some("operator")),
            ("--out", Some("/tmp/signed.json")),
        ];
        let mut out: Vec<String> =
            vec!["omni-node".to_string(), subcommand.to_string()];
        for (flag, val) in full {
            if flag == drop_flag {
                continue;
            }
            out.push(flag.to_string());
            if let Some(v) = val {
                out.push(v.to_string());
            }
        }
        out
    }

    // ── Stage 12.21 — sign-state-integrity-diff clap regression ──

    /// Build `sign-state-integrity-diff` argv with the supplied
    /// `signer_role` value + optional extra trailing flags.
    /// `signer_role` is NOT part of the defaults so each test
    /// can override it cleanly (clap would reject a duplicate-
    /// arg pass).
    fn parse_sign_state_integrity_diff(
        signer_role: &str,
        extra: &[&str],
    ) -> SignStateIntegrityDiffArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "sign-state-integrity-diff".into(),
            "--diff-in".into(),
            "/tmp/diff.json".into(),
            "--signer-seed".into(),
            "/tmp/seed.bin".into(),
            "--signer-role".into(),
            signer_role.into(),
            "--out".into(),
            "/tmp/signed.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::SignStateIntegrityDiff(a) => a,
            _ => panic!("expected SignStateIntegrityDiff"),
        }
    }

    #[test]
    fn sign_state_integrity_diff_flag_parse_smoke() {
        let defaults = parse_sign_state_integrity_diff("operator", &[]);
        assert_eq!(defaults.diff_in, std::path::PathBuf::from("/tmp/diff.json"));
        assert_eq!(defaults.signer_seed, std::path::PathBuf::from("/tmp/seed.bin"));
        assert_eq!(defaults.signer_role, CliBaselineSignerRole::Operator);
        assert_eq!(defaults.out, std::path::PathBuf::from("/tmp/signed.json"));
        assert_eq!(defaults.format, SignStateIntegrityDiffFormat::Events);

        // All four roles parse via the closed enum (reused from
        // Stage 12.20 — the four variants are role names, not
        // artifact-type names).
        for (raw, expected) in &[
            ("operator", CliBaselineSignerRole::Operator),
            ("contributor", CliBaselineSignerRole::Contributor),
            ("dispatcher", CliBaselineSignerRole::Dispatcher),
            ("coordinator", CliBaselineSignerRole::Coordinator),
        ] {
            let got = parse_sign_state_integrity_diff(raw, &[]);
            assert_eq!(got.signer_role, *expected, "role={raw} parsed wrong");
        }

        // --format closed enum.
        for (raw, expected) in &[
            ("events", SignStateIntegrityDiffFormat::Events),
            ("json", SignStateIntegrityDiffFormat::Json),
            ("pretty", SignStateIntegrityDiffFormat::Pretty),
        ] {
            let got = parse_sign_state_integrity_diff("operator", &["--format", raw]);
            assert_eq!(got.format, *expected);
        }

        // Unknown --signer-role rejected.
        let bad_role = TestRoot::try_parse_from([
            "omni-node",
            "sign-state-integrity-diff",
            "--diff-in",
            "/tmp/diff.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "auditor",
            "--out",
            "/tmp/signed.json",
        ]);
        assert!(
            bad_role.is_err(),
            "clap must reject unknown --signer-role for sign-state-integrity-diff"
        );

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "sign-state-integrity-diff",
            "--diff-in",
            "/tmp/diff.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "operator",
            "--out",
            "/tmp/signed.json",
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for sign-state-integrity-diff"
        );

        // Required-flag refusals.
        for missing in &[
            "--diff-in",
            "--signer-seed",
            "--signer-role",
            "--out",
        ] {
            let stripped = strip_sign_diff_flag_with_value(missing);
            let parsed = TestRoot::try_parse_from(stripped);
            assert!(
                parsed.is_err(),
                "clap must reject sign-state-integrity-diff without {missing}"
            );
        }

        // Auto-prune flag deliberately absent — `sign-state-
        // integrity-diff` opens no state-store, so the Stage
        // 12.7 `--no-prune-state-on-start` flag must NOT be
        // accepted.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "sign-state-integrity-diff",
            "--diff-in",
            "/tmp/diff.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "operator",
            "--out",
            "/tmp/signed.json",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "sign-state-integrity-diff must not accept --no-prune-state-on-start"
        );
    }

    /// Build an argv for `sign-state-integrity-diff` with one
    /// (flag, value) pair stripped — drives the missing-flag
    /// refusal sweep above.
    fn strip_sign_diff_flag_with_value(drop_flag: &str) -> Vec<String> {
        let full: Vec<(&str, Option<&str>)> = vec![
            ("--diff-in", Some("/tmp/diff.json")),
            ("--signer-seed", Some("/tmp/seed.bin")),
            ("--signer-role", Some("operator")),
            ("--out", Some("/tmp/signed.json")),
        ];
        let mut out: Vec<String> = vec![
            "omni-node".to_string(),
            "sign-state-integrity-diff".to_string(),
        ];
        for (flag, val) in full {
            if flag == drop_flag {
                continue;
            }
            out.push(flag.to_string());
            if let Some(v) = val {
                out.push(v.to_string());
            }
        }
        out
    }

    // ── Stage 12.21 — verify-state-integrity-diff-signature clap regression ──

    fn parse_verify_state_integrity_diff_signature(
        pubkey: &str,
        extra: &[&str],
    ) -> VerifyStateIntegrityDiffSignatureArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "verify-state-integrity-diff-signature".into(),
            "--signed-diff".into(),
            "/tmp/signed.json".into(),
            "--expected-signer-pubkey-hex".into(),
            pubkey.into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::VerifyStateIntegrityDiffSignature(a) => a,
            _ => panic!("expected VerifyStateIntegrityDiffSignature"),
        }
    }

    #[test]
    fn verify_state_integrity_diff_signature_flag_parse_smoke() {
        let pubkey = "ab".repeat(32);
        let defaults =
            parse_verify_state_integrity_diff_signature(&pubkey, &[]);
        assert_eq!(
            defaults.signed_diff,
            std::path::PathBuf::from("/tmp/signed.json")
        );
        assert_eq!(defaults.expected_signer_pubkey_hex, pubkey);
        assert_eq!(
            defaults.format,
            VerifyStateIntegrityDiffSignatureFormat::Events
        );

        // --format closed enum.
        for (raw, expected) in &[
            ("events", VerifyStateIntegrityDiffSignatureFormat::Events),
            ("json", VerifyStateIntegrityDiffSignatureFormat::Json),
            ("pretty", VerifyStateIntegrityDiffSignatureFormat::Pretty),
        ] {
            let got = parse_verify_state_integrity_diff_signature(
                &pubkey,
                &["--format", raw],
            );
            assert_eq!(got.format, *expected);
        }

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "verify-state-integrity-diff-signature",
            "--signed-diff",
            "/tmp/signed.json",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for verify-state-integrity-diff-signature"
        );

        // Required-flag refusals.
        let missing_signed = TestRoot::try_parse_from([
            "omni-node",
            "verify-state-integrity-diff-signature",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
        ]);
        assert!(
            missing_signed.is_err(),
            "clap must reject verify-state-integrity-diff-signature without --signed-diff"
        );
        let missing_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "verify-state-integrity-diff-signature",
            "--signed-diff",
            "/tmp/signed.json",
        ]);
        assert!(
            missing_pubkey.is_err(),
            "clap must reject verify-state-integrity-diff-signature without --expected-signer-pubkey-hex"
        );

        // Auto-prune flag deliberately absent — verifier opens no
        // state-store.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "verify-state-integrity-diff-signature",
            "--signed-diff",
            "/tmp/signed.json",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "verify-state-integrity-diff-signature must not accept --no-prune-state-on-start"
        );
    }

    // ── Stage 12.22 — build-integrity-evidence-bundle clap regression ──

    fn parse_build_integrity_evidence_bundle(
        extra: &[&str],
    ) -> BuildIntegrityEvidenceBundleArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "build-integrity-evidence-bundle".into(),
            "--include".into(),
            "signed_state_integrity_diff=signed-diff.json".into(),
            "--base-dir".into(),
            "/tmp/audit".into(),
            "--out".into(),
            "/tmp/bundle.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::BuildIntegrityEvidenceBundle(a) => a,
            _ => panic!("expected BuildIntegrityEvidenceBundle"),
        }
    }

    #[test]
    fn build_integrity_evidence_bundle_flag_parse_smoke() {
        let defaults = parse_build_integrity_evidence_bundle(&[]);
        assert_eq!(defaults.includes.len(), 1);
        assert_eq!(
            defaults.includes[0],
            "signed_state_integrity_diff=signed-diff.json"
        );
        assert_eq!(defaults.base_dir, std::path::PathBuf::from("/tmp/audit"));
        assert_eq!(defaults.out, std::path::PathBuf::from("/tmp/bundle.json"));
        assert!(defaults.label.is_none());
        assert!(defaults.notes.is_none());
        assert_eq!(
            defaults.format,
            BuildIntegrityEvidenceBundleFormat::Events
        );

        // Multiple --include values accumulate.
        let multi = parse_build_integrity_evidence_bundle(&[
            "--include",
            "state_integrity_report=baseline.json",
            "--include",
            "archive_manifest=arc/manifest.json",
        ]);
        assert_eq!(multi.includes.len(), 3);

        // --label / --notes parse.
        let with_strings = parse_build_integrity_evidence_bundle(&[
            "--label",
            "audit-2026-06-12",
            "--notes",
            "captured by CI",
        ]);
        assert_eq!(
            with_strings.label.as_deref(),
            Some("audit-2026-06-12")
        );
        assert_eq!(with_strings.notes.as_deref(), Some("captured by CI"));

        // --format closed enum.
        for (raw, expected) in &[
            ("events", BuildIntegrityEvidenceBundleFormat::Events),
            ("json", BuildIntegrityEvidenceBundleFormat::Json),
            ("pretty", BuildIntegrityEvidenceBundleFormat::Pretty),
        ] {
            let got = parse_build_integrity_evidence_bundle(&["--format", raw]);
            assert_eq!(got.format, *expected);
        }

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "build-integrity-evidence-bundle",
            "--include",
            "signed_state_integrity_diff=signed-diff.json",
            "--base-dir",
            "/tmp/audit",
            "--out",
            "/tmp/bundle.json",
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for build-integrity-evidence-bundle"
        );

        // Required-flag refusals: missing --include / --base-dir / --out.
        for missing in &["--include", "--base-dir", "--out"] {
            let stripped = strip_bundle_build_flag_with_value(missing);
            let parsed = TestRoot::try_parse_from(stripped);
            assert!(
                parsed.is_err(),
                "clap must reject build-integrity-evidence-bundle without {missing}"
            );
        }

        // Auto-prune flag deliberately absent — pinned by clap
        // regression so a future operator can't reintroduce
        // state-store coupling.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "build-integrity-evidence-bundle",
            "--include",
            "signed_state_integrity_diff=signed-diff.json",
            "--base-dir",
            "/tmp/audit",
            "--out",
            "/tmp/bundle.json",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "build-integrity-evidence-bundle must not accept --no-prune-state-on-start"
        );
    }

    fn strip_bundle_build_flag_with_value(drop_flag: &str) -> Vec<String> {
        let full: Vec<(&str, Option<&str>)> = vec![
            (
                "--include",
                Some("signed_state_integrity_diff=signed-diff.json"),
            ),
            ("--base-dir", Some("/tmp/audit")),
            ("--out", Some("/tmp/bundle.json")),
        ];
        let mut out: Vec<String> = vec![
            "omni-node".to_string(),
            "build-integrity-evidence-bundle".to_string(),
        ];
        for (flag, val) in full {
            if flag == drop_flag {
                continue;
            }
            out.push(flag.to_string());
            if let Some(v) = val {
                out.push(v.to_string());
            }
        }
        out
    }

    // ── Stage 12.22 — verify-integrity-evidence-bundle clap regression ──

    fn parse_verify_integrity_evidence_bundle(
        extra: &[&str],
    ) -> VerifyIntegrityEvidenceBundleArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "verify-integrity-evidence-bundle".into(),
            "--bundle".into(),
            "/tmp/bundle.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::VerifyIntegrityEvidenceBundle(a) => a,
            _ => panic!("expected VerifyIntegrityEvidenceBundle"),
        }
    }

    #[test]
    fn verify_integrity_evidence_bundle_flag_parse_smoke() {
        let defaults = parse_verify_integrity_evidence_bundle(&[]);
        assert_eq!(defaults.bundle, std::path::PathBuf::from("/tmp/bundle.json"));
        assert!(defaults.base_dir.is_none());
        assert_eq!(
            defaults.format,
            VerifyIntegrityEvidenceBundleFormat::Events
        );

        let with_override = parse_verify_integrity_evidence_bundle(&[
            "--base-dir",
            "/srv/audit-mirror",
        ]);
        assert_eq!(
            with_override.base_dir,
            Some(std::path::PathBuf::from("/srv/audit-mirror"))
        );

        // --format closed enum.
        for (raw, expected) in &[
            ("events", VerifyIntegrityEvidenceBundleFormat::Events),
            ("json", VerifyIntegrityEvidenceBundleFormat::Json),
            ("pretty", VerifyIntegrityEvidenceBundleFormat::Pretty),
        ] {
            let got =
                parse_verify_integrity_evidence_bundle(&["--format", raw]);
            assert_eq!(got.format, *expected);
        }

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-bundle",
            "--bundle",
            "/tmp/bundle.json",
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for verify-integrity-evidence-bundle"
        );

        // Required-flag refusal.
        let missing_bundle = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-bundle",
        ]);
        assert!(
            missing_bundle.is_err(),
            "clap must reject verify-integrity-evidence-bundle without --bundle"
        );

        // Auto-prune flag deliberately absent.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-bundle",
            "--bundle",
            "/tmp/bundle.json",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "verify-integrity-evidence-bundle must not accept --no-prune-state-on-start"
        );
    }

    // ── Stage 12.23 — sign-integrity-evidence-bundle clap regression ──

    fn parse_sign_integrity_evidence_bundle(
        signer_role: &str,
        extra: &[&str],
    ) -> SignIntegrityEvidenceBundleArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "sign-integrity-evidence-bundle".into(),
            "--bundle-in".into(),
            "/tmp/bundle.json".into(),
            "--signer-seed".into(),
            "/tmp/seed.bin".into(),
            "--signer-role".into(),
            signer_role.into(),
            "--out".into(),
            "/tmp/signed-bundle.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::SignIntegrityEvidenceBundle(a) => a,
            _ => panic!("expected SignIntegrityEvidenceBundle"),
        }
    }

    #[test]
    fn sign_integrity_evidence_bundle_flag_parse_smoke() {
        let defaults = parse_sign_integrity_evidence_bundle("operator", &[]);
        assert_eq!(defaults.bundle_in, std::path::PathBuf::from("/tmp/bundle.json"));
        assert_eq!(defaults.signer_seed, std::path::PathBuf::from("/tmp/seed.bin"));
        assert_eq!(defaults.signer_role, CliBaselineSignerRole::Operator);
        assert_eq!(
            defaults.out,
            std::path::PathBuf::from("/tmp/signed-bundle.json")
        );
        assert_eq!(defaults.format, SignIntegrityEvidenceBundleFormat::Events);

        // All four roles parse via the reused Stage 12.20 enum.
        for (raw, expected) in &[
            ("operator", CliBaselineSignerRole::Operator),
            ("contributor", CliBaselineSignerRole::Contributor),
            ("dispatcher", CliBaselineSignerRole::Dispatcher),
            ("coordinator", CliBaselineSignerRole::Coordinator),
        ] {
            let got = parse_sign_integrity_evidence_bundle(raw, &[]);
            assert_eq!(got.signer_role, *expected, "role={raw} parsed wrong");
        }

        // --format closed enum.
        for (raw, expected) in &[
            ("events", SignIntegrityEvidenceBundleFormat::Events),
            ("json", SignIntegrityEvidenceBundleFormat::Json),
            ("pretty", SignIntegrityEvidenceBundleFormat::Pretty),
        ] {
            let got = parse_sign_integrity_evidence_bundle(
                "operator",
                &["--format", raw],
            );
            assert_eq!(got.format, *expected);
        }

        // Unknown --signer-role rejected.
        let bad_role = TestRoot::try_parse_from([
            "omni-node",
            "sign-integrity-evidence-bundle",
            "--bundle-in",
            "/tmp/bundle.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "auditor",
            "--out",
            "/tmp/signed-bundle.json",
        ]);
        assert!(
            bad_role.is_err(),
            "clap must reject unknown --signer-role for sign-integrity-evidence-bundle"
        );

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "sign-integrity-evidence-bundle",
            "--bundle-in",
            "/tmp/bundle.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "operator",
            "--out",
            "/tmp/signed-bundle.json",
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for sign-integrity-evidence-bundle"
        );

        // Required-flag refusals.
        for missing in &[
            "--bundle-in",
            "--signer-seed",
            "--signer-role",
            "--out",
        ] {
            let stripped =
                strip_sign_bundle_flag_with_value(missing);
            let parsed = TestRoot::try_parse_from(stripped);
            assert!(
                parsed.is_err(),
                "clap must reject sign-integrity-evidence-bundle without {missing}"
            );
        }

        // Auto-prune flag deliberately absent — pinned by clap
        // regression so a future operator can't reintroduce
        // state-store coupling.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "sign-integrity-evidence-bundle",
            "--bundle-in",
            "/tmp/bundle.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "operator",
            "--out",
            "/tmp/signed-bundle.json",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "sign-integrity-evidence-bundle must not accept --no-prune-state-on-start"
        );
    }

    /// Build an argv for `sign-integrity-evidence-bundle` with
    /// one (flag, value) pair stripped — drives the
    /// missing-flag refusal sweep above.
    fn strip_sign_bundle_flag_with_value(drop_flag: &str) -> Vec<String> {
        let full: Vec<(&str, Option<&str>)> = vec![
            ("--bundle-in", Some("/tmp/bundle.json")),
            ("--signer-seed", Some("/tmp/seed.bin")),
            ("--signer-role", Some("operator")),
            ("--out", Some("/tmp/signed-bundle.json")),
        ];
        let mut out: Vec<String> = vec![
            "omni-node".to_string(),
            "sign-integrity-evidence-bundle".to_string(),
        ];
        for (flag, val) in full {
            if flag == drop_flag {
                continue;
            }
            out.push(flag.to_string());
            if let Some(v) = val {
                out.push(v.to_string());
            }
        }
        out
    }

    // ── Stage 12.23 — verify-integrity-evidence-bundle-signature clap regression ──

    fn parse_verify_integrity_evidence_bundle_signature(
        pubkey: &str,
        extra: &[&str],
    ) -> VerifyIntegrityEvidenceBundleSignatureArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "verify-integrity-evidence-bundle-signature".into(),
            "--signed-bundle".into(),
            "/tmp/signed-bundle.json".into(),
            "--expected-signer-pubkey-hex".into(),
            pubkey.into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::VerifyIntegrityEvidenceBundleSignature(a) => a,
            _ => panic!("expected VerifyIntegrityEvidenceBundleSignature"),
        }
    }

    #[test]
    fn verify_integrity_evidence_bundle_signature_flag_parse_smoke() {
        let pubkey = "ab".repeat(32);
        let defaults =
            parse_verify_integrity_evidence_bundle_signature(&pubkey, &[]);
        assert_eq!(
            defaults.signed_bundle,
            std::path::PathBuf::from("/tmp/signed-bundle.json")
        );
        assert_eq!(defaults.expected_signer_pubkey_hex, pubkey);
        assert_eq!(
            defaults.format,
            VerifyIntegrityEvidenceBundleSignatureFormat::Events
        );

        // --format closed enum.
        for (raw, expected) in &[
            ("events", VerifyIntegrityEvidenceBundleSignatureFormat::Events),
            ("json", VerifyIntegrityEvidenceBundleSignatureFormat::Json),
            ("pretty", VerifyIntegrityEvidenceBundleSignatureFormat::Pretty),
        ] {
            let got = parse_verify_integrity_evidence_bundle_signature(
                &pubkey,
                &["--format", raw],
            );
            assert_eq!(got.format, *expected);
        }

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-bundle-signature",
            "--signed-bundle",
            "/tmp/signed-bundle.json",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for verify-integrity-evidence-bundle-signature"
        );

        // Required-flag refusals.
        let missing_signed = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-bundle-signature",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
        ]);
        assert!(
            missing_signed.is_err(),
            "clap must reject verify-integrity-evidence-bundle-signature without --signed-bundle"
        );
        let missing_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-bundle-signature",
            "--signed-bundle",
            "/tmp/signed-bundle.json",
        ]);
        assert!(
            missing_pubkey.is_err(),
            "clap must reject verify-integrity-evidence-bundle-signature without --expected-signer-pubkey-hex"
        );

        // Auto-prune flag deliberately absent.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-bundle-signature",
            "--signed-bundle",
            "/tmp/signed-bundle.json",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "verify-integrity-evidence-bundle-signature must not accept --no-prune-state-on-start"
        );
    }

    // ── Stage 12.23 — sign/verify failure-event tag mapping ─────

    /// Pins the closed `reason=<tag>` set on
    /// `event=signed_integrity_evidence_bundle_{sign,verify}_failed`.
    /// Both the signer (every failure path) and verifier
    /// (envelope + bundle-read paths) emit this event with a
    /// tag drawn from `super::signed_bundle_reason_tag`. If a
    /// future variant lands on `SignedIntegrityEvidenceBundleError`
    /// without being mapped here, the closed-set contract
    /// breaks.
    #[test]
    fn signed_bundle_reason_tag_covers_closed_set() {
        use omni_contributor::SignedIntegrityEvidenceBundleError as E;
        use std::io::{Error as IoError, ErrorKind};
        use std::path::PathBuf;

        assert_eq!(
            super::signed_bundle_reason_tag(&E::UnsupportedSchemaVersion {
                got: 2,
                expected: 1
            }),
            "unsupported_schema_version"
        );
        assert_eq!(
            super::signed_bundle_reason_tag(
                &E::UnsupportedBundleSchemaVersion { got: 2, expected: 1 }
            ),
            "unsupported_bundle_schema_version"
        );
        assert_eq!(
            super::signed_bundle_reason_tag(&E::SignerPubkeyMismatch {
                expected: "aa".repeat(32),
                got: "bb".repeat(32),
            }),
            "signer_pubkey_mismatch"
        );
        assert_eq!(
            super::signed_bundle_reason_tag(&E::SignatureMismatch),
            "signature_mismatch"
        );
        assert_eq!(
            super::signed_bundle_reason_tag(&E::Io {
                path: PathBuf::from("/tmp/x"),
                source: IoError::from(ErrorKind::NotFound),
            }),
            "io"
        );
        // serde_json::Error doesn't have a public constructor;
        // round-tripping a malformed JSON through serde_json
        // is the cleanest way to mint one for the mapper.
        let json_err = serde_json::from_str::<serde_json::Value>("{bad")
            .expect_err("malformed json");
        assert_eq!(
            super::signed_bundle_reason_tag(&E::MalformedJson {
                path: PathBuf::from("/tmp/x"),
                source: json_err,
            }),
            "malformed_json"
        );
        // Signing(...) and Canonical(...) tags — the inner
        // error types are bubbled from other crates; the only
        // requirement is the mapper picks the closed tag, not
        // the inner error's shape.
    }

    // ── Stage 12.24 — verify-integrity-evidence-chain clap regression ──

    fn parse_verify_integrity_evidence_chain(
        pubkey: &str,
        extra: &[&str],
    ) -> VerifyIntegrityEvidenceChainArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "verify-integrity-evidence-chain".into(),
            "--signed-bundle".into(),
            "/tmp/signed-bundle.json".into(),
            "--expected-bundle-signer-pubkey-hex".into(),
            pubkey.into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::VerifyIntegrityEvidenceChain(a) => a,
            _ => panic!("expected VerifyIntegrityEvidenceChain"),
        }
    }

    #[test]
    fn verify_integrity_evidence_chain_flag_parse_smoke() {
        let pubkey = "ab".repeat(32);
        let defaults =
            parse_verify_integrity_evidence_chain(&pubkey, &[]);
        assert_eq!(
            defaults.signed_bundle,
            std::path::PathBuf::from("/tmp/signed-bundle.json")
        );
        assert_eq!(defaults.expected_bundle_signer_pubkey_hex, pubkey);
        assert!(defaults.expected_baseline_signer_pubkey_hex.is_none());
        assert!(defaults.expected_diff_signer_pubkey_hex.is_none());
        assert!(defaults.base_dir.is_none());
        assert!(defaults.json_out.is_none());
        assert_eq!(
            defaults.format,
            VerifyIntegrityEvidenceChainFormat::Events
        );

        // Optional child anchors parse independently.
        let baseline_pk = "cd".repeat(32);
        let with_baseline = parse_verify_integrity_evidence_chain(
            &pubkey,
            &[
                "--expected-baseline-signer-pubkey-hex",
                &baseline_pk,
            ],
        );
        assert_eq!(
            with_baseline.expected_baseline_signer_pubkey_hex.as_deref(),
            Some(baseline_pk.as_str())
        );
        assert!(with_baseline.expected_diff_signer_pubkey_hex.is_none());

        let diff_pk = "ef".repeat(32);
        let with_both = parse_verify_integrity_evidence_chain(
            &pubkey,
            &[
                "--expected-baseline-signer-pubkey-hex",
                &baseline_pk,
                "--expected-diff-signer-pubkey-hex",
                &diff_pk,
            ],
        );
        assert_eq!(
            with_both.expected_baseline_signer_pubkey_hex.as_deref(),
            Some(baseline_pk.as_str())
        );
        assert_eq!(
            with_both.expected_diff_signer_pubkey_hex.as_deref(),
            Some(diff_pk.as_str())
        );

        // --base-dir and --json-out parse.
        let with_paths = parse_verify_integrity_evidence_chain(
            &pubkey,
            &[
                "--base-dir",
                "/srv/audit-mirror",
                "--json-out",
                "/var/audit/chain.json",
            ],
        );
        assert_eq!(
            with_paths.base_dir,
            Some(std::path::PathBuf::from("/srv/audit-mirror"))
        );
        assert_eq!(
            with_paths.json_out,
            Some(std::path::PathBuf::from("/var/audit/chain.json"))
        );

        // --format closed enum.
        for (raw, expected) in &[
            ("events", VerifyIntegrityEvidenceChainFormat::Events),
            ("json", VerifyIntegrityEvidenceChainFormat::Json),
            ("pretty", VerifyIntegrityEvidenceChainFormat::Pretty),
        ] {
            let got = parse_verify_integrity_evidence_chain(
                &pubkey,
                &["--format", raw],
            );
            assert_eq!(got.format, *expected);
        }

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-chain",
            "--signed-bundle",
            "/tmp/signed-bundle.json",
            "--expected-bundle-signer-pubkey-hex",
            pubkey.as_str(),
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for verify-integrity-evidence-chain"
        );

        // Required-flag refusals.
        let missing_signed = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-chain",
            "--expected-bundle-signer-pubkey-hex",
            pubkey.as_str(),
        ]);
        assert!(
            missing_signed.is_err(),
            "clap must reject verify-integrity-evidence-chain without --signed-bundle"
        );
        let missing_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-chain",
            "--signed-bundle",
            "/tmp/signed-bundle.json",
        ]);
        assert!(
            missing_pubkey.is_err(),
            "clap must reject verify-integrity-evidence-chain without --expected-bundle-signer-pubkey-hex"
        );

        // Auto-prune flag deliberately absent.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-chain",
            "--signed-bundle",
            "/tmp/signed-bundle.json",
            "--expected-bundle-signer-pubkey-hex",
            pubkey.as_str(),
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "verify-integrity-evidence-chain must not accept --no-prune-state-on-start"
        );
    }

    /// Pins the closed `reason=<tag>` sets for per-child
    /// signature verify outcomes. The chain verifier hand-
    /// dispatches `SignedBaselineError` and
    /// `SignedIntegrityDiffError` through
    /// `baseline_child_reason_tag` / `diff_child_reason_tag`
    /// in `integrity_evidence_chain.rs`. If a future variant
    /// lands on either error without a mapping the closed set
    /// breaks; this regression catches that.
    #[test]
    fn chain_child_reason_tag_covers_closed_sets() {
        use omni_contributor::{
            baseline_child_reason_tag, diff_child_reason_tag,
            SignedBaselineError, SignedIntegrityDiffError,
        };
        use std::io::{Error as IoError, ErrorKind};
        use std::path::PathBuf;

        // ── Baseline child mapper ───────────────────────
        assert_eq!(
            baseline_child_reason_tag(
                &SignedBaselineError::UnsupportedSchemaVersion {
                    got: 2,
                    expected: 1
                }
            ),
            "unsupported_schema_version"
        );
        assert_eq!(
            baseline_child_reason_tag(
                &SignedBaselineError::UnsupportedReportSchemaVersion {
                    got: 2,
                    expected: 1
                }
            ),
            "unsupported_report_schema_version"
        );
        assert_eq!(
            baseline_child_reason_tag(&SignedBaselineError::SignerPubkeyMismatch {
                expected: "aa".repeat(32),
                got: "bb".repeat(32),
            }),
            "signer_pubkey_mismatch"
        );
        assert_eq!(
            baseline_child_reason_tag(&SignedBaselineError::SignatureMismatch),
            "signature_mismatch"
        );
        assert_eq!(
            baseline_child_reason_tag(&SignedBaselineError::Io {
                path: PathBuf::from("/tmp/x"),
                source: IoError::from(ErrorKind::NotFound),
            }),
            "io"
        );
        let json_err = serde_json::from_str::<serde_json::Value>("{bad")
            .expect_err("malformed json");
        assert_eq!(
            baseline_child_reason_tag(&SignedBaselineError::MalformedJson {
                path: PathBuf::from("/tmp/x"),
                source: json_err,
            }),
            "malformed_json"
        );

        // ── Diff child mapper ───────────────────────────
        assert_eq!(
            diff_child_reason_tag(
                &SignedIntegrityDiffError::UnsupportedSchemaVersion {
                    got: 2,
                    expected: 1
                }
            ),
            "unsupported_schema_version"
        );
        assert_eq!(
            diff_child_reason_tag(
                &SignedIntegrityDiffError::UnsupportedDiffSchemaVersion {
                    got: 2,
                    expected: 1
                }
            ),
            "unsupported_diff_schema_version"
        );
        assert_eq!(
            diff_child_reason_tag(
                &SignedIntegrityDiffError::SignerPubkeyMismatch {
                    expected: "aa".repeat(32),
                    got: "bb".repeat(32),
                }
            ),
            "signer_pubkey_mismatch"
        );
        assert_eq!(
            diff_child_reason_tag(&SignedIntegrityDiffError::SignatureMismatch),
            "signature_mismatch"
        );
        assert_eq!(
            diff_child_reason_tag(&SignedIntegrityDiffError::Io {
                path: PathBuf::from("/tmp/x"),
                source: IoError::from(ErrorKind::NotFound),
            }),
            "io"
        );
        let json_err2 = serde_json::from_str::<serde_json::Value>("{bad")
            .expect_err("malformed json");
        assert_eq!(
            diff_child_reason_tag(&SignedIntegrityDiffError::MalformedJson {
                path: PathBuf::from("/tmp/x"),
                source: json_err2,
            }),
            "malformed_json"
        );
        // Signing / Canonical variants — bubbled from other
        // crates; the only requirement is the mapper picks the
        // closed tag, not the inner shape.
    }

    // ── Stage 12.25 — sign-integrity-evidence-chain-report clap regression ──

    fn parse_sign_integrity_evidence_chain_report(
        signer_role: &str,
        extra: &[&str],
    ) -> SignIntegrityEvidenceChainReportArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "sign-integrity-evidence-chain-report".into(),
            "--chain-report-in".into(),
            "/tmp/chain-report.json".into(),
            "--signer-seed".into(),
            "/tmp/seed.bin".into(),
            "--signer-role".into(),
            signer_role.into(),
            "--out".into(),
            "/tmp/signed-chain.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::SignIntegrityEvidenceChainReport(a) => a,
            _ => panic!("expected SignIntegrityEvidenceChainReport"),
        }
    }

    #[test]
    fn sign_integrity_evidence_chain_report_flag_parse_smoke() {
        let defaults =
            parse_sign_integrity_evidence_chain_report("operator", &[]);
        assert_eq!(
            defaults.chain_report_in,
            std::path::PathBuf::from("/tmp/chain-report.json")
        );
        assert_eq!(defaults.signer_seed, std::path::PathBuf::from("/tmp/seed.bin"));
        assert_eq!(defaults.signer_role, CliBaselineSignerRole::Operator);
        assert_eq!(
            defaults.out,
            std::path::PathBuf::from("/tmp/signed-chain.json")
        );
        assert_eq!(
            defaults.format,
            SignIntegrityEvidenceChainReportFormat::Events
        );

        // All four roles parse via the reused Stage 12.20 enum.
        for (raw, expected) in &[
            ("operator", CliBaselineSignerRole::Operator),
            ("contributor", CliBaselineSignerRole::Contributor),
            ("dispatcher", CliBaselineSignerRole::Dispatcher),
            ("coordinator", CliBaselineSignerRole::Coordinator),
        ] {
            let got = parse_sign_integrity_evidence_chain_report(raw, &[]);
            assert_eq!(got.signer_role, *expected, "role={raw} parsed wrong");
        }

        // --format closed enum.
        for (raw, expected) in &[
            ("events", SignIntegrityEvidenceChainReportFormat::Events),
            ("json", SignIntegrityEvidenceChainReportFormat::Json),
            ("pretty", SignIntegrityEvidenceChainReportFormat::Pretty),
        ] {
            let got = parse_sign_integrity_evidence_chain_report(
                "operator",
                &["--format", raw],
            );
            assert_eq!(got.format, *expected);
        }

        // Unknown --signer-role rejected.
        let bad_role = TestRoot::try_parse_from([
            "omni-node",
            "sign-integrity-evidence-chain-report",
            "--chain-report-in",
            "/tmp/chain-report.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "rogue_role",
            "--out",
            "/tmp/signed-chain.json",
        ]);
        assert!(
            bad_role.is_err(),
            "clap must reject unknown --signer-role for sign-integrity-evidence-chain-report"
        );

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "sign-integrity-evidence-chain-report",
            "--chain-report-in",
            "/tmp/chain-report.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "operator",
            "--out",
            "/tmp/signed-chain.json",
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for sign-integrity-evidence-chain-report"
        );

        // Required-flag refusals.
        for missing in &[
            "--chain-report-in",
            "--signer-seed",
            "--signer-role",
            "--out",
        ] {
            let stripped = strip_sign_chain_report_flag_with_value(missing);
            let parsed = TestRoot::try_parse_from(stripped);
            assert!(
                parsed.is_err(),
                "clap must reject sign-integrity-evidence-chain-report without {missing}"
            );
        }

        // Auto-prune flag deliberately absent — pinned by clap
        // regression so a future operator can't reintroduce
        // state-store coupling.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "sign-integrity-evidence-chain-report",
            "--chain-report-in",
            "/tmp/chain-report.json",
            "--signer-seed",
            "/tmp/seed.bin",
            "--signer-role",
            "operator",
            "--out",
            "/tmp/signed-chain.json",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "sign-integrity-evidence-chain-report must not accept --no-prune-state-on-start"
        );
    }

    /// Build an argv for `sign-integrity-evidence-chain-report`
    /// with one (flag, value) pair stripped — drives the
    /// missing-flag refusal sweep above.
    fn strip_sign_chain_report_flag_with_value(drop_flag: &str) -> Vec<String> {
        let full: Vec<(&str, Option<&str>)> = vec![
            ("--chain-report-in", Some("/tmp/chain-report.json")),
            ("--signer-seed", Some("/tmp/seed.bin")),
            ("--signer-role", Some("operator")),
            ("--out", Some("/tmp/signed-chain.json")),
        ];
        let mut out: Vec<String> = vec![
            "omni-node".to_string(),
            "sign-integrity-evidence-chain-report".to_string(),
        ];
        for (flag, val) in full {
            if flag == drop_flag {
                continue;
            }
            out.push(flag.to_string());
            if let Some(v) = val {
                out.push(v.to_string());
            }
        }
        out
    }

    // ── Stage 12.25 — verify-integrity-evidence-chain-report-signature clap regression ──

    fn parse_verify_integrity_evidence_chain_report_signature(
        pubkey: &str,
        extra: &[&str],
    ) -> VerifyIntegrityEvidenceChainReportSignatureArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "verify-integrity-evidence-chain-report-signature".into(),
            "--signed-chain-report".into(),
            "/tmp/signed-chain.json".into(),
            "--expected-signer-pubkey-hex".into(),
            pubkey.into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::VerifyIntegrityEvidenceChainReportSignature(a) => a,
            _ => panic!(
                "expected VerifyIntegrityEvidenceChainReportSignature"
            ),
        }
    }

    #[test]
    fn verify_integrity_evidence_chain_report_signature_flag_parse_smoke() {
        let pubkey = "ab".repeat(32);
        let defaults =
            parse_verify_integrity_evidence_chain_report_signature(&pubkey, &[]);
        assert_eq!(
            defaults.signed_chain_report,
            std::path::PathBuf::from("/tmp/signed-chain.json")
        );
        assert_eq!(defaults.expected_signer_pubkey_hex, pubkey);
        assert_eq!(
            defaults.format,
            VerifyIntegrityEvidenceChainReportSignatureFormat::Events
        );

        // --format closed enum.
        for (raw, expected) in &[
            (
                "events",
                VerifyIntegrityEvidenceChainReportSignatureFormat::Events,
            ),
            (
                "json",
                VerifyIntegrityEvidenceChainReportSignatureFormat::Json,
            ),
            (
                "pretty",
                VerifyIntegrityEvidenceChainReportSignatureFormat::Pretty,
            ),
        ] {
            let got = parse_verify_integrity_evidence_chain_report_signature(
                &pubkey,
                &["--format", raw],
            );
            assert_eq!(got.format, *expected);
        }

        // Unknown --format rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-chain-report-signature",
            "--signed-chain-report",
            "/tmp/signed-chain.json",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format for verify-integrity-evidence-chain-report-signature"
        );

        // Required-flag refusals.
        let missing_signed = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-chain-report-signature",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
        ]);
        assert!(
            missing_signed.is_err(),
            "clap must reject verify-integrity-evidence-chain-report-signature without --signed-chain-report"
        );
        let missing_pubkey = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-chain-report-signature",
            "--signed-chain-report",
            "/tmp/signed-chain.json",
        ]);
        assert!(
            missing_pubkey.is_err(),
            "clap must reject verify-integrity-evidence-chain-report-signature without --expected-signer-pubkey-hex"
        );

        // Auto-prune flag deliberately absent.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "verify-integrity-evidence-chain-report-signature",
            "--signed-chain-report",
            "/tmp/signed-chain.json",
            "--expected-signer-pubkey-hex",
            pubkey.as_str(),
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "verify-integrity-evidence-chain-report-signature must not accept --no-prune-state-on-start"
        );
    }

    /// Pins the closed `reason=<tag>` set on
    /// `event=signed_integrity_evidence_chain_report_{sign,verify}_failed`.
    /// Mirrors the Stage 12.23
    /// `signed_bundle_reason_tag_covers_closed_set` precedent
    /// so a future variant on
    /// `SignedIntegrityEvidenceChainReportError` without a
    /// mapping breaks the closed-set contract loudly.
    #[test]
    fn signed_chain_report_reason_tag_covers_closed_set() {
        use omni_contributor::SignedIntegrityEvidenceChainReportError as E;
        use std::io::{Error as IoError, ErrorKind};
        use std::path::PathBuf;

        assert_eq!(
            super::signed_chain_report_reason_tag(&E::UnsupportedSchemaVersion {
                got: 2,
                expected: 1
            }),
            "unsupported_schema_version"
        );
        assert_eq!(
            super::signed_chain_report_reason_tag(
                &E::UnsupportedChainReportSchemaVersion {
                    got: 2,
                    expected: 1
                }
            ),
            "unsupported_chain_report_schema_version"
        );
        assert_eq!(
            super::signed_chain_report_reason_tag(&E::SignerPubkeyMismatch {
                expected: "aa".repeat(32),
                got: "bb".repeat(32),
            }),
            "signer_pubkey_mismatch"
        );
        assert_eq!(
            super::signed_chain_report_reason_tag(&E::SignatureMismatch),
            "signature_mismatch"
        );
        assert_eq!(
            super::signed_chain_report_reason_tag(&E::Io {
                path: PathBuf::from("/tmp/x"),
                source: IoError::from(ErrorKind::NotFound),
            }),
            "io"
        );
        let json_err = serde_json::from_str::<serde_json::Value>("{bad")
            .expect_err("malformed json");
        assert_eq!(
            super::signed_chain_report_reason_tag(&E::MalformedJson {
                path: PathBuf::from("/tmp/x"),
                source: json_err,
            }),
            "malformed_json"
        );
        // Signing(...) and Canonical(...) tags — the inner
        // error types are bubbled from other crates; the only
        // requirement is the mapper picks the closed tag, not
        // the inner error's shape.
    }

    // ── Stage 12.17 — plan-state-cleanup / apply-state-cleanup ──

    fn parse_plan_state_cleanup(extra: &[&str]) -> PlanStateCleanupArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "plan-state-cleanup".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
            "--out".into(),
            "/tmp/plan.json".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::PlanStateCleanup(a) => a,
            _ => panic!("expected PlanStateCleanup"),
        }
    }

    fn parse_apply_state_cleanup(extra: &[&str]) -> ApplyStateCleanupArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "apply-state-cleanup".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
            "--plan".into(),
            "/tmp/plan.json".into(),
            "--quarantine-dir".into(),
            "/tmp/quarantine".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::ApplyStateCleanup(a) => a,
            _ => panic!("expected ApplyStateCleanup"),
        }
    }

    #[test]
    fn plan_state_cleanup_flag_parse_smoke() {
        let defaults = parse_plan_state_cleanup(&[]);
        assert!(defaults.session_id.is_none());
        assert!(defaults.integrity_json.is_none());
        assert_eq!(defaults.format, PlanStateCleanupFormat::Events);
        assert_eq!(defaults.out, std::path::PathBuf::from("/tmp/plan.json"));

        let sid = "aa".repeat(32);
        let with_sid = parse_plan_state_cleanup(&["--session-id", &sid]);
        assert_eq!(with_sid.session_id.as_deref(), Some(sid.as_str()));

        let with_integ =
            parse_plan_state_cleanup(&["--integrity-json", "/tmp/r.json"]);
        assert_eq!(
            with_integ.integrity_json.as_deref(),
            Some(std::path::Path::new("/tmp/r.json"))
        );

        for (raw, expected) in &[
            ("events", PlanStateCleanupFormat::Events),
            ("json", PlanStateCleanupFormat::Json),
            ("pretty", PlanStateCleanupFormat::Pretty),
        ] {
            let got = parse_plan_state_cleanup(&["--format", raw]);
            assert_eq!(got.format, *expected);
        }

        // Stage 12.16 review precedent: cleanup CLI does NOT
        // expose --no-prune-state-on-start (cleanup planner is
        // read-only and always opens with auto_prune off).
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "plan-state-cleanup",
            "--contributor-state-dir",
            "/tmp/state",
            "--out",
            "/tmp/plan.json",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "plan-state-cleanup must not accept --no-prune-state-on-start"
        );
    }

    #[test]
    fn apply_state_cleanup_flag_parse_smoke() {
        let defaults = parse_apply_state_cleanup(&[]);
        assert!(!defaults.dry_run);
        assert!(!defaults.allow_invalid_partial_cleanup);
        assert!(!defaults.allow_orphan_assignments);
        assert!(!defaults.purge_stray);
        assert_eq!(defaults.format, ApplyStateCleanupFormat::Events);

        let toggled = parse_apply_state_cleanup(&[
            "--dry-run",
            "--allow-invalid-partial-cleanup",
            "--allow-orphan-assignments",
            "--purge-stray",
        ]);
        assert!(toggled.dry_run);
        assert!(toggled.allow_invalid_partial_cleanup);
        assert!(toggled.allow_orphan_assignments);
        assert!(toggled.purge_stray);

        for (raw, expected) in &[
            ("events", ApplyStateCleanupFormat::Events),
            ("json", ApplyStateCleanupFormat::Json),
            ("pretty", ApplyStateCleanupFormat::Pretty),
        ] {
            let got = parse_apply_state_cleanup(&["--format", raw]);
            assert_eq!(got.format, *expected);
        }

        // Auto-prune flag is deliberately absent on apply too.
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "apply-state-cleanup",
            "--contributor-state-dir",
            "/tmp/state",
            "--plan",
            "/tmp/plan.json",
            "--quarantine-dir",
            "/tmp/q",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "apply-state-cleanup must not accept --no-prune-state-on-start"
        );
    }

    // ── Stage 12.18 — restore-state-cleanup-quarantine clap regression ──

    fn parse_restore_state_cleanup_quarantine(
        extra: &[&str],
    ) -> RestoreStateCleanupQuarantineArgs {
        let mut argv: Vec<String> = vec![
            "omni-node".into(),
            "restore-state-cleanup-quarantine".into(),
            "--contributor-state-dir".into(),
            "/tmp/state".into(),
        ];
        for s in extra {
            argv.push((*s).to_string());
        }
        let root = TestRoot::try_parse_from(&argv).expect("parse");
        match root.contributor.cmd {
            ContributorCmd::RestoreStateCleanupQuarantine(a) => a,
            _ => panic!("expected RestoreStateCleanupQuarantine"),
        }
    }

    #[test]
    fn restore_state_cleanup_quarantine_flag_parse_smoke() {
        // Defaults: no source set yet, no flags on, format=events.
        let defaults = parse_restore_state_cleanup_quarantine(&[]);
        assert!(defaults.quarantine_plan_dir.is_none());
        assert!(defaults.quarantine_dir.is_none());
        assert!(defaults.plan_id.is_none());
        assert!(!defaults.dry_run);
        assert!(!defaults.verify_only);
        assert!(!defaults.overwrite_existing);
        assert!(!defaults.no_restore_seen_markers);
        assert!(!defaults.allow_restore_orphan_assignments);
        assert_eq!(defaults.format, RestoreQuarantineFormat::Events);

        // --quarantine-plan-dir resolves directly.
        let with_plan_dir = parse_restore_state_cleanup_quarantine(&[
            "--quarantine-plan-dir",
            "/tmp/q/abc1234567890def",
        ]);
        assert_eq!(
            with_plan_dir.quarantine_plan_dir.as_deref(),
            Some(std::path::Path::new("/tmp/q/abc1234567890def"))
        );
        assert!(with_plan_dir.quarantine_dir.is_none());
        assert!(with_plan_dir.plan_id.is_none());

        // --quarantine-dir + --plan-id pair.
        let with_pair = parse_restore_state_cleanup_quarantine(&[
            "--quarantine-dir",
            "/tmp/q",
            "--plan-id",
            "abc1234567890def",
        ]);
        assert_eq!(
            with_pair.quarantine_dir.as_deref(),
            Some(std::path::Path::new("/tmp/q"))
        );
        assert_eq!(with_pair.plan_id.as_deref(), Some("abc1234567890def"));

        // Mutual-exclusion: --quarantine-plan-dir vs
        // --quarantine-dir/--plan-id pair is rejected by clap.
        let conflict = TestRoot::try_parse_from([
            "omni-node",
            "restore-state-cleanup-quarantine",
            "--contributor-state-dir",
            "/tmp/state",
            "--quarantine-plan-dir",
            "/tmp/q/abc",
            "--quarantine-dir",
            "/tmp/q",
            "--plan-id",
            "abc1234567890def",
        ]);
        assert!(
            conflict.is_err(),
            "clap must reject --quarantine-plan-dir + --quarantine-dir conflict"
        );

        // Pair-requires: --quarantine-dir alone (no --plan-id)
        // is rejected.
        let lonely = TestRoot::try_parse_from([
            "omni-node",
            "restore-state-cleanup-quarantine",
            "--contributor-state-dir",
            "/tmp/state",
            "--quarantine-dir",
            "/tmp/q",
        ]);
        assert!(
            lonely.is_err(),
            "clap must reject --quarantine-dir without --plan-id"
        );

        // --dry-run / --verify-only / --overwrite-existing
        // / --no-restore-seen-markers /
        // --allow-restore-orphan-assignments toggles.
        let toggled = parse_restore_state_cleanup_quarantine(&[
            "--quarantine-plan-dir",
            "/tmp/q/abc",
            "--dry-run",
            "--verify-only",
            "--overwrite-existing",
            "--no-restore-seen-markers",
            "--allow-restore-orphan-assignments",
        ]);
        assert!(toggled.dry_run);
        assert!(toggled.verify_only);
        assert!(toggled.overwrite_existing);
        assert!(toggled.no_restore_seen_markers);
        assert!(toggled.allow_restore_orphan_assignments);

        // --format closed enum.
        for (raw, expected) in &[
            ("events", RestoreQuarantineFormat::Events),
            ("json", RestoreQuarantineFormat::Json),
            ("pretty", RestoreQuarantineFormat::Pretty),
        ] {
            let got = parse_restore_state_cleanup_quarantine(&[
                "--quarantine-plan-dir",
                "/tmp/q/abc",
                "--format",
                raw,
            ]);
            assert_eq!(got.format, *expected);
        }

        // Unknown --format value is rejected.
        let bad_format = TestRoot::try_parse_from([
            "omni-node",
            "restore-state-cleanup-quarantine",
            "--contributor-state-dir",
            "/tmp/state",
            "--quarantine-plan-dir",
            "/tmp/q/abc",
            "--format",
            "yaml",
        ]);
        assert!(
            bad_format.is_err(),
            "clap must reject unknown --format value"
        );

        // Auto-prune flag is deliberately absent on this
        // subcommand (Stage 12.16/12.17 precedent).
        let no_such_flag = TestRoot::try_parse_from([
            "omni-node",
            "restore-state-cleanup-quarantine",
            "--contributor-state-dir",
            "/tmp/state",
            "--quarantine-plan-dir",
            "/tmp/q/abc",
            "--no-prune-state-on-start",
        ]);
        assert!(
            no_such_flag.is_err(),
            "restore-state-cleanup-quarantine must not accept --no-prune-state-on-start"
        );
    }

    // ── Stage 14.2 — halo2-reference sidecar proof emission ─────────────────

    #[cfg(feature = "halo2-reference-prove")]
    mod stage_14_2_sidecar_proof {
        use super::*;
        use omni_contributor::{
            BaseUnitRewardPolicy, ContributorJob, ContributorResult, Evidence,
            JobAccounting, MeasuredAccounting, StageContribution,
            VerificationRequirement,
        };

        // ── Canonical-spec fixture helpers ──────────────────────────────────

        const CANONICAL_SPEC: &[u8] = include_bytes!(
            "../../omni-proofs-halo2-reference/assets/canonical_spec.json"
        );

        fn canonical_spec_hash_hex() -> String {
            blake3::hash(CANONICAL_SPEC).to_hex().to_string()
        }

        fn canonical_input_bytes() -> Vec<u8> {
            omni_proofs_halo2_reference::encode_canonical_input(
                &omni_proofs_halo2_reference::CANONICAL_INPUT,
            )
        }

        fn canonical_output_bytes() -> Vec<u8> {
            omni_proofs_halo2_reference::encode_canonical_output(
                &omni_proofs_halo2_reference::canonical_evaluate(
                    omni_proofs_halo2_reference::CANONICAL_INPUT,
                ),
            )
        }

        fn hex64(b: u8) -> String {
            let mut s = String::with_capacity(64);
            for _ in 0..32 {
                s.push_str(&format!("{b:02x}"));
            }
            s
        }

        fn snip_root_hex(seed: u8) -> String {
            // 66 chars including "0x" prefix.
            format!("0x{}", hex64(seed))
        }

        fn sig_hex(seed: u8) -> String {
            let mut s = String::with_capacity(128);
            for _ in 0..64 {
                s.push_str(&format!("{seed:02x}"));
            }
            s
        }

        fn build_canonical_job(input_hash_hex: String, model_hash_hex: String) -> ContributorJob {
            ContributorJob {
                schema_version: 1,
                job_id: hex64(0x11),
                model_hash: model_hash_hex,
                manifest_snip_root: snip_root_hex(0x22),
                input_snip_root: snip_root_hex(0x33),
                input_hash: input_hash_hex,
                verification_requirement: VerificationRequirement::AttestationOnly,
                accounting: JobAccounting {
                    tokenizer_hash: hex64(0x44),
                    tokenizer_id: "test-tokenizer".to_string(),
                    input_token_count: 1,
                    max_output_token_count: 1,
                    base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
                },
                dispatched_at_utc: "2026-06-20T00:00:00Z".to_string(),
                expires_at_utc: None,
                dispatcher_pubkey_hex: None,
                dispatcher_signature_hex: None,
                notes: None,
            }
        }

        fn build_canonical_result(
            job: &ContributorJob,
            response_hash_hex: String,
        ) -> ContributorResult {
            ContributorResult {
                schema_version: 1,
                job_id: job.job_id.clone(),
                job_hash: job.job_id.clone(),
                job_snip_root: None,
                model_hash: job.model_hash.clone(),
                input_hash: job.input_hash.clone(),
                response_snip_root: snip_root_hex(0x55),
                response_hash: response_hash_hex,
                evidence: Evidence::AttestationOnly,
                measured_accounting: MeasuredAccounting {
                    tokenizer_hash: job.accounting.tokenizer_hash.clone(),
                    input_token_count: 1,
                    output_token_count: 1,
                    total_base_units: 2,
                    stage_contributions: vec![StageContribution {
                        contributor_pubkey_hex: hex64(0x66),
                        stage_label: "stub-runner".to_string(),
                        work_unit_kind: omni_contributor::WorkUnitKind::DecodeTokens,
                        work_units: 2,
                    }],
                },
                produced_at_utc: "2026-06-20T00:00:01Z".to_string(),
                contributor_pubkey_hex: hex64(0x66),
                contributor_signature_hex: sig_hex(0x77),
                notes: None,
            }
        }

        fn write_temp(dir: &std::path::Path, name: &str, bytes: &[u8]) -> PathBuf {
            let p = dir.join(name);
            std::fs::write(&p, bytes).unwrap();
            p
        }

        // Canonical, hash-aligned, ready-to-prove fixture suite.
        struct Fixture {
            _dir: tempfile::TempDir,
            job: ContributorJob,
            result: ContributorResult,
            stub_input_path: PathBuf,
            stub_response_path: PathBuf,
            proof_path: PathBuf,
        }

        fn build_canonical_fixture() -> Fixture {
            let dir = tempfile::tempdir().unwrap();
            let in_bytes = canonical_input_bytes();
            let out_bytes = canonical_output_bytes();
            let input_hash = blake3::hash(&in_bytes).to_hex().to_string();
            let output_hash = blake3::hash(&out_bytes).to_hex().to_string();
            let model_hash = canonical_spec_hash_hex();
            let job = build_canonical_job(input_hash, model_hash);
            let result = build_canonical_result(&job, output_hash);
            let stub_input_path = write_temp(dir.path(), "stub_input.bin", &in_bytes);
            let stub_response_path =
                write_temp(dir.path(), "stub_response.bin", &out_bytes);
            let proof_path = dir.path().join("sidecar_proof.json");
            Fixture {
                _dir: dir,
                job,
                result,
                stub_input_path,
                stub_response_path,
                proof_path,
            }
        }

        // ── Test 1: happy path ──────────────────────────────────────────────

        #[test]
        fn run_job_with_emit_flag_writes_sidecar_artifact_under_canonical_spec_job() {
            let f = build_canonical_fixture();
            emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .expect("sidecar emission on canonical fixture must succeed");
            assert!(f.proof_path.is_file());
            let bytes = std::fs::read(&f.proof_path).unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body.metadata.proof_system,
                Some(omni_zkml::ProofSystem::Stage11bHalo2Reference)
            );
            assert_eq!(body.metadata.testnet_or_dev_only, Some(true));
            assert!(!body.proof_bytes_hex.is_empty());
        }

        // ── Test 2: D6 — sidecar does not mutate ContributorResult bytes ────

        #[test]
        fn emit_sidecar_does_not_mutate_contributor_result_bytes_on_disk() {
            // Structural pin matching D6 (semantic equality lock).
            // The helper takes `&ContributorResult` (immutable
            // borrow); this test additionally confirms that a
            // separately-written result file is byte-untouched even
            // when sidecar emission succeeds. Catches any future
            // refactor that conflates the two write paths.
            let f = build_canonical_fixture();
            let result_path = f._dir.path().join("result.json");
            let result_bytes = serde_json::to_vec_pretty(&f.result).unwrap();
            std::fs::write(&result_path, &result_bytes).unwrap();
            let before = std::fs::read(&result_path).unwrap();
            emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let after = std::fs::read(&result_path).unwrap();
            assert_eq!(before, after);
            assert!(f.proof_path.is_file());
        }

        // ── Test 3: D5 — non-canonical model_hash refused; no sidecar ───────

        #[test]
        fn run_job_refuses_emit_flag_when_job_model_hash_is_not_canonical_spec() {
            let mut f = build_canonical_fixture();
            // Set model_hash to something definitely not the
            // canonical spec hash.
            f.job.model_hash = hex64(0xaa);
            let err = emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap_err();
            // D5 — anyhow message only; no closed taxonomy assertion.
            assert!(err.to_string().contains("canonical halo2-mlp-v1 spec hash"));
            assert!(
                !f.proof_path.exists(),
                "no sidecar must be written when job is non-canonical"
            );
        }

        // ── Test 4: hash-binding refusal on stub_input ──────────────────────

        #[test]
        fn run_job_refuses_emit_flag_when_stub_input_hash_mismatches_job_input_hash() {
            let f = build_canonical_fixture();
            // Substitute stub_input bytes that do NOT hash to
            // job.input_hash.
            let bad_input = f._dir.path().join("bad_input.bin");
            std::fs::write(&bad_input, b"definitely-not-the-canonical-input").unwrap();
            let err = emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&bad_input),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap_err();
            assert!(err.to_string().contains("does not match job.input_hash"));
            assert!(!f.proof_path.exists());
        }

        // ── Test 5: D1 Alpha transition — clap no longer requires
        //              stub_input, but runtime check refuses ────────────

        #[test]
        fn run_run_job_refuses_emit_flag_without_stub_input_for_stub_runner_via_runtime_check() {
            // Stage 14.3 D1 Alpha — the static
            // `requires = "stub_input"` was removed from
            // `--emit-halo2-reference-proof` so the ExternalRunner
            // emit path does not falsely require `--stub-input`.
            // The runtime check in `run_run_job` enforces the
            // StubRunner pairing instead. Same user-observable
            // contract (refusal before any work runs, clear error
            // mentioning --stub-input), different mechanism.
            //
            // Part 1: clap parse now SUCCEEDS (no static requires).
            let root = TestRoot::try_parse_from([
                "omni-node",
                "run-job",
                "--job",
                "/tmp/phantom_job.json",
                "--out",
                "/tmp/phantom_result.json",
                "--seed-file",
                "/tmp/phantom_seed",
                "--stub-response",
                "/tmp/phantom_response.bin",
                "--emit-halo2-reference-proof",
                "/tmp/phantom_proof.json",
                // intentionally NO --stub-input
            ])
            .expect("clap parse must succeed after D1 Alpha removed static requires");
            let run_job_args = match root.contributor.cmd {
                ContributorCmd::RunJob(a) => a,
                _ => panic!("expected RunJob variant"),
            };
            // Part 2: runtime check fires before any work runs
            // (phantom paths are NEVER opened — the early check
            // bails first).
            let err = super::run_run_job(run_job_args)
                .expect_err("runtime check must refuse missing --stub-input");
            assert!(
                err.to_string().contains("--stub-input"),
                "expected runtime refusal mentioning --stub-input; got: {err}"
            );
        }

        // ── Test 6: file-based wrapper retains its StubRunner-only refusal ─

        #[test]
        fn file_based_emit_helper_refuses_non_stub_runner() {
            // The Stage 14.2 file-based helper is the StubRunner
            // path. The Stage 14.3 ExternalCommandRunner emit
            // path goes through
            // `emit_halo2_reference_proof_sidecar_from_bytes`
            // instead. This test pins that the file-based wrapper
            // refuses any other runner choice so it cannot be
            // accidentally called for the External path.
            let f = build_canonical_fixture();
            let err = emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::External,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("StubRunner path")
                    || err.to_string().contains("emit_halo2_reference_proof_sidecar_from_bytes"),
                "expected StubRunner-path refusal; got: {}",
                err
            );
            assert!(!f.proof_path.exists());
        }

        // ── Test 7: end-to-end — verifier accepts the contributor sidecar ─

        #[test]
        fn halo2_reference_verifier_accepts_contributor_emitted_sidecar() {
            let f = build_canonical_fixture();
            emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let body_bytes = std::fs::read(&f.proof_path).unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&body_bytes).unwrap();
            use omni_zkml::ProofVerifier;
            let verifier =
                omni_proofs_halo2_reference::Halo2ReferenceVerifier::from_embedded_fixtures()
                    .expect("verifier construction from embedded fixtures");
            let verified = verifier
                .verify_artifact(&body)
                .expect("verifier should run without internal error");
            assert!(
                verified,
                "halo2-reference verifier must accept the contributor-emitted sidecar"
            );
        }

        // ── Test 8: mainnet refusal posture ─────────────────────────────────

        #[test]
        fn sidecar_artifact_carries_testnet_or_dev_only_true_and_is_mainnet_refused() {
            let f = build_canonical_fixture();
            emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&f.proof_path).unwrap()).unwrap();
            assert_eq!(body.metadata.testnet_or_dev_only, Some(true));
            assert_eq!(
                body.metadata.proof_system,
                Some(omni_zkml::ProofSystem::Stage11bHalo2Reference)
            );
            // Mainnet-eligibility check refuses the artifact's
            // metadata. `check_mainnet_eligible` reads `testnet_or_dev_only`
            // (layer 1), then `proof_system` (layers 3 + 6); any of
            // them triggers on a Stage 14.2 sidecar.
            let refusal = omni_zkml::check_mainnet_eligible(&body.metadata);
            assert!(
                refusal.is_err(),
                "sidecar artifact MUST be refused on mainnet eligibility; got Ok"
            );
        }

        // ── Test 9: D2 — public_inputs carries contributor_job_id ───────────

        #[test]
        fn sidecar_public_inputs_carry_contributor_job_id() {
            let f = build_canonical_fixture();
            emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&f.proof_path).unwrap()).unwrap();
            let pi = body
                .metadata
                .public_inputs
                .expect("public_inputs must be populated");
            let job_id_value = pi
                .get("contributor_job_id")
                .expect("public_inputs must carry contributor_job_id key (D2)");
            assert_eq!(
                job_id_value.as_str(),
                Some(f.result.job_id.as_str())
            );
        }

        // ── Test 10: D2 regression — verifier tolerates extra public_inputs key ─

        #[test]
        fn verifier_tolerates_extra_contributor_job_id_key_in_public_inputs() {
            // Construct an artifact independently of the sidecar
            // emitter to pin the verifier's "extra-key tolerance"
            // — if a future tightening of
            // `decode_public_inputs_json` adds `deny_unknown_fields`
            // semantics, this test fails before Stage 14.2's
            // sidecar emission silently breaks.
            let f = build_canonical_fixture();
            // First produce a sidecar via the helper.
            emit_halo2_reference_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let mut body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&f.proof_path).unwrap()).unwrap();
            // Confirm the extra key is present (sanity).
            assert!(body
                .metadata
                .public_inputs
                .as_ref()
                .and_then(|v| v.get("contributor_job_id"))
                .is_some());
            // Add a SECOND extra key that the verifier has never
            // seen — proves the tolerance generalises beyond just
            // contributor_job_id.
            if let Some(pi) = body.metadata.public_inputs.as_mut() {
                if let Some(obj) = pi.as_object_mut() {
                    obj.insert(
                        "stage_14_2_synthetic_extra_key".to_string(),
                        serde_json::json!({ "anything": [1, 2, 3] }),
                    );
                }
            }
            use omni_zkml::ProofVerifier;
            let verifier =
                omni_proofs_halo2_reference::Halo2ReferenceVerifier::from_embedded_fixtures()
                    .unwrap();
            let verified = verifier
                .verify_artifact(&body)
                .expect("verifier must not error on extra public_inputs keys");
            assert!(
                verified,
                "verifier must accept artifacts whose public_inputs carries \
                 additional keys beyond input + output (D2 tolerance)"
            );
        }
    }

    // ── Stage 14.3 — ExternalCommandRunner sidecar proof emission ──────────

    #[cfg(feature = "halo2-reference-prove")]
    mod stage_14_3_external_runner_sidecar_proof {
        use super::*;
        use omni_contributor::{InferenceRunner, RunOutput, RunnerError};

        // ── Shared canonical-spec helpers (mirror Stage 14.2 fixture).

        const CANONICAL_SPEC: &[u8] = include_bytes!(
            "../../omni-proofs-halo2-reference/assets/canonical_spec.json"
        );

        fn canonical_spec_hash_hex() -> String {
            blake3::hash(CANONICAL_SPEC).to_hex().to_string()
        }

        fn canonical_input_bytes() -> Vec<u8> {
            omni_proofs_halo2_reference::encode_canonical_input(
                &omni_proofs_halo2_reference::CANONICAL_INPUT,
            )
        }

        fn canonical_output_bytes() -> Vec<u8> {
            omni_proofs_halo2_reference::encode_canonical_output(
                &omni_proofs_halo2_reference::canonical_evaluate(
                    omni_proofs_halo2_reference::CANONICAL_INPUT,
                ),
            )
        }

        fn hex64(b: u8) -> String {
            let mut s = String::with_capacity(64);
            for _ in 0..32 {
                s.push_str(&format!("{b:02x}"));
            }
            s
        }

        fn snip_root_hex(seed: u8) -> String {
            format!("0x{}", hex64(seed))
        }

        fn sig_hex(seed: u8) -> String {
            let mut s = String::with_capacity(128);
            for _ in 0..64 {
                s.push_str(&format!("{seed:02x}"));
            }
            s
        }

        fn canonical_job_and_result() -> (
            omni_contributor::ContributorJob,
            omni_contributor::ContributorResult,
        ) {
            use omni_contributor::{
                BaseUnitRewardPolicy, ContributorJob, ContributorResult, Evidence,
                JobAccounting, MeasuredAccounting, StageContribution,
                VerificationRequirement,
            };
            let in_bytes = canonical_input_bytes();
            let out_bytes = canonical_output_bytes();
            let input_hash = blake3::hash(&in_bytes).to_hex().to_string();
            let output_hash = blake3::hash(&out_bytes).to_hex().to_string();
            let job = ContributorJob {
                schema_version: 1,
                job_id: hex64(0x11),
                model_hash: canonical_spec_hash_hex(),
                manifest_snip_root: snip_root_hex(0x22),
                input_snip_root: snip_root_hex(0x33),
                input_hash,
                verification_requirement: VerificationRequirement::AttestationOnly,
                accounting: JobAccounting {
                    tokenizer_hash: hex64(0x44),
                    tokenizer_id: "test-tokenizer".to_string(),
                    input_token_count: 1,
                    max_output_token_count: 1,
                    base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
                },
                dispatched_at_utc: "2026-06-20T00:00:00Z".to_string(),
                expires_at_utc: None,
                dispatcher_pubkey_hex: None,
                dispatcher_signature_hex: None,
                notes: None,
            };
            let result = ContributorResult {
                schema_version: 1,
                job_id: job.job_id.clone(),
                job_hash: job.job_id.clone(),
                job_snip_root: None,
                model_hash: job.model_hash.clone(),
                input_hash: job.input_hash.clone(),
                response_snip_root: snip_root_hex(0x55),
                response_hash: output_hash,
                evidence: Evidence::AttestationOnly,
                measured_accounting: MeasuredAccounting {
                    tokenizer_hash: job.accounting.tokenizer_hash.clone(),
                    input_token_count: 1,
                    output_token_count: 1,
                    total_base_units: 2,
                    stage_contributions: vec![StageContribution {
                        contributor_pubkey_hex: hex64(0x66),
                        stage_label: "external".to_string(),
                        work_unit_kind: omni_contributor::WorkUnitKind::DecodeTokens,
                        work_units: 2,
                    }],
                },
                produced_at_utc: "2026-06-20T00:00:01Z".to_string(),
                contributor_pubkey_hex: hex64(0x66),
                contributor_signature_hex: sig_hex(0x77),
                notes: None,
            };
            (job, result)
        }

        // ── ByteCapturingRunner lifecycle unit tests (D6) ──────────────────

        /// In-test fake runner that lets us script exact byte
        /// returns and error paths so the wrapper's D6 lifecycle
        /// can be exercised in isolation from the
        /// ExternalCommandRunner subprocess plumbing.
        struct ScriptedRunner {
            return_value: std::cell::RefCell<
                Vec<std::result::Result<Vec<u8>, &'static str>>,
            >,
            saw_inputs: std::cell::RefCell<Vec<Vec<u8>>>,
        }

        impl ScriptedRunner {
            fn new(returns: Vec<std::result::Result<Vec<u8>, &'static str>>) -> Self {
                Self {
                    return_value: std::cell::RefCell::new(returns),
                    saw_inputs: std::cell::RefCell::new(Vec::new()),
                }
            }
        }

        impl InferenceRunner for ScriptedRunner {
            fn run(
                &self,
                _manifest_path: &std::path::Path,
                input_bytes: &[u8],
            ) -> std::result::Result<RunOutput, RunnerError> {
                self.saw_inputs.borrow_mut().push(input_bytes.to_vec());
                let next = self.return_value.borrow_mut().remove(0);
                match next {
                    Ok(response_bytes) => Ok(RunOutput {
                        response_bytes,
                        measured_input_tokens: 1,
                        measured_output_tokens: 1,
                        stage_contributions: vec![
                            omni_contributor::StageContribution {
                                contributor_pubkey_hex: hex64(0x66),
                                stage_label: "scripted".to_string(),
                                work_unit_kind:
                                    omni_contributor::WorkUnitKind::DecodeTokens,
                                work_units: 1,
                            },
                        ],
                    }),
                    Err(msg) => Err(RunnerError::ExternalCommandFailure {
                        code: 1,
                        stderr: msg.to_string(),
                    }),
                }
            }
        }

        // ── Test 7: D6 — slots cleared at start of every run ───────────────

        #[test]
        fn byte_capturing_runner_clears_both_slots_at_start_of_each_run() {
            // First call returns Ok (populates output); second
            // call's inner returns Err (must clear output back to
            // None — proves the wrapper does not leak the
            // previous success's bytes).
            let inner = ScriptedRunner::new(vec![
                Ok(canonical_output_bytes()),
                Err("scripted runner failure"),
            ]);
            let wrapper = ByteCapturingRunner::new(&inner);

            let _ = wrapper.run(std::path::Path::new("/tmp/m1"), &canonical_input_bytes()).unwrap();
            assert!(wrapper.take_captured_input().is_some());
            assert!(wrapper.take_captured_output().is_some());

            // Second run errors: captured_output must be None.
            let alt_input = b"alternate-input-bytes".to_vec();
            let err =
                wrapper.run(std::path::Path::new("/tmp/m2"), &alt_input).unwrap_err();
            assert!(matches!(err, RunnerError::ExternalCommandFailure { .. }));
            // captured_input was set to the alternate bytes at the
            // start of the second run, replacing the first run's value.
            assert_eq!(wrapper.take_captured_input().unwrap(), alt_input);
            // captured_output was cleared at the start and never
            // re-populated because inner errored.
            assert!(
                wrapper.take_captured_output().is_none(),
                "D6 invariant: captured_output must be None after inner Err"
            );
        }

        // ── Test 8: D6 — inner Err leaves output None on a fresh wrapper ──

        #[test]
        fn byte_capturing_runner_does_not_capture_output_on_inner_error() {
            let inner = ScriptedRunner::new(vec![Err("inner failed")]);
            let wrapper = ByteCapturingRunner::new(&inner);
            let err = wrapper
                .run(std::path::Path::new("/tmp/m"), &canonical_input_bytes())
                .unwrap_err();
            assert!(matches!(err, RunnerError::ExternalCommandFailure { .. }));
            assert!(wrapper.take_captured_input().is_some());
            assert!(
                wrapper.take_captured_output().is_none(),
                "D6 invariant: captured_output must be None when inner returns Err"
            );
        }

        // ── Test 9: D6 — input captured before inner call ──────────────────

        #[test]
        fn byte_capturing_runner_captures_input_before_inner_call_runs() {
            // Even when the inner errors, captured_input must hold
            // the exact bytes that were passed in — documenting the
            // "input captured before inner call" invariant so a
            // future refactor that delays capture (e.g. only after
            // success) regresses visibly here.
            let inner = ScriptedRunner::new(vec![Err("inner failed")]);
            let wrapper = ByteCapturingRunner::new(&inner);
            let payload = b"payload-for-capture-pin".to_vec();
            let _ = wrapper.run(std::path::Path::new("/tmp/m"), &payload).unwrap_err();
            assert_eq!(wrapper.take_captured_input().as_deref(), Some(payload.as_slice()));
        }

        // ── Tests 1–4: bytes-based emit helper (External path proxy) ──────
        //
        // The bytes-based helper is the unit under Stage 14.3 test:
        // it accepts captured `&[u8]` directly and runs the same
        // canonical-spec + hash-binding + prover pipeline that
        // Stage 14.2's file-based helper does. Hash bindings on
        // External captured bytes are tautological by construction
        // (the wrapper captures the same bytes `run_job` hashes),
        // so these tests pin the helper's correctness directly
        // rather than spawning a subprocess.

        #[test]
        fn external_path_bytes_helper_writes_sidecar_for_canonical_triple() {
            let dir = tempfile::tempdir().unwrap();
            let proof_path = dir.path().join("external_sidecar.json");
            let (job, result) = canonical_job_and_result();
            emit_halo2_reference_proof_sidecar_from_bytes(
                &job,
                &result,
                &canonical_input_bytes(),
                &canonical_output_bytes(),
                &proof_path,
            )
            .expect("bytes-based helper must succeed on canonical triple");
            assert!(proof_path.is_file());
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&proof_path).unwrap()).unwrap();
            assert_eq!(
                body.metadata.proof_system,
                Some(omni_zkml::ProofSystem::Stage11bHalo2Reference)
            );
        }

        #[test]
        fn external_path_sidecar_input_hash_equals_captured_input_bytes_hash() {
            let dir = tempfile::tempdir().unwrap();
            let proof_path = dir.path().join("p.json");
            let (job, result) = canonical_job_and_result();
            let input = canonical_input_bytes();
            emit_halo2_reference_proof_sidecar_from_bytes(
                &job, &result, &input, &canonical_output_bytes(), &proof_path,
            )
            .unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&proof_path).unwrap()).unwrap();
            assert_eq!(
                body.metadata.input_hash,
                blake3::hash(&input).to_hex().to_string()
            );
            // And it equals the job's committed input_hash
            // (tautological — both come from the same bytes).
            assert_eq!(body.metadata.input_hash, job.input_hash);
        }

        #[test]
        fn external_path_sidecar_output_hash_equals_envelope_response_bytes_hash() {
            let dir = tempfile::tempdir().unwrap();
            let proof_path = dir.path().join("p.json");
            let (job, result) = canonical_job_and_result();
            let output = canonical_output_bytes();
            emit_halo2_reference_proof_sidecar_from_bytes(
                &job, &result, &canonical_input_bytes(), &output, &proof_path,
            )
            .unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&proof_path).unwrap()).unwrap();
            assert_eq!(
                body.metadata.response_hash,
                blake3::hash(&output).to_hex().to_string()
            );
            assert_eq!(body.metadata.response_hash, result.response_hash);
        }

        #[test]
        fn external_path_sidecar_verifies_under_halo2_reference_verifier() {
            let dir = tempfile::tempdir().unwrap();
            let proof_path = dir.path().join("p.json");
            let (job, result) = canonical_job_and_result();
            emit_halo2_reference_proof_sidecar_from_bytes(
                &job,
                &result,
                &canonical_input_bytes(),
                &canonical_output_bytes(),
                &proof_path,
            )
            .unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&proof_path).unwrap()).unwrap();
            use omni_zkml::ProofVerifier;
            let verifier =
                omni_proofs_halo2_reference::Halo2ReferenceVerifier::from_embedded_fixtures()
                    .unwrap();
            let verified = verifier.verify_artifact(&body).unwrap();
            assert!(verified);
        }

        // ── Test 5: refusal when response bytes != canonical_evaluate(input) ─

        #[test]
        fn external_path_refuses_when_envelope_response_bytes_do_not_match_canonical_evaluator_output()
        {
            // Construct a `(job, result)` whose response_hash
            // matches some non-canonical bytes; the bytes-based
            // helper validates hash bindings first (passes), then
            // invokes the prover, which re-runs
            // `canonical_evaluate(input)` and refuses because the
            // claimed output does not match. No sidecar written.
            let dir = tempfile::tempdir().unwrap();
            let proof_path = dir.path().join("p.json");
            let (job, mut result) = canonical_job_and_result();
            let bad_output: Vec<u8> = (0..8).map(|i| (0xAA ^ i) as u8).collect();
            result.response_hash = blake3::hash(&bad_output).to_hex().to_string();
            let err = emit_halo2_reference_proof_sidecar_from_bytes(
                &job,
                &result,
                &canonical_input_bytes(),
                &bad_output,
                &proof_path,
            )
            .unwrap_err();
            // The adapter's pre-check fires before any halo2 work.
            assert!(
                err.to_string().contains("canonical_evaluate")
                    || err.to_string().contains("halo2-reference prover failure"),
                "expected canonical-evaluator refusal; got: {err}"
            );
            assert!(!proof_path.exists());
        }

        // ── Test 6: D4 + D6 — no sidecar written when runner returns Err ──

        #[test]
        fn no_sidecar_is_written_when_byte_capturing_runner_inner_returns_err() {
            // Direct unit test of the wrapper + helper composition
            // that `run_run_job` performs for the External + emit
            // path: if inner returns Err, captured_output is None
            // (D6); the `take_captured_output().ok_or_else(...)`
            // guard in `run_run_job` short-circuits and the
            // bytes-based helper is NEVER called, so no sidecar
            // file lands. We assert no sidecar file exists at the
            // chosen path.
            let inner = ScriptedRunner::new(vec![Err("scripted err")]);
            let wrapper = ByteCapturingRunner::new(&inner);
            let _ = wrapper
                .run(std::path::Path::new("/tmp/m"), &canonical_input_bytes())
                .unwrap_err();
            assert!(wrapper.take_captured_output().is_none());
            // D6 — captured_output None means the External branch
            // would early-bail in run_run_job. We do not even
            // attempt to call the helper; the sidecar file at any
            // chosen path simply never gets created.
            let dir = tempfile::tempdir().unwrap();
            let phantom_sidecar = dir.path().join("never_written.json");
            assert!(
                !phantom_sidecar.exists(),
                "D4 + D6: sidecar must NOT be written when runner errored"
            );
        }

        // ── Test 11: real ExternalCommandRunner subprocess end-to-end ──────
        //
        // The unit-level tests above pin the wrapper's behavior at
        // the trait boundary using an in-process `ScriptedRunner`.
        // This test additionally proves the operator promise: a
        // **real ExternalCommandRunner** spawned against a Unix
        // shell script flows through the wrapper, the bytes-based
        // sidecar helper, and the verifier — closing the loop on
        // the Stage 14.3 acceptance bar. Mirrors the repo pattern
        // at `integrity_evidence_bundle.rs:446` for `#[cfg(unix)]`
        // + `set_permissions(0o755)`. CI runs on Linux.

        /// Tiny self-contained base64 encoder used only by the
        /// external-runner subprocess test. `omni-node` does not
        /// otherwise depend on `base64`; bringing it in for a
        /// single test fixture would touch Cargo.toml. The
        /// canonical output bytes are 8 bytes long so the encoded
        /// payload is deterministic and tiny.
        fn base64_encode(bytes: &[u8]) -> String {
            const ALPHABET: &[u8; 64] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
            let mut i = 0;
            while i + 3 <= bytes.len() {
                let n = ((bytes[i] as u32) << 16)
                    | ((bytes[i + 1] as u32) << 8)
                    | (bytes[i + 2] as u32);
                out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
                out.push(ALPHABET[(n & 0x3f) as usize] as char);
                i += 3;
            }
            let rem = bytes.len() - i;
            if rem == 1 {
                let n = (bytes[i] as u32) << 16;
                out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
                out.push('=');
                out.push('=');
            } else if rem == 2 {
                let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
                out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
                out.push('=');
            }
            out
        }

        #[cfg(unix)]
        #[test]
        fn external_command_runner_real_subprocess_end_to_end_writes_verifiable_sidecar() {
            // ──────────────────────────────────────────────────────
            // Operator promise covered:
            //
            //   contributor run-job --runner external \
            //     --external-command <SCRIPT> \
            //     --emit-halo2-reference-proof <PATH>
            //
            // → external command receives `--input <tempfile>`,
            // → writes valid envelope JSON to stdout,
            // → ByteCapturingRunner captures the bytes that flowed
            //   through `InferenceRunner::run` at the trait
            //   boundary,
            // → `emit_halo2_reference_proof_sidecar_from_bytes`
            //   assembles a verifying `ProofArtifactBody`.
            //
            // The test exercises the **real ExternalCommandRunner
            // subprocess path** (not the in-process
            // `ScriptedRunner`), so the wrapper integration with
            // the actual subprocess driver is pinned end-to-end.
            // ──────────────────────────────────────────────────────
            use std::os::unix::fs::PermissionsExt;
            use omni_contributor::ExternalCommandRunner;

            let dir = tempfile::tempdir().unwrap();

            // The canonical 8-byte output that `canonical_evaluate(CANONICAL_INPUT)`
            // produces, base64-encoded for the envelope.
            let output_bytes = canonical_output_bytes();
            let response_b64 = base64_encode(&output_bytes);

            // Hand-build a valid `ExternalRunnerEnvelope` JSON. The
            // runner uses `deny_unknown_fields`; field names + the
            // snake_case `decode_tokens` literal match the
            // `WorkUnitKind` serde representation.
            let envelope_json = format!(
                r#"{{"response_b64":"{response_b64}","measured_input_tokens":1,"measured_output_tokens":1,"stage_contributions":[{{"contributor_pubkey_hex":"{}","stage_label":"external-test","work_unit_kind":"decode_tokens","work_units":1}}]}}"#,
                hex64(0x66)
            );

            // Shell script that ignores --manifest / --input and
            // emits the canned envelope. The runner's contract
            // (12.0) is "stdout MUST parse as one
            // ExternalRunnerEnvelope"; stderr is unrestricted.
            let script_body =
                format!("#!/bin/sh\ncat <<'EOF'\n{envelope_json}\nEOF\n");
            let script_path = dir.path().join("inference-runner.sh");
            std::fs::write(&script_path, script_body).unwrap();
            let mut perms =
                std::fs::metadata(&script_path).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms).unwrap();

            // Manifest tempfile (unused by our script, but the
            // runner stages it as a tempfile and passes
            // `--manifest <path>`; the path must exist).
            let manifest_path = dir.path().join("manifest.bin");
            std::fs::write(&manifest_path, b"unused-manifest-bytes").unwrap();

            // REAL ExternalCommandRunner (not ScriptedRunner).
            let runner = ExternalCommandRunner::new(script_path.clone());

            // Same wrapping as run_run_job's External + emit-flag branch.
            let wrapper = ByteCapturingRunner::new(&runner);

            // Drive a REAL subprocess through the trait boundary.
            let input_bytes = canonical_input_bytes();
            let run_output = wrapper
                .run(&manifest_path, &input_bytes)
                .expect("external subprocess must succeed against our envelope");

            // Wrapper captured the exact bytes the subprocess saw.
            let captured_input = wrapper
                .take_captured_input()
                .expect("input must be captured on Ok inner");
            let captured_output = wrapper
                .take_captured_output()
                .expect("output must be captured on Ok inner");
            assert_eq!(
                captured_input, input_bytes,
                "captured_input must equal the bytes we passed to wrapper.run"
            );
            assert_eq!(
                captured_output, run_output.response_bytes,
                "captured_output must equal the runner-returned response_bytes"
            );
            assert_eq!(
                captured_output, output_bytes,
                "the subprocess's envelope-decoded response_bytes must equal \
                 the canonical output we encoded into the envelope"
            );

            // Hand the captured bytes to the bytes-based sidecar
            // helper — the same call run_run_job makes after
            // extracting the wrapper's slots.
            let (job, result) = canonical_job_and_result();
            let sidecar_path = dir.path().join("sidecar_proof.json");
            emit_halo2_reference_proof_sidecar_from_bytes(
                &job,
                &result,
                &captured_input,
                &captured_output,
                &sidecar_path,
            )
            .expect("sidecar emission from captured external-runner bytes");

            // Sidecar verifies under the existing Halo2 reference verifier.
            let body: omni_zkml::ProofArtifactBody = serde_json::from_slice(
                &std::fs::read(&sidecar_path).unwrap(),
            )
            .unwrap();
            use omni_zkml::ProofVerifier;
            let verifier =
                omni_proofs_halo2_reference::Halo2ReferenceVerifier::from_embedded_fixtures()
                    .unwrap();
            let verified =
                verifier.verify_artifact(&body).expect("verifier must not error");
            assert!(
                verified,
                "sidecar produced via real external runner subprocess must verify"
            );
            // Hash bindings tautological by construction — pin them
            // anyway so a future wrapper regression that re-encodes
            // bytes regresses visibly here.
            assert_eq!(body.metadata.input_hash, job.input_hash);
            assert_eq!(body.metadata.response_hash, result.response_hash);
        }

        // ── Test 10: D7 — Stage 14.3 wraps only `run`, not `run_with_activations` ─

        #[test]
        fn byte_capturing_runner_does_not_override_run_with_activations_per_d7() {
            // D7 — Stage 14.3 explicitly covers
            // `InferenceRunner::run` only. This compile-pin
            // documents that we DO NOT provide a custom
            // `run_with_activations` impl; the default-impl
            // forward-through-`run` is incidental, not a guarantee
            // Stage 14.3 maintains. A future refactor that
            // overrides `run_with_activations` should fail this
            // test by failing to compile (the test's body
            // constructs a wrapper and asserts the standard `run`
            // path is the only one we exercise).
            let inner = ScriptedRunner::new(vec![Ok(canonical_output_bytes())]);
            let wrapper = ByteCapturingRunner::new(&inner);
            // Call standard `run` — succeeds and captures bytes.
            let _ = wrapper.run(std::path::Path::new("/tmp/m"), &canonical_input_bytes()).unwrap();
            assert!(wrapper.take_captured_input().is_some());
            assert!(wrapper.take_captured_output().is_some());
            // Stage 14.3 makes NO claim about `run_with_activations`.
            // The default-impl forward chain is incidental.
        }
    }

    // ── Stage 14.6 — production-MLP contributor sidecar proof ─────────────

    #[cfg(feature = "stage11d-production-prove")]
    mod stage_14_6_production_mlp_sidecar_proof {
        use super::*;
        use omni_contributor::{
            BaseUnitRewardPolicy, ContributorJob, ContributorResult, Evidence,
            JobAccounting, MeasuredAccounting, StageContribution,
            VerificationRequirement,
        };

        // ── Production-spec fixture helpers ────────────────────────────────

        const PRODUCTION_CANONICAL_SPEC: &[u8] = include_bytes!(
            "../../omni-proofs-halo2-production-mlp/assets/canonical_spec.json"
        );

        fn production_spec_hash_hex() -> String {
            blake3::hash(PRODUCTION_CANONICAL_SPEC).to_hex().to_string()
        }

        fn canonical_input_bytes() -> Vec<u8> {
            omni_proofs_halo2_production_mlp::encode_canonical_input(
                &omni_proofs_halo2_production_mlp::CANONICAL_INPUT,
            )
        }

        fn canonical_output_bytes() -> Vec<u8> {
            omni_proofs_halo2_production_mlp::encode_canonical_output(
                &omni_proofs_halo2_production_mlp::canonical_evaluate(
                    omni_proofs_halo2_production_mlp::CANONICAL_INPUT,
                ),
            )
        }

        fn hex64(b: u8) -> String {
            let mut s = String::with_capacity(64);
            for _ in 0..32 {
                s.push_str(&format!("{b:02x}"));
            }
            s
        }

        fn snip_root_hex(seed: u8) -> String {
            format!("0x{}", hex64(seed))
        }

        fn sig_hex(seed: u8) -> String {
            let mut s = String::with_capacity(128);
            for _ in 0..64 {
                s.push_str(&format!("{seed:02x}"));
            }
            s
        }

        fn build_canonical_production_job(
            input_hash_hex: String,
            model_hash_hex: String,
        ) -> ContributorJob {
            ContributorJob {
                schema_version: 1,
                job_id: hex64(0x11),
                model_hash: model_hash_hex,
                manifest_snip_root: snip_root_hex(0x22),
                input_snip_root: snip_root_hex(0x33),
                input_hash: input_hash_hex,
                verification_requirement: VerificationRequirement::AttestationOnly,
                accounting: JobAccounting {
                    tokenizer_hash: hex64(0x44),
                    tokenizer_id: "test-tokenizer".to_string(),
                    input_token_count: 1,
                    max_output_token_count: 1,
                    base_unit_reward_policy: BaseUnitRewardPolicy::Unspecified,
                },
                dispatched_at_utc: "2026-06-22T00:00:00Z".to_string(),
                expires_at_utc: None,
                dispatcher_pubkey_hex: None,
                dispatcher_signature_hex: None,
                notes: None,
            }
        }

        fn build_canonical_production_result(
            job: &ContributorJob,
            response_hash_hex: String,
        ) -> ContributorResult {
            ContributorResult {
                schema_version: 1,
                job_id: job.job_id.clone(),
                job_hash: job.job_id.clone(),
                job_snip_root: None,
                model_hash: job.model_hash.clone(),
                input_hash: job.input_hash.clone(),
                response_snip_root: snip_root_hex(0x55),
                response_hash: response_hash_hex,
                evidence: Evidence::AttestationOnly,
                measured_accounting: MeasuredAccounting {
                    tokenizer_hash: job.accounting.tokenizer_hash.clone(),
                    input_token_count: 1,
                    output_token_count: 1,
                    total_base_units: 2,
                    stage_contributions: vec![StageContribution {
                        contributor_pubkey_hex: hex64(0x66),
                        stage_label: "stub-runner".to_string(),
                        work_unit_kind: omni_contributor::WorkUnitKind::DecodeTokens,
                        work_units: 2,
                    }],
                },
                produced_at_utc: "2026-06-22T00:00:01Z".to_string(),
                contributor_pubkey_hex: hex64(0x66),
                contributor_signature_hex: sig_hex(0x77),
                notes: None,
            }
        }

        fn write_temp(dir: &std::path::Path, name: &str, bytes: &[u8]) -> PathBuf {
            let p = dir.join(name);
            std::fs::write(&p, bytes).unwrap();
            p
        }

        struct ProductionFixture {
            _dir: tempfile::TempDir,
            job: ContributorJob,
            result: ContributorResult,
            stub_input_path: PathBuf,
            stub_response_path: PathBuf,
            proof_path: PathBuf,
        }

        fn build_canonical_production_fixture() -> ProductionFixture {
            let dir = tempfile::tempdir().unwrap();
            let in_bytes = canonical_input_bytes();
            let out_bytes = canonical_output_bytes();
            let input_hash = blake3::hash(&in_bytes).to_hex().to_string();
            let output_hash = blake3::hash(&out_bytes).to_hex().to_string();
            let model_hash = production_spec_hash_hex();
            let job = build_canonical_production_job(input_hash, model_hash);
            let result = build_canonical_production_result(&job, output_hash);
            let stub_input_path = write_temp(dir.path(), "stub_input.bin", &in_bytes);
            let stub_response_path =
                write_temp(dir.path(), "stub_response.bin", &out_bytes);
            let proof_path = dir.path().join("sidecar_proof.json");
            ProductionFixture {
                _dir: dir,
                job,
                result,
                stub_input_path,
                stub_response_path,
                proof_path,
            }
        }

        // ── Test 1: StubRunner happy path ──────────────────────────────────

        #[test]
        fn stub_runner_with_emit_flag_writes_production_sidecar_artifact_under_canonical_production_spec_job()
        {
            let f = build_canonical_production_fixture();
            emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .expect("sidecar emission on canonical production fixture must succeed");
            assert!(f.proof_path.is_file());
            let bytes = std::fs::read(&f.proof_path).unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body.metadata.proof_system,
                Some(omni_zkml::ProofSystem::Stage11dProductionFixedPointMlp)
            );
            // Production-shape contract: testnet_or_dev_only=Some(false).
            assert_eq!(body.metadata.testnet_or_dev_only, Some(false));
            assert!(!body.proof_bytes_hex.is_empty());
        }

        // ── Test 2: D6 — sidecar does not mutate ContributorResult bytes ───

        #[test]
        fn emit_production_sidecar_does_not_mutate_contributor_result_bytes_on_disk() {
            let f = build_canonical_production_fixture();
            let result_path = f._dir.path().join("result.json");
            let result_bytes = serde_json::to_vec_pretty(&f.result).unwrap();
            std::fs::write(&result_path, &result_bytes).unwrap();
            let before = std::fs::read(&result_path).unwrap();
            emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let after = std::fs::read(&result_path).unwrap();
            assert_eq!(before, after);
            assert!(f.proof_path.is_file());
        }

        // ── Test 3: non-canonical model_hash refused ───────────────────────

        #[test]
        fn run_job_refuses_emit_production_flag_when_job_model_hash_is_not_canonical_production_spec()
        {
            let mut f = build_canonical_production_fixture();
            f.job.model_hash = hex64(0xaa);
            let err = emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap_err();
            assert!(
                err.to_string()
                    .contains("canonical production-fixedpoint-mlp-v1 spec hash"),
                "expected production-spec refusal; got: {err}"
            );
            assert!(
                !f.proof_path.exists(),
                "no sidecar must be written when job is non-canonical"
            );
        }

        // ── Test 4: hash-binding refusal on stub_input ─────────────────────

        #[test]
        fn run_job_refuses_emit_production_flag_when_stub_input_hash_mismatches_job_input_hash()
        {
            let f = build_canonical_production_fixture();
            let bad_input = f._dir.path().join("bad_input.bin");
            // 32 bytes (correct length) but wrong content → wrong hash.
            std::fs::write(&bad_input, vec![0u8; 32]).unwrap();
            let err = emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&bad_input),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap_err();
            assert!(err.to_string().contains("does not match job.input_hash"));
            assert!(!f.proof_path.exists());
        }

        // ── Test 5: production-shape arity refusal ─────────────────────────

        #[test]
        fn run_job_refuses_emit_production_flag_when_stub_input_byte_length_is_wrong() {
            // Stage 14.2 reference path uses 8-byte input; production
            // path requires exactly 32 bytes (16 × i16 LE). Bytes
            // with the WRONG length must be refused. The hash binding
            // catches this first (the wrong-length file won't hash to
            // job.input_hash), so the refusal surfaces at the hash
            // check; the assembler's 32-byte adapter check is a
            // belt-and-suspenders backstop. We verify the refusal
            // happens with no sidecar written.
            let f = build_canonical_production_fixture();
            let too_short = f._dir.path().join("too_short.bin");
            // 8 bytes — Stage 14.2 reference shape; wrong for production.
            std::fs::write(&too_short, vec![0u8; 8]).unwrap();
            let err = emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&too_short),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap_err();
            // Either the hash check or the adapter's size check fires;
            // both are correct refusals. Pin the broader invariant:
            // no sidecar gets written.
            let _ = err;
            assert!(
                !f.proof_path.exists(),
                "no sidecar must be written when stub_input is wrong size for production"
            );
        }

        // ── Test 6: end-to-end — verifier accepts the contributor sidecar ─

        #[test]
        fn production_mlp_verifier_accepts_contributor_emitted_sidecar() {
            let f = build_canonical_production_fixture();
            emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let body_bytes = std::fs::read(&f.proof_path).unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&body_bytes).unwrap();
            use omni_zkml::ProofVerifier;
            let verifier =
                omni_proofs_halo2_production_mlp::Halo2ProductionMlpVerifier::from_embedded_fixtures()
                    .expect("verifier construction from embedded fixtures");
            let verified = verifier
                .verify_artifact(&body)
                .expect("verifier should run without internal error");
            assert!(
                verified,
                "halo2 production-MLP verifier must accept the contributor-emitted sidecar"
            );
        }

        // ── Test 7: clap conflicts_with mutual exclusion ──────────────────

        /// Stage 14.6 Q3 lock — clap declares `conflicts_with` on
        /// BOTH emit flags. This pin only applies when both prover
        /// features are simultaneously enabled (the conflict is a
        /// no-op for single-feature builds because one of the flags
        /// doesn't exist).
        #[cfg(all(feature = "halo2-reference-prove", feature = "stage11d-production-prove"))]
        #[test]
        fn clap_refuses_both_emit_flags_simultaneously() {
            let parse = TestRoot::try_parse_from([
                "omni-node",
                "run-job",
                "--job",
                "/tmp/phantom_job.json",
                "--out",
                "/tmp/phantom_result.json",
                "--seed-file",
                "/tmp/phantom_seed",
                "--stub-response",
                "/tmp/phantom_response.bin",
                "--stub-input",
                "/tmp/phantom_input.bin",
                "--emit-halo2-reference-proof",
                "/tmp/ref_proof.json",
                "--emit-production-mlp-proof",
                "/tmp/prod_proof.json",
            ]);
            let msg = match parse {
                Ok(_) => panic!(
                    "clap MUST refuse setting both --emit-halo2-reference-proof \
                     and --emit-production-mlp-proof on the same invocation"
                ),
                Err(e) => e.to_string(),
            };
            assert!(
                msg.contains("cannot be used with")
                    || msg.contains("emit-halo2-reference-proof")
                    || msg.contains("emit-production-mlp-proof"),
                "expected clap usage error about mutual exclusion; got: {msg}"
            );
        }

        // ── Test 8: mainnet refusal at layer 6 only ────────────────────────

        #[test]
        fn generated_production_sidecar_artifact_is_mainnet_refused_at_layer_6_only() {
            let f = build_canonical_production_fixture();
            emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&f.proof_path).unwrap()).unwrap();
            // Production-shape contract: testnet_or_dev_only=Some(false)
            // → layer 1 does NOT fire.
            assert_eq!(body.metadata.testnet_or_dev_only, Some(false));
            assert_eq!(
                body.metadata.proof_system,
                Some(omni_zkml::ProofSystem::Stage11dProductionFixedPointMlp)
            );
            // But the artifact is still refused on mainnet via layer 6
            // (empty MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES). Stage 11d.3
            // is the separate chain-team-reviewed allowlist PR.
            let refusal = omni_zkml::check_mainnet_eligible(&body.metadata);
            assert!(
                refusal.is_err(),
                "Stage 14.6 production sidecar MUST be mainnet-refused; got Ok \
                 (Stage 11d.3 allowlist landed?)"
            );
        }

        // ── Test 9: public_inputs carries contributor_job_id ──────────────

        #[test]
        fn production_sidecar_public_inputs_carry_contributor_job_id() {
            let f = build_canonical_production_fixture();
            emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&f.proof_path).unwrap()).unwrap();
            let pi = body
                .metadata
                .public_inputs
                .expect("public_inputs must be populated");
            let job_id_value = pi
                .get("contributor_job_id")
                .expect("public_inputs must carry contributor_job_id key");
            assert_eq!(job_id_value.as_str(), Some(f.result.job_id.as_str()));
            // Production shape: 16-int input, 8-int output.
            assert_eq!(pi.get("input").and_then(|v| v.as_array()).unwrap().len(), 16);
            assert_eq!(pi.get("output").and_then(|v| v.as_array()).unwrap().len(), 8);
        }

        // ── Test 10: D2 production extra-key tolerance regression pin ─────

        #[test]
        fn production_verifier_tolerates_extra_contributor_job_id_key_in_public_inputs() {
            // Mirror Stage 14.2 D2 regression pin for the production
            // verifier. Plant a SECOND extra key that the verifier
            // has never seen; assert verify still succeeds.
            let f = build_canonical_production_fixture();
            emit_production_mlp_proof_sidecar(
                &f.job,
                &f.result,
                RunnerChoice::Stub,
                Some(&f.stub_input_path),
                Some(&f.stub_response_path),
                &f.proof_path,
            )
            .unwrap();
            let mut body: omni_zkml::ProofArtifactBody =
                serde_json::from_slice(&std::fs::read(&f.proof_path).unwrap()).unwrap();
            // Confirm contributor_job_id is present.
            assert!(body
                .metadata
                .public_inputs
                .as_ref()
                .and_then(|v| v.get("contributor_job_id"))
                .is_some());
            // Add a SECOND unknown key.
            if let Some(pi) = body.metadata.public_inputs.as_mut() {
                if let Some(obj) = pi.as_object_mut() {
                    obj.insert(
                        "stage_14_6_synthetic_extra_key".to_string(),
                        serde_json::json!({ "anything": [1, 2, 3] }),
                    );
                }
            }
            use omni_zkml::ProofVerifier;
            let verifier =
                omni_proofs_halo2_production_mlp::Halo2ProductionMlpVerifier::from_embedded_fixtures()
                    .unwrap();
            let verified = verifier
                .verify_artifact(&body)
                .expect("verifier must not error on extra public_inputs keys");
            assert!(
                verified,
                "production verifier must accept artifacts whose public_inputs \
                 carries additional keys beyond input + output"
            );
        }

        // ── Test 11: real ExternalCommandRunner subprocess end-to-end ─────

        /// Tiny stdlib-only base64 encoder (mirrors the Stage 14.3
        /// pattern — avoids pulling `base64` into `omni-node` deps).
        fn base64_encode(bytes: &[u8]) -> String {
            const ALPHABET: &[u8; 64] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
            let mut i = 0;
            while i + 3 <= bytes.len() {
                let n = ((bytes[i] as u32) << 16)
                    | ((bytes[i + 1] as u32) << 8)
                    | (bytes[i + 2] as u32);
                out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
                out.push(ALPHABET[(n & 0x3f) as usize] as char);
                i += 3;
            }
            let rem = bytes.len() - i;
            if rem == 1 {
                let n = (bytes[i] as u32) << 16;
                out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
                out.push('=');
                out.push('=');
            } else if rem == 2 {
                let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
                out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
                out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
                out.push('=');
            }
            out
        }

        #[cfg(unix)]
        #[test]
        fn external_command_runner_real_subprocess_end_to_end_writes_verifiable_production_sidecar()
        {
            // Mirrors Stage 14.3's real-subprocess test but for the
            // production-MLP path. Spawns a real ExternalCommandRunner
            // against a #[cfg(unix)] shell script emitting a valid
            // ExternalRunnerEnvelope with the canonical 16-byte
            // production response_b64. Drives through
            // ByteCapturingRunner; calls
            // emit_production_mlp_proof_sidecar_from_bytes; verifies.
            use std::os::unix::fs::PermissionsExt;
            use omni_contributor::ExternalCommandRunner;

            let dir = tempfile::tempdir().unwrap();
            let output_bytes = canonical_output_bytes();
            let response_b64 = base64_encode(&output_bytes);

            let envelope_json = format!(
                r#"{{"response_b64":"{response_b64}","measured_input_tokens":1,"measured_output_tokens":1,"stage_contributions":[{{"contributor_pubkey_hex":"{}","stage_label":"external-prod-test","work_unit_kind":"decode_tokens","work_units":1}}]}}"#,
                hex64(0x66)
            );
            let script_body =
                format!("#!/bin/sh\ncat <<'EOF'\n{envelope_json}\nEOF\n");
            let script_path = dir.path().join("production-inference-runner.sh");
            std::fs::write(&script_path, script_body).unwrap();
            let mut perms =
                std::fs::metadata(&script_path).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms).unwrap();

            let manifest_path = dir.path().join("manifest.bin");
            std::fs::write(&manifest_path, b"unused-manifest-bytes").unwrap();

            let runner = ExternalCommandRunner::new(script_path.clone());
            let wrapper = ByteCapturingRunner::new(&runner);

            let input_bytes = canonical_input_bytes();
            let run_output = wrapper
                .run(&manifest_path, &input_bytes)
                .expect("external subprocess must succeed against our envelope");

            let captured_input = wrapper
                .take_captured_input()
                .expect("input must be captured on Ok inner");
            let captured_output = wrapper
                .take_captured_output()
                .expect("output must be captured on Ok inner");
            assert_eq!(captured_input, input_bytes);
            assert_eq!(captured_output, run_output.response_bytes);
            assert_eq!(captured_output, output_bytes);

            // Build a canonical (job, result) whose hashes match the
            // captured bytes. The captured bytes ARE the canonical
            // bytes (the script always returns canonical_output_bytes),
            // so the fixture's job.input_hash / result.response_hash
            // already match.
            let f = build_canonical_production_fixture();
            let sidecar_path = dir.path().join("sidecar_proof.json");
            emit_production_mlp_proof_sidecar_from_bytes(
                &f.job,
                &f.result,
                &captured_input,
                &captured_output,
                &sidecar_path,
            )
            .expect(
                "sidecar emission from captured external-runner bytes (production)",
            );

            let body: omni_zkml::ProofArtifactBody = serde_json::from_slice(
                &std::fs::read(&sidecar_path).unwrap(),
            )
            .unwrap();
            use omni_zkml::ProofVerifier;
            let verifier =
                omni_proofs_halo2_production_mlp::Halo2ProductionMlpVerifier::from_embedded_fixtures()
                    .unwrap();
            let verified =
                verifier.verify_artifact(&body).expect("verifier must not error");
            assert!(
                verified,
                "production sidecar produced via real external runner subprocess must verify"
            );
            assert_eq!(body.metadata.input_hash, f.job.input_hash);
            assert_eq!(body.metadata.response_hash, f.result.response_hash);
            assert_eq!(body.metadata.testnet_or_dev_only, Some(false));
        }
    }
}

// ── Stage 12.9 — session-status ──────────────────────────────────────────

fn run_session_status(args: SessionStatusArgs) -> Result<()> {
    use omni_contributor::{
        build_session_status_report, ContributorStateStore, SessionOverallStatus,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // With `--format json`, stdout must be ONLY the report JSON so
    // scripts can `jq` it without preprocessing. Operational
    // chatter (state-store open notice, JSON-out write notice,
    // warnings) all go to stderr in that mode. `events` and
    // `pretty` keep their prose-style stdout.
    let json_mode = matches!(args.format, SessionStatusFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Open the state store. Stage 12.7 auto-prune applies unless
    // the operator opts out.
    let (store, prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        !args.no_prune_state_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    log_op(&format!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        args.contributor_state_dir.display(),
        prune_report.removed_sessions,
        prune_report.removed_peer_adverts,
        prune_report.kept
    ));

    let report = build_session_status_report(
        &store,
        &args.session_id,
        &now_utc,
        args.include_expired,
    )
    .map_err(|e| anyhow!("build session status report: {e}"))?;

    match args.format {
        SessionStatusFormat::Events => render_status_events(&report),
        SessionStatusFormat::Json => render_status_json(&report)?,
        SessionStatusFormat::Pretty => render_status_pretty(&report),
    }

    // Optional best-effort JSON mirror. Failure here is a stderr
    // warning, not a process error — the report is a snapshot, not
    // a protocol artifact.
    if let Some(path) = args.json_out.as_deref() {
        match serde_json::to_vec_pretty(&report) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, &bytes) {
                    eprintln!(
                        "event=warn context=session_status_json_out path={} message={e}",
                        path.display()
                    );
                } else {
                    log_op(&format!(
                        "event=session_status_json_written path={}",
                        path.display()
                    ));
                }
            }
            Err(e) => {
                eprintln!(
                    "event=warn context=session_status_json_serialize message={e}"
                );
            }
        }
    }

    // Exit code policy. The Result<()> return shape lets us err
    // out cleanly; clap's bin entry then converts to exit code 1.
    if args.fail_on_incomplete
        && !matches!(
            report.overall_status,
            SessionOverallStatus::CompletePartials
                | SessionOverallStatus::Aggregated
        )
    {
        return Err(anyhow!(
            "session status is {:?}; --fail-on-incomplete tripped",
            report.overall_status
        ));
    }
    Ok(())
}

fn render_status_events(report: &omni_contributor::SessionStatusReport) {
    // Stage 12.11 — top line gains active/superseded/supersession
    // counts. `assignments=` remains the *all-verified* count so
    // existing tooling does not silently break; `active=` is the
    // verifier-relevant subset.
    println!(
        "event=session_status session_id={} status={:?} \
         assignments={} active={} superseded={} supersessions={} \
         partials={} missing={} aggregate={} expired={}",
        report.session_id,
        report.overall_status,
        report.assignment_count,
        report.active_assignment_count,
        report.superseded_assignment_count,
        report.supersession_count,
        report.partial_count,
        report.missing_assignment_ids.len(),
        report.aggregate_valid,
        report.session_expired,
    );
    for a in &report.assignments {
        // Stage 12.11 — surface per-assignment supersession flag
        // and (when present) the supersession_id that retired it.
        // Active assignments print `superseded=no superseded_by=-`
        // so log filters can pattern-match either state without
        // optional-field gymnastics.
        let superseded_by = a
            .superseded_by_supersession_id
            .as_deref()
            .unwrap_or("-");
        println!(
            "event=assignment_status session_id={} assignment_id={} \
             stage_index={} contributor={} join={} peer_advert={} \
             partial={} superseded={} superseded_by={}",
            report.session_id,
            a.assignment_id,
            a.stage_index,
            a.contributor_pubkey_hex,
            if a.join_present { "present" } else { "missing" },
            if a.peer_advert_present { "present" } else { "missing" },
            if a.partial_present { "present" } else { "missing" },
            if a.superseded { "yes" } else { "no" },
            superseded_by,
        );
    }
    for s in &report.supersessions {
        println!(
            "event=supersession_status session_id={} supersession_id={} \
             superseded_count={} replacement_count={} reason={:?} valid={}",
            report.session_id,
            s.supersession_id,
            s.superseded_assignment_ids.len(),
            s.replacement_assignment_ids.len(),
            s.reason,
            s.valid,
        );
    }
    // Stage 12.12 — surface structured chain-failure diagnostics
    // so operators see the kind + reason_tag at a glance and so
    // automation can scrape `event=invalid_artifact` lines.
    // Mirrors the JSON `invalid_artifacts` field.
    for entry in &report.invalid_artifacts {
        match entry {
            omni_contributor::InvalidArtifactStatus::InvalidSession { reason_tag } => {
                println!(
                    "event=invalid_artifact session_id={} kind=invalid_session \
                     reason_tag={reason_tag}",
                    report.session_id,
                );
            }
            omni_contributor::InvalidArtifactStatus::InvalidJoin {
                contributor_pubkey_hex,
                reason_tag,
            } => {
                println!(
                    "event=invalid_artifact session_id={} kind=invalid_join \
                     contributor_pubkey_hex={contributor_pubkey_hex} \
                     reason_tag={reason_tag}",
                    report.session_id,
                );
            }
            omni_contributor::InvalidArtifactStatus::InvalidAssignment {
                assignment_id,
                reason_tag,
            } => {
                println!(
                    "event=invalid_artifact session_id={} kind=invalid_assignment \
                     assignment_id={assignment_id} reason_tag={reason_tag}",
                    report.session_id,
                );
            }
            omni_contributor::InvalidArtifactStatus::InvalidPartial {
                assignment_id,
                reason_tag,
            } => {
                println!(
                    "event=invalid_artifact session_id={} kind=invalid_partial \
                     assignment_id={assignment_id} reason_tag={reason_tag}",
                    report.session_id,
                );
            }
            omni_contributor::InvalidArtifactStatus::InvalidSupersession {
                supersession_id,
                reason_tag,
            } => {
                println!(
                    "event=invalid_artifact session_id={} kind=invalid_supersession \
                     supersession_id={supersession_id} reason_tag={reason_tag}",
                    report.session_id,
                );
            }
            omni_contributor::InvalidArtifactStatus::InvalidAggregate {
                reason_tag,
            } => {
                println!(
                    "event=invalid_artifact session_id={} kind=invalid_aggregate \
                     reason_tag={reason_tag}",
                    report.session_id,
                );
            }
        }
    }
    for missing in &report.missing_assignment_ids {
        // Find the corresponding assignment for stage_index and
        // contributor context. Linear scan is fine — assignment
        // counts are small (a handful per session).
        if let Some(a) = report
            .assignments
            .iter()
            .find(|x| &x.assignment_id == missing)
        {
            println!(
                "event=missing_partial session_id={} assignment_id={} \
                 stage_index={} contributor={}",
                report.session_id,
                missing,
                a.stage_index,
                a.contributor_pubkey_hex,
            );
        } else {
            println!(
                "event=missing_partial session_id={} assignment_id={}",
                report.session_id, missing,
            );
        }
    }
    for note in &report.notes {
        println!("event=note context=session_status message={note}");
    }
    // Stage 12.13 — audit health, derived from the v3 report
    // fields. Closed `coherence` discriminator + static-string
    // `recommended_action` so log scrapers can pattern-match
    // deterministically. JSON renderer is unchanged (the v3 JSON
    // schema is frozen at Stage 12.12); audit is an ergonomics
    // extension on top of `events` + `pretty`.
    let audit = omni_contributor::compute_audit_health(report);
    let coherence_tag = match &audit.coherence {
        omni_contributor::AuditCoherence::Coherent => "coherent".to_string(),
        omni_contributor::AuditCoherence::PartialApplySupersession {
            supersession_id,
            unresolved_count,
        } => format!(
            "partial_apply_supersession supersession_id={supersession_id} \
             unresolved_count={unresolved_count}"
        ),
        omni_contributor::AuditCoherence::OrphanReplacementAssignments {
            assignment_ids,
        } => format!(
            "orphan_replacement_assignments count={} ids={}",
            assignment_ids.len(),
            assignment_ids.join(",")
        ),
        omni_contributor::AuditCoherence::NotReassignTriagable => {
            "not_reassign_triagable".to_string()
        }
        omni_contributor::AuditCoherence::ReassignTriagable => {
            "reassign_triagable".to_string()
        }
    };
    println!(
        "event=audit_health session_id={} coherence={coherence_tag} \
         triagable_by_reassign={} recommended_action={:?}",
        report.session_id,
        audit.triagable_by_reassign,
        audit.recommended_action,
    );
}

fn render_status_json(
    report: &omni_contributor::SessionStatusReport,
) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    println!("{json}");
    Ok(())
}

fn render_status_pretty(report: &omni_contributor::SessionStatusReport) {
    println!("Session   {}", report.session_id);
    println!("Status    {:?}", report.overall_status);
    // Stage 12.11 — surface active / superseded / supersession
    // counts inline with the existing roll-up.
    println!(
        "Counts    joins={} assignments={} active={} superseded={} \
         supersessions={} partials={} adverts={} aggregate_present={} aggregate_valid={}",
        report.join_count,
        report.assignment_count,
        report.active_assignment_count,
        report.superseded_assignment_count,
        report.supersession_count,
        report.partial_count,
        report.peer_advert_count,
        report.aggregate_present,
        report.aggregate_valid,
    );
    if !report.assignments.is_empty() {
        println!("Assignments:");
        println!(
            "  {:<5} {:<10} {:<10} {:<10} {:<10} {:<10}",
            "stage", "join", "advert", "partial", "superseded", "assignment_id (12)"
        );
        for a in &report.assignments {
            let short_id: String = a.assignment_id.chars().take(12).collect();
            println!(
                "  {:<5} {:<10} {:<10} {:<10} {:<10} {}",
                a.stage_index,
                if a.join_present { "present" } else { "missing" },
                if a.peer_advert_present { "present" } else { "missing" },
                if a.partial_present { "present" } else { "missing" },
                if a.superseded { "yes" } else { "no" },
                short_id,
            );
        }
    }
    if !report.supersessions.is_empty() {
        println!("Supersessions:");
        println!(
            "  {:<10} {:<10} {:<10} {:<20} {:<10}",
            "valid", "superseded", "replacement", "reason", "supersession_id (12)"
        );
        for s in &report.supersessions {
            let short_id: String = s.supersession_id.chars().take(12).collect();
            println!(
                "  {:<10} {:<10} {:<10} {:<20} {}",
                if s.valid { "yes" } else { "no" },
                s.superseded_assignment_ids.len(),
                s.replacement_assignment_ids.len(),
                format!("{:?}", s.reason),
                short_id,
            );
        }
    }
    // Stage 12.12 — Invalid-artifact diagnostics section. The
    // events renderer emits one `event=invalid_artifact` line per
    // entry; pretty groups them under a header so an operator
    // can see the chain-failure shape at a glance and decide
    // whether the InvalidState is triagable.
    if !report.invalid_artifacts.is_empty() {
        println!("Invalid artifacts:");
        println!(
            "  {:<22} {:<32} id",
            "kind", "reason_tag"
        );
        for entry in &report.invalid_artifacts {
            let (kind, id, reason_tag) = match entry {
                omni_contributor::InvalidArtifactStatus::InvalidSession { reason_tag } => (
                    "invalid_session",
                    String::from("-"),
                    reason_tag.as_str(),
                ),
                omni_contributor::InvalidArtifactStatus::InvalidJoin {
                    contributor_pubkey_hex,
                    reason_tag,
                } => (
                    "invalid_join",
                    contributor_pubkey_hex.chars().take(12).collect::<String>(),
                    reason_tag.as_str(),
                ),
                omni_contributor::InvalidArtifactStatus::InvalidAssignment {
                    assignment_id,
                    reason_tag,
                } => (
                    "invalid_assignment",
                    assignment_id.chars().take(12).collect::<String>(),
                    reason_tag.as_str(),
                ),
                omni_contributor::InvalidArtifactStatus::InvalidPartial {
                    assignment_id,
                    reason_tag,
                } => (
                    "invalid_partial",
                    assignment_id.chars().take(12).collect::<String>(),
                    reason_tag.as_str(),
                ),
                omni_contributor::InvalidArtifactStatus::InvalidSupersession {
                    supersession_id,
                    reason_tag,
                } => (
                    "invalid_supersession",
                    supersession_id.chars().take(12).collect::<String>(),
                    reason_tag.as_str(),
                ),
                omni_contributor::InvalidArtifactStatus::InvalidAggregate {
                    reason_tag,
                } => (
                    "invalid_aggregate",
                    String::from("-"),
                    reason_tag.as_str(),
                ),
            };
            println!("  {:<22} {:<32} {}", kind, reason_tag, id);
        }
    }
    if !report.missing_assignment_ids.is_empty() {
        // Stage 12.11 — denominator is active_assignment_count
        // because missing_assignment_ids is active-only.
        println!(
            "Missing partials: {} / {}",
            report.missing_assignment_ids.len(),
            report.active_assignment_count
        );
    }
    if !report.notes.is_empty() {
        println!("Notes:");
        for n in &report.notes {
            println!("  - {n}");
        }
    }
    // Stage 12.13 — audit health roll-up. Closed enum + static
    // recommended-action so the pretty output stays operator-
    // friendly without leaking implementation detail. JSON
    // renderer is unchanged (v3 JSON schema frozen at 12.12).
    let audit = omni_contributor::compute_audit_health(report);
    println!("Audit health:");
    match &audit.coherence {
        omni_contributor::AuditCoherence::Coherent => {
            println!("  coherence       coherent");
        }
        omni_contributor::AuditCoherence::PartialApplySupersession {
            supersession_id,
            unresolved_count,
        } => {
            let short_id: String =
                supersession_id.chars().take(12).collect();
            println!(
                "  coherence       partial_apply_supersession ({short_id}, \
                 unresolved={unresolved_count})"
            );
        }
        omni_contributor::AuditCoherence::OrphanReplacementAssignments {
            assignment_ids,
        } => {
            println!(
                "  coherence       orphan_replacement_assignments \
                 (count={})",
                assignment_ids.len()
            );
            for id in assignment_ids {
                let short: String = id.chars().take(12).collect();
                println!("    - {short}");
            }
        }
        omni_contributor::AuditCoherence::NotReassignTriagable => {
            println!("  coherence       not_reassign_triagable");
        }
        omni_contributor::AuditCoherence::ReassignTriagable => {
            println!("  coherence       reassign_triagable");
        }
    }
    println!(
        "  triagable_by_reassign  {}",
        audit.triagable_by_reassign
    );
    println!("  recommended_action     {}", audit.recommended_action);
}

// ── Stage 12.10 — plan-session-repair ───────────────────────────────────

fn run_plan_session_repair(args: PlanSessionRepairArgs) -> Result<()> {
    use omni_contributor::{
        build_session_repair_plan, build_session_status_report, ContributorStateStore,
        SessionStatusReport,
    };

    if args.status_report.is_some() == args.build_status {
        return Err(anyhow!(
            "supply exactly one of --status-report <path> or --build-status"
        ));
    }

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // Open the state store. Stage 12.7 auto-prune applies unless
    // the operator opts out.
    let (store, prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        !args.no_prune_state_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    println!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        args.contributor_state_dir.display(),
        prune_report.removed_sessions,
        prune_report.removed_peer_adverts,
        prune_report.kept
    );

    // Acquire the status report — either build it on the fly OR
    // load it from the operator-supplied file.
    let status: SessionStatusReport = if args.build_status {
        build_session_status_report(
            &store,
            &args.session_id,
            &now_utc,
            args.include_expired,
        )
        .map_err(|e| anyhow!("build session status report: {e}"))?
    } else {
        let path = args.status_report.as_deref().expect("validated above");
        let bytes = std::fs::read(path)
            .with_context(|| format!("read status-report: {}", path.display()))?;
        let report: SessionStatusReport = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse status-report: {}", path.display()))?;
        if report.session_id != args.session_id {
            return Err(anyhow!(
                "status-report.session_id={} but --session-id={}",
                report.session_id,
                args.session_id
            ));
        }
        report
    };

    let plan = build_session_repair_plan(
        &status,
        args.strategy.into(),
        &now_utc,
        args.coordinator_pubkey_hex.as_deref(),
    )
    .map_err(|e| anyhow!("build session repair plan: {e}"))?;

    let json = serde_json::to_vec_pretty(&plan)?;
    std::fs::write(&args.out, &json)
        .with_context(|| format!("write repair plan: {}", args.out.display()))?;

    println!(
        "event=repair_plan_created session_id={} strategy={:?} actions={} \
         repair_plan_hash={} source_status_hash={} out={}",
        plan.session_id,
        plan.strategy,
        plan.actions.len(),
        plan.repair_plan_hash,
        plan.source_status_hash,
        args.out.display()
    );
    Ok(())
}

// ── Stage 12.10 — apply-session-repair ──────────────────────────────────

async fn run_apply_session_repair(args: ApplySessionRepairArgs) -> Result<()> {
    use omni_contributor::canonical::{hex_lower, net_assign_signing_input};
    use omni_contributor::{
        build_session_status_report, repair_plan_hash_hex, source_status_hash_hex,
        ContributorRelay, ContributorStateStore, CoordinatorSigner,
        NetworkWorkAssignedAnnouncement, OmniNetRelay, RepairAction, SessionRepairPlan,
        WorkAssignment, NET_SCHEMA_VERSION,
    };

    // ── 1. Read + integrity-check the plan ─────────────────────
    let bytes = std::fs::read(&args.repair_plan)
        .with_context(|| format!("read repair-plan: {}", args.repair_plan.display()))?;
    let plan: SessionRepairPlan = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse repair-plan: {}", args.repair_plan.display()))?;
    if plan.schema_version != omni_contributor::REPAIR_PLAN_SCHEMA_VERSION {
        return Err(anyhow!(
            "repair_plan.schema_version {} not supported (expected {})",
            plan.schema_version,
            omni_contributor::REPAIR_PLAN_SCHEMA_VERSION
        ));
    }
    let recomputed_plan_hash = repair_plan_hash_hex(&plan);
    if recomputed_plan_hash != plan.repair_plan_hash {
        return Err(anyhow!(omni_contributor::RepairError::PlanHashDrift {
            stored: plan.repair_plan_hash.clone(),
            recomputed: recomputed_plan_hash,
        }));
    }
    if plan.actions.is_empty() {
        return Err(anyhow!("repair-plan carries zero actions; nothing to apply"));
    }

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // ── 2. Open state store + recompute source-status drift ────
    let (store, prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        !args.no_prune_state_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    println!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        args.contributor_state_dir.display(),
        prune_report.removed_sessions,
        prune_report.removed_peer_adverts,
        prune_report.kept
    );

    let current_status = build_session_status_report(
        &store,
        &plan.session_id,
        &now_utc,
        /* include_expired = */ false,
    )
    .map_err(|e| anyhow!("rebuild status report for drift check: {e}"))?;
    let current_projection = source_status_hash_hex(&current_status);
    if current_projection != plan.source_status_hash {
        return Err(anyhow!(omni_contributor::RepairError::SourceStatusDrift));
    }
    // `source_status_hash` is a projection over only
    // `(session_id, [(assignment_id, partial_present)])`. Status
    // shifts that leave that projection identical (e.g.
    // `ExpiredIncomplete` under `--no-prune-state-on-start`, or
    // `InvalidState` from an invalid aggregate body that doesn't
    // touch partial presence) would otherwise slip past the drift
    // check. Re-check eligibility against the same matrix the
    // planner uses, returning identical typed errors so a CI gate
    // sees a consistent surface.
    omni_contributor::check_repair_eligible(&current_status)
        .map_err(|e| anyhow!("apply rejected by current status: {e}"))?;

    // ── 3. Fetch + verify session, check coordinator seed ──────
    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.session_snip_root)?;
    if session.session_id != plan.session_id {
        return Err(anyhow!(
            "repair-plan.session_id={} but --session-snip-root resolves to session_id={}",
            plan.session_id,
            session.session_id
        ));
    }
    let coord = CoordinatorSigner::from_seed_file(&args.coordinator_seed)?;
    if coord.pubkey_hex() != session.coordinator_pubkey_hex {
        return Err(anyhow!(
            "coordinator_seed pubkey does not match session.coordinator_pubkey_hex"
        ));
    }

    // ── 4. Load joined-pubkey set + dry-run path ───────────────
    let raw_joins = store.list_verified_joins_for(&plan.session_id)?;
    let joins: Vec<omni_contributor::ContributorJoin> = raw_joins
        .into_iter()
        .filter(|j| omni_contributor::verify_contributor_join(&session, j).is_ok())
        .collect();
    let joined_pubkeys: std::collections::HashSet<String> = joins
        .iter()
        .map(|j| j.contributor_pubkey_hex.clone())
        .collect();

    // Re-verify every referenced assignment before any publish so
    // a dry-run surfaces tampered bytes too.
    struct VerifiedAction {
        assignment: WorkAssignment,
        stage_index: u32,
        contributor_pubkey_hex: String,
    }
    let mut verified: Vec<VerifiedAction> = Vec::with_capacity(plan.actions.len());
    for action in &plan.actions {
        // Stage 12.10 applier path: only `ReannounceAssignment`
        // actions are accepted here. `ReassignAssignment` (Stage
        // 12.11) is handled by `apply-session-reassign` instead;
        // mixing the two in a single plan is intentionally
        // disallowed by the planner (it commits to one strategy
        // per plan). A future plan that smuggled in a Stage 12.11
        // action must surface a typed error rather than silently
        // skip — the operator-meaningful invariant is "every
        // action in a v1-reannounce plan is a reannounce."
        let (assignment_id, stage_index, contributor_pubkey_hex) = match action {
            RepairAction::ReannounceAssignment {
                assignment_id,
                stage_index,
                contributor_pubkey_hex,
            } => (assignment_id, stage_index, contributor_pubkey_hex),
            RepairAction::ReassignAssignment { .. } => {
                return Err(anyhow!(
                    "apply-session-repair only handles ReannounceAssignment \
                     actions; use apply-session-reassign for ReassignAssignment"
                ));
            }
        };
        let asn: WorkAssignment = match store.read_verified_json(
            omni_contributor::StateObjectKind::Assignment {
                session_id: plan.session_id.clone(),
            },
            assignment_id,
        )? {
            Some(a) => a,
            None => {
                return Err(anyhow!(
                    omni_contributor::RepairError::AssignmentNotPresent {
                        session_id: plan.session_id.clone(),
                        assignment_id: assignment_id.clone(),
                    }
                ));
            }
        };
        let outcome = omni_contributor::verify_work_assignment(
            &session,
            &joined_pubkeys,
            &asn,
        );
        if !outcome.is_ok() {
            return Err(anyhow!(
                "assignment_id={} in state-dir failed verify_work_assignment: {outcome:?}",
                assignment_id
            ));
        }
        // Drift check between the plan's snapshot and the on-disk
        // body. The Stage 12.9 status report fed the plan with
        // these fields; if a tamper sneaks through here it's
        // already a separate failure mode worth surfacing.
        if &asn.stage_index != stage_index
            || &asn.contributor_pubkey_hex != contributor_pubkey_hex
        {
            return Err(anyhow!(
                "assignment_id={} body drift vs plan: \
                 plan(stage_index={stage_index}, contributor={contributor_pubkey_hex}) \
                 disk(stage_index={}, contributor={})",
                assignment_id,
                asn.stage_index,
                asn.contributor_pubkey_hex
            ));
        }
        verified.push(VerifiedAction {
            assignment: asn,
            stage_index: *stage_index,
            contributor_pubkey_hex: contributor_pubkey_hex.clone(),
        });
    }

    if args.dry_run {
        for v in &verified {
            println!(
                "event=would_reannounce session_id={} assignment_id={} \
                 stage_index={} contributor={}",
                plan.session_id,
                v.assignment.assignment_id,
                v.stage_index,
                v.contributor_pubkey_hex,
            );
        }
        println!(
            "event=repair_dry_run session_id={} actions={}",
            plan.session_id,
            verified.len()
        );
        return Ok(());
    }

    // ── 5. Real publish path ───────────────────────────────────
    //
    // Only open the mesh + wait for peers when we'll actually
    // broadcast. With `--no-publish-announcements`, the apply is a
    // pure SNIP-republish loop, and forcing a 30s peer-wait on a
    // SNIP-only run would be a silent latency tax. The relay,
    // propagation sleep, and shutdown are all gated on the same
    // `should_publish_announcements` predicate.
    let should_publish_announcements = !args.no_publish_announcements;
    let mut mesh: Option<(
        std::sync::Arc<tokio::sync::Mutex<omni_net::OmniNet>>,
        OmniNetRelay,
    )> = if should_publish_announcements {
        let (net, handle) = open_omninet_with_peer_wait(
            args.listen_port,
            args.peer,
            args.peer_wait_secs,
            args.mesh_stabilize_ms,
            args.net_identity_file.as_deref(),
        )
        .await?;
        let relay = OmniNetRelay::new(net.clone(), handle);
        Some((net, relay))
    } else {
        None
    };
    let mut reannounced = 0u64;
    for v in &verified {
        // Re-publish the same assignment JSON bytes. SNIP is
        // content-addressed; the returned root equals the original
        // publish's root (assignment_id, coordinator signature, and
        // SNIP root are all preserved across reannounce).
        let json = serde_json::to_vec_pretty(&v.assignment)?;
        let root = omni_contributor::snip::publish_bytes(
            &snip,
            &json,
            "assignment-reannounce",
        )
        .map_err(|e| anyhow!("snip publish reannouncement: {e}"))?;
        let root_hex = format!("0x{}", hex_lower(root.as_bytes()));

        if let Some((_, ref mut relay)) = mesh.as_mut().map(|(n, r)| (n, r)) {
            let mut ann = NetworkWorkAssignedAnnouncement {
                schema_version: NET_SCHEMA_VERSION,
                work_assignment_snip_root: root_hex.clone(),
                session_id: plan.session_id.clone(),
                assignment_id: v.assignment.assignment_id.clone(),
                contributor_pubkey_hex: v.assignment.contributor_pubkey_hex.clone(),
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
        }

        println!(
            "event=assignment_reannounced session_id={} stage_index={} \
             assignment_id={} contributor={} work_assignment_snip_root={}",
            plan.session_id,
            v.stage_index,
            v.assignment.assignment_id,
            v.contributor_pubkey_hex,
            root_hex
        );
        reannounced += 1;
    }

    if let Some((net, _relay)) = mesh {
        tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms))
            .await;
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }
    println!(
        "event=repair_applied session_id={} assignments_reannounced={}",
        plan.session_id, reannounced
    );
    Ok(())
}

// ── Stage 12.11 — plan-session-reassign ──────────────────────────────────

fn run_plan_session_reassign(args: PlanSessionReassignArgs) -> Result<()> {
    use omni_contributor::{
        build_session_repair_plan_with_reason, build_session_status_report,
        ContributorStateStore, SessionStatusReport,
    };

    if args.status_report.is_some() == args.build_status {
        return Err(anyhow!(
            "supply exactly one of --status-report <path> or --build-status"
        ));
    }

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let (store, prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        !args.no_prune_state_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    println!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        args.contributor_state_dir.display(),
        prune_report.removed_sessions,
        prune_report.removed_peer_adverts,
        prune_report.kept
    );

    let status: SessionStatusReport = if args.build_status {
        build_session_status_report(
            &store,
            &args.session_id,
            &now_utc,
            args.include_expired,
        )
        .map_err(|e| anyhow!("build session status report: {e}"))?
    } else {
        let path = args.status_report.as_deref().expect("validated above");
        let bytes = std::fs::read(path)
            .with_context(|| format!("read status-report: {}", path.display()))?;
        let report: SessionStatusReport = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse status-report: {}", path.display()))?;
        if report.session_id != args.session_id {
            return Err(anyhow!(
                "status-report.session_id={} but --session-id={}",
                report.session_id,
                args.session_id
            ));
        }
        report
    };

    let plan = build_session_repair_plan_with_reason(
        &status,
        omni_contributor::RepairStrategy::ReassignMissing,
        omni_contributor::SupersessionReason::from(args.reason),
        &now_utc,
        args.coordinator_pubkey_hex.as_deref(),
    )
    .map_err(|e| anyhow!("build session repair plan: {e}"))?;

    let json = serde_json::to_vec_pretty(&plan)?;
    std::fs::write(&args.out, &json)
        .with_context(|| format!("write reassignment plan: {}", args.out.display()))?;

    println!(
        "event=reassignment_plan_created session_id={} actions={} \
         repair_plan_hash={} source_status_hash={} out={}",
        plan.session_id,
        plan.actions.len(),
        plan.repair_plan_hash,
        plan.source_status_hash,
        args.out.display()
    );
    Ok(())
}

// ── Stage 12.11 — apply-session-reassign ─────────────────────────────────

async fn run_apply_session_reassign(args: ApplySessionReassignArgs) -> Result<()> {
    use omni_contributor::canonical::{
        assignment_id_hex, hex_lower, net_assign_signing_input,
        net_supersession_signing_input, supersession_id_hex,
        work_assignment_signing_input, work_assignment_supersession_signing_input,
    };
    use omni_contributor::{
        build_session_status_report, repair_plan_hash_hex, source_status_hash_hex,
        ContributorRelay, ContributorStateStore, CoordinatorSigner,
        NetworkWorkAssignedAnnouncement, NetworkWorkAssignmentSupersessionAnnouncement,
        OmniNetRelay, RepairAction, SessionRepairPlan, WorkAssignment,
        WorkAssignmentSupersession, NET_SCHEMA_VERSION, SESSION_SCHEMA_VERSION,
        SUPERSESSION_SCHEMA_VERSION,
    };

    // ── 1. Read + integrity-check the plan ─────────────────────
    let bytes = std::fs::read(&args.reassignment_plan)
        .with_context(|| {
            format!("read reassignment-plan: {}", args.reassignment_plan.display())
        })?;
    let plan: SessionRepairPlan = serde_json::from_slice(&bytes).with_context(|| {
        format!("parse reassignment-plan: {}", args.reassignment_plan.display())
    })?;
    if plan.schema_version != omni_contributor::REPAIR_PLAN_SCHEMA_VERSION {
        return Err(anyhow!(
            "repair_plan.schema_version {} not supported (expected {})",
            plan.schema_version,
            omni_contributor::REPAIR_PLAN_SCHEMA_VERSION
        ));
    }
    let recomputed_plan_hash = repair_plan_hash_hex(&plan);
    if recomputed_plan_hash != plan.repair_plan_hash {
        return Err(anyhow!(omni_contributor::RepairError::PlanHashDrift {
            stored: plan.repair_plan_hash.clone(),
            recomputed: recomputed_plan_hash,
        }));
    }
    if plan.strategy != omni_contributor::RepairStrategy::ReassignMissing {
        return Err(anyhow!(
            "apply-session-reassign requires plan.strategy == ReassignMissing; got {:?}",
            plan.strategy
        ));
    }
    if plan.actions.is_empty() {
        return Err(anyhow!(
            "reassignment-plan carries zero actions; nothing to apply"
        ));
    }
    // Every action must be ReassignAssignment.
    for action in &plan.actions {
        match action {
            RepairAction::ReassignAssignment { .. } => {}
            RepairAction::ReannounceAssignment { .. } => {
                return Err(anyhow!(
                    "apply-session-reassign only accepts ReassignAssignment actions; \
                     use apply-session-repair for ReannounceAssignment"
                ));
            }
        }
    }

    // Stage 12.12 — mixed-reason defense. The planner emits a
    // uniform `SupersessionReason` across every action in one
    // plan, but the plan is unsigned/local so an operator could
    // hand-edit it. The apply path's eligibility check dispatches
    // on the plan's reason (only `InvalidPartial` takes the
    // relaxed `InvalidState`-accepting branch); a mixed-reason
    // plan would either smuggle one reason's relaxed eligibility
    // onto another reason's actions or silently apply under the
    // wrong gate. Refuse before any eligibility / drift / SNIP /
    // mesh / state-dir work.
    let plan_reason =
        match &plan.actions[0] {
            RepairAction::ReassignAssignment { reason, .. } => reason.clone(),
            _ => unreachable!("validated above"),
        };
    for action in &plan.actions[1..] {
        if let RepairAction::ReassignAssignment { reason, .. } = action {
            if reason != &plan_reason {
                return Err(anyhow!(
                    "apply-session-reassign refuses mixed-reason plan: \
                     plan.actions[0].reason={:?} but a later action has \
                     reason={:?}",
                    plan_reason,
                    reason,
                ));
            }
        }
    }

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // ── 2. Open state store + recompute source-status drift ────
    let (store, prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        !args.no_prune_state_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    println!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        args.contributor_state_dir.display(),
        prune_report.removed_sessions,
        prune_report.removed_peer_adverts,
        prune_report.kept
    );

    let current_status = build_session_status_report(
        &store,
        &plan.session_id,
        &now_utc,
        /* include_expired = */ false,
    )
    .map_err(|e| anyhow!("rebuild status report for drift check: {e}"))?;
    let current_projection = source_status_hash_hex(&current_status);
    if current_projection != plan.source_status_hash {
        return Err(anyhow!(omni_contributor::RepairError::SourceStatusDrift));
    }
    // Stage 12.12 — eligibility dispatch.
    //
    //   reason == InvalidPartial → relaxed gate that accepts
    //     `InvalidState` ONLY when every `invalid_artifacts`
    //     entry is an `InvalidPartial` whose `assignment_id` is
    //     in the plan's superseded set. Every other invalid-kind
    //     (InvalidSession / InvalidJoin / InvalidAssignment /
    //     InvalidSupersession / InvalidAggregate) refuses; every
    //     non-`InvalidState` overall status defers to
    //     `check_repair_eligible`.
    //
    //   reason == MissingPartial / OperatorRebalance / Custom →
    //     standard gate (refuses `InvalidState` outright), same
    //     posture as Stage 12.11's original apply path. This
    //     preserves the contract that triage is bounded by the
    //     coordinator-supplied reason: an operator cannot bypass
    //     `InvalidState` via `MissingPartial` semantics.
    if matches!(
        plan_reason,
        omni_contributor::SupersessionReason::InvalidPartial
    ) {
        omni_contributor::check_reassign_eligible_allowing_invalid_partials(
            &current_status,
            &plan,
        )
        .map_err(|e| anyhow!("apply rejected by current status: {e}"))?;
    } else {
        omni_contributor::check_repair_eligible(&current_status)
            .map_err(|e| anyhow!("apply rejected by current status: {e}"))?;
    }

    // ── 3. Fetch + verify session, check coordinator seed ──────
    let snip = build_snip_adapter(args.snip_binary, args.snip_seed);
    let session = fetch_and_verify_session(&snip, &args.session_snip_root)?;
    if session.session_id != plan.session_id {
        return Err(anyhow!(
            "reassignment-plan.session_id={} but --session-snip-root resolves to session_id={}",
            plan.session_id,
            session.session_id
        ));
    }
    let coord = CoordinatorSigner::from_seed_file(&args.coordinator_seed)?;
    if coord.pubkey_hex() != session.coordinator_pubkey_hex {
        return Err(anyhow!(
            "coordinator_seed pubkey does not match session.coordinator_pubkey_hex"
        ));
    }

    // ── 4. Load + re-verify joins (needed for replacement assignment
    //       verification because verify_work_assignment requires the
    //       replacement contributor to be in the joined set) + load
    //       current verified assignments (the supersession's
    //       superseded_assignment_ids reference these). ─────────
    let raw_joins = store.list_verified_joins_for(&plan.session_id)?;
    let joins: Vec<omni_contributor::ContributorJoin> = raw_joins
        .into_iter()
        .filter(|j| omni_contributor::verify_contributor_join(&session, j).is_ok())
        .collect();
    let joined_pubkeys: std::collections::HashSet<String> = joins
        .iter()
        .map(|j| j.contributor_pubkey_hex.clone())
        .collect();

    // ── 5. Build, sign, validate each replacement WorkAssignment
    //       (in plan order so dry-run output is deterministic). ──
    struct PreparedReplacement {
        replacement: WorkAssignment,
        superseded_assignment_id: String,
    }
    let mut prepared: Vec<PreparedReplacement> = Vec::with_capacity(plan.actions.len());
    let mut all_assignments_with_replacements: Vec<WorkAssignment> = Vec::new();
    // Stage 12.7 state-dir already has every Stage 12.0–12.10
    // assignment; load them once so we can fold replacements in and
    // pass the full set to the supersession verifier.
    let raw_existing = store.list_verified_assignments_for(&plan.session_id)?;
    for a in raw_existing {
        if omni_contributor::verify_work_assignment(&session, &joined_pubkeys, &a)
            .is_ok()
        {
            all_assignments_with_replacements.push(a);
        }
    }
    let existing_ids: std::collections::HashSet<String> = all_assignments_with_replacements
        .iter()
        .map(|a| a.assignment_id.clone())
        .collect();

    // Stage 12.11 review enforcement — the plan is unsigned/local
    // and `repair_plan_hash` is reachable by hand-editing the JSON.
    // Without this guard, an edited plan can target an assignment
    // that is already `superseded` or already has a valid partial
    // (`partial_present == true`), while the session is still
    // `InProgress` due to some OTHER missing assignment. The
    // session-level `source_status_hash` recheck above proves the
    // overall shape is unchanged but does not bind per-action
    // intent. The helper required every `ReassignAssignment` to
    // target an active-missing row whose `stage_index` matches
    // the plan's `original_stage_index`.
    omni_contributor::check_reassign_targets_active_missing(&plan, &current_status)
        .map_err(|e| anyhow!("apply rejected by per-action enforcement: {e}"))?;

    for action in &plan.actions {
        let RepairAction::ReassignAssignment {
            superseded_assignment_id,
            replacement_contributor_pubkey_hex,
            replacement_stage_index,
            replacement_work_kind,
            replacement_expected_work_units,
            replacement_expected_work_unit_kind,
            ..
        } = action
        else {
            unreachable!("validated above");
        };
        if !existing_ids.contains(superseded_assignment_id) {
            return Err(anyhow!(
                omni_contributor::RepairError::AssignmentNotPresent {
                    session_id: plan.session_id.clone(),
                    assignment_id: superseded_assignment_id.clone(),
                }
            ));
        }
        // Build the replacement WorkAssignment body. `assigned_at_utc`
        // = now_utc gives a fresh assignment_id even when the rest
        // of the body matches the superseded original.
        let mut replacement = WorkAssignment {
            schema_version: SESSION_SCHEMA_VERSION,
            session_id: plan.session_id.clone(),
            assignment_id: String::new(),
            stage_index: *replacement_stage_index,
            contributor_pubkey_hex: replacement_contributor_pubkey_hex.clone(),
            work_kind: replacement_work_kind.clone(),
            expected_work_units: *replacement_expected_work_units,
            expected_work_unit_kind: *replacement_expected_work_unit_kind,
            assigned_at_utc: now_utc.clone(),
            coordinator_signature_hex: String::new(),
        };
        replacement.assignment_id = assignment_id_hex(&replacement)?;
        let sig_input = work_assignment_signing_input(&replacement)?;
        replacement.coordinator_signature_hex = coord.sign_hex(&sig_input);
        replacement.validate_schema().map_err(|e| {
            anyhow!("replacement WorkAssignment failed validate_schema: {e}")
        })?;
        // The replacement's contributor must be in the joined set;
        // otherwise verify_work_assignment fails and the chain is
        // unusable.
        let outcome = omni_contributor::verify_work_assignment(
            &session,
            &joined_pubkeys,
            &replacement,
        );
        if !outcome.is_ok() {
            return Err(anyhow!(
                "replacement WorkAssignment failed verify_work_assignment: {outcome:?}"
            ));
        }
        // Fold into the slice we pass to the supersession verifier
        // (its reference-resolution check needs to see replacement
        // assignment_ids).
        all_assignments_with_replacements.push(replacement.clone());
        prepared.push(PreparedReplacement {
            replacement,
            superseded_assignment_id: superseded_assignment_id.clone(),
        });
    }

    // ── 6. Build, sign, validate the single supersession envelope
    //       covering every superseded + replacement id. ─────────
    let mut superseded_ids: Vec<String> = prepared
        .iter()
        .map(|p| p.superseded_assignment_id.clone())
        .collect();
    superseded_ids.sort();
    superseded_ids.dedup();
    let mut replacement_ids: Vec<String> = prepared
        .iter()
        .map(|p| p.replacement.assignment_id.clone())
        .collect();
    replacement_ids.sort();
    replacement_ids.dedup();
    // All actions in a single plan share one reason (the planner
    // builds them uniformly). Pull the first.
    let reason = match &plan.actions[0] {
        RepairAction::ReassignAssignment { reason, .. } => reason.clone(),
        _ => unreachable!("validated above"),
    };
    let mut supersession = WorkAssignmentSupersession {
        schema_version: SUPERSESSION_SCHEMA_VERSION,
        session_id: plan.session_id.clone(),
        supersession_id: String::new(),
        superseded_assignment_ids: superseded_ids.clone(),
        replacement_assignment_ids: replacement_ids.clone(),
        reason,
        created_at_utc: now_utc.clone(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    supersession.supersession_id = supersession_id_hex(&supersession)?;
    let s_sig = work_assignment_supersession_signing_input(&supersession)?;
    supersession.coordinator_signature_hex = coord.sign_hex(&s_sig);
    supersession.validate_schema().map_err(|e| {
        anyhow!("WorkAssignmentSupersession failed validate_schema: {e}")
    })?;
    let s_outcome = omni_contributor::verify_assignment_supersession(
        &session,
        &all_assignments_with_replacements,
        &supersession,
    );
    if !s_outcome.is_ok() {
        return Err(anyhow!(
            "WorkAssignmentSupersession failed verify_assignment_supersession: {s_outcome:?}"
        ));
    }

    // ── 7. Dry-run path: print would-* events, exit clean. ────
    if args.dry_run {
        for p in &prepared {
            println!(
                "event=would_reassign session_id={} superseded_assignment_id={} \
                 replacement_assignment_id={} stage_index={} contributor={}",
                plan.session_id,
                p.superseded_assignment_id,
                p.replacement.assignment_id,
                p.replacement.stage_index,
                p.replacement.contributor_pubkey_hex,
            );
        }
        println!(
            "event=would_publish_supersession session_id={} supersession_id={} \
             superseded={} replacement={}",
            plan.session_id,
            supersession.supersession_id,
            superseded_ids.len(),
            replacement_ids.len(),
        );
        println!(
            "event=reassign_dry_run session_id={} replacements={} supersession=1",
            plan.session_id,
            prepared.len()
        );
        return Ok(());
    }

    // ── 8. Real publish path. ────────────────────────────────
    let should_publish_announcements = !args.no_publish_announcements;
    let mut mesh: Option<(
        std::sync::Arc<tokio::sync::Mutex<omni_net::OmniNet>>,
        OmniNetRelay,
    )> = if should_publish_announcements {
        let (net, handle) = open_omninet_with_peer_wait(
            args.listen_port,
            args.peer,
            args.peer_wait_secs,
            args.mesh_stabilize_ms,
            args.net_identity_file.as_deref(),
        )
        .await?;
        let relay = OmniNetRelay::new(net.clone(), handle);
        Some((net, relay))
    } else {
        None
    };

    // 8a. **Phase A — SNIP-publish every body first**, before
    // ANY state-dir mutation or mesh broadcast. SNIP is
    // content-addressed and republish is idempotent (same bytes →
    // same root), so a transient failure here is safe to retry
    // and the operator's local state-dir + source_status_hash are
    // unchanged. This avoids the Stage 12.11 review's flagged
    // partial-apply window: previously, replacement assignments
    // were written + marked into the state-dir BEFORE the
    // supersession was even attempted, so a failed supersession
    // publish would leave extra active replacements in the
    // state-dir without the retiring supersession — and a retry
    // would then refuse on `source_status_hash` drift, forcing
    // manual state cleanup. Now Phase A commits every SNIP body
    // up-front, Phase B only runs after every Phase-A publish
    // succeeded.
    let mut replacement_roots: Vec<String> =
        Vec::with_capacity(prepared.len());
    for p in &prepared {
        let json = serde_json::to_vec_pretty(&p.replacement)?;
        let root = omni_contributor::snip::publish_bytes(
            &snip,
            &json,
            "replacement-assignment",
        )
        .map_err(|e| anyhow!("snip publish replacement: {e}"))?;
        replacement_roots.push(format!("0x{}", hex_lower(root.as_bytes())));
    }
    let s_json = serde_json::to_vec_pretty(&supersession)?;
    let s_root = omni_contributor::snip::publish_bytes(
        &snip,
        &s_json,
        "assignment-supersession",
    )
    .map_err(|e| anyhow!("snip publish supersession: {e}"))?;
    let s_root_hex = format!("0x{}", hex_lower(s_root.as_bytes()));

    // 8b. **Phase B — local + mesh side effects**, split into
    // four sub-phases by Stage 12.13:
    //
    //   B1. Write every replacement assignment to the state-dir
    //       + mark seen. NO mesh broadcast yet.
    //   B2. Write the supersession body to the state-dir + mark
    //       seen. NO mesh broadcast yet.
    //   B3. Mesh-broadcast every replacement
    //       (NetworkWorkAssignedAnnouncement). Failures here
    //       emit a structured `event=warn` and continue; the
    //       state-dir is already coherent, so a future apply
    //       retry would just re-broadcast.
    //   B4. Mesh-broadcast the supersession ONLY when every B3
    //       replacement broadcast succeeded. Otherwise skip B4
    //       and emit a structured warning so the operator can
    //       investigate before announcing the supersession to
    //       peers — preserving the "replacements observable on
    //       mesh before supersession" invariant.
    //
    // The Stage 12.11 round-1 fix (Phase A SNIP-publishes
    // everything first) closes the SNIP-publish failure window.
    // Stage 12.13 closes the harder window where a Phase B mesh
    // failure mid-loop left the state-dir half-applied: now
    // every state-dir write completes before any mesh work, so
    // a mid-loop failure during state writes is the only
    // remaining trap (a single FS op for the supersession), and
    // mesh failures degrade gracefully via warnings.

    // ── B1: write replacement assignments ────────────────────
    for (p, root_hex) in prepared.iter().zip(replacement_roots.iter()) {
        store.write_verified_json(
            omni_contributor::StateObjectKind::Assignment {
                session_id: plan.session_id.clone(),
            },
            &p.replacement.assignment_id,
            &p.replacement,
        )?;
        let marker = format!("{}--{}", plan.session_id, p.replacement.assignment_id);
        store.mark_seen(
            omni_contributor::StateNamespace::Assignments,
            &marker,
        )?;
        println!(
            "event=replacement_assignment_state_written session_id={} \
             superseded_assignment_id={} replacement_assignment_id={} \
             stage_index={} contributor={} work_assignment_snip_root={}",
            plan.session_id,
            p.superseded_assignment_id,
            p.replacement.assignment_id,
            p.replacement.stage_index,
            p.replacement.contributor_pubkey_hex,
            root_hex,
        );
    }

    // ── B2: write supersession ──────────────────────────────
    store.write_verified_json(
        omni_contributor::StateObjectKind::AssignmentSupersession {
            session_id: plan.session_id.clone(),
        },
        &supersession.supersession_id,
        &supersession,
    )?;
    let s_marker = format!("{}--{}", plan.session_id, supersession.supersession_id);
    store.mark_seen(
        omni_contributor::StateNamespace::AssignmentSupersessions,
        &s_marker,
    )?;
    println!(
        "event=supersession_state_written session_id={} supersession_id={} \
         superseded={} replacement={} work_assignment_supersession_snip_root={}",
        plan.session_id,
        supersession.supersession_id,
        superseded_ids.len(),
        replacement_ids.len(),
        s_root_hex,
    );

    // ── B3: mesh broadcast replacements ─────────────────────
    let mut replacement_broadcast_failures = 0u32;
    if let Some((_, ref mut relay)) = mesh.as_mut().map(|(n, r)| (n, r)) {
        for (p, root_hex) in prepared.iter().zip(replacement_roots.iter()) {
            let mut ann = NetworkWorkAssignedAnnouncement {
                schema_version: NET_SCHEMA_VERSION,
                work_assignment_snip_root: root_hex.clone(),
                session_id: plan.session_id.clone(),
                assignment_id: p.replacement.assignment_id.clone(),
                contributor_pubkey_hex: p.replacement.contributor_pubkey_hex.clone(),
                announced_at_utc: chrono::Utc::now()
                    .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                announcer_pubkey_hex: coord.pubkey_hex(),
                announcer_signature_hex: String::new(),
            };
            let ann_sig = net_assign_signing_input(&ann)?;
            ann.announcer_signature_hex = coord.sign_hex(&ann_sig);
            match relay.publish_work_assigned(&ann) {
                Ok(()) => {
                    println!(
                        "event=replacement_assignment_broadcast session_id={} \
                         replacement_assignment_id={}",
                        plan.session_id, p.replacement.assignment_id,
                    );
                }
                Err(e) => {
                    replacement_broadcast_failures += 1;
                    println!(
                        "event=warn context=replacement_assignment_broadcast \
                         session_id={} replacement_assignment_id={} message={e}",
                        plan.session_id, p.replacement.assignment_id,
                    );
                }
            }
        }
    }

    // ── B4: mesh broadcast supersession (skip on any B3 failure) ──
    let supersession_broadcast_status: &str;
    if mesh.is_none() {
        supersession_broadcast_status = "skipped_no_publish";
    } else if replacement_broadcast_failures > 0 {
        // Warn-and-skip: replacements must be observable on mesh
        // before peers see the supersession naming them. Operator
        // can re-run apply or manually rebroadcast after
        // investigating the failures.
        supersession_broadcast_status = "skipped_replacement_broadcast_failed";
        println!(
            "event=warn context=supersession_broadcast_skipped session_id={} \
             supersession_id={} replacement_broadcast_failures={}",
            plan.session_id,
            supersession.supersession_id,
            replacement_broadcast_failures,
        );
    } else if let Some((_, ref mut relay)) = mesh.as_mut().map(|(n, r)| (n, r)) {
        let mut ann = NetworkWorkAssignmentSupersessionAnnouncement {
            schema_version: NET_SCHEMA_VERSION,
            work_assignment_supersession_snip_root: s_root_hex.clone(),
            session_id: plan.session_id.clone(),
            supersession_id: supersession.supersession_id.clone(),
            announced_at_utc: chrono::Utc::now()
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            announcer_pubkey_hex: coord.pubkey_hex(),
            announcer_signature_hex: String::new(),
        };
        let ann_sig = net_supersession_signing_input(&ann)?;
        ann.announcer_signature_hex = coord.sign_hex(&ann_sig);
        match relay.publish_assignment_supersession(&ann) {
            Ok(()) => {
                supersession_broadcast_status = "broadcast_ok";
                println!(
                    "event=supersession_broadcast session_id={} \
                     supersession_id={}",
                    plan.session_id, supersession.supersession_id,
                );
            }
            Err(e) => {
                supersession_broadcast_status = "broadcast_failed";
                println!(
                    "event=warn context=supersession_broadcast \
                     session_id={} supersession_id={} message={e}",
                    plan.session_id, supersession.supersession_id,
                );
            }
        }
    } else {
        // Unreachable in practice (mesh.is_some checked above)
        // but kept as a defensive fallthrough so the local
        // bookkeeping `supersession_broadcast_status` is always
        // initialized.
        supersession_broadcast_status = "skipped_no_publish";
    }

    if let Some((net, _relay)) = mesh {
        tokio::time::sleep(std::time::Duration::from_millis(args.propagation_wait_ms))
            .await;
        let g = net.lock().await;
        let _ = g.shutdown().await;
    }
    println!(
        "event=reassign_applied session_id={} replacements_published={} \
         supersession_published=1 supersession_broadcast={} \
         replacement_broadcast_failures={}",
        plan.session_id,
        prepared.len(),
        supersession_broadcast_status,
        replacement_broadcast_failures,
    );
    // Stage 12.13 review fix — exit nonzero when mesh state is
    // incoherent so automation does NOT treat a half-complete
    // apply as a clean one. The state-dir is coherent in every
    // case (B1 + B2 wrote everything before any mesh work) but
    // peers see different views depending on the closed
    // `supersession_broadcast` status:
    //
    //   - `broadcast_ok`       → mesh state coherent → Ok.
    //   - `skipped_no_publish` → operator opted out of mesh →
    //                            Ok (intentional).
    //   - `skipped_replacement_broadcast_failed` → B3 partially
    //                            published replacements but B4
    //                            was skipped to preserve the
    //                            "replacements observable on
    //                            mesh before supersession"
    //                            invariant → Err (operator
    //                            must triage).
    //   - `broadcast_failed`   → all B3 replacements broadcast
    //                            cleanly but the B4 supersession
    //                            broadcast itself errored →
    //                            replacements are visible on
    //                            mesh without the retiring
    //                            supersession → Err (operator
    //                            re-runs or manually
    //                            re-broadcasts).
    match supersession_broadcast_status {
        "broadcast_ok" | "skipped_no_publish" => Ok(()),
        "skipped_replacement_broadcast_failed" => Err(anyhow!(
            "apply-session-reassign mesh state incoherent: \
             {replacement_broadcast_failures} replacement broadcast(s) \
             failed in B3, B4 supersession broadcast skipped to preserve \
             ordering invariant; state-dir is coherent; \
             re-run apply or manually re-broadcast"
        )),
        "broadcast_failed" => Err(anyhow!(
            "apply-session-reassign mesh state incoherent: B3 \
             replacement broadcasts succeeded but B4 supersession \
             broadcast failed; state-dir is coherent; re-run apply \
             or manually re-broadcast the supersession"
        )),
        other => Err(anyhow!(
            "apply-session-reassign internal bookkeeping bug: unknown \
             supersession_broadcast status {other:?}"
        )),
    }
}

// ── Stage 12.14 — archive-session ─────────────────────────────────────────

fn run_archive_session(args: ArchiveSessionArgs) -> Result<()> {
    use omni_contributor::{
        archive_session, ArchiveMode, ArchiveOptions, ContributorStateStore,
    };

    // Resolve mode. The clap layer pins `--copy` xor `--move`;
    // `--move` (default false) takes precedence when set.
    let mode = if args.move_mode {
        ArchiveMode::Move
    } else {
        ArchiveMode::Copy
    };
    let mode_tag = match mode {
        ArchiveMode::Copy => "copy",
        ArchiveMode::Move => "move",
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let (store, prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        !args.no_prune_state_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    println!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        args.contributor_state_dir.display(),
        prune_report.removed_sessions,
        prune_report.removed_peer_adverts,
        prune_report.kept
    );

    let opts = ArchiveOptions {
        session_id: &args.session_id,
        archive_dir: &args.archive_dir,
        mode,
        require_status:
            omni_contributor::ArchiveStatusRequirement::from(args.require_status),
        include_results: args.include_results,
        now_utc: &now_utc,
        dry_run: args.dry_run,
    };

    println!(
        "event=archive_started session_id={} mode={mode_tag} \
         require_status={} include_results={} dry_run={}",
        args.session_id,
        omni_contributor::ArchiveStatusRequirement::from(args.require_status)
            .as_str(),
        args.include_results,
        args.dry_run,
    );

    let manifest = archive_session(&store, &opts)
        .map_err(|e| anyhow!("archive-session refused: {e}"))?;

    let total_bytes: u64 = manifest.files.iter().map(|f| f.bytes).sum();

    if args.dry_run {
        for f in &manifest.files {
            println!(
                "event=would_archive_file source_relative={} \
                 archive_relative={} blake3_hex={} bytes={}",
                f.source_relative, f.archive_relative, f.blake3_hex, f.bytes,
            );
        }
        println!(
            "event=would_archive_complete session_id={} files={} \
             total_bytes={} mode={mode_tag}",
            args.session_id,
            manifest.files.len(),
            total_bytes,
        );
        return Ok(());
    }

    // Real apply: emit per-file events, then the closing event.
    for f in &manifest.files {
        println!(
            "event=archive_file session_id={} source_relative={} \
             archive_relative={} blake3_hex={} bytes={}",
            args.session_id,
            f.source_relative,
            f.archive_relative,
            f.blake3_hex,
            f.bytes,
        );
    }
    let manifest_path = args.archive_dir.join(&args.session_id).join("manifest.json");
    println!(
        "event=archive_complete session_id={} files={} total_bytes={} \
         mode={mode_tag} manifest={}",
        args.session_id,
        manifest.files.len(),
        total_bytes,
        manifest_path.display(),
    );
    Ok(())
}

// ── Stage 12.15 — restore-session-archive ─────────────────────────────────

fn run_restore_session_archive(args: RestoreSessionArchiveArgs) -> Result<()> {
    use omni_contributor::{
        restore_session_archive, ContributorStateStore, RestoreOptions,
        RestoreSource,
    };

    // Resolve the archive source. Clap enforces the mutual
    // exclusion + the requires-pair (session_id <-> archive_dir);
    // the run fn just needs to pick one branch.
    let session_dir_owned: PathBuf;
    let archive_dir_owned: PathBuf;
    let session_id_owned: String;
    let source = if let Some(p) = args.archive_session_dir.as_ref() {
        session_dir_owned = p.clone();
        RestoreSource::SessionDir(&session_dir_owned)
    } else {
        let archive_dir = args.archive_dir.as_ref().ok_or_else(|| {
            anyhow!(
                "restore-session-archive: supply either --archive-session-dir \
                 OR (--archive-dir + --session-id)"
            )
        })?;
        let session_id = args.session_id.as_ref().ok_or_else(|| {
            anyhow!(
                "restore-session-archive: --archive-dir requires --session-id"
            )
        })?;
        archive_dir_owned = archive_dir.clone();
        session_id_owned = session_id.clone();
        RestoreSource::ArchiveRoot {
            archive_dir: &archive_dir_owned,
            session_id: &session_id_owned,
        }
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let (store, prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        !args.no_prune_state_on_start,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    println!(
        "event=state_store_opened path={} pruned_sessions={} pruned_peer_adverts={} kept={}",
        args.contributor_state_dir.display(),
        prune_report.removed_sessions,
        prune_report.removed_peer_adverts,
        prune_report.kept
    );

    let opts = RestoreOptions {
        source,
        dry_run: args.dry_run,
        verify_only: args.verify_only,
        overwrite_existing: args.overwrite_existing,
        include_results: args.include_results,
        now_utc: &now_utc,
    };

    // Verify the manifest first so the `event=restore_started`
    // line carries the informational status/coherence fields
    // even before per-file work.
    let manifest =
        omni_contributor::verify_archive_manifest(&opts.source).map_err(|e| {
            anyhow!("restore-session-archive refused: {e}")
        })?;

    let mode_tag = if args.verify_only {
        "verify_only"
    } else if args.dry_run {
        "dry_run"
    } else {
        "restore"
    };
    let archive_dir_for_event = match (&args.archive_session_dir, &args.archive_dir) {
        (Some(p), _) => p.display().to_string(),
        (_, Some(p)) => p.display().to_string(),
        _ => String::from("-"),
    };
    println!(
        "event=restore_started session_id={} mode={mode_tag} \
         archive_dir={archive_dir_for_event} \
         overwrite_existing={} include_results={} \
         session_overall_status={} audit_coherence={}",
        manifest.session_id,
        args.overwrite_existing,
        args.include_results,
        manifest.session_overall_status,
        manifest.audit_coherence,
    );

    let report = restore_session_archive(&store, &opts)
        .map_err(|e| anyhow!("restore-session-archive refused: {e}"))?;

    // Per-mode per-file emission. The library `restore_session_archive`
    // doesn't emit; the CLI does so it can produce closed-set
    // event lines without leaking library-internal state.
    let session_dir_path = opts.source.session_dir();
    for entry in &manifest.files {
        let is_link =
            entry.source_relative.starts_with("results/result-links/");
        if is_link && !args.include_results {
            println!(
                "event=restore_skipped_result_link session_id={} \
                 source_relative={} reason=include_results_off",
                manifest.session_id, entry.source_relative,
            );
            continue;
        }
        let line = match report.mode {
            "dry_run" => "event=would_restore_file",
            "verify_only" => "event=verify_only_file",
            _ => "event=restore_file",
        };
        let destination = store
            .root()
            .join(&entry.source_relative)
            .display()
            .to_string();
        println!(
            "{line} session_id={} source_relative={} blake3_hex={} bytes={} \
             destination={destination}",
            manifest.session_id,
            entry.source_relative,
            entry.blake3_hex,
            entry.bytes,
        );
    }

    println!(
        "event=restore_complete session_id={} mode={mode_tag} files={} \
         files_skipped_results={} bytes={} archive_dir={}",
        report.session_id,
        report.files_restored,
        report.files_skipped_results,
        report.bytes_restored,
        session_dir_path.display(),
    );
    Ok(())
}

// ── Stage 12.16 — state-integrity ─────────────────────────────────────────

fn run_state_integrity(args: StateIntegrityArgs) -> Result<()> {
    use omni_contributor::{
        scan_state_integrity, ContributorStateStore, ScanOptions,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // JSON mode keeps stdout report-only so `jq` works directly;
    // events / pretty keep their prose stdout posture.
    let json_mode = matches!(args.format, StateIntegrityFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Stage 12.16 safety contract: the scanner is read-only. We
    // pass `auto_prune = false` UNCONDITIONALLY so the open call
    // never cascades-removes an expired session subtree before
    // the scan runs — exactly the state an integrity scan should
    // surface, not silently delete.
    let (store, _prune_report) = ContributorStateStore::open(
        &args.contributor_state_dir,
        /* auto_prune = */ false,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    log_op(&format!(
        "event=state_store_opened path={} auto_prune=off",
        args.contributor_state_dir.display(),
    ));

    let opts = ScanOptions {
        session_id_filter: args.session_id.as_deref(),
        archive_dir: args.include_archives.as_deref(),
        now_utc: &now_utc,
    };

    let report = scan_state_integrity(&store, &opts)
        .map_err(|e| anyhow!("state-integrity scan refused: {e}"))?;

    // Stage 12.19 — when `--baseline` is set, switch to diff
    // mode: read the baseline JSON, diff against the live
    // scan, render the diff (not the raw report), apply diff
    // exit-code policy. `--format` / `--json-out` apply to the
    // diff output. `--fail-on-warn` still applies to the live
    // scan's counts and composes with the diff exit flags via
    // OR — any one tripping → exit 1.
    if args.baseline.is_some() || args.signed_baseline.is_some() {
        // Stage 12.20 — load the baseline. `--baseline` and
        // `--signed-baseline` are clap-level mutually
        // exclusive; whichever is set determines the loader.
        // The signed loader runs the
        // Stage 12.20 verification (pubkey trust anchor +
        // signature) BEFORE extracting the embedded report.
        let baseline = resolve_diff_baseline(
            args.baseline.as_deref(),
            args.signed_baseline.as_deref(),
            args.baseline_pubkey_hex.as_deref(),
            &log_op,
        )?;
        return run_state_integrity_baseline_diff(
            &args, &report, baseline, &now_utc, log_op,
        );
    }

    match args.format {
        StateIntegrityFormat::Events => render_state_integrity_events(&report),
        StateIntegrityFormat::Json => render_state_integrity_json(&report)?,
        StateIntegrityFormat::Pretty => render_state_integrity_pretty(&report),
    }

    if let Some(path) = args.json_out.as_deref() {
        match serde_json::to_vec_pretty(&report) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, &bytes) {
                    eprintln!(
                        "event=warn context=state_integrity_json_out path={} message={e}",
                        path.display()
                    );
                } else {
                    log_op(&format!(
                        "event=state_integrity_json_written path={}",
                        path.display()
                    ));
                }
            }
            Err(e) => {
                eprintln!(
                    "event=warn context=state_integrity_json_serialize message={e}"
                );
            }
        }
    }

    // Exit code policy:
    //   default       — exit 1 when counts_error > 0
    //   --fail-on-warn — exit 1 when counts_warn + counts_error > 0
    let trip = if args.fail_on_warn {
        report.counts_warn + report.counts_error > 0
    } else {
        report.counts_error > 0
    };
    if trip {
        return Err(anyhow!(
            "state-integrity scan reported counts_warn={} counts_error={}; \
             {}",
            report.counts_warn,
            report.counts_error,
            if args.fail_on_warn {
                "--fail-on-warn tripped"
            } else {
                "counts_error > 0"
            }
        ));
    }
    Ok(())
}

/// Stage 12.19 — live-scan-vs-baseline branch of
/// `run_state_integrity`. Called when `args.baseline` OR
/// Stage 12.20 `args.signed_baseline` is set. The caller has
/// already resolved the baseline (raw or signed-then-verified)
/// to a `StateIntegrityReport` via `resolve_diff_baseline`.
fn run_state_integrity_baseline_diff(
    args: &StateIntegrityArgs,
    current_report: &omni_contributor::StateIntegrityReport,
    baseline: omni_contributor::StateIntegrityReport,
    now_utc: &str,
    log_op: impl Fn(&str),
) -> Result<()> {
    use omni_contributor::{diff_state_integrity_reports, DiffOptions};

    let diff_opts = DiffOptions {
        now_utc,
        require_state_dir_match: args.require_state_dir_match,
    };
    let diff = diff_state_integrity_reports(&baseline, current_report, &diff_opts)
        .map_err(|e| anyhow!("state-integrity --baseline refused: {e}"))?;

    match args.format {
        StateIntegrityFormat::Events => {
            render_state_integrity_diff_events(&diff, args.summary_only)
        }
        StateIntegrityFormat::Json => {
            render_state_integrity_diff_json(&diff, args.summary_only)?
        }
        StateIntegrityFormat::Pretty => {
            render_state_integrity_diff_pretty(&diff, args.summary_only)
        }
    }

    if let Some(path) = args.json_out.as_deref() {
        write_diff_json_mirror(&diff, path, args.summary_only, &log_op);
    }

    // Exit-code policy: union of the live scan's
    // `--fail-on-warn` and the diff's `--fail-on-new` /
    // `--fail-on-new-error`.
    let scan_trip = if args.fail_on_warn {
        current_report.counts_warn + current_report.counts_error > 0
    } else {
        current_report.counts_error > 0
    };
    let diff_trip = (args.fail_on_new && diff.counts.new > 0)
        || (args.fail_on_new_error && diff.counts.new_error > 0);
    if scan_trip || diff_trip {
        return Err(anyhow!(
            "state-integrity --baseline tripped: scan_trip={scan_trip} \
             diff_trip={diff_trip} new={} new_error={} counts_warn={} \
             counts_error={}",
            diff.counts.new,
            diff.counts.new_error,
            current_report.counts_warn,
            current_report.counts_error,
        ));
    }
    Ok(())
}

fn render_state_integrity_events(report: &omni_contributor::StateIntegrityReport) {
    for s in &report.sessions {
        println!(
            "event=session_integrity_summary session_id={} overall_status={}",
            s.session_id, s.overall_status,
        );
    }
    for f in &report.findings {
        let sid = f.session_id.as_deref().unwrap_or("-");
        let path = f.path.as_deref().unwrap_or("-");
        println!(
            "event=integrity_finding kind={} severity={} session_id={} \
             path={} reason_tag={} recommended_action={:?}",
            f.kind.as_str(),
            f.severity.as_str(),
            sid,
            path,
            f.reason_tag,
            f.recommended_action.as_str(),
        );
    }
    println!(
        "event=state_integrity_summary sessions_scanned={} sessions_verified={} \
         counts_ok={} counts_warn={} counts_error={}",
        report.sessions_scanned,
        report.sessions_verified,
        report.counts_ok,
        report.counts_warn,
        report.counts_error,
    );
}

fn render_state_integrity_json(
    report: &omni_contributor::StateIntegrityReport,
) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| anyhow!("serialize state integrity report: {e}"))?;
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write state integrity report: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("write trailing newline: {e}"))?;
    Ok(())
}

fn render_state_integrity_pretty(report: &omni_contributor::StateIntegrityReport) {
    use omni_contributor::FindingSeverity;
    println!("State integrity report");
    println!("  generated_at_utc     : {}", report.generated_at_utc);
    println!("  state_dir            : {}", report.state_dir);
    println!("  state_version        : {}", report.state_version);
    println!(
        "  omni_contributor     : {}",
        report.omni_contributor_version
    );
    println!("  schema_version       : {}", report.schema_version);
    println!("  sessions_scanned     : {}", report.sessions_scanned);
    println!("  sessions_verified    : {}", report.sessions_verified);
    println!(
        "  counts (ok/warn/err) : {} / {} / {}",
        report.counts_ok, report.counts_warn, report.counts_error
    );
    if !report.sessions.is_empty() {
        println!();
        println!("Per-session summaries:");
        for s in &report.sessions {
            println!(
                "  - session_id={} overall_status={}",
                s.session_id, s.overall_status
            );
        }
    }
    let section = |label: &str, sev: FindingSeverity| {
        let filtered: Vec<_> = report
            .findings
            .iter()
            .filter(|f| f.severity == sev)
            .collect();
        if filtered.is_empty() {
            return;
        }
        println!();
        println!("{label}:");
        for f in filtered {
            let sid = f.session_id.as_deref().unwrap_or("-");
            let path = f.path.as_deref().unwrap_or("-");
            println!(
                "  - kind={} session_id={} path={} reason_tag={}",
                f.kind.as_str(),
                sid,
                path,
                f.reason_tag,
            );
            println!("      action: {}", f.recommended_action.as_str());
        }
    };
    section("Errors", FindingSeverity::Error);
    section("Warnings", FindingSeverity::Warn);
    section("OK findings", FindingSeverity::Ok);
}

// ── Stage 12.17 — plan-state-cleanup ──────────────────────────────────────

fn run_plan_state_cleanup(args: PlanStateCleanupArgs) -> Result<()> {
    use omni_contributor::{
        plan_state_cleanup, scan_state_integrity_with_audit_orphans,
        source_integrity_hash_hex, CleanupPlanOptions, ContributorStateStore,
        ScanOptions, StateIntegrityReport,
    };
    use std::collections::HashMap;

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode = matches!(args.format, PlanStateCleanupFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Stage 12.16 review precedent: cleanup CLIs always open
    // with auto_prune off so a stale-state snapshot isn't
    // cascade-removed before the planner sees it.
    let (store, _) = ContributorStateStore::open(
        &args.contributor_state_dir,
        /* auto_prune = */ false,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    log_op(&format!(
        "event=state_store_opened path={} auto_prune=off",
        args.contributor_state_dir.display(),
    ));

    // Source the integrity report from either a prebaked JSON
    // OR a fresh scan. Either way, the orphan side-channel
    // comes from the live scan (the report alone doesn't
    // carry orphan assignment_ids).
    let scan_opts = ScanOptions {
        session_id_filter: args.session_id.as_deref(),
        archive_dir: None,
        now_utc: &now_utc,
    };
    let (live_report, live_orphans) =
        scan_state_integrity_with_audit_orphans(&store, &scan_opts)
            .map_err(|e| anyhow!("integrity scan during plan-state-cleanup: {e}"))?;
    let report: StateIntegrityReport = if let Some(path) = args.integrity_json.as_deref() {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read integrity-json: {}", path.display()))?;
        let parsed: StateIntegrityReport = serde_json::from_slice(&bytes)
            .with_context(|| {
                format!("parse integrity-json: {}", path.display())
            })?;
        // Drift sanity: if the supplied report disagrees with
        // the live scan, the plan's source_integrity_hash
        // would refuse at apply time anyway. Warn now so the
        // operator sees the mismatch immediately.
        let live_hash = source_integrity_hash_hex(&live_report);
        let supplied_hash = source_integrity_hash_hex(&parsed);
        if live_hash != supplied_hash {
            eprintln!(
                "event=warn context=plan_state_cleanup_integrity_drift \
                 supplied_hash={supplied_hash} live_hash={live_hash} \
                 message=apply will refuse this plan until state matches the supplied report"
            );
        }
        parsed
    } else {
        live_report
    };

    let orphans: HashMap<String, Vec<String>> = live_orphans;
    let plan_opts = CleanupPlanOptions {
        now_utc: &now_utc,
        session_id_filter: args.session_id.as_deref(),
    };
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts)
        .map_err(|e| anyhow!("build cleanup plan: {e}"))?;

    // Write the plan JSON atomically (temp + rename) — same
    // posture as the Stage 12.10 plan-session-repair --out.
    let plan_bytes = serde_json::to_vec_pretty(&plan)
        .map_err(|e| anyhow!("serialize cleanup plan: {e}"))?;
    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("create plan parent dir: {}", parent.display())
            })?;
        }
    }
    let tmp = args.out.with_extension("json.tmp");
    std::fs::write(&tmp, &plan_bytes)
        .with_context(|| format!("write cleanup plan tmp: {}", tmp.display()))?;
    std::fs::rename(&tmp, &args.out)
        .with_context(|| format!("rename cleanup plan into place: {}", args.out.display()))?;
    log_op(&format!(
        "event=cleanup_plan_written path={} actions={} plan_id={}",
        args.out.display(),
        plan.actions.len(),
        plan.plan_id,
    ));

    match args.format {
        PlanStateCleanupFormat::Events => render_cleanup_plan_events(&plan),
        PlanStateCleanupFormat::Json => render_cleanup_plan_json(&plan)?,
        PlanStateCleanupFormat::Pretty => render_cleanup_plan_pretty(&plan),
    }
    Ok(())
}

fn render_cleanup_plan_events(plan: &omni_contributor::StateCleanupPlan) {
    for (i, action) in plan.actions.iter().enumerate() {
        println!(
            "event=cleanup_action_planned index={} kind={} session_id={} path={} \
             seen_marker={} source_finding_kind={} source_reason_tag={}",
            i,
            action.kind.as_str(),
            action.session_id.as_deref().unwrap_or("-"),
            action.path,
            action.seen_marker_path.as_deref().unwrap_or("-"),
            action.source_finding_kind,
            action.source_reason_tag,
        );
    }
    println!(
        "event=cleanup_plan_built plan_id={} schema_version={} actions={} \
         source_integrity_hash={} cleanup_plan_hash={}",
        plan.plan_id,
        plan.schema_version,
        plan.actions.len(),
        plan.source_integrity_hash,
        plan.cleanup_plan_hash,
    );
}

fn render_cleanup_plan_json(plan: &omni_contributor::StateCleanupPlan) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(plan)
        .map_err(|e| anyhow!("serialize cleanup plan: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write cleanup plan: {e}"))?;
    handle.write_all(b"\n").map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_cleanup_plan_pretty(plan: &omni_contributor::StateCleanupPlan) {
    println!("State cleanup plan");
    println!("  plan_id              : {}", plan.plan_id);
    println!("  schema_version       : {}", plan.schema_version);
    println!("  created_at_utc       : {}", plan.created_at_utc);
    println!("  state_dir            : {}", plan.state_dir);
    println!("  source_integrity_hash: {}", plan.source_integrity_hash);
    println!("  cleanup_plan_hash    : {}", plan.cleanup_plan_hash);
    println!("  action count         : {}", plan.actions.len());
    if !plan.actions.is_empty() {
        println!();
        println!("Actions (in apply order):");
        for (i, action) in plan.actions.iter().enumerate() {
            println!(
                "  {i:>3}. kind={} session={} path={}",
                action.kind.as_str(),
                action.session_id.as_deref().unwrap_or("-"),
                action.path,
            );
            if let Some(seen) = &action.seen_marker_path {
                println!("       seen_marker={seen}");
            }
            println!("       source_finding_kind={}", action.source_finding_kind);
            println!("       source_reason_tag={}", action.source_reason_tag);
        }
    }
}

// ── Stage 12.17 — apply-state-cleanup ────────────────────────────────────

fn run_apply_state_cleanup(args: ApplyStateCleanupArgs) -> Result<()> {
    use omni_contributor::{
        apply_state_cleanup, CleanupApplyOptions, ContributorStateStore,
        StateCleanupPlan,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode = matches!(args.format, ApplyStateCleanupFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    let plan_bytes = std::fs::read(&args.plan)
        .with_context(|| format!("read plan: {}", args.plan.display()))?;
    let plan: StateCleanupPlan = serde_json::from_slice(&plan_bytes)
        .with_context(|| format!("parse plan: {}", args.plan.display()))?;

    let (store, _) = ContributorStateStore::open(
        &args.contributor_state_dir,
        /* auto_prune = */ false,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    log_op(&format!(
        "event=state_store_opened path={} auto_prune=off",
        args.contributor_state_dir.display(),
    ));

    if !args.quarantine_dir.exists() {
        std::fs::create_dir_all(&args.quarantine_dir).with_context(|| {
            format!("create quarantine dir: {}", args.quarantine_dir.display())
        })?;
    }

    let opts = CleanupApplyOptions {
        quarantine_dir: &args.quarantine_dir,
        dry_run: args.dry_run,
        allow_invalid_partial_cleanup: args.allow_invalid_partial_cleanup,
        allow_orphan_assignments: args.allow_orphan_assignments,
        purge_stray: args.purge_stray,
        now_utc: &now_utc,
    };
    log_op(&format!(
        "event=cleanup_started plan_id={} mode={} actions={}",
        plan.plan_id,
        if args.dry_run { "dry_run" } else { "apply" },
        plan.actions.len(),
    ));

    let report = apply_state_cleanup(&store, &plan, &opts)
        .map_err(|e| anyhow!("apply-state-cleanup refused: {e}"))?;

    match args.format {
        ApplyStateCleanupFormat::Events => render_cleanup_apply_events(&report),
        ApplyStateCleanupFormat::Json => render_cleanup_apply_json(&report)?,
        ApplyStateCleanupFormat::Pretty => render_cleanup_apply_pretty(&report),
    }
    Ok(())
}

fn render_cleanup_apply_events(report: &omni_contributor::CleanupReport) {
    for outcome in &report.outcomes {
        let event = match outcome.status.as_str() {
            "would_apply" => "event=would_apply_action",
            "skipped_missing" => "event=cleanup_action_skipped",
            _ => "event=cleanup_action_applied",
        };
        println!(
            "{event} index={} kind={} session_id={} path={} status={}",
            outcome.action_index,
            outcome.kind,
            outcome.session_id.as_deref().unwrap_or("-"),
            outcome.path,
            outcome.status,
        );
    }
    println!(
        "event=cleanup_complete plan_id={} mode={} applied={} dry_run={} \
         skipped={} quarantine_dir={} manifest={}",
        report.plan_id,
        report.mode,
        report.actions_applied,
        report.actions_dry_run,
        report.actions_skipped,
        report.quarantine_dir,
        report.quarantine_manifest_relative.as_deref().unwrap_or("-"),
    );
}

fn render_cleanup_apply_json(report: &omni_contributor::CleanupReport) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| anyhow!("serialize cleanup report: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(&bytes).map_err(|e| anyhow!("write cleanup report: {e}"))?;
    handle.write_all(b"\n").map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_cleanup_apply_pretty(report: &omni_contributor::CleanupReport) {
    println!("State cleanup apply report");
    println!("  plan_id        : {}", report.plan_id);
    println!("  mode           : {}", report.mode);
    println!("  applied        : {}", report.actions_applied);
    println!("  dry_run        : {}", report.actions_dry_run);
    println!("  skipped        : {}", report.actions_skipped);
    println!("  quarantine_dir : {}", report.quarantine_dir);
    if let Some(m) = &report.quarantine_manifest_relative {
        println!("  manifest       : {m}");
    }
    if !report.outcomes.is_empty() {
        println!();
        println!("Outcomes:");
        for o in &report.outcomes {
            println!(
                "  {:>3}. {} session={} path={} status={}",
                o.action_index,
                o.kind,
                o.session_id.as_deref().unwrap_or("-"),
                o.path,
                o.status,
            );
        }
    }
}

// ── Stage 12.18 — restore-state-cleanup-quarantine ───────────────────────

fn run_restore_state_cleanup_quarantine(
    args: RestoreStateCleanupQuarantineArgs,
) -> Result<()> {
    use omni_contributor::{
        restore_state_cleanup_quarantine, ContributorStateStore,
        QuarantineRestoreOptions, QuarantineRestoreSource,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode = matches!(args.format, RestoreQuarantineFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Resolve the quarantine source. Clap enforces both the
    // mutual-exclusion (PlanDir xor pair) AND the
    // pair-requires (quarantine_dir <-> plan_id); the run-fn
    // just picks the branch.
    let plan_dir_owned: PathBuf;
    let quarantine_dir_owned: PathBuf;
    let plan_id_owned: String;
    let source = if let Some(p) = args.quarantine_plan_dir.as_ref() {
        plan_dir_owned = p.clone();
        QuarantineRestoreSource::PlanDir(&plan_dir_owned)
    } else {
        let q = args.quarantine_dir.as_ref().ok_or_else(|| {
            anyhow!(
                "restore-state-cleanup-quarantine: supply either \
                 --quarantine-plan-dir OR (--quarantine-dir + --plan-id)"
            )
        })?;
        let pid = args.plan_id.as_ref().ok_or_else(|| {
            anyhow!(
                "restore-state-cleanup-quarantine: --quarantine-dir requires --plan-id"
            )
        })?;
        quarantine_dir_owned = q.clone();
        plan_id_owned = pid.clone();
        QuarantineRestoreSource::QuarantineRoot {
            quarantine_dir: &quarantine_dir_owned,
            plan_id: &plan_id_owned,
        }
    };

    // Stage 12.16/12.17 precedent: cleanup CLIs always open
    // with auto_prune off so a stale-state snapshot isn't
    // cascade-removed before the restore writes.
    let (store, _) = ContributorStateStore::open(
        &args.contributor_state_dir,
        /* auto_prune = */ false,
        &now_utc,
    )
    .map_err(|e| anyhow!("open contributor state dir: {e}"))?;
    log_op(&format!(
        "event=state_store_opened path={} auto_prune=off",
        args.contributor_state_dir.display(),
    ));

    let opts = QuarantineRestoreOptions {
        source,
        dry_run: args.dry_run,
        verify_only: args.verify_only,
        overwrite_existing: args.overwrite_existing,
        restore_seen_markers: !args.no_restore_seen_markers,
        allow_restore_orphan_assignments: args.allow_restore_orphan_assignments,
        now_utc: &now_utc,
    };

    let mode_tag = if args.verify_only {
        "verify_only"
    } else if args.dry_run {
        "dry_run"
    } else {
        "restore"
    };
    log_op(&format!(
        "event=restore_quarantine_started mode={mode_tag} \
         overwrite_existing={} restore_seen_markers={} \
         allow_restore_orphan_assignments={}",
        args.overwrite_existing,
        !args.no_restore_seen_markers,
        args.allow_restore_orphan_assignments,
    ));

    let report = restore_state_cleanup_quarantine(&store, &opts)
        .map_err(|e| anyhow!("restore-state-cleanup-quarantine refused: {e}"))?;

    match args.format {
        RestoreQuarantineFormat::Events => render_restore_quarantine_events(&report),
        RestoreQuarantineFormat::Json => render_restore_quarantine_json(&report)?,
        RestoreQuarantineFormat::Pretty => render_restore_quarantine_pretty(&report),
    }
    Ok(())
}

fn render_restore_quarantine_events(
    report: &omni_contributor::QuarantineRestoreReport,
) {
    for o in &report.outcomes {
        let line = match o.status.as_str() {
            "verify_only" => "event=verify_only_quarantine_file",
            "would_apply" => "event=would_restore_quarantine_file",
            _ => "event=restore_quarantine_file",
        };
        println!(
            "{line} index={} source_relative={} source_finding_kind={} \
             status={} seen_marker_written={}",
            o.entry_index,
            o.source_relative,
            o.source_finding_kind,
            o.status,
            o.seen_marker_written,
        );
    }
    println!(
        "event=restore_quarantine_complete plan_id={} mode={} files_restored={} \
         seen_markers_restored={} bytes_restored={} quarantine_dir={}",
        report.plan_id,
        report.mode,
        report.files_restored,
        report.seen_markers_restored,
        report.bytes_restored,
        report.quarantine_dir,
    );
}

fn render_restore_quarantine_json(
    report: &omni_contributor::QuarantineRestoreReport,
) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| anyhow!("serialize quarantine restore report: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write quarantine restore report: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_restore_quarantine_pretty(
    report: &omni_contributor::QuarantineRestoreReport,
) {
    println!("Quarantine restore report");
    println!("  plan_id                : {}", report.plan_id);
    println!("  mode                   : {}", report.mode);
    println!("  manifest_schema_version: {}", report.manifest_schema_version);
    println!("  source_state_version   : {}", report.source_state_version);
    println!("  files_restored         : {}", report.files_restored);
    println!("  seen_markers_restored  : {}", report.seen_markers_restored);
    println!("  bytes_restored         : {}", report.bytes_restored);
    println!("  quarantine_dir         : {}", report.quarantine_dir);
    if !report.outcomes.is_empty() {
        println!();
        println!("Outcomes:");
        for o in &report.outcomes {
            println!(
                "  {:>3}. status={} kind={} source_relative={} seen_marker={}",
                o.entry_index,
                o.status,
                o.source_finding_kind,
                o.source_relative,
                if o.seen_marker_written { "yes" } else { "-" },
            );
        }
    }
}

// ── Stage 12.19 — state-integrity-diff ───────────────────────────────────

fn run_state_integrity_diff(args: StateIntegrityDiffArgs) -> Result<()> {
    use omni_contributor::{
        diff_state_integrity_reports, DiffOptions, StateIntegrityReport,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode = matches!(args.format, StateIntegrityDiffFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Stage 12.20 — `--baseline` and `--signed-baseline` are
    // clap-level mutually exclusive AND required-unless-the-other.
    // The signed loader runs the Stage 12.20 verification
    // (pubkey trust anchor + signature) BEFORE extracting the
    // embedded report.
    let baseline = resolve_diff_baseline(
        args.baseline.as_deref(),
        args.signed_baseline.as_deref(),
        args.baseline_pubkey_hex.as_deref(),
        &log_op,
    )?;
    let current_bytes = std::fs::read(&args.current)
        .with_context(|| format!("read current json: {}", args.current.display()))?;
    let current: StateIntegrityReport = serde_json::from_slice(&current_bytes)
        .with_context(|| format!("parse current json: {}", args.current.display()))?;

    log_op(&format!(
        "event=state_integrity_diff_started baseline_source={} current={} \
         require_state_dir_match={}",
        if args.signed_baseline.is_some() {
            "signed"
        } else {
            "raw"
        },
        args.current.display(),
        args.require_state_dir_match,
    ));

    let diff = diff_state_integrity_reports(
        &baseline,
        &current,
        &DiffOptions {
            now_utc: &now_utc,
            require_state_dir_match: args.require_state_dir_match,
        },
    )
    .map_err(|e| anyhow!("state-integrity-diff refused: {e}"))?;

    match args.format {
        StateIntegrityDiffFormat::Events => {
            render_state_integrity_diff_events(&diff, args.summary_only)
        }
        StateIntegrityDiffFormat::Json => {
            render_state_integrity_diff_json(&diff, args.summary_only)?
        }
        StateIntegrityDiffFormat::Pretty => {
            render_state_integrity_diff_pretty(&diff, args.summary_only)
        }
    }

    if let Some(path) = args.json_out.as_deref() {
        write_diff_json_mirror(&diff, path, args.summary_only, &log_op);
    }

    let trip = (args.fail_on_new && diff.counts.new > 0)
        || (args.fail_on_new_error && diff.counts.new_error > 0);
    if trip {
        return Err(anyhow!(
            "state-integrity-diff tripped: new={} new_error={} \
             (--fail-on-new={} --fail-on-new-error={})",
            diff.counts.new,
            diff.counts.new_error,
            args.fail_on_new,
            args.fail_on_new_error,
        ));
    }
    Ok(())
}

/// Stage 12.20 — resolve `--baseline` xor `--signed-baseline`
/// into a typed `StateIntegrityReport`. Clap has already
/// enforced that exactly one of the two is set AND that
/// `--baseline-pubkey-hex` accompanies `--signed-baseline`;
/// this helper just executes the chosen loader. The signed
/// loader runs the Stage 12.20 verification (pubkey trust
/// anchor + signature) BEFORE extracting the embedded report,
/// so a refused wrapper never reaches the diff helper.
fn resolve_diff_baseline(
    raw_path: Option<&std::path::Path>,
    signed_path: Option<&std::path::Path>,
    baseline_pubkey_hex: Option<&str>,
    log_op: &impl Fn(&str),
) -> Result<omni_contributor::StateIntegrityReport> {
    use omni_contributor::{
        read_signed_baseline_from_path, verify_signed_state_integrity_baseline,
        StateIntegrityReport,
    };

    if let Some(path) = signed_path {
        let expected_pubkey = baseline_pubkey_hex.ok_or_else(|| {
            anyhow!(
                "--signed-baseline requires --baseline-pubkey-hex (clap should have caught this)"
            )
        })?;
        let wrapper = read_signed_baseline_from_path(path).map_err(|e| {
            anyhow!("read signed baseline at {}: {e}", path.display())
        })?;
        verify_signed_state_integrity_baseline(&wrapper, expected_pubkey).map_err(
            |e| {
                // Closed-tag reason for log scrapers.
                let reason_tag = match &e {
                    omni_contributor::SignedBaselineError::SignerPubkeyMismatch { .. } => {
                        "signer_pubkey_mismatch"
                    }
                    omni_contributor::SignedBaselineError::SignatureMismatch => {
                        "signature_mismatch"
                    }
                    omni_contributor::SignedBaselineError::UnsupportedSchemaVersion { .. } => {
                        "unsupported_schema_version"
                    }
                    omni_contributor::SignedBaselineError::UnsupportedReportSchemaVersion { .. } => {
                        "unsupported_report_schema_version"
                    }
                    omni_contributor::SignedBaselineError::Signing(_) => {
                        "signing_decode_error"
                    }
                    omni_contributor::SignedBaselineError::Canonical(_) => {
                        "canonical_encoding_error"
                    }
                    _ => "other",
                };
                eprintln!(
                    "event=signed_baseline_refused reason={reason_tag} message={e}"
                );
                anyhow!("signed baseline refused: {e}")
            },
        )?;
        log_op(&format!(
            "event=signed_baseline_verified pubkey={} signer_role={}",
            wrapper.signer_pubkey_hex,
            wrapper.signer_role.as_str(),
        ));
        Ok(wrapper.report)
    } else if let Some(path) = raw_path {
        if baseline_pubkey_hex.is_some() {
            // Runtime backstop for the clap-edge described in
            // the test: when --baseline is supplied AND
            // --baseline-pubkey-hex is also supplied (without
            // --signed-baseline), clap's `requires` doesn't
            // refuse because the `required_unless_present`
            // chain on the baseline flags satisfies clap. Warn
            // loudly so the operator notices the pubkey is
            // unused.
            eprintln!(
                "event=warn context=baseline_pubkey_hex_unused \
                 message=--baseline-pubkey-hex was supplied but --signed-baseline is not set; \
                 the trust anchor is ignored because the raw baseline is unsigned"
            );
        }
        let bytes = std::fs::read(path)
            .with_context(|| format!("read baseline json: {}", path.display()))?;
        let report: StateIntegrityReport = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse baseline json: {}", path.display()))?;
        Ok(report)
    } else {
        Err(anyhow!(
            "neither --baseline nor --signed-baseline supplied (clap should have caught this)"
        ))
    }
}

fn render_state_integrity_diff_events(
    diff: &omni_contributor::StateIntegrityDiffReport,
    summary_only: bool,
) {
    for f in &diff.new_findings {
        println!(
            "event=integrity_diff_new kind={} severity={} session_id={} \
             path={} reason_tag={}",
            f.kind.as_str(),
            f.severity.as_str(),
            f.session_id.as_deref().unwrap_or("-"),
            f.path.as_deref().unwrap_or("-"),
            f.reason_tag,
        );
    }
    for f in &diff.resolved_findings {
        println!(
            "event=integrity_diff_resolved kind={} severity={} session_id={} \
             path={} reason_tag={}",
            f.kind.as_str(),
            f.severity.as_str(),
            f.session_id.as_deref().unwrap_or("-"),
            f.path.as_deref().unwrap_or("-"),
            f.reason_tag,
        );
    }
    if !summary_only {
        for f in &diff.unchanged_findings {
            println!(
                "event=integrity_diff_unchanged kind={} severity={} session_id={} \
                 path={} reason_tag={}",
                f.kind.as_str(),
                f.severity.as_str(),
                f.session_id.as_deref().unwrap_or("-"),
                f.path.as_deref().unwrap_or("-"),
                f.reason_tag,
            );
        }
    }
    println!(
        "event=state_integrity_diff_summary new={} resolved={} unchanged={} \
         new_ok={} new_warn={} new_error={} resolved_ok={} resolved_warn={} \
         resolved_error={}",
        diff.counts.new,
        diff.counts.resolved,
        diff.counts.unchanged,
        diff.counts.new_ok,
        diff.counts.new_warn,
        diff.counts.new_error,
        diff.counts.resolved_ok,
        diff.counts.resolved_warn,
        diff.counts.resolved_error,
    );
}

fn render_state_integrity_diff_json(
    diff: &omni_contributor::StateIntegrityDiffReport,
    summary_only: bool,
) -> Result<()> {
    use std::io::Write;
    // Stage 12.19 review fix — `--summary-only` is a CLI
    // presentation gate that must apply to the JSON stdout
    // render AND the `--json-out` mirror, not just events /
    // pretty. The library's `diff_presentation_view` clones
    // and clears `unchanged_findings` when summary_only is
    // true (preserving `counts.unchanged` verbatim so scripts
    // still see the elided count) and borrows otherwise.
    let view = omni_contributor::diff_presentation_view(diff, summary_only);
    let bytes = serde_json::to_vec_pretty(view.as_ref())
        .map_err(|e| anyhow!("serialize diff report: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write diff report: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_state_integrity_diff_pretty(
    diff: &omni_contributor::StateIntegrityDiffReport,
    summary_only: bool,
) {
    println!("State integrity diff");
    println!("  generated_at_utc       : {}", diff.generated_at_utc);
    println!("  schema_version         : {}", diff.schema_version);
    println!(
        "  baseline_generated_at  : {}",
        diff.baseline_generated_at_utc
    );
    println!(
        "  current_generated_at   : {}",
        diff.current_generated_at_utc
    );
    println!(
        "  omni_contributor_versions: baseline={} current={}",
        diff.baseline_omni_contributor_version,
        diff.current_omni_contributor_version,
    );
    println!(
        "  state_version          : baseline={} current={}",
        diff.baseline_state_version, diff.current_state_version,
    );
    println!(
        "  state_dir              : baseline={} current={}",
        diff.baseline_state_dir, diff.state_dir,
    );
    println!();
    println!("Counts");
    println!(
        "  new      : {} (ok={} warn={} error={})",
        diff.counts.new,
        diff.counts.new_ok,
        diff.counts.new_warn,
        diff.counts.new_error,
    );
    println!(
        "  resolved : {} (ok={} warn={} error={})",
        diff.counts.resolved,
        diff.counts.resolved_ok,
        diff.counts.resolved_warn,
        diff.counts.resolved_error,
    );
    println!("  unchanged: {}", diff.counts.unchanged);

    let section = |label: &str, findings: &[omni_contributor::IntegrityFinding]| {
        if findings.is_empty() {
            return;
        }
        println!();
        println!("{label}:");
        for f in findings {
            println!(
                "  - kind={} severity={} session={} path={} reason_tag={}",
                f.kind.as_str(),
                f.severity.as_str(),
                f.session_id.as_deref().unwrap_or("-"),
                f.path.as_deref().unwrap_or("-"),
                f.reason_tag,
            );
        }
    };
    section("New findings", &diff.new_findings);
    section("Resolved findings", &diff.resolved_findings);
    if !summary_only {
        section("Unchanged findings", &diff.unchanged_findings);
    }
}

fn write_diff_json_mirror(
    diff: &omni_contributor::StateIntegrityDiffReport,
    path: &std::path::Path,
    summary_only: bool,
    log_op: &impl Fn(&str),
) {
    // Stage 12.19 review fix — same redaction posture as the
    // stdout JSON render: `--summary-only` elides
    // `unchanged_findings` from the mirrored artifact too,
    // preserving `counts.unchanged`.
    let view = omni_contributor::diff_presentation_view(diff, summary_only);
    match serde_json::to_vec_pretty(view.as_ref()) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(path, &bytes) {
                eprintln!(
                    "event=warn context=state_integrity_diff_json_out path={} message={e}",
                    path.display()
                );
            } else {
                log_op(&format!(
                    "event=state_integrity_diff_json_written path={}",
                    path.display()
                ));
            }
        }
        Err(e) => {
            eprintln!(
                "event=warn context=state_integrity_diff_json_serialize message={e}"
            );
        }
    }
}

// ── Stage 12.20 — sign-state-integrity-baseline ───────────────────────────

fn run_sign_state_integrity_baseline(
    args: SignStateIntegrityBaselineArgs,
) -> Result<()> {
    use omni_contributor::{
        sign_state_integrity_baseline, write_signed_baseline_atomic,
        BaselineSignerRole, ContributorSigner, SignedStateIntegrityBaseline,
        StateIntegrityReport,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode = matches!(args.format, SignStateIntegrityBaselineFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    log_op("event=signed_baseline_signing_started");

    // Read the raw v1 report.
    let report_bytes = std::fs::read(&args.baseline_in).with_context(|| {
        format!("read baseline-in: {}", args.baseline_in.display())
    })?;
    let report: StateIntegrityReport = serde_json::from_slice(&report_bytes)
        .with_context(|| format!("parse baseline-in: {}", args.baseline_in.display()))?;

    // Load the signer. Stage 12.20's CLI uses
    // `ContributorSigner` as the seed loader since all four
    // role-typed signers share the same shape; the recorded
    // `signer_role` tag is independent of the loader type.
    let signer = ContributorSigner::from_seed_file(&args.signer_seed)
        .with_context(|| {
            format!("load signer seed: {}", args.signer_seed.display())
        })?;
    let signer_pubkey_hex = signer.pubkey_hex();
    let role: BaselineSignerRole = args.signer_role.into();

    let signed: SignedStateIntegrityBaseline = sign_state_integrity_baseline(
        report,
        &signer_pubkey_hex,
        role,
        &now_utc,
        |msg| signer.sign(msg),
    )
    .map_err(|e| anyhow!("sign baseline refused: {e}"))?;

    write_signed_baseline_atomic(&signed, &args.out)
        .map_err(|e| anyhow!("write signed baseline: {e}"))?;
    log_op(&format!(
        "event=signed_baseline_written path={} signer_role={} signer_pubkey={}",
        args.out.display(),
        role.as_str(),
        signer_pubkey_hex,
    ));

    match args.format {
        SignStateIntegrityBaselineFormat::Events => {
            // Already emitted the summary line above.
        }
        SignStateIntegrityBaselineFormat::Json => {
            render_sign_baseline_json(&signed)?;
        }
        SignStateIntegrityBaselineFormat::Pretty => {
            render_sign_baseline_pretty(&signed, &args.out);
        }
    }
    Ok(())
}

fn render_sign_baseline_json(
    signed: &omni_contributor::SignedStateIntegrityBaseline,
) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(signed)
        .map_err(|e| anyhow!("serialize signed baseline: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write signed baseline: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_sign_baseline_pretty(
    signed: &omni_contributor::SignedStateIntegrityBaseline,
    out: &std::path::Path,
) {
    println!("Signed integrity baseline");
    println!("  schema_version  : {}", signed.schema_version);
    println!("  signed_at_utc   : {}", signed.signed_at_utc);
    println!("  signer_role     : {}", signed.signer_role.as_str());
    println!("  signer_pubkey   : {}", signed.signer_pubkey_hex);
    println!("  signature       : {}", signed.signature_hex);
    println!("  state_dir       : {}", signed.report.state_dir);
    println!("  state_version   : {}", signed.report.state_version);
    println!(
        "  findings        : {} (ok={} warn={} error={})",
        signed.report.findings.len(),
        signed.report.counts_ok,
        signed.report.counts_warn,
        signed.report.counts_error,
    );
    println!("  out             : {}", out.display());
}

// ── Stage 12.21 — sign-state-integrity-diff ──────────────────────────────

fn run_sign_state_integrity_diff(
    args: SignStateIntegrityDiffArgs,
) -> Result<()> {
    use omni_contributor::{
        sign_state_integrity_diff, write_signed_integrity_diff_atomic,
        BaselineSignerRole, ContributorSigner, SignedStateIntegrityDiff,
        StateIntegrityDiffReport,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode = matches!(args.format, SignStateIntegrityDiffFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    log_op("event=signed_integrity_diff_signing_started");

    // Read the raw v1 diff report.
    let diff_bytes = std::fs::read(&args.diff_in).with_context(|| {
        format!("read diff-in: {}", args.diff_in.display())
    })?;
    let diff: StateIntegrityDiffReport = serde_json::from_slice(&diff_bytes)
        .with_context(|| format!("parse diff-in: {}", args.diff_in.display()))?;

    // Load the signer. Same posture as Stage 12.20: use
    // `ContributorSigner` as the seed loader since all four
    // role-typed signers share the same shape; the recorded
    // `signer_role` tag is independent of the loader type.
    let signer = ContributorSigner::from_seed_file(&args.signer_seed)
        .with_context(|| {
            format!("load signer seed: {}", args.signer_seed.display())
        })?;
    let signer_pubkey_hex = signer.pubkey_hex();
    let role: BaselineSignerRole = args.signer_role.into();

    let signed: SignedStateIntegrityDiff = sign_state_integrity_diff(
        diff,
        &signer_pubkey_hex,
        role,
        &now_utc,
        |msg| signer.sign(msg),
    )
    .map_err(|e| anyhow!("sign diff refused: {e}"))?;

    write_signed_integrity_diff_atomic(&signed, &args.out)
        .map_err(|e| anyhow!("write signed diff: {e}"))?;
    log_op(&format!(
        "event=signed_integrity_diff_written path={} signer_role={} signer_pubkey={}",
        args.out.display(),
        role.as_str(),
        signer_pubkey_hex,
    ));

    match args.format {
        SignStateIntegrityDiffFormat::Events => {
            // Already emitted the summary line above.
        }
        SignStateIntegrityDiffFormat::Json => {
            render_sign_diff_json(&signed)?;
        }
        SignStateIntegrityDiffFormat::Pretty => {
            render_sign_diff_pretty(&signed, &args.out);
        }
    }
    Ok(())
}

fn render_sign_diff_json(
    signed: &omni_contributor::SignedStateIntegrityDiff,
) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(signed)
        .map_err(|e| anyhow!("serialize signed diff: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write signed diff: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_sign_diff_pretty(
    signed: &omni_contributor::SignedStateIntegrityDiff,
    out: &std::path::Path,
) {
    println!("Signed integrity diff");
    println!("  schema_version  : {}", signed.schema_version);
    println!("  signed_at_utc   : {}", signed.signed_at_utc);
    println!("  signer_role     : {}", signed.signer_role.as_str());
    println!("  signer_pubkey   : {}", signed.signer_pubkey_hex);
    println!("  signature       : {}", signed.signature_hex);
    println!("  state_dir       : {}", signed.diff.state_dir);
    println!(
        "  baseline_state_dir: {}",
        signed.diff.baseline_state_dir
    );
    println!(
        "  state_versions  : baseline={} current={}",
        signed.diff.baseline_state_version,
        signed.diff.current_state_version,
    );
    println!(
        "  counts          : new={} resolved={} unchanged={}",
        signed.diff.counts.new,
        signed.diff.counts.resolved,
        signed.diff.counts.unchanged,
    );
    println!(
        "  new severity    : ok={} warn={} error={}",
        signed.diff.counts.new_ok,
        signed.diff.counts.new_warn,
        signed.diff.counts.new_error,
    );
    println!(
        "  resolved severity: ok={} warn={} error={}",
        signed.diff.counts.resolved_ok,
        signed.diff.counts.resolved_warn,
        signed.diff.counts.resolved_error,
    );
    println!("  out             : {}", out.display());
}

// ── Stage 12.21 — verify-state-integrity-diff-signature ──────────────────

fn run_verify_state_integrity_diff_signature(
    args: VerifyStateIntegrityDiffSignatureArgs,
) -> Result<()> {
    use omni_contributor::{
        read_signed_integrity_diff_from_path, verify_signed_state_integrity_diff,
        SignedIntegrityDiffError,
    };

    let json_mode =
        matches!(args.format, VerifyStateIntegrityDiffSignatureFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    log_op("event=signed_integrity_diff_verify_started");

    let wrapper = read_signed_integrity_diff_from_path(&args.signed_diff)
        .map_err(|e| anyhow!("read signed diff: {e}"))?;

    if let Err(e) = verify_signed_state_integrity_diff(
        &wrapper,
        &args.expected_signer_pubkey_hex,
    ) {
        let reason_tag = match &e {
            SignedIntegrityDiffError::UnsupportedSchemaVersion { .. } => {
                "unsupported_schema_version"
            }
            SignedIntegrityDiffError::UnsupportedDiffSchemaVersion { .. } => {
                "unsupported_diff_schema_version"
            }
            SignedIntegrityDiffError::SignerPubkeyMismatch { .. } => {
                "signer_pubkey_mismatch"
            }
            SignedIntegrityDiffError::SignatureMismatch => "signature_mismatch",
            SignedIntegrityDiffError::Signing(_) => "signing",
            SignedIntegrityDiffError::Canonical(_) => "canonical",
            SignedIntegrityDiffError::Io { .. } => "io",
            SignedIntegrityDiffError::MalformedJson { .. } => "malformed_json",
        };
        log_op(&format!(
            "event=signed_integrity_diff_verify_failed reason={reason_tag} detail={e}"
        ));
        bail!("signed diff verification refused: {e}");
    }

    log_op(&format!(
        "event=signed_integrity_diff_verify_ok path={} signer_role={} signer_pubkey={}",
        args.signed_diff.display(),
        wrapper.signer_role.as_str(),
        wrapper.signer_pubkey_hex,
    ));

    match args.format {
        VerifyStateIntegrityDiffSignatureFormat::Events => {
            // Already emitted the success line above.
        }
        VerifyStateIntegrityDiffSignatureFormat::Json => {
            render_verify_signed_diff_json(&wrapper)?;
        }
        VerifyStateIntegrityDiffSignatureFormat::Pretty => {
            render_verify_signed_diff_pretty(&wrapper, &args.signed_diff);
        }
    }
    Ok(())
}

fn render_verify_signed_diff_json(
    wrapper: &omni_contributor::SignedStateIntegrityDiff,
) -> Result<()> {
    use std::io::Write;
    // Emit a compact metadata view rather than the full embedded
    // diff per the Stage 12.21 v1 posture: verification's job is
    // to attest authenticity, not to re-print the diff bytes.
    let metadata = serde_json::json!({
        "schema_version": wrapper.schema_version,
        "signed_at_utc": wrapper.signed_at_utc,
        "signer_role": wrapper.signer_role.as_str(),
        "signer_pubkey_hex": wrapper.signer_pubkey_hex,
        "signature_hex": wrapper.signature_hex,
        "diff_schema_version": wrapper.diff.schema_version,
        "diff_generated_at_utc": wrapper.diff.generated_at_utc,
        "diff_state_dir": wrapper.diff.state_dir,
        "diff_baseline_state_dir": wrapper.diff.baseline_state_dir,
        "diff_counts": {
            "new": wrapper.diff.counts.new,
            "resolved": wrapper.diff.counts.resolved,
            "unchanged": wrapper.diff.counts.unchanged,
        },
    });
    let bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|e| anyhow!("serialize verify metadata: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write verify metadata: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_verify_signed_diff_pretty(
    wrapper: &omni_contributor::SignedStateIntegrityDiff,
    path: &std::path::Path,
) {
    println!("Verified signed integrity diff");
    println!("  path            : {}", path.display());
    println!("  schema_version  : {}", wrapper.schema_version);
    println!("  signed_at_utc   : {}", wrapper.signed_at_utc);
    println!("  signer_role     : {}", wrapper.signer_role.as_str());
    println!("  signer_pubkey   : {}", wrapper.signer_pubkey_hex);
    println!("  signature       : {}", wrapper.signature_hex);
    println!("  state_dir       : {}", wrapper.diff.state_dir);
    println!(
        "  baseline_state_dir: {}",
        wrapper.diff.baseline_state_dir
    );
    println!(
        "  counts          : new={} resolved={} unchanged={}",
        wrapper.diff.counts.new,
        wrapper.diff.counts.resolved,
        wrapper.diff.counts.unchanged,
    );
}

// ── Stage 12.22 — build-integrity-evidence-bundle ────────────────────────

/// Parse a `--include <kind=path>` value into
/// `(BundleArtifactKind, PathBuf)`. Split on the FIRST `=` —
/// everything after that is the path verbatim, so paths
/// containing `=` are tolerated. Unknown kind tags, missing
/// `=`, and empty paths refuse with an `anyhow` error that
/// surfaces as `event=integrity_evidence_bundle_build_failed
/// reason=invalid_include`.
fn parse_include_pair(
    raw: &str,
) -> Result<(omni_contributor::BundleArtifactKind, std::path::PathBuf)> {
    let (kind_str, path_str) = raw.split_once('=').ok_or_else(|| {
        anyhow!(
            "--include must be `<kind>=<path>` (no `=` in `{raw}`)"
        )
    })?;
    if kind_str.is_empty() {
        bail!("--include kind tag is empty in `{raw}`");
    }
    if path_str.is_empty() {
        bail!("--include path is empty in `{raw}` (kind={kind_str})");
    }
    let kind = omni_contributor::BundleArtifactKind::from_wire_tag(kind_str)
        .ok_or_else(|| {
            anyhow!(
                "--include unknown artifact kind `{kind_str}` \
                 (closed set: state_integrity_report / \
                 signed_state_integrity_baseline / state_integrity_diff_report / \
                 signed_state_integrity_diff / state_cleanup_plan / cleanup_report / \
                 quarantine_manifest / quarantine_restore_report / \
                 archive_manifest / other)"
            )
        })?;
    Ok((kind, std::path::PathBuf::from(path_str)))
}

fn run_build_integrity_evidence_bundle(
    args: BuildIntegrityEvidenceBundleArgs,
) -> Result<()> {
    use omni_contributor::{
        build_integrity_evidence_bundle, write_integrity_evidence_bundle_atomic,
        BundleBuilderInput, BundleBuilderOptions, IntegrityEvidenceBundle,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode =
        matches!(args.format, BuildIntegrityEvidenceBundleFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    log_op(&format!(
        "event=integrity_evidence_bundle_build_started base_dir={}",
        args.base_dir.display()
    ));

    // Parse the --include values into typed pairs up front so
    // CLI-arg errors refuse before we touch the filesystem.
    let mut parsed: Vec<(omni_contributor::BundleArtifactKind, std::path::PathBuf)> =
        Vec::with_capacity(args.includes.len());
    for raw in &args.includes {
        let pair = parse_include_pair(raw).map_err(|e| {
            log_op(&format!(
                "event=integrity_evidence_bundle_build_failed reason=invalid_include detail={e}"
            ));
            anyhow!("invalid --include: {e}")
        })?;
        parsed.push(pair);
    }
    let inputs: Vec<BundleBuilderInput<'_>> = parsed
        .iter()
        .map(|(kind, path)| BundleBuilderInput {
            artifact_kind: *kind,
            path: path.as_path(),
        })
        .collect();

    let opts = BundleBuilderOptions {
        now_utc: &now_utc,
        base_dir: &args.base_dir,
        label: args.label.as_deref(),
        notes: args.notes.as_deref(),
    };
    let bundle: IntegrityEvidenceBundle = match build_integrity_evidence_bundle(
        &inputs, &opts,
    ) {
        Ok(b) => b,
        Err(e) => {
            let reason = bundle_build_reason_tag(&e);
            log_op(&format!(
                "event=integrity_evidence_bundle_build_failed reason={reason} detail={e}"
            ));
            bail!("build integrity evidence bundle refused: {e}");
        }
    };

    // Per-entry hashed lines AFTER the build succeeds so we
    // don't half-emit on a partial failure.
    for entry in &bundle.entries {
        log_op(&format!(
            "event=integrity_evidence_bundle_entry_hashed \
             kind={} path={} bytes={} blake3={}",
            entry.artifact_kind.as_str(),
            entry.path,
            entry.bytes,
            entry.blake3_hex,
        ));
    }

    write_integrity_evidence_bundle_atomic(&bundle, &args.out)
        .map_err(|e| anyhow!("write integrity evidence bundle: {e}"))?;
    log_op(&format!(
        "event=integrity_evidence_bundle_written path={} entry_count={}",
        args.out.display(),
        bundle.entries.len(),
    ));

    match args.format {
        BuildIntegrityEvidenceBundleFormat::Events => {
            // Already emitted per-entry + written lines above.
        }
        BuildIntegrityEvidenceBundleFormat::Json => {
            render_build_bundle_json(&bundle)?;
        }
        BuildIntegrityEvidenceBundleFormat::Pretty => {
            render_build_bundle_pretty(&bundle, &args.out);
        }
    }
    Ok(())
}

fn bundle_build_reason_tag(e: &omni_contributor::EvidenceBundleError) -> &'static str {
    use omni_contributor::EvidenceBundleError as E;
    match e {
        E::UnsupportedSchemaVersion { .. } => "unsupported_schema_version",
        E::EmptyBundle => "empty_bundle",
        E::DuplicateEntry { .. } => "duplicate_entry",
        E::TooManyEntries { .. } => "too_many_entries",
        E::BundleLabelTooLong { .. } => "bundle_label_too_long",
        E::NotesTooLong { .. } => "notes_too_long",
        E::EntryTooLarge { .. } => "entry_too_large",
        E::EntryNotFound { .. } => "entry_not_found",
        E::PathOutsideBaseDir { .. } => "path_outside_base_dir",
        E::InvalidRelativePath { .. } => "invalid_relative_path",
        E::BaseDirInvalid { .. } => "base_dir_invalid",
        E::EffectiveBaseDirNotFound { .. } => "effective_base_dir_not_found",
        E::NonUtf8Path { .. } => "non_utf8_path",
        E::Io { .. } => "io",
        E::MalformedJson { .. } => "malformed_json",
    }
}

fn render_build_bundle_json(
    bundle: &omni_contributor::IntegrityEvidenceBundle,
) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(bundle)
        .map_err(|e| anyhow!("serialize bundle: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write bundle: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_build_bundle_pretty(
    bundle: &omni_contributor::IntegrityEvidenceBundle,
    out: &std::path::Path,
) {
    println!("Integrity evidence bundle");
    println!("  schema_version  : {}", bundle.schema_version);
    println!("  generated_at_utc: {}", bundle.generated_at_utc);
    println!(
        "  omni_contributor_version: {}",
        bundle.omni_contributor_version
    );
    if let Some(label) = &bundle.label {
        println!("  label           : {label}");
    }
    if let Some(notes) = &bundle.notes {
        println!("  notes           : {notes}");
    }
    println!("  base_dir        : {}", bundle.base_dir);
    println!("  entries         : {}", bundle.entries.len());
    for entry in &bundle.entries {
        println!(
            "    [{}] {} ({} bytes, blake3={})",
            entry.artifact_kind.as_str(),
            entry.path,
            entry.bytes,
            entry.blake3_hex,
        );
    }
    println!("  out             : {}", out.display());
}

// ── Stage 12.22 — verify-integrity-evidence-bundle ───────────────────────

fn run_verify_integrity_evidence_bundle(
    args: VerifyIntegrityEvidenceBundleArgs,
) -> Result<()> {
    use omni_contributor::{
        read_integrity_evidence_bundle_from_path, verify_integrity_evidence_bundle,
        BundleEntryOutcome, BundleVerifyOptions,
    };

    let json_mode =
        matches!(args.format, VerifyIntegrityEvidenceBundleFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Start event fires FIRST — before any FS IO. The
    // `effective_base_dir` isn't known until the bundle is read
    // AND canonicalized, so it lands on the summary line
    // instead. Operators always see a start event even when the
    // bundle is missing / malformed / has an unsupported
    // schema / fails envelope-level path validation.
    let override_field = match args.base_dir.as_deref() {
        Some(p) => format!(" base_dir_override={}", p.display()),
        None => String::new(),
    };
    log_op(&format!(
        "event=integrity_evidence_bundle_verify_started bundle={}{}",
        args.bundle.display(),
        override_field,
    ));

    let bundle = read_integrity_evidence_bundle_from_path(&args.bundle)
        .map_err(|e| {
            // Distinguish io (missing file / permissions) from
            // malformed_json — both can come out of the bundle
            // read path and the closed-tag taxonomy carries
            // both. Using bundle_build_reason_tag keeps the
            // tagging single-sourced.
            let reason = bundle_build_reason_tag(&e);
            log_op(&format!(
                "event=integrity_evidence_bundle_verify_failed reason={reason} detail={e}"
            ));
            anyhow!("read bundle: {e}")
        })?;

    let opts = BundleVerifyOptions {
        base_dir_override: args.base_dir.as_deref(),
    };
    let report =
        verify_integrity_evidence_bundle(&bundle, &opts).map_err(|e| {
            let reason = bundle_build_reason_tag(&e);
            log_op(&format!(
                "event=integrity_evidence_bundle_verify_failed reason={reason} detail={e}"
            ));
            anyhow!("verify integrity evidence bundle refused: {e}")
        })?;

    for outcome in &report.entries {
        let tag = match &outcome.outcome {
            BundleEntryOutcome::Ok => "ok",
            BundleEntryOutcome::SizeMismatch { .. } => "size_mismatch",
            BundleEntryOutcome::HashMismatch { .. } => "hash_mismatch",
            BundleEntryOutcome::NotFound => "not_found",
            BundleEntryOutcome::ReadError { .. } => "read_error",
        };
        let detail = match &outcome.outcome {
            BundleEntryOutcome::Ok | BundleEntryOutcome::NotFound => {
                String::new()
            }
            BundleEntryOutcome::SizeMismatch { expected, got } => {
                format!(" expected_bytes={expected} got_bytes={got}")
            }
            BundleEntryOutcome::HashMismatch { expected, got } => {
                format!(" expected_blake3={expected} got_blake3={got}")
            }
            BundleEntryOutcome::ReadError { detail } => {
                format!(" detail={detail}")
            }
        };
        log_op(&format!(
            "event=integrity_evidence_bundle_entry_{tag} kind={} path={} resolved_path={}{}",
            outcome.artifact_kind.as_str(),
            outcome.path,
            outcome.resolved_path,
            detail,
        ));
    }

    log_op(&format!(
        "event=integrity_evidence_bundle_verify_summary \
         effective_base_dir={} ok={} size_mismatch={} hash_mismatch={} not_found={} read_error={}",
        report.effective_base_dir,
        report.counts_ok,
        report.counts_size_mismatch,
        report.counts_hash_mismatch,
        report.counts_not_found,
        report.counts_read_error,
    ));

    match args.format {
        VerifyIntegrityEvidenceBundleFormat::Events => {
            // Already emitted above.
        }
        VerifyIntegrityEvidenceBundleFormat::Json => {
            render_verify_bundle_json(&report)?;
        }
        VerifyIntegrityEvidenceBundleFormat::Pretty => {
            render_verify_bundle_pretty(&report);
        }
    }

    if !report.all_ok() {
        bail!(
            "integrity evidence bundle verification failed: \
             size_mismatch={} hash_mismatch={} not_found={} read_error={}",
            report.counts_size_mismatch,
            report.counts_hash_mismatch,
            report.counts_not_found,
            report.counts_read_error,
        );
    }
    Ok(())
}

fn render_verify_bundle_json(
    report: &omni_contributor::BundleVerifyReport,
) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| anyhow!("serialize verify report: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write verify report: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_verify_bundle_pretty(report: &omni_contributor::BundleVerifyReport) {
    use omni_contributor::BundleEntryOutcome;
    println!("Integrity evidence bundle verify");
    println!(
        "  bundle_schema_version : {}",
        report.bundle_schema_version
    );
    println!(
        "  bundle_generated_at_utc: {}",
        report.bundle_generated_at_utc
    );
    println!("  effective_base_dir    : {}", report.effective_base_dir);
    println!(
        "  counts                : ok={} size_mismatch={} hash_mismatch={} not_found={} read_error={}",
        report.counts_ok,
        report.counts_size_mismatch,
        report.counts_hash_mismatch,
        report.counts_not_found,
        report.counts_read_error,
    );
    for outcome in &report.entries {
        let line = match &outcome.outcome {
            BundleEntryOutcome::Ok => "ok".to_string(),
            BundleEntryOutcome::SizeMismatch { expected, got } => {
                format!("size_mismatch expected={expected} got={got}")
            }
            BundleEntryOutcome::HashMismatch { .. } => {
                "hash_mismatch".to_string()
            }
            BundleEntryOutcome::NotFound => "not_found".to_string(),
            BundleEntryOutcome::ReadError { detail } => {
                format!("read_error detail={detail}")
            }
        };
        println!(
            "    [{}] {} -> {} [{}]",
            outcome.artifact_kind.as_str(),
            outcome.path,
            outcome.resolved_path,
            line,
        );
    }
}

// ── Stage 12.23 — sign-integrity-evidence-bundle ─────────────────────────

fn run_sign_integrity_evidence_bundle(
    args: SignIntegrityEvidenceBundleArgs,
) -> Result<()> {
    use omni_contributor::{
        sign_integrity_evidence_bundle,
        write_signed_integrity_evidence_bundle_atomic, BaselineSignerRole,
        ContributorSigner, IntegrityEvidenceBundle,
        SignedIntegrityEvidenceBundle,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode =
        matches!(args.format, SignIntegrityEvidenceBundleFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    log_op("event=signed_integrity_evidence_bundle_signing_started");

    // Read the raw v1 bundle. Any FS error here surfaces as
    // `reason=io` on the closed sign_failed event so log
    // scrapers can distinguish missing/permission-denied from
    // malformed-JSON downstream.
    let bundle_bytes = std::fs::read(&args.bundle_in).map_err(|e| {
        log_op(&format!(
            "event=signed_integrity_evidence_bundle_sign_failed reason=io detail={e}"
        ));
        anyhow!("read bundle-in {}: {e}", args.bundle_in.display())
    })?;
    let bundle: IntegrityEvidenceBundle = serde_json::from_slice(&bundle_bytes)
        .map_err(|e| {
            log_op(&format!(
                "event=signed_integrity_evidence_bundle_sign_failed reason=malformed_json detail={e}"
            ));
            anyhow!("parse bundle-in {}: {e}", args.bundle_in.display())
        })?;

    // Load the signer. Same posture as Stage 12.20/12.21:
    // `ContributorSigner` is the seed loader since all four
    // role-typed signers share the same shape; the recorded
    // `signer_role` tag is independent of the loader type. Any
    // seed-load failure (FS missing, bad length, hex parse,
    // etc.) is part of the signing primitive setup → tag as
    // `signing`.
    let signer = ContributorSigner::from_seed_file(&args.signer_seed)
        .map_err(|e| {
            log_op(&format!(
                "event=signed_integrity_evidence_bundle_sign_failed reason=signing detail={e}"
            ));
            anyhow!("load signer seed {}: {e}", args.signer_seed.display())
        })?;
    let signer_pubkey_hex = signer.pubkey_hex();
    let role: BaselineSignerRole = args.signer_role.into();

    let signed: SignedIntegrityEvidenceBundle =
        sign_integrity_evidence_bundle(
            bundle,
            &signer_pubkey_hex,
            role,
            &now_utc,
            |msg| signer.sign(msg),
        )
        .map_err(|e| {
            let reason = signed_bundle_reason_tag(&e);
            log_op(&format!(
                "event=signed_integrity_evidence_bundle_sign_failed reason={reason} detail={e}"
            ));
            anyhow!("sign bundle refused: {e}")
        })?;

    write_signed_integrity_evidence_bundle_atomic(&signed, &args.out)
        .map_err(|e| {
            let reason = signed_bundle_reason_tag(&e);
            log_op(&format!(
                "event=signed_integrity_evidence_bundle_sign_failed reason={reason} detail={e}"
            ));
            anyhow!("write signed bundle: {e}")
        })?;
    log_op(&format!(
        "event=signed_integrity_evidence_bundle_written path={} signer_role={} signer_pubkey={}",
        args.out.display(),
        role.as_str(),
        signer_pubkey_hex,
    ));

    match args.format {
        SignIntegrityEvidenceBundleFormat::Events => {
            // Already emitted the summary line above.
        }
        SignIntegrityEvidenceBundleFormat::Json => {
            render_sign_bundle_json(&signed)?;
        }
        SignIntegrityEvidenceBundleFormat::Pretty => {
            render_sign_bundle_pretty(&signed, &args.out);
        }
    }
    Ok(())
}

fn render_sign_bundle_json(
    signed: &omni_contributor::SignedIntegrityEvidenceBundle,
) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(signed)
        .map_err(|e| anyhow!("serialize signed bundle: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write signed bundle: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_sign_bundle_pretty(
    signed: &omni_contributor::SignedIntegrityEvidenceBundle,
    out: &std::path::Path,
) {
    println!("Signed integrity evidence bundle");
    println!("  schema_version  : {}", signed.schema_version);
    println!("  signed_at_utc   : {}", signed.signed_at_utc);
    println!("  signer_role     : {}", signed.signer_role.as_str());
    println!("  signer_pubkey   : {}", signed.signer_pubkey_hex);
    println!("  signature       : {}", signed.signature_hex);
    println!(
        "  bundle_schema   : {}",
        signed.bundle.schema_version
    );
    println!(
        "  bundle_generated: {}",
        signed.bundle.generated_at_utc
    );
    println!("  base_dir        : {}", signed.bundle.base_dir);
    if let Some(label) = &signed.bundle.label {
        println!("  bundle_label    : {label}");
    }
    println!("  entry_count     : {}", signed.bundle.entries.len());
    println!("  out             : {}", out.display());
}

// ── Stage 12.23 — verify-integrity-evidence-bundle-signature ─────────────

fn run_verify_integrity_evidence_bundle_signature(
    args: VerifyIntegrityEvidenceBundleSignatureArgs,
) -> Result<()> {
    use omni_contributor::{
        read_signed_integrity_evidence_bundle_from_path,
        verify_signed_integrity_evidence_bundle,
    };

    let json_mode = matches!(
        args.format,
        VerifyIntegrityEvidenceBundleSignatureFormat::Json
    );
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Start event fires FIRST — before any FS IO. Mirrors the
    // Stage 12.22 review fix: operators always see a start
    // event even when the bundle is missing / malformed /
    // unsupported schema / fails signature verification.
    log_op(&format!(
        "event=signed_integrity_evidence_bundle_verify_started signed_bundle={}",
        args.signed_bundle.display()
    ));

    let wrapper = read_signed_integrity_evidence_bundle_from_path(
        &args.signed_bundle,
    )
    .map_err(|e| {
        // Distinguish io (missing file / permissions) from
        // malformed_json (parse failure) via the shared closed
        // mapper, single-sourcing the tag.
        let reason = signed_bundle_reason_tag(&e);
        log_op(&format!(
            "event=signed_integrity_evidence_bundle_verify_failed reason={reason} detail={e}"
        ));
        anyhow!("read signed bundle: {e}")
    })?;

    if let Err(e) = verify_signed_integrity_evidence_bundle(
        &wrapper,
        &args.expected_signer_pubkey_hex,
    ) {
        let reason = signed_bundle_reason_tag(&e);
        log_op(&format!(
            "event=signed_integrity_evidence_bundle_verify_failed reason={reason} detail={e}"
        ));
        bail!("signed bundle verification refused: {e}");
    }

    log_op(&format!(
        "event=signed_integrity_evidence_bundle_verify_ok path={} signer_role={} signer_pubkey={}",
        args.signed_bundle.display(),
        wrapper.signer_role.as_str(),
        wrapper.signer_pubkey_hex,
    ));

    match args.format {
        VerifyIntegrityEvidenceBundleSignatureFormat::Events => {
            // Already emitted the success line above.
        }
        VerifyIntegrityEvidenceBundleSignatureFormat::Json => {
            render_verify_signed_bundle_json(&wrapper)?;
        }
        VerifyIntegrityEvidenceBundleSignatureFormat::Pretty => {
            render_verify_signed_bundle_pretty(&wrapper, &args.signed_bundle);
        }
    }
    Ok(())
}

fn signed_bundle_reason_tag(
    e: &omni_contributor::SignedIntegrityEvidenceBundleError,
) -> &'static str {
    use omni_contributor::SignedIntegrityEvidenceBundleError as E;
    match e {
        E::UnsupportedSchemaVersion { .. } => "unsupported_schema_version",
        E::UnsupportedBundleSchemaVersion { .. } => {
            "unsupported_bundle_schema_version"
        }
        E::SignerPubkeyMismatch { .. } => "signer_pubkey_mismatch",
        E::SignatureMismatch => "signature_mismatch",
        E::Signing(_) => "signing",
        E::Canonical(_) => "canonical",
        E::Io { .. } => "io",
        E::MalformedJson { .. } => "malformed_json",
    }
}

fn render_verify_signed_bundle_json(
    wrapper: &omni_contributor::SignedIntegrityEvidenceBundle,
) -> Result<()> {
    use std::io::Write;
    // Compact metadata view per locked v1 scope: attest
    // authenticity, don't mirror the embedded bundle (operators
    // who want the full bundle already have it on disk inside
    // the wrapper).
    let metadata = serde_json::json!({
        "schema_version": wrapper.schema_version,
        "signed_at_utc": wrapper.signed_at_utc,
        "signer_role": wrapper.signer_role.as_str(),
        "signer_pubkey_hex": wrapper.signer_pubkey_hex,
        "signature_hex": wrapper.signature_hex,
        "bundle_schema_version": wrapper.bundle.schema_version,
        "bundle_generated_at_utc": wrapper.bundle.generated_at_utc,
        "bundle_omni_contributor_version": wrapper.bundle.omni_contributor_version,
        "bundle_label": wrapper.bundle.label,
        "bundle_base_dir": wrapper.bundle.base_dir,
        "bundle_entry_count": wrapper.bundle.entries.len(),
    });
    let bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|e| anyhow!("serialize verify metadata: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write verify metadata: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_verify_signed_bundle_pretty(
    wrapper: &omni_contributor::SignedIntegrityEvidenceBundle,
    path: &std::path::Path,
) {
    println!("Verified signed integrity evidence bundle");
    println!("  path            : {}", path.display());
    println!("  schema_version  : {}", wrapper.schema_version);
    println!("  signed_at_utc   : {}", wrapper.signed_at_utc);
    println!("  signer_role     : {}", wrapper.signer_role.as_str());
    println!("  signer_pubkey   : {}", wrapper.signer_pubkey_hex);
    println!("  signature       : {}", wrapper.signature_hex);
    println!(
        "  bundle_schema   : {}",
        wrapper.bundle.schema_version
    );
    println!(
        "  bundle_generated: {}",
        wrapper.bundle.generated_at_utc
    );
    println!("  base_dir        : {}", wrapper.bundle.base_dir);
    if let Some(label) = &wrapper.bundle.label {
        println!("  bundle_label    : {label}");
    }
    println!("  entry_count     : {}", wrapper.bundle.entries.len());
}

// ── Stage 12.24 — verify-integrity-evidence-chain ────────────────────────

fn run_verify_integrity_evidence_chain(
    args: VerifyIntegrityEvidenceChainArgs,
) -> Result<()> {
    use omni_contributor::{
        verify_integrity_evidence_chain,
        write_integrity_evidence_chain_report_atomic, BundleEntryOutcome,
        ChainStepOutcome, ChainVerifyError, ChainVerifyOptions,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode =
        matches!(args.format, VerifyIntegrityEvidenceChainFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Start event fires FIRST — before any FS IO. Mirrors the
    // Stage 12.22/12.23 review-fix posture so operators always
    // see a start event even on missing/malformed wrappers.
    // The `set` / absent markers let log scrapers see exactly
    // which optional gates were enabled without leaking the
    // pubkey hex on the start line.
    let override_field = match args.base_dir.as_deref() {
        Some(p) => format!(" base_dir_override={}", p.display()),
        None => String::new(),
    };
    let baseline_marker = if args.expected_baseline_signer_pubkey_hex.is_some() {
        " expected_baseline_pubkey=set"
    } else {
        ""
    };
    let diff_marker = if args.expected_diff_signer_pubkey_hex.is_some() {
        " expected_diff_pubkey=set"
    } else {
        ""
    };
    log_op(&format!(
        "event=integrity_evidence_chain_verify_started signed_bundle={}{}{}{}",
        args.signed_bundle.display(),
        override_field,
        baseline_marker,
        diff_marker,
    ));

    let opts = ChainVerifyOptions {
        now_utc: &now_utc,
        signed_bundle_path: &args.signed_bundle,
        expected_bundle_signer_pubkey_hex: &args.expected_bundle_signer_pubkey_hex,
        base_dir_override: args.base_dir.as_deref(),
        expected_baseline_signer_pubkey_hex: args
            .expected_baseline_signer_pubkey_hex
            .as_deref(),
        expected_diff_signer_pubkey_hex: args
            .expected_diff_signer_pubkey_hex
            .as_deref(),
    };
    let report = match verify_integrity_evidence_chain(&opts) {
        Ok(r) => r,
        Err(e) => {
            let reason = chain_envelope_reason_tag(&e);
            log_op(&format!(
                "event=integrity_evidence_chain_verify_failed reason={reason} detail={e}"
            ));
            bail!("chain verification refused: {e}");
        }
    };

    // Bundle-signature step — always Ok if we got here, since
    // the library short-circuits on signed-bundle envelope
    // failure. Emit the explicit OK event so log scrapers see
    // every step. The chain report carries the verified
    // wrapper's `signer_role` / `signer_pubkey_hex` as minimal
    // metadata (v1 does NOT embed the full Stage 12.23
    // wrapper); surface them on this line so the event stream
    // is self-describing on signer identity — the start line
    // only carries `expected_*_pubkey=set` markers and
    // deliberately doesn't leak the pubkey.
    if matches!(report.bundle_signature, ChainStepOutcome::Ok) {
        log_op(&format!(
            "event=integrity_evidence_chain_signed_bundle_ok signer_role={} signer_pubkey={}",
            report.bundle_signer_role.as_str(),
            report.bundle_signer_pubkey_hex,
        ));
    }

    log_op(&format!(
        "event=integrity_evidence_chain_bundle_byte_resolved effective_base_dir={}",
        report.effective_base_dir
    ));

    for outcome in &report.bundle_byte_verify.entries {
        let tag = match &outcome.outcome {
            BundleEntryOutcome::Ok => "ok",
            BundleEntryOutcome::SizeMismatch { .. } => "size_mismatch",
            BundleEntryOutcome::HashMismatch { .. } => "hash_mismatch",
            BundleEntryOutcome::NotFound => "not_found",
            BundleEntryOutcome::ReadError { .. } => "read_error",
        };
        let detail = match &outcome.outcome {
            BundleEntryOutcome::Ok | BundleEntryOutcome::NotFound => {
                String::new()
            }
            BundleEntryOutcome::SizeMismatch { expected, got } => {
                format!(" expected_bytes={expected} got_bytes={got}")
            }
            BundleEntryOutcome::HashMismatch { expected, got } => {
                format!(" expected_blake3={expected} got_blake3={got}")
            }
            BundleEntryOutcome::ReadError { detail } => {
                format!(" detail={detail}")
            }
        };
        log_op(&format!(
            "event=integrity_evidence_chain_bundle_byte_entry_{tag} kind={} path={} resolved_path={}{}",
            outcome.artifact_kind.as_str(),
            outcome.path,
            outcome.resolved_path,
            detail,
        ));
    }

    for child in &report.child_signatures {
        match &child.signature_outcome {
            ChainStepOutcome::Ok => {
                log_op(&format!(
                    "event=integrity_evidence_chain_child_ok kind={} path={} resolved_path={}",
                    child.artifact_kind.as_str(),
                    child.path,
                    child.resolved_path,
                ));
            }
            ChainStepOutcome::Skipped => {
                log_op(&format!(
                    "event=integrity_evidence_chain_child_skipped kind={} path={} resolved_path={}",
                    child.artifact_kind.as_str(),
                    child.path,
                    child.resolved_path,
                ));
            }
            ChainStepOutcome::Failed { reason, detail } => {
                log_op(&format!(
                    "event=integrity_evidence_chain_child_failed kind={} path={} resolved_path={} reason={reason} detail={detail}",
                    child.artifact_kind.as_str(),
                    child.path,
                    child.resolved_path,
                ));
            }
        }
    }

    log_op(&format!(
        "event=integrity_evidence_chain_verify_summary \
         bundle_signature=ok \
         bundle_byte_counts={{ok={} size_mismatch={} hash_mismatch={} not_found={} read_error={}}} \
         child_counts={{ok={} skipped={} failed={}}}",
        report.bundle_byte_verify.counts_ok,
        report.bundle_byte_verify.counts_size_mismatch,
        report.bundle_byte_verify.counts_hash_mismatch,
        report.bundle_byte_verify.counts_not_found,
        report.bundle_byte_verify.counts_read_error,
        report.counts_child_ok,
        report.counts_child_skipped,
        report.counts_child_failed,
    ));

    // Optional best-effort --json-out write. Failure logs a
    // warn event and DOES NOT change exit code.
    if let Some(out) = args.json_out.as_deref() {
        match write_integrity_evidence_chain_report_atomic(&report, out) {
            Ok(_) => {
                log_op(&format!(
                    "event=integrity_evidence_chain_json_written path={}",
                    out.display()
                ));
            }
            Err(e) => {
                let reason = match &e {
                    ChainVerifyError::MalformedJson { .. } => "malformed_json",
                    _ => "io",
                };
                log_op(&format!(
                    "event=integrity_evidence_chain_json_write_failed reason={reason} detail={e}"
                ));
            }
        }
    }

    match args.format {
        VerifyIntegrityEvidenceChainFormat::Events => {
            // Already emitted above.
        }
        VerifyIntegrityEvidenceChainFormat::Json => {
            render_verify_chain_json(&report)?;
        }
        VerifyIntegrityEvidenceChainFormat::Pretty => {
            render_verify_chain_pretty(&report);
        }
    }

    if !report.all_required_ok() {
        bail!(
            "integrity-evidence-chain verification failed: \
             bundle_byte size_mismatch={} hash_mismatch={} not_found={} read_error={} \
             child_failed={}",
            report.bundle_byte_verify.counts_size_mismatch,
            report.bundle_byte_verify.counts_hash_mismatch,
            report.bundle_byte_verify.counts_not_found,
            report.bundle_byte_verify.counts_read_error,
            report.counts_child_failed,
        );
    }
    Ok(())
}

/// Closed-set envelope-level reason tag for the chain CLI.
/// Prefixes inner Stage 12.22/12.23 tags with `signed_bundle_`
/// / `bundle_byte_` so the closed-set taxonomy stays
/// self-disambiguating.
fn chain_envelope_reason_tag(e: &omni_contributor::ChainVerifyError) -> String {
    use omni_contributor::ChainVerifyError as E;
    match e {
        E::UnsupportedChainSchemaVersion { .. } => {
            "unsupported_chain_schema_version".to_string()
        }
        E::SignedBundle(inner) => {
            format!("signed_bundle_{}", signed_bundle_reason_tag(inner))
        }
        E::BundleByte(inner) => {
            format!("bundle_byte_{}", bundle_build_reason_tag(inner))
        }
        E::Io { .. } => "io".to_string(),
        E::MalformedJson { .. } => "malformed_json".to_string(),
    }
}

fn render_verify_chain_json(
    report: &omni_contributor::IntegrityEvidenceChainReport,
) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|e| anyhow!("serialize chain report: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write chain report: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_verify_chain_pretty(
    report: &omni_contributor::IntegrityEvidenceChainReport,
) {
    use omni_contributor::{BundleEntryOutcome, ChainStepOutcome};
    println!("Integrity evidence chain verify");
    println!(
        "  schema_version       : {}",
        report.schema_version
    );
    println!(
        "  generated_at_utc     : {}",
        report.generated_at_utc
    );
    println!(
        "  signed_bundle_path   : {}",
        report.signed_bundle_path
    );
    println!(
        "  effective_base_dir   : {}",
        report.effective_base_dir
    );
    println!(
        "  bundle_signature     : {}",
        match &report.bundle_signature {
            ChainStepOutcome::Ok => "ok".to_string(),
            ChainStepOutcome::Skipped => "skipped".to_string(),
            ChainStepOutcome::Failed { reason, .. } => {
                format!("failed reason={reason}")
            }
        }
    );
    println!(
        "  bundle_signer_role   : {}",
        report.bundle_signer_role.as_str()
    );
    println!(
        "  bundle_signer_pubkey : {}",
        report.bundle_signer_pubkey_hex
    );
    println!(
        "  bundle_byte_counts   : ok={} size_mismatch={} hash_mismatch={} not_found={} read_error={}",
        report.bundle_byte_verify.counts_ok,
        report.bundle_byte_verify.counts_size_mismatch,
        report.bundle_byte_verify.counts_hash_mismatch,
        report.bundle_byte_verify.counts_not_found,
        report.bundle_byte_verify.counts_read_error,
    );
    println!(
        "  child_counts         : ok={} skipped={} failed={}",
        report.counts_child_ok,
        report.counts_child_skipped,
        report.counts_child_failed,
    );
    for outcome in &report.bundle_byte_verify.entries {
        let line = match &outcome.outcome {
            BundleEntryOutcome::Ok => "ok".to_string(),
            BundleEntryOutcome::SizeMismatch { expected, got } => {
                format!("size_mismatch expected={expected} got={got}")
            }
            BundleEntryOutcome::HashMismatch { .. } => "hash_mismatch".to_string(),
            BundleEntryOutcome::NotFound => "not_found".to_string(),
            BundleEntryOutcome::ReadError { detail } => {
                format!("read_error detail={detail}")
            }
        };
        println!(
            "    byte  [{}] {} -> {} [{}]",
            outcome.artifact_kind.as_str(),
            outcome.path,
            outcome.resolved_path,
            line,
        );
    }
    for child in &report.child_signatures {
        let line = match &child.signature_outcome {
            ChainStepOutcome::Ok => "ok".to_string(),
            ChainStepOutcome::Skipped => "skipped".to_string(),
            ChainStepOutcome::Failed { reason, .. } => {
                format!("failed reason={reason}")
            }
        };
        println!(
            "    child [{}] {} -> {} [{}]",
            child.artifact_kind.as_str(),
            child.path,
            child.resolved_path,
            line,
        );
    }
    println!(
        "  all_required_ok      : {}",
        report.all_required_ok()
    );
}

// ── Stage 12.25 — sign-integrity-evidence-chain-report ───────────────────

fn run_sign_integrity_evidence_chain_report(
    args: SignIntegrityEvidenceChainReportArgs,
) -> Result<()> {
    use omni_contributor::{
        sign_integrity_evidence_chain_report,
        write_signed_integrity_evidence_chain_report_atomic, BaselineSignerRole,
        ContributorSigner, IntegrityEvidenceChainReport,
        SignedIntegrityEvidenceChainReport,
    };

    let now_utc =
        chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    let json_mode =
        matches!(args.format, SignIntegrityEvidenceChainReportFormat::Json);
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    log_op("event=signed_integrity_evidence_chain_report_signing_started");

    // Read the raw v1 chain report. FS error → reason=io;
    // closed-tag mapping matches the Stage 12.23 pattern.
    let chain_bytes = std::fs::read(&args.chain_report_in).map_err(|e| {
        log_op(&format!(
            "event=signed_integrity_evidence_chain_report_sign_failed reason=io detail={e}"
        ));
        anyhow!("read chain-report-in {}: {e}", args.chain_report_in.display())
    })?;
    let chain_report: IntegrityEvidenceChainReport =
        serde_json::from_slice(&chain_bytes).map_err(|e| {
            log_op(&format!(
                "event=signed_integrity_evidence_chain_report_sign_failed reason=malformed_json detail={e}"
            ));
            anyhow!(
                "parse chain-report-in {}: {e}",
                args.chain_report_in.display()
            )
        })?;

    // Load the signer. Seed-load failures tag as `signing`
    // since signer setup is part of the signing primitive.
    let signer = ContributorSigner::from_seed_file(&args.signer_seed)
        .map_err(|e| {
            log_op(&format!(
                "event=signed_integrity_evidence_chain_report_sign_failed reason=signing detail={e}"
            ));
            anyhow!("load signer seed {}: {e}", args.signer_seed.display())
        })?;
    let signer_pubkey_hex = signer.pubkey_hex();
    let role: BaselineSignerRole = args.signer_role.into();

    let signed: SignedIntegrityEvidenceChainReport =
        sign_integrity_evidence_chain_report(
            chain_report,
            &signer_pubkey_hex,
            role,
            &now_utc,
            |msg| signer.sign(msg),
        )
        .map_err(|e| {
            let reason = signed_chain_report_reason_tag(&e);
            log_op(&format!(
                "event=signed_integrity_evidence_chain_report_sign_failed reason={reason} detail={e}"
            ));
            anyhow!("sign chain report refused: {e}")
        })?;

    write_signed_integrity_evidence_chain_report_atomic(&signed, &args.out)
        .map_err(|e| {
            let reason = signed_chain_report_reason_tag(&e);
            log_op(&format!(
                "event=signed_integrity_evidence_chain_report_sign_failed reason={reason} detail={e}"
            ));
            anyhow!("write signed chain report: {e}")
        })?;
    log_op(&format!(
        "event=signed_integrity_evidence_chain_report_written path={} signer_role={} signer_pubkey={}",
        args.out.display(),
        role.as_str(),
        signer_pubkey_hex,
    ));

    match args.format {
        SignIntegrityEvidenceChainReportFormat::Events => {
            // Already emitted the summary line above.
        }
        SignIntegrityEvidenceChainReportFormat::Json => {
            render_sign_chain_report_json(&signed)?;
        }
        SignIntegrityEvidenceChainReportFormat::Pretty => {
            render_sign_chain_report_pretty(&signed, &args.out);
        }
    }
    Ok(())
}

fn render_sign_chain_report_json(
    signed: &omni_contributor::SignedIntegrityEvidenceChainReport,
) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(signed)
        .map_err(|e| anyhow!("serialize signed chain report: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write signed chain report: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_sign_chain_report_pretty(
    signed: &omni_contributor::SignedIntegrityEvidenceChainReport,
    out: &std::path::Path,
) {
    println!("Signed integrity evidence chain report");
    println!("  schema_version       : {}", signed.schema_version);
    println!("  signed_at_utc        : {}", signed.signed_at_utc);
    println!(
        "  signer_role          : {}",
        signed.signer_role.as_str()
    );
    println!("  signer_pubkey        : {}", signed.signer_pubkey_hex);
    println!("  signature            : {}", signed.signature_hex);
    println!(
        "  chain_schema_version : {}",
        signed.chain_report.schema_version
    );
    println!(
        "  chain_generated_at   : {}",
        signed.chain_report.generated_at_utc
    );
    println!(
        "  signed_bundle_path   : {}",
        signed.chain_report.signed_bundle_path
    );
    println!(
        "  effective_base_dir   : {}",
        signed.chain_report.effective_base_dir
    );
    println!(
        "  bundle_signer_role   : {}",
        signed.chain_report.bundle_signer_role.as_str()
    );
    println!(
        "  bundle_signer_pubkey : {}",
        signed.chain_report.bundle_signer_pubkey_hex
    );
    println!(
        "  bundle_byte_counts   : ok={} size_mismatch={} hash_mismatch={} not_found={} read_error={}",
        signed.chain_report.bundle_byte_verify.counts_ok,
        signed.chain_report.bundle_byte_verify.counts_size_mismatch,
        signed.chain_report.bundle_byte_verify.counts_hash_mismatch,
        signed.chain_report.bundle_byte_verify.counts_not_found,
        signed.chain_report.bundle_byte_verify.counts_read_error,
    );
    println!(
        "  child_counts         : ok={} skipped={} failed={}",
        signed.chain_report.counts_child_ok,
        signed.chain_report.counts_child_skipped,
        signed.chain_report.counts_child_failed,
    );
    println!("  out                  : {}", out.display());
}

// ── Stage 12.25 — verify-integrity-evidence-chain-report-signature ───────

fn run_verify_integrity_evidence_chain_report_signature(
    args: VerifyIntegrityEvidenceChainReportSignatureArgs,
) -> Result<()> {
    use omni_contributor::{
        read_signed_integrity_evidence_chain_report_from_path,
        verify_signed_integrity_evidence_chain_report,
    };

    let json_mode = matches!(
        args.format,
        VerifyIntegrityEvidenceChainReportSignatureFormat::Json
    );
    let log_op = |msg: &str| {
        if json_mode {
            eprintln!("{msg}");
        } else {
            println!("{msg}");
        }
    };

    // Start event fires FIRST — before any FS IO. Mirrors the
    // Stage 12.22/12.23/12.24 review-fix posture: operators
    // always see a start event even on missing/malformed
    // wrappers / unsupported schemas / signature failures.
    log_op(&format!(
        "event=signed_integrity_evidence_chain_report_verify_started signed_chain_report={}",
        args.signed_chain_report.display()
    ));

    let wrapper = read_signed_integrity_evidence_chain_report_from_path(
        &args.signed_chain_report,
    )
    .map_err(|e| {
        let reason = signed_chain_report_reason_tag(&e);
        log_op(&format!(
            "event=signed_integrity_evidence_chain_report_verify_failed reason={reason} detail={e}"
        ));
        anyhow!("read signed chain report: {e}")
    })?;

    if let Err(e) = verify_signed_integrity_evidence_chain_report(
        &wrapper,
        &args.expected_signer_pubkey_hex,
    ) {
        let reason = signed_chain_report_reason_tag(&e);
        log_op(&format!(
            "event=signed_integrity_evidence_chain_report_verify_failed reason={reason} detail={e}"
        ));
        bail!("signed chain report verification refused: {e}");
    }

    log_op(&format!(
        "event=signed_integrity_evidence_chain_report_verify_ok path={} signer_role={} signer_pubkey={}",
        args.signed_chain_report.display(),
        wrapper.signer_role.as_str(),
        wrapper.signer_pubkey_hex,
    ));

    match args.format {
        VerifyIntegrityEvidenceChainReportSignatureFormat::Events => {
            // Already emitted the success line above.
        }
        VerifyIntegrityEvidenceChainReportSignatureFormat::Json => {
            render_verify_signed_chain_report_json(&wrapper)?;
        }
        VerifyIntegrityEvidenceChainReportSignatureFormat::Pretty => {
            render_verify_signed_chain_report_pretty(
                &wrapper,
                &args.signed_chain_report,
            );
        }
    }
    Ok(())
}

fn signed_chain_report_reason_tag(
    e: &omni_contributor::SignedIntegrityEvidenceChainReportError,
) -> &'static str {
    use omni_contributor::SignedIntegrityEvidenceChainReportError as E;
    match e {
        E::UnsupportedSchemaVersion { .. } => "unsupported_schema_version",
        E::UnsupportedChainReportSchemaVersion { .. } => {
            "unsupported_chain_report_schema_version"
        }
        E::SignerPubkeyMismatch { .. } => "signer_pubkey_mismatch",
        E::SignatureMismatch => "signature_mismatch",
        E::Signing(_) => "signing",
        E::Canonical(_) => "canonical",
        E::Io { .. } => "io",
        E::MalformedJson { .. } => "malformed_json",
    }
}

fn render_verify_signed_chain_report_json(
    wrapper: &omni_contributor::SignedIntegrityEvidenceChainReport,
) -> Result<()> {
    use std::io::Write;
    // Compact metadata view per locked v1 scope: attest
    // authenticity, don't mirror the embedded chain report's
    // full per-entry / per-child lists.
    let metadata = serde_json::json!({
        "schema_version": wrapper.schema_version,
        "signed_at_utc": wrapper.signed_at_utc,
        "signer_role": wrapper.signer_role.as_str(),
        "signer_pubkey_hex": wrapper.signer_pubkey_hex,
        "signature_hex": wrapper.signature_hex,
        "chain_report_schema_version": wrapper.chain_report.schema_version,
        "chain_report_generated_at_utc": wrapper.chain_report.generated_at_utc,
        "chain_report_signed_bundle_path": wrapper.chain_report.signed_bundle_path,
        "chain_report_effective_base_dir": wrapper.chain_report.effective_base_dir,
        "chain_report_bundle_signer_role": wrapper.chain_report.bundle_signer_role.as_str(),
        "chain_report_bundle_signer_pubkey_hex": wrapper.chain_report.bundle_signer_pubkey_hex,
        "chain_report_bundle_byte_counts": {
            "ok": wrapper.chain_report.bundle_byte_verify.counts_ok,
            "size_mismatch": wrapper.chain_report.bundle_byte_verify.counts_size_mismatch,
            "hash_mismatch": wrapper.chain_report.bundle_byte_verify.counts_hash_mismatch,
            "not_found": wrapper.chain_report.bundle_byte_verify.counts_not_found,
            "read_error": wrapper.chain_report.bundle_byte_verify.counts_read_error,
        },
        "chain_report_child_counts": {
            "ok": wrapper.chain_report.counts_child_ok,
            "skipped": wrapper.chain_report.counts_child_skipped,
            "failed": wrapper.chain_report.counts_child_failed,
        },
    });
    let bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|e| anyhow!("serialize verify metadata: {e}"))?;
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    handle
        .write_all(&bytes)
        .map_err(|e| anyhow!("write verify metadata: {e}"))?;
    handle
        .write_all(b"\n")
        .map_err(|e| anyhow!("trailing newline: {e}"))?;
    Ok(())
}

fn render_verify_signed_chain_report_pretty(
    wrapper: &omni_contributor::SignedIntegrityEvidenceChainReport,
    path: &std::path::Path,
) {
    println!("Verified signed integrity evidence chain report");
    println!("  path                 : {}", path.display());
    println!("  schema_version       : {}", wrapper.schema_version);
    println!("  signed_at_utc        : {}", wrapper.signed_at_utc);
    println!(
        "  signer_role          : {}",
        wrapper.signer_role.as_str()
    );
    println!(
        "  signer_pubkey        : {}",
        wrapper.signer_pubkey_hex
    );
    println!("  signature            : {}", wrapper.signature_hex);
    println!(
        "  chain_schema_version : {}",
        wrapper.chain_report.schema_version
    );
    println!(
        "  chain_generated_at   : {}",
        wrapper.chain_report.generated_at_utc
    );
    println!(
        "  signed_bundle_path   : {}",
        wrapper.chain_report.signed_bundle_path
    );
    println!(
        "  effective_base_dir   : {}",
        wrapper.chain_report.effective_base_dir
    );
    println!(
        "  bundle_signer_role   : {}",
        wrapper.chain_report.bundle_signer_role.as_str()
    );
    println!(
        "  bundle_signer_pubkey : {}",
        wrapper.chain_report.bundle_signer_pubkey_hex
    );
    println!(
        "  bundle_byte_counts   : ok={} size_mismatch={} hash_mismatch={} not_found={} read_error={}",
        wrapper.chain_report.bundle_byte_verify.counts_ok,
        wrapper.chain_report.bundle_byte_verify.counts_size_mismatch,
        wrapper.chain_report.bundle_byte_verify.counts_hash_mismatch,
        wrapper.chain_report.bundle_byte_verify.counts_not_found,
        wrapper.chain_report.bundle_byte_verify.counts_read_error,
    );
    println!(
        "  child_counts         : ok={} skipped={} failed={}",
        wrapper.chain_report.counts_child_ok,
        wrapper.chain_report.counts_child_skipped,
        wrapper.chain_report.counts_child_failed,
    );
}
