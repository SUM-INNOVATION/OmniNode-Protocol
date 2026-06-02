//! Stage 12.0 — typed error surface for the contributor crate.

use thiserror::Error;

/// Top-level error for the contributor crate. Wraps the sub-error
/// types so callers (e.g. the `omni-node operator contributor` CLI)
/// can match on category without losing detail.
#[derive(Debug, Error)]
pub enum ContributorError {
    #[error("schema error: {0}")]
    Schema(#[from] SchemaError),

    #[error("canonical encoding error: {0}")]
    Canonical(#[from] CanonicalError),

    #[error("signing error: {0}")]
    Signing(#[from] SigningError),

    #[error("snip error: {0}")]
    Snip(#[from] SnipError),

    #[error("inference runner error: {0}")]
    Runner(#[from] RunnerError),

    #[error("verification error: {0}")]
    Verify(#[from] VerifyError),

    #[error("discovery error: {0}")]
    Discover(#[from] DiscoverError),

    #[error("relay error: {0}")]
    Relay(#[from] RelayError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error(
        "job carries verification_requirement::Stage11dProductionFixedPointMlpProof, \
         which is NOT a Stage 12.0 v1 variant. Production-proof evidence is reserved \
         for a future schema_version: 2 migration."
    )]
    Stage12_0DoesNotProduceProductionProofs,
}

/// Schema-level validation errors (hex widths, ISO 8601, enum
/// well-formedness, half-set dispatcher identity, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SchemaError {
    #[error("unsupported schema_version {got}; this crate ships v1 only")]
    UnsupportedVersion { got: u32 },

    #[error("malformed hash field '{field}': expected 64 lowercase hex chars, got {got:?}")]
    MalformedHash { field: &'static str, got: String },

    #[error(
        "malformed Ed25519 public-key hex field '{field}': expected 64 lowercase hex chars, \
         got {got:?}"
    )]
    MalformedPubkey { field: &'static str, got: String },

    #[error(
        "malformed Ed25519 signature hex field '{field}': expected 128 lowercase hex chars, \
         got {got:?}"
    )]
    MalformedSignature { field: &'static str, got: String },

    #[error("malformed timestamp '{field}': expected RFC 3339 / ISO 8601, got {got:?}")]
    MalformedTimestamp { field: &'static str, got: String },

    #[error(
        "inconsistent dispatcher identity: dispatcher_pubkey_hex is {pubkey_set} but \
         dispatcher_signature_hex is {signature_set}; both must be Some or both None"
    )]
    InconsistentDispatcherIdentity {
        pubkey_set: &'static str,
        signature_set: &'static str,
    },

    #[error("empty tokenizer_id: must be a non-empty UTF-8 identifier")]
    EmptyTokenizerId,

    #[error("empty stage_contributions: result must report at least one stage")]
    EmptyStageContributions,

    #[error("job_id must equal lowercase_hex(job_hash); got job_id={job_id:?}, derived={derived:?}")]
    JobIdMismatch { job_id: String, derived: String },

    // Stage 12.3 — session schema errors.

    #[error("WorkKind::Layers range inverted or empty: start={start}, end={end}")]
    WorkKindLayersInverted { start: u32, end: u32 },

    #[error("WorkKind::Custom label must be non-empty")]
    WorkKindCustomEmptyLabel,

    #[error("WorkKind::Custom label length {got} exceeds bound {max}")]
    WorkKindCustomLabelTooLong { got: usize, max: usize },

    #[error("WorkKind::Custom label must be printable ASCII (0x20..=0x7E)")]
    WorkKindCustomLabelNotPrintableAscii,

    #[error("ContributorJoin.supported_work_unit_kinds must be non-empty")]
    EmptySupportedWorkUnitKinds,

    #[error("ContributorJoin.runner_kind must be non-empty")]
    EmptyRunnerKind,

    #[error("ContributorJoin.runner_kind length {got} exceeds bound {max}")]
    RunnerKindTooLong { got: usize, max: usize },

    #[error("ContributorJoin.runner_kind must be printable ASCII (0x20..=0x7E)")]
    RunnerKindNotPrintableAscii,

    #[error("WorkAssignment.expected_work_units must be > 0")]
    AssignmentZeroExpectedWorkUnits,

    #[error(
        "PartialContributorResult.measured_accounting.stage_contributions must have \
         exactly one entry; got {got}"
    )]
    PartialMustHaveOneStageContribution { got: usize },

    #[error(
        "PartialContributorResult.measured_accounting.stage_contributions[0].contributor_pubkey_hex \
         must equal partial.contributor_pubkey_hex"
    )]
    PartialStageContributorMismatch,

    #[error("AggregatedContributorResult.partial_refs must be non-empty")]
    AggregateEmptyPartialRefs,

    #[error("AggregatedContributorResult.partial_refs[{index}] invalid: {inner}")]
    AggregatePartialRefInvalid {
        index: usize,
        #[source]
        inner: Box<SchemaError>,
    },

    // Stage 12.4 — activation-handoff schema errors.

    #[error("ActivationHandoff.shape must be non-empty")]
    HandoffShapeEmpty,

    #[error("ActivationHandoff.shape rank {got} exceeds bound {max}")]
    HandoffShapeRankTooLarge { got: usize, max: usize },

    #[error("ActivationHandoff.shape contains a zero dimension")]
    HandoffShapeZeroDim,

    #[error("ActivationHandoff.byte_len must be > 0")]
    HandoffByteLenZero,

    #[error("ActivationHandoff.byte_len {got} exceeds bound {max}")]
    HandoffByteLenTooLarge { got: u64, max: u64 },

    #[error(
        "ActivationHandoff.byte_len {declared} does not equal shape-product * dtype \
         bytes-per-element {expected_from_shape_and_dtype}"
    )]
    HandoffByteLenMismatch {
        declared: u64,
        expected_from_shape_and_dtype: u64,
    },

    #[error("ActivationHandoff.chunk_count must be > 0")]
    HandoffChunkCountZero,

    #[error("ActivationHandoff.chunk_count {got} exceeds bound {max}")]
    HandoffChunkCountTooLarge { got: u32, max: u32 },

    #[error("ActivationHandoff.chunk_index {index} out of range for chunk_count {count}")]
    HandoffChunkIndexOutOfRange { index: u32, count: u32 },

    #[error("ActivationHandoff.tensor_chunk_bytes must be non-empty")]
    HandoffChunkBytesEmpty,

    #[error("ActivationHandoff.tensor_chunk_bytes len {got} exceeds bound {max}")]
    HandoffChunkBytesTooLarge { got: u64, max: u64 },

    #[error(
        "single-chunk ActivationHandoff: tensor_chunk_bytes len {chunk_len} must equal \
         byte_len {byte_len}"
    )]
    HandoffSingleChunkLenMismatch { chunk_len: u64, byte_len: u64 },

    // Stage 12.5 — peer advertisement schema errors.

    #[error("ContributorPeerAdvertisement.libp2p_peer_id is not a valid base58 libp2p PeerId: {got:?}")]
    PeerAdvertLibp2pPeerIdMalformed { got: String },

    #[error("ContributorPeerAdvertisement.listen_multiaddrs[{index}] is not a valid multiaddr: {got:?}")]
    PeerAdvertMultiaddrMalformed { index: usize, got: String },

    #[error(
        "ContributorPeerAdvertisement.listen_multiaddrs[{index}] /p2p/{multiaddr_peer} does \
         not match libp2p_peer_id {advertised_peer}"
    )]
    PeerAdvertMultiaddrP2pMismatch {
        index: usize,
        multiaddr_peer: String,
        advertised_peer: String,
    },

    #[error("PeerCapabilities.max_handoff_chunk_bytes must be > 0")]
    PeerAdvertChunkCapZero,

    #[error("PeerCapabilities.max_handoff_chunk_bytes {got} exceeds HANDOFF_CHUNK_MAX_BYTES {max}")]
    PeerAdvertChunkCapTooLarge { got: u64, max: u64 },

    #[error("PeerCapabilities.supported_dtypes must be non-empty")]
    PeerAdvertSupportedDtypesEmpty,

    #[error(
        "ContributorPeerAdvertisement.expires_at_utc {expires_at} must be strictly after \
         advertised_at_utc {advertised_at}"
    )]
    PeerAdvertExpiryNotAfterAdvertised {
        advertised_at: String,
        expires_at: String,
    },

    #[error(
        "ContributorPeerAdvertisement.expires_at_utc {expires_at} exceeds \
         advertised_at_utc {advertised_at} + {max_lifetime_secs} seconds (24h freshness cap)"
    )]
    PeerAdvertExpiryTooFar {
        advertised_at: String,
        expires_at: String,
        max_lifetime_secs: i64,
    },

    // Stage 12.11 — assignment supersession schema errors.

    #[error("WorkAssignmentSupersession.superseded_assignment_ids must be non-empty")]
    SupersessionEmptySuperseded,

    /// v1 replacement-only scope. Stage 12.11 forbids abandonment;
    /// a partial-aggregate / cancellation envelope is deferred to a
    /// later stage.
    #[error(
        "WorkAssignmentSupersession.replacement_assignment_ids must be non-empty \
         in v1 (replacement-only scope; abandonment deferred)"
    )]
    SupersessionEmptyReplacement,

    #[error(
        "WorkAssignmentSupersession.{field} duplicate entry: assignment_id={assignment_id}"
    )]
    SupersessionDuplicateId {
        field: &'static str,
        assignment_id: String,
    },

    #[error(
        "WorkAssignmentSupersession.{field} must be sorted ascending by hex value"
    )]
    SupersessionIdsNotSorted { field: &'static str },

    #[error(
        "WorkAssignmentSupersession: assignment_id {assignment_id} appears in BOTH \
         superseded_assignment_ids and replacement_assignment_ids"
    )]
    SupersessionSupersededAndReplacement { assignment_id: String },

    #[error("SupersessionReason::Custom label must be non-empty")]
    SupersessionReasonCustomEmptyLabel,

    #[error("SupersessionReason::Custom label length {got} exceeds bound {max}")]
    SupersessionReasonCustomLabelTooLong { got: usize, max: usize },

    #[error("SupersessionReason::Custom label must be printable ASCII (0x20..=0x7E)")]
    SupersessionReasonCustomLabelNotPrintableAscii,
}

/// Stage 12.7 — typed errors from the contributor state store.
#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("state-dir io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("state-dir json error at {path}: {source}")]
    Json {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// `meta/state_version.json` carries a version this binary
    /// does not support. Forward-compat with future migrations.
    #[error(
        "state-dir version {got} not supported by this binary (expected {expected})"
    )]
    UnsupportedVersion { got: u32, expected: u32 },

    /// Surfaced by the CLI when an operator supplies both
    /// `--contributor-state-dir` AND the legacy
    /// `--peer-advert-dir` / `--joins-dir` flags on the same
    /// `run-assignment --resolve-downstream-peer-from-session`
    /// invocation. Pick one source of truth.
    #[error(
        "ambiguous source of truth: --contributor-state-dir conflicts with \
         {legacy_flag}; supply one or the other"
    )]
    AmbiguousSource { legacy_flag: &'static str },
}

/// Stage 12.8 — typed errors from the local assignment planner.
///
/// The planner is a coordinator-side hint generator. Errors here are
/// operator-facing — they describe why a plan could not be produced
/// from the supplied inputs. None of these errors are network-visible
/// or chain-visible; the `AssignmentPlan` artifact is local-only and
/// unsigned in v1.
#[derive(Debug, thiserror::Error)]
pub enum PlannerError {
    #[error("planner schema_version {got} not supported (expected {expected})")]
    UnsupportedSchemaVersion { got: u32, expected: u32 },

    #[error("model-plan schema_version {got} not supported (expected {expected})")]
    UnsupportedModelPlanVersion { got: u32, expected: u32 },

    /// No `ContributorJoin` survived eligibility filtering (RAM,
    /// dtype-via-advert, live-routing requirement).
    #[error(
        "no eligible contributors after filtering: \
         {joins_total} joined, {filtered_out} filtered \
         (reason: {reason})"
    )]
    NoEligibleContributors {
        joins_total: usize,
        filtered_out: usize,
        reason: String,
    },

    /// Strategy + inputs don't agree on how to shape the work.
    /// E.g. `sequential-layers` with neither a model-plan nor
    /// `--layer-count`, or a model-plan with zero stages.
    #[error("planner inputs are inconsistent: {reason}")]
    InconsistentInputs { reason: String },

    /// A `ModelPlan` stage carries a `WorkKind::Layers { start, end }`
    /// with `start >= end`, a zero `expected_work_units`, or
    /// out-of-order `stage_index` values.
    #[error("model-plan stage {stage_index} invalid: {reason}")]
    InvalidModelPlanStage { stage_index: u32, reason: String },

    /// The planner was asked to split N layers across M contributors
    /// where the equal-split would round down to zero layers for at
    /// least one contributor. Operator should reduce `--max-assignments`
    /// or supply an explicit model-plan.
    #[error(
        "equal layer split would assign zero layers to at least one \
         contributor: layer_count={layer_count}, contributor_count={contributor_count}"
    )]
    EqualSplitProducesEmptyStage {
        layer_count: u32,
        contributor_count: u32,
    },

    /// `--max-assignments` is below the number of assignments the
    /// chosen strategy needs to cover the requested work. Silently
    /// truncating would drop layers / stages, leaving an incomplete
    /// plan; refuse instead. `single-contributor` with a model-plan
    /// requires `stages.len()` assignments; `sequential-layers`
    /// with a model-plan requires `stages.len()`; `round-robin`
    /// requires `stages.len()` (model-plan) or `layer_count`
    /// (`--layer-count` fallback). For `sequential-layers` without
    /// a model-plan, `--max-assignments` is a contributor cap and
    /// does NOT raise this error.
    #[error(
        "--max-assignments={max} is below the {required} assignments the \
         {strategy} strategy needs to cover the requested work; \
         increase the cap or refine the strategy"
    )]
    MaxAssignmentsTooSmall {
        strategy: &'static str,
        required: u32,
        max: u32,
    },

    /// `now_utc >= session.expires_at_utc`. Planner refuses to
    /// emit assignments against expired sessions, and
    /// `assign-session-plan` refuses to publish against one. The
    /// pre-12.7 path (no `--contributor-state-dir`) already refused
    /// expired sessions through the verifier; this variant covers
    /// the state-dir + `--no-prune-state-on-start` case.
    #[error(
        "session is expired: now_utc={now_utc}, expires_at_utc={expires_at_utc}"
    )]
    SessionExpired {
        now_utc: String,
        expires_at_utc: String,
    },
}

/// Stage 12.9 — typed errors from the local session-status reporter.
///
/// `SessionStatusReport` is a local read-only snapshot — it never
/// leaves the operator's machine, is never signed, and is never
/// SNIP-published. Errors here describe why the reporter could not
/// load + re-verify enough state to produce the report at all. Bad
/// individual artifacts (forged joins, tampered partials) do NOT
/// produce errors — they produce report `notes` + `InvalidState`
/// overall status.
#[derive(Debug, thiserror::Error)]
pub enum StatusError {
    #[error("state-dir error: {0}")]
    State(#[from] StateError),

    #[error(
        "status reporter schema_version {got} not supported (expected {expected})"
    )]
    UnsupportedSchemaVersion { got: u32, expected: u32 },
}

/// Stage 12.10 — typed errors from the local pooled-session repair
/// planner / applier.
///
/// `SessionRepairPlan` is a local read-only operator hint — never
/// signed, never SNIP-published, never network-visible. Errors here
/// describe why the planner refused to emit actions, or why the
/// applier refused to publish them.
#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    #[error(
        "repair planner schema_version {got} not supported (expected {expected})"
    )]
    UnsupportedSchemaVersion { got: u32, expected: u32 },

    /// The supplied `SessionStatusReport` describes a session that
    /// isn't on disk in the state-dir.
    #[error("no session: session_id={session_id} is not in the state-dir")]
    SessionNotPresent { session_id: String },

    /// Status is `CompletePartials`, `Aggregated`, or `NoAssignments`
    /// — the session does not need a reannounce-missing repair.
    #[error("nothing to repair: session status is {status:?}")]
    NothingToRepair { status: String },

    /// Status is `InvalidState`. Operators must clean tampered
    /// artifacts before repair planning. Stage 12.10 does NOT
    /// surface an `--allow-invalid-state` flag.
    #[error(
        "session has InvalidState; clean tampered artifacts before \
         repair (see Stage 12.9 status report notes)"
    )]
    InvalidState,

    /// Status is `ExpiredIncomplete`. Reannouncing past-expiry
    /// assignments is mostly noise; operators wanting to repair an
    /// expired session must extend it via `open-session` first.
    #[error(
        "session is expired (ExpiredIncomplete); reannouncing past \
         expiry is rejected"
    )]
    SessionExpired,

    /// Apply-time: the on-disk state's assignment-vs-partial shape
    /// has drifted from the plan's `source_status_hash` projection.
    /// Operator should re-plan from a fresh status report.
    #[error(
        "source status drift: plan was built against a session shape \
         that no longer matches the state-dir (a partial may have \
         arrived); re-plan from a fresh status report"
    )]
    SourceStatusDrift,

    /// Apply-time: the plan's `repair_plan_hash` does not match the
    /// recomputed hash of the loaded bytes. The plan was edited
    /// after creation.
    #[error(
        "repair_plan_hash drift: stored={stored} recomputed={recomputed} \
         — the plan may have been edited after creation"
    )]
    PlanHashDrift { stored: String, recomputed: String },

    /// Apply-time: a referenced assignment_id is not on disk in the
    /// state-dir's `verified/sessions/<id>/assignments/`.
    #[error(
        "assignment not in state-dir: session_id={session_id} \
         assignment_id={assignment_id}"
    )]
    AssignmentNotPresent {
        session_id: String,
        assignment_id: String,
    },

    /// Stage 12.11 apply-time enforcement. A `ReassignAssignment`
    /// action targeted an assignment that, in the current rebuilt
    /// status, is NOT active-missing: it is either already
    /// completed (`partial_present == true`), already superseded
    /// by a prior verified `WorkAssignmentSupersession`, missing
    /// from the status report entirely, or has a `stage_index`
    /// that disagrees with the plan's `original_stage_index`.
    ///
    /// The plan is unsigned and local — a hand-edited plan can
    /// recompute its `repair_plan_hash` so the integrity check
    /// passes, so the applier must independently re-verify that
    /// every targeted assignment is still safe to retire.
    #[error(
        "reassign target not active-missing: session_id={session_id} \
         assignment_id={assignment_id} reason={reason}"
    )]
    ReassignTargetNotActiveMissing {
        session_id: String,
        assignment_id: String,
        /// `not_in_status` / `already_superseded` / `already_completed` /
        /// `stage_index_mismatch`.
        reason: &'static str,
    },
}

/// Canonical-bytes / hash encoding errors.
#[derive(Debug, Error)]
pub enum CanonicalError {
    #[error("bincode serialization failed: {0}")]
    Bincode(String),
}

impl From<bincode1::Error> for CanonicalError {
    fn from(e: bincode1::Error) -> Self {
        CanonicalError::Bincode(e.to_string())
    }
}

/// Ed25519 signing / pubkey-derivation errors.
#[derive(Debug, Error)]
pub enum SigningError {
    #[error("seed file is not exactly 32 bytes: {got} bytes")]
    SeedWrongLength { got: usize },

    #[error("Ed25519 secret-key decode failed: {0}")]
    SecretDecode(String),

    #[error("Ed25519 public-key decode failed: {0}")]
    PublicDecode(String),

    #[error("Ed25519 signature decode failed: {0}")]
    SignatureDecode(String),

    #[error("seed file io error: {0}")]
    Io(#[from] std::io::Error),
}

/// SNIP wrapper errors. All produced by the tempfile-backed
/// `publish_bytes` / `fetch_bytes` helpers.
#[derive(Debug, Error)]
pub enum SnipError {
    #[error("snip adapter error: {0}")]
    Adapter(String),

    #[error("snip integrity check failed for {label}: expected {expected}, got {got}")]
    Integrity {
        label: &'static str,
        expected: String,
        got: String,
    },

    #[error("snip tempfile io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<omni_store::SnipV2Error> for SnipError {
    fn from(e: omni_store::SnipV2Error) -> Self {
        SnipError::Adapter(e.to_string())
    }
}

/// Errors produced by an `InferenceRunner` impl.
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("external runner command failed (exit code {code}): {stderr}")]
    ExternalCommandFailure { code: i32, stderr: String },

    #[error("external runner stdout could not be parsed as the documented JSON envelope: {0}")]
    ExternalEnvelopeMalformed(String),

    #[error("external runner stdout response_b64 decode failed: {0}")]
    ExternalResponseDecode(String),

    #[error("external runner timed out after {0:?}")]
    ExternalTimeout(std::time::Duration),

    #[error("runner io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors produced by the verification pipeline. Each variant maps to
/// one of the checks in [`crate::verify::verify_result`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifyError {
    #[error("schema error during verification: {0}")]
    Schema(#[from] SchemaError),

    #[error("job_hash mismatch: result.job_hash={result_job_hash}, recomputed={recomputed}")]
    JobHashMismatch {
        result_job_hash: String,
        recomputed: String,
    },

    #[error("dispatcher signature did not verify against dispatcher_pubkey_hex")]
    DispatcherSignatureFailed,

    #[error("model_hash mismatch: job={job_model_hash}, result={result_model_hash}")]
    ModelHashMismatch {
        job_model_hash: String,
        result_model_hash: String,
    },

    #[error("input_hash mismatch between job/result/SNIP-fetched bytes; field={field}")]
    InputHashMismatch { field: &'static str },

    #[error("response_hash mismatch between result and SNIP-fetched bytes")]
    ResponseHashMismatch,

    #[error("tokenizer_hash mismatch: job={job_tokenizer_hash}, result={result_tokenizer_hash}")]
    TokenizerHashMismatch {
        job_tokenizer_hash: String,
        result_tokenizer_hash: String,
    },

    #[error(
        "accounting input_token_count mismatch: job declared {job_count}, \
         result measured {result_count}"
    )]
    AccountingInputMismatch { job_count: u64, result_count: u64 },

    #[error(
        "accounting output_token_count {output} exceeds job's max_output_token_count {max}"
    )]
    AccountingOutputOverCap { output: u64, max: u64 },

    #[error(
        "accounting total_base_units inconsistency: {total} != input {input} + output {output}"
    )]
    AccountingTotalInconsistent {
        total: u64,
        input: u64,
        output: u64,
    },

    #[error("contributor signature did not verify against contributor_pubkey_hex")]
    ContributorSignatureFailed,

    #[error(
        "accounting overflow: input_token_count {input} + output_token_count {output} \
         overflows u64"
    )]
    AccountingTotalOverflow { input: u64, output: u64 },

    #[error(
        "verification_requirement {requirement:?} not satisfied by evidence {evidence:?}"
    )]
    RequirementUnsatisfied {
        requirement: String,
        evidence: String,
    },

    #[error("job expired_at_utc {expires_at} is past now() {now}")]
    JobExpired { expires_at: String, now: String },
}

/// Stage 12.1 — errors specific to the discovery / watch surface.
/// Cost-cap refusals are recoverable (the contributor logs and
/// skips); IO / parse / schema errors during discovery typically
/// indicate a malformed posted-job file and are also recoverable.
#[derive(Debug, Error)]
pub enum DiscoverError {
    #[error("posted-job file io error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse posted-job JSON at {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("posted-job schema error at {path}: {source}")]
    Schema {
        path: String,
        #[source]
        source: SchemaError,
    },

    #[error("posted_id mismatch at {path}: posted_id={posted_id}, recomputed={recomputed}")]
    PostedIdMismatch {
        path: String,
        posted_id: String,
        recomputed: String,
    },

    #[error("posted-job posted_at_utc {posted_at} is past expires_at_utc {expires_at}")]
    PostedJobExpired {
        posted_at: String,
        expires_at: String,
    },

    #[error("poster signature did not verify against poster_pubkey_hex (at {path})")]
    PosterSignatureFailed { path: String },

    #[error(
        "cost cap exceeded for {field}: job has {value}, contributor cap is {cap}"
    )]
    CostCapExceeded {
        field: &'static str,
        value: u64,
        cap: u64,
    },

    #[error(
        "filesystem source error: {0}"
    )]
    FilesystemSourceOther(String),

    #[error("network announcement signature did not verify against announcer_pubkey_hex")]
    AnnouncerSignatureFailed,

    #[error(
        "drift between network announcement and SNIP-fetched envelope: field={field}, \
         announcement={announcement}, fetched={fetched}"
    )]
    AnnouncementDrift {
        field: &'static str,
        announcement: String,
        fetched: String,
    },

    #[error("malformed network announcement JSON: {0}")]
    MalformedAnnouncement(String),
}

/// Stage 12.2 — errors specific to the network-relay surface.
/// Distinct from `DiscoverError` so the watch loop can distinguish
/// "the mesh is broken" (RelayError) from "this one announcement
/// was bad" (DiscoverError).
#[derive(Debug, Error)]
pub enum RelayError {
    #[error("relay publish failed: {0}")]
    Publish(String),

    #[error("relay poll failed: {0}")]
    Poll(String),

    #[error("relay serialization failed: {0}")]
    Serialization(String),
}

impl From<serde_json::Error> for RelayError {
    fn from(e: serde_json::Error) -> Self {
        RelayError::Serialization(e.to_string())
    }
}
