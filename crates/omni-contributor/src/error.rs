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

    /// Stage 12.15 — `write_archived_bytes` refused the supplied
    /// `source_relative` path because it contains traversal
    /// (`..`), is absolute, or uses a Windows-style backslash.
    /// Restore callers should bubble this up as
    /// `RestoreError::UnsafeRelativePath`.
    #[error("unsafe relative path: {path}")]
    UnsafeRelativePath { path: String },

    /// Stage 12.15 — `write_archived_bytes` refused the supplied
    /// `source_relative` path because it does not match the
    /// Stage 12.14 archive whitelist (the leading components
    /// must address a `verified/sessions/<session_id>/...`
    /// subtree, a session-keyed `seen/*` marker, or
    /// `results/result-links/<posted_id>.link.json`).
    #[error("disallowed relative path: {path}")]
    DisallowedRelativePath { path: String },

    /// Stage 12.15 — `write_archived_bytes` refused because the
    /// destination already exists and `overwrite_existing == false`.
    /// Surfaced as `RestoreError::DestinationExists` upstream.
    #[error("destination already exists: {path}")]
    DestinationExists { path: std::path::PathBuf },
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

    /// Stage 12.12 apply-time refusal — `InvalidState` is not
    /// triagable via the `--reason invalid-partial` reassign path
    /// because of an additional non-`InvalidPartial` chain failure
    /// (e.g. invalid join, invalid assignment, invalid aggregate,
    /// invalid session, invalid supersession), OR because an
    /// `InvalidPartial` entry exists for an assignment NOT in the
    /// reassignment plan's superseded set.
    ///
    /// `kind` is one of: `"invalid_session"`, `"invalid_join"`,
    /// `"invalid_assignment"`, `"invalid_aggregate"`,
    /// `"invalid_supersession"`, `"invalid_partial_not_in_plan"`.
    /// `context` carries the relevant id (`assignment_id=...`,
    /// `contributor_pubkey_hex=...`, `supersession_id=...`) or is
    /// empty for `invalid_session` / `invalid_aggregate`. Both
    /// fields are stable across `schema_version: 3` for
    /// scripting.
    #[error("InvalidState not triagable via reassign: kind={kind}{context}")]
    InvalidStateNotTriagable {
        kind: &'static str,
        /// Optional id context. When non-empty, MUST start with a
        /// leading space (e.g. `" assignment_id=ff..."`) so the
        /// `Display` impl renders cleanly.
        context: String,
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

/// Stage 12.14 — local session archival errors. Every variant is
/// operator-actionable and carries enough context to triage
/// without re-running the command.
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// The state-dir holds no `verified/sessions/<session_id>/`
    /// subtree. The session was never seen by this watcher, or has
    /// already been pruned/archived/cascaded out.
    #[error("session not present in state-dir: session_id={session_id}")]
    SessionNotPresent { session_id: String },

    /// The rebuilt `SessionStatusReport.overall_status` did not
    /// satisfy the `--require-status` policy. `got` and
    /// `requirement` are the stable Debug-stringified discriminators
    /// (closed sets) so scripts can pattern-match.
    #[error(
        "session overall_status {got} does not satisfy --require-status {requirement}"
    )]
    StatusRequirementUnmet {
        got: String,
        requirement: String,
    },

    /// The archive destination already holds a subtree named
    /// `<session_id>/`. Stage 12.14 refuses to overwrite — operators
    /// must move/rename/delete the existing archive directory before
    /// re-running.
    #[error("archive directory already contains session: {path}")]
    ArchiveAlreadyExists { path: std::path::PathBuf },

    /// BLAKE3 of the destination file did not match the BLAKE3
    /// computed at copy time. Fail-fast (no retry); operator must
    /// triage the FS / hardware before retrying.
    #[error(
        "blake3 mismatch on copied file: path={path} expected={expected} got={got}"
    )]
    BlakeMismatch {
        path: std::path::PathBuf,
        expected: String,
        got: String,
    },

    /// Generic FS error from a `fs::read` / `fs::write` /
    /// `fs::copy` call. The `path` field names the artifact that
    /// failed so operators can `ls` it.
    #[error("archive io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Bubbled-up `StatusError` from
    /// `build_session_status_report`. The archive entry point
    /// builds the status report once (to enforce
    /// `--require-status` + populate the manifest's
    /// `session_overall_status` field).
    #[error("status build for archive: {0}")]
    Status(#[from] StatusError),

    /// Bubbled-up `StateError`.
    #[error("state error: {0}")]
    State(#[from] StateError),
}

/// Stage 12.15 — local session archive restore errors. The
/// inverse of `ArchiveError`: each variant names exactly one
/// safety check the restore path enforces before any state-dir
/// write.
#[derive(Debug, thiserror::Error)]
pub enum RestoreError {
    /// The supplied archive directory does not exist on disk.
    #[error("archive directory not found: {path}")]
    ArchiveNotFound { path: std::path::PathBuf },

    /// The expected `manifest.json` is missing from the archive
    /// directory. Stage 12.14 writes the manifest LAST, so a
    /// missing manifest typically means the archive was created
    /// by a tool that crashed mid-copy.
    #[error("manifest.json missing at {path}")]
    ManifestMissing { path: std::path::PathBuf },

    /// `manifest.json` did not parse as a v1 `ArchiveManifest`.
    /// Bubbled `serde_json::Error` carries the column/line.
    #[error("malformed archive manifest at {path}: {source}")]
    MalformedManifest {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// `manifest.schema_version` is not `ARCHIVE_MANIFEST_SCHEMA_VERSION`.
    /// Stage 12.15 v1 accepts exactly version `1`.
    #[error(
        "unsupported archive manifest schema_version: got={got} expected={expected}"
    )]
    UnsupportedManifestVersion { got: u32, expected: u32 },

    /// The `manifest.session_id` field does not match the
    /// session_id derived from the archive directory name (or
    /// the operator-supplied `--session-id`). Defends against
    /// hand-renamed archive directories.
    #[error(
        "session_id mismatch: manifest.session_id={manifest_session_id} \
         dir.session_id={dir_session_id}"
    )]
    SessionIdMismatch {
        manifest_session_id: String,
        dir_session_id: String,
    },

    /// `manifest.source_state_version != STATE_VERSION`. Stage
    /// 12.15 v1 enforces strict equality — restoring across a
    /// state-dir version boundary would require a migration
    /// story that does not exist yet.
    #[error(
        "incompatible source state-dir version: archive={archive} current={current}"
    )]
    IncompatibleSourceStateVersion { archive: u32, current: u32 },

    /// `source_relative` contains `..`, is absolute, or uses a
    /// backslash. Bubbled from `StateError::UnsafeRelativePath`.
    #[error("unsafe relative path in manifest: {path}")]
    UnsafeRelativePath { path: String },

    /// `source_relative` does not match the Stage 12.14 archive
    /// whitelist. Bubbled from `StateError::DisallowedRelativePath`.
    #[error("disallowed relative path in manifest: {path}")]
    DisallowedRelativePath { path: String },

    /// A file named in the manifest does not exist under the
    /// archive directory. Restore is fail-fast — operator
    /// triages then re-runs.
    #[error("archive file missing for manifest entry: {archive_path}")]
    ManifestFileMissing { archive_path: std::path::PathBuf },

    /// BLAKE3 of the archive file did not match the manifest's
    /// `blake3_hex`. Fail-fast (no retry); operator triages
    /// the FS / archive integrity.
    #[error(
        "blake3 mismatch on archive file: path={path} expected={expected} got={got}"
    )]
    BlakeMismatch {
        path: std::path::PathBuf,
        expected: String,
        got: String,
    },

    /// Preflight check found a destination file that already
    /// exists in the state-dir AND `--overwrite-existing` was
    /// false. Stage 12.15 enforces all-or-nothing: any
    /// pre-existing destination refuses BEFORE any state-dir
    /// write happens. Operator re-runs with
    /// `--overwrite-existing` or cleans state-dir first.
    #[error("destination already exists: {path} (re-run with --overwrite-existing)")]
    DestinationExists { path: std::path::PathBuf },

    /// Generic FS error.
    #[error("restore io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Bubbled `StateError` — typically from
    /// `write_archived_bytes`.
    #[error("state error: {0}")]
    State(#[from] StateError),
}

/// Stage 12.16 — local state-dir integrity scan errors. These are
/// **scanner-aborting** only: per-artifact problems (a tampered
/// session, a stale seen marker, a corrupt archive file) are
/// captured as findings in the
/// [`crate::integrity::StateIntegrityReport`], not as variants
/// here. `IntegrityError` is reserved for failures that prevent
/// the scan from even running (e.g. the state-dir itself
/// can't be walked).
#[derive(Debug, thiserror::Error)]
pub enum IntegrityError {
    /// The Stage 12.7 `ContributorStateStore` returned an error
    /// before the scan could start. Wraps the upstream typed
    /// error (e.g. `StateError::UnsupportedVersion`,
    /// `StateError::Io`) so operators see the underlying cause.
    #[error("state error: {0}")]
    State(#[from] StateError),

    /// `build_session_status_report` failed mid-scan. The
    /// scanner runs status build per session to drive the
    /// Stage 12.13 audit projection; a build failure indicates
    /// the state-dir walked OK but a per-session re-verify
    /// chain hit an internal error.
    #[error("status build during integrity scan: {0}")]
    Status(#[from] StatusError),

    /// Generic FS error encountered during stray-file detection
    /// or the optional `--include-archives` directory walk.
    #[error("integrity scan io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Stage 12.17 — local state-dir cleanup planner / applier
/// errors. The cleanup flow is a closed-set composition of
/// Stage 12.16 findings + Stage 12.13 audit projection +
/// Stage 12.14-shaped quarantine writes; this enum surfaces
/// the failures specific to that flow.
///
/// Per-action failures (one action's apply tripped) are NOT
/// captured here — they propagate as the variant the failing
/// primitive returned (`State`, `Status`, `Integrity`, `Io`).
/// The applier records action-level outcomes in its returned
/// report.
#[derive(Debug, thiserror::Error)]
pub enum CleanupError {
    /// `scan_state_integrity` itself failed (state-dir won't
    /// walk, status build crashed, FS error during stray
    /// detection). Plan-time and apply-time entry both wrap
    /// this so callers see one consistent surface.
    #[error("integrity scan during cleanup: {0}")]
    Integrity(#[from] IntegrityError),

    /// Wraps a status build failure that bubbled up through
    /// audit re-projection at apply time.
    #[error("status build during cleanup: {0}")]
    Status(#[from] StatusError),

    /// State-store primitive (`remove_verified_relative`,
    /// `unmark_seen`, `write_archived_bytes`) refused.
    #[error("state error during cleanup: {0}")]
    State(#[from] StateError),

    /// Apply-time `source_integrity_hash` re-projection
    /// disagreed with the plan's recorded hash. The state-dir
    /// has changed between plan and apply; the operator must
    /// re-plan.
    #[error(
        "source integrity drift: plan expected {expected}, current state \
         hashes to {got}; re-run plan-state-cleanup"
    )]
    SourceIntegrityDrift { expected: String, got: String },

    /// The plan file's `cleanup_plan_hash` doesn't match the
    /// BLAKE3 of the canonical body. The plan was hand-edited
    /// or corrupted after write.
    #[error(
        "plan hash mismatch: stored {stored}, recomputed {recomputed}"
    )]
    PlanHashMismatch {
        stored: String,
        recomputed: String,
    },

    /// A gated action (`QuarantineAndUnmarkPartial` /
    /// `QuarantineAndUnmarkOrphanAssignment`) is present in the
    /// plan but the operator did not pass the corresponding
    /// `--allow-…` flag. Plan-time always emits gated actions
    /// when the finding warrants them; apply-time refuses
    /// unless the gate flag is present.
    #[error(
        "gated action {kind} requires {flag}; pass the flag explicitly to apply"
    )]
    GateRequired { kind: String, flag: String },

    /// The audit projection's orphan-assignment set for a
    /// session changed between plan and apply. Mirrors the
    /// `SourceIntegrityDrift` posture but at finer granularity:
    /// the integrity hash may still match if the change is
    /// confined to non-finding fields, so the apply re-checks
    /// `compute_audit_health` per gated session and refuses
    /// when the orphan id set diverges.
    #[error(
        "orphan-assignment audit drift for session {session_id}: plan listed \
         {plan_count} orphans, current projection lists {current_count}; \
         re-run plan-state-cleanup"
    )]
    OrphanAuditDrift {
        session_id: String,
        plan_count: u32,
        current_count: u32,
    },

    /// A quarantine destination already exists. The applier
    /// refuses to overwrite — operator must clear the
    /// quarantine subtree (or pass a fresh `--quarantine-dir`)
    /// before re-running.
    #[error("quarantine destination already exists: {path}")]
    QuarantineCollision { path: std::path::PathBuf },

    /// Malformed plan JSON (schema violation, unknown fields,
    /// missing required fields).
    #[error("malformed plan at {path}: {source}")]
    MalformedPlan {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// Plan-schema version is not supported by this binary.
    /// Stage 12.17 v1 accepts schema_version == 1 only.
    #[error(
        "unsupported cleanup plan schema_version: got {got}, this binary supports {expected}"
    )]
    UnsupportedPlanVersion { got: u32, expected: u32 },

    /// One of the plan's `path` / `seen_marker_path` strings
    /// violates the per-kind whitelist (`verified/sessions/...`
    /// for tier B and `WriteSeenMarker`; `seen/...` for tier A
    /// stray/remove actions; no `..`, no absolute, no backslash,
    /// no empty segments). A self-consistent
    /// `cleanup_plan_hash` does not vouch for path safety —
    /// hand-edited plans can produce malicious paths whose
    /// hash recomputes correctly. Apply-time refuses BEFORE
    /// any IO when this fires.
    #[error("unsafe path in cleanup plan ({reason}): {path}")]
    UnsafePlanPath {
        path: String,
        reason: &'static str,
    },

    /// Generic FS error encountered while reading the plan,
    /// writing quarantine bytes, or walking the state-dir.
    #[error("cleanup io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Stage 12.18 — local cleanup-quarantine restore errors.
/// Consumes the existing v1 `QuarantineManifest` written by
/// Stage 12.17's `apply_state_cleanup`; no manifest schema bump.
///
/// Per-entry path / BLAKE3 / destination-existence problems are
/// fail-fast: the all-or-nothing preflight runs BEFORE any
/// state-dir write, so a single bad entry refuses the whole
/// restore without partially mutating the state-dir.
#[derive(Debug, thiserror::Error)]
pub enum QuarantineRestoreError {
    /// The supplied quarantine plan directory does not exist
    /// on disk.
    #[error("quarantine plan directory not found: {path}")]
    QuarantineDirNotFound { path: std::path::PathBuf },

    /// The expected `quarantine-manifest.json` is missing
    /// from the supplied plan directory. Stage 12.17 writes
    /// the manifest LAST under the Phase A→B→C ordering, so a
    /// missing manifest typically means the Stage 12.17 apply
    /// crashed between Phase A and Phase B.
    #[error("quarantine-manifest.json missing at {path}")]
    ManifestMissing { path: std::path::PathBuf },

    /// `quarantine-manifest.json` did not parse as a v1
    /// `QuarantineManifest`.
    #[error("malformed quarantine manifest at {path}: {source}")]
    MalformedManifest {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// `manifest.schema_version` is not
    /// `QUARANTINE_MANIFEST_SCHEMA_VERSION`. Stage 12.18 v1
    /// accepts exactly version `1`.
    #[error(
        "unsupported quarantine manifest schema_version: got={got} expected={expected}"
    )]
    UnsupportedManifestVersion { got: u32, expected: u32 },

    /// `manifest.source_state_version != STATE_VERSION`. Same
    /// strict-equality posture as Stage 12.15 archive restore.
    #[error(
        "incompatible source state-dir version: manifest={manifest} current={current}"
    )]
    IncompatibleSourceStateVersion { manifest: u32, current: u32 },

    /// `manifest.plan_id` does not match the
    /// caller-supplied `plan_id` (via `--quarantine-dir +
    /// --plan-id`) OR the directory name supplied via
    /// `--quarantine-plan-dir`. Defends against hand-renamed
    /// quarantine directories.
    #[error(
        "plan_id mismatch: manifest.plan_id={manifest_plan_id} \
         supplied={supplied_plan_id}"
    )]
    PlanIdMismatch {
        manifest_plan_id: String,
        supplied_plan_id: String,
    },

    /// An entry's `source_relative` or `quarantine_relative`
    /// contains `..`, is absolute, uses a backslash, has an
    /// empty segment, or fails the closed-set
    /// `verified/sessions/<64hex>/...` whitelist. Hand-edited
    /// manifests with malicious paths get this BEFORE any IO.
    #[error("unsafe relative path in manifest ({reason}): {path}")]
    UnsafeRelativePath {
        path: String,
        reason: &'static str,
    },

    /// A file named in the manifest does not exist under the
    /// quarantine plan directory. Restore is fail-fast.
    #[error("quarantine file missing for manifest entry: {path}")]
    ManifestFileMissing { path: std::path::PathBuf },

    /// BLAKE3 of the quarantine file did not match the
    /// manifest's `blake3_hex`. Fail-fast (no retry); operator
    /// triages the FS / quarantine integrity.
    #[error(
        "blake3 mismatch on quarantine file: path={path} expected={expected} got={got}"
    )]
    BlakeMismatch {
        path: std::path::PathBuf,
        expected: String,
        got: String,
    },

    /// All-or-nothing preflight found a destination that
    /// already exists in the state-dir AND
    /// `--overwrite-existing` was false. Mirrors Stage 12.15.
    #[error(
        "destination already exists: {path} (re-run with --overwrite-existing)"
    )]
    DestinationExists { path: std::path::PathBuf },

    /// All-or-nothing preflight found a seen-marker
    /// destination that is occupied by a non-file (typically
    /// a directory) so `store.mark_seen` would have failed
    /// mid-restore. Refused BEFORE any body write so a marker
    /// problem cannot partially mutate the state-dir.
    /// `--overwrite-existing` does NOT cover this — marker
    /// preflight is unconditional whenever
    /// `restore_seen_markers == true`.
    #[error("seen marker path blocked at {path}: {reason}")]
    SeenMarkerPathBlocked {
        path: std::path::PathBuf,
        reason: &'static str,
    },

    /// A `source_finding_kind` in the manifest matched a
    /// closed-set tag whose restore requires an opt-in flag
    /// the operator didn't pass (e.g. orphan-assignment
    /// entries without `--allow-restore-orphan-assignments`).
    /// Refused BEFORE any FS interaction.
    #[error("gated restore requires {flag}: source_finding_kind={kind}")]
    GatedRestoreRequired {
        kind: &'static str,
        flag: &'static str,
    },

    /// An entry's `source_finding_kind` is not in the Stage
    /// 12.17 closed set. Hand-edited manifest or future-stage
    /// tag — refused so v1 doesn't silently skip new
    /// variants.
    #[error("unknown source_finding_kind in manifest: {kind}")]
    UnknownFindingKind { kind: String },

    /// Bubbled `StateError` — typically from
    /// `write_archived_bytes` or `mark_seen`.
    #[error("state error: {0}")]
    State(#[from] StateError),

    /// Generic FS error.
    #[error("quarantine restore io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Stage 12.19 — integrity-report diff errors. The differ is a
/// pure JSON-to-JSON comparison; it never opens a state-store
/// or writes state-dir bytes. These variants surface schema /
/// state-version refusals, optional state_dir-pinning refusal,
/// and the v1 "shouldn't happen" finding-metadata-drift guard.
#[derive(Debug, thiserror::Error)]
pub enum IntegrityDiffError {
    /// The baseline report's `schema_version` is not
    /// `STATE_INTEGRITY_REPORT_SCHEMA_VERSION`. Stage 12.19 v1
    /// accepts exactly v1.
    #[error(
        "unsupported baseline schema_version: got={got} expected={expected}"
    )]
    UnsupportedBaselineSchemaVersion { got: u32, expected: u32 },

    /// Same as above but for the `current` report.
    #[error(
        "unsupported current schema_version: got={got} expected={expected}"
    )]
    UnsupportedCurrentSchemaVersion { got: u32, expected: u32 },

    /// `baseline.state_version != current.state_version`.
    /// Cross-state-dir-version diff isn't meaningful without a
    /// migration story.
    #[error(
        "incompatible state-dir version: baseline={baseline} current={current}"
    )]
    IncompatibleStateVersion { baseline: u32, current: u32 },

    /// `baseline.state_dir != current.state_dir` AND
    /// `--require-state-dir-match` was set. CI baselines are
    /// commonly captured on a different host so this is OFF by
    /// default; operators who want host pinning opt in.
    #[error(
        "state_dir mismatch: baseline={baseline} current={current} \
         (re-run without --require-state-dir-match to ignore)"
    )]
    StateDirMismatch { baseline: String, current: String },

    /// Two findings share the same identity tuple
    /// `(kind, session_id, path, reason_tag)` but disagree on
    /// `severity` or `recommended_action`. v1 treats this as a
    /// structural inconsistency rather than silently collapsing
    /// it; the closed-set scanner deterministically maps each
    /// identity to a fixed (severity, action) pair, so a drift
    /// here means one of the reports was tampered with or
    /// produced by a non-Stage-12.16 tool.
    #[error(
        "finding metadata drift for identity={identity}: \
         severity baseline={baseline_severity} current={current_severity}; \
         action baseline={baseline_recommended_action} \
         current={current_recommended_action}"
    )]
    FindingMetadataDrift {
        identity: String,
        baseline_severity: String,
        current_severity: String,
        baseline_recommended_action: String,
        current_recommended_action: String,
    },

    /// Baseline JSON failed to parse as a v1
    /// `StateIntegrityReport`. Bubbled `serde_json::Error`
    /// carries column/line.
    #[error("malformed baseline at {path}: {source}")]
    MalformedBaseline {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// `current` JSON (typically supplied via
    /// `state-integrity-diff --current <path>`) failed to parse
    /// as a v1 `StateIntegrityReport`.
    #[error("malformed current at {path}: {source}")]
    MalformedCurrent {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// Generic FS error encountered while reading a report
    /// JSON.
    #[error("integrity diff io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Stage 12.20 — signed integrity-baseline errors. The signed
/// baseline is a local-only wrapper around a v1
/// `StateIntegrityReport`; this enum covers signing, verifying,
/// and consuming a wrapper. Wire-protocol surfaces are NOT
/// affected — Stage 12.20 introduces no new envelope.
#[derive(Debug, thiserror::Error)]
pub enum SignedBaselineError {
    /// Wrapper's `schema_version` is not
    /// `SIGNED_BASELINE_SCHEMA_VERSION = 1`. v1 binary refuses
    /// future-stage wrappers.
    #[error(
        "unsupported signed-baseline schema_version: got={got} expected={expected}"
    )]
    UnsupportedSchemaVersion { got: u32, expected: u32 },

    /// The wrapper's embedded `report.schema_version` is not
    /// `STATE_INTEGRITY_REPORT_SCHEMA_VERSION = 1`. The
    /// wrapper might be v1 but the report it carries is from
    /// a future Stage 12.16 lineage; refuse to deserialize.
    #[error(
        "unsupported embedded report schema_version: got={got} expected={expected}"
    )]
    UnsupportedReportSchemaVersion { got: u32, expected: u32 },

    /// Bincode encoding of the canonical body failed. Bubbled
    /// from `CanonicalError`. Should be impossible in practice
    /// because the canonical body is closed-set.
    #[error("canonical encoding error: {0}")]
    Canonical(#[from] CanonicalError),

    /// Hex parse or Ed25519 signing/verification primitive
    /// returned an error. Bubbled from `SigningError`.
    #[error("signing error: {0}")]
    Signing(#[from] SigningError),

    /// Ed25519 verification of the wrapper's `signature_hex`
    /// against the canonical body and the wrapper's
    /// `signer_pubkey_hex` returned `Ok(false)` — the
    /// signature does not match. Refused after the
    /// `SignerPubkeyMismatch` cheap pre-check.
    #[error(
        "signature mismatch: wrapper signature does not verify against embedded pubkey"
    )]
    SignatureMismatch,

    /// The `expected_signer_pubkey_hex` the caller passed to
    /// `verify_signed_state_integrity_baseline` does NOT
    /// equal the wrapper's `signer_pubkey_hex`. The signature
    /// might be valid for some other key, but it's not the key
    /// the verifier was told to trust. Cheap pre-check — runs
    /// BEFORE crypto verification so a malicious wrapper with a
    /// forged pubkey can't burn cycles.
    #[error(
        "signer pubkey mismatch: expected={expected} got={got} \
         (the wrapper was signed by a different key than the trust anchor)"
    )]
    SignerPubkeyMismatch { expected: String, got: String },

    /// Generic FS error reading the wrapper JSON.
    #[error("signed-baseline io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Wrapper JSON did not parse as a v1
    /// `SignedStateIntegrityBaseline`.
    #[error("malformed signed-baseline at {path}: {source}")]
    MalformedJson {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

/// Stage 12.21 — local-only signed integrity-diff errors.
/// Consumes the existing Stage 12.19 `StateIntegrityDiffReport`
/// unchanged and signs a new wrapper around it. Mirrors the
/// Stage 12.20 `SignedBaselineError` shape verbatim — the same
/// two-step refusal posture (cheap `SignerPubkeyMismatch`
/// pre-check, then crypto verify) applies.
#[derive(Debug, thiserror::Error)]
pub enum SignedIntegrityDiffError {
    /// Wrapper's `schema_version` is not
    /// `SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION = 1`. v1 binary
    /// refuses future-stage wrappers.
    #[error(
        "unsupported signed-diff schema_version: got={got} expected={expected}"
    )]
    UnsupportedSchemaVersion { got: u32, expected: u32 },

    /// The wrapper's embedded `diff.schema_version` is not
    /// `STATE_INTEGRITY_DIFF_SCHEMA_VERSION = 1`. The wrapper
    /// might be v1 but the diff it carries is from a future
    /// Stage 12.19 lineage; refuse to deserialize.
    #[error(
        "unsupported embedded diff schema_version: got={got} expected={expected}"
    )]
    UnsupportedDiffSchemaVersion { got: u32, expected: u32 },

    /// Bincode encoding of the canonical body failed. Bubbled
    /// from `CanonicalError`. Should be impossible in practice
    /// because the canonical body is closed-set.
    #[error("canonical encoding error: {0}")]
    Canonical(#[from] CanonicalError),

    /// Hex parse or Ed25519 signing/verification primitive
    /// returned an error. Bubbled from `SigningError`.
    #[error("signing error: {0}")]
    Signing(#[from] SigningError),

    /// Ed25519 verification of the wrapper's `signature_hex`
    /// against the canonical body and the wrapper's
    /// `signer_pubkey_hex` returned `Ok(false)` — the
    /// signature does not match. Refused after the
    /// `SignerPubkeyMismatch` cheap pre-check.
    #[error(
        "signature mismatch: wrapper signature does not verify against embedded pubkey"
    )]
    SignatureMismatch,

    /// The `expected_signer_pubkey_hex` the caller passed to
    /// `verify_signed_state_integrity_diff` does NOT equal
    /// the wrapper's `signer_pubkey_hex`. Cheap pre-check —
    /// runs BEFORE crypto verification so a malicious wrapper
    /// with a forged pubkey can't burn cycles.
    #[error(
        "signer pubkey mismatch: expected={expected} got={got} \
         (the wrapper was signed by a different key than the trust anchor)"
    )]
    SignerPubkeyMismatch { expected: String, got: String },

    /// Generic FS error reading the wrapper JSON.
    #[error("signed-diff io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Wrapper JSON did not parse as a v1
    /// `SignedStateIntegrityDiff`.
    #[error("malformed signed-diff at {path}: {source}")]
    MalformedJson {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },
}
