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
