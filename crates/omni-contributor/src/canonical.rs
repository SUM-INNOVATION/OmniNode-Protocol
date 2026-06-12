//! Stage 12.0 — canonical bytes / hash / signing-input encoding for
//! `ContributorJob` and `ContributorResult`.
//!
//! Reuses the Stage 6 `chain_wire` design pattern (bincode 1.3 via the
//! `bincode1` crate alias) so the codebase has one canonical-encoding
//! philosophy. Domain separators are **distinct** from any chain-wire
//! domain tag — these are off-chain identifiers and do not collide.
//!
//! ## Frozen layout
//!
//! Field declaration order on `JobCanonicalBody` / `ResultCanonicalBody`
//! is the bincode wire order; frozen for `schema_version: 1`. Any
//! reorder is a `schema_version: 2` migration.
//!
//! ## job_hash construction
//!
//! ```text
//! body  = JobCanonicalBody { schema_version, model_hash,
//!                            manifest_snip_root, input_snip_root,
//!                            input_hash, verification_requirement,
//!                            accounting, dispatched_at_utc,
//!                            expires_at_utc, dispatcher_pubkey_hex,
//!                            notes }
//!         // EXCLUDES job_id and dispatcher_signature_hex.
//! bytes = JOB_DOMAIN || bincode1::serialize(&body)
//! hash  = BLAKE3(bytes)                 // [u8; 32]
//! hex   = lowercase_hex(hash)           // 64-char string (no 0x prefix)
//! ```
//!
//! `job_id` is exactly this `hex` value; the validator asserts equality.
//!
//! ## Dispatcher signing input
//!
//! `dispatcher_signing_input` is the same byte sequence
//! (`JOB_DOMAIN || bincode_body`) — the dispatcher signs the body, not
//! the BLAKE3 of it. (This matches Stage 6's `signing_input_bytes`
//! convention: the signer covers the canonical wire bytes directly,
//! letting Ed25519 do its own hashing.)
//!
//! ## Result signing input
//!
//! ```text
//! body  = ResultCanonicalBody { every field except contributor_signature_hex }
//! bytes = RESULT_DOMAIN || bincode1::serialize(&body)
//! ```
//!
//! The contributor signs this byte sequence; the verifier reconstructs
//! it and Ed25519-verifies against `contributor_pubkey_hex`.

use serde::Serialize;

use crate::error::CanonicalError;
use crate::job::{BaseUnitRewardPolicy, ContributorJob, JobAccounting, VerificationRequirement};
use crate::result::{
    ContributorResult, Evidence, MeasuredAccounting, StageContribution, WorkUnitKind,
};

/// Domain separator for the canonical job-body byte sequence. 28
/// ASCII bytes; no null terminator, no length prefix.
pub const JOB_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-JOB:v1:";

/// Domain separator for the canonical result-body byte sequence. 31
/// ASCII bytes.
pub const RESULT_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-RESULT:v1:";

// ── Job canonical body ────────────────────────────────────────────────────

/// Frozen-layout view of a `ContributorJob` for canonical encoding.
/// Excludes `job_id` (derived from `job_hash`) and
/// `dispatcher_signature_hex` (the signer can't include its own
/// signature in its signing input).
///
/// Field order is the bincode wire order; do NOT reorder without a
/// `schema_version: 2` migration.
#[derive(Debug, Serialize)]
struct JobCanonicalBody<'a> {
    schema_version: u32,
    model_hash: &'a str,
    manifest_snip_root: &'a str,
    input_snip_root: &'a str,
    input_hash: &'a str,
    verification_requirement: &'a VerificationRequirement,
    accounting: AccountingCanonical<'a>,
    dispatched_at_utc: &'a str,
    expires_at_utc: Option<&'a str>,
    dispatcher_pubkey_hex: Option<&'a str>,
    notes: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct AccountingCanonical<'a> {
    tokenizer_hash: &'a str,
    tokenizer_id: &'a str,
    input_token_count: u64,
    max_output_token_count: u64,
    base_unit_reward_policy: &'a BaseUnitRewardPolicy,
}

impl<'a> From<&'a JobAccounting> for AccountingCanonical<'a> {
    fn from(a: &'a JobAccounting) -> Self {
        Self {
            tokenizer_hash: &a.tokenizer_hash,
            tokenizer_id: &a.tokenizer_id,
            input_token_count: a.input_token_count,
            max_output_token_count: a.max_output_token_count,
            base_unit_reward_policy: &a.base_unit_reward_policy,
        }
    }
}

impl<'a> From<&'a ContributorJob> for JobCanonicalBody<'a> {
    fn from(j: &'a ContributorJob) -> Self {
        Self {
            schema_version: j.schema_version,
            model_hash: &j.model_hash,
            manifest_snip_root: &j.manifest_snip_root,
            input_snip_root: &j.input_snip_root,
            input_hash: &j.input_hash,
            verification_requirement: &j.verification_requirement,
            accounting: (&j.accounting).into(),
            dispatched_at_utc: &j.dispatched_at_utc,
            expires_at_utc: j.expires_at_utc.as_deref(),
            dispatcher_pubkey_hex: j.dispatcher_pubkey_hex.as_deref(),
            notes: j.notes.as_deref(),
        }
    }
}

/// `JOB_DOMAIN || bincode1::serialize(&canonical_body)`.
pub fn canonical_job_bytes(job: &ContributorJob) -> Result<Vec<u8>, CanonicalError> {
    let body: JobCanonicalBody = job.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(JOB_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(JOB_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// `BLAKE3(canonical_job_bytes(job))` as 32 raw bytes.
pub fn job_hash_bytes(job: &ContributorJob) -> Result<[u8; 32], CanonicalError> {
    let bytes = canonical_job_bytes(job)?;
    let hash = blake3::hash(&bytes);
    Ok(*hash.as_bytes())
}

/// `BLAKE3(canonical_job_bytes(job))` as 64-char lowercase hex (no
/// `0x` prefix). This is the value stored in `ContributorJob.job_id`
/// (and equivalently `ContributorResult.job_hash`).
pub fn job_hash_hex(job: &ContributorJob) -> Result<String, CanonicalError> {
    Ok(hex_lower(&job_hash_bytes(job)?))
}

/// Byte sequence the dispatcher signs over (same bytes as
/// `canonical_job_bytes`).
pub fn dispatcher_signing_input(job: &ContributorJob) -> Result<Vec<u8>, CanonicalError> {
    canonical_job_bytes(job)
}

// ── Result canonical body ─────────────────────────────────────────────────

/// Frozen-layout view of a `ContributorResult` for canonical encoding.
/// Excludes `contributor_signature_hex`.
#[derive(Debug, Serialize)]
struct ResultCanonicalBody<'a> {
    schema_version: u32,
    job_id: &'a str,
    job_hash: &'a str,
    job_snip_root: Option<&'a str>,
    model_hash: &'a str,
    input_hash: &'a str,
    response_snip_root: &'a str,
    response_hash: &'a str,
    evidence: &'a Evidence,
    measured_accounting: MeasuredAccountingCanonical<'a>,
    produced_at_utc: &'a str,
    contributor_pubkey_hex: &'a str,
    notes: Option<&'a str>,
}

#[derive(Debug, Serialize)]
struct MeasuredAccountingCanonical<'a> {
    tokenizer_hash: &'a str,
    input_token_count: u64,
    output_token_count: u64,
    total_base_units: u64,
    stage_contributions: Vec<StageContributionCanonical<'a>>,
}

#[derive(Debug, Serialize)]
struct StageContributionCanonical<'a> {
    contributor_pubkey_hex: &'a str,
    stage_label: &'a str,
    work_unit_kind: &'a WorkUnitKind,
    work_units: u64,
}

impl<'a> From<&'a StageContribution> for StageContributionCanonical<'a> {
    fn from(s: &'a StageContribution) -> Self {
        Self {
            contributor_pubkey_hex: &s.contributor_pubkey_hex,
            stage_label: &s.stage_label,
            work_unit_kind: &s.work_unit_kind,
            work_units: s.work_units,
        }
    }
}

impl<'a> From<&'a MeasuredAccounting> for MeasuredAccountingCanonical<'a> {
    fn from(m: &'a MeasuredAccounting) -> Self {
        Self {
            tokenizer_hash: &m.tokenizer_hash,
            input_token_count: m.input_token_count,
            output_token_count: m.output_token_count,
            total_base_units: m.total_base_units,
            stage_contributions: m.stage_contributions.iter().map(Into::into).collect(),
        }
    }
}

impl<'a> From<&'a ContributorResult> for ResultCanonicalBody<'a> {
    fn from(r: &'a ContributorResult) -> Self {
        Self {
            schema_version: r.schema_version,
            job_id: &r.job_id,
            job_hash: &r.job_hash,
            job_snip_root: r.job_snip_root.as_deref(),
            model_hash: &r.model_hash,
            input_hash: &r.input_hash,
            response_snip_root: &r.response_snip_root,
            response_hash: &r.response_hash,
            evidence: &r.evidence,
            measured_accounting: (&r.measured_accounting).into(),
            produced_at_utc: &r.produced_at_utc,
            contributor_pubkey_hex: &r.contributor_pubkey_hex,
            notes: r.notes.as_deref(),
        }
    }
}

/// `RESULT_DOMAIN || bincode1::serialize(&canonical_body)`.
pub fn canonical_result_bytes(result: &ContributorResult) -> Result<Vec<u8>, CanonicalError> {
    let body: ResultCanonicalBody = result.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(RESULT_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(RESULT_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Byte sequence the contributor signs over (same bytes as
/// `canonical_result_bytes`).
pub fn contributor_signing_input(
    result: &ContributorResult,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_result_bytes(result)
}

// ── Stage 12.1 — posted-envelope canonical bytes ───────────────────────────
//
// `PostedJob` and `PostedResultLink` get their own domain separators
// distinct from Stage 12.0's JOB_DOMAIN / RESULT_DOMAIN and distinct
// from any chain-wire tag. Bincode 1.3 wire layout; field order frozen
// for schema_version: 1.

use crate::posted::{PostedJob, PostedResultLink};

/// Domain separator for the canonical PostedJob-body byte sequence
/// (35 ASCII bytes; no null terminator, no length prefix).
pub const POSTED_JOB_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-POSTED-JOB:v1:";

/// Domain separator for the canonical PostedResultLink-body byte
/// sequence (43 ASCII bytes).
pub const POSTED_RESULT_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-POSTED-RESULT-LINK:v1:";

/// Frozen-layout view of a `PostedJob` for canonical encoding.
/// Excludes `posted_id` (derived from this hash) and
/// `poster_signature_hex` (signer can't include its own signature).
#[derive(Debug, Serialize)]
struct PostedJobCanonicalBody<'a> {
    schema_version: u32,
    job_snip_root: &'a str,
    job_hash: &'a str,
    model_hash: &'a str,
    posted_at_utc: &'a str,
    expires_at_utc: Option<&'a str>,
    poster_pubkey_hex: Option<&'a str>,
    notes: Option<&'a str>,
}

impl<'a> From<&'a PostedJob> for PostedJobCanonicalBody<'a> {
    fn from(p: &'a PostedJob) -> Self {
        Self {
            schema_version: p.schema_version,
            job_snip_root: &p.job_snip_root,
            job_hash: &p.job_hash,
            model_hash: &p.model_hash,
            posted_at_utc: &p.posted_at_utc,
            expires_at_utc: p.expires_at_utc.as_deref(),
            poster_pubkey_hex: p.poster_pubkey_hex.as_deref(),
            notes: p.notes.as_deref(),
        }
    }
}

/// `POSTED_JOB_DOMAIN || bincode1::serialize(&canonical_body)`.
pub fn canonical_posted_job_bytes(p: &PostedJob) -> Result<Vec<u8>, CanonicalError> {
    let body: PostedJobCanonicalBody = p.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(POSTED_JOB_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(POSTED_JOB_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// 32-byte BLAKE3 of `canonical_posted_job_bytes`.
pub fn posted_job_hash_bytes(p: &PostedJob) -> Result<[u8; 32], CanonicalError> {
    let bytes = canonical_posted_job_bytes(p)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

/// 64-char lowercase hex of `posted_job_hash_bytes`. Value stored in
/// `PostedJob.posted_id`.
pub fn posted_id_hex(p: &PostedJob) -> Result<String, CanonicalError> {
    Ok(hex_lower(&posted_job_hash_bytes(p)?))
}

/// Bytes the poster signs over (same as `canonical_posted_job_bytes`).
pub fn poster_signing_input(p: &PostedJob) -> Result<Vec<u8>, CanonicalError> {
    canonical_posted_job_bytes(p)
}

/// Frozen-layout view of a `PostedResultLink` for canonical encoding.
/// Excludes `contributor_signature_hex`.
#[derive(Debug, Serialize)]
struct PostedResultLinkCanonicalBody<'a> {
    schema_version: u32,
    posted_id: &'a str,
    result_snip_root: &'a str,
    result_canonical_hash: &'a str,
    contributor_pubkey_hex: &'a str,
    published_at_utc: &'a str,
}

impl<'a> From<&'a PostedResultLink> for PostedResultLinkCanonicalBody<'a> {
    fn from(r: &'a PostedResultLink) -> Self {
        Self {
            schema_version: r.schema_version,
            posted_id: &r.posted_id,
            result_snip_root: &r.result_snip_root,
            result_canonical_hash: &r.result_canonical_hash,
            contributor_pubkey_hex: &r.contributor_pubkey_hex,
            published_at_utc: &r.published_at_utc,
        }
    }
}

/// `POSTED_RESULT_DOMAIN || bincode1::serialize(&canonical_body)`.
pub fn canonical_posted_result_link_bytes(
    r: &PostedResultLink,
) -> Result<Vec<u8>, CanonicalError> {
    let body: PostedResultLinkCanonicalBody = r.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(POSTED_RESULT_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(POSTED_RESULT_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Bytes the contributor signs over (same as
/// `canonical_posted_result_link_bytes`).
pub fn posted_result_link_signing_input(
    r: &PostedResultLink,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_posted_result_link_bytes(r)
}

// ── Stage 12.2 — network announcement canonical bytes ─────────────────────
//
// `NetworkPostedJobAnnouncement` and `NetworkPostedResultAnnouncement`
// each get their own domain separator, distinct from the 12.0/12.1
// separators and from any chain-wire tag. Bincode 1.3 wire layout;
// field order frozen for `schema_version: 1`.

use crate::net::{NetworkPostedJobAnnouncement, NetworkPostedResultAnnouncement};

/// Domain separator for the canonical network-posted-job-announcement
/// byte sequence (32 ASCII bytes).
pub const NET_JOB_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-NET-JOB:v1:";

/// Domain separator for the canonical network-posted-result-announcement
/// byte sequence (35 ASCII bytes).
pub const NET_RESULT_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-NET-RESULT:v1:";

/// Frozen-layout view of a `NetworkPostedJobAnnouncement` for
/// canonical encoding. Excludes `announcer_signature_hex` (the
/// signer can't include its own signature in its signing input);
/// **includes** `announcer_pubkey_hex` so the signature binds to
/// the claimed pubkey.
#[derive(Debug, Serialize)]
struct NetworkJobCanonicalBody<'a> {
    schema_version: u32,
    posted_job_snip_root: &'a str,
    posted_id: &'a str,
    job_hash: &'a str,
    model_hash: &'a str,
    tokenizer_hash: Option<&'a str>,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a NetworkPostedJobAnnouncement> for NetworkJobCanonicalBody<'a> {
    fn from(a: &'a NetworkPostedJobAnnouncement) -> Self {
        Self {
            schema_version: a.schema_version,
            posted_job_snip_root: &a.posted_job_snip_root,
            posted_id: &a.posted_id,
            job_hash: &a.job_hash,
            model_hash: &a.model_hash,
            tokenizer_hash: a.tokenizer_hash.as_deref(),
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

/// `NET_JOB_DOMAIN || bincode1::serialize(&canonical_body)`.
pub fn canonical_network_job_announcement_bytes(
    a: &NetworkPostedJobAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetworkJobCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_JOB_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_JOB_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Bytes the announcer signs over (same as
/// `canonical_network_job_announcement_bytes`).
pub fn network_job_announcement_signing_input(
    a: &NetworkPostedJobAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_network_job_announcement_bytes(a)
}

/// Frozen-layout view of a `NetworkPostedResultAnnouncement` for
/// canonical encoding. Excludes `announcer_signature_hex`.
#[derive(Debug, Serialize)]
struct NetworkResultCanonicalBody<'a> {
    schema_version: u32,
    posted_id: &'a str,
    posted_result_link_snip_root: &'a str,
    result_canonical_hash: &'a str,
    contributor_pubkey_hex: &'a str,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a NetworkPostedResultAnnouncement> for NetworkResultCanonicalBody<'a> {
    fn from(a: &'a NetworkPostedResultAnnouncement) -> Self {
        Self {
            schema_version: a.schema_version,
            posted_id: &a.posted_id,
            posted_result_link_snip_root: &a.posted_result_link_snip_root,
            result_canonical_hash: &a.result_canonical_hash,
            contributor_pubkey_hex: &a.contributor_pubkey_hex,
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

/// `NET_RESULT_DOMAIN || bincode1::serialize(&canonical_body)`.
pub fn canonical_network_result_announcement_bytes(
    a: &NetworkPostedResultAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetworkResultCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_RESULT_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_RESULT_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Bytes the announcer signs over (same as
/// `canonical_network_result_announcement_bytes`).
pub fn network_result_announcement_signing_input(
    a: &NetworkPostedResultAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_network_result_announcement_bytes(a)
}

// ── Stage 12.3 — session canonical bytes ──────────────────────────────────
//
// Five new domain separators, one per inner envelope. Frozen layout for
// `schema_version: 1`. Signer's own signature excluded from the
// canonical body; `session_id` / `assignment_id` self-hash fields also
// excluded (they're derived from the canonical bytes here).

use crate::session::{
    AggregatedContributorResult, AggregatedPartialRef, ContributorJoin, ExecutionSession,
    PartialContributorResult, WorkAssignment, WorkKind,
};

/// 32 ASCII bytes.
pub const SESSION_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-SESSION:v1:";

/// 29 ASCII bytes.
pub const JOIN_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-JOIN:v1:";

/// 35 ASCII bytes.
pub const ASSIGNMENT_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-ASSIGNMENT:v1:";

/// 32 ASCII bytes.
pub const PARTIAL_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-PARTIAL:v1:";

/// 35 ASCII bytes.
pub const AGGREGATED_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-AGGREGATED:v1:";

/// Stage 12.20 — domain separator for the local-only signed
/// integrity-baseline wrapper. Mirrors the existing wire-envelope
/// convention but is consumed only by `signed_baseline.rs` —
/// no protocol surface, no gossipsub topic, no SNIP wire.
pub const SIGNED_BASELINE_DOMAIN: &[u8] =
    b"OMNINODE-CONTRIBUTOR-SIGNED-INTEGRITY-BASELINE:v1:";

/// Stage 12.21 — domain separator for the local-only signed
/// integrity-diff wrapper. Same posture as
/// `SIGNED_BASELINE_DOMAIN`: consumed only by `signed_diff.rs`;
/// no protocol surface, no gossipsub topic, no SNIP wire.
pub const SIGNED_INTEGRITY_DIFF_DOMAIN: &[u8] =
    b"OMNINODE-CONTRIBUTOR-SIGNED-INTEGRITY-DIFF:v1:";

// --- ExecutionSession ---

#[derive(Debug, Serialize)]
struct ExecutionSessionCanonicalBody<'a> {
    schema_version: u32,
    posted_id: &'a str,
    job_hash: &'a str,
    model_hash: &'a str,
    tokenizer_hash: Option<&'a str>,
    coordinator_pubkey_hex: &'a str,
    created_at_utc: &'a str,
    expires_at_utc: &'a str,
}

impl<'a> From<&'a ExecutionSession> for ExecutionSessionCanonicalBody<'a> {
    fn from(s: &'a ExecutionSession) -> Self {
        Self {
            schema_version: s.schema_version,
            posted_id: &s.posted_id,
            job_hash: &s.job_hash,
            model_hash: &s.model_hash,
            tokenizer_hash: s.tokenizer_hash.as_deref(),
            coordinator_pubkey_hex: &s.coordinator_pubkey_hex,
            created_at_utc: &s.created_at_utc,
            expires_at_utc: &s.expires_at_utc,
        }
    }
}

pub fn canonical_execution_session_bytes(
    s: &ExecutionSession,
) -> Result<Vec<u8>, CanonicalError> {
    let body: ExecutionSessionCanonicalBody = s.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(SESSION_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(SESSION_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// 32-byte BLAKE3 over `canonical_execution_session_bytes`.
pub fn execution_session_hash_bytes(
    s: &ExecutionSession,
) -> Result<[u8; 32], CanonicalError> {
    let bytes = canonical_execution_session_bytes(s)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

/// 64-char lowercase hex. Value stored in `ExecutionSession.session_id`.
pub fn session_id_hex(s: &ExecutionSession) -> Result<String, CanonicalError> {
    Ok(hex_lower(&execution_session_hash_bytes(s)?))
}

pub fn execution_session_signing_input(
    s: &ExecutionSession,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_execution_session_bytes(s)
}

// --- ContributorJoin ---

#[derive(Debug, Serialize)]
struct ContributorJoinCanonicalBody<'a> {
    schema_version: u32,
    session_id: &'a str,
    contributor_pubkey_hex: &'a str,
    available_ram_bytes: u64,
    max_input_tokens: u64,
    max_output_tokens: u64,
    supported_work_unit_kinds: &'a [crate::result::WorkUnitKind],
    runner_kind: &'a str,
    joined_at_utc: &'a str,
}

impl<'a> From<&'a ContributorJoin> for ContributorJoinCanonicalBody<'a> {
    fn from(j: &'a ContributorJoin) -> Self {
        Self {
            schema_version: j.schema_version,
            session_id: &j.session_id,
            contributor_pubkey_hex: &j.contributor_pubkey_hex,
            available_ram_bytes: j.available_ram_bytes,
            max_input_tokens: j.max_input_tokens,
            max_output_tokens: j.max_output_tokens,
            supported_work_unit_kinds: &j.supported_work_unit_kinds,
            runner_kind: &j.runner_kind,
            joined_at_utc: &j.joined_at_utc,
        }
    }
}

pub fn canonical_contributor_join_bytes(
    j: &ContributorJoin,
) -> Result<Vec<u8>, CanonicalError> {
    let body: ContributorJoinCanonicalBody = j.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(JOIN_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(JOIN_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn contributor_join_signing_input(
    j: &ContributorJoin,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_contributor_join_bytes(j)
}

// --- WorkAssignment ---

#[derive(Debug, Serialize)]
struct WorkAssignmentCanonicalBody<'a> {
    schema_version: u32,
    session_id: &'a str,
    stage_index: u32,
    contributor_pubkey_hex: &'a str,
    work_kind: &'a WorkKind,
    expected_work_units: u64,
    expected_work_unit_kind: &'a crate::result::WorkUnitKind,
    assigned_at_utc: &'a str,
}

impl<'a> From<&'a WorkAssignment> for WorkAssignmentCanonicalBody<'a> {
    fn from(a: &'a WorkAssignment) -> Self {
        Self {
            schema_version: a.schema_version,
            session_id: &a.session_id,
            stage_index: a.stage_index,
            contributor_pubkey_hex: &a.contributor_pubkey_hex,
            work_kind: &a.work_kind,
            expected_work_units: a.expected_work_units,
            expected_work_unit_kind: &a.expected_work_unit_kind,
            assigned_at_utc: &a.assigned_at_utc,
        }
    }
}

pub fn canonical_work_assignment_bytes(
    a: &WorkAssignment,
) -> Result<Vec<u8>, CanonicalError> {
    let body: WorkAssignmentCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(ASSIGNMENT_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(ASSIGNMENT_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn work_assignment_hash_bytes(
    a: &WorkAssignment,
) -> Result<[u8; 32], CanonicalError> {
    let bytes = canonical_work_assignment_bytes(a)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

/// 64-char lowercase hex. Value stored in `WorkAssignment.assignment_id`.
pub fn assignment_id_hex(a: &WorkAssignment) -> Result<String, CanonicalError> {
    Ok(hex_lower(&work_assignment_hash_bytes(a)?))
}

pub fn work_assignment_signing_input(
    a: &WorkAssignment,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_work_assignment_bytes(a)
}

// --- PartialContributorResult ---

#[derive(Debug, Serialize)]
struct PartialResultCanonicalBody<'a> {
    schema_version: u32,
    session_id: &'a str,
    assignment_id: &'a str,
    contributor_pubkey_hex: &'a str,
    partial_artifact_snip_root: &'a str,
    partial_artifact_hash: &'a str,
    measured_accounting: PartialMeasuredAccountingCanonical<'a>,
    produced_at_utc: &'a str,
}

/// Partial-domain view of `MeasuredAccounting`. Distinct from the
/// 12.0 `MeasuredAccountingCanonical` (which lives under RESULT_DOMAIN
/// and serializes `StageContributionCanonical`) — they're independent
/// canonical bodies for independent domain separators.
#[derive(Debug, Serialize)]
struct PartialMeasuredAccountingCanonical<'a> {
    tokenizer_hash: &'a str,
    input_token_count: u64,
    output_token_count: u64,
    total_base_units: u64,
    stage_contributions: &'a [StageContribution],
}

impl<'a> From<&'a MeasuredAccounting> for PartialMeasuredAccountingCanonical<'a> {
    fn from(m: &'a MeasuredAccounting) -> Self {
        Self {
            tokenizer_hash: &m.tokenizer_hash,
            input_token_count: m.input_token_count,
            output_token_count: m.output_token_count,
            total_base_units: m.total_base_units,
            stage_contributions: &m.stage_contributions,
        }
    }
}

impl<'a> From<&'a PartialContributorResult> for PartialResultCanonicalBody<'a> {
    fn from(p: &'a PartialContributorResult) -> Self {
        Self {
            schema_version: p.schema_version,
            session_id: &p.session_id,
            assignment_id: &p.assignment_id,
            contributor_pubkey_hex: &p.contributor_pubkey_hex,
            partial_artifact_snip_root: &p.partial_artifact_snip_root,
            partial_artifact_hash: &p.partial_artifact_hash,
            measured_accounting: (&p.measured_accounting).into(),
            produced_at_utc: &p.produced_at_utc,
        }
    }
}

pub fn canonical_partial_result_bytes(
    p: &PartialContributorResult,
) -> Result<Vec<u8>, CanonicalError> {
    let body: PartialResultCanonicalBody = p.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(PARTIAL_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(PARTIAL_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn partial_result_signing_input(
    p: &PartialContributorResult,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_partial_result_bytes(p)
}

// --- AggregatedContributorResult ---

#[derive(Debug, Serialize)]
struct AggregatedCanonicalBody<'a> {
    schema_version: u32,
    session_id: &'a str,
    posted_id: &'a str,
    final_result_snip_root: &'a str,
    final_result_canonical_hash: &'a str,
    partial_refs: &'a [AggregatedPartialRef],
    aggregated_at_utc: &'a str,
    coordinator_pubkey_hex: &'a str,
}

impl<'a> From<&'a AggregatedContributorResult> for AggregatedCanonicalBody<'a> {
    fn from(a: &'a AggregatedContributorResult) -> Self {
        Self {
            schema_version: a.schema_version,
            session_id: &a.session_id,
            posted_id: &a.posted_id,
            final_result_snip_root: &a.final_result_snip_root,
            final_result_canonical_hash: &a.final_result_canonical_hash,
            partial_refs: &a.partial_refs,
            aggregated_at_utc: &a.aggregated_at_utc,
            coordinator_pubkey_hex: &a.coordinator_pubkey_hex,
        }
    }
}

pub fn canonical_aggregated_result_bytes(
    a: &AggregatedContributorResult,
) -> Result<Vec<u8>, CanonicalError> {
    let body: AggregatedCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(AGGREGATED_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(AGGREGATED_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn aggregated_result_signing_input(
    a: &AggregatedContributorResult,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_aggregated_result_bytes(a)
}

// ── Stage 12.11 — WorkAssignmentSupersession canonical bytes ──────────────

use crate::supersession::{SupersessionReason, WorkAssignmentSupersession};

/// 47 ASCII bytes.
pub const SUPERSESSION_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-SESSION-SUPERSESSION:v1:";

#[derive(Debug, Serialize)]
struct SupersessionCanonicalBody<'a> {
    schema_version: u32,
    session_id: &'a str,
    superseded_assignment_ids: &'a [String],
    replacement_assignment_ids: &'a [String],
    reason: &'a SupersessionReason,
    created_at_utc: &'a str,
    coordinator_pubkey_hex: &'a str,
}

impl<'a> From<&'a WorkAssignmentSupersession> for SupersessionCanonicalBody<'a> {
    fn from(s: &'a WorkAssignmentSupersession) -> Self {
        Self {
            schema_version: s.schema_version,
            session_id: &s.session_id,
            superseded_assignment_ids: &s.superseded_assignment_ids,
            replacement_assignment_ids: &s.replacement_assignment_ids,
            reason: &s.reason,
            created_at_utc: &s.created_at_utc,
            coordinator_pubkey_hex: &s.coordinator_pubkey_hex,
        }
    }
}

pub fn canonical_work_assignment_supersession_bytes(
    s: &WorkAssignmentSupersession,
) -> Result<Vec<u8>, CanonicalError> {
    let body: SupersessionCanonicalBody = s.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(SUPERSESSION_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(SUPERSESSION_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// 64-char lowercase hex. Value stored in
/// `WorkAssignmentSupersession.supersession_id`.
pub fn supersession_id_hex(
    s: &WorkAssignmentSupersession,
) -> Result<String, CanonicalError> {
    let bytes = canonical_work_assignment_supersession_bytes(s)?;
    Ok(hex_lower(blake3::hash(&bytes).as_bytes()))
}

pub fn work_assignment_supersession_signing_input(
    s: &WorkAssignmentSupersession,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_work_assignment_supersession_bytes(s)
}

// ── Stage 12.3 — session network announcement canonical bytes ────────────

use crate::net::{
    NetworkAggregatedResultAnnouncement, NetworkContributorJoinedAnnouncement,
    NetworkPartialResultAnnouncement, NetworkSessionOpenedAnnouncement,
    NetworkWorkAssignedAnnouncement,
};

/// 36 ASCII bytes.
pub const NET_SESSION_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-NET-SESSION:v1:";

/// 33 ASCII bytes.
pub const NET_JOIN_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-NET-JOIN:v1:";

/// 35 ASCII bytes.
pub const NET_ASSIGN_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-NET-ASSIGN:v1:";

/// 36 ASCII bytes.
pub const NET_PARTIAL_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-NET-PARTIAL:v1:";

/// 39 ASCII bytes.
pub const NET_AGGREGATED_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-NET-AGGREGATED:v1:";

// --- NetworkSessionOpenedAnnouncement ---

#[derive(Debug, Serialize)]
struct NetSessionOpenedCanonicalBody<'a> {
    schema_version: u32,
    execution_session_snip_root: &'a str,
    session_id: &'a str,
    posted_id: &'a str,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a NetworkSessionOpenedAnnouncement> for NetSessionOpenedCanonicalBody<'a> {
    fn from(a: &'a NetworkSessionOpenedAnnouncement) -> Self {
        Self {
            schema_version: a.schema_version,
            execution_session_snip_root: &a.execution_session_snip_root,
            session_id: &a.session_id,
            posted_id: &a.posted_id,
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

pub fn canonical_net_session_opened_bytes(
    a: &NetworkSessionOpenedAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetSessionOpenedCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_SESSION_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_SESSION_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn net_session_opened_signing_input(
    a: &NetworkSessionOpenedAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_net_session_opened_bytes(a)
}

// --- NetworkContributorJoinedAnnouncement ---

#[derive(Debug, Serialize)]
struct NetJoinCanonicalBody<'a> {
    schema_version: u32,
    contributor_join_snip_root: &'a str,
    session_id: &'a str,
    contributor_pubkey_hex: &'a str,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a NetworkContributorJoinedAnnouncement> for NetJoinCanonicalBody<'a> {
    fn from(a: &'a NetworkContributorJoinedAnnouncement) -> Self {
        Self {
            schema_version: a.schema_version,
            contributor_join_snip_root: &a.contributor_join_snip_root,
            session_id: &a.session_id,
            contributor_pubkey_hex: &a.contributor_pubkey_hex,
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

pub fn canonical_net_join_bytes(
    a: &NetworkContributorJoinedAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetJoinCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_JOIN_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_JOIN_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn net_join_signing_input(
    a: &NetworkContributorJoinedAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_net_join_bytes(a)
}

// --- NetworkWorkAssignedAnnouncement ---

#[derive(Debug, Serialize)]
struct NetAssignCanonicalBody<'a> {
    schema_version: u32,
    work_assignment_snip_root: &'a str,
    session_id: &'a str,
    assignment_id: &'a str,
    contributor_pubkey_hex: &'a str,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a NetworkWorkAssignedAnnouncement> for NetAssignCanonicalBody<'a> {
    fn from(a: &'a NetworkWorkAssignedAnnouncement) -> Self {
        Self {
            schema_version: a.schema_version,
            work_assignment_snip_root: &a.work_assignment_snip_root,
            session_id: &a.session_id,
            assignment_id: &a.assignment_id,
            contributor_pubkey_hex: &a.contributor_pubkey_hex,
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

pub fn canonical_net_assign_bytes(
    a: &NetworkWorkAssignedAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetAssignCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_ASSIGN_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_ASSIGN_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn net_assign_signing_input(
    a: &NetworkWorkAssignedAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_net_assign_bytes(a)
}

// --- NetworkPartialResultAnnouncement ---

#[derive(Debug, Serialize)]
struct NetPartialCanonicalBody<'a> {
    schema_version: u32,
    partial_result_snip_root: &'a str,
    session_id: &'a str,
    assignment_id: &'a str,
    contributor_pubkey_hex: &'a str,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a NetworkPartialResultAnnouncement> for NetPartialCanonicalBody<'a> {
    fn from(a: &'a NetworkPartialResultAnnouncement) -> Self {
        Self {
            schema_version: a.schema_version,
            partial_result_snip_root: &a.partial_result_snip_root,
            session_id: &a.session_id,
            assignment_id: &a.assignment_id,
            contributor_pubkey_hex: &a.contributor_pubkey_hex,
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

pub fn canonical_net_partial_bytes(
    a: &NetworkPartialResultAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetPartialCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_PARTIAL_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_PARTIAL_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn net_partial_signing_input(
    a: &NetworkPartialResultAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_net_partial_bytes(a)
}

// --- NetworkAggregatedResultAnnouncement ---

#[derive(Debug, Serialize)]
struct NetAggregatedCanonicalBody<'a> {
    schema_version: u32,
    aggregated_result_snip_root: &'a str,
    session_id: &'a str,
    posted_id: &'a str,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a NetworkAggregatedResultAnnouncement> for NetAggregatedCanonicalBody<'a> {
    fn from(a: &'a NetworkAggregatedResultAnnouncement) -> Self {
        Self {
            schema_version: a.schema_version,
            aggregated_result_snip_root: &a.aggregated_result_snip_root,
            session_id: &a.session_id,
            posted_id: &a.posted_id,
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

pub fn canonical_net_aggregated_bytes(
    a: &NetworkAggregatedResultAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetAggregatedCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_AGGREGATED_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_AGGREGATED_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn net_aggregated_signing_input(
    a: &NetworkAggregatedResultAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_net_aggregated_bytes(a)
}

// ── Stage 12.4 — activation handoff canonical bytes ──────────────────────
//
// The canonical signing body EXCLUDES `tensor_chunk_bytes` and
// `sender_signature_hex`. The signature binds `tensor_hash` +
// `byte_len` + chunk metadata + the from/to assignment / pubkey
// identifiers. The receiver is required to re-hash reassembled bytes
// and reject on mismatch (`handoff_verify::process_activation_handoff`).

use crate::handoff::{ActivationHandoff, TensorDtype};

/// 33 ASCII bytes.
pub const HANDOFF_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-HANDOFF:v1:";

#[derive(Debug, Serialize)]
struct ActivationHandoffCanonicalBody<'a> {
    schema_version: u32,
    session_id: &'a str,
    from_assignment_id: &'a str,
    to_assignment_id: &'a str,
    from_contributor_pubkey_hex: &'a str,
    to_contributor_pubkey_hex: &'a str,
    dtype: &'a TensorDtype,
    shape: &'a [u64],
    byte_len: u64,
    tensor_hash: &'a str,
    chunk_index: u32,
    chunk_count: u32,
    produced_at_utc: &'a str,
}

impl<'a> From<&'a ActivationHandoff> for ActivationHandoffCanonicalBody<'a> {
    fn from(h: &'a ActivationHandoff) -> Self {
        Self {
            schema_version: h.schema_version,
            session_id: &h.session_id,
            from_assignment_id: &h.from_assignment_id,
            to_assignment_id: &h.to_assignment_id,
            from_contributor_pubkey_hex: &h.from_contributor_pubkey_hex,
            to_contributor_pubkey_hex: &h.to_contributor_pubkey_hex,
            dtype: &h.dtype,
            shape: &h.shape,
            byte_len: h.byte_len,
            tensor_hash: &h.tensor_hash,
            chunk_index: h.chunk_index,
            chunk_count: h.chunk_count,
            produced_at_utc: &h.produced_at_utc,
        }
    }
}

pub fn canonical_activation_handoff_bytes(
    h: &ActivationHandoff,
) -> Result<Vec<u8>, CanonicalError> {
    let body: ActivationHandoffCanonicalBody = h.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(HANDOFF_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(HANDOFF_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Bytes the sender signs over. Equal to
/// `canonical_activation_handoff_bytes`.
pub fn activation_handoff_signing_input(
    h: &ActivationHandoff,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_activation_handoff_bytes(h)
}

// ── Stage 12.5 — peer advertisement canonical bytes ──────────────────────
//
// The canonical signing body EXCLUDES `advertisement_id` (which IS
// the BLAKE3 of this body, lower-hex) and `contributor_signature_hex`.
// Bincode 1.3 wire layout; field order frozen for
// `schema_version: 1`. Capabilities are serialized as a sub-struct
// in declaration order.

use crate::peer_advert::{ContributorPeerAdvertisement, PeerCapabilities};

/// 36 ASCII bytes.
pub const PEER_ADVERT_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-PEER-ADVERT:v1:";

#[derive(Debug, Serialize)]
struct PeerAdvertCanonicalBody<'a> {
    schema_version: u32,
    session_id: &'a str,
    contributor_pubkey_hex: &'a str,
    libp2p_peer_id: &'a str,
    listen_multiaddrs: &'a [String],
    capabilities: PeerCapabilitiesCanonical<'a>,
    advertised_at_utc: &'a str,
    expires_at_utc: &'a str,
}

#[derive(Debug, Serialize)]
struct PeerCapabilitiesCanonical<'a> {
    supports_live_handoff: bool,
    max_handoff_chunk_bytes: u64,
    supported_dtypes: &'a [TensorDtype],
}

impl<'a> From<&'a PeerCapabilities> for PeerCapabilitiesCanonical<'a> {
    fn from(c: &'a PeerCapabilities) -> Self {
        Self {
            supports_live_handoff: c.supports_live_handoff,
            max_handoff_chunk_bytes: c.max_handoff_chunk_bytes,
            supported_dtypes: &c.supported_dtypes,
        }
    }
}

impl<'a> From<&'a ContributorPeerAdvertisement> for PeerAdvertCanonicalBody<'a> {
    fn from(a: &'a ContributorPeerAdvertisement) -> Self {
        Self {
            schema_version: a.schema_version,
            session_id: &a.session_id,
            contributor_pubkey_hex: &a.contributor_pubkey_hex,
            libp2p_peer_id: &a.libp2p_peer_id,
            listen_multiaddrs: &a.listen_multiaddrs,
            capabilities: (&a.capabilities).into(),
            advertised_at_utc: &a.advertised_at_utc,
            expires_at_utc: &a.expires_at_utc,
        }
    }
}

pub fn canonical_peer_advertisement_bytes(
    a: &ContributorPeerAdvertisement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: PeerAdvertCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(PEER_ADVERT_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(PEER_ADVERT_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// 32-byte BLAKE3 of `canonical_peer_advertisement_bytes`.
pub fn peer_advertisement_hash_bytes(
    a: &ContributorPeerAdvertisement,
) -> Result<[u8; 32], CanonicalError> {
    let bytes = canonical_peer_advertisement_bytes(a)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

/// 64-char lowercase hex. Value stored in
/// `ContributorPeerAdvertisement.advertisement_id`.
pub fn advertisement_id_hex(
    a: &ContributorPeerAdvertisement,
) -> Result<String, CanonicalError> {
    Ok(hex_lower(&peer_advertisement_hash_bytes(a)?))
}

/// Bytes the contributor signs over. Equal to
/// `canonical_peer_advertisement_bytes`.
pub fn peer_advertisement_signing_input(
    a: &ContributorPeerAdvertisement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_peer_advertisement_bytes(a)
}

// --- NetworkPeerAdvertisementAnnouncement ---

use crate::net::NetworkPeerAdvertisementAnnouncement;

/// 40 ASCII bytes.
pub const NET_PEER_ADVERT_DOMAIN: &[u8] = b"OMNINODE-CONTRIBUTOR-NET-PEER-ADVERT:v1:";

#[derive(Debug, Serialize)]
struct NetPeerAdvertCanonicalBody<'a> {
    schema_version: u32,
    peer_advertisement_snip_root: &'a str,
    advertisement_id: &'a str,
    session_id: &'a str,
    contributor_pubkey_hex: &'a str,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a NetworkPeerAdvertisementAnnouncement> for NetPeerAdvertCanonicalBody<'a> {
    fn from(a: &'a NetworkPeerAdvertisementAnnouncement) -> Self {
        Self {
            schema_version: a.schema_version,
            peer_advertisement_snip_root: &a.peer_advertisement_snip_root,
            advertisement_id: &a.advertisement_id,
            session_id: &a.session_id,
            contributor_pubkey_hex: &a.contributor_pubkey_hex,
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

pub fn canonical_net_peer_advert_bytes(
    a: &NetworkPeerAdvertisementAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetPeerAdvertCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_PEER_ADVERT_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_PEER_ADVERT_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn net_peer_advert_signing_input(
    a: &NetworkPeerAdvertisementAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_net_peer_advert_bytes(a)
}

// --- NetworkWorkAssignmentSupersessionAnnouncement (Stage 12.11) ---

pub const NET_SUPERSESSION_DOMAIN: &[u8] =
    b"OMNINODE-CONTRIBUTOR-NET-SUPERSESSION:v1:";

#[derive(Debug, Serialize)]
struct NetSupersessionCanonicalBody<'a> {
    schema_version: u32,
    work_assignment_supersession_snip_root: &'a str,
    session_id: &'a str,
    supersession_id: &'a str,
    announced_at_utc: &'a str,
    announcer_pubkey_hex: &'a str,
}

impl<'a> From<&'a crate::net::NetworkWorkAssignmentSupersessionAnnouncement>
    for NetSupersessionCanonicalBody<'a>
{
    fn from(
        a: &'a crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
    ) -> Self {
        Self {
            schema_version: a.schema_version,
            work_assignment_supersession_snip_root: &a
                .work_assignment_supersession_snip_root,
            session_id: &a.session_id,
            supersession_id: &a.supersession_id,
            announced_at_utc: &a.announced_at_utc,
            announcer_pubkey_hex: &a.announcer_pubkey_hex,
        }
    }
}

pub fn canonical_net_supersession_bytes(
    a: &crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    let body: NetSupersessionCanonicalBody = a.into();
    let body_bytes = bincode1::serialize(&body)?;
    let mut out = Vec::with_capacity(NET_SUPERSESSION_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(NET_SUPERSESSION_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

pub fn net_supersession_signing_input(
    a: &crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
) -> Result<Vec<u8>, CanonicalError> {
    canonical_net_supersession_bytes(a)
}

// ── Hex helpers ───────────────────────────────────────────────────────────

/// Lowercase-hex encode raw bytes (no `0x` prefix). Used throughout
/// the contributor crate for the bare-64-char BLAKE3 hash convention
/// (distinct from SNIP V2 IDs which are `0x`-prefixed). Exposed
/// publicly so test fixtures and CLI tooling can produce matching
/// strings without re-implementing.
pub fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(nibble(b >> 4));
        s.push(nibble(b & 0x0F));
    }
    s
}

fn nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + n - 10) as char,
        _ => unreachable!("nibble must be 0..=15"),
    }
}
