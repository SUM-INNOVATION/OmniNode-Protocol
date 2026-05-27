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
