//! Stage 12.2 — network announcement envelopes for the contributor
//! mesh.
//!
//! Two signed structs announce SNIP-pointed posted envelopes:
//!
//!   - [`NetworkPostedJobAnnouncement`] — a dispatcher (or relayer)
//!     announces a `PostedJob` SNIP root to the contributor mesh.
//!     Receivers fetch the inner `PostedJob` from SNIP, validate it
//!     under the existing Stage 12.1 schema/signature/hash rules,
//!     then drop into the existing watch pipeline.
//!
//!   - [`NetworkPostedResultAnnouncement`] — a contributor announces
//!     a `PostedResultLink` SNIP root. Receivers fetch the link
//!     envelope, validate it, and (in 12.2) write it to disk for
//!     later inspection.
//!
//! Both messages **announce pointers only** — never job bodies,
//! input bytes, model bytes, result bytes. SNIP remains the content
//! store. The network signature is anti-spam + provenance; it does
//! NOT replace the existing 12.0/12.1 inner-envelope validators.
//!
//! Both schemas are frozen at `schema_version: 1`. Extensions are a
//! v2 migration (same policy as the rest of the contributor crate).

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;
use crate::job::{check_blake3_hex, check_iso_8601, check_pubkey_hex, check_signature_hex, check_snip_root_hex};

/// Pinned at 1 for Stage 12.2. Each closed-enum extension or field
/// reorder is a `schema_version: 2` migration.
pub const NET_SCHEMA_VERSION: u32 = 1;

// ── NetworkPostedJobAnnouncement ──────────────────────────────────────────

/// Signed network announcement for a `PostedJob` SNIP root. Carries
/// drift-guard copies (`posted_id`, `job_hash`, `model_hash`) so a
/// receiver can pre-filter announcements before incurring the SNIP
/// fetch + the inner-envelope validation cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkPostedJobAnnouncement {
    pub schema_version: u32,

    /// SNIP V2 Merkle root of the `PostedJob` JSON (`0x`-prefixed
    /// lowercase hex, 66 chars).
    pub posted_job_snip_root: String,

    /// 64-char lowercase hex copy of the `PostedJob.posted_id`.
    /// Drift guard: a receiver fetches `posted_job_snip_root` from
    /// SNIP, recomputes `posted_id`, and refuses on mismatch.
    pub posted_id: String,

    /// 64-char lowercase hex copy of the `PostedJob.job_hash`.
    pub job_hash: String,

    /// 64-char lowercase hex copy of the `PostedJob.model_hash`.
    /// Permits cheap pre-fetch filtering against an accept-list.
    pub model_hash: String,

    /// Optional 64-char lowercase hex of the inner
    /// `ContributorJob.accounting.tokenizer_hash`. Populated only if
    /// the announcer fetched the inner job before broadcasting;
    /// `None` is acceptable. Receivers that filter on tokenizer
    /// must fetch the inner job to confirm.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tokenizer_hash: Option<String>,

    /// RFC 3339 UTC (`Z` suffix).
    pub announced_at_utc: String,

    /// 64-char lowercase hex Ed25519 public key of the announcer.
    /// Always present (announcer signature is **required** in 12.2
    /// — different from `PostedJob.poster_signature_hex` which is
    /// optional).
    pub announcer_pubkey_hex: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// announcement body (which excludes this field).
    pub announcer_signature_hex: String,
}

impl NetworkPostedJobAnnouncement {
    /// Schema-level validation. Does NOT verify the announcer
    /// signature (callers do that against the canonical signing
    /// input) or fetch the SNIP-pointed `PostedJob`.
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_snip_root_hex("posted_job_snip_root", &self.posted_job_snip_root)?;
        check_blake3_hex("posted_id", &self.posted_id)?;
        check_blake3_hex("job_hash", &self.job_hash)?;
        check_blake3_hex("model_hash", &self.model_hash)?;
        if let Some(ref t) = self.tokenizer_hash {
            check_blake3_hex("tokenizer_hash", t)?;
        }
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex("announcer_signature_hex", &self.announcer_signature_hex)?;
        Ok(())
    }
}

// ── NetworkPostedResultAnnouncement ───────────────────────────────────────

/// Signed network announcement for a `PostedResultLink` SNIP root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkPostedResultAnnouncement {
    pub schema_version: u32,

    /// 64-char lowercase hex copy of the `PostedJob.posted_id` this
    /// result answers. Permits dispatcher-side filtering of results
    /// to a specific job.
    pub posted_id: String,

    /// SNIP V2 Merkle root of the `PostedResultLink` JSON
    /// (`0x`-prefixed lowercase hex, 66 chars).
    pub posted_result_link_snip_root: String,

    /// 64-char lowercase hex copy of the
    /// `PostedResultLink.result_canonical_hash`. Drift guard between
    /// the announcement and the SNIP-fetched link envelope.
    pub result_canonical_hash: String,

    /// 64-char lowercase hex Ed25519 public key of the contributor
    /// (copy of `PostedResultLink.contributor_pubkey_hex`).
    pub contributor_pubkey_hex: String,

    /// RFC 3339 UTC (`Z` suffix).
    pub announced_at_utc: String,

    /// Announcer identity (may or may not equal `contributor_pubkey_hex`
    /// — anyone may relay an announcement they observed). Always
    /// required.
    pub announcer_pubkey_hex: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// announcement body (excludes this field).
    pub announcer_signature_hex: String,
}

impl NetworkPostedResultAnnouncement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("posted_id", &self.posted_id)?;
        check_snip_root_hex(
            "posted_result_link_snip_root",
            &self.posted_result_link_snip_root,
        )?;
        check_blake3_hex("result_canonical_hash", &self.result_canonical_hash)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex("announcer_signature_hex", &self.announcer_signature_hex)?;
        Ok(())
    }
}
