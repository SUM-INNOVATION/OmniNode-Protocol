//! Stage 12.1 — posted-envelope schemas for off-chain job discovery.
//!
//! Two envelopes:
//!
//!   - [`PostedJob`] — a dispatcher's announcement that a
//!     `ContributorJob` exists at a SNIP root. Written to a
//!     filesystem directory that a contributor watches; the body
//!     also includes drift-guard hashes so a contributor can refuse
//!     a posting whose claimed `job_hash` / `model_hash` doesn't
//!     match the SNIP-fetched job content.
//!
//!   - [`PostedResultLink`] — a contributor's announcement that a
//!     `ContributorResult` has been published to SNIP in response
//!     to a specific `posted_id`. Optional in the watch loop;
//!     useful for dispatchers polling for results without scanning
//!     SNIP blindly.
//!
//! Both schemas use the same canonical-bytes regime as Stage 12.0:
//! bincode 1.3, domain separators in `canonical.rs`, frozen field
//! order. `posted_id` is the lowercase hex of
//! `BLAKE3(POSTED_JOB_DOMAIN || canonical_body)`. The
//! `result_canonical_hash` field on `PostedResultLink` is BLAKE3 of
//! the **canonical** result bytes (i.e., the signature-domain
//! hash) — distinct from BLAKE3 of the on-disk JSON bytes; lets a
//! consumer detect schema/signature drift without re-encoding.

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;
use crate::job::{check_blake3_hex, check_iso_8601, check_pubkey_hex, check_signature_hex, check_snip_root_hex};

/// Pinned at 1 for Stage 12.1. Same migration policy as Stage 12.0:
/// any closed-enum extension or field reorder is a v2 migration.
pub const POSTED_SCHEMA_VERSION: u32 = 1;

// ── PostedJob ─────────────────────────────────────────────────────────────

/// On-wire `PostedJob` envelope. Written by `post-job` to a directory
/// that a contributor's `watch-jobs --source fs --jobs-dir <path>`
/// observes. The body also points at the SNIP root of the actual
/// `ContributorJob` JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostedJob {
    pub schema_version: u32,

    /// 64-char lowercase hex of
    /// `BLAKE3(POSTED_JOB_DOMAIN || canonical_body)`. Validator
    /// asserts equality against the recomputed value.
    pub posted_id: String,

    /// SNIP V2 Merkle root of the `ContributorJob` JSON bytes
    /// (`0x`-prefixed lowercase hex, 66 chars).
    pub job_snip_root: String,

    /// 64-char lowercase hex copy of the inner job's `job_hash`.
    /// Drift guard: a contributor that fetches the job from SNIP
    /// recomputes its `job_hash` and refuses the posting if they
    /// disagree.
    pub job_hash: String,

    /// 64-char lowercase hex copy of the inner job's `model_hash`.
    /// Permits index-time filtering by model without fetching the
    /// full job from SNIP.
    pub model_hash: String,

    /// RFC 3339 UTC (`Z` suffix).
    pub posted_at_utc: String,

    /// Optional posting expiry. If present and past `now()` at watch
    /// time, the contributor skips the entry.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub expires_at_utc: Option<String>,

    /// Optional 64-char lowercase hex Ed25519 public key. Iff
    /// present, `poster_signature_hex` must also be present.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub poster_pubkey_hex: Option<String>,

    /// Optional 128-char lowercase hex Ed25519 signature over the
    /// canonical posted-job body. Iff present,
    /// `poster_pubkey_hex` must also be present.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub poster_signature_hex: Option<String>,

    /// Free-form audit string. Part of the canonical signing input.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
}

impl PostedJob {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != POSTED_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("posted_id", &self.posted_id)?;
        check_snip_root_hex("job_snip_root", &self.job_snip_root)?;
        check_blake3_hex("job_hash", &self.job_hash)?;
        check_blake3_hex("model_hash", &self.model_hash)?;
        check_iso_8601("posted_at_utc", &self.posted_at_utc)?;
        if let Some(ref s) = self.expires_at_utc {
            check_iso_8601("expires_at_utc", s)?;
        }
        match (&self.poster_pubkey_hex, &self.poster_signature_hex) {
            (Some(pk), Some(sig)) => {
                check_pubkey_hex("poster_pubkey_hex", pk)?;
                check_signature_hex("poster_signature_hex", sig)?;
            }
            (None, None) => {}
            (Some(_), None) => {
                return Err(SchemaError::InconsistentDispatcherIdentity {
                    pubkey_set: "Some",
                    signature_set: "None",
                });
            }
            (None, Some(_)) => {
                return Err(SchemaError::InconsistentDispatcherIdentity {
                    pubkey_set: "None",
                    signature_set: "Some",
                });
            }
        }
        Ok(())
    }
}

// ── PostedResultLink ──────────────────────────────────────────────────────

/// On-wire `PostedResultLink` envelope. Published by a contributor
/// to SNIP (and optionally also persisted locally) to signal that a
/// `ContributorResult` has been produced for a given `posted_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PostedResultLink {
    pub schema_version: u32,

    /// 64-char lowercase hex — copy of the `PostedJob.posted_id`
    /// this result answers.
    pub posted_id: String,

    /// SNIP V2 Merkle root of the `ContributorResult` JSON bytes
    /// (`0x`-prefixed lowercase hex, 66 chars).
    pub result_snip_root: String,

    /// 64-char lowercase hex of
    /// `BLAKE3(canonical_result_bytes(result))` — the
    /// signature-domain hash. Distinct from BLAKE3 of the on-disk
    /// JSON bytes; lets a consumer detect schema/signature drift
    /// without re-encoding.
    pub result_canonical_hash: String,

    /// 64-char lowercase hex Ed25519 public key of the contributor
    /// who produced the result.
    pub contributor_pubkey_hex: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// posted-result-link body.
    pub contributor_signature_hex: String,

    /// RFC 3339 UTC (`Z` suffix).
    pub published_at_utc: String,
}

impl PostedResultLink {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != POSTED_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("posted_id", &self.posted_id)?;
        check_snip_root_hex("result_snip_root", &self.result_snip_root)?;
        check_blake3_hex("result_canonical_hash", &self.result_canonical_hash)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_signature_hex(
            "contributor_signature_hex",
            &self.contributor_signature_hex,
        )?;
        check_iso_8601("published_at_utc", &self.published_at_utc)?;
        Ok(())
    }
}
