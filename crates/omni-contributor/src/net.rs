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

// ── Stage 12.3 — session network announcements ───────────────────────────
//
// Five pointer-only announcements, one per session event. Each
// carries drift-guard copies of the inner envelope's identity fields
// and a required announcer signature. Receivers fetch the inner
// body from SNIP and run the local verifier (`session_verify`) on
// what they get.

/// Announcement: a new `ExecutionSession` was published to SNIP.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkSessionOpenedAnnouncement {
    pub schema_version: u32,
    /// SNIP V2 Merkle root of the `ExecutionSession` JSON.
    pub execution_session_snip_root: String,
    /// 64-char lowercase hex — copy of `ExecutionSession.session_id`.
    pub session_id: String,
    /// 64-char lowercase hex — copy of `ExecutionSession.posted_id`.
    pub posted_id: String,
    /// RFC 3339 UTC (`Z` suffix).
    pub announced_at_utc: String,
    pub announcer_pubkey_hex: String,
    pub announcer_signature_hex: String,
}

impl NetworkSessionOpenedAnnouncement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion { got: self.schema_version });
        }
        check_snip_root_hex("execution_session_snip_root", &self.execution_session_snip_root)?;
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("posted_id", &self.posted_id)?;
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex("announcer_signature_hex", &self.announcer_signature_hex)?;
        Ok(())
    }
}

/// Announcement: a `ContributorJoin` was published to SNIP.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkContributorJoinedAnnouncement {
    pub schema_version: u32,
    /// SNIP V2 Merkle root of the `ContributorJoin` JSON.
    pub contributor_join_snip_root: String,
    pub session_id: String,
    /// Drift-guarded against the inner join.
    pub contributor_pubkey_hex: String,
    pub announced_at_utc: String,
    pub announcer_pubkey_hex: String,
    pub announcer_signature_hex: String,
}

impl NetworkContributorJoinedAnnouncement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion { got: self.schema_version });
        }
        check_snip_root_hex("contributor_join_snip_root", &self.contributor_join_snip_root)?;
        check_blake3_hex("session_id", &self.session_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex("announcer_signature_hex", &self.announcer_signature_hex)?;
        Ok(())
    }
}

/// Announcement: a `WorkAssignment` was published to SNIP.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkWorkAssignedAnnouncement {
    pub schema_version: u32,
    pub work_assignment_snip_root: String,
    pub session_id: String,
    pub assignment_id: String,
    pub contributor_pubkey_hex: String,
    pub announced_at_utc: String,
    pub announcer_pubkey_hex: String,
    pub announcer_signature_hex: String,
}

impl NetworkWorkAssignedAnnouncement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion { got: self.schema_version });
        }
        check_snip_root_hex("work_assignment_snip_root", &self.work_assignment_snip_root)?;
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("assignment_id", &self.assignment_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex("announcer_signature_hex", &self.announcer_signature_hex)?;
        Ok(())
    }
}

/// Announcement: a `PartialContributorResult` was published to SNIP.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkPartialResultAnnouncement {
    pub schema_version: u32,
    pub partial_result_snip_root: String,
    pub session_id: String,
    pub assignment_id: String,
    pub contributor_pubkey_hex: String,
    pub announced_at_utc: String,
    pub announcer_pubkey_hex: String,
    pub announcer_signature_hex: String,
}

impl NetworkPartialResultAnnouncement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion { got: self.schema_version });
        }
        check_snip_root_hex("partial_result_snip_root", &self.partial_result_snip_root)?;
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("assignment_id", &self.assignment_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex("announcer_signature_hex", &self.announcer_signature_hex)?;
        Ok(())
    }
}

/// Announcement: an `AggregatedContributorResult` was published to SNIP.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkAggregatedResultAnnouncement {
    pub schema_version: u32,
    pub aggregated_result_snip_root: String,
    pub session_id: String,
    pub posted_id: String,
    pub announced_at_utc: String,
    pub announcer_pubkey_hex: String,
    pub announcer_signature_hex: String,
}

impl NetworkAggregatedResultAnnouncement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion { got: self.schema_version });
        }
        check_snip_root_hex("aggregated_result_snip_root", &self.aggregated_result_snip_root)?;
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("posted_id", &self.posted_id)?;
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex("announcer_signature_hex", &self.announcer_signature_hex)?;
        Ok(())
    }
}

/// Stage 12.5 — pointer-only mesh announcement for a
/// `ContributorPeerAdvertisement` published to SNIP. The announcer
/// signature here is anti-spam + provenance; the inner advertisement
/// is verified separately by the routing-cache processor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkPeerAdvertisementAnnouncement {
    pub schema_version: u32,
    /// SNIP V2 Merkle root of the `ContributorPeerAdvertisement` JSON.
    pub peer_advertisement_snip_root: String,
    pub advertisement_id: String,
    pub session_id: String,
    pub contributor_pubkey_hex: String,
    pub announced_at_utc: String,
    pub announcer_pubkey_hex: String,
    pub announcer_signature_hex: String,
}

impl NetworkPeerAdvertisementAnnouncement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion { got: self.schema_version });
        }
        check_snip_root_hex(
            "peer_advertisement_snip_root",
            &self.peer_advertisement_snip_root,
        )?;
        check_blake3_hex("advertisement_id", &self.advertisement_id)?;
        check_blake3_hex("session_id", &self.session_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex("announcer_signature_hex", &self.announcer_signature_hex)?;
        Ok(())
    }
}

/// Stage 12.11 — pointer-only mesh announcement for a
/// `WorkAssignmentSupersession` published to SNIP. Same posture as
/// every other Stage 12.x announcement: the announcer signature is
/// anti-spam + provenance; the inner supersession envelope is
/// verified separately by the aggregate verifier
/// (`verify_aggregated_result_with_supersessions`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkWorkAssignmentSupersessionAnnouncement {
    pub schema_version: u32,
    /// SNIP V2 Merkle root of the `WorkAssignmentSupersession` JSON.
    pub work_assignment_supersession_snip_root: String,
    pub session_id: String,
    pub supersession_id: String,
    pub announced_at_utc: String,
    pub announcer_pubkey_hex: String,
    pub announcer_signature_hex: String,
}

impl NetworkWorkAssignmentSupersessionAnnouncement {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != NET_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion { got: self.schema_version });
        }
        check_snip_root_hex(
            "work_assignment_supersession_snip_root",
            &self.work_assignment_supersession_snip_root,
        )?;
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("supersession_id", &self.supersession_id)?;
        check_iso_8601("announced_at_utc", &self.announced_at_utc)?;
        check_pubkey_hex("announcer_pubkey_hex", &self.announcer_pubkey_hex)?;
        check_signature_hex(
            "announcer_signature_hex",
            &self.announcer_signature_hex,
        )?;
        Ok(())
    }
}
