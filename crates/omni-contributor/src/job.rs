//! Stage 12.0 — `ContributorJob` schema.
//!
//! On-wire JSON shape (snake_case external) for the job envelope a
//! dispatcher hands to a contributor. The contributor signs a
//! `ContributorResult` that binds to this job's content (via
//! `job_hash`); the verifier recomputes `job_hash` from the canonical
//! bytes (see `canonical.rs`) and asserts the binding.
//!
//! All hash / pubkey / signature fields are lowercase hex of the raw
//! bytes (no `0x` prefix). 64 hex chars for BLAKE3 hashes and Ed25519
//! public keys; 128 hex chars for Ed25519 signatures.
//!
//! See `docs/stage12-contributor-protocol.md` for the protocol-level
//! description and canonical-bytes layout.

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;

/// Pinned at 1 for Stage 12.0. Any closed-enum extension
/// (`VerificationRequirement`, `Evidence`, `WorkUnitKind`,
/// `BaseUnitRewardPolicy`) is a `schema_version: 2` migration.
pub const SCHEMA_VERSION: u32 = 1;

/// On-wire `ContributorJob` envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContributorJob {
    /// Pinned at [`SCHEMA_VERSION`].
    pub schema_version: u32,

    /// Lowercase hex of the canonical `job_hash`. Redundant with
    /// `job_hash` (logically `job_id == job_hash`); kept as a
    /// dedicated field for filename / log-line ergonomics. Validator
    /// asserts equality.
    pub job_id: String,

    // ── What to run ──
    /// 64-char lowercase hex BLAKE3 of the model bytes the
    /// dispatcher claims to be requesting against. Same shape as
    /// `omni_zkml::ProofMetadata.model_hash`.
    pub model_hash: String,

    /// Lowercase hex (66 chars including `0x` prefix) of the SNIP V2
    /// Merkle root for the manifest tree.
    pub manifest_snip_root: String,

    /// Lowercase hex (66 chars including `0x` prefix) of the SNIP V2
    /// Merkle root for the input bytes.
    pub input_snip_root: String,

    /// 64-char lowercase hex BLAKE3 of the input bytes (drift guard;
    /// verifier recomputes after fetching from SNIP).
    pub input_hash: String,

    // ── Evidence requirement ──
    pub verification_requirement: VerificationRequirement,

    // ── Forward-compatible accounting placeholders ──
    pub accounting: JobAccounting,

    // ── Provenance ──
    /// RFC 3339 / ISO 8601 UTC timestamp (`Z` suffix).
    pub dispatched_at_utc: String,

    /// Optional RFC 3339 UTC expiry. If present and past `now()` at
    /// verification time, the verifier refuses.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub expires_at_utc: Option<String>,

    /// Optional 64-char lowercase hex of the dispatcher's Ed25519
    /// public key. Iff present, `dispatcher_signature_hex` must also
    /// be present and is verified over the canonical signing input.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dispatcher_pubkey_hex: Option<String>,

    /// Optional 128-char lowercase hex Ed25519 signature. Iff present,
    /// `dispatcher_pubkey_hex` must also be present.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dispatcher_signature_hex: Option<String>,

    /// Free-form audit string. **Part of the canonical signing
    /// input** — the dispatcher's signature covers this field, so a
    /// downstream party cannot quietly alter the notes a contributor
    /// (or auditor) saw. If the dispatcher wants mutable audit
    /// metadata, that's a separate side channel; on-envelope notes
    /// are signed.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
}

/// Accounting placeholders. Stage 12.0 records and verifies these
/// numbers structurally but does NOT compute, distribute, or settle
/// any reward. The future payment engine (Stage 12.x+) consumes
/// these fields plus `MeasuredAccounting` from the result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobAccounting {
    /// 64-char lowercase hex BLAKE3 of the canonical tokenizer bytes
    /// (config + vocab tables, encoder-specific). Verifier asserts
    /// structural equality with the contributor's measured
    /// `tokenizer_hash`; **does NOT recompute tokens**.
    pub tokenizer_hash: String,

    /// Free-form, namespace-style tokenizer identifier (e.g.
    /// `"tiktoken/cl100k_base"`, `"llama/tokenizer-v3"`). Non-empty
    /// UTF-8. Audit-only — the verifier does not consult this string.
    pub tokenizer_id: String,

    /// Number of input tokens the dispatcher declares. Verifier
    /// asserts strict equality with the contributor's measured count
    /// (meaningful only because both are tied to the same
    /// `tokenizer_hash`).
    pub input_token_count: u64,

    /// Upper bound on output tokens this job will be accounted for.
    /// Verifier asserts the contributor's measured output count is
    /// `≤ max_output_token_count`.
    pub max_output_token_count: u64,

    /// Forward-compatible reward policy. Stage 12.0 ships only
    /// `Unspecified`; future variants (e.g. flat-per-base-unit) are
    /// a `schema_version: 2` migration.
    pub base_unit_reward_policy: BaseUnitRewardPolicy,
}

/// Closed enum. Adding a variant is a `schema_version: 2` migration.
/// Stage 12.0 ships only `Unspecified` — Stage 12.0 does not
/// implement payment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BaseUnitRewardPolicy {
    /// No reward policy declared. Stage 12.0 records but does not
    /// settle any reward.
    Unspecified,
}

/// Closed enum. Stage 12.0 ships only `AttestationOnly`. The
/// production-proof variant (`Stage11dProductionFixedPointMlpProof`)
/// is a reserved future name documented in
/// `docs/stage12-contributor-protocol.md` and is NOT present in this
/// Rust enum — adding it is a `schema_version: 2` migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum VerificationRequirement {
    AttestationOnly,
}

// ── Validation ────────────────────────────────────────────────────────────

impl ContributorJob {
    /// Schema-level validation: hex widths, ISO 8601, half-set
    /// dispatcher identity, non-empty tokenizer_id, schema_version
    /// pin. Does NOT recompute `job_hash` (that lives in
    /// `canonical::recompute_job_hash_hex` so the verifier can call
    /// it once and reuse the recomputed value).
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("model_hash", &self.model_hash)?;
        check_blake3_hex("input_hash", &self.input_hash)?;
        check_snip_root_hex("manifest_snip_root", &self.manifest_snip_root)?;
        check_snip_root_hex("input_snip_root", &self.input_snip_root)?;
        check_blake3_hex("job_id", &self.job_id)?;
        self.accounting.validate_schema()?;
        check_iso_8601("dispatched_at_utc", &self.dispatched_at_utc)?;
        if let Some(ref s) = self.expires_at_utc {
            check_iso_8601("expires_at_utc", s)?;
        }
        match (&self.dispatcher_pubkey_hex, &self.dispatcher_signature_hex) {
            (Some(pk), Some(sig)) => {
                check_pubkey_hex("dispatcher_pubkey_hex", pk)?;
                check_signature_hex("dispatcher_signature_hex", sig)?;
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

impl JobAccounting {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        check_blake3_hex("tokenizer_hash", &self.tokenizer_hash)?;
        if self.tokenizer_id.is_empty() {
            return Err(SchemaError::EmptyTokenizerId);
        }
        Ok(())
    }
}

// ── Internal hex / timestamp validators ───────────────────────────────────

pub(crate) fn check_blake3_hex(field: &'static str, value: &str) -> Result<(), SchemaError> {
    if value.len() != 64 || !value.chars().all(is_lower_hex) {
        return Err(SchemaError::MalformedHash {
            field,
            got: value.to_string(),
        });
    }
    Ok(())
}

pub(crate) fn check_pubkey_hex(field: &'static str, value: &str) -> Result<(), SchemaError> {
    if value.len() != 64 || !value.chars().all(is_lower_hex) {
        return Err(SchemaError::MalformedPubkey {
            field,
            got: value.to_string(),
        });
    }
    Ok(())
}

pub(crate) fn check_signature_hex(field: &'static str, value: &str) -> Result<(), SchemaError> {
    if value.len() != 128 || !value.chars().all(is_lower_hex) {
        return Err(SchemaError::MalformedSignature {
            field,
            got: value.to_string(),
        });
    }
    Ok(())
}

/// SNIP V2 object IDs are `0x`-prefixed lowercase hex (66 chars
/// total — see `omni_types::phase5::SnipV2ObjectId`). Distinct shape
/// from the bare 64-char BLAKE3 hex used for `model_hash` etc.
pub(crate) fn check_snip_root_hex(field: &'static str, value: &str) -> Result<(), SchemaError> {
    if value.len() != 66 || !value.starts_with("0x")
        || !value[2..].chars().all(is_lower_hex)
    {
        return Err(SchemaError::MalformedHash {
            field,
            got: value.to_string(),
        });
    }
    Ok(())
}

pub(crate) fn check_iso_8601(field: &'static str, value: &str) -> Result<(), SchemaError> {
    // Stage 12.0 protocol pins the `Z` UTC suffix. Other RFC 3339
    // offsets (`+00:00`, `+02:00`, etc.) are rejected so the on-disk
    // / on-wire timestamp shape is exactly one canonical form. A
    // mixed-offset corpus is harder to audit; an exact-string match
    // on `Z`-terminated UTC keeps `produced_at_utc` / `dispatched_at_utc`
    // grep-able and unambiguous.
    if !value.ends_with('Z') {
        return Err(SchemaError::MalformedTimestamp {
            field,
            got: value.to_string(),
        });
    }
    chrono::DateTime::parse_from_rfc3339(value).map_err(|_| SchemaError::MalformedTimestamp {
        field,
        got: value.to_string(),
    })?;
    Ok(())
}

fn is_lower_hex(c: char) -> bool {
    matches!(c, '0'..='9' | 'a'..='f')
}
