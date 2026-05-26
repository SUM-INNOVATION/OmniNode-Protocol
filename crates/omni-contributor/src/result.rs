//! Stage 12.0 — `ContributorResult` schema.
//!
//! On-wire JSON shape the contributor publishes after running a job.
//! Carries the response SNIP root, evidence artifact, measured
//! accounting numbers, and an Ed25519 signature by the contributor
//! over the canonical signing input.
//!
//! See `docs/stage12-contributor-protocol.md` for the protocol-level
//! description.

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;
use crate::job::{
    check_blake3_hex, check_iso_8601, check_pubkey_hex, check_signature_hex, check_snip_root_hex,
    SCHEMA_VERSION,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContributorResult {
    pub schema_version: u32,

    // ── Binding to the originating job ──
    pub job_id: String,
    /// 64-char lowercase hex BLAKE3 of the canonical job bytes.
    /// Verifier recomputes from a separately-fetched job and asserts
    /// equality.
    pub job_hash: String,
    /// Optional convenience: the SNIP root a verifier may use to
    /// fetch the job. `verify-result` does NOT implicitly trust this
    /// field; it always requires the verifier to supply the job
    /// explicitly via `--job`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub job_snip_root: Option<String>,

    // ── Copies of job binding fields (verifier asserts equality with
    // ── the fetched job and recomputes the *_hash fields from
    // ── SNIP-fetched bytes).
    pub model_hash: String,
    pub input_hash: String,

    // ── The response ──
    pub response_snip_root: String,
    pub response_hash: String,

    // ── Evidence ──
    pub evidence: Evidence,

    // ── Measured accounting ──
    pub measured_accounting: MeasuredAccounting,

    // ── Contributor identity + signature ──
    pub produced_at_utc: String,
    pub contributor_pubkey_hex: String,
    pub contributor_signature_hex: String,

    /// Free-form audit string. **Part of the canonical signing
    /// input** — the contributor's signature covers this field.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredAccounting {
    /// Must equal `job.accounting.tokenizer_hash` exactly. Structural
    /// only — the verifier does NOT recompute tokens.
    pub tokenizer_hash: String,

    /// Measured at runtime by the inference runner. Verifier asserts
    /// `== job.accounting.input_token_count`.
    pub input_token_count: u64,

    /// Measured at runtime. Verifier asserts
    /// `≤ job.accounting.max_output_token_count`.
    pub output_token_count: u64,

    /// Total base units the job consumed under the protocol rule
    /// `1 base unit = 1 token`. Stage 12.0 enforces
    /// `total_base_units == input_token_count + output_token_count`.
    /// Stored explicitly so a future accounting variant (e.g.
    /// prompt-cache discounts) can diverge under a
    /// `schema_version: 2` migration.
    pub total_base_units: u64,

    /// Per-stage contribution records. Must be non-empty. For a
    /// single-runner job the list has one entry attributing all work
    /// to the top-level `contributor_pubkey_hex`; multi-contributor
    /// pipeline splits (future stage) emit one entry per participant.
    pub stage_contributions: Vec<StageContribution>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageContribution {
    /// 64-char lowercase hex of the Ed25519 public key of whoever
    /// performed this stage's work. For a single-runner job this
    /// equals the top-level `contributor_pubkey_hex`.
    pub contributor_pubkey_hex: String,

    /// Free-form identifier (e.g. `"stub-runner"`, `"prefill"`,
    /// `"decode"`, `"layers-0-31"`). Audit-only.
    pub stage_label: String,

    /// What unit the `work_units` count is in. Closed enum;
    /// extending is a `schema_version: 2` migration. Critical for
    /// future B/C/D reward-split engines to interpret entries
    /// consistently across runtimes.
    pub work_unit_kind: WorkUnitKind,

    /// Measured contribution count under `work_unit_kind`.
    pub work_units: u64,
}

/// Closed enum. Adding a variant is a `schema_version: 2` migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkUnitKind {
    /// Total tokens (input + output combined, or generic token count
    /// when the runner does not distinguish prefill/decode).
    Tokens,
    /// Tokens consumed at prefill.
    PrefillTokens,
    /// Tokens produced at decode.
    DecodeTokens,
    /// Count of transformer layers executed.
    Layers,
    /// Estimated FLOPs as a `u64` count (NOT a dollarized cost).
    FlopsEstimate,
}

/// Closed enum. Stage 12.0 ships only `AttestationOnly`. The
/// production-proof variant
/// (`Stage11dProductionFixedPointMlpProof`) is a reserved future name
/// documented in `docs/stage12-contributor-protocol.md` and is NOT
/// present in this Rust enum — adding it is a `schema_version: 2`
/// migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Evidence {
    AttestationOnly,
}

// ── Validation ────────────────────────────────────────────────────────────

impl ContributorResult {
    /// Schema-level validation. Does NOT recompute `job_hash` or
    /// verify signatures — those are the verifier's job in
    /// `verify::verify_result`.
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("job_id", &self.job_id)?;
        check_blake3_hex("job_hash", &self.job_hash)?;
        if let Some(ref s) = self.job_snip_root {
            check_snip_root_hex("job_snip_root", s)?;
        }
        check_blake3_hex("model_hash", &self.model_hash)?;
        check_blake3_hex("input_hash", &self.input_hash)?;
        check_snip_root_hex("response_snip_root", &self.response_snip_root)?;
        check_blake3_hex("response_hash", &self.response_hash)?;
        self.measured_accounting.validate_schema()?;
        check_iso_8601("produced_at_utc", &self.produced_at_utc)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_signature_hex("contributor_signature_hex", &self.contributor_signature_hex)?;
        Ok(())
    }
}

impl MeasuredAccounting {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        check_blake3_hex("tokenizer_hash", &self.tokenizer_hash)?;
        if self.stage_contributions.is_empty() {
            return Err(SchemaError::EmptyStageContributions);
        }
        for (i, sc) in self.stage_contributions.iter().enumerate() {
            sc.validate_schema(i)?;
        }
        Ok(())
    }
}

impl StageContribution {
    pub fn validate_schema(&self, _index: usize) -> Result<(), SchemaError> {
        check_pubkey_hex("stage_contribution.contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        // stage_label and work_unit_kind are free-form / closed-enum;
        // no further checks needed.
        Ok(())
    }
}
