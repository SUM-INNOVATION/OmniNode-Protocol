//! Stage 12.3 — multi-contributor pooled-memory inference sessions.
//!
//! Signed coordination shell letting multiple contributor devices
//! cooperate on one posted job, recording each participant's join,
//! work assignment, and partial result under their own signatures,
//! and producing one aggregated `ContributorResult` that the
//! existing 12.0 verifier accepts unchanged.
//!
//! Stage 12.3 is **coordination, not live pooled RAM**. Inter-stage
//! artifact handoff goes through SNIP; actual low-latency shared
//! tensor / activation transport between participants is Stage 12.4.
//!
//! Five new on-wire envelopes, each pinned to `schema_version: 1`
//! with frozen domain separators in `crate::canonical`:
//!
//!   - [`ExecutionSession`]            — coordinator opens a session
//!                                       for a posted_id.
//!   - [`ContributorJoin`]             — contributor advertises
//!                                       capability + RAM hints.
//!   - [`WorkAssignment`]              — coordinator assigns one
//!                                       stage of work to a joined
//!                                       contributor.
//!   - [`PartialContributorResult`]    — contributor publishes a
//!                                       signed artifact for one
//!                                       assignment.
//!   - [`AggregatedContributorResult`] — coordinator wraps a
//!                                       standalone (v1) final
//!                                       `ContributorResult` with
//!                                       the partial chain it was
//!                                       built from.
//!
//! ## Accounting (Stage 12.3 rule)
//!
//! The final `ContributorResult.measured_accounting.total_base_units`
//! is the job-level `input_token_count + output_token_count`,
//! exactly as in 12.0. **It is NOT a sum of partial totals.**
//! Partials record per-stage `work_units` (used later to split the
//! base-unit pool); the verifier checks partial accounting for
//! structural validity but does not require numerical equality with
//! the final total.

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;
use crate::job::{check_blake3_hex, check_iso_8601, check_pubkey_hex, check_signature_hex,
                 check_snip_root_hex};
use crate::result::{MeasuredAccounting, WorkUnitKind};

/// Pinned `schema_version` for every Stage 12.3 envelope.
pub const SESSION_SCHEMA_VERSION: u32 = 1;

/// Maximum length (in chars) of a `WorkKind::Custom { label }`. Bounded
/// so canonical bytes stay reasonable and gossip messages stay small.
pub const WORK_KIND_CUSTOM_LABEL_MAX: usize = 64;

// ── WorkKind ──────────────────────────────────────────────────────────────

/// Closed enum (with a forward-compat `Custom` escape hatch) describing
/// what slice of the model an assignment covers.
///
/// `Custom(label)` exists because we cannot enumerate every real-world
/// split strategy at v1 freezing — e.g. `Custom("kv-cache-shard")`,
/// `Custom("expert-7")`. The label is validated as non-empty printable
/// ASCII, bounded length, so canonical bytes stay tight.
///
/// Adding a non-`Custom` variant is a `schema_version: 2` migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkKind {
    /// Prompt-phase compute over the full input.
    Prefill,
    /// Autoregressive decode phase.
    Decode,
    /// A contiguous half-open range of transformer layers `[start, end)`.
    Layers { start: u32, end: u32 },
    /// A model shard identified by zero-based index.
    Shard { index: u32 },
    /// Forward-compat custom label.
    Custom { label: String },
}

impl WorkKind {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        match self {
            WorkKind::Prefill | WorkKind::Decode => Ok(()),
            WorkKind::Layers { start, end } => {
                if start >= end {
                    return Err(SchemaError::WorkKindLayersInverted {
                        start: *start,
                        end: *end,
                    });
                }
                Ok(())
            }
            WorkKind::Shard { .. } => Ok(()),
            WorkKind::Custom { label } => {
                if label.is_empty() {
                    return Err(SchemaError::WorkKindCustomEmptyLabel);
                }
                if label.len() > WORK_KIND_CUSTOM_LABEL_MAX {
                    return Err(SchemaError::WorkKindCustomLabelTooLong {
                        got: label.len(),
                        max: WORK_KIND_CUSTOM_LABEL_MAX,
                    });
                }
                if !label
                    .as_bytes()
                    .iter()
                    .all(|b| (0x20..=0x7E).contains(b))
                {
                    return Err(SchemaError::WorkKindCustomLabelNotPrintableAscii);
                }
                Ok(())
            }
        }
    }
}

// ── ExecutionSession ──────────────────────────────────────────────────────

/// Signed session envelope opened by a coordinator for one posted job.
/// `session_id` is the lower-hex BLAKE3 of the canonical session body
/// (the same domain-separated bincode bytes the coordinator signs),
/// excluding `session_id` itself and `coordinator_signature_hex`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionSession {
    pub schema_version: u32,

    /// 64-char lowercase hex — derived from `session_id_hex(self)`.
    pub session_id: String,

    /// 64-char lowercase hex — copy of the `PostedJob.posted_id` this
    /// session executes. Drift-guarded by verifiers.
    pub posted_id: String,

    /// 64-char lowercase hex — copy of the inner `ContributorJob.job_hash`.
    pub job_hash: String,

    /// 64-char lowercase hex — copy of `ContributorJob.model_hash`.
    pub model_hash: String,

    /// 64-char lowercase hex — copy of
    /// `ContributorJob.accounting.tokenizer_hash`. Optional because
    /// some posted jobs may not yet record it.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tokenizer_hash: Option<String>,

    /// 64-char lowercase hex Ed25519 public key.
    pub coordinator_pubkey_hex: String,

    /// RFC 3339 UTC (`Z` suffix).
    pub created_at_utc: String,

    /// RFC 3339 UTC (`Z` suffix). Required — sessions are bounded.
    pub expires_at_utc: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// session body.
    pub coordinator_signature_hex: String,
}

impl ExecutionSession {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != SESSION_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("posted_id", &self.posted_id)?;
        check_blake3_hex("job_hash", &self.job_hash)?;
        check_blake3_hex("model_hash", &self.model_hash)?;
        if let Some(ref t) = self.tokenizer_hash {
            check_blake3_hex("tokenizer_hash", t)?;
        }
        check_pubkey_hex("coordinator_pubkey_hex", &self.coordinator_pubkey_hex)?;
        check_iso_8601("created_at_utc", &self.created_at_utc)?;
        check_iso_8601("expires_at_utc", &self.expires_at_utc)?;
        check_signature_hex("coordinator_signature_hex", &self.coordinator_signature_hex)?;
        Ok(())
    }
}

// ── ContributorJoin ───────────────────────────────────────────────────────

/// Signed join envelope: contributor advertises their capability +
/// RAM hints for a specific session. Joining is voluntary; no
/// exclusive claim, no commitment to be assigned work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContributorJoin {
    pub schema_version: u32,

    /// 64-char lowercase hex — copy of `ExecutionSession.session_id`.
    pub session_id: String,

    /// 64-char lowercase hex Ed25519 public key.
    pub contributor_pubkey_hex: String,

    /// Advertised free RAM at join time. Hint only; the protocol does
    /// not (and cannot) verify this.
    pub available_ram_bytes: u64,

    /// Contributor's per-job input-token ceiling. Coordinators picking
    /// assignments should respect this.
    pub max_input_tokens: u64,

    /// Contributor's per-job output-token ceiling.
    pub max_output_tokens: u64,

    /// Closed enum from 12.0. Non-empty — a contributor must support
    /// at least one work-unit kind to be useful.
    pub supported_work_unit_kinds: Vec<WorkUnitKind>,

    /// Free-form runner kind hint, e.g. `"stub"`, `"external"`,
    /// `"vllm-shim"`. Non-empty printable ASCII; bounded length.
    pub runner_kind: String,

    /// RFC 3339 UTC (`Z` suffix).
    pub joined_at_utc: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// join body.
    pub contributor_signature_hex: String,
}

/// Same printable-ASCII + length bound as [`WORK_KIND_CUSTOM_LABEL_MAX`]
/// — keeps gossip messages tight and forbids control chars.
pub const RUNNER_KIND_MAX: usize = 64;

impl ContributorJoin {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != SESSION_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("session_id", &self.session_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        if self.supported_work_unit_kinds.is_empty() {
            return Err(SchemaError::EmptySupportedWorkUnitKinds);
        }
        if self.runner_kind.is_empty() {
            return Err(SchemaError::EmptyRunnerKind);
        }
        if self.runner_kind.len() > RUNNER_KIND_MAX {
            return Err(SchemaError::RunnerKindTooLong {
                got: self.runner_kind.len(),
                max: RUNNER_KIND_MAX,
            });
        }
        if !self
            .runner_kind
            .as_bytes()
            .iter()
            .all(|b| (0x20..=0x7E).contains(b))
        {
            return Err(SchemaError::RunnerKindNotPrintableAscii);
        }
        check_iso_8601("joined_at_utc", &self.joined_at_utc)?;
        check_signature_hex("contributor_signature_hex", &self.contributor_signature_hex)?;
        Ok(())
    }
}

// ── WorkAssignment ────────────────────────────────────────────────────────

/// Signed assignment envelope: coordinator assigns one stage of work
/// to a joined contributor. The assignment carries no
/// `coordinator_pubkey_hex` — the coordinator is identified by the
/// session it references, and verifiers must check the assignment
/// signature against `session.coordinator_pubkey_hex`.
///
/// Assignments are cooperation hints within a session, **not**
/// exclusive claims across the network. Other sessions for the same
/// posted_id may assign overlapping work to different contributors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkAssignment {
    pub schema_version: u32,

    /// 64-char lowercase hex — copy of `ExecutionSession.session_id`.
    pub session_id: String,

    /// 64-char lowercase hex — derived from `assignment_id_hex(self)`.
    pub assignment_id: String,

    /// Zero-based pipeline order. The contributor that produces the
    /// user-visible final response sits at the highest `stage_index`.
    pub stage_index: u32,

    /// 64-char lowercase hex Ed25519 public key — must match a
    /// `ContributorJoin.contributor_pubkey_hex` in the same session.
    pub contributor_pubkey_hex: String,

    pub work_kind: WorkKind,

    /// Coordinator's pre-declared expected work-unit count for this
    /// assignment (in `expected_work_unit_kind` units). Partial
    /// results are bounded by this.
    pub expected_work_units: u64,

    pub expected_work_unit_kind: WorkUnitKind,

    /// RFC 3339 UTC (`Z` suffix).
    pub assigned_at_utc: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// assignment body. Verified against the *session's*
    /// `coordinator_pubkey_hex`.
    pub coordinator_signature_hex: String,
}

impl WorkAssignment {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != SESSION_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("assignment_id", &self.assignment_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        self.work_kind.validate_schema()?;
        if self.expected_work_units == 0 {
            return Err(SchemaError::AssignmentZeroExpectedWorkUnits);
        }
        check_iso_8601("assigned_at_utc", &self.assigned_at_utc)?;
        check_signature_hex("coordinator_signature_hex", &self.coordinator_signature_hex)?;
        Ok(())
    }
}

// ── PartialContributorResult ──────────────────────────────────────────────

/// Signed partial result published by a contributor for one
/// assignment. Carries a SNIP root pointing at whatever artifact
/// bytes the contributor's runner emitted (hidden-state activations,
/// next-stage prefill input, intermediate logits — opaque to this
/// envelope).
///
/// `measured_accounting.stage_contributions` must contain exactly one
/// entry whose `contributor_pubkey_hex` equals this partial's
/// `contributor_pubkey_hex`. The protocol does NOT sum partial
/// totals into the final job-level `total_base_units` — see
/// crate-level docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialContributorResult {
    pub schema_version: u32,

    /// 64-char lowercase hex.
    pub session_id: String,

    /// 64-char lowercase hex — copy of `WorkAssignment.assignment_id`.
    pub assignment_id: String,

    /// 64-char lowercase hex Ed25519 public key — must match the
    /// referenced assignment's `contributor_pubkey_hex`.
    pub contributor_pubkey_hex: String,

    /// SNIP V2 Merkle root of the partial artifact bytes (`0x`-prefixed
    /// lowercase hex, 66 chars).
    pub partial_artifact_snip_root: String,

    /// 64-char lowercase hex BLAKE3 of the partial artifact bytes.
    pub partial_artifact_hash: String,

    /// Per-stage accounting. Exactly one stage_contributions entry.
    pub measured_accounting: MeasuredAccounting,

    /// RFC 3339 UTC (`Z` suffix).
    pub produced_at_utc: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// partial body.
    pub contributor_signature_hex: String,
}

impl PartialContributorResult {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != SESSION_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("assignment_id", &self.assignment_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_snip_root_hex("partial_artifact_snip_root", &self.partial_artifact_snip_root)?;
        check_blake3_hex("partial_artifact_hash", &self.partial_artifact_hash)?;
        self.measured_accounting.validate_schema()?;
        if self.measured_accounting.stage_contributions.len() != 1 {
            return Err(SchemaError::PartialMustHaveOneStageContribution {
                got: self.measured_accounting.stage_contributions.len(),
            });
        }
        let sc = &self.measured_accounting.stage_contributions[0];
        if sc.contributor_pubkey_hex != self.contributor_pubkey_hex {
            return Err(SchemaError::PartialStageContributorMismatch);
        }
        check_iso_8601("produced_at_utc", &self.produced_at_utc)?;
        check_signature_hex("contributor_signature_hex", &self.contributor_signature_hex)?;
        Ok(())
    }
}

// ── AggregatedContributorResult ───────────────────────────────────────────

/// Reference to one partial result, copied into the aggregate so the
/// aggregate's canonical bytes bind to a frozen view of the chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregatedPartialRef {
    /// 64-char lowercase hex.
    pub assignment_id: String,

    pub stage_index: u32,

    /// 64-char lowercase hex Ed25519 public key (drift-guarded copy).
    pub contributor_pubkey_hex: String,

    /// SNIP V2 Merkle root of the partial envelope bytes.
    pub partial_snip_root: String,

    /// 64-char lowercase hex BLAKE3 of canonical partial bytes
    /// (signature-domain hash).
    pub partial_canonical_hash: String,
}

impl AggregatedPartialRef {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        check_blake3_hex("assignment_id", &self.assignment_id)?;
        check_pubkey_hex("contributor_pubkey_hex", &self.contributor_pubkey_hex)?;
        check_snip_root_hex("partial_snip_root", &self.partial_snip_root)?;
        check_blake3_hex("partial_canonical_hash", &self.partial_canonical_hash)?;
        Ok(())
    }
}

/// Signed aggregate envelope produced by the coordinator after every
/// assignment in the session has exactly one valid partial. The
/// `final_result_snip_root` points at a normal (Stage 12.0 v1)
/// `ContributorResult` JSON; the existing `verify_result` pipeline
/// runs unchanged against it.
///
/// Stage 12.3 verification additionally checks structural integrity:
/// every assignment has a partial here; every partial signature is
/// valid; the chain references resolve via SNIP; and the coordinator's
/// signature over this aggregate verifies against the session's
/// `coordinator_pubkey_hex`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregatedContributorResult {
    pub schema_version: u32,

    /// 64-char lowercase hex.
    pub session_id: String,

    /// 64-char lowercase hex — drift-guarded against
    /// `ExecutionSession.posted_id`.
    pub posted_id: String,

    /// SNIP V2 Merkle root of the standalone final `ContributorResult`
    /// JSON.
    pub final_result_snip_root: String,

    /// 64-char lowercase hex BLAKE3 of canonical (signature-domain)
    /// `ContributorResult` bytes.
    pub final_result_canonical_hash: String,

    /// Ordered by ascending `stage_index`. Must cover every assignment
    /// in the session exactly once (verified separately).
    pub partial_refs: Vec<AggregatedPartialRef>,

    /// RFC 3339 UTC (`Z` suffix).
    pub aggregated_at_utc: String,

    /// 64-char lowercase hex Ed25519 public key — must equal the
    /// session's `coordinator_pubkey_hex`.
    pub coordinator_pubkey_hex: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// aggregate body.
    pub coordinator_signature_hex: String,
}

impl AggregatedContributorResult {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != SESSION_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("posted_id", &self.posted_id)?;
        check_snip_root_hex("final_result_snip_root", &self.final_result_snip_root)?;
        check_blake3_hex("final_result_canonical_hash", &self.final_result_canonical_hash)?;
        if self.partial_refs.is_empty() {
            return Err(SchemaError::AggregateEmptyPartialRefs);
        }
        for (i, p) in self.partial_refs.iter().enumerate() {
            p.validate_schema()
                .map_err(|e| SchemaError::AggregatePartialRefInvalid {
                    index: i,
                    inner: Box::new(e),
                })?;
        }
        check_iso_8601("aggregated_at_utc", &self.aggregated_at_utc)?;
        check_pubkey_hex("coordinator_pubkey_hex", &self.coordinator_pubkey_hex)?;
        check_signature_hex("coordinator_signature_hex", &self.coordinator_signature_hex)?;
        Ok(())
    }
}
