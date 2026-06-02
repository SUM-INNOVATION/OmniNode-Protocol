//! Stage 12.11 — signed `WorkAssignmentSupersession` envelope.
//!
//! A coordinator-signed declaration that one or more existing
//! `WorkAssignment`s are superseded by one or more replacement
//! `WorkAssignment`s. The aggregate verifier
//! ([`crate::session_verify::verify_aggregated_result_with_supersessions`])
//! uses this evidence to ignore the superseded set in its coverage
//! check; without a signed supersession, the original Stage 12.3
//! "every assignment must be covered exactly once" rule still
//! applies (Stage 12.10 halt-finding).
//!
//! ## v1 scope: replacement-only
//!
//! `replacement_assignment_ids` must be **non-empty** in v1.
//! Stage 12.11 is reassignment/supersession, not abandonment or
//! cancellation. If the operator wants to remove a stage entirely
//! from a session, that requires a partial-aggregate /
//! cancellation envelope which is explicitly out of scope here.
//!
//! ## What this envelope IS
//!
//! - A signed, schema-versioned, SNIP-publishable artifact that
//!   declares: "as the coordinator of this session, I have
//!   reassigned the following work."
//! - The trust anchor for the aggregate verifier's "ignore these
//!   superseded assignments" coverage relaxation.
//!
//! ## What this envelope is NOT
//!
//! - **NOT a cancellation declaration.** v1 forbids empty
//!   `replacement_assignment_ids`.
//! - **NOT a penalty / slashing artifact.** The `InvalidPartial`
//!   reason variant is the coordinator's local verdict, not a
//!   chain-anchored proof of misconduct.
//! - **NOT a chain wire envelope.** No transaction, no on-chain
//!   anchor.
//! - **NOT a state mutation primitive.** The verifier only reads
//!   bodies.
//!
//! ## Canonical body
//!
//! Domain separator
//! `OMNINODE-CONTRIBUTOR-SESSION-SUPERSESSION:v1:`. The canonical
//! signing input excludes both `supersession_id` and
//! `coordinator_signature_hex`, mirroring every other Stage 12.x
//! signing pattern. The `supersession_id` is the lowercase hex
//! BLAKE3 of the canonical body bytes.

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;
use crate::job::{check_blake3_hex, check_iso_8601, check_pubkey_hex, check_signature_hex};

/// Pinned v1. Adding e.g. partial-aggregate semantics is a
/// `schema_version: 2` migration.
pub const SUPERSESSION_SCHEMA_VERSION: u32 = 1;

/// Same printable-ASCII + length bound as `WORK_KIND_CUSTOM_LABEL_MAX`.
pub const SUPERSESSION_REASON_CUSTOM_LABEL_MAX: usize = 64;

// ── SupersessionReason ──────────────────────────────────────────

/// Closed enum (with a `Custom` forward-compat escape hatch)
/// describing WHY the coordinator superseded the work.
///
/// `Custom` exists because v1 cannot enumerate every operator
/// rationale at freezing time. Same shape rules as
/// `WorkKind::Custom`: non-empty printable ASCII, bounded length.
///
/// Adding a non-`Custom` variant is a `schema_version: 2`
/// migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SupersessionReason {
    /// Contributor never delivered a valid partial within the
    /// operator's review window.
    MissingPartial,
    /// Contributor delivered a partial that failed verification
    /// (tampered signature, bad shape, drift).
    InvalidPartial,
    /// Operator-initiated rebalance (e.g. peer churn, RAM
    /// constraint discovered late).
    OperatorRebalance,
    /// Forward-compat escape hatch.
    Custom { label: String },
}

impl SupersessionReason {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        match self {
            SupersessionReason::MissingPartial
            | SupersessionReason::InvalidPartial
            | SupersessionReason::OperatorRebalance => Ok(()),
            SupersessionReason::Custom { label } => {
                if label.is_empty() {
                    return Err(SchemaError::SupersessionReasonCustomEmptyLabel);
                }
                if label.len() > SUPERSESSION_REASON_CUSTOM_LABEL_MAX {
                    return Err(SchemaError::SupersessionReasonCustomLabelTooLong {
                        got: label.len(),
                        max: SUPERSESSION_REASON_CUSTOM_LABEL_MAX,
                    });
                }
                if !label
                    .as_bytes()
                    .iter()
                    .all(|b| (0x20..=0x7E).contains(b))
                {
                    return Err(
                        SchemaError::SupersessionReasonCustomLabelNotPrintableAscii,
                    );
                }
                Ok(())
            }
        }
    }
}

// ── WorkAssignmentSupersession ─────────────────────────────────

/// Signed envelope. Published to SNIP; pointer announced on the
/// new `omni/contributor/session/assignment-supersession/v1`
/// gossipsub topic.
///
/// Verifier rules (see
/// [`crate::session_verify::verify_assignment_supersession`]):
///
/// 1. `schema_version == 1`.
/// 2. `session_id == session.session_id`.
/// 3. `coordinator_pubkey_hex == session.coordinator_pubkey_hex`.
/// 4. `supersession_id` equals the recomputed canonical hash.
/// 5. `coordinator_signature_hex` verifies over the canonical
///    signing input.
/// 6. `superseded_assignment_ids ⊆ assignments`.
/// 7. `replacement_assignment_ids ⊆ assignments`.
/// 8. `superseded_assignment_ids ∩ replacement_assignment_ids == ∅`.
/// 9. `superseded_assignment_ids` non-empty.
/// 10. `replacement_assignment_ids` non-empty (v1 replacement-only
///     scope; abandonment deferred).
///
/// Cross-supersession invariants are checked at aggregate-verifier
/// time: no `assignment_id` appears in `superseded_assignment_ids`
/// of more than one supersession in the supplied slice (no
/// double-supersession). Chains via sequential supersessions
/// (`A→B`, then `B→C`) ARE permitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkAssignmentSupersession {
    pub schema_version: u32,

    /// 64-char lowercase hex — copy of `ExecutionSession.session_id`.
    pub session_id: String,

    /// 64-char lowercase hex — derived from `supersession_id_hex(self)`.
    pub supersession_id: String,

    /// Non-empty, sorted ascending by hex value, no duplicates.
    /// All entries must be 64-char lowercase hex `assignment_id`s
    /// in the same session.
    pub superseded_assignment_ids: Vec<String>,

    /// **Non-empty** in v1 (replacement-only scope). Sorted
    /// ascending by hex value, no duplicates. All entries must be
    /// 64-char lowercase hex `assignment_id`s in the same session.
    /// Must be disjoint from `superseded_assignment_ids`.
    pub replacement_assignment_ids: Vec<String>,

    pub reason: SupersessionReason,

    /// RFC 3339 UTC (`Z` suffix). Supersessions inherit their
    /// session's `expires_at_utc`; no separate supersession
    /// expiry. Pruned with the session when its
    /// `expires_at_utc` passes.
    pub created_at_utc: String,

    /// 64-char lowercase hex Ed25519 public key. Must equal
    /// `session.coordinator_pubkey_hex` at verify time.
    pub coordinator_pubkey_hex: String,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// signing body (which excludes `supersession_id` and this
    /// field).
    pub coordinator_signature_hex: String,
}

impl WorkAssignmentSupersession {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != SUPERSESSION_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("supersession_id", &self.supersession_id)?;
        check_pubkey_hex(
            "coordinator_pubkey_hex",
            &self.coordinator_pubkey_hex,
        )?;
        check_signature_hex(
            "coordinator_signature_hex",
            &self.coordinator_signature_hex,
        )?;
        check_iso_8601("created_at_utc", &self.created_at_utc)?;

        // Cardinality + duplicate + ordering checks.
        if self.superseded_assignment_ids.is_empty() {
            return Err(SchemaError::SupersessionEmptySuperseded);
        }
        if self.replacement_assignment_ids.is_empty() {
            // v1 replacement-only scope. Stage 12.11 review
            // hardened this: abandonment requires a separate
            // partial-aggregate / cancellation envelope that does
            // not exist in v1.
            return Err(SchemaError::SupersessionEmptyReplacement);
        }
        check_ids_sorted_unique_hex(
            "superseded_assignment_ids",
            &self.superseded_assignment_ids,
        )?;
        check_ids_sorted_unique_hex(
            "replacement_assignment_ids",
            &self.replacement_assignment_ids,
        )?;

        // Disjointness check. Both lists are individually sorted,
        // so the merge-style intersection is O(n + m) — but n + m
        // ≤ a session's full assignment count, which is small in
        // practice. Use a HashSet for clarity.
        let superseded_set: std::collections::HashSet<&String> =
            self.superseded_assignment_ids.iter().collect();
        for r in &self.replacement_assignment_ids {
            if superseded_set.contains(r) {
                return Err(SchemaError::SupersessionSupersededAndReplacement {
                    assignment_id: r.clone(),
                });
            }
        }

        self.reason.validate_schema()?;
        Ok(())
    }
}

fn check_ids_sorted_unique_hex(
    field: &'static str,
    ids: &[String],
) -> Result<(), SchemaError> {
    let mut prev: Option<&str> = None;
    for id in ids {
        check_blake3_hex(field, id)?;
        if let Some(p) = prev {
            match id.as_str().cmp(p) {
                std::cmp::Ordering::Greater => {}
                std::cmp::Ordering::Equal => {
                    return Err(SchemaError::SupersessionDuplicateId {
                        field,
                        assignment_id: id.clone(),
                    });
                }
                std::cmp::Ordering::Less => {
                    return Err(SchemaError::SupersessionIdsNotSorted { field });
                }
            }
        }
        prev = Some(id.as_str());
    }
    Ok(())
}
