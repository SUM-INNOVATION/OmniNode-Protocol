//! Phase 5 Stage 7a — chain JSON status payload → `AttestationStatus` mapping.
//!
//! Strict matching against the chain's five lowercase status variants.
//! Unknown variants are rejected with `ChainClientError::Other(_)` —
//! treating an unrecognized status as `Unknown` would mask real
//! protocol drift.

use omni_zkml::{AttestationStatus, ChainClientError};

use crate::dto::InferenceAttestationStatusInfo;

/// Convert a chain `InferenceAttestationStatusInfo` into the
/// `omni-zkml` `AttestationStatus` enum that the Stage 5 query workflow
/// consumes.
///
/// Stage 5.1 contracts preserved here:
/// - `Unknown` maps cleanly and is **non-terminal**; the workflow leaves
///   the record unchanged and logs a warning.
/// - The chain does **not** return `Dropped`. Local `Dropped` is set
///   only by Stage 5.2 staleness detection, never by this mapper.
pub fn map_status_info(
    info: InferenceAttestationStatusInfo,
) -> std::result::Result<AttestationStatus, ChainClientError> {
    match info.status.as_str() {
        "submitted" => Ok(AttestationStatus::Submitted),
        "included" => Ok(AttestationStatus::Included),
        "finalized" => Ok(AttestationStatus::Finalized),
        "failed" => {
            let reason = info.reason.ok_or_else(|| {
                ChainClientError::Other(
                    "chain returned status=\"failed\" without a `reason` \
                     field; chain-RPC contract requires reason for failed"
                        .into(),
                )
            })?;
            Ok(AttestationStatus::Failed { reason })
        }
        "unknown" => Ok(AttestationStatus::Unknown),
        other => Err(ChainClientError::Other(format!(
            "unexpected status variant from chain RPC: {other:?} \
             (expected lowercase: submitted|included|finalized|failed|unknown)"
        ))),
    }
}
