//! Phase 5 Stage 5 — chain client abstraction (with Stage-5.1 alignment).
//!
//! Defines a synchronous trait that future chain implementations
//! (a real SUM Chain adapter, or any other chain target) implement to
//! submit and query attestations. **No real RPC, no tx encoding.**
//!
//! Strings on the wire (`tx_id`, `Failed::reason`) are opaque. Chain-
//! specific encoding is pending and Stage 5 makes no commitment to hex
//! vs base58 vs bech32, or to EVM/ABI receipt shapes.
//!
//! The trait is **sync** for now: the rest of `omni-zkml` is sync, and no
//! real chain client exists yet. A future async chain implementation can
//! wrap its async work with `tokio::runtime::Handle::block_on` /
//! `block_in_place`, or we can introduce an async variant later.
//!
//! ### Stage 5.1 — SUM Chain v1 status alignment
//!
//! - [`ChainClient::query_attestation_status`] is keyed by `tx_id: &str`,
//!   matching the chain RPC `sum_getInferenceAttestationStatus(tx_hash)`.
//!   The registry workflow looks up the stored `SubmissionReceipt::tx_id`
//!   before calling this method.
//! - [`AttestationStatus`] reflects the chain v1 five-state enum:
//!   `Submitted | Included | Finalized | Failed { reason } | Unknown`.
//!   There is no chain-side `Dropped`; `Unknown` covers mempool eviction
//!   and unrecognized tx hashes.

use serde::{Deserialize, Serialize};

use omni_types::phase5::InferenceAttestation;

use crate::error::ChainClientError;

// ── Chain client trait ───────────────────────────────────────────────────────

pub trait ChainClient {
    /// Submit an attestation. Returns an implementation-specific receipt
    /// on success.
    fn submit_attestation(
        &self,
        attestation: &InferenceAttestation,
    ) -> std::result::Result<SubmissionReceipt, ChainClientError>;

    /// Query the chain-side status of a previously-submitted attestation.
    ///
    /// **Keyed by `tx_id`**, matching the SUM Chain v1 RPC
    /// `sum_getInferenceAttestationStatus(tx_hash)`. Callers that hold a
    /// local registry [`crate::registry::AttestationId`] (e.g.
    /// [`crate::registry::query_attestation_workflow`]) must look up the
    /// stored [`SubmissionReceipt::tx_id`] on the local record before
    /// invoking this method.
    fn query_attestation_status(
        &self,
        tx_id: &str,
    ) -> std::result::Result<AttestationStatus, ChainClientError>;
}

// ── Placeholder receipt & status types ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionReceipt {
    /// Opaque chain-side identifier (tx hash, request id, …). Format is
    /// implementation-defined; chain encoding is pending. This is the
    /// value passed to [`ChainClient::query_attestation_status`] on
    /// subsequent reads.
    pub tx_id: String,

    /// Optional implementation-specific diagnostic. Stage-5 fake clients
    /// use this for test assertions; a real chain client may leave it
    /// `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Chain-side status reported by [`ChainClient::query_attestation_status`].
///
/// Matches the SUM Chain v1 five-state model. There is intentionally no
/// `Pending` variant — from the chain's perspective, the moment a query
/// succeeds the submission has at least reached `Submitted`. `Pending` is
/// local-only (it means "we built a record but haven't submitted yet").
///
/// There is also no chain-side `Dropped` — chain v1 does not track tx
/// drops directly. Mempool eviction and unrecognized tx hashes both
/// surface as [`AttestationStatus::Unknown`]; clients infer drops from
/// `Unknown` + age via local staleness detection (Stage 5.2, not part of
/// Stage 5.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationStatus {
    Submitted,
    Included,
    Finalized,
    Failed { reason: String },
    /// Chain reports it does not recognize this `tx_hash`. Could mean
    /// mempool eviction, a never-seen tx, or chain-side lag. Carries no
    /// payload — the chain has no diagnostic "reason" for an unknown tx.
    /// Clients MUST NOT treat `Unknown` as terminal; staleness handling
    /// is a separate client-side concern (Stage 5.2).
    Unknown,
}
