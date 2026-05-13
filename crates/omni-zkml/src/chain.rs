//! Phase 5 Stage 5 — chain client abstraction.
//!
//! Defines a synchronous trait that future chain implementations
//! (a real SUM Chain adapter, or any other chain target) implement to
//! submit and query attestations. **No real RPC, no tx encoding.**
//!
//! Strings on the wire (`tx_id`, `Failed::reason`, `Dropped::reason`) are
//! opaque. Chain-specific encoding is pending and Stage 5 makes no
//! commitment to hex vs base58 vs bech32, or to EVM/ABI receipt shapes.
//!
//! The trait is **sync** for now: the rest of `omni-zkml` is sync, and no
//! real chain client exists yet. A future async chain implementation can
//! wrap its async work with `tokio::runtime::Handle::block_on` /
//! `block_in_place`, or we can introduce an async variant later.

use serde::{Deserialize, Serialize};

use omni_types::phase5::InferenceAttestation;

use crate::error::ChainClientError;
use crate::registry::AttestationId;

// ── Chain client trait ───────────────────────────────────────────────────────

pub trait ChainClient {
    /// Submit an attestation. Returns an implementation-specific receipt
    /// on success.
    fn submit_attestation(
        &self,
        attestation: &InferenceAttestation,
    ) -> std::result::Result<SubmissionReceipt, ChainClientError>;

    /// Query the chain-side status of a previously-submitted attestation.
    fn query_attestation_status(
        &self,
        id: &AttestationId,
    ) -> std::result::Result<AttestationStatus, ChainClientError>;
}

// ── Placeholder receipt & status types ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionReceipt {
    /// Opaque chain-side identifier (tx hash, request id, …). Format is
    /// implementation-defined; chain encoding is pending.
    pub tx_id: String,

    /// Optional implementation-specific diagnostic. Stage-5 fake clients
    /// use this for test assertions; a real chain client may leave it
    /// `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Chain-side status reported by [`ChainClient::query_attestation_status`].
///
/// There is intentionally no `Pending` variant — from the chain's
/// perspective, the moment a query succeeds the submission has at least
/// reached `Submitted`. `Pending` is local-only (it means "we built a
/// record but haven't submitted yet").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationStatus {
    Submitted,
    Included,
    Finalized,
    Failed { reason: String },
    Dropped { reason: Option<String> },
}
