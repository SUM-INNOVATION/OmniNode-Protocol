//! Phase 5 Stage 7a — `SumChainClient`, the SUM Chain adapter.
//!
//! Implements the [`ChainClient`] trait from `omni-zkml` for SUM
//! Chain's JSON-RPC surface. Stage 7a is **read/query only** — the
//! `submit_attestation` trait method is a typed stub; Stage 7b will
//! replace its body once chain confirms the outer-tx primitive
//! dependency strategy.
//!
//! `SumChainClient` is generic over a [`JsonRpcTransport`] so default
//! `cargo test` runs fully hermetically against
//! [`FakeJsonRpcTransport`]. The type alias `SumChainClient` (no
//! explicit parameter) defaults to [`UreqTransport`] for ergonomic
//! production construction via [`SumChainClient::new`].

use omni_types::phase5::InferenceAttestation;
use omni_zkml::{
    AttestationStatus, ChainClient, ChainClientError, SubmissionReceipt,
};

use crate::dto::{
    BlockFinality, BlockHeightInfo, ChainParamsInfo, InferenceAttestationInfo,
    InferenceAttestationStatusInfo,
};
use crate::rpc::{JsonRpcTransport, UreqTransport};
use crate::status::map_status_info;
use crate::tx::build_and_submit_signed_transaction;

// ── SumChainClient ────────────────────────────────────────────────────────────

/// Adapter against a single SUM Chain JSON-RPC endpoint. Generic over a
/// [`JsonRpcTransport`] so tests can substitute
/// [`crate::FakeJsonRpcTransport`] without touching the network.
///
/// **Stage 7a is read/query only.** `submit_attestation` returns a
/// typed `ChainClientError::Other(_)` describing exactly why; do not
/// wire `SumChainClient` into a production submit workflow until
/// Stage 7b lands.
pub struct SumChainClient<T: JsonRpcTransport = UreqTransport> {
    seed: [u8; 32],
    transport: T,
}

impl SumChainClient<UreqTransport> {
    /// Build a production client with the default [`UreqTransport`]
    /// against `rpc_url`. The `seed` is an Ed25519 seed used for
    /// inner-digest signing (Stage 6) and, in Stage 7b, for outer-tx
    /// signing. Stage 7a does not invoke either signing path — the
    /// seed is held for forward-compat.
    pub fn new(rpc_url: String, seed: [u8; 32]) -> Self {
        Self::with_transport(seed, UreqTransport::new(rpc_url))
    }
}

impl<T: JsonRpcTransport> SumChainClient<T> {
    /// Construct with an explicit transport. Primary use case: tests
    /// inject a [`crate::FakeJsonRpcTransport`].
    pub fn with_transport(seed: [u8; 32], transport: T) -> Self {
        Self { seed, transport }
    }

    /// Borrow the underlying transport. Tests use this to inspect
    /// recorded calls on a fake.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Borrow the configured Ed25519 seed (32 bytes). Not used by
    /// Stage 7a but exposed for symmetry with Stage 7b's eventual
    /// outer-signing path.
    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    // ── Inherent read helpers ────────────────────────────────────────

    /// `sum_getInferenceAttestationStatus(tx_id)` returning the raw
    /// chain DTO. Used by Stage 5.2's planned staleness detection
    /// (which needs `included_at_height`). For Stage 5.1's
    /// `query_workflow`, the trait method's enum output is sufficient.
    pub fn query_attestation_status_full(
        &self,
        tx_id: &str,
    ) -> std::result::Result<InferenceAttestationStatusInfo, ChainClientError> {
        let result = self
            .transport
            .call("sum_getInferenceAttestationStatus", serde_json::json!([tx_id]))?;
        serde_json::from_value(result).map_err(|e| {
            ChainClientError::Other(format!(
                "failed to parse sum_getInferenceAttestationStatus response: {e}"
            ))
        })
    }

    /// `sum_getInferenceAttestation(session_id, verifier_address)`.
    /// Returns `Ok(None)` when the chain has no record under the pair
    /// (the chain emits JSON `null` for that case).
    pub fn get_attestation(
        &self,
        session_id: &str,
        verifier_address: &str,
    ) -> std::result::Result<Option<InferenceAttestationInfo>, ChainClientError> {
        let result = self.transport.call(
            "sum_getInferenceAttestation",
            serde_json::json!([session_id, verifier_address]),
        )?;
        if result.is_null() {
            return Ok(None);
        }
        serde_json::from_value(result).map(Some).map_err(|e| {
            ChainClientError::Other(format!(
                "failed to parse sum_getInferenceAttestation response: {e}"
            ))
        })
    }

    /// `sum_listInferenceAttestations(session_id)`.
    pub fn list_attestations(
        &self,
        session_id: &str,
    ) -> std::result::Result<Vec<InferenceAttestationInfo>, ChainClientError> {
        let result = self
            .transport
            .call("sum_listInferenceAttestations", serde_json::json!([session_id]))?;
        serde_json::from_value(result).map_err(|e| {
            ChainClientError::Other(format!(
                "failed to parse sum_listInferenceAttestations response: {e}"
            ))
        })
    }

    /// `chain_getChainParams`. Returns `finality_depth`, `min_fee`,
    /// `chain_id`, and (post-patch) `omninode_enabled_from_height`.
    /// For the chain id, call `params.chain_id` — there is no
    /// dedicated `get_chain_id` helper because the local mirror's id
    /// is stable across calls and callers may cache.
    pub fn get_chain_params(
        &self,
    ) -> std::result::Result<ChainParamsInfo, ChainClientError> {
        let result = self
            .transport
            .call("chain_getChainParams", serde_json::json!([]))?;
        serde_json::from_value(result).map_err(|e| {
            ChainClientError::Other(format!(
                "failed to parse chain_getChainParams response: {e}"
            ))
        })
    }

    /// `chain_getBlockHeight(finality_token)`.
    pub fn get_block_height(
        &self,
        finality: BlockFinality,
    ) -> std::result::Result<BlockHeightInfo, ChainClientError> {
        let result = self.transport.call(
            "chain_getBlockHeight",
            serde_json::json!([finality.as_token()]),
        )?;
        serde_json::from_value(result).map_err(|e| {
            ChainClientError::Other(format!(
                "failed to parse chain_getBlockHeight response: {e}"
            ))
        })
    }

    /// `sum_getNonce(address)` — returns the next nonce the address
    /// should use when submitting a transaction. Stage 7b will call
    /// this from `build_signed_transaction`. Stage 7a exposes it for
    /// operators / live tests to verify their funded account.
    pub fn get_nonce(
        &self,
        address: &str,
    ) -> std::result::Result<u64, ChainClientError> {
        let result = self
            .transport
            .call("sum_getNonce", serde_json::json!([address]))?;
        serde_json::from_value(result).map_err(|e| {
            ChainClientError::Other(format!(
                "failed to parse sum_getNonce response: {e}"
            ))
        })
    }

    /// `Ok(true)` only when the chain has activated OmniNode
    /// transactions AND has progressed past `omninode_enabled_from_height`.
    ///
    /// Pre-patch (when `omninode_enabled_from_height` is not exposed
    /// in `chain_getChainParams`), this method returns `Ok(false)`
    /// unconditionally because the parser deserialises the missing
    /// field as `None`. Callers MUST NOT treat that as proof of
    /// "chain disabled", only as proof of "OmniNode cannot yet
    /// confirm activation". Stage 7b's `submit_attestation` will
    /// re-check this and refuse to submit unless it returns
    /// `Ok(true)`.
    pub fn omninode_is_active(
        &self,
    ) -> std::result::Result<bool, ChainClientError> {
        let params = self.get_chain_params()?;
        match params.omninode_enabled_from_height {
            None => Ok(false),
            Some(activation) => {
                let head = self.get_block_height(BlockFinality::Latest)?.height;
                Ok(head >= activation)
            }
        }
    }
}

// ── ChainClient trait impl ───────────────────────────────────────────────────

impl<T: JsonRpcTransport> ChainClient for SumChainClient<T> {
    /// Stage 7a — implemented.
    ///
    /// Posts `sum_getInferenceAttestationStatus(tx_id)` and maps the
    /// chain's confirmed `InferenceAttestationStatusInfo` shape into
    /// the Stage-5 `AttestationStatus` enum via
    /// [`crate::status::map_status_info`].
    fn query_attestation_status(
        &self,
        tx_id: &str,
    ) -> std::result::Result<AttestationStatus, ChainClientError> {
        let info = self.query_attestation_status_full(tx_id)?;
        map_status_info(info)
    }

    /// Stage 7a — **typed stub.**
    ///
    /// Returns a `ChainClientError::Other(_)` with a clear message.
    /// Stage 7b will replace this body with the 9-step outer
    /// `SignedTransaction` construction once the chain primitive
    /// vendoring strategy is confirmed.
    ///
    /// The Stage 5.1 workflow already handles this gracefully: the
    /// error surfaces as `RegistryError::ChainClient(_)` and leaves
    /// the local record at `Pending`.
    fn submit_attestation(
        &self,
        attestation: &InferenceAttestation,
    ) -> std::result::Result<SubmissionReceipt, ChainClientError> {
        // Delegated so Stage 7b's diff localises to `tx.rs`. The
        // function currently returns the documented "unimplemented"
        // typed error; see the module rustdoc on `tx` for the
        // construction sequence that will replace it.
        build_and_submit_signed_transaction(&self.seed, attestation)
    }
}
