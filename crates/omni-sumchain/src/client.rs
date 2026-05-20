//! Phase 5 Stage 7a + 7b — `SumChainClient`, the SUM Chain adapter.
//!
//! Implements the [`ChainClient`] trait from `omni-zkml` for SUM
//! Chain's JSON-RPC surface. Both the read/query surface (Stage 7a)
//! and the real `submit_attestation` flow (Stage 7b) are live; the
//! submit path delegates to [`crate::tx::build_and_submit_signed_transaction`]
//! against vendored chain primitives at rev `d83e45a4`.
//!
//! `SumChainClient` is generic over a [`JsonRpcTransport`] so default
//! `cargo test` runs fully hermetically against
//! [`FakeJsonRpcTransport`]. The type alias `SumChainClient` (no
//! explicit parameter) defaults to [`UreqTransport`] for ergonomic
//! production construction via [`SumChainClient::new`].

use omni_types::phase5::InferenceAttestation;
use omni_zkml::{
    AttestationStatus, ChainClient, ChainClientError, OrchestrationClient, SubmissionReceipt,
};

use crate::dto::{
    BlockFinality, BlockHeightInfo, ChainParamsInfo, InferenceAttestationInfo,
    InferenceAttestationStatusInfo,
};
use crate::rpc::{JsonRpcTransport, UreqTransport};
use crate::status::map_status_info;
#[cfg(feature = "submit")]
use crate::tx::build_and_submit_signed_transaction;

// ── SumChainClient ────────────────────────────────────────────────────────────

/// Adapter against a single SUM Chain JSON-RPC endpoint. Generic over a
/// [`JsonRpcTransport`] so tests can substitute
/// [`crate::FakeJsonRpcTransport`] without touching the network.
///
/// Implements both the Stage 7a read surface and the Stage 7b real
/// submit flow. `submit_attestation` enforces four pre-flight gates
/// (OmniNode activation, V2 activation, verifier-address consistency)
/// before any state-mutating RPC, delegates the construction to
/// [`crate::tx::build_and_submit_signed_transaction`], and posts the
/// resulting `SignedTransaction` via `sum_sendRawTransaction`.
pub struct SumChainClient<T: JsonRpcTransport = UreqTransport> {
    seed: [u8; 32],
    transport: T,
}

impl SumChainClient<UreqTransport> {
    /// Build a production client with the default [`UreqTransport`]
    /// against `rpc_url`. The `seed` is the Ed25519 seed used for
    /// Stage 6 inner-digest signing and Stage 7b outer-tx signing; it
    /// must derive (via `omni_zkml::signer_chain_address_base58`) to
    /// the same chain address embedded in the attestations submitted
    /// through this client, or the verifier-address gate refuses to
    /// submit.
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

    /// Borrow the configured Ed25519 seed (32 bytes). Used by Stage 7b
    /// for both the Stage 6 inner-digest signing call
    /// (`sign_chain_attestation_digest`) and the outer
    /// `TransactionV2::signing_hash()` Ed25519 sign.
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
    /// should use when submitting a transaction. Stage 7b's submit
    /// flow calls this once, after all four pre-flight gates pass.
    /// Exposed as an inherent helper so operators / live tests can
    /// inspect the nonce of a funded account independently.
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
        self.activation_satisfied(params.omninode_enabled_from_height)
    }

    /// `Ok(true)` only when the chain has activated the V2 transaction
    /// envelope AND has progressed past `v2_enabled_from_height`.
    ///
    /// Stage 7b addition. Symmetric to [`Self::omninode_is_active`].
    /// `submit_attestation` requires **both** active before transmitting
    /// — V2 activation gates the outer-tx format itself, OmniNode
    /// activation gates the `TxPayload::InferenceAttestation` variant.
    pub fn v2_is_active(
        &self,
    ) -> std::result::Result<bool, ChainClientError> {
        let params = self.get_chain_params()?;
        self.activation_satisfied(params.v2_enabled_from_height)
    }

    /// Stage 7b helper: given an `Option<u64>` activation height from
    /// `chain_getChainParams`, return `Ok(true)` iff it is `Some(h)` and
    /// the chain's latest head is `>= h`. `None` (field absent / pre-
    /// patch) maps to `Ok(false)`; no `chain_getBlockHeight` call is
    /// made in that case.
    pub(crate) fn activation_satisfied(
        &self,
        activation: Option<u64>,
    ) -> std::result::Result<bool, ChainClientError> {
        match activation {
            None => Ok(false),
            Some(h) => {
                let head = self.get_block_height(BlockFinality::Latest)?.height;
                Ok(head >= h)
            }
        }
    }

    /// Derive the chain address from `self.seed` using Stage 6's helper.
    /// Stage 7b uses this both for the `sender == verifier` consistency
    /// check inside `submit_attestation` and for the `sum_getNonce`
    /// lookup parameter.
    pub fn derived_verifier_address(
        &self,
    ) -> std::result::Result<String, ChainClientError> {
        omni_zkml::signer_chain_address_base58(&self.seed).map_err(|e| {
            ChainClientError::Other(format!("seed → address derivation failed: {e}"))
        })
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

    /// Stage 7b — real implementation, gated behind the `submit`
    /// feature (Stage 9a). With `submit`, delegates to
    /// [`crate::tx::build_and_submit_signed_transaction`] (four
    /// pre-flight gates → Stage 6 inner pipeline → vendored chain
    /// types → outer-sign via `sumchain-crypto` →
    /// `sum_sendRawTransaction`). Without `submit`, returns a typed
    /// `ChainClientError::Other` so the `ChainClient` trait stays
    /// satisfied for the read-only operator surface (`query`,
    /// `poll_attestations_workflow`, etc.) while default builds need
    /// no access to the private `SUM-INNOVATION/sum-chain` repo.
    ///
    /// Stage 5.1 contract preserved either way: any error returned
    /// here surfaces through `submit_attestation_workflow` as
    /// `RegistryError::ChainClient(_)` and leaves the local record at
    /// its pre-submit state.
    fn submit_attestation(
        &self,
        attestation: &InferenceAttestation,
    ) -> std::result::Result<SubmissionReceipt, ChainClientError> {
        #[cfg(feature = "submit")]
        {
            build_and_submit_signed_transaction(self, attestation)
        }
        #[cfg(not(feature = "submit"))]
        {
            let _ = attestation;
            Err(ChainClientError::Other(
                "omni-sumchain built without the `submit` feature; \
                 rebuild with --features submit to enable \
                 sum_sendRawTransaction"
                    .into(),
            ))
        }
    }
}

// ── OrchestrationClient trait impl (Stage 5.3) ───────────────────────────────

impl<T: JsonRpcTransport> OrchestrationClient for SumChainClient<T> {
    /// Stage 5.3 surface for `omni-zkml::orchestration`. Delegates to
    /// the existing inherent helper `get_block_height(BlockFinality::Latest)`
    /// and returns the `.height` field. `Latest` (not `Finalized`) is
    /// the natural finality token for staleness and block-aware
    /// submit — `Finalized` lags inclusion and would over-aggressively
    /// declare records stale.
    fn get_latest_block_height(&self) -> std::result::Result<u64, ChainClientError> {
        self.get_block_height(BlockFinality::Latest).map(|h| h.height)
    }
}
