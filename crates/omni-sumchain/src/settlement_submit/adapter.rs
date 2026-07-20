//! Extension method on `SumChainClient<T>` for the settlement
//! builder RPC.
//!
//! Uses only `serde_json` on the wire — no chain-crypto or chain-
//! primitives coupling here. The chain-primitives coupling lives in
//! [`super::tx`], which decodes the returned `unsigned_tx` hex.

use crate::client::SumChainClient;
use crate::rpc::JsonRpcTransport;

use super::error::SettlementSubmitError;
use super::wire::{
    BuildClaimRewardRaw, BuildClaimRewardRequest, BuildRegisterVerifierRequest,
    SettlementBuilderEnvelope,
};

impl<T: JsonRpcTransport> SumChainClient<T> {
    /// Issue #87 — call `omninode_buildClaimInferenceReward`.
    ///
    /// This is a chain-side write-preparation RPC. It does NOT mutate
    /// chain state. The chain returns the canonical unsigned tx,
    /// signing hash, and envelope; the caller signs and submits via
    /// the normal `sum_sendRawTransaction` path.
    ///
    /// **This adapter method does no local precheck.** The precheck
    /// pipeline (dormancy, attestation, authority, maturity, bond)
    /// lives in the CLI-level driver; every one of those gates fires
    /// BEFORE this method is called, so a call reaching here has
    /// already cleared them.
    pub fn omninode_build_claim_inference_reward(
        &self,
        request: &BuildClaimRewardRequest,
    ) -> Result<BuildClaimRewardRaw, SettlementSubmitError> {
        let params = serde_json::to_value(request).map_err(|e| {
            SettlementSubmitError::WireDecode(format!(
                "failed to serialize BuildClaimRewardRequest: {e}"
            ))
        })?;
        // JSON-RPC params is an array. Wrap the single-object request
        // to match chain-team spec.
        let result = self
            .transport()
            .call("omninode_buildClaimInferenceReward", serde_json::json!([params]))
            .map_err(SettlementSubmitError::BuilderRpc)?;
        serde_json::from_value(result).map_err(|e| {
            SettlementSubmitError::WireDecode(format!(
                "failed to parse omninode_buildClaimInferenceReward response: {e}"
            ))
        })
    }

    /// Issue #100 — call `omninode_buildRegisterVerifier`.
    ///
    /// Chain-side write-preparation RPC (no state mutation). The chain
    /// returns the canonical unsigned `RegisterVerifier(bond)` tx plus
    /// the signing hash and envelope; the caller signs and submits via
    /// the normal `sum_sendRawTransaction` path.
    ///
    /// The response envelope is the generic settlement build response
    /// (`OmniSettlementBuildResponse`), modeled here by the generic
    /// [`SettlementBuilderEnvelope`] — the same envelope the claim
    /// builder returns — so the shared
    /// [`super::tx::verify_builder_envelope`] / [`super::tx::decode_unsigned_tx`]
    /// helpers apply. Local prechecks (chain-id, dormancy) live in the
    /// CLI driver and fire BEFORE this method is called.
    pub fn omninode_build_register_verifier(
        &self,
        request: &BuildRegisterVerifierRequest,
    ) -> Result<SettlementBuilderEnvelope, SettlementSubmitError> {
        let params = serde_json::to_value(request).map_err(|e| {
            SettlementSubmitError::WireDecode(format!(
                "failed to serialize BuildRegisterVerifierRequest: {e}"
            ))
        })?;
        let result = self
            .transport()
            .call("omninode_buildRegisterVerifier", serde_json::json!([params]))
            .map_err(SettlementSubmitError::BuilderRpc)?;
        serde_json::from_value(result).map_err(|e| {
            SettlementSubmitError::WireDecode(format!(
                "failed to parse omninode_buildRegisterVerifier response: {e}"
            ))
        })
    }
}
