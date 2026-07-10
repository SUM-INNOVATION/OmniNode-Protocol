//! Extension methods on `SumChainClient<T>` for the two dispute
//! builder RPCs.
//!
//! No local prechecks here — the dispute pipeline in the CLI-level
//! driver runs every gate BEFORE these RPCs are called.

use crate::client::SumChainClient;
use crate::rpc::JsonRpcTransport;

use super::error::SettlementDisputeError;
use super::wire::{
    BuildOpenInferenceDisputeRaw, BuildOpenInferenceDisputeRequest,
    BuildResolveInferenceDisputeRaw, BuildResolveInferenceDisputeRequest,
};

impl<T: JsonRpcTransport> SumChainClient<T> {
    /// `omninode_buildOpenInferenceDispute` — chain-authoritative
    /// unsigned OpenDispute tx.
    pub fn omninode_build_open_inference_dispute(
        &self,
        request: &BuildOpenInferenceDisputeRequest,
    ) -> Result<BuildOpenInferenceDisputeRaw, SettlementDisputeError> {
        let params = serde_json::to_value(request).map_err(|e| {
            SettlementDisputeError::WireDecode(format!(
                "failed to serialize BuildOpenInferenceDisputeRequest: {e}"
            ))
        })?;
        let result = self
            .transport()
            .call(
                "omninode_buildOpenInferenceDispute",
                serde_json::json!([params]),
            )
            .map_err(SettlementDisputeError::BuilderRpc)?;
        serde_json::from_value(result).map_err(|e| {
            SettlementDisputeError::WireDecode(format!(
                "failed to parse omninode_buildOpenInferenceDispute response: {e}"
            ))
        })
    }

    /// `omninode_buildResolveInferenceDispute` — chain-authoritative
    /// unsigned ResolveDispute tx.
    pub fn omninode_build_resolve_inference_dispute(
        &self,
        request: &BuildResolveInferenceDisputeRequest,
    ) -> Result<BuildResolveInferenceDisputeRaw, SettlementDisputeError> {
        let params = serde_json::to_value(request).map_err(|e| {
            SettlementDisputeError::WireDecode(format!(
                "failed to serialize BuildResolveInferenceDisputeRequest: {e}"
            ))
        })?;
        let result = self
            .transport()
            .call(
                "omninode_buildResolveInferenceDispute",
                serde_json::json!([params]),
            )
            .map_err(SettlementDisputeError::BuilderRpc)?;
        serde_json::from_value(result).map_err(|e| {
            SettlementDisputeError::WireDecode(format!(
                "failed to parse omninode_buildResolveInferenceDispute response: {e}"
            ))
        })
    }
}
