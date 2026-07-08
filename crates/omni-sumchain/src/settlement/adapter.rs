//! Issue #83 — extension methods on the existing generic
//! `SumChainClient<T: JsonRpcTransport>` for the InferenceSettlement
//! read RPCs. Every gated method obeys a single ordering invariant:
//!
//! 1. `chain_getChainParams` — fetches the three gate values (they
//!    ride on `dto::ChainParamsInfo`).
//! 2. `chain_getBlockHeight` — fetches the current head.
//! 3. Local gate check — refuses with [`SettlementReadError::Dormant`]
//!    if any required gate is dormant.
//! 4. Gated RPC issued only after the local check passes.
//!
//! The five attestation / block / params reads that the chain exposes
//! independently of settlement activation (
//! `chain_getChainParams`, `chain_getBlockHeight`,
//! `sum_getInferenceAttestation`, `sum_listInferenceAttestations`,
//! `sum_getInferenceAttestationStatus`) already live on
//! [`crate::SumChainClient`] and are not re-implemented here.

use omni_zkml::ChainClientError;

use crate::client::SumChainClient;
use crate::dto::{BlockFinality, ChainParamsInfo};
use crate::rpc::JsonRpcTransport;

use super::dormancy::{check_gate_active, SettlementGate, SettlementReadError};
use super::wire::{
    ClaimableRewardRaw, InferenceClaimsRaw, InferenceConsistencyRaw,
    InferenceDisputesRaw, InferenceSessionRaw, VerifierRegistryRaw,
};

impl<T: JsonRpcTransport> SumChainClient<T> {
    /// Fetch chain params + head in a single ordered pair. Both are
    /// re-fetched on every gated adapter call — no cross-call caching
    /// in this PR — so views always reflect the current head.
    fn fetch_params_and_head(
        &self,
    ) -> Result<(ChainParamsInfo, u64), SettlementReadError> {
        let params = self.get_chain_params()?;
        let head = self.get_block_height(BlockFinality::Latest)?.height;
        Ok((params, head))
    }

    /// Extract the `Option<u64>` for a specific gate from a fetched
    /// [`ChainParamsInfo`] snapshot.
    fn observed_gate(params: &ChainParamsInfo, gate: SettlementGate) -> Option<u64> {
        match gate {
            SettlementGate::Settlement => params.inference_settlement_enabled_from_height,
            SettlementGate::Consistency => {
                params.inference_settlement_consistency_enabled_from_height
            }
            SettlementGate::Bonding => params.inference_verifier_bonding_enabled_from_height,
        }
    }

    /// Check a single gate against a fetched params + head snapshot.
    fn require_gate(
        params: &ChainParamsInfo,
        head: u64,
        gate: SettlementGate,
    ) -> Result<(), SettlementReadError> {
        check_gate_active(gate, Self::observed_gate(params, gate), head)
    }

    /// `omninode_getInferenceSession(session_id)` — returns the base
    /// session record or `Ok(None)` when the chain has no such session.
    /// Requires the settlement gate at the RPC level.
    pub fn omninode_get_inference_session(
        &self,
        session_id: &str,
    ) -> Result<Option<InferenceSessionRaw>, SettlementReadError> {
        let (params, head) = self.fetch_params_and_head()?;
        Self::require_gate(&params, head, SettlementGate::Settlement)?;
        let result = self
            .transport()
            .call("omninode_getInferenceSession", serde_json::json!([session_id]))?;
        if result.is_null() {
            return Ok(None);
        }
        serde_json::from_value(result).map(Some).map_err(|e| {
            SettlementReadError::WireParse(format!(
                "failed to parse omninode_getInferenceSession response: {e}"
            ))
        })
    }

    /// `omninode_getInferenceClaims(session_id)` — returns the full
    /// list of claims recorded for `session_id`, possibly empty.
    /// Requires the settlement gate at the RPC level.
    pub fn omninode_get_inference_claims(
        &self,
        session_id: &str,
    ) -> Result<InferenceClaimsRaw, SettlementReadError> {
        let (params, head) = self.fetch_params_and_head()?;
        Self::require_gate(&params, head, SettlementGate::Settlement)?;
        let result = self
            .transport()
            .call("omninode_getInferenceClaims", serde_json::json!([session_id]))?;
        serde_json::from_value(result).map_err(|e| {
            SettlementReadError::WireParse(format!(
                "failed to parse omninode_getInferenceClaims response: {e}"
            ))
        })
    }

    /// `omninode_getInferenceDisputes(session_id)` — returns the full
    /// list of disputes opened against verifiers on `session_id`,
    /// possibly empty. Requires the settlement gate at the RPC level.
    pub fn omninode_get_inference_disputes(
        &self,
        session_id: &str,
    ) -> Result<InferenceDisputesRaw, SettlementReadError> {
        let (params, head) = self.fetch_params_and_head()?;
        Self::require_gate(&params, head, SettlementGate::Settlement)?;
        let result = self
            .transport()
            .call("omninode_getInferenceDisputes", serde_json::json!([session_id]))?;
        serde_json::from_value(result).map_err(|e| {
            SettlementReadError::WireParse(format!(
                "failed to parse omninode_getInferenceDisputes response: {e}"
            ))
        })
    }

    /// `omninode_getClaimableReward(session_id, verifier_address)` —
    /// chain-side pre-computed maturity + eligibility. Requires the
    /// settlement gate at the RPC level ONLY; view-level composition
    /// (in [`crate::settlement::view`]) may additionally require the
    /// consistency / bonding gate based on observed session mode.
    pub fn omninode_get_claimable_reward(
        &self,
        session_id: &str,
        verifier_address: &str,
    ) -> Result<ClaimableRewardRaw, SettlementReadError> {
        let (params, head) = self.fetch_params_and_head()?;
        Self::require_gate(&params, head, SettlementGate::Settlement)?;
        let result = self.transport().call(
            "omninode_getClaimableReward",
            serde_json::json!([session_id, verifier_address]),
        )?;
        serde_json::from_value(result).map_err(|e| {
            SettlementReadError::WireParse(format!(
                "failed to parse omninode_getClaimableReward response: {e}"
            ))
        })
    }

    /// `omninode_getInferenceConsistency(session_id)` — returns
    /// digest-tuple-keyed consistency groups. Requires BOTH the
    /// settlement gate AND the consistency gate at the RPC level;
    /// consistency is only meaningful when its gate is active.
    ///
    /// Gate-check order: settlement first, then consistency, so a
    /// dormant settlement gate produces
    /// `Dormant { gate: Settlement, .. }` even if consistency is also
    /// dormant.
    pub fn omninode_get_inference_consistency(
        &self,
        session_id: &str,
    ) -> Result<InferenceConsistencyRaw, SettlementReadError> {
        let (params, head) = self.fetch_params_and_head()?;
        Self::require_gate(&params, head, SettlementGate::Settlement)?;
        Self::require_gate(&params, head, SettlementGate::Consistency)?;
        let result = self.transport().call(
            "omninode_getInferenceConsistency",
            serde_json::json!([session_id]),
        )?;
        serde_json::from_value(result).map_err(|e| {
            SettlementReadError::WireParse(format!(
                "failed to parse omninode_getInferenceConsistency response: {e}"
            ))
        })
    }

    /// `omninode_getVerifier(address)` — verifier registry record.
    /// Returns `Ok(None)` when the address has no registry entry.
    /// Requires the bonding gate at the RPC level.
    ///
    /// **Independent of the settlement + consistency gates** — the
    /// chain's verifier registry has its own bonding activation.
    pub fn omninode_get_verifier(
        &self,
        address: &str,
    ) -> Result<Option<VerifierRegistryRaw>, SettlementReadError> {
        let (params, head) = self.fetch_params_and_head()?;
        Self::require_gate(&params, head, SettlementGate::Bonding)?;
        let result = self
            .transport()
            .call("omninode_getVerifier", serde_json::json!([address]))?;
        if result.is_null() {
            return Ok(None);
        }
        serde_json::from_value(result).map(Some).map_err(|e| {
            SettlementReadError::WireParse(format!(
                "failed to parse omninode_getVerifier response: {e}"
            ))
        })
    }
}

// Suppress unused-import warnings on `ChainClientError` in builds where
// only downstream users invoke the `?` operator against transport
// errors — the compiler already resolves the `From<ChainClientError>`
// impl via the `dormancy` module.
#[allow(dead_code)]
type _EnsureChainClientErrorReferenced = ChainClientError;
