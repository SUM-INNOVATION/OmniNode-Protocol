//! Issue #83 — locally-enforced settlement-track dormancy detection.
//!
//! Three independent chain-param gates control whether their RPCs
//! return meaningful state. OmniNode fetches the gate values from
//! `chain_getChainParams`, compares to the current head, and refuses
//! to issue a gated RPC when the gate is dormant.
//!
//! ## Local-truth rationale
//!
//! Chain read RPCs may return empty state even while the gate is
//! dormant — a session query returning `null` is ambiguous between
//! "settlement dormant" and "session doesn't exist". The local check
//! disambiguates these before any gated RPC is issued.
//!
//! ## Gates
//!
//! - [`SettlementGate::Settlement`] — `inference_settlement_enabled_from_height`.
//!   Required for the `omninode_get{InferenceSession,InferenceClaims,
//!   InferenceDisputes,ClaimableReward}` methods.
//! - [`SettlementGate::Consistency`] — `inference_settlement_consistency_enabled_from_height`.
//!   Required for `omninode_getInferenceConsistency` and for view-
//!   layer composition of consistency-mode sessions.
//! - [`SettlementGate::Bonding`] — `inference_verifier_bonding_enabled_from_height`.
//!   Required for `omninode_getVerifier` and for view-layer
//!   composition of bond-required sessions.

use omni_zkml::ChainClientError;

/// The three independent InferenceSettlement chain-param gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SettlementGate {
    /// `inference_settlement_enabled_from_height`.
    Settlement,
    /// `inference_settlement_consistency_enabled_from_height`.
    Consistency,
    /// `inference_verifier_bonding_enabled_from_height`.
    Bonding,
}

impl SettlementGate {
    /// The `chain_getChainParams` field name that governs this gate.
    /// Used in dormant-error messages to name the exact chain field
    /// operators should inspect.
    pub fn param_field_name(&self) -> &'static str {
        match self {
            Self::Settlement => "inference_settlement_enabled_from_height",
            Self::Consistency => "inference_settlement_consistency_enabled_from_height",
            Self::Bonding => "inference_verifier_bonding_enabled_from_height",
        }
    }
}

/// Typed error surface for the settlement-read adapter and view layer.
#[derive(Debug, thiserror::Error)]
pub enum SettlementReadError {
    /// RPC-level gate dormant — the caller invoked an adapter method
    /// whose gate is either unset (`observed: None`) or scheduled
    /// beyond the current head (`Some(N)` with `head < N`). The gated
    /// RPC is NOT issued in either case.
    #[error(
        "settlement RPC gate '{}' dormant: observed={observed:?}, head={head}",
        gate.param_field_name(),
    )]
    Dormant {
        gate: SettlementGate,
        observed: Option<u64>,
        head: u64,
    },

    /// View-level gate missing — the underlying RPCs succeeded but the
    /// session's mode requires an additional gate to be active for a
    /// fully-normalized view. E.g. a consistency-mode session with the
    /// consistency gate dormant; or a bond-required session with the
    /// bonding gate dormant.
    #[error(
        "settlement view for session '{session_id}' requires gate '{}' to be active: {reason}",
        missing_gate.param_field_name(),
    )]
    ViewIncomplete {
        missing_gate: SettlementGate,
        session_id: String,
        reason: String,
    },

    /// Underlying transport or JSON-RPC error surfaced from the
    /// generic `SumChainClient<T>` methods (`get_chain_params`,
    /// `get_block_height`, or a gated RPC call itself).
    #[error("chain RPC failure: {0}")]
    Rpc(ChainClientError),

    /// Wire response did not deserialize into the declared DTO shape.
    /// If this fires, either the chain wire diverges from the DTOs in
    /// [`crate::settlement::wire`] or the fake transport returned a
    /// malformed pre-seeded response.
    #[error("settlement wire parse failure: {0}")]
    WireParse(String),
}

impl From<ChainClientError> for SettlementReadError {
    fn from(err: ChainClientError) -> Self {
        Self::Rpc(err)
    }
}

/// Check that `observed` (from chain params) is active at `head`.
/// Returns [`SettlementReadError::Dormant`] otherwise.
///
/// - `None` → dormant (gate never set).
/// - `Some(N)` with `head < N` → scheduled but not yet reached.
/// - `Some(N)` with `head >= N` → active.
pub fn check_gate_active(
    gate: SettlementGate,
    observed: Option<u64>,
    head: u64,
) -> Result<(), SettlementReadError> {
    match observed {
        None => Err(SettlementReadError::Dormant { gate, observed: None, head }),
        Some(n) if head < n => Err(SettlementReadError::Dormant {
            gate,
            observed: Some(n),
            head,
        }),
        Some(_) => Ok(()),
    }
}
