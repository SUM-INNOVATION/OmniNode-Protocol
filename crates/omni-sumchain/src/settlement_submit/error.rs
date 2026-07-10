//! Typed error surface for the settlement claim submit flow.
//!
//! Each variant maps to a distinct refusal/failure category. Callers
//! (the CLI at `omni-node/src/settlement_cli.rs`) match on the
//! variant to emit the correct `event=settlement_claim_*` marker per
//! Issue #85's taxonomy and the extension for Issue #87.

use omni_zkml::ChainClientError;

/// Settlement claim submit errors. Every non-happy path routes
/// through one of these variants.
///
/// Contract with the CLI marker layer (Issue #85 / #87):
///
/// | Variant                    | Marker                                      |
/// | -------------------------- | ------------------------------------------- |
/// | `Dormant`                  | `settlement_claim_refused_dormancy`         |
/// | `AttestationNotFound`      | `settlement_claim_failed` category=attestation_not_found |
/// | `AuthorityMismatch`        | `settlement_claim_refused_authority`        |
/// | `Immature`                 | `settlement_claim_refused_maturity`         |
/// | `BondPrecheckFailed`       | `settlement_claim_refused_bond_precheck`    |
/// | `BuilderRpc`               | `settlement_claim_failed` category=chain_rpc (during build) |
/// | `BuilderEnvelopeMismatch`  | `settlement_claim_failed` category=builder_mismatch |
/// | `WireDecode`               | `settlement_claim_failed` category=wire_decode |
/// | `DecodedTransactionMismatch` | `settlement_claim_failed` category=builder_mismatch |
/// | `SubmitRpc`                | `settlement_claim_failed` category=chain_rpc (after signing) |
#[derive(Debug, thiserror::Error)]
pub enum SettlementSubmitError {
    #[error(
        "settlement gate 'inference_settlement_enabled_from_height' dormant: \
         observed={observed:?}, head={head}"
    )]
    Dormant { observed: Option<u64>, head: u64 },

    #[error(
        "no attestation found on chain for (session_id={session_id}, \
         verifier={verifier})"
    )]
    AttestationNotFound { session_id: String, verifier: String },

    #[error(
        "authority mismatch: attestation verifier={attestation_verifier}, \
         derived signer address={derived}, --verifier flag value={explicit:?}"
    )]
    AuthorityMismatch {
        attestation_verifier: String,
        derived: String,
        explicit: Option<String>,
    },

    #[error(
        "claim not yet mature: claim_ready_block={claim_ready_block}, \
         head={head}, blocks_until_ready={blocks_until_ready}"
    )]
    Immature {
        claim_ready_block: u64,
        head: u64,
        blocks_until_ready: u64,
    },

    #[error("bond precheck refused: {outcome_kind}")]
    BondPrecheckFailed {
        /// Debug-formatted `BondPrecheckOutcome` variant name from #80.
        /// Kept as a string here to avoid pulling omni-node's
        /// `BondPrecheckOutcome` type into the omni-sumchain error
        /// surface (a cross-crate direction we intentionally avoid).
        outcome_kind: String,
    },

    #[error("omninode_buildClaimInferenceReward RPC failure: {0}")]
    BuilderRpc(ChainClientError),

    #[error(
        "builder response envelope mismatch on field '{field}': \
         expected={expected}, got={got}"
    )]
    BuilderEnvelopeMismatch {
        field: &'static str,
        expected: String,
        got: String,
    },

    #[error("wire decode failure: {0}")]
    WireDecode(String),

    #[error(
        "decoded TransactionV2 mismatch on field '{field}': \
         expected={expected}, got={got}"
    )]
    DecodedTransactionMismatch {
        field: &'static str,
        expected: String,
        got: String,
    },

    #[error("chain-address derivation failure: {0}")]
    AddressDerivation(String),

    #[error("sum_sendRawTransaction failure: {0}")]
    SubmitRpc(ChainClientError),

    #[error("settlement submit response parse failure: {0}")]
    SubmitResponseMalformed(String),
}

impl From<ChainClientError> for SettlementSubmitError {
    /// Fallback conversion — used at the raw RPC boundaries where
    /// intermediate `?` needs an implicit conversion. Categorised
    /// as `BuilderRpc` because that's the most-likely surface;
    /// the more-specific `SubmitRpc` variant is used explicitly by
    /// the submit path via `.map_err(...)` and does not go through
    /// this `From`.
    fn from(err: ChainClientError) -> Self {
        Self::BuilderRpc(err)
    }
}
