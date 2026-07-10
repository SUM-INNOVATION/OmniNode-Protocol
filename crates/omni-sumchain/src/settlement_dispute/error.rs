//! Typed error surface for the settlement dispute open + resolve
//! write path.
//!
//! Contract with the CLI marker layer:
//!
//! | Variant                          | Marker                                  |
//! | -------------------------------- | --------------------------------------- |
//! | `Dormant`                        | `settlement_dispute_refused_dormancy`   |
//! | `SessionNotFound`                | `settlement_dispute_refused_state`      |
//! | `FunderMismatch`                 | `settlement_dispute_refused_authority`  |
//! | `AttestationNotFound`            | `settlement_dispute_refused_state`      |
//! | `MaturityWindowClosed`           | `settlement_dispute_refused_maturity`   |
//! | `DuplicateDispute`               | `settlement_dispute_refused_state`      |
//! | `DisputeNotOpen`                 | `settlement_dispute_refused_state`      |
//! | `ApprovalsParseFailed`           | `settlement_dispute_failed`             |
//! | `EvidenceParseFailed`            | `settlement_dispute_failed`             |
//! | `BuilderRpc`                     | `settlement_dispute_failed` (chain_rpc) |
//! | `BuilderEnvelopeMismatch`        | `settlement_dispute_failed` (builder_mismatch) |
//! | `WireDecode`                     | `settlement_dispute_failed` (wire_decode) |
//! | `DecodedTransactionMismatch`     | `settlement_dispute_failed` (builder_mismatch) |
//! | `SubmitRpc`                      | `settlement_dispute_failed` (chain_rpc) |

use omni_zkml::ChainClientError;

#[derive(Debug, thiserror::Error)]
pub enum SettlementDisputeError {
    #[error(
        "settlement gate 'inference_settlement_enabled_from_height' dormant: \
         observed={observed:?}, head={head}"
    )]
    Dormant { observed: Option<u64>, head: u64 },

    #[error("session '{session_id}' not found on chain")]
    SessionNotFound { session_id: String },

    #[error(
        "OpenDispute funder mismatch: derived signer address={derived}, \
         session.funder={session_funder}"
    )]
    FunderMismatch {
        derived: String,
        session_funder: String,
    },

    #[error(
        "no attestation found for (session_id={session_id}, \
         verifier={verifier})"
    )]
    AttestationNotFound { session_id: String, verifier: String },

    #[error(
        "dispute window closed: attestation.included_at_height={included_at_height} \
         + params.finality_depth={finality_depth} + \
         session.dispute_window_blocks={dispute_window_blocks} = \
         maturity={maturity}, but head={head} >= maturity"
    )]
    MaturityWindowClosed {
        included_at_height: u64,
        finality_depth: u64,
        dispute_window_blocks: u64,
        maturity: u64,
        head: u64,
    },

    #[error(
        "duplicate dispute: an active dispute already exists for verifier={verifier} \
         on session={session_id}"
    )]
    DuplicateDispute {
        session_id: String,
        verifier: String,
    },

    #[error(
        "resolve refused: no `Open` dispute found for verifier={verifier} on \
         session={session_id}"
    )]
    DisputeNotOpen {
        session_id: String,
        verifier: String,
    },

    #[error("approvals JSON parse failed: {0}")]
    ApprovalsParseFailed(String),

    /// Empty approvals array (`[]`) refused locally. Chain-team
    /// confirmed (sum-chain#110) that an empty approvals array will
    /// build a tx but ALWAYS fail authority at execution because it
    /// cannot meet the validator-quorum threshold. Refusing locally
    /// prevents the operator from burning a fee on a predictably-
    /// rejected tx.
    #[error(
        "approvals list is empty: ResolveDispute requires a validator-quorum \
         approvals bundle; chain-side execution would refuse and consume the fee"
    )]
    ApprovalsEmpty,

    #[error("evidence hex parse failed: {0}")]
    EvidenceParseFailed(String),

    #[error("dispute builder RPC failure: {0}")]
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

    #[error("settlement dispute submit response parse failure: {0}")]
    SubmitResponseMalformed(String),
}

impl From<ChainClientError> for SettlementDisputeError {
    fn from(err: ChainClientError) -> Self {
        Self::BuilderRpc(err)
    }
}
