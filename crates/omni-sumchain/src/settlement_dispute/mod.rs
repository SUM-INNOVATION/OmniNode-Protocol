//! Issue #81 — settlement dispute WRITE path (open + resolve).
//!
//! Feature-gated on `settlement-dispute` (superset of
//! `settlement-submit`, which is a superset of `settlement-read` +
//! `submit`). Adds NO new external crates on top of
//! `settlement-submit`.
//!
//! ## Chain-authoritative contract
//!
//! Confirmed via `sum-chain#110`:
//!
//! - `omninode_buildOpenInferenceDispute` and
//!   `omninode_buildResolveInferenceDispute` are the authoritative
//!   tx constructors. OmniNode never builds these txs locally in
//!   production code.
//! - Both builders return the same six-field unsigned tx response
//!   as `omninode_buildClaimInferenceReward` (see #87):
//!   `unsigned_tx`, `signing_hash`, `from`, `nonce`, `fee`,
//!   `chain_id`.
//! - `unsigned_tx = bincode(TransactionV2)`.
//! - Submit path: `SignedTransaction::new_v2(...).to_hex()` then
//!   `sum_sendRawTransaction`. No raw concat.
//!
//! ## Local prechecks (open subcommand)
//!
//! Three hard prechecks fire BEFORE the builder call:
//!
//! 1. **Funder authority**: `omninode_getInferenceSession(session_id)`
//!    returns `funder`. Refuse if `derived != session.funder`.
//! 2. **Maturity window**: compute
//!    `maturity = attestation.included_at_height +
//!    params.finality_depth + session.dispute_window_blocks`.
//!    Refuse if `head >= maturity` — the dispute window has closed.
//! 3. **Duplicate dispute**: `omninode_getInferenceDisputes(session_id)`
//!    lists existing disputes. Refuse if any already targets this
//!    verifier.
//!
//! ## Non-goals
//!
//! - No resolver key; no single-resolver address.
//! - No validator-approval collection or signing. `--approvals` is
//!   an already-signed input file; OmniNode only relays it.
//! - No claim that OmniNode decides disputes. OmniNode only submits
//!   chain-authorized dispute transactions.
//! - No runtime semantic claims beyond what local prechecks verify.
//! - No dispute-penalty or bond-penalty surface in this write path.

pub mod adapter;
pub mod error;
pub mod tx;
pub mod wire;

pub use error::SettlementDisputeError;
pub use tx::{
    decode_unsigned_tx, decode_unsigned_tx_open, decode_unsigned_tx_resolve,
    sign_and_submit_dispute, verify_open_builder_envelope,
    verify_open_decoded_transaction, verify_resolve_builder_envelope,
    verify_resolve_decoded_transaction, DisputeSubmitReceipt,
};
pub use wire::{
    ApprovalWire, BuildOpenInferenceDisputeRaw, BuildOpenInferenceDisputeRequest,
    BuildResolveInferenceDisputeRaw, BuildResolveInferenceDisputeRequest,
    ParsedApproval,
};

// Re-export the chain-primitives types the settlement dispute path
// needs and downstream consumers may match on. These are the real
// public-API chain surfaces; nothing test-only in this list.
pub use sumchain_primitives::inference_settlement::{
    InferenceSettlementOperation, InferenceSettlementTxData,
    OpenInferenceDisputeRequest, ResolveInferenceDisputeRequest,
};
pub use sumchain_primitives::validator_authority::ValidatorApproval;
pub use sumchain_primitives::{Address, SignedTransaction, TransactionV2, TxPayload};
