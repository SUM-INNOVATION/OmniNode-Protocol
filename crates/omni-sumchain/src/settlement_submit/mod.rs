//! Issue #87 — verifier self-claim submission (chain-authoritative).
//!
//! Feature-gated on `settlement-submit` (superset of `settlement-read`
//! + `submit`). Adds NO new external crates on top of `submit`.
//!
//! ## Chain-authoritative contract
//!
//! Chain team owns tx construction. OmniNode:
//!
//! 1. Calls `omninode_buildClaimInferenceReward` — chain returns the
//!    canonical unsigned `TransactionV2` bytes plus the pre-computed
//!    `signing_hash`, `from`, `nonce`, `fee`, `chain_id`.
//! 2. Decodes `unsigned_tx` as `bincode(TransactionV2)` using the
//!    bumped `sumchain-primitives = 0.2.0`. The new
//!    `TxPayload::InferenceSettlement(_)` variant (ordinal 24) lives
//!    there; local decode is now type-safe.
//! 3. Verifies every decodable envelope field against both the
//!    builder envelope AND the local caller's expectations BEFORE
//!    any signing key is loaded.
//! 4. Signs via [`crate::outer_sign::outer_sign_transaction_v2`],
//!    which internally re-computes `tx.signing_hash()`, signs, and
//!    wraps in `SignedTransaction::new_v2(tx, sig, pubkey)`. The
//!    canonical wire format is `SignedTransaction::to_hex()` — no
//!    raw concat.
//! 5. Submits via `sum_sendRawTransaction` using the same transport
//!    the attestation submit path uses. Response parsed by the
//!    existing [`crate::tx::parse_send_raw_transaction_result`]
//!    helper (promoted to `pub(crate)` by this PR).
//!
//! ## Non-goals
//!
//! - No raw concat of `unsigned_tx || sig || pubkey`. Use
//!   `SignedTransaction::to_hex()` verbatim.
//! - No local construction of the settlement payload. Chain owns it.
//! - No local computation of `signing_hash` bypassing the chain-
//!   returned value — but we DO cross-verify by re-running
//!   `tx.signing_hash()` locally and asserting equality against
//!   the builder's value.
//! - No coordinator claim, no claim-on-behalf, no contributor-
//!   triggered claim, no auto-daemon.
//! - No dispute writes, no bonding writes, no session-admin writes.
//! - No dispute-penalty or bond-penalty surface in this submit path.

pub mod adapter;
pub mod error;
pub mod tx;
pub mod wire;

pub use error::SettlementSubmitError;
pub use tx::{
    decode_unsigned_tx, sign_and_submit, verify_builder_envelope,
    verify_decoded_transaction, ClaimSubmitReceipt,
};
pub use wire::{BuildClaimRewardRaw, BuildClaimRewardRequest};

// Re-export the chain-primitives types downstream consumers
// (`omni-node`'s `run_claim` handler + its tests) need to inspect and
// round-trip a submitted hex payload without adding a direct
// `sumchain-primitives` dep on `omni-node`. These are the real
// public-API chain surfaces — nothing test-only in this list.
pub use sumchain_primitives::inference_settlement::{
    ClaimInferenceRewardRequest, InferenceSettlementOperation,
    InferenceSettlementTxData,
};
pub use sumchain_primitives::{Address, SignedTransaction, TransactionV2, TxPayload};
