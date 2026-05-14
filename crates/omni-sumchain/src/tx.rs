//! Phase 5 Stage 7b (planned) — outer `SignedTransaction` construction.
//!
//! Stage 7a ships this file as a placeholder. The `build_signed_transaction`
//! function below is **unimplemented** and returns a typed
//! `ChainClientError::Other(_)`. Once chain confirms the primitive
//! vendoring strategy (Open Q C-4), Stage 7b replaces the body with the
//! construction sequence documented below.
//!
//! # Stage 7b construction sequence (chain-confirmed)
//!
//! ```text
//! // 1. Stage 6 inner pipeline (already in omni-zkml).
//! let digest    = omni_zkml::commitment_to_chain_digest(&att.commitment)?;
//! let inner_sig = omni_zkml::sign_chain_attestation_digest(&seed, &digest)?;
//!
//! // 2. Wrap as chain inner payload. Stage 6's `InferenceAttestationTxData`
//! //    is byte-identical to chain's; reused as-is.
//! let tx_data = InferenceAttestationTxData { digest, verifier_signature: inner_sig };
//!
//! // 3-4. Build TransactionV2. Every input has a known source:
//! let params   = client.get_chain_params()?;             // chain_id, min_fee
//! let pubkey   = omni_zkml::signer_pubkey_bytes(&seed)?;
//! let address  = omni_zkml::signer_chain_address_base58(&seed)?;
//! let nonce    = client.get_nonce(&address)?;
//! let fee      = caller_override.unwrap_or(params.min_fee);
//! let tx = TransactionV2 {
//!     chain_id: params.chain_id,
//!     from:     Address::from_public_key(pubkey),
//!     fee,
//!     nonce,
//!     payload:  TxPayload::InferenceAttestation(tx_data),
//! };
//!
//! // 5-7. Outer canonical = raw bincode 1.3 of TransactionV2 (no domain
//! //      tag — chain_id provides replay protection). Outer signature
//! //      signs the 32-byte BLAKE3 of the canonical bytes (different
//! //      shape from Stage 6's inner pipeline, which signs raw bytes).
//! let outer_bytes = bincode1::serialize(&tx)?;
//! let outer_hash  = blake3::hash(&outer_bytes);
//! let outer_sig   = ed25519_sign_hash(&seed, outer_hash.as_bytes())?;
//!
//! // 8. Assemble.
//! let signed = SignedTransaction::new_v2(tx, outer_sig, pubkey);
//!
//! // 9. Encode + submit. `to_hex()` returns BARE hex (no 0x prefix);
//! //    sum_sendRawTransaction accepts bare hex. Response tx_hash is
//! //    0x-prefixed and propagates to SubmissionReceipt::tx_id as-is.
//! let hex     = signed.to_hex();
//! let tx_hash = transport.call("sum_sendRawTransaction", json!([hex]))?;
//! ```
//!
//! # Stage 7b dependencies (still gated)
//!
//! - `TransactionV2`, `TxPayload`, `SignedTransaction`, `Address` —
//!   either vendored from a publishable chain primitives crate (preferred)
//!   or mirrored locally with a parity fixture provided by the chain team.
//!   See open question C-4 in the Stage 7 plan.

use omni_types::phase5::InferenceAttestation;
use omni_zkml::{ChainClientError, SubmissionReceipt};

/// Stage 7a placeholder — always returns the typed "unimplemented"
/// `ChainClientError`. Stage 7b will replace this body with the
/// 9-step sequence documented in the module-level rustdoc.
pub(crate) fn build_and_submit_signed_transaction(
    _seed: &[u8; 32],
    _attestation: &InferenceAttestation,
) -> std::result::Result<SubmissionReceipt, ChainClientError> {
    Err(ChainClientError::Other(
        "build_and_submit_signed_transaction is unimplemented in Stage 7a; \
         outer SignedTransaction construction lands in Stage 7b once the \
         chain primitive vendoring strategy (Open Q C-4) is confirmed."
            .into(),
    ))
}
