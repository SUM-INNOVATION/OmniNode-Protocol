//! Raw JSON-RPC DTOs for `omninode_buildClaimInferenceReward`.
//!
//! Request fields per chain-team spec:
//!
//! ```json
//! { "from": "<verifier address>",
//!   "session_id": "<session id>",
//!   "fee": 1000 }         // optional
//! ```
//!
//! Response fields:
//!
//! ```json
//! { "unsigned_tx":  "0x...",
//!   "signing_hash": "0x...",
//!   "from":         "<verifier address>",
//!   "nonce":        123,
//!   "fee":          1000,
//!   "chain_id":     1 }
//! ```
//!
//! Both types stay JSON-oriented — no bincode round-trip through the
//! chain payload here. The chain payload interior is decoded
//! separately by [`super::tx::decode_unsigned_tx`] against the
//! bumped `sumchain-primitives = 0.2.0` types.

use serde::{Deserialize, Serialize};

/// Request body for `omninode_buildClaimInferenceReward`. Emitted as
/// the first (and only) element of the JSON-RPC `params` array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuildClaimRewardRequest {
    /// Verifier base58 chain address. Chain rejects the request if
    /// this doesn't match a verifier holding an attestation for the
    /// session. OmniNode always sends the address it derived from its
    /// own configured seed.
    pub from: String,

    /// Session identifier from the CLI input.
    pub session_id: String,

    /// Optional operator-supplied fee. When absent, chain applies its
    /// default. `#[serde(skip_serializing_if = "Option::is_none")]`
    /// so an omitted `--fee` on the CLI doesn't send `"fee": null`
    /// (chain-team spec says the field is omitted when absent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fee: Option<u128>,
}

/// Response body for `omninode_buildClaimInferenceReward`. All hex
/// strings are `0x`-prefixed lowercase.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BuildClaimRewardRaw {
    /// Hex-encoded `bincode(TransactionV2)`. Decoded by
    /// [`super::tx::decode_unsigned_tx`] into a `TransactionV2` for
    /// local verification and signing.
    pub unsigned_tx: String,

    /// Hex-encoded 32-byte outer signing hash. OmniNode locally
    /// re-computes `tx.signing_hash()` after decoding `unsigned_tx`
    /// and asserts equality against this field.
    pub signing_hash: String,

    /// Base58 verifier chain address the chain-side builder used.
    /// Asserted to match the caller's derived address by
    /// [`super::tx::verify_builder_envelope`].
    pub from: String,

    /// Chain-supplied account nonce. Asserted to equal the decoded
    /// `tx.nonce`.
    pub nonce: u64,

    /// Chain-supplied fee. Asserted to equal the decoded `tx.fee`
    /// AND (if the operator supplied `--fee`) to equal that value.
    pub fee: u128,

    /// Chain id. Asserted to equal the decoded `tx.chain_id` AND
    /// the caller's `chain_getChainParams.chain_id`.
    pub chain_id: u64,
}
