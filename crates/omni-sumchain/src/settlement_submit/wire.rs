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

/// Generic settlement builder-envelope returned by every chain-side
/// `omninode_build*` settlement RPC. Structurally identical across the
/// claim (`omninode_buildClaimInferenceReward`) and register-verifier
/// (`omninode_buildRegisterVerifier`) builders — it carries only the
/// unsigned tx bytes, the signing hash, and the account/chain envelope
/// fields, none of which are operation-specific. Both the claim and
/// register paths decode + verify against this one type so the shared
/// [`super::tx::verify_builder_envelope`] / [`super::tx::decode_unsigned_tx`]
/// helpers apply unchanged.
///
/// All hex strings are `0x`-prefixed lowercase.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SettlementBuilderEnvelope {
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

/// Backwards-compatibility alias. Issue #87's claim path (and its
/// tests) refer to the builder envelope as `BuildClaimRewardRaw`; the
/// type was generalized to [`SettlementBuilderEnvelope`] in Issue #100
/// so the register-verifier path can share it. This alias keeps every
/// existing claim-path reference compiling and behaviorally identical.
pub type BuildClaimRewardRaw = SettlementBuilderEnvelope;

/// Request body for `omninode_buildRegisterVerifier` (Issue #100).
/// Emitted as the first (and only) element of the JSON-RPC `params`
/// array. Mirrors the chain-side `OmniBuildRegisterVerifierRequest`
/// (`from`, `bond`, `fee?`).
///
/// The chain-side builder returns an [`OmniSettlementBuildResponse`],
/// which this module models with the generic
/// [`SettlementBuilderEnvelope`]; the register-verifier build reuses
/// that one envelope type so the shared `verify_builder_envelope` /
/// `decode_unsigned_tx` helpers apply unchanged.
///
/// [`OmniSettlementBuildResponse`]: <chain crate `rpc::inference_settlement_types`>
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuildRegisterVerifierRequest {
    /// Verifier base58 chain address — the address OmniNode derived
    /// from its configured `OMNINODE_VERIFIER_SEED_HEX` seed. The
    /// registered verifier == the tx signer.
    pub from: String,

    /// Bond amount (native Koppa) to lock. The chain executor rejects
    /// a zero bond (`Failed(365)`), so operators pass a positive
    /// value; the builder itself embeds whatever is sent here.
    pub bond: u128,

    /// Optional operator-supplied fee. When absent, chain applies its
    /// default. `skip_serializing_if` keeps an omitted `--fee` from
    /// sending `"fee": null` (chain-team spec omits the field when
    /// absent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fee: Option<u128>,
}
