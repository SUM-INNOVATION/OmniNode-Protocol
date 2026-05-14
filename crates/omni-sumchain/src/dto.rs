//! Phase 5 Stage 7a — chain RPC read-view DTOs.
//!
//! These are the **read-view** types `omni-sumchain` parses out of
//! SUM Chain JSON-RPC responses. They are NOT the same as the inner
//! chain-wire types (`InferenceAttestationDigest`,
//! `InferenceAttestationTxData`) from Stage 6, which live in
//! `omni-zkml::chain_wire` and are the **write-side** byte format used
//! when constructing a `SignedTransaction`.
//!
//! All hash / signature / tx hex fields in the read DTOs are emitted by
//! the chain with the `0x` prefix in lowercase. Stage 7a stores them
//! as-emitted; downstream consumers that need bare hex strip the prefix
//! themselves.

use serde::{Deserialize, Serialize};

// ── Status DTO ────────────────────────────────────────────────────────────────

/// Response shape for `sum_getInferenceAttestationStatus(tx_id)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceAttestationStatusInfo {
    /// Lowercase: `"submitted"` | `"included"` | `"finalized"` |
    /// `"failed"` | `"unknown"`. Other values are rejected at the
    /// status mapping boundary (see [`crate::status::map_status_info`]).
    pub status: String,
    /// `Some` once the chain has placed the tx in a block; `None` for
    /// `"submitted"` and `"unknown"`. Optional only on this DTO —
    /// read DTOs for already-included attestations carry it
    /// unconditionally.
    pub included_at_height: Option<u64>,
    /// Required for `"failed"`. May be present alongside other variants
    /// but Stage 7a does not consume it there.
    pub reason: Option<String>,
}

// ── Attestation DTO ──────────────────────────────────────────────────────────

/// Response shape for `sum_getInferenceAttestation` and
/// `sum_listInferenceAttestations`.
///
/// The chain only returns this DTO for attestations it has actually
/// included in a block, so `included_at_height` is **not** optional.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceAttestationInfo {
    pub session_id: String,
    pub verifier_address: String,
    /// `0x`-prefixed lowercase hex; 64 hex chars after the prefix
    /// (32 bytes).
    pub model_hash: String,
    /// `0x`-prefixed lowercase hex; 64 hex chars after the prefix
    /// (32 bytes).
    pub manifest_root: String,
    /// `0x`-prefixed lowercase hex; 64 hex chars after the prefix
    /// (32 bytes).
    pub response_hash: String,
    /// `0x`-prefixed lowercase hex; 64 hex chars after the prefix
    /// (32 bytes).
    pub proof_root: String,
    /// `0x`-prefixed lowercase hex; 128 hex chars after the prefix
    /// (64 bytes).
    pub verifier_signature: String,
    /// Required: the chain only emits this DTO for included
    /// attestations.
    pub included_at_height: u64,
    /// `0x`-prefixed lowercase hex; opaque to OmniNode.
    pub tx_hash: String,
    pub finalized: bool,
}

// ── Block-height DTO ──────────────────────────────────────────────────────────

/// Response shape for `chain_getBlockHeight(finality_token)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockHeightInfo {
    pub height: u64,
    /// `"latest"` or `"finalized"`. Echoed back from the request param.
    pub finality: String,
}

/// Finality token for `chain_getBlockHeight`. Serialised on the wire as
/// the lowercase string token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockFinality {
    Latest,
    Finalized,
}

impl BlockFinality {
    /// On-wire token for the JSON-RPC `params` array.
    pub fn as_token(&self) -> &'static str {
        match self {
            Self::Latest => "latest",
            Self::Finalized => "finalized",
        }
    }
}

// ── Chain params DTO ─────────────────────────────────────────────────────────

/// Response shape for `chain_getChainParams`.
///
/// `finality_depth`, `min_fee`, and `chain_id` are required (chain has
/// confirmed they are present at this RPC today).
/// `omninode_enabled_from_height` is the chain follow-up patch's
/// activation flag for the OmniNode subprotocol;
/// `v2_enabled_from_height` (Stage 7b) is the symmetric activation flag
/// for the chain's V2 transaction envelope itself. Both are
/// `#[serde(default)]` to keep the parser forward-compat with mirrors
/// that don't yet expose them.
///
/// Chain rev `d83e45a4`'s local mirror config emits both at value `0`,
/// meaning activation from genesis.
///
/// Note on `min_fee` width: the chain's on-tx fee field is `Balance`
/// (= `u128`); this DTO exposes it as `u64` because the JSON-RPC
/// emission fits in 64 bits for any practical local-mirror fee. Stage 7b
/// widens to `u128` via `as` cast at the `TransactionV2` construction
/// site.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainParamsInfo {
    pub finality_depth: u64,
    pub min_fee: u64,
    pub chain_id: u64,

    #[serde(default)]
    pub omninode_enabled_from_height: Option<u64>,

    /// Stage 7b addition. Chain rev `d83e45a4` exposes this alongside
    /// `omninode_enabled_from_height`. `submit_attestation` requires
    /// **both** activation flags to be `Some(h)` with `head >= h` before
    /// transmitting.
    #[serde(default)]
    pub v2_enabled_from_height: Option<u64>,
}
