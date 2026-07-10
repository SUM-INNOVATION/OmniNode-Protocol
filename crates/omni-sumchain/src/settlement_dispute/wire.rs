//! Raw JSON-RPC DTOs for `omninode_buildOpenInferenceDispute` +
//! `omninode_buildResolveInferenceDispute`.
//!
//! Chain-team confirmation (sum-chain#110):
//!
//! - Both builders return the same six-field response as
//!   `omninode_buildClaimInferenceReward`.
//! - Approvals wire in Option B (hex strings). OmniNode accepts
//!   operator input with OR without `0x` prefix and normalizes to
//!   the `0x`-prefixed form on the wire.

use serde::{Deserialize, Serialize};

use super::error::SettlementDisputeError;

// ── OpenDispute ──────────────────────────────────────────────────────────────

/// Request body for `omninode_buildOpenInferenceDispute`.
///
/// The `from` field is the funder base58 address; chain rejects if
/// this doesn't match the session's funder. OmniNode enforces the
/// same check locally first so the operator doesn't burn a fee on a
/// tx chain will refuse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuildOpenInferenceDisputeRequest {
    /// Funder base58 address.
    pub from: String,
    pub session_id: String,
    /// Target verifier base58 address.
    pub verifier: String,
    /// `0x`-prefixed lowercase hex, exactly 64 chars after prefix.
    pub evidence_commitment: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fee: Option<u128>,
}

/// Response body for `omninode_buildOpenInferenceDispute`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BuildOpenInferenceDisputeRaw {
    pub unsigned_tx: String,
    pub signing_hash: String,
    pub from: String,
    pub nonce: u64,
    pub fee: u128,
    pub chain_id: u64,
}

// ── ResolveDispute ───────────────────────────────────────────────────────────

/// Request body for `omninode_buildResolveInferenceDispute`.
///
/// The `from` field is the fee-payer base58 address — **not** an
/// authority signer. Authority for ResolveDispute comes from
/// `approvals` (validator quorum). Chain accepts submission from any
/// funded address whose approvals meet the configured quorum
/// threshold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuildResolveInferenceDisputeRequest {
    /// Fee-payer base58 address. Chain does NOT require this to be
    /// any specific role; ANY funded operator can submit a resolve
    /// tx if their approvals list meets the validator quorum.
    pub from: String,
    pub session_id: String,
    /// Target verifier base58 address of the dispute being resolved.
    pub verifier: String,
    /// True → resolve in favor of the verifier (allow claim to
    /// proceed). False → deny claim.
    pub allow_claim: bool,
    /// Validator approvals collected off-chain. Encoded per Q3's
    /// Option B (hex strings). See [`ApprovalWire`].
    pub approvals: Vec<ApprovalWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fee: Option<u128>,
}

/// Response body for `omninode_buildResolveInferenceDispute`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BuildResolveInferenceDisputeRaw {
    pub unsigned_tx: String,
    pub signing_hash: String,
    pub from: String,
    pub nonce: u64,
    pub fee: u128,
    pub chain_id: u64,
}

// ── Approvals ────────────────────────────────────────────────────────────────

/// Wire shape for a single validator approval — hex-encoded per
/// Q3 Option B.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ApprovalWire {
    /// `0x`-prefixed lowercase hex, exactly 64 chars after prefix
    /// (32-byte Ed25519 public key).
    pub pubkey: String,
    /// `0x`-prefixed lowercase hex, exactly 128 chars after prefix
    /// (64-byte Ed25519 signature).
    pub signature: String,
}

/// Parsed form of an approval line — byte arrays, ready to compare
/// against a decoded `ValidatorApproval` from the returned tx.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedApproval {
    pub pubkey: [u8; 32],
    pub signature: [u8; 64],
}

impl ParsedApproval {
    /// Convert to the on-wire `ApprovalWire` (hex-encoded with `0x`
    /// prefix).
    pub fn to_wire(&self) -> ApprovalWire {
        ApprovalWire {
            pubkey: format!("0x{}", encode_hex(&self.pubkey)),
            signature: format!("0x{}", encode_hex(&self.signature)),
        }
    }
}

/// Parse an operator-facing approvals JSON string into a
/// `Vec<ParsedApproval>`. Accepts each hex field with OR without the
/// `0x` prefix per Q3.
///
/// Input shape:
///
/// ```json
/// [
///   { "pubkey": "0x<64 hex>",   "signature": "0x<128 hex>" },
///   { "pubkey": "<64 hex>",     "signature": "<128 hex>" }
/// ]
/// ```
///
/// Returns `Ok(vec![])` for `[]`.
pub fn parse_approvals_json(
    json_str: &str,
) -> Result<Vec<ParsedApproval>, SettlementDisputeError> {
    #[derive(Deserialize)]
    struct Raw {
        pubkey: String,
        signature: String,
    }
    let raws: Vec<Raw> = serde_json::from_str(json_str).map_err(|e| {
        SettlementDisputeError::ApprovalsParseFailed(format!(
            "top-level JSON parse failed: {e}"
        ))
    })?;
    raws.into_iter()
        .enumerate()
        .map(|(i, r)| {
            let pubkey = decode_hex_flex(&r.pubkey, i, "pubkey", 32)?;
            let signature = decode_hex_flex(&r.signature, i, "signature", 64)?;
            let mut pk_arr = [0u8; 32];
            pk_arr.copy_from_slice(&pubkey);
            let mut sig_arr = [0u8; 64];
            sig_arr.copy_from_slice(&signature);
            Ok(ParsedApproval {
                pubkey: pk_arr,
                signature: sig_arr,
            })
        })
        .collect()
}

fn decode_hex_flex(
    s: &str,
    index: usize,
    field: &'static str,
    expected_bytes: usize,
) -> Result<Vec<u8>, SettlementDisputeError> {
    let body = s.strip_prefix("0x").unwrap_or(s);
    let expected_chars = expected_bytes * 2;
    if body.len() != expected_chars {
        return Err(SettlementDisputeError::ApprovalsParseFailed(format!(
            "approvals[{index}].{field}: expected {expected_chars} hex chars \
             (with or without 0x prefix), got {}",
            body.len()
        )));
    }
    (0..body.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&body[i..i + 2], 16).map_err(|e| {
                SettlementDisputeError::ApprovalsParseFailed(format!(
                    "approvals[{index}].{field}: invalid hex at char {i}: {e}"
                ))
            })
        })
        .collect()
}

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

// ── Evidence commitment hex helper ───────────────────────────────────────────

/// Parse the operator's `--evidence` argument as 32-byte hex. Accepts
/// with OR without `0x` prefix.
pub fn parse_evidence_hex(s: &str) -> Result<[u8; 32], SettlementDisputeError> {
    let body = s.strip_prefix("0x").unwrap_or(s);
    if body.len() != 64 {
        return Err(SettlementDisputeError::EvidenceParseFailed(format!(
            "expected 64 hex chars (with or without 0x prefix), got {}",
            body.len()
        )));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&body[i * 2..i * 2 + 2], 16).map_err(|e| {
            SettlementDisputeError::EvidenceParseFailed(format!(
                "invalid hex at char {}: {e}",
                i * 2
            ))
        })?;
    }
    Ok(out)
}

/// Encode 32 bytes as `0x`-prefixed lowercase hex for wire
/// transmission of `evidence_commitment`.
pub fn evidence_bytes_to_wire(bytes: &[u8; 32]) -> String {
    format!("0x{}", encode_hex(bytes))
}
