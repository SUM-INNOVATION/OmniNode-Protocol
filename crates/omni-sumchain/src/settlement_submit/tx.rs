//! Decode + verify + sign + submit for the settlement claim tx.
//!
//! Chain-authoritative construction: chain returns
//! `bincode(TransactionV2)` as `unsigned_tx` hex. OmniNode decodes
//! locally (v0.2.0 `sumchain-primitives` has the
//! `TxPayload::InferenceSettlement` variant), verifies every
//! decodable field, then signs via the existing
//! [`crate::outer_sign::outer_sign_transaction_v2`] helper and calls
//! `SignedTransaction::to_hex()` — no raw byte concat.

use omni_zkml::ChainClientError;
use sumchain_primitives::{
    inference_settlement::InferenceSettlementOperation, SignedTransaction,
    TransactionV2, TxPayload,
};

use crate::client::SumChainClient;
use crate::outer_sign::outer_sign_transaction_v2;
use crate::rpc::JsonRpcTransport;
use crate::tx::parse_send_raw_transaction_result;

use super::error::SettlementSubmitError;
use super::wire::{BuildClaimRewardRaw, SettlementBuilderEnvelope};

// ── Small hex helpers ────────────────────────────────────────────────────────
//
// Chosen over a `hex` crate dep so `settlement-submit` adds zero new
// external crates on top of what `submit` already pulls.

pub(crate) fn decode_hex_prefixed(s: &str) -> Result<Vec<u8>, SettlementSubmitError> {
    let body = s.strip_prefix("0x").unwrap_or(s);
    if body.len() % 2 != 0 {
        return Err(SettlementSubmitError::WireDecode(format!(
            "hex string length must be even, got {}",
            body.len()
        )));
    }
    (0..body.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&body[i..i + 2], 16).map_err(|e| {
                SettlementSubmitError::WireDecode(format!(
                    "invalid hex digit at index {i}: {e}"
                ))
            })
        })
        .collect()
}

pub(crate) fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

/// Issue #100 — status of a settlement write submission.
///
/// A successful `sum_sendRawTransaction` returns a tx hash, which the
/// chain treats as accepted into the mempool ([`SubmitStatus::Submitted`]).
/// The [`SubmitStatus::Pending`] variant is reserved for a future
/// builder/RPC that acknowledges a hash without confirming mempool
/// acceptance; the current path only ever produces `Submitted`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitStatus {
    /// `sum_sendRawTransaction` accepted the raw tx and returned a hash.
    Submitted,
    /// The tx hash is known but not yet acknowledged as accepted.
    Pending,
}

impl SubmitStatus {
    /// `true` when the chain accepted the raw tx into the mempool.
    pub fn is_submitted(self) -> bool {
        matches!(self, SubmitStatus::Submitted)
    }
}

/// Issue #100 — generic settlement submission receipt.
///
/// Carries only what any settlement write submission yields: the
/// chain-returned tx hash and the [`SubmitStatus`]. Operation-specific
/// receipts (e.g. the claim path's [`ClaimSubmitReceipt`]) wrap this
/// and add their own fields; paths with no extra fields to report (the
/// register-verifier path) use this directly rather than fabricating
/// claim-only fields such as `session_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettlementSubmitReceipt {
    /// The `0x`-prefixed chain-returned tx hash.
    pub tx_hash: String,
    /// Submission status reported by `sum_sendRawTransaction`.
    pub status: SubmitStatus,
}

/// Successful claim submission receipt. Superset of the generic
/// [`SettlementSubmitReceipt`] — adds the claim-only session /
/// verifier / envelope fields the claim CLI reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimSubmitReceipt {
    /// The `0x`-prefixed chain-returned tx hash.
    pub tx_hash: String,
    pub session_id: String,
    pub verifier: String,
    pub chain_id: u64,
    pub nonce: u64,
    pub fee: u128,
}

// ── Envelope + decoded verification ──────────────────────────────────────────

/// Verify the JSON-decodable envelope fields on the builder response
/// against the caller's known-good values. Runs BEFORE any hex decode
/// or bincode parse — a mismatch here means the chain-side builder
/// returned a shape we can't trust, and we refuse before allocating
/// signing material.
///
/// Called right after `omninode_buildClaimInferenceReward`.
pub fn verify_builder_envelope(
    response: &BuildClaimRewardRaw,
    expected_from: &str,
    expected_chain_id: u64,
    requested_fee: Option<u128>,
) -> Result<(), SettlementSubmitError> {
    if response.from != expected_from {
        return Err(SettlementSubmitError::BuilderEnvelopeMismatch {
            field: "from",
            expected: expected_from.to_string(),
            got: response.from.clone(),
        });
    }
    if response.chain_id != expected_chain_id {
        return Err(SettlementSubmitError::BuilderEnvelopeMismatch {
            field: "chain_id",
            expected: expected_chain_id.to_string(),
            got: response.chain_id.to_string(),
        });
    }
    if let Some(fee) = requested_fee {
        if response.fee != fee {
            return Err(SettlementSubmitError::BuilderEnvelopeMismatch {
                field: "fee",
                expected: fee.to_string(),
                got: response.fee.to_string(),
            });
        }
    }
    // Shape-only checks on `signing_hash` and `unsigned_tx`:
    if !response.signing_hash.starts_with("0x")
        || response.signing_hash.len() != 2 + 64
    {
        return Err(SettlementSubmitError::BuilderEnvelopeMismatch {
            field: "signing_hash",
            expected: "0x + 64 lowercase hex chars".to_string(),
            got: response.signing_hash.clone(),
        });
    }
    if !response.unsigned_tx.starts_with("0x")
        || response.unsigned_tx.len() < 4
        || response.unsigned_tx.len() % 2 != 0
    {
        return Err(SettlementSubmitError::BuilderEnvelopeMismatch {
            field: "unsigned_tx",
            expected: "0x-prefixed non-empty even-length hex".to_string(),
            got: response.unsigned_tx.clone(),
        });
    }
    Ok(())
}

/// Hex-decode `response.unsigned_tx` and bincode-deserialize into a
/// `TransactionV2`. Uses the workspace-standard `bincode` 1.3 config
/// via the crates.io `bincode` crate — matches the format the chain
/// itself serializes with.
///
/// A decode failure here indicates either (a) chain-side wire drift
/// against the pinned `sumchain-primitives` version, or (b) hex
/// encoding drift. Either way, we refuse before signing.
pub fn decode_unsigned_tx(
    response: &BuildClaimRewardRaw,
) -> Result<TransactionV2, SettlementSubmitError> {
    let bytes = decode_hex_prefixed(&response.unsigned_tx)?;
    // `TransactionV2::from_bytes` is the crate's own bincode-1
    // deserialise entrypoint — same wire config the chain writes
    // with. No local bincode dep required.
    TransactionV2::from_bytes(&bytes).map_err(|e| {
        SettlementSubmitError::WireDecode(format!(
            "failed to bincode-decode unsigned_tx into TransactionV2: {e}"
        ))
    })
}

/// Verify every field on the decoded `TransactionV2` against the
/// builder envelope AND the caller's local expectations. Fires
/// BEFORE any signing seed is loaded.
///
/// Checks (in order):
/// 1. `tx.chain_id` == `envelope.chain_id`.
/// 2. `tx.from.to_base58()` == `envelope.from` == `expected_from`.
/// 3. `tx.nonce` == `envelope.nonce`.
/// 4. `tx.fee` == `envelope.fee`.
/// 5. `tx.payload` is `TxPayload::InferenceSettlement(_)`.
/// 6. Inner operation is `InferenceSettlementOperation::ClaimReward(_)`.
/// 7. `request.session_id` == `expected_session_id`.
/// 8. Locally-computed `tx.signing_hash().as_bytes()` (32 bytes,
///    hex-encoded) matches `envelope.signing_hash`.
///
/// Any check failure returns `DecodedTransactionMismatch` with a
/// specific `field` label so the CLI can render a precise error.
pub fn verify_decoded_transaction(
    tx: &TransactionV2,
    envelope: &BuildClaimRewardRaw,
    expected_from: &str,
    expected_session_id: &str,
) -> Result<(), SettlementSubmitError> {
    // 1. chain_id.
    if u64::from(tx.chain_id) != envelope.chain_id {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.chain_id",
            expected: envelope.chain_id.to_string(),
            got: u64::from(tx.chain_id).to_string(),
        });
    }

    // 2. from.
    let tx_from_b58 = tx.from.to_base58();
    if tx_from_b58 != envelope.from {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.from vs envelope.from",
            expected: envelope.from.clone(),
            got: tx_from_b58.clone(),
        });
    }
    if tx_from_b58 != expected_from {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.from vs derived verifier",
            expected: expected_from.to_string(),
            got: tx_from_b58,
        });
    }

    // 3. nonce.
    if u64::from(tx.nonce) != envelope.nonce {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.nonce",
            expected: envelope.nonce.to_string(),
            got: u64::from(tx.nonce).to_string(),
        });
    }

    // 4. fee.
    if u128::from(tx.fee) != envelope.fee {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.fee",
            expected: envelope.fee.to_string(),
            got: u128::from(tx.fee).to_string(),
        });
    }

    // 5. payload variant.
    let settlement = match &tx.payload {
        TxPayload::InferenceSettlement(inner) => inner,
        other => {
            return Err(SettlementSubmitError::DecodedTransactionMismatch {
                field: "tx.payload variant",
                expected: "TxPayload::InferenceSettlement".to_string(),
                got: format!("{other:?}"),
            })
        }
    };

    // 6. inner operation variant.
    let claim_request = match &settlement.operation {
        InferenceSettlementOperation::ClaimReward(req) => req,
        other => {
            return Err(SettlementSubmitError::DecodedTransactionMismatch {
                field: "tx.payload.operation variant",
                expected: "InferenceSettlementOperation::ClaimReward".to_string(),
                got: format!("{other:?}"),
            })
        }
    };

    // 7. session_id.
    if claim_request.session_id != expected_session_id {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.payload.operation.session_id",
            expected: expected_session_id.to_string(),
            got: claim_request.session_id.clone(),
        });
    }

    // 8. Cross-verify signing_hash. This is the operator-side
    //    defense-in-depth check: even if the chain returned a
    //    signing_hash mismatched with the tx it also returned, the
    //    local recomputation catches it. Prevents a compromised or
    //    buggy builder from getting OmniNode to sign a hash that
    //    doesn't correspond to the tx bytes we're about to submit.
    let local_hash = tx.signing_hash();
    let local_hex = format!("0x{}", encode_hex(local_hash.as_bytes()));
    if local_hex != envelope.signing_hash {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.signing_hash() vs envelope.signing_hash",
            expected: envelope.signing_hash.clone(),
            got: local_hex,
        });
    }

    Ok(())
}

/// Issue #100 — verify a decoded `TransactionV2` is the expected
/// `RegisterVerifier(bond)` transaction. Runs BEFORE any signing seed
/// is loaded, mirroring [`verify_decoded_transaction`] for the claim
/// path but asserting the register-verifier operation shape instead.
///
/// Checks (in order):
/// 1. `tx.chain_id` == `envelope.chain_id`.
/// 2. `tx.from.to_base58()` == `envelope.from` == `expected_from`.
/// 3. `tx.nonce` == `envelope.nonce`.
/// 4. `tx.fee` == `envelope.fee`.
/// 5. `tx.payload` is `TxPayload::InferenceSettlement(_)`.
/// 6. Inner operation is `InferenceSettlementOperation::RegisterVerifier(_)`.
/// 7. `request.bond` == `expected_bond` (the operator's `--bond`).
/// 8. Locally-computed `tx.signing_hash()` matches `envelope.signing_hash`.
pub fn verify_register_verifier_transaction(
    tx: &TransactionV2,
    envelope: &SettlementBuilderEnvelope,
    expected_from: &str,
    expected_bond: u128,
) -> Result<(), SettlementSubmitError> {
    // 1. chain_id.
    if u64::from(tx.chain_id) != envelope.chain_id {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.chain_id",
            expected: envelope.chain_id.to_string(),
            got: u64::from(tx.chain_id).to_string(),
        });
    }

    // 2. from.
    let tx_from_b58 = tx.from.to_base58();
    if tx_from_b58 != envelope.from {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.from vs envelope.from",
            expected: envelope.from.clone(),
            got: tx_from_b58.clone(),
        });
    }
    if tx_from_b58 != expected_from {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.from vs derived verifier",
            expected: expected_from.to_string(),
            got: tx_from_b58,
        });
    }

    // 3. nonce.
    if u64::from(tx.nonce) != envelope.nonce {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.nonce",
            expected: envelope.nonce.to_string(),
            got: u64::from(tx.nonce).to_string(),
        });
    }

    // 4. fee.
    if u128::from(tx.fee) != envelope.fee {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.fee",
            expected: envelope.fee.to_string(),
            got: u128::from(tx.fee).to_string(),
        });
    }

    // 5. payload variant.
    let settlement = match &tx.payload {
        TxPayload::InferenceSettlement(inner) => inner,
        other => {
            return Err(SettlementSubmitError::DecodedTransactionMismatch {
                field: "tx.payload variant",
                expected: "TxPayload::InferenceSettlement".to_string(),
                got: format!("{other:?}"),
            })
        }
    };

    // 6. inner operation variant.
    let register_request = match &settlement.operation {
        InferenceSettlementOperation::RegisterVerifier(req) => req,
        other => {
            return Err(SettlementSubmitError::DecodedTransactionMismatch {
                field: "tx.payload.operation variant",
                expected: "InferenceSettlementOperation::RegisterVerifier".to_string(),
                got: format!("{other:?}"),
            })
        }
    };

    // 7. bond.
    if register_request.bond != expected_bond {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.payload.operation.bond",
            expected: expected_bond.to_string(),
            got: register_request.bond.to_string(),
        });
    }

    // 8. Cross-verify signing_hash — defense-in-depth against a builder
    //    that returns a signing_hash mismatched with the tx bytes.
    let local_hash = tx.signing_hash();
    let local_hex = format!("0x{}", encode_hex(local_hash.as_bytes()));
    if local_hex != envelope.signing_hash {
        return Err(SettlementSubmitError::DecodedTransactionMismatch {
            field: "tx.signing_hash() vs envelope.signing_hash",
            expected: envelope.signing_hash.clone(),
            got: local_hex,
        });
    }

    Ok(())
}

// ── Sign + submit ────────────────────────────────────────────────────────────

/// Issue #100 — generic sign + submit for ANY settlement
/// `TransactionV2`. Signs the decoded tx with the verifier seed and
/// submits via `sum_sendRawTransaction`, returning only the generic
/// [`SettlementSubmitReceipt`] (tx hash + [`SubmitStatus`]). This is
/// the shared core the claim path's [`sign_and_submit`] wraps; the
/// register-verifier path calls it directly since it has no
/// session/verifier fields to report.
///
/// Reuses [`crate::outer_sign::outer_sign_transaction_v2`] — which
/// internally re-computes `tx.signing_hash()`, signs with Ed25519, and
/// returns a `SignedTransaction::new_v2(tx, sig_bytes, pubkey_bytes)`.
/// The wire bytes are produced by `SignedTransaction::to_hex()` —
/// **no raw concat anywhere in this module**.
///
/// This function is the ONLY place in the settlement submit paths that
/// touches seed bytes. All prechecks (dormancy / gates / envelope /
/// decoded-tx, plus the claim-only attestation/authority/maturity/bond
/// gates) must fire BEFORE this function is invoked.
pub fn sign_and_submit_tx<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    tx: &TransactionV2,
    seed: &[u8; 32],
) -> Result<SettlementSubmitReceipt, SettlementSubmitError> {
    let signed = outer_sign_transaction_v2(seed, tx).map_err(|e| {
        // outer_sign returns ChainClientError; map to a submit-side
        // variant so the CLI's dispatcher classifies it correctly.
        SettlementSubmitError::AddressDerivation(format!(
            "outer_sign_transaction_v2 failed: {e}"
        ))
    })?;
    let signed_hex = signed.to_hex();

    // Sanity: SignedTransaction::from_hex(&signed_hex) must round-
    // trip. Not a security check; a fast fail-fast against wire
    // drift in the local `sumchain-primitives` version.
    if let Err(e) = SignedTransaction::from_hex(&signed_hex) {
        return Err(SettlementSubmitError::WireDecode(format!(
            "SignedTransaction::to_hex() round-trip broken: {e}"
        )));
    }

    let response = client
        .transport()
        .call(
            "sum_sendRawTransaction",
            serde_json::json!([signed_hex]),
        )
        .map_err(SettlementSubmitError::SubmitRpc)?;
    let tx_hash =
        parse_send_raw_transaction_result(&response).map_err(|e| match e {
            ChainClientError::Other(msg) => {
                SettlementSubmitError::SubmitResponseMalformed(msg)
            }
        })?;

    Ok(SettlementSubmitReceipt {
        tx_hash,
        status: SubmitStatus::Submitted,
    })
}

/// Sign the decoded claim `TransactionV2` with the verifier seed and
/// submit via `sum_sendRawTransaction`, returning the claim-specific
/// [`ClaimSubmitReceipt`]. Thin wrapper over the generic
/// [`sign_and_submit_tx`] that adds the claim-only session / verifier /
/// envelope fields.
///
/// This function is the ONLY caller in the settlement claim path that
/// touches seed bytes. All prechecks (dormancy, attestation,
/// authority, maturity, bond, envelope, decoded-tx) must fire BEFORE
/// this function is invoked.
pub fn sign_and_submit<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    tx: &TransactionV2,
    seed: &[u8; 32],
    session_id: &str,
    verifier: &str,
) -> Result<ClaimSubmitReceipt, SettlementSubmitError> {
    let receipt = sign_and_submit_tx(client, tx, seed)?;
    Ok(ClaimSubmitReceipt {
        tx_hash: receipt.tx_hash,
        session_id: session_id.to_string(),
        verifier: verifier.to_string(),
        chain_id: u64::from(tx.chain_id),
        nonce: u64::from(tx.nonce),
        fee: u128::from(tx.fee),
    })
}
