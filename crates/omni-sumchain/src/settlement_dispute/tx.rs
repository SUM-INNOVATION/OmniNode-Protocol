//! Decode + verify + sign + submit for OpenDispute + ResolveDispute.
//!
//! Same chain-authoritative pattern as `settlement_submit/tx.rs`:
//! chain returns `bincode(TransactionV2)` as `unsigned_tx` hex, we
//! decode locally against pinned `sumchain-primitives = 0.2.0` (which
//! has the `TxPayload::InferenceSettlement` variant + both
//! `OpenDispute` and `ResolveDispute` operation variants), verify
//! every decodable field, sign via
//! [`crate::outer_sign::outer_sign_transaction_v2`], and call
//! `SignedTransaction::to_hex()` — NO raw concat.

use omni_zkml::ChainClientError;
use sumchain_primitives::{
    inference_settlement::InferenceSettlementOperation, SignedTransaction,
    TransactionV2, TxPayload,
};

use crate::client::SumChainClient;
use crate::outer_sign::outer_sign_transaction_v2;
use crate::rpc::JsonRpcTransport;
use crate::tx::parse_send_raw_transaction_result;

use super::error::SettlementDisputeError;
use super::wire::{
    BuildOpenInferenceDisputeRaw, BuildResolveInferenceDisputeRaw, ParsedApproval,
};

/// Successful submission receipt (open OR resolve).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisputeSubmitReceipt {
    pub tx_hash: String,
    /// Phase this receipt belongs to — `"open"` or `"resolve"`.
    pub phase: &'static str,
    pub session_id: String,
    pub verifier: String,
    pub chain_id: u64,
    pub nonce: u64,
    pub fee: u128,
}

// ── Small hex helpers (kept local to avoid a new hex dep) ────────────────────

fn decode_hex_prefixed(s: &str) -> Result<Vec<u8>, SettlementDisputeError> {
    let body = s.strip_prefix("0x").unwrap_or(s);
    if body.len() % 2 != 0 {
        return Err(SettlementDisputeError::WireDecode(format!(
            "hex string length must be even, got {}",
            body.len()
        )));
    }
    (0..body.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&body[i..i + 2], 16).map_err(|e| {
                SettlementDisputeError::WireDecode(format!(
                    "invalid hex digit at index {i}: {e}"
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

// ── Envelope + decoded verification — shared shape helpers ───────────────────

fn verify_envelope_shape(
    from: &str,
    chain_id: u64,
    fee: u128,
    unsigned_tx: &str,
    signing_hash: &str,
    expected_from: &str,
    expected_chain_id: u64,
    requested_fee: Option<u128>,
) -> Result<(), SettlementDisputeError> {
    if from != expected_from {
        return Err(SettlementDisputeError::BuilderEnvelopeMismatch {
            field: "from",
            expected: expected_from.to_string(),
            got: from.to_string(),
        });
    }
    if chain_id != expected_chain_id {
        return Err(SettlementDisputeError::BuilderEnvelopeMismatch {
            field: "chain_id",
            expected: expected_chain_id.to_string(),
            got: chain_id.to_string(),
        });
    }
    if let Some(f) = requested_fee {
        if fee != f {
            return Err(SettlementDisputeError::BuilderEnvelopeMismatch {
                field: "fee",
                expected: f.to_string(),
                got: fee.to_string(),
            });
        }
    }
    if !signing_hash.starts_with("0x") || signing_hash.len() != 2 + 64 {
        return Err(SettlementDisputeError::BuilderEnvelopeMismatch {
            field: "signing_hash",
            expected: "0x + 64 lowercase hex chars".to_string(),
            got: signing_hash.to_string(),
        });
    }
    if !unsigned_tx.starts_with("0x")
        || unsigned_tx.len() < 4
        || unsigned_tx.len() % 2 != 0
    {
        return Err(SettlementDisputeError::BuilderEnvelopeMismatch {
            field: "unsigned_tx",
            expected: "0x-prefixed non-empty even-length hex".to_string(),
            got: unsigned_tx.to_string(),
        });
    }
    Ok(())
}

pub fn verify_open_builder_envelope(
    response: &BuildOpenInferenceDisputeRaw,
    expected_from: &str,
    expected_chain_id: u64,
    requested_fee: Option<u128>,
) -> Result<(), SettlementDisputeError> {
    verify_envelope_shape(
        &response.from,
        response.chain_id,
        response.fee,
        &response.unsigned_tx,
        &response.signing_hash,
        expected_from,
        expected_chain_id,
        requested_fee,
    )
}

pub fn verify_resolve_builder_envelope(
    response: &BuildResolveInferenceDisputeRaw,
    expected_from: &str,
    expected_chain_id: u64,
    requested_fee: Option<u128>,
) -> Result<(), SettlementDisputeError> {
    verify_envelope_shape(
        &response.from,
        response.chain_id,
        response.fee,
        &response.unsigned_tx,
        &response.signing_hash,
        expected_from,
        expected_chain_id,
        requested_fee,
    )
}

// ── Decode + verify shared logic ─────────────────────────────────────────────

fn decode_hex_and_bincode(
    unsigned_tx_hex: &str,
) -> Result<TransactionV2, SettlementDisputeError> {
    let bytes = decode_hex_prefixed(unsigned_tx_hex)?;
    TransactionV2::from_bytes(&bytes).map_err(|e| {
        SettlementDisputeError::WireDecode(format!(
            "failed to bincode-decode unsigned_tx into TransactionV2: {e}"
        ))
    })
}

pub fn decode_unsigned_tx_open(
    response: &BuildOpenInferenceDisputeRaw,
) -> Result<TransactionV2, SettlementDisputeError> {
    decode_hex_and_bincode(&response.unsigned_tx)
}

pub fn decode_unsigned_tx_resolve(
    response: &BuildResolveInferenceDisputeRaw,
) -> Result<TransactionV2, SettlementDisputeError> {
    decode_hex_and_bincode(&response.unsigned_tx)
}

/// Convenience re-export: `decode_unsigned_tx` is the generic entry
/// used by higher-level tests. Both response types decode from
/// `unsigned_tx` hex; this helper picks the field automatically.
pub fn decode_unsigned_tx<R: HasUnsignedTx>(
    response: &R,
) -> Result<TransactionV2, SettlementDisputeError> {
    decode_hex_and_bincode(response.unsigned_tx())
}

pub trait HasUnsignedTx {
    fn unsigned_tx(&self) -> &str;
}

impl HasUnsignedTx for BuildOpenInferenceDisputeRaw {
    fn unsigned_tx(&self) -> &str {
        &self.unsigned_tx
    }
}

impl HasUnsignedTx for BuildResolveInferenceDisputeRaw {
    fn unsigned_tx(&self) -> &str {
        &self.unsigned_tx
    }
}

fn common_decoded_checks(
    tx: &TransactionV2,
    envelope_chain_id: u64,
    envelope_from: &str,
    envelope_nonce: u64,
    envelope_fee: u128,
    envelope_signing_hash: &str,
    expected_from: &str,
) -> Result<(), SettlementDisputeError> {
    if u64::from(tx.chain_id) != envelope_chain_id {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.chain_id",
            expected: envelope_chain_id.to_string(),
            got: u64::from(tx.chain_id).to_string(),
        });
    }
    let tx_from_b58 = tx.from.to_base58();
    if tx_from_b58 != envelope_from {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.from vs envelope.from",
            expected: envelope_from.to_string(),
            got: tx_from_b58,
        });
    }
    if tx_from_b58 != expected_from {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.from vs expected signer",
            expected: expected_from.to_string(),
            got: tx_from_b58,
        });
    }
    if u64::from(tx.nonce) != envelope_nonce {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.nonce",
            expected: envelope_nonce.to_string(),
            got: u64::from(tx.nonce).to_string(),
        });
    }
    if u128::from(tx.fee) != envelope_fee {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.fee",
            expected: envelope_fee.to_string(),
            got: u128::from(tx.fee).to_string(),
        });
    }
    let local_hash = tx.signing_hash();
    let local_hex = format!("0x{}", encode_hex(local_hash.as_bytes()));
    if local_hex != envelope_signing_hash {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.signing_hash() vs envelope.signing_hash",
            expected: envelope_signing_hash.to_string(),
            got: local_hex,
        });
    }
    Ok(())
}

/// Verify a decoded `TransactionV2` against an OpenDispute envelope +
/// caller expectations.
pub fn verify_open_decoded_transaction(
    tx: &TransactionV2,
    envelope: &BuildOpenInferenceDisputeRaw,
    expected_from: &str,
    expected_session_id: &str,
    expected_verifier: &str,
    expected_evidence: &[u8; 32],
) -> Result<(), SettlementDisputeError> {
    common_decoded_checks(
        tx,
        envelope.chain_id,
        &envelope.from,
        envelope.nonce,
        envelope.fee,
        &envelope.signing_hash,
        expected_from,
    )?;

    let settlement = match &tx.payload {
        TxPayload::InferenceSettlement(inner) => inner,
        other => {
            return Err(SettlementDisputeError::DecodedTransactionMismatch {
                field: "tx.payload variant",
                expected: "TxPayload::InferenceSettlement".to_string(),
                got: format!("{other:?}"),
            });
        }
    };

    let open_request = match &settlement.operation {
        InferenceSettlementOperation::OpenDispute(req) => req,
        other => {
            return Err(SettlementDisputeError::DecodedTransactionMismatch {
                field: "tx.payload.operation variant",
                expected: "InferenceSettlementOperation::OpenDispute".to_string(),
                got: format!("{other:?}"),
            });
        }
    };

    if open_request.session_id != expected_session_id {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.payload.operation.session_id",
            expected: expected_session_id.to_string(),
            got: open_request.session_id.clone(),
        });
    }
    let request_verifier_b58 = open_request.verifier.to_base58();
    if request_verifier_b58 != expected_verifier {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.payload.operation.verifier",
            expected: expected_verifier.to_string(),
            got: request_verifier_b58,
        });
    }
    if &open_request.evidence_commitment != expected_evidence {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.payload.operation.evidence_commitment",
            expected: format!("0x{}", encode_hex(expected_evidence)),
            got: format!("0x{}", encode_hex(&open_request.evidence_commitment)),
        });
    }
    Ok(())
}

/// Verify a decoded `TransactionV2` against a ResolveDispute envelope +
/// caller expectations, including approval byte-equality (order-sensitive).
pub fn verify_resolve_decoded_transaction(
    tx: &TransactionV2,
    envelope: &BuildResolveInferenceDisputeRaw,
    expected_from: &str,
    expected_session_id: &str,
    expected_verifier: &str,
    expected_allow_claim: bool,
    expected_approvals: &[ParsedApproval],
) -> Result<(), SettlementDisputeError> {
    common_decoded_checks(
        tx,
        envelope.chain_id,
        &envelope.from,
        envelope.nonce,
        envelope.fee,
        &envelope.signing_hash,
        expected_from,
    )?;

    let settlement = match &tx.payload {
        TxPayload::InferenceSettlement(inner) => inner,
        other => {
            return Err(SettlementDisputeError::DecodedTransactionMismatch {
                field: "tx.payload variant",
                expected: "TxPayload::InferenceSettlement".to_string(),
                got: format!("{other:?}"),
            });
        }
    };

    let resolve_request = match &settlement.operation {
        InferenceSettlementOperation::ResolveDispute(req) => req,
        other => {
            return Err(SettlementDisputeError::DecodedTransactionMismatch {
                field: "tx.payload.operation variant",
                expected: "InferenceSettlementOperation::ResolveDispute".to_string(),
                got: format!("{other:?}"),
            });
        }
    };

    if resolve_request.session_id != expected_session_id {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.payload.operation.session_id",
            expected: expected_session_id.to_string(),
            got: resolve_request.session_id.clone(),
        });
    }
    let request_verifier_b58 = resolve_request.verifier.to_base58();
    if request_verifier_b58 != expected_verifier {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.payload.operation.verifier",
            expected: expected_verifier.to_string(),
            got: request_verifier_b58,
        });
    }
    if resolve_request.allow_claim != expected_allow_claim {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.payload.operation.allow_claim",
            expected: expected_allow_claim.to_string(),
            got: resolve_request.allow_claim.to_string(),
        });
    }
    // Approvals: exact-count, exact-byte, order-sensitive.
    if resolve_request.approvals.len() != expected_approvals.len() {
        return Err(SettlementDisputeError::DecodedTransactionMismatch {
            field: "tx.payload.operation.approvals.len",
            expected: expected_approvals.len().to_string(),
            got: resolve_request.approvals.len().to_string(),
        });
    }
    for (i, (chain_appr, expected)) in resolve_request
        .approvals
        .iter()
        .zip(expected_approvals.iter())
        .enumerate()
    {
        if chain_appr.pubkey != expected.pubkey {
            return Err(SettlementDisputeError::DecodedTransactionMismatch {
                field: "tx.payload.operation.approvals[..].pubkey",
                expected: format!(
                    "approvals[{i}].pubkey = 0x{}",
                    encode_hex(&expected.pubkey)
                ),
                got: format!(
                    "approvals[{i}].pubkey = 0x{}",
                    encode_hex(&chain_appr.pubkey)
                ),
            });
        }
        if chain_appr.signature != expected.signature {
            return Err(SettlementDisputeError::DecodedTransactionMismatch {
                field: "tx.payload.operation.approvals[..].signature",
                expected: format!(
                    "approvals[{i}].signature = 0x{}",
                    encode_hex(&expected.signature)
                ),
                got: format!(
                    "approvals[{i}].signature = 0x{}",
                    encode_hex(&chain_appr.signature)
                ),
            });
        }
    }
    Ok(())
}

// ── Sign + submit ────────────────────────────────────────────────────────────

/// Sign the decoded `TransactionV2` and submit via
/// `sum_sendRawTransaction`. Shared by both open + resolve — they only
/// differ in the payload variant, which the caller has already verified.
pub fn sign_and_submit_dispute<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    tx: &TransactionV2,
    seed: &[u8; 32],
    phase: &'static str,
    session_id: &str,
    verifier: &str,
) -> Result<DisputeSubmitReceipt, SettlementDisputeError> {
    let signed = outer_sign_transaction_v2(seed, tx).map_err(|e| {
        SettlementDisputeError::AddressDerivation(format!(
            "outer_sign_transaction_v2 failed: {e}"
        ))
    })?;
    let signed_hex = signed.to_hex();

    if let Err(e) = SignedTransaction::from_hex(&signed_hex) {
        return Err(SettlementDisputeError::WireDecode(format!(
            "SignedTransaction::to_hex() round-trip broken: {e}"
        )));
    }

    let response = client
        .transport()
        .call(
            "sum_sendRawTransaction",
            serde_json::json!([signed_hex]),
        )
        .map_err(SettlementDisputeError::SubmitRpc)?;

    let tx_hash =
        parse_send_raw_transaction_result(&response).map_err(|e| match e {
            ChainClientError::Other(msg) => {
                SettlementDisputeError::SubmitResponseMalformed(msg)
            }
        })?;

    Ok(DisputeSubmitReceipt {
        tx_hash,
        phase,
        session_id: session_id.to_string(),
        verifier: verifier.to_string(),
        chain_id: u64::from(tx.chain_id),
        nonce: u64::from(tx.nonce),
        fee: u128::from(tx.fee),
    })
}
