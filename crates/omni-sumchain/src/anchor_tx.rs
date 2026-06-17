//! Phase 5 Stage 13.2 — adapter call sites for the
//! integrity-evidence-anchor RPCs.
//!
//! Two thin helpers consumed by `SumChainClient`'s
//! `EvidenceAnchorChainClient` impl:
//!
//! - [`build_and_submit_anchor`] — adapter-layer gates +
//!   bincode-1 serialize + POST `sum_submitIntegrityEvidenceAnchor`
//!   + lenient response parse.
//! - [`query_anchor_status`] — POST
//!   `sum_getIntegrityEvidenceAnchorStatus` + map to
//!   `omni_zkml::AnchorStatus` (lowercase status strings;
//!   `included_at_height` parsed-but-dropped per Stage 13.2 Q5).
//!
//! **Adapter-owned gates** (before any RPC):
//!
//! 1. Activation defense-in-depth via
//!    [`crate::SumChainClient::integrity_evidence_anchor_is_active`].
//!    On `false`, refuse with
//!    `ChainClientError::Other("integrity_evidence_anchor not activated")` —
//!    `classify_chain_client_error` returns
//!    `ChainErrorCategory::AdapterNotActivated` and the CLI emits
//!    `reason=not_activated`.
//! 2. Same-key submitter defense-in-depth — `self.seed()` must
//!    derive `tx_data.digest.signer_pubkey`. Stage 13.0's
//!    workflow already enforces this; the adapter re-checks at
//!    the boundary so non-CLI callers cannot bypass it.
//!
//! **Chain-id is NOT gated here.** The trait does not carry an
//! `expected_chain_id` and `SumChainClient` does not store one;
//! the CLI preflight owns chain-id enforcement.

use omni_zkml::{AnchorStatus, ChainClientError};
#[cfg(feature = "submit")]
use omni_zkml::{
    bincode1_serialize_anchor_tx_data, AnchorSubmissionReceipt,
    IntegrityEvidenceAnchorTxData,
};

use crate::anchor_dto::AnchorStatusResult;
#[cfg(feature = "submit")]
use crate::anchor_dto::parse_submit_anchor_result;
use crate::client::SumChainClient;
use crate::rpc::{error_prefixes, JsonRpcTransport};

/// Hex-encode bytes with a `0x` prefix. Inlined helper to keep
/// the `hex` crate out of the `omni-sumchain` dep graph; the
/// 148-byte anchor payload is small enough that a hand-rolled
/// encoder pays for itself.
#[cfg(feature = "submit")]
fn to_0x_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(2 + bytes.len() * 2);
    out.push_str("0x");
    for b in bytes {
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

/// Submit an anchor transaction. Adapter-layer gates run BEFORE
/// the RPC; on gate failure no submit RPC is reached.
#[cfg(feature = "submit")]
pub(crate) fn build_and_submit_anchor<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    tx_data: &IntegrityEvidenceAnchorTxData,
) -> std::result::Result<AnchorSubmissionReceipt, ChainClientError> {
    // ── Gate 1: activation defense-in-depth ──────────────────
    if !client.integrity_evidence_anchor_is_active()? {
        return Err(ChainClientError::Other(
            error_prefixes::ADAPTER_NOT_ACTIVATED.to_string(),
        ));
    }

    // ── Gate 2: same-key submitter defense-in-depth ──────────
    let derived = omni_zkml::anchor_signer_pubkey_bytes(client.seed())
        .map_err(|e| ChainClientError::Other(format!("derive pubkey: {e}")))?;
    if derived != tx_data.digest.signer_pubkey {
        let derived_hex = omni_zkml::anchor_hex_lower(&derived);
        let declared_hex = omni_zkml::anchor_hex_lower(&tx_data.digest.signer_pubkey);
        return Err(ChainClientError::Other(format!(
            "{prefix}seed derives {derived_hex}, digest declares {declared_hex}",
            prefix = error_prefixes::ADAPTER_SAME_KEY_FAIL,
        )));
    }

    // ── Step 3: bincode-1 serialize the locked 148-byte payload ─
    let bytes = bincode1_serialize_anchor_tx_data(tx_data).map_err(|e| {
        ChainClientError::Other(format!("bincode-1 serialize anchor tx_data: {e}"))
    })?;
    debug_assert_eq!(bytes.len(), 148, "Stage 13.0 wire shape is 148 bytes");
    let hex = to_0x_hex(&bytes);

    // ── Step 4: POST ─────────────────────────────────────────
    // Per Stage 13.2 Q1 contract assumption: single positional
    // hex string. Chain handles envelope / nonce / fee
    // accounting internally; pre-success rejects consume no fee,
    // no nonce, no state.
    let result = client
        .transport()
        .call("sum_submitIntegrityEvidenceAnchor", serde_json::json!([hex]))?;

    // ── Step 5: lenient response parse ───────────────────────
    let tx_hash = parse_submit_anchor_result(&result).ok_or_else(|| {
        ChainClientError::Other(format!(
            "{prefix}{value}",
            prefix = error_prefixes::ADAPTER_MALFORMED_SUBMIT_RESP,
            value = result
        ))
    })?;

    tracing::info!(
        tx_hash = %tx_hash,
        "submitted IntegrityEvidenceAnchor to SUM Chain"
    );

    Ok(AnchorSubmissionReceipt {
        tx_id: tx_hash,
        note: None,
    })
}

/// Query an anchor's chain-side status by `tx_hash`.
///
/// Lowercase status strings per Stage 13.2 Q4 contract
/// assumption. `included_at_height` is parsed-but-dropped per
/// Stage 13.2 Q5 (Stage 13.0 `AnchorStatus` enum unchanged).
/// Foreign / non-anchor tx_hash returns `Unknown` per chain
/// contract; consumer treats this as observation-only.
pub(crate) fn query_anchor_status<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    tx_id: &str,
) -> std::result::Result<AnchorStatus, ChainClientError> {
    let result = client.transport().call(
        "sum_getIntegrityEvidenceAnchorStatus",
        serde_json::json!([tx_id]),
    )?;
    let parsed: AnchorStatusResult = serde_json::from_value(result).map_err(|e| {
        ChainClientError::Other(format!(
            "{prefix}{e}",
            prefix = error_prefixes::ADAPTER_MALFORMED_STATUS_RESP,
        ))
    })?;
    Ok(match parsed.status.as_str() {
        "submitted" => AnchorStatus::Submitted,
        "included" => AnchorStatus::Included,
        "finalized" => AnchorStatus::Finalized,
        "failed" => AnchorStatus::Failed {
            reason: parsed
                .reason
                .unwrap_or_else(|| "no reason provided".to_string()),
        },
        "unknown" => AnchorStatus::Unknown,
        other => {
            return Err(ChainClientError::Other(format!(
                "{prefix}{other:?}",
                prefix = error_prefixes::ADAPTER_UNRECOGNIZED_STATUS,
            )));
        }
    })
}

#[cfg(all(test, feature = "submit"))]
mod tests {
    use super::*;

    #[test]
    fn to_0x_hex_pads_each_byte_to_two_chars() {
        assert_eq!(to_0x_hex(&[0x00, 0xab, 0xff]), "0x00abff");
        assert_eq!(to_0x_hex(&[]), "0x");
    }
}
