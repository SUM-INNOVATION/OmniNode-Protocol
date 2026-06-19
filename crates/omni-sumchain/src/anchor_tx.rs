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

use omni_zkml::{
    canonicalize_tx_hash, AnchorStatus, AnchorStatusReport, AnchoredArtifactKind,
    BatchStatusItem, ChainClientError, TupleLookupResult,
    FAILED_REASON_NULL_FALLBACK,
};
#[cfg(feature = "submit")]
use omni_zkml::{
    bincode1_serialize_anchor_tx_data, AnchorSubmissionReceipt,
    IntegrityEvidenceAnchorTxData,
};

use crate::anchor_dto::{AnchorStatusResult, BatchStatusItem as BatchItemDto, ByTupleResult};
#[cfg(feature = "submit")]
use crate::anchor_dto::parse_submit_anchor_result;
use crate::client::SumChainClient;
use crate::rpc::{error_prefixes, JsonRpcTransport};

/// Stage 13.9 batch chunk size — chain contract max.
pub(crate) const ANCHOR_STATUS_BATCH_MAX: usize = 100;

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

// ── Stage 13.9 — DTO → report conversion ─────────────────────────────────────

/// Convert a parsed `AnchorStatusResult` into the omni-zkml
/// `AnchorStatusReport`. Maps lowercase status strings to
/// `AnchorStatus` variants; when status is `failed` with
/// `reason: null`, substitutes [`FAILED_REASON_NULL_FALLBACK`]
/// (Stage 13.9 implementation lock) so the existing
/// `AnchorStatus::Failed { reason: String }` shape stays
/// compatible. The chain's `code` is preserved independently
/// on the report.
fn dto_to_status_report(parsed: AnchorStatusResult) -> Result<AnchorStatusReport, ChainClientError> {
    let status = match parsed.status.as_str() {
        "submitted" => AnchorStatus::Submitted,
        "included" => AnchorStatus::Included,
        "finalized" => AnchorStatus::Finalized,
        "failed" => AnchorStatus::Failed {
            reason: parsed
                .reason
                .clone()
                .unwrap_or_else(|| FAILED_REASON_NULL_FALLBACK.to_string()),
        },
        "unknown" => AnchorStatus::Unknown,
        other => {
            return Err(ChainClientError::Other(format!(
                "{prefix}{other:?}",
                prefix = error_prefixes::ADAPTER_UNRECOGNIZED_STATUS,
            )));
        }
    };
    Ok(AnchorStatusReport {
        status,
        included_at_height: parsed.included_at_height,
        code: parsed.code,
        reason: parsed.reason,
    })
}

/// Stage 13.9 — query a single anchor's chain-side status with
/// the richer report fields (`included_at_height`, `code`,
/// opaque `reason`). Calls the same single-record RPC as
/// [`query_anchor_status`] but returns the unprojected report
/// so callers can surface chain metadata in event lines.
pub(crate) fn query_anchor_status_report<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    tx_id: &str,
) -> std::result::Result<AnchorStatusReport, ChainClientError> {
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
    dto_to_status_report(parsed)
}

/// Stage 13.9 — batch status RPC for many `tx_id`s at once.
///
/// Wire shape:
/// ```text
/// "method": "sum_getIntegrityEvidenceAnchorStatusBatch"
/// "params": [[ "0x...", "0x...", ... ]]
/// ```
/// (one positional param = the array of hashes).
///
/// Per-call constraints:
/// - The caller (reconcile workflow) chunks at
///   [`ANCHOR_STATUS_BATCH_MAX`] before this function is invoked.
///   This function refuses oversize chunks via debug_assert.
/// - Response order MUST match request order. Verified via
///   [`canonicalize_tx_hash`] so cosmetic `0x` / case
///   differences are tolerated.
/// - Length mismatch / order mismatch → `Malformed` (mapped to
///   `chain_response_malformed` upstream).
pub(crate) fn query_anchor_status_batch<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    tx_ids: &[String],
) -> std::result::Result<Vec<BatchStatusItem>, ChainClientError> {
    debug_assert!(
        tx_ids.len() <= ANCHOR_STATUS_BATCH_MAX,
        "chunk caller must split at {ANCHOR_STATUS_BATCH_MAX}"
    );
    if tx_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Positional array form: params = [[..hashes..]] (Stage 13.9
    // REJECT-fix Finding 1).
    let result = client.transport().call(
        "sum_getIntegrityEvidenceAnchorStatusBatch",
        serde_json::json!([tx_ids]),
    )?;

    let items: Vec<BatchItemDto> = serde_json::from_value(result).map_err(|e| {
        ChainClientError::Other(format!(
            "{prefix}{e}",
            prefix = error_prefixes::ADAPTER_MALFORMED_STATUS_BATCH_RESP,
        ))
    })?;

    if items.len() != tx_ids.len() {
        return Err(ChainClientError::Other(format!(
            "{prefix}requested {} hashes; got {}",
            tx_ids.len(),
            items.len(),
            prefix = error_prefixes::ADAPTER_BATCH_LENGTH_MISMATCH,
        )));
    }

    let mut out = Vec::with_capacity(items.len());
    for (idx, dto) in items.into_iter().enumerate() {
        // Canonical-form order verification — Stage 13.9
        // implementation lock. Cosmetic differences (`0x`
        // prefix / case) must not cause false
        // `chain_response_malformed`; AND malformed input
        // hashes echoed back verbatim must not cause it either
        // (the chain's per-item `error` field is the right
        // surface for malformed inputs, NOT a whole-batch
        // refusal).
        //
        // Accept order if:
        //   - both sides canonicalize and the canonical forms
        //     match, OR
        //   - the raw request and raw response strings are
        //     equal (covers the malformed-input echo case).
        let requested = &tx_ids[idx];
        let req_canon = canonicalize_tx_hash(requested);
        let resp_canon = canonicalize_tx_hash(&dto.tx_hash);
        let in_order = match (&req_canon, &resp_canon) {
            (Some(a), Some(b)) if a == b => true,
            _ => requested == &dto.tx_hash,
        };
        if !in_order {
            return Err(ChainClientError::Other(format!(
                "{prefix}position {idx}: requested {requested}, got {response}",
                prefix = error_prefixes::ADAPTER_BATCH_ORDER_MISMATCH,
                response = dto.tx_hash,
            )));
        }
        let report = match dto.result {
            Some(parsed) => Some(dto_to_status_report(parsed)?),
            None => None,
        };
        out.push(BatchStatusItem {
            tx_hash: dto.tx_hash,
            result: report,
            error: dto.error,
        });
    }
    Ok(out)
}

/// Stage 13.9 — by-tuple lookup.
///
/// Wire shape (positional, NOT named):
/// ```text
/// "method": "sum_getIntegrityEvidenceAnchorByTuple"
/// "params": [
///   anchor_schema_version,      // u32
///   artifact_kind_tag,           // u32 (bincode-1 enum discriminant)
///   artifact_schema_version,     // u32
///   artifact_hash_0x_hex,        // String "0x..."
///   signer_pubkey_0x_hex         // String "0x..."
/// ]
/// ```
/// Returns `Ok(None)` when the chain's `result` is `null` (no
/// chain anchor for the tuple).
pub(crate) fn lookup_anchor_by_tuple<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    anchor_schema_version: u32,
    artifact_kind: AnchoredArtifactKind,
    artifact_schema_version: u32,
    artifact_hash: &[u8; 32],
    signer_pubkey: &[u8; 32],
) -> std::result::Result<Option<TupleLookupResult>, ChainClientError> {
    let artifact_hash_hex = format_0x_hex_32(artifact_hash);
    let signer_pubkey_hex = format_0x_hex_32(signer_pubkey);
    let result = client.transport().call(
        "sum_getIntegrityEvidenceAnchorByTuple",
        serde_json::json!([
            anchor_schema_version,
            artifact_kind.to_chain_tag_u32(),
            artifact_schema_version,
            artifact_hash_hex,
            signer_pubkey_hex,
        ]),
    )?;
    if result.is_null() {
        return Ok(None);
    }
    let parsed: ByTupleResult = serde_json::from_value(result).map_err(|e| {
        ChainClientError::Other(format!(
            "{prefix}{e}",
            prefix = error_prefixes::ADAPTER_MALFORMED_BY_TUPLE_RESP,
        ))
    })?;
    Ok(Some(TupleLookupResult {
        tx_hash: parsed.tx_hash,
        included_at_height: parsed.included_at_height,
    }))
}

fn format_0x_hex_32(bytes: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(2 + 64);
    out.push_str("0x");
    for b in bytes {
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

#[cfg(all(test, feature = "submit"))]
mod tests {
    use super::*;

    #[test]
    fn to_0x_hex_pads_each_byte_to_two_chars() {
        assert_eq!(to_0x_hex(&[0x00, 0xab, 0xff]), "0x00abff");
        assert_eq!(to_0x_hex(&[]), "0x");
    }

    #[test]
    fn format_0x_hex_32_produces_64_lower_hex_with_prefix() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0xab;
        bytes[31] = 0x12;
        let hex = format_0x_hex_32(&bytes);
        assert_eq!(hex.len(), 66);
        assert!(hex.starts_with("0xab"));
        assert!(hex.ends_with("12"));
        assert!(hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f' | b'x')));
    }

    #[test]
    fn dto_to_status_report_uses_failed_reason_null_fallback() {
        let dto = AnchorStatusResult {
            status: "failed".to_string(),
            included_at_height: None,
            code: Some(60),
            reason: None,
        };
        let report = dto_to_status_report(dto).unwrap();
        match &report.status {
            AnchorStatus::Failed { reason } => {
                assert_eq!(reason, FAILED_REASON_NULL_FALLBACK);
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(report.code, Some(60));
    }

    #[test]
    fn batch_chunk_max_is_100() {
        assert_eq!(ANCHOR_STATUS_BATCH_MAX, 100);
    }
}
