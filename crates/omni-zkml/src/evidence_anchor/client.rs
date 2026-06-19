//! Phase 5 Stage 13.0 — chain client trait + in-memory stub for
//! integrity-evidence anchoring.
//!
//! Mirrors the Stage 5 [`crate::chain::ChainClient`] shape: a
//! synchronous trait that future chain adapters will implement,
//! plus a hermetic stub used by tests and the CLI's stub mode.
//!
//! Stage 13.0 deliberately ships **no real SUM Chain adapter**.
//! The wire spec frozen by [`crate::evidence_anchor::wire`] is
//! reviewed by the chain team before Stage 13.1 plugs in the real
//! submission path.
//!
//! ## Status model
//!
//! Anchor status mirrors the chain v1 five-state model
//! ([`crate::chain::AttestationStatus`]) verbatim:
//! `Submitted | Included | Finalized | Failed { reason } | Unknown`.
//! Keeping the model identical lets the orchestration layer reuse
//! the Stage 5 semantics (no chain-side `Dropped`, `Unknown` is
//! observation-only) without divergence.

use std::cell::RefCell;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::ChainClientError;
use crate::evidence_anchor::wire::IntegrityEvidenceAnchorTxData;

// ── Chain client trait ────────────────────────────────────────────────────────

/// Stage 13.0 anchor chain-client trait. Future chain adapters
/// implement this to submit and query anchor records. **No real
/// RPC, no tx encoding in Stage 13.0.**
///
/// `tx_id` is opaque (string). The stub uses a deterministic
/// `"anchor-{counter:08x}-{artifact_hash_hex[..12]}"` form; a real
/// adapter would use the chain's native tx hash.
///
/// ## Stage 13.9 extension
///
/// Three default-impl'd methods were added — `query_anchor_status_report`,
/// `query_anchor_status_batch`, `lookup_anchor_by_tuple` — for batch
/// reads and by-tuple lookup. The defaults keep every existing
/// `EvidenceAnchorChainClient` impl compiling unchanged:
///
/// - `query_anchor_status_report` wraps `query_anchor_status` with
///   `Option<u64>` / `Option<u32>` / `Option<String>` chain fields
///   defaulted to `None`. Real adapter overrides for richer info.
/// - `query_anchor_status_batch` loops `query_anchor_status` and
///   **fails fast on the first `ChainClientError`** (Stage 13.9
///   REJECT-fix Finding 3 — symmetric with real chunk-level batch
///   failure; reconcile fans out per-record from the `Err`).
/// - `lookup_anchor_by_tuple` returns
///   `Err(ChainClientError::Other("by-tuple lookup not supported by this client"))`.
pub trait EvidenceAnchorChainClient {
    /// Submit a signed anchor wire payload. Returns an
    /// implementation-specific receipt on success.
    fn submit_anchor(
        &self,
        tx_data: &IntegrityEvidenceAnchorTxData,
    ) -> std::result::Result<AnchorSubmissionReceipt, ChainClientError>;

    /// Query the chain-side status of a previously-submitted
    /// anchor. Keyed by `tx_id`, matching the Stage 5.1 chain-
    /// keying contract.
    fn query_anchor_status(
        &self,
        tx_id: &str,
    ) -> std::result::Result<AnchorStatus, ChainClientError>;

    /// Stage 13.9 — single-record richer status report. Default
    /// impl wraps [`Self::query_anchor_status`] and fills the
    /// chain-only fields with `None`. Real adapter overrides for
    /// actual `included_at_height` / `code` / `reason`.
    fn query_anchor_status_report(
        &self,
        tx_id: &str,
    ) -> std::result::Result<AnchorStatusReport, ChainClientError> {
        let status = self.query_anchor_status(tx_id)?;
        Ok(AnchorStatusReport {
            status,
            included_at_height: None,
            code: None,
            reason: None,
        })
    }

    /// Stage 13.9 — batch status read. Default impl loops
    /// [`Self::query_anchor_status`] and **fails fast** on the
    /// first per-call transport / client error. This is
    /// semantically symmetric with real chunk-level batch
    /// failure — reconcile's chunk-level handler fans out per-
    /// record from the `Err`, not from item errors (per-item
    /// errors only originate from a real batch RPC returning
    /// per-item `error` fields).
    ///
    /// Implementations that send a real batch RPC override this
    /// method.
    fn query_anchor_status_batch(
        &self,
        tx_ids: &[String],
    ) -> std::result::Result<Vec<BatchStatusItem>, ChainClientError> {
        let mut out = Vec::with_capacity(tx_ids.len());
        for tx_id in tx_ids {
            let status = self.query_anchor_status(tx_id)?;
            out.push(BatchStatusItem {
                tx_hash: tx_id.clone(),
                result: Some(AnchorStatusReport {
                    status,
                    included_at_height: None,
                    code: None,
                    reason: None,
                }),
                error: None,
            });
        }
        Ok(out)
    }

    /// Stage 13.9 — by-tuple lookup. Default impl returns an
    /// `Err("not supported")` so existing stubs that never call
    /// this method keep compiling. Real adapter overrides.
    ///
    /// Tuple field order matches the chain contract:
    /// `(anchor_schema_version, artifact_kind, artifact_schema_version,
    /// artifact_hash, signer_pubkey)`.
    fn lookup_anchor_by_tuple(
        &self,
        anchor_schema_version: u32,
        artifact_kind: crate::evidence_anchor::wire::AnchoredArtifactKind,
        artifact_schema_version: u32,
        artifact_hash: &[u8; 32],
        signer_pubkey: &[u8; 32],
    ) -> std::result::Result<Option<TupleLookupResult>, ChainClientError> {
        let _ = (
            anchor_schema_version,
            artifact_kind,
            artifact_schema_version,
            artifact_hash,
            signer_pubkey,
        );
        Err(ChainClientError::Other(
            "by-tuple lookup not supported by this client".to_string(),
        ))
    }
}

// ── Receipt + status ──────────────────────────────────────────────────────────

/// Opaque chain-side identifier for a submitted anchor. Format
/// is implementation-defined (stub mode uses a deterministic
/// string; Stage 13.1 swaps in real chain tx hashes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorSubmissionReceipt {
    pub tx_id: String,

    /// Optional implementation-specific diagnostic. Stub clients
    /// use this for test assertions; a real chain client may
    /// leave it `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Chain-side status mirror of [`crate::chain::AttestationStatus`].
/// Kept identical so the orchestration layer reuses Stage 5
/// semantics; see `crate::chain` docs for the full state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnchorStatus {
    Submitted,
    Included,
    Finalized,
    Failed {
        reason: String,
    },
    /// Chain reports it does not recognize this `tx_id`. Stage
    /// 13.0 leaves the record unchanged on `Unknown`; staleness
    /// detection is a future-stage concern.
    Unknown,
}

// ── Stage 13.9 — richer chain-status DTOs ────────────────────────────────────

/// Stage 13.9 — single-record richer status response. Carries
/// the chain-only metadata (`included_at_height` / `code` /
/// `reason`) without growing the closed [`AnchorStatus`] enum.
/// The Stage 13.0 surface is unchanged; new callers consume
/// this type when they want richer event-line output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorStatusReport {
    pub status: AnchorStatus,
    pub included_at_height: Option<u64>,
    /// Stable on `Failed` per chain contract:
    /// `60` not activated, `61` duplicate 5-tuple,
    /// `62` invalid submitter signature, `63` `tx.from !=
    /// address(signer_pubkey)`. Other failures may carry
    /// `None`. Operators read `code` as a numeric token, not as
    /// a parsed enum — the closed-set lives in operator-runbook
    /// docs.
    pub code: Option<u32>,
    /// Opaque human-readable text. Never parsed; surfaced
    /// verbatim. When the chain returns `failed` with
    /// `reason: null`, reconcile uses
    /// [`FAILED_REASON_NULL_FALLBACK`] so the existing
    /// `AnchorStatus::Failed { reason: String }` shape stays
    /// compatible (Stage 13.9 implementation lock).
    pub reason: Option<String>,
}

/// Stage 13.9 — per-item entry of the batch-status response.
/// `tx_hash` is the chain-echoed hash (used for response-order
/// verification). Exactly one of `result` / `error` is non-`None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchStatusItem {
    pub tx_hash: String,
    pub result: Option<AnchorStatusReport>,
    pub error: Option<String>,
}

/// Stage 13.9 — by-tuple lookup successful result. Returned in
/// the `Some(...)` arm of `lookup_anchor_by_tuple` when the
/// chain has an anchor matching the 5-tuple.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TupleLookupResult {
    /// Canonical `0x`-prefixed 32-byte lowercase hex of the
    /// final committed `TransactionV2` envelope.
    pub tx_hash: String,
    pub included_at_height: u64,
}

/// Stage 13.9 implementation lock — fallback `reason` text when
/// the chain returns `failed` with `reason: null`. Keeps
/// [`AnchorStatus::Failed { reason: String }`] backward-
/// compatible (no Stage 13.0 schema change) while still
/// surfacing the chain's `code` independently via event lines.
pub const FAILED_REASON_NULL_FALLBACK: &str =
    "chain returned failed with no reason";

/// Stage 13.9 batch chunk size — chain contract max. Mirrors
/// the omni-sumchain `ANCHOR_STATUS_BATCH_MAX`. The reconcile
/// workflow chunks at this size before calling
/// [`EvidenceAnchorChainClient::query_anchor_status_batch`].
pub const ANCHOR_STATUS_BATCH_MAX: usize = 100;

/// Stage 13.9 implementation lock — canonicalize a tx hash for
/// batch-response order verification. Strips an optional `0x`
/// prefix and lowercases. Returns `None` when the canonical
/// form is not exactly 64 lower-hex characters (caller routes
/// this through `chain_response_malformed`).
///
/// This handles the practical case where local `tx_id`s and
/// the chain's echoed `tx_hash` differ only in `0x` prefix or
/// case — a valid response should NOT be falsely rejected for
/// cosmetic differences.
pub fn canonicalize_tx_hash(s: &str) -> Option<String> {
    let stripped = s.strip_prefix("0x").unwrap_or(s);
    if stripped.len() != 64 {
        return None;
    }
    let lower = stripped.to_ascii_lowercase();
    if !lower
        .bytes()
        .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return None;
    }
    Some(lower)
}

// ── In-memory stub client (tests + CLI stub mode) ─────────────────────────────

/// Hermetic in-memory [`EvidenceAnchorChainClient`] used by the
/// Stage 13.0 CLI in stub mode and by integration tests. Returns
/// deterministic `tx_id`s; tracks per-`tx_id` status outcomes
/// configurable via [`Self::set_status_for`].
pub struct StubEvidenceAnchorChainClient {
    next_counter: RefCell<u32>,
    status_per_tx: RefCell<HashMap<String, AnchorStatus>>,
    default_status: AnchorStatus,
}

impl StubEvidenceAnchorChainClient {
    /// Construct a stub that defaults all queries to
    /// [`AnchorStatus::Submitted`] until overridden via
    /// [`Self::set_status_for`].
    pub fn new() -> Self {
        Self {
            next_counter: RefCell::new(0),
            status_per_tx: RefCell::new(HashMap::new()),
            default_status: AnchorStatus::Submitted,
        }
    }

    /// Override the status returned by future
    /// [`Self::query_anchor_status`] calls for a specific
    /// `tx_id`.
    pub fn set_status_for(&self, tx_id: &str, status: AnchorStatus) {
        self.status_per_tx
            .borrow_mut()
            .insert(tx_id.to_string(), status);
    }
}

impl Default for StubEvidenceAnchorChainClient {
    fn default() -> Self {
        Self::new()
    }
}

impl EvidenceAnchorChainClient for StubEvidenceAnchorChainClient {
    fn submit_anchor(
        &self,
        tx_data: &IntegrityEvidenceAnchorTxData,
    ) -> std::result::Result<AnchorSubmissionReceipt, ChainClientError> {
        let mut counter = self.next_counter.borrow_mut();
        let n = *counter;
        *counter += 1;
        let hash_hex =
            crate::evidence_anchor::wire::anchor_hex_lower(&tx_data.digest.artifact_hash);
        let prefix: String = hash_hex.chars().take(12).collect();
        let tx_id = format!("anchor-{n:08x}-{prefix}");
        Ok(AnchorSubmissionReceipt {
            tx_id,
            note: Some("stub".to_string()),
        })
    }

    fn query_anchor_status(
        &self,
        tx_id: &str,
    ) -> std::result::Result<AnchorStatus, ChainClientError> {
        if let Some(s) = self.status_per_tx.borrow().get(tx_id).cloned() {
            return Ok(s);
        }
        Ok(self.default_status.clone())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_anchor::wire::{
        AnchoredArtifactKind, INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
        IntegrityEvidenceAnchorDigest, anchor_signer_pubkey_bytes, sign_anchor_digest,
    };

    fn build_tx(seed: [u8; 32]) -> IntegrityEvidenceAnchorTxData {
        let pubkey = anchor_signer_pubkey_bytes(&seed).unwrap();
        let digest = IntegrityEvidenceAnchorDigest {
            anchor_schema_version: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
            artifact_kind: AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
            artifact_schema_version: 1,
            artifact_hash: [0x11; 32],
            signer_pubkey: pubkey,
            signed_at_utc_unix: 1_700_000_000,
        };
        let sig = sign_anchor_digest(&seed, &digest).unwrap();
        IntegrityEvidenceAnchorTxData {
            digest,
            submitter_signature: sig,
        }
    }

    #[test]
    fn stub_submit_returns_deterministic_tx_id() {
        let client = StubEvidenceAnchorChainClient::new();
        let tx1 = build_tx([7u8; 32]);
        let receipt1 = client.submit_anchor(&tx1).unwrap();
        // Counter is 0; hash prefix is the first 12 hex chars of [0x11; 32].
        assert_eq!(receipt1.tx_id, "anchor-00000000-111111111111");
        let receipt2 = client.submit_anchor(&tx1).unwrap();
        assert_eq!(receipt2.tx_id, "anchor-00000001-111111111111");
    }

    #[test]
    fn stub_query_defaults_to_submitted() {
        let client = StubEvidenceAnchorChainClient::new();
        let status = client.query_anchor_status("anchor-xyz").unwrap();
        assert!(matches!(status, AnchorStatus::Submitted));
    }

    #[test]
    fn stub_query_uses_configured_status() {
        let client = StubEvidenceAnchorChainClient::new();
        client.set_status_for("anchor-finalized", AnchorStatus::Finalized);
        let status = client.query_anchor_status("anchor-finalized").unwrap();
        assert!(matches!(status, AnchorStatus::Finalized));
    }
}
