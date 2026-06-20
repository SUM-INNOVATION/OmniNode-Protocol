//! Phase 5 Stage 13.9 — integration tests for the
//! batch-reconcile + by-tuple-lookup contract on the
//! `EvidenceAnchorChainClient` trait. Hermetic; uses an
//! in-test stub client.

use omni_zkml::{
    anchor_signer_pubkey_bytes, build_anchor_digest, lookup_anchor_by_tuple_workflow,
    reconcile_evidence_anchors_workflow, submit_evidence_anchor_workflow, AnchorRecord,
    AnchorSelector, AnchorStatus, AnchorStatusReport, AnchoredArtifactKind,
    BatchStatusItem, ChainClientError, EvidenceAnchorChainClient,
    EvidenceAnchorError, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
    StubEvidenceAnchorChainClient, TupleLookupOutcome, TupleLookupResult,
    VerifiedWrapperMetadata, FAILED_REASON_NULL_FALLBACK,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

// ── Test helpers ─────────────────────────────────────────────────────────────

fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry) {
    let dir = tempfile::tempdir().unwrap();
    let reg = LocalEvidenceAnchorRegistry::open(dir.path().join("anchors")).unwrap();
    (dir, reg)
}

fn seed(
    reg: &LocalEvidenceAnchorRegistry,
    stub: &StubEvidenceAnchorChainClient,
    seed: [u8; 32],
    marker: u8,
) -> AnchorRecord {
    let raw = vec![marker; 32];
    let metadata = VerifiedWrapperMetadata {
        artifact_schema_version: 1,
        signer_pubkey: anchor_signer_pubkey_bytes(&seed).unwrap(),
        signed_at_utc_unix: 1_750_000_000,
    };
    let digest = build_anchor_digest(&metadata, &raw);
    submit_evidence_anchor_workflow(reg, stub, digest, &seed).unwrap()
}

/// Fake chain client that overrides `query_anchor_status_batch`
/// with per-item containment (matches real `omni-sumchain`
/// semantics: per-item errors land in the per-item slot; the
/// batch as a whole succeeds).
#[derive(Default)]
struct BatchCapableFakeClient {
    statuses: RefCell<HashMap<String, AnchorStatusReport>>,
    per_item_errors: RefCell<HashMap<String, String>>,
    batch_call_count: RefCell<usize>,
    by_tuple_result: RefCell<Option<TupleLookupResult>>,
}

impl BatchCapableFakeClient {
    fn new() -> Self {
        Self::default()
    }
    fn set_status_report(&self, tx_id: &str, report: AnchorStatusReport) {
        self.statuses
            .borrow_mut()
            .insert(tx_id.to_string(), report);
    }
    fn set_per_item_error(&self, tx_id: &str, msg: &str) {
        self.per_item_errors
            .borrow_mut()
            .insert(tx_id.to_string(), msg.to_string());
    }
    fn set_by_tuple_found(&self, result: TupleLookupResult) {
        *self.by_tuple_result.borrow_mut() = Some(result);
    }
    fn batch_call_count(&self) -> usize {
        *self.batch_call_count.borrow()
    }
}

impl EvidenceAnchorChainClient for BatchCapableFakeClient {
    fn submit_anchor(
        &self,
        _: &omni_zkml::IntegrityEvidenceAnchorTxData,
    ) -> Result<omni_zkml::AnchorSubmissionReceipt, ChainClientError> {
        unimplemented!("not used in reconcile tests")
    }
    fn query_anchor_status(&self, tx_id: &str) -> Result<AnchorStatus, ChainClientError> {
        Ok(self
            .statuses
            .borrow()
            .get(tx_id)
            .map(|r| r.status.clone())
            .unwrap_or(AnchorStatus::Submitted))
    }
    fn query_anchor_status_batch(
        &self,
        tx_ids: &[String],
    ) -> Result<Vec<BatchStatusItem>, ChainClientError> {
        *self.batch_call_count.borrow_mut() += 1;
        let mut out = Vec::with_capacity(tx_ids.len());
        for tx_id in tx_ids {
            if let Some(err) = self.per_item_errors.borrow().get(tx_id) {
                out.push(BatchStatusItem {
                    tx_hash: tx_id.clone(),
                    result: None,
                    error: Some(err.clone()),
                });
                continue;
            }
            let report = self
                .statuses
                .borrow()
                .get(tx_id)
                .cloned()
                .unwrap_or(AnchorStatusReport {
                    status: AnchorStatus::Submitted,
                    included_at_height: None,
                    code: None,
                    reason: None,
                });
            out.push(BatchStatusItem {
                tx_hash: tx_id.clone(),
                result: Some(report),
                error: None,
            });
        }
        Ok(out)
    }
    fn lookup_anchor_by_tuple(
        &self,
        _: u32,
        _: AnchoredArtifactKind,
        _: u32,
        _: &[u8; 32],
        _: &[u8; 32],
    ) -> Result<Option<TupleLookupResult>, ChainClientError> {
        Ok(self.by_tuple_result.borrow().clone())
    }
}

/// Fake chain client that DOESN'T override the batch method —
/// exercises the default trait fallback (fail-fast).
#[derive(Default)]
struct DefaultFallbackFakeClient {
    statuses: RefCell<HashMap<String, AnchorStatus>>,
    transport_errors: RefCell<HashMap<String, String>>,
    single_call_count: RefCell<usize>,
}

impl DefaultFallbackFakeClient {
    fn new() -> Self {
        Self::default()
    }
    fn set_status(&self, tx_id: &str, status: AnchorStatus) {
        self.statuses
            .borrow_mut()
            .insert(tx_id.to_string(), status);
    }
    fn set_transport_error(&self, tx_id: &str, msg: &str) {
        self.transport_errors
            .borrow_mut()
            .insert(tx_id.to_string(), msg.to_string());
    }
    fn single_call_count(&self) -> usize {
        *self.single_call_count.borrow()
    }
}

impl EvidenceAnchorChainClient for DefaultFallbackFakeClient {
    fn submit_anchor(
        &self,
        _: &omni_zkml::IntegrityEvidenceAnchorTxData,
    ) -> Result<omni_zkml::AnchorSubmissionReceipt, ChainClientError> {
        unimplemented!("not used")
    }
    fn query_anchor_status(&self, tx_id: &str) -> Result<AnchorStatus, ChainClientError> {
        *self.single_call_count.borrow_mut() += 1;
        if let Some(msg) = self.transport_errors.borrow().get(tx_id) {
            return Err(ChainClientError::Other(msg.clone()));
        }
        Ok(self
            .statuses
            .borrow()
            .get(tx_id)
            .cloned()
            .unwrap_or(AnchorStatus::Submitted))
    }
    // No batch / no by-tuple overrides — default trait impl runs.
}

// ── Stage 13.9 v2 lock — REJECT-fix Finding 3 + Finding 6 + Finding 7 ──────

#[test]
fn default_trait_impl_returns_err_on_first_transport_error_for_batch_fallback() {
    // REJECT-fix Finding 3 — the default batch fallback fails
    // fast on the first per-call error (symmetric with real
    // chunk-level batch failure).
    let client = DefaultFallbackFakeClient::new();
    client.set_transport_error("anchor-1", "HTTP transport failure: timed out");
    client.set_status("anchor-2", AnchorStatus::Included);
    let tx_ids: Vec<String> = vec!["anchor-1".to_string(), "anchor-2".to_string()];
    let result = client.query_anchor_status_batch(&tx_ids);
    assert!(result.is_err(), "fallback fails fast on first call error");
    // The second tx_id was never queried.
    assert_eq!(
        client.single_call_count(),
        1,
        "fail-fast: stops at first error, does not continue"
    );
}

#[test]
fn default_trait_impl_provides_single_status_report_wrapping_query_anchor_status() {
    // REJECT-fix Finding 6 — the default
    // `query_anchor_status_report` impl wraps
    // `query_anchor_status` with `None` chain fields.
    let stub = StubEvidenceAnchorChainClient::new();
    stub.set_status_for("anchor-1", AnchorStatus::Finalized);
    let report = stub.query_anchor_status_report("anchor-1").unwrap();
    assert!(matches!(report.status, AnchorStatus::Finalized));
    assert_eq!(report.included_at_height, None);
    assert_eq!(report.code, None);
    assert_eq!(report.reason, None);
}

#[test]
fn default_trait_impl_returns_unsupported_error_for_by_tuple() {
    let stub = StubEvidenceAnchorChainClient::new();
    let err = stub
        .lookup_anchor_by_tuple(
            1,
            AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
            1,
            &[0u8; 32],
            &[0u8; 32],
        )
        .unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.contains("not supported"),
        "default impl returns a not-supported error: {msg}"
    );
}

// ── Reconcile — batch + transition table ─────────────────────────────────────

#[test]
fn reconcile_uses_batch_method_when_client_overrides_it() {
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let r1 = seed(&reg, &stub, [1u8; 32], 0x11);
    let r2 = seed(&reg, &stub, [2u8; 32], 0x22);

    let real = BatchCapableFakeClient::new();
    real.set_status_report(
        &r1.receipt.tx_id,
        AnchorStatusReport {
            status: AnchorStatus::Finalized,
            included_at_height: Some(100),
            code: None,
            reason: None,
        },
    );
    real.set_status_report(
        &r2.receipt.tx_id,
        AnchorStatusReport {
            status: AnchorStatus::Included,
            included_at_height: Some(50),
            code: None,
            reason: None,
        },
    );
    let out = reconcile_evidence_anchors_workflow(&reg, &real).unwrap();
    assert_eq!(out.len(), 2);
    // Single chunk → single batch call.
    assert_eq!(real.batch_call_count(), 1);
    // Outcomes carry the rich chain fields.
    let finalized_outcome = out
        .iter()
        .find(|(h, _)| *h == r1.artifact_hash_hex)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert!(matches!(
        finalized_outcome.chain_status,
        AnchorStatus::Finalized
    ));
    assert_eq!(finalized_outcome.included_at_height, Some(100));
    assert!(finalized_outcome.local_status_transitioned);
}

#[test]
fn reconcile_chunks_at_100_tx_ids() {
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    // Seed 101 records (101 distinct seeds + markers).
    for i in 0..101u8 {
        let mut seed_bytes = [0u8; 32];
        seed_bytes[0] = i.wrapping_add(1);
        let _ = seed(&reg, &stub, seed_bytes, i.wrapping_add(0x10));
    }
    let real = BatchCapableFakeClient::new();
    // Default `Submitted` for every tx_id — observation-only,
    // no transitions.
    let out = reconcile_evidence_anchors_workflow(&reg, &real).unwrap();
    assert_eq!(out.len(), 101);
    // 101 records → 2 chunks (100 + 1).
    assert_eq!(real.batch_call_count(), 2);
}

#[test]
fn submitted_from_chain_is_observation_only_for_local_records() {
    // Stage 13.9 Finding 7 lock — `Submitted` from chain
    // leaves local unchanged regardless of prior state. No
    // reorg-aware downgrade in 13.x.
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let r = seed(&reg, &stub, [3u8; 32], 0x33);
    // Manually transition local to Included to test no-downgrade.
    reg.update_status(&r.artifact_hash_hex, LocalAnchorStatus::Included)
        .unwrap();
    let real = BatchCapableFakeClient::new();
    real.set_status_report(
        &r.receipt.tx_id,
        AnchorStatusReport {
            status: AnchorStatus::Submitted,
            included_at_height: None,
            code: None,
            reason: None,
        },
    );
    let out = reconcile_evidence_anchors_workflow(&reg, &real).unwrap();
    assert_eq!(out.len(), 1);
    let outcome = out[0].1.as_ref().unwrap();
    assert!(matches!(outcome.chain_status, AnchorStatus::Submitted));
    assert!(
        !outcome.local_status_transitioned,
        "Submitted from chain MUST NOT downgrade Included local record"
    );
    let reloaded = reg.load_by_artifact_hash(&r.artifact_hash_hex).unwrap().unwrap();
    assert_eq!(reloaded.status, LocalAnchorStatus::Included);
}

#[test]
fn reconcile_unknown_status_remains_observation_only() {
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let r = seed(&reg, &stub, [4u8; 32], 0x44);
    let real = BatchCapableFakeClient::new();
    real.set_status_report(
        &r.receipt.tx_id,
        AnchorStatusReport {
            status: AnchorStatus::Unknown,
            included_at_height: None,
            code: None,
            reason: None,
        },
    );
    let out = reconcile_evidence_anchors_workflow(&reg, &real).unwrap();
    let outcome = out[0].1.as_ref().unwrap();
    assert!(!outcome.local_status_transitioned);
    let reloaded = reg.load_by_artifact_hash(&r.artifact_hash_hex).unwrap().unwrap();
    assert_eq!(reloaded.status, LocalAnchorStatus::Submitted);
}

#[test]
fn reconcile_failed_carries_code_and_opaque_reason_in_outcome() {
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let r = seed(&reg, &stub, [5u8; 32], 0x55);
    let real = BatchCapableFakeClient::new();
    real.set_status_report(
        &r.receipt.tx_id,
        AnchorStatusReport {
            status: AnchorStatus::Failed {
                reason: "duplicate 5-tuple".to_string(),
            },
            included_at_height: None,
            code: Some(61),
            reason: Some("duplicate 5-tuple".to_string()),
        },
    );
    let out = reconcile_evidence_anchors_workflow(&reg, &real).unwrap();
    let outcome = out[0].1.as_ref().unwrap();
    assert_eq!(outcome.code, Some(61));
    assert!(matches!(outcome.chain_status, AnchorStatus::Failed { .. }));
}

#[test]
fn reconcile_real_batch_per_item_error_containment_keeps_sibling_records_succeeding() {
    // With a real batch-capable client (per-item errors),
    // sibling records still succeed even when one tx_id fails.
    // This is the contract the user wanted preserved.
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let r1 = seed(&reg, &stub, [6u8; 32], 0x66);
    let r2 = seed(&reg, &stub, [7u8; 32], 0x77);
    let real = BatchCapableFakeClient::new();
    real.set_per_item_error(&r1.receipt.tx_id, "Invalid hash");
    real.set_status_report(
        &r2.receipt.tx_id,
        AnchorStatusReport {
            status: AnchorStatus::Finalized,
            included_at_height: Some(99),
            code: None,
            reason: None,
        },
    );
    let out = reconcile_evidence_anchors_workflow(&reg, &real).unwrap();
    let oks = out.iter().filter(|(_, r)| r.is_ok()).count();
    let errs = out.iter().filter(|(_, r)| r.is_err()).count();
    assert_eq!(oks, 1, "r2 still succeeds");
    assert_eq!(errs, 1, "r1 surfaces as per-record Err from per-item error");
}

// ── By-tuple workflow ────────────────────────────────────────────────────────

#[test]
fn lookup_anchor_by_tuple_workflow_returns_found_when_chain_has_canonical_anchor() {
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let r = seed(&reg, &stub, [8u8; 32], 0x88);
    let real = BatchCapableFakeClient::new();
    let canonical = "0x".to_string() + &"f".repeat(64);
    real.set_by_tuple_found(TupleLookupResult {
        tx_hash: canonical.clone(),
        included_at_height: 4807033,
    });
    let outcome = lookup_anchor_by_tuple_workflow(
        &reg,
        &real,
        AnchorSelector::ArtifactHashHex(&r.artifact_hash_hex),
    )
    .unwrap();
    match outcome {
        TupleLookupOutcome::Found {
            canonical_tx_hash,
            included_at_height,
            local_record_tx_id,
        } => {
            assert_eq!(canonical_tx_hash, canonical);
            assert_eq!(included_at_height, 4807033);
            assert_eq!(local_record_tx_id, r.receipt.tx_id);
        }
        other => panic!("expected Found, got {other:?}"),
    }
}

#[test]
fn lookup_anchor_by_tuple_workflow_returns_not_found_when_chain_returns_null() {
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let r = seed(&reg, &stub, [9u8; 32], 0x99);
    let real = BatchCapableFakeClient::new();
    // Don't call set_by_tuple_found → by_tuple_result is None.
    let outcome = lookup_anchor_by_tuple_workflow(
        &reg,
        &real,
        AnchorSelector::ArtifactHashHex(&r.artifact_hash_hex),
    )
    .unwrap();
    assert!(matches!(outcome, TupleLookupOutcome::NotFound));
}

#[test]
fn lookup_anchor_by_tuple_workflow_does_not_mutate_registry() {
    let (_dir, reg) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let r = seed(&reg, &stub, [10u8; 32], 0xaa);
    let real = BatchCapableFakeClient::new();
    real.set_by_tuple_found(TupleLookupResult {
        tx_hash: "0x".to_string() + &"a".repeat(64),
        included_at_height: 1,
    });
    let pre_tx_id = r.receipt.tx_id.clone();
    let pre_status = r.status.clone();
    let _ = lookup_anchor_by_tuple_workflow(
        &reg,
        &real,
        AnchorSelector::ArtifactHashHex(&r.artifact_hash_hex),
    )
    .unwrap();
    let post = reg.load_by_artifact_hash(&r.artifact_hash_hex).unwrap().unwrap();
    assert_eq!(post.receipt.tx_id, pre_tx_id, "tx_id unchanged");
    assert_eq!(post.status, pre_status, "status unchanged");
}

// ── Failed-reason null fallback (implementation lock) ────────────────────────

#[test]
fn failed_reason_null_fallback_constant_is_stable() {
    assert_eq!(
        FAILED_REASON_NULL_FALLBACK,
        "chain returned failed with no reason"
    );
}

// ── Selector miss inside by-tuple workflow ───────────────────────────────────

#[test]
fn lookup_anchor_by_tuple_workflow_refuses_when_selector_misses_with_anchor_not_found() {
    let (_dir, reg) = fresh_registry();
    let real = BatchCapableFakeClient::new();
    let err = lookup_anchor_by_tuple_workflow(
        &reg,
        &real,
        AnchorSelector::TxId("nonexistent-tx"),
    )
    .unwrap_err();
    match err {
        EvidenceAnchorError::AnchorNotFound { .. } => {}
        other => panic!("expected AnchorNotFound, got {other:?}"),
    }
}

// ── Trait method counts and back-compat ──────────────────────────────────────

#[test]
fn stub_anchor_client_compiles_without_overriding_new_methods() {
    // Compile-time pin — `StubEvidenceAnchorChainClient` from
    // Stage 13.0 keeps compiling without any change. The default
    // trait impls cover the three new Stage 13.9 methods.
    let stub = StubEvidenceAnchorChainClient::new();
    let _: Result<AnchorStatus, _> = stub.query_anchor_status("anchor-x");
    let _: Result<AnchorStatusReport, _> = stub.query_anchor_status_report("anchor-x");
    let _: Result<Vec<BatchStatusItem>, _> =
        stub.query_anchor_status_batch(&["anchor-x".to_string()]);
    let _: Result<Option<TupleLookupResult>, _> = stub.lookup_anchor_by_tuple(
        1,
        AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
        1,
        &[0u8; 32],
        &[0u8; 32],
    );
}

// Unused import to keep PathBuf in scope for potential test
// growth without a warning churn.
#[allow(dead_code)]
fn _path_buf_in_scope() -> PathBuf {
    PathBuf::new()
}
