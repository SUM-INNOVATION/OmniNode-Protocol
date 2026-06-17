//! Phase 5 Stage 13.2 — reconcile workflow integration tests.
//!
//! Hand-rolled `FakeEvidenceAnchorChainClient` (no transport)
//! mirroring Stage 5.3's `FakeOrchestrationClient` pattern;
//! the `omni-sumchain` `FakeJsonRpcTransport` adapter integration
//! already covers the wire/decode lanes, so this file focuses on
//! the orchestration semantics:
//!
//! - Submitted → Included → Finalized transitions land on the
//!   correct local records.
//! - Chain `Unknown` leaves records unchanged (Stage 5.1
//!   observation-only contract, mirrored verbatim in
//!   reconcile_evidence_anchors_workflow).
//! - Per-record RPC failures land as `Err` entries in the result
//!   vec; the sweep continues.
//! - Non-Submitted / Non-Included records are skipped (omitted
//!   from the result vec).
//! - Empty registry → empty result, zero chain calls.

use std::cell::RefCell;
use std::collections::HashMap;

use omni_zkml::{
    anchor_signer_pubkey_bytes, build_anchor_digest, reconcile_evidence_anchors_workflow,
    submit_evidence_anchor_workflow, AnchorStatus, AnchorSubmissionReceipt, ChainClientError,
    EvidenceAnchorChainClient, IntegrityEvidenceAnchorTxData, LocalAnchorStatus,
    LocalEvidenceAnchorRegistry, StubEvidenceAnchorChainClient, VerifiedWrapperMetadata,
};

// ── Fake client with per-tx status overrides + call log ──────────────────────

struct FakeAnchorClient {
    submit_counter: RefCell<u32>,
    status_by_tx: RefCell<HashMap<String, AnchorStatus>>,
    submit_calls: RefCell<u32>,
    query_calls: RefCell<Vec<String>>,
    query_error_by_tx: RefCell<HashMap<String, ChainClientError>>,
}

impl FakeAnchorClient {
    fn new() -> Self {
        Self {
            submit_counter: RefCell::new(0),
            status_by_tx: RefCell::new(HashMap::new()),
            submit_calls: RefCell::new(0),
            query_calls: RefCell::new(Vec::new()),
            query_error_by_tx: RefCell::new(HashMap::new()),
        }
    }
    fn set_status(&self, tx_id: &str, status: AnchorStatus) {
        self.status_by_tx
            .borrow_mut()
            .insert(tx_id.to_string(), status);
    }
    fn set_query_error(&self, tx_id: &str, err: ChainClientError) {
        self.query_error_by_tx
            .borrow_mut()
            .insert(tx_id.to_string(), err);
    }
    fn query_call_count(&self) -> usize {
        self.query_calls.borrow().len()
    }
}

impl EvidenceAnchorChainClient for FakeAnchorClient {
    fn submit_anchor(
        &self,
        _tx_data: &IntegrityEvidenceAnchorTxData,
    ) -> Result<AnchorSubmissionReceipt, ChainClientError> {
        let mut c = self.submit_counter.borrow_mut();
        *c += 1;
        *self.submit_calls.borrow_mut() += 1;
        Ok(AnchorSubmissionReceipt {
            tx_id: format!("0xfake-{c}"),
            note: None,
        })
    }
    fn query_anchor_status(&self, tx_id: &str) -> Result<AnchorStatus, ChainClientError> {
        self.query_calls.borrow_mut().push(tx_id.to_string());
        if let Some(e) = self.query_error_by_tx.borrow().get(tx_id) {
            return Err(e.clone());
        }
        Ok(self
            .status_by_tx
            .borrow()
            .get(tx_id)
            .cloned()
            .unwrap_or(AnchorStatus::Submitted))
    }
}

// ── Fixtures ──────────────────────────────────────────────────────────────────

fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry) {
    let dir = tempfile::tempdir().unwrap();
    let reg = LocalEvidenceAnchorRegistry::open(dir.path().join("anchors")).unwrap();
    (dir, reg)
}

fn seed_record(
    registry: &LocalEvidenceAnchorRegistry,
    client: &impl EvidenceAnchorChainClient,
    seed: [u8; 32],
    raw_bytes_marker: u8,
) -> String {
    let raw = vec![raw_bytes_marker; 16];
    let metadata = VerifiedWrapperMetadata {
        artifact_schema_version: 1,
        signer_pubkey: anchor_signer_pubkey_bytes(&seed).unwrap(),
        signed_at_utc_unix: 1_750_000_000,
    };
    let digest = build_anchor_digest(&metadata, &raw);
    let record =
        submit_evidence_anchor_workflow(registry, client, digest, &seed).unwrap();
    record.artifact_hash_hex
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn reconcile_empty_registry_returns_empty_with_zero_chain_calls() {
    let (_dir, registry) = fresh_registry();
    let client = FakeAnchorClient::new();
    let out = reconcile_evidence_anchors_workflow(&registry, &client).unwrap();
    assert!(out.is_empty());
    assert_eq!(client.query_call_count(), 0);
}

#[test]
fn reconcile_drives_submitted_through_finalized() {
    let (_dir, registry) = fresh_registry();
    // Stage records via the stub client so the registry has a
    // `Submitted` row to drive.
    let stub = StubEvidenceAnchorChainClient::new();
    let hash_a = seed_record(&registry, &stub, [7u8; 32], 0x11);

    let real = FakeAnchorClient::new();
    // First reconcile pass: chain still says Submitted → no
    // transition.
    let out = reconcile_evidence_anchors_workflow(&registry, &real).unwrap();
    assert_eq!(out.len(), 1);
    let (h, r) = &out[0];
    assert_eq!(h, &hash_a);
    let outcome = r.as_ref().unwrap();
    assert!(matches!(outcome.chain_status, AnchorStatus::Submitted));
    assert!(!outcome.local_status_transitioned);

    // Second pass: chain reports Included → local transitions.
    let tx_id = registry
        .load_by_artifact_hash(&hash_a)
        .unwrap()
        .unwrap()
        .receipt
        .tx_id;
    real.set_status(&tx_id, AnchorStatus::Included);
    let out = reconcile_evidence_anchors_workflow(&registry, &real).unwrap();
    assert_eq!(out.len(), 1);
    let outcome = out[0].1.as_ref().unwrap();
    assert!(matches!(outcome.chain_status, AnchorStatus::Included));
    assert!(outcome.local_status_transitioned);

    // Third pass: chain reports Finalized → terminal.
    real.set_status(&tx_id, AnchorStatus::Finalized);
    let out = reconcile_evidence_anchors_workflow(&registry, &real).unwrap();
    assert_eq!(out.len(), 1);
    let outcome = out[0].1.as_ref().unwrap();
    assert!(matches!(outcome.chain_status, AnchorStatus::Finalized));
    assert!(outcome.local_status_transitioned);

    // Fourth pass: Finalized record is skipped — omitted from
    // result vec.
    let out = reconcile_evidence_anchors_workflow(&registry, &real).unwrap();
    assert!(out.is_empty(), "Finalized records must be skipped from sweep");
}

#[test]
fn reconcile_chain_unknown_is_observation_only() {
    let (_dir, registry) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let hash_a = seed_record(&registry, &stub, [7u8; 32], 0x11);
    let tx_id = registry
        .load_by_artifact_hash(&hash_a)
        .unwrap()
        .unwrap()
        .receipt
        .tx_id;

    let real = FakeAnchorClient::new();
    real.set_status(&tx_id, AnchorStatus::Unknown);
    let out = reconcile_evidence_anchors_workflow(&registry, &real).unwrap();
    assert_eq!(out.len(), 1);
    let outcome = out[0].1.as_ref().unwrap();
    assert!(matches!(outcome.chain_status, AnchorStatus::Unknown));
    assert!(!outcome.local_status_transitioned);

    // Local record still Submitted.
    let reloaded = registry.load_by_artifact_hash(&hash_a).unwrap().unwrap();
    assert_eq!(reloaded.status, LocalAnchorStatus::Submitted);
}

#[test]
fn reconcile_per_record_rpc_failure_does_not_abort_sweep() {
    let (_dir, registry) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let _hash_a = seed_record(&registry, &stub, [7u8; 32], 0x11);
    let _hash_b = seed_record(&registry, &stub, [8u8; 32], 0x22);

    // One record fails its query; the other gets Included.
    let real = FakeAnchorClient::new();
    let all_records: Vec<_> = registry
        .list()
        .unwrap()
        .into_iter()
        .map(|r| (r.artifact_hash_hex, r.receipt.tx_id))
        .collect();
    assert_eq!(all_records.len(), 2);
    let (failing_hash, failing_tx) = &all_records[0];
    let (ok_hash, ok_tx) = &all_records[1];
    real.set_query_error(
        failing_tx,
        ChainClientError::Other("HTTP transport failure: timed out".into()),
    );
    real.set_status(ok_tx, AnchorStatus::Included);

    let out = reconcile_evidence_anchors_workflow(&registry, &real).unwrap();
    assert_eq!(out.len(), 2, "both records appear in the result vec");

    let oks = out.iter().filter(|(_, r)| r.is_ok()).count();
    let errs = out.iter().filter(|(_, r)| r.is_err()).count();
    assert_eq!(oks, 1);
    assert_eq!(errs, 1);

    let _ = (failing_hash, ok_hash);
    // The failing record is unchanged on disk (Stage 5.1 contract).
    let reloaded_failing = registry
        .load_by_artifact_hash(failing_hash)
        .unwrap()
        .unwrap();
    assert_eq!(reloaded_failing.status, LocalAnchorStatus::Submitted);

    let reloaded_ok = registry
        .load_by_artifact_hash(ok_hash)
        .unwrap()
        .unwrap();
    assert_eq!(reloaded_ok.status, LocalAnchorStatus::Included);
}

#[test]
fn reconcile_skips_non_submitted_and_non_included_records() {
    let (_dir, registry) = fresh_registry();
    let stub = StubEvidenceAnchorChainClient::new();
    let hash_a = seed_record(&registry, &stub, [7u8; 32], 0x11);

    // Mark record terminal locally (simulate prior finalization).
    registry
        .update_status(&hash_a, LocalAnchorStatus::Finalized)
        .unwrap();

    let real = FakeAnchorClient::new();
    let out = reconcile_evidence_anchors_workflow(&registry, &real).unwrap();
    assert!(out.is_empty(), "Finalized records must not appear in the sweep");
    assert_eq!(real.query_call_count(), 0);
}
