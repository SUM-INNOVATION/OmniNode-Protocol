//! Phase 5 Stage 5.3 — end-to-end attestation orchestration.
//!
//! Stitches Stage 5 (registry + workflows), Stage 5.1 (tx-id query
//! alignment), Stage 5.2 (block-aware submit + staleness policy), and
//! Stage 7b (real SUM Chain submit/query) into one operator-facing
//! surface. **No chain protocol changes.** The `ChainClient` trait
//! stays at two methods; Stage 5.3 introduces a sibling
//! [`OrchestrationClient`] that extends it with one additional method,
//! `get_latest_block_height`, used by the block-aware submit path and
//! the staleness sweep.
//!
//! The four operator helpers:
//!
//! | Helper                                       | Source-state filter | Per-record RPC budget          | Aggregate RPC budget                          |
//! |----------------------------------------------|---------------------|--------------------------------|-----------------------------------------------|
//! | [`submit_attestation_workflow_with_block`]   | `Pending`, `Dropped` | 1× height + 1× submit          | (single record)                               |
//! | [`poll_attestations_workflow`]               | `Submitted`, `Included` | 1× query                    | 1× per queryable record                       |
//! | [`sweep_stale_attestations_workflow`]        | `Submitted`         | 0 (pure registry write on stale) | 1× height (up-front, shared)                |
//! | [`retry_dropped_attestations_workflow`]      | `Dropped`           | 1× height + 1× submit           | 1× height + 1× submit per `Dropped` record   |
//!
//! ## Invariants preserved from earlier stages
//!
//! - **Stage 5 idempotency.** `submit_attestation_workflow_with_block`
//!   calls [`AttestationRegistry::insert`] first; a record already at
//!   `Submitted` / `Included` / `Finalized` / `Failed` returns
//!   unchanged with **zero** chain calls. A byte-different attestation
//!   under the same `(session_id, verifier_address)` key surfaces as
//!   [`RegistryError::ConflictingAttestation`] without reaching the
//!   chain.
//! - **Stage 5.1 `Unknown` is observation-only.**
//!   `poll_attestations_workflow` delegates to
//!   [`query_attestation_workflow`] per record, which leaves
//!   `Submitted` / `Included` records unchanged on a chain-returned
//!   `Unknown` and emits a `tracing::warn!`. The sweep never
//!   terminalises a record on `Unknown`.
//! - **Stage 5.1 RPC-failure containment.** Per-record RPC failures
//!   in the sweep helpers surface as the corresponding entry's `Err`
//!   in the returned vec; the local record is **not** mutated.
//! - **Stage 5.2 `Dropped` is local-only.** Only
//!   `sweep_stale_attestations_workflow` writes
//!   `Submitted → Dropped`, and only via
//!   [`mark_stale_if_overdue`], which itself only fires when the
//!   caller-constructed [`StalenessPolicy`] is satisfied.
//!
//! ## Error model
//!
//! The three sweep helpers return
//! `Vec<(AttestationId, RegistryResult<AttestationRecord>)>`. A
//! transient RPC failure on one record yields an `Err` entry for that
//! record only; the rest of the sweep continues. Records that are
//! skipped by the source-state filter are **omitted** from the result
//! vec entirely (operators wanting the full registry can call
//! [`AttestationRegistry::list`]).
//!
//! The staleness sweep does an up-front
//! `client.get_latest_block_height()`; if **that** call fails, the
//! whole sweep returns `Err(RegistryError::ChainClient(_))` and no
//! per-record processing is attempted (we have no reference height to
//! compare against).
//!
//! ## What Stage 5.3 deliberately does not do
//!
//! - No live polling loop, scheduler, retry backoff, or
//!   max-attempts cap. Operators wire the cadence.
//! - No `ChainClient` trait change — `OrchestrationClient` is a
//!   sibling that extends it via supertrait.
//! - No async. Everything stays sync.
//! - No new dependencies.
//! - No mainnet config; no hardcoded chain id, finality depth, or
//!   block time.

use omni_types::phase5::InferenceAttestation;

use crate::chain::ChainClient;
use crate::error::{ChainClientError, RegistryError, RegistryResult};
use crate::registry::{
    query_attestation_workflow, AttestationId, AttestationRecord, AttestationRegistry,
    LocalAttestationStatus,
};
use crate::staleness::{mark_stale_if_overdue, StalenessPolicy};

// ── Trait ────────────────────────────────────────────────────────────────────

/// Stage 5.3 extension trait. A chain adapter that already implements
/// [`ChainClient`] (submit + per-tx-id query) opts into orchestration
/// by also exposing the chain's latest block height.
///
/// `omni-sumchain`'s `SumChainClient<T>` implements this via
/// `get_block_height(BlockFinality::Latest)` and is the canonical
/// production impl. Test fakes implement it directly.
///
/// Only the **latest** finality token is exposed — staleness wants
/// `Latest`, not `Finalized` (Stage 5.2 docs); callers needing
/// `Finalized` can read it via the chain adapter's inherent helper if
/// the adapter offers one.
pub trait OrchestrationClient: ChainClient {
    /// Return the chain's latest block height.
    fn get_latest_block_height(&self) -> std::result::Result<u64, ChainClientError>;
}

// ── 1. Block-aware submit (single record) ────────────────────────────────────

/// Stage 5.3 block-aware submit. Stitches:
///
/// 1. `registry.insert(attestation.clone())` — Stage 5 idempotency:
///    a byte-equal existing record returns unchanged; a byte-different
///    record under the same key surfaces
///    [`RegistryError::ConflictingAttestation`] without any chain call.
/// 2. If the inserted/loaded record's status is **not** `Pending` or
///    `Dropped`, return it unchanged. **No chain calls.** Existing
///    `Submitted` / `Included` / `Finalized` / `Failed` records are
///    inert under this helper.
/// 3. `client.get_latest_block_height()` — fetched **before** submit so
///    the stamp reflects the head as the chain sees it at submit time.
///    On failure, surfaces as `RegistryError::ChainClient(_)` and the
///    record stays at its prior `Pending` / `Dropped` state (Stage 5.1
///    invariant: RPC failures don't mutate local state).
/// 4. `client.submit_attestation(&attestation)`. Same RPC-failure
///    containment as step 3.
/// 5. `registry.mark_submitted_with_block(&id, receipt, current_block)`
///    — stamps `submitted_at_block = Some(current_block)` and clears
///    any prior `error_message`.
///
/// Emits `tracing::info!` on the successful state transition (step 5);
/// does not log step 1's no-op outcome.
pub fn submit_attestation_workflow_with_block<C: OrchestrationClient>(
    registry: &AttestationRegistry,
    client: &C,
    attestation: InferenceAttestation,
) -> RegistryResult<AttestationRecord> {
    let record = registry.insert(attestation.clone())?;
    match &record.status {
        LocalAttestationStatus::Pending | LocalAttestationStatus::Dropped { .. } => {
            // proceed to chain
        }
        _ => return Ok(record),
    }

    let current_block = client
        .get_latest_block_height()
        .map_err(RegistryError::ChainClient)?;

    let receipt = client
        .submit_attestation(&attestation)
        .map_err(RegistryError::ChainClient)?;

    let updated = registry.mark_submitted_with_block(&record.id, receipt, current_block)?;

    tracing::info!(
        id = %updated.id,
        current_block,
        tx_id = updated
            .receipt
            .as_ref()
            .map(|r| r.tx_id.as_str())
            .unwrap_or("<none>"),
        "submit_attestation_workflow_with_block: Submitted at chain head"
    );

    Ok(updated)
}

// ── 2. Poll / reconcile sweep ────────────────────────────────────────────────

/// Iterate the registry; for every `Submitted` or `Included` record,
/// call [`query_attestation_workflow`] and collect the result. Other
/// statuses are skipped and **not** present in the returned vec
/// (operators wanting the full registry can call
/// [`AttestationRegistry::list`]).
///
/// Per-record RPC failures land as `Err` entries in the returned vec;
/// the sweep continues. Stage 5.1 semantics are preserved exactly —
/// `Unknown` leaves the record unchanged and is **not** terminalised.
///
/// `Submitted → Included → Finalized` / `Submitted → Failed{reason}`
/// transitions emit `tracing::info!`. Skipped records emit
/// `tracing::debug!`.
pub fn poll_attestations_workflow<C: ChainClient>(
    registry: &AttestationRegistry,
    client: &C,
) -> RegistryResult<Vec<(AttestationId, RegistryResult<AttestationRecord>)>> {
    let mut out = Vec::new();
    for record in registry.list()? {
        let id = record.id;
        let from = record.status.clone();
        let is_queryable = matches!(
            from,
            LocalAttestationStatus::Submitted | LocalAttestationStatus::Included
        );
        if !is_queryable {
            tracing::debug!(
                id = %id,
                status = ?from,
                "poll_attestations_workflow: skipping non-queryable record"
            );
            continue;
        }

        let result = query_attestation_workflow(registry, client, &id);
        if let Ok(ref to) = result {
            if to.status != from {
                tracing::info!(
                    id = %id,
                    from = ?from,
                    to = ?to.status,
                    "poll_attestations_workflow: state transition"
                );
            }
        }
        out.push((id, result));
    }
    Ok(out)
}

// ── 3. Staleness sweep ───────────────────────────────────────────────────────

/// Fetch the chain's latest block height **once** up-front, then
/// iterate the registry; for every `Submitted` record call
/// [`mark_stale_if_overdue`] with the shared height + the provided
/// policy. Non-`Submitted` records are skipped (omitted from the
/// returned vec).
///
/// If the up-front `get_latest_block_height` fails, the sweep returns
/// `Err(RegistryError::ChainClient(_))` without touching any record —
/// without a reference height there's nothing to compare against. A
/// per-record `mark_stale_if_overdue` error lands as an `Err` entry
/// for that record; the sweep continues.
///
/// `Submitted → Dropped` transitions emit `tracing::info!`. Skipped
/// records emit `tracing::debug!`.
pub fn sweep_stale_attestations_workflow<C: OrchestrationClient>(
    registry: &AttestationRegistry,
    client: &C,
    policy: &StalenessPolicy,
) -> RegistryResult<Vec<(AttestationId, RegistryResult<AttestationRecord>)>> {
    let current_block = client
        .get_latest_block_height()
        .map_err(RegistryError::ChainClient)?;

    let mut out = Vec::new();
    for record in registry.list()? {
        let id = record.id;
        let from = record.status.clone();
        if !matches!(from, LocalAttestationStatus::Submitted) {
            tracing::debug!(
                id = %id,
                status = ?from,
                "sweep_stale_attestations_workflow: skipping non-Submitted record"
            );
            continue;
        }

        let result = mark_stale_if_overdue(registry, &id, current_block, policy);
        if let Ok(ref to) = result {
            if matches!(to.status, LocalAttestationStatus::Dropped { .. }) {
                tracing::info!(
                    id = %id,
                    current_block,
                    "sweep_stale_attestations_workflow: dropped stale record"
                );
            }
        }
        out.push((id, result));
    }
    Ok(out)
}

// ── 4. Retry dropped records ─────────────────────────────────────────────────

/// Iterate the registry; for every `Dropped` record call
/// [`submit_attestation_workflow_with_block`] with that record's
/// stored attestation. Each retry independently fetches the chain
/// head so its `submitted_at_block` stamp reflects the head at the
/// retry's own submit time. Non-`Dropped` records are skipped
/// (omitted from the returned vec).
///
/// No retry cap, no backoff. Operators wanting either build their own
/// loop on top of this helper.
///
/// `Dropped → Submitted` transitions emit `tracing::info!`. Skipped
/// records emit `tracing::debug!`.
pub fn retry_dropped_attestations_workflow<C: OrchestrationClient>(
    registry: &AttestationRegistry,
    client: &C,
) -> RegistryResult<Vec<(AttestationId, RegistryResult<AttestationRecord>)>> {
    let mut out = Vec::new();
    for record in registry.list()? {
        let id = record.id;
        let from = record.status.clone();
        if !matches!(from, LocalAttestationStatus::Dropped { .. }) {
            tracing::debug!(
                id = %id,
                status = ?from,
                "retry_dropped_attestations_workflow: skipping non-Dropped record"
            );
            continue;
        }

        let result =
            submit_attestation_workflow_with_block(registry, client, record.attestation.clone());
        if let Ok(ref to) = result {
            if matches!(to.status, LocalAttestationStatus::Submitted) {
                tracing::info!(
                    id = %id,
                    tx_id = to
                        .receipt
                        .as_ref()
                        .map(|r| r.tx_id.as_str())
                        .unwrap_or("<none>"),
                    "retry_dropped_attestations_workflow: resubmitted dropped record"
                );
            }
        }
        out.push((id, result));
    }
    Ok(out)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};

    use omni_types::phase5::{InferenceCommitment, SnipV2ObjectId};

    use crate::chain::{AttestationStatus, SubmissionReceipt};
    use crate::registry::compute_attestation_id;

    // ── Fake client ──────────────────────────────────────────────────────

    /// A single recorded call against the fake. Used by tests that
    /// need to pin ordering (e.g. height-before-submit) or count
    /// per-helper RPC budget (e.g. retry fetches height per record).
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Call {
        Height,
        Submit { attestation_session: String },
        Query { tx_id: String },
    }

    /// Hermetic fake implementing both [`ChainClient`] and
    /// [`OrchestrationClient`]. Configurable via builder-style setters
    /// for submit / query / height outcomes; queues let one fake serve
    /// multiple consecutive calls in retry/lifecycle tests.
    struct FakeOrchestrationClient {
        submit_queue: RefCell<VecDeque<std::result::Result<SubmissionReceipt, ChainClientError>>>,
        submit_default: RefCell<std::result::Result<SubmissionReceipt, ChainClientError>>,

        query_per_tx: RefCell<HashMap<String, AttestationStatus>>,
        query_default: RefCell<std::result::Result<AttestationStatus, ChainClientError>>,

        height_queue: RefCell<VecDeque<std::result::Result<u64, ChainClientError>>>,
        height_default: RefCell<std::result::Result<u64, ChainClientError>>,

        call_log: RefCell<Vec<Call>>,
    }

    impl FakeOrchestrationClient {
        fn new() -> Self {
            Self {
                submit_queue: RefCell::new(VecDeque::new()),
                submit_default: RefCell::new(Err(ChainClientError::Other(
                    "no submit outcome configured".into(),
                ))),
                query_per_tx: RefCell::new(HashMap::new()),
                query_default: RefCell::new(Ok(AttestationStatus::Submitted)),
                height_queue: RefCell::new(VecDeque::new()),
                height_default: RefCell::new(Err(ChainClientError::Other(
                    "no height outcome configured".into(),
                ))),
                call_log: RefCell::new(Vec::new()),
            }
        }

        fn set_submit_default(
            &self,
            outcome: std::result::Result<SubmissionReceipt, ChainClientError>,
        ) {
            *self.submit_default.borrow_mut() = outcome;
        }

        fn enqueue_submit(
            &self,
            outcome: std::result::Result<SubmissionReceipt, ChainClientError>,
        ) {
            self.submit_queue.borrow_mut().push_back(outcome);
        }

        fn set_query_for(&self, tx_id: &str, status: AttestationStatus) {
            self.query_per_tx
                .borrow_mut()
                .insert(tx_id.to_string(), status);
        }

        fn set_query_default(
            &self,
            outcome: std::result::Result<AttestationStatus, ChainClientError>,
        ) {
            *self.query_default.borrow_mut() = outcome;
        }

        fn set_height_default(
            &self,
            outcome: std::result::Result<u64, ChainClientError>,
        ) {
            *self.height_default.borrow_mut() = outcome;
        }

        fn enqueue_height(
            &self,
            outcome: std::result::Result<u64, ChainClientError>,
        ) {
            self.height_queue.borrow_mut().push_back(outcome);
        }

        fn calls(&self) -> Vec<Call> {
            self.call_log.borrow().clone()
        }

        fn count_calls<F: Fn(&Call) -> bool>(&self, pred: F) -> usize {
            self.calls().iter().filter(|c| pred(c)).count()
        }
    }

    impl ChainClient for FakeOrchestrationClient {
        fn submit_attestation(
            &self,
            attestation: &InferenceAttestation,
        ) -> std::result::Result<SubmissionReceipt, ChainClientError> {
            self.call_log.borrow_mut().push(Call::Submit {
                attestation_session: attestation.commitment.session_id.clone(),
            });
            if let Some(r) = self.submit_queue.borrow_mut().pop_front() {
                return r;
            }
            self.submit_default.borrow().clone()
        }

        fn query_attestation_status(
            &self,
            tx_id: &str,
        ) -> std::result::Result<AttestationStatus, ChainClientError> {
            self.call_log.borrow_mut().push(Call::Query {
                tx_id: tx_id.to_string(),
            });
            if let Some(s) = self.query_per_tx.borrow().get(tx_id).cloned() {
                return Ok(s);
            }
            self.query_default.borrow().clone()
        }
    }

    impl OrchestrationClient for FakeOrchestrationClient {
        fn get_latest_block_height(&self) -> std::result::Result<u64, ChainClientError> {
            self.call_log.borrow_mut().push(Call::Height);
            if let Some(r) = self.height_queue.borrow_mut().pop_front() {
                return r;
            }
            self.height_default.borrow().clone()
        }
    }

    // ── Fixtures ─────────────────────────────────────────────────────────

    fn snip_id(byte: u8) -> SnipV2ObjectId {
        let mut b = [0u8; 32];
        b.fill(byte);
        SnipV2ObjectId::from_bytes(b)
    }

    fn make_attestation(session: &str, address: &str, signature: &str) -> InferenceAttestation {
        InferenceAttestation {
            commitment: InferenceCommitment {
                session_id: session.into(),
                model_hash: "a".repeat(64),
                manifest_snip_root: snip_id(0x11),
                response_hash: "b".repeat(64),
                proof_snip_root: snip_id(0x22),
            },
            verifier_address: address.into(),
            verifier_signature: signature.into(),
        }
    }

    fn open_temp_registry() -> (tempfile::TempDir, AttestationRegistry) {
        let dir = tempfile::tempdir().unwrap();
        let reg = AttestationRegistry::open(dir.path().join("attestations")).unwrap();
        (dir, reg)
    }

    fn receipt(tx: &str) -> SubmissionReceipt {
        SubmissionReceipt {
            tx_id: tx.into(),
            note: None,
        }
    }

    fn happy_client(tx: &str, height: u64) -> FakeOrchestrationClient {
        let c = FakeOrchestrationClient::new();
        c.set_submit_default(Ok(receipt(tx)));
        c.set_height_default(Ok(height));
        c
    }

    // ── 1. submit_attestation_workflow_with_block (6) ────────────────────

    #[test]
    fn submit_with_block_stamps_submitted_at_block_on_happy_path() {
        let (_dir, reg) = open_temp_registry();
        let client = happy_client("tx-1", 42);
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let updated = submit_attestation_workflow_with_block(&reg, &client, att).unwrap();

        assert_eq!(updated.status, LocalAttestationStatus::Submitted);
        assert_eq!(updated.submitted_at_block, Some(42));
        assert_eq!(updated.receipt.as_ref().unwrap().tx_id, "tx-1");
    }

    #[test]
    fn submit_with_block_fetches_height_before_submit() {
        let (_dir, reg) = open_temp_registry();
        let client = happy_client("tx-1", 42);
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        submit_attestation_workflow_with_block(&reg, &client, att).unwrap();

        let calls = client.calls();
        // Find positions; height must precede submit.
        let height_idx = calls.iter().position(|c| matches!(c, Call::Height));
        let submit_idx = calls.iter().position(|c| matches!(c, Call::Submit { .. }));
        assert!(height_idx.is_some() && submit_idx.is_some());
        assert!(
            height_idx.unwrap() < submit_idx.unwrap(),
            "height must be fetched before submit; got call order: {calls:?}"
        );
    }

    #[test]
    fn submit_with_block_retry_from_dropped_records_fresh_head() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-retry", "addr-1", "sig-1");

        // First submit at block 10.
        let client = happy_client("tx-old", 10);
        let first = submit_attestation_workflow_with_block(&reg, &client, att.clone()).unwrap();
        assert_eq!(first.submitted_at_block, Some(10));
        reg.mark_dropped(&first.id, Some("evicted".into())).unwrap();

        // Retry: client reconfigured to return a fresh receipt + height.
        client.set_submit_default(Ok(receipt("tx-new")));
        client.set_height_default(Ok(110));
        let retried = submit_attestation_workflow_with_block(&reg, &client, att).unwrap();
        assert_eq!(retried.status, LocalAttestationStatus::Submitted);
        assert_eq!(retried.submitted_at_block, Some(110));
        assert_eq!(retried.receipt.as_ref().unwrap().tx_id, "tx-new");
        assert!(retried.error_message.is_none());
    }

    #[test]
    fn submit_with_block_is_noop_for_already_submitted_record() {
        let (_dir, reg) = open_temp_registry();
        let client = happy_client("tx-1", 42);
        let att = make_attestation("sess-1", "addr-1", "sig-1");

        // First call: submits.
        submit_attestation_workflow_with_block(&reg, &client, att.clone()).unwrap();
        let first_call_count = client.calls().len();
        assert!(first_call_count >= 2, "expected height + submit on first call");

        // Second call with the same attestation: insert is idempotent;
        // status is already Submitted; helper must NOT call height or submit.
        let again = submit_attestation_workflow_with_block(&reg, &client, att).unwrap();
        assert_eq!(again.status, LocalAttestationStatus::Submitted);
        assert_eq!(
            client.calls().len(),
            first_call_count,
            "second invocation must add zero chain calls; got call log: {:?}",
            client.calls()
        );
    }

    #[test]
    fn submit_with_block_height_failure_leaves_record_pending() {
        let (_dir, reg) = open_temp_registry();
        let client = FakeOrchestrationClient::new();
        client.set_submit_default(Ok(receipt("tx-1")));
        client.set_height_default(Err(ChainClientError::Other("height rpc down".into())));

        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let err = submit_attestation_workflow_with_block(&reg, &client, att.clone()).unwrap_err();
        match err {
            RegistryError::ChainClient(ChainClientError::Other(msg)) => {
                assert_eq!(msg, "height rpc down");
            }
            other => panic!("expected ChainClient(Other(..)), got {other:?}"),
        }

        // Record is still Pending; submit was never reached.
        let id = compute_attestation_id(&att).unwrap();
        let loaded = reg.load(&id).unwrap();
        assert_eq!(loaded.status, LocalAttestationStatus::Pending);
        assert_eq!(loaded.submitted_at_block, None);
        assert_eq!(
            client.count_calls(|c| matches!(c, Call::Submit { .. })),
            0,
            "submit must NOT be reached when height fetch fails"
        );
    }

    #[test]
    fn submit_with_block_submit_failure_leaves_record_pending() {
        let (_dir, reg) = open_temp_registry();
        let client = FakeOrchestrationClient::new();
        client.set_height_default(Ok(42));
        client.set_submit_default(Err(ChainClientError::Other("mempool full".into())));

        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let err = submit_attestation_workflow_with_block(&reg, &client, att.clone()).unwrap_err();
        match err {
            RegistryError::ChainClient(ChainClientError::Other(msg)) => {
                assert_eq!(msg, "mempool full");
            }
            other => panic!("expected ChainClient(Other(..)), got {other:?}"),
        }

        // Record is still Pending; mark_submitted_with_block was not
        // reached (no submitted_at_block stamp).
        let id = compute_attestation_id(&att).unwrap();
        let loaded = reg.load(&id).unwrap();
        assert_eq!(loaded.status, LocalAttestationStatus::Pending);
        assert_eq!(loaded.submitted_at_block, None);
    }

    // ── 2. poll_attestations_workflow (5) ────────────────────────────────

    /// Insert records covering every status; only `Submitted` and
    /// `Included` should appear in the sweep result vec.
    #[test]
    fn poll_queries_only_submitted_and_included() {
        let (_dir, reg) = open_temp_registry();

        let pending = reg.insert(make_attestation("p", "addr", "s")).unwrap();
        let submitted = reg.insert(make_attestation("s", "addr", "s")).unwrap();
        reg.mark_submitted(&submitted.id, receipt("tx-s")).unwrap();
        let included = reg.insert(make_attestation("i", "addr", "s")).unwrap();
        reg.mark_submitted(&included.id, receipt("tx-i")).unwrap();
        reg.mark_included(&included.id).unwrap();
        let finalized = reg.insert(make_attestation("f", "addr", "s")).unwrap();
        reg.mark_submitted(&finalized.id, receipt("tx-f")).unwrap();
        reg.mark_finalized(&finalized.id).unwrap();
        let failed = reg.insert(make_attestation("x", "addr", "s")).unwrap();
        reg.mark_submitted(&failed.id, receipt("tx-x")).unwrap();
        reg.mark_failed(&failed.id, "reverted".into()).unwrap();
        let dropped = reg.insert(make_attestation("d", "addr", "s")).unwrap();
        reg.mark_submitted(&dropped.id, receipt("tx-d")).unwrap();
        reg.mark_dropped(&dropped.id, Some("evicted".into())).unwrap();

        let client = FakeOrchestrationClient::new();
        // Default query result for any tx is Submitted (no transition).
        let out = poll_attestations_workflow(&reg, &client).unwrap();

        // Exactly two queried entries — submitted + included.
        assert_eq!(out.len(), 2, "got: {out:?}");
        let queried_ids: std::collections::HashSet<_> =
            out.iter().map(|(id, _)| *id).collect();
        assert!(queried_ids.contains(&submitted.id));
        assert!(queried_ids.contains(&included.id));
        assert!(!queried_ids.contains(&pending.id));
        assert!(!queried_ids.contains(&finalized.id));
        assert!(!queried_ids.contains(&failed.id));
        assert!(!queried_ids.contains(&dropped.id));
    }

    #[test]
    fn poll_preserves_unknown_as_observation_only() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let r = reg.insert(att).unwrap();
        reg.mark_submitted(&r.id, receipt("tx-unknown")).unwrap();

        let client = FakeOrchestrationClient::new();
        client.set_query_for("tx-unknown", AttestationStatus::Unknown);

        let out = poll_attestations_workflow(&reg, &client).unwrap();
        assert_eq!(out.len(), 1);
        let (id, result) = &out[0];
        assert_eq!(id, &r.id);
        let returned = result.as_ref().expect("query must succeed");
        assert_eq!(returned.status, LocalAttestationStatus::Submitted);

        // Persisted state also unchanged.
        let reloaded = reg.load(&r.id).unwrap();
        assert_eq!(reloaded.status, LocalAttestationStatus::Submitted);
    }

    #[test]
    fn poll_per_record_failure_does_not_abort_sweep() {
        let (_dir, reg) = open_temp_registry();

        let a = reg.insert(make_attestation("a", "addr", "s")).unwrap();
        reg.mark_submitted(&a.id, receipt("tx-a")).unwrap();
        let b = reg.insert(make_attestation("b", "addr", "s")).unwrap();
        reg.mark_submitted(&b.id, receipt("tx-b")).unwrap();

        let client = FakeOrchestrationClient::new();
        // tx-a → RPC failure; tx-b → Finalized.
        client.set_query_default(Err(ChainClientError::Other("rpc gone".into())));
        client.set_query_for("tx-b", AttestationStatus::Finalized);

        let out = poll_attestations_workflow(&reg, &client).unwrap();
        assert_eq!(out.len(), 2);

        let by_id: HashMap<AttestationId, &RegistryResult<AttestationRecord>> =
            out.iter().map(|(id, r)| (*id, r)).collect();
        // a: Err; record unchanged.
        let a_result = by_id[&a.id];
        assert!(matches!(
            a_result,
            Err(RegistryError::ChainClient(ChainClientError::Other(msg))) if msg == "rpc gone"
        ));
        assert_eq!(
            reg.load(&a.id).unwrap().status,
            LocalAttestationStatus::Submitted,
            "RPC failure must not mutate local state"
        );

        // b: Ok, transitioned to Finalized.
        let b_result = by_id[&b.id];
        assert_eq!(
            b_result.as_ref().unwrap().status,
            LocalAttestationStatus::Finalized
        );
    }

    #[test]
    fn poll_empty_registry_returns_empty_with_zero_chain_calls() {
        let (_dir, reg) = open_temp_registry();
        let client = FakeOrchestrationClient::new();
        let out = poll_attestations_workflow(&reg, &client).unwrap();
        assert!(out.is_empty());
        assert!(client.calls().is_empty());
    }

    /// `submit_attestation_workflow_with_block` followed by
    /// `poll_attestations_workflow` should sequentially observe
    /// transition Submitted → Finalized when the chain confirms.
    #[test]
    fn poll_drives_submitted_to_finalized() {
        let (_dir, reg) = open_temp_registry();
        let client = happy_client("tx-1", 42);
        let att = make_attestation("sess-1", "addr-1", "sig-1");
        let r = submit_attestation_workflow_with_block(&reg, &client, att).unwrap();

        client.set_query_for("tx-1", AttestationStatus::Finalized);
        let out = poll_attestations_workflow(&reg, &client).unwrap();
        assert_eq!(out.len(), 1);
        let (id, result) = &out[0];
        assert_eq!(id, &r.id);
        assert_eq!(
            result.as_ref().unwrap().status,
            LocalAttestationStatus::Finalized
        );
    }

    // ── 3. sweep_stale_attestations_workflow (5) ─────────────────────────

    /// One Submitted record marked at block 10; sweep at block 50
    /// with threshold 10 → stale → Dropped.
    #[test]
    fn staleness_sweep_drops_stale_submitted_records() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-stale", "addr", "sig");
        let r = reg.insert(att.clone()).unwrap();
        reg.mark_submitted_with_block(&r.id, receipt("tx-stale"), 10)
            .unwrap();

        let client = FakeOrchestrationClient::new();
        client.set_height_default(Ok(50));
        let policy = StalenessPolicy::new(10).unwrap();

        let out = sweep_stale_attestations_workflow(&reg, &client, &policy).unwrap();
        assert_eq!(out.len(), 1);
        let (id, result) = &out[0];
        assert_eq!(id, &r.id);
        assert!(matches!(
            result.as_ref().unwrap().status,
            LocalAttestationStatus::Dropped { .. }
        ));

        // Persisted to disk.
        let reloaded = reg.load(&r.id).unwrap();
        assert!(matches!(reloaded.status, LocalAttestationStatus::Dropped { .. }));
    }

    /// Multiple Submitted records: the sweep fetches height exactly
    /// once regardless of record count.
    #[test]
    fn staleness_sweep_fetches_height_exactly_once() {
        let (_dir, reg) = open_temp_registry();
        for s in ["a", "b", "c", "d"] {
            let att = make_attestation(s, "addr", "sig");
            let r = reg.insert(att).unwrap();
            reg.mark_submitted_with_block(&r.id, receipt(&format!("tx-{s}")), 10)
                .unwrap();
        }

        let client = FakeOrchestrationClient::new();
        client.set_height_default(Ok(50));
        let policy = StalenessPolicy::new(10).unwrap();

        sweep_stale_attestations_workflow(&reg, &client, &policy).unwrap();

        assert_eq!(
            client.count_calls(|c| matches!(c, Call::Height)),
            1,
            "height must be fetched exactly once for the whole sweep"
        );
    }

    /// Non-`Submitted` records are skipped (omitted from the result
    /// vec), regardless of block height.
    #[test]
    fn staleness_sweep_omits_non_submitted_records_from_result() {
        let (_dir, reg) = open_temp_registry();
        // Pending only — never reached Submitted.
        let _p = reg.insert(make_attestation("p", "addr", "s")).unwrap();
        // Finalized.
        let f = reg.insert(make_attestation("f", "addr", "s")).unwrap();
        reg.mark_submitted_with_block(&f.id, receipt("tx-f"), 1).unwrap();
        reg.mark_finalized(&f.id).unwrap();
        // Dropped.
        let d = reg.insert(make_attestation("d", "addr", "s")).unwrap();
        reg.mark_submitted_with_block(&d.id, receipt("tx-d"), 1).unwrap();
        reg.mark_dropped(&d.id, Some("manual".into())).unwrap();

        let client = FakeOrchestrationClient::new();
        client.set_height_default(Ok(1_000_000));
        let policy = StalenessPolicy::new(1).unwrap();

        let out = sweep_stale_attestations_workflow(&reg, &client, &policy).unwrap();
        assert!(
            out.is_empty(),
            "non-Submitted records must be omitted; got: {out:?}"
        );
    }

    #[test]
    fn staleness_sweep_height_failure_aborts_without_touching_records() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess", "addr", "sig");
        let r = reg.insert(att).unwrap();
        reg.mark_submitted_with_block(&r.id, receipt("tx"), 10).unwrap();

        let client = FakeOrchestrationClient::new();
        client.set_height_default(Err(ChainClientError::Other("height down".into())));
        let policy = StalenessPolicy::new(1).unwrap();

        let err = sweep_stale_attestations_workflow(&reg, &client, &policy).unwrap_err();
        match err {
            RegistryError::ChainClient(ChainClientError::Other(msg)) => {
                assert_eq!(msg, "height down");
            }
            other => panic!("expected ChainClient(Other(..)), got {other:?}"),
        }

        let reloaded = reg.load(&r.id).unwrap();
        assert_eq!(reloaded.status, LocalAttestationStatus::Submitted);
    }

    /// Two policies applied to the same fixtures produce different
    /// drop sets — tight threshold drops, loose threshold doesn't.
    #[test]
    fn staleness_sweep_honours_policy_threshold() {
        for (threshold, expect_drop) in [(10u64, true), (10_000u64, false)] {
            let (_dir, reg) = open_temp_registry();
            let att = make_attestation("sess", "addr", "sig");
            let r = reg.insert(att).unwrap();
            reg.mark_submitted_with_block(&r.id, receipt("tx"), 10).unwrap();

            let client = FakeOrchestrationClient::new();
            client.set_height_default(Ok(50));
            let policy = StalenessPolicy::new(threshold).unwrap();

            let out = sweep_stale_attestations_workflow(&reg, &client, &policy).unwrap();
            assert_eq!(out.len(), 1);
            let (_id, result) = &out[0];
            let returned = result.as_ref().unwrap();
            if expect_drop {
                assert!(
                    matches!(returned.status, LocalAttestationStatus::Dropped { .. }),
                    "threshold={threshold}: expected Dropped, got {:?}",
                    returned.status
                );
            } else {
                assert_eq!(
                    returned.status,
                    LocalAttestationStatus::Submitted,
                    "threshold={threshold}: expected unchanged Submitted"
                );
            }
        }
    }

    // ── 4. retry_dropped_attestations_workflow (5) ───────────────────────

    #[test]
    fn retry_resubmits_only_dropped_records() {
        let (_dir, reg) = open_temp_registry();
        let pending = reg.insert(make_attestation("p", "addr", "s")).unwrap();
        let submitted = reg.insert(make_attestation("s", "addr", "s")).unwrap();
        reg.mark_submitted_with_block(&submitted.id, receipt("tx-s"), 1)
            .unwrap();
        let dropped = reg.insert(make_attestation("d", "addr", "s")).unwrap();
        reg.mark_submitted_with_block(&dropped.id, receipt("tx-old"), 1)
            .unwrap();
        reg.mark_dropped(&dropped.id, Some("stale".into())).unwrap();

        let client = happy_client("tx-new", 99);
        let out = retry_dropped_attestations_workflow(&reg, &client).unwrap();

        assert_eq!(out.len(), 1, "only Dropped records appear; got: {out:?}");
        let (id, result) = &out[0];
        assert_eq!(id, &dropped.id);
        let returned = result.as_ref().unwrap();
        assert_eq!(returned.status, LocalAttestationStatus::Submitted);
        assert_eq!(returned.submitted_at_block, Some(99));
        assert_eq!(returned.receipt.as_ref().unwrap().tx_id, "tx-new");

        // Pending and Submitted records are untouched.
        assert_eq!(
            reg.load(&pending.id).unwrap().status,
            LocalAttestationStatus::Pending
        );
        assert_eq!(
            reg.load(&submitted.id).unwrap().status,
            LocalAttestationStatus::Submitted
        );
    }

    /// Each Dropped record's retry independently fetches the chain
    /// head (per-record RPC budget).
    #[test]
    fn retry_fetches_height_per_record() {
        let (_dir, reg) = open_temp_registry();
        for s in ["d1", "d2", "d3"] {
            let r = reg.insert(make_attestation(s, "addr", "sig")).unwrap();
            reg.mark_submitted_with_block(&r.id, receipt(&format!("tx-{s}-old")), 1)
                .unwrap();
            reg.mark_dropped(&r.id, Some("stale".into())).unwrap();
        }

        let client = FakeOrchestrationClient::new();
        // Per-retry: height N then receipt tx-N.
        for n in 0..3 {
            client.enqueue_height(Ok(100 + n));
            client.enqueue_submit(Ok(receipt(&format!("tx-new-{n}"))));
        }

        retry_dropped_attestations_workflow(&reg, &client).unwrap();

        assert_eq!(
            client.count_calls(|c| matches!(c, Call::Height)),
            3,
            "retry must fetch height per record (per-record RPC budget)"
        );
        assert_eq!(
            client.count_calls(|c| matches!(c, Call::Submit { .. })),
            3,
        );
    }

    #[test]
    fn retry_stamps_fresh_submitted_at_block_and_clears_error_message() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("d", "addr", "sig");
        let r = reg.insert(att).unwrap();
        reg.mark_submitted_with_block(&r.id, receipt("tx-old"), 10)
            .unwrap();
        reg.mark_dropped(&r.id, Some("stale".into())).unwrap();

        // After mark_dropped, error_message is "stale".
        assert_eq!(reg.load(&r.id).unwrap().error_message.as_deref(), Some("stale"));

        let client = happy_client("tx-new", 500);
        retry_dropped_attestations_workflow(&reg, &client).unwrap();

        let reloaded = reg.load(&r.id).unwrap();
        assert_eq!(reloaded.status, LocalAttestationStatus::Submitted);
        assert_eq!(reloaded.submitted_at_block, Some(500));
        assert!(reloaded.error_message.is_none());
    }

    /// One retry's submit fails; the other still completes.
    #[test]
    fn retry_per_record_failure_does_not_abort_sweep() {
        let (_dir, reg) = open_temp_registry();
        let a = reg.insert(make_attestation("a", "addr", "sig")).unwrap();
        reg.mark_submitted_with_block(&a.id, receipt("tx-a-old"), 1)
            .unwrap();
        reg.mark_dropped(&a.id, Some("stale".into())).unwrap();
        let b = reg.insert(make_attestation("b", "addr", "sig")).unwrap();
        reg.mark_submitted_with_block(&b.id, receipt("tx-b-old"), 1)
            .unwrap();
        reg.mark_dropped(&b.id, Some("stale".into())).unwrap();

        let client = FakeOrchestrationClient::new();
        // Both retries fetch height = 50; first submit fails, second succeeds.
        // Note: list() ordering is by id hex; we can't predict which retry
        // runs first, so configure both submit outcomes via a queue and
        // pin "exactly one Err, exactly one Ok" rather than per-id outcomes.
        client.enqueue_height(Ok(50));
        client.enqueue_height(Ok(50));
        client.enqueue_submit(Err(ChainClientError::Other("mempool full".into())));
        client.enqueue_submit(Ok(receipt("tx-new")));

        let out = retry_dropped_attestations_workflow(&reg, &client).unwrap();
        assert_eq!(out.len(), 2);

        let oks = out.iter().filter(|(_, r)| r.is_ok()).count();
        let errs = out.iter().filter(|(_, r)| r.is_err()).count();
        assert_eq!(oks, 1, "exactly one retry must succeed; got: {out:?}");
        assert_eq!(errs, 1, "exactly one retry must fail; got: {out:?}");
    }

    #[test]
    fn retry_empty_registry_returns_empty_with_zero_chain_calls() {
        let (_dir, reg) = open_temp_registry();
        let client = FakeOrchestrationClient::new();
        let out = retry_dropped_attestations_workflow(&reg, &client).unwrap();
        assert!(out.is_empty());
        assert!(client.calls().is_empty());
    }

    // ── E2E lifecycle (1) ────────────────────────────────────────────────

    /// Walk one record through every Stage-5.3 helper:
    /// submit → poll(Submitted, no transition) → simulate chain
    /// forgetting → poll(Unknown, unchanged) → staleness sweep drops
    /// → retry resubmits with fresh head → poll observes Finalized.
    #[test]
    fn lifecycle_submit_poll_drop_retry_finalize() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-lifecycle", "addr-1", "sig-1");

        // Step 1: submit at block 10.
        let client = FakeOrchestrationClient::new();
        client.enqueue_height(Ok(10));
        client.enqueue_submit(Ok(receipt("tx-1")));
        client.set_query_default(Ok(AttestationStatus::Submitted));
        let r = submit_attestation_workflow_with_block(&reg, &client, att.clone()).unwrap();
        assert_eq!(r.submitted_at_block, Some(10));

        // Step 2: poll — chain reports Submitted (no transition).
        let out = poll_attestations_workflow(&reg, &client).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].1.as_ref().unwrap().status,
            LocalAttestationStatus::Submitted
        );

        // Step 3: chain "forgets" tx-1 (mempool eviction simulation).
        client.set_query_for("tx-1", AttestationStatus::Unknown);
        let out = poll_attestations_workflow(&reg, &client).unwrap();
        // Stage 5.1: Unknown is observation-only.
        assert_eq!(
            out[0].1.as_ref().unwrap().status,
            LocalAttestationStatus::Submitted
        );

        // Step 4: staleness sweep at block 100, threshold 10 → Dropped.
        client.enqueue_height(Ok(100));
        let policy = StalenessPolicy::new(10).unwrap();
        let out = sweep_stale_attestations_workflow(&reg, &client, &policy).unwrap();
        assert_eq!(out.len(), 1);
        assert!(matches!(
            out[0].1.as_ref().unwrap().status,
            LocalAttestationStatus::Dropped { .. }
        ));

        // Step 5: retry — resubmit at fresh head 110 → tx-2.
        client.enqueue_height(Ok(110));
        client.enqueue_submit(Ok(receipt("tx-2")));
        let out = retry_dropped_attestations_workflow(&reg, &client).unwrap();
        assert_eq!(out.len(), 1);
        let retried = out[0].1.as_ref().unwrap();
        assert_eq!(retried.status, LocalAttestationStatus::Submitted);
        assert_eq!(retried.submitted_at_block, Some(110));
        assert_eq!(retried.receipt.as_ref().unwrap().tx_id, "tx-2");

        // Step 6: chain finalises tx-2.
        client.set_query_for("tx-2", AttestationStatus::Finalized);
        let out = poll_attestations_workflow(&reg, &client).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].1.as_ref().unwrap().status,
            LocalAttestationStatus::Finalized
        );

        // Confirm persisted state.
        let id = compute_attestation_id(&att).unwrap();
        let final_record = reg.load(&id).unwrap();
        assert_eq!(final_record.status, LocalAttestationStatus::Finalized);
        assert_eq!(final_record.submitted_at_block, Some(110));
    }
}
