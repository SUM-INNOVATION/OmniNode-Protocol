//! Phase 5 Stage 5.2 — client-local staleness / retry policy.
//!
//! SUM Chain v1 does **not** report a chain-side `Dropped` state.
//! Mempool eviction and unrecognized tx hashes both surface from the
//! chain as [`crate::chain::AttestationStatus::Unknown`], which the
//! Stage 5.1 query workflow treats as observation-only. This module
//! is the *only* writer that transitions a local record
//! `Submitted → Dropped`, and the resulting `Dropped` is a
//! **client-side synthetic** decision made by OmniNode based on a
//! caller-supplied policy.
//!
//! ## Policy
//!
//! [`StalenessPolicy`] holds a single knob,
//! `submitted_threshold_blocks: u64`. Construction is fallible:
//! [`StalenessPolicy::new`] returns [`StalenessPolicyError::ZeroThreshold`]
//! for `0`, which would mean "stale as soon as any new block ticks
//! past submit" — almost certainly a config bug. The library does
//! **not** hardcode any formula like `finality_depth * N`; deriving
//! the threshold from `chain_getChainParams` and operational
//! expectations is the caller's responsibility.
//!
//! ## Stale predicate
//!
//! A record is stale iff all three hold:
//!
//! 1. `record.status == LocalAttestationStatus::Submitted`
//! 2. `record.submitted_at_block == Some(b)`
//! 3. `current_block.saturating_sub(b) > policy.submitted_threshold_blocks()`
//!
//! Strict `>` (not `>=`): with `threshold = N`, the record becomes
//! stale only after **more than** `N` blocks have elapsed since
//! `submitted_at_block`. With `threshold = 1` and `submitted_at_block
//! = 10`, the record is stale starting at `current_block = 12`.
//!
//! `saturating_sub` collapses any height regression (`current_block <
//! b`, e.g. due to a chain re-org or a misconfigured caller passing a
//! stale head) to `0`, which compares as **not stale**. This is the
//! conservative direction — a momentary head regression should not
//! drop a record that may still be fine. [`is_record_stale`] is a
//! pure function: it returns `bool` and does not log. Diagnostic
//! `tracing::warn!`s for height regression and legacy-record cases
//! live in [`mark_stale_if_overdue`], which has the record id and
//! both heights in scope.
//!
//! ## Block height source
//!
//! This module makes no chain RPC calls; the `ChainClient` trait is
//! untouched in Stage 5.2. Callers fetch the current head via their
//! chain client's own helper (for the SUM Chain adapter, that is
//! `SumChainClient::get_block_height(BlockFinality::Latest)`) and
//! pass the resulting `u64` into [`mark_stale_if_overdue`] and into
//! [`crate::registry::AttestationRegistry::mark_submitted_with_block`].
//! `Latest` is the natural pick for staleness; `Finalized` lags the
//! actual mempool / inclusion state and would declare records stale
//! more aggressively than intended.

use crate::registry::{
    AttestationId, AttestationRecord, AttestationRegistry, LocalAttestationStatus,
};
use crate::error::RegistryResult;

// ── Policy ───────────────────────────────────────────────────────────────────

/// Construction error for [`StalenessPolicy`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StalenessPolicyError {
    /// `submitted_threshold_blocks == 0` is rejected because it would
    /// mark records stale on the first block past submit, which is
    /// almost always a config bug. Pick `>= 1`.
    #[error("submitted_threshold_blocks must be >= 1; got 0")]
    ZeroThreshold,
}

/// Caller-constructed staleness policy. Today the only knob is
/// `submitted_threshold_blocks`; future policy fields (per-status
/// thresholds, age windows, etc.) can be added without breaking the
/// constructor invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StalenessPolicy {
    submitted_threshold_blocks: u64,
}

impl StalenessPolicy {
    /// Construct a policy. Returns [`StalenessPolicyError::ZeroThreshold`]
    /// if `submitted_threshold_blocks == 0`.
    pub fn new(
        submitted_threshold_blocks: u64,
    ) -> std::result::Result<Self, StalenessPolicyError> {
        if submitted_threshold_blocks == 0 {
            return Err(StalenessPolicyError::ZeroThreshold);
        }
        Ok(Self {
            submitted_threshold_blocks,
        })
    }

    /// The configured "blocks since submit before stale" threshold.
    /// Always `>= 1` by construction.
    pub fn submitted_threshold_blocks(&self) -> u64 {
        self.submitted_threshold_blocks
    }
}

// ── Pure detection ───────────────────────────────────────────────────────────

/// `true` iff the record satisfies all three predicates listed in the
/// module doc:
///
/// 1. `status == Submitted`
/// 2. `submitted_at_block == Some(b)`
/// 3. `current_block.saturating_sub(b) > policy.submitted_threshold_blocks()`
///
/// Pure function. Returns `false` (conservative) for:
///
/// - any non-`Submitted` status,
/// - `Submitted` records with `submitted_at_block == None`
///   (pre-Stage-5.2 legacy records have no reference point),
/// - `current_block < submitted_at_block` (height regression).
///
/// No logging — see [`mark_stale_if_overdue`] for the workflow-level
/// `tracing::warn!`s.
pub fn is_record_stale(
    record: &AttestationRecord,
    current_block: u64,
    policy: &StalenessPolicy,
) -> bool {
    let LocalAttestationStatus::Submitted = record.status else {
        return false;
    };
    let Some(submitted_at_block) = record.submitted_at_block else {
        return false;
    };
    let elapsed = current_block.saturating_sub(submitted_at_block);
    elapsed > policy.submitted_threshold_blocks()
}

// ── Workflow ─────────────────────────────────────────────────────────────────

/// If the loaded record is `Submitted` and [`is_record_stale`] returns
/// `true`, transition it `Submitted → Dropped` with a reason string
/// embedding the block-height triple
/// `(submitted_at_block, current_block, threshold)`. Otherwise return
/// the record unchanged.
///
/// Behaviour table:
///
/// | Source state                                  | Behaviour                                                        |
/// |-----------------------------------------------|------------------------------------------------------------------|
/// | `Submitted`, stale                            | `mark_dropped(Some("stale: …"))` → `Dropped`                     |
/// | `Submitted`, not stale                        | unchanged                                                        |
/// | `Submitted`, `submitted_at_block == None`     | unchanged + `tracing::warn!` (legacy record; no reference point) |
/// | `Submitted`, `current_block < submitted_at`   | unchanged + `tracing::warn!` (chain height regressed)            |
/// | `Pending`, `Included`, `Finalized`, `Failed`, `Dropped` | unchanged (silent no-op; mirrors `query_attestation_workflow`) |
///
/// Preserves `receipt` and `submitted_at_block` on the dropped record
/// so the retry hinge (`mark_submitted` / `mark_submitted_with_block`)
/// has the prior submission's traceability available.
///
/// Returns `Err(RegistryError::RecordNotFound)` if `id` is not on disk;
/// other I/O / serialisation errors propagate as the corresponding
/// `RegistryError` variant. No chain RPC is invoked.
pub fn mark_stale_if_overdue(
    registry: &AttestationRegistry,
    id: &AttestationId,
    current_block: u64,
    policy: &StalenessPolicy,
) -> RegistryResult<AttestationRecord> {
    let record = registry.load(id)?;

    // Only Submitted records are candidates. Non-Submitted states are
    // silent no-ops to mirror the precedent set by
    // query_attestation_workflow for irrelevant inputs.
    if !matches!(record.status, LocalAttestationStatus::Submitted) {
        return Ok(record);
    }

    let Some(submitted_at_block) = record.submitted_at_block else {
        tracing::warn!(
            id = %id,
            "mark_stale_if_overdue: Submitted record has no \
             submitted_at_block (legacy record predating Stage 5.2 or \
             submitted via the non-block-aware mark_submitted); leaving \
             unchanged"
        );
        return Ok(record);
    };

    if current_block < submitted_at_block {
        tracing::warn!(
            id = %id,
            submitted_at_block,
            current_block,
            "mark_stale_if_overdue: chain head regressed below the \
             record's submitted_at_block; treating as not stale \
             (conservative)"
        );
        return Ok(record);
    }

    if !is_record_stale(&record, current_block, policy) {
        return Ok(record);
    }

    let reason = format!(
        "stale: submitted_at_block={}, current_block={}, threshold_blocks={}",
        submitted_at_block,
        current_block,
        policy.submitted_threshold_blocks(),
    );
    registry.mark_dropped(id, Some(reason))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::Utc;
    use omni_types::phase5::{InferenceAttestation, InferenceCommitment, SnipV2ObjectId};

    use crate::attestation::compute_digest;
    use crate::chain::SubmissionReceipt;
    use crate::registry::{compute_attestation_id, AttestationRegistry};

    // ── Fixtures (mirror those in registry::tests; staleness is a
    // sibling module so we can't share private fixtures) ───────────────

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

    fn dummy_receipt(tx: &str) -> SubmissionReceipt {
        SubmissionReceipt {
            tx_id: tx.into(),
            note: None,
        }
    }

    /// Build a hand-constructed `AttestationRecord` with explicit
    /// status / `submitted_at_block`, bypassing the registry's
    /// transition machinery. Used for the pure-detection tests that
    /// need to exercise statuses the transition API would refuse to
    /// reach from `Pending`.
    fn make_record(
        status: LocalAttestationStatus,
        submitted_at_block: Option<u64>,
    ) -> AttestationRecord {
        let att = make_attestation("sess-pure", "addr-1", "sig-1");
        let id = compute_attestation_id(&att).unwrap();
        let digest = compute_digest(&att.commitment).unwrap();
        let now = Utc::now();
        AttestationRecord {
            id,
            digest,
            attestation: att,
            created_at: now,
            updated_at: now,
            status,
            receipt: None,
            error_message: None,
            submitted_at_block,
        }
    }

    fn submitted_record_at_block(submitted_at_block: u64) -> AttestationRecord {
        make_record(
            LocalAttestationStatus::Submitted,
            Some(submitted_at_block),
        )
    }

    // ── Policy construction ──────────────────────────────────────────────

    #[test]
    fn policy_new_rejects_zero_threshold() {
        let err = StalenessPolicy::new(0).unwrap_err();
        assert_eq!(err, StalenessPolicyError::ZeroThreshold);
    }

    #[test]
    fn policy_new_accepts_one_and_above() {
        for n in [1u64, 2, 10, 100, u64::MAX] {
            let p = StalenessPolicy::new(n)
                .unwrap_or_else(|_| panic!("threshold {n} must be accepted"));
            assert_eq!(p.submitted_threshold_blocks(), n);
        }
    }

    // ── Pure detection — non-Submitted statuses ──────────────────────────

    #[test]
    fn is_record_stale_returns_false_for_pending() {
        let r = make_record(LocalAttestationStatus::Pending, Some(10));
        let p = StalenessPolicy::new(1).unwrap();
        // Even with massive elapsed, Pending must not be stale.
        assert!(!is_record_stale(&r, 1_000_000, &p));
    }

    #[test]
    fn is_record_stale_returns_false_for_included() {
        let r = make_record(LocalAttestationStatus::Included, Some(10));
        let p = StalenessPolicy::new(1).unwrap();
        assert!(!is_record_stale(&r, 1_000_000, &p));
    }

    #[test]
    fn is_record_stale_returns_false_for_finalized() {
        let r = make_record(LocalAttestationStatus::Finalized, Some(10));
        let p = StalenessPolicy::new(1).unwrap();
        assert!(!is_record_stale(&r, 1_000_000, &p));
    }

    #[test]
    fn is_record_stale_returns_false_for_failed() {
        let r = make_record(
            LocalAttestationStatus::Failed {
                reason: "execution reverted".into(),
            },
            Some(10),
        );
        let p = StalenessPolicy::new(1).unwrap();
        assert!(!is_record_stale(&r, 1_000_000, &p));
    }

    #[test]
    fn is_record_stale_returns_false_for_dropped() {
        let r = make_record(
            LocalAttestationStatus::Dropped {
                reason: Some("stale-earlier".into()),
            },
            Some(10),
        );
        let p = StalenessPolicy::new(1).unwrap();
        assert!(!is_record_stale(&r, 1_000_000, &p));
    }

    // ── Pure detection — Submitted edge cases ────────────────────────────

    #[test]
    fn is_record_stale_returns_false_when_submitted_at_block_is_none() {
        let r = make_record(LocalAttestationStatus::Submitted, None);
        let p = StalenessPolicy::new(1).unwrap();
        assert!(!is_record_stale(&r, 1_000_000, &p));
    }

    #[test]
    fn is_record_stale_returns_false_when_current_eq_submitted() {
        let r = submitted_record_at_block(10);
        let p = StalenessPolicy::new(1).unwrap();
        assert!(!is_record_stale(&r, 10, &p));
    }

    #[test]
    fn is_record_stale_returns_false_when_within_threshold() {
        let r = submitted_record_at_block(10);
        let p = StalenessPolicy::new(10).unwrap();
        // submitted=10, current=15, elapsed=5, threshold=10 → not stale.
        assert!(!is_record_stale(&r, 15, &p));
    }

    /// Boundary: `elapsed == threshold` is **not** stale (strict `>`).
    /// With `threshold = 10` and `submitted_at_block = 10`, the record
    /// is stale starting at `current_block = 21`, not `20`.
    #[test]
    fn is_record_stale_treats_boundary_as_not_stale() {
        let r = submitted_record_at_block(10);
        let p = StalenessPolicy::new(10).unwrap();
        // elapsed=10, threshold=10 → not stale (10 > 10 is false).
        assert!(!is_record_stale(&r, 20, &p));
    }

    #[test]
    fn is_record_stale_returns_true_when_strictly_past_threshold() {
        let r = submitted_record_at_block(10);
        let p = StalenessPolicy::new(10).unwrap();
        // elapsed=11, threshold=10 → stale.
        assert!(is_record_stale(&r, 21, &p));
    }

    #[test]
    fn is_record_stale_returns_false_on_height_regression() {
        let r = submitted_record_at_block(100);
        let p = StalenessPolicy::new(1).unwrap();
        // current < submitted → saturating_sub → 0 → 0 > 1 is false.
        assert!(!is_record_stale(&r, 50, &p));
    }

    /// `threshold = 1` only requires `elapsed >= 2` (strict `>`). At
    /// `elapsed = 1` the record is still not stale.
    #[test]
    fn is_record_stale_minimum_threshold_one_requires_two_block_gap() {
        let r = submitted_record_at_block(10);
        let p = StalenessPolicy::new(1).unwrap();
        assert!(!is_record_stale(&r, 11, &p)); // elapsed=1, threshold=1 → not stale
        assert!(is_record_stale(&r, 12, &p)); // elapsed=2, threshold=1 → stale
    }

    // ── Workflow — happy paths ───────────────────────────────────────────

    /// Insert a record, `mark_submitted_with_block(.., submitted)`,
    /// then `mark_stale_if_overdue(.., current, ..)` with stale
    /// parameters. Returns the registry + id for further assertions.
    fn setup_submitted_at_block(
        reg: &AttestationRegistry,
        session: &str,
        submitted_at_block: u64,
    ) -> AttestationId {
        let att = make_attestation(session, "addr-1", "sig-1");
        let r = reg.insert(att).unwrap();
        reg.mark_submitted_with_block(&r.id, dummy_receipt("tx-x"), submitted_at_block)
            .unwrap();
        r.id
    }

    #[test]
    fn mark_stale_if_overdue_drops_submitted_record_when_stale() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-stale", 10);
        let policy = StalenessPolicy::new(10).unwrap();

        // elapsed=15, threshold=10 → stale.
        let updated = mark_stale_if_overdue(&reg, &id, 25, &policy).unwrap();
        assert!(matches!(
            updated.status,
            LocalAttestationStatus::Dropped { ref reason } if reason.is_some()
        ));

        // Persisted to disk.
        let reloaded = reg.load(&id).unwrap();
        assert_eq!(updated, reloaded);
    }

    #[test]
    fn mark_stale_if_overdue_drop_reason_message_carries_block_context() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-reason", 100);
        let policy = StalenessPolicy::new(5).unwrap();

        let updated = mark_stale_if_overdue(&reg, &id, 200, &policy).unwrap();
        let LocalAttestationStatus::Dropped { reason } = &updated.status else {
            panic!("expected Dropped, got {:?}", updated.status);
        };
        let msg = reason.as_deref().expect("reason must be Some");
        assert!(msg.contains("submitted_at_block=100"), "got: {msg}");
        assert!(msg.contains("current_block=200"), "got: {msg}");
        assert!(msg.contains("threshold_blocks=5"), "got: {msg}");
    }

    /// Dropped via staleness must preserve `receipt` and
    /// `submitted_at_block` for traceability. The retry hinge depends
    /// on operators being able to look up what was previously
    /// submitted.
    #[test]
    fn mark_stale_if_overdue_preserves_receipt_and_submitted_at_block_on_drop() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-preserve", 50);
        let policy = StalenessPolicy::new(1).unwrap();

        let updated = mark_stale_if_overdue(&reg, &id, 100, &policy).unwrap();
        assert!(matches!(updated.status, LocalAttestationStatus::Dropped { .. }));
        assert_eq!(updated.receipt.as_ref().unwrap().tx_id, "tx-x");
        assert_eq!(updated.submitted_at_block, Some(50));
    }

    #[test]
    fn mark_stale_if_overdue_leaves_submitted_record_unchanged_when_not_stale() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-young", 100);
        let policy = StalenessPolicy::new(10).unwrap();

        // elapsed=5, threshold=10 → not stale.
        let updated = mark_stale_if_overdue(&reg, &id, 105, &policy).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Submitted);
        assert_eq!(updated.submitted_at_block, Some(100));
    }

    #[test]
    fn mark_stale_if_overdue_leaves_submitted_record_unchanged_when_no_submitted_block() {
        let (_dir, reg) = open_temp_registry();
        // Legacy path: mark_submitted (no block height).
        let att = make_attestation("sess-legacy", "addr-1", "sig-1");
        let r = reg.insert(att).unwrap();
        reg.mark_submitted(&r.id, dummy_receipt("tx-legacy")).unwrap();
        let policy = StalenessPolicy::new(1).unwrap();

        // Even at an enormous current_block, no submitted_at_block →
        // no reference point → unchanged.
        let updated = mark_stale_if_overdue(&reg, &r.id, 1_000_000, &policy).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Submitted);
        assert_eq!(updated.submitted_at_block, None);
    }

    #[test]
    fn mark_stale_if_overdue_leaves_record_unchanged_on_height_regression() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-regress", 100);
        let policy = StalenessPolicy::new(1).unwrap();

        // current < submitted: regression path. Must NOT transition.
        let updated = mark_stale_if_overdue(&reg, &id, 50, &policy).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Submitted);
        assert_eq!(updated.submitted_at_block, Some(100));
    }

    // ── Workflow — non-Submitted no-ops ──────────────────────────────────

    #[test]
    fn mark_stale_if_overdue_is_noop_for_pending_record() {
        let (_dir, reg) = open_temp_registry();
        let att = make_attestation("sess-pending", "addr-1", "sig-1");
        let r = reg.insert(att).unwrap();
        let policy = StalenessPolicy::new(1).unwrap();

        let updated = mark_stale_if_overdue(&reg, &r.id, 1_000_000, &policy).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Pending);
    }

    #[test]
    fn mark_stale_if_overdue_is_noop_for_included_record() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-included", 10);
        reg.mark_included(&id).unwrap();
        let policy = StalenessPolicy::new(1).unwrap();

        let updated = mark_stale_if_overdue(&reg, &id, 1_000_000, &policy).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Included);
    }

    #[test]
    fn mark_stale_if_overdue_is_noop_for_finalized_record() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-final", 10);
        reg.mark_finalized(&id).unwrap();
        let policy = StalenessPolicy::new(1).unwrap();

        let updated = mark_stale_if_overdue(&reg, &id, 1_000_000, &policy).unwrap();
        assert_eq!(updated.status, LocalAttestationStatus::Finalized);
    }

    #[test]
    fn mark_stale_if_overdue_is_noop_for_failed_record() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-failed", 10);
        reg.mark_failed(&id, "boom".into()).unwrap();
        let policy = StalenessPolicy::new(1).unwrap();

        let updated = mark_stale_if_overdue(&reg, &id, 1_000_000, &policy).unwrap();
        assert!(matches!(
            updated.status,
            LocalAttestationStatus::Failed { ref reason } if reason == "boom"
        ));
    }

    #[test]
    fn mark_stale_if_overdue_is_noop_for_already_dropped_record() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-dropped", 10);
        reg.mark_dropped(&id, Some("first-pass".into())).unwrap();
        let policy = StalenessPolicy::new(1).unwrap();

        let updated = mark_stale_if_overdue(&reg, &id, 1_000_000, &policy).unwrap();
        assert!(matches!(
            updated.status,
            LocalAttestationStatus::Dropped { ref reason } if reason.as_deref() == Some("first-pass")
        ));
    }

    // ── Retry hinge — Dropped → Submitted with new block ─────────────────

    /// After Stage 5.2 drops a record, a retry via
    /// `mark_submitted_with_block` must stamp the new height and clear
    /// the previous staleness `error_message`.
    #[test]
    fn retry_after_stale_drop_records_new_block() {
        let (_dir, reg) = open_temp_registry();
        let id = setup_submitted_at_block(&reg, "sess-retry", 10);
        let policy = StalenessPolicy::new(1).unwrap();

        // First pass: drop as stale.
        mark_stale_if_overdue(&reg, &id, 100, &policy).unwrap();
        let dropped = reg.load(&id).unwrap();
        assert!(matches!(dropped.status, LocalAttestationStatus::Dropped { .. }));

        // Retry from Dropped: new submission with a fresher block.
        let retried = reg
            .mark_submitted_with_block(&id, dummy_receipt("tx-retry"), 110)
            .unwrap();
        assert_eq!(retried.status, LocalAttestationStatus::Submitted);
        assert_eq!(retried.receipt.as_ref().unwrap().tx_id, "tx-retry");
        assert_eq!(retried.submitted_at_block, Some(110));
        assert!(retried.error_message.is_none());

        // Same threshold=1; elapsed = 111-110 = 1; strict `>` says
        // not stale yet. End-to-end sanity: the retry's fresh
        // `submitted_at_block` is now the reference point, not the
        // pre-drop one.
        let still_submitted =
            mark_stale_if_overdue(&reg, &id, 111, &policy).unwrap();
        assert_eq!(still_submitted.status, LocalAttestationStatus::Submitted);
    }
}
