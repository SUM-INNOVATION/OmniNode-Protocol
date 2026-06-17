//! Phase 5 Stage 13.3 — integration tests for the
//! operator-facing operations helpers: registry summary,
//! registry health, and time-based stale detection.
//!
//! All hermetic. No chain interaction. Tests exercise the
//! helpers on real on-disk registries seeded via the public
//! workflow API (so the on-disk shape is byte-equal to what
//! production produces) and via direct file writes (for the
//! health-check edge cases: malformed records, orphan tx_index
//! entries, orphan `.tmp` files).
//!
//! Pinned: summary helper does NOT mutate the registry.

use std::path::PathBuf;

use chrono::{Duration, Utc};
use omni_zkml::{
    anchor_signer_pubkey_bytes, build_anchor_digest,
    check_evidence_anchor_registry_health, list_evidence_anchors_by_status,
    list_stale_submitted_or_included, submit_evidence_anchor_workflow,
    EvidenceAnchorRegistryHealth, EvidenceAnchorRegistrySummary, LocalAnchorStatus,
    LocalEvidenceAnchorRegistry, StubEvidenceAnchorChainClient, VerifiedWrapperMetadata,
};

// ── Fixtures ──────────────────────────────────────────────────────────────────

fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("anchors");
    let reg = LocalEvidenceAnchorRegistry::open(root.clone()).unwrap();
    (dir, reg, root)
}

/// Submit one anchor through the real workflow + stub client.
/// The on-disk record lands as `Submitted` with `submitted_at =
/// Utc::now()`. Returns the artifact-hash hex so the caller can
/// post-process the record (e.g. flip its status) if needed.
fn seed_submitted(
    registry: &LocalEvidenceAnchorRegistry,
    seed: [u8; 32],
    raw_marker: u8,
) -> String {
    let client = StubEvidenceAnchorChainClient::new();
    let raw = vec![raw_marker; 32 + raw_marker as usize % 7];
    let metadata = VerifiedWrapperMetadata {
        artifact_schema_version: 1,
        signer_pubkey: anchor_signer_pubkey_bytes(&seed).unwrap(),
        signed_at_utc_unix: 1_750_000_000,
    };
    let digest = build_anchor_digest(&metadata, &raw);
    let record = submit_evidence_anchor_workflow(registry, &client, digest, &seed).unwrap();
    record.artifact_hash_hex
}

/// Flip a record's status by reading, mutating, and rewriting
/// its JSON file directly. Mirrors what
/// `LocalEvidenceAnchorRegistry::update_status` would do; using
/// the public method here keeps the integration test honest.
fn set_status(
    registry: &LocalEvidenceAnchorRegistry,
    artifact_hash_hex: &str,
    target: LocalAnchorStatus,
) {
    registry.update_status(artifact_hash_hex, target).unwrap();
}

/// Backdate a record's `submitted_at` field directly via JSON
/// rewrite. Used to make stale-detection tests deterministic
/// without sleeping.
fn backdate_submitted_at(
    registry: &LocalEvidenceAnchorRegistry,
    artifact_hash_hex: &str,
    seconds_ago: u64,
) {
    let path = registry
        .root()
        .join(format!("{artifact_hash_hex}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let past = Utc::now() - Duration::seconds(seconds_ago as i64);
    value["submitted_at"] = serde_json::Value::String(past.to_rfc3339());
    let new_bytes = serde_json::to_vec_pretty(&value).unwrap();
    std::fs::write(&path, new_bytes).unwrap();
}

// ── 1. summary_counts_records_by_status ──────────────────────────────────────

#[test]
fn summary_counts_records_by_status() {
    let (_dir, reg, _root) = fresh_registry();
    let h_sub = seed_submitted(&reg, [1u8; 32], 0x11);
    let h_inc = seed_submitted(&reg, [2u8; 32], 0x22);
    let h_fin = seed_submitted(&reg, [3u8; 32], 0x33);
    let h_fail = seed_submitted(&reg, [4u8; 32], 0x44);
    let _ = seed_submitted(&reg, [5u8; 32], 0x55); // second Submitted

    set_status(&reg, &h_inc, LocalAnchorStatus::Included);
    set_status(&reg, &h_fin, LocalAnchorStatus::Finalized);
    set_status(
        &reg,
        &h_fail,
        LocalAnchorStatus::Failed {
            reason: "test".into(),
        },
    );

    let s = list_evidence_anchors_by_status(&reg).unwrap();
    assert_eq!(
        s,
        EvidenceAnchorRegistrySummary {
            total: 5,
            submitted: 2,
            included: 1,
            finalized: 1,
            failed: 1,
        }
    );
    let _ = h_sub;
}

// ── 2. summary_handles_empty_registry ────────────────────────────────────────

#[test]
fn summary_handles_empty_registry() {
    let (_dir, reg, _root) = fresh_registry();
    let s = list_evidence_anchors_by_status(&reg).unwrap();
    assert_eq!(s.total, 0);
    assert_eq!(s.submitted, 0);
    assert_eq!(s.included, 0);
    assert_eq!(s.finalized, 0);
    assert_eq!(s.failed, 0);
}

// ── 3. stale_detection_finds_records_past_threshold ──────────────────────────

#[test]
fn stale_detection_finds_records_past_threshold() {
    let (_dir, reg, _root) = fresh_registry();
    let stale_hash = seed_submitted(&reg, [1u8; 32], 0x11);
    let fresh_hash = seed_submitted(&reg, [2u8; 32], 0x22);

    // Backdate one record 3600 seconds; leave the other fresh.
    backdate_submitted_at(&reg, &stale_hash, 3600);

    // Threshold = 60 seconds: stale_hash should report; fresh
    // should not.
    let stale = list_stale_submitted_or_included(&reg, Utc::now(), 60).unwrap();
    assert_eq!(stale.len(), 1, "expected exactly one stale entry");
    assert_eq!(stale[0].artifact_hash_hex, stale_hash);
    assert!(stale[0].age_secs >= 3600);
    assert!(matches!(stale[0].status, LocalAnchorStatus::Submitted));
    let _ = fresh_hash;
}

// ── 4. stale_detection_skips_finalized_and_failed_records ────────────────────

#[test]
fn stale_detection_skips_finalized_and_failed_records() {
    let (_dir, reg, _root) = fresh_registry();
    let h_fin = seed_submitted(&reg, [1u8; 32], 0x11);
    let h_fail = seed_submitted(&reg, [2u8; 32], 0x22);
    let h_inc = seed_submitted(&reg, [3u8; 32], 0x33);

    // Backdate all three so they would qualify on age alone.
    backdate_submitted_at(&reg, &h_fin, 9_000);
    backdate_submitted_at(&reg, &h_fail, 9_000);
    backdate_submitted_at(&reg, &h_inc, 9_000);

    // Flip the terminal ones; leave the Included one open.
    set_status(&reg, &h_fin, LocalAnchorStatus::Finalized);
    set_status(
        &reg,
        &h_fail,
        LocalAnchorStatus::Failed {
            reason: "terminal".into(),
        },
    );
    set_status(&reg, &h_inc, LocalAnchorStatus::Included);

    let stale = list_stale_submitted_or_included(&reg, Utc::now(), 60).unwrap();
    assert_eq!(stale.len(), 1, "only Included should report stale");
    assert_eq!(stale[0].artifact_hash_hex, h_inc);
    assert!(matches!(stale[0].status, LocalAnchorStatus::Included));
}

// ── 5. stale_detection_handles_threshold_zero_gracefully ─────────────────────

#[test]
fn stale_detection_handles_threshold_zero_gracefully() {
    let (_dir, reg, _root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);
    let _ = seed_submitted(&reg, [2u8; 32], 0x22);
    set_status(
        &reg,
        &seed_submitted(&reg, [3u8; 32], 0x33),
        LocalAnchorStatus::Finalized,
    );

    let stale = list_stale_submitted_or_included(&reg, Utc::now(), 0).unwrap();
    // Threshold 0 reports every Submitted/Included record (age
    // is always >= 0). Terminal records still skipped.
    assert_eq!(stale.len(), 2);
}

// ── 5b. stale_detection_skips_future_dated_submitted_at ──────────────────────

#[test]
fn stale_detection_skips_future_dated_submitted_at() {
    // Defends the clock-skew underflow guard documented on
    // `list_stale_submitted_or_included`.
    let (_dir, reg, _root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    // Set submitted_at to 60 seconds in the future.
    let future = Utc::now() + Duration::seconds(60);
    let path = reg.root().join(format!("{h}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["submitted_at"] = serde_json::Value::String(future.to_rfc3339());
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

    let stale = list_stale_submitted_or_included(&reg, Utc::now(), 0).unwrap();
    assert!(
        stale.is_empty(),
        "future-dated submitted_at must NOT be reported as stale"
    );
}

// ── 6. health_reports_orphan_tx_index_entries ────────────────────────────────

#[test]
fn health_reports_orphan_tx_index_entries() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);

    // Hand-edit tx_index.json to add an entry mapping a phantom
    // tx_id to a phantom artifact-hash that doesn't exist on
    // disk.
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let phantom_hash = "ff".repeat(32);
    idx["by_tx_id"]["phantom-tx"] = serde_json::Value::String(phantom_hash);
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();

    let h = check_evidence_anchor_registry_health(&reg).unwrap();
    assert_eq!(h.records, 1);
    assert_eq!(h.orphan_tx_index_entries, 1);
    assert_eq!(h.orphan_tmp_files, 0);
    assert_eq!(h.malformed_records, 0);
}

// ── 7. health_reports_orphan_tmp_files ───────────────────────────────────────

#[test]
fn health_reports_orphan_tmp_files() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);

    // Drop a stray `<hash>.json.tmp` in the registry root.
    let tmp_path = root.join("deadbeef.json.tmp");
    std::fs::write(&tmp_path, b"{stale}").unwrap();

    let h = check_evidence_anchor_registry_health(&reg).unwrap();
    assert_eq!(h.records, 1);
    assert_eq!(h.orphan_tmp_files, 1);
    assert_eq!(h.orphan_tx_index_entries, 0);
    assert_eq!(h.malformed_records, 0);
}

// ── 8. health_reports_malformed_records ──────────────────────────────────────

#[test]
fn health_reports_malformed_records() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);

    // Drop a 64-hex-named JSON file that doesn't parse as
    // AnchorRecord.
    let bogus_name = format!("{}.json", "a".repeat(64));
    std::fs::write(root.join(&bogus_name), b"{not anchor json}").unwrap();

    let h = check_evidence_anchor_registry_health(&reg).unwrap();
    assert_eq!(h.records, 1);
    assert_eq!(h.malformed_records, 1);
    assert_eq!(h.orphan_tmp_files, 0);
    assert_eq!(h.orphan_tx_index_entries, 0);
}

// ── 9. summary_helper_does_not_mutate_registry ───────────────────────────────

#[test]
fn summary_helper_does_not_mutate_registry() {
    let (_dir, reg, root) = fresh_registry();
    let h_a = seed_submitted(&reg, [1u8; 32], 0x11);
    let h_b = seed_submitted(&reg, [2u8; 32], 0x22);

    let mut pre: Vec<(std::ffi::OsString, Vec<u8>)> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            let p = e.path();
            (
                p.file_name().unwrap().to_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    pre.sort_by(|a, b| a.0.cmp(&b.0));

    // Call all three helpers.
    let _ = list_evidence_anchors_by_status(&reg).unwrap();
    let _ = check_evidence_anchor_registry_health(&reg).unwrap();
    let _ = list_stale_submitted_or_included(&reg, Utc::now(), 0).unwrap();

    let mut post: Vec<(std::ffi::OsString, Vec<u8>)> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            let p = e.path();
            (
                p.file_name().unwrap().to_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    post.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(
        pre, post,
        "summary / health / stale helpers must not mutate the registry"
    );
    let _ = (h_a, h_b);
}

// ── Bonus: health on an empty registry ───────────────────────────────────────

#[test]
fn health_empty_registry_returns_zero_counts() {
    let (_dir, reg, _root) = fresh_registry();
    let h = check_evidence_anchor_registry_health(&reg).unwrap();
    assert_eq!(
        h,
        EvidenceAnchorRegistryHealth {
            records: 0,
            malformed_records: 0,
            orphan_tx_index_entries: 0,
            orphan_tmp_files: 0,
        }
    );
}
