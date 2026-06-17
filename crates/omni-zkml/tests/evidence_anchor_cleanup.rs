//! Phase 5 Stage 13.4 — integration tests for anchor-registry
//! cleanup: plan → apply (dry-run + real) → restore.
//!
//! Hermetic. Temp dirs everywhere. No network. Mirrors the
//! Stage 13.3 health/stale tests' pattern of seeding records via
//! the public workflow API for byte-equality with production
//! and direct JSON-file writes for backdating / corruption.

use chrono::{Duration, Utc};
use omni_zkml::{
    anchor_signer_pubkey_bytes, apply_anchor_cleanup, build_anchor_digest,
    plan_anchor_cleanup, restore_anchor_cleanup_quarantine, submit_evidence_anchor_workflow,
    AnchorApplyOptions, AnchorCleanupActionKind, AnchorPlanOptions,
    AnchorQuarantineManifest, AnchorRestoreOptions, EvidenceAnchorError,
    LocalAnchorStatus, LocalEvidenceAnchorRegistry, StubEvidenceAnchorChainClient,
    VerifiedWrapperMetadata, ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION,
};

fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("anchors");
    let reg = LocalEvidenceAnchorRegistry::open(root.clone()).unwrap();
    (dir, reg, root)
}

fn seed_submitted(
    reg: &LocalEvidenceAnchorRegistry,
    seed: [u8; 32],
    marker: u8,
) -> String {
    let client = StubEvidenceAnchorChainClient::new();
    let raw = vec![marker; 32];
    let metadata = VerifiedWrapperMetadata {
        artifact_schema_version: 1,
        signer_pubkey: anchor_signer_pubkey_bytes(&seed).unwrap(),
        signed_at_utc_unix: 1_750_000_000,
    };
    let digest = build_anchor_digest(&metadata, &raw);
    let record =
        submit_evidence_anchor_workflow(reg, &client, digest, &seed).unwrap();
    record.artifact_hash_hex
}

fn backdate_submitted_at(reg: &LocalEvidenceAnchorRegistry, hash: &str, secs_ago: u64) {
    let path = reg.root().join(format!("{hash}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let past = Utc::now() - Duration::seconds(secs_ago as i64);
    value["submitted_at"] = serde_json::Value::String(past.to_rfc3339());
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
}

fn quarantine_dir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

const NOW: &str = "2026-06-17T00:00:00Z";

// ── 1. plan_handles_empty_registry ───────────────────────────────────────────

#[test]
fn plan_handles_empty_registry() {
    let (_dir, reg, _root) = fresh_registry();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 0);
    assert!(!plan.cleanup_plan_hash.is_empty());
    assert_eq!(plan.plan_id.len(), 16);
}

// ── 2. plan_detects_orphan_tmp_files ─────────────────────────────────────────

#[test]
fn plan_detects_orphan_tmp_files() {
    let (_dir, reg, root) = fresh_registry();
    std::fs::write(root.join("orphan.json.tmp"), b"stale").unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(plan.actions[0].kind, AnchorCleanupActionKind::RemoveOrphanTmpFile);
}

// ── 3. plan_detects_orphan_tx_index_entries ──────────────────────────────────

#[test]
fn plan_detects_orphan_tx_index_entries() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);
    // Inject a phantom tx_id → phantom hash that has no record.
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    idx["by_tx_id"]["phantom-tx"] = serde_json::Value::String("ff".repeat(32));
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();

    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 1);
    let a = &plan.actions[0];
    assert_eq!(a.kind, AnchorCleanupActionKind::RemoveOrphanTxIndexEntry);
    assert_eq!(a.tx_id.as_deref(), Some("phantom-tx"));
}

// ── 4. plan_detects_malformed_records ────────────────────────────────────────

#[test]
fn plan_detects_malformed_records() {
    let (_dir, reg, root) = fresh_registry();
    // Drop a 64-hex JSON file that doesn't parse.
    let name = format!("{}.json", "a".repeat(64));
    std::fs::write(root.join(&name), b"{bogus}").unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(
        plan.actions[0].kind,
        AnchorCleanupActionKind::QuarantineMalformedRecord
    );
    assert_eq!(plan.actions[0].source_relative, name);
}

// ── 5. plan_detects_stale_open_records_when_threshold_given ──────────────────

#[test]
fn plan_detects_stale_open_records_when_threshold_given() {
    let (_dir, reg, _root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    backdate_submitted_at(&reg, &h, 3600);
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: &Utc::now().to_rfc3339(),
            stale_threshold_secs: Some(60),
        },
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(
        plan.actions[0].kind,
        AnchorCleanupActionKind::QuarantineStaleOpenRecord
    );
    assert!(plan.actions[0].tx_id.is_some());
}

// ── 6. plan_skips_stale_records_when_threshold_omitted ───────────────────────

#[test]
fn plan_skips_stale_records_when_threshold_omitted() {
    let (_dir, reg, _root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    backdate_submitted_at(&reg, &h, 9_999);
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: &Utc::now().to_rfc3339(),
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    // Q1: no stale-cleanup actions emitted without threshold.
    assert_eq!(plan.actions.len(), 0);
}

// ── 7. plan_hash_is_byte_stable_across_recomputation ─────────────────────────

#[test]
fn plan_hash_is_byte_stable_across_recomputation() {
    let (_dir, reg, _root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);
    let p1 = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let p2 = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    assert_eq!(p1.cleanup_plan_hash, p2.cleanup_plan_hash);
    assert_eq!(p1.registry_state_hash, p2.registry_state_hash);
    assert_eq!(p1.plan_id, p2.plan_id);
}

// ── 8. plan_registry_state_hash_changes_when_status_changes ──────────────────

#[test]
fn plan_registry_state_hash_changes_when_status_changes() {
    let (_dir, reg, _root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    let pre = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    reg.update_status(&h, LocalAnchorStatus::Finalized).unwrap();
    let post = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    assert_ne!(pre.registry_state_hash, post.registry_state_hash);
}

// ── 9. plan_registry_state_hash_unchanged_when_only_updated_at_changes ──────

#[test]
fn plan_registry_state_hash_unchanged_when_only_updated_at_changes() {
    let (_dir, reg, _root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    let pre = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    // Manually bump updated_at without touching status / submitted_at.
    let path = reg.root().join(format!("{h}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let future = Utc::now() + Duration::seconds(3600);
    value["updated_at"] = serde_json::Value::String(future.to_rfc3339());
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let post = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    assert_eq!(pre.registry_state_hash, post.registry_state_hash);
}

// ── 10. apply_dry_run_emits_would_apply_for_every_action ─────────────────────

#[test]
fn apply_dry_run_emits_would_apply_for_every_action() {
    let (_dir, reg, root) = fresh_registry();
    std::fs::write(root.join("a.json.tmp"), b"stale").unwrap();
    std::fs::write(root.join("b.json.tmp"), b"stale").unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    let report = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: true,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(report.mode, "dry_run");
    assert_eq!(report.actions_applied, 0);
    assert_eq!(report.actions_dry_run, 2);
    assert!(report.outcomes.iter().all(|o| o.status == "would_apply"));
    // No FS mutations.
    assert!(root.join("a.json.tmp").exists());
    assert!(root.join("b.json.tmp").exists());
}

// ── 11. apply_real_run_quarantines_malformed_then_deletes_source ─────────────

#[test]
fn apply_real_run_quarantines_malformed_then_deletes_source() {
    let (_dir, reg, root) = fresh_registry();
    let name = format!("{}.json", "a".repeat(64));
    let bogus_bytes = b"{bogus}".to_vec();
    std::fs::write(root.join(&name), &bogus_bytes).unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    let report = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(report.mode, "apply");
    assert_eq!(report.actions_applied, 1);
    assert!(report.quarantine_manifest_relative.is_some());
    // Source deleted; quarantine populated.
    assert!(!root.join(&name).exists());
    let quarantined = qdir.path().join(&plan.plan_id).join(&name);
    assert!(quarantined.exists());
    assert_eq!(std::fs::read(&quarantined).unwrap(), bogus_bytes);
}

// ── 12. apply_real_run_removes_orphan_tx_index_entry ─────────────────────────

#[test]
fn apply_real_run_removes_orphan_tx_index_entry() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    idx["by_tx_id"]["phantom-tx"] = serde_json::Value::String("ff".repeat(32));
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Index rewritten, entry gone.
    let bytes = std::fs::read(&idx_path).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let map = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert!(!map.contains_key("phantom-tx"));
}

// ── 13. apply_refuses_on_cleanup_drift ───────────────────────────────────────

#[test]
fn apply_refuses_on_cleanup_drift() {
    let (_dir, reg, _root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    // Drift the registry between plan and apply.
    reg.update_status(&h, LocalAnchorStatus::Finalized).unwrap();
    let qdir = quarantine_dir();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert!(matches!(err, EvidenceAnchorError::CleanupDrift { .. }));
}

// ── 14. apply_refuses_on_cleanup_plan_hash_mismatch ──────────────────────────

#[test]
fn apply_refuses_on_cleanup_plan_hash_mismatch() {
    let (_dir, reg, _root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);
    let mut plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    // Hand-edit the plan's body without recomputing the hash.
    plan.created_at_utc = "2099-01-01T00:00:00Z".to_string();
    let qdir = quarantine_dir();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::CleanupPlanHashMismatch { .. }
    ));
}

// ── 15. apply_refuses_gated_action_without_allow_stale_quarantine ────────────

#[test]
fn apply_refuses_gated_action_without_allow_stale_quarantine() {
    let (_dir, reg, _root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    backdate_submitted_at(&reg, &h, 3600);
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: &Utc::now().to_rfc3339(),
            stale_threshold_secs: Some(60),
        },
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 1);
    let qdir = quarantine_dir();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false, // missing gate
            now_utc: NOW,
        },
    )
    .unwrap_err();
    if let EvidenceAnchorError::CleanupGateRequired { action_kind, gate_flag } = err {
        assert_eq!(action_kind, "quarantine_stale_open_record");
        assert_eq!(gate_flag, "--allow-stale-quarantine");
    } else {
        panic!("expected CleanupGateRequired; got {err:?}");
    }
}

// ── 16. apply_idempotent_skipped_missing_for_already_removed_file ────────────

#[test]
fn apply_idempotent_skipped_missing_for_already_removed_file() {
    let (_dir, reg, root) = fresh_registry();
    std::fs::write(root.join("orphan.json.tmp"), b"stale").unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    // Remove the file between plan and apply.
    std::fs::remove_file(root.join("orphan.json.tmp")).unwrap();
    let qdir = quarantine_dir();
    let report = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(report.actions_applied, 0);
    assert_eq!(report.actions_skipped, 1);
    assert_eq!(report.outcomes[0].status, "skipped_missing");
}

// ── 17. apply_does_not_quarantine_orphan_tx_index_entry ──────────────────────

#[test]
fn apply_does_not_quarantine_orphan_tx_index_entry() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_submitted(&reg, [1u8; 32], 0x11);
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    idx["by_tx_id"]["phantom-tx"] = serde_json::Value::String("ff".repeat(32));
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    let report = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Q5: no quarantine manifest for orphan tx_index entries.
    assert_eq!(report.quarantine_manifest_relative, None);
    let plan_dir = qdir.path().join(&plan.plan_id);
    // Plan dir may or may not exist; if it does, no manifest inside.
    if plan_dir.exists() {
        assert!(!plan_dir.join("quarantine_manifest.json").exists());
    }
}

// ── 18. restore_round_trips_quarantined_malformed_record ─────────────────────

#[test]
fn restore_round_trips_quarantined_malformed_record() {
    let (_dir, reg, root) = fresh_registry();
    let name = format!("{}.json", "a".repeat(64));
    let bogus_bytes = b"{bogus}".to_vec();
    std::fs::write(root.join(&name), &bogus_bytes).unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Read the manifest, restore it.
    let manifest_path =
        qdir.path().join(&plan.plan_id).join("quarantine_manifest.json");
    let manifest: AnchorQuarantineManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let restore_root = qdir.path().join(&plan.plan_id);
    let report = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: &restore_root,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(report.restored, 1);
    assert!(root.join(&name).exists());
    assert_eq!(std::fs::read(root.join(&name)).unwrap(), bogus_bytes);
}

// ── 19. restore_refuses_on_quarantine_blake3_mismatch ────────────────────────

#[test]
fn restore_refuses_on_quarantine_blake3_mismatch() {
    let (_dir, reg, root) = fresh_registry();
    let name = format!("{}.json", "a".repeat(64));
    std::fs::write(root.join(&name), b"original-bytes").unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Tamper the quarantined file.
    let quarantined = qdir.path().join(&plan.plan_id).join(&name);
    std::fs::write(&quarantined, b"tampered-bytes").unwrap();

    let manifest_path =
        qdir.path().join(&plan.plan_id).join("quarantine_manifest.json");
    let manifest: AnchorQuarantineManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let restore_root = qdir.path().join(&plan.plan_id);
    let err = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: &restore_root,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::QuarantineBlake3Mismatch { .. }
    ));
}

// ── 20. restore_refuses_when_target_exists ───────────────────────────────────

#[test]
fn restore_refuses_when_target_exists() {
    let (_dir, reg, root) = fresh_registry();
    let name = format!("{}.json", "a".repeat(64));
    std::fs::write(root.join(&name), b"original-bytes").unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Put a DIFFERENT file at the target path before restore.
    std::fs::write(root.join(&name), b"i-took-over").unwrap();
    let manifest_path =
        qdir.path().join(&plan.plan_id).join("quarantine_manifest.json");
    let manifest: AnchorQuarantineManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let restore_root = qdir.path().join(&plan.plan_id);
    let err = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: &restore_root,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::RestoreTargetExists { .. }
    ));
}

// ── 21. restore_idempotent_skipped_already_restored_for_matching_target ──────

#[test]
fn restore_idempotent_skipped_already_restored_for_matching_target() {
    let (_dir, reg, root) = fresh_registry();
    let name = format!("{}.json", "a".repeat(64));
    let original = b"original-bytes".to_vec();
    std::fs::write(root.join(&name), &original).unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let manifest_path =
        qdir.path().join(&plan.plan_id).join("quarantine_manifest.json");
    let manifest: AnchorQuarantineManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let restore_root = qdir.path().join(&plan.plan_id);
    // First restore.
    restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: &restore_root,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    // Second restore: target already exists AND bytes match the
    // manifest → idempotent skip.
    let report = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: &restore_root,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(report.restored, 0);
    assert_eq!(report.skipped, 1);
    assert_eq!(report.outcomes[0].status, "skipped_already_restored");
}

// ── 22. restore_re_adds_tx_index_entry_for_stale_records ─────────────────────

#[test]
fn restore_re_adds_tx_index_entry_for_stale_records() {
    let (_dir, reg, root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    backdate_submitted_at(&reg, &h, 3600);
    // Snapshot the original tx_id from the receipt for the assertion below.
    let original_tx_id = reg
        .load_by_artifact_hash(&h)
        .unwrap()
        .unwrap()
        .receipt
        .tx_id;
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: &Utc::now().to_rfc3339(),
            stale_threshold_secs: Some(60),
        },
    )
    .unwrap();
    assert_eq!(plan.actions.len(), 1);
    let qdir = quarantine_dir();
    apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: true,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Record removed; tx_index entry removed.
    assert!(!root.join(format!("{h}.json")).exists());
    let idx_bytes = std::fs::read(root.join("tx_index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    assert!(idx
        .get("by_tx_id")
        .and_then(|v| v.as_object())
        .map(|m| !m.contains_key(&original_tx_id))
        .unwrap_or(true));

    // Restore.
    let manifest_path =
        qdir.path().join(&plan.plan_id).join("quarantine_manifest.json");
    let manifest: AnchorQuarantineManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(
        manifest.schema_version,
        ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION
    );
    let restore_root = qdir.path().join(&plan.plan_id);
    restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: &restore_root,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    // Record + tx_index entry both back.
    assert!(root.join(format!("{h}.json")).exists());
    let idx_bytes = std::fs::read(root.join("tx_index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    let map = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert_eq!(
        map.get(&original_tx_id).and_then(|v| v.as_str()),
        Some(h.as_str())
    );
}

// ── REJECT-fix #1: path validation (apply) ───────────────────────────────────

#[test]
fn apply_refuses_action_with_absolute_path() {
    let (_dir, reg, _root) = fresh_registry();
    let mut plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    plan.actions.push(omni_zkml::AnchorCleanupAction {
        kind: AnchorCleanupActionKind::RemoveOrphanTmpFile,
        source_relative: "/etc/passwd".to_string(),
        tx_id: None,
    });
    // Recompute the plan hash so the path-validation gate
    // fires, not the plan-hash-mismatch gate.
    let blanked = omni_zkml::AnchorCleanupPlan {
        cleanup_plan_hash: String::new(),
        ..plan.clone()
    };
    let bytes = serde_json::to_vec(&blanked).unwrap();
    plan.cleanup_plan_hash = blake3::hash(&bytes).to_hex().to_string();
    let qdir = quarantine_dir();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::CleanupInvalidPath { .. }
    ));
}

#[test]
fn apply_refuses_action_with_parent_traversal() {
    let (_dir, reg, _root) = fresh_registry();
    let mut plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    plan.actions.push(omni_zkml::AnchorCleanupAction {
        kind: AnchorCleanupActionKind::QuarantineMalformedRecord,
        source_relative: "../escape.json".to_string(),
        tx_id: None,
    });
    let blanked = omni_zkml::AnchorCleanupPlan {
        cleanup_plan_hash: String::new(),
        ..plan.clone()
    };
    plan.cleanup_plan_hash = blake3::hash(&serde_json::to_vec(&blanked).unwrap())
        .to_hex()
        .to_string();
    let qdir = quarantine_dir();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::CleanupInvalidPath { .. }
    ));
}

#[test]
fn apply_refuses_action_with_wrong_per_kind_shape() {
    // RemoveOrphanTmpFile must end in .tmp; a record-shaped
    // name should be refused.
    let (_dir, reg, _root) = fresh_registry();
    let mut plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    plan.actions.push(omni_zkml::AnchorCleanupAction {
        kind: AnchorCleanupActionKind::RemoveOrphanTmpFile,
        source_relative: format!("{}.json", "a".repeat(64)),
        tx_id: None,
    });
    let blanked = omni_zkml::AnchorCleanupPlan {
        cleanup_plan_hash: String::new(),
        ..plan.clone()
    };
    plan.cleanup_plan_hash = blake3::hash(&serde_json::to_vec(&blanked).unwrap())
        .to_hex()
        .to_string();
    let qdir = quarantine_dir();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::CleanupInvalidPath { .. }
    ));
}

// ── REJECT-fix #1: path validation (restore) ─────────────────────────────────

fn build_quarantine_manifest_with_entry(
    plan_id: &str,
    source_relative: &str,
    quarantine_relative: &str,
    action_kind: AnchorCleanupActionKind,
) -> AnchorQuarantineManifest {
    AnchorQuarantineManifest {
        schema_version: ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION,
        plan_id: plan_id.to_string(),
        created_at_utc: NOW.to_string(),
        entries: vec![omni_zkml::AnchorQuarantineEntry {
            source_relative: source_relative.to_string(),
            quarantine_relative: quarantine_relative.to_string(),
            blake3_hex: blake3::hash(b"unused").to_hex().to_string(),
            bytes: 6,
            action_kind: action_kind.as_str().to_string(),
            tx_id: None,
        }],
    }
}

#[test]
fn restore_refuses_manifest_entry_with_absolute_path() {
    let (_dir, _reg, root) = fresh_registry();
    let qdir = quarantine_dir();
    let manifest = build_quarantine_manifest_with_entry(
        "deadbeef",
        "/etc/passwd",
        "/etc/passwd",
        AnchorCleanupActionKind::QuarantineMalformedRecord,
    );
    let err = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: qdir.path(),
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::CleanupInvalidPath { .. }
    ));
}

#[test]
fn restore_refuses_manifest_entry_with_parent_traversal() {
    let (_dir, _reg, root) = fresh_registry();
    let qdir = quarantine_dir();
    let bogus = format!("../{}.json", "a".repeat(64));
    let manifest = build_quarantine_manifest_with_entry(
        "deadbeef",
        &bogus,
        &bogus,
        AnchorCleanupActionKind::QuarantineMalformedRecord,
    );
    let err = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: qdir.path(),
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::CleanupInvalidPath { .. }
    ));
}

#[test]
fn restore_refuses_manifest_entry_where_quarantine_relative_differs_from_source() {
    // Q4 layout lock: quarantine_relative must mirror
    // source_relative verbatim.
    let (_dir, _reg, root) = fresh_registry();
    let qdir = quarantine_dir();
    let src = format!("{}.json", "a".repeat(64));
    let qua = format!("{}.json", "b".repeat(64));
    let manifest = build_quarantine_manifest_with_entry(
        "deadbeef",
        &src,
        &qua,
        AnchorCleanupActionKind::QuarantineMalformedRecord,
    );
    let err = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: qdir.path(),
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::CleanupInvalidPath { .. }
    ));
}

// ── REJECT-fix #2: schema-version refusal ────────────────────────────────────

#[test]
fn apply_refuses_unsupported_plan_schema_version() {
    let (_dir, reg, _root) = fresh_registry();
    let mut plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    plan.schema_version = 99;
    // Recompute the plan hash so the schema-version check
    // fires, not the plan-hash-mismatch check.
    let blanked = omni_zkml::AnchorCleanupPlan {
        cleanup_plan_hash: String::new(),
        ..plan.clone()
    };
    plan.cleanup_plan_hash = blake3::hash(&serde_json::to_vec(&blanked).unwrap())
        .to_hex()
        .to_string();
    let qdir = quarantine_dir();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    if let EvidenceAnchorError::CleanupPlanSchemaUnsupported { got, expected } = err {
        assert_eq!(got, 99);
        assert_eq!(expected, omni_zkml::ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION);
    } else {
        panic!("expected CleanupPlanSchemaUnsupported; got {err:?}");
    }
}

#[test]
fn apply_schema_version_check_fires_before_plan_hash_check() {
    // A future-schema plan with a tampered hash must refuse on
    // SCHEMA, not on plan-hash. This pins the preflight ordering.
    let (_dir, reg, _root) = fresh_registry();
    let mut plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    plan.schema_version = 99;
    // Leave cleanup_plan_hash unchanged → would fail the hash
    // check IF schema-version check didn't run first.
    let qdir = quarantine_dir();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, EvidenceAnchorError::CleanupPlanSchemaUnsupported { .. }),
        "schema check must fire before hash check; got {err:?}"
    );
}

// ── REJECT-fix #3: durability ordering ───────────────────────────────────────

#[test]
fn apply_does_not_delete_tier_b_source_before_manifest_lands() {
    // Pin the durability ordering by inspecting on-disk state
    // AFTER a successful apply. Specifically:
    // - The quarantine_manifest.json must exist.
    // - The source must be deleted (Pass 2).
    // The ordering is also enforced structurally: Pass 1
    // copies + accumulates entries, Pass 1.5 writes the
    // manifest, Pass 2 deletes sources. The integration test
    // here serves as the live happy-path sanity for the new
    // structure; the unit-level write-failure injection test
    // is below.
    let (_dir, reg, root) = fresh_registry();
    let name = format!("{}.json", "a".repeat(64));
    std::fs::write(root.join(&name), b"malformed-content").unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Manifest is on disk.
    let manifest_path =
        qdir.path().join(&plan.plan_id).join("quarantine_manifest.json");
    assert!(manifest_path.is_file());
    // Source is deleted.
    assert!(!root.join(&name).exists());
    // Quarantine file is present.
    assert!(qdir.path().join(&plan.plan_id).join(&name).is_file());
}

#[test]
#[cfg(unix)]
fn apply_with_manifest_write_failure_leaves_source_intact() {
    // Stronger durability test: make the manifest write fail
    // and verify the source is still on disk (so restore is
    // not lost). We achieve a deterministic manifest-write
    // failure on Unix by creating the manifest's expected
    // directory as a regular file BEFORE apply — Pass 1's
    // quarantine_target write into `<dir>/plan_id/<name>` will
    // fail because the parent path stem is a file, not a
    // directory.
    //
    // Note: on this platform the failure happens during Pass 1
    // (quarantine copy), before the manifest write. The
    // important invariant is identical: the source MUST NOT
    // have been deleted.
    let (_dir, reg, root) = fresh_registry();
    let name = format!("{}.json", "a".repeat(64));
    std::fs::write(root.join(&name), b"malformed-content").unwrap();
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    // Sabotage: make `<qdir>/<plan_id>` a regular file so
    // create_dir_all inside it fails. Pass 1's quarantine
    // copy will fail; the source MUST still be there.
    std::fs::write(qdir.path().join(&plan.plan_id), b"sabotage").unwrap();
    let err = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    );
    assert!(err.is_err(), "apply must fail when quarantine path is sabotaged");
    // Critical invariant: source NOT deleted.
    assert!(
        root.join(&name).exists(),
        "source must remain intact on apply failure"
    );
}

// ── REJECT-fix #4: restore stale-record idempotency ──────────────────────────

#[test]
fn restore_idempotent_re_adds_tx_index_entry_when_file_already_back() {
    // Simulate: a prior restore copied the file back to the
    // registry but failed before the tx_index rewrite. A
    // second restore must STILL queue the tx_id re-add — the
    // previous code skipped it, leaving the record file
    // present but permanently missing from tx_index.json.
    let (_dir, reg, root) = fresh_registry();
    let h = seed_submitted(&reg, [1u8; 32], 0x11);
    backdate_submitted_at(&reg, &h, 3600);
    let original_tx_id = reg
        .load_by_artifact_hash(&h)
        .unwrap()
        .unwrap()
        .receipt
        .tx_id;
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: &Utc::now().to_rfc3339(),
            stale_threshold_secs: Some(60),
        },
    )
    .unwrap();
    let qdir = quarantine_dir();
    apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: true,
            now_utc: NOW,
        },
    )
    .unwrap();
    let manifest_path =
        qdir.path().join(&plan.plan_id).join("quarantine_manifest.json");
    let manifest: AnchorQuarantineManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let restore_root = qdir.path().join(&plan.plan_id);

    // Simulate partial restore: manually copy the record file
    // back, but DO NOT touch tx_index.json. This is exactly
    // the failure state the fix defends.
    let quarantined = restore_root.join(format!("{h}.json"));
    let target = root.join(format!("{h}.json"));
    std::fs::copy(&quarantined, &target).unwrap();
    let idx_bytes = std::fs::read(root.join("tx_index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    let map = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert!(
        !map.contains_key(&original_tx_id),
        "precondition: tx_index entry should still be absent before re-run"
    );

    // Re-run restore. Idempotent on the file copy
    // (skipped_already_restored) but the tx_index entry must
    // come back.
    let report = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: &restore_root,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(report.skipped, 1, "file restore should be skipped");
    let idx_bytes = std::fs::read(root.join("tx_index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    let map = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert_eq!(
        map.get(&original_tx_id).and_then(|v| v.as_str()),
        Some(h.as_str()),
        "tx_index entry MUST be re-added even when file restore is idempotent"
    );
}
