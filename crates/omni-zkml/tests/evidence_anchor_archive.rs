//! Phase 5 Stage 13.7 — integration tests for local terminal-
//! anchor archive / restore. Hermetic. Temp dirs everywhere. No
//! network.

use chrono::{Duration, Utc};
use omni_zkml::{
    anchor_signer_pubkey_bytes, apply_anchor_archive, build_anchor_digest,
    plan_anchor_archive, restore_anchor_archive, submit_evidence_anchor_workflow,
    AnchorArchiveApplyOptions, AnchorArchiveManifest, AnchorArchivePlan,
    AnchorArchivePlanOptions, AnchorArchiveRestoreOptions, AnchorArchiveSelection,
    EvidenceAnchorError, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
    StubEvidenceAnchorChainClient, VerifiedWrapperMetadata,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fresh_registry() -> (
    tempfile::TempDir,
    LocalEvidenceAnchorRegistry,
    std::path::PathBuf,
) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("anchors");
    let reg = LocalEvidenceAnchorRegistry::open(root.clone()).unwrap();
    (dir, reg, root)
}

/// Seed a record, then transition it to a chosen terminal
/// status. Returns (artifact_hash_hex, tx_id).
fn seed_terminal(
    reg: &LocalEvidenceAnchorRegistry,
    seed: [u8; 32],
    marker: u8,
    status: LocalAnchorStatus,
) -> (String, String) {
    let client = StubEvidenceAnchorChainClient::new();
    let raw = vec![marker; 32];
    let metadata = VerifiedWrapperMetadata {
        artifact_schema_version: 1,
        signer_pubkey: anchor_signer_pubkey_bytes(&seed).unwrap(),
        signed_at_utc_unix: 1_750_000_000,
    };
    let digest = build_anchor_digest(&metadata, &raw);
    let record = submit_evidence_anchor_workflow(reg, &client, digest, &seed).unwrap();
    let hash = record.artifact_hash_hex.clone();
    let tx_id = record.receipt.tx_id.clone();
    if !matches!(status, LocalAnchorStatus::Submitted) {
        reg.update_status(&hash, status).unwrap();
    }
    (hash, tx_id)
}

/// Seed a record left in `Submitted` (non-terminal).
fn seed_submitted(
    reg: &LocalEvidenceAnchorRegistry,
    seed: [u8; 32],
    marker: u8,
) -> (String, String) {
    seed_terminal(reg, seed, marker, LocalAnchorStatus::Submitted)
}

fn archive_dir() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

const NOW: &str = "2026-06-18T00:00:00Z";

fn plan_opts<'a>(
    statuses: Vec<LocalAnchorStatus>,
    selection: &'a AnchorArchiveSelection,
    before: Option<&'a str>,
) -> AnchorArchivePlanOptions<'a> {
    AnchorArchivePlanOptions {
        now_utc: NOW,
        statuses,
        before_utc: before,
        selection,
    }
}

fn assert_reason_tag(err: &EvidenceAnchorError, expected: &str) {
    let tag = omni_zkml::evidence_anchor_reason_tag(err);
    assert_eq!(tag, expected, "wrong tag on {err:?}");
}

fn read_tx_index(root: &std::path::Path) -> serde_json::Value {
    let bytes = std::fs::read(root.join("tx_index.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn read_manifest(archive_root_for_plan_id: &std::path::Path) -> AnchorArchiveManifest {
    let bytes = std::fs::read(archive_root_for_plan_id.join("archive_manifest.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn read_plan_from_disk(p: &std::path::Path) -> AnchorArchivePlan {
    let bytes = std::fs::read(p).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

// ── Plan / selection ──────────────────────────────────────────────────────────

#[test]
fn plan_defaults_to_finalized_only_when_no_status_supplied() {
    let (_dir, reg, _) = fresh_registry();
    let (h_finalized, _) = seed_terminal(&reg, [1u8; 32], 0x11, LocalAnchorStatus::Finalized);
    let _ = seed_terminal(&reg, [2u8; 32], 0x22, LocalAnchorStatus::Failed {
        reason: "x".into(),
    });
    let _ = seed_submitted(&reg, [3u8; 32], 0x33);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(plan.actions[0].artifact_hash_hex, h_finalized);
    assert_eq!(plan.actions[0].status, "finalized");
}

#[test]
fn plan_includes_failed_with_explicit_status_failed_opt_in() {
    let (_dir, reg, _) = fresh_registry();
    let (h_finalized, _) = seed_terminal(&reg, [4u8; 32], 0x44, LocalAnchorStatus::Finalized);
    let (h_failed, _) = seed_terminal(&reg, [5u8; 32], 0x55, LocalAnchorStatus::Failed {
        reason: "y".into(),
    });
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(
        &reg,
        &plan_opts(
            vec![
                LocalAnchorStatus::Finalized,
                LocalAnchorStatus::Failed { reason: String::new() },
            ],
            &sel,
            None,
        ),
    )
    .unwrap();
    let hashes: Vec<&str> = plan.actions.iter().map(|a| a.artifact_hash_hex.as_str()).collect();
    assert!(hashes.contains(&h_finalized.as_str()));
    assert!(hashes.contains(&h_failed.as_str()));
    assert_eq!(plan.actions.len(), 2);
}

#[test]
fn plan_selector_miss_uses_anchor_not_found() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [6u8; 32], 0x66, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection {
        tx_ids: vec!["anchor-ghost".to_string()],
        ..Default::default()
    };
    let err = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap_err();
    assert_reason_tag(&err, "anchor_not_found");
}

#[test]
fn plan_selector_miss_on_non_terminal_record_uses_anchor_not_found_with_operator_readable_detail() {
    let (_dir, reg, _) = fresh_registry();
    // Submitted (non-terminal) record.
    let (_h, tx_id) = seed_submitted(&reg, [7u8; 32], 0x77);
    let sel = AnchorArchiveSelection {
        tx_ids: vec![tx_id.clone()],
        ..Default::default()
    };
    let err = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap_err();
    assert_reason_tag(&err, "anchor_not_found");
    if let EvidenceAnchorError::AnchorNotFound { selector } = err {
        assert!(
            selector.contains("not terminal"),
            "operator-readable detail must name the reason"
        );
    } else {
        panic!("expected AnchorNotFound");
    }
}

#[test]
fn plan_selector_miss_after_before_filter_uses_anchor_not_found_with_operator_readable_detail() {
    let (_dir, reg, _) = fresh_registry();
    let (_h, tx_id) = seed_terminal(&reg, [8u8; 32], 0x88, LocalAnchorStatus::Finalized);
    // Set --before in the past so the just-seeded record's
    // updated_at is NEWER than --before → excluded.
    let before = (Utc::now() - Duration::days(7)).to_rfc3339();
    let sel = AnchorArchiveSelection {
        tx_ids: vec![tx_id.clone()],
        ..Default::default()
    };
    let err = plan_anchor_archive(
        &reg,
        &plan_opts(vec![], &sel, Some(&before)),
    )
    .unwrap_err();
    assert_reason_tag(&err, "anchor_not_found");
    if let EvidenceAnchorError::AnchorNotFound { selector } = err {
        assert!(
            selector.contains("--before"),
            "operator-readable detail must name --before"
        );
    } else {
        panic!("expected AnchorNotFound");
    }
}

// ── Plan integrity ───────────────────────────────────────────────────────────

#[test]
fn archive_plan_hash_is_byte_stable_across_recomputation() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [9u8; 32], 0x99, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let p1 = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let p2 = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    assert_eq!(p1.archive_plan_hash, p2.archive_plan_hash);
    assert!(!p1.archive_plan_hash.is_empty());
}

#[test]
fn archive_plan_hash_blanks_self_field_in_canonical_bytes() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [10u8; 32], 0xa0, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let mut plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let h1 = plan.archive_plan_hash.clone();
    plan.archive_plan_hash = "0".repeat(64);
    // Recomputing via apply's preflight would catch a real
    // mismatch — but for this pin we verify the hash differs
    // when fields change.
    let dir = archive_dir();
    let err = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: true,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_plan_hash_mismatch");
    assert!(!h1.is_empty());
}

#[test]
fn registry_state_hash_changes_when_selected_record_status_changes() {
    let (_dir, reg, _) = fresh_registry();
    let (h, _) = seed_terminal(&reg, [11u8; 32], 0xa1, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let p1 = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let hash_before = p1.registry_state_hash;
    reg.update_status(&h, LocalAnchorStatus::Failed { reason: "x".into() }).unwrap();
    let p2 = plan_anchor_archive(
        &reg,
        &plan_opts(
            vec![
                LocalAnchorStatus::Finalized,
                LocalAnchorStatus::Failed { reason: String::new() },
            ],
            &sel,
            None,
        ),
    )
    .unwrap();
    assert_ne!(hash_before, p2.registry_state_hash);
}

#[test]
fn registry_state_hash_changes_when_selected_record_updated_at_changes() {
    // Q3 lock pin — Stage 13.7 INCLUDES updated_at_unix in the
    // state hash (deliberate divergence from Stage 13.4 because
    // of the --before selector).
    let (_dir, reg, root) = fresh_registry();
    let (h, _) = seed_terminal(&reg, [12u8; 32], 0xa2, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let p1 = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let hash_before = p1.registry_state_hash;
    // Mutate updated_at directly on disk.
    let path = root.join(format!("{h}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let past = (Utc::now() - Duration::days(30)).to_rfc3339();
    value["updated_at"] = serde_json::Value::String(past);
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let p2 = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    assert_ne!(
        hash_before, p2.registry_state_hash,
        "registry_state_hash must change when updated_at changes (Stage 13.7 Q3 lock)"
    );
}

#[test]
fn registry_state_hash_changes_when_tx_index_entry_changes_independently() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_terminal(&reg, [13u8; 32], 0xa3, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let p1 = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let hash_before = p1.registry_state_hash;
    // Inject a phantom tx_index entry.
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    idx["by_tx_id"]["phantom"] = serde_json::Value::String("ff".repeat(32));
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();
    let p2 = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    assert_ne!(hash_before, p2.registry_state_hash);
}

// ── Apply preflight refusals ─────────────────────────────────────────────────

#[test]
fn apply_refuses_unsupported_archive_plan_schema_version() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [14u8; 32], 0xa4, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let mut plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    plan.schema_version = 2;
    let dir = archive_dir();
    let err = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "unsupported_archive_plan_schema_version");
}

#[test]
fn apply_schema_version_check_fires_before_plan_hash_check() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [15u8; 32], 0xa5, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let mut plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    plan.schema_version = 2;
    // Also scramble the hash. Schema check should win.
    plan.archive_plan_hash = "ff".repeat(32);
    let dir = archive_dir();
    let err = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "unsupported_archive_plan_schema_version");
}

#[test]
fn apply_refuses_on_archive_plan_hash_mismatch() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [16u8; 32], 0xa6, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let mut plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    plan.archive_plan_hash = "ff".repeat(32);
    let dir = archive_dir();
    let err = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_plan_hash_mismatch");
}

#[test]
fn apply_refuses_on_archive_drift() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [17u8; 32], 0xa7, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    // Seed a NEW record after plan → state hash changes → drift.
    let _ = seed_terminal(&reg, [18u8; 32], 0xa8, LocalAnchorStatus::Finalized);
    let dir = archive_dir();
    let err = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_drift");
}

// ── Apply path validation ────────────────────────────────────────────────────

#[test]
fn apply_refuses_action_with_absolute_path_using_archive_invalid_path() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [19u8; 32], 0xa9, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let mut plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    plan.actions[0].source_relative = "/etc/passwd".to_string();
    // Recompute plan_hash so we reach the path-validation step.
    plan.archive_plan_hash = recompute_plan_hash(&plan);
    let dir = archive_dir();
    let err = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: true,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_invalid_path");
}

#[test]
fn apply_refuses_action_with_parent_traversal_using_archive_invalid_path() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [20u8; 32], 0xb0, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let mut plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    plan.actions[0].source_relative = "../etc/passwd".to_string();
    plan.archive_plan_hash = recompute_plan_hash(&plan);
    let dir = archive_dir();
    let err = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: true,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_invalid_path");
}

#[test]
fn apply_refuses_action_with_wrong_per_kind_shape_using_archive_invalid_path() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [21u8; 32], 0xb1, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let mut plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    plan.actions[0].source_relative = "abc.json".to_string(); // not 64-hex
    plan.archive_plan_hash = recompute_plan_hash(&plan);
    let dir = archive_dir();
    let err = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: true,
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_invalid_path");
}

// ── Apply happy paths + durability ───────────────────────────────────────────

#[test]
fn apply_dry_run_emits_would_archive_outcomes_and_does_not_mutate_registry() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _) = seed_terminal(&reg, [22u8; 32], 0xb2, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let dir = archive_dir();
    let report = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: true,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(report.mode, "dry_run");
    assert_eq!(report.actions_would_archive, 1);
    assert_eq!(report.actions_archived, 0);
    assert_eq!(report.outcomes[0].outcome, "would_archive");
    // Hot registry untouched.
    assert!(root.join(format!("{h}.json")).is_file());
    let idx = read_tx_index(&root);
    let by_tx = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert!(!by_tx.is_empty(), "tx_index entry must remain in dry-run");
}

#[test]
fn apply_real_run_copies_to_archive_then_deletes_source_record() {
    let (_dir, reg, root) = fresh_registry();
    let (h, tx_id) = seed_terminal(&reg, [23u8; 32], 0xb3, LocalAnchorStatus::Finalized);
    let source_bytes = std::fs::read(root.join(format!("{h}.json"))).unwrap();
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let dir = archive_dir();
    let report = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(report.mode, "apply");
    assert_eq!(report.actions_archived, 1);
    // Source record gone from hot registry.
    assert!(!root.join(format!("{h}.json")).is_file());
    // Archive subtree has byte-equal record at the documented
    // path.
    let archived = dir.path().join(format!("{}/anchors/{h}.json", plan.plan_id));
    let archived_bytes = std::fs::read(&archived).unwrap();
    assert_eq!(archived_bytes, source_bytes);
    // tx_index entry removed.
    let idx = read_tx_index(&root);
    assert!(idx.get("by_tx_id").unwrap().get(&tx_id).is_none());
}

#[test]
fn apply_writes_manifest_before_deleting_any_source_record() {
    // Stage 13.7 REJECT-fix Finding 2 — Phase 1 guarantee.
    // Sabotage the manifest write so it fails AFTER quarantine
    // copies have landed but BEFORE any source deletion. The
    // hot registry must remain byte-identical to its pre-apply
    // state.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, reg, root) = fresh_registry();
        let (h, _) = seed_terminal(&reg, [24u8; 32], 0xb4, LocalAnchorStatus::Finalized);
        let source_bytes_before =
            std::fs::read(root.join(format!("{h}.json"))).unwrap();
        let sel = AnchorArchiveSelection::default();
        let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
        let dir = archive_dir();
        // Pre-create the per-plan archive dir AND set it RO so
        // the manifest write fails after Pass 1 copies land.
        let plan_subdir = dir.path().join(&plan.plan_id);
        std::fs::create_dir_all(&plan_subdir).unwrap();
        // First copy the source bytes into a writeable
        // `anchors/` subdir so Pass 1 succeeds.
        let anchors_subdir = plan_subdir.join("anchors");
        std::fs::create_dir_all(&anchors_subdir).unwrap();
        // Now make the plan-subdir RO so Pass 1.5's manifest
        // write fails when it tries to land
        // `<plan_id>/archive_manifest.json`.
        let mut perms = std::fs::metadata(&plan_subdir).unwrap().permissions();
        perms.set_mode(0o555);
        std::fs::set_permissions(&plan_subdir, perms).unwrap();
        let result = apply_anchor_archive(
            &plan,
            &AnchorArchiveApplyOptions {
                archive_dir: dir.path(),
                dry_run: false,
                now_utc: NOW,
            },
        );
        assert!(result.is_err());
        // Restore perms for cleanup.
        let mut perms = std::fs::metadata(&plan_subdir).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&plan_subdir, perms).unwrap();
        // Hot registry MUST be byte-identical to pre-apply.
        let source_bytes_after = std::fs::read(root.join(format!("{h}.json"))).unwrap();
        assert_eq!(
            source_bytes_before, source_bytes_after,
            "Phase-1 guarantee: hot registry intact on manifest-write failure"
        );
    }
}

#[test]
fn apply_removes_archived_records_tx_index_entries() {
    let (_dir, reg, root) = fresh_registry();
    let (_h, tx_id_archived) =
        seed_terminal(&reg, [25u8; 32], 0xb5, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let dir = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let idx = read_tx_index(&root);
    let by_tx = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert!(by_tx.get(&tx_id_archived).is_none());
}

#[test]
fn apply_preserves_unrelated_tx_index_entries() {
    let (_dir, reg, root) = fresh_registry();
    let (_h_a, tx_a) = seed_terminal(&reg, [26u8; 32], 0xb6, LocalAnchorStatus::Finalized);
    let (_h_b, tx_b) = seed_terminal(&reg, [27u8; 32], 0xb7, LocalAnchorStatus::Failed {
        reason: String::new(),
    });
    // Default plan (statuses = [Finalized] only) → archives A.
    // B's tx_index entry must survive.
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let dir = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let idx = read_tx_index(&root);
    let by_tx = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert!(by_tx.get(&tx_a).is_none(), "archived record's tx_id removed");
    assert!(
        by_tx.get(&tx_b).is_some(),
        "unrelated tx_index entry preserved (D3 merge-not-rebuild)"
    );
}

// ── Manifest portability (Stage 13.7 REJECT-fix Finding 4) ───────────────────

#[test]
fn archive_manifest_does_not_contain_anchor_registry_dir_field() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [28u8; 32], 0xb8, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let dir = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let manifest_bytes = std::fs::read(
        dir.path().join(format!("{}/archive_manifest.json", plan.plan_id)),
    )
    .unwrap();
    let text = std::str::from_utf8(&manifest_bytes).unwrap();
    assert!(
        !text.contains("anchor_registry_dir"),
        "manifest must not leak host-local registry path (Finding 4)"
    );
}

// ── Phase-2 recovery via restore ─────────────────────────────────────────────

#[test]
fn restore_via_manifest_recovers_archived_records_after_apply() {
    // Stage 13.7 REJECT-fix Finding 2 — restore IS the recovery
    // path for partial Phase-2 apply. This test demonstrates the
    // happy-case symmetry: archive then restore brings every
    // record back idempotently. (The cfg-unix sabotage test
    // covers the actual mid-Phase-2 partial-failure scenario.)
    let (_dir, reg, root) = fresh_registry();
    let (h_a, tx_a) = seed_terminal(&reg, [29u8; 32], 0xb9, LocalAnchorStatus::Finalized);
    let (h_b, tx_b) = seed_terminal(&reg, [30u8; 32], 0xc0, LocalAnchorStatus::Finalized);
    let source_a = std::fs::read(root.join(format!("{h_a}.json"))).unwrap();
    let source_b = std::fs::read(root.join(format!("{h_b}.json"))).unwrap();

    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert!(!root.join(format!("{h_a}.json")).exists());
    assert!(!root.join(format!("{h_b}.json")).exists());

    // Restore from manifest.
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let _ = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();

    let restored_a = std::fs::read(root.join(format!("{h_a}.json"))).unwrap();
    let restored_b = std::fs::read(root.join(format!("{h_b}.json"))).unwrap();
    assert_eq!(restored_a, source_a);
    assert_eq!(restored_b, source_b);

    // tx_index entries restored.
    let idx = read_tx_index(&root);
    let by_tx = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert!(by_tx.get(&tx_a).is_some());
    assert!(by_tx.get(&tx_b).is_some());
}

// ── Restore happy paths ──────────────────────────────────────────────────────

#[test]
fn restore_round_trips_archived_record_byte_for_byte() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _) = seed_terminal(&reg, [31u8; 32], 0xc1, LocalAnchorStatus::Finalized);
    let source_bytes = std::fs::read(root.join(format!("{h}.json"))).unwrap();
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let _ = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    let restored_bytes = std::fs::read(root.join(format!("{h}.json"))).unwrap();
    assert_eq!(
        source_bytes, restored_bytes,
        "byte-preserve: restored bytes must equal source bytes"
    );
}

#[test]
fn restore_re_adds_tx_index_entry_for_archived_record() {
    let (_dir, reg, root) = fresh_registry();
    let (_h, tx_id) = seed_terminal(&reg, [32u8; 32], 0xc2, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let _ = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    let idx = read_tx_index(&root);
    let by_tx = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert!(by_tx.get(&tx_id).is_some());
}

#[test]
fn restore_preserves_unrelated_tx_index_entries() {
    let (_dir, reg, root) = fresh_registry();
    let (_h, _tx) = seed_terminal(&reg, [33u8; 32], 0xc3, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Inject an unrelated tx_index entry between apply and
    // restore.
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    idx["by_tx_id"]["unrelated"] = serde_json::Value::String("99".repeat(32));
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();

    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let _ = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    let idx = read_tx_index(&root);
    let by_tx = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert_eq!(
        by_tx.get("unrelated").and_then(|v| v.as_str()),
        Some("99".repeat(32).as_str()),
        "unrelated tx_index entry preserved on restore (D3 merge-not-rebuild)"
    );
}

// ── Restore conflict matrix ──────────────────────────────────────────────────

#[test]
fn restore_idempotent_skipped_already_restored_for_byte_equal_record() {
    let (_dir, reg, root) = fresh_registry();
    let (_h, _) = seed_terminal(&reg, [34u8; 32], 0xc4, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let _ = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    // Second restore — row 2.
    let report = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(report.skipped_already_restored, 1);
    assert_eq!(report.outcomes[0].outcome, "skipped_already_restored");
}

#[test]
fn restore_re_adds_missing_tx_index_entry_when_record_file_already_present_byte_equal_without_rewriting_file()
 {
    let (_dir, reg, root) = fresh_registry();
    let (h, tx_id) = seed_terminal(&reg, [35u8; 32], 0xc5, LocalAnchorStatus::Finalized);
    let source_bytes = std::fs::read(root.join(format!("{h}.json"))).unwrap();
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Manually replace the record file in the hot registry
    // (simulating a Phase-2 partial-failure recovery scenario
    // where Pass 2 deleted A but not the tx_index rewrite).
    std::fs::write(root.join(format!("{h}.json")), &source_bytes).unwrap();
    let mtime_before = std::fs::metadata(root.join(format!("{h}.json")))
        .unwrap()
        .modified()
        .unwrap();

    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let report = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(report.re_added_tx_index_entry, 1);
    assert_eq!(report.outcomes[0].outcome, "re_added_tx_index_entry");

    // The byte-equal record file MUST not be rewritten.
    let mtime_after = std::fs::metadata(root.join(format!("{h}.json")))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(
        mtime_before, mtime_after,
        "row 3 must not rewrite the byte-equal record file"
    );
    // tx_index entry is now present.
    let idx = read_tx_index(&root);
    assert!(idx.get("by_tx_id").unwrap().get(&tx_id).is_some());
}

#[test]
fn restore_refuses_when_target_record_has_different_bytes_using_archive_target_exists_artifact_hash_field()
 {
    let (_dir, reg, root) = fresh_registry();
    let (h, _) = seed_terminal(&reg, [36u8; 32], 0xc6, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Plant a byte-different file at the target slot.
    std::fs::write(root.join(format!("{h}.json")), b"divergent").unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let err = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_target_exists");
    if let EvidenceAnchorError::ArchiveTargetExists { field, .. } = err {
        assert_eq!(field, "artifact_hash");
    } else {
        panic!("expected ArchiveTargetExists field=artifact_hash");
    }
}

#[test]
fn restore_refuses_when_tx_index_maps_same_tx_id_to_different_hash_using_archive_target_exists_tx_id_field()
 {
    let (_dir, reg, root) = fresh_registry();
    let (_h, tx_id) = seed_terminal(&reg, [37u8; 32], 0xc7, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Seed target tx_index with SAME tx_id → DIFFERENT hash.
    let phantom_hash = "00".repeat(32);
    std::fs::write(
        root.join("tx_index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "by_tx_id": { tx_id.clone(): phantom_hash }
        }))
        .unwrap(),
    )
    .unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let err = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_target_exists");
    if let EvidenceAnchorError::ArchiveTargetExists { field, .. } = err {
        assert_eq!(field, "tx_id");
    } else {
        panic!("expected ArchiveTargetExists field=tx_id");
    }
}

#[test]
fn restore_refuses_on_archive_blake3_mismatch() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _) = seed_terminal(&reg, [38u8; 32], 0xc8, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Tamper the archived file's bytes.
    let archived_path = arch.path().join(format!("{}/anchors/{h}.json", plan.plan_id));
    std::fs::write(&archived_path, b"hand-edited archive bytes").unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let err = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_blake3_mismatch");
}

// ── Restore path validation ──────────────────────────────────────────────────

#[test]
fn restore_refuses_manifest_entry_with_absolute_path_using_archive_invalid_path() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_terminal(&reg, [39u8; 32], 0xc9, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let mut manifest = read_manifest(&manifest_dir);
    manifest.entries[0].archive_relative = "/etc/passwd".to_string();
    std::fs::write(
        manifest_dir.join("archive_manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let err = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_invalid_path");
}

#[test]
fn restore_refuses_manifest_entry_with_parent_traversal_using_archive_invalid_path() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed_terminal(&reg, [40u8; 32], 0xca, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let mut manifest = read_manifest(&manifest_dir);
    manifest.entries[0].archive_relative = "anchors/../etc/passwd".to_string();
    let err = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_invalid_path");
}

// ── Preflight-all-before-mutate (carry-forward from Stage 13.6) ──────────────

#[test]
fn restore_does_not_modify_target_when_any_entry_refuses_with_target_exists() {
    let (_dir, reg, root) = fresh_registry();
    let (h_a, _) = seed_terminal(&reg, [41u8; 32], 0xcb, LocalAnchorStatus::Finalized);
    let (h_b, _) = seed_terminal(&reg, [42u8; 32], 0xcc, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Entry B is byte-different in the hot registry → row 4
    // collision.
    std::fs::write(root.join(format!("{h_b}.json")), b"squatter").unwrap();
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let err = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "archive_target_exists");
    // A must NOT have been restored.
    assert!(!root.join(format!("{h_a}.json")).exists());
}

// ── Restore dry-run (Stage 13.7 REJECT-fix Finding 1) ────────────────────────

#[test]
fn restore_dry_run_emits_would_restore_outcomes_and_does_not_mutate_target_registry() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _) = seed_terminal(&reg, [43u8; 32], 0xcd, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Hot registry has had the record removed.
    assert!(!root.join(format!("{h}.json")).exists());
    let manifest_dir = arch.path().join(&plan.plan_id);
    let manifest = read_manifest(&manifest_dir);
    let report = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: true,
        },
    )
    .unwrap();
    assert_eq!(report.mode, "dry_run");
    assert_eq!(report.would_restore, 1);
    assert_eq!(report.restored, 0);
    assert_eq!(report.outcomes[0].outcome, "would_restore");
    // No FS mutation on the hot registry.
    assert!(!root.join(format!("{h}.json")).exists());
    // tx_index.json was removed by apply too; dry-run mustn't
    // re-create or rewrite it.
    let idx_path = root.join("tx_index.json");
    if idx_path.exists() {
        // tx_index might still exist as an empty-ish file from
        // apply's merge; what matters is that dry-run didn't
        // ADD the archived tx_id back.
        let idx: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&idx_path).unwrap()).unwrap();
        if let Some(by_tx) = idx.get("by_tx_id").and_then(|v| v.as_object()) {
            assert!(
                by_tx.is_empty() || by_tx.keys().all(|k| k != &manifest.entries[0].tx_id),
                "restore dry-run must not add tx_index entries"
            );
        }
    }
}

// ── Defense-in-depth (Stage 13.0 verifier on restore) ────────────────────────

#[test]
fn restore_refuses_when_archived_record_has_tampered_signature_using_submitter_signature_invalid() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _) = seed_terminal(&reg, [44u8; 32], 0xce, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let arch = archive_dir();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Hand-edit the archived record's signature.
    let archived_path = arch.path().join(format!("{}/anchors/{h}.json", plan.plan_id));
    let bytes = std::fs::read(&archived_path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let sig = value["tx_data"]["submitter_signature"].as_array_mut().unwrap();
    sig[0] = serde_json::Value::Number(0u8.into());
    let new_bytes = serde_json::to_vec_pretty(&value).unwrap();
    std::fs::write(&archived_path, &new_bytes).unwrap();
    // Update the manifest's blake3 + bytes so we get past the
    // archive_blake3_mismatch check and reach the signature
    // verifier.
    let manifest_dir = arch.path().join(&plan.plan_id);
    let mut manifest = read_manifest(&manifest_dir);
    manifest.entries[0].blake3_hex = blake3::hash(&new_bytes).to_hex().to_string();
    manifest.entries[0].bytes = new_bytes.len() as u64;
    std::fs::write(
        manifest_dir.join("archive_manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let err = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "submitter_signature_invalid");
}

// ── Plan-to-disk round-trip ──────────────────────────────────────────────────

#[test]
fn plan_round_trips_through_disk_and_applies_cleanly() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_terminal(&reg, [45u8; 32], 0xcf, LocalAnchorStatus::Finalized);
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let plan_path = tmp.path().join("plan.json");
    std::fs::write(&plan_path, serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
    let reloaded = read_plan_from_disk(&plan_path);
    assert_eq!(reloaded.archive_plan_hash, plan.archive_plan_hash);
    let arch = archive_dir();
    let report = apply_anchor_archive(
        &reloaded,
        &AnchorArchiveApplyOptions {
            archive_dir: arch.path(),
            dry_run: true,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(report.actions_would_archive, 1);
}

// ── REJECT-fix Finding 2: zero-action apply requires no archive_dir ──

#[test]
fn apply_zero_action_plan_succeeds_in_dry_run_with_empty_archive_dir() {
    // Empty registry → zero-action plan. The library must accept
    // an empty archive_dir path without touching it.
    let (_dir, reg, _) = fresh_registry();
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    assert_eq!(plan.actions.len(), 0);
    let report = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: std::path::Path::new(""),
            dry_run: true,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(report.actions_archived, 0);
    assert_eq!(report.actions_would_archive, 0);
    assert_eq!(report.mode, "dry_run");
}

#[test]
fn apply_zero_action_plan_succeeds_in_real_run_with_empty_archive_dir() {
    // Real-run version: no FS effects because Pass 1 / 1.5 / 2
    // all loop zero times.
    let (_dir, reg, _) = fresh_registry();
    let sel = AnchorArchiveSelection::default();
    let plan = plan_anchor_archive(&reg, &plan_opts(vec![], &sel, None)).unwrap();
    assert_eq!(plan.actions.len(), 0);
    let report = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: std::path::Path::new(""),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(report.actions_archived, 0);
    assert_eq!(report.actions_would_archive, 0);
    assert_eq!(report.mode, "apply");
    // No manifest landed (entries vector was empty).
    assert!(report.archive_manifest_relative.is_none());
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn recompute_plan_hash(plan: &AnchorArchivePlan) -> String {
    let blanked = AnchorArchivePlan {
        archive_plan_hash: String::new(),
        ..plan.clone()
    };
    let bytes = serde_json::to_vec(&blanked).unwrap();
    blake3::hash(&bytes).to_hex().to_string()
}
