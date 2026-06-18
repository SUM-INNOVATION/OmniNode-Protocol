//! Phase 5 Stage 13.8 — integration tests for the local
//! consistency report. Hermetic. Temp dirs. No network.

use chrono::{Duration, Utc};
use omni_zkml::{
    anchor_signer_pubkey_bytes, apply_anchor_archive, apply_anchor_export,
    build_anchor_consistency_report, build_anchor_digest, plan_anchor_archive,
    submit_evidence_anchor_workflow, AnchorArchiveApplyOptions,
    AnchorArchivePlanOptions, AnchorArchiveSelection,
    AnchorConsistencyFindingKind, AnchorConsistencyOptions,
    AnchorConsistencySeverity, AnchorExportOptions, AnchorExportSelection,
    EvidenceAnchorError, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
    StubEvidenceAnchorChainClient, VerifiedWrapperMetadata,
    ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION,
};
use std::path::PathBuf;

// ── Helpers ──────────────────────────────────────────────────────────────────

const NOW: &str = "2026-06-18T00:00:00Z";

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

fn seed(
    reg: &LocalEvidenceAnchorRegistry,
    seed: [u8; 32],
    marker: u8,
    status: Option<LocalAnchorStatus>,
) -> (String, String, Vec<u8>) {
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
    if let Some(s) = status {
        reg.update_status(&hash, s).unwrap();
    }
    (hash, tx_id, raw)
}

fn build_report(root: &std::path::Path) -> omni_zkml::AnchorConsistencyReport {
    build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: root,
        archive_dirs: &[],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap()
}

fn find_kind(
    report: &omni_zkml::AnchorConsistencyReport,
    kind: AnchorConsistencyFindingKind,
) -> Vec<&omni_zkml::AnchorConsistencyFinding> {
    report.findings.iter().filter(|f| f.kind == kind).collect()
}

// ── Hot registry — basic ──────────────────────────────────────────────────────

#[test]
fn empty_hot_registry_produces_zero_counts_and_no_error_findings() {
    let (_dir, _reg, root) = fresh_registry();
    let report = build_report(&root);
    assert_eq!(report.summary.hot_total, 0);
    assert_eq!(report.summary.findings_by_severity_error, 0);
    assert!(report.findings.is_empty());
}

#[test]
fn valid_hot_registry_summary_counts_by_status() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed(&reg, [1u8; 32], 0x11, None); // submitted
    let _ = seed(&reg, [2u8; 32], 0x22, Some(LocalAnchorStatus::Included));
    let _ = seed(&reg, [3u8; 32], 0x33, Some(LocalAnchorStatus::Finalized));
    let _ = seed(
        &reg,
        [4u8; 32],
        0x44,
        Some(LocalAnchorStatus::Failed {
            reason: "x".into(),
        }),
    );
    let report = build_report(&root);
    assert_eq!(report.summary.hot_total, 4);
    assert_eq!(report.summary.hot_submitted, 1);
    assert_eq!(report.summary.hot_included, 1);
    assert_eq!(report.summary.hot_finalized, 1);
    assert_eq!(report.summary.hot_failed, 1);
}

#[test]
fn malformed_hot_record_yields_hot_record_malformed_error_severity() {
    let (_dir, _reg, root) = fresh_registry();
    let fake_hash = "aa".repeat(32);
    std::fs::write(root.join(format!("{fake_hash}.json")), b"NOT JSON").unwrap();
    let report = build_report(&root);
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotRecordMalformed);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn tampered_hot_record_signature_yields_hot_record_signature_invalid_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _, _) = seed(&reg, [5u8; 32], 0x55, None);
    let path = root.join(format!("{h}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let sig = value["tx_data"]["submitter_signature"].as_array_mut().unwrap();
    sig[0] = serde_json::Value::Number(0u8.into());
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let report = build_report(&root);
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotRecordSignatureInvalid);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn hot_record_with_unsupported_schema_version_yields_hot_record_schema_unsupported_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _, _) = seed(&reg, [6u8; 32], 0x66, None);
    let path = root.join(format!("{h}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["tx_data"]["digest"]["anchor_schema_version"] =
        serde_json::Value::Number(99u32.into());
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let report = build_report(&root);
    let f = find_kind(
        &report,
        AnchorConsistencyFindingKind::HotRecordSchemaUnsupported,
    );
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn hot_filename_hash_mismatch_yields_hot_filename_hash_mismatch_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _, _) = seed(&reg, [7u8; 32], 0x77, None);
    // Move record to a wrong filename.
    let src = root.join(format!("{h}.json"));
    let dst = root.join(format!("{}.json", "00".repeat(32)));
    std::fs::rename(&src, &dst).unwrap();
    let report = build_report(&root);
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotFilenameHashMismatch);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn tx_index_orphan_yields_hot_tx_index_orphan_warning_severity() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed(&reg, [8u8; 32], 0x88, None);
    // Inject orphan.
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    idx["by_tx_id"]["phantom"] = serde_json::Value::String("ff".repeat(32));
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();
    let report = build_report(&root);
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotTxIndexOrphan);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Warning);
}

#[test]
fn tx_index_mapping_mismatch_yields_hot_tx_index_mismatch_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let (h_a, tx_a, _) = seed(&reg, [9u8; 32], 0x99, None);
    let (h_b, _, _) = seed(&reg, [10u8; 32], 0xa0, None);
    // Rewrite tx_index so tx_a maps to h_b instead of h_a.
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    idx["by_tx_id"][&tx_a] = serde_json::Value::String(h_b.clone());
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();
    let report = build_report(&root);
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotTxIndexMismatch);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
    assert_eq!(f[0].tx_id.as_deref(), Some(tx_a.as_str()));
    // The finding's artifact_hash_hex carries the tx_index's
    // CLAIMED hash (which we rewrote to h_b), not the record's
    // actual hash (h_a). Both surfaces are reported via the
    // detail string.
    assert_eq!(f[0].artifact_hash_hex.as_deref(), Some(h_b.as_str()));
    let _ = h_a;
}

#[test]
fn tx_index_entry_to_existing_record_with_different_inner_tx_id_yields_hot_tx_index_mismatch() {
    // REJECT-fix regression — forward-direction
    // HotTxIndexMismatch. tx_index maps `phantom-tx-a → h_b`,
    // but h_b.json's `receipt.tx_id` is `tx_b` (NOT phantom-tx-a).
    // Lookup by `phantom-tx-a` via tx_index would route to a
    // record that doesn't claim it. Previously missed because
    // the reverse-direction check (record-claiming-tx-a-under-
    // different-hash) returns None when no record claims
    // phantom-tx-a.
    let (_dir, reg, root) = fresh_registry();
    let (h_b, tx_b, _) = seed(&reg, [80u8; 32], 0xb0, None);
    let idx_path = root.join("tx_index.json");
    let bytes = std::fs::read(&idx_path).unwrap();
    let mut idx: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Remove the auto-generated tx_b → h_b entry so only the
    // phantom mapping remains.
    idx["by_tx_id"].as_object_mut().unwrap().remove(&tx_b);
    idx["by_tx_id"]["phantom-tx-a"] = serde_json::Value::String(h_b.clone());
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();
    let report = build_report(&root);
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotTxIndexMismatch);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
    assert_eq!(f[0].tx_id.as_deref(), Some("phantom-tx-a"));
    assert_eq!(f[0].artifact_hash_hex.as_deref(), Some(h_b.as_str()));
    assert!(
        f[0].detail.contains(&tx_b),
        "detail must name the record's actual receipt.tx_id ({tx_b}); got: {}",
        f[0].detail
    );
}

#[test]
fn tx_id_duplicate_across_records_yields_hot_tx_id_duplicate_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let (h_a, tx_a, _) = seed(&reg, [11u8; 32], 0xb1, None);
    let (h_b, _, _) = seed(&reg, [12u8; 32], 0xb2, None);
    // Hand-edit record B so its receipt.tx_id equals tx_a.
    let path_b = root.join(format!("{h_b}.json"));
    let bytes = std::fs::read(&path_b).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["receipt"]["tx_id"] = serde_json::Value::String(tx_a.clone());
    std::fs::write(&path_b, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let report = build_report(&root);
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotTxIdDuplicate);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
    let _ = h_a;
}

#[test]
fn tmp_orphan_yields_hot_tmp_orphan_warning_severity() {
    let (_dir, _reg, root) = fresh_registry();
    std::fs::write(root.join("orphan.json.tmp"), b"stale").unwrap();
    let report = build_report(&root);
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotTmpOrphan);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Warning);
}

#[test]
fn stale_open_record_emitted_when_threshold_supplied_warning_severity() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _, _) = seed(&reg, [13u8; 32], 0xb3, None); // submitted
    // Backdate submitted_at by 30 days.
    let path = root.join(format!("{h}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let past = (Utc::now() - Duration::days(30)).to_rfc3339();
    value["submitted_at"] = serde_json::Value::String(past);
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let now_utc = Utc::now().to_rfc3339();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[],
        stale_threshold_secs: Some(60 * 60 * 24), // 1 day
        now_utc: &now_utc,
    })
    .unwrap();
    let f = find_kind(&report, AnchorConsistencyFindingKind::HotStaleOpenRecord);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Warning);
}

// ── Archive ──────────────────────────────────────────────────────────────────

fn make_archive(
    reg: &LocalEvidenceAnchorRegistry,
    status: LocalAnchorStatus,
    archive_dir: &std::path::Path,
) -> String {
    // Transition a record to terminal status and archive it.
    let (_h, _, _) = seed(reg, [120u8; 32], 0xc0, Some(status));
    let plan = plan_anchor_archive(
        reg,
        &AnchorArchivePlanOptions {
            now_utc: NOW,
            statuses: vec![
                LocalAnchorStatus::Finalized,
                LocalAnchorStatus::Failed {
                    reason: String::new(),
                },
            ],
            before_utc: None,
            selection: &AnchorArchiveSelection::default(),
        },
    )
    .unwrap();
    apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir,
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    plan.plan_id
}

#[test]
fn archive_concrete_dir_scan_finds_manifest_directly() {
    let (_dir, reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let plan_id = make_archive(&reg, LocalAnchorStatus::Finalized, arch_dir.path());
    let concrete = arch_dir.path().join(&plan_id);
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[concrete],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    assert_eq!(report.summary.archive_manifests_scanned, 1);
    assert_eq!(report.summary.archive_entries_scanned, 1);
}

#[test]
fn archive_root_scan_finds_child_archive_manifests() {
    let (_dir, reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let _plan_id = make_archive(&reg, LocalAnchorStatus::Finalized, arch_dir.path());
    // Pass the ROOT, not the concrete plan_id dir.
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[arch_dir.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    assert_eq!(report.summary.archive_manifests_scanned, 1);
}

#[test]
fn archive_dir_with_neither_manifest_nor_child_yields_archive_dir_no_manifest_warning_severity() {
    let (_dir, _reg, root) = fresh_registry();
    let empty = tempfile::tempdir().unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[empty.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(&report, AnchorConsistencyFindingKind::ArchiveDirNoManifest);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Warning);
}

#[test]
fn archive_dir_unreadable_yields_warning_finding_not_command_failure() {
    let (_dir, _reg, root) = fresh_registry();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[PathBuf::from("/no/such/archive/dir")],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let no_manifest = find_kind(&report, AnchorConsistencyFindingKind::ArchiveDirNoManifest);
    let unreadable = find_kind(&report, AnchorConsistencyFindingKind::ArchiveDirUnreadable);
    // Either kind is acceptable for a non-existent path; both must
    // be `warning` severity. We accept either one fires.
    assert_eq!(no_manifest.len() + unreadable.len(), 1);
    let f = if !no_manifest.is_empty() {
        no_manifest[0]
    } else {
        unreadable[0]
    };
    assert_eq!(f.severity, AnchorConsistencySeverity::Warning);
}

#[test]
fn archive_manifest_malformed_yields_archive_manifest_malformed_error_severity() {
    let (_dir, _reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let plan_subdir = arch_dir.path().join("abcd0000abcd0000");
    std::fs::create_dir_all(&plan_subdir).unwrap();
    std::fs::write(plan_subdir.join("archive_manifest.json"), b"BAD JSON").unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[arch_dir.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(&report, AnchorConsistencyFindingKind::ArchiveManifestMalformed);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn archive_manifest_schema_unsupported_yields_archive_manifest_schema_unsupported_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let plan_id = make_archive(&reg, LocalAnchorStatus::Finalized, arch_dir.path());
    let manifest_path = arch_dir.path().join(format!("{plan_id}/archive_manifest.json"));
    let bytes = std::fs::read(&manifest_path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["schema_version"] = serde_json::Value::Number(99u32.into());
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[arch_dir.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(
        &report,
        AnchorConsistencyFindingKind::ArchiveManifestSchemaUnsupported,
    );
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn archive_entry_blake3_mismatch_yields_archive_entry_blake3_mismatch_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let plan_id = make_archive(&reg, LocalAnchorStatus::Finalized, arch_dir.path());
    // Find the archived file and tamper.
    let anchors_dir = arch_dir.path().join(format!("{plan_id}/anchors"));
    let entry = std::fs::read_dir(&anchors_dir).unwrap().next().unwrap().unwrap();
    std::fs::write(entry.path(), b"tampered archived bytes").unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[arch_dir.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(&report, AnchorConsistencyFindingKind::ArchiveEntryBlake3Mismatch);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn archive_entry_missing_file_yields_archive_entry_missing_file_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let plan_id = make_archive(&reg, LocalAnchorStatus::Finalized, arch_dir.path());
    let anchors_dir = arch_dir.path().join(format!("{plan_id}/anchors"));
    let entry = std::fs::read_dir(&anchors_dir).unwrap().next().unwrap().unwrap();
    std::fs::remove_file(entry.path()).unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[arch_dir.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(&report, AnchorConsistencyFindingKind::ArchiveEntryMissingFile);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn archive_hot_same_bytes_collision_yields_warning_severity() {
    let (_dir, reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let plan_id = make_archive(&reg, LocalAnchorStatus::Finalized, arch_dir.path());
    // Restore the same record back into hot by copying the
    // archive bytes directly (simulating a partial-Phase-2 state).
    let anchors_dir = arch_dir.path().join(format!("{plan_id}/anchors"));
    let entry = std::fs::read_dir(&anchors_dir).unwrap().next().unwrap().unwrap();
    let stem = entry
        .path()
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let bytes = std::fs::read(entry.path()).unwrap();
    std::fs::write(root.join(format!("{stem}.json")), &bytes).unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[arch_dir.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(
        &report,
        AnchorConsistencyFindingKind::ArchiveHotCollisionSameBytes,
    );
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Warning);
}

#[test]
fn archive_hot_different_bytes_collision_yields_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let plan_id = make_archive(&reg, LocalAnchorStatus::Finalized, arch_dir.path());
    // Plant a DIFFERENT-bytes record under the same hash in hot.
    let anchors_dir = arch_dir.path().join(format!("{plan_id}/anchors"));
    let entry = std::fs::read_dir(&anchors_dir).unwrap().next().unwrap().unwrap();
    let stem = entry
        .path()
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    std::fs::write(root.join(format!("{stem}.json")), b"divergent bytes").unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[arch_dir.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(
        &report,
        AnchorConsistencyFindingKind::ArchiveHotCollisionDifferentBytes,
    );
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

#[test]
fn archive_tx_id_collision_yields_error_severity() {
    let (_dir, reg, root) = fresh_registry();
    let arch_dir = tempfile::tempdir().unwrap();
    let plan_id = make_archive(&reg, LocalAnchorStatus::Finalized, arch_dir.path());
    // Find the manifest entry's tx_id.
    let manifest_path = arch_dir.path().join(format!("{plan_id}/archive_manifest.json"));
    let bytes = std::fs::read(&manifest_path).unwrap();
    let manifest: omni_zkml::AnchorArchiveManifest =
        serde_json::from_slice(&bytes).unwrap();
    let tx_id_arch = &manifest.entries[0].tx_id;
    // Plant a hot tx_index entry that maps tx_id → a DIFFERENT hash.
    std::fs::write(
        root.join("tx_index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "by_tx_id": { tx_id_arch.clone(): "00".repeat(32) }
        }))
        .unwrap(),
    )
    .unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[arch_dir.path().to_path_buf()],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(&report, AnchorConsistencyFindingKind::ArchiveTxIdCollision);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
}

// ── Export ───────────────────────────────────────────────────────────────────

fn make_export(reg: &LocalEvidenceAnchorRegistry) -> tempfile::TempDir {
    let out = tempfile::tempdir().unwrap();
    let _ = apply_anchor_export(
        reg,
        &AnchorExportOptions {
            export_out: out.path(),
            selection: &AnchorExportSelection {
                statuses: vec![LocalAnchorStatus::Submitted],
                ..Default::default()
            },
            artifact_bytes_inclusions: &[],
            signed_chain_report_paths: &[],
            label: "",
            notes: "",
            now_utc: NOW,
        },
    )
    .unwrap();
    out
}

#[test]
fn export_verify_failed_yields_export_verify_failed_with_stage_13_5_reason_tag_in_detail() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed(&reg, [50u8; 32], 0xd0, None);
    let out = make_export(&reg);
    // Tamper the export manifest to break blake3-of-manifest.
    let manifest_path = out
        .path()
        .join(omni_zkml::EXPORT_MANIFEST_FILENAME);
    let bytes = std::fs::read(&manifest_path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["notes"] = serde_json::Value::String("tampered".into());
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[out.path().to_path_buf()],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(&report, AnchorConsistencyFindingKind::ExportVerifyFailed);
    assert_eq!(f.len(), 1);
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Error);
    assert!(
        f[0].detail.contains("export_manifest_hash_mismatch"),
        "detail must carry the Stage 13.5 reason tag"
    );
}

#[test]
fn valid_export_contributes_to_export_summary_counts() {
    let (_dir, reg, root) = fresh_registry();
    let (_h, _, _) = seed(&reg, [51u8; 32], 0xd1, None);
    let out = make_export(&reg);
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[out.path().to_path_buf()],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    assert_eq!(report.summary.export_manifests_scanned, 1);
    assert!(report.summary.export_entries_scanned >= 1);
}

#[test]
fn export_dir_unreadable_yields_warning_finding_not_command_failure() {
    let (_dir, _reg, root) = fresh_registry();
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[PathBuf::from("/no/such/export/dir")],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let f = find_kind(&report, AnchorConsistencyFindingKind::ExportDirUnreadable);
    assert_eq!(f.len(), 1);
    // V2 Finding 1 lock — was `error` in v1, now `warning`.
    assert_eq!(f[0].severity, AnchorConsistencySeverity::Warning);
}

// ── Cross-surface overlaps (summary-only — V2 Finding 4) ─────────────────────

#[test]
fn hot_export_byte_equal_overlap_appears_in_summary_hot_export_overlaps_counter() {
    let (_dir, reg, root) = fresh_registry();
    let (h, _, _) = seed(&reg, [60u8; 32], 0xe0, None);
    let out = make_export(&reg);
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[out.path().to_path_buf()],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    assert_eq!(report.summary.hot_export_overlaps, 1);
    let _ = h;
}

#[test]
fn cross_surface_duplicates_emit_no_per_item_findings_in_v1_by_default() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed(&reg, [61u8; 32], 0xe1, None);
    let out = make_export(&reg);
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[out.path().to_path_buf()],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    // No info-level cross-surface findings; the overlap lives in
    // the summary counter only.
    assert_eq!(report.summary.findings_by_severity_info, 0);
}

// ── Schema + serde pins ──────────────────────────────────────────────────────

#[test]
fn report_schema_version_is_one() {
    let (_dir, _reg, root) = fresh_registry();
    let report = build_report(&root);
    assert_eq!(report.schema_version, ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION);
    assert_eq!(report.schema_version, 1);
}

#[test]
fn every_finding_kind_round_trips_through_serde() {
    use AnchorConsistencyFindingKind::*;
    let kinds = [
        HotRecordMalformed,
        HotRecordSignatureInvalid,
        HotRecordSchemaUnsupported,
        HotFilenameHashMismatch,
        HotTxIndexOrphan,
        HotTxIndexMismatch,
        HotTxIdDuplicate,
        HotStaleOpenRecord,
        HotTmpOrphan,
        ArchiveDirUnreadable,
        ArchiveDirNoManifest,
        ArchiveManifestMalformed,
        ArchiveManifestSchemaUnsupported,
        ArchiveEntryInvalidPath,
        ArchiveEntryMissingFile,
        ArchiveEntryBlake3Mismatch,
        ArchiveEntryRecordMalformed,
        ArchiveEntrySignatureInvalid,
        ArchiveEntryMetadataMismatch,
        ArchiveHotCollisionSameBytes,
        ArchiveHotCollisionDifferentBytes,
        ArchiveTxIdCollision,
        ExportDirUnreadable,
        ExportVerifyFailed,
    ];
    assert_eq!(kinds.len(), 24);
    for kind in kinds {
        let s = serde_json::to_string(&kind).unwrap();
        let round: AnchorConsistencyFindingKind = serde_json::from_str(&s).unwrap();
        assert_eq!(kind, round);
    }
}

#[test]
fn every_severity_round_trips_through_serde() {
    for sev in [
        AnchorConsistencySeverity::Info,
        AnchorConsistencySeverity::Warning,
        AnchorConsistencySeverity::Error,
    ] {
        let s = serde_json::to_string(&sev).unwrap();
        let round: AnchorConsistencySeverity = serde_json::from_str(&s).unwrap();
        assert_eq!(sev, round);
    }
}

#[test]
fn findings_by_severity_counts_match_emitted_findings() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed(&reg, [70u8; 32], 0xf0, None);
    std::fs::write(root.join("orphan.json.tmp"), b"stale").unwrap();
    let report = build_report(&root);
    let info_count = report
        .findings
        .iter()
        .filter(|f| f.severity == AnchorConsistencySeverity::Info)
        .count() as u64;
    let warn_count = report
        .findings
        .iter()
        .filter(|f| f.severity == AnchorConsistencySeverity::Warning)
        .count() as u64;
    let err_count = report
        .findings
        .iter()
        .filter(|f| f.severity == AnchorConsistencySeverity::Error)
        .count() as u64;
    assert_eq!(report.summary.findings_by_severity_info, info_count);
    assert_eq!(report.summary.findings_by_severity_warning, warn_count);
    assert_eq!(report.summary.findings_by_severity_error, err_count);
}

// ── Boundary: required registry unreadable returns io error ──────────────────

#[test]
fn unreadable_required_anchor_registry_dir_returns_io_error() {
    let bad = std::path::PathBuf::from("/no/such/anchor/registry");
    let err = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &bad,
        archive_dirs: &[],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap_err();
    let tag = omni_zkml::evidence_anchor_reason_tag(&err);
    assert_eq!(tag, "io");
    match err {
        EvidenceAnchorError::Io { .. } => {}
        other => panic!("expected Io, got {other:?}"),
    }
}
