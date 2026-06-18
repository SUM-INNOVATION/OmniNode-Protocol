//! Phase 5 Stage 13.6 — integration tests for local-only
//! export-import / registry restore.
//!
//! Hermetic. Temp dirs everywhere. No network. The seeding
//! pattern mirrors Stage 13.5's export tests: anchors are
//! inserted via the public Stage 13.0 workflow API, then a
//! Stage 13.5 export is built and used as the import source.

use omni_zkml::{
    anchor_signer_pubkey_bytes, apply_anchor_export, apply_anchor_export_import,
    build_anchor_digest, plan_anchor_export_import, submit_evidence_anchor_workflow,
    AnchorExportManifest, AnchorExportOptions, AnchorExportSelection,
    AnchorImportOptions, AnchorImportSelection, ArtifactBytesInclusion,
    EvidenceAnchorError, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
    StubEvidenceAnchorChainClient, VerifiedWrapperMetadata,
    EXPORT_MANIFEST_FILENAME,
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

fn seed_submitted(
    reg: &LocalEvidenceAnchorRegistry,
    seed: [u8; 32],
    marker: u8,
) -> (String, Vec<u8>) {
    let client = StubEvidenceAnchorChainClient::new();
    let raw = vec![marker; 32];
    let metadata = VerifiedWrapperMetadata {
        artifact_schema_version: 1,
        signer_pubkey: anchor_signer_pubkey_bytes(&seed).unwrap(),
        signed_at_utc_unix: 1_750_000_000,
    };
    let digest = build_anchor_digest(&metadata, &raw);
    let record = submit_evidence_anchor_workflow(reg, &client, digest, &seed).unwrap();
    (record.artifact_hash_hex, raw)
}

fn build_export(
    source_reg: &LocalEvidenceAnchorRegistry,
    selection: AnchorExportSelection,
    inclusions: &[ArtifactBytesInclusion],
) -> tempfile::TempDir {
    let out = tempfile::tempdir().unwrap();
    apply_anchor_export(
        source_reg,
        &AnchorExportOptions {
            export_out: out.path(),
            selection: &selection,
            artifact_bytes_inclusions: inclusions,
            signed_chain_report_paths: &[],
            label: "",
            notes: "",
            now_utc: NOW,
        },
    )
    .unwrap();
    out
}

fn read_manifest(export_dir: &std::path::Path) -> AnchorExportManifest {
    let bytes = std::fs::read(export_dir.join(EXPORT_MANIFEST_FILENAME)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn write_manifest(export_dir: &std::path::Path, manifest: &AnchorExportManifest) {
    let bytes = serde_json::to_vec_pretty(manifest).unwrap();
    std::fs::write(export_dir.join(EXPORT_MANIFEST_FILENAME), bytes).unwrap();
}

fn recompute_manifest_hash(manifest: &AnchorExportManifest) -> String {
    let mut blanked = manifest.clone();
    blanked.export_manifest_hash = String::new();
    let bytes = serde_json::to_vec(&blanked).unwrap();
    blake3::hash(&bytes).to_hex().to_string()
}

fn fresh_target_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry) {
    let dir = tempfile::tempdir().unwrap();
    let reg = LocalEvidenceAnchorRegistry::open(dir.path().join("target")).unwrap();
    (dir, reg)
}

fn assert_reason_tag(err: &EvidenceAnchorError, expected: &str) {
    let tag = omni_zkml::evidence_anchor_reason_tag(err);
    assert_eq!(tag, expected, "wrong tag on {err:?}");
}

const NOW: &str = "2026-06-18T00:00:00Z";

fn import_opts<'a>(
    selection: &'a AnchorImportSelection,
    dry_run: bool,
    strict: bool,
) -> AnchorImportOptions<'a> {
    AnchorImportOptions {
        dry_run,
        strict,
        selection,
        now_utc: NOW,
    }
}

fn select_all() -> AnchorImportSelection {
    AnchorImportSelection::default()
}

fn export_with_all_submitted(reg: &LocalEvidenceAnchorRegistry) -> tempfile::TempDir {
    build_export(
        reg,
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
    )
}

// ── Plan / selection ──────────────────────────────────────────────────────────

#[test]
fn plan_selects_all_anchor_records_when_no_selector_given() {
    // D6 lock — no-selector default imports all anchor_record entries
    // from the manifest. Asymmetric to Stage 13.5 export.
    let (_dir_src, src, _) = fresh_registry();
    let _ = seed_submitted(&src, [1u8; 32], 0x11);
    let _ = seed_submitted(&src, [2u8; 32], 0x22);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let plan = plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    assert_eq!(plan.actions.len(), 2);
}

#[test]
fn plan_selects_by_status_only() {
    let (_dir_src, src, _) = fresh_registry();
    let (h1, _) = seed_submitted(&src, [1u8; 32], 0x11);
    let (h2, _) = seed_submitted(&src, [2u8; 32], 0x22);
    src.update_status(&h2, LocalAnchorStatus::Finalized).unwrap();
    let export = build_export(
        &src,
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted, LocalAnchorStatus::Finalized],
            ..Default::default()
        },
        &[],
    );
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = AnchorImportSelection {
        statuses: vec![LocalAnchorStatus::Submitted],
        ..Default::default()
    };
    let plan = plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(plan.actions[0].artifact_hash_hex, h1);
}

#[test]
fn plan_selects_by_tx_id_only() {
    let (_dir_src, src, _) = fresh_registry();
    let _ = seed_submitted(&src, [3u8; 32], 0x33);
    let (h2, _) = seed_submitted(&src, [4u8; 32], 0x44);
    let record_h2 = src.load_by_artifact_hash(&h2).unwrap().unwrap();
    let tx_id = record_h2.receipt.tx_id.clone();
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = AnchorImportSelection {
        tx_ids: vec![tx_id],
        ..Default::default()
    };
    let plan = plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(plan.actions[0].artifact_hash_hex, h2);
}

#[test]
fn plan_selects_by_artifact_hash_only() {
    let (_dir_src, src, _) = fresh_registry();
    let (h1, _) = seed_submitted(&src, [5u8; 32], 0x55);
    let _ = seed_submitted(&src, [6u8; 32], 0x66);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = AnchorImportSelection {
        artifact_hashes: vec![h1.clone()],
        ..Default::default()
    };
    let plan = plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(plan.actions[0].artifact_hash_hex, h1);
}

#[test]
fn plan_combines_status_and_artifact_hash_intersection() {
    let (_dir_src, src, _) = fresh_registry();
    let (h1, _) = seed_submitted(&src, [7u8; 32], 0x77);
    let (h2, _) = seed_submitted(&src, [8u8; 32], 0x88);
    src.update_status(&h2, LocalAnchorStatus::Finalized).unwrap();
    let export = build_export(
        &src,
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted, LocalAnchorStatus::Finalized],
            ..Default::default()
        },
        &[],
    );
    let (_dir_tgt, tgt) = fresh_target_registry();
    // status=Submitted AND artifact_hash=h2 → empty intersection
    // (h2 is finalized).
    let sel = AnchorImportSelection {
        statuses: vec![LocalAnchorStatus::Submitted],
        artifact_hashes: vec![h2.clone()],
        ..Default::default()
    };
    let plan = plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    assert_eq!(plan.actions.len(), 0);
    // status=Submitted AND artifact_hash=h1 → just h1.
    let sel2 = AnchorImportSelection {
        statuses: vec![LocalAnchorStatus::Submitted],
        artifact_hashes: vec![h1.clone()],
        ..Default::default()
    };
    let plan2 = plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel2, true, false)).unwrap();
    assert_eq!(plan2.actions.len(), 1);
    assert_eq!(plan2.actions[0].artifact_hash_hex, h1);
}

// ── Verify-first invariant ────────────────────────────────────────────────────

#[test]
fn plan_refuses_when_export_verification_fails_with_export_blake3_mismatch_tag() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [9u8; 32], 0x99);
    let export = export_with_all_submitted(&src);
    // Replace the record file with different bytes; leave the
    // manifest unchanged so verify catches the BLAKE3 drift.
    std::fs::write(
        export.path().join(format!("anchors/{h}.json")),
        b"different bytes",
    )
    .unwrap();
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let err = plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap_err();
    assert_reason_tag(&err, "export_blake3_mismatch");
}

#[test]
fn apply_re_verifies_export_before_writing_so_post_plan_tamper_refuses() {
    // Plan succeeds, operator hand-edits the export between plan
    // and apply, apply re-runs verify and refuses with the Stage
    // 13.5 tag — durability fence per the plan.
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [10u8; 32], 0xaa);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let _ok_plan =
        plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    // Tamper now.
    std::fs::write(
        export.path().join(format!("anchors/{h}.json")),
        b"hand-edited after plan",
    )
    .unwrap();
    let err = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false))
        .unwrap_err();
    assert_reason_tag(&err, "export_blake3_mismatch");
}

// ── Selector miss ─────────────────────────────────────────────────────────────

#[test]
fn plan_refuses_when_selector_misses_manifest_entry_with_anchor_not_found_tag() {
    let (_dir_src, src, _) = fresh_registry();
    let _ = seed_submitted(&src, [11u8; 32], 0xbb);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = AnchorImportSelection {
        tx_ids: vec!["anchor-ghost".to_string()],
        ..Default::default()
    };
    let err = plan_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap_err();
    assert_reason_tag(&err, "anchor_not_found");
}

// ── Byte-preserve (D5 lock; prompt requirement) ───────────────────────────────

#[test]
fn import_preserves_record_bytes_exactly() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [12u8; 32], 0xcc);
    let source_bytes = std::fs::read(src.root().join(format!("{h}.json"))).unwrap();
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let _ = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false)).unwrap();
    let target_bytes = std::fs::read(tgt.root().join(format!("{h}.json"))).unwrap();
    assert_eq!(
        source_bytes, target_bytes,
        "imported record bytes must equal source record bytes EXACTLY (D5)"
    );
}

// ── Conflict matrix — row 1 (fresh import) ────────────────────────────────────

#[test]
fn import_imports_when_target_has_no_conflict() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [13u8; 32], 0xdd);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let report = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false)).unwrap();
    assert_eq!(report.actions_imported, 1);
    assert_eq!(report.actions_skipped_already_imported, 0);
    assert_eq!(report.outcomes[0].outcome, "imported");
    assert!(tgt.root().join(format!("{h}.json")).is_file());
    // tx_index.json should have been created with the imported tx_id.
    let idx_bytes = std::fs::read(tgt.root().join("tx_index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    let tx_id = idx.get("by_tx_id").unwrap().as_object().unwrap().keys().next().cloned().unwrap();
    assert!(!tx_id.is_empty());
}

// ── Conflict matrix — row 2 (skipped_already_imported) ────────────────────────

#[test]
fn import_idempotent_skipped_already_imported_for_byte_equal_record() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [14u8; 32], 0xee);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let _ = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false)).unwrap();
    // Re-run with no changes.
    let report = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false)).unwrap();
    assert_eq!(report.actions_imported, 0);
    assert_eq!(report.actions_skipped_already_imported, 1);
    assert_eq!(report.outcomes[0].outcome, "skipped_already_imported");
    // Record file still present, byte-equal to the source.
    let source_bytes = std::fs::read(src.root().join(format!("{h}.json"))).unwrap();
    let target_bytes = std::fs::read(tgt.root().join(format!("{h}.json"))).unwrap();
    assert_eq!(source_bytes, target_bytes);
}

// ── Conflict matrix — row 3 (re_added_tx_index_entry) ─────────────────────────

#[test]
fn import_re_adds_missing_tx_index_entry_when_record_file_already_present() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [15u8; 32], 0xa1);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    // Place the byte-equal record file directly into the target,
    // WITHOUT seeding the tx_index.json.
    let source_bytes = std::fs::read(src.root().join(format!("{h}.json"))).unwrap();
    std::fs::write(tgt.root().join(format!("{h}.json")), &source_bytes).unwrap();
    let mtime_before = std::fs::metadata(tgt.root().join(format!("{h}.json")))
        .unwrap()
        .modified()
        .unwrap();

    let sel = select_all();
    let report = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false)).unwrap();
    assert_eq!(report.actions_re_added_tx_index_entry, 1);
    assert_eq!(report.actions_imported, 0);
    assert_eq!(report.outcomes[0].outcome, "re_added_tx_index_entry");

    // Implementation note 2 — the record file must NOT have been
    // rewritten. Confirm via mtime preservation.
    let mtime_after = std::fs::metadata(tgt.root().join(format!("{h}.json")))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(
        mtime_before, mtime_after,
        "row-3 must NOT rewrite the byte-equal record file (implementation note 2)"
    );

    // tx_index.json now contains the tx_id.
    let idx_bytes = std::fs::read(tgt.root().join("tx_index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    let tx_id_count = idx.get("by_tx_id").unwrap().as_object().unwrap().len();
    assert_eq!(tx_id_count, 1);
}

// ── Conflict matrix — row 4 (artifact_hash collision, byte-different) ─────────

#[test]
fn import_refuses_when_target_artifact_hash_has_different_bytes() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [16u8; 32], 0xa2);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    // Place a DIFFERENT-bytes file at the target's artifact-hash
    // slot. Real-world cause: target had its own legitimate
    // record under the same hash with a different signer.
    std::fs::write(tgt.root().join(format!("{h}.json")), b"different record bytes").unwrap();

    let sel = select_all();
    let err = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false))
        .unwrap_err();
    assert_reason_tag(&err, "import_target_exists");
    // Pin the field discriminator.
    match err {
        EvidenceAnchorError::ImportTargetExists { field, .. } => {
            assert_eq!(field, "artifact_hash");
        }
        other => panic!("expected ImportTargetExists, got {other:?}"),
    }
}

// ── Conflict matrix — row 5 (tx_id collision, different hash) ─────────────────

#[test]
fn import_refuses_when_tx_id_in_target_maps_to_different_artifact_hash() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [17u8; 32], 0xa3);
    let record = src.load_by_artifact_hash(&h).unwrap().unwrap();
    let tx_id = record.receipt.tx_id.clone();
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    // Seed the target's tx_index.json with the SAME tx_id mapped
    // to a DIFFERENT artifact_hash.
    let phantom_hash = "00".repeat(32);
    std::fs::write(
        tgt.root().join("tx_index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "by_tx_id": { tx_id.clone(): phantom_hash }
        }))
        .unwrap(),
    )
    .unwrap();

    let sel = select_all();
    let err = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false))
        .unwrap_err();
    assert_reason_tag(&err, "import_target_exists");
    match err {
        EvidenceAnchorError::ImportTargetExists { field, .. } => {
            assert_eq!(field, "tx_id");
        }
        other => panic!("expected ImportTargetExists, got {other:?}"),
    }
}

// ── tx_index merge (D3 lock) ──────────────────────────────────────────────────

#[test]
fn import_preserves_unrelated_tx_index_entries_in_target_registry() {
    let (_dir_src, src, _) = fresh_registry();
    let (_h_src, _) = seed_submitted(&src, [18u8; 32], 0xa4);
    let export = export_with_all_submitted(&src);

    let (_dir_tgt, tgt) = fresh_target_registry();
    // Seed an unrelated tx_index entry that's already in the
    // target registry. The import must NOT clobber it (D3).
    let unrelated_tx_id = "unrelated-tx".to_string();
    let unrelated_hash = "11".repeat(32);
    std::fs::write(
        tgt.root().join("tx_index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "by_tx_id": { unrelated_tx_id.clone(): unrelated_hash.clone() }
        }))
        .unwrap(),
    )
    .unwrap();

    let sel = select_all();
    let _ = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false)).unwrap();

    let idx_bytes = std::fs::read(tgt.root().join("tx_index.json")).unwrap();
    let idx: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
    let by_tx_id = idx.get("by_tx_id").unwrap().as_object().unwrap();
    // Unrelated entry preserved.
    assert_eq!(
        by_tx_id.get(&unrelated_tx_id).and_then(|v| v.as_str()),
        Some(unrelated_hash.as_str()),
        "merge must preserve unrelated tx_index entries (D3 lock)"
    );
    // Imported entry also present.
    assert_eq!(by_tx_id.len(), 2);
}

// ── Dry-run outcomes ──────────────────────────────────────────────────────────

#[test]
fn apply_dry_run_emits_would_import_outcomes_and_does_not_mutate_target_registry() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [19u8; 32], 0xa5);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let report = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    assert_eq!(report.mode, "dry_run");
    assert_eq!(report.actions_would_import, 1);
    assert_eq!(report.actions_imported, 0);
    assert_eq!(report.outcomes[0].outcome, "would_import");
    assert!(!tgt.root().join(format!("{h}.json")).exists());
    assert!(!tgt.root().join("tx_index.json").exists());
}

#[test]
fn apply_dry_run_emits_would_re_add_tx_index_entry_for_row_3_case() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [20u8; 32], 0xa6);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let source_bytes = std::fs::read(src.root().join(format!("{h}.json"))).unwrap();
    std::fs::write(tgt.root().join(format!("{h}.json")), &source_bytes).unwrap();
    let sel = select_all();
    let report = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    assert_eq!(report.actions_would_re_add_tx_index_entry, 1);
    assert_eq!(report.outcomes[0].outcome, "would_re_add_tx_index_entry");
    // tx_index.json still absent.
    assert!(!tgt.root().join("tx_index.json").exists());
}

#[test]
fn apply_dry_run_emits_skipped_already_imported_for_row_2_case_with_no_would_prefix() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [21u8; 32], 0xa7);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    // Seed both the file and the tx_index for row-2 state.
    let source_bytes = std::fs::read(src.root().join(format!("{h}.json"))).unwrap();
    std::fs::write(tgt.root().join(format!("{h}.json")), &source_bytes).unwrap();
    let record = src.load_by_artifact_hash(&h).unwrap().unwrap();
    std::fs::write(
        tgt.root().join("tx_index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "by_tx_id": { record.receipt.tx_id.clone(): h.clone() }
        }))
        .unwrap(),
    )
    .unwrap();

    let sel = select_all();
    let report = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, true, false)).unwrap();
    // Row-2 outcome string is identical in apply AND dry-run —
    // no `would_` prefix because the state is already idempotent.
    assert_eq!(report.outcomes[0].outcome, "skipped_already_imported");
    assert_eq!(report.actions_skipped_already_imported, 1);
}

// ── Strict mode passthrough ───────────────────────────────────────────────────

#[test]
fn apply_strict_mode_calls_verify_anchor_export_with_strict_true() {
    // Export has an anchor_record but no paired artifact_bytes;
    // import with --strict should refuse via the Stage 13.5
    // strict-mode tag.
    let (_dir_src, src, _) = fresh_registry();
    let _ = seed_submitted(&src, [22u8; 32], 0xa8);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let err = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, true))
        .unwrap_err();
    assert_reason_tag(&err, "export_strict_mode_artifact_bytes_missing");
}

// ── Metadata-mismatch tag reuse (D7 lock) ─────────────────────────────────────

#[test]
fn apply_refuses_when_manifest_artifact_hash_does_not_match_record_field_with_export_entry_metadata_mismatch_tag()
 {
    let (_dir_src, src, _) = fresh_registry();
    let (_h, _) = seed_submitted(&src, [23u8; 32], 0xa9);
    let export = export_with_all_submitted(&src);
    // Hand-edit the manifest's entry: claim a different
    // artifact_hash_hex than the record carries. We must also
    // re-compute the manifest hash so the change passes the
    // manifest-hash gate and reaches the per-entry cross-check.
    let mut manifest = read_manifest(export.path());
    manifest.entries[0].artifact_hash_hex = Some("00".repeat(32));
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(export.path(), &manifest);

    let (_dir_tgt, tgt) = fresh_target_registry();
    let sel = select_all();
    let err = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false))
        .unwrap_err();
    // The Stage 13.5 verify_anchor_export step catches the
    // metadata mismatch on the way in — refusal routes through
    // the existing tag per D7.
    assert_reason_tag(&err, "export_entry_metadata_mismatch");
}

// ── field= disambiguator on import_target_exists ──────────────────────────────

#[test]
fn target_exists_refusal_has_field_artifact_hash_when_hash_collides() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [24u8; 32], 0xb1);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    // Different-bytes record at the artifact-hash slot. No
    // tx_index entry — so the tx_id row-5 check passes; row 4
    // fires.
    std::fs::write(tgt.root().join(format!("{h}.json")), b"divergent").unwrap();
    let sel = select_all();
    let err = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false))
        .unwrap_err();
    match err {
        EvidenceAnchorError::ImportTargetExists { field, .. } => {
            assert_eq!(field, "artifact_hash");
        }
        other => panic!("expected ImportTargetExists{{field=artifact_hash}}, got {other:?}"),
    }
}

#[test]
fn target_exists_refusal_has_field_tx_id_when_tx_id_collides() {
    let (_dir_src, src, _) = fresh_registry();
    let (h, _) = seed_submitted(&src, [25u8; 32], 0xb2);
    let record = src.load_by_artifact_hash(&h).unwrap().unwrap();
    let tx_id = record.receipt.tx_id.clone();
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    // Target's tx_index maps the same tx_id to a DIFFERENT hash.
    // No file at the artifact-hash slot — row 5 wins regardless.
    let phantom_hash = "22".repeat(32);
    std::fs::write(
        tgt.root().join("tx_index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "by_tx_id": { tx_id.clone(): phantom_hash }
        }))
        .unwrap(),
    )
    .unwrap();
    let sel = select_all();
    let err = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false))
        .unwrap_err();
    match err {
        EvidenceAnchorError::ImportTargetExists { field, .. } => {
            assert_eq!(field, "tx_id");
        }
        other => panic!("expected ImportTargetExists{{field=tx_id}}, got {other:?}"),
    }
}

// ── Apply preflight-all-before-mutate (note 1 lock) ───────────────────────────

#[test]
fn apply_does_not_modify_target_when_any_action_refuses_with_target_exists() {
    // Two-record export. Record A would import cleanly; record B
    // trips a row-4 collision. Preflight-all-before-mutate says
    // the conflict on B refuses with NO write of A.
    let (_dir_src, src, _) = fresh_registry();
    let (h_a, _) = seed_submitted(&src, [26u8; 32], 0xb3);
    let (h_b, _) = seed_submitted(&src, [27u8; 32], 0xb4);
    let export = export_with_all_submitted(&src);
    let (_dir_tgt, tgt) = fresh_target_registry();
    // Plant a row-4 collision for record B.
    std::fs::write(tgt.root().join(format!("{h_b}.json")), b"divergent").unwrap();
    // Pre-state snapshot: only the B-collision file exists; A's
    // file does NOT exist in the target.
    assert!(!tgt.root().join(format!("{h_a}.json")).exists());

    let sel = select_all();
    let err = apply_anchor_export_import(export.path(), &tgt, &import_opts(&sel, false, false))
        .unwrap_err();
    assert_reason_tag(&err, "import_target_exists");
    // Implementation note 1 — A must NOT have been written.
    assert!(
        !tgt.root().join(format!("{h_a}.json")).exists(),
        "preflight-all-before-mutate: a fresh-import action must not write \
         if a later action would refuse"
    );
    // The B file (the squatter) must still be there with its
    // original (divergent) bytes.
    let target_b = std::fs::read(tgt.root().join(format!("{h_b}.json"))).unwrap();
    assert_eq!(target_b, b"divergent");
    // No tx_index.json was created either.
    assert!(!tgt.root().join("tx_index.json").exists());
}
