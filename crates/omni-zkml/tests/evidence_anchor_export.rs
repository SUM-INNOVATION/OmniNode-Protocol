//! Phase 5 Stage 13.5 — integration tests for local-only anchor
//! export / import verification.
//!
//! Hermetic. Temp dirs everywhere. No network. Seeding mirrors
//! the Stage 13.4 cleanup tests: records are inserted via the
//! public workflow API for byte-equality with production; direct
//! JSON-file mutations stage tampering scenarios.

use std::path::PathBuf;

use omni_zkml::{
    anchor_hex_lower, anchor_signer_pubkey_bytes, apply_anchor_export,
    build_anchor_digest, submit_evidence_anchor_workflow, verify_anchor_export,
    AnchorExportManifest, AnchorExportOptions, AnchorExportSelection,
    AnchorExportVerifyOptions, ArtifactBytesInclusion, EvidenceAnchorError,
    LocalAnchorStatus, LocalEvidenceAnchorRegistry, StubEvidenceAnchorChainClient,
    VerifiedWrapperMetadata, EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION,
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

/// Seed a `Submitted` record by running the real submit workflow.
/// Returns (`artifact_hash_hex`, raw_artifact_bytes).
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

fn bump_status(reg: &LocalEvidenceAnchorRegistry, hash: &str, to: LocalAnchorStatus) {
    reg.update_status(hash, to).unwrap();
}

fn export_out() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn read_manifest(export_dir: &std::path::Path) -> AnchorExportManifest {
    let bytes = std::fs::read(export_dir.join(EXPORT_MANIFEST_FILENAME)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn write_manifest(export_dir: &std::path::Path, manifest: &AnchorExportManifest) {
    let bytes = serde_json::to_vec_pretty(manifest).unwrap();
    std::fs::write(export_dir.join(EXPORT_MANIFEST_FILENAME), bytes).unwrap();
}

const NOW: &str = "2026-06-17T00:00:00Z";

fn run_export(
    reg: &LocalEvidenceAnchorRegistry,
    out: &std::path::Path,
    selection: AnchorExportSelection,
    inclusions: &[ArtifactBytesInclusion],
    scrs: &[PathBuf],
) -> omni_zkml::AnchorExportReport {
    apply_anchor_export(
        reg,
        &AnchorExportOptions {
            export_out: out,
            selection: &selection,
            artifact_bytes_inclusions: inclusions,
            signed_chain_report_paths: scrs,
            label: "",
            notes: "",
            now_utc: NOW,
        },
    )
    .unwrap()
}

fn assert_reason_tag(err: &EvidenceAnchorError, expected: &str) {
    let tag = omni_zkml::evidence_anchor_reason_tag(err);
    assert_eq!(tag, expected, "wrong tag on {err:?}");
}

// ── Plan / selection ──────────────────────────────────────────────────────────

#[test]
fn export_selects_records_by_status_only() {
    let (_dir, reg, _) = fresh_registry();
    let (h1, _) = seed_submitted(&reg, [1u8; 32], 0x11);
    let (h2, _) = seed_submitted(&reg, [2u8; 32], 0x22);
    bump_status(&reg, &h2, LocalAnchorStatus::Finalized);

    let out = export_out();
    let report = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    assert_eq!(report.anchors_written, 1);

    let manifest = read_manifest(out.path());
    assert_eq!(manifest.entries.len(), 1);
    assert_eq!(
        manifest.entries[0].artifact_hash_hex.as_deref(),
        Some(h1.as_str())
    );
}

#[test]
fn export_selects_records_by_tx_id_only() {
    let (_dir, reg, _) = fresh_registry();
    let (_, _) = seed_submitted(&reg, [3u8; 32], 0x33);
    let (h2, _) = seed_submitted(&reg, [4u8; 32], 0x44);
    let record_h2 = reg.load_by_artifact_hash(&h2).unwrap().unwrap();
    let tx_id = record_h2.receipt.tx_id.clone();

    let out = export_out();
    let report = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            tx_ids: vec![tx_id],
            ..Default::default()
        },
        &[],
        &[],
    );
    assert_eq!(report.anchors_written, 1);
    let manifest = read_manifest(out.path());
    assert_eq!(
        manifest.entries[0].artifact_hash_hex.as_deref(),
        Some(h2.as_str())
    );
}

#[test]
fn export_selects_records_by_artifact_hash_only() {
    let (_dir, reg, _) = fresh_registry();
    let (h1, _) = seed_submitted(&reg, [5u8; 32], 0x55);
    let _ = seed_submitted(&reg, [6u8; 32], 0x66);

    let out = export_out();
    let report = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            artifact_hashes: vec![h1.clone()],
            ..Default::default()
        },
        &[],
        &[],
    );
    assert_eq!(report.anchors_written, 1);
    let manifest = read_manifest(out.path());
    assert_eq!(
        manifest.entries[0].artifact_hash_hex.as_deref(),
        Some(h1.as_str())
    );
}

#[test]
fn export_combines_status_and_tx_id_intersection() {
    let (_dir, reg, _) = fresh_registry();
    let (_, _) = seed_submitted(&reg, [7u8; 32], 0x77);
    let (h2, _) = seed_submitted(&reg, [8u8; 32], 0x88);
    let record_h2 = reg.load_by_artifact_hash(&h2).unwrap().unwrap();
    let tx_id_h2 = record_h2.receipt.tx_id.clone();
    bump_status(&reg, &h2, LocalAnchorStatus::Finalized);

    let out = export_out();
    // Selection: status=Submitted AND tx_id=<h2's tx_id>. Since h2
    // is now Finalized, the intersection is empty.
    let report = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            tx_ids: vec![tx_id_h2.clone()],
            ..Default::default()
        },
        &[],
        &[],
    );
    assert_eq!(report.anchors_written, 0);
}

#[test]
fn export_refuses_when_tx_id_selector_misses_record_with_anchor_not_found_tag() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [9u8; 32], 0x99);

    let out = export_out();
    let err = apply_anchor_export(
        &reg,
        &AnchorExportOptions {
            export_out: out.path(),
            selection: &AnchorExportSelection {
                tx_ids: vec!["anchor-ghost".to_string()],
                ..Default::default()
            },
            artifact_bytes_inclusions: &[],
            signed_chain_report_paths: &[],
            label: "",
            notes: "",
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "anchor_not_found");
}

#[test]
fn export_refuses_when_artifact_hash_selector_misses_record_with_anchor_not_found_tag() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [10u8; 32], 0xaa);

    let out = export_out();
    let err = apply_anchor_export(
        &reg,
        &AnchorExportOptions {
            export_out: out.path(),
            selection: &AnchorExportSelection {
                artifact_hashes: vec!["bb".repeat(32)],
                ..Default::default()
            },
            artifact_bytes_inclusions: &[],
            signed_chain_report_paths: &[],
            label: "",
            notes: "",
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "anchor_not_found");
}

#[test]
fn export_refuses_when_export_out_is_non_empty() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [11u8; 32], 0xbb);

    let out = export_out();
    std::fs::write(out.path().join("squatter.txt"), b"hi").unwrap();
    let err = apply_anchor_export(
        &reg,
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
    .unwrap_err();
    // No-clobber surfaces as `reason=io` with AlreadyExists kind.
    assert_reason_tag(&err, "io");
}

// ── Manifest invariants ───────────────────────────────────────────────────────

#[test]
fn export_manifest_schema_version_is_one() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [12u8; 32], 0xcc);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let manifest = read_manifest(out.path());
    assert_eq!(
        manifest.schema_version,
        EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION
    );
    assert_eq!(manifest.schema_version, 1);
}

#[test]
fn export_manifest_hash_is_byte_stable_across_recomputation() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [13u8; 32], 0xdd);
    let out1 = export_out();
    let out2 = export_out();
    let _ = run_export(
        &reg,
        out1.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let _ = run_export(
        &reg,
        out2.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let m1 = read_manifest(out1.path());
    let m2 = read_manifest(out2.path());
    // Same selection + same NOW + same registry contents → same hash.
    assert_eq!(m1.export_manifest_hash, m2.export_manifest_hash);
}

#[test]
fn export_manifest_does_not_contain_anchor_registry_dir_field() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [14u8; 32], 0xee);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let bytes = std::fs::read(out.path().join(EXPORT_MANIFEST_FILENAME)).unwrap();
    let raw_text = std::str::from_utf8(&bytes).unwrap();
    assert!(
        !raw_text.contains("anchor_registry_dir"),
        "manifest must not leak host-local registry path \
         (Stage 13.5 REJECT-fix Finding 2)"
    );
}

#[test]
fn export_id_is_deterministic_for_same_inputs_and_timestamp() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [15u8; 32], 0xa1);
    let out1 = export_out();
    let out2 = export_out();
    let r1 = run_export(
        &reg,
        out1.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let r2 = run_export(
        &reg,
        out2.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    assert_eq!(r1.export_id, r2.export_id);
    assert_eq!(r1.export_id.len(), 16);
}

#[test]
fn export_entries_are_sorted_by_relative_path_ascending() {
    let (_dir, reg, _) = fresh_registry();
    let (_, _) = seed_submitted(&reg, [16u8; 32], 0xa2);
    let (_, _) = seed_submitted(&reg, [17u8; 32], 0xa3);
    let (_, _) = seed_submitted(&reg, [18u8; 32], 0xa4);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let manifest = read_manifest(out.path());
    let paths: Vec<&str> = manifest.entries.iter().map(|e| e.relative_path.as_str()).collect();
    let mut sorted = paths.clone();
    sorted.sort();
    assert_eq!(paths, sorted, "entries must be sorted by relative_path asc");
}

// ── Layout ────────────────────────────────────────────────────────────────────

#[test]
fn export_layout_has_anchors_subdir() {
    let (_dir, reg, _) = fresh_registry();
    let (h, _) = seed_submitted(&reg, [19u8; 32], 0xa5);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    assert!(out.path().join("anchors").is_dir());
    assert!(out.path().join(format!("anchors/{h}.json")).is_file());
}

#[test]
fn export_does_not_copy_tx_index_json() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [20u8; 32], 0xa6);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    assert!(!out.path().join("tx_index.json").exists());
    assert!(!out.path().join("anchors/tx_index.json").exists());
}

// ── Optional include ─────────────────────────────────────────────────────────

#[test]
fn export_include_artifact_bytes_copies_files_and_emits_manifest_entry() {
    let (_dir, reg, _) = fresh_registry();
    let (h, raw) = seed_submitted(&reg, [21u8; 32], 0xa7);
    let bytes_dir = tempfile::tempdir().unwrap();
    let bytes_path = bytes_dir.path().join("artifact.bin");
    std::fs::write(&bytes_path, &raw).unwrap();

    let out = export_out();
    let report = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[ArtifactBytesInclusion {
            path: bytes_path.clone(),
            artifact_hash_hex: h.clone(),
        }],
        &[],
    );
    assert_eq!(report.artifact_bytes_written, 1);
    assert!(out.path().join("artifacts").is_dir());
    assert!(out.path().join(format!("artifacts/{h}")).is_file());
}

#[test]
fn export_include_signed_chain_report_copies_files_and_emits_manifest_entry() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [22u8; 32], 0xa8);
    let scr_dir = tempfile::tempdir().unwrap();
    let scr_path = scr_dir.path().join("chain_report.json");
    std::fs::write(&scr_path, b"{\"fake\":true}").unwrap();

    let out = export_out();
    let report = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        std::slice::from_ref(&scr_path),
    );
    assert_eq!(report.signed_chain_reports_written, 1);
    assert!(out.path().join("signed_chain_reports/chain_report.json").is_file());
}

#[test]
fn export_include_artifact_bytes_refuses_when_claimed_hash_doesnt_match_file() {
    let (_dir, reg, _) = fresh_registry();
    let (_, _) = seed_submitted(&reg, [23u8; 32], 0xa9);
    let bytes_dir = tempfile::tempdir().unwrap();
    let bytes_path = bytes_dir.path().join("artifact.bin");
    std::fs::write(&bytes_path, b"not the right bytes").unwrap();

    let out = export_out();
    let err = apply_anchor_export(
        &reg,
        &AnchorExportOptions {
            export_out: out.path(),
            selection: &AnchorExportSelection {
                statuses: vec![LocalAnchorStatus::Submitted],
                ..Default::default()
            },
            artifact_bytes_inclusions: &[ArtifactBytesInclusion {
                path: bytes_path,
                // Lie about the hash.
                artifact_hash_hex: "00".repeat(32),
            }],
            signed_chain_report_paths: &[],
            label: "",
            notes: "",
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "export_entry_metadata_mismatch");
}

// ── Duplicate relative_path defense ──────────────────────────────────────────

#[test]
fn export_refuses_when_two_signed_chain_reports_have_duplicate_basename() {
    // Two distinct source files share the same basename ("scr.json"),
    // so the planner derives the same `signed_chain_reports/scr.json`
    // relative path for both. Without the dedup, apply would
    // overwrite the first with the second AND record two manifest
    // entries with different `blake3_hex`s — the export would fail
    // its own verifier. Refuse at plan time with
    // `reason=export_invalid_path`.
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [60u8; 32], 0xd0);
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let p_a = dir_a.path().join("scr.json");
    let p_b = dir_b.path().join("scr.json");
    std::fs::write(&p_a, b"{\"version\":\"A\"}").unwrap();
    std::fs::write(&p_b, b"{\"version\":\"B-different-bytes\"}").unwrap();
    let out = export_out();
    let err = apply_anchor_export(
        &reg,
        &AnchorExportOptions {
            export_out: out.path(),
            selection: &AnchorExportSelection {
                statuses: vec![LocalAnchorStatus::Submitted],
                ..Default::default()
            },
            artifact_bytes_inclusions: &[],
            signed_chain_report_paths: &[p_a, p_b],
            label: "",
            notes: "",
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "export_invalid_path");
}

#[test]
fn export_refuses_when_two_include_artifact_bytes_inclusions_target_same_hash() {
    // Two `--include-artifact-bytes` inclusions claim the same
    // artifact_hash_hex — they'd collide in the artifacts/ subdir.
    // Both files genuinely BLAKE3 to the claimed hash (so the
    // per-inclusion check passes), but the resulting plan has a
    // duplicate `artifacts/<hash>` relative_path. Refuse with
    // `reason=export_invalid_path`.
    let (_dir, reg, _) = fresh_registry();
    let (h, raw) = seed_submitted(&reg, [61u8; 32], 0xd1);
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let p_a = dir_a.path().join("artifact_a.bin");
    let p_b = dir_b.path().join("artifact_b.bin");
    // Same bytes → both genuinely match the claimed hash.
    std::fs::write(&p_a, &raw).unwrap();
    std::fs::write(&p_b, &raw).unwrap();
    let out = export_out();
    let err = apply_anchor_export(
        &reg,
        &AnchorExportOptions {
            export_out: out.path(),
            selection: &AnchorExportSelection {
                statuses: vec![LocalAnchorStatus::Submitted],
                ..Default::default()
            },
            artifact_bytes_inclusions: &[
                ArtifactBytesInclusion {
                    path: p_a,
                    artifact_hash_hex: h.clone(),
                },
                ArtifactBytesInclusion {
                    path: p_b,
                    artifact_hash_hex: h.clone(),
                },
            ],
            signed_chain_report_paths: &[],
            label: "",
            notes: "",
            now_utc: NOW,
        },
    )
    .unwrap_err();
    assert_reason_tag(&err, "export_invalid_path");
}

#[test]
fn verify_refuses_on_duplicate_relative_path_in_manifest_for_signed_chain_reports() {
    // Hand-edit a previously-valid manifest to point two
    // signed_chain_report entries at the same relative_path. The
    // verifier's duplicate guard fires BEFORE any FS read.
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [62u8; 32], 0xd2);
    let scr_dir = tempfile::tempdir().unwrap();
    let scr_path = scr_dir.path().join("chain_report.json");
    std::fs::write(&scr_path, b"{\"any\":\"json\"}").unwrap();
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        std::slice::from_ref(&scr_path),
    );
    let mut manifest = read_manifest(out.path());
    // Append a second entry with the same relative_path as the
    // first signed_chain_report entry. Use the exact same
    // blake3/bytes so we'd land on the duplicate-relative_path
    // refusal, not blake3 mismatch.
    let scr_entry = manifest
        .entries
        .iter()
        .find(|e| e.kind == "signed_chain_report")
        .cloned()
        .unwrap();
    manifest.entries.push(scr_entry);
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_invalid_path");
}

#[test]
fn verify_refuses_on_duplicate_relative_path_in_manifest_for_anchor_records() {
    // Hand-edit a previously-valid manifest to add a second
    // anchor_record entry pointing at the same relative_path as
    // an existing one. The verifier's duplicate guard fires
    // BEFORE any FS read of that record.
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [63u8; 32], 0xd3);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    let dupe = manifest.entries[0].clone();
    manifest.entries.push(dupe);
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_invalid_path");
}

// ── Verify happy paths ────────────────────────────────────────────────────────

#[test]
fn verify_export_with_anchor_records_only_succeeds() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [24u8; 32], 0xb1);
    let _ = seed_submitted(&reg, [25u8; 32], 0xb2);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let report = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap();
    assert_eq!(report.anchor_records_verified, 2);
    assert_eq!(report.artifact_bytes_verified, 0);
    assert_eq!(report.pairings_artifact_hash_bound, 0);
}

#[test]
fn verify_export_with_artifact_bytes_runs_artifact_hash_binding() {
    let (_dir, reg, _) = fresh_registry();
    let (h, raw) = seed_submitted(&reg, [26u8; 32], 0xb3);
    let bytes_dir = tempfile::tempdir().unwrap();
    let bytes_path = bytes_dir.path().join("artifact.bin");
    std::fs::write(&bytes_path, &raw).unwrap();

    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[ArtifactBytesInclusion {
            path: bytes_path,
            artifact_hash_hex: h,
        }],
        &[],
    );
    let report = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap();
    assert_eq!(report.anchor_records_verified, 1);
    assert_eq!(report.artifact_bytes_verified, 1);
    assert_eq!(report.pairings_artifact_hash_bound, 1);
}

#[test]
fn verify_export_with_signed_chain_report_validates_blake3_only() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [27u8; 32], 0xb4);
    let scr_dir = tempfile::tempdir().unwrap();
    let scr_path = scr_dir.path().join("chain_report.json");
    std::fs::write(&scr_path, b"{\"any\":\"json\"}").unwrap();
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[scr_path],
    );
    let report = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap();
    assert_eq!(report.signed_chain_reports_verified, 1);
}

#[test]
fn verify_export_with_all_three_kinds_succeeds() {
    let (_dir, reg, _) = fresh_registry();
    let (h1, raw1) = seed_submitted(&reg, [28u8; 32], 0xb5);
    let (_h2, _raw2) = seed_submitted(&reg, [29u8; 32], 0xb6);
    let bytes_dir = tempfile::tempdir().unwrap();
    let bytes_path = bytes_dir.path().join("artifact1.bin");
    std::fs::write(&bytes_path, &raw1).unwrap();
    let scr_dir = tempfile::tempdir().unwrap();
    let scr_path = scr_dir.path().join("chain_report.json");
    std::fs::write(&scr_path, b"{\"any\":\"json\"}").unwrap();
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[ArtifactBytesInclusion {
            path: bytes_path,
            artifact_hash_hex: h1,
        }],
        &[scr_path],
    );
    let report = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap();
    assert_eq!(report.anchor_records_verified, 2);
    assert_eq!(report.artifact_bytes_verified, 1);
    assert_eq!(report.signed_chain_reports_verified, 1);
    assert_eq!(report.pairings_artifact_hash_bound, 1);
}

// ── Verify refusals — reused tags ─────────────────────────────────────────────

#[test]
fn verify_refuses_on_tampered_anchor_record_signature() {
    let (_dir, reg, _) = fresh_registry();
    let (h, _) = seed_submitted(&reg, [30u8; 32], 0xb7);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    // Tamper: flip a byte inside the submitter_signature (preserves
    // length, breaks signature). Update the entry's blake3_hex AND
    // the manifest hash to keep prior checks satisfied so the
    // refusal lands on submitter_signature_invalid, not earlier.
    let record_path = out.path().join(format!("anchors/{h}.json"));
    let record_bytes = std::fs::read(&record_path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&record_bytes).unwrap();
    let sig = value["tx_data"]["submitter_signature"].as_array_mut().unwrap();
    sig[0] = serde_json::Value::Number(0u8.into());
    let new_bytes = serde_json::to_vec_pretty(&value).unwrap();
    std::fs::write(&record_path, &new_bytes).unwrap();
    rehash_manifest(out.path(), &h, &new_bytes);

    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "submitter_signature_invalid");
}

#[test]
fn verify_refuses_on_malformed_anchor_record_json() {
    let (_dir, reg, _) = fresh_registry();
    let (h, _) = seed_submitted(&reg, [31u8; 32], 0xb8);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let record_path = out.path().join(format!("anchors/{h}.json"));
    let new_bytes = b"NOT JSON".to_vec();
    std::fs::write(&record_path, &new_bytes).unwrap();
    rehash_manifest(out.path(), &h, &new_bytes);

    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "malformed_json");
}

#[test]
fn verify_refuses_on_missing_entry_file() {
    let (_dir, reg, _) = fresh_registry();
    let (h, _) = seed_submitted(&reg, [32u8; 32], 0xb9);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let record_path = out.path().join(format!("anchors/{h}.json"));
    std::fs::remove_file(&record_path).unwrap();
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "io");
}

#[test]
fn verify_refuses_on_manifest_parse_failure() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [33u8; 32], 0xba);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    std::fs::write(out.path().join(EXPORT_MANIFEST_FILENAME), b"BAD JSON").unwrap();
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "malformed_json");
}

// ── Verify refusals — new tags ────────────────────────────────────────────────

#[test]
fn verify_refuses_on_manifest_schema_version_drift() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [34u8; 32], 0xbb);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    manifest.schema_version = 2;
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "unsupported_export_manifest_schema_version");
}

#[test]
fn verify_refuses_on_manifest_hash_mismatch() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [35u8; 32], 0xbc);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    manifest.notes = "tampered after the fact".to_string();
    // Note: we do NOT recompute export_manifest_hash, so the
    // declared hash no longer matches the recomputed canonical
    // bytes.
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_manifest_hash_mismatch");
}

#[test]
fn verify_refuses_on_per_entry_blake3_mismatch() {
    let (_dir, reg, _) = fresh_registry();
    let (h, _) = seed_submitted(&reg, [36u8; 32], 0xbd);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    // Replace the file but leave the manifest hash unchanged.
    let record_path = out.path().join(format!("anchors/{h}.json"));
    std::fs::write(&record_path, b"different bytes of same-shape json").unwrap();
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_blake3_mismatch");
}

#[test]
fn verify_refuses_on_per_entry_relative_path_absolute() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [37u8; 32], 0xbe);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    manifest.entries[0].relative_path = "/etc/passwd".to_string();
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_invalid_path");
}

#[test]
fn verify_refuses_on_per_entry_relative_path_parent_traversal() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [38u8; 32], 0xbf);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    manifest.entries[0].relative_path = "anchors/../etc/passwd".to_string();
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_invalid_path");
}

#[test]
fn verify_refuses_on_per_entry_relative_path_wrong_per_kind_shape() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [39u8; 32], 0xc0);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    // Wrong subdir for anchor_record kind.
    manifest.entries[0].relative_path = "artifacts/whatever.json".to_string();
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_invalid_path");
}

#[test]
fn verify_refuses_on_metadata_mismatch_artifact_hash() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [40u8; 32], 0xc1);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    manifest.entries[0].artifact_hash_hex = Some("00".repeat(32));
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_entry_metadata_mismatch");
}

#[test]
fn verify_refuses_on_metadata_mismatch_tx_id() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [41u8; 32], 0xc2);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    manifest.entries[0].tx_id = Some("anchor-completely-different".to_string());
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_entry_metadata_mismatch");
}

#[test]
fn verify_refuses_on_metadata_mismatch_status() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [42u8; 32], 0xc3);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    manifest.entries[0].status = Some("finalized".to_string());
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_entry_metadata_mismatch");
}

// ── Ordering pin ──────────────────────────────────────────────────────────────

#[test]
fn verify_schema_version_check_fires_before_manifest_hash_check() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [43u8; 32], 0xc4);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let mut manifest = read_manifest(out.path());
    // Bump schema_version AND scramble the hash. If the
    // hash check ran first we'd see export_manifest_hash_mismatch;
    // the documented preflight order fires schema first.
    manifest.schema_version = 2;
    manifest.export_manifest_hash = "ff".repeat(32);
    write_manifest(out.path(), &manifest);
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "unsupported_export_manifest_schema_version");
}

// ── Strict mode ───────────────────────────────────────────────────────────────

#[test]
fn verify_strict_mode_refuses_when_artifact_bytes_missing_for_anchor_record() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed_submitted(&reg, [44u8; 32], 0xc5);
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[],
        &[],
    );
    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: true,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_strict_mode_artifact_bytes_missing");
}

#[test]
fn verify_strict_mode_succeeds_when_every_anchor_record_has_artifact_bytes() {
    let (_dir, reg, _) = fresh_registry();
    let (h, raw) = seed_submitted(&reg, [45u8; 32], 0xc6);
    let bytes_dir = tempfile::tempdir().unwrap();
    let bytes_path = bytes_dir.path().join("artifact.bin");
    std::fs::write(&bytes_path, &raw).unwrap();
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[ArtifactBytesInclusion {
            path: bytes_path,
            artifact_hash_hex: h,
        }],
        &[],
    );
    let report = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: true,
    })
    .unwrap();
    assert_eq!(report.pairings_artifact_hash_bound, 1);
    assert!(report.strict);
}

// ── Honesty-of-binding pins ──────────────────────────────────────────────────

#[test]
fn verify_with_artifact_bytes_proves_artifact_hash_binding_not_wrapper_signer_binding() {
    // What Stage 13.5 verify SHOULD prove with raw artifact bytes:
    // blake3(bytes) == record.tx_data.digest.artifact_hash. It
    // explicitly does NOT prove that the bytes were signed by any
    // specific "wrapper signer" — because no wrapper context is
    // supplied. If a future refactor sneaks the tautological
    // signer-self check back in via
    // `verify_anchor_file_against_artifact_bytes`, this test still
    // passes (the digest signer DOES equal itself); the negative
    // pin below catches the lie about the contract.
    let (_dir, reg, _) = fresh_registry();
    let (h, raw) = seed_submitted(&reg, [46u8; 32], 0xc7);
    let bytes_dir = tempfile::tempdir().unwrap();
    let bytes_path = bytes_dir.path().join("artifact.bin");
    std::fs::write(&bytes_path, &raw).unwrap();
    let out = export_out();
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[ArtifactBytesInclusion {
            path: bytes_path,
            artifact_hash_hex: h.clone(),
        }],
        &[],
    );
    let report = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap();
    assert_eq!(report.pairings_artifact_hash_bound, 1);
}

#[test]
fn verify_with_artifact_bytes_refuses_when_bytes_diverge_from_anchor_artifact_hash() {
    // Forge: a paired artifact_bytes file whose BLAKE3 equals the
    // operator-claimed hash (declared at export time) but does NOT
    // equal the anchor record's digest.artifact_hash. This stages a
    // post-export tamper of the manifest's artifact_hash claim while
    // the file's BLAKE3 stays consistent with the claim. The
    // verifier's paired artifact-hash binding check (step 9) catches
    // it via `export_entry_metadata_mismatch` on the
    // artifact_hash_binding field.
    let (_dir, reg, _) = fresh_registry();
    let (h, _raw) = seed_submitted(&reg, [47u8; 32], 0xc8);
    // Real bytes for *some other* hash.
    let forged_bytes = b"forged artifact bytes".to_vec();
    let forged_hash = anchor_hex_lower(blake3::hash(&forged_bytes).as_bytes());

    let bytes_dir = tempfile::tempdir().unwrap();
    let bytes_path = bytes_dir.path().join("artifact.bin");
    std::fs::write(&bytes_path, &forged_bytes).unwrap();

    let out = export_out();
    // Re-keying the artifact bytes inclusion to the forged_hash
    // gets us past export-time validation; then we surgically
    // rewrite the manifest entry's artifact_hash_hex to equal the
    // anchor record's `h` so verify believes they pair. Step 9
    // recomputes blake3 of the bytes file (= forged_hash !=
    // anchor.digest.artifact_hash = h) and refuses.
    let _ = run_export(
        &reg,
        out.path(),
        AnchorExportSelection {
            statuses: vec![LocalAnchorStatus::Submitted],
            ..Default::default()
        },
        &[ArtifactBytesInclusion {
            path: bytes_path,
            artifact_hash_hex: forged_hash.clone(),
        }],
        &[],
    );
    // Surgical rewrite — move the forged file into the anchor's
    // artifact-hash slot AND rewrite the manifest entry to pair
    // them.
    let dst = out.path().join(format!("artifacts/{h}"));
    std::fs::rename(out.path().join(format!("artifacts/{forged_hash}")), &dst).unwrap();
    let mut manifest = read_manifest(out.path());
    for e in manifest.entries.iter_mut() {
        if e.kind == "artifact_bytes" {
            e.relative_path = format!("artifacts/{h}");
            e.artifact_hash_hex = Some(h.clone());
            // Recompute file blake3 (still forged_bytes).
            e.blake3_hex = anchor_hex_lower(blake3::hash(&forged_bytes).as_bytes());
            e.bytes = forged_bytes.len() as u64;
        }
    }
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(out.path(), &manifest);

    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: out.path(),
        strict: false,
    })
    .unwrap_err();
    assert_reason_tag(&err, "export_entry_metadata_mismatch");
}

// ── Helpers for tampering tests ──────────────────────────────────────────────

fn recompute_manifest_hash(manifest: &AnchorExportManifest) -> String {
    let mut blanked = manifest.clone();
    blanked.export_manifest_hash = String::new();
    let bytes = serde_json::to_vec(&blanked).unwrap();
    anchor_hex_lower(blake3::hash(&bytes).as_bytes())
}

/// Rewrite the manifest entry for `anchors/<h>.json` so its
/// blake3 + length match `new_bytes`, then recompute the
/// manifest's own hash. Used by tests that mutate a record file
/// and need to land tampering downstream of steps 1–5 of verify.
fn rehash_manifest(export_dir: &std::path::Path, h: &str, new_bytes: &[u8]) {
    let mut manifest = read_manifest(export_dir);
    let target_rel = format!("anchors/{h}.json");
    for e in manifest.entries.iter_mut() {
        if e.relative_path == target_rel {
            e.blake3_hex = anchor_hex_lower(blake3::hash(new_bytes).as_bytes());
            e.bytes = new_bytes.len() as u64;
        }
    }
    manifest.export_manifest_hash = recompute_manifest_hash(&manifest);
    write_manifest(export_dir, &manifest);
}
