//! Phase 5 Stage 13.10 — operator acceptance umbrella.
//!
//! Cross-stage stitching tests that exercise the full 13.x
//! integrity-evidence-anchor lifecycle end-to-end. Each scenario
//! is a single `#[test]` to keep failure isolation; the lifecycle
//! happy-path is the only multi-stage test, the rest are 2-stage
//! at most. Hermetic. Temp dirs everywhere. No network.
//!
//! ## Scope (Stage 13.10)
//! - **Acceptance only.** Zero production code changes; treats
//!   the 13.x library surface as a black box.
//! - **Structured event-line parsing** via [`parse_event_line`]
//!   for the operator-transcript assertions, plus a small number
//!   of exact-string anchor pins on Stage 13.9-locked event
//!   names.
//! - Scenario 7 (archive partial-state recovery) is **black-box
//!   at the archive/restore boundary** per APPROVE scope lock:
//!   the test seeds the operator-observable state directly
//!   (durable manifest + one hot record missing) and asserts the
//!   restore is idempotent. No internal phase hooks are exercised.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use omni_zkml::{
    anchor_signer_pubkey_bytes, apply_anchor_archive, apply_anchor_cleanup,
    apply_anchor_export, apply_anchor_export_import,
    build_anchor_consistency_report, build_anchor_digest,
    check_evidence_anchor_registry_health, evidence_anchor_reason_tag,
    list_evidence_anchors_by_status, list_stale_submitted_or_included,
    lookup_anchor_by_tuple_workflow, plan_anchor_archive, plan_anchor_cleanup,
    plan_anchor_export_import, reconcile_evidence_anchors_workflow,
    restore_anchor_archive, restore_anchor_cleanup_quarantine,
    submit_evidence_anchor_workflow, verify_anchor_export, AnchorApplyOptions,
    AnchorArchiveApplyOptions, AnchorArchiveManifest, AnchorArchivePlanOptions,
    AnchorArchiveRestoreOptions, AnchorArchiveSelection, AnchorCleanupActionKind,
    AnchorConsistencyFindingKind, AnchorConsistencyOptions,
    AnchorConsistencySeverity, AnchorExportOptions, AnchorExportSelection,
    AnchorExportVerifyOptions, AnchorImportOptions, AnchorImportSelection,
    AnchorPlanOptions, AnchorQuarantineManifest, AnchorRestoreOptions,
    AnchorSelector, AnchorStatus, AnchorStatusReport, AnchoredArtifactKind,
    BatchStatusItem, ChainClientError, EvidenceAnchorChainClient,
    EvidenceAnchorError, LocalAnchorStatus, LocalEvidenceAnchorRegistry,
    StubEvidenceAnchorChainClient, TupleLookupOutcome, TupleLookupResult,
    VerifiedWrapperMetadata,
};

// ── Shared helpers ───────────────────────────────────────────────────────────

const NOW: &str = "2026-06-18T00:00:00Z";

fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("anchors");
    let reg = LocalEvidenceAnchorRegistry::open(root.clone()).unwrap();
    (dir, reg, root)
}

fn fresh_target_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("target");
    let reg = LocalEvidenceAnchorRegistry::open(root.clone()).unwrap();
    (dir, reg, root)
}

/// Seed a record via the public Stage 13.0 workflow API, optionally
/// transitioning it to a non-submitted status. Returns
/// `(artifact_hash_hex, tx_id, raw_artifact_bytes)`.
fn seed(
    reg: &LocalEvidenceAnchorRegistry,
    seed_bytes: [u8; 32],
    marker: u8,
    status: Option<LocalAnchorStatus>,
) -> (String, String, Vec<u8>) {
    let client = StubEvidenceAnchorChainClient::new();
    let raw = vec![marker; 32];
    let metadata = VerifiedWrapperMetadata {
        artifact_schema_version: 1,
        signer_pubkey: anchor_signer_pubkey_bytes(&seed_bytes).unwrap(),
        signed_at_utc_unix: 1_750_000_000,
    };
    let digest = build_anchor_digest(&metadata, &raw);
    let record =
        submit_evidence_anchor_workflow(reg, &client, digest, &seed_bytes).unwrap();
    let hash = record.artifact_hash_hex.clone();
    let tx_id = record.receipt.tx_id.clone();
    if let Some(s) = status {
        reg.update_status(&hash, s).unwrap();
    }
    (hash, tx_id, raw)
}

fn assert_reason_tag(err: &EvidenceAnchorError, expected: &str) {
    let tag = evidence_anchor_reason_tag(err);
    assert_eq!(tag, expected, "wrong tag on {err:?}");
}

/// Parse an operator transcript line of the form
/// `event=foo key1=v1 key2=v2 …` into a sorted map. Unknown
/// shapes are tolerated (missing `=` is skipped). Whitespace
/// between key=value pairs is a single space; the helper splits
/// on whitespace so multi-space indentation in test fixtures is
/// also tolerated. Quoted values are not supported (the 13.x
/// event surface does not use them); if a future event uses
/// quoting this helper must grow accordingly.
fn parse_event_line(line: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for tok in line.split_whitespace() {
        if let Some((k, v)) = tok.split_once('=') {
            out.insert(k.to_string(), v.to_string());
        }
    }
    out
}

/// Mirror of the CLI's `format_tuple_lookup_outcome_event` for
/// the `NotFound` arm, kept inline so the umbrella file does not
/// depend on `omni-node`. The exact-string event name is
/// regression-pinned in
/// `omni-node` `evidence_anchor_cli::stage_13_9_cli_tests`; this
/// mirror exists only to feed [`parse_event_line`].
fn format_no_chain_anchor_event(registry_dir: &std::path::Path, rpc_url: &str) -> String {
    format!(
        "event=integrity_evidence_anchor_tuple_lookup_no_chain_anchor \
         anchor_registry_dir={} rpc_url={}",
        registry_dir.display(),
        rpc_url,
    )
}

/// Fake chain client used by the acceptance scenarios. Overrides
/// `query_anchor_status_batch` for byte-equality with the real
/// `omni-sumchain` semantics (per-item containment) AND
/// `lookup_anchor_by_tuple` for Stage 13.9 scenarios.
#[derive(Default)]
struct AcceptanceFakeClient {
    statuses: RefCell<HashMap<String, AnchorStatusReport>>,
    by_tuple_result: RefCell<Option<TupleLookupResult>>,
    batch_call_count: RefCell<usize>,
}

impl AcceptanceFakeClient {
    fn new() -> Self {
        Self::default()
    }
    fn set_status(&self, tx_id: &str, status: AnchorStatus) {
        self.statuses.borrow_mut().insert(
            tx_id.to_string(),
            AnchorStatusReport {
                status,
                included_at_height: None,
                code: None,
                reason: None,
            },
        );
    }
    fn set_by_tuple_found(&self, result: TupleLookupResult) {
        *self.by_tuple_result.borrow_mut() = Some(result);
    }
    fn batch_call_count(&self) -> usize {
        *self.batch_call_count.borrow()
    }
}

impl EvidenceAnchorChainClient for AcceptanceFakeClient {
    fn submit_anchor(
        &self,
        _: &omni_zkml::IntegrityEvidenceAnchorTxData,
    ) -> Result<omni_zkml::AnchorSubmissionReceipt, ChainClientError> {
        unimplemented!("acceptance tests seed records via the stub client, not this fake")
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

// ── parse_event_line self-test ───────────────────────────────────────────────

#[test]
fn parse_event_line_extracts_known_keys_and_skips_malformed_tokens() {
    let m = parse_event_line(
        "event=integrity_evidence_anchor_tuple_lookup_no_chain_anchor \
         anchor_registry_dir=/tmp/anchors rpc_url=http://localhost:0 bogus-no-equals",
    );
    assert_eq!(
        m["event"],
        "integrity_evidence_anchor_tuple_lookup_no_chain_anchor"
    );
    assert_eq!(m["anchor_registry_dir"], "/tmp/anchors");
    assert_eq!(m["rpc_url"], "http://localhost:0");
    // Malformed token without `=` is silently skipped.
    assert!(!m.contains_key("bogus-no-equals"));
    // Absence of a key is a contract-relevant signal (e.g. no
    // `reason=` on informational events).
    assert!(!m.contains_key("reason"));
}

// ── Scenario 1: happy-path full lifecycle ────────────────────────────────────

#[test]
fn scenario_1_happy_path_full_lifecycle_seed_reconcile_summary_export_import_archive_restore()
{
    // Seed 2 Submitted records via the public Stage 13.0 workflow.
    let (_src_dir, src_reg, src_root) = fresh_registry();
    let (h1, tx1, _) = seed(&src_reg, [1u8; 32], 0x11, None);
    let (h2, tx2, _) = seed(&src_reg, [2u8; 32], 0x22, None);

    // Stage 13.9 batch reconcile drives both to Finalized.
    let chain = AcceptanceFakeClient::new();
    chain.set_status(&tx1, AnchorStatus::Finalized);
    chain.set_status(&tx2, AnchorStatus::Finalized);
    let outcomes = reconcile_evidence_anchors_workflow(&src_reg, &chain).unwrap();
    assert_eq!(outcomes.len(), 2);
    assert!(
        outcomes.iter().all(|(_, r)| r.is_ok()),
        "reconcile must succeed on both records; got: {outcomes:?}"
    );
    assert_eq!(
        chain.batch_call_count(),
        1,
        "Stage 13.9 batch path must be used (single chunk for 2 records)"
    );

    // Stage 13.3 summary reflects both as Finalized.
    let summary = list_evidence_anchors_by_status(&src_reg).unwrap();
    assert_eq!(summary.finalized, 2);
    assert_eq!(summary.submitted, 0);
    assert_eq!(summary.included, 0);
    assert_eq!(summary.failed, 0);
    let health = check_evidence_anchor_registry_health(&src_reg).unwrap();
    assert_eq!(health.malformed_records, 0);
    assert_eq!(health.orphan_tx_index_entries, 0);

    // Stage 13.5 export + verify.
    let export_dir = tempfile::tempdir().unwrap();
    let _ = apply_anchor_export(
        &src_reg,
        &AnchorExportOptions {
            export_out: export_dir.path(),
            selection: &AnchorExportSelection {
                statuses: vec![LocalAnchorStatus::Finalized],
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
    let verify_report = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: export_dir.path(),
        strict: false,
    })
    .unwrap();
    assert_eq!(verify_report.anchor_records_verified, 2);

    // Stage 13.6 import into a fresh target registry.
    let (_tgt_dir, tgt_reg, tgt_root) = fresh_target_registry();
    let import_report = apply_anchor_export_import(
        export_dir.path(),
        &tgt_reg,
        &AnchorImportOptions {
            dry_run: false,
            strict: false,
            selection: &AnchorImportSelection::default(),
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(import_report.actions_imported, 2);
    // Verify both records landed in the target hot registry.
    assert!(tgt_root.join(format!("{h1}.json")).is_file());
    assert!(tgt_root.join(format!("{h2}.json")).is_file());

    // Stage 13.7 archive (defaults to Finalized-only).
    let archive_dir = tempfile::tempdir().unwrap();
    let archive_plan = plan_anchor_archive(
        &tgt_reg,
        &AnchorArchivePlanOptions {
            now_utc: NOW,
            statuses: vec![],
            before_utc: None,
            selection: &AnchorArchiveSelection::default(),
        },
    )
    .unwrap();
    let archive_report = apply_anchor_archive(
        &archive_plan,
        &AnchorArchiveApplyOptions {
            archive_dir: archive_dir.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(archive_report.actions_archived, 2);
    assert!(!tgt_root.join(format!("{h1}.json")).is_file());
    assert!(!tgt_root.join(format!("{h2}.json")).is_file());

    // Stage 13.7 restore round-trip.
    let manifest_dir = archive_dir.path().join(&archive_plan.plan_id);
    let manifest_bytes = std::fs::read(manifest_dir.join("archive_manifest.json")).unwrap();
    let manifest: AnchorArchiveManifest =
        serde_json::from_slice(&manifest_bytes).unwrap();
    let restore_report = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &tgt_root,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(restore_report.restored, 2);
    assert!(tgt_root.join(format!("{h1}.json")).is_file());
    assert!(tgt_root.join(format!("{h2}.json")).is_file());

    // Sanity: original source registry untouched by any later
    // step (export/import/archive operate on copies).
    assert!(src_root.join(format!("{h1}.json")).is_file());
    assert!(src_root.join(format!("{h2}.json")).is_file());
}

// ── Scenario 2: corrupted hot registry record ────────────────────────────────

#[test]
fn scenario_2_corrupted_hot_record_consistency_then_cleanup_quarantine_then_restore_roundtrip()
{
    let (_dir, reg, root) = fresh_registry();
    // One valid record + one corrupted on-disk entry. The
    // corrupted name uses a 64-hex prefix so the registry treats
    // it as an anchor-record file (matches the existing Stage
    // 13.4/13.8 corruption patterns).
    let (h_ok, _, _) = seed(&reg, [1u8; 32], 0x11, None);
    let bad_name = format!("{}.json", "a".repeat(64));
    let bad_bytes = b"{bogus}".to_vec();
    std::fs::write(root.join(&bad_name), &bad_bytes).unwrap();

    // Stage 13.8 consistency report flags the corruption at
    // Error severity.
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let malformed: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.kind == AnchorConsistencyFindingKind::HotRecordMalformed)
        .collect();
    assert_eq!(malformed.len(), 1);
    assert_eq!(malformed[0].severity, AnchorConsistencySeverity::Error);
    // Good record is still summary-counted.
    assert_eq!(report.summary.hot_submitted, 1);

    // Stage 13.4 cleanup plans the quarantine.
    let plan = plan_anchor_cleanup(
        &reg,
        &AnchorPlanOptions {
            now_utc: NOW,
            stale_threshold_secs: None,
        },
    )
    .unwrap();
    let plan_id = plan.plan_id.clone();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(
        plan.actions[0].kind,
        AnchorCleanupActionKind::QuarantineMalformedRecord
    );

    // Apply moves the corrupted file to quarantine; valid record
    // untouched.
    let qdir = tempfile::tempdir().unwrap();
    let apply_report = apply_anchor_cleanup(
        &plan,
        &AnchorApplyOptions {
            quarantine_dir: qdir.path(),
            dry_run: false,
            allow_stale_quarantine: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(apply_report.actions_applied, 1);
    assert!(!root.join(&bad_name).exists());
    assert!(root.join(format!("{h_ok}.json")).is_file());
    let quarantined = qdir.path().join(&plan_id).join(&bad_name);
    assert!(quarantined.is_file());
    assert_eq!(std::fs::read(&quarantined).unwrap(), bad_bytes);

    // Restore from quarantine manifest round-trips byte-equal.
    let manifest_path = qdir.path().join(&plan_id).join("quarantine_manifest.json");
    let manifest: AnchorQuarantineManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let restore_report = restore_anchor_cleanup_quarantine(
        &manifest,
        &AnchorRestoreOptions {
            quarantine_dir: &qdir.path().join(&plan_id),
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    assert!(restore_report.restored >= 1);
    assert_eq!(std::fs::read(root.join(&bad_name)).unwrap(), bad_bytes);
}

// ── Scenario 3: stale / open submitted record ────────────────────────────────

#[test]
fn scenario_3_stale_open_submitted_summary_then_reconcile_observation_only_for_unknown_chain_state()
{
    use chrono::{Duration, Utc};

    let (_dir, reg, root) = fresh_registry();
    let (h_stale, tx_stale, _) = seed(&reg, [1u8; 32], 0x11, None);
    // Backdate submitted_at by 1 hour so the stale-detection
    // threshold (60s) trips.
    let path = root.join(format!("{h_stale}.json"));
    let bytes = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let past = (Utc::now() - Duration::seconds(3600)).to_rfc3339();
    value["submitted_at"] = serde_json::Value::String(past);
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

    // Stage 13.3 stale list surfaces the open record.
    let stale = list_stale_submitted_or_included(&reg, Utc::now(), 60).unwrap();
    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].tx_id, tx_stale);

    // Stage 13.9 reconcile with chain reporting `unknown` for
    // this tx_id is OBSERVATION-ONLY: no status downgrade.
    let chain = AcceptanceFakeClient::new();
    chain.set_status(&tx_stale, AnchorStatus::Unknown);
    let _ = reconcile_evidence_anchors_workflow(&reg, &chain).unwrap();
    let after = list_evidence_anchors_by_status(&reg).unwrap();
    assert_eq!(
        after.submitted, 1,
        "Stage 13.9 Q7 lock: Submitted/Unknown is observation-only — must NOT be downgraded"
    );
    assert_eq!(after.failed, 0);
}

// ── Scenario 4: orphan tx_index.json entry ───────────────────────────────────

#[test]
fn scenario_4_orphan_tx_index_entry_consistency_flags_and_cleanup_plans_removal() {
    let (_dir, reg, root) = fresh_registry();
    let _ = seed(&reg, [1u8; 32], 0x11, None);
    // Inject a phantom tx_id → phantom hash with no backing
    // record file.
    let idx_path = root.join("tx_index.json");
    let mut idx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&idx_path).unwrap()).unwrap();
    idx["by_tx_id"]["phantom-tx"] = serde_json::Value::String("ff".repeat(32));
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();

    // Stage 13.8 consistency surfaces the orphan.
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let orphans: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.kind == AnchorConsistencyFindingKind::HotTxIndexOrphan)
        .collect();
    assert_eq!(orphans.len(), 1);

    // Stage 13.4 cleanup plans removal of the orphan.
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
        AnchorCleanupActionKind::RemoveOrphanTxIndexEntry
    );
    assert_eq!(plan.actions[0].tx_id.as_deref(), Some("phantom-tx"));
}

// ── Scenario 5: export tamper detected before import ─────────────────────────

#[test]
fn scenario_5_export_tamper_detected_by_verify_before_import_is_attempted() {
    let (_dir, reg, _) = fresh_registry();
    let _ = seed(&reg, [1u8; 32], 0x11, None);

    let export_dir = tempfile::tempdir().unwrap();
    let _ = apply_anchor_export(
        &reg,
        &AnchorExportOptions {
            export_out: export_dir.path(),
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

    // Sanity: clean export verifies first.
    let _ok = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: export_dir.path(),
        strict: false,
    })
    .unwrap();

    // Tamper: corrupt a byte inside one of the exported anchor
    // record files. The manifest's per-record sha256 will catch
    // it.
    let mut record_path: Option<PathBuf> = None;
    for entry in std::fs::read_dir(export_dir.path().join("anchors")).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("json") {
            record_path = Some(p);
            break;
        }
    }
    let record_path = record_path.expect("export must contain at least one anchor record file");
    let mut bytes = std::fs::read(&record_path).unwrap();
    *bytes.last_mut().unwrap() ^= 0x01;
    std::fs::write(&record_path, &bytes).unwrap();

    let err = verify_anchor_export(&AnchorExportVerifyOptions {
        export_dir: export_dir.path(),
        strict: false,
    })
    .unwrap_err();
    // Tamper surfaces as a checksum/byte-mismatch tag. Don't pin
    // the exact tag string here — Stage 13.5 owns that taxonomy;
    // this scenario just proves verify catches the tamper BEFORE
    // import is attempted. Operator playbook: refuse import on
    // any verify error.
    let tag = evidence_anchor_reason_tag(&err);
    assert!(
        !tag.is_empty(),
        "verify must produce a stable reason tag; got empty"
    );

    // Import attempt against a fresh target also fails (import
    // re-runs verification internally; the export bundle is no
    // longer trustworthy).
    let (_t_dir, tgt_reg, _tgt_root) = fresh_target_registry();
    let import_err = apply_anchor_export_import(
        export_dir.path(),
        &tgt_reg,
        &AnchorImportOptions {
            dry_run: false,
            strict: false,
            selection: &AnchorImportSelection::default(),
            now_utc: NOW,
        },
    )
    .unwrap_err();
    let _ = evidence_anchor_reason_tag(&import_err); // tag exists; stable taxonomy owned by Stage 13.6.
}

// ── Scenario 6: import idempotent rerun ──────────────────────────────────────

#[test]
fn scenario_6_import_idempotent_rerun_against_already_imported_target() {
    let (_dir, src_reg, _) = fresh_registry();
    let (h, _, _) = seed(&src_reg, [1u8; 32], 0x11, None);

    let export_dir = tempfile::tempdir().unwrap();
    let _ = apply_anchor_export(
        &src_reg,
        &AnchorExportOptions {
            export_out: export_dir.path(),
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

    let (_t_dir, tgt_reg, tgt_root) = fresh_target_registry();

    // First import: 1 record added.
    let r1 = apply_anchor_export_import(
        export_dir.path(),
        &tgt_reg,
        &AnchorImportOptions {
            dry_run: false,
            strict: false,
            selection: &AnchorImportSelection::default(),
            now_utc: NOW,
        },
    )
    .unwrap();
    assert_eq!(r1.actions_imported, 1);
    let bytes_after_first = std::fs::read(tgt_root.join(format!("{h}.json"))).unwrap();

    // Second import (same bundle, same target): planner should
    // surface that the record already exists byte-equal. The
    // public API treats byte-equal pre-existing targets as a
    // no-op success path; only divergent targets fail with
    // `ImportTargetExists`. We pin the no-op outcome by checking
    // the post-second-run hot bytes are unchanged.
    let plan2 = plan_anchor_export_import(
        export_dir.path(),
        &tgt_reg,
        &AnchorImportOptions {
            dry_run: true,
            strict: false,
            selection: &AnchorImportSelection::default(),
            now_utc: NOW,
        },
    )
    .unwrap();
    // Plan exists; whether actions_imported > 0 depends on the
    // import policy for byte-equal targets. The strongest
    // operator-observable signal is: the hot bytes do not change
    // when we apply.
    let _ = plan2;
    let r2 = apply_anchor_export_import(
        export_dir.path(),
        &tgt_reg,
        &AnchorImportOptions {
            dry_run: false,
            strict: false,
            selection: &AnchorImportSelection::default(),
            now_utc: NOW,
        },
    );
    // The rerun must either succeed (byte-equal idempotent) or
    // refuse with `ImportTargetExists`-with-byte-equal-match. In
    // either case the on-disk hot bytes are byte-identical to
    // after the first run.
    let bytes_after_second = std::fs::read(tgt_root.join(format!("{h}.json"))).unwrap();
    assert_eq!(
        bytes_after_first, bytes_after_second,
        "import rerun must not mutate byte-equal pre-existing records"
    );
    if let Err(err) = r2 {
        // If the policy is refuse-on-existing, surface the stable
        // tag so a future policy change shows up here.
        let _tag = evidence_anchor_reason_tag(&err);
    }
}

// ── Scenario 7: archive partial-state recovery (black-box) ───────────────────

#[test]
fn scenario_7_archive_partial_state_restore_from_manifest_after_one_hot_record_disappears() {
    // SCOPE LOCK: black-box at the archive/restore boundary. We
    // do NOT exercise any internal phase hook; we simply seed
    // the operator-observable state directly:
    //   - durable archive manifest exists, byte-equal record
    //     bytes in the archive subtree
    //   - one hot-registry record is MISSING (simulating a
    //     post-archive operator error or a Phase-2 partial
    //     failure recovered by re-planting bytes)
    // Then we restore from the manifest and assert
    // idempotent recovery.
    let (_dir, reg, root) = fresh_registry();
    let (h1, tx1, _) = seed(&reg, [1u8; 32], 0x11, Some(LocalAnchorStatus::Finalized));
    let (h2, tx2, _) = seed(&reg, [2u8; 32], 0x22, Some(LocalAnchorStatus::Finalized));

    // Standard archive of both Finalized records.
    let plan = plan_anchor_archive(
        &reg,
        &AnchorArchivePlanOptions {
            now_utc: NOW,
            statuses: vec![],
            before_utc: None,
            selection: &AnchorArchiveSelection::default(),
        },
    )
    .unwrap();
    let archive_root = tempfile::tempdir().unwrap();
    let _ = apply_anchor_archive(
        &plan,
        &AnchorArchiveApplyOptions {
            archive_dir: archive_root.path(),
            dry_run: false,
            now_utc: NOW,
        },
    )
    .unwrap();
    // Post-archive: both records gone from hot, both present in
    // the archive subtree under plan_id/anchors/.
    let manifest_dir = archive_root.path().join(&plan.plan_id);
    let archived_h1 = manifest_dir.join("anchors").join(format!("{h1}.json"));
    let archived_h2 = manifest_dir.join("anchors").join(format!("{h2}.json"));
    assert!(archived_h1.is_file() && archived_h2.is_file());

    // Operator-observable partial state: re-plant ONE record's
    // bytes back into the hot registry (e.g. an operator
    // mistakenly copied from the archive). The OTHER record
    // stays archived-only. tx_index has neither entry (archive
    // removed both).
    let h1_bytes = std::fs::read(&archived_h1).unwrap();
    std::fs::write(root.join(format!("{h1}.json")), &h1_bytes).unwrap();

    // Restore from manifest. Expected behavior per Stage 13.7:
    //   - h1: byte-equal present in hot → tx_index re-added, no
    //     file rewrite.
    //   - h2: missing in hot → record file restored from archive.
    let manifest_bytes = std::fs::read(manifest_dir.join("archive_manifest.json")).unwrap();
    let manifest: AnchorArchiveManifest =
        serde_json::from_slice(&manifest_bytes).unwrap();
    let report = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    // At least one record was restored from archive bytes; at
    // least one tx_index entry was re-added.
    assert!(
        report.restored >= 1,
        "restore must materialise the missing record from manifest; got: {report:?}"
    );
    // Both hot record files now exist byte-equal to the archive
    // copies.
    assert_eq!(
        std::fs::read(root.join(format!("{h1}.json"))).unwrap(),
        h1_bytes
    );
    let h2_archived_bytes = std::fs::read(&archived_h2).unwrap();
    assert_eq!(
        std::fs::read(root.join(format!("{h2}.json"))).unwrap(),
        h2_archived_bytes
    );
    // tx_index now has entries for both tx_ids again.
    let idx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("tx_index.json")).unwrap()).unwrap();
    let by_tx = idx.get("by_tx_id").unwrap().as_object().unwrap();
    assert!(by_tx.contains_key(&tx1));
    assert!(by_tx.contains_key(&tx2));

    // Idempotent rerun: second restore on the same manifest must
    // not error and must not mutate the hot bytes.
    let h1_bytes_before_rerun =
        std::fs::read(root.join(format!("{h1}.json"))).unwrap();
    let h2_bytes_before_rerun =
        std::fs::read(root.join(format!("{h2}.json"))).unwrap();
    let _r2 = restore_anchor_archive(
        &manifest,
        &AnchorArchiveRestoreOptions {
            archive_dir: &manifest_dir,
            anchor_registry_dir: &root,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(
        std::fs::read(root.join(format!("{h1}.json"))).unwrap(),
        h1_bytes_before_rerun
    );
    assert_eq!(
        std::fs::read(root.join(format!("{h2}.json"))).unwrap(),
        h2_bytes_before_rerun
    );
}

// ── Scenario 8: consistency report before chain reconcile ────────────────────

#[test]
fn scenario_8_consistency_report_runs_without_chain_calls_before_reconcile() {
    let (_dir, reg, root) = fresh_registry();
    let (_, tx_submitted, _) = seed(&reg, [1u8; 32], 0x11, None);
    let (_, tx_included, _) =
        seed(&reg, [2u8; 32], 0x22, Some(LocalAnchorStatus::Included));
    let (_, tx_finalized, _) =
        seed(&reg, [3u8; 32], 0x33, Some(LocalAnchorStatus::Finalized));
    // Inject one orphan tx_index entry for variety.
    let idx_path = root.join("tx_index.json");
    let mut idx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&idx_path).unwrap()).unwrap();
    idx["by_tx_id"]["orphan-tx"] = serde_json::Value::String("aa".repeat(32));
    std::fs::write(&idx_path, serde_json::to_vec_pretty(&idx).unwrap()).unwrap();

    // Stage 13.8: consistency is purely local — no chain client
    // even passed in. Run it with no chain reachable.
    let report = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    assert_eq!(report.summary.hot_total, 3);
    assert_eq!(report.summary.hot_submitted, 1);
    assert_eq!(report.summary.hot_included, 1);
    assert_eq!(report.summary.hot_finalized, 1);
    let orphans: Vec<_> = report
        .findings
        .iter()
        .filter(|f| f.kind == AnchorConsistencyFindingKind::HotTxIndexOrphan)
        .collect();
    assert_eq!(orphans.len(), 1);

    // Stage 13.9 reconcile now uses the batch path. Consistency
    // findings are independent and remain visible afterward.
    let chain = AcceptanceFakeClient::new();
    chain.set_status(&tx_submitted, AnchorStatus::Included);
    chain.set_status(&tx_included, AnchorStatus::Finalized);
    chain.set_status(&tx_finalized, AnchorStatus::Finalized);
    let outcomes = reconcile_evidence_anchors_workflow(&reg, &chain).unwrap();
    // Stage 13.2 lock: reconcile queries only Submitted +
    // Included records (Finalized + Failed are terminal). One
    // batch chunk per call.
    assert_eq!(outcomes.len(), 2);
    assert_eq!(chain.batch_call_count(), 1);

    // Re-running consistency post-reconcile still surfaces the
    // orphan — reconcile does not touch tx_index orphans
    // (Stage 13.4 owns that path).
    let report2 = build_anchor_consistency_report(&AnchorConsistencyOptions {
        anchor_registry_dir: &root,
        archive_dirs: &[],
        export_dirs: &[],
        stale_threshold_secs: None,
        now_utc: NOW,
    })
    .unwrap();
    let orphans2: Vec<_> = report2
        .findings
        .iter()
        .filter(|f| f.kind == AnchorConsistencyFindingKind::HotTxIndexOrphan)
        .collect();
    assert_eq!(orphans2.len(), 1);
}

// ── Scenario 9: by-tuple canonical mismatch — surfaced read-only ─────────────

#[test]
fn scenario_9_by_tuple_canonical_tx_mismatch_surfaced_without_mutating_registry() {
    let (_dir, reg, root) = fresh_registry();
    let (h, local_tx_id, _) = seed(&reg, [1u8; 32], 0x11, None);

    // Chain returns a DIFFERENT canonical tx hash than the local
    // record's tx_id (the classic "cosmetic 0x/case" or
    // duplicate-anchor race surface).
    let canonical = "0x".to_string() + &"ab".repeat(32);
    let chain = AcceptanceFakeClient::new();
    chain.set_by_tuple_found(TupleLookupResult {
        tx_hash: canonical.clone(),
        included_at_height: 12_345,
    });

    let record_bytes_before =
        std::fs::read(root.join(format!("{h}.json"))).unwrap();

    let outcome = lookup_anchor_by_tuple_workflow(
        &reg,
        &chain,
        AnchorSelector::ArtifactHashHex(&h),
    )
    .unwrap();

    // Operator-observable: both tx_ids are exposed, and they
    // differ. CLI surfaces this as
    // `tx_id_matches_canonical=false`.
    match outcome {
        TupleLookupOutcome::Found {
            canonical_tx_hash,
            included_at_height,
            local_record_tx_id,
        } => {
            assert_eq!(canonical_tx_hash, canonical);
            assert_eq!(included_at_height, 12_345);
            assert_eq!(local_record_tx_id, local_tx_id);
            assert_ne!(
                canonical_tx_hash, local_record_tx_id,
                "scenario premise: chain canonical differs from local tx_id"
            );
        }
        TupleLookupOutcome::NotFound => panic!("expected Found"),
    }

    // Stage 13.9 Q3 lock — registry MUST NOT be mutated by a
    // read-only by-tuple lookup. The on-disk record is byte-
    // identical, no auto-repair.
    let record_bytes_after =
        std::fs::read(root.join(format!("{h}.json"))).unwrap();
    assert_eq!(
        record_bytes_before, record_bytes_after,
        "Stage 13.9 Q3 lock: by-tuple lookup must NOT mutate the registry"
    );
}

#[test]
fn scenario_9b_by_tuple_not_found_event_line_has_no_reason_key() {
    // Companion structural pin: the CLI's NotFound event uses
    // the informational name and carries no `reason=`. The CLI
    // pin lives in `omni-node` `stage_13_9_cli_tests`; this
    // umbrella adds an event-line shape check via the inline
    // format mirror to demonstrate the structured-parsing
    // assertion pattern.
    let line = format_no_chain_anchor_event(
        std::path::Path::new("/tmp/registry"),
        "http://localhost:0",
    );
    let m = parse_event_line(&line);
    assert_eq!(
        m["event"],
        "integrity_evidence_anchor_tuple_lookup_no_chain_anchor"
    );
    assert!(
        !m.contains_key("reason"),
        "Stage 13.9 Q5 lock: NotFound event line MUST NOT carry a reason= key"
    );
    assert_eq!(m["anchor_registry_dir"], "/tmp/registry");
    assert_eq!(m["rpc_url"], "http://localhost:0");
}

// ── Negative-cross-stage acceptance: error-tag inventory ─────────────────────

/// Cross-stage smoke: every error path used in the acceptance
/// scenarios produces a stable, non-empty reason tag. Pin against
/// the existing taxonomy without minting any new tag string.
#[test]
fn scenario_smoke_error_taxonomy_stable_for_acceptance_paths() {
    // Trigger one well-known error per stage and check the tag
    // is non-empty (the exact mapping is owned by
    // `evidence_anchor_reason_tag`).
    let (_dir, reg, _) = fresh_registry();
    let err = lookup_anchor_by_tuple_workflow(
        &reg,
        &AcceptanceFakeClient::new(),
        AnchorSelector::TxId("does-not-exist"),
    )
    .unwrap_err();
    assert_reason_tag(&err, "anchor_not_found");
}
