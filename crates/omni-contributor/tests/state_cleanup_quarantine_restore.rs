//! Stage 12.18 — integration tests for the cleanup-quarantine
//! restore. Reuses the Stage 12.17 cleanup builder pattern: seed
//! an aggregated session, perturb it, run cleanup, then run
//! restore and assert closed-set behavior.

use std::path::Path;

use omni_contributor::{
    apply_state_cleanup,
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, partial_result_signing_input,
        session_id_hex, work_assignment_signing_input,
    },
    plan_state_cleanup,
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    restore_state_cleanup_quarantine, scan_state_integrity_with_audit_orphans,
    session::{
        AggregatedContributorResult, AggregatedPartialRef, ContributorJoin,
        ExecutionSession, PartialContributorResult, WorkAssignment, WorkKind,
    },
    verify_quarantine_manifest, CleanupApplyOptions, CleanupPlanOptions,
    ContributorSigner, ContributorStateStore, CoordinatorSigner, FindingKind,
    QuarantineRestoreOptions, QuarantineRestoreSource, ScanOptions,
    SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.18-restore-coord-seed-32";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.18-restore-contrib-a-32!";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.18-restore-contrib-b-32!";
const NOW_UTC: &str = "2026-06-07T01:00:00Z";
const FAR_FUTURE: &str = "2026-12-31T23:59:59Z";

fn fresh_dir() -> tempfile::TempDir {
    tempfile::TempDir::new().expect("tempdir")
}

fn open_store(root: &Path) -> ContributorStateStore {
    let (s, _) = ContributorStateStore::open(root, false, NOW_UTC).unwrap();
    s
}

fn build_session(expires_at_utc: &str) -> ExecutionSession {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-06-02T00:00:00Z".into(),
        expires_at_utc: expires_at_utc.into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn build_join(s: &ExecutionSession, c: &ContributorSigner) -> ContributorJoin {
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: s.session_id.clone(),
        contributor_pubkey_hex: c.pubkey_hex(),
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 100,
        max_output_tokens: 100,
        supported_work_unit_kinds: vec![WorkUnitKind::Layers],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-06-02T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = c.sign_hex(&si);
    j
}

fn build_assignment(
    s: &ExecutionSession,
    c: &ContributorSigner,
    coord: &CoordinatorSigner,
    stage_index: u32,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: s.session_id.clone(),
        assignment_id: String::new(),
        stage_index,
        contributor_pubkey_hex: c.pubkey_hex(),
        work_kind: WorkKind::Layers {
            start: stage_index * 8,
            end: stage_index * 8 + 8,
        },
        expected_work_units: 8,
        expected_work_unit_kind: WorkUnitKind::Layers,
        assigned_at_utc: "2026-06-02T00:00:05Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

fn build_partial(
    asn: &WorkAssignment,
    contrib: &ContributorSigner,
    stage_label: &str,
) -> PartialContributorResult {
    let mut p = PartialContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: asn.session_id.clone(),
        assignment_id: asn.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        partial_artifact_snip_root: format!("0x{}", "aa".repeat(32)),
        partial_artifact_hash: hex_lower(blake3::hash(b"artifact").as_bytes()),
        measured_accounting: MeasuredAccounting {
            tokenizer_hash: "44".repeat(32),
            input_token_count: 0,
            output_token_count: 0,
            total_base_units: 8,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: contrib.pubkey_hex(),
                stage_label: stage_label.into(),
                work_unit_kind: WorkUnitKind::Layers,
                work_units: 8,
            }],
        },
        produced_at_utc: "2026-06-02T00:00:20Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = partial_result_signing_input(&p).unwrap();
    p.contributor_signature_hex = contrib.sign_hex(&si);
    p
}

fn build_aggregate(
    session: &ExecutionSession,
    items: &[(&WorkAssignment, &PartialContributorResult)],
    coord: &CoordinatorSigner,
) -> AggregatedContributorResult {
    let mut partial_refs: Vec<AggregatedPartialRef> = items
        .iter()
        .map(|(a, p)| {
            let bytes = canonical_partial_result_bytes(p).unwrap();
            AggregatedPartialRef {
                assignment_id: a.assignment_id.clone(),
                stage_index: a.stage_index,
                contributor_pubkey_hex: p.contributor_pubkey_hex.clone(),
                partial_snip_root: format!("0x{}", "cc".repeat(32)),
                partial_canonical_hash: hex_lower(blake3::hash(&bytes).as_bytes()),
            }
        })
        .collect();
    partial_refs.sort_by_key(|r| r.stage_index);
    let mut g = AggregatedContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        final_result_snip_root: format!("0x{}", "bb".repeat(32)),
        final_result_canonical_hash: hex_lower(blake3::hash(b"final").as_bytes()),
        partial_refs,
        aggregated_at_utc: "2026-06-02T00:00:30Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let si = aggregated_result_signing_input(&g).unwrap();
    g.coordinator_signature_hex = coord.sign_hex(&si);
    g
}

fn seed_aggregated_session(store: &ContributorStateStore) -> ExecutionSession {
    use omni_contributor::{StateNamespace, StateObjectKind};
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib_a = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = build_session(FAR_FUTURE);
    let join_a = build_join(&session, &contrib_a);
    let join_b = build_join(&session, &contrib_b);
    let asn_0 = build_assignment(&session, &contrib_a, &coord, 0);
    let asn_1 = build_assignment(&session, &contrib_b, &coord, 1);
    let p_0 = build_partial(&asn_0, &contrib_a, "stage-0");
    let p_1 = build_partial(&asn_1, &contrib_b, "stage-1");
    let agg = build_aggregate(&session, &[(&asn_0, &p_0), (&asn_1, &p_1)], &coord);

    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .mark_seen(StateNamespace::Sessions, &session.session_id)
        .unwrap();
    for j in [&join_a, &join_b] {
        store
            .write_verified_json(
                StateObjectKind::Join {
                    session_id: session.session_id.clone(),
                },
                &j.contributor_pubkey_hex,
                j,
            )
            .unwrap();
        let marker = format!("{}--{}", session.session_id, j.contributor_pubkey_hex);
        store.mark_seen(StateNamespace::Joins, &marker).unwrap();
    }
    for a in [&asn_0, &asn_1] {
        store
            .write_verified_json(
                StateObjectKind::Assignment {
                    session_id: session.session_id.clone(),
                },
                &a.assignment_id,
                a,
            )
            .unwrap();
        let marker = format!("{}--{}", session.session_id, a.assignment_id);
        store
            .mark_seen(StateNamespace::Assignments, &marker)
            .unwrap();
    }
    for p in [&p_0, &p_1] {
        store
            .write_verified_json(
                StateObjectKind::Partial {
                    session_id: session.session_id.clone(),
                },
                &p.assignment_id,
                p,
            )
            .unwrap();
        let marker = format!("{}--{}", session.session_id, p.assignment_id);
        store.mark_seen(StateNamespace::Partials, &marker).unwrap();
    }
    store
        .write_verified_json(StateObjectKind::Aggregate, &session.session_id, &agg)
        .unwrap();
    store
        .mark_seen(StateNamespace::Aggregates, &session.session_id)
        .unwrap();
    session
}

fn default_scan<'a>(now_utc: &'a str) -> ScanOptions<'a> {
    ScanOptions {
        session_id_filter: None,
        archive_dir: None,
        now_utc,
    }
}

fn plan_opts<'a>(now_utc: &'a str) -> CleanupPlanOptions<'a> {
    CleanupPlanOptions {
        now_utc,
        session_id_filter: None,
    }
}

fn apply_opts<'a>(
    quarantine_dir: &'a Path,
    now_utc: &'a str,
) -> CleanupApplyOptions<'a> {
    CleanupApplyOptions {
        quarantine_dir,
        dry_run: false,
        allow_invalid_partial_cleanup: false,
        allow_orphan_assignments: false,
        purge_stray: false,
        now_utc,
    }
}

fn restore_opts<'a>(
    source: QuarantineRestoreSource<'a>,
    now_utc: &'a str,
) -> QuarantineRestoreOptions<'a> {
    QuarantineRestoreOptions {
        source,
        dry_run: false,
        verify_only: false,
        overwrite_existing: false,
        restore_seen_markers: true,
        allow_restore_orphan_assignments: false,
        now_utc,
    }
}

fn tamper_signature(p: &Path) {
    let mut s = std::fs::read_to_string(p).unwrap();
    let keys = ["coordinator_signature_hex", "contributor_signature_hex"];
    for key in keys {
        let needle = format!("\"{key}\": \"");
        if let Some(idx) = s.find(&needle) {
            let pos = idx + needle.len();
            let bytes = unsafe { s.as_bytes_mut() };
            let old = bytes[pos];
            let new = match old {
                b'0' => b'1',
                b'1'..=b'8' => old + 1,
                b'9' => b'a',
                b'a'..=b'e' => old + 1,
                b'f' => b'0',
                _ => continue,
            };
            bytes[pos] = new;
            std::fs::write(p, s.as_bytes()).unwrap();
            return;
        }
    }
    panic!("no signature_hex field found in {p:?}");
}

/// Drive a clean cleanup that quarantines a tampered join.
/// Returns `(state-dir tempdir, quarantine tempdir, plan_id,
/// pubkey_stem, session)` for the caller to drive a restore.
fn cleanup_tampered_join(
) -> (tempfile::TempDir, tempfile::TempDir, String, String, ExecutionSession) {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let joins_dir = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins");
    let join_path = std::fs::read_dir(&joins_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let pubkey_stem = join_path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    tamper_signature(&join_path);

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let quarantine = fresh_dir();
    let mut opts = apply_opts(quarantine.path(), NOW_UTC);
    // Tampering a join cascades — also clears the dependent
    // partial. Pass the gate so the full plan applies cleanly.
    opts.allow_invalid_partial_cleanup = true;
    let _ = apply_state_cleanup(&store, &plan, &opts).unwrap();
    (state_dir, quarantine, plan.plan_id, pubkey_stem, session)
}

// ── 1. verify_quarantine_manifest happy path ──────────────────

#[test]
fn verify_quarantine_manifest_parses_v1_manifest() {
    let (_state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let plan_dir = quarantine.path().join(&plan_id);
    let manifest =
        verify_quarantine_manifest(&QuarantineRestoreSource::PlanDir(&plan_dir))
            .unwrap();
    assert_eq!(manifest.plan_id, plan_id);
    assert_eq!(manifest.schema_version, 1);
    assert!(!manifest.files.is_empty());
}

// ── 2. End-to-end restore: tampered join round-trip ───────────

#[test]
fn tampered_join_cleanup_then_restore_round_trips() {
    use omni_contributor::StateNamespace;
    let (state_dir, quarantine, plan_id, pubkey_stem, session) =
        cleanup_tampered_join();
    let plan_dir = quarantine.path().join(&plan_id);

    // Reopen store post-cleanup. Restore should reintroduce
    // the tampered join body + matching seen marker.
    let store = open_store(state_dir.path());
    let report = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(QuarantineRestoreSource::PlanDir(&plan_dir), NOW_UTC),
    )
    .unwrap();
    assert_eq!(report.mode, "restore");
    assert!(report.files_restored >= 1);
    assert!(report.seen_markers_restored >= 1);

    // Body is back on disk.
    let join_path = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join(format!("{pubkey_stem}.json"));
    assert!(join_path.is_file());
    // Seen marker is back.
    let marker_key = format!("{}--{}", session.session_id, pubkey_stem);
    assert!(store
        .is_seen(StateNamespace::Joins, &marker_key)
        .unwrap());
}

// ── 3. Verify-only on intact quarantine ───────────────────────

#[test]
fn verify_only_intact_quarantine_writes_nothing() {
    let (state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let plan_dir = quarantine.path().join(&plan_id);
    let store = open_store(state_dir.path());
    let mut opts = restore_opts(QuarantineRestoreSource::PlanDir(&plan_dir), NOW_UTC);
    opts.verify_only = true;
    let report = restore_state_cleanup_quarantine(&store, &opts).unwrap();
    assert_eq!(report.mode, "verify_only");
    assert_eq!(report.files_restored, 0);
    assert!(report
        .outcomes
        .iter()
        .all(|o| o.status == "verify_only"));
}

// ── 4. Dry-run writes nothing ─────────────────────────────────

#[test]
fn dry_run_writes_nothing_and_status_would_apply() {
    let (state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let plan_dir = quarantine.path().join(&plan_id);
    let store = open_store(state_dir.path());
    let mut opts = restore_opts(QuarantineRestoreSource::PlanDir(&plan_dir), NOW_UTC);
    opts.dry_run = true;
    let report = restore_state_cleanup_quarantine(&store, &opts).unwrap();
    assert_eq!(report.mode, "dry_run");
    assert_eq!(report.files_restored, 0);
    assert!(report
        .outcomes
        .iter()
        .all(|o| o.status == "would_apply"));
}

// ── 5. Verify-only wins when both flags set ───────────────────

#[test]
fn verify_only_wins_over_dry_run() {
    let (state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let plan_dir = quarantine.path().join(&plan_id);
    let store = open_store(state_dir.path());
    let mut opts = restore_opts(QuarantineRestoreSource::PlanDir(&plan_dir), NOW_UTC);
    opts.dry_run = true;
    opts.verify_only = true;
    let report = restore_state_cleanup_quarantine(&store, &opts).unwrap();
    assert_eq!(report.mode, "verify_only");
}

// ── 6. BLAKE3 mismatch refusal ────────────────────────────────

#[test]
fn corrupt_quarantine_byte_emits_blake_mismatch() {
    let (state_dir, quarantine, plan_id, pubkey_stem, session) =
        cleanup_tampered_join();
    // Corrupt one byte in the quarantine copy of the join body.
    let q_body = quarantine
        .path()
        .join(&plan_id)
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join(format!("{pubkey_stem}.json"));
    let mut bytes = std::fs::read(&q_body).unwrap();
    if let Some(b) = bytes.last_mut() {
        *b = b.wrapping_add(1);
    }
    std::fs::write(&q_body, &bytes).unwrap();

    let store = open_store(state_dir.path());
    let plan_dir = quarantine.path().join(&plan_id);
    let mut opts = restore_opts(
        QuarantineRestoreSource::PlanDir(&plan_dir),
        NOW_UTC,
    );
    opts.verify_only = true;
    let err = restore_state_cleanup_quarantine(&store, &opts).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.starts_with("blake3 mismatch"),
        "expected BlakeMismatch, got: {msg}"
    );
}

// ── 7. Path-traversal refusal in manifest ─────────────────────

#[test]
fn manifest_with_traversal_path_is_refused_before_io() {
    let (state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let manifest_path = quarantine
        .path()
        .join(&plan_id)
        .join("quarantine-manifest.json");
    // Hand-edit one entry's source_relative to a traversal
    // payload. Manifest parses; v1 doesn't carry a self-hash
    // so no rebuild is needed.
    let s = std::fs::read_to_string(&manifest_path).unwrap();
    let tampered = s.replacen(
        "verified/sessions/",
        "../../etc/passwd/sessions/",
        1,
    );
    std::fs::write(&manifest_path, &tampered).unwrap();

    let store = open_store(state_dir.path());
    let err = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(
            QuarantineRestoreSource::PlanDir(&quarantine.path().join(&plan_id)),
            NOW_UTC,
        ),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsafe relative path"),
        "expected UnsafeRelativePath, got: {msg}"
    );
}

// ── 8. Schema-version mismatch refusal ────────────────────────

#[test]
fn schema_version_mismatch_is_refused() {
    let (state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let manifest_path = quarantine
        .path()
        .join(&plan_id)
        .join("quarantine-manifest.json");
    let s = std::fs::read_to_string(&manifest_path).unwrap();
    let tampered = s.replacen("\"schema_version\": 1", "\"schema_version\": 2", 1);
    std::fs::write(&manifest_path, &tampered).unwrap();

    let store = open_store(state_dir.path());
    let err = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(
            QuarantineRestoreSource::PlanDir(&quarantine.path().join(&plan_id)),
            NOW_UTC,
        ),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported quarantine manifest schema_version"),
        "expected UnsupportedManifestVersion, got: {msg}"
    );
}

// ── 9. plan_id mismatch refusal ───────────────────────────────

#[test]
fn plan_id_mismatch_in_paired_source_is_refused() {
    let (state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let store = open_store(state_dir.path());
    let wrong_plan_id = "f".repeat(16);
    assert_ne!(wrong_plan_id, plan_id);
    // Use the paired source form with a wrong plan_id. The
    // path resolves to a non-existent dir, so we expect
    // QuarantineDirNotFound rather than a mismatch — but if we
    // point to a real dir with a wrong supplied plan_id we
    // exercise PlanIdMismatch:
    let plan_dir = quarantine.path().join(&plan_id);
    // Use PlanDir with a SYMLINK-style rename. Simulate by
    // copying the manifest into a different named dir.
    let renamed = quarantine.path().join("renamed_plan_id_dir");
    let _ = std::fs::create_dir_all(&renamed);
    // Copy manifest only — enough to exercise the plan-id
    // pinning check at Phase A.
    std::fs::copy(
        plan_dir.join("quarantine-manifest.json"),
        renamed.join("quarantine-manifest.json"),
    )
    .unwrap();
    let err = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(QuarantineRestoreSource::PlanDir(&renamed), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("plan_id mismatch"),
        "expected PlanIdMismatch, got: {msg}"
    );
}

// ── 10. Destination-exists preflight refusal ──────────────────

#[test]
fn destination_exists_refuses_without_overwrite() {
    let (state_dir, quarantine, plan_id, pubkey_stem, session) =
        cleanup_tampered_join();
    // Manually drop a sentinel file at the destination so the
    // preflight fires.
    let dest = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join(format!("{pubkey_stem}.json"));
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    std::fs::write(&dest, b"sentinel").unwrap();

    let store = open_store(state_dir.path());
    let err = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(
            QuarantineRestoreSource::PlanDir(&quarantine.path().join(&plan_id)),
            NOW_UTC,
        ),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("destination already exists"),
        "expected DestinationExists, got: {msg}"
    );
    // Sentinel still on disk: refusal preserved state-dir.
    assert_eq!(std::fs::read(&dest).unwrap(), b"sentinel");
}

// ── 11. --overwrite-existing accepts ──────────────────────────

#[test]
fn overwrite_existing_replaces_destination() {
    let (state_dir, quarantine, plan_id, pubkey_stem, session) =
        cleanup_tampered_join();
    let dest = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join(format!("{pubkey_stem}.json"));
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    std::fs::write(&dest, b"old").unwrap();

    let store = open_store(state_dir.path());
    let plan_dir = quarantine.path().join(&plan_id);
    let mut opts = restore_opts(
        QuarantineRestoreSource::PlanDir(&plan_dir),
        NOW_UTC,
    );
    opts.overwrite_existing = true;
    // The tampered partial that was also quarantined will
    // collide too; allow overwrite covers both. We don't need
    // to gate orphan assignments — this fixture has none.
    let report = restore_state_cleanup_quarantine(&store, &opts).unwrap();
    assert!(report.files_restored >= 1);
    let post = std::fs::read(&dest).unwrap();
    assert_ne!(post, b"old", "overwrite_existing must replace bytes");
}

// ── 12. --no-restore-seen-markers skips markers ───────────────

#[test]
fn no_restore_seen_markers_keeps_body_only() {
    use omni_contributor::StateNamespace;
    let (state_dir, quarantine, plan_id, pubkey_stem, session) =
        cleanup_tampered_join();
    let store = open_store(state_dir.path());
    let plan_dir = quarantine.path().join(&plan_id);
    let mut opts = restore_opts(
        QuarantineRestoreSource::PlanDir(&plan_dir),
        NOW_UTC,
    );
    opts.restore_seen_markers = false;
    let report = restore_state_cleanup_quarantine(&store, &opts).unwrap();
    assert!(report.files_restored >= 1);
    assert_eq!(
        report.seen_markers_restored, 0,
        "no-restore-seen-markers must zero markers"
    );
    // Body is back …
    let body = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join(format!("{pubkey_stem}.json"));
    assert!(body.is_file());
    // … but seen marker is NOT.
    let marker_key = format!("{}--{}", session.session_id, pubkey_stem);
    assert!(!store
        .is_seen(StateNamespace::Joins, &marker_key)
        .unwrap());
}

// ── 13. Unknown source_finding_kind refusal ───────────────────

#[test]
fn unknown_source_finding_kind_is_refused() {
    let (state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let manifest_path = quarantine
        .path()
        .join(&plan_id)
        .join("quarantine-manifest.json");
    let s = std::fs::read_to_string(&manifest_path).unwrap();
    // Inject a future-stage tag into the first entry's
    // source_finding_kind.
    let tampered = s.replacen(
        "\"invalid_join\"",
        "\"future_stage_thing\"",
        1,
    );
    std::fs::write(&manifest_path, &tampered).unwrap();

    let store = open_store(state_dir.path());
    let plan_dir = quarantine.path().join(&plan_id);
    let err = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(QuarantineRestoreSource::PlanDir(&plan_dir), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown source_finding_kind"),
        "expected UnknownFindingKind, got: {msg}"
    );
}

// ── 14. Orphan-assignment gate refusal + accept ───────────────
//
// We can't easily fixture a real orphan-assignment cleanup
// without a Phase B partial-apply setup; instead, simulate by
// hand-editing the manifest to retag one entry as
// `orphan_replacement_assignments`. The applier's Phase A still
// refuses without the gate.

#[test]
fn orphan_assignment_entry_requires_allow_flag() {
    let (state_dir, quarantine, plan_id, _, _) = cleanup_tampered_join();
    let manifest_path = quarantine
        .path()
        .join(&plan_id)
        .join("quarantine-manifest.json");
    let s = std::fs::read_to_string(&manifest_path).unwrap();
    let tampered = s.replacen(
        "\"invalid_join\"",
        "\"orphan_replacement_assignments\"",
        1,
    );
    std::fs::write(&manifest_path, &tampered).unwrap();

    let store = open_store(state_dir.path());
    let plan_dir = quarantine.path().join(&plan_id);
    let err = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(QuarantineRestoreSource::PlanDir(&plan_dir), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("gated restore requires --allow-restore-orphan-assignments"),
        "expected GatedRestoreRequired, got: {msg}"
    );

    // With the flag, restore proceeds (modulo unrelated
    // destination existence — tampered fixture leaves the
    // destination empty, so write succeeds).
    let mut opts = restore_opts(
        QuarantineRestoreSource::PlanDir(&plan_dir),
        NOW_UTC,
    );
    opts.allow_restore_orphan_assignments = true;
    let report = restore_state_cleanup_quarantine(&store, &opts).unwrap();
    assert!(report.files_restored >= 1);
}

// ── 15. Hash-tight rollback ───────────────────────────────────
//
// With seen-marker restore ON and no tier-A actions in the
// plan, the post-restore `source_integrity_hash` must exactly
// equal the pre-cleanup `source_integrity_hash`. Proves the
// restore is a bit-exact undo of the cleanup.

#[test]
fn rollback_restores_pre_cleanup_source_integrity_hash() {
    use omni_contributor::source_integrity_hash_hex;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let joins_dir = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins");
    let join_path = std::fs::read_dir(&joins_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    tamper_signature(&join_path);

    // Snapshot pre-cleanup integrity hash.
    let (pre_report, _) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let pre_hash = source_integrity_hash_hex(&pre_report);

    // Plan + apply cleanup.
    let (report_for_plan, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan =
        plan_state_cleanup(&report_for_plan, &orphans, &plan_opts(NOW_UTC)).unwrap();
    // No tier-A actions in this plan (only tier-B), so the
    // hash-tight invariant holds.
    assert!(plan
        .actions
        .iter()
        .all(|a| matches!(
            a.kind,
            omni_contributor::CleanupActionKind::QuarantineVerifiedFile
                | omni_contributor::CleanupActionKind::QuarantineAndUnmarkJoin
                | omni_contributor::CleanupActionKind::QuarantineAndUnmarkAssignment
                | omni_contributor::CleanupActionKind::QuarantineAndUnmarkPartial
                | omni_contributor::CleanupActionKind::QuarantineAndUnmarkSupersession
                | omni_contributor::CleanupActionKind::QuarantineAndUnmarkOrphanAssignment
        )), "{:?}", plan.actions.iter().map(|a| a.kind).collect::<Vec<_>>());
    let quarantine = fresh_dir();
    let mut apply_options = apply_opts(quarantine.path(), NOW_UTC);
    apply_options.allow_invalid_partial_cleanup = true;
    let _ = apply_state_cleanup(&store, &plan, &apply_options).unwrap();

    // Restore.
    let plan_dir = quarantine.path().join(&plan.plan_id);
    let report = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(QuarantineRestoreSource::PlanDir(&plan_dir), NOW_UTC),
    )
    .unwrap();
    assert!(report.files_restored >= 1);

    // Post-restore integrity hash must match pre-cleanup
    // exactly.
    let (post_report, _) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let post_hash = source_integrity_hash_hex(&post_report);
    assert_eq!(
        post_hash, pre_hash,
        "hash-tight rollback: post-restore source_integrity_hash must equal pre-cleanup"
    );

    // Sanity: the post-restore scan re-detects the original
    // InvalidJoin finding (because cleanup → restore undid
    // itself).
    assert!(post_report
        .findings
        .iter()
        .any(|f| matches!(f.kind, FindingKind::InvalidJoin)));
}

// ── 16. Stage 12.18 review fix — marker preflight invariant ──
//
// When `restore_seen_markers == true` (the default), the
// applier preflights every seen-marker destination BEFORE any
// body write. A directory planted at the marker path refuses
// the whole restore with `SeenMarkerPathBlocked`, and the body
// destinations stay untouched. Proves the all-or-nothing
// invariant survives marker hazards.

fn count_files(dir: &Path) -> usize {
    fn rec(d: &Path, n: &mut usize) {
        if let Ok(entries) = std::fs::read_dir(d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    rec(&p, n);
                } else if p.is_file() {
                    *n += 1;
                }
            }
        }
    }
    let mut n = 0;
    rec(dir, &mut n);
    n
}

#[test]
fn marker_path_blocked_refuses_before_any_body_write() {
    let (state_dir, quarantine, plan_id, pubkey_stem, session) =
        cleanup_tampered_join();

    // Plant a DIRECTORY at the seen-marker path the restore
    // would write. Cleanup removed the marker; the restore
    // tries to recreate it; the directory blocks the recreate.
    let marker_key = format!("{}--{}", session.session_id, pubkey_stem);
    let marker_path = state_dir
        .path()
        .join("seen")
        .join("joins")
        .join(&marker_key);
    std::fs::create_dir_all(&marker_path).unwrap();
    assert!(
        marker_path.is_dir(),
        "fixture: marker path must be a directory blocker"
    );

    // Sanity: the body destination is currently empty (the
    // cleanup removed it), so any post-error file presence
    // would prove a partial mutation.
    let body_path = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join(format!("{pubkey_stem}.json"));
    assert!(
        !body_path.is_file(),
        "fixture: tampered join body must have been removed by cleanup"
    );
    let files_before = count_files(state_dir.path());

    let store = open_store(state_dir.path());
    let plan_dir = quarantine.path().join(&plan_id);
    let err = restore_state_cleanup_quarantine(
        &store,
        &restore_opts(QuarantineRestoreSource::PlanDir(&plan_dir), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("seen marker path blocked"),
        "expected SeenMarkerPathBlocked, got: {msg}"
    );

    // The all-or-nothing invariant: no body restored, no
    // state-dir mutation of any kind.
    assert!(
        !body_path.is_file(),
        "body must NOT be restored when marker preflight fails"
    );
    let files_after = count_files(state_dir.path());
    assert_eq!(
        files_before, files_after,
        "state-dir file count must be unchanged after marker preflight failure"
    );

    // --no-restore-seen-markers escapes the marker preflight
    // (because no marker write is planned). The body restore
    // then succeeds even with the marker dir in place.
    let mut opts = restore_opts(
        QuarantineRestoreSource::PlanDir(&plan_dir),
        NOW_UTC,
    );
    opts.restore_seen_markers = false;
    let report = restore_state_cleanup_quarantine(&store, &opts).unwrap();
    assert!(report.files_restored >= 1);
    assert!(body_path.is_file());
}
