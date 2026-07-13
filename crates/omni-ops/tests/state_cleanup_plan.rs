//! Stage 12.17 — integration tests for the local state-dir
//! cleanup planner / applier. Each test seeds a state-dir via
//! the same builder pattern as `state_integrity_scan.rs`,
//! introduces one perturbation, plans, applies, and asserts
//! both the closed-set actions and the post-apply integrity
//! state.

use std::path::Path;

use omni_contributor::{
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower,
        partial_result_signing_input, session_id_hex,
        work_assignment_signing_input,
    },
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    session::{
        AggregatedContributorResult, AggregatedPartialRef, ContributorJoin,
        ExecutionSession, PartialContributorResult, WorkAssignment, WorkKind,
    },
    ContributorSigner, ContributorStateStore, CoordinatorSigner,
    SESSION_SCHEMA_VERSION,
};
use omni_ops::{
    apply_state_cleanup, archive_session, cleanup_plan_hash_hex,
    plan_state_cleanup, scan_state_integrity,
    scan_state_integrity_with_audit_orphans, source_integrity_hash_hex,
    ArchiveMode, ArchiveOptions, ArchiveStatusRequirement, CleanupActionKind,
    CleanupApplyOptions, CleanupPlanOptions, FindingKind, ScanOptions,
};

const COORD_SEED: [u8; 32] = *b"stage12.17-cleanup-coord-seed-32";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.17-cleanup-contrib-a-32!";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.17-cleanup-contrib-b-32!";
const NOW_UTC: &str = "2026-06-06T01:00:00Z";
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

// ── 1. Clean state-dir → empty plan ───────────────────────────

#[test]
fn clean_state_produces_empty_plan() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let _session = seed_aggregated_session(&store);
    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    assert!(plan.actions.is_empty(), "{:?}", plan.actions);
    assert_eq!(plan.schema_version, 1);
}

// ── 2. Plan hash + drift refusal ──────────────────────────────

#[test]
fn plan_hash_is_self_consistent_and_drift_aware() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let join_file = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins");
    let join_path = std::fs::read_dir(&join_file)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    tamper_signature(&join_path);

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    assert_eq!(plan.cleanup_plan_hash, cleanup_plan_hash_hex(&plan));

    let supplied_hash = source_integrity_hash_hex(&report);
    assert_eq!(plan.source_integrity_hash, supplied_hash);
}

// ── 3. Tier A round-trip: stale seen marker ───────────────────

#[test]
fn stale_seen_marker_is_removed_by_cleanup_apply() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    // Plant a stale marker.
    let phantom = "ee".repeat(32);
    store
        .mark_seen(StateNamespace::Sessions, &phantom)
        .unwrap();
    let marker_path = state_dir
        .path()
        .join("seen")
        .join("sessions")
        .join(&phantom);
    assert!(marker_path.is_file());

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    assert!(report
        .findings
        .iter()
        .any(|f| matches!(f.kind, FindingKind::StaleSeenMarker)));
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    assert!(plan
        .actions
        .iter()
        .any(|a| a.kind == CleanupActionKind::RemoveSeenMarker));

    let quarantine = fresh_dir();
    let apply_report =
        apply_state_cleanup(&store, &plan, &apply_opts(quarantine.path(), NOW_UTC))
            .unwrap();
    assert_eq!(apply_report.actions_applied, plan.actions.len() as u32);
    assert!(!marker_path.is_file(), "marker must be gone after apply");

    // Re-scan: no StaleSeenMarker any more.
    let post = scan_state_integrity(&store, &default_scan(NOW_UTC)).unwrap();
    assert!(!post
        .findings
        .iter()
        .any(|f| matches!(f.kind, FindingKind::StaleSeenMarker)));
}

// ── 4. Tier A round-trip: missing seen marker ─────────────────

#[test]
fn missing_seen_marker_is_written_by_cleanup_apply() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    // Delete the aggregates marker; body is still present.
    let marker = state_dir
        .path()
        .join("seen")
        .join("aggregates")
        .join(&session.session_id);
    std::fs::remove_file(&marker).unwrap();

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let quarantine = fresh_dir();
    let apply_report =
        apply_state_cleanup(&store, &plan, &apply_opts(quarantine.path(), NOW_UTC))
            .unwrap();
    assert!(apply_report.actions_applied >= 1);
    assert!(store
        .is_seen(StateNamespace::Aggregates, &session.session_id)
        .unwrap());
}

// ── 5. Tier B round-trip: tampered join ───────────────────────

#[test]
fn tampered_join_is_quarantined_and_unmarked() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let joins_dir = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins");
    let join_file = std::fs::read_dir(&joins_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let join_bytes = std::fs::read(&join_file).unwrap();
    tamper_signature(&join_file);

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let quarantine = fresh_dir();
    // Tampering a join cascades: the assignment loses its
    // joined-pubkey reference (InvalidAssignment) and the
    // dependent partial becomes orphaned (InvalidPartial). The
    // InvalidPartial cleanup is gated, so flip the gate flag
    // for the round-trip — same operator posture as in the CLI.
    let mut opts = apply_opts(quarantine.path(), NOW_UTC);
    opts.allow_invalid_partial_cleanup = true;
    let apply_report = apply_state_cleanup(&store, &plan, &opts).unwrap();
    assert!(apply_report.actions_applied >= 1);

    // Source removed.
    assert!(!join_file.is_file());
    // Quarantine has the original tampered bytes.
    let q_root = quarantine.path().join(&plan.plan_id);
    let pubkey_stem = join_file.file_stem().unwrap().to_string_lossy();
    let q_body = q_root
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join(format!("{pubkey_stem}.json"));
    assert!(q_body.is_file(), "quarantine copy missing: {q_body:?}");
    let q_bytes = std::fs::read(&q_body).unwrap();
    assert_ne!(q_bytes, join_bytes, "tampered copy lands in quarantine");
    let manifest_path = q_root.join("quarantine-manifest.json");
    assert!(manifest_path.is_file());
    // Seen marker gone.
    let pubkey = pubkey_stem.into_owned();
    let marker_key = format!("{}--{}", session.session_id, pubkey);
    assert!(!store
        .is_seen(StateNamespace::Joins, &marker_key)
        .unwrap());

    // Re-scan no longer reports the InvalidJoin.
    let post = scan_state_integrity(&store, &default_scan(NOW_UTC)).unwrap();
    assert!(!post
        .findings
        .iter()
        .any(|f| matches!(f.kind, FindingKind::InvalidJoin)));
}

// ── 6. Source-integrity drift refusal ─────────────────────────

#[test]
fn apply_refuses_on_source_integrity_drift() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let phantom = "11".repeat(32);
    store.mark_seen(StateNamespace::Sessions, &phantom).unwrap();

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();

    // Mutate state between plan and apply: add a second stale marker.
    let phantom2 = "22".repeat(32);
    store.mark_seen(StateNamespace::Sessions, &phantom2).unwrap();
    let _ = session; // silence warning

    let quarantine = fresh_dir();
    let err = apply_state_cleanup(
        &store,
        &plan,
        &apply_opts(quarantine.path(), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("source integrity drift"),
        "expected drift refusal, got: {msg}"
    );
}

// ── 7. Plan hash mismatch refusal ─────────────────────────────

#[test]
fn apply_refuses_on_plan_hash_mismatch() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    store.mark_seen(StateNamespace::Sessions, &"aa".repeat(32)).unwrap();
    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let mut plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    // Tamper a field that's covered by the plan hash.
    plan.created_at_utc = "2099-01-01T00:00:00Z".to_string();

    let quarantine = fresh_dir();
    let err = apply_state_cleanup(
        &store,
        &plan,
        &apply_opts(quarantine.path(), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("plan hash mismatch"),
        "expected plan-hash refusal, got: {msg}"
    );
}

// ── 8. Gated InvalidPartial action refused without flag ───────

#[test]
fn invalid_partial_cleanup_is_gated() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let partials = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("partials");
    let p_file = std::fs::read_dir(&partials)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    tamper_signature(&p_file);

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    assert!(plan
        .actions
        .iter()
        .any(|a| a.kind == CleanupActionKind::QuarantineAndUnmarkPartial));

    let quarantine = fresh_dir();
    let err = apply_state_cleanup(
        &store,
        &plan,
        &apply_opts(quarantine.path(), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("--allow-invalid-partial-cleanup"),
        "expected gate refusal, got: {msg}"
    );

    // With the flag, apply succeeds.
    let mut opts = apply_opts(quarantine.path(), NOW_UTC);
    opts.allow_invalid_partial_cleanup = true;
    let report = apply_state_cleanup(&store, &plan, &opts).unwrap();
    assert!(report.actions_applied >= 1);
}

// ── 9. Dry-run writes nothing ─────────────────────────────────

#[test]
fn dry_run_writes_nothing() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    store.mark_seen(StateNamespace::Sessions, &"bb".repeat(32)).unwrap();
    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let quarantine = fresh_dir();

    let files_state_before = count_files(state_dir.path());
    let files_q_before = count_files(quarantine.path());
    let mut opts = apply_opts(quarantine.path(), NOW_UTC);
    opts.dry_run = true;
    let report = apply_state_cleanup(&store, &plan, &opts).unwrap();
    let files_state_after = count_files(state_dir.path());
    let files_q_after = count_files(quarantine.path());

    assert_eq!(report.mode, "dry_run");
    assert_eq!(report.actions_dry_run, plan.actions.len() as u32);
    assert_eq!(files_state_before, files_state_after, "state-dir untouched");
    assert_eq!(files_q_before, files_q_after, "quarantine untouched");
}

// ── 10. Quarantine collision refusal ──────────────────────────

#[test]
fn apply_refuses_on_existing_quarantine_dir() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    store.mark_seen(StateNamespace::Sessions, &"cc".repeat(32)).unwrap();
    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let quarantine = fresh_dir();
    // Pre-create the plan-id subdir.
    std::fs::create_dir_all(quarantine.path().join(&plan.plan_id)).unwrap();
    let err = apply_state_cleanup(
        &store,
        &plan,
        &apply_opts(quarantine.path(), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("quarantine destination already exists"),
        "expected collision refusal, got: {msg}"
    );
}

// ── 11. Stray verified file → quarantine + remove ─────────────

#[test]
fn stray_verified_file_quarantine_round_trip() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let junk = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join("not-hex-suffix.txt");
    std::fs::write(&junk, b"garbage").unwrap();

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    assert!(plan
        .actions
        .iter()
        .any(|a| a.kind == CleanupActionKind::QuarantineVerifiedFile));
    let quarantine = fresh_dir();
    let report = apply_state_cleanup(
        &store,
        &plan,
        &apply_opts(quarantine.path(), NOW_UTC),
    )
    .unwrap();
    assert!(report.actions_applied >= 1);
    assert!(!junk.is_file());
}

// ── 12. Plan ⇄ JSON round-trip ───────────────────────────────

#[test]
fn plan_json_roundtrip_preserves_hash() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    store.mark_seen(StateNamespace::Sessions, &"dd".repeat(32)).unwrap();
    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let s = serde_json::to_string(&plan).unwrap();
    let round: omni_ops::StateCleanupPlan =
        serde_json::from_str(&s).unwrap();
    assert_eq!(round.cleanup_plan_hash, plan.cleanup_plan_hash);
    assert_eq!(round.source_integrity_hash, plan.source_integrity_hash);
    assert_eq!(round.plan_id, plan.plan_id);
    assert_eq!(round.actions.len(), plan.actions.len());
}

// ── 13. Out-of-scope findings produce no actions ──────────────

#[test]
fn invalid_session_finding_produces_no_action() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let session_body = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("session.json");
    tamper_signature(&session_body);

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    assert!(report
        .findings
        .iter()
        .any(|f| matches!(f.kind, FindingKind::InvalidSession)));
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    // InvalidSession is out-of-scope for v1 → no matching action.
    assert!(!plan
        .actions
        .iter()
        .any(|a| a.source_finding_kind == "invalid_session"));
}

// ── 14. Archive integration: tier-B cleanup composes with
//       Stage 12.14 archive workflow (sanity that the
//       quarantine subtree doesn't collide with archive layout).

#[test]
fn cleanup_quarantine_does_not_collide_with_archive_layout() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    // Sanity: archive the clean session somewhere unrelated.
    let archive_dir = fresh_dir();
    archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_dir.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();

    // Plant a stray, plan, apply.
    let junk = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("assignments")
        .join("not-hex-suffix.bak");
    std::fs::write(&junk, b"x").unwrap();
    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let quarantine = fresh_dir();
    let _ = apply_state_cleanup(
        &store,
        &plan,
        &apply_opts(quarantine.path(), NOW_UTC),
    )
    .unwrap();

    // The archive subtree at `<archive>/<sid>/...` must not be
    // affected by the quarantine writes.
    assert!(archive_dir
        .path()
        .join(&session.session_id)
        .join("manifest.json")
        .is_file());
}

// ── 15. Stage 12.17 review fix — path-traversal regressions ──
//
// `cleanup_plan_hash` proves the plan body wasn't tampered
// since the planner stamped it, but anyone can hand-write a
// plan AND re-stamp the hash. The applier must validate every
// path it consumes BEFORE any FS interaction. Each test below
// recomputes the hash on the tampered body so the
// `PlanHashMismatch` check passes — the only thing that should
// catch the attack is `UnsafePlanPath`.

#[test]
fn apply_refuses_traversal_path_in_remove_seen_file() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let _store2 = open_store(state_dir.path());
    // Plant a stray seen file so plan_state_cleanup emits a
    // `RemoveSeenFile` action.
    let junk = state_dir.path().join("seen").join("garbage.tmp");
    std::fs::create_dir_all(junk.parent().unwrap()).unwrap();
    std::fs::write(&junk, b"x").unwrap();

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let mut plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let target = plan
        .actions
        .iter_mut()
        .find(|a| a.kind == CleanupActionKind::RemoveSeenFile)
        .expect("plan must contain a RemoveSeenFile action");
    target.path = "../../etc/passwd".to_string();
    plan.cleanup_plan_hash = cleanup_plan_hash_hex(&plan);

    let quarantine = fresh_dir();
    let err = apply_state_cleanup(
        &store,
        &plan,
        &apply_opts(quarantine.path(), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsafe path") && msg.contains("../../etc/passwd"),
        "expected UnsafePlanPath refusal, got: {msg}"
    );
}

#[test]
fn apply_refuses_traversal_path_in_remove_seen_marker() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let phantom = "ee".repeat(32);
    store.mark_seen(StateNamespace::Sessions, &phantom).unwrap();

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let mut plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let target = plan
        .actions
        .iter_mut()
        .find(|a| a.kind == CleanupActionKind::RemoveSeenMarker)
        .expect("plan must contain a RemoveSeenMarker action");
    target.path = "seen/../../sensitive".to_string();
    plan.cleanup_plan_hash = cleanup_plan_hash_hex(&plan);

    let quarantine = fresh_dir();
    let err = apply_state_cleanup(
        &store,
        &plan,
        &apply_opts(quarantine.path(), NOW_UTC),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsafe path") && msg.contains("parent-directory traversal"),
        "expected UnsafePlanPath refusal, got: {msg}"
    );
}

#[test]
fn apply_refuses_traversal_path_in_tier_b_action() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    // Tamper a join so a tier-B action lands in the plan.
    let joins = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins");
    let join_path = std::fs::read_dir(&joins)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    tamper_signature(&join_path);

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let mut plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let target = plan
        .actions
        .iter_mut()
        .find(|a| a.kind == CleanupActionKind::QuarantineAndUnmarkJoin)
        .expect("plan must contain a QuarantineAndUnmarkJoin action");
    target.path = "../../tmp/sensitive".to_string();
    plan.cleanup_plan_hash = cleanup_plan_hash_hex(&plan);

    let quarantine = fresh_dir();
    let mut opts = apply_opts(quarantine.path(), NOW_UTC);
    opts.allow_invalid_partial_cleanup = true; // tampered join cascades

    let err = apply_state_cleanup(&store, &plan, &opts).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsafe path") && msg.contains("../../tmp/sensitive"),
        "expected UnsafePlanPath refusal, got: {msg}"
    );
}

#[test]
fn apply_refuses_traversal_path_in_seen_marker_path_field() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let joins = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins");
    let join_path = std::fs::read_dir(&joins)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    tamper_signature(&join_path);

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let mut plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();
    let target = plan
        .actions
        .iter_mut()
        .find(|a| a.kind == CleanupActionKind::QuarantineAndUnmarkJoin)
        .expect("plan must contain a QuarantineAndUnmarkJoin action");
    target.seen_marker_path = Some("seen/joins/../../sensitive".to_string());
    plan.cleanup_plan_hash = cleanup_plan_hash_hex(&plan);

    let quarantine = fresh_dir();
    let mut opts = apply_opts(quarantine.path(), NOW_UTC);
    opts.allow_invalid_partial_cleanup = true;
    let err = apply_state_cleanup(&store, &plan, &opts).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsafe path") && msg.contains("parent-directory traversal"),
        "expected UnsafePlanPath refusal, got: {msg}"
    );
}

// ── 16. Stage 12.17 review fix — quarantine-write failure ──
//
// The new Phase A → Phase B → Phase C ordering guarantees that
// if the quarantine write or manifest write fails, NO source
// has been removed yet. This test triggers a Phase A failure
// (quarantine destination is unwritable because the supplied
// `--quarantine-dir` is actually a regular file) and asserts:
//   (a) apply returns an Io error,
//   (b) the source verified body is still on disk,
//   (c) the matching seen marker still exists,
//   (d) the state-dir file count is unchanged.

#[test]
fn quarantine_write_failure_leaves_state_dir_untouched() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let joins = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins");
    let join_path = std::fs::read_dir(&joins)
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
    let marker_key = format!("{}--{}", session.session_id, pubkey_stem);
    tamper_signature(&join_path);

    let (report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_scan(NOW_UTC))
            .unwrap();
    let plan = plan_state_cleanup(&report, &orphans, &plan_opts(NOW_UTC)).unwrap();

    // Quarantine destination is a regular FILE, not a directory.
    // Phase A will try `create_dir_all(<file>/<plan_id>/...)`
    // which fails because the parent isn't a directory.
    let quarantine_dir_holder = fresh_dir();
    let q_file = quarantine_dir_holder.path().join("not-a-dir");
    std::fs::write(&q_file, b"i am a file").unwrap();

    let files_before = count_files(state_dir.path());
    let mut opts = apply_opts(&q_file, NOW_UTC);
    opts.allow_invalid_partial_cleanup = true;
    let err = apply_state_cleanup(&store, &plan, &opts).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.starts_with("cleanup io error"),
        "expected Io error, got: {msg}"
    );

    // (a) Source still on disk + bytes unchanged.
    assert!(
        join_path.is_file(),
        "tampered join body must still be on disk after a quarantine-write failure"
    );
    // (b) Matching seen marker still exists.
    assert!(
        store
            .is_seen(StateNamespace::Joins, &marker_key)
            .unwrap(),
        "seen marker must still exist after a quarantine-write failure"
    );
    // (c) State-dir file count unchanged.
    let files_after = count_files(state_dir.path());
    assert_eq!(
        files_before, files_after,
        "state-dir must be byte-identical after a quarantine-write failure"
    );
}
