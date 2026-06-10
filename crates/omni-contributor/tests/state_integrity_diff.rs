//! Stage 12.19 — integration tests for the integrity-report
//! diff. Uses lightweight synthetic `StateIntegrityReport`
//! values where possible (no state-dir fixture needed), and
//! one end-to-end live-vs-baseline test that reuses the Stage
//! 12.16 builder pattern.

use std::path::Path;

use omni_contributor::{
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, partial_result_signing_input,
        session_id_hex, work_assignment_signing_input,
    },
    diff_state_integrity_reports,
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    scan_state_integrity,
    session::{
        AggregatedContributorResult, AggregatedPartialRef, ContributorJoin,
        ExecutionSession, PartialContributorResult, WorkAssignment, WorkKind,
    },
    ContributorSigner, ContributorStateStore, CoordinatorSigner, DiffOptions,
    FindingKind, FindingSeverity, IntegrityFinding, RecommendedAction, ScanOptions,
    SessionIntegritySummary, StateIntegrityReport, SESSION_SCHEMA_VERSION,
    STATE_INTEGRITY_DIFF_SCHEMA_VERSION, STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
};

const NOW_UTC: &str = "2026-06-10T01:00:00Z";

fn diff_opts<'a>(now_utc: &'a str) -> DiffOptions<'a> {
    DiffOptions {
        now_utc,
        require_state_dir_match: false,
    }
}

/// Build a synthetic v1 `StateIntegrityReport` with the supplied
/// findings. State-dir / version / counts are filled with safe
/// defaults so tests can focus on the finding-set semantics.
fn synth_report(
    generated_at_utc: &str,
    state_dir: &str,
    omni_version: &str,
    findings: Vec<IntegrityFinding>,
) -> StateIntegrityReport {
    let counts_ok = findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Ok)
        .count() as u32;
    let counts_warn = findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Warn)
        .count() as u32;
    let counts_error = findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Error)
        .count() as u32;
    StateIntegrityReport {
        schema_version: STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
        generated_at_utc: generated_at_utc.to_string(),
        state_dir: state_dir.to_string(),
        state_version: 2,
        omni_contributor_version: omni_version.to_string(),
        sessions_scanned: 0,
        sessions_verified: 0,
        counts_ok,
        counts_warn,
        counts_error,
        sessions: Vec::new(),
        findings,
    }
}

/// Build a closed-set finding. `sid` is `Some(hex)` for
/// session-scoped findings; `path` is the relative path; the
/// kind/severity/action map matches what the Stage 12.16 scanner
/// would actually emit.
fn finding(
    kind: FindingKind,
    severity: FindingSeverity,
    sid: Option<&str>,
    path: Option<&str>,
    reason_tag: &str,
    recommended_action: RecommendedAction,
) -> IntegrityFinding {
    IntegrityFinding {
        kind,
        severity,
        session_id: sid.map(|s| s.to_string()),
        path: path.map(|s| s.to_string()),
        reason_tag: reason_tag.to_string(),
        recommended_action,
    }
}

// ── 1. Identical reports → empty diff ─────────────────────────

#[test]
fn identical_reports_produce_empty_diff_with_full_unchanged() {
    let f = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&"aa".repeat(32)),
        Some(&format!("verified/sessions/{}/joins/{}.json", "aa".repeat(32), "bb".repeat(32))),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let a = synth_report("2026-06-09T00:00:00Z", "/state", "0.1.0", vec![f.clone()]);
    let b = synth_report("2026-06-10T00:00:00Z", "/state", "0.1.0", vec![f.clone()]);
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(diff.counts.new, 0);
    assert_eq!(diff.counts.resolved, 0);
    assert_eq!(diff.counts.unchanged, 1);
    assert_eq!(diff.unchanged_findings.len(), 1);
    assert!(diff.new_findings.is_empty());
    assert!(diff.resolved_findings.is_empty());
    assert_eq!(diff.schema_version, STATE_INTEGRITY_DIFF_SCHEMA_VERSION);
}

// ── 2. Empty baseline + non-empty current → all new ───────────

#[test]
fn empty_baseline_classifies_everything_as_new() {
    let f1 = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&"aa".repeat(32)),
        Some("verified/sessions/.../joins/x.json"),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let f2 = finding(
        FindingKind::StaleSeenMarker,
        FindingSeverity::Warn,
        Some(&"bb".repeat(32)),
        Some("seen/sessions/bb..."),
        "verified_body_missing",
        RecommendedAction::DELETE_STALE_SEEN_MARKER,
    );
    let a = synth_report("2026-06-09T00:00:00Z", "/state", "0.1.0", vec![]);
    let b = synth_report(
        "2026-06-10T00:00:00Z",
        "/state",
        "0.1.0",
        vec![f1, f2],
    );
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(diff.counts.new, 2);
    assert_eq!(diff.counts.new_error, 1);
    assert_eq!(diff.counts.new_warn, 1);
    assert_eq!(diff.counts.resolved, 0);
    assert!(diff.resolved_findings.is_empty());
    assert!(diff.unchanged_findings.is_empty());
}

// ── 3. Non-empty baseline + empty current → all resolved ──────

#[test]
fn empty_current_classifies_everything_as_resolved() {
    let f = finding(
        FindingKind::InvalidPartial,
        FindingSeverity::Error,
        Some(&"cc".repeat(32)),
        Some("verified/sessions/.../partials/x.json"),
        "tampered",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let a = synth_report("2026-06-09T00:00:00Z", "/state", "0.1.0", vec![f.clone()]);
    let b = synth_report("2026-06-10T00:00:00Z", "/state", "0.1.0", vec![]);
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(diff.counts.new, 0);
    assert_eq!(diff.counts.resolved, 1);
    assert_eq!(diff.counts.resolved_error, 1);
    assert_eq!(diff.resolved_findings.len(), 1);
}

// ── 4. Mixed new / resolved / unchanged ───────────────────────

#[test]
fn mixed_diff_populates_all_three_buckets() {
    let shared = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&"aa".repeat(32)),
        Some("verified/sessions/aa/joins/x.json"),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let only_baseline = finding(
        FindingKind::StaleSeenMarker,
        FindingSeverity::Warn,
        Some(&"bb".repeat(32)),
        Some("seen/sessions/bb"),
        "verified_body_missing",
        RecommendedAction::DELETE_STALE_SEEN_MARKER,
    );
    let only_current = finding(
        FindingKind::InvalidAssignment,
        FindingSeverity::Error,
        Some(&"cc".repeat(32)),
        Some("verified/sessions/cc/assignments/x.json"),
        "BindingMismatch",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );

    let a = synth_report(
        "2026-06-09T00:00:00Z",
        "/state",
        "0.1.0",
        vec![shared.clone(), only_baseline.clone()],
    );
    let b = synth_report(
        "2026-06-10T00:00:00Z",
        "/state",
        "0.1.0",
        vec![shared.clone(), only_current.clone()],
    );
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(diff.counts.new, 1);
    assert_eq!(diff.counts.resolved, 1);
    assert_eq!(diff.counts.unchanged, 1);
    assert_eq!(diff.new_findings[0].kind, FindingKind::InvalidAssignment);
    assert_eq!(diff.resolved_findings[0].kind, FindingKind::StaleSeenMarker);
    assert_eq!(diff.unchanged_findings[0].kind, FindingKind::InvalidJoin);
}

// ── 5. Sort stability: diff is byte-stable on identical inputs ─

#[test]
fn diff_output_is_byte_stable_on_identical_inputs() {
    let f1 = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&"aa".repeat(32)),
        Some("verified/sessions/aa/joins/x.json"),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let f2 = finding(
        FindingKind::StaleSeenMarker,
        FindingSeverity::Warn,
        Some(&"bb".repeat(32)),
        Some("seen/sessions/bb"),
        "verified_body_missing",
        RecommendedAction::DELETE_STALE_SEEN_MARKER,
    );
    let a = synth_report("2026-06-09T00:00:00Z", "/state", "0.1.0", vec![f1.clone(), f2.clone()]);
    let b = synth_report("2026-06-10T00:00:00Z", "/state", "0.1.0", vec![f2.clone(), f1.clone()]);
    let d1 = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    let d2 = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(
        serde_json::to_string(&d1).unwrap(),
        serde_json::to_string(&d2).unwrap()
    );
}

// ── 6. Schema-version mismatch refusal (baseline) ─────────────

#[test]
fn baseline_schema_v2_is_refused() {
    let mut a = synth_report("x", "/state", "0.1.0", vec![]);
    a.schema_version = 2;
    let b = synth_report("y", "/state", "0.1.0", vec![]);
    let err = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported baseline schema_version"),
        "got: {msg}"
    );
}

// ── 7. Schema-version mismatch refusal (current) ──────────────

#[test]
fn current_schema_v2_is_refused() {
    let a = synth_report("x", "/state", "0.1.0", vec![]);
    let mut b = synth_report("y", "/state", "0.1.0", vec![]);
    b.schema_version = 2;
    let err = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported current schema_version"),
        "got: {msg}"
    );
}

// ── 8. State-version mismatch refusal ─────────────────────────

#[test]
fn state_version_mismatch_is_refused() {
    let mut a = synth_report("x", "/state", "0.1.0", vec![]);
    a.state_version = 1;
    let b = synth_report("y", "/state", "0.1.0", vec![]);
    let err = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("incompatible state-dir version"), "got: {msg}");
}

// ── 9. state_dir mismatch — flag off (ok) / on (refuse) ───────

#[test]
fn state_dir_mismatch_passes_without_flag_and_refuses_with_flag() {
    let a = synth_report("x", "/host-a", "0.1.0", vec![]);
    let b = synth_report("y", "/host-b", "0.1.0", vec![]);
    let ok =
        diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(ok.baseline_state_dir, "/host-a");
    assert_eq!(ok.state_dir, "/host-b");

    let mut strict = diff_opts(NOW_UTC);
    strict.require_state_dir_match = true;
    let err = diff_state_integrity_reports(&a, &b, &strict).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("state_dir mismatch"), "got: {msg}");
}

// ── 10. omni_contributor_version drift is informational only ──

#[test]
fn omni_version_drift_is_informational() {
    let a = synth_report("x", "/state", "0.1.0", vec![]);
    let b = synth_report("y", "/state", "0.2.0", vec![]);
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(diff.baseline_omni_contributor_version, "0.1.0");
    assert_eq!(diff.current_omni_contributor_version, "0.2.0");
}

// ── 11. Metadata drift refusal — severity ─────────────────────

#[test]
fn finding_severity_drift_is_refused() {
    let key_sid = "aa".repeat(32);
    let key_path = "verified/sessions/aa/joins/x.json";
    let baseline_f = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&key_sid),
        Some(key_path),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let current_f = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Warn, // drift!
        Some(&key_sid),
        Some(key_path),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let a = synth_report("x", "/state", "0.1.0", vec![baseline_f]);
    let b = synth_report("y", "/state", "0.1.0", vec![current_f]);
    let err = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("finding metadata drift") && msg.contains("severity baseline=error current=warn"),
        "got: {msg}"
    );
}

// ── 12. Metadata drift refusal — recommended_action ───────────

#[test]
fn finding_action_drift_is_refused() {
    let key_sid = "aa".repeat(32);
    let key_path = "verified/sessions/aa/joins/x.json";
    let baseline_f = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&key_sid),
        Some(key_path),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let current_f = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&key_sid),
        Some(key_path),
        "ContributorSignatureFailed",
        RecommendedAction::DELETE_STALE_SEEN_MARKER, // drift!
    );
    let a = synth_report("x", "/state", "0.1.0", vec![baseline_f]);
    let b = synth_report("y", "/state", "0.1.0", vec![current_f]);
    let err = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("finding metadata drift"), "got: {msg}");
    assert!(msg.contains("action baseline="), "got: {msg}");
}

// ── 13. JSON round-trip ───────────────────────────────────────

#[test]
fn diff_json_round_trip_preserves_report() {
    let f = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&"aa".repeat(32)),
        Some("verified/sessions/aa/joins/x.json"),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let a = synth_report("x", "/state", "0.1.0", vec![]);
    let b = synth_report("y", "/state", "0.1.0", vec![f]);
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();

    let s = serde_json::to_string(&diff).unwrap();
    let round: omni_contributor::StateIntegrityDiffReport =
        serde_json::from_str(&s).unwrap();
    assert_eq!(round, diff);
}

// ── 14. Per-severity sub-counts ───────────────────────────────

#[test]
fn diff_counts_split_by_severity() {
    let ok_finding = finding(
        FindingKind::ArchiveCoveredSession,
        FindingSeverity::Ok,
        Some(&"aa".repeat(32)),
        None,
        "covered",
        RecommendedAction::NONE,
    );
    let warn_finding = finding(
        FindingKind::StaleSeenMarker,
        FindingSeverity::Warn,
        Some(&"bb".repeat(32)),
        Some("seen/sessions/bb"),
        "verified_body_missing",
        RecommendedAction::DELETE_STALE_SEEN_MARKER,
    );
    let err_finding = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&"cc".repeat(32)),
        Some("verified/sessions/cc/joins/x.json"),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let a = synth_report("x", "/state", "0.1.0", vec![]);
    let b = synth_report(
        "y",
        "/state",
        "0.1.0",
        vec![ok_finding, warn_finding, err_finding],
    );
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(diff.counts.new, 3);
    assert_eq!(diff.counts.new_ok, 1);
    assert_eq!(diff.counts.new_warn, 1);
    assert_eq!(diff.counts.new_error, 1);
}

// ── 15. End-to-end live-vs-baseline composition ───────────────

const COORD_SEED: [u8; 32] = *b"stage12.19-diff-coord-seed-32--!";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.19-diff-contrib-a-32----";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.19-diff-contrib-b-32----";
const FAR_FUTURE: &str = "2026-12-31T23:59:59Z";

fn fresh_dir() -> tempfile::TempDir {
    tempfile::TempDir::new().expect("tempdir")
}

fn open_store(root: &Path) -> ContributorStateStore {
    let (s, _) = ContributorStateStore::open(root, false, NOW_UTC).unwrap();
    s
}

fn build_session() -> ExecutionSession {
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
        expires_at_utc: FAR_FUTURE.into(),
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
    let session = build_session();
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

fn default_scan<'a>(now_utc: &'a str) -> ScanOptions<'a> {
    ScanOptions {
        session_id_filter: None,
        archive_dir: None,
        now_utc,
    }
}

#[test]
fn live_scan_against_baseline_round_trips_via_json() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);

    // Clean baseline: scan + persist via to_vec_pretty (same path
    // the CLI's `--json-out` uses).
    let baseline = scan_state_integrity(&store, &default_scan(NOW_UTC)).unwrap();
    let baseline_bytes = serde_json::to_vec_pretty(&baseline).unwrap();
    let baseline_round: StateIntegrityReport =
        serde_json::from_slice(&baseline_bytes).unwrap();
    assert_eq!(baseline_round, baseline);

    // Tamper a join + re-scan.
    let joins = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins");
    let join_file = std::fs::read_dir(&joins)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    tamper_signature(&join_file);
    let current = scan_state_integrity(&store, &default_scan(NOW_UTC)).unwrap();
    assert!(current
        .findings
        .iter()
        .any(|f| matches!(f.kind, FindingKind::InvalidJoin)));

    // Diff: exactly one InvalidJoin appears as new; nothing
    // resolved.
    let diff =
        diff_state_integrity_reports(&baseline_round, &current, &diff_opts(NOW_UTC))
            .unwrap();
    assert!(diff.counts.new >= 1);
    assert_eq!(diff.counts.resolved, 0);
    assert!(diff
        .new_findings
        .iter()
        .any(|f| matches!(f.kind, FindingKind::InvalidJoin)));
}

// ── 16. SessionIntegritySummary drift is informational ────────
//
// The diff doesn't read or compare `sessions: Vec<...>` — it's
// derived state that varies across runs. Confirm a difference
// there doesn't break the diff.

#[test]
fn session_summary_diff_does_not_affect_finding_diff() {
    let mut a = synth_report("x", "/state", "0.1.0", vec![]);
    a.sessions = vec![SessionIntegritySummary {
        session_id: "aa".repeat(32),
        overall_status: "Aggregated".to_string(),
    }];
    let b = synth_report("y", "/state", "0.1.0", vec![]);
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();
    assert_eq!(diff.counts.new, 0);
    assert_eq!(diff.counts.resolved, 0);
    assert_eq!(diff.counts.unchanged, 0);
}

// ── 17. Stage 12.19 review fix — --summary-only redaction ──
//
// The CLI `--summary-only` flag is a *presentation* gate that
// must apply to events, pretty, JSON stdout, AND `--json-out`
// mirroring alike. The library helper `diff_presentation_view`
// is the single source of truth — it clones the diff and
// clears `unchanged_findings` when summary_only is set, while
// preserving `counts.unchanged` so scripts still see the
// elided count.

#[test]
fn diff_presentation_view_elides_unchanged_when_summary_only() {
    use omni_contributor::diff_presentation_view;

    let shared = finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&"aa".repeat(32)),
        Some("verified/sessions/aa/joins/x.json"),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let new_finding = finding(
        FindingKind::InvalidAssignment,
        FindingSeverity::Error,
        Some(&"bb".repeat(32)),
        Some("verified/sessions/bb/assignments/x.json"),
        "BindingMismatch",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    );
    let a = synth_report("x", "/state", "0.1.0", vec![shared.clone()]);
    let b = synth_report(
        "y",
        "/state",
        "0.1.0",
        vec![shared.clone(), new_finding.clone()],
    );
    let diff = diff_state_integrity_reports(&a, &b, &diff_opts(NOW_UTC)).unwrap();

    // Sanity: the library helper always populates every vector.
    assert_eq!(diff.counts.unchanged, 1);
    assert_eq!(diff.unchanged_findings.len(), 1);
    assert_eq!(diff.counts.new, 1);
    assert_eq!(diff.new_findings.len(), 1);

    // summary_only == false → zero-copy borrow, unchanged
    // intact.
    let view_full = diff_presentation_view(&diff, false);
    assert_eq!(view_full.unchanged_findings.len(), 1);
    assert_eq!(view_full.counts.unchanged, 1);

    // summary_only == true → owned clone, `unchanged_findings`
    // cleared, but `counts.unchanged` preserved.
    let view_summary = diff_presentation_view(&diff, true);
    assert!(
        view_summary.unchanged_findings.is_empty(),
        "summary view must clear unchanged_findings"
    );
    assert_eq!(
        view_summary.counts.unchanged, 1,
        "counts.unchanged must be preserved so scripts can read the elided count"
    );
    // New / resolved untouched on the summary view.
    assert_eq!(view_summary.new_findings.len(), 1);
    assert_eq!(view_summary.counts.new, 1);
    assert_eq!(view_summary.resolved_findings.len(), 0);

    // JSON serialization carries the same redaction — proves
    // the CLI JSON stdout render + `--json-out` mirror will
    // both elide unchanged on summary_only.
    let full_json = serde_json::to_string(view_full.as_ref()).unwrap();
    let summary_json = serde_json::to_string(view_summary.as_ref()).unwrap();
    assert!(
        full_json.contains("\"unchanged_findings\":[{"),
        "full JSON must contain unchanged_findings entries; got: {full_json}"
    );
    assert!(
        summary_json.contains("\"unchanged_findings\":[]"),
        "summary JSON must have empty unchanged_findings array; got: {summary_json}"
    );
    assert!(
        summary_json.contains("\"unchanged\":1"),
        "summary JSON must preserve counts.unchanged; got: {summary_json}"
    );

    // The original diff is untouched (library helper does not
    // mutate its input).
    assert_eq!(diff.unchanged_findings.len(), 1);
}
