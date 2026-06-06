//! Stage 12.16 — integration tests for `scan_state_integrity`.
//!
//! Each test stands up a temp state-dir, optionally seeds an
//! aggregated session via the same builder pattern as
//! `restore_session_archive.rs` / `archive_session_integration.rs`,
//! optionally perturbs one body / marker / file, then drives the
//! scan and asserts on closed-set discriminators in the resulting
//! `StateIntegrityReport`. The scanner must never mutate the
//! state-dir or archive-dir — a file-count diff guards that.

use std::path::Path;

use omni_contributor::{
    archive_session,
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, partial_result_signing_input,
        session_id_hex, work_assignment_signing_input,
    },
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    scan_state_integrity,
    session::{
        AggregatedContributorResult, AggregatedPartialRef, ContributorJoin,
        ExecutionSession, PartialContributorResult, WorkAssignment, WorkKind,
    },
    ArchiveMode, ArchiveOptions, ArchiveStatusRequirement, ContributorSigner,
    ContributorStateStore, CoordinatorSigner, FindingKind, FindingSeverity,
    scan_state_integrity_with_audit_orphans, ScanOptions, SESSION_SCHEMA_VERSION,
    STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.16-integrity-coord-seed!";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.16-integrity-contrib-a-!";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.16-integrity-contrib-b-!";
const NOW_UTC: &str = "2026-06-05T01:00:00Z";
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

fn build_join(
    session: &ExecutionSession,
    contrib: &ContributorSigner,
) -> ContributorJoin {
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 100,
        max_output_tokens: 100,
        supported_work_unit_kinds: vec![WorkUnitKind::Layers],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-06-02T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

fn build_assignment(
    session: &ExecutionSession,
    contrib: &ContributorSigner,
    coord: &CoordinatorSigner,
    stage_index: u32,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index,
        contributor_pubkey_hex: contrib.pubkey_hex(),
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

fn default_opts<'a>(now_utc: &'a str) -> ScanOptions<'a> {
    ScanOptions {
        session_id_filter: None,
        archive_dir: None,
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

// ── 1. Clean state-dir + scanner-writes-nothing ───────────────

#[test]
fn clean_state_produces_no_findings_and_writes_nothing() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let _session = seed_aggregated_session(&store);

    let files_before = count_files(state_dir.path());
    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let files_after = count_files(state_dir.path());

    assert_eq!(report.schema_version, STATE_INTEGRITY_REPORT_SCHEMA_VERSION);
    assert_eq!(report.sessions_scanned, 1);
    assert_eq!(report.sessions_verified, 1);
    assert_eq!(report.counts_error, 0, "{report:?}");
    assert_eq!(report.counts_warn, 0, "{report:?}");
    assert!(report.findings.is_empty(), "{:?}", report.findings);
    assert_eq!(
        files_before, files_after,
        "scanner must not mutate the state-dir"
    );
}

// ── 2. Verifier-failure findings: tampered bodies ─────────────

/// Corrupt the signature on a verified body in place. Keeps the
/// JSON parseable (so Stage 12.7 parse-only loaders still return
/// the body) and flips exactly one hex char of the first
/// signature_hex field found, so the verifier sees a bad sig and
/// rejects via Stage 12.13's rejection-note pipeline.
fn tamper_signature(p: &Path) {
    let mut s = std::fs::read_to_string(p).unwrap();
    // The `serde_json::to_vec_pretty` format used by
    // `write_verified_json` produces lines like:
    //   "coordinator_signature_hex": "abcdef..."
    // — quoted key, colon-space, quoted value. The first hex
    // char of the value is therefore at the byte right after
    // the value-opening quote.
    let keys = ["coordinator_signature_hex", "contributor_signature_hex"];
    for key in keys {
        let needle = format!("\"{key}\": \"");
        if let Some(idx) = s.find(&needle) {
            let pos = idx + needle.len();
            // SAFETY: pos points at an ASCII hex char inside a
            // hex string literal in a UTF-8 JSON file.
            let bytes = unsafe { s.as_bytes_mut() };
            let old = bytes[pos];
            let new = match old {
                b'0' => b'1',
                b'1'..=b'8' => old + 1,
                b'9' => b'a',
                b'a'..=b'e' => old + 1,
                b'f' => b'0',
                _ => panic!("unexpected non-hex byte at signature pos {pos}: {old}"),
            };
            bytes[pos] = new;
            std::fs::write(p, s.as_bytes()).unwrap();
            return;
        }
    }
    panic!(
        "no signature_hex field found in {p:?}; cannot tamper"
    );
}

/// Arbitrary byte flip — used to corrupt archived bodies so the
/// Stage 12.15 BLAKE3 walk surfaces an `ArchiveBlakeMismatch`
/// finding regardless of which byte changed.
fn tamper_byte(p: &Path) {
    let mut bytes = std::fs::read(p).unwrap();
    if let Some(b) = bytes.last_mut() {
        *b = b.wrapping_add(1);
    }
    std::fs::write(p, &bytes).unwrap();
}

#[test]
fn tampered_session_body_emits_invalid_session_finding() {
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

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert!(
        report.findings.iter().any(|f| matches!(
            f.kind,
            FindingKind::InvalidSession
        ) && f.severity == FindingSeverity::Error),
        "expected InvalidSession; got {:?}",
        report.findings
    );
    assert!(report.counts_error > 0);
}

#[test]
fn tampered_join_body_emits_invalid_join_finding() {
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
    tamper_signature(&join_file);

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| matches!(f.kind, FindingKind::InvalidJoin)),
        "{:?}",
        report.findings
    );
    assert!(report.counts_error > 0);
}

#[test]
fn tampered_assignment_body_emits_invalid_assignment_finding() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let asn_dir = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("assignments");
    let asn_file = std::fs::read_dir(&asn_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    tamper_signature(&asn_file);

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| matches!(f.kind, FindingKind::InvalidAssignment)),
        "{:?}",
        report.findings
    );
    assert!(report.counts_error > 0);
}

#[test]
fn tampered_partial_body_emits_invalid_partial_finding() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let partials_dir = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("partials");
    let p_file = std::fs::read_dir(&partials_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    tamper_signature(&p_file);

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| matches!(f.kind, FindingKind::InvalidPartial)),
        "{:?}",
        report.findings
    );
    assert!(report.counts_error > 0);
}

#[test]
fn tampered_aggregate_body_emits_invalid_aggregate_finding() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let agg_body = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("aggregated.json");
    tamper_signature(&agg_body);

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| matches!(f.kind, FindingKind::InvalidAggregate)),
        "{:?}",
        report.findings
    );
    assert!(report.counts_error > 0);
}

// ── 3. Seen-marker ↔ body consistency ─────────────────────────

#[test]
fn stale_seen_marker_without_body_emits_finding() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    // Mark a session marker whose verified body never existed.
    let phantom_sid = "ff".repeat(32);
    store
        .mark_seen(StateNamespace::Sessions, &phantom_sid)
        .unwrap();

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert!(
        report.findings.iter().any(|f| matches!(
            f.kind,
            FindingKind::StaleSeenMarker
        ) && f.severity == FindingSeverity::Warn),
        "{:?}",
        report.findings
    );
    assert!(report.counts_warn > 0);
}

#[test]
fn missing_seen_marker_with_body_emits_finding() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);

    // Delete the `seen/aggregates/<sid>` marker so the body
    // is verified but the marker is missing.
    let marker_path = state_dir
        .path()
        .join("seen")
        .join("aggregates")
        .join(&session.session_id);
    std::fs::remove_file(&marker_path).unwrap();
    assert!(!store
        .is_seen(StateNamespace::Aggregates, &session.session_id)
        .unwrap());

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| matches!(f.kind, FindingKind::MissingSeenMarker)),
        "{:?}",
        report.findings
    );
}

// ── 4. Stray-file walker ──────────────────────────────────────

#[test]
fn stray_verified_file_emits_finding() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    // Drop a junk file inside joins/.
    let junk = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("joins")
        .join("not-hex.txt");
    std::fs::write(&junk, b"garbage").unwrap();

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| matches!(f.kind, FindingKind::StrayVerifiedFile)),
        "{:?}",
        report.findings
    );
}

// ── 5. session-id filter ──────────────────────────────────────

#[test]
fn session_id_filter_restricts_session_scoped_findings() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    // Tamper a join — should be reported normally.
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
    tamper_signature(&join_file);

    // Filter for an UNRELATED sid: the session-scoped findings
    // must drop out (the snapshot has no matching session).
    let unrelated = "ab".repeat(32);
    let opts = ScanOptions {
        session_id_filter: Some(&unrelated),
        archive_dir: None,
        now_utc: NOW_UTC,
    };
    let report = scan_state_integrity(&store, &opts).unwrap();
    let invalid_join_count = report
        .findings
        .iter()
        .filter(|f| matches!(f.kind, FindingKind::InvalidJoin))
        .count();
    assert_eq!(
        invalid_join_count, 0,
        "session_id filter must hide session-scoped findings for unrelated sid; got {:?}",
        report.findings
    );
    assert_eq!(report.sessions_scanned, 0);
}

// ── 6. --include-archives clean walk ──────────────────────────

#[test]
fn clean_archive_dir_via_include_archives_emits_no_archive_findings() {
    let state_dir = fresh_dir();
    let archive_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);

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

    let opts = ScanOptions {
        session_id_filter: None,
        archive_dir: Some(archive_dir.path()),
        now_utc: NOW_UTC,
    };
    let report = scan_state_integrity(&store, &opts).unwrap();
    // No archive-related errors / warnings — the only allowed
    // archive finding is the `Ok` ArchiveCoveredSession.
    let archive_problems = report
        .findings
        .iter()
        .filter(|f| matches!(
            f.kind,
            FindingKind::ArchiveBlakeMismatch | FindingKind::ArchiveManifestMalformed
        ))
        .count();
    assert_eq!(archive_problems, 0, "{:?}", report.findings);
    assert!(report.findings.iter().any(|f| matches!(
        f.kind,
        FindingKind::ArchiveCoveredSession
    ) && f.severity == FindingSeverity::Ok));
}

// ── 7. --include-archives BLAKE3 mismatch ─────────────────────

#[test]
fn corrupt_archive_blake3_emits_archive_blake_mismatch_finding() {
    let state_dir = fresh_dir();
    let archive_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);

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
    // Corrupt a body in the archive.
    let archived_session_body = archive_dir
        .path()
        .join(&session.session_id)
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("session.json");
    tamper_byte(&archived_session_body);

    let opts = ScanOptions {
        session_id_filter: None,
        archive_dir: Some(archive_dir.path()),
        now_utc: NOW_UTC,
    };
    let report = scan_state_integrity(&store, &opts).unwrap();
    assert!(
        report.findings.iter().any(|f| matches!(
            f.kind,
            FindingKind::ArchiveBlakeMismatch
        ) && f.severity == FindingSeverity::Error),
        "{:?}",
        report.findings
    );
    assert!(report.counts_error > 0);
}

// ── 8. --include-archives malformed manifest ──────────────────

#[test]
fn missing_archive_manifest_emits_manifest_malformed_finding() {
    let state_dir = fresh_dir();
    let archive_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);

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
    // Remove the manifest entirely so RestoreError surfaces.
    let manifest_path = archive_dir
        .path()
        .join(&session.session_id)
        .join("manifest.json");
    std::fs::remove_file(&manifest_path).unwrap();

    let opts = ScanOptions {
        session_id_filter: None,
        archive_dir: Some(archive_dir.path()),
        now_utc: NOW_UTC,
    };
    let report = scan_state_integrity(&store, &opts).unwrap();
    assert!(
        report.findings.iter().any(|f| matches!(
            f.kind,
            FindingKind::ArchiveManifestMalformed
        ) && f.severity == FindingSeverity::Error),
        "{:?}",
        report.findings
    );
}

// ── 9. JSON round-trip preserves report ───────────────────────

#[test]
fn json_roundtrip_preserves_report() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);
    let agg_body = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("aggregated.json");
    tamper_signature(&agg_body);

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let json = serde_json::to_string(&report).unwrap();
    let round: omni_contributor::StateIntegrityReport =
        serde_json::from_str(&json).unwrap();
    assert_eq!(round.schema_version, report.schema_version);
    assert_eq!(round.sessions_scanned, report.sessions_scanned);
    assert_eq!(round.counts_error, report.counts_error);
    assert_eq!(round.findings.len(), report.findings.len());
    for (a, b) in round.findings.iter().zip(report.findings.iter()) {
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.severity, b.severity);
        assert_eq!(a.session_id, b.session_id);
        assert_eq!(a.path, b.path);
        assert_eq!(a.reason_tag, b.reason_tag);
        assert_eq!(a.recommended_action.as_str(), b.recommended_action.as_str());
    }
}

// ── 10. Findings are deterministically ordered ────────────────

#[test]
fn findings_are_deterministically_ordered_across_repeat_runs() {
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

    let a = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let b = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let aj = serde_json::to_string(&a.findings).unwrap();
    let bj = serde_json::to_string(&b.findings).unwrap();
    assert_eq!(aj, bj);
}

// ── 11. Empty state-dir scans cleanly ─────────────────────────

#[test]
fn empty_state_dir_produces_clean_report() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    assert_eq!(report.sessions_scanned, 0);
    assert_eq!(report.counts_error, 0);
    assert_eq!(report.counts_warn, 0);
    assert!(report.findings.is_empty());
}

// ── 12. Supersession seen marker round-trips clean ────────────
//
// Regression for the Stage 12.16 review finding that the
// reverse-walk mapped `seen/assignment-supersessions/<sid>--<id>`
// to a non-existent `verified/sessions/<sid>/assignment-supersessions/`
// path (the real verified leaf is `supersessions/`), producing a
// false `StaleSeenMarker` for every clean supersession marker.

#[test]
fn supersession_seen_marker_with_body_emits_no_stale_finding() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let session = seed_aggregated_session(&store);

    // The reverse-walk only checks on-disk presence at the
    // expected path — so a synthetic file with non-parseable
    // content is enough to exercise the path mapping. (The
    // Stage 12.7 parse-only loader silently skips it, so no
    // rejection note fires.)
    let sup_id = "ab".repeat(32);
    let sup_dir = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id)
        .join("supersessions");
    std::fs::create_dir_all(&sup_dir).unwrap();
    let sup_body = sup_dir.join(format!("{sup_id}.json"));
    std::fs::write(&sup_body, b"{}").unwrap();
    let marker = format!("{}--{}", session.session_id, sup_id);
    store
        .mark_seen(StateNamespace::AssignmentSupersessions, &marker)
        .unwrap();

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let stale_sup: Vec<_> = report
        .findings
        .iter()
        .filter(|f| {
            matches!(f.kind, FindingKind::StaleSeenMarker)
                && f.path
                    .as_deref()
                    .unwrap_or("")
                    .contains("seen/assignment-supersessions/")
        })
        .collect();
    assert!(
        stale_sup.is_empty(),
        "supersession seen marker must not be reported stale when its body \
         exists at verified/sessions/<sid>/supersessions/<id>.json; got {:?}",
        stale_sup
    );
}

// ── 13. Scanner leaves expired sessions on disk ───────────────
//
// Regression for the Stage 12.16 review finding that the CLI was
// opening the state-store with auto-prune ENABLED by default,
// which silently cascades-removes expired session subtrees before
// the scan ever runs. Stage 12.16's contract is "reads, never
// writes" — the scanner must surface expired/incomplete state,
// not delete it. The fix forces `auto_prune = false` in
// `run_state_integrity`. This test mirrors that posture and
// proves the scanner library leaves the subtree intact.

#[test]
fn scanner_leaves_expired_session_subtree_on_disk() {
    use omni_contributor::{StateNamespace, StateObjectKind};
    let state_dir = fresh_dir();
    // Open with auto_prune=false — same posture the CLI's
    // run_state_integrity uses after the review fix.
    let (store, _) =
        ContributorStateStore::open(state_dir.path(), false, NOW_UTC).unwrap();

    // Seed a session whose `expires_at_utc` is BEFORE `NOW_UTC`.
    // With auto_prune enabled this would be cascaded away on
    // open(); with auto_prune disabled it survives.
    let expired_at = "2026-06-01T00:00:00Z";
    assert!(expired_at < NOW_UTC, "fixture sanity");
    let expired = build_session(expired_at);
    store
        .write_verified_json(StateObjectKind::Session, &expired.session_id, &expired)
        .unwrap();
    store
        .mark_seen(StateNamespace::Sessions, &expired.session_id)
        .unwrap();

    let session_body = state_dir
        .path()
        .join("verified")
        .join("sessions")
        .join(&expired.session_id)
        .join("session.json");
    assert!(session_body.is_file(), "fixture sanity: body present");

    let files_before = count_files(state_dir.path());
    let _report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let files_after = count_files(state_dir.path());

    assert_eq!(
        files_before, files_after,
        "scanner must leave the disk unchanged"
    );
    assert!(
        session_body.is_file(),
        "expired session.json must still be on disk after the scan"
    );
}

// ── 14. Stage 12.17 — StraySeenFile emission ──────────────────
//
// Three forms of stray-seen the scanner must surface:
//   (a) a file directly under `seen/` (no namespace dir),
//   (b) a file under an unknown-namespace dir,
//   (c) a shape-malformed key inside a valid namespace dir.

#[test]
fn stray_seen_file_under_seen_root_emits_finding() {
    let state_dir = fresh_dir();
    let _store = open_store(state_dir.path());
    let junk = state_dir.path().join("seen").join("garbage.tmp");
    std::fs::write(&junk, b"junk").unwrap();

    let store = open_store(state_dir.path());
    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let strays: Vec<_> = report
        .findings
        .iter()
        .filter(|f| {
            matches!(f.kind, FindingKind::StraySeenFile)
                && f.path.as_deref() == Some("seen/garbage.tmp")
        })
        .collect();
    assert_eq!(
        strays.len(),
        1,
        "expected one StraySeenFile for seen/garbage.tmp; got {:?}",
        report.findings
    );
}

#[test]
fn stray_seen_file_under_unknown_namespace_emits_finding() {
    let state_dir = fresh_dir();
    let _store = open_store(state_dir.path());
    let unknown_ns = state_dir.path().join("seen").join("unknown-ns");
    std::fs::create_dir_all(&unknown_ns).unwrap();
    std::fs::write(unknown_ns.join("whatever"), b"x").unwrap();

    let store = open_store(state_dir.path());
    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let strays: Vec<_> = report
        .findings
        .iter()
        .filter(|f| matches!(f.kind, FindingKind::StraySeenFile))
        .filter(|f| {
            f.path
                .as_deref()
                .unwrap_or("")
                .starts_with("seen/unknown-ns/")
        })
        .collect();
    assert_eq!(
        strays.len(),
        1,
        "expected one StraySeenFile for unknown-ns; got {:?}",
        report.findings
    );
}

#[test]
fn shape_malformed_seen_key_emits_stray_not_stale() {
    use omni_contributor::StateNamespace;
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    // A "session" marker whose key isn't 64-hex.
    store
        .mark_seen(StateNamespace::Sessions, "not_64_hex_at_all")
        .unwrap();

    let report = scan_state_integrity(&store, &default_opts(NOW_UTC)).unwrap();
    let strays: Vec<_> = report
        .findings
        .iter()
        .filter(|f| matches!(f.kind, FindingKind::StraySeenFile))
        .collect();
    let stales: Vec<_> = report
        .findings
        .iter()
        .filter(|f| matches!(f.kind, FindingKind::StaleSeenMarker))
        .collect();
    assert_eq!(strays.len(), 1, "{:?}", report.findings);
    assert!(
        stales.is_empty(),
        "shape-malformed key must not double-emit StaleSeenMarker; got {:?}",
        stales
    );
}

// ── 15. Stage 12.17 — orphan side-channel ─────────────────────

#[test]
fn audit_orphan_side_channel_is_empty_on_clean_state() {
    let state_dir = fresh_dir();
    let store = open_store(state_dir.path());
    let _session = seed_aggregated_session(&store);
    let (_report, orphans) =
        scan_state_integrity_with_audit_orphans(&store, &default_opts(NOW_UTC))
            .unwrap();
    assert!(
        orphans.is_empty(),
        "clean state must not surface any orphan ids; got {:?}",
        orphans
    );
}
