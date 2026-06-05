//! Stage 12.15 — integration tests for `restore_session_archive`.
//!
//! Each test stands up a temp state-dir, archives a session to a
//! temp archive-dir via Stage 12.14's `archive_session`, then
//! drives the restore path through one accept/refuse arm.
//! Session-chain helpers (`build_session`, `build_join`,
//! `build_assignment`, …) are locally duplicated — same
//! convention as the Stage 12.10–12.14 test files.

use std::path::PathBuf;

use omni_contributor::{
    archive_session,
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, partial_result_signing_input,
        session_id_hex, work_assignment_signing_input,
    },
    restore_session_archive,
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    session::{
        AggregatedContributorResult, AggregatedPartialRef, ContributorJoin,
        ExecutionSession, PartialContributorResult, WorkAssignment, WorkKind,
    },
    verify_archive_manifest, ArchiveManifest, ArchiveMode, ArchiveOptions,
    ArchiveStatusRequirement, ContributorSigner, ContributorStateStore,
    CoordinatorSigner, RestoreError, RestoreOptions, RestoreSource,
    SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.15-restore-coord-seed32!";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.15-restore-contrib-a-32!";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.15-restore-contrib-b-32!";
const NOW_UTC: &str = "2026-06-03T01:00:00Z";
const FAR_FUTURE: &str = "2026-12-31T23:59:59Z";

fn fresh_dir() -> tempfile::TempDir {
    tempfile::TempDir::new().expect("tempdir")
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

fn seed_aggregated_session(
    store: &ContributorStateStore,
) -> ExecutionSession {
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
        store.mark_seen(StateNamespace::Assignments, &marker).unwrap();
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
    }
    store
        .write_verified_json(StateObjectKind::Aggregate, &session.session_id, &agg)
        .unwrap();
    store
        .mark_seen(StateNamespace::Sessions, &session.session_id)
        .unwrap();
    store
        .mark_seen(StateNamespace::Aggregates, &session.session_id)
        .unwrap();
    session
}

fn open_store(root: &std::path::Path) -> ContributorStateStore {
    let (s, _) = ContributorStateStore::open(root, false, NOW_UTC).unwrap();
    s
}

fn archive_to(
    store: &ContributorStateStore,
    archive_root: &std::path::Path,
    session_id: &str,
    mode: ArchiveMode,
    include_results: bool,
) -> ArchiveManifest {
    archive_session(
        store,
        &ArchiveOptions {
            session_id,
            archive_dir: archive_root,
            mode,
            require_status: ArchiveStatusRequirement::Complete,
            include_results,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap()
}

fn manifest_path(archive_root: &std::path::Path, session_id: &str) -> PathBuf {
    archive_root.join(session_id).join("manifest.json")
}

fn read_manifest(archive_root: &std::path::Path, session_id: &str) -> ArchiveManifest {
    let bytes = std::fs::read(manifest_path(archive_root, session_id)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn write_manifest(
    archive_root: &std::path::Path,
    session_id: &str,
    manifest: &ArchiveManifest,
) {
    let bytes = serde_json::to_vec_pretty(manifest).unwrap();
    std::fs::write(manifest_path(archive_root, session_id), bytes).unwrap();
}

fn restore_root(state_root: &std::path::Path) -> ContributorStateStore {
    // The destination state-dir for restore — distinct from the
    // source state-dir we archived FROM, so the test exercises a
    // true "import" rather than a self-overwrite.
    let _ = state_root; // kept for symmetry with archive tests
    open_store(state_root)
}

fn count_files(dir: &std::path::Path) -> usize {
    fn rec(d: &std::path::Path, n: &mut usize) {
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

// ── 1. Dry-run writes nothing ──────────────────────────────────

#[test]
fn dry_run_validates_manifest_without_touching_state_dir() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );

    let dest_store = restore_root(dest_state.path());
    // Count files BEFORE the restore (state-store open created
    // `meta/state_version.json` + the empty top-level subdirs).
    let before = count_files(dest_state.path());
    let report = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();
    assert_eq!(report.mode, "dry_run");
    assert_eq!(report.files_restored, 0);
    // No new files written.
    assert_eq!(count_files(dest_state.path()), before);
}

// ── 2. Verify-only hashes intact archive without writing ───────

#[test]
fn verify_only_hashes_intact_archive_without_writing() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );

    let dest_store = restore_root(dest_state.path());
    let before = count_files(dest_state.path());
    let report = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: true,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();
    assert_eq!(report.mode, "verify_only");
    assert_eq!(report.files_restored, 0);
    assert_eq!(count_files(dest_state.path()), before);
}

// ── 3. Restore round-trips bytes ───────────────────────────────

#[test]
fn restore_round_trips_bytes_byte_for_byte() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);

    let manifest = archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let dest_store = restore_root(dest_state.path());
    let _ = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();

    // For each manifest entry, the destination file must exist
    // and its BLAKE3 must match the manifest.
    for entry in &manifest.files {
        let dest = dest_state.path().join(&entry.source_relative);
        assert!(dest.is_file(), "missing destination: {entry:?}");
        let bytes = std::fs::read(&dest).unwrap();
        let mut got = String::with_capacity(64);
        for b in blake3::hash(&bytes).as_bytes() {
            got.push_str(&format!("{b:02x}"));
        }
        assert_eq!(got, entry.blake3_hex, "{entry:?}");
    }
}

// ── 4. Restore after archive --move recreates the state subtree ──

#[test]
fn restore_after_archive_move_recreates_full_state_subtree() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);

    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Move,
        false,
    );
    // Source state-dir no longer has the session subtree.
    let src_session_dir = archive_source
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id);
    assert!(!src_session_dir.exists());

    // Restore into a FRESH state-dir.
    let dest_state = fresh_dir();
    let dest_store = restore_root(dest_state.path());
    let report = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();
    assert!(report.files_restored > 0);
    let dest_session_dir = dest_state
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id);
    assert!(dest_session_dir.is_dir());
    assert!(dest_session_dir.join("session.json").is_file());
}

// ── 5. Restored session accepted by load_verified_restart_snapshot ──

#[test]
fn restored_session_is_accepted_by_load_verified_restart_snapshot() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let dest_store = restore_root(dest_state.path());
    restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();

    // Stage 12.13 restart preload must accept the restored
    // session — the bytes are canonical-equivalent, so every
    // verifier passes.
    let (snapshot, report) =
        omni_contributor::load_verified_restart_snapshot(&dest_store).unwrap();
    assert_eq!(report.sessions_accepted, 1);
    assert_eq!(report.assignments_accepted, 2);
    assert!(snapshot.sessions.contains_key(&session.session_id));
    assert_eq!(
        snapshot
            .assignments_by_session
            .get(&session.session_id)
            .map(|v| v.len())
            .unwrap_or(0),
        2
    );
    // No rejections for THIS session.
    let any_rej_for_us = report
        .rejection_notes
        .iter()
        .any(|n| n.contains(&session.session_id));
    assert!(!any_rej_for_us, "{:?}", report.rejection_notes);
}

// ── 6. BLAKE3 mismatch refusal ─────────────────────────────────

#[test]
fn refuses_when_archive_file_blake3_mismatches_manifest() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    let manifest = archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );

    // Tamper with the first non-marker file in the archive.
    let target = manifest
        .files
        .iter()
        .find(|f| f.archive_relative.starts_with("verified/"))
        .unwrap();
    let path = archive_dir
        .path()
        .join(&session.session_id)
        .join(&target.archive_relative);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(b'!');
    std::fs::write(&path, &bytes).unwrap();

    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: true,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(matches!(err, RestoreError::BlakeMismatch { .. }));
}

// ── 7. Missing archive file refusal ────────────────────────────

#[test]
fn refuses_when_manifest_references_missing_archive_file() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    let manifest = archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    // Delete one archive file.
    let target = manifest
        .files
        .iter()
        .find(|f| f.archive_relative.starts_with("verified/"))
        .unwrap();
    std::fs::remove_file(
        archive_dir
            .path()
            .join(&session.session_id)
            .join(&target.archive_relative),
    )
    .unwrap();
    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: true,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(matches!(err, RestoreError::ManifestFileMissing { .. }));
}

// ── 8. Path traversal refusal ──────────────────────────────────

#[test]
fn refuses_path_traversal_in_manifest_entry() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    // Hand-edit the manifest's first file to a traversal path.
    let mut m = read_manifest(archive_dir.path(), &session.session_id);
    m.files[0].source_relative = "verified/sessions/../../etc/passwd".into();
    write_manifest(archive_dir.path(), &session.session_id, &m);
    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(matches!(err, RestoreError::UnsafeRelativePath { .. }));
}

// ── 9. Absolute path refusal ───────────────────────────────────

#[test]
fn refuses_absolute_path_in_manifest_entry() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let mut m = read_manifest(archive_dir.path(), &session.session_id);
    m.files[0].source_relative = "/etc/passwd".into();
    write_manifest(archive_dir.path(), &session.session_id, &m);
    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(matches!(err, RestoreError::UnsafeRelativePath { .. }));
}

// ── 10. Disallowed other-session path refusal ─────────────────

#[test]
fn refuses_path_outside_session_whitelist() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let mut m = read_manifest(archive_dir.path(), &session.session_id);
    // Whitelist requires `verified/sessions/<sid>/...` etc.
    // `verified/something-else/...` fails the prefix gate.
    m.files[0].source_relative = "verified/something-else/x.json".into();
    write_manifest(archive_dir.path(), &session.session_id, &m);
    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(matches!(err, RestoreError::DisallowedRelativePath { .. }));
}

// ── 11. Existing destination refusal (all-or-nothing preflight) ──

#[test]
fn refuses_when_any_destination_exists_without_overwrite_existing() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    let manifest = archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let dest_store = restore_root(dest_state.path());
    // Plant a single existing destination file via
    // write_archived_bytes (the same primitive the restore would
    // use). Pick the first verified file.
    let target = manifest
        .files
        .iter()
        .find(|f| f.source_relative.starts_with("verified/"))
        .unwrap();
    dest_store
        .write_archived_bytes(&target.source_relative, b"preexisting", true)
        .unwrap();
    // Capture the file count BEFORE the restore attempt.
    let before = count_files(dest_state.path());

    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(matches!(err, RestoreError::DestinationExists { .. }));
    // Critical: NO new files were written. The preflight is
    // all-or-nothing.
    assert_eq!(count_files(dest_state.path()), before);
}

// ── 12. Overwrite-existing replaces files ──────────────────────

#[test]
fn overwrite_existing_replaces_destination_files() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    let manifest = archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let dest_store = restore_root(dest_state.path());
    // Plant a junk file at every destination — overwrite must
    // replace every one of them with the archive's bytes.
    for entry in &manifest.files {
        dest_store
            .write_archived_bytes(&entry.source_relative, b"junk", true)
            .unwrap();
    }
    restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: false,
            overwrite_existing: true,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();
    // Each destination now matches the manifest's BLAKE3.
    for entry in &manifest.files {
        let dest = dest_state.path().join(&entry.source_relative);
        let bytes = std::fs::read(&dest).unwrap();
        let mut got = String::with_capacity(64);
        for b in blake3::hash(&bytes).as_bytes() {
            got.push_str(&format!("{b:02x}"));
        }
        assert_eq!(got, entry.blake3_hex, "{entry:?}");
    }
}

// ── 13. Result-link skipped unless --include-results ───────────

#[test]
fn result_link_skipped_unless_include_results_true() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    // Plant a result-link in the source state-dir so the
    // archive picks it up.
    let link_dir = archive_source.path().join("results").join("result-links");
    std::fs::create_dir_all(&link_dir).unwrap();
    std::fs::write(
        link_dir.join(format!("{}.link.json", session.posted_id)),
        b"{\"link\":\"x\"}",
    )
    .unwrap();
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        /* include_results = */ true,
    );

    // Restore WITHOUT --include-results → link must be skipped.
    let dest_store = restore_root(dest_state.path());
    let report = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();
    assert_eq!(report.files_skipped_results, 1);
    let dest_link = dest_state
        .path()
        .join("results")
        .join("result-links")
        .join(format!("{}.link.json", session.posted_id));
    assert!(!dest_link.exists());
}

// ── 14. Result-link restored when --include-results ────────────

#[test]
fn result_link_restored_when_include_results_true() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    let link_dir = archive_source.path().join("results").join("result-links");
    std::fs::create_dir_all(&link_dir).unwrap();
    std::fs::write(
        link_dir.join(format!("{}.link.json", session.posted_id)),
        b"{\"link\":\"x\"}",
    )
    .unwrap();
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        true,
    );
    let dest_store = restore_root(dest_state.path());
    restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: false,
            verify_only: false,
            overwrite_existing: false,
            include_results: true,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();
    let dest_link = dest_state
        .path()
        .join("results")
        .join("result-links")
        .join(format!("{}.link.json", session.posted_id));
    assert!(dest_link.is_file());
}

// ── 15. Incompatible source_state_version ──────────────────────

#[test]
fn refuses_incompatible_source_state_version() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let mut m = read_manifest(archive_dir.path(), &session.session_id);
    m.source_state_version = 99;
    write_manifest(archive_dir.path(), &session.session_id, &m);
    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        RestoreError::IncompatibleSourceStateVersion { .. }
    ));
}

// ── 16. Unsupported manifest schema_version ────────────────────

#[test]
fn refuses_unsupported_manifest_schema_version() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let mut m = read_manifest(archive_dir.path(), &session.session_id);
    m.schema_version = 99;
    write_manifest(archive_dir.path(), &session.session_id, &m);
    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        RestoreError::UnsupportedManifestVersion { .. }
    ));
}

// ── 17. Session-id mismatch ────────────────────────────────────

#[test]
fn refuses_when_manifest_session_id_mismatches_supplied_session_id() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    // Operator supplies a different `--session-id` than the
    // directory holds. The session-binding check refuses.
    let fake_id = "00".repeat(32);
    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &fake_id,
            },
            dry_run: true,
            verify_only: false,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    // Either ArchiveNotFound (fake_id has no directory) OR
    // SessionIdMismatch — depends on whether the operator
    // pointed at the right archive dir but the wrong session.
    // Here the archive_dir has the real session, so we're
    // looking for ArchiveNotFound (the resolver builds
    // archive_dir/<fake_id>/, which doesn't exist).
    assert!(
        matches!(err, RestoreError::ArchiveNotFound { .. })
            || matches!(err, RestoreError::SessionIdMismatch { .. })
    );
}

// ── 18. verify_archive_manifest helper ─────────────────────────

#[test]
fn verify_archive_manifest_returns_typed_manifest_for_valid_archive() {
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    let original = archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let m = verify_archive_manifest(&RestoreSource::ArchiveRoot {
        archive_dir: archive_dir.path(),
        session_id: &session.session_id,
    })
    .unwrap();
    assert_eq!(m, original);
}

// ── Stage 12.15 review fix — verify-only wins when paired with dry-run ──

#[test]
fn dry_run_plus_verify_only_still_catches_blake3_mismatch() {
    // Reviewer-flagged case: passing both `--dry-run` and
    // `--verify-only` previously took the dry-run gate and
    // skipped the BLAKE3 walk silently, returning a
    // `verify_only` mode tag without actually hashing any
    // archive bytes. The fix runs verification whenever
    // `verify_only` is true regardless of `dry_run`. This test
    // would fail silently against the pre-fix gate.
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    let manifest = archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    // Tamper with the first verified file in the archive.
    let target = manifest
        .files
        .iter()
        .find(|f| f.archive_relative.starts_with("verified/"))
        .unwrap();
    let path = archive_dir
        .path()
        .join(&session.session_id)
        .join(&target.archive_relative);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.push(b'!');
    std::fs::write(&path, &bytes).unwrap();

    let dest_store = restore_root(dest_state.path());
    // BOTH flags set.
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: true,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, RestoreError::BlakeMismatch { .. }),
        "verify-only must catch BLAKE3 mismatch even when --dry-run is also set; \
         got {err:?}"
    );
}

#[test]
fn dry_run_plus_verify_only_still_catches_missing_archive_file() {
    // Same scenario for the missing-file refusal: with verify-only
    // dominant, a missing archive file MUST raise
    // ManifestFileMissing even if dry_run is also set.
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    let manifest = archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let target = manifest
        .files
        .iter()
        .find(|f| f.archive_relative.starts_with("verified/"))
        .unwrap();
    std::fs::remove_file(
        archive_dir
            .path()
            .join(&session.session_id)
            .join(&target.archive_relative),
    )
    .unwrap();
    let dest_store = restore_root(dest_state.path());
    let err = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: true,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, RestoreError::ManifestFileMissing { .. }),
        "verify-only must catch missing archive file even when --dry-run is \
         also set; got {err:?}"
    );
}

#[test]
fn dry_run_plus_verify_only_accepts_intact_archive_and_writes_nothing() {
    // Positive control: passing both flags on a clean archive
    // returns Ok, walks every BLAKE3 successfully, but writes
    // nothing to the state-dir (verify-only's no-write
    // contract is preserved).
    let archive_source = fresh_dir();
    let archive_dir = fresh_dir();
    let dest_state = fresh_dir();
    let src_store = open_store(archive_source.path());
    let session = seed_aggregated_session(&src_store);
    archive_to(
        &src_store,
        archive_dir.path(),
        &session.session_id,
        ArchiveMode::Copy,
        false,
    );
    let dest_store = restore_root(dest_state.path());
    let before = count_files(dest_state.path());
    let report = restore_session_archive(
        &dest_store,
        &RestoreOptions {
            source: RestoreSource::ArchiveRoot {
                archive_dir: archive_dir.path(),
                session_id: &session.session_id,
            },
            dry_run: true,
            verify_only: true,
            overwrite_existing: false,
            include_results: false,
            now_utc: NOW_UTC,
        },
    )
    .unwrap();
    assert_eq!(
        report.mode, "verify_only",
        "mode resolution: verify-only wins over dry-run"
    );
    assert_eq!(report.files_restored, 0);
    assert_eq!(count_files(dest_state.path()), before);
}
