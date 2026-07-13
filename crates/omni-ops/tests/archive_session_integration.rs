//! Stage 12.14 — integration tests for `archive_session`.
//!
//! Each test stands up a temp state-dir + a temp archive-dir,
//! seeds a session chain, and exercises one accept/refuse path
//! of `archive_session`. The session chain helpers
//! (`build_session`, `build_join`, `build_assignment`, etc.) are
//! locally duplicated rather than imported across test files —
//! this keeps the suite self-contained and matches the existing
//! Stage 12.10–12.13 test conventions.
//!
//! Hard rules pinned: no protocol envelope, no canonical-byte
//! changes, no schema_version bump, no STATE_VERSION bump,
//! no omni-net topic changes, no SNIP wire change, no chain /
//! proof / payment / marketplace surfaces.

use std::path::PathBuf;

use omni_contributor::{
    canonical::{
        aggregated_result_signing_input, assignment_id_hex,
        canonical_partial_result_bytes, contributor_join_signing_input,
        execution_session_signing_input, hex_lower,
        partial_result_signing_input, session_id_hex, supersession_id_hex,
        work_assignment_signing_input,
        work_assignment_supersession_signing_input,
    },
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    session::{
        AggregatedContributorResult, AggregatedPartialRef, ContributorJoin,
        ExecutionSession, PartialContributorResult, WorkAssignment, WorkKind,
    },
    supersession::{SupersessionReason, WorkAssignmentSupersession},
    ContributorSigner, ContributorStateStore, CoordinatorSigner,
    StateNamespace, StateObjectKind, SESSION_SCHEMA_VERSION,
    SUPERSESSION_SCHEMA_VERSION,
};
use omni_ops::{
    archive_session, ArchiveError, ArchiveManifest, ArchiveMode,
    ArchiveOptions, ArchiveStatusRequirement, RepairError as _RepairError,
    ARCHIVE_MANIFEST_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.14-archive-coord-seed32!";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.14-archive-contrib-a-32!";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.14-archive-contrib-b-32!";
const NOW_UTC: &str = "2026-06-03T01:00:00Z";
const FAR_FUTURE: &str = "2026-12-31T23:59:59Z";
const PAST: &str = "2026-05-01T00:00:00Z";

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

fn build_supersession(
    session: &ExecutionSession,
    coord: &CoordinatorSigner,
    superseded: Vec<String>,
    replacement: Vec<String>,
) -> WorkAssignmentSupersession {
    let mut superseded = superseded;
    superseded.sort();
    let mut replacement = replacement;
    replacement.sort();
    let mut s = WorkAssignmentSupersession {
        schema_version: SUPERSESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        supersession_id: String::new(),
        superseded_assignment_ids: superseded,
        replacement_assignment_ids: replacement,
        reason: SupersessionReason::MissingPartial,
        created_at_utc: "2026-06-02T00:00:25Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    s.supersession_id = supersession_id_hex(&s).unwrap();
    let si = work_assignment_supersession_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn seed_aggregated_session(
    store: &ContributorStateStore,
) -> (ExecutionSession, Vec<WorkAssignment>) {
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

    (session, vec![asn_0, asn_1])
}

fn open_store(root: &std::path::Path) -> ContributorStateStore {
    let (s, _) =
        ContributorStateStore::open(root, false, NOW_UTC).unwrap();
    s
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

fn manifest_for(session_id: &str, archive_root: &std::path::Path) -> ArchiveManifest {
    let p = archive_root.join(session_id).join("manifest.json");
    let bytes = std::fs::read(&p).expect("read manifest");
    serde_json::from_slice(&bytes).expect("parse manifest")
}

// ── 1. Dry-run writes nothing ──────────────────────────────────

#[test]
fn dry_run_archives_nothing() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let (session, _) = seed_aggregated_session(&store);
    let archive_before = count_files(archive_root.path());

    let manifest = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: true,
        },
    )
    .unwrap();

    assert_eq!(manifest.schema_version, ARCHIVE_MANIFEST_SCHEMA_VERSION);
    assert!(!manifest.files.is_empty());
    // No files written.
    assert_eq!(count_files(archive_root.path()), archive_before);
    // Source state-dir still has the session.
    let session_dir = state_root
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id);
    assert!(session_dir.is_dir());
}

// ── 2. Copy writes manifest + verified BLAKE3 ──────────────────

#[test]
fn copy_writes_manifest_and_verifies_blake3() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let (session, _) = seed_aggregated_session(&store);

    let manifest = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();

    // Manifest written.
    let on_disk = manifest_for(&session.session_id, archive_root.path());
    assert_eq!(on_disk, manifest);
    // Every file in the manifest exists in the archive AND its
    // BLAKE3 matches the manifest entry (the archiver already
    // verified, but pin the file-on-disk shape here too).
    for entry in &on_disk.files {
        let path = archive_root
            .path()
            .join(&session.session_id)
            .join(&entry.archive_relative);
        let bytes = std::fs::read(&path).expect("read archived file");
        let mut got = String::with_capacity(64);
        for b in blake3::hash(&bytes).as_bytes() {
            got.push_str(&format!("{b:02x}"));
        }
        assert_eq!(got, entry.blake3_hex, "{}", entry.archive_relative);
    }
    // session.json IS in the manifest.
    assert!(on_disk
        .files
        .iter()
        .any(|f| f.archive_relative.ends_with("/session.json")));
}

// ── 3. Copy preserves source bit-for-bit ───────────────────────

#[test]
fn copy_preserves_source_state_dir() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let (session, _) = seed_aggregated_session(&store);

    // Capture every file's bytes in the session subtree before
    // archive.
    fn capture(
        root: &std::path::Path,
        dir: &std::path::Path,
        out: &mut Vec<(PathBuf, Vec<u8>)>,
    ) {
        if let Ok(es) = std::fs::read_dir(dir) {
            for e in es.flatten() {
                let p = e.path();
                if p.is_dir() {
                    capture(root, &p, out);
                } else if p.is_file() {
                    let rel = p.strip_prefix(root).unwrap().to_path_buf();
                    let bytes = std::fs::read(&p).unwrap();
                    out.push((rel, bytes));
                }
            }
        }
    }
    let mut before: Vec<(PathBuf, Vec<u8>)> = Vec::new();
    capture(
        state_root.path(),
        &state_root.path().join("verified"),
        &mut before,
    );

    let _ = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();

    let mut after: Vec<(PathBuf, Vec<u8>)> = Vec::new();
    capture(
        state_root.path(),
        &state_root.path().join("verified"),
        &mut after,
    );
    before.sort_by(|a, b| a.0.cmp(&b.0));
    after.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(before, after, "source files must be untouched by --copy");
}

// ── 4. Move removes source + seen markers ──────────────────────

#[test]
fn move_removes_source_session_subtree_and_seen_markers() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let (session, asns) = seed_aggregated_session(&store);

    let session_dir = state_root
        .path()
        .join("verified")
        .join("sessions")
        .join(&session.session_id);
    assert!(session_dir.is_dir());
    assert!(store
        .is_seen(StateNamespace::Sessions, &session.session_id)
        .unwrap());

    let _ = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Move,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();

    // Subtree gone.
    assert!(!session_dir.exists());
    // Session-keyed seen markers cascaded.
    assert!(!store
        .is_seen(StateNamespace::Sessions, &session.session_id)
        .unwrap());
    assert!(!store
        .is_seen(StateNamespace::Aggregates, &session.session_id)
        .unwrap());
    for a in &asns {
        let marker = format!("{}--{}", session.session_id, a.assignment_id);
        assert!(!store
            .is_seen(StateNamespace::Assignments, &marker)
            .unwrap());
    }
    // Manifest IS in the archive (move ran cascade only after
    // manifest write).
    let manifest_path = archive_root
        .path()
        .join(&session.session_id)
        .join("manifest.json");
    assert!(manifest_path.is_file());
}

// ── 5. Refusal: InProgress under default require_status=complete ──

#[test]
fn refuses_in_progress_when_require_status_complete() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = build_session(FAR_FUTURE);
    let join = build_join(&session, &contrib);
    let asn = build_assignment(&session, &contrib, &coord, 0);
    // No partial → status InProgress (with active-missing).
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session.session_id.clone(),
            },
            &join.contributor_pubkey_hex,
            &join,
        )
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Assignment {
                session_id: session.session_id.clone(),
            },
            &asn.assignment_id,
            &asn,
        )
        .unwrap();

    let err = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap_err();
    match err {
        ArchiveError::StatusRequirementUnmet { got, requirement } => {
            assert_eq!(got, "InProgress");
            assert_eq!(requirement, "complete");
        }
        other => panic!("expected StatusRequirementUnmet, got {other:?}"),
    }
    // Nothing written.
    assert_eq!(count_files(archive_root.path()), 0);
}

// ── 6. Any accepts InvalidState (escape valve) ─────────────────

#[test]
fn require_status_any_accepts_invalid_state() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let wrong = CoordinatorSigner::from_seed_bytes(&[0xAB; 32]).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = build_session(FAR_FUTURE);
    let join = build_join(&session, &contrib);
    // Forge the assignment with the wrong coord — reporter
    // emits InvalidState + invalid_assignment.
    let bad = build_assignment(&session, &contrib, &wrong, 0);
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session.session_id.clone(),
            },
            &join.contributor_pubkey_hex,
            &join,
        )
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Assignment {
                session_id: session.session_id.clone(),
            },
            &bad.assignment_id,
            &bad,
        )
        .unwrap();
    let _ = (&coord,); // silence unused; coord intentionally unused in this branch.

    let manifest = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Any,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(manifest.session_overall_status, "InvalidState");
}

// ── 7. Missing session ─────────────────────────────────────────

#[test]
fn refuses_when_session_not_in_state_dir() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let session_id = "aa".repeat(32);
    let err = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert!(matches!(err, ArchiveError::SessionNotPresent { .. }));
}

// ── 8. Existing archive dir refusal ────────────────────────────

#[test]
fn refuses_when_archive_dir_already_has_this_session() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let (session, _) = seed_aggregated_session(&store);

    // First archive succeeds.
    archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();

    // Second archive into the same dir refuses.
    let err = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap_err();
    assert!(matches!(err, ArchiveError::ArchiveAlreadyExists { .. }));
}

// ── 9. --include-results copies the matching result-link ───────

#[test]
fn include_results_copies_posted_result_link_only() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let (session, _) = seed_aggregated_session(&store);

    // Plant a result-link file matching session.posted_id, AND
    // an unrelated contributor-result.
    let link_dir = state_root.path().join("results").join("result-links");
    std::fs::create_dir_all(&link_dir).unwrap();
    let link_path = link_dir.join(format!("{}.link.json", session.posted_id));
    std::fs::write(&link_path, b"{\"link\":\"ok\"}").unwrap();
    let cr_dir = state_root.path().join("results").join("contributor-results");
    std::fs::create_dir_all(&cr_dir).unwrap();
    let cr_path = cr_dir.join("some-job.json");
    std::fs::write(&cr_path, b"{\"unrelated\":true}").unwrap();

    let manifest = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: true,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();

    let has_link = manifest
        .files
        .iter()
        .any(|f| f.archive_relative.contains("result-links"));
    let has_contrib_result = manifest
        .files
        .iter()
        .any(|f| f.archive_relative.contains("contributor-results"));
    assert!(has_link, "result-link must be archived under --include-results");
    assert!(
        !has_contrib_result,
        "contributor-results must NOT be archived (per-job, not per-session)"
    );
}

// ── 10. Other sessions untouched ───────────────────────────────

#[test]
fn does_not_touch_other_sessions() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let (session_a, _) = seed_aggregated_session(&store);
    // Build a second, independent session with its own coord
    // seed so the session_id differs.
    let coord_b = CoordinatorSigner::from_seed_bytes(&[0xBB; 32]).unwrap();
    let contrib_b = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let mut session_b = build_session(FAR_FUTURE);
    session_b.coordinator_pubkey_hex = coord_b.pubkey_hex();
    session_b.session_id = session_id_hex(&session_b).unwrap();
    let si = execution_session_signing_input(&session_b).unwrap();
    session_b.coordinator_signature_hex = coord_b.sign_hex(&si);
    store
        .write_verified_json(
            StateObjectKind::Session,
            &session_b.session_id,
            &session_b,
        )
        .unwrap();
    let join_b = build_join(&session_b, &contrib_b);
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session_b.session_id.clone(),
            },
            &join_b.contributor_pubkey_hex,
            &join_b,
        )
        .unwrap();

    archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session_a.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Move,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();

    let b_dir = state_root
        .path()
        .join("verified")
        .join("sessions")
        .join(&session_b.session_id);
    assert!(b_dir.is_dir(), "session_b must NOT be touched by archiving session_a");
    let b_archive = archive_root.path().join(&session_b.session_id);
    assert!(
        !b_archive.exists(),
        "session_b must NOT appear in the archive root"
    );
}

// ── 11. Manifest serde round-trip ──────────────────────────────

#[test]
fn manifest_serde_roundtrip() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let (session, _) = seed_aggregated_session(&store);
    let m = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::Complete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: true,
        },
    )
    .unwrap();
    let json = serde_json::to_vec_pretty(&m).unwrap();
    let parsed: ArchiveManifest = serde_json::from_slice(&json).unwrap();
    assert_eq!(parsed, m);
}

// ── 12. ExpiredIncomplete acceptance under expired-incomplete ──

#[test]
fn expired_incomplete_accepted_under_expired_incomplete_requirement() {
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = build_session(PAST);
    let join = build_join(&session, &contrib);
    let asn = build_assignment(&session, &contrib, &coord, 0);
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Join {
                session_id: session.session_id.clone(),
            },
            &join.contributor_pubkey_hex,
            &join,
        )
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::Assignment {
                session_id: session.session_id.clone(),
            },
            &asn.assignment_id,
            &asn,
        )
        .unwrap();
    let manifest = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            require_status: ArchiveStatusRequirement::ExpiredIncomplete,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();
    assert_eq!(manifest.session_overall_status, "ExpiredIncomplete");
}

// ── 13. Supersession files + seen markers are archived ────────

#[test]
fn supersession_files_and_seen_markers_are_archived() {
    // Pins that the archive walker covers the Stage 12.11
    // `supersessions/` subdirectory + the
    // `seen/assignment-supersessions/` namespace. The chain
    // shape itself is intentionally minimal (no partials, no
    // aggregate) — this test exercises the file-walker contract
    // under `--require-status any` (the escape valve), not the
    // status decision tree.
    let state_root = fresh_dir();
    let archive_root = fresh_dir();
    let store = open_store(state_root.path());
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let session = build_session(FAR_FUTURE);
    // A minimal supersession (schema-valid; references two
    // synthetic ids that are NOT in the chain, so
    // `verify_assignment_supersession` rejects on reference
    // resolution — but the file walker still picks it up).
    let sup = build_supersession(
        &session,
        &coord,
        vec!["aa".repeat(32)],
        vec!["cd".repeat(32)],
    );
    store
        .write_verified_json(StateObjectKind::Session, &session.session_id, &session)
        .unwrap();
    store
        .write_verified_json(
            StateObjectKind::AssignmentSupersession {
                session_id: session.session_id.clone(),
            },
            &sup.supersession_id,
            &sup,
        )
        .unwrap();
    let sup_marker = format!("{}--{}", session.session_id, sup.supersession_id);
    store
        .mark_seen(StateNamespace::AssignmentSupersessions, &sup_marker)
        .unwrap();

    let manifest = archive_session(
        &store,
        &ArchiveOptions {
            session_id: &session.session_id,
            archive_dir: archive_root.path(),
            mode: ArchiveMode::Copy,
            // `any` because the synthetic supersession refers to
            // unknown ids → InvalidState; this test exercises the
            // file walker, not the status policy.
            require_status: ArchiveStatusRequirement::Any,
            include_results: false,
            now_utc: NOW_UTC,
            dry_run: false,
        },
    )
    .unwrap();
    // Supersession body archived.
    assert!(
        manifest
            .files
            .iter()
            .any(|f| f.archive_relative.contains("/supersessions/")),
        "supersession body must be archived under verified/.../supersessions/"
    );
    // Supersession seen marker archived.
    assert!(
        manifest
            .files
            .iter()
            .any(|f| f
                .archive_relative
                .contains("seen/assignment-supersessions/")),
        "supersession seen marker must be archived"
    );
}

// silence unused import lint for RepairError (kept available for
// future tests of cross-policy interactions).
#[allow(dead_code)]
fn _ensure_repair_error_in_scope(_x: _RepairError) {}
