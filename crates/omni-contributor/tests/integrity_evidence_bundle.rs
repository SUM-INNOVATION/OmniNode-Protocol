//! Stage 12.22 — integration tests for the local-only
//! integrity evidence bundle.

use std::fs;
use std::path::{Path, PathBuf};

use omni_contributor::{
    build_integrity_evidence_bundle, read_integrity_evidence_bundle_from_path,
    verify_integrity_evidence_bundle, write_integrity_evidence_bundle_atomic,
    BundleArtifactKind, BundleBuilderInput, BundleBuilderOptions,
    BundleEntryOutcome, BundleVerifyOptions, EvidenceBundleError,
    IntegrityEvidenceBundle, BUNDLE_ENTRY_MAX_BYTES, BUNDLE_LABEL_MAX,
    BUNDLE_MAX_ENTRIES, BUNDLE_NOTES_MAX,
    INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
};

const NOW_UTC: &str = "2026-06-12T12:00:00Z";

fn write_file(dir: &Path, rel: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&p, bytes).unwrap();
    p
}

fn canon(p: &Path) -> PathBuf {
    fs::canonicalize(p).unwrap()
}

fn opts<'a>(base_dir: &'a Path) -> BundleBuilderOptions<'a> {
    BundleBuilderOptions {
        now_utc: NOW_UTC,
        base_dir,
        label: None,
        notes: None,
    }
}

fn input<'a>(
    kind: BundleArtifactKind,
    path: &'a Path,
) -> BundleBuilderInput<'a> {
    BundleBuilderInput { artifact_kind: kind, path }
}

// ── 1. Build + verify happy path (3 mixed kinds) ──────────────

#[test]
fn build_and_verify_happy_path_three_kinds() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "baseline.json", b"{\"schema_version\":1}");
    write_file(base, "baseline.signed.json", b"signed-baseline-bytes");
    write_file(base, "diff.signed.json", b"signed-diff-bytes");

    let inputs = vec![
        input(
            BundleArtifactKind::StateIntegrityReport,
            Path::new("baseline.json"),
        ),
        input(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        ),
        input(
            BundleArtifactKind::SignedStateIntegrityDiff,
            Path::new("diff.signed.json"),
        ),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    assert_eq!(bundle.schema_version, INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION);
    assert_eq!(bundle.entries.len(), 3);

    let report =
        verify_integrity_evidence_bundle(&bundle, &BundleVerifyOptions::default())
            .unwrap();
    assert!(report.all_ok(), "got: {report:?}");
    assert_eq!(report.counts_ok, 3);
}

// ── 2. Determinism (same inputs + now_utc → byte-identical) ───

#[test]
fn determinism_same_inputs_produce_byte_identical_bundle() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "a.json", b"alpha");
    write_file(base, "b.json", b"bravo");

    let inputs = vec![
        input(BundleArtifactKind::StateIntegrityReport, Path::new("a.json")),
        input(BundleArtifactKind::Other, Path::new("b.json")),
    ];
    let a = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    let b = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    assert_eq!(a, b);
    let ja = serde_json::to_vec(&a).unwrap();
    let jb = serde_json::to_vec(&b).unwrap();
    assert_eq!(ja, jb);
}

// ── 3. Sort order — (kind, path) regardless of input order ────

#[test]
fn entries_sorted_by_kind_then_path() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "z.json", b"z");
    write_file(base, "a.json", b"a");
    write_file(base, "m.json", b"m");

    // Pass in deliberately scrambled order.
    let inputs = vec![
        input(BundleArtifactKind::Other, Path::new("m.json")),
        input(BundleArtifactKind::Other, Path::new("a.json")),
        input(BundleArtifactKind::ArchiveManifest, Path::new("z.json")),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    // archive_manifest < other (alphabetical wire-tag order),
    // then within other: a < m.
    assert_eq!(
        bundle.entries[0].artifact_kind,
        BundleArtifactKind::ArchiveManifest
    );
    assert_eq!(bundle.entries[1].artifact_kind, BundleArtifactKind::Other);
    assert_eq!(bundle.entries[1].path, "a.json");
    assert_eq!(bundle.entries[2].artifact_kind, BundleArtifactKind::Other);
    assert_eq!(bundle.entries[2].path, "m.json");
}

// ── 4. Empty inputs → EmptyBundle ─────────────────────────────

#[test]
fn empty_inputs_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let err = build_integrity_evidence_bundle(&[], &opts(dir.path())).unwrap_err();
    assert!(
        matches!(err, EvidenceBundleError::EmptyBundle),
        "got: {err:?}"
    );
}

// ── 5. Duplicate (kind, path) refused; same-kind-different-path
//      and same-path-different-kind both OK ──────────────────

#[test]
fn duplicate_kind_path_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "x.json", b"x");

    // Same (kind, path) twice → refuse.
    let dup = vec![
        input(BundleArtifactKind::StateIntegrityReport, Path::new("x.json")),
        input(BundleArtifactKind::StateIntegrityReport, Path::new("x.json")),
    ];
    let err = build_integrity_evidence_bundle(&dup, &opts(base)).unwrap_err();
    assert!(
        matches!(err, EvidenceBundleError::DuplicateEntry { .. }),
        "got: {err:?}"
    );
}

#[test]
fn same_kind_different_paths_accepted() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "x.json", b"x");
    write_file(base, "y.json", b"y");
    let inputs = vec![
        input(BundleArtifactKind::Other, Path::new("x.json")),
        input(BundleArtifactKind::Other, Path::new("y.json")),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    assert_eq!(bundle.entries.len(), 2);
}

#[test]
fn same_path_different_kinds_accepted_in_v1() {
    // v1 contract: identity is (kind, path). Same path under
    // two kinds is semantically odd but bytes-only; we allow it.
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "shared.json", b"shared");
    let inputs = vec![
        input(BundleArtifactKind::StateIntegrityReport, Path::new("shared.json")),
        input(BundleArtifactKind::Other, Path::new("shared.json")),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    assert_eq!(bundle.entries.len(), 2);
}

// ── 6. EntryTooLarge refusal at cap (via sparse file) ─────────

#[test]
fn entry_too_large_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    let big = base.join("big.bin");
    let f = fs::File::create(&big).unwrap();
    // Sparse file: only metadata().len() grows; no real bytes
    // are allocated. The builder's size check trips BEFORE
    // std::fs::read so we never actually OOM.
    f.set_len(BUNDLE_ENTRY_MAX_BYTES + 1).unwrap();
    drop(f);

    let inputs = vec![input(BundleArtifactKind::Other, Path::new("big.bin"))];
    let err = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::EntryTooLarge { bytes, max, .. }
            if bytes == BUNDLE_ENTRY_MAX_BYTES + 1 && max == BUNDLE_ENTRY_MAX_BYTES
        ),
        "got: {err:?}"
    );
}

// ── 7. TooManyEntries refusal at cap ──────────────────────────

#[test]
fn too_many_entries_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    // Build BUNDLE_MAX_ENTRIES+1 input descriptors WITHOUT
    // actually creating files — the cap check fires before the
    // FS walk so we don't need to write 1025 files. We DO need
    // the input slice though.
    let paths: Vec<PathBuf> = (0..=BUNDLE_MAX_ENTRIES)
        .map(|i| PathBuf::from(format!("entry-{i}.json")))
        .collect();
    let inputs: Vec<BundleBuilderInput<'_>> = paths
        .iter()
        .map(|p| BundleBuilderInput {
            artifact_kind: BundleArtifactKind::Other,
            path: p.as_path(),
        })
        .collect();
    let err = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::TooManyEntries { count, max }
            if count == BUNDLE_MAX_ENTRIES + 1 && max == BUNDLE_MAX_ENTRIES
        ),
        "got: {err:?}"
    );
}

// ── 8. BundleLabelTooLong / NotesTooLong refusals ─────────────

#[test]
fn bundle_label_too_long_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "x.json", b"x");
    let label = "x".repeat(BUNDLE_LABEL_MAX + 1);
    let mut o = opts(base);
    o.label = Some(&label);
    let inputs = vec![input(BundleArtifactKind::Other, Path::new("x.json"))];
    let err = build_integrity_evidence_bundle(&inputs, &o).unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::BundleLabelTooLong { len, max }
            if len == BUNDLE_LABEL_MAX + 1 && max == BUNDLE_LABEL_MAX
        ),
        "got: {err:?}"
    );
}

#[test]
fn notes_too_long_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "x.json", b"x");
    let notes = "y".repeat(BUNDLE_NOTES_MAX + 1);
    let mut o = opts(base);
    o.notes = Some(&notes);
    let inputs = vec![input(BundleArtifactKind::Other, Path::new("x.json"))];
    let err = build_integrity_evidence_bundle(&inputs, &o).unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::NotesTooLong { len, max }
            if len == BUNDLE_NOTES_MAX + 1 && max == BUNDLE_NOTES_MAX
        ),
        "got: {err:?}"
    );
}

// ── 9. Absolute input NOT under base_dir → PathOutsideBaseDir ─

#[test]
fn absolute_input_outside_base_dir_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    // Create an outside-base file so canonicalize succeeds.
    let outside_dir = tempfile::TempDir::new().unwrap();
    let outside = write_file(outside_dir.path(), "outsider.json", b"outside");
    let inputs = vec![input(BundleArtifactKind::Other, &outside)];
    let err = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap_err();
    assert!(
        matches!(err, EvidenceBundleError::PathOutsideBaseDir { .. }),
        "got: {err:?}"
    );
}

// ── 10. Absolute input UNDER base_dir → recorded relative form

#[test]
fn absolute_input_under_base_dir_is_recorded_relative() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    let abs = write_file(base, "deep/nested/file.json", b"deep");
    let inputs = vec![input(BundleArtifactKind::Other, &abs)];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    assert_eq!(bundle.entries[0].path, "deep/nested/file.json");
    // The recorded base_dir is the canonical form, not whatever
    // string the operator typed.
    let canon_base = canon(base).to_string_lossy().replace('\\', "/");
    assert_eq!(bundle.base_dir, canon_base);
}

// ── 11. Verify happy path → all Ok ────────────────────────────

#[test]
fn verify_all_ok_when_nothing_changed() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "a.json", b"hello");
    write_file(base, "b.json", b"world");
    let inputs = vec![
        input(BundleArtifactKind::Other, Path::new("a.json")),
        input(BundleArtifactKind::Other, Path::new("b.json")),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    let report =
        verify_integrity_evidence_bundle(&bundle, &BundleVerifyOptions::default())
            .unwrap();
    assert!(report.all_ok());
    assert_eq!(report.counts_ok, 2);
    assert!(report.entries.iter().all(|e| matches!(e.outcome, BundleEntryOutcome::Ok)));
}

// ── 12. Hash mismatch — collect-all, others stay Ok ───────────

#[test]
fn hash_mismatch_does_not_short_circuit_other_entries() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    let a = write_file(base, "a.json", b"original-a");
    write_file(base, "b.json", b"original-b");
    let inputs = vec![
        input(BundleArtifactKind::Other, Path::new("a.json")),
        input(BundleArtifactKind::Other, Path::new("b.json")),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    // Flip a byte but KEEP the same length → forces hash check.
    fs::write(&a, b"tampered-A").unwrap();
    let report =
        verify_integrity_evidence_bundle(&bundle, &BundleVerifyOptions::default())
            .unwrap();
    assert!(!report.all_ok());
    assert_eq!(report.counts_hash_mismatch, 1);
    assert_eq!(report.counts_ok, 1);
    let a_entry = report
        .entries
        .iter()
        .find(|e| e.path == "a.json")
        .unwrap();
    assert!(matches!(
        a_entry.outcome,
        BundleEntryOutcome::HashMismatch { .. }
    ));
    let b_entry = report
        .entries
        .iter()
        .find(|e| e.path == "b.json")
        .unwrap();
    assert!(matches!(b_entry.outcome, BundleEntryOutcome::Ok));
}

// ── 13. Size mismatch — cheap pre-check, no hash work ─────────

#[test]
fn size_mismatch_is_reported_without_hash_burn() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    let a = write_file(base, "a.json", b"twelve-bytes");
    write_file(base, "b.json", b"untouched");
    let inputs = vec![
        input(BundleArtifactKind::Other, Path::new("a.json")),
        input(BundleArtifactKind::Other, Path::new("b.json")),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    fs::write(&a, b"short").unwrap(); // size changes 12 → 5
    let report =
        verify_integrity_evidence_bundle(&bundle, &BundleVerifyOptions::default())
            .unwrap();
    let a_entry = report
        .entries
        .iter()
        .find(|e| e.path == "a.json")
        .unwrap();
    assert!(
        matches!(
            a_entry.outcome,
            BundleEntryOutcome::SizeMismatch { expected: 12, got: 5 }
        ),
        "got: {:?}",
        a_entry.outcome
    );
    assert_eq!(report.counts_size_mismatch, 1);
    assert_eq!(report.counts_ok, 1);
}

// ── 14. NotFound — collect-all ────────────────────────────────

#[test]
fn missing_entry_is_not_found_without_short_circuit() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    let a = write_file(base, "a.json", b"a");
    write_file(base, "b.json", b"b");
    let inputs = vec![
        input(BundleArtifactKind::Other, Path::new("a.json")),
        input(BundleArtifactKind::Other, Path::new("b.json")),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    fs::remove_file(&a).unwrap();
    let report =
        verify_integrity_evidence_bundle(&bundle, &BundleVerifyOptions::default())
            .unwrap();
    assert_eq!(report.counts_not_found, 1);
    assert_eq!(report.counts_ok, 1);
}

// ── 15. ReadError (chmod 000) — Unix-only ─────────────────────

#[cfg(unix)]
#[test]
fn unreadable_entry_is_read_error_without_short_circuit() {
    use std::os::unix::fs::PermissionsExt;
    // Skip under root because root bypasses permission checks.
    if nix_is_root() {
        return;
    }
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    let a = write_file(base, "a.json", b"a");
    write_file(base, "b.json", b"b");
    let inputs = vec![
        input(BundleArtifactKind::Other, Path::new("a.json")),
        input(BundleArtifactKind::Other, Path::new("b.json")),
    ];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    let mut perms = fs::metadata(&a).unwrap().permissions();
    perms.set_mode(0o000);
    fs::set_permissions(&a, perms).unwrap();
    let report =
        verify_integrity_evidence_bundle(&bundle, &BundleVerifyOptions::default())
            .unwrap();
    // Restore perms before any assertion failure so cleanup
    // doesn't leak unreadable files.
    let mut restore = fs::metadata(&a).unwrap().permissions();
    restore.set_mode(0o644);
    fs::set_permissions(&a, restore).unwrap();
    assert_eq!(report.counts_read_error, 1);
    assert_eq!(report.counts_ok, 1);
    let a_entry = report
        .entries
        .iter()
        .find(|e| e.path == "a.json")
        .unwrap();
    assert!(
        matches!(a_entry.outcome, BundleEntryOutcome::ReadError { .. }),
        "got: {:?}",
        a_entry.outcome
    );
}

#[cfg(unix)]
fn nix_is_root() -> bool {
    // Best-effort: getuid() == 0.
    // Avoids pulling `nix` as a dep — `libc::getuid()` is
    // available via std on Unix targets via the `libc` re-export.
    // Using std::process::id() is NOT the same; check $USER instead.
    std::env::var("USER").map(|u| u == "root").unwrap_or(false)
}

// ── 16. JSON round-trip preserves bundle ──────────────────────

#[test]
fn json_round_trip_preserves_bundle() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "a.json", b"a");
    let inputs = vec![input(BundleArtifactKind::Other, Path::new("a.json"))];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    let s = serde_json::to_string(&bundle).unwrap();
    let round: IntegrityEvidenceBundle = serde_json::from_str(&s).unwrap();
    assert_eq!(round, bundle);
}

// ── 17. Atomic writer round-trips via disk ────────────────────

#[test]
fn write_atomic_round_trips_via_disk() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "a.json", b"a");
    let inputs = vec![input(BundleArtifactKind::Other, Path::new("a.json"))];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    let out_dir = tempfile::TempDir::new().unwrap();
    let out = out_dir.path().join("bundle.json");
    write_integrity_evidence_bundle_atomic(&bundle, &out).unwrap();
    let round = read_integrity_evidence_bundle_from_path(&out).unwrap();
    assert_eq!(round, bundle);
}

// ── 18. Wrapper schema_version != 1 refused ───────────────────

#[test]
fn wrapper_schema_v2_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "a.json", b"a");
    let inputs = vec![input(BundleArtifactKind::Other, Path::new("a.json"))];
    let mut bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    bundle.schema_version = 2;
    let err = verify_integrity_evidence_bundle(&bundle, &BundleVerifyOptions::default())
        .unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::UnsupportedSchemaVersion { got: 2, expected: 1 }
        ),
        "got: {err:?}"
    );
}

// ── 19. --base-dir override at verify time relocates root ─────

#[test]
fn verify_base_dir_override_relocates_resolution_root() {
    let dir_a = tempfile::TempDir::new().unwrap();
    let dir_b = tempfile::TempDir::new().unwrap();
    let base_a = dir_a.path();
    let base_b = dir_b.path();
    write_file(base_a, "a.json", b"contents");
    write_file(base_b, "a.json", b"contents"); // same contents
    let inputs = vec![input(BundleArtifactKind::Other, Path::new("a.json"))];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base_a)).unwrap();
    // Bundle was built against dir_a; override to dir_b at
    // verify time — should still verify Ok because the file
    // contents are identical at the same relative offset.
    let report = verify_integrity_evidence_bundle(
        &bundle,
        &BundleVerifyOptions {
            base_dir_override: Some(base_b),
        },
    )
    .unwrap();
    assert!(report.all_ok());
    let canon_b = canon(base_b).to_string_lossy().replace('\\', "/");
    assert_eq!(report.effective_base_dir, canon_b);
}

// ── 20a. Path traversal: relative `..` refused before hashing ─

#[test]
fn relative_input_with_dotdot_refused_before_hashing() {
    let outer = tempfile::TempDir::new().unwrap();
    // Lay out:  outer/safe/   (the base_dir)
    //           outer/outside/file.json   (the escape target)
    let base = outer.path().join("safe");
    fs::create_dir_all(&base).unwrap();
    write_file(outer.path(), "outside/file.json", b"escape-target");
    // Sanity: the escape target does exist outside base, so any
    // failure must come from the validator, NOT from "not found".
    assert!(outer.path().join("outside/file.json").exists());

    let inputs = vec![input(
        BundleArtifactKind::Other,
        Path::new("../outside/file.json"),
    )];
    let err = build_integrity_evidence_bundle(&inputs, &opts(&base)).unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::InvalidRelativePath { ref reason, .. }
            if *reason == "dotdot_segment"
        ),
        "expected InvalidRelativePath{{reason:\"dotdot_segment\"}}, got: {err:?}"
    );
}

// ── 20b. Path traversal: backslash / `.` segment / empty segment

#[test]
fn relative_input_with_backslash_dot_or_empty_segments_refused() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "foo.json", b"x");

    // Backslash — not a valid forward-slash-normalized path on
    // the bundle wire.
    let err = build_integrity_evidence_bundle(
        &[input(BundleArtifactKind::Other, Path::new("a\\b.json"))],
        &opts(base),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::InvalidRelativePath { ref reason, .. }
            if *reason == "backslash"
        ),
        "expected backslash, got: {err:?}"
    );

    // `.` segment.
    let err = build_integrity_evidence_bundle(
        &[input(BundleArtifactKind::Other, Path::new("./foo.json"))],
        &opts(base),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::InvalidRelativePath { ref reason, .. }
            if *reason == "dot_segment"
        ),
        "expected dot_segment, got: {err:?}"
    );

    // Empty segment (`//`).
    let err = build_integrity_evidence_bundle(
        &[input(BundleArtifactKind::Other, Path::new("a//b.json"))],
        &opts(base),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::InvalidRelativePath { ref reason, .. }
            if *reason == "empty_segment"
        ),
        "expected empty_segment, got: {err:?}"
    );
}

// ── 20c. Hand-edited bundle with `..` refused at verify envelope

#[test]
fn hand_edited_bundle_with_traversal_refused_at_verify_envelope() {
    let outer = tempfile::TempDir::new().unwrap();
    let base = outer.path().join("safe");
    fs::create_dir_all(&base).unwrap();
    let escape_target =
        write_file(outer.path(), "outside/file.json", b"escape-target");

    // Build a legit bundle first, then HAND-EDIT entries[0].path
    // to point outside the base_dir.
    write_file(&base, "ok.json", b"ok");
    let inputs = vec![input(BundleArtifactKind::Other, Path::new("ok.json"))];
    let mut bundle =
        build_integrity_evidence_bundle(&inputs, &opts(&base)).unwrap();
    // Mutate to simulate a hand-edit. Even if the operator
    // tries to point to a real file outside base_dir, the
    // verifier must refuse BEFORE walking entries.
    bundle.entries[0].path = "../outside/file.json".to_string();
    assert!(escape_target.exists());

    let err = verify_integrity_evidence_bundle(
        &bundle,
        &BundleVerifyOptions::default(),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            EvidenceBundleError::InvalidRelativePath { ref reason, .. }
            if *reason == "dotdot_segment"
        ),
        "expected InvalidRelativePath at envelope level, got: {err:?}"
    );
}

// ── 20d. Absolute input UNDER base_dir still succeeds ─────────

#[test]
fn absolute_input_under_base_dir_still_succeeds_after_validator() {
    // Defense-in-depth check: the validator runs on the
    // stripped form, which should never trip for a canonical
    // path under base_dir. Pinning so a future canonicalize
    // regression is loud.
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    let abs = write_file(base, "deep/nested/ok.json", b"deep-ok");
    let inputs = vec![input(BundleArtifactKind::Other, &abs)];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    assert_eq!(bundle.entries[0].path, "deep/nested/ok.json");
    let report =
        verify_integrity_evidence_bundle(&bundle, &BundleVerifyOptions::default())
            .unwrap();
    assert!(report.all_ok());
}

// ── 20e. Bundle read distinguishes Io (missing file) from MalformedJson

#[test]
fn bundle_read_missing_file_returns_io_not_malformed_json() {
    // Pins the library variant the CLI's reason-tag mapping
    // depends on: a missing bundle file must surface as `Io`
    // (which the CLI tags `reason=io`), NOT `MalformedJson`
    // (which would tag `reason=malformed_json` and confuse
    // operators who haven't yet written the file).
    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("never-written.json");
    assert!(!missing.exists());
    let err = read_integrity_evidence_bundle_from_path(&missing).unwrap_err();
    assert!(
        matches!(err, EvidenceBundleError::Io { ref source, .. }
            if source.kind() == std::io::ErrorKind::NotFound),
        "expected Io {{ NotFound }}, got: {err:?}"
    );
}

#[test]
fn bundle_read_invalid_json_returns_malformed_json() {
    // Counterpart pin: a present-but-unparseable file must
    // surface as `MalformedJson`, NOT `Io`.
    let dir = tempfile::TempDir::new().unwrap();
    let bad = dir.path().join("bad.json");
    fs::write(&bad, b"{not valid json at all").unwrap();
    let err = read_integrity_evidence_bundle_from_path(&bad).unwrap_err();
    assert!(
        matches!(err, EvidenceBundleError::MalformedJson { .. }),
        "expected MalformedJson, got: {err:?}"
    );
}

// ── 20. Effective base_dir doesn't exist → typed envelope refusal

#[test]
fn effective_base_dir_not_found_is_envelope_level_refusal() {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "a.json", b"a");
    let inputs = vec![input(BundleArtifactKind::Other, Path::new("a.json"))];
    let bundle = build_integrity_evidence_bundle(&inputs, &opts(base)).unwrap();
    let bogus = std::env::temp_dir().join("definitely-not-here-stage12.22");
    let _ = fs::remove_dir_all(&bogus); // ensure absent
    let err = verify_integrity_evidence_bundle(
        &bundle,
        &BundleVerifyOptions {
            base_dir_override: Some(&bogus),
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, EvidenceBundleError::EffectiveBaseDirNotFound { .. }),
        "got: {err:?}"
    );
}
