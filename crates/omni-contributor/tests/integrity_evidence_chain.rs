//! Stage 12.24 — integration tests for the local-only
//! integrity-evidence chain verifier.

use std::fs;
use std::path::{Path, PathBuf};

use omni_contributor::{
    build_integrity_evidence_bundle, diff_state_integrity_reports,
    sign_integrity_evidence_bundle, sign_state_integrity_baseline,
    sign_state_integrity_diff, verify_integrity_evidence_chain,
    write_signed_baseline_atomic, write_signed_integrity_diff_atomic,
    write_signed_integrity_evidence_bundle_atomic, BaselineSignerRole,
    BundleArtifactKind, BundleBuilderInput, BundleBuilderOptions,
    ChainStepOutcome, ChainVerifyError, ChainVerifyOptions, ContributorSigner,
    DiffOptions, FindingKind, FindingSeverity, IntegrityEvidenceChainReport,
    IntegrityFinding, RecommendedAction, StateIntegrityReport,
    INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION,
    STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
};

const NOW_UTC: &str = "2026-06-15T12:00:00Z";

const SEED_BUNDLE: [u8; 32] = *b"stage12.24-chain-bundle-seed-32!";
const SEED_BASELINE: [u8; 32] = *b"stage12.24-chain-baseline-seed3!";
const SEED_DIFF: [u8; 32] = *b"stage12.24-chain-diff-seed-32-3!";
const SEED_OTHER: [u8; 32] = *b"stage12.24-chain-OTHER-seed-32!!";

fn signer(seed: &[u8; 32]) -> ContributorSigner {
    ContributorSigner::from_seed_bytes(seed).unwrap()
}

fn synth_report(findings: Vec<IntegrityFinding>) -> StateIntegrityReport {
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
        generated_at_utc: "2026-06-15T11:00:00Z".to_string(),
        state_dir: "/state".to_string(),
        state_version: 2,
        omni_contributor_version: "0.1.0".to_string(),
        sessions_scanned: 0,
        sessions_verified: 0,
        counts_ok,
        counts_warn,
        counts_error,
        sessions: Vec::new(),
        findings,
    }
}

fn sample_finding() -> IntegrityFinding {
    IntegrityFinding {
        kind: FindingKind::InvalidJoin,
        severity: FindingSeverity::Error,
        session_id: Some("aa".repeat(32)),
        path: Some("verified/sessions/aa/joins/x.json".to_string()),
        reason_tag: "ContributorSignatureFailed".to_string(),
        recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    }
}

/// Build a real Stage 12.20 SignedStateIntegrityBaseline on
/// disk under `dir` and return its path. Caller supplies the
/// seed so tests can mismatch trust anchors.
fn write_signed_baseline(
    dir: &Path,
    rel: &str,
    seed: &[u8; 32],
) -> PathBuf {
    let report = synth_report(vec![sample_finding()]);
    let s = signer(seed);
    let signed = sign_state_integrity_baseline(
        report,
        &s.pubkey_hex(),
        BaselineSignerRole::Operator,
        "2026-06-15T10:00:00Z",
        |msg| s.sign(msg),
    )
    .unwrap();
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    write_signed_baseline_atomic(&signed, &path).unwrap();
    path
}

/// Build a real Stage 12.21 SignedStateIntegrityDiff on disk.
fn write_signed_diff(dir: &Path, rel: &str, seed: &[u8; 32]) -> PathBuf {
    let baseline = synth_report(vec![]);
    let current = synth_report(vec![sample_finding()]);
    let diff = diff_state_integrity_reports(
        &baseline,
        &current,
        &DiffOptions {
            now_utc: "2026-06-15T10:30:00Z",
            require_state_dir_match: false,
        },
    )
    .unwrap();
    let s = signer(seed);
    let signed = sign_state_integrity_diff(
        diff,
        &s.pubkey_hex(),
        BaselineSignerRole::Operator,
        "2026-06-15T10:30:00Z",
        |msg| s.sign(msg),
    )
    .unwrap();
    let path = dir.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    write_signed_integrity_diff_atomic(&signed, &path).unwrap();
    path
}

/// Build a full Stage 12.22 bundle pointing at the supplied
/// `--include` triples, sign it with `bundle_seed`, write the
/// signed wrapper to disk, and return (`tempdir`, signed wrapper
/// path, signed-bundle pubkey hex).
fn build_signed_bundle(
    dir: &Path,
    inputs: &[(BundleArtifactKind, &Path)],
    bundle_seed: &[u8; 32],
) -> (PathBuf, String) {
    let builder_inputs: Vec<BundleBuilderInput<'_>> = inputs
        .iter()
        .map(|(kind, path)| BundleBuilderInput {
            artifact_kind: *kind,
            path,
        })
        .collect();
    let bundle = build_integrity_evidence_bundle(
        &builder_inputs,
        &BundleBuilderOptions {
            now_utc: "2026-06-15T11:30:00Z",
            base_dir: dir,
            label: Some("stage-12.24-chain-test"),
            notes: None,
        },
    )
    .unwrap();
    let s = signer(bundle_seed);
    let pubkey = s.pubkey_hex();
    let signed = sign_integrity_evidence_bundle(
        bundle,
        &pubkey,
        BaselineSignerRole::Operator,
        NOW_UTC,
        |msg| s.sign(msg),
    )
    .unwrap();
    let out_dir = tempfile::TempDir::new().unwrap();
    let path = out_dir.path().join("signed-bundle.json");
    write_signed_integrity_evidence_bundle_atomic(&signed, &path).unwrap();
    // Keep the out_dir alive by leaking via std::mem::forget?
    // Simpler: copy the wrapper into the base dir so the test
    // owns it via the caller's tempdir lifetime.
    let final_path = dir.join("signed-bundle.json");
    fs::copy(&path, &final_path).unwrap();
    drop(out_dir);
    (final_path, pubkey)
}

fn run_chain<'a>(
    signed_bundle: &'a Path,
    bundle_anchor: &'a str,
    baseline_anchor: Option<&'a str>,
    diff_anchor: Option<&'a str>,
) -> Result<IntegrityEvidenceChainReport, ChainVerifyError> {
    verify_integrity_evidence_chain(&ChainVerifyOptions {
        now_utc: NOW_UTC,
        signed_bundle_path: signed_bundle,
        expected_bundle_signer_pubkey_hex: bundle_anchor,
        base_dir_override: None,
        expected_baseline_signer_pubkey_hex: baseline_anchor,
        expected_diff_signer_pubkey_hex: diff_anchor,
    })
}

// ── 1. Happy path — all three anchors supplied ────────────────

#[test]
fn chain_happy_path_all_three_anchors() {
    let dir = tempfile::TempDir::new().unwrap();
    let baseline_path =
        write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let diff_path =
        write_signed_diff(dir.path(), "diff.signed.json", &SEED_DIFF);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[
            (
                BundleArtifactKind::SignedStateIntegrityBaseline,
                Path::new("baseline.signed.json"),
            ),
            (
                BundleArtifactKind::SignedStateIntegrityDiff,
                Path::new("diff.signed.json"),
            ),
        ],
        &SEED_BUNDLE,
    );
    assert!(baseline_path.exists() && diff_path.exists());

    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let diff_pk = signer(&SEED_DIFF).pubkey_hex();
    let report = run_chain(
        &bundle_path,
        &bundle_pk,
        Some(&baseline_pk),
        Some(&diff_pk),
    )
    .unwrap();

    assert_eq!(
        report.schema_version,
        INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION
    );
    assert!(matches!(report.bundle_signature, ChainStepOutcome::Ok));
    assert!(report.bundle_byte_verify.all_ok());
    assert_eq!(report.counts_child_ok, 2);
    assert_eq!(report.counts_child_skipped, 0);
    assert_eq!(report.counts_child_failed, 0);
    assert!(report.all_required_ok());
}

// ── 2. Skip on omitted child anchors ──────────────────────────

#[test]
fn chain_skip_when_child_anchors_omitted() {
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    write_signed_diff(dir.path(), "diff.signed.json", &SEED_DIFF);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[
            (
                BundleArtifactKind::SignedStateIntegrityBaseline,
                Path::new("baseline.signed.json"),
            ),
            (
                BundleArtifactKind::SignedStateIntegrityDiff,
                Path::new("diff.signed.json"),
            ),
        ],
        &SEED_BUNDLE,
    );

    let report = run_chain(&bundle_path, &bundle_pk, None, None).unwrap();
    assert!(matches!(report.bundle_signature, ChainStepOutcome::Ok));
    assert!(report.bundle_byte_verify.all_ok());
    assert_eq!(report.counts_child_ok, 0);
    assert_eq!(report.counts_child_skipped, 2);
    assert_eq!(report.counts_child_failed, 0);
    assert!(
        report.all_required_ok(),
        "skipped children must NOT fail all_required_ok"
    );
    assert!(report
        .child_signatures
        .iter()
        .all(|c| matches!(c.signature_outcome, ChainStepOutcome::Skipped)));
}

// ── 3. Bundle signature mismatch → envelope refusal ───────────

#[test]
fn chain_signed_bundle_pubkey_mismatch_short_circuits() {
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    write_signed_diff(dir.path(), "diff.signed.json", &SEED_DIFF);
    let (bundle_path, _bundle_pk) = build_signed_bundle(
        dir.path(),
        &[
            (
                BundleArtifactKind::SignedStateIntegrityBaseline,
                Path::new("baseline.signed.json"),
            ),
            (
                BundleArtifactKind::SignedStateIntegrityDiff,
                Path::new("diff.signed.json"),
            ),
        ],
        &SEED_BUNDLE,
    );

    // Pass a DIFFERENT pubkey as the bundle anchor.
    let wrong_pk = signer(&SEED_OTHER).pubkey_hex();
    let err = run_chain(&bundle_path, &wrong_pk, None, None).unwrap_err();
    assert!(
        matches!(
            err,
            ChainVerifyError::SignedBundle(
                omni_contributor::SignedIntegrityEvidenceBundleError::SignerPubkeyMismatch { .. }
            )
        ),
        "got: {err:?}"
    );
}

// ── 4. Bundle-byte hash_mismatch on one artifact ──────────────

#[test]
fn chain_bundle_byte_hash_mismatch_still_runs_child_verify() {
    let dir = tempfile::TempDir::new().unwrap();
    let baseline_path =
        write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    write_signed_diff(dir.path(), "diff.signed.json", &SEED_DIFF);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[
            (
                BundleArtifactKind::SignedStateIntegrityBaseline,
                Path::new("baseline.signed.json"),
            ),
            (
                BundleArtifactKind::SignedStateIntegrityDiff,
                Path::new("diff.signed.json"),
            ),
        ],
        &SEED_BUNDLE,
    );
    // Tamper the baseline file's bytes — keep the SAME length
    // so we trip hash_mismatch, NOT size_mismatch. The signed
    // baseline JSON is large and varied; appending one space
    // before the closing brace works on any well-formed JSON.
    let original = fs::read(&baseline_path).unwrap();
    let mut tampered = original.clone();
    // Flip a single byte deep in the middle to keep length.
    let mid = tampered.len() / 2;
    tampered[mid] = if tampered[mid] == b'a' { b'b' } else { b'a' };
    fs::write(&baseline_path, &tampered).unwrap();

    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let diff_pk = signer(&SEED_DIFF).pubkey_hex();
    let report = run_chain(
        &bundle_path,
        &bundle_pk,
        Some(&baseline_pk),
        Some(&diff_pk),
    )
    .unwrap();

    // Bundle-byte view: hash_mismatch on the baseline.
    assert_eq!(report.bundle_byte_verify.counts_hash_mismatch, 1);
    assert!(!report.all_required_ok());

    // Child-signature view: still ran on BOTH children. The
    // tampered baseline likely no longer parses or no longer
    // verifies; the diff still verifies.
    assert_eq!(report.child_signatures.len(), 2);
    let baseline_outcome = report
        .child_signatures
        .iter()
        .find(|c| c.path == "baseline.signed.json")
        .unwrap();
    assert!(
        !matches!(baseline_outcome.signature_outcome, ChainStepOutcome::Skipped),
        "tampered baseline must NOT be skipped — child verify is collect-all"
    );
    let diff_outcome = report
        .child_signatures
        .iter()
        .find(|c| c.path == "diff.signed.json")
        .unwrap();
    assert!(
        matches!(diff_outcome.signature_outcome, ChainStepOutcome::Ok),
        "untouched diff must still verify Ok"
    );
}

// ── 5. Bundle-byte envelope refusal (effective base_dir bogus) ─

#[test]
fn chain_bundle_byte_effective_base_dir_not_found_is_envelope() {
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        )],
        &SEED_BUNDLE,
    );

    // Override to a definitely-absent dir.
    let bogus = std::env::temp_dir().join("definitely-not-here-stage12.24");
    let _ = fs::remove_dir_all(&bogus);
    let err = verify_integrity_evidence_chain(&ChainVerifyOptions {
        now_utc: NOW_UTC,
        signed_bundle_path: &bundle_path,
        expected_bundle_signer_pubkey_hex: &bundle_pk,
        base_dir_override: Some(&bogus),
        expected_baseline_signer_pubkey_hex: None,
        expected_diff_signer_pubkey_hex: None,
    })
    .unwrap_err();
    assert!(
        matches!(
            err,
            ChainVerifyError::BundleByte(
                omni_contributor::EvidenceBundleError::EffectiveBaseDirNotFound { .. }
            )
        ),
        "got: {err:?}"
    );
}

// ── 6. Bundle-byte envelope refusal (traversal in entry path) ──

#[test]
fn chain_bundle_byte_invalid_relative_path_is_envelope() {
    use omni_contributor::{
        read_signed_integrity_evidence_bundle_from_path,
        write_signed_integrity_evidence_bundle_atomic,
    };

    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        )],
        &SEED_BUNDLE,
    );
    // Hand-edit the signed wrapper to insert a traversal path
    // in entries[0].path. The Stage 12.23 signature will NOT
    // match anymore — but Stage 12.22's envelope-level path
    // validator runs BEFORE the bundle-byte walk, and only
    // AFTER Stage 12.23 signature verification. So we'd need
    // a wrapper whose signature still verifies. To preserve
    // signatures, re-sign the bundle with the traversal path
    // BEFORE the bundle gets sealed. Simplest: tamper the
    // already-written wrapper, then re-sign on the fly using
    // the same SEED_BUNDLE.
    let mut wrapper =
        read_signed_integrity_evidence_bundle_from_path(&bundle_path).unwrap();
    wrapper.bundle.entries[0].path = "../escape/baseline.signed.json".to_string();
    // Re-sign so the chain reaches the Stage 12.22 gate.
    let s = signer(&SEED_BUNDLE);
    let re_signed = sign_integrity_evidence_bundle(
        wrapper.bundle,
        &s.pubkey_hex(),
        BaselineSignerRole::Operator,
        NOW_UTC,
        |msg| s.sign(msg),
    )
    .unwrap();
    write_signed_integrity_evidence_bundle_atomic(&re_signed, &bundle_path)
        .unwrap();

    let err = run_chain(&bundle_path, &bundle_pk, None, None).unwrap_err();
    assert!(
        matches!(
            err,
            ChainVerifyError::BundleByte(
                omni_contributor::EvidenceBundleError::InvalidRelativePath { .. }
            )
        ),
        "got: {err:?}"
    );
}

// ── 7. Child baseline signer mismatch → child Failed ──────────

#[test]
fn chain_child_baseline_signer_mismatch_does_not_short_circuit() {
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    write_signed_diff(dir.path(), "diff.signed.json", &SEED_DIFF);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[
            (
                BundleArtifactKind::SignedStateIntegrityBaseline,
                Path::new("baseline.signed.json"),
            ),
            (
                BundleArtifactKind::SignedStateIntegrityDiff,
                Path::new("diff.signed.json"),
            ),
        ],
        &SEED_BUNDLE,
    );
    // Wrong baseline anchor — but correct diff anchor.
    let wrong_baseline_pk = signer(&SEED_OTHER).pubkey_hex();
    let diff_pk = signer(&SEED_DIFF).pubkey_hex();
    let report = run_chain(
        &bundle_path,
        &bundle_pk,
        Some(&wrong_baseline_pk),
        Some(&diff_pk),
    )
    .unwrap();
    assert!(report.bundle_byte_verify.all_ok());
    let baseline_outcome = report
        .child_signatures
        .iter()
        .find(|c| c.path == "baseline.signed.json")
        .unwrap();
    assert!(
        matches!(
            &baseline_outcome.signature_outcome,
            ChainStepOutcome::Failed { reason, .. }
            if reason == "signer_pubkey_mismatch"
        ),
        "got: {:?}",
        baseline_outcome.signature_outcome
    );
    let diff_outcome = report
        .child_signatures
        .iter()
        .find(|c| c.path == "diff.signed.json")
        .unwrap();
    assert!(
        matches!(diff_outcome.signature_outcome, ChainStepOutcome::Ok),
        "diff child must still verify Ok"
    );
    assert_eq!(report.counts_child_failed, 1);
    assert_eq!(report.counts_child_ok, 1);
    assert!(!report.all_required_ok());
}

// ── 8. Child file missing → both bundle-byte NotFound + child IO

#[test]
fn chain_child_file_missing_yields_both_bundle_byte_and_child_io() {
    let dir = tempfile::TempDir::new().unwrap();
    let baseline_path =
        write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        )],
        &SEED_BUNDLE,
    );
    // Delete the baseline AFTER signing so the bundle's entry
    // is still recorded but the file is gone at chain-verify
    // time.
    fs::remove_file(&baseline_path).unwrap();

    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let report = run_chain(
        &bundle_path,
        &bundle_pk,
        Some(&baseline_pk),
        None,
    )
    .unwrap();
    // Bundle-byte view: NotFound.
    assert_eq!(report.bundle_byte_verify.counts_not_found, 1);
    // Child-signature view: io (read attempt failed).
    let baseline_outcome = report
        .child_signatures
        .iter()
        .find(|c| c.path == "baseline.signed.json")
        .unwrap();
    assert!(
        matches!(
            &baseline_outcome.signature_outcome,
            ChainStepOutcome::Failed { reason, .. }
            if reason == "io"
        ),
        "got: {:?}",
        baseline_outcome.signature_outcome
    );
    assert!(!report.all_required_ok());
}

// ── 9. Child malformed JSON → child Failed reason=malformed_json

#[test]
fn chain_child_malformed_json_recorded_as_failed_with_tag() {
    let dir = tempfile::TempDir::new().unwrap();
    let baseline_path =
        write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        )],
        &SEED_BUNDLE,
    );
    // Replace with garbage AFTER signing — bundle-byte will
    // mismatch (different bytes); child read will refuse to
    // parse.
    fs::write(&baseline_path, b"{not even close to valid json").unwrap();

    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let report = run_chain(
        &bundle_path,
        &bundle_pk,
        Some(&baseline_pk),
        None,
    )
    .unwrap();
    let baseline_outcome = report
        .child_signatures
        .iter()
        .find(|c| c.path == "baseline.signed.json")
        .unwrap();
    assert!(
        matches!(
            &baseline_outcome.signature_outcome,
            ChainStepOutcome::Failed { reason, .. }
            if reason == "malformed_json"
        ),
        "got: {:?}",
        baseline_outcome.signature_outcome
    );
    assert!(!report.all_required_ok());
}

// ── 10. All-Skipped scenario ──────────────────────────────────

#[test]
fn chain_all_signed_children_skipped_when_no_anchors() {
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "b1.signed.json", &SEED_BASELINE);
    write_signed_baseline(dir.path(), "b2.signed.json", &SEED_BASELINE);
    write_signed_diff(dir.path(), "d1.signed.json", &SEED_DIFF);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[
            (
                BundleArtifactKind::SignedStateIntegrityBaseline,
                Path::new("b1.signed.json"),
            ),
            (
                BundleArtifactKind::SignedStateIntegrityBaseline,
                Path::new("b2.signed.json"),
            ),
            (
                BundleArtifactKind::SignedStateIntegrityDiff,
                Path::new("d1.signed.json"),
            ),
        ],
        &SEED_BUNDLE,
    );
    let report = run_chain(&bundle_path, &bundle_pk, None, None).unwrap();
    assert!(report.bundle_byte_verify.all_ok());
    assert_eq!(report.counts_child_skipped, 3);
    assert_eq!(report.counts_child_ok, 0);
    assert_eq!(report.counts_child_failed, 0);
    assert!(report.all_required_ok());
}

// ── 11. Bundle with zero signed children ──────────────────────

#[test]
fn chain_bundle_with_zero_signed_children_passes_cleanly() {
    let dir = tempfile::TempDir::new().unwrap();
    // Use a non-signed kind: plain state_integrity_report
    // (Stage 12.16-style — operator-supplied path).
    fs::write(dir.path().join("plain-report.json"), b"plain bytes").unwrap();
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::StateIntegrityReport,
            Path::new("plain-report.json"),
        )],
        &SEED_BUNDLE,
    );
    // Supply child anchors anyway — they should be silently
    // unused.
    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let diff_pk = signer(&SEED_DIFF).pubkey_hex();
    let report = run_chain(
        &bundle_path,
        &bundle_pk,
        Some(&baseline_pk),
        Some(&diff_pk),
    )
    .unwrap();
    assert!(report.bundle_byte_verify.all_ok());
    assert_eq!(report.child_signatures.len(), 0);
    assert_eq!(report.counts_child_ok, 0);
    assert_eq!(report.counts_child_skipped, 0);
    assert_eq!(report.counts_child_failed, 0);
    assert!(report.all_required_ok());
}

// ── 12. Determinism ───────────────────────────────────────────

#[test]
fn chain_report_is_byte_identical_on_repeat_runs() {
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    write_signed_diff(dir.path(), "diff.signed.json", &SEED_DIFF);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[
            (
                BundleArtifactKind::SignedStateIntegrityBaseline,
                Path::new("baseline.signed.json"),
            ),
            (
                BundleArtifactKind::SignedStateIntegrityDiff,
                Path::new("diff.signed.json"),
            ),
        ],
        &SEED_BUNDLE,
    );
    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let diff_pk = signer(&SEED_DIFF).pubkey_hex();
    let a = run_chain(
        &bundle_path,
        &bundle_pk,
        Some(&baseline_pk),
        Some(&diff_pk),
    )
    .unwrap();
    let b = run_chain(
        &bundle_path,
        &bundle_pk,
        Some(&baseline_pk),
        Some(&diff_pk),
    )
    .unwrap();
    assert_eq!(a, b);
    let ja = serde_json::to_vec(&a).unwrap();
    let jb = serde_json::to_vec(&b).unwrap();
    assert_eq!(ja, jb, "chain report JSON must be byte-stable");
}

// ── 13. JSON round-trip ───────────────────────────────────────

#[test]
fn chain_report_json_round_trip_preserves_report() {
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        )],
        &SEED_BUNDLE,
    );
    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let report =
        run_chain(&bundle_path, &bundle_pk, Some(&baseline_pk), None).unwrap();
    let s = serde_json::to_string(&report).unwrap();
    let round: IntegrityEvidenceChainReport = serde_json::from_str(&s).unwrap();
    assert_eq!(round, report);
}

// ── 14. all_required_ok semantics ─────────────────────────────

#[test]
fn all_required_ok_is_true_with_skipped_children_only() {
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        )],
        &SEED_BUNDLE,
    );
    let report = run_chain(&bundle_path, &bundle_pk, None, None).unwrap();
    assert_eq!(report.counts_child_skipped, 1);
    assert!(
        report.all_required_ok(),
        "skipped child must NOT fail all_required_ok"
    );
}

// ── 14b. Verified signer metadata flows into the chain report ─

#[test]
fn chain_report_records_verified_bundle_signer_metadata() {
    // Pins the contract that `verify_integrity_evidence_chain`
    // surfaces the verified Stage 12.23 wrapper's
    // `signer_role` and `signer_pubkey_hex` as minimal chain
    // report metadata — without embedding the full wrapper.
    // The CLI's `..._signed_bundle_ok` event consumes these
    // fields; this regression guards against a future refactor
    // dropping them silently.
    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        )],
        &SEED_BUNDLE,
    );
    let report = run_chain(&bundle_path, &bundle_pk, None, None).unwrap();
    // Role recorded matches what build_signed_bundle signed
    // with — Operator per the test helper.
    assert_eq!(report.bundle_signer_role, BaselineSignerRole::Operator);
    // Pubkey hex matches the verified anchor.
    assert_eq!(report.bundle_signer_pubkey_hex, bundle_pk);
    // 64-char lowercase hex sanity (Stage 12.20 convention).
    assert_eq!(report.bundle_signer_pubkey_hex.len(), 64);
    assert!(report
        .bundle_signer_pubkey_hex
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
}

// ── 15. Atomic --json-out helper round-trips via disk ─────────

#[test]
fn write_chain_report_atomic_round_trips_via_disk() {
    use omni_contributor::write_integrity_evidence_chain_report_atomic;

    let dir = tempfile::TempDir::new().unwrap();
    write_signed_baseline(dir.path(), "baseline.signed.json", &SEED_BASELINE);
    let (bundle_path, bundle_pk) = build_signed_bundle(
        dir.path(),
        &[(
            BundleArtifactKind::SignedStateIntegrityBaseline,
            Path::new("baseline.signed.json"),
        )],
        &SEED_BUNDLE,
    );
    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let report =
        run_chain(&bundle_path, &bundle_pk, Some(&baseline_pk), None).unwrap();
    let out_dir = tempfile::TempDir::new().unwrap();
    let out = out_dir.path().join("chain.json");
    write_integrity_evidence_chain_report_atomic(&report, &out).unwrap();
    let bytes = fs::read(&out).unwrap();
    let round: IntegrityEvidenceChainReport =
        serde_json::from_slice(&bytes).unwrap();
    assert_eq!(round, report);
}
