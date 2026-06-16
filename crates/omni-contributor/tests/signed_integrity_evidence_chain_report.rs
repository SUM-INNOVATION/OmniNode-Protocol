//! Stage 12.25 — integration tests for the local-only signed
//! integrity-evidence-chain-report wrapper.

use std::fs;
use std::path::{Path, PathBuf};

use omni_contributor::{
    build_integrity_evidence_bundle, diff_state_integrity_reports,
    sign_integrity_evidence_bundle, sign_integrity_evidence_chain_report,
    sign_state_integrity_baseline, sign_state_integrity_diff,
    signed_integrity_evidence_chain_report_signing_input,
    verify_integrity_evidence_chain, verify_signed_integrity_evidence_chain_report,
    write_signed_baseline_atomic, write_signed_integrity_diff_atomic,
    write_signed_integrity_evidence_bundle_atomic, BaselineSignerRole,
    BundleArtifactKind, BundleBuilderInput, BundleBuilderOptions, ChainStepOutcome,
    ChainVerifyOptions, ContributorSigner, DiffOptions, FindingKind,
    FindingSeverity, IntegrityEvidenceChainReport, IntegrityFinding,
    RecommendedAction, SignedIntegrityEvidenceChainReport,
    SignedIntegrityEvidenceChainReportError, StateIntegrityReport,
    SIGNED_INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION,
    STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
};

const NOW_UTC: &str = "2026-06-16T12:00:00Z";

const SEED_CHAIN: [u8; 32] = *b"stage12.25-chain-report-sign-32!";
const SEED_OTHER: [u8; 32] = *b"stage12.25-OTHER-chain-seed-32-!";
const SEED_BUNDLE: [u8; 32] = *b"stage12.25-bundle-sign-seed-32!!";
const SEED_BASELINE: [u8; 32] = *b"stage12.25-baseline-sign-seed3!!";
const SEED_DIFF: [u8; 32] = *b"stage12.25-diff-sign-seed-32-3!!";

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
        generated_at_utc: "2026-06-16T11:00:00Z".to_string(),
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

/// Run the full Stage 12.20→12.24 pipeline and return a real
/// chain report. The TempDir is returned so its contents
/// survive for the test lifetime — child files live under it.
fn produce_chain_report() -> (tempfile::TempDir, IntegrityEvidenceChainReport, String) {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();

    // Stage 12.20 signed baseline.
    let baseline_report = synth_report(vec![sample_finding()]);
    let baseline_s = signer(&SEED_BASELINE);
    let signed_baseline = sign_state_integrity_baseline(
        baseline_report,
        &baseline_s.pubkey_hex(),
        BaselineSignerRole::Operator,
        "2026-06-16T10:00:00Z",
        |msg| baseline_s.sign(msg),
    )
    .unwrap();
    let baseline_path = base.join("baseline.signed.json");
    write_signed_baseline_atomic(&signed_baseline, &baseline_path).unwrap();

    // Stage 12.21 signed diff.
    let baseline = synth_report(vec![]);
    let current = synth_report(vec![sample_finding()]);
    let diff = diff_state_integrity_reports(
        &baseline,
        &current,
        &DiffOptions {
            now_utc: "2026-06-16T10:30:00Z",
            require_state_dir_match: false,
        },
    )
    .unwrap();
    let diff_s = signer(&SEED_DIFF);
    let signed_diff = sign_state_integrity_diff(
        diff,
        &diff_s.pubkey_hex(),
        BaselineSignerRole::Operator,
        "2026-06-16T10:30:00Z",
        |msg| diff_s.sign(msg),
    )
    .unwrap();
    let diff_path = base.join("diff.signed.json");
    write_signed_integrity_diff_atomic(&signed_diff, &diff_path).unwrap();

    // Stage 12.22 bundle.
    let bundle = build_integrity_evidence_bundle(
        &[
            BundleBuilderInput {
                artifact_kind: BundleArtifactKind::SignedStateIntegrityBaseline,
                path: Path::new("baseline.signed.json"),
            },
            BundleBuilderInput {
                artifact_kind: BundleArtifactKind::SignedStateIntegrityDiff,
                path: Path::new("diff.signed.json"),
            },
        ],
        &BundleBuilderOptions {
            now_utc: "2026-06-16T11:00:00Z",
            base_dir: base,
            label: Some("stage-12.25-chain-test"),
            notes: None,
        },
    )
    .unwrap();

    // Stage 12.23 signed bundle.
    let bundle_s = signer(&SEED_BUNDLE);
    let bundle_pk = bundle_s.pubkey_hex();
    let signed_bundle = sign_integrity_evidence_bundle(
        bundle,
        &bundle_pk,
        BaselineSignerRole::Operator,
        "2026-06-16T11:30:00Z",
        |msg| bundle_s.sign(msg),
    )
    .unwrap();
    let signed_bundle_path = base.join("bundle.signed.json");
    write_signed_integrity_evidence_bundle_atomic(&signed_bundle, &signed_bundle_path)
        .unwrap();

    // Stage 12.24 chain verify.
    let baseline_pk = signer(&SEED_BASELINE).pubkey_hex();
    let diff_pk = signer(&SEED_DIFF).pubkey_hex();
    let chain_report = verify_integrity_evidence_chain(&ChainVerifyOptions {
        now_utc: NOW_UTC,
        signed_bundle_path: &signed_bundle_path,
        expected_bundle_signer_pubkey_hex: &bundle_pk,
        base_dir_override: None,
        expected_baseline_signer_pubkey_hex: Some(&baseline_pk),
        expected_diff_signer_pubkey_hex: Some(&diff_pk),
    })
    .unwrap();

    (dir, chain_report, bundle_pk)
}

fn sign_with(
    chain_report: IntegrityEvidenceChainReport,
    seed: &[u8; 32],
    role: BaselineSignerRole,
) -> SignedIntegrityEvidenceChainReport {
    let s = signer(seed);
    let pubkey = s.pubkey_hex();
    sign_integrity_evidence_chain_report(
        chain_report,
        &pubkey,
        role,
        NOW_UTC,
        |msg| s.sign(msg),
    )
    .unwrap()
}

// ── 1. Happy path: sign + verify ──────────────────────────────

#[test]
fn sign_and_verify_round_trips() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let signed =
        sign_with(chain_report.clone(), &SEED_CHAIN, BaselineSignerRole::Operator);
    assert_eq!(
        signed.schema_version,
        SIGNED_INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION
    );
    assert_eq!(signed.signer_role, BaselineSignerRole::Operator);
    assert_eq!(signed.signed_at_utc, NOW_UTC);
    assert_eq!(signed.chain_report, chain_report);
    verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap();
}

// ── 2. All four signer-role variants round-trip ───────────────

#[test]
fn all_signer_role_variants_round_trip() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    for role in [
        BaselineSignerRole::Operator,
        BaselineSignerRole::Contributor,
        BaselineSignerRole::Dispatcher,
        BaselineSignerRole::Coordinator,
    ] {
        let signed = sign_with(chain_report.clone(), &SEED_CHAIN, role);
        verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
            .unwrap();
        assert_eq!(signed.signer_role, role);
    }
}

// ── 3. Pubkey mismatch refused (cheap pre-check) ──────────────

#[test]
fn pubkey_mismatch_is_refused_before_crypto() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let signed = sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    let other_pubkey = signer(&SEED_OTHER).pubkey_hex();
    let err = verify_signed_integrity_evidence_chain_report(&signed, &other_pubkey)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signer pubkey mismatch"),
        "expected SignerPubkeyMismatch, got: {msg}"
    );
}

// ── 4. Signature tamper refused ───────────────────────────────

#[test]
fn signature_tamper_is_refused() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let mut signed =
        sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    let bytes = unsafe { signed.signature_hex.as_bytes_mut() };
    bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
    let err = verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch, got: {msg}"
    );
}

// ── 5. Embedded chain-report scalar field tamper refused ──────

#[test]
fn embedded_chain_report_scalar_tamper_is_refused() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let mut signed =
        sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    // Flip the counts_child_ok scalar from 2 to 99. A
    // hand-edited bundle is the typical attack vector here —
    // the signature must catch it.
    signed.chain_report.counts_child_ok = 99;
    let err = verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch (scalar field is in canonical body), got: {msg}"
    );
}

// ── 6. Embedded child-signature enum payload tamper refused ───

#[test]
fn embedded_chain_report_child_enum_tamper_is_refused() {
    // Pins the contract that nested enum payloads inside the
    // chain report are bincode-1-encoded into the canonical
    // body. Flips one child's signature_outcome from Ok to
    // Skipped — a hand-edit that would convert a real
    // verification into a fake "we skipped this" record. The
    // signature must catch it.
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    assert!(!chain_report.child_signatures.is_empty(),
        "test setup expects at least one signed child to mutate");
    let mut signed =
        sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    signed.chain_report.child_signatures[0].signature_outcome =
        ChainStepOutcome::Skipped;
    let err = verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch (nested enum payload is in canonical body), got: {msg}"
    );
}

// ── 7. signer_role tamper refused ─────────────────────────────

#[test]
fn signer_role_tamper_is_refused() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let mut signed =
        sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    signed.signer_role = BaselineSignerRole::Coordinator;
    let err = verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch (role is part of canonical body), got: {msg}"
    );
}

// ── 8. signed_at_utc tamper refused ───────────────────────────

#[test]
fn signed_at_utc_tamper_is_refused() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let mut signed =
        sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    signed.signed_at_utc = "2099-01-01T00:00:00Z".to_string();
    let err = verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch (signed_at_utc is in canonical body), got: {msg}"
    );
}

// ── 9. Stage 12.24 bundle_signer_pubkey_hex tamper refused ────

#[test]
fn embedded_chain_report_bundle_signer_pubkey_tamper_is_refused() {
    // Pins the contract that the Stage 12.24 minimal
    // signer-metadata fields (bundle_signer_role,
    // bundle_signer_pubkey_hex) are in the canonical body. A
    // hand-edit that swaps the recorded bundle signer for a
    // different key must be caught by the chain-report
    // wrapper signature — even though the bundle's OWN
    // signature lives outside the chain report's view.
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let mut signed =
        sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    signed.chain_report.bundle_signer_pubkey_hex = "ff".repeat(32);
    let err = verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch (Stage 12.24 signer metadata is in canonical body), got: {msg}"
    );
}

// ── 10. Wrapper schema_version != 1 refused ───────────────────

#[test]
fn wrapper_schema_v2_is_refused() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let mut signed =
        sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    signed.schema_version = 2;
    let err = verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceChainReportError::UnsupportedSchemaVersion {
                got: 2,
                expected: 1,
            }
        ),
        "got: {err:?}"
    );
}

// ── 11. Embedded chain_report.schema_version != 1 refused ─────

#[test]
fn embedded_chain_report_schema_v2_is_refused() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let mut signed =
        sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    signed.chain_report.schema_version = 2;
    let err = verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceChainReportError::UnsupportedChainReportSchemaVersion {
                got: 2,
                expected: 1,
            }
        ),
        "got: {err:?}"
    );
}

// ── 12. Sign refuses non-v1 input chain report ────────────────

#[test]
fn sign_refuses_non_v1_chain_report() {
    let (_dir, mut chain_report, _bundle_pk) = produce_chain_report();
    chain_report.schema_version = 2;
    let s = signer(&SEED_CHAIN);
    let err = sign_integrity_evidence_chain_report(
        chain_report,
        &s.pubkey_hex(),
        BaselineSignerRole::Operator,
        NOW_UTC,
        |msg| s.sign(msg),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceChainReportError::UnsupportedChainReportSchemaVersion {
                got: 2,
                expected: 1,
            }
        ),
        "got: {err:?}"
    );
}

// ── 13. JSON round-trip preserves signature ──────────────────

#[test]
fn json_round_trip_preserves_signature() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let signed = sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    let s = serde_json::to_string(&signed).unwrap();
    let round: SignedIntegrityEvidenceChainReport =
        serde_json::from_str(&s).unwrap();
    assert_eq!(round, signed);
    verify_signed_integrity_evidence_chain_report(&round, &round.signer_pubkey_hex)
        .unwrap();
}

// ── 14. Determinism ───────────────────────────────────────────

#[test]
fn same_inputs_produce_byte_identical_wrapper() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let a =
        sign_with(chain_report.clone(), &SEED_CHAIN, BaselineSignerRole::Operator);
    let b = sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    assert_eq!(a, b);
    let ja = serde_json::to_vec(&a).unwrap();
    let jb = serde_json::to_vec(&b).unwrap();
    assert_eq!(ja, jb, "JSON serialization must be byte-stable");
}

// ── 15. Canonical signing-input byte stability + domain ───────

#[test]
fn signing_input_is_byte_stable_across_recomputation() {
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let signed =
        sign_with(chain_report.clone(), &SEED_CHAIN, BaselineSignerRole::Operator);
    let a = signed_integrity_evidence_chain_report_signing_input(
        signed.schema_version,
        &signed.signed_at_utc,
        &signed.signer_pubkey_hex,
        signed.signer_role,
        &signed.chain_report,
    )
    .unwrap();
    let b = signed_integrity_evidence_chain_report_signing_input(
        SIGNED_INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION,
        NOW_UTC,
        &signed.signer_pubkey_hex,
        BaselineSignerRole::Operator,
        &chain_report,
    )
    .unwrap();
    assert_eq!(a, b);
    // And it must START with the domain separator.
    let dom = b"OMNINODE-CONTRIBUTOR-SIGNED-INTEGRITY-EVIDENCE-CHAIN-REPORT:v1:";
    assert!(
        a.starts_with(dom),
        "canonical input must lead with domain separator"
    );
}

// ── 16. Atomic write helper round-trips via disk ──────────────

#[test]
fn write_signed_atomic_round_trips_via_disk() {
    use omni_contributor::{
        read_signed_integrity_evidence_chain_report_from_path,
        write_signed_integrity_evidence_chain_report_atomic,
    };

    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let signed = sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);
    let out_dir = tempfile::TempDir::new().unwrap();
    let out = out_dir.path().join("chain-report.signed.json");
    write_signed_integrity_evidence_chain_report_atomic(&signed, &out).unwrap();
    let round =
        read_signed_integrity_evidence_chain_report_from_path(&out).unwrap();
    assert_eq!(round, signed);
    verify_signed_integrity_evidence_chain_report(&round, &round.signer_pubkey_hex)
        .unwrap();
}

// ── 17. Six-stage composition ─────────────────────────────────

#[test]
fn six_stage_chain_composition_end_to_end() {
    // The strongest "the whole stack composes" regression:
    //   Stage 12.20 sign baseline
    //   Stage 12.21 sign diff
    //   Stage 12.22 build bundle
    //   Stage 12.23 sign bundle
    //   Stage 12.24 chain verify
    //   Stage 12.25 sign chain report
    //   Stage 12.25 verify chain-report signature
    // produce_chain_report already runs stages 12.20-12.24.
    // We extend with 12.25 sign + verify and assert the
    // embedded chain report round-trips bit-for-bit.
    let (_dir, chain_report, _bundle_pk) = produce_chain_report();
    let original = chain_report.clone();

    // Stage 12.25 sign.
    let signed = sign_with(chain_report, &SEED_CHAIN, BaselineSignerRole::Operator);

    // Stage 12.25 verify signature.
    verify_signed_integrity_evidence_chain_report(&signed, &signed.signer_pubkey_hex)
        .unwrap();

    // Embedded chain report survives unchanged.
    assert_eq!(signed.chain_report, original);
    // The Stage 12.24 chain report's outcomes must all be Ok
    // because we built and signed every artifact with matching
    // anchors.
    assert!(
        matches!(signed.chain_report.bundle_signature, ChainStepOutcome::Ok),
        "expected bundle_signature Ok, got: {:?}",
        signed.chain_report.bundle_signature
    );
    assert!(
        signed.chain_report.bundle_byte_verify.all_ok(),
        "expected bundle_byte_verify all_ok"
    );
    assert!(
        signed.chain_report.all_required_ok(),
        "expected chain report all_required_ok"
    );
    assert_eq!(signed.chain_report.counts_child_failed, 0);
}

// ── 18. Wrapper read distinguishes Io from MalformedJson ──────

#[test]
fn signed_chain_report_read_missing_file_returns_io_not_malformed_json() {
    use omni_contributor::read_signed_integrity_evidence_chain_report_from_path;
    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("never-written.json");
    assert!(!missing.exists());
    let err = read_signed_integrity_evidence_chain_report_from_path(&missing)
        .unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceChainReportError::Io { ref source, .. }
            if source.kind() == std::io::ErrorKind::NotFound
        ),
        "expected Io {{ NotFound }}, got: {err:?}"
    );
}

#[test]
fn signed_chain_report_read_invalid_json_returns_malformed_json() {
    use omni_contributor::read_signed_integrity_evidence_chain_report_from_path;
    let dir = tempfile::TempDir::new().unwrap();
    let bad = dir.path().join("bad.json");
    fs::write(&bad, b"{not valid json at all").unwrap();
    let err = read_signed_integrity_evidence_chain_report_from_path(&bad)
        .unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceChainReportError::MalformedJson { .. }
        ),
        "expected MalformedJson, got: {err:?}"
    );
}

// Silence unused-imports warnings for helpers used only by
// fixture construction.
#[allow(dead_code)]
fn _silence_unused(_: PathBuf) {}
