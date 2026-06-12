//! Stage 12.21 — integration tests for the signed
//! integrity-diff wrapper.

use omni_contributor::{
    diff_state_integrity_reports, sign_state_integrity_diff,
    signed_integrity_diff_signing_input, verify_signed_state_integrity_diff,
    BaselineSignerRole, ContributorSigner, DiffOptions, FindingKind,
    FindingSeverity, IntegrityFinding, RecommendedAction,
    SignedStateIntegrityDiff, StateIntegrityDiffReport, StateIntegrityReport,
    SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION, STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
};

const NOW_UTC: &str = "2026-06-12T01:00:00Z";

const SEED_A: [u8; 32] = *b"stage12.21-diff-signer-aaaaa-32!";
const SEED_B: [u8; 32] = *b"stage12.21-diff-signer-bbbbb-32!";

fn signer(seed: &[u8; 32]) -> ContributorSigner {
    ContributorSigner::from_seed_bytes(seed).unwrap()
}

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
        generated_at_utc: "2026-06-11T00:00:00Z".to_string(),
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
    finding(
        FindingKind::InvalidJoin,
        FindingSeverity::Error,
        Some(&"aa".repeat(32)),
        Some("verified/sessions/aa/joins/x.json"),
        "ContributorSignatureFailed",
        RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    )
}

/// Build a Stage 12.19 v1 `StateIntegrityDiffReport` by diffing
/// two synthesized reports. Using the real helper keeps the
/// test exercising the same diff schema the CLI produces.
fn synth_diff(
    baseline_findings: Vec<IntegrityFinding>,
    current_findings: Vec<IntegrityFinding>,
) -> StateIntegrityDiffReport {
    let baseline = synth_report(baseline_findings);
    let current = synth_report(current_findings);
    diff_state_integrity_reports(
        &baseline,
        &current,
        &DiffOptions {
            now_utc: NOW_UTC,
            require_state_dir_match: false,
        },
    )
    .unwrap()
}

fn sign_with(
    diff: StateIntegrityDiffReport,
    seed: &[u8; 32],
    role: BaselineSignerRole,
) -> SignedStateIntegrityDiff {
    let s = signer(seed);
    let pubkey = s.pubkey_hex();
    sign_state_integrity_diff(diff, &pubkey, role, NOW_UTC, |msg| s.sign(msg))
        .unwrap()
}

// ── 1. Happy path: sign + verify ──────────────────────────────

#[test]
fn sign_and_verify_round_trips() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let signed = sign_with(diff.clone(), &SEED_A, BaselineSignerRole::Operator);
    assert_eq!(signed.schema_version, SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION);
    assert_eq!(signed.signer_role, BaselineSignerRole::Operator);
    assert_eq!(signed.signed_at_utc, NOW_UTC);
    assert_eq!(signed.diff, diff);
    verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex).unwrap();
}

// ── 2. All four signer-role variants round-trip ───────────────

#[test]
fn all_signer_role_variants_round_trip() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    for role in [
        BaselineSignerRole::Operator,
        BaselineSignerRole::Contributor,
        BaselineSignerRole::Dispatcher,
        BaselineSignerRole::Coordinator,
    ] {
        let signed = sign_with(diff.clone(), &SEED_A, role);
        verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex)
            .unwrap();
        assert_eq!(signed.signer_role, role);
    }
}

// ── 3. Pubkey mismatch refused (cheap pre-check) ──────────────

#[test]
fn pubkey_mismatch_is_refused_before_crypto() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    let key_b_pubkey = signer(&SEED_B).pubkey_hex();
    let err = verify_signed_state_integrity_diff(&signed, &key_b_pubkey).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signer pubkey mismatch"),
        "expected SignerPubkeyMismatch, got: {msg}"
    );
}

// ── 4. Signature tamper refused ───────────────────────────────

#[test]
fn signature_tamper_is_refused() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let mut signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    let bytes = unsafe { signed.signature_hex.as_bytes_mut() };
    bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
    let err = verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch, got: {msg}"
    );
}

// ── 5. Embedded diff tamper refused ───────────────────────────

#[test]
fn embedded_diff_tamper_is_refused() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let mut signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    // Inflate the new-counts counter on the embedded diff. The
    // signature is over the original bytes; the verifier
    // recomputes the canonical body from the tampered field
    // and gets a mismatch.
    signed.diff.counts.new += 7;
    let err = verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch, got: {msg}"
    );
}

// ── 6. signer_role tamper refused ─────────────────────────────

#[test]
fn signer_role_tamper_is_refused() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let mut signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    signed.signer_role = BaselineSignerRole::Coordinator;
    let err = verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch (role is part of canonical body), got: {msg}"
    );
}

// ── 7. signed_at_utc tamper refused ───────────────────────────

#[test]
fn signed_at_utc_tamper_is_refused() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let mut signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    signed.signed_at_utc = "2099-01-01T00:00:00Z".to_string();
    let err = verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch (signed_at_utc is in canonical body), got: {msg}"
    );
}

// ── 8. Wrapper schema_version != 1 refused ────────────────────

#[test]
fn wrapper_schema_v2_is_refused() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let mut signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    signed.schema_version = 2;
    let err = verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported signed-diff schema_version"),
        "got: {msg}"
    );
}

// ── 9. Embedded diff.schema_version != 1 refused ──────────────

#[test]
fn embedded_diff_schema_v2_is_refused() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let mut signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    signed.diff.schema_version = 2;
    let err = verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported embedded diff schema_version"),
        "got: {msg}"
    );
}

// ── 10. JSON round-trip preserves signature ──────────────────

#[test]
fn json_round_trip_preserves_signature() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    let s = serde_json::to_string(&signed).unwrap();
    let round: SignedStateIntegrityDiff = serde_json::from_str(&s).unwrap();
    assert_eq!(round, signed);
    verify_signed_state_integrity_diff(&round, &round.signer_pubkey_hex).unwrap();
}

// ── 11. Determinism ───────────────────────────────────────────

#[test]
fn same_inputs_produce_byte_identical_wrapper() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let a = sign_with(diff.clone(), &SEED_A, BaselineSignerRole::Operator);
    let b = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    assert_eq!(a, b);
    let ja = serde_json::to_vec(&a).unwrap();
    let jb = serde_json::to_vec(&b).unwrap();
    assert_eq!(ja, jb, "JSON serialization must be byte-stable");
}

// ── 12. Canonical signing-input byte stability ────────────────

#[test]
fn signing_input_is_byte_stable_across_recomputation() {
    let diff = synth_diff(vec![], vec![sample_finding()]);
    let signed = sign_with(diff.clone(), &SEED_A, BaselineSignerRole::Operator);
    let a = signed_integrity_diff_signing_input(
        signed.schema_version,
        &signed.signed_at_utc,
        &signed.signer_pubkey_hex,
        signed.signer_role,
        &signed.diff,
    )
    .unwrap();
    let b = signed_integrity_diff_signing_input(
        SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION,
        NOW_UTC,
        &signed.signer_pubkey_hex,
        BaselineSignerRole::Operator,
        &diff,
    )
    .unwrap();
    assert_eq!(a, b);
    // And it must START with the domain separator.
    let dom = b"OMNINODE-CONTRIBUTOR-SIGNED-INTEGRITY-DIFF:v1:";
    assert!(a.starts_with(dom), "canonical input must lead with domain separator");
}

// ── 13. Sign refuses a non-v1 diff ────────────────────────────

#[test]
fn sign_refuses_non_v1_diff() {
    let mut diff = synth_diff(vec![], vec![sample_finding()]);
    diff.schema_version = 2;
    let s = signer(&SEED_A);
    let err = sign_state_integrity_diff(
        diff,
        &s.pubkey_hex(),
        BaselineSignerRole::Operator,
        NOW_UTC,
        |msg| s.sign(msg),
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported embedded diff schema_version"),
        "got: {msg}"
    );
}

// ── 14. Empty-diff wrapper round-trips ────────────────────────

#[test]
fn empty_diff_wrapper_round_trips() {
    // baseline == current → counts are all zero.
    let diff = synth_diff(vec![], vec![]);
    assert_eq!(diff.counts.new, 0);
    assert_eq!(diff.counts.resolved, 0);
    assert_eq!(diff.counts.unchanged, 0);
    let signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    verify_signed_state_integrity_diff(&signed, &signed.signer_pubkey_hex).unwrap();
    let s = serde_json::to_string(&signed).unwrap();
    let round: SignedStateIntegrityDiff = serde_json::from_str(&s).unwrap();
    assert_eq!(round, signed);
}

// ── 15. Atomic write helper round-trips ───────────────────────

#[test]
fn write_signed_integrity_diff_atomic_round_trips_via_disk() {
    use omni_contributor::{
        read_signed_integrity_diff_from_path, write_signed_integrity_diff_atomic,
    };

    let diff = synth_diff(vec![], vec![sample_finding()]);
    let signed = sign_with(diff, &SEED_A, BaselineSignerRole::Operator);
    let dir = tempfile::TempDir::new().unwrap();
    let out = dir.path().join("diff.signed.json");
    write_signed_integrity_diff_atomic(&signed, &out).unwrap();
    let round = read_signed_integrity_diff_from_path(&out).unwrap();
    assert_eq!(round, signed);
    verify_signed_state_integrity_diff(&round, &round.signer_pubkey_hex).unwrap();
}
