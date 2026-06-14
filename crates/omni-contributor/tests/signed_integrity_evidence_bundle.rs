//! Stage 12.23 — integration tests for the signed
//! integrity-evidence-bundle wrapper.

use std::fs;
use std::path::Path;

use omni_contributor::{
    build_integrity_evidence_bundle, sign_integrity_evidence_bundle,
    signed_integrity_evidence_bundle_signing_input,
    verify_integrity_evidence_bundle,
    verify_signed_integrity_evidence_bundle, BaselineSignerRole,
    BundleArtifactKind, BundleBuilderInput, BundleBuilderOptions,
    BundleVerifyOptions, ContributorSigner, IntegrityEvidenceBundle,
    SignedIntegrityEvidenceBundle, SignedIntegrityEvidenceBundleError,
    SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
};

const NOW_UTC: &str = "2026-06-14T12:00:00Z";

const SEED_A: [u8; 32] = *b"stage12.23-bundle-signer-aaa-32!";
const SEED_B: [u8; 32] = *b"stage12.23-bundle-signer-bbb-32!";

fn signer(seed: &[u8; 32]) -> ContributorSigner {
    ContributorSigner::from_seed_bytes(seed).unwrap()
}

fn write_file(dir: &Path, rel: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = dir.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&p, bytes).unwrap();
    p
}

/// Build a real v1 IntegrityEvidenceBundle in a temp dir so
/// each test signs an authentic Stage 12.22 artifact. Returns
/// (TempDir kept alive for the test lifetime, bundle).
fn synth_bundle_with_two_entries() -> (tempfile::TempDir, IntegrityEvidenceBundle) {
    let dir = tempfile::TempDir::new().unwrap();
    let base = dir.path();
    write_file(base, "baseline.json", b"baseline-bytes");
    write_file(base, "diff.json", b"diff-bytes");
    let inputs = vec![
        BundleBuilderInput {
            artifact_kind: BundleArtifactKind::StateIntegrityReport,
            path: Path::new("baseline.json"),
        },
        BundleBuilderInput {
            artifact_kind: BundleArtifactKind::StateIntegrityDiffReport,
            path: Path::new("diff.json"),
        },
    ];
    let bundle = build_integrity_evidence_bundle(
        &inputs,
        &BundleBuilderOptions {
            now_utc: "2026-06-14T11:30:00Z",
            base_dir: base,
            label: Some("test-bundle"),
            notes: None,
        },
    )
    .unwrap();
    (dir, bundle)
}

fn sign_with(
    bundle: IntegrityEvidenceBundle,
    seed: &[u8; 32],
    role: BaselineSignerRole,
) -> SignedIntegrityEvidenceBundle {
    let s = signer(seed);
    let pubkey = s.pubkey_hex();
    sign_integrity_evidence_bundle(bundle, &pubkey, role, NOW_UTC, |msg| {
        s.sign(msg)
    })
    .unwrap()
}

// ── 1. Happy path: sign + verify ──────────────────────────────

#[test]
fn sign_and_verify_round_trips() {
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let signed = sign_with(bundle.clone(), &SEED_A, BaselineSignerRole::Operator);
    assert_eq!(
        signed.schema_version,
        SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION
    );
    assert_eq!(signed.signer_role, BaselineSignerRole::Operator);
    assert_eq!(signed.signed_at_utc, NOW_UTC);
    assert_eq!(signed.bundle, bundle);
    verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
        .unwrap();
}

// ── 2. All four signer-role variants round-trip ───────────────

#[test]
fn all_signer_role_variants_round_trip() {
    let (_dir, bundle) = synth_bundle_with_two_entries();
    for role in [
        BaselineSignerRole::Operator,
        BaselineSignerRole::Contributor,
        BaselineSignerRole::Dispatcher,
        BaselineSignerRole::Coordinator,
    ] {
        let signed = sign_with(bundle.clone(), &SEED_A, role);
        verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
            .unwrap();
        assert_eq!(signed.signer_role, role);
    }
}

// ── 3. Pubkey mismatch refused (cheap pre-check) ──────────────

#[test]
fn pubkey_mismatch_is_refused_before_crypto() {
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    let key_b_pubkey = signer(&SEED_B).pubkey_hex();
    let err = verify_signed_integrity_evidence_bundle(&signed, &key_b_pubkey)
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
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let mut signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    let bytes = unsafe { signed.signature_hex.as_bytes_mut() };
    bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
    let err = verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("signature mismatch"),
        "expected SignatureMismatch, got: {msg}"
    );
}

// ── 5. Embedded-bundle tamper refused ─────────────────────────

#[test]
fn embedded_bundle_tamper_is_refused() {
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let mut signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    // Flip a byte of an embedded entry's blake3_hex — proves
    // every byte of the bundle (including per-entry hashes) is
    // in the canonical body.
    let hex = &mut signed.bundle.entries[0].blake3_hex;
    let bytes = unsafe { hex.as_bytes_mut() };
    bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
    let err = verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
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
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let mut signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    signed.signer_role = BaselineSignerRole::Coordinator;
    let err = verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
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
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let mut signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    signed.signed_at_utc = "2099-01-01T00:00:00Z".to_string();
    let err = verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
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
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let mut signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    signed.schema_version = 2;
    let err = verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceBundleError::UnsupportedSchemaVersion {
                got: 2,
                expected: 1
            }
        ),
        "got: {err:?}"
    );
}

// ── 9. Embedded bundle schema_version != 1 refused ────────────

#[test]
fn embedded_bundle_schema_v2_is_refused() {
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let mut signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    signed.bundle.schema_version = 2;
    let err = verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
        .unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceBundleError::UnsupportedBundleSchemaVersion {
                got: 2,
                expected: 1
            }
        ),
        "got: {err:?}"
    );
}

// ── 10. Sign refuses non-v1 input bundle ──────────────────────

#[test]
fn sign_refuses_non_v1_bundle() {
    let (_dir, mut bundle) = synth_bundle_with_two_entries();
    bundle.schema_version = 2;
    let s = signer(&SEED_A);
    let err = sign_integrity_evidence_bundle(
        bundle,
        &s.pubkey_hex(),
        BaselineSignerRole::Operator,
        NOW_UTC,
        |msg| s.sign(msg),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceBundleError::UnsupportedBundleSchemaVersion {
                got: 2,
                expected: 1
            }
        ),
        "got: {err:?}"
    );
}

// ── 11. JSON round-trip preserves signature ──────────────────

#[test]
fn json_round_trip_preserves_signature() {
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    let s = serde_json::to_string(&signed).unwrap();
    let round: SignedIntegrityEvidenceBundle = serde_json::from_str(&s).unwrap();
    assert_eq!(round, signed);
    verify_signed_integrity_evidence_bundle(&round, &round.signer_pubkey_hex)
        .unwrap();
}

// ── 12. Determinism ───────────────────────────────────────────

#[test]
fn same_inputs_produce_byte_identical_wrapper() {
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let a = sign_with(bundle.clone(), &SEED_A, BaselineSignerRole::Operator);
    let b = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    assert_eq!(a, b);
    let ja = serde_json::to_vec(&a).unwrap();
    let jb = serde_json::to_vec(&b).unwrap();
    assert_eq!(ja, jb, "JSON serialization must be byte-stable");
}

// ── 13. Canonical signing-input byte stability + domain ───────

#[test]
fn signing_input_is_byte_stable_across_recomputation() {
    let (_dir, bundle) = synth_bundle_with_two_entries();
    let signed = sign_with(bundle.clone(), &SEED_A, BaselineSignerRole::Operator);
    let a = signed_integrity_evidence_bundle_signing_input(
        signed.schema_version,
        &signed.signed_at_utc,
        &signed.signer_pubkey_hex,
        signed.signer_role,
        &signed.bundle,
    )
    .unwrap();
    let b = signed_integrity_evidence_bundle_signing_input(
        SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        NOW_UTC,
        &signed.signer_pubkey_hex,
        BaselineSignerRole::Operator,
        &bundle,
    )
    .unwrap();
    assert_eq!(a, b);
    // And it must START with the domain separator.
    let dom = b"OMNINODE-CONTRIBUTOR-SIGNED-INTEGRITY-EVIDENCE-BUNDLE:v1:";
    assert!(
        a.starts_with(dom),
        "canonical input must lead with domain separator"
    );
}

// ── 14. Atomic write helper round-trips via disk ──────────────

#[test]
fn write_signed_atomic_round_trips_via_disk() {
    use omni_contributor::{
        read_signed_integrity_evidence_bundle_from_path,
        write_signed_integrity_evidence_bundle_atomic,
    };

    let (_dir, bundle) = synth_bundle_with_two_entries();
    let signed = sign_with(bundle, &SEED_A, BaselineSignerRole::Operator);
    let out_dir = tempfile::TempDir::new().unwrap();
    let out = out_dir.path().join("bundle.signed.json");
    write_signed_integrity_evidence_bundle_atomic(&signed, &out).unwrap();
    let round = read_signed_integrity_evidence_bundle_from_path(&out).unwrap();
    assert_eq!(round, signed);
    verify_signed_integrity_evidence_bundle(&round, &round.signer_pubkey_hex)
        .unwrap();
}

// ── 15. Composition with Stage 12.22 ──────────────────────────

#[test]
fn signed_wrapper_composes_with_stage_12_22_bundle_verifier() {
    // End-to-end chain:
    //   1. build_integrity_evidence_bundle (Stage 12.22)
    //   2. sign_integrity_evidence_bundle  (Stage 12.23)
    //   3. verify_signed_integrity_evidence_bundle (Stage 12.23 sig OK)
    //   4. verify_integrity_evidence_bundle on the embedded
    //      bundle (Stage 12.22 byte verify OK — no entries
    //      touched between build and verify)
    // All four must pass without re-hashing artifact files
    // anywhere in the signature verification stage.
    let (dir, bundle) = synth_bundle_with_two_entries();
    let signed = sign_with(bundle.clone(), &SEED_A, BaselineSignerRole::Operator);

    // Stage 12.23 — signature OK.
    verify_signed_integrity_evidence_bundle(&signed, &signed.signer_pubkey_hex)
        .unwrap();

    // Stage 12.22 — byte verify on the embedded bundle still
    // resolves through the canonical base_dir. The embedded
    // bundle was built against `dir.path()`; its base_dir is
    // recorded canonically.
    let report = verify_integrity_evidence_bundle(
        &signed.bundle,
        &BundleVerifyOptions::default(),
    )
    .unwrap();
    assert!(report.all_ok(), "stage 12.22 verify on embedded bundle: {report:?}");

    // Sanity: the embedded bundle still matches what
    // build_integrity_evidence_bundle produced.
    assert_eq!(signed.bundle, bundle);
    drop(dir); // keep the temp dir alive until here
}

// ── 16. Bundle read distinguishes Io from MalformedJson ───────

#[test]
fn signed_bundle_read_missing_file_returns_io_not_malformed_json() {
    use omni_contributor::read_signed_integrity_evidence_bundle_from_path;
    // Mirrors the Stage 12.22 review-fix regression: pins the
    // library variant the CLI's `reason=io` tag depends on. A
    // missing wrapper file must surface as `Io`, NOT
    // `MalformedJson`.
    let dir = tempfile::TempDir::new().unwrap();
    let missing = dir.path().join("never-written.json");
    assert!(!missing.exists());
    let err =
        read_signed_integrity_evidence_bundle_from_path(&missing).unwrap_err();
    assert!(
        matches!(
            err,
            SignedIntegrityEvidenceBundleError::Io { ref source, .. }
            if source.kind() == std::io::ErrorKind::NotFound
        ),
        "expected Io {{ NotFound }}, got: {err:?}"
    );
}

#[test]
fn signed_bundle_read_invalid_json_returns_malformed_json() {
    use omni_contributor::read_signed_integrity_evidence_bundle_from_path;
    let dir = tempfile::TempDir::new().unwrap();
    let bad = dir.path().join("bad.json");
    fs::write(&bad, b"{not valid json at all").unwrap();
    let err =
        read_signed_integrity_evidence_bundle_from_path(&bad).unwrap_err();
    assert!(
        matches!(err, SignedIntegrityEvidenceBundleError::MalformedJson { .. }),
        "expected MalformedJson, got: {err:?}"
    );
}
