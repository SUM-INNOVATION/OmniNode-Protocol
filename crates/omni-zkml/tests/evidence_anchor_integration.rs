//! Phase 5 Stage 13.0 — integration tests for the chain-anchor
//! surface. Exercises the library's public API the same way the
//! CLI does:
//!
//! 1. Build a real Stage 12.25 `SignedIntegrityEvidenceChainReport`
//!    via `omni-contributor` (dev-dep).
//! 2. Serialize it to bytes and feed the bytes through
//!    `build_anchor_digest` / `submit_evidence_anchor_workflow`.
//! 3. Drive the stub chain client + persistent registry through
//!    the locked-plan refusal taxonomy.
//!
//! Mirrors the test list locked at plan time:
//!
//! - `evidence_anchor_digest_roundtrip`
//! - `evidence_anchor_signed_wire_roundtrip`
//! - `evidence_anchor_artifact_hash_binds_raw_bytes`
//! - `evidence_anchor_artifact_hash_binds_formatting`
//! - `evidence_anchor_rejects_unverified_wrapper`
//! - `evidence_anchor_rejects_seed_file_pubkey_mismatch`
//! - `evidence_anchor_rejects_malformed_seed_file`
//! - `evidence_anchor_rejects_bad_submitter_signature`
//! - `evidence_anchor_rejects_unsupported_anchor_schema_version`
//! - `evidence_anchor_rejects_unsupported_artifact_kind`
//! - `evidence_anchor_rejects_malformed_wrapper_json`
//! - `evidence_anchor_rejects_malformed_anchor_json`
//! - `evidence_anchor_submit_then_query_status_transitions`
//! - `evidence_anchor_verify_default_hash_lookup_ok`
//! - `evidence_anchor_verify_tx_id_lookup_ok`
//! - `evidence_anchor_verify_anchor_not_found`
//! - `evidence_anchor_signing_input_starts_with_domain_tag`

use std::path::PathBuf;

use omni_contributor::{
    BaselineSignerRole, BundleVerifyReport, ChainStepOutcome, ContributorSigner,
    INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION, IntegrityEvidenceChainReport,
    SignedIntegrityEvidenceChainReport, sign_integrity_evidence_chain_report,
};
use omni_zkml::{
    AnchorSelector, AnchorStatus, AnchoredArtifactKind, EVIDENCE_ANCHOR_DOMAIN,
    EvidenceAnchorError, INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION, IntegrityEvidenceAnchorTxData,
    LocalAnchorStatus, LocalEvidenceAnchorRegistry, StubEvidenceAnchorChainClient,
    VerifiedWrapperMetadata, anchor_hex_lower, anchor_signer_pubkey_bytes,
    anchor_signing_input_for_digest, build_anchor_digest, canonical_anchor_bytes,
    evidence_anchor_reason_tag, query_evidence_anchor_workflow, submit_evidence_anchor_workflow,
    verify_anchor_against_registry, verify_anchor_file_against_artifact_bytes,
    verify_anchor_tx_data,
};

// ── Fixture helpers ───────────────────────────────────────────────────────────

/// Build a minimal v1 chain report that successfully sets `all_required_ok = true`.
/// The exact contents don't matter to Stage 13.0 (we only hash
/// the wrapper's raw bytes); we just need a real wrapper the
/// `omni-contributor` signer accepts.
fn fixture_chain_report() -> IntegrityEvidenceChainReport {
    IntegrityEvidenceChainReport {
        schema_version: INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION,
        generated_at_utc: "2026-06-15T12:00:00Z".to_string(),
        omni_contributor_version: "0.1.0".to_string(),
        signed_bundle_path: "/tmp/signed_bundle.json".to_string(),
        effective_base_dir: "/tmp/evidence".to_string(),
        bundle_signature: ChainStepOutcome::Ok,
        bundle_signer_role: BaselineSignerRole::Operator,
        bundle_signer_pubkey_hex: "a".repeat(64),
        bundle_byte_verify: BundleVerifyReport {
            bundle_schema_version: 1,
            bundle_generated_at_utc: "2026-06-15T11:59:00Z".to_string(),
            effective_base_dir: "/tmp/evidence".to_string(),
            counts_ok: 0,
            counts_size_mismatch: 0,
            counts_hash_mismatch: 0,
            counts_not_found: 0,
            counts_read_error: 0,
            entries: vec![],
        },
        child_signatures: vec![],
        counts_child_ok: 0,
        counts_child_skipped: 0,
        counts_child_failed: 0,
    }
}

/// Sign a fixture chain report with the supplied seed and return
/// the persisted JSON bytes (compact serde_json) as the CLI would
/// write them to disk via the Stage 12.25 atomic writer (which
/// uses `serde_json::to_vec_pretty`). We use pretty here so the
/// "formatting" test below has a separate compact byte sequence
/// for the same parsed wrapper.
fn build_signed_wrapper_pretty_bytes(seed: &[u8; 32]) -> Vec<u8> {
    let signer = ContributorSigner::from_seed_bytes(seed).unwrap();
    let wrapper = sign_integrity_evidence_chain_report(
        fixture_chain_report(),
        &signer.pubkey_hex(),
        BaselineSignerRole::Operator,
        "2026-06-15T12:00:00Z",
        |msg| signer.sign(msg),
    )
    .unwrap();
    serde_json::to_vec_pretty(&wrapper).unwrap()
}

fn build_signed_wrapper_compact_bytes(seed: &[u8; 32]) -> Vec<u8> {
    let signer = ContributorSigner::from_seed_bytes(seed).unwrap();
    let wrapper = sign_integrity_evidence_chain_report(
        fixture_chain_report(),
        &signer.pubkey_hex(),
        BaselineSignerRole::Operator,
        "2026-06-15T12:00:00Z",
        |msg| signer.sign(msg),
    )
    .unwrap();
    serde_json::to_vec(&wrapper).unwrap()
}

fn parse_wrapper(bytes: &[u8]) -> SignedIntegrityEvidenceChainReport {
    serde_json::from_slice(bytes).expect("wrapper must parse")
}

fn wrapper_metadata_for_anchor(
    wrapper: &SignedIntegrityEvidenceChainReport,
) -> VerifiedWrapperMetadata {
    let mut signer_pubkey = [0u8; 32];
    let bytes: Vec<u8> = (0..32)
        .map(|i| {
            let s = &wrapper.signer_pubkey_hex[i * 2..i * 2 + 2];
            u8::from_str_radix(s, 16).unwrap()
        })
        .collect();
    signer_pubkey.copy_from_slice(&bytes);
    let signed_at_utc_unix = chrono::DateTime::parse_from_rfc3339(&wrapper.signed_at_utc)
        .unwrap()
        .timestamp();
    VerifiedWrapperMetadata {
        artifact_schema_version: wrapper.schema_version,
        signer_pubkey,
        signed_at_utc_unix,
    }
}

fn fresh_registry() -> (tempfile::TempDir, LocalEvidenceAnchorRegistry) {
    let dir = tempfile::tempdir().unwrap();
    let reg = LocalEvidenceAnchorRegistry::open(dir.path().join("anchors")).unwrap();
    (dir, reg)
}

// ── 1. Digest + signing-input determinism ─────────────────────────────────────

#[test]
fn evidence_anchor_digest_roundtrip() {
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let d1 = build_anchor_digest(&metadata, &raw);
    let d2 = build_anchor_digest(&metadata, &raw);
    assert_eq!(d1, d2);
    let a = canonical_anchor_bytes(&d1).unwrap();
    let b = canonical_anchor_bytes(&d2).unwrap();
    assert_eq!(a, b);
}

#[test]
fn evidence_anchor_signing_input_starts_with_domain_tag() {
    let seed = [9u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let digest = build_anchor_digest(&metadata, &raw);
    let signing = anchor_signing_input_for_digest(&digest).unwrap();
    assert_eq!(
        &signing[..EVIDENCE_ANCHOR_DOMAIN.len()],
        EVIDENCE_ANCHOR_DOMAIN
    );
}

// ── 2. Signed wire roundtrip ──────────────────────────────────────────────────

#[test]
fn evidence_anchor_signed_wire_roundtrip() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let digest = build_anchor_digest(&metadata, &raw);

    let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
    // Verify the persisted record's tx_data signature under
    // its own signer_pubkey (same-key submitter rule).
    verify_anchor_tx_data(&record.tx_data).unwrap();
}

// ── 3. Raw-byte hash sensitivity (Finding 1 from REJECT v1) ───────────────────

#[test]
fn evidence_anchor_artifact_hash_binds_raw_bytes() {
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let digest_original = build_anchor_digest(&metadata, &raw);

    // Mutate ONE byte of the raw on-disk artifact bytes.
    let mut mutated = raw.clone();
    let last = mutated.len() - 2;
    mutated[last] ^= 0x20;
    let digest_mutated = build_anchor_digest(&metadata, &mutated);

    assert_ne!(
        digest_original.artifact_hash, digest_mutated.artifact_hash,
        "single-byte mutation must produce a different artifact_hash"
    );
}

/// Same parsed wrapper, two different on-disk byte sequences
/// (pretty vs compact JSON) → two different anchors. Pins
/// "hash the bytes, not the structure".
#[test]
fn evidence_anchor_artifact_hash_binds_formatting() {
    let seed = [7u8; 32];
    let pretty = build_signed_wrapper_pretty_bytes(&seed);
    let compact = build_signed_wrapper_compact_bytes(&seed);
    assert_ne!(pretty, compact, "pretty and compact JSON must differ");

    let wrapper_pretty = parse_wrapper(&pretty);
    let wrapper_compact = parse_wrapper(&compact);
    // Sanity: structurally the wrappers are identical.
    assert_eq!(wrapper_pretty, wrapper_compact);

    let m = wrapper_metadata_for_anchor(&wrapper_pretty);
    let d_pretty = build_anchor_digest(&m, &pretty);
    let d_compact = build_anchor_digest(&m, &compact);
    assert_ne!(
        d_pretty.artifact_hash, d_compact.artifact_hash,
        "different on-disk formatting must produce different anchors"
    );
}

// ── 4. Wrapper-signature refusal (Finding 2 from REJECT v1) ───────────────────
//
// Library-level analogue: feed bytes whose embedded
// `signature_hex` does NOT verify under `signer_pubkey_hex`.
// The CLI runs `verify_signed_integrity_evidence_chain_report`
// up-front and refuses with reason=wrapper_signature_invalid;
// the library never sees the wrapper type, so this test asserts
// the contributor verifier's refusal directly.

#[test]
fn evidence_anchor_rejects_unverified_wrapper() {
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let mut wrapper = parse_wrapper(&raw);
    // Corrupt the signature hex.
    wrapper.signature_hex.replace_range(0..2, "00");
    let err = omni_contributor::verify_signed_integrity_evidence_chain_report(
        &wrapper,
        &wrapper.signer_pubkey_hex,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        omni_contributor::SignedIntegrityEvidenceChainReportError::SignatureMismatch
    ));
}

// ── 5. Submitter-seed refusal (same-key submitter rule) ──────────────────────

#[test]
fn evidence_anchor_rejects_seed_file_pubkey_mismatch() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let wrapper_seed = [7u8; 32];
    let bad_submitter_seed = [9u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&wrapper_seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let digest = build_anchor_digest(&metadata, &raw);
    let err =
        submit_evidence_anchor_workflow(&reg, &client, digest, &bad_submitter_seed).unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::SubmitterPubkeyMismatch { .. }
    ));
    assert_eq!(
        evidence_anchor_reason_tag(&err),
        "submitter_pubkey_mismatch"
    );
}

// ── 6. Malformed seed file (CLI-side concern, but library
//      surfaces a typed error if asked) ─────────────────────────

#[test]
fn evidence_anchor_rejects_malformed_seed_file() {
    // Simulate the CLI's seed-file read by constructing the error
    // directly. The CLI's `read_seed_file` helper is gated under
    // `submit`, so we exercise the typed error path here.
    let path = PathBuf::from("/tmp/bogus_seed");
    let err = EvidenceAnchorError::MalformedSeedFile {
        path: path.clone(),
        reason: "expected 32 bytes, got 5".to_string(),
    };
    assert_eq!(evidence_anchor_reason_tag(&err), "malformed_seed_file");
}

// ── 7. Tampered submitter signature ──────────────────────────────────────────

#[test]
fn evidence_anchor_rejects_bad_submitter_signature() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let signer = metadata.signer_pubkey;
    let digest = build_anchor_digest(&metadata, &raw);
    let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();

    // Tamper the submitter signature and run verify against the
    // wire payload directly (standalone-JSON path).
    let mut bad = record.tx_data.clone();
    bad.submitter_signature[5] ^= 0x01;
    let err = verify_anchor_file_against_artifact_bytes(&bad, &raw, &signer).unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::SubmitterSignatureInvalid
    ));
    assert_eq!(
        evidence_anchor_reason_tag(&err),
        "submitter_signature_invalid"
    );
}

// ── 8. Unsupported anchor schema version ──────────────────────────────────────

#[test]
fn evidence_anchor_rejects_unsupported_anchor_schema_version() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let digest = build_anchor_digest(&metadata, &raw);
    let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();

    // Tamper the stored wire payload's schema version.
    let mut bad = record.tx_data.clone();
    bad.digest.anchor_schema_version = 999;
    let err = verify_anchor_tx_data(&bad).unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::UnsupportedAnchorSchemaVersion {
            got: 999,
            expected: 1
        }
    ));
    assert_eq!(
        evidence_anchor_reason_tag(&err),
        "unsupported_anchor_schema_version"
    );
}

// ── 9. Unsupported artifact kind (serde refuses unknown variants) ─────────────

#[test]
fn evidence_anchor_rejects_unsupported_artifact_kind() {
    // The CLI parses anchor JSON via serde; an unknown
    // `artifact_kind` string is refused at parse time.
    let bogus = serde_json::json!({
        "digest": {
            "anchor_schema_version": 1,
            "artifact_kind": "rogue_artifact_kind",
            "artifact_schema_version": 1,
            "artifact_hash": vec![0u8; 32],
            "signer_pubkey": vec![0u8; 32],
            "signed_at_utc_unix": 0i64,
        },
        "submitter_signature": vec![0u8; 64],
    });
    let result: Result<IntegrityEvidenceAnchorTxData, _> = serde_json::from_value(bogus);
    assert!(
        result.is_err(),
        "unknown artifact_kind must refuse at parse time"
    );
}

// ── 10. Malformed wrapper JSON ────────────────────────────────────────────────

#[test]
fn evidence_anchor_rejects_malformed_wrapper_json() {
    let bogus_bytes = b"{this is not JSON";
    let parsed: Result<SignedIntegrityEvidenceChainReport, _> = serde_json::from_slice(bogus_bytes);
    assert!(parsed.is_err());
}

// ── 11. Malformed anchor JSON ─────────────────────────────────────────────────

#[test]
fn evidence_anchor_rejects_malformed_anchor_json() {
    let bogus_bytes = b"{still not JSON";
    let parsed: Result<IntegrityEvidenceAnchorTxData, _> = serde_json::from_slice(bogus_bytes);
    assert!(parsed.is_err());
}

// ── 12. Submit → query status transitions ────────────────────────────────────

#[test]
fn evidence_anchor_submit_then_query_status_transitions() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let digest = build_anchor_digest(&metadata, &raw);
    let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
    assert_eq!(record.status, LocalAnchorStatus::Submitted);

    // chain transitions: Submitted → Included → Finalized.
    client.set_status_for(&record.receipt.tx_id, AnchorStatus::Included);
    let outcome =
        query_evidence_anchor_workflow(&reg, &client, AnchorSelector::TxId(&record.receipt.tx_id))
            .unwrap();
    assert_eq!(outcome.record.status, LocalAnchorStatus::Included);

    client.set_status_for(&record.receipt.tx_id, AnchorStatus::Finalized);
    let outcome =
        query_evidence_anchor_workflow(&reg, &client, AnchorSelector::TxId(&record.receipt.tx_id))
            .unwrap();
    assert_eq!(outcome.record.status, LocalAnchorStatus::Finalized);

    // Persisted: reload by hash.
    let by_hash = reg
        .load_by_artifact_hash(&record.artifact_hash_hex)
        .unwrap()
        .unwrap();
    assert_eq!(by_hash.status, LocalAnchorStatus::Finalized);
}

// ── 13. verify by default hash lookup ─────────────────────────────────────────

#[test]
fn evidence_anchor_verify_default_hash_lookup_ok() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let signer = metadata.signer_pubkey;
    let digest = build_anchor_digest(&metadata, &raw);
    submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
    let record = verify_anchor_against_registry(&reg, &raw, &signer, None).unwrap();
    assert_eq!(record.status, LocalAnchorStatus::Submitted);
    assert_eq!(
        record.artifact_hash_hex,
        anchor_hex_lower(blake3::hash(&raw).as_bytes())
    );
}

// ── 14. verify by tx_id lookup ────────────────────────────────────────────────

#[test]
fn evidence_anchor_verify_tx_id_lookup_ok() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let wrapper = parse_wrapper(&raw);
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let signer = metadata.signer_pubkey;
    let digest = build_anchor_digest(&metadata, &raw);
    let submitted = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
    let record =
        verify_anchor_against_registry(&reg, &raw, &signer, Some(&submitted.receipt.tx_id))
            .unwrap();
    assert_eq!(record.artifact_hash_hex, submitted.artifact_hash_hex);
}

// ── 15. verify miss → anchor_not_found ───────────────────────────────────────

#[test]
fn evidence_anchor_verify_anchor_not_found() {
    let (_dir, reg) = fresh_registry();
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);
    let signer = anchor_signer_pubkey_bytes(&seed).unwrap();
    // Nothing has been submitted — verify must miss.
    let err = verify_anchor_against_registry(&reg, &raw, &signer, None).unwrap_err();
    assert!(matches!(err, EvidenceAnchorError::AnchorNotFound { .. }));
    assert_eq!(evidence_anchor_reason_tag(&err), "anchor_not_found");
}

// ── 15a. same-hash, wrong-but-valid signer (registry-backed) ──────────────────
//
// Defends the same-key submitter rule at verify time: a
// hand-edited registry record (or a record submitted by a
// different key that happens to share the artifact hash)
// must be refused when the wrapper signer doesn't match the
// anchor's `digest.signer_pubkey`.

#[test]
fn evidence_anchor_verify_registry_refuses_wrong_signer_with_same_hash() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let wrapper_seed = [7u8; 32];
    let attacker_seed = [9u8; 32];
    // The on-disk wrapper is built / signed by `wrapper_seed`.
    let raw = build_signed_wrapper_pretty_bytes(&wrapper_seed);
    let wrapper = parse_wrapper(&raw);
    let wrapper_signer = wrapper_metadata_for_anchor(&wrapper).signer_pubkey;
    // The anchor in the registry was submitted by the ATTACKER
    // key (committing the same artifact hash). Library-level
    // submit doesn't refuse this; the CLI's pre-submit gate
    // (same-key submitter) does — but a registry mutated
    // afterwards or imported from another host can carry
    // exactly this shape.
    let attacker_metadata = VerifiedWrapperMetadata {
        artifact_schema_version: wrapper.schema_version,
        signer_pubkey: anchor_signer_pubkey_bytes(&attacker_seed).unwrap(),
        signed_at_utc_unix: chrono::DateTime::parse_from_rfc3339(&wrapper.signed_at_utc)
            .unwrap()
            .timestamp(),
    };
    let attacker_digest = build_anchor_digest(&attacker_metadata, &raw);
    let attacker_record =
        submit_evidence_anchor_workflow(&reg, &client, attacker_digest, &attacker_seed).unwrap();

    let err = verify_anchor_against_registry(
        &reg,
        &raw,
        &wrapper_signer,
        Some(&attacker_record.receipt.tx_id),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::AnchoredSignerPubkeyMismatch { .. }
    ));
    assert_eq!(
        evidence_anchor_reason_tag(&err),
        "anchored_signer_pubkey_mismatch"
    );
}

// ── 15b. same-hash, wrong-but-valid signer (standalone JSON) ──────────────────

#[test]
fn evidence_anchor_verify_file_refuses_wrong_signer_with_same_hash() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let wrapper_seed = [7u8; 32];
    let attacker_seed = [9u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&wrapper_seed);
    let wrapper = parse_wrapper(&raw);
    let wrapper_signer = wrapper_metadata_for_anchor(&wrapper).signer_pubkey;

    let attacker_metadata = VerifiedWrapperMetadata {
        artifact_schema_version: wrapper.schema_version,
        signer_pubkey: anchor_signer_pubkey_bytes(&attacker_seed).unwrap(),
        signed_at_utc_unix: chrono::DateTime::parse_from_rfc3339(&wrapper.signed_at_utc)
            .unwrap()
            .timestamp(),
    };
    let attacker_digest = build_anchor_digest(&attacker_metadata, &raw);
    let attacker_record =
        submit_evidence_anchor_workflow(&reg, &client, attacker_digest, &attacker_seed).unwrap();
    let bad_tx_data = attacker_record.tx_data.clone();

    let err =
        verify_anchor_file_against_artifact_bytes(&bad_tx_data, &raw, &wrapper_signer)
            .unwrap_err();
    assert!(matches!(
        err,
        EvidenceAnchorError::AnchoredSignerPubkeyMismatch { .. }
    ));
    assert_eq!(
        evidence_anchor_reason_tag(&err),
        "anchored_signer_pubkey_mismatch"
    );
}

// ── 16. Schema-version constant is exactly 1 ──────────────────────────────────

#[test]
fn evidence_anchor_schema_version_is_locked_at_1() {
    assert_eq!(INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION, 1);
}

// ── 17. Closed-set kind enum has exactly one variant for Stage 13.0 ─────────

#[test]
fn evidence_anchor_artifact_kind_is_closed_and_single_variant_for_stage_13_0() {
    let kind = AnchoredArtifactKind::SignedIntegrityEvidenceChainReport;
    assert_eq!(kind.as_str(), "signed_integrity_evidence_chain_report");
    let s = serde_json::to_string(&kind).unwrap();
    assert_eq!(s, "\"signed_integrity_evidence_chain_report\"");
}

// ── 18. End-to-end CLI-shaped flow ────────────────────────────────────────────
//
// Reproduces the CLI's full pipeline at the library level: parse
// raw wrapper bytes → verify wrapper signature → extract
// metadata → submit anchor → query → verify (registry-backed) →
// verify-file (standalone JSON). One test, one fixture, six
// steps — covers the operator's golden path end-to-end.

#[test]
fn evidence_anchor_end_to_end_cli_shaped_flow() {
    let (_dir, reg) = fresh_registry();
    let client = StubEvidenceAnchorChainClient::new();
    let seed = [7u8; 32];
    let raw = build_signed_wrapper_pretty_bytes(&seed);

    // Step 1: parse + verify wrapper (CLI's pre-submit gates).
    let wrapper = parse_wrapper(&raw);
    omni_contributor::verify_signed_integrity_evidence_chain_report(
        &wrapper,
        &wrapper.signer_pubkey_hex,
    )
    .unwrap();

    // Step 2: extract metadata + build digest.
    let metadata = wrapper_metadata_for_anchor(&wrapper);
    let signer_pubkey = metadata.signer_pubkey;
    let digest = build_anchor_digest(&metadata, &raw);
    assert_eq!(
        digest.signer_pubkey,
        anchor_signer_pubkey_bytes(&seed).unwrap()
    );

    // Step 3: submit anchor.
    let record = submit_evidence_anchor_workflow(&reg, &client, digest, &seed).unwrap();
    assert_eq!(record.status, LocalAnchorStatus::Submitted);

    // Step 4: query (default Submitted).
    let outcome = query_evidence_anchor_workflow(
        &reg,
        &client,
        AnchorSelector::ArtifactHashHex(&record.artifact_hash_hex),
    )
    .unwrap();
    assert_eq!(outcome.record.status, LocalAnchorStatus::Submitted);

    // Step 5: registry-backed verify (default hash lookup) —
    // bound to the wrapper signer.
    let verified = verify_anchor_against_registry(&reg, &raw, &signer_pubkey, None).unwrap();
    assert_eq!(verified.artifact_hash_hex, record.artifact_hash_hex);

    // Step 6: standalone-JSON verify against the persisted
    // wire payload — also bound to the wrapper signer.
    verify_anchor_file_against_artifact_bytes(&record.tx_data, &raw, &signer_pubkey).unwrap();
}
