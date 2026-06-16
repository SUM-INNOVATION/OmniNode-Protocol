//! Phase 5 Stage 13.1 — chain-team review wire vectors for the
//! `IntegrityEvidenceAnchorTxData` payload frozen in Stage 13.0.
//!
//! Three deterministic test vectors the chain team consumes to
//! assert byte-for-byte parity with their planned
//! `sumchain-primitives::integrity_evidence_anchor::IntegrityEvidenceAnchorTxData`
//! module.
//!
//! Mirrors the Stage 6 `chain_attestation_vectors.rs` posture:
//!
//! - Default mode: **verify**. Compute the three wire vectors via
//!   the actual `omni-zkml` implementation path and assert
//!   byte-equality against the committed fixture at
//!   `tests/fixtures/evidence_anchor_wire_vectors.json`.
//! - Regen mode: set `OMNINODE_REGEN_VECTORS=1` to overwrite the
//!   committed fixture with the freshly-computed vectors.
//!   Explicit-only — a developer runs it once when intentionally
//!   bumping the spec, then commits the regenerated JSON.
//!
//! ## What "frozen" means here
//!
//! These vectors pin the Stage 13.0 wire under
//! `INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION = 1` and
//! `EVIDENCE_ANCHOR_DOMAIN = b"OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:"`.
//! Any drift in canonical bytes / signing input / signature for the
//! same inputs is a wire-breaking change; the test fails with a
//! pointer to the regen toggle and a reminder that regen is
//! reserved for an intentional version bump.

use std::path::PathBuf;

use omni_zkml::{
    anchor_hex_lower, anchor_signing_input_for_digest, canonical_anchor_bytes,
    parse_anchor_hex_32, sign_anchor_digest, AnchoredArtifactKind,
    IntegrityEvidenceAnchorDigest, IntegrityEvidenceAnchorTxData,
    INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
};

/// One input row for a Stage 13.1 wire vector. All hex strings are
/// 64-char lowercase (= 32 bytes). `signed_at_utc_unix` is the
/// `i64` value embedded in the digest, recorded verbatim.
///
/// Note `signer_pubkey_hex` is deliberately NOT an input — it is
/// derived from `submitter_seed_hex` by the test (Stage 13.0
/// same-key submitter rule: the chain payload's
/// `digest.signer_pubkey` MUST equal the submitter's pubkey).
/// Recording the derived value in the fixture lets the chain team
/// rebuild and verify it from scratch.
struct Input {
    label: &'static str,
    artifact_schema_version: u32,
    artifact_hash_hex: &'static str,
    signed_at_utc_unix: i64,
    submitter_seed_hex: &'static str,
}

/// Three locked vectors (per the Stage 13.1 approval):
/// 1. **normal** — realistic submit shape: non-zero artifact
///    hash, a non-trivial seed (every byte = `0x01`), contemporary
///    Unix timestamp.
/// 2. **minimal** — all-zero artifact hash, all-zero seed,
///    `signed_at_utc_unix = 0` (Unix epoch).
/// 3. **high-entropy** — every byte of the artifact hash and seed
///    set to `0xFF`, large positive `signed_at_utc_unix`.
///
/// Each row's `signer_pubkey` is **derived** from the seed via
/// `omni_zkml::anchor_signer_pubkey_bytes` and recorded in the
/// JSON fixture. This binds every vector to the Stage 13.0
/// same-key submitter rule (chain payload's
/// `digest.signer_pubkey` always equals the submitter's pubkey)
/// and lets a chain-team reviewer rebuild the entire vector from
/// just the inputs.
const VECTORS: &[Input] = &[
    Input {
        label: "stage13.1-vec-1-normal",
        artifact_schema_version: 1,
        // Non-zero, non-degenerate artifact hash.
        artifact_hash_hex:
            "1739d0e5e3e3b2bc63b67ef58ca7a99a4b50a5f3cf08af3c5e9d5d5d28d1c4b3",
        signed_at_utc_unix: 1_750_000_000,
        submitter_seed_hex:
            "0101010101010101010101010101010101010101010101010101010101010101",
    },
    Input {
        label: "stage13.1-vec-2-minimal",
        artifact_schema_version: 1,
        artifact_hash_hex:
            "0000000000000000000000000000000000000000000000000000000000000000",
        signed_at_utc_unix: 0,
        submitter_seed_hex:
            "0000000000000000000000000000000000000000000000000000000000000000",
    },
    Input {
        label: "stage13.1-vec-3-high-entropy",
        artifact_schema_version: 1,
        artifact_hash_hex:
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        signed_at_utc_unix: 4_102_444_800, // 2100-01-01T00:00:00Z
        submitter_seed_hex:
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    },
];

/// Output JSON shape — one record per vector. Field order and
/// names are stable; the chain team reads this verbatim and is
/// free to assume `serde_json::from_str::<Vec<Vector>>` works.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct Vector {
    label: String,

    // ── Inputs (verbatim from the test source) ─────────────────
    artifact_schema_version: u32,
    artifact_hash_hex: String,
    signer_pubkey_hex: String,
    signed_at_utc_unix: i64,
    submitter_seed_hex: String,

    // ── Constants used (so the chain team has them in-line) ───
    anchor_schema_version: u32,
    artifact_kind: String, // = "signed_integrity_evidence_chain_report"
    evidence_anchor_domain_ascii: String, // = "OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:"

    // ── Derived (the contract pin) ─────────────────────────────
    canonical_digest_bytes_hex: String,
    signing_input_bytes_hex: String,
    submitter_signature_hex: String,
    /// The full signed wire payload serialized via `serde_json`
    /// for the chain team's eyeball check / quick rebuild.
    wire_payload_json: serde_json::Value,
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("evidence_anchor_wire_vectors.json")
}

fn compute_vector(input: &Input) -> Vector {
    let artifact_hash = parse_anchor_hex_32(input.artifact_hash_hex)
        .unwrap_or_else(|e| panic!("vector {}: artifact_hash_hex: {e}", input.label));
    let seed = parse_anchor_hex_32(input.submitter_seed_hex)
        .unwrap_or_else(|e| panic!("vector {}: submitter_seed_hex: {e}", input.label));

    // Same-key submitter rule (Stage 13.0): the chain payload's
    // `digest.signer_pubkey` MUST equal the submitter's derived
    // pubkey. Derive it deterministically here.
    let signer_pubkey = omni_zkml::anchor_signer_pubkey_bytes(&seed)
        .unwrap_or_else(|e| panic!("vector {}: derive submitter pubkey: {e}", input.label));

    let digest = IntegrityEvidenceAnchorDigest {
        anchor_schema_version: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
        artifact_kind: AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
        artifact_schema_version: input.artifact_schema_version,
        artifact_hash,
        signer_pubkey,
        signed_at_utc_unix: input.signed_at_utc_unix,
    };
    let canonical = canonical_anchor_bytes(&digest).unwrap();
    let signing_input = anchor_signing_input_for_digest(&digest).unwrap();
    let signature = sign_anchor_digest(&seed, &digest).unwrap();
    let tx_data = IntegrityEvidenceAnchorTxData {
        digest: digest.clone(),
        submitter_signature: signature,
    };
    let wire_payload_json =
        serde_json::to_value(&tx_data).expect("wire payload serializes as JSON");

    Vector {
        label: input.label.to_string(),
        artifact_schema_version: input.artifact_schema_version,
        artifact_hash_hex: input.artifact_hash_hex.to_string(),
        signer_pubkey_hex: anchor_hex_lower(&signer_pubkey),
        signed_at_utc_unix: input.signed_at_utc_unix,
        submitter_seed_hex: input.submitter_seed_hex.to_string(),

        anchor_schema_version: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
        artifact_kind: AnchoredArtifactKind::SignedIntegrityEvidenceChainReport
            .as_str()
            .to_string(),
        evidence_anchor_domain_ascii:
            "OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:".to_string(),

        canonical_digest_bytes_hex: anchor_hex_lower(&canonical),
        signing_input_bytes_hex: anchor_hex_lower(&signing_input),
        submitter_signature_hex: anchor_hex_lower(&signature),
        wire_payload_json,
    }
}

fn compute_all() -> Vec<Vector> {
    VECTORS.iter().map(compute_vector).collect()
}

#[test]
fn three_evidence_anchor_wire_vectors_match_committed_fixture() {
    let computed = compute_all();
    let path = fixture_path();

    if std::env::var("OMNINODE_REGEN_VECTORS").as_deref() == Ok("1") {
        // Regen mode: overwrite the committed fixture and stop here.
        std::fs::create_dir_all(path.parent().unwrap()).expect("create fixtures dir");
        let bytes = serde_json::to_vec_pretty(&computed).expect("pretty-print JSON");
        std::fs::write(&path, &bytes).expect("write fixture");
        eprintln!(
            "[regen] wrote {} vectors to {}",
            computed.len(),
            path.display()
        );
        return;
    }

    // Verify mode (default).
    let committed_bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture at {}: {e} — run with OMNINODE_REGEN_VECTORS=1 to regenerate \
             (regen is reserved for intentional Stage 13.x wire bumps)",
            path.display()
        )
    });
    let committed: Vec<Vector> = serde_json::from_slice(&committed_bytes)
        .unwrap_or_else(|e| panic!("fixture at {} is not valid JSON: {e}", path.display()));

    assert_eq!(
        committed.len(),
        computed.len(),
        "fixture vector count mismatch (computed {}, committed {})",
        computed.len(),
        committed.len(),
    );
    for (i, (got, want)) in computed.iter().zip(committed.iter()).enumerate() {
        assert_eq!(
            got, want,
            "evidence-anchor wire vector {i} ({}) drifted from committed fixture — \
             this means a Stage 13.0 wire-breaking change. If intentional, regen via \
             OMNINODE_REGEN_VECTORS=1 AND bump INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION + \
             the EVIDENCE_ANCHOR_DOMAIN suffix.",
            got.label
        );
    }
}
