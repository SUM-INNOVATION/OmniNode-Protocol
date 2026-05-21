//! Phase 5 Stage 11a — proof pipeline byte-stable vectors.
//!
//! Default mode: **verify**. Compute the full proof-pipeline output for
//! each input vector via the actual Stage 11a implementation path —
//! `MockProofBackend` → `ProofArtifactBody` → canonical JSON envelope →
//! BLAKE3 root → `InferenceCommitment` → chain-wire digest → Ed25519
//! signature — and assert byte-equality against the committed fixture at
//! `tests/fixtures/proof_pipeline_vectors.json`. This is what runs under
//! `cargo test` and what the CI `stage11a-fixture-check` gate enforces.
//!
//! Regen mode: set `OMNINODE_REGEN_PROOF_PIPELINE_VECTORS=1` to overwrite
//! the committed fixture. Explicit-only — a developer runs it once when
//! intentionally bumping the spec, then commits the regenerated JSON.
//!
//! ## What this fixture pins
//!
//! Per Stage 11a correction 4: the fixture includes every byte the
//! pipeline produces, so a drift at any layer (mock backend output,
//! canonical JSON serialization, BLAKE3 hashing, commitment field
//! ordering, chain-wire bincode, Ed25519 signing) fails this test
//! loudly. The committed JSON is the authoritative byte contract for
//! the mock proof pipeline.
//!
//! ## Why a "manifest_root" derived from model bytes, not a real publish?
//!
//! Stage 11a is about the proof side of the pipeline; Stage 2 already
//! covers real manifest publishing through `omni-store`. To keep the
//! fixture hermetic and reproducible without spawning sum-node, the
//! `manifest_root` is computed as a deterministic BLAKE3 over a small
//! domain tag + model bytes. The chain-wire digest treats it the same
//! as any other 32-byte SNIP V2 root, so the test exercises every
//! Stage 4/5/6 surface unchanged.

use std::path::PathBuf;

use omni_types::phase5::SnipV2ObjectId;
use serde::{Deserialize, Serialize};

use omni_zkml::{
    compute_chain_attestation_vector, derive_chain_address_base58, signer_pubkey_bytes,
    MockProofBackend, ProofArtifactBody, ProofBackend, ProofMetadata,
};

// ── Inputs ──────────────────────────────────────────────────────────────────

/// Compile-time table of pipeline inputs. Three vectors covering:
///
/// - vector 0: short body, lowest-byte seed (`0x01`).
/// - vector 1: empty input bytes (edge case — caller proves "no input").
/// - vector 2: ASCII-only longer body matching the Stage 6
///   `omninode-stage6-vec-3-…` session-id shape so we have at least one
///    vector at a realistic operator-facing length.
const VECTORS: &[Input] = &[
    Input {
        name: "stage11a-vector-0",
        session_id: "omninode-stage11a-vec-0",
        model_bytes: b"stage11a.vec-0.model.bytes",
        input_bytes: b"stage11a.vec-0.input.bytes",
        output_bytes: b"stage11a.vec-0.output.bytes",
        verifier_seed: "0101010101010101010101010101010101010101010101010101010101010101",
    },
    Input {
        name: "stage11a-vector-1-empty-input",
        session_id: "omninode-stage11a-vec-1-empty-input",
        model_bytes: b"stage11a.vec-1.model.bytes",
        input_bytes: b"",
        output_bytes: b"stage11a.vec-1.output.bytes",
        verifier_seed: "0202020202020202020202020202020202020202020202020202020202020202",
    },
    Input {
        name: "stage11a-vector-2-longer-body",
        session_id: "omninode-stage11a-vec-2-abcdef-0123456789",
        model_bytes:
            b"stage11a.vec-2.model.bytes-with-extra-content-for-shape-coverage",
        input_bytes:
            b"stage11a.vec-2.input.bytes-with-extra-content-for-shape-coverage",
        output_bytes:
            b"stage11a.vec-2.output.bytes-with-extra-content-for-shape-coverage",
        verifier_seed: "0303030303030303030303030303030303030303030303030303030303030303",
    },
];

struct Input {
    name: &'static str,
    session_id: &'static str,
    model_bytes: &'static [u8],
    input_bytes: &'static [u8],
    output_bytes: &'static [u8],
    /// 64-char lowercase hex (32 bytes).
    verifier_seed: &'static str,
}

// ── Expected output shape ───────────────────────────────────────────────────

/// One row of the committed fixture. Every field is `expected_*` because
/// the test re-derives them and asserts byte-equality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Stage11aVector {
    name: String,
    session_id: String,
    backend_id: String,

    // Raw inputs (committed verbatim so the fixture is self-contained).
    model_bytes_hex: String,
    input_bytes_hex: String,
    output_bytes_hex: String,
    verifier_seed_hex: String,

    // BLAKE3 of raw inputs.
    expected_model_hash_hex: String,
    expected_input_hash_hex: String,
    expected_response_hash_hex: String,

    // Proof + envelope.
    expected_proof_bytes_hex: String,
    expected_proof_artifact_body_canonical_hex: String,
    expected_proof_artifact_root_hex: String,
    expected_manifest_root_hex: String,

    // Final commitment + chain-wire bytes.
    expected_commitment_session_id: String,
    expected_commitment_model_hash_hex: String,
    expected_commitment_manifest_snip_root_hex: String,
    expected_commitment_response_hash_hex: String,
    expected_commitment_proof_snip_root_hex: String,
    expected_chain_canonical_digest_bytes_hex: String,
    expected_chain_signing_input_bytes_hex: String,
    expected_signature_bytes_hex: String,
    expected_signer_address_base58: String,
    expected_signer_pubkey_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixtureFile {
    comment: String,
    vectors: Vec<Stage11aVector>,
}

// ── Manifest root: deterministic from model bytes ───────────────────────────

/// Synthetic but deterministic manifest_root for a fixture vector. Stage
/// 11a's job is the proof side of the pipeline; Stage 2 covers real
/// manifest publishing. Hashing a domain tag + model bytes gives a
/// reproducible 32-byte root that exercises every Stage 4/5/6 code path
/// without needing a sum-node binary or real `omni-store` plumbing.
fn fixture_manifest_root(model_bytes: &[u8]) -> SnipV2ObjectId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"omninode.stage11a.fixture.manifest");
    hasher.update(model_bytes);
    let mut id = [0u8; 32];
    id.copy_from_slice(hasher.finalize().as_bytes());
    SnipV2ObjectId::from_bytes(id)
}

// ── Hex helpers ─────────────────────────────────────────────────────────────

fn encode_hex_lower(bytes: &[u8]) -> String {
    const NIBBLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(NIBBLE[(b >> 4) as usize] as char);
        s.push(NIBBLE[(b & 0x0f) as usize] as char);
    }
    s
}

fn decode_blake3_seed(s: &str) -> [u8; 32] {
    assert_eq!(s.len(), 64, "seed must be 64 hex chars");
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .expect("seed must be lowercase hex");
    }
    out
}

// ── Vector computation ─────────────────────────────────────────────────────

fn compute_vector(input: &Input) -> Stage11aVector {
    // 1. Hash the raw inputs.
    let model_hash = blake3::hash(input.model_bytes);
    let input_hash = blake3::hash(input.input_bytes);
    let response_hash = blake3::hash(input.output_bytes);

    let model_hash_hex = model_hash.to_hex().to_string();
    let input_hash_hex = input_hash.to_hex().to_string();
    let response_hash_hex = response_hash.to_hex().to_string();

    // 2. Mock backend proof bytes.
    let proof_bytes = MockProofBackend
        .prove(input.model_bytes, input.input_bytes, input.output_bytes)
        .expect("MockProofBackend is infallible");

    // 3. Canonical ProofArtifactBody envelope → BLAKE3 = proof_snip_root.
    let metadata = ProofMetadata {
        backend_id: MockProofBackend.backend_id().to_string(),
        model_hash: model_hash_hex.clone(),
        input_hash: input_hash_hex.clone(),
        response_hash: response_hash_hex.clone(),
    };
    let body = ProofArtifactBody::from_components(metadata, &proof_bytes);
    let body_bytes = body
        .to_canonical_bytes()
        .expect("canonical envelope serialization is infallible for this input");
    let mut proof_root = [0u8; 32];
    proof_root.copy_from_slice(blake3::hash(&body_bytes).as_bytes());
    let proof_snip_root = SnipV2ObjectId::from_bytes(proof_root);

    // 4. Manifest root (deterministic from model bytes).
    let manifest_snip_root = fixture_manifest_root(input.model_bytes);

    // 5. Chain-wire digest + Ed25519 signature.
    //
    //    `compute_chain_attestation_vector` is the Stage 6 helper that
    //    builds the chain digest from hex inputs, signs, and returns the
    //    full attestation vector. We rebuild the canonical-digest /
    //    signing-input / signature bytes through this single entry point
    //    so the Stage 11a fixture exercises the same code path the Stage
    //    6 fixture pins.
    let mr_hex = encode_hex_lower(manifest_snip_root.as_bytes());
    let pr_hex = encode_hex_lower(proof_snip_root.as_bytes());
    let chain_vec = compute_chain_attestation_vector(
        input.session_id,
        &model_hash_hex,
        &mr_hex,
        &response_hash_hex,
        &pr_hex,
        input.verifier_seed,
    )
    .expect("compute_chain_attestation_vector for fixture inputs");

    // 6. Re-derive signer address + pubkey via the same helpers the
    //    operator binary uses, so the fixture pins both sides.
    let seed = decode_blake3_seed(input.verifier_seed);
    let pubkey = signer_pubkey_bytes(&seed).expect("pubkey derive");
    let address = derive_chain_address_base58(&pubkey);

    Stage11aVector {
        name: input.name.to_string(),
        session_id: input.session_id.to_string(),
        backend_id: MockProofBackend.backend_id().to_string(),

        model_bytes_hex: encode_hex_lower(input.model_bytes),
        input_bytes_hex: encode_hex_lower(input.input_bytes),
        output_bytes_hex: encode_hex_lower(input.output_bytes),
        verifier_seed_hex: input.verifier_seed.to_string(),

        expected_model_hash_hex: model_hash_hex.clone(),
        expected_input_hash_hex: input_hash_hex,
        expected_response_hash_hex: response_hash_hex.clone(),

        expected_proof_bytes_hex: encode_hex_lower(&proof_bytes),
        expected_proof_artifact_body_canonical_hex: encode_hex_lower(&body_bytes),
        expected_proof_artifact_root_hex: encode_hex_lower(&proof_root),
        expected_manifest_root_hex: encode_hex_lower(manifest_snip_root.as_bytes()),

        expected_commitment_session_id: input.session_id.to_string(),
        expected_commitment_model_hash_hex: model_hash_hex,
        expected_commitment_manifest_snip_root_hex: mr_hex,
        expected_commitment_response_hash_hex: response_hash_hex,
        expected_commitment_proof_snip_root_hex: pr_hex,
        expected_chain_canonical_digest_bytes_hex: chain_vec.canonical_digest_bytes,
        expected_chain_signing_input_bytes_hex: chain_vec.signing_input_bytes,
        expected_signature_bytes_hex: chain_vec.signature_bytes,
        expected_signer_address_base58: address,
        expected_signer_pubkey_hex: encode_hex_lower(&pubkey),
    }
}

fn compute_all() -> FixtureFile {
    FixtureFile {
        comment: "Stage 11a — proof pipeline byte-stable vectors. Regenerate with OMNINODE_REGEN_PROOF_PIPELINE_VECTORS=1.".into(),
        vectors: VECTORS.iter().map(compute_vector).collect(),
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("proof_pipeline_vectors.json")
}

// ── Test ───────────────────────────────────────────────────────────────────

#[test]
fn stage11a_proof_pipeline_vectors_match_committed_fixture() {
    let computed = compute_all();
    let path = fixture_path();

    if std::env::var("OMNINODE_REGEN_PROOF_PIPELINE_VECTORS").as_deref() == Ok("1") {
        std::fs::create_dir_all(path.parent().unwrap()).expect("create fixtures dir");
        let bytes = serde_json::to_vec_pretty(&computed).expect("pretty-print JSON");
        std::fs::write(&path, &bytes).expect("write fixture");
        eprintln!(
            "[regen] wrote {} vectors to {}",
            computed.vectors.len(),
            path.display()
        );
        return;
    }

    let committed_bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture at {}: {e} — run with \
             OMNINODE_REGEN_PROOF_PIPELINE_VECTORS=1 to regenerate",
            path.display()
        )
    });
    let committed: FixtureFile = serde_json::from_slice(&committed_bytes)
        .unwrap_or_else(|e| panic!("fixture at {} is not valid JSON: {e}", path.display()));

    assert_eq!(
        committed.vectors.len(),
        computed.vectors.len(),
        "fixture vector count mismatch"
    );
    for (i, (got, want)) in computed
        .vectors
        .iter()
        .zip(committed.vectors.iter())
        .enumerate()
    {
        assert_eq!(
            got, want,
            "proof pipeline vector {i} ({}) drifted from committed fixture",
            got.name
        );
    }
}

// ── Sanity tests over the helper paths the fixture depends on ─────────────

#[test]
fn fixture_manifest_root_is_deterministic_and_distinct_per_model() {
    let a = fixture_manifest_root(b"model-a");
    let b = fixture_manifest_root(b"model-b");
    assert_eq!(a, fixture_manifest_root(b"model-a"));
    assert_ne!(a, b);
}

#[test]
fn compute_vector_output_fields_are_self_consistent() {
    // Pick the first vector and assert the cross-field invariants hold
    // (e.g. expected_commitment_model_hash_hex == expected_model_hash_hex).
    // If `compute_vector` ever drops a field copy, this fires before the
    // committed fixture rejects it.
    let v = compute_vector(&VECTORS[0]);
    assert_eq!(v.expected_commitment_model_hash_hex, v.expected_model_hash_hex);
    assert_eq!(
        v.expected_commitment_response_hash_hex,
        v.expected_response_hash_hex
    );
    assert_eq!(
        v.expected_commitment_manifest_snip_root_hex, v.expected_manifest_root_hex
    );
    // proof_snip_root and proof_artifact_root commit the same canonical
    // envelope so they must be byte-identical.
    assert_eq!(
        v.expected_commitment_proof_snip_root_hex, v.expected_proof_artifact_root_hex
    );
}
