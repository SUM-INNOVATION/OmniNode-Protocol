//! Phase 5 Stage 11a — proof generation backend trait + mock impl + orchestrator.
//!
//! This module introduces the first real proof-generation surface in OmniNode.
//! Stages 1-10a treated the proof bytes as opaque: the [`crate::artifact`]
//! module published whatever bytes a caller handed it, [`crate::build_commitment`]
//! emitted the SNIP V2 root of those bytes as `proof_snip_root`, and the
//! operator binary's `smoke --synthetic` flow hardcoded synthetic placeholder
//! bytes (`model_hash="a".repeat(64)`, `proof_snip_root=[0x22;32]`, etc. —
//! see the 2026-05-19 mainnet smoke audit for the on-chain evidence).
//!
//! Stage 11a replaces that placeholder with an honest end-to-end pipeline:
//!
//! 1. A [`ProofBackend`] trait abstracts "produce proof bytes given a
//!    `(model, input, output)` triple". Stage 11a ships [`MockProofBackend`],
//!    a deterministic non-cryptographic stand-in usable for local/dev/CI
//!    only. Stage 11b will plug in real systems (ezkl / risc0 / sp1).
//! 2. A [`ProofVerifier`] trait abstracts "verify proof bytes against
//!    `PublicInputs`". [`MockProofVerifier`] re-derives the deterministic
//!    mock bytes and byte-compares — that's not a SNARK check, but it is a
//!    real "tampering breaks verification" surface for tests.
//! 3. A [`ProofMetadata`] struct carries `(backend_id, model_hash,
//!    input_hash, response_hash)` **inside** the SNIP V2 proof artifact —
//!    committed by the existing `proof_snip_root` on the chain digest. This
//!    binds the input to the proof without changing Stage 6 chain-wire
//!    format. Off-chain verifiers fetch the proof artifact via SNIP V2,
//!    recover `ProofMetadata` + proof bytes, and run [`ProofVerifier::verify`].
//! 4. The [`produce_proof_artifact`] orchestrator wraps backend invocation,
//!    metadata composition, canonical JSON serialization, file write, and
//!    SNIP V2 publish via the existing [`crate::artifact::publish_proof_artifacts`]
//!    plumbing. It returns `(ProofArtifact, ResponseArtifact, ProofMetadata)`
//!    so the caller can hand the artifacts to the unchanged
//!    [`crate::build_commitment`] and continue through the Stage 4
//!    `build_attestation` + Stage 5.3 submit-workflow path with zero
//!    further edits.
//!
//! ## What stays unchanged
//!
//! - Stage 6 chain-wire bytes ([`crate::chain_wire`]). The on-chain digest
//!   still carries `(session_id, model_hash, manifest_root, response_hash,
//!   proof_root)` and only those. `proof_root` is still a SNIP V2 Merkle
//!   root — but now it's the root of a structured JSON envelope containing
//!   metadata + opaque proof bytes, rather than the root of synthetic
//!   placeholder bytes.
//! - Stage 7b transaction construction ([`omni_sumchain::tx`]).
//! - The [`crate::ProofArtifact`] / [`crate::ResponseArtifact`] structs.
//! - The [`crate::build_commitment`] signature. Its callers feed it richer
//!   artifacts now, but the function itself doesn't know proofs got real.
//!
//! ## Mainnet guard (operator-level, enforced in `omni-node`)
//!
//! `MockProofBackend::backend_id() == "mock-v1"` is non-cryptographic by
//! design. The operator binary refuses `chain_id == 1` submissions whose
//! backend is `mock-v1`, **including** with `--allow-mainnet-submit`. This
//! is a hard Stage 11a requirement. See `crates/omni-node/src/operator.rs`
//! for the enforcement site.

use std::path::Path;

use blake3;
use omni_store::SnipV2Adapter;
use serde::{Deserialize, Serialize};

use crate::artifact::{publish_proof_artifacts, ProofArtifact, ResponseArtifact};
use crate::error::{ProofBackendError, ProofPipelineError, ProofVerifierError};

// ── Stable backend ids ──────────────────────────────────────────────────────

/// Stage 11a's mock backend id. The mainnet guard in `omni-node` keys off
/// this constant; do NOT rename without coordinating the operator-side check.
pub const MOCK_BACKEND_ID: &str = "mock-v1";

// ── PublicInputs ────────────────────────────────────────────────────────────

/// Inputs a proof commits to. Typed (not opaque `&[u8]`) per Stage 11a OQ3
/// — picks up future backends adding fields without forcing a rewrite of
/// every verifier impl.
///
/// All three fields are 32-byte BLAKE3 hashes computed from the raw bytes of
/// the corresponding artifact. Sources at the operator side:
///
/// - `model_hash`: BLAKE3 of the model file (matches
///   [`omni_types::model::ModelManifest::model_hash`], which is the hex
///   rendering of the same bytes).
/// - `input_hash`: BLAKE3 of the inference input bytes.
/// - `output_hash`: BLAKE3 of the response bytes (equal to
///   [`omni_types::phase5::InferenceCommitment::response_hash`] decoded).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicInputs {
    pub model_hash: [u8; 32],
    pub input_hash: [u8; 32],
    pub output_hash: [u8; 32],
}

// ── ProofBackend ────────────────────────────────────────────────────────────

/// A backend that can produce proof bytes given `(model, input, output)`.
///
/// Stage 11a contract: implementations must be **deterministic** for the
/// same inputs (same `(model, input, output)` ⇒ same proof bytes). Stage
/// 11b may relax this if the chosen real backend has a salt / randomness
/// parameter, but that's an explicit decision then.
pub trait ProofBackend {
    /// Produce proof bytes binding `(model, input, output)`.
    fn prove(
        &self,
        model: &[u8],
        input: &[u8],
        output: &[u8],
    ) -> std::result::Result<Vec<u8>, ProofBackendError>;

    /// Stable identifier for this backend (e.g. `"mock-v1"`, `"ezkl-halo2-v1"`).
    /// Recorded in [`ProofMetadata::backend_id`] so verifiers can pick the
    /// matching [`ProofVerifier`].
    fn backend_id(&self) -> &'static str;
}

// ── ProofVerifier ───────────────────────────────────────────────────────────

/// Verifier companion of [`ProofBackend`]. Returns `Ok(true)` on a valid
/// proof, `Ok(false)` on a structurally well-formed but failing proof, and
/// a typed error only for backend-internal failures (parse, panic, runtime).
pub trait ProofVerifier {
    fn verify(
        &self,
        proof: &[u8],
        public_inputs: &PublicInputs,
    ) -> std::result::Result<bool, ProofVerifierError>;
}

// ── Mock backend / verifier ─────────────────────────────────────────────────

/// Deterministic, non-cryptographic Stage 11a backend. Computes a 64-byte
/// "proof" as two domain-separated BLAKE3 hashes of `(model_hash ||
/// input_hash || output_hash)`. The hash chain makes the output vary with
/// inputs (so a tampered fixture fails the byte-roundtrip test) and lets
/// [`MockProofVerifier`] re-derive and byte-compare without needing the
/// original raw bytes — `PublicInputs` alone suffice.
///
/// **Not cryptographic.** This backend produces no soundness guarantee and
/// must not be used for mainnet submissions. The operator binary refuses
/// `mock-v1` on `chain_id == 1` (Stage 11a guardrail).
pub struct MockProofBackend;

impl ProofBackend for MockProofBackend {
    fn prove(
        &self,
        model: &[u8],
        input: &[u8],
        output: &[u8],
    ) -> std::result::Result<Vec<u8>, ProofBackendError> {
        let mh = blake3::hash(model);
        let ih = blake3::hash(input);
        let oh = blake3::hash(output);
        Ok(mock_proof_bytes(mh.as_bytes(), ih.as_bytes(), oh.as_bytes()))
    }

    fn backend_id(&self) -> &'static str {
        MOCK_BACKEND_ID
    }
}

/// Companion verifier for [`MockProofBackend`]. Re-derives the deterministic
/// mock bytes from `PublicInputs` and byte-compares against the supplied
/// proof.
pub struct MockProofVerifier;

impl ProofVerifier for MockProofVerifier {
    fn verify(
        &self,
        proof: &[u8],
        public_inputs: &PublicInputs,
    ) -> std::result::Result<bool, ProofVerifierError> {
        let expected = mock_proof_bytes(
            &public_inputs.model_hash,
            &public_inputs.input_hash,
            &public_inputs.output_hash,
        );
        Ok(proof == expected.as_slice())
    }
}

/// Domain-separated 64-byte deterministic "proof". Two BLAKE3 invocations
/// with disjoint domain tags so the two halves can't be swapped without
/// detection. The tags include the `mock-v1` backend id so a future
/// `mock-v2` would never collide.
fn mock_proof_bytes(
    model_hash: &[u8; 32],
    input_hash: &[u8; 32],
    output_hash: &[u8; 32],
) -> Vec<u8> {
    fn part(tag: &'static [u8], mh: &[u8; 32], ih: &[u8; 32], oh: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(tag);
        hasher.update(mh);
        hasher.update(ih);
        hasher.update(oh);
        *hasher.finalize().as_bytes()
    }
    let p0 = part(
        b"omninode.mock-v1.proof.part0",
        model_hash,
        input_hash,
        output_hash,
    );
    let p1 = part(
        b"omninode.mock-v1.proof.part1",
        model_hash,
        input_hash,
        output_hash,
    );
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(&p0);
    out.extend_from_slice(&p1);
    out
}

// ── ProofMetadata + ProofArtifactBody ───────────────────────────────────────

/// Metadata embedded inside the SNIP V2 proof artifact. Committed by
/// `proof_snip_root` on the chain digest (which is the SNIP V2 Merkle root
/// of the canonical JSON envelope produced below). Off-chain verifiers
/// recover this struct to drive [`ProofVerifier::verify`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Stable identifier of the backend that produced the proof.
    pub backend_id: String,
    /// BLAKE3 hex of the model bytes (bare lowercase, 64 chars). Equal to
    /// [`omni_types::model::ModelManifest::model_hash`].
    pub model_hash: String,
    /// BLAKE3 hex of the inference input bytes (bare lowercase, 64 chars).
    /// **This is the binding for the "input" in
    /// `(model, input, output)`** — committed here, not on the chain
    /// digest, per Stage 11a OQ4.
    pub input_hash: String,
    /// BLAKE3 hex of the response/output bytes (bare lowercase, 64 chars).
    /// Equal to [`omni_types::phase5::InferenceCommitment::response_hash`].
    pub response_hash: String,
}

impl ProofMetadata {
    /// Decode the three hex hashes into the typed [`PublicInputs`] struct
    /// the verifier consumes. Returns a typed `ProofVerifierError` on any
    /// hex parse failure so verifier callers don't have to convert errors.
    pub fn public_inputs(&self) -> std::result::Result<PublicInputs, ProofVerifierError> {
        Ok(PublicInputs {
            model_hash: decode_blake3_hex_lower("model_hash", &self.model_hash)?,
            input_hash: decode_blake3_hex_lower("input_hash", &self.input_hash)?,
            output_hash: decode_blake3_hex_lower("response_hash", &self.response_hash)?,
        })
    }
}

/// The canonical-JSON envelope written to the proof file before SNIP V2
/// ingest. The file's BLAKE3 (which is what the SNIP V2 fake adapter uses
/// for its `merkle_root` in tests, and what the real `sum-node ingest-v2`
/// emits in production via the same algorithm at the byte level) becomes
/// the `proof_snip_root` on the chain digest.
///
/// `serde_json::to_vec(&self)` produces deterministic output: field order
/// is taken from the struct definition, no trailing newline, no random
/// keying. Two identical `ProofArtifactBody` values therefore produce
/// byte-identical files and identical SNIP V2 roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofArtifactBody {
    pub metadata: ProofMetadata,
    /// Hex of the proof bytes returned by the backend (bare lowercase, no
    /// `0x` prefix; variable length).
    pub proof_bytes_hex: String,
}

impl ProofArtifactBody {
    /// Compose an envelope from already-computed components.
    pub fn from_components(metadata: ProofMetadata, proof_bytes: &[u8]) -> Self {
        Self {
            metadata,
            proof_bytes_hex: encode_hex_lower(proof_bytes),
        }
    }

    /// Decode the embedded proof bytes back into the byte buffer the
    /// backend originally returned.
    pub fn proof_bytes(&self) -> std::result::Result<Vec<u8>, ProofVerifierError> {
        decode_hex_lower("proof_bytes_hex", &self.proof_bytes_hex)
    }

    /// Canonical JSON serialization. Deterministic for the same input.
    pub fn to_canonical_bytes(&self) -> std::result::Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

// ── Orchestrator ────────────────────────────────────────────────────────────

/// Inputs to [`produce_proof_artifact`].
#[derive(Debug, Clone, Copy)]
pub struct ProofPipelineInputs<'a> {
    pub model_bytes: &'a [u8],
    pub input_bytes: &'a [u8],
    pub output_bytes: &'a [u8],
}

/// Outputs of [`produce_proof_artifact`]: the published artifacts the
/// caller hands to [`crate::build_commitment`] unchanged, plus the
/// metadata for any further off-chain verification work.
#[derive(Debug, Clone)]
pub struct ProofPipelineOutputs {
    pub response_artifact: ResponseArtifact,
    pub proof_artifact: ProofArtifact,
    pub metadata: ProofMetadata,
}

/// End-to-end Stage 11a orchestrator. Steps:
///
/// 1. BLAKE3 the model / input / output bytes (gives us the `ProofMetadata`
///    fields without re-hashing the response later).
/// 2. Invoke `backend.prove(model, input, output)`.
/// 3. Assemble a [`ProofArtifactBody`] (metadata + hex proof bytes).
/// 4. Canonical JSON-serialize the envelope and write it to
///    `artifact_dir/proof.json`. Also write the response bytes to
///    `artifact_dir/response.bin` so [`publish_proof_artifacts`] can
///    independently ingest+hash the response per its Stage 3 contract.
/// 5. Construct fresh [`ResponseArtifact`] + [`ProofArtifact`] pointing at
///    those files and call [`publish_proof_artifacts`] — which populates
///    both SNIP V2 refs and the response BLAKE3 hash via the existing
///    Stage 3 plumbing, unchanged.
/// 6. Return the populated artifacts and the metadata.
///
/// The caller then composes with the unchanged
/// [`crate::build_commitment`] / [`crate::build_attestation`] /
/// [`crate::submit_attestation_workflow_with_block`] pipeline.
///
/// **`build_commitment` is NOT modified** — proof concerns live here
/// (Stage 11a correction 1).
pub fn produce_proof_artifact<A, B>(
    adapter: &A,
    backend: &B,
    inputs: ProofPipelineInputs<'_>,
    artifact_dir: &Path,
) -> std::result::Result<ProofPipelineOutputs, ProofPipelineError>
where
    A: SnipV2Adapter,
    B: ProofBackend + ?Sized,
{
    // 1. Compute the three BLAKE3 hashes once.
    let model_hash = blake3::hash(inputs.model_bytes);
    let input_hash = blake3::hash(inputs.input_bytes);
    let response_hash = blake3::hash(inputs.output_bytes);

    // 2. Backend produces proof bytes.
    let proof_bytes =
        backend.prove(inputs.model_bytes, inputs.input_bytes, inputs.output_bytes)?;

    // 3. Assemble metadata + body.
    let metadata = ProofMetadata {
        backend_id: backend.backend_id().to_string(),
        model_hash: model_hash.to_hex().to_string(),
        input_hash: input_hash.to_hex().to_string(),
        response_hash: response_hash.to_hex().to_string(),
    };
    let body = ProofArtifactBody::from_components(metadata.clone(), &proof_bytes);
    let body_bytes = body.to_canonical_bytes()?;

    // 4. Write the canonical envelope + response bytes to disk.
    std::fs::create_dir_all(artifact_dir)?;
    let proof_path = artifact_dir.join("proof.json");
    let response_path = artifact_dir.join("response.bin");
    std::fs::write(&proof_path, &body_bytes)?;
    std::fs::write(&response_path, inputs.output_bytes)?;

    // 5. Hand the new artifacts to the existing Stage 3 publish plumbing.
    //    No file-format changes; publish_proof_artifacts treats both files
    //    as opaque bytes, which is exactly the Stage 11a contract.
    let mut response_artifact = ResponseArtifact::new(&response_path);
    let mut proof_artifact = ProofArtifact::new(&proof_path);
    publish_proof_artifacts(adapter, &mut response_artifact, &mut proof_artifact)?;

    Ok(ProofPipelineOutputs {
        response_artifact,
        proof_artifact,
        metadata,
    })
}

// ── hex helpers (local, lowercase-strict) ───────────────────────────────────
//
// `omni-zkml` doesn't depend on the `hex` crate; existing code uses
// blake3's `to_hex()` for 32-byte values and hand-rolled lowercase nibble
// decoders for parse paths (see `attestation::CommitmentDigest::from_hex`).
// Mirroring that style here keeps the crate dep-clean.

fn encode_hex_lower(bytes: &[u8]) -> String {
    const NIBBLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(NIBBLE[(b >> 4) as usize] as char);
        s.push(NIBBLE[(b & 0x0f) as usize] as char);
    }
    s
}

fn decode_hex_lower(
    field: &'static str,
    s: &str,
) -> std::result::Result<Vec<u8>, ProofVerifierError> {
    if s.len() % 2 != 0 {
        return Err(ProofVerifierError::VerifierInternal(format!(
            "field {field}: odd-length hex ({} chars)",
            s.len()
        )));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in 0..(s.len() / 2) {
        let hi = decode_nibble_lower(field, bytes[i * 2])?;
        let lo = decode_nibble_lower(field, bytes[i * 2 + 1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn decode_blake3_hex_lower(
    field: &'static str,
    s: &str,
) -> std::result::Result<[u8; 32], ProofVerifierError> {
    if s.len() != 64 {
        return Err(ProofVerifierError::VerifierInternal(format!(
            "field {field}: expected 64 hex chars (32 bytes), got {}",
            s.len()
        )));
    }
    let mut out = [0u8; 32];
    let bytes = s.as_bytes();
    for i in 0..32 {
        let hi = decode_nibble_lower(field, bytes[i * 2])?;
        let lo = decode_nibble_lower(field, bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn decode_nibble_lower(field: &'static str, b: u8) -> std::result::Result<u8, ProofVerifierError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Err(ProofVerifierError::VerifierInternal(format!(
            "field {field}: uppercase hex '{}' not allowed (bare lowercase contract)",
            b as char
        ))),
        _ => Err(ProofVerifierError::VerifierInternal(format!(
            "field {field}: invalid hex char '{}'",
            b as char
        ))),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Mutex;

    use omni_store::snip_v2::SnipV2Error;
    use omni_types::phase5::{SnipV2Lifecycle, SnipV2ObjectId, SnipV2ObjectRef};

    /// Minimal in-memory `SnipV2Adapter` for proof orchestrator tests.
    /// Content-addresses by BLAKE3 of the file bytes (mirrors what
    /// `sum-node ingest-v2` produces in production at the byte level).
    /// `download_public` is implemented in case future tests need it.
    struct LocalFakeSnipV2Adapter {
        store: Mutex<HashMap<SnipV2ObjectId, Vec<u8>>>,
    }
    impl LocalFakeSnipV2Adapter {
        fn new() -> Self {
            Self {
                store: Mutex::new(HashMap::new()),
            }
        }
    }
    impl omni_store::SnipV2Adapter for LocalFakeSnipV2Adapter {
        fn ingest_public(
            &self,
            path: &Path,
        ) -> std::result::Result<SnipV2ObjectRef, SnipV2Error> {
            let bytes = std::fs::read(path).map_err(|e| SnipV2Error::CommandSpawn(e))?;
            let hash = blake3::hash(&bytes);
            let mut id_bytes = [0u8; 32];
            id_bytes.copy_from_slice(hash.as_bytes());
            let id = SnipV2ObjectId::from_bytes(id_bytes);
            self.store.lock().unwrap().insert(id, bytes.clone());
            Ok(SnipV2ObjectRef {
                merkle_root: id,
                lifecycle: SnipV2Lifecycle::Active,
                plaintext_size_bytes: Some(bytes.len() as u64),
            })
        }

        fn download_public(
            &self,
            root: &SnipV2ObjectId,
            output_path: &Path,
        ) -> std::result::Result<(), SnipV2Error> {
            let bytes = self
                .store
                .lock()
                .unwrap()
                .get(root)
                .cloned()
                .ok_or_else(|| {
                    SnipV2Error::CommandSpawn(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "no such object in fake store",
                    ))
                })?;
            std::fs::write(output_path, bytes).map_err(SnipV2Error::CommandSpawn)
        }
    }

    #[test]
    fn mock_backend_id_is_stable() {
        assert_eq!(MockProofBackend.backend_id(), "mock-v1");
        assert_eq!(MOCK_BACKEND_ID, "mock-v1");
    }

    #[test]
    fn mock_backend_prove_is_deterministic_and_64_bytes() {
        let p1 = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        let p2 = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 64);
    }

    #[test]
    fn mock_backend_prove_varies_with_inputs() {
        let p_baseline = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        let p_model = MockProofBackend
            .prove(b"MODEL", b"input", b"output")
            .unwrap();
        let p_input = MockProofBackend
            .prove(b"model", b"INPUT", b"output")
            .unwrap();
        let p_output = MockProofBackend
            .prove(b"model", b"input", b"OUTPUT")
            .unwrap();
        assert_ne!(p_baseline, p_model);
        assert_ne!(p_baseline, p_input);
        assert_ne!(p_baseline, p_output);
        // And each varied input is distinct from the others (domain tags
        // bind the position of each hash).
        assert_ne!(p_model, p_input);
        assert_ne!(p_input, p_output);
    }

    #[test]
    fn mock_verifier_accepts_matching_proof() {
        let proof = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        let pi = PublicInputs {
            model_hash: *blake3::hash(b"model").as_bytes(),
            input_hash: *blake3::hash(b"input").as_bytes(),
            output_hash: *blake3::hash(b"output").as_bytes(),
        };
        assert_eq!(MockProofVerifier.verify(&proof, &pi).unwrap(), true);
    }

    #[test]
    fn mock_verifier_rejects_tampered_proof() {
        let mut proof = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        proof[0] ^= 0x01;
        let pi = PublicInputs {
            model_hash: *blake3::hash(b"model").as_bytes(),
            input_hash: *blake3::hash(b"input").as_bytes(),
            output_hash: *blake3::hash(b"output").as_bytes(),
        };
        assert_eq!(MockProofVerifier.verify(&proof, &pi).unwrap(), false);
    }

    #[test]
    fn mock_verifier_rejects_mismatched_public_inputs() {
        let proof = MockProofBackend
            .prove(b"model", b"input", b"output")
            .unwrap();
        let pi = PublicInputs {
            model_hash: *blake3::hash(b"DIFFERENT_MODEL").as_bytes(),
            input_hash: *blake3::hash(b"input").as_bytes(),
            output_hash: *blake3::hash(b"output").as_bytes(),
        };
        assert_eq!(MockProofVerifier.verify(&proof, &pi).unwrap(), false);
    }

    #[test]
    fn proof_artifact_body_roundtrips_canonical_bytes() {
        let body = ProofArtifactBody::from_components(
            ProofMetadata {
                backend_id: "mock-v1".to_string(),
                model_hash: blake3::hash(b"m").to_hex().to_string(),
                input_hash: blake3::hash(b"i").to_hex().to_string(),
                response_hash: blake3::hash(b"o").to_hex().to_string(),
            },
            &[0x01, 0x02, 0xab, 0xcd],
        );
        let bytes = body.to_canonical_bytes().unwrap();
        let body_back: ProofArtifactBody = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body_back, body);
        // Round-trip is byte-identical (deterministic serialization).
        assert_eq!(body_back.to_canonical_bytes().unwrap(), bytes);
        // Embedded proof bytes decode cleanly.
        assert_eq!(body.proof_bytes().unwrap(), vec![0x01, 0x02, 0xab, 0xcd]);
    }

    #[test]
    fn proof_metadata_public_inputs_decodes_hex() {
        let metadata = ProofMetadata {
            backend_id: "mock-v1".to_string(),
            model_hash: blake3::hash(b"m").to_hex().to_string(),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
        };
        let pi = metadata.public_inputs().unwrap();
        assert_eq!(pi.model_hash, *blake3::hash(b"m").as_bytes());
        assert_eq!(pi.input_hash, *blake3::hash(b"i").as_bytes());
        assert_eq!(pi.output_hash, *blake3::hash(b"o").as_bytes());
    }

    #[test]
    fn proof_metadata_public_inputs_rejects_uppercase_hex() {
        let metadata = ProofMetadata {
            backend_id: "mock-v1".to_string(),
            // Uppercase — must fail the bare-lowercase contract.
            model_hash: blake3::hash(b"m").to_hex().to_string().to_uppercase(),
            input_hash: blake3::hash(b"i").to_hex().to_string(),
            response_hash: blake3::hash(b"o").to_hex().to_string(),
        };
        let err = metadata.public_inputs().unwrap_err();
        assert!(
            matches!(err, ProofVerifierError::VerifierInternal(ref s) if s.contains("uppercase")),
            "expected uppercase rejection, got {err:?}"
        );
    }

    #[test]
    fn end_to_end_orchestrator_publishes_via_fake_adapter() {
        // Hermetic end-to-end smoke against a local in-memory SNIP V2
        // adapter (omni-store's fake is private). The orchestrator
        // exercises the real publish_proof_artifacts path without a
        // sum-node binary.
        let adapter = LocalFakeSnipV2Adapter::new();
        let tmp = tempfile::tempdir().unwrap();

        let inputs = ProofPipelineInputs {
            model_bytes: b"stage11a-orchestrator-test-model",
            input_bytes: b"stage11a-orchestrator-test-input",
            output_bytes: b"stage11a-orchestrator-test-output",
        };
        let out =
            produce_proof_artifact(&adapter, &MockProofBackend, inputs, tmp.path()).unwrap();

        // ProofArtifact and ResponseArtifact both have populated SNIP refs.
        assert!(out.proof_artifact.snip_v2.is_some());
        assert!(out.response_artifact.snip_v2.is_some());
        assert!(out.response_artifact.blake3_hash.is_some());

        // ProofMetadata fields match the deterministic hashes of the inputs.
        assert_eq!(out.metadata.backend_id, "mock-v1");
        assert_eq!(
            out.metadata.model_hash,
            blake3::hash(inputs.model_bytes).to_hex().to_string()
        );
        assert_eq!(
            out.metadata.input_hash,
            blake3::hash(inputs.input_bytes).to_hex().to_string()
        );
        assert_eq!(
            out.metadata.response_hash,
            blake3::hash(inputs.output_bytes).to_hex().to_string()
        );

        // Response artifact's BLAKE3 matches the metadata response_hash.
        assert_eq!(
            out.response_artifact.blake3_hash.as_deref().unwrap(),
            out.metadata.response_hash
        );

        // The proof file on disk parses back into a ProofArtifactBody with
        // the same metadata and proof bytes that the backend produced.
        let proof_path_bytes = std::fs::read(&out.proof_artifact.local_path).unwrap();
        let parsed: ProofArtifactBody = serde_json::from_slice(&proof_path_bytes).unwrap();
        assert_eq!(parsed.metadata, out.metadata);

        // Verifier accepts the proof bytes recovered from the artifact.
        let recovered_proof = parsed.proof_bytes().unwrap();
        let pi = parsed.metadata.public_inputs().unwrap();
        assert_eq!(MockProofVerifier.verify(&recovered_proof, &pi).unwrap(), true);
    }
}
