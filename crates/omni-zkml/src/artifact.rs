//! Phase 5 Stage 3 — proof / response artifact publishing and commitment
//! construction.
//!
//! Stage 3 is byte-shovel infrastructure: it accepts opaque proof and
//! response files from a caller, publishes them to SNIP V2 Public through
//! the existing [`SnipV2Adapter`] from `omni-store`, computes the response
//! BLAKE3 hash, and assembles an [`InferenceCommitment`] ready for a future
//! stage to sign and submit. There is intentionally **no proof generation,
//! no verifier, no signing, no chain client** in this module.
//!
//! Idempotency: the publish flow inspects the optional fields on the
//! artifacts and skips every side-effect that would re-establish an already
//! supplied piece of state. A second call with the same arguments after a
//! successful first call performs zero adapter calls and zero filesystem
//! reads. See the per-case table in the module rustdoc on
//! [`publish_proof_artifacts`].

use std::path::PathBuf;

use omni_store::SnipV2Adapter;
use omni_types::model::ModelManifest;
use omni_types::phase5::{InferenceCommitment, SnipV2ObjectRef};

use crate::error::{ProofArtifactError, Result};

// ── Data types ────────────────────────────────────────────────────────────────

/// Local opaque proof byte file plus an optional SNIP V2 reference.
///
/// `snip_v2 == None` means "not yet published to SNIP". When already
/// populated the publish flow skips the proof side entirely and does not
/// consult `local_path`.
#[derive(Debug, Clone)]
pub struct ProofArtifact {
    pub local_path: PathBuf,
    pub snip_v2: Option<SnipV2ObjectRef>,
}

impl ProofArtifact {
    pub fn new(local_path: impl Into<PathBuf>) -> Self {
        Self { local_path: local_path.into(), snip_v2: None }
    }
}

/// Local response byte file plus an optional SNIP V2 reference and an
/// optional BLAKE3 hex hash.
///
/// `blake3_hash` is the value that ends up in `InferenceCommitment::response_hash`.
/// Format: **bare lowercase 64-char hex** (no `0x` prefix), matching the
/// existing convention used by `ModelManifest::model_hash` and
/// `ShardDescriptor::blake3_hash`.
#[derive(Debug, Clone)]
pub struct ResponseArtifact {
    pub local_path: PathBuf,
    pub snip_v2: Option<SnipV2ObjectRef>,
    pub blake3_hash: Option<String>,
}

impl ResponseArtifact {
    pub fn new(local_path: impl Into<PathBuf>) -> Self {
        Self { local_path: local_path.into(), snip_v2: None, blake3_hash: None }
    }
}

/// Returned by [`publish_proof_artifacts`] on full success. Each boolean
/// reports whether the corresponding side-effect actually fired.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProofPublishReport {
    pub response_published: bool,
    pub response_hash_computed: bool,
    pub proof_published: bool,
}

// ── Publish ───────────────────────────────────────────────────────────────────

/// Publish a response artifact and a proof artifact to SNIP V2 Public, and
/// compute the response BLAKE3 hash.
///
/// File-existence preconditions (only enforced when the corresponding piece
/// of state is missing):
///
/// | `response.snip_v2` | `response.blake3_hash` | Response file required? |
/// |---|---|---|
/// | `Some` | `Some` | No — response side fully skipped |
/// | `Some` | `None` | Yes — needed for hash computation |
/// | `None` | `Some` | Yes — needed for ingest |
/// | `None` | `None` | Yes — needed for both |
///
/// | `proof.snip_v2` | Proof file required? |
/// |---|---|
/// | `Some` | No — proof side fully skipped |
/// | `None` | Yes — needed for ingest |
///
/// A missing file always surfaces as the typed
/// [`ProofArtifactError::ResponseFileNotFound`] /
/// [`ProofArtifactError::ProofFileNotFound`], never as a generic `Io`.
/// Genuine I/O failures on files that existed at preflight time (permission
/// revoked, FS unmount) surface as [`ProofArtifactError::Io`].
///
/// Bails on first failure; the `&mut` artifacts are left partially updated
/// so that a re-call after fixing the root cause resumes via the same
/// idempotency checks.
pub fn publish_proof_artifacts<A: SnipV2Adapter>(
    adapter: &A,
    response: &mut ResponseArtifact,
    proof: &mut ProofArtifact,
) -> Result<ProofPublishReport> {
    let mut report = ProofPublishReport::default();

    // ── Response — Step A: SNIP ingest ──────────────────────────────────
    if response.snip_v2.is_none() {
        if !response.local_path.is_file() {
            return Err(ProofArtifactError::ResponseFileNotFound {
                path: response.local_path.clone(),
            });
        }
        let object_ref = adapter.ingest_public(&response.local_path)?;
        let file_len = std::fs::metadata(&response.local_path)?.len();
        let merkle_root = object_ref.merkle_root;
        response.snip_v2 = Some(SnipV2ObjectRef {
            merkle_root,
            lifecycle: object_ref.lifecycle,
            plaintext_size_bytes: Some(file_len),
        });
        tracing::info!(
            merkle = %merkle_root,
            "published response artifact to SNIP V2"
        );
        report.response_published = true;
    }

    // ── Response — Step B: BLAKE3 hash ──────────────────────────────────
    // Independent preflight: Step A may have been skipped if `snip_v2` was
    // pre-supplied, so we cannot rely on Step A's preflight having run.
    // Missing file here must surface as `ResponseFileNotFound`, not as a
    // generic `Io` leaked from `std::fs::read`.
    if response.blake3_hash.is_none() {
        if !response.local_path.is_file() {
            return Err(ProofArtifactError::ResponseFileNotFound {
                path: response.local_path.clone(),
            });
        }
        let bytes = std::fs::read(&response.local_path)?;
        response.blake3_hash = Some(blake3::hash(&bytes).to_hex().to_string());
        report.response_hash_computed = true;
    }

    // ── Proof side ──────────────────────────────────────────────────────
    if proof.snip_v2.is_none() {
        if !proof.local_path.is_file() {
            return Err(ProofArtifactError::ProofFileNotFound {
                path: proof.local_path.clone(),
            });
        }
        let object_ref = adapter.ingest_public(&proof.local_path)?;
        let file_len = std::fs::metadata(&proof.local_path)?.len();
        let merkle_root = object_ref.merkle_root;
        proof.snip_v2 = Some(SnipV2ObjectRef {
            merkle_root,
            lifecycle: object_ref.lifecycle,
            plaintext_size_bytes: Some(file_len),
        });
        tracing::info!(
            merkle = %merkle_root,
            "published proof artifact to SNIP V2"
        );
        report.proof_published = true;
    }

    Ok(report)
}

// ── Build commitment ──────────────────────────────────────────────────────────

/// Assemble an [`InferenceCommitment`] from a session id, a model manifest,
/// a response artifact, and a proof artifact.
///
/// Strict pre-conditions, each mapping to a typed error:
///
/// - `!session_id.is_empty()` → otherwise [`ProofArtifactError::EmptySessionId`].
/// - `manifest.snip_v2.is_some()` → otherwise
///   [`ProofArtifactError::ManifestLacksSnipRoot`].
/// - `response.blake3_hash.is_some()` → otherwise
///   [`ProofArtifactError::ResponseLacksHash`].
/// - `proof.snip_v2.is_some()` → otherwise
///   [`ProofArtifactError::ProofLacksSnipRoot`].
///
/// Note: `response.snip_v2` is **not** required — [`InferenceCommitment`]
/// does not carry a response SNIP root, only `response_hash`. Callers who
/// want the response stored in SNIP run [`publish_proof_artifacts`] first;
/// that populates the hash as a side effect.
///
/// Restored manifests: [`omni_store::OmniStore::restore_manifest_from_snip`]
/// returns a manifest whose top-level `snip_v2` is `None` by Stage-2 design
/// (the canonical SNIP bytes never embed a self-pointer). To pass this
/// function's strict check, callers must annotate the manifest locally with
/// the root they restored from before calling. Example:
///
/// ```ignore
/// let manifest = store.restore_manifest_from_snip(&adapter, &root, &dest)?;
/// manifest.snip_v2 = Some(SnipV2ObjectRef {
///     merkle_root: root,
///     lifecycle: SnipV2Lifecycle::Active,
///     plaintext_size_bytes: Some(std::fs::metadata(&dest)?.len()),
/// });
/// let commitment = build_commitment(session_id, &manifest, &response, &proof)?;
/// ```
pub fn build_commitment(
    session_id: String,
    manifest: &ModelManifest,
    response: &ResponseArtifact,
    proof: &ProofArtifact,
) -> Result<InferenceCommitment> {
    if session_id.is_empty() {
        return Err(ProofArtifactError::EmptySessionId);
    }
    let manifest_snip = manifest
        .snip_v2
        .as_ref()
        .ok_or(ProofArtifactError::ManifestLacksSnipRoot)?;
    let response_hash = response
        .blake3_hash
        .as_ref()
        .ok_or(ProofArtifactError::ResponseLacksHash)?;
    let proof_snip = proof
        .snip_v2
        .as_ref()
        .ok_or(ProofArtifactError::ProofLacksSnipRoot)?;

    Ok(InferenceCommitment {
        session_id,
        model_hash: manifest.model_hash.clone(),
        manifest_snip_root: manifest_snip.merkle_root,
        response_hash: response_hash.clone(),
        proof_snip_root: proof_snip.merkle_root,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::fs;
    use std::path::Path;

    use omni_store::SnipV2Error;
    use omni_types::model::ModelManifest;
    use omni_types::phase5::{SnipV2Lifecycle, SnipV2ObjectId};

    // ── Local fake adapter ───────────────────────────────────────────────

    /// Stage-3 fake: deterministic content-addressed ingest, no download
    /// support (Stage 3 never downloads). Mirrors the Stage-2 pattern but
    /// is intentionally smaller. Stage-3 tests trigger missing-file failure
    /// paths via filesystem absence rather than adapter-side injection, so
    /// no forced-failure knob is exposed.
    struct FakeSnipV2Adapter {
        state: RefCell<FakeState>,
    }

    #[derive(Default)]
    struct FakeState {
        ingest_calls: Vec<PathBuf>,
    }

    impl FakeSnipV2Adapter {
        fn new() -> Self {
            Self { state: RefCell::new(FakeState::default()) }
        }
        fn ingest_calls(&self) -> Vec<PathBuf> {
            self.state.borrow().ingest_calls.clone()
        }
    }

    impl SnipV2Adapter for FakeSnipV2Adapter {
        fn ingest_public(&self, path: &Path) -> std::result::Result<SnipV2ObjectRef, SnipV2Error> {
            self.state.borrow_mut().ingest_calls.push(path.to_path_buf());
            let bytes = fs::read(path).map_err(SnipV2Error::CommandSpawn)?;
            let hash = blake3::hash(&bytes);
            let mut id_bytes = [0u8; 32];
            id_bytes.copy_from_slice(hash.as_bytes());
            Ok(SnipV2ObjectRef {
                merkle_root: SnipV2ObjectId::from_bytes(id_bytes),
                lifecycle: SnipV2Lifecycle::Active,
                plaintext_size_bytes: None,
            })
        }

        fn download_public(
            &self,
            _root: &SnipV2ObjectId,
            _output_path: &Path,
        ) -> std::result::Result<(), SnipV2Error> {
            // Stage-3 tests never download; surface an unambiguous error if
            // anyone reaches this path.
            Err(SnipV2Error::DownloadFailed {
                code: 1,
                stderr: "fake adapter: download not used by Stage 3 tests".into(),
            })
        }
    }

    // ── Test fixture helpers ─────────────────────────────────────────────

    fn write_temp_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }

    fn dummy_object_ref(byte: u8) -> SnipV2ObjectRef {
        let mut b = [0u8; 32];
        b.fill(byte);
        SnipV2ObjectRef {
            merkle_root: SnipV2ObjectId::from_bytes(b),
            lifecycle: SnipV2Lifecycle::Active,
            plaintext_size_bytes: Some(0),
        }
    }

    fn manifest_with_snip_root(root: SnipV2ObjectId) -> ModelManifest {
        ModelManifest {
            model_name: "test".into(),
            model_hash: "a".repeat(64),
            architecture: "llama".into(),
            total_layers: 4,
            quantization: "F16".into(),
            total_size_bytes: 0,
            gguf_version: 3,
            shards: vec![],
            snip_v2: Some(SnipV2ObjectRef {
                merkle_root: root,
                lifecycle: SnipV2Lifecycle::Active,
                plaintext_size_bytes: Some(1024),
            }),
        }
    }

    fn manifest_without_snip_root() -> ModelManifest {
        ModelManifest {
            model_name: "test".into(),
            model_hash: "a".repeat(64),
            architecture: "llama".into(),
            total_layers: 4,
            quantization: "F16".into(),
            total_size_bytes: 0,
            gguf_version: 3,
            shards: vec![],
            snip_v2: None,
        }
    }

    // ── 1. publish response populates ref AND hash ──────────────────────

    #[test]
    fn publish_response_populates_ref_and_hash() {
        let dir = tempfile::tempdir().unwrap();
        let resp_path = write_temp_file(dir.path(), "response.bin", b"hello response bytes");
        let proof_path = write_temp_file(dir.path(), "proof.bin", b"opaque proof blob");

        let mut response = ResponseArtifact::new(&resp_path);
        let mut proof = ProofArtifact::new(&proof_path);
        let fake = FakeSnipV2Adapter::new();

        let report = publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap();

        assert!(response.snip_v2.is_some());
        assert!(response.blake3_hash.is_some());
        assert!(report.response_published);
        assert!(report.response_hash_computed);
        assert!(report.proof_published);

        // Ingest was called for the response path.
        assert!(fake.ingest_calls().contains(&resp_path));

        // Plaintext size on the SNIP ref matches the file size.
        let resp_size = fs::metadata(&resp_path).unwrap().len();
        assert_eq!(
            response.snip_v2.as_ref().unwrap().plaintext_size_bytes,
            Some(resp_size)
        );
    }

    // ── 2. publish proof populates ref ──────────────────────────────────

    #[test]
    fn publish_proof_populates_ref() {
        let dir = tempfile::tempdir().unwrap();
        let resp_path = write_temp_file(dir.path(), "r.bin", b"resp");
        let proof_path = write_temp_file(dir.path(), "p.bin", b"proof");

        let mut response = ResponseArtifact::new(&resp_path);
        let mut proof = ProofArtifact::new(&proof_path);
        let fake = FakeSnipV2Adapter::new();

        publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap();

        assert!(proof.snip_v2.is_some());
        assert!(fake.ingest_calls().contains(&proof_path));

        let proof_size = fs::metadata(&proof_path).unwrap().len();
        assert_eq!(
            proof.snip_v2.as_ref().unwrap().plaintext_size_bytes,
            Some(proof_size)
        );
    }

    // ── 3. publish skips response when ref AND hash are pre-provided ────

    #[test]
    fn publish_skips_when_response_already_has_ref_and_hash() {
        let dir = tempfile::tempdir().unwrap();
        // Response file deliberately does NOT exist — proves the response
        // side is skipped entirely.
        let resp_path = dir.path().join("nonexistent.bin");
        let proof_path = write_temp_file(dir.path(), "p.bin", b"proof");

        let mut response = ResponseArtifact::new(&resp_path);
        response.snip_v2 = Some(dummy_object_ref(0xAA));
        response.blake3_hash = Some("c".repeat(64));
        let pre_ref = response.snip_v2.clone().unwrap();
        let pre_hash = response.blake3_hash.clone().unwrap();
        let mut proof = ProofArtifact::new(&proof_path);
        let fake = FakeSnipV2Adapter::new();

        let report = publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap();

        assert!(!report.response_published);
        assert!(!report.response_hash_computed);
        assert_eq!(response.snip_v2.as_ref().unwrap(), &pre_ref);
        assert_eq!(response.blake3_hash.as_ref().unwrap(), &pre_hash);
        // Adapter never saw the response path.
        assert!(!fake.ingest_calls().contains(&resp_path));
    }

    // ── 4. publish skips proof when ref is pre-provided ─────────────────

    #[test]
    fn publish_skips_when_proof_already_has_ref() {
        let dir = tempfile::tempdir().unwrap();
        let resp_path = write_temp_file(dir.path(), "r.bin", b"resp");
        let proof_path = dir.path().join("nonexistent_proof.bin");

        let mut response = ResponseArtifact::new(&resp_path);
        let mut proof = ProofArtifact::new(&proof_path);
        proof.snip_v2 = Some(dummy_object_ref(0xBB));
        let pre_ref = proof.snip_v2.clone().unwrap();
        let fake = FakeSnipV2Adapter::new();

        let report = publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap();

        assert!(!report.proof_published);
        assert_eq!(proof.snip_v2.as_ref().unwrap(), &pre_ref);
        assert!(!fake.ingest_calls().contains(&proof_path));
    }

    // ── 5. publish computes hash even when response ref is pre-provided ─

    #[test]
    fn publish_computes_hash_even_when_response_ref_pre_provided() {
        let dir = tempfile::tempdir().unwrap();
        let resp_bytes = b"response bytes for hashing only";
        let resp_path = write_temp_file(dir.path(), "r.bin", resp_bytes);
        let proof_path = write_temp_file(dir.path(), "p.bin", b"proof");

        let mut response = ResponseArtifact::new(&resp_path);
        response.snip_v2 = Some(dummy_object_ref(0x11)); // pre-provided
        // blake3_hash deliberately None — Step B must run and compute it.
        let mut proof = ProofArtifact::new(&proof_path);
        let fake = FakeSnipV2Adapter::new();

        let report = publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap();

        assert!(!report.response_published);
        assert!(report.response_hash_computed);
        assert_eq!(
            response.blake3_hash.as_ref().unwrap(),
            &blake3::hash(resp_bytes).to_hex().to_string()
        );
        // Step A was skipped — adapter did NOT see the response path.
        assert!(!fake.ingest_calls().contains(&resp_path));
    }

    // ── 6. response file missing surfaces as typed error in BOTH cases ──

    #[test]
    fn publish_fails_with_typed_error_when_response_file_missing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.bin");
        let proof_path = write_temp_file(dir.path(), "p.bin", b"proof");

        // Case 1: response fully empty + missing file → caught by Step A.
        {
            let mut response = ResponseArtifact::new(&missing);
            let mut proof = ProofArtifact::new(&proof_path);
            let fake = FakeSnipV2Adapter::new();
            let err =
                publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap_err();
            match err {
                ProofArtifactError::ResponseFileNotFound { path } => {
                    assert_eq!(path, missing);
                }
                other => panic!("expected ResponseFileNotFound, got {other:?}"),
            }
        }

        // Case 2: response.snip_v2 pre-provided + missing file →
        // caught by Step B's own preflight (NOT as a generic Io).
        {
            let mut response = ResponseArtifact::new(&missing);
            response.snip_v2 = Some(dummy_object_ref(0x22));
            // blake3_hash None → Step B runs.
            let mut proof = ProofArtifact::new(&proof_path);
            let fake = FakeSnipV2Adapter::new();
            let err =
                publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap_err();
            match err {
                ProofArtifactError::ResponseFileNotFound { path } => {
                    assert_eq!(path, missing);
                }
                ProofArtifactError::Io(_) => {
                    panic!("Step B leaked a generic Io error instead of ResponseFileNotFound");
                }
                other => panic!("expected ResponseFileNotFound, got {other:?}"),
            }
        }
    }

    // ── 7. proof file missing surfaces as typed error ───────────────────

    #[test]
    fn publish_fails_with_typed_error_when_proof_file_missing() {
        let dir = tempfile::tempdir().unwrap();
        let resp_path = write_temp_file(dir.path(), "r.bin", b"resp");
        let proof_missing = dir.path().join("missing_proof.bin");

        let mut response = ResponseArtifact::new(&resp_path);
        let mut proof = ProofArtifact::new(&proof_missing);
        let fake = FakeSnipV2Adapter::new();
        let err = publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap_err();
        match err {
            ProofArtifactError::ProofFileNotFound { path } => assert_eq!(path, proof_missing),
            other => panic!("expected ProofFileNotFound, got {other:?}"),
        }
    }

    // ── 8. response BLAKE3 matches independent computation ──────────────

    #[test]
    fn response_blake3_hash_matches_independent_computation() {
        let dir = tempfile::tempdir().unwrap();
        let resp_bytes = b"specific byte stream for the test to hash and verify against";
        let resp_path = write_temp_file(dir.path(), "r.bin", resp_bytes);
        let proof_path = write_temp_file(dir.path(), "p.bin", b"unused");

        let mut response = ResponseArtifact::new(&resp_path);
        let mut proof = ProofArtifact::new(&proof_path);
        let fake = FakeSnipV2Adapter::new();

        publish_proof_artifacts(&fake, &mut response, &mut proof).unwrap();

        let expected = blake3::hash(resp_bytes).to_hex().to_string();
        assert_eq!(response.blake3_hash.as_ref().unwrap(), &expected);
        // And the hash length / shape is exactly what InferenceCommitment expects.
        assert_eq!(expected.len(), 64);
        assert!(expected.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()));
    }

    // ── 9. build_commitment success ─────────────────────────────────────

    #[test]
    fn build_commitment_success() {
        let mut id_bytes = [0u8; 32];
        id_bytes.fill(0x33);
        let manifest_root = SnipV2ObjectId::from_bytes(id_bytes);
        let manifest = manifest_with_snip_root(manifest_root);

        let response = ResponseArtifact {
            local_path: PathBuf::from("/unused/in/this/test"),
            snip_v2: None,             // not required by build_commitment
            blake3_hash: Some("d".repeat(64)),
        };
        let proof = ProofArtifact {
            local_path: PathBuf::from("/unused"),
            snip_v2: Some(dummy_object_ref(0x44)),
        };

        let commitment =
            build_commitment("session-abc".into(), &manifest, &response, &proof).unwrap();
        assert_eq!(commitment.session_id, "session-abc");
        assert_eq!(commitment.model_hash, manifest.model_hash);
        assert_eq!(commitment.manifest_snip_root, manifest_root);
        assert_eq!(commitment.response_hash, "d".repeat(64));
        assert_eq!(commitment.proof_snip_root, proof.snip_v2.as_ref().unwrap().merkle_root);
    }

    // ── 10. build_commitment fails on each missing pre-condition ────────

    #[test]
    fn build_commitment_fails_on_empty_session_id() {
        let manifest = manifest_with_snip_root(SnipV2ObjectId::from_bytes([0; 32]));
        let response = ResponseArtifact {
            local_path: PathBuf::from("/x"),
            snip_v2: None,
            blake3_hash: Some("e".repeat(64)),
        };
        let proof = ProofArtifact {
            local_path: PathBuf::from("/y"),
            snip_v2: Some(dummy_object_ref(0x55)),
        };
        let err = build_commitment(String::new(), &manifest, &response, &proof).unwrap_err();
        assert!(matches!(err, ProofArtifactError::EmptySessionId));
    }

    #[test]
    fn build_commitment_fails_when_manifest_lacks_snip_root() {
        let manifest = manifest_without_snip_root();
        let response = ResponseArtifact {
            local_path: PathBuf::from("/x"),
            snip_v2: None,
            blake3_hash: Some("e".repeat(64)),
        };
        let proof = ProofArtifact {
            local_path: PathBuf::from("/y"),
            snip_v2: Some(dummy_object_ref(0x55)),
        };
        let err =
            build_commitment("sess".into(), &manifest, &response, &proof).unwrap_err();
        assert!(matches!(err, ProofArtifactError::ManifestLacksSnipRoot));
    }

    #[test]
    fn build_commitment_fails_when_response_lacks_hash() {
        let manifest = manifest_with_snip_root(SnipV2ObjectId::from_bytes([0; 32]));
        let response = ResponseArtifact {
            local_path: PathBuf::from("/x"),
            snip_v2: None,
            blake3_hash: None,
        };
        let proof = ProofArtifact {
            local_path: PathBuf::from("/y"),
            snip_v2: Some(dummy_object_ref(0x55)),
        };
        let err =
            build_commitment("sess".into(), &manifest, &response, &proof).unwrap_err();
        assert!(matches!(err, ProofArtifactError::ResponseLacksHash));
    }

    #[test]
    fn build_commitment_fails_when_proof_lacks_snip_root() {
        let manifest = manifest_with_snip_root(SnipV2ObjectId::from_bytes([0; 32]));
        let response = ResponseArtifact {
            local_path: PathBuf::from("/x"),
            snip_v2: None,
            blake3_hash: Some("e".repeat(64)),
        };
        let proof = ProofArtifact {
            local_path: PathBuf::from("/y"),
            snip_v2: None,
        };
        let err =
            build_commitment("sess".into(), &manifest, &response, &proof).unwrap_err();
        assert!(matches!(err, ProofArtifactError::ProofLacksSnipRoot));
    }

    // ── 11. restored-manifest top-level None requires annotation ────────

    #[test]
    fn restored_manifest_requires_annotation_before_commitment() {
        // Simulate the Stage-2 restore post-condition: top-level None.
        let mut manifest = manifest_without_snip_root();
        let response = ResponseArtifact {
            local_path: PathBuf::from("/x"),
            snip_v2: None,
            blake3_hash: Some("f".repeat(64)),
        };
        let proof = ProofArtifact {
            local_path: PathBuf::from("/y"),
            snip_v2: Some(dummy_object_ref(0x66)),
        };

        // First call: strict check fails.
        let err = build_commitment("sess".into(), &manifest, &response, &proof).unwrap_err();
        assert!(matches!(err, ProofArtifactError::ManifestLacksSnipRoot));

        // Caller annotates the manifest with the root they restored from.
        let mut known_root_bytes = [0u8; 32];
        known_root_bytes.fill(0x77);
        let known_root = SnipV2ObjectId::from_bytes(known_root_bytes);
        manifest.snip_v2 = Some(SnipV2ObjectRef {
            merkle_root: known_root,
            lifecycle: SnipV2Lifecycle::Active,
            plaintext_size_bytes: Some(2048),
        });

        // Second call: strict check passes, commitment uses the annotated root.
        let commitment =
            build_commitment("sess".into(), &manifest, &response, &proof).unwrap();
        assert_eq!(commitment.manifest_snip_root, known_root);
        assert_eq!(commitment.session_id, "sess");
        assert_eq!(commitment.response_hash, "f".repeat(64));
    }
}
