//! Phase 5 Stage 4 — local verifier attestation envelope.
//!
//! Pipeline: `InferenceCommitment` → domain-separated canonical bytes
//! (bincode 2.0 with `config::standard()`) → BLAKE3 digest → signed by a
//! [`Signer`] → [`InferenceAttestation`].
//!
//! No real chain semantics this stage:
//! - Signature/address string encodings are implementation-defined; chain
//!   encoding is pending.
//! - No proof verification, no chain submission, no `Verifier` companion
//!   trait.
//! - The 32-byte digest is the signing input; implementations that prefer
//!   raw bytes can hash internally.
//!
//! Domain separation: the [`DOMAIN_TAG`] string is encoded as the **first**
//! field of [`CommitmentPayload`], so any change to either the tag (version
//! bump) or to the commitment fields perturbs the digest. The
//! `pub(crate) compute_canonical_bytes_with_domain` seam exists so tests
//! can prove this directly.

use serde::{Deserialize, Serialize};

use omni_types::phase5::{InferenceAttestation, InferenceCommitment};

use crate::error::{AttestationError, AttestationResult, SignerError};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Domain tag baked into canonical commitment bytes for domain separation.
/// Bumping the trailing `vN` is the contract for any breaking change to the
/// canonical envelope.
pub const DOMAIN_TAG: &str = "omninode.inference_attestation.v1";

// ── Envelope + digest types ──────────────────────────────────────────────────

/// Domain-separated envelope encoded to produce the canonical bytes a
/// [`CommitmentDigest`] is computed over.
///
/// `domain` is the **first** field on purpose: bincode encodes fields in
/// struct declaration order, so the domain tag appears before any
/// commitment bytes in the serialized output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentPayload {
    pub domain: String,
    pub commitment: InferenceCommitment,
}

/// 32-byte BLAKE3 digest of the canonical commitment bytes. The signing
/// input for any [`Signer`] implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommitmentDigest([u8; 32]);

impl CommitmentDigest {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase 64-char hex without `0x` prefix, matching the existing
    /// `model_hash` / `blake3_hash` convention used elsewhere in the
    /// protocol.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in &self.0 {
            use std::fmt::Write;
            let _ = write!(&mut s, "{:02x}", b);
        }
        s
    }
}

// ── Canonical bytes & digest helpers ─────────────────────────────────────────

/// Compute the canonical bytes for `commitment` using [`DOMAIN_TAG`].
///
/// Byte-stable across runs and platforms for a fixed schema (bincode 2.0
/// with `config::standard()`, struct field declaration order).
pub fn compute_canonical_bytes(
    commitment: &InferenceCommitment,
) -> AttestationResult<Vec<u8>> {
    compute_canonical_bytes_with_domain(commitment, DOMAIN_TAG)
}

/// Crate-internal domain-parameterised seam. Tests use this to prove that
/// changing the domain string perturbs the digest even when the commitment
/// is bit-identical. Not re-exported from the crate root.
pub(crate) fn compute_canonical_bytes_with_domain(
    commitment: &InferenceCommitment,
    domain: &str,
) -> AttestationResult<Vec<u8>> {
    let payload = CommitmentPayload {
        domain: domain.to_string(),
        commitment: commitment.clone(),
    };
    bincode::serde::encode_to_vec(&payload, bincode::config::standard())
        .map_err(|e| AttestationError::Serialization(e.to_string()))
}

/// Compute the BLAKE3 [`CommitmentDigest`] over [`compute_canonical_bytes`].
pub fn compute_digest(
    commitment: &InferenceCommitment,
) -> AttestationResult<CommitmentDigest> {
    let bytes = compute_canonical_bytes(commitment)?;
    let mut out = [0u8; 32];
    out.copy_from_slice(blake3::hash(&bytes).as_bytes());
    Ok(CommitmentDigest(out))
}

// ── Signer trait ─────────────────────────────────────────────────────────────

/// Produces a verifier signature over a 32-byte commitment digest.
///
/// Both methods return strings whose encodings (hex, base58, base64, …) are
/// implementation-defined. Chain encoding is pending; Stage 4 only enforces
/// non-empty at the [`build_attestation`] boundary.
pub trait Signer {
    /// Stable identifier of this verifier; chain encoding is pending.
    fn verifier_address(&self) -> String;

    /// Sign a 32-byte commitment digest; chain encoding is pending.
    fn sign(&self, digest: &CommitmentDigest) -> std::result::Result<String, SignerError>;
}

// ── Builder ──────────────────────────────────────────────────────────────────

/// Consume an [`InferenceCommitment`] and produce a signed
/// [`InferenceAttestation`].
///
/// Light pre-validation catches obvious hand-built mistakes without
/// re-implementing the Stage-3 build path: `session_id`, `model_hash`, and
/// `response_hash` must each be non-empty. The two SNIP roots are
/// type-safe `SnipV2ObjectId`s and cannot be empty.
///
/// The signer is called once with the BLAKE3 digest of the canonical
/// bytes; signer failures propagate as [`AttestationError::Signer`] with
/// the inner [`SignerError`] message preserved verbatim. The returned
/// `verifier_address` and `verifier_signature` strings are both required
/// to be non-empty.
pub fn build_attestation<S: Signer>(
    commitment: InferenceCommitment,
    signer: &S,
) -> AttestationResult<InferenceAttestation> {
    if commitment.session_id.is_empty() {
        return Err(AttestationError::EmptySessionId);
    }
    if commitment.model_hash.is_empty() {
        return Err(AttestationError::EmptyModelHash);
    }
    if commitment.response_hash.is_empty() {
        return Err(AttestationError::EmptyResponseHash);
    }

    let digest = compute_digest(&commitment)?;

    let verifier_address = signer.verifier_address();
    if verifier_address.is_empty() {
        return Err(AttestationError::EmptyVerifierAddress);
    }

    let verifier_signature = signer.sign(&digest)?;
    if verifier_signature.is_empty() {
        return Err(AttestationError::EmptySignature);
    }

    Ok(InferenceAttestation {
        commitment,
        verifier_address,
        verifier_signature,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;

    use omni_types::phase5::SnipV2ObjectId;

    // ── Fake signer ──────────────────────────────────────────────────────

    struct FakeSigner {
        address: String,
        signature_to_emit: std::result::Result<String, SignerError>,
        recorded_digest: RefCell<Option<CommitmentDigest>>,
    }

    impl FakeSigner {
        fn ok(address: &str, signature: &str) -> Self {
            Self {
                address: address.to_string(),
                signature_to_emit: Ok(signature.to_string()),
                recorded_digest: RefCell::new(None),
            }
        }
        fn fail(address: &str, err: SignerError) -> Self {
            Self {
                address: address.to_string(),
                signature_to_emit: Err(err),
                recorded_digest: RefCell::new(None),
            }
        }
    }

    impl Signer for FakeSigner {
        fn verifier_address(&self) -> String {
            self.address.clone()
        }
        fn sign(&self, digest: &CommitmentDigest) -> std::result::Result<String, SignerError> {
            *self.recorded_digest.borrow_mut() = Some(*digest);
            self.signature_to_emit.clone()
        }
    }

    // ── Test fixtures ────────────────────────────────────────────────────

    fn id(byte: u8) -> SnipV2ObjectId {
        let mut b = [0u8; 32];
        b.fill(byte);
        SnipV2ObjectId::from_bytes(b)
    }

    fn sample_commitment() -> InferenceCommitment {
        InferenceCommitment {
            session_id: "session-42".into(),
            model_hash: "a".repeat(64),
            manifest_snip_root: id(0x11),
            response_hash: "b".repeat(64),
            proof_snip_root: id(0x22),
        }
    }

    // ── 1. canonical bytes are deterministic ─────────────────────────────

    #[test]
    fn canonical_bytes_are_deterministic() {
        let c = sample_commitment();
        let a = compute_canonical_bytes(&c).unwrap();
        let b = compute_canonical_bytes(&c).unwrap();
        assert_eq!(a, b);
    }

    // ── 2. canonical bytes contain the literal DOMAIN_TAG string ─────────

    #[test]
    fn canonical_bytes_start_with_domain_tag_string() {
        let c = sample_commitment();
        let bytes = compute_canonical_bytes(&c).unwrap();
        // bincode encodes a leading String as varint-length + UTF-8 bytes,
        // so the literal tag appears as a contiguous substring.
        let needle = DOMAIN_TAG.as_bytes();
        let found = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "DOMAIN_TAG not found in canonical bytes");
    }

    // ── 3. domain/version change perturbs digest ─────────────────────────

    #[test]
    fn digest_changes_when_domain_tag_changes() {
        let c = sample_commitment();
        let v1 = compute_canonical_bytes_with_domain(&c, DOMAIN_TAG).unwrap();
        let vx = compute_canonical_bytes_with_domain(
            &c,
            "omninode.inference_attestation.vX",
        )
        .unwrap();
        assert_ne!(v1, vx, "different domain must produce different bytes");
        let d_v1 = blake3::hash(&v1);
        let d_vx = blake3::hash(&vx);
        assert_ne!(
            d_v1.as_bytes(),
            d_vx.as_bytes(),
            "different domain must produce different digest"
        );
    }

    // ── 4. digest matches independent BLAKE3 of canonical bytes ──────────

    #[test]
    fn digest_matches_independent_blake3_of_canonical_bytes() {
        let c = sample_commitment();
        let bytes = compute_canonical_bytes(&c).unwrap();
        let independent = blake3::hash(&bytes);
        let library = compute_digest(&c).unwrap();
        assert_eq!(library.as_bytes(), independent.as_bytes());
    }

    // ── 5–9. digest changes when each commitment field changes ──────────

    fn assert_digests_differ(modified: InferenceCommitment) {
        let base = compute_digest(&sample_commitment()).unwrap();
        let other = compute_digest(&modified).unwrap();
        assert_ne!(base, other);
    }

    #[test]
    fn digest_changes_when_session_id_changes() {
        let mut c = sample_commitment();
        c.session_id = "different-session".into();
        assert_digests_differ(c);
    }

    #[test]
    fn digest_changes_when_model_hash_changes() {
        let mut c = sample_commitment();
        c.model_hash = "c".repeat(64);
        assert_digests_differ(c);
    }

    #[test]
    fn digest_changes_when_manifest_snip_root_changes() {
        let mut c = sample_commitment();
        c.manifest_snip_root = id(0x99);
        assert_digests_differ(c);
    }

    #[test]
    fn digest_changes_when_response_hash_changes() {
        let mut c = sample_commitment();
        c.response_hash = "d".repeat(64);
        assert_digests_differ(c);
    }

    #[test]
    fn digest_changes_when_proof_snip_root_changes() {
        let mut c = sample_commitment();
        c.proof_snip_root = id(0xAA);
        assert_digests_differ(c);
    }

    // ── 10. build_attestation fills address and signature ────────────────

    #[test]
    fn build_attestation_fills_address_and_signature() {
        let signer = FakeSigner::ok("addr-1", "sig-1");
        let c = sample_commitment();
        let att = build_attestation(c.clone(), &signer).unwrap();
        assert_eq!(att.commitment, c);
        assert_eq!(att.verifier_address, "addr-1");
        assert_eq!(att.verifier_signature, "sig-1");
    }

    // ── 11. signer receives the digest (not raw canonical bytes) ─────────

    #[test]
    fn build_attestation_passes_digest_not_raw_bytes_to_signer() {
        let signer = FakeSigner::ok("addr", "sig");
        let c = sample_commitment();
        let expected = compute_digest(&c).unwrap();
        let _ = build_attestation(c, &signer).unwrap();
        let recorded = signer.recorded_digest.borrow().unwrap();
        assert_eq!(recorded, expected);
    }

    // ── 12. same commitment + signer → same attestation ──────────────────

    #[test]
    fn build_attestation_is_deterministic() {
        let signer = FakeSigner::ok("addr", "sig");
        let c = sample_commitment();
        let a = build_attestation(c.clone(), &signer).unwrap();
        let b = build_attestation(c, &signer).unwrap();
        assert_eq!(a.commitment, b.commitment);
        assert_eq!(a.verifier_address, b.verifier_address);
        assert_eq!(a.verifier_signature, b.verifier_signature);
    }

    // ── 13–15. build rejects empty commitment fields ─────────────────────

    #[test]
    fn build_attestation_rejects_empty_session_id() {
        let signer = FakeSigner::ok("addr", "sig");
        let mut c = sample_commitment();
        c.session_id = String::new();
        let err = build_attestation(c, &signer).unwrap_err();
        assert!(matches!(err, AttestationError::EmptySessionId));
        // Signer was never invoked.
        assert!(signer.recorded_digest.borrow().is_none());
    }

    #[test]
    fn build_attestation_rejects_empty_model_hash() {
        let signer = FakeSigner::ok("addr", "sig");
        let mut c = sample_commitment();
        c.model_hash = String::new();
        let err = build_attestation(c, &signer).unwrap_err();
        assert!(matches!(err, AttestationError::EmptyModelHash));
        assert!(signer.recorded_digest.borrow().is_none());
    }

    #[test]
    fn build_attestation_rejects_empty_response_hash() {
        let signer = FakeSigner::ok("addr", "sig");
        let mut c = sample_commitment();
        c.response_hash = String::new();
        let err = build_attestation(c, &signer).unwrap_err();
        assert!(matches!(err, AttestationError::EmptyResponseHash));
        assert!(signer.recorded_digest.borrow().is_none());
    }

    // ── 16. signer returning empty address rejected ──────────────────────

    #[test]
    fn build_attestation_rejects_empty_verifier_address_from_signer() {
        let signer = FakeSigner::ok("", "sig");
        let c = sample_commitment();
        let err = build_attestation(c, &signer).unwrap_err();
        assert!(matches!(err, AttestationError::EmptyVerifierAddress));
    }

    // ── 17. signer returning empty signature rejected ────────────────────

    #[test]
    fn build_attestation_rejects_empty_signature_from_signer() {
        let signer = FakeSigner::ok("addr", "");
        let c = sample_commitment();
        let err = build_attestation(c, &signer).unwrap_err();
        assert!(matches!(err, AttestationError::EmptySignature));
    }

    // ── 18. signer failure propagates with the inner message preserved ──

    #[test]
    fn build_attestation_propagates_signer_failure_with_message() {
        let signer = FakeSigner::fail("addr", SignerError::Failed("hsm offline".into()));
        let c = sample_commitment();
        let err = build_attestation(c, &signer).unwrap_err();
        match err {
            AttestationError::Signer(SignerError::Failed(msg)) => {
                assert_eq!(msg, "hsm offline");
            }
            other => panic!("expected Signer(Failed(\"hsm offline\")), got {other:?}"),
        }
    }
}
