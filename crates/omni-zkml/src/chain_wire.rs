//! Phase 5 Stage 6 — chain wire fixture and signing-spec deliverables.
//!
//! Implements the **chain v1** payload shape and signing pipeline so the
//! chain team can lock down `InferenceAttestation` integration with three
//! deterministic test vectors.
//!
//! Pipeline:
//!
//! ```text
//! InferenceCommitment
//!   └─► commitment_to_chain_digest  ─► InferenceAttestationDigest
//!         └─► canonical_digest_bytes   = bincode(digest)
//!               └─► signing_input_bytes = DOMAIN_TAG.as_bytes() || canonical_digest_bytes
//!                     └─► sign_chain_attestation_digest(seed, &digest) -> [u8; 64]
//! ```
//!
//! **Compatibility with Stage 4.** Stage 4's [`crate::attestation`] surface
//! is byte-stable and unchanged. The local Stage-4 [`crate::CommitmentDigest`]
//! pipeline and the chain-wire pipeline produce **different** bytes for the
//! same [`omni_types::phase5::InferenceCommitment`] (different schema → different
//! bincode output → different downstream hashes/signatures). Both pipelines
//! reuse the same versioned domain string
//! ([`crate::attestation::DOMAIN_TAG`]) so a future version bump propagates
//! to both consistently.
//!
//! **Chain address derivation.** `signer_address_base58` follows the chain's
//! exact rule:
//!
//! ```text
//! address_bytes = BLAKE3(raw_ed25519_pubkey)[12..32]   // last 20 bytes
//! checksum      = BLAKE3(BLAKE3(address_bytes))[0..4]
//! payload       = address_bytes || checksum            // 24 bytes
//! signer_addr   = bs58::encode(payload).into_string()
//! ```
//!
//! No libp2p `PeerId` is used; the address is the chain's Bitcoin-base58
//! checksum-payload string and nothing else.

use serde::{Deserialize, Serialize};

use omni_types::phase5::InferenceCommitment;

use crate::attestation::DOMAIN_TAG;
use crate::error::{ChainWireError, ChainWireResult};

// ── Limits ────────────────────────────────────────────────────────────────────

/// Maximum byte length (UTF-8) of `InferenceAttestationDigest::session_id`
/// accepted by [`commitment_to_chain_digest`].
pub const MAX_SESSION_ID_BYTES: usize = 256;

// ── Wire types (chain v1) ────────────────────────────────────────────────────

/// Chain v1 digest. Field order is the on-wire bincode order; declaration
/// order is significant and **frozen** for v1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceAttestationDigest {
    pub session_id: String,
    pub model_hash: [u8; 32],
    pub manifest_root: [u8; 32],
    pub response_hash: [u8; 32],
    pub proof_root: [u8; 32],
}

/// Chain v1 tx-data envelope: digest + 64-byte Ed25519 signature.
/// Defined for type completeness and chain-team handoff; Stage 6 does
/// **not** serialize this for any wire purpose — outer SignedTransaction
/// encoding is explicitly out of scope.
///
/// `verifier_signature` uses a custom serde helper because serde 1.0's
/// built-in array impls only cover sizes up to 32. The helper emits a
/// fixed 64-element tuple of `u8`, which bincode `config::standard()`
/// encodes as a raw 64-byte sequence (no length prefix).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceAttestationTxData {
    pub digest: InferenceAttestationDigest,
    #[serde(with = "serde_signature_64")]
    pub verifier_signature: [u8; 64],
}

mod serde_signature_64 {
    use serde::de::{Error, SeqAccess, Visitor};
    use serde::ser::SerializeTuple;
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(
        arr: &[u8; 64],
        ser: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut tup = ser.serialize_tuple(64)?;
        for b in arr {
            tup.serialize_element(b)?;
        }
        tup.end()
    }

    struct Sig64Visitor;

    impl<'de> Visitor<'de> for Sig64Visitor {
        type Value = [u8; 64];

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a fixed-size array of 64 bytes")
        }

        fn visit_seq<A: SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> std::result::Result<[u8; 64], A::Error> {
            let mut out = [0u8; 64];
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(i, &self))?;
            }
            Ok(out)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        de: D,
    ) -> std::result::Result<[u8; 64], D::Error> {
        de.deserialize_tuple(64, Sig64Visitor)
    }
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert an [`InferenceCommitment`] into the chain v1 digest with strict
/// validation: bare 64-char lowercase hex on both string hashes, byte
/// length cap on `session_id`.
pub fn commitment_to_chain_digest(
    commitment: &InferenceCommitment,
) -> ChainWireResult<InferenceAttestationDigest> {
    if commitment.session_id.len() > MAX_SESSION_ID_BYTES {
        return Err(ChainWireError::SessionIdTooLong {
            got: commitment.session_id.len(),
            max: MAX_SESSION_ID_BYTES,
        });
    }
    let model_hash = parse_bare_hex_32("model_hash", &commitment.model_hash)?;
    let response_hash = parse_bare_hex_32("response_hash", &commitment.response_hash)?;
    let manifest_root = *commitment.manifest_snip_root.as_bytes();
    let proof_root = *commitment.proof_snip_root.as_bytes();
    Ok(InferenceAttestationDigest {
        session_id: commitment.session_id.clone(),
        model_hash,
        manifest_root,
        response_hash,
        proof_root,
    })
}

// ── Canonical bytes and signing input ────────────────────────────────────────

/// `bincode(digest)` under the same `config::standard()` the workspace
/// uses everywhere else.
pub fn canonical_digest_bytes(
    digest: &InferenceAttestationDigest,
) -> ChainWireResult<Vec<u8>> {
    bincode::serde::encode_to_vec(digest, bincode::config::standard())
        .map_err(|e| ChainWireError::Serialization(e.to_string()))
}

/// `DOMAIN_TAG.as_bytes() || canonical_digest_bytes(digest)` — the exact
/// byte sequence the chain team signs over.
pub fn signing_input_bytes(
    digest: &InferenceAttestationDigest,
) -> ChainWireResult<Vec<u8>> {
    let canonical = canonical_digest_bytes(digest)?;
    let domain = DOMAIN_TAG.as_bytes();
    let mut out = Vec::with_capacity(domain.len() + canonical.len());
    out.extend_from_slice(domain);
    out.extend_from_slice(&canonical);
    Ok(out)
}

// ── Ed25519 signing ──────────────────────────────────────────────────────────

/// Sign the chain signing-input bytes with an Ed25519 keypair derived from
/// a 32-byte seed. Returns the raw 64-byte signature.
pub fn sign_chain_attestation_digest(
    seed: &[u8; 32],
    digest: &InferenceAttestationDigest,
) -> ChainWireResult<[u8; 64]> {
    let signing_input = signing_input_bytes(digest)?;
    let keypair = ed25519_keypair_from_seed(seed)?;
    let sig_vec = keypair.sign(&signing_input);
    if sig_vec.len() != 64 {
        return Err(ChainWireError::Signing(format!(
            "unexpected Ed25519 signature length: {}",
            sig_vec.len()
        )));
    }
    let mut out = [0u8; 64];
    out.copy_from_slice(&sig_vec);
    Ok(out)
}

fn ed25519_keypair_from_seed(
    seed: &[u8; 32],
) -> ChainWireResult<libp2p_identity::ed25519::Keypair> {
    let mut seed_copy = *seed;
    let secret = libp2p_identity::ed25519::SecretKey::try_from_bytes(&mut seed_copy)
        .map_err(|e| ChainWireError::Signing(format!("Ed25519 secret decode failed: {e}")))?;
    Ok(libp2p_identity::ed25519::Keypair::from(secret))
}

/// Derive the raw 32-byte Ed25519 public key from a seed.
pub fn signer_pubkey_bytes(seed: &[u8; 32]) -> ChainWireResult<[u8; 32]> {
    let kp = ed25519_keypair_from_seed(seed)?;
    Ok(kp.public().to_bytes())
}

/// Lowercase 64-char hex of [`signer_pubkey_bytes`].
pub fn signer_pubkey_hex(seed: &[u8; 32]) -> ChainWireResult<String> {
    Ok(to_lower_hex(&signer_pubkey_bytes(seed)?))
}

// ── Chain address derivation (the chain spec, exactly) ──────────────────────

/// Pure, infallible derivation of the chain's base58 address from a raw
/// 32-byte Ed25519 public key. Implements:
///
/// ```text
/// address_bytes = BLAKE3(pubkey)[12..32]            // last 20 bytes
/// checksum      = BLAKE3(BLAKE3(address_bytes))[0..4]
/// payload       = address_bytes || checksum          // 24 bytes
/// address_b58   = bs58::encode(payload).into_string()
/// ```
pub fn derive_chain_address_base58(pubkey: &[u8; 32]) -> String {
    let pk_hash = blake3::hash(pubkey);
    let pk_hash_bytes = pk_hash.as_bytes();
    let address_bytes: &[u8] = &pk_hash_bytes[12..32]; // 20 bytes — the LAST 20

    let inner = blake3::hash(address_bytes);
    let outer = blake3::hash(inner.as_bytes());
    let checksum: &[u8] = &outer.as_bytes()[0..4];

    let mut payload = [0u8; 24];
    payload[..20].copy_from_slice(address_bytes);
    payload[20..].copy_from_slice(checksum);

    bs58::encode(payload).into_string()
}

/// Convenience: derive the chain address straight from an Ed25519 seed.
pub fn signer_chain_address_base58(seed: &[u8; 32]) -> ChainWireResult<String> {
    let pubkey = signer_pubkey_bytes(seed)?;
    Ok(derive_chain_address_base58(&pubkey))
}

// ── Hex helpers ──────────────────────────────────────────────────────────────

/// Lowercase hex (no `0x` prefix), 2 chars per byte.
pub fn to_lower_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

/// Parse a bare 64-char lowercase hex string into 32 bytes. Strict on
/// length (must be exactly 64), case (lowercase only), and digit shape.
fn parse_bare_hex_32(field: &'static str, s: &str) -> ChainWireResult<[u8; 32]> {
    if s.len() != 64 {
        return Err(ChainWireError::InvalidHex {
            field,
            reason: format!("expected 64 chars, got {}", s.len()),
        });
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = decode_lower_nibble(field, bytes[i * 2], i * 2)?;
        let lo = decode_lower_nibble(field, bytes[i * 2 + 1], i * 2 + 1)?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn decode_lower_nibble(field: &'static str, b: u8, offset: usize) -> ChainWireResult<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Err(ChainWireError::InvalidHex {
            field,
            reason: format!(
                "uppercase hex not allowed at offset {offset}: '{}'",
                b as char
            ),
        }),
        _ => Err(ChainWireError::InvalidHex {
            field,
            reason: format!("invalid hex digit at offset {offset}: '{}'", b as char),
        }),
    }
}

// ── Test vector struct + builder ─────────────────────────────────────────────

/// One frozen test vector for the chain team. All hex fields are bare
/// lowercase (no `0x` prefix). `signer_address_base58` uses the chain's
/// exact derivation (see [`derive_chain_address_base58`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainAttestationVector {
    pub session_id: String,
    pub model_hash: String,
    pub manifest_root: String,
    pub response_hash: String,
    pub proof_root: String,
    pub verifier_ed25519_seed: String,
    pub canonical_digest_bytes: String,
    pub signing_input_bytes: String,
    pub signature_bytes: String,
    pub signer_address_base58: String,
    pub signer_pubkey_hex: String,
}

/// Build one chain attestation test vector from hex inputs. The single
/// entry point used by the integration test to produce the JSON fixture
/// the chain team consumes.
pub fn compute_chain_attestation_vector(
    session_id: &str,
    model_hash_hex: &str,
    manifest_root_hex: &str,
    response_hash_hex: &str,
    proof_root_hex: &str,
    verifier_seed_hex: &str,
) -> ChainWireResult<ChainAttestationVector> {
    if session_id.len() > MAX_SESSION_ID_BYTES {
        return Err(ChainWireError::SessionIdTooLong {
            got: session_id.len(),
            max: MAX_SESSION_ID_BYTES,
        });
    }
    let model_hash = parse_bare_hex_32("model_hash", model_hash_hex)?;
    let manifest_root = parse_bare_hex_32("manifest_root", manifest_root_hex)?;
    let response_hash = parse_bare_hex_32("response_hash", response_hash_hex)?;
    let proof_root = parse_bare_hex_32("proof_root", proof_root_hex)?;
    let seed = parse_bare_hex_32("verifier_ed25519_seed", verifier_seed_hex)?;

    let digest = InferenceAttestationDigest {
        session_id: session_id.to_string(),
        model_hash,
        manifest_root,
        response_hash,
        proof_root,
    };

    let canonical = canonical_digest_bytes(&digest)?;
    let signing_input = signing_input_bytes(&digest)?;
    let signature = sign_chain_attestation_digest(&seed, &digest)?;
    let pubkey = signer_pubkey_bytes(&seed)?;
    let pubkey_hex = to_lower_hex(&pubkey);
    let address = derive_chain_address_base58(&pubkey);

    Ok(ChainAttestationVector {
        session_id: session_id.to_string(),
        model_hash: model_hash_hex.to_string(),
        manifest_root: manifest_root_hex.to_string(),
        response_hash: response_hash_hex.to_string(),
        proof_root: proof_root_hex.to_string(),
        verifier_ed25519_seed: verifier_seed_hex.to_string(),
        canonical_digest_bytes: to_lower_hex(&canonical),
        signing_input_bytes: to_lower_hex(&signing_input),
        signature_bytes: to_lower_hex(&signature),
        signer_address_base58: address,
        signer_pubkey_hex: pubkey_hex,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use omni_types::phase5::{InferenceCommitment, SnipV2ObjectId};

    // ── Fixtures ─────────────────────────────────────────────────────────

    fn snip_id(byte: u8) -> SnipV2ObjectId {
        let mut b = [0u8; 32];
        b.fill(byte);
        SnipV2ObjectId::from_bytes(b)
    }

    fn sample_commitment() -> InferenceCommitment {
        InferenceCommitment {
            session_id: "session-stage6".into(),
            model_hash: "0".repeat(64),
            manifest_snip_root: snip_id(0x11),
            response_hash: "1".repeat(64),
            proof_snip_root: snip_id(0x22),
        }
    }

    fn sample_digest() -> InferenceAttestationDigest {
        commitment_to_chain_digest(&sample_commitment()).unwrap()
    }

    // ── Conversion (5) ───────────────────────────────────────────────────

    #[test]
    fn commitment_to_chain_digest_succeeds() {
        let d = sample_digest();
        assert_eq!(d.session_id, "session-stage6");
        assert_eq!(d.model_hash, [0x00; 32]);
        assert_eq!(d.manifest_root, [0x11; 32]);
        assert_eq!(d.response_hash, [0x11; 32]);
        assert_eq!(d.proof_root, [0x22; 32]);
    }

    #[test]
    fn commitment_to_chain_digest_rejects_uppercase_model_hash() {
        let mut c = sample_commitment();
        c.model_hash = "A".repeat(64);
        let err = commitment_to_chain_digest(&c).unwrap_err();
        match err {
            ChainWireError::InvalidHex { field, .. } => assert_eq!(field, "model_hash"),
            other => panic!("expected InvalidHex {{ field: \"model_hash\" }}, got {other:?}"),
        }
    }

    #[test]
    fn commitment_to_chain_digest_rejects_short_model_hash() {
        let mut c = sample_commitment();
        c.model_hash = "0".repeat(63);
        let err = commitment_to_chain_digest(&c).unwrap_err();
        match err {
            ChainWireError::InvalidHex { field, reason } => {
                assert_eq!(field, "model_hash");
                assert!(reason.contains("63"), "reason was: {reason}");
            }
            other => panic!("expected InvalidHex, got {other:?}"),
        }
    }

    #[test]
    fn commitment_to_chain_digest_rejects_non_hex_model_hash() {
        let mut c = sample_commitment();
        c.model_hash = "g".repeat(64);
        let err = commitment_to_chain_digest(&c).unwrap_err();
        match err {
            ChainWireError::InvalidHex { field, .. } => assert_eq!(field, "model_hash"),
            other => panic!("expected InvalidHex, got {other:?}"),
        }
    }

    #[test]
    fn commitment_to_chain_digest_rejects_invalid_response_hash() {
        // Umbrella: uppercase + short + non-hex sub-cases all surface as
        // InvalidHex { field: "response_hash" }.
        for bad in [
            "A".repeat(64),
            "0".repeat(65),
            "z".repeat(64),
        ] {
            let mut c = sample_commitment();
            c.response_hash = bad.clone();
            let err = commitment_to_chain_digest(&c).unwrap_err();
            match err {
                ChainWireError::InvalidHex { field, .. } => {
                    assert_eq!(field, "response_hash", "bad input: {bad:?}");
                }
                other => panic!("expected InvalidHex on {bad:?}, got {other:?}"),
            }
        }
    }

    // ── Session-id length (3) ────────────────────────────────────────────

    #[test]
    fn commitment_to_chain_digest_rejects_session_id_over_256_bytes() {
        let mut c = sample_commitment();
        c.session_id = "a".repeat(257);
        let err = commitment_to_chain_digest(&c).unwrap_err();
        assert!(matches!(
            err,
            ChainWireError::SessionIdTooLong { got: 257, max: 256 }
        ));
    }

    #[test]
    fn commitment_to_chain_digest_session_id_exactly_256_bytes_allowed() {
        let mut c = sample_commitment();
        c.session_id = "a".repeat(256);
        let d = commitment_to_chain_digest(&c).unwrap();
        assert_eq!(d.session_id.len(), 256);
    }

    #[test]
    fn session_id_byte_length_is_counted_not_character_count() {
        // The Greek capital Alpha 'Α' (U+0391) is 2 bytes in UTF-8.
        // 200 characters × 2 bytes/char = 400 bytes, well over the 256 cap.
        let mut c = sample_commitment();
        c.session_id = "\u{0391}".repeat(200);
        assert_eq!(c.session_id.chars().count(), 200);
        assert_eq!(c.session_id.len(), 400);
        let err = commitment_to_chain_digest(&c).unwrap_err();
        assert!(matches!(
            err,
            ChainWireError::SessionIdTooLong { got: 400, max: 256 }
        ));
    }

    // ── Canonical bytes & signing input (5) ──────────────────────────────

    #[test]
    fn canonical_digest_bytes_is_deterministic() {
        let d = sample_digest();
        let a = canonical_digest_bytes(&d).unwrap();
        let b = canonical_digest_bytes(&d).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_digest_bytes_changes_when_any_field_changes() {
        let base = canonical_digest_bytes(&sample_digest()).unwrap();

        let mut d = sample_digest();
        d.session_id = "different".into();
        assert_ne!(canonical_digest_bytes(&d).unwrap(), base);

        let mut d = sample_digest();
        d.model_hash = [0xFF; 32];
        assert_ne!(canonical_digest_bytes(&d).unwrap(), base);

        let mut d = sample_digest();
        d.manifest_root = [0xFF; 32];
        assert_ne!(canonical_digest_bytes(&d).unwrap(), base);

        let mut d = sample_digest();
        d.response_hash = [0xFF; 32];
        assert_ne!(canonical_digest_bytes(&d).unwrap(), base);

        let mut d = sample_digest();
        d.proof_root = [0xFF; 32];
        assert_ne!(canonical_digest_bytes(&d).unwrap(), base);
    }

    #[test]
    fn signing_input_starts_with_exact_stage4_domain_string() {
        let d = sample_digest();
        let signing = signing_input_bytes(&d).unwrap();
        let domain = b"omninode.inference_attestation.v1";
        assert_eq!(&signing[..domain.len()], domain);
    }

    #[test]
    fn signing_input_equals_domain_bytes_concatenated_with_canonical_digest_bytes() {
        let d = sample_digest();
        let canonical = canonical_digest_bytes(&d).unwrap();
        let signing = signing_input_bytes(&d).unwrap();
        let mut expected = Vec::new();
        expected.extend_from_slice(DOMAIN_TAG.as_bytes());
        expected.extend_from_slice(&canonical);
        assert_eq!(signing, expected);
    }

    #[test]
    fn signing_input_length_equals_domain_len_plus_canonical_len() {
        let d = sample_digest();
        let canonical = canonical_digest_bytes(&d).unwrap();
        let signing = signing_input_bytes(&d).unwrap();
        assert_eq!(signing.len(), DOMAIN_TAG.as_bytes().len() + canonical.len());
    }

    // ── Signing (4) ──────────────────────────────────────────────────────

    #[test]
    fn ed25519_signature_is_64_bytes() {
        let seed = [7u8; 32];
        let sig = sign_chain_attestation_digest(&seed, &sample_digest()).unwrap();
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn ed25519_signature_is_deterministic_for_fixed_seed_and_digest() {
        let seed = [7u8; 32];
        let d = sample_digest();
        let a = sign_chain_attestation_digest(&seed, &d).unwrap();
        let b = sign_chain_attestation_digest(&seed, &d).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ed25519_signature_can_be_verified_with_derived_public_key() {
        let seed = [7u8; 32];
        let d = sample_digest();
        let sig = sign_chain_attestation_digest(&seed, &d).unwrap();
        let pubkey_bytes = signer_pubkey_bytes(&seed).unwrap();
        let signing_input = signing_input_bytes(&d).unwrap();

        let pubkey = libp2p_identity::ed25519::PublicKey::try_from_bytes(&pubkey_bytes)
            .expect("valid Ed25519 public key");
        assert!(pubkey.verify(&signing_input, &sig));
    }

    #[test]
    fn ed25519_signature_changes_when_digest_changes() {
        let seed = [7u8; 32];
        let d1 = sample_digest();
        let mut d2 = sample_digest();
        d2.proof_root = [0xFE; 32];
        let s1 = sign_chain_attestation_digest(&seed, &d1).unwrap();
        let s2 = sign_chain_attestation_digest(&seed, &d2).unwrap();
        assert_ne!(s1, s2);
    }

    // ── Pubkey & address (3) ─────────────────────────────────────────────

    #[test]
    fn signer_pubkey_hex_is_64_lowercase_hex_chars() {
        let hex = signer_pubkey_hex(&[7u8; 32]).unwrap();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
    }

    #[test]
    fn signer_pubkey_hex_is_deterministic_for_fixed_seed() {
        let a = signer_pubkey_hex(&[7u8; 32]).unwrap();
        let b = signer_pubkey_hex(&[7u8; 32]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn signer_chain_address_base58_is_deterministic_for_fixed_seed() {
        let a = signer_chain_address_base58(&[7u8; 32]).unwrap();
        let b = signer_chain_address_base58(&[7u8; 32]).unwrap();
        assert_eq!(a, b);
    }

    // ── Chain-address checksum invariants (1, all assertions in one) ─────

    #[test]
    fn derive_chain_address_uses_last_20_bytes_of_blake3_and_correct_checksum() {
        let pubkey: [u8; 32] = [42u8; 32];
        let address_b58 = derive_chain_address_base58(&pubkey);

        // Decode with bs58 — chain alphabet, no built-in checksum scheme.
        let decoded = bs58::decode(&address_b58)
            .into_vec()
            .expect("address must be valid base58");
        assert_eq!(
            decoded.len(),
            24,
            "decoded chain address payload must be 24 bytes (20-byte addr + 4-byte checksum)"
        );

        let (address_bytes, checksum) = decoded.split_at(20);
        assert_eq!(address_bytes.len(), 20);
        assert_eq!(checksum.len(), 4);

        // Recompute double-BLAKE3 checksum.
        let inner = blake3::hash(address_bytes);
        let outer = blake3::hash(inner.as_bytes());
        let recomputed = &outer.as_bytes()[..4];
        assert_eq!(checksum, recomputed);

        // Positive: address is the LAST 20 bytes of BLAKE3(pubkey).
        let pk_hash = blake3::hash(&pubkey);
        assert_eq!(address_bytes, &pk_hash.as_bytes()[12..32]);

        // Negative: address must NOT be the FIRST 20 bytes.
        assert_ne!(address_bytes, &pk_hash.as_bytes()[0..20]);
    }
}
