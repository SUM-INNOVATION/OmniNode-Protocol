//! Phase 5 Stage 13.0 — chain-anchor wire payload for integrity
//! evidence artifacts.
//!
//! Anchors a **specific local Stage 12.25
//! `SignedIntegrityEvidenceChainReport` artifact** to SUM Chain by
//! committing to:
//!
//! - the BLAKE3 hash of the artifact's **raw on-disk bytes**,
//! - the artifact's embedded `signer_pubkey` (Ed25519 32-byte
//!   public key),
//! - the artifact's `signed_at_utc` (Unix seconds),
//! - the artifact kind tag and schema versions.
//!
//! The full JSON wrapper never leaves the operator's host. The
//! chain stores a 32-byte hash + ~70 bytes of metadata; an
//! operator who finds the anchor on chain re-runs
//! `verify-integrity-evidence-anchor` locally to bind the chain
//! record to their on-disk artifact.
//!
//! ## Same-key submitter (locked)
//!
//! Stage 13.0 requires the anchor submitter to equal the Stage
//! 12.25 wrapper signer. The wire payload carries the artifact
//! signer pubkey on the digest; the submitter signature is
//! verified under that **same** key. There is no separate
//! `submitter_pubkey` field — keeps the on-chain payload smaller
//! and makes the refusal taxonomy unambiguous. Relay / separate-
//! submitter flows are a deferred future-stage extension and
//! would require an `anchor_schema_version` bump.
//!
//! ## Canonical bytes
//!
//! ```text
//! canonical_bytes = EVIDENCE_ANCHOR_DOMAIN
//!                || bincode1::serialize(&digest)
//! submitter_signature = Ed25519 signature over canonical_bytes
//!                       (verified under digest.signer_pubkey)
//! ```
//!
//! Uses **bincode 1.3** via the crate-local `bincode1` alias to
//! match the Stage 6 chain-wire encoder posture. Fixed-size
//! `[u8; 32]` and `u32` / `i64` fields carry no per-field length
//! prefix.
//!
//! ## What is anchored, what is not
//!
//! - **Anchored**: existence of `(artifact_hash, signer_pubkey,
//!   signed_at, artifact_kind, schema versions)`.
//! - **NOT anchored**: any semantic correctness of the underlying
//!   evidence. Chain inclusion proves "this operator submitted a
//!   commitment to this artifact at this time" — not that the
//!   evidence inside the artifact passes any verification gate.
//!   Re-running Stage 12.24 / 12.25 verification against the
//!   artifact bytes is unchanged.

use serde::{Deserialize, Serialize};

use crate::error::{EvidenceAnchorError, EvidenceAnchorResult};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Stage 13.0 anchor wire schema version. Bumping is a forward-
/// incompatible change to the on-chain payload shape; future
/// extensions (e.g. separate-submitter relay flows, new artifact
/// kinds) require a bump.
pub const INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION: u32 = 1;

/// Domain separator prepended to canonical digest bytes before
/// Ed25519 signing. Versioned per OmniNode convention.
pub const EVIDENCE_ANCHOR_DOMAIN: &[u8] = b"OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:";

// ── Closed-set artifact kind ──────────────────────────────────────────────────

/// Stage 13.0 anchored artifact kind. Closed-set enum with one
/// variant per locked plan; new variants require an
/// `INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION` bump.
///
/// `#[serde(rename_all = "snake_case")]` keeps the JSON form
/// stable across the wire (`"signed_integrity_evidence_chain_report"`)
/// while the variant name follows Rust convention.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchoredArtifactKind {
    /// Stage 12.25 `SignedIntegrityEvidenceChainReport` JSON
    /// wrapper.
    SignedIntegrityEvidenceChainReport,
}

impl AnchoredArtifactKind {
    /// Stable string tag for events and JSON / pretty output.
    /// Identical to the serde wire form.
    pub fn as_str(&self) -> &'static str {
        match self {
            AnchoredArtifactKind::SignedIntegrityEvidenceChainReport => {
                "signed_integrity_evidence_chain_report"
            }
        }
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

/// Stage 13.0 anchor digest. Field order is the on-wire bincode
/// order and is **frozen** for v1. Self-describing: the wire form
/// carries its own `anchor_schema_version`, so a chain adapter or
/// off-chain reader does not need to infer the schema from the
/// domain tag string.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntegrityEvidenceAnchorDigest {
    /// = [`INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION`]. Explicit
    /// wire field; defense-in-depth alongside the versioned
    /// [`EVIDENCE_ANCHOR_DOMAIN`].
    pub anchor_schema_version: u32,

    /// Closed-set kind of the artifact being anchored.
    pub artifact_kind: AnchoredArtifactKind,

    /// Schema version of the wrapped artifact (e.g. Stage 12.25
    /// wrapper `schema_version`).
    pub artifact_schema_version: u32,

    /// BLAKE3 over the artifact's **raw on-disk bytes**. Binds
    /// the chain record to the exact byte sequence the operator
    /// holds — any reformatting / key-order change in the JSON
    /// file produces a different anchor.
    pub artifact_hash: [u8; 32],

    /// Ed25519 public key embedded in the Stage 12.25 wrapper.
    /// Same-key-submitter rule (Stage 13.0): this is also the
    /// key the [`IntegrityEvidenceAnchorTxData::submitter_signature`]
    /// is verified under.
    pub signer_pubkey: [u8; 32],

    /// Unix seconds (UTC) parsed from the Stage 12.25 wrapper's
    /// `signed_at_utc` RFC 3339 string. Signed integer to keep
    /// the wire compatible with pre-epoch timestamps (the
    /// integrity-evidence pipeline does not produce these, but
    /// the on-chain shape stays canonical and bincode-friendly).
    pub signed_at_utc_unix: i64,
}

/// Stage 13.0 anchor transaction data: signed wire payload.
///
/// `submitter_signature` is a 64-byte Ed25519 signature over the
/// canonical bytes of [`Self::digest`] (see
/// [`canonical_anchor_bytes`]). Verified under
/// `digest.signer_pubkey` per the same-key-submitter rule.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IntegrityEvidenceAnchorTxData {
    pub digest: IntegrityEvidenceAnchorDigest,

    /// Ed25519 signature over
    /// `EVIDENCE_ANCHOR_DOMAIN || bincode1::serialize(&digest)`,
    /// verified under `digest.signer_pubkey`. Custom serde helper
    /// matches the Stage 6 64-byte tuple encoding (serde 1.0's
    /// built-in array impls only cover sizes up to 32).
    #[serde(with = "serde_signature_64")]
    pub submitter_signature: [u8; 64],
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

// ── Canonical bytes + signing primitives ──────────────────────────────────────

/// Canonical bytes of `digest` for signing / verification.
///
/// Uses **bincode 1.3** `bincode::serialize` (imported as
/// `bincode1`) to match the Stage 6 chain-wire encoder. Layout
/// (positional, no per-field tag):
///
/// ```text
/// [u32 LE  anchor_schema_version]      // 4 bytes
/// [u32 LE  artifact_kind discriminant] // 4 bytes (bincode-1 enum tag)
/// [u32 LE  artifact_schema_version]    // 4 bytes
/// [artifact_hash]                      // 32 bytes
/// [signer_pubkey]                      // 32 bytes
/// [i64 LE  signed_at_utc_unix]         // 8 bytes
/// ```
///
/// Total size: 84 bytes; deterministic and frozen for v1.
pub fn canonical_anchor_bytes(
    digest: &IntegrityEvidenceAnchorDigest,
) -> EvidenceAnchorResult<Vec<u8>> {
    bincode1::serialize(digest)
        .map_err(|e| EvidenceAnchorError::CanonicalSerialization(e.to_string()))
}

/// `EVIDENCE_ANCHOR_DOMAIN || canonical_anchor_bytes(digest)` — the
/// exact byte sequence the submitter signs. Equivalent form for
/// the `omni-zkml` chain-wire convention.
pub fn anchor_signing_input_bytes(
    digest: &IntegrityEvidenceAnchorDigest,
) -> EvidenceAnchorResult<Vec<u8>> {
    let canonical = canonical_anchor_bytes(digest)?;
    let mut out = Vec::with_capacity(EVIDENCE_ANCHOR_DOMAIN.len() + canonical.len());
    out.extend_from_slice(EVIDENCE_ANCHOR_DOMAIN);
    out.extend_from_slice(&canonical);
    Ok(out)
}

/// Sign the anchor signing-input bytes with a 32-byte Ed25519
/// seed. Returns the raw 64-byte signature. Pure; the caller
/// holds the seed.
pub fn sign_anchor_digest(
    seed: &[u8; 32],
    digest: &IntegrityEvidenceAnchorDigest,
) -> EvidenceAnchorResult<[u8; 64]> {
    let signing_input = anchor_signing_input_bytes(digest)?;
    let keypair = ed25519_keypair_from_seed(seed)?;
    let sig_vec = keypair.sign(&signing_input);
    if sig_vec.len() != 64 {
        return Err(EvidenceAnchorError::Signing(format!(
            "unexpected Ed25519 signature length: {}",
            sig_vec.len()
        )));
    }
    let mut out = [0u8; 64];
    out.copy_from_slice(&sig_vec);
    Ok(out)
}

/// Verify the submitter signature on `tx_data` under
/// `tx_data.digest.signer_pubkey` (same-key-submitter rule).
/// Returns `Ok(())` on a valid signature; refuses with the
/// closed-set [`EvidenceAnchorError::SubmitterSignatureInvalid`]
/// otherwise. Decode failures (bad pubkey bytes) surface as
/// [`EvidenceAnchorError::Signing`].
pub fn verify_anchor_tx_data(tx_data: &IntegrityEvidenceAnchorTxData) -> EvidenceAnchorResult<()> {
    if tx_data.digest.anchor_schema_version != INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION {
        return Err(EvidenceAnchorError::UnsupportedAnchorSchemaVersion {
            got: tx_data.digest.anchor_schema_version,
            expected: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
        });
    }
    let signing_input = anchor_signing_input_bytes(&tx_data.digest)?;
    let pubkey = libp2p_identity::ed25519::PublicKey::try_from_bytes(&tx_data.digest.signer_pubkey)
        .map_err(|e| EvidenceAnchorError::Signing(format!("pubkey decode failed: {e}")))?;
    if pubkey.verify(&signing_input, &tx_data.submitter_signature) {
        Ok(())
    } else {
        Err(EvidenceAnchorError::SubmitterSignatureInvalid)
    }
}

/// Derive the raw 32-byte Ed25519 public key from a seed. Same
/// shape as Stage 6 [`crate::chain_wire::signer_pubkey_bytes`];
/// duplicated here to keep evidence_anchor a self-contained
/// module that doesn't reach into Stage 6 internals.
pub fn anchor_signer_pubkey_bytes(seed: &[u8; 32]) -> EvidenceAnchorResult<[u8; 32]> {
    let kp = ed25519_keypair_from_seed(seed)?;
    Ok(kp.public().to_bytes())
}

fn ed25519_keypair_from_seed(
    seed: &[u8; 32],
) -> EvidenceAnchorResult<libp2p_identity::ed25519::Keypair> {
    let mut seed_copy = *seed;
    let secret = libp2p_identity::ed25519::SecretKey::try_from_bytes(&mut seed_copy)
        .map_err(|e| EvidenceAnchorError::Signing(format!("Ed25519 secret decode failed: {e}")))?;
    Ok(libp2p_identity::ed25519::Keypair::from(secret))
}

// ── Hex helpers ──────────────────────────────────────────────────────────────

/// Lowercase hex (no `0x` prefix), 2 chars per byte. Mirrors the
/// Stage 6 helper so the evidence_anchor surface is
/// self-contained.
pub fn anchor_hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

/// Parse a bare 64-char lowercase hex string into 32 bytes.
/// Strict on length, case, and digit shape. Mirrors Stage 6.
pub fn parse_anchor_hex_32(s: &str) -> Result<[u8; 32], String> {
    if s.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", s.len()));
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = decode_lower_nibble(bytes[i * 2], i * 2)?;
        let lo = decode_lower_nibble(bytes[i * 2 + 1], i * 2 + 1)?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn decode_lower_nibble(b: u8, offset: usize) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Err(format!(
            "uppercase hex not allowed at offset {offset}: '{}'",
            b as char
        )),
        _ => Err(format!(
            "invalid hex digit at offset {offset}: '{}'",
            b as char
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_digest() -> IntegrityEvidenceAnchorDigest {
        IntegrityEvidenceAnchorDigest {
            anchor_schema_version: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
            artifact_kind: AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
            artifact_schema_version: 1,
            artifact_hash: [0x11; 32],
            signer_pubkey: [0x22; 32],
            signed_at_utc_unix: 1_700_000_000,
        }
    }

    #[test]
    fn canonical_anchor_bytes_is_deterministic() {
        let d = sample_digest();
        let a = canonical_anchor_bytes(&d).unwrap();
        let b = canonical_anchor_bytes(&d).unwrap();
        assert_eq!(a, b);
    }

    /// Pins the exact bincode-1.3 wire layout. Total 84 bytes:
    /// 4 (u32 schema) + 4 (u32 enum tag) + 4 (u32 artifact schema) +
    /// 32 (artifact_hash) + 32 (signer_pubkey) + 8 (i64 unix).
    #[test]
    fn canonical_anchor_bytes_has_frozen_84_byte_layout() {
        let d = sample_digest();
        let canonical = canonical_anchor_bytes(&d).unwrap();
        assert_eq!(canonical.len(), 84);

        // u32 anchor_schema_version little-endian.
        assert_eq!(
            u32::from_le_bytes(canonical[0..4].try_into().unwrap()),
            INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION
        );
        // u32 enum discriminant for the one variant = 0.
        assert_eq!(u32::from_le_bytes(canonical[4..8].try_into().unwrap()), 0);
        // u32 artifact_schema_version little-endian.
        assert_eq!(u32::from_le_bytes(canonical[8..12].try_into().unwrap()), 1);
        // 32 bytes artifact_hash.
        assert_eq!(&canonical[12..44], &d.artifact_hash);
        // 32 bytes signer_pubkey.
        assert_eq!(&canonical[44..76], &d.signer_pubkey);
        // i64 signed_at_utc_unix little-endian.
        assert_eq!(
            i64::from_le_bytes(canonical[76..84].try_into().unwrap()),
            d.signed_at_utc_unix
        );
    }

    #[test]
    fn canonical_anchor_bytes_changes_when_any_field_changes() {
        let base = canonical_anchor_bytes(&sample_digest()).unwrap();

        let mut d = sample_digest();
        d.anchor_schema_version = 999;
        assert_ne!(canonical_anchor_bytes(&d).unwrap(), base);

        let mut d = sample_digest();
        d.artifact_schema_version = 2;
        assert_ne!(canonical_anchor_bytes(&d).unwrap(), base);

        let mut d = sample_digest();
        d.artifact_hash = [0xFF; 32];
        assert_ne!(canonical_anchor_bytes(&d).unwrap(), base);

        let mut d = sample_digest();
        d.signer_pubkey = [0xFF; 32];
        assert_ne!(canonical_anchor_bytes(&d).unwrap(), base);

        let mut d = sample_digest();
        d.signed_at_utc_unix += 1;
        assert_ne!(canonical_anchor_bytes(&d).unwrap(), base);
    }

    #[test]
    fn anchor_signing_input_starts_with_domain_tag() {
        let d = sample_digest();
        let signing = anchor_signing_input_bytes(&d).unwrap();
        assert_eq!(
            &signing[..EVIDENCE_ANCHOR_DOMAIN.len()],
            EVIDENCE_ANCHOR_DOMAIN
        );
    }

    #[test]
    fn anchor_signature_is_64_bytes_and_deterministic() {
        let seed = [7u8; 32];
        let d = sample_digest();
        let a = sign_anchor_digest(&seed, &d).unwrap();
        let b = sign_anchor_digest(&seed, &d).unwrap();
        assert_eq!(a.len(), 64);
        assert_eq!(a, b);
    }

    #[test]
    fn verify_anchor_tx_data_roundtrips_under_signer_pubkey() {
        let seed = [7u8; 32];
        let pubkey = anchor_signer_pubkey_bytes(&seed).unwrap();
        let mut d = sample_digest();
        d.signer_pubkey = pubkey;
        let sig = sign_anchor_digest(&seed, &d).unwrap();
        let tx = IntegrityEvidenceAnchorTxData {
            digest: d,
            submitter_signature: sig,
        };
        verify_anchor_tx_data(&tx).unwrap();
    }

    #[test]
    fn verify_anchor_tx_data_refuses_tampered_signature() {
        let seed = [7u8; 32];
        let pubkey = anchor_signer_pubkey_bytes(&seed).unwrap();
        let mut d = sample_digest();
        d.signer_pubkey = pubkey;
        let mut sig = sign_anchor_digest(&seed, &d).unwrap();
        sig[0] ^= 0x01;
        let tx = IntegrityEvidenceAnchorTxData {
            digest: d,
            submitter_signature: sig,
        };
        let err = verify_anchor_tx_data(&tx).unwrap_err();
        assert!(matches!(
            err,
            EvidenceAnchorError::SubmitterSignatureInvalid
        ));
    }

    #[test]
    fn verify_anchor_tx_data_refuses_unsupported_anchor_schema_version() {
        let seed = [7u8; 32];
        let pubkey = anchor_signer_pubkey_bytes(&seed).unwrap();
        let mut d = sample_digest();
        d.signer_pubkey = pubkey;
        let sig = sign_anchor_digest(&seed, &d).unwrap();
        // Tamper the schema version AFTER signing; verify must
        // refuse before reaching the crypto step.
        d.anchor_schema_version = 999;
        let tx = IntegrityEvidenceAnchorTxData {
            digest: d,
            submitter_signature: sig,
        };
        let err = verify_anchor_tx_data(&tx).unwrap_err();
        assert!(matches!(
            err,
            EvidenceAnchorError::UnsupportedAnchorSchemaVersion {
                got: 999,
                expected: 1
            }
        ));
    }

    #[test]
    fn anchored_artifact_kind_serializes_as_snake_case() {
        let kind = AnchoredArtifactKind::SignedIntegrityEvidenceChainReport;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"signed_integrity_evidence_chain_report\"");
        let round: AnchoredArtifactKind = serde_json::from_str(&json).unwrap();
        assert_eq!(round, kind);
        assert_eq!(kind.as_str(), "signed_integrity_evidence_chain_report");
    }

    #[test]
    fn anchor_hex_lower_round_trips_with_parse_anchor_hex_32() {
        let bytes = [0xAB; 32];
        let hex = anchor_hex_lower(&bytes);
        assert_eq!(hex.len(), 64);
        let back = parse_anchor_hex_32(&hex).unwrap();
        assert_eq!(back, bytes);
    }

    #[test]
    fn parse_anchor_hex_32_refuses_uppercase_and_short_and_non_hex() {
        assert!(parse_anchor_hex_32(&"A".repeat(64)).is_err());
        assert!(parse_anchor_hex_32(&"0".repeat(63)).is_err());
        assert!(parse_anchor_hex_32(&"g".repeat(64)).is_err());
    }
}
