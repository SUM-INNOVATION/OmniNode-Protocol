//! Phase 5 — SNIP V2 storage and proof-commitment types.
//!
//! These types describe references to SNIP V2 objects (`sum-node` storage)
//! and the data containers OmniNode emits when committing a verified
//! inference result. They are intentionally additive: existing CIDv1 /
//! BLAKE3 identifiers used by `omni-store` remain in use and are not
//! replaced.
//!
//! Wire conventions:
//! - `SnipV2ObjectId` serializes as a `0x`-prefixed lowercase hex string.
//! - `SnipV2Lifecycle` serializes via the default serde representation
//!   (variant name as a bare string). Parsing CLI stdout uses [`FromStr`]
//!   directly and does not route through serde.
//!
//! Stage-1 scope: data only. No chain client, no proof backend.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

// ── SnipV2ParseError ──────────────────────────────────────────────────────────

/// Reason a string failed to parse as a [`SnipV2ObjectId`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SnipV2ParseError {
    #[error("SNIP V2 object ID missing required `0x` hex prefix")]
    MissingHexPrefix,

    #[error("SNIP V2 object ID has wrong length: expected 66 (`0x` + 64 hex chars), got {got}")]
    WrongLength { got: usize },

    #[error("SNIP V2 object ID contains non-lowercase hex character at offset {offset}")]
    NonLowercaseHex { offset: usize },

    #[error("SNIP V2 object ID contains invalid hex digit '{ch}' at offset {offset}")]
    InvalidHexDigit { offset: usize, ch: char },
}

// ── SnipV2ObjectId ────────────────────────────────────────────────────────────

/// A SNIP V2 object identifier: a 32-byte BLAKE3 Merkle root.
///
/// Rendered on the wire and on disk as a `0x`-prefixed lowercase hex string
/// (66 characters total). This is **not** a CIDv1 and **not** the raw BLAKE3
/// hash of a file; CIDs and SNIP V2 roots address different things and may
/// coexist on the same shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnipV2ObjectId([u8; 32]);

impl SnipV2ObjectId {
    /// Build from raw 32 bytes (no validation needed — the byte array length
    /// is enforced by the type system).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Parse the canonical `0x`-prefixed lowercase hex form.
    pub fn from_hex(s: &str) -> Result<Self, SnipV2ParseError> {
        if !s.starts_with("0x") {
            return Err(SnipV2ParseError::MissingHexPrefix);
        }
        if s.len() != 66 {
            return Err(SnipV2ParseError::WrongLength { got: s.len() });
        }
        let bytes = s.as_bytes();
        let mut out = [0u8; 32];
        for i in 0..32 {
            let hi_off = 2 + i * 2;
            let lo_off = hi_off + 1;
            let hi = decode_lower_nibble(bytes[hi_off], hi_off)?;
            let lo = decode_lower_nibble(bytes[lo_off], lo_off)?;
            out[i] = (hi << 4) | lo;
        }
        Ok(Self(out))
    }

    /// Render as `0x` + 64 lowercase hex chars.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(66);
        s.push_str("0x");
        for b in &self.0 {
            s.push(nibble_to_lower_hex(b >> 4));
            s.push(nibble_to_lower_hex(b & 0x0F));
        }
        s
    }
}

fn decode_lower_nibble(b: u8, offset: usize) -> Result<u8, SnipV2ParseError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Err(SnipV2ParseError::NonLowercaseHex { offset }),
        _ => Err(SnipV2ParseError::InvalidHexDigit {
            offset,
            ch: b as char,
        }),
    }
}

fn nibble_to_lower_hex(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + n - 10) as char,
        _ => unreachable!("nibble must be 0..=15"),
    }
}

impl FromStr for SnipV2ObjectId {
    type Err = SnipV2ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_hex(s)
    }
}

impl fmt::Display for SnipV2ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl Serialize for SnipV2ObjectId {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for SnipV2ObjectId {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        SnipV2ObjectId::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

// ── SnipV2Lifecycle ───────────────────────────────────────────────────────────

/// Lifecycle status reported by `sum-node` for a SNIP V2 object.
///
/// Two parsing paths exist and are intentionally independent:
/// - [`Serialize`]/[`Deserialize`] (default representation) — used when an
///   `InferenceCommitment` / `ModelManifest` is round-tripped through
///   `serde_json` or `ciborium`.
/// - [`FromStr`] — used by the SNIP V2 CLI stdout parser in `omni-store`.
///   The parser does not go through serde.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SnipV2Lifecycle {
    Active,
    Pending,
    Abandoned,
}

/// Error returned by [`SnipV2Lifecycle::from_str`] when the token is not one
/// of the three documented variants. The rejected token is preserved so the
/// caller can surface it in higher-level error messages.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown SNIP V2 lifecycle token: '{0}'")]
pub struct LifecycleParseError(pub String);

impl FromStr for SnipV2Lifecycle {
    type Err = LifecycleParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Active" => Ok(Self::Active),
            "Pending" => Ok(Self::Pending),
            "Abandoned" => Ok(Self::Abandoned),
            other => Err(LifecycleParseError(other.to_string())),
        }
    }
}

impl fmt::Display for SnipV2Lifecycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Active => "Active",
            Self::Pending => "Pending",
            Self::Abandoned => "Abandoned",
        })
    }
}

// ── SnipV2ObjectRef ───────────────────────────────────────────────────────────

/// Reference to a SNIP V2 object: its Merkle root plus the lifecycle the
/// publisher last observed. Optional `plaintext_size_bytes` is populated when
/// the caller already knows the size (e.g. from the local file that was
/// ingested); absence means "size not recorded here".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnipV2ObjectRef {
    pub merkle_root: SnipV2ObjectId,
    pub lifecycle: SnipV2Lifecycle,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plaintext_size_bytes: Option<u64>,
}

// ── InferenceCommitment ───────────────────────────────────────────────────────

/// Data the verifier signs over after checking a stage's proof.
///
/// Hex conventions in this struct are deliberately mixed and that mix
/// reflects the protocol:
/// - `model_hash` and `response_hash` are **bare lowercase hex** (no `0x`
///   prefix), matching `ModelManifest.model_hash` and `ShardDescriptor.blake3_hash`.
/// - `manifest_snip_root` and `proof_snip_root` are `SnipV2ObjectId`, which
///   serialize as **`0x`-prefixed lowercase hex** strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceCommitment {
    /// Pipeline session identifier; same string shape as
    /// `PipelineSchedule.session_id`, typically `Uuid::new_v4().to_string()`.
    pub session_id: String,
    /// BLAKE3 hex of the original model GGUF file (no `0x` prefix).
    pub model_hash: String,
    /// SNIP V2 Merkle root of the published model manifest.
    pub manifest_snip_root: SnipV2ObjectId,
    /// BLAKE3 hex of the final response payload (no `0x` prefix).
    pub response_hash: String,
    /// SNIP V2 Merkle root of the published proof artifact.
    pub proof_snip_root: SnipV2ObjectId,
}

// ── InferenceAttestation ──────────────────────────────────────────────────────

/// A signed commitment.
///
/// The exact chain encoding of `verifier_address` and `verifier_signature`
/// (signature scheme, address format, base64 vs hex) is **not finalized** —
/// both fields are intentionally `String` so the wire format can be locked in
/// once the SUM Chain specification is confirmed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceAttestation {
    pub commitment: InferenceCommitment,
    /// Verifier identity; chain encoding pending.
    pub verifier_address: String,
    /// Signature over `commitment`; encoding pending.
    pub verifier_signature: String,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_HEX: &str =
        "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn object_id_parses_valid_lowercase_hex() {
        let id = SnipV2ObjectId::from_hex(VALID_HEX).unwrap();
        assert_eq!(id.to_hex(), VALID_HEX);
    }

    #[test]
    fn object_id_rejects_missing_prefix() {
        let stripped = &VALID_HEX[2..];
        let err = SnipV2ObjectId::from_hex(stripped).unwrap_err();
        assert_eq!(err, SnipV2ParseError::MissingHexPrefix);
    }

    #[test]
    fn object_id_rejects_wrong_length() {
        let short = format!("0x{}", "a".repeat(63));
        assert!(matches!(
            SnipV2ObjectId::from_hex(&short),
            Err(SnipV2ParseError::WrongLength { got: 65 })
        ));
        let long = format!("0x{}", "a".repeat(65));
        assert!(matches!(
            SnipV2ObjectId::from_hex(&long),
            Err(SnipV2ParseError::WrongLength { got: 67 })
        ));
    }

    #[test]
    fn object_id_rejects_uppercase() {
        let upper = format!("0x{}", "A".repeat(64));
        assert!(matches!(
            SnipV2ObjectId::from_hex(&upper),
            Err(SnipV2ParseError::NonLowercaseHex { offset: 2 })
        ));
    }

    #[test]
    fn object_id_rejects_non_hex() {
        let s = format!("0x{}", "g".repeat(64));
        assert!(matches!(
            SnipV2ObjectId::from_hex(&s),
            Err(SnipV2ParseError::InvalidHexDigit { offset: 2, ch: 'g' })
        ));
    }

    #[test]
    fn object_id_serde_round_trip() {
        let id = SnipV2ObjectId::from_hex(VALID_HEX).unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{VALID_HEX}\""));
        let back: SnipV2ObjectId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn object_id_serde_rejects_invalid_string() {
        let result: Result<SnipV2ObjectId, _> = serde_json::from_str("\"0xZZZ\"");
        assert!(result.is_err());
    }

    #[test]
    fn lifecycle_from_str_each_variant() {
        assert_eq!(
            SnipV2Lifecycle::from_str("Active").unwrap(),
            SnipV2Lifecycle::Active
        );
        assert_eq!(
            SnipV2Lifecycle::from_str("Pending").unwrap(),
            SnipV2Lifecycle::Pending
        );
        assert_eq!(
            SnipV2Lifecycle::from_str("Abandoned").unwrap(),
            SnipV2Lifecycle::Abandoned
        );
        let err = SnipV2Lifecycle::from_str("Garbage").unwrap_err();
        assert_eq!(err.0, "Garbage");
    }

    #[test]
    fn lifecycle_serde_round_trip() {
        let json = serde_json::to_string(&SnipV2Lifecycle::Active).unwrap();
        assert_eq!(json, "\"Active\"");
        let back: SnipV2Lifecycle = serde_json::from_str(&json).unwrap();
        assert_eq!(back, SnipV2Lifecycle::Active);
    }

    #[test]
    fn object_ref_serde_round_trip_with_and_without_size() {
        let id = SnipV2ObjectId::from_hex(VALID_HEX).unwrap();

        let with = SnipV2ObjectRef {
            merkle_root: id,
            lifecycle: SnipV2Lifecycle::Active,
            plaintext_size_bytes: Some(12_345),
        };
        let json = serde_json::to_string(&with).unwrap();
        assert!(json.contains("\"plaintext_size_bytes\":12345"));
        let back: SnipV2ObjectRef = serde_json::from_str(&json).unwrap();
        assert_eq!(with, back);

        let without = SnipV2ObjectRef {
            merkle_root: id,
            lifecycle: SnipV2Lifecycle::Active,
            plaintext_size_bytes: None,
        };
        let json = serde_json::to_string(&without).unwrap();
        assert!(!json.contains("plaintext_size_bytes"));
        let back: SnipV2ObjectRef = serde_json::from_str(&json).unwrap();
        assert_eq!(without, back);
    }

    #[test]
    fn inference_commitment_serde_round_trip() {
        let id = SnipV2ObjectId::from_hex(VALID_HEX).unwrap();
        let c = InferenceCommitment {
            session_id: "test-session".into(),
            model_hash: "a".repeat(64),
            manifest_snip_root: id,
            response_hash: "b".repeat(64),
            proof_snip_root: id,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: InferenceCommitment = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn inference_attestation_serde_round_trip() {
        let id = SnipV2ObjectId::from_hex(VALID_HEX).unwrap();
        let att = InferenceAttestation {
            commitment: InferenceCommitment {
                session_id: "sess".into(),
                model_hash: "a".repeat(64),
                manifest_snip_root: id,
                response_hash: "b".repeat(64),
                proof_snip_root: id,
            },
            verifier_address: "pending-verifier-address".into(),
            verifier_signature: "pending-signature".into(),
        };
        let json = serde_json::to_string(&att).unwrap();
        let back: InferenceAttestation = serde_json::from_str(&json).unwrap();
        assert_eq!(att, back);
    }
}
