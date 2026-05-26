//! Stage 12.0 — Ed25519 signing for contributors and dispatchers.
//!
//! Uses `libp2p-identity::ed25519` **directly** (no `omni-zkml`
//! dependency). The contributor crate stays clean of chain/proof
//! concerns; the seed handling, pubkey derivation, and sign/verify
//! primitives are duplicated here rather than re-exported from
//! `omni-zkml`. The duplication is intentional and small (~40 lines)
//! — it keeps the contributor role separable from chain-attestation
//! signing keys.
//!
//! Seed handling: each signer loads a raw 32-byte seed file. The
//! recommended practice (documented in
//! `docs/stage12-contributor-protocol.md`) is to use a seed
//! **distinct** from any chain-attestation seed (Stage 6's
//! `sign_chain_attestation_digest` seed) — different role, less
//! blast radius if compromised.

use std::path::Path;

use libp2p_identity::ed25519;

use crate::canonical::hex_lower;
use crate::error::SigningError;

/// 32-byte Ed25519 seed loaded from disk. Held opaquely so callers
/// don't get to inspect the raw bytes.
pub struct ContributorSigner {
    keypair: ed25519::Keypair,
}

/// Dispatcher signer. Identical surface to `ContributorSigner`; the
/// distinct type keeps role mistakes from compiling silently.
pub struct DispatcherSigner {
    keypair: ed25519::Keypair,
}

impl ContributorSigner {
    /// Load a 32-byte seed file. Refuses any file whose length is
    /// not exactly 32 bytes.
    pub fn from_seed_file(path: &Path) -> Result<Self, SigningError> {
        let bytes = std::fs::read(path)?;
        let keypair = keypair_from_seed_bytes(&bytes)?;
        Ok(Self { keypair })
    }

    /// In-process constructor (tests). Refuses non-32-byte slices.
    pub fn from_seed_bytes(bytes: &[u8]) -> Result<Self, SigningError> {
        let keypair = keypair_from_seed_bytes(bytes)?;
        Ok(Self { keypair })
    }

    pub fn pubkey_bytes(&self) -> [u8; 32] {
        self.keypair.public().to_bytes()
    }

    pub fn pubkey_hex(&self) -> String {
        hex_lower(&self.pubkey_bytes())
    }

    /// Ed25519-sign `msg`. Returns the raw 64-byte signature.
    pub fn sign(&self, msg: &[u8]) -> [u8; 64] {
        sign_via_keypair(&self.keypair, msg)
    }

    /// Hex-encoded 128-char signature.
    pub fn sign_hex(&self, msg: &[u8]) -> String {
        hex_lower(&self.sign(msg))
    }
}

impl DispatcherSigner {
    pub fn from_seed_file(path: &Path) -> Result<Self, SigningError> {
        let bytes = std::fs::read(path)?;
        let keypair = keypair_from_seed_bytes(&bytes)?;
        Ok(Self { keypair })
    }

    pub fn from_seed_bytes(bytes: &[u8]) -> Result<Self, SigningError> {
        let keypair = keypair_from_seed_bytes(bytes)?;
        Ok(Self { keypair })
    }

    pub fn pubkey_bytes(&self) -> [u8; 32] {
        self.keypair.public().to_bytes()
    }

    pub fn pubkey_hex(&self) -> String {
        hex_lower(&self.pubkey_bytes())
    }

    pub fn sign(&self, msg: &[u8]) -> [u8; 64] {
        sign_via_keypair(&self.keypair, msg)
    }

    pub fn sign_hex(&self, msg: &[u8]) -> String {
        hex_lower(&self.sign(msg))
    }
}

// ── Verification helpers (no role-specific wrappers; verifiers are
//    callers and don't hold private keys).

/// Verify a 64-byte Ed25519 signature against a 32-byte public key.
/// Decode failures (bad hex widths upstream, bad signature bytes,
/// bad pubkey bytes) return `Err`; cryptographic verification
/// failure returns `Ok(false)`.
pub fn verify_signature(
    pubkey_bytes: &[u8; 32],
    msg: &[u8],
    signature_bytes: &[u8; 64],
) -> Result<bool, SigningError> {
    let pubkey = ed25519::PublicKey::try_from_bytes(pubkey_bytes)
        .map_err(|e| SigningError::PublicDecode(e.to_string()))?;
    Ok(pubkey.verify(msg, signature_bytes))
}

/// Verify a hex-encoded signature against a hex-encoded public key.
/// Hex parse failures return `Err`; cryptographic verification
/// failure returns `Ok(false)`.
pub fn verify_signature_hex(
    pubkey_hex: &str,
    msg: &[u8],
    signature_hex: &str,
) -> Result<bool, SigningError> {
    let pubkey_bytes: [u8; 32] = parse_hex_fixed::<32>(pubkey_hex)
        .map_err(|e| SigningError::PublicDecode(e.to_string()))?;
    let sig_bytes: [u8; 64] = parse_hex_fixed::<64>(signature_hex)
        .map_err(|e| SigningError::SignatureDecode(e.to_string()))?;
    verify_signature(&pubkey_bytes, msg, &sig_bytes)
}

// ── Internal helpers ──────────────────────────────────────────────────────

fn keypair_from_seed_bytes(bytes: &[u8]) -> Result<ed25519::Keypair, SigningError> {
    if bytes.len() != 32 {
        return Err(SigningError::SeedWrongLength { got: bytes.len() });
    }
    let mut seed_copy = [0u8; 32];
    seed_copy.copy_from_slice(bytes);
    let secret = ed25519::SecretKey::try_from_bytes(&mut seed_copy)
        .map_err(|e| SigningError::SecretDecode(e.to_string()))?;
    Ok(ed25519::Keypair::from(secret))
}

fn sign_via_keypair(keypair: &ed25519::Keypair, msg: &[u8]) -> [u8; 64] {
    let sig_vec = keypair.sign(msg);
    debug_assert_eq!(sig_vec.len(), 64, "Ed25519 signature must be 64 bytes");
    let mut out = [0u8; 64];
    out.copy_from_slice(&sig_vec);
    out
}

fn parse_hex_fixed<const N: usize>(s: &str) -> Result<[u8; N], &'static str> {
    if s.len() != N * 2 {
        return Err("wrong hex length");
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; N];
    for i in 0..N {
        let hi = decode_nibble(bytes[i * 2])?;
        let lo = decode_nibble(bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn decode_nibble(b: u8) -> Result<u8, &'static str> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Err("uppercase hex not allowed"),
        _ => Err("invalid hex digit"),
    }
}
