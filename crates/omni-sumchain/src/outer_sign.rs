//! Phase 5 Stage 7b — outer `SignedTransaction` construction.
//!
//! Wraps the call into `sumchain-crypto` for the outer Ed25519
//! signature. Encapsulated here so Stage 7b's [`crate::tx`] focuses on
//! the OmniNode-side construction and the chain-side primitive
//! interaction is in one place.
//!
//! **Outer signing input is the 32-byte BLAKE3 hash of the bincode-1.3
//! serialisation of `TransactionV2`**, computed by the chain's
//! `TransactionV2::signing_hash()`. This is intentionally different
//! from Stage 6's inner pipeline (which signs raw
//! `DOMAIN_TAG || canonical_digest_bytes`); the chain's outer rule is
//! "sign the hash", and replay protection is provided by the
//! `chain_id` field of `TransactionV2` rather than by a separate
//! domain tag.

use omni_zkml::ChainClientError;
use sumchain_crypto::{sign, PrivateKey};
use sumchain_primitives::{SignedTransaction, TransactionV2};
// `Address::from_public_key` lives at the crate root via `pub use`,
// and `Hash` similarly; `TransactionV2::signing_hash` returns a
// `Hash` which carries `as_bytes() -> &[u8; 32]`.

/// Sign a `TransactionV2` with the operator's Ed25519 seed and produce
/// the assembled `SignedTransaction` ready for hex-encode + submit.
///
/// Flow (matches the chain's canonical convention):
/// 1. `outer_hash = TransactionV2::signing_hash()` —
///    `BLAKE3(bincode_1_3(tx))`, returns a 32-byte `Hash`.
/// 2. `outer_signature = sumchain_crypto::sign(outer_hash.as_bytes(), &PrivateKey::from_bytes(seed))`
///    — Ed25519 signature over the 32-byte hash.
/// 3. `pubkey = PrivateKey::from_bytes(seed).public_key()` — raw 32-byte
///    public key for the `SignedTransaction.public_key` field.
/// 4. `SignedTransaction::new_v2(tx, outer_signature_bytes, pubkey_bytes)`.
///
/// Returns a typed `ChainClientError` on the (practically unreachable)
/// path where converting from the chain's `Signature` newtype to raw
/// `[u8; 64]` somehow fails.
pub(crate) fn outer_sign_transaction_v2(
    seed: &[u8; 32],
    tx: &TransactionV2,
) -> std::result::Result<SignedTransaction, ChainClientError> {
    let private_key = PrivateKey::from_bytes(*seed);
    let public_key = private_key.public_key();

    let signing_hash = tx.signing_hash();
    let signature = sign(signing_hash.as_bytes(), &private_key);

    Ok(SignedTransaction::new_v2(
        tx.clone(),
        signature.to_bytes(),
        *public_key.as_bytes(),
    ))
}
