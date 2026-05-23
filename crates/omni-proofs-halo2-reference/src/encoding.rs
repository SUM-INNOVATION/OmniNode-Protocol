//! Stage 11b.1.a — canonical byte encodings for the bounded MLP's
//! input and output tensors.
//!
//! **Encoding contract (frozen):** each tensor is a `[i16; 4]`
//! serialized as **little-endian i16** with **no header, no length
//! prefix, no padding**. Exactly 8 bytes per tensor (`4 × 2`).
//!
//! These bytes are what get BLAKE3'd into the universal
//! `input_hash` / `response_hash` fields on a `ProofArtifactBody`'s
//! metadata. Every framework manifest commits to the same byte
//! shape; the cross-framework equivalence test re-derives the
//! hashes from the manifest's integer values and asserts they match
//! the universal hashes.

use thiserror::Error;

use crate::canonical::{CanonicalInput, CanonicalOutput};

/// Errors decoding a `[u8; 8]` back into a `[i16; 4]`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorDecodeError {
    #[error("expected exactly 8 bytes (4 × i16 little-endian), got {0}")]
    WrongLength(usize),
}

/// Encode a 4-element i16 tensor as 8 bytes, little-endian per
/// element, no header. Returns owned `Vec<u8>` (always length 8).
pub fn encode_tensor_4xi16_le(tensor: &[i16; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8);
    for v in tensor {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Canonical encoder for the input tensor (alias for clarity).
pub fn encode_canonical_input(input: &CanonicalInput) -> Vec<u8> {
    encode_tensor_4xi16_le(input)
}

/// Canonical encoder for the output tensor (alias for clarity).
pub fn encode_canonical_output(output: &CanonicalOutput) -> Vec<u8> {
    encode_tensor_4xi16_le(output)
}

/// Decode 8 bytes (little-endian i16 × 4) back into a `[i16; 4]`.
/// Returns a typed error on wrong length.
pub fn decode_tensor_4xi16_le(bytes: &[u8]) -> Result<[i16; 4], TensorDecodeError> {
    if bytes.len() != 8 {
        return Err(TensorDecodeError::WrongLength(bytes.len()));
    }
    let mut out = [0i16; 4];
    for i in 0..4 {
        let lo = bytes[i * 2];
        let hi = bytes[i * 2 + 1];
        out[i] = i16::from_le_bytes([lo, hi]);
    }
    Ok(out)
}

pub fn decode_canonical_input(bytes: &[u8]) -> Result<CanonicalInput, TensorDecodeError> {
    decode_tensor_4xi16_le(bytes)
}

pub fn decode_canonical_output(bytes: &[u8]) -> Result<CanonicalOutput, TensorDecodeError> {
    decode_tensor_4xi16_le(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{CANONICAL_INPUT, CANONICAL_OUTPUT};

    #[test]
    fn encode_canonical_input_is_8_bytes_little_endian() {
        let bytes = encode_canonical_input(&CANONICAL_INPUT);
        assert_eq!(bytes.len(), 8);
        // Spot-check: CANONICAL_INPUT[0] = -5 → 0xFB 0xFF (LE i16).
        assert_eq!(bytes[0], 0xFB);
        assert_eq!(bytes[1], 0xFF);
        // CANONICAL_INPUT[3] = -100 → 0x9C 0xFF (LE i16).
        assert_eq!(bytes[6], 0x9C);
        assert_eq!(bytes[7], 0xFF);
    }

    #[test]
    fn encode_canonical_output_is_8_bytes_little_endian() {
        let bytes = encode_canonical_output(&CANONICAL_OUTPUT);
        assert_eq!(bytes.len(), 8);
        // CANONICAL_OUTPUT[0] = 33 → 0x21 0x00 (LE i16).
        assert_eq!(bytes[0], 0x21);
        assert_eq!(bytes[1], 0x00);
        // CANONICAL_OUTPUT[1] = -32 → 0xE0 0xFF (LE i16).
        assert_eq!(bytes[2], 0xE0);
        assert_eq!(bytes[3], 0xFF);
        // CANONICAL_OUTPUT[3] = 7 → 0x07 0x00 (LE i16).
        assert_eq!(bytes[6], 0x07);
        assert_eq!(bytes[7], 0x00);
    }

    #[test]
    fn encode_decode_round_trip() {
        let input = CANONICAL_INPUT;
        let bytes = encode_canonical_input(&input);
        let back = decode_canonical_input(&bytes).unwrap();
        assert_eq!(input, back);

        let output = CANONICAL_OUTPUT;
        let bytes = encode_canonical_output(&output);
        let back = decode_canonical_output(&bytes).unwrap();
        assert_eq!(output, back);
    }

    #[test]
    fn decode_rejects_wrong_length() {
        assert!(matches!(
            decode_tensor_4xi16_le(&[1, 2, 3]),
            Err(TensorDecodeError::WrongLength(3)),
        ));
        assert!(matches!(
            decode_tensor_4xi16_le(&[1; 9]),
            Err(TensorDecodeError::WrongLength(9)),
        ));
    }

    #[test]
    fn encoding_is_deterministic() {
        let a = encode_canonical_input(&CANONICAL_INPUT);
        let b = encode_canonical_input(&CANONICAL_INPUT);
        assert_eq!(a, b);
    }
}
