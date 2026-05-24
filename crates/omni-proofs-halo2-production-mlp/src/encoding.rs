//! Stage 11d.2 — canonical byte encodings for the production MLP's
//! 16-input / 8-output tensors.
//!
//! **Encoding contract (frozen):** each tensor is a slice of i16
//! serialized as little-endian i16 with no header, no length prefix,
//! no padding. Input: 16 × 2 = 32 bytes. Output: 8 × 2 = 16 bytes.
//!
//! These bytes are what get BLAKE3'd into the artifact's `input_hash`
//! / `response_hash` fields and (per the Stage 11d.2 plan §1) into a
//! synthetic `InferenceAttestationDigest` for the optional R9
//! chain-digest-roundtrip test — without modifying any chain wire
//! surface.

use thiserror::Error;

pub type CanonicalInput = [i16; 16];
pub type CanonicalOutput = [i16; 8];

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorDecodeError {
    #[error("expected exactly {expected} bytes ({n} × i16 little-endian), got {got}")]
    WrongLength { expected: usize, n: usize, got: usize },
}

pub fn encode_canonical_input(input: &CanonicalInput) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    for v in input {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn encode_canonical_output(output: &CanonicalOutput) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    for v in output {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn decode_canonical_input(bytes: &[u8]) -> Result<CanonicalInput, TensorDecodeError> {
    if bytes.len() != 32 {
        return Err(TensorDecodeError::WrongLength { expected: 32, n: 16, got: bytes.len() });
    }
    let mut out = [0i16; 16];
    for i in 0..16 {
        out[i] = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
    }
    Ok(out)
}

pub fn decode_canonical_output(bytes: &[u8]) -> Result<CanonicalOutput, TensorDecodeError> {
    if bytes.len() != 16 {
        return Err(TensorDecodeError::WrongLength { expected: 16, n: 8, got: bytes.len() });
    }
    let mut out = [0i16; 8];
    for i in 0..8 {
        out[i] = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{CANONICAL_INPUT, CANONICAL_OUTPUT};

    #[test]
    fn encode_input_is_32_bytes_little_endian() {
        let bytes = encode_canonical_input(&CANONICAL_INPUT);
        assert_eq!(bytes.len(), 32);
        // Spot-check: CANONICAL_INPUT[0] = -5 → 0xFB 0xFF (LE i16)
        assert_eq!(bytes[0], 0xFB);
        assert_eq!(bytes[1], 0xFF);
    }

    #[test]
    fn encode_output_is_16_bytes_little_endian() {
        let bytes = encode_canonical_output(&CANONICAL_OUTPUT);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn encode_decode_round_trip() {
        let input = CANONICAL_INPUT;
        let back = decode_canonical_input(&encode_canonical_input(&input)).unwrap();
        assert_eq!(input, back);
        let output = CANONICAL_OUTPUT;
        let back2 = decode_canonical_output(&encode_canonical_output(&output)).unwrap();
        assert_eq!(output, back2);
    }

    #[test]
    fn decode_rejects_wrong_length() {
        assert!(decode_canonical_input(&[0u8; 31]).is_err());
        assert!(decode_canonical_input(&[0u8; 33]).is_err());
        assert!(decode_canonical_output(&[0u8; 15]).is_err());
        assert!(decode_canonical_output(&[0u8; 17]).is_err());
    }
}
