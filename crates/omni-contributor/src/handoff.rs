//! Stage 12.4 — live activation / tensor handoff between adjacent
//! contributors in a Stage 12.3 session.
//!
//! Sends opaque tensor bytes peer-to-peer (via omni-net's existing
//! request/response transport) while SNIP keeps its role as durable
//! audit / fallback storage. This is the v1 protocol envelope; the
//! transport itself lives in [`crate::tensor_transport`], and the
//! receive-side verifier lives in [`crate::handoff_verify`].
//!
//! ## Scope
//!
//! - Linear pipeline only (`to.stage_index == from.stage_index + 1`).
//! - Sender + tensor-content integrity (signature binds `tensor_hash`,
//!   `byte_len`, chunk metadata, and the from/to assignments).
//! - Chunking: a single `ActivationHandoff` envelope carries one
//!   chunk; the receiver concatenates all `chunk_count` chunks in
//!   `chunk_index` order, BLAKE3s the result, and rejects on hash
//!   or length mismatch.
//! - Opaque bytes — no model-specific tensor semantics.
//! - AttestationOnly. No proof claim about the activation's
//!   semantic correctness.
//!
//! ## Posture (unchanged)
//!
//! No chain wire / Stage 7b tx / SUM Chain RPC / payment / marketplace
//! / exclusive claim / proof / on-chain verification. SNIP only for
//! durable storage / fallback.

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;
use crate::job::{check_blake3_hex, check_iso_8601, check_pubkey_hex, check_signature_hex};

/// Pinned at 1. Closed-enum extensions or field reorders are
/// `schema_version: 2` migrations.
pub const HANDOFF_SCHEMA_VERSION: u32 = 1;

/// Maximum bytes for a single chunk's `tensor_chunk_bytes`. Defended
/// in `validate_schema`; receivers MAY additionally cap below this.
pub const HANDOFF_CHUNK_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum `chunk_count` per handoff. At the default
/// `HANDOFF_CHUNK_MAX_BYTES`, this caps total tensor size at
/// 64 MiB × 256 = 16 GiB.
pub const HANDOFF_CHUNK_COUNT_MAX: u32 = 256;

/// Maximum `shape.len()`. Stops absurd-rank tensors from blowing up
/// canonical bytes / signing input.
pub const HANDOFF_SHAPE_RANK_MAX: usize = 8;

/// Maximum total `byte_len`. Mirrors `HANDOFF_CHUNK_MAX_BYTES *
/// HANDOFF_CHUNK_COUNT_MAX`.
pub const HANDOFF_BYTE_LEN_MAX: u64 = HANDOFF_CHUNK_MAX_BYTES * HANDOFF_CHUNK_COUNT_MAX as u64;

/// Closed enum for tensor element dtype. Mirrors `omni-net`'s
/// `TensorRequest.dtype` u8 discriminant but typed at the
/// `omni-contributor` boundary. Frozen at v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorDtype {
    F16,
    Bf16,
    F32,
}

impl TensorDtype {
    /// Bytes per element. Used to range-check `shape.product() *
    /// bytes_per_element == byte_len`.
    pub fn bytes_per_element(self) -> u64 {
        match self {
            TensorDtype::F16 | TensorDtype::Bf16 => 2,
            TensorDtype::F32 => 4,
        }
    }
}

/// Stage 12.4 activation handoff envelope. Signed by the *from*
/// contributor. One envelope carries one chunk; multi-chunk
/// handoffs share `(session_id, from_assignment_id, to_assignment_id)`
/// and use `chunk_index` / `chunk_count` to order + bound reassembly.
///
/// `tensor_chunk_bytes` is **excluded** from the canonical signing
/// body — the signature binds `tensor_hash` instead. The receiver
/// is required to re-hash reassembled bytes and reject on mismatch
/// (`handoff_verify::process_activation_handoff`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationHandoff {
    pub schema_version: u32,

    /// 64-char lowercase hex — copy of `ExecutionSession.session_id`.
    pub session_id: String,

    /// 64-char lowercase hex.
    pub from_assignment_id: String,
    /// 64-char lowercase hex.
    pub to_assignment_id: String,

    /// 64-char lowercase hex Ed25519 pubkey of the sender.
    pub from_contributor_pubkey_hex: String,
    /// 64-char lowercase hex Ed25519 pubkey of the intended receiver.
    pub to_contributor_pubkey_hex: String,

    pub dtype: TensorDtype,

    /// Tensor shape. Non-empty; bounded by [`HANDOFF_SHAPE_RANK_MAX`].
    /// Element values must be > 0.
    pub shape: Vec<u64>,

    /// Total length (in bytes) of the *reassembled* tensor across all
    /// `chunk_count` chunks. Bounded by [`HANDOFF_BYTE_LEN_MAX`].
    pub byte_len: u64,

    /// 64-char lowercase hex BLAKE3 of the reassembled tensor bytes.
    /// The sender signs this hash; the receiver re-hashes after
    /// reassembly and rejects on mismatch.
    pub tensor_hash: String,

    /// Zero-based chunk position. `< chunk_count`.
    pub chunk_index: u32,
    /// Total chunks (>= 1). Bounded by [`HANDOFF_CHUNK_COUNT_MAX`].
    pub chunk_count: u32,

    /// RFC 3339 UTC (`Z` suffix).
    pub produced_at_utc: String,

    /// THIS chunk's bytes. Bounded by [`HANDOFF_CHUNK_MAX_BYTES`].
    /// **Excluded from the canonical signing body** — the signature
    /// binds `tensor_hash` (full reassembled) instead.
    pub tensor_chunk_bytes: Vec<u8>,

    /// 128-char lowercase hex Ed25519 signature over the canonical
    /// body (which excludes `tensor_chunk_bytes` and this field).
    pub sender_signature_hex: String,
}

impl ActivationHandoff {
    pub fn validate_schema(&self) -> Result<(), SchemaError> {
        if self.schema_version != HANDOFF_SCHEMA_VERSION {
            return Err(SchemaError::UnsupportedVersion {
                got: self.schema_version,
            });
        }
        check_blake3_hex("session_id", &self.session_id)?;
        check_blake3_hex("from_assignment_id", &self.from_assignment_id)?;
        check_blake3_hex("to_assignment_id", &self.to_assignment_id)?;
        check_pubkey_hex("from_contributor_pubkey_hex", &self.from_contributor_pubkey_hex)?;
        check_pubkey_hex("to_contributor_pubkey_hex", &self.to_contributor_pubkey_hex)?;
        check_blake3_hex("tensor_hash", &self.tensor_hash)?;
        check_iso_8601("produced_at_utc", &self.produced_at_utc)?;
        check_signature_hex("sender_signature_hex", &self.sender_signature_hex)?;

        // Shape: non-empty, bounded rank, every dim > 0.
        if self.shape.is_empty() {
            return Err(SchemaError::HandoffShapeEmpty);
        }
        if self.shape.len() > HANDOFF_SHAPE_RANK_MAX {
            return Err(SchemaError::HandoffShapeRankTooLarge {
                got: self.shape.len(),
                max: HANDOFF_SHAPE_RANK_MAX,
            });
        }
        if self.shape.iter().any(|d| *d == 0) {
            return Err(SchemaError::HandoffShapeZeroDim);
        }

        // byte_len bounds + dtype consistency.
        if self.byte_len == 0 {
            return Err(SchemaError::HandoffByteLenZero);
        }
        if self.byte_len > HANDOFF_BYTE_LEN_MAX {
            return Err(SchemaError::HandoffByteLenTooLarge {
                got: self.byte_len,
                max: HANDOFF_BYTE_LEN_MAX,
            });
        }
        let element_count: Option<u64> = self
            .shape
            .iter()
            .copied()
            .try_fold(1u64, |acc, d| acc.checked_mul(d));
        let expected_byte_len = element_count
            .and_then(|n| n.checked_mul(self.dtype.bytes_per_element()));
        match expected_byte_len {
            Some(expected) if expected == self.byte_len => {}
            _ => {
                return Err(SchemaError::HandoffByteLenMismatch {
                    declared: self.byte_len,
                    expected_from_shape_and_dtype: expected_byte_len.unwrap_or(0),
                });
            }
        }

        // Chunk fields.
        if self.chunk_count == 0 {
            return Err(SchemaError::HandoffChunkCountZero);
        }
        if self.chunk_count > HANDOFF_CHUNK_COUNT_MAX {
            return Err(SchemaError::HandoffChunkCountTooLarge {
                got: self.chunk_count,
                max: HANDOFF_CHUNK_COUNT_MAX,
            });
        }
        if self.chunk_index >= self.chunk_count {
            return Err(SchemaError::HandoffChunkIndexOutOfRange {
                index: self.chunk_index,
                count: self.chunk_count,
            });
        }

        // tensor_chunk_bytes bound.
        let chunk_len = self.tensor_chunk_bytes.len() as u64;
        if chunk_len == 0 {
            return Err(SchemaError::HandoffChunkBytesEmpty);
        }
        if chunk_len > HANDOFF_CHUNK_MAX_BYTES {
            return Err(SchemaError::HandoffChunkBytesTooLarge {
                got: chunk_len,
                max: HANDOFF_CHUNK_MAX_BYTES,
            });
        }
        // A single-chunk handoff's tensor_chunk_bytes must equal byte_len.
        if self.chunk_count == 1 && chunk_len != self.byte_len {
            return Err(SchemaError::HandoffSingleChunkLenMismatch {
                chunk_len,
                byte_len: self.byte_len,
            });
        }
        Ok(())
    }
}
