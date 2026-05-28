//! Stage 12.4 — receive-side verifier + chunk reassembler.
//!
//! Two layers:
//!
//!   - [`verify_activation_handoff`] verifies a single envelope's
//!     schema, sender signature, and structural binding to a 12.3
//!     session + from/to assignments. It does NOT touch
//!     `tensor_chunk_bytes` beyond schema bounds — content integrity
//!     is the reassembler's job.
//!
//!   - [`HandoffReceiver`] accumulates chunks for a given handoff
//!     identity (`session_id` + `from_assignment_id` + `to_assignment_id`),
//!     deduplicates `chunk_index`, and once `chunk_count` chunks
//!     have arrived, reassembles, BLAKE3s, and emits the full
//!     reconstructed tensor bytes — or a typed rejection if hash or
//!     length disagree.
//!
//! Outer `omni-net::TensorRequest` fields are advisory routing
//! hints. The verifier here NEVER trusts them over the signed inner
//! envelope; callers are expected to pass the inner `ActivationHandoff`
//! decoded from `TensorRequest.data`.

use std::collections::HashMap;

use crate::canonical::activation_handoff_signing_input;
use crate::canonical::hex_lower;
use crate::handoff::{ActivationHandoff, TensorDtype};
use crate::session::{ExecutionSession, WorkAssignment};
use crate::signing::verify_signature_hex;

/// Per-envelope verification outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandoffVerifyOutcome {
    Ok,
    SchemaMalformed(String),
    SessionIdMismatch,
    FromAssignmentMismatch,
    ToAssignmentMismatch,
    /// `to.stage_index != from.stage_index + 1`. v1 enforces strict
    /// linear pipeline; non-adjacent / branching topologies are
    /// Stage 12.5+.
    StageOrderInvalid {
        from_stage_index: u32,
        to_stage_index: u32,
    },
    FromContributorMismatch,
    ToContributorMismatch,
    /// Sender signature did not verify against
    /// `from_contributor_pubkey_hex`.
    SenderSignatureFailed,
    /// `produced_at_utc >= session.expires_at_utc`.
    SessionExpired { produced_at: String, expires_at: String },
}

impl HandoffVerifyOutcome {
    pub fn is_ok(&self) -> bool {
        matches!(self, HandoffVerifyOutcome::Ok)
    }
}

/// Verify a single `ActivationHandoff` envelope against a verified
/// 12.3 session + the from/to assignments.
///
/// The caller is responsible for ensuring the supplied assignments
/// are themselves valid against the session — this verifier focuses
/// on the handoff-specific binding.
pub fn verify_activation_handoff(
    session: &ExecutionSession,
    from_assignment: &WorkAssignment,
    to_assignment: &WorkAssignment,
    handoff: &ActivationHandoff,
) -> HandoffVerifyOutcome {
    if let Err(e) = handoff.validate_schema() {
        return HandoffVerifyOutcome::SchemaMalformed(e.to_string());
    }
    if handoff.session_id != session.session_id {
        return HandoffVerifyOutcome::SessionIdMismatch;
    }
    if handoff.from_assignment_id != from_assignment.assignment_id {
        return HandoffVerifyOutcome::FromAssignmentMismatch;
    }
    if handoff.to_assignment_id != to_assignment.assignment_id {
        return HandoffVerifyOutcome::ToAssignmentMismatch;
    }
    // v1: strict linear pipeline. to == from + 1.
    if to_assignment.stage_index != from_assignment.stage_index.wrapping_add(1) {
        return HandoffVerifyOutcome::StageOrderInvalid {
            from_stage_index: from_assignment.stage_index,
            to_stage_index: to_assignment.stage_index,
        };
    }
    if handoff.from_contributor_pubkey_hex != from_assignment.contributor_pubkey_hex {
        return HandoffVerifyOutcome::FromContributorMismatch;
    }
    if handoff.to_contributor_pubkey_hex != to_assignment.contributor_pubkey_hex {
        return HandoffVerifyOutcome::ToContributorMismatch;
    }
    // Session-time bound. RFC 3339 Z lex-compare (same precision).
    if handoff.produced_at_utc >= session.expires_at_utc {
        return HandoffVerifyOutcome::SessionExpired {
            produced_at: handoff.produced_at_utc.clone(),
            expires_at: session.expires_at_utc.clone(),
        };
    }
    // Sender signature.
    let signing_input = match activation_handoff_signing_input(handoff) {
        Ok(b) => b,
        Err(e) => return HandoffVerifyOutcome::SchemaMalformed(e.to_string()),
    };
    let ok = verify_signature_hex(
        &handoff.from_contributor_pubkey_hex,
        &signing_input,
        &handoff.sender_signature_hex,
    )
    .unwrap_or(false);
    if !ok {
        return HandoffVerifyOutcome::SenderSignatureFailed;
    }
    HandoffVerifyOutcome::Ok
}

// ── Reassembler ──────────────────────────────────────────────────────────

/// Identity of a handoff stream (one session / one pipe between two
/// adjacent assignments). Multiple chunks share this identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HandoffStreamKey {
    pub session_id: String,
    pub from_assignment_id: String,
    pub to_assignment_id: String,
}

impl HandoffStreamKey {
    pub fn from_envelope(h: &ActivationHandoff) -> Self {
        Self {
            session_id: h.session_id.clone(),
            from_assignment_id: h.from_assignment_id.clone(),
            to_assignment_id: h.to_assignment_id.clone(),
        }
    }
}

/// Outcome of feeding a single chunk into the reassembler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkOutcome {
    /// Chunk accepted; more chunks expected.
    Accepted,
    /// All `chunk_count` chunks have arrived AND reconstructed BLAKE3
    /// equals the signed `tensor_hash` AND total length equals
    /// `byte_len`. The full tensor bytes are returned.
    Complete { tensor_bytes: Vec<u8> },
    /// Duplicate `chunk_index` for the same stream (idempotent
    /// reject; the receiver should NOT update its state).
    DuplicateChunkIndex { chunk_index: u32 },
    /// A chunk's metadata disagreed with the first chunk on the
    /// stream (`chunk_count`, `byte_len`, `tensor_hash`, `dtype`,
    /// `shape`, `from_contributor_pubkey_hex`,
    /// `to_contributor_pubkey_hex`). The receiver should drop the
    /// whole stream.
    StreamMetadataDrift { field: &'static str },
    /// All chunks arrived but the reconstructed BLAKE3 did not equal
    /// the signed `tensor_hash`. The receiver should drop the
    /// whole stream.
    TensorHashMismatch { expected: String, recomputed: String },
    /// All chunks arrived but total length disagrees with the
    /// signed `byte_len`.
    ByteLenMismatch { expected: u64, recomputed: u64 },
}

#[derive(Debug, Clone)]
struct ReassemblyState {
    /// Frozen at first-chunk arrival.
    chunk_count: u32,
    byte_len: u64,
    tensor_hash: String,
    dtype: TensorDtype,
    shape: Vec<u64>,
    from_contributor_pubkey_hex: String,
    to_contributor_pubkey_hex: String,
    /// `chunks[i]` is `Some` once chunk `i` has been accepted.
    chunks: Vec<Option<Vec<u8>>>,
    received_count: u32,
}

/// In-memory chunk reassembler. Stateful; one instance per
/// receiver process. Holds at most one in-flight stream per
/// `HandoffStreamKey`; emitting `Complete` removes the entry.
#[derive(Debug, Default)]
pub struct HandoffReceiver {
    streams: HashMap<HandoffStreamKey, ReassemblyState>,
}

impl HandoffReceiver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk into the reassembler. Caller must have already
    /// run [`verify_activation_handoff`] on the envelope.
    ///
    /// `Complete` clears the stream's state; further chunks on the
    /// same key after `Complete` will start a new stream entry.
    /// `TensorHashMismatch` / `ByteLenMismatch` / `StreamMetadataDrift`
    /// also clear the stream — the receiver is expected to log + drop.
    pub fn feed(&mut self, h: &ActivationHandoff) -> ChunkOutcome {
        let key = HandoffStreamKey::from_envelope(h);
        if self.streams.contains_key(&key) {
            // Take ownership so we can mutate freely + remove without
            // holding two mutable borrows of `self.streams`.
            let mut state = self.streams.remove(&key).unwrap();
            // Drift-check against the frozen first-chunk values.
            if state.chunk_count != h.chunk_count {
                return ChunkOutcome::StreamMetadataDrift { field: "chunk_count" };
            }
            if state.byte_len != h.byte_len {
                return ChunkOutcome::StreamMetadataDrift { field: "byte_len" };
            }
            if state.tensor_hash != h.tensor_hash {
                return ChunkOutcome::StreamMetadataDrift { field: "tensor_hash" };
            }
            if state.dtype != h.dtype {
                return ChunkOutcome::StreamMetadataDrift { field: "dtype" };
            }
            if state.shape != h.shape {
                return ChunkOutcome::StreamMetadataDrift { field: "shape" };
            }
            if state.from_contributor_pubkey_hex != h.from_contributor_pubkey_hex {
                return ChunkOutcome::StreamMetadataDrift {
                    field: "from_contributor_pubkey_hex",
                };
            }
            if state.to_contributor_pubkey_hex != h.to_contributor_pubkey_hex {
                return ChunkOutcome::StreamMetadataDrift {
                    field: "to_contributor_pubkey_hex",
                };
            }
            let idx = h.chunk_index as usize;
            if state.chunks[idx].is_some() {
                // Re-insert; duplicate is idempotent reject, not stream
                // destruction.
                self.streams.insert(key, state);
                return ChunkOutcome::DuplicateChunkIndex {
                    chunk_index: h.chunk_index,
                };
            }
            state.chunks[idx] = Some(h.tensor_chunk_bytes.clone());
            state.received_count += 1;
            if state.received_count == state.chunk_count {
                // Reassemble.
                let mut buf: Vec<u8> = Vec::with_capacity(state.byte_len as usize);
                for slot in state.chunks.iter() {
                    // Every slot must be Some at this point; received_count == chunk_count.
                    buf.extend_from_slice(slot.as_ref().unwrap());
                }
                let recomputed_len = buf.len() as u64;
                if recomputed_len != state.byte_len {
                    return ChunkOutcome::ByteLenMismatch {
                        expected: state.byte_len,
                        recomputed: recomputed_len,
                    };
                }
                let recomputed_hash = hex_lower(blake3::hash(&buf).as_bytes());
                if recomputed_hash != state.tensor_hash {
                    return ChunkOutcome::TensorHashMismatch {
                        expected: state.tensor_hash,
                        recomputed: recomputed_hash,
                    };
                }
                return ChunkOutcome::Complete { tensor_bytes: buf };
            }
            // Still in-flight — put the state back.
            self.streams.insert(key, state);
            ChunkOutcome::Accepted
        } else {
            // First chunk on this stream — bring up state.
            let mut chunks: Vec<Option<Vec<u8>>> =
                (0..h.chunk_count).map(|_| None).collect();
            chunks[h.chunk_index as usize] = Some(h.tensor_chunk_bytes.clone());
            let state = ReassemblyState {
                chunk_count: h.chunk_count,
                byte_len: h.byte_len,
                tensor_hash: h.tensor_hash.clone(),
                dtype: h.dtype,
                shape: h.shape.clone(),
                from_contributor_pubkey_hex: h.from_contributor_pubkey_hex.clone(),
                to_contributor_pubkey_hex: h.to_contributor_pubkey_hex.clone(),
                chunks,
                received_count: 1,
            };
            if h.chunk_count == 1 {
                // Single-chunk fast path: the schema validator
                // already enforced `tensor_chunk_bytes.len() == byte_len`,
                // but we still hash-check to bind the bytes to the
                // signed `tensor_hash`.
                let buf = h.tensor_chunk_bytes.clone();
                let recomputed_hash = hex_lower(blake3::hash(&buf).as_bytes());
                if recomputed_hash != state.tensor_hash {
                    return ChunkOutcome::TensorHashMismatch {
                        expected: state.tensor_hash,
                        recomputed: recomputed_hash,
                    };
                }
                return ChunkOutcome::Complete { tensor_bytes: buf };
            }
            self.streams.insert(key, state);
            ChunkOutcome::Accepted
        }
    }

    /// Number of streams currently in flight. Used by tests.
    pub fn pending_streams(&self) -> usize {
        self.streams.len()
    }
}
