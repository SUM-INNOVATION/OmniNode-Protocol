//! The candidate-neutral statement — adopted B0 `R0ComputationStatementV2`
//! (**996 bytes**), only ever built as a zero-spec-hash **template**.
//!
//! The 1066-byte OmniNode `R0ComputationStatementV1` and its
//! `OMNINODE.R0.STATEMENT.V1` domain are **deleted** — this crate has a single
//! statement format. The statement is candidate-neutral: it carries no
//! `candidate_id`, vkey, or image-id (those live only in
//! [`crate::envelope::ProductionProofEnvelopeV1`]).
//!
//! ## Public boundary: no raw statement is re-exposed
//!
//! The raw B0 statement type is imported **crate-privately** and is deliberately
//! NOT part of this crate's public API — no public function, method, field,
//! alias, or re-export mentions it. The only public statement surface is
//! [`SyntheticJournal`], which exposes read-only inspection plus the
//! zero-template-only [`SyntheticJournal::from_template_bytes`] ingestion path.
//! This crate therefore re-exposes no finalization capability (no
//! `materialize_final`, no raw `Writer`, no constructible/encodable raw
//! statement). The following does not compile:
//!
//! ```compile_fail
//! use r0_zkvm_bench::statement::R0ComputationStatementV2;
//! ```
//!
//! Callers that genuinely need to build/mutate a raw statement may depend on
//! `sumchain-wire` directly (the dependency inherently exposes the type) and then
//! return through this zero-template-only boundary.
//!
//! ## The synthetic journal (never a final statement)
//!
//! A journal is exactly **996 template bytes** with `b0_pre_spec_hash` (bytes
//! 34..66) zeroed. Those bytes ARE the synthetic public output; `journal.len() ==
//! 996`, and its `computation_statement_hash` is `BLAKE3` recomputed over exactly
//! those bytes. No code path calls B0's `materialize_final` or writes a non-zero
//! spec hash, so a *final* (selection-relevant) statement is never produced here.

pub use sumchain_wire::b0::derived_input::DerivedInputV1;
pub use sumchain_wire::b0::enums::UnitKind;
// Crate-private: the raw statement type is available to this crate's internals
// (the reference executor / fixtures build it) but is NOT part of the public API.
pub(crate) use sumchain_wire::b0::statement::R0ComputationStatementV2;

use crate::b0::codec::DecodeError;
use crate::b0::statement as b0_statement;

/// The synthetic public output (journal): exactly 996 zero-spec-hash template
/// bytes, plus the recomputed `computation_statement_hash`.
///
/// The public API exposes only read-only inspection and the zero-template-only
/// [`from_template_bytes`](Self::from_template_bytes) ingestion path; there is no
/// public route to a constructible/encodable raw statement or a materialized/
/// final statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntheticJournal {
    /// Exactly 996 bytes, `b0_pre_spec_hash` zeroed.
    template_bytes: Vec<u8>,
}

impl SyntheticJournal {
    /// Build the journal from an in-crate statement by producing its zero-spec-
    /// hash template (validates every embedded commitment's object kind first).
    /// Always yields a template — never a final statement. **Crate-private**: it
    /// mentions the raw statement type, so it is not part of the public API.
    pub(crate) fn from_statement_template(
        statement: R0ComputationStatementV2,
    ) -> Result<Self, DecodeError> {
        let template_bytes = b0_statement::template_bytes(statement)?;
        debug_assert_eq!(template_bytes.len(), R0ComputationStatementV2::LEN);
        Ok(Self { template_bytes })
    }

    /// Ingest externally-produced template bytes through the zero-template-only
    /// boundary. This is the ONLY public byte-ingestion route, and it accepts a
    /// **template** only — never a materialized/final statement. It requires:
    ///
    /// * exactly 996 bytes;
    /// * an all-zero `b0_pre_spec_hash` field (bytes 34..66) — a nonzero spec hash
    ///   (i.e. a materialized/final statement) is rejected;
    /// * a strict canonical decode with NO trailing bytes and valid embedded
    ///   commitments / object kinds;
    /// * that canonical re-encoding reproduces the supplied bytes (noncanonical
    ///   input is rejected).
    pub fn from_template_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() != R0ComputationStatementV2::LEN {
            return Err(DecodeError::BadValue {
                ctx: "SyntheticJournal.template_len",
            });
        }
        // Reject any nonzero spec hash: a template's spec-hash field is all zero,
        // so this refuses materialized/final statement bytes up front.
        if bytes[R0ComputationStatementV2::SPEC_HASH_RANGE]
            .iter()
            .any(|&b| b != 0)
        {
            return Err(DecodeError::BadValue {
                ctx: "SyntheticJournal.nonzero_spec_hash",
            });
        }
        // Strict decode (rejects bad tags/scalars/kinds, lying chunk counts, and
        // trailing bytes), then require canonical re-encoding to match exactly.
        let decoded = R0ComputationStatementV2::decode_exact(bytes)?;
        let reencoded = decoded.try_encode()?;
        if reencoded.as_slice() != bytes {
            return Err(DecodeError::BadValue {
                ctx: "SyntheticJournal.noncanonical",
            });
        }
        Ok(Self {
            template_bytes: bytes.to_vec(),
        })
    }

    /// The exact 996 template bytes — the synthetic journal / public output.
    pub fn bytes(&self) -> &[u8] {
        &self.template_bytes
    }

    /// Journal length — always exactly 996.
    pub fn len(&self) -> usize {
        self.template_bytes.len()
    }

    /// Always `false` (the journal is a fixed 996 bytes).
    pub fn is_empty(&self) -> bool {
        self.template_bytes.is_empty()
    }

    /// The statement's `b0_pre_spec_hash` (bytes 34..66) — all zero for a
    /// template. Bound by the envelope / partial proof.
    pub fn spec_hash(&self) -> [u8; 32] {
        let mut h = [0u8; 32];
        h.copy_from_slice(&self.template_bytes[R0ComputationStatementV2::SPEC_HASH_RANGE]);
        h
    }

    /// `computation_statement_hash = BLAKE3(the exact 996 template bytes)`.
    pub fn computation_statement_hash(&self) -> [u8; 32] {
        R0ComputationStatementV2::computation_statement_hash(&self.template_bytes)
    }

    /// Round-trip the journal bytes back through the strict B0 decoder.
    /// **Crate-private**: it returns the raw statement type, so it is not part of
    /// the public API.
    #[cfg(test)]
    pub(crate) fn decode(&self) -> Result<R0ComputationStatementV2, DecodeError> {
        R0ComputationStatementV2::decode_exact(&self.template_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture;

    /// Absolute byte offset of the embedded `model_commitment.chunk_count` field:
    /// statement header up to the model commitment = 236, plus the commitment's
    /// own `chunk_count` offset 44.
    const MODEL_COMMITMENT_CHUNK_COUNT_LSB: usize = 236 + 44;

    #[test]
    fn journal_is_996_byte_zero_spec_hash_template() {
        let j = fixture::journal().unwrap();
        assert_eq!(j.len(), 996);
        // Template discipline: the spec-hash field is all zero.
        assert_eq!(j.spec_hash(), [0u8; 32]);
        assert!(j.bytes()[34..66].iter().all(|&b| b == 0));
    }

    #[test]
    fn statement_hash_is_blake3_of_the_exact_996_bytes() {
        let j = fixture::journal().unwrap();
        assert_eq!(
            j.computation_statement_hash(),
            *blake3::hash(j.bytes()).as_bytes()
        );
    }

    // ── from_template_bytes ingestion boundary ────────────────────────────────

    #[test]
    fn from_template_bytes_accepts_valid_zero_spec_template() {
        let j = fixture::journal().unwrap();
        let round = SyntheticJournal::from_template_bytes(j.bytes()).unwrap();
        assert_eq!(round, j);
        assert_eq!(round.len(), 996);
        assert_eq!(round.spec_hash(), [0u8; 32]);
        assert_eq!(
            round.computation_statement_hash(),
            j.computation_statement_hash()
        );
    }

    #[test]
    fn from_template_bytes_rejects_nonzero_spec_hash() {
        let j = fixture::journal().unwrap();
        let mut bytes = j.bytes().to_vec();
        bytes[34] = 0x01; // first byte of the b0_pre_spec_hash field (34..66)
        assert!(matches!(
            SyntheticJournal::from_template_bytes(&bytes),
            Err(DecodeError::BadValue {
                ctx: "SyntheticJournal.nonzero_spec_hash"
            })
        ));
    }

    #[test]
    fn from_template_bytes_rejects_wrong_length() {
        let j = fixture::journal().unwrap();
        let mut short = j.bytes().to_vec();
        short.pop(); // 995 bytes (truncated)
        assert!(matches!(
            SyntheticJournal::from_template_bytes(&short),
            Err(DecodeError::BadValue {
                ctx: "SyntheticJournal.template_len"
            })
        ));
        let mut long = j.bytes().to_vec();
        long.push(0); // 997 bytes (trailing)
        assert!(matches!(
            SyntheticJournal::from_template_bytes(&long),
            Err(DecodeError::BadValue {
                ctx: "SyntheticJournal.template_len"
            })
        ));
    }

    #[test]
    fn from_template_bytes_rejects_noncanonical_embedded_commitment() {
        let j = fixture::journal().unwrap();
        let mut bytes = j.bytes().to_vec();
        // Corrupt the embedded model_commitment's chunk_count so it no longer
        // equals ceil(byte_len / CHUNK): the strict decoder rejects the lie.
        bytes[MODEL_COMMITMENT_CHUNK_COUNT_LSB] =
            bytes[MODEL_COMMITMENT_CHUNK_COUNT_LSB].wrapping_add(1);
        assert!(matches!(
            SyntheticJournal::from_template_bytes(&bytes),
            Err(DecodeError::Inconsistent {
                ctx: "ObjectCommitmentV1.chunk_count"
            })
        ));
    }

    #[test]
    fn decode_roundtrips_through_the_template_route() {
        let j = fixture::journal().unwrap();
        let s = j.decode().unwrap();
        assert_eq!(
            SyntheticJournal::from_statement_template(s)
                .unwrap()
                .bytes(),
            j.bytes()
        );
    }
}
