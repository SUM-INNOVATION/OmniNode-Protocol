//! Object commitments — the adopted B0 [`ObjectCommitmentV1`] (80 bytes).
//!
//! The bespoke OmniNode `ObjectCommitmentV1` / `ObjectKind` codec and the
//! `OMNINODE.R0.OBJCOMMIT.V1` domain are **deleted**; every object field now uses
//! the frozen `sumchain_wire::b0` type. Fields are private and the only
//! constructors are the checked [`ObjectCommitmentV1::commit`] /
//! [`ObjectCommitmentV1::empty`] / decoders, so a value can never hold
//! decoder-inconsistent state. The leading domain is the frozen
//! `SUMCHAIN/R0/OBJECT/v1` tag and `object_kind` maps through the frozen
//! [`ObjectKind`] enum, in which `Tokenizer = 2` and `Slot = 8` are RESERVED and
//! rejected on decode.

pub use sumchain_wire::b0::enums::ObjectKind;
pub use sumchain_wire::b0::object_commitment::ObjectCommitmentV1;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::b0::codec::DecodeError;
    use crate::merkle::CHUNK;

    #[test]
    fn canonical_len_is_80_and_roundtrips() {
        let c = ObjectCommitmentV1::commit(ObjectKind::Model, b"weights").unwrap();
        assert_eq!(c.encode().len(), 80);
        assert_eq!(ObjectCommitmentV1::decode_exact(&c.encode()).unwrap(), c);
    }

    #[test]
    fn empty_object_is_canonical_empty_descriptor() {
        let e = ObjectCommitmentV1::empty(ObjectKind::TokenPrefix);
        assert_eq!(e.byte_len(), 0);
        assert_eq!(e.chunk_count(), 0);
        assert_eq!(e.merkle_root(), [0u8; 32]);
        // commit(kind, &[]) equals empty(kind).
        assert_eq!(
            ObjectCommitmentV1::commit(ObjectKind::TokenPrefix, &[]).unwrap(),
            e
        );
    }

    #[test]
    fn object_kind_substitution_changes_identity() {
        let a = ObjectCommitmentV1::commit(ObjectKind::Model, b"payload").unwrap();
        let b = ObjectCommitmentV1::commit(ObjectKind::TokenSeq, b"payload").unwrap();
        assert_eq!(a.merkle_root(), b.merkle_root()); // same bytes/root
        assert_ne!(a.identity(), b.identity()); // different kind → different identity
    }

    #[test]
    fn reserved_tokenizer_and_slot_kinds_are_rejected_on_decode() {
        // The reserved kinds cannot even be named as a variant; feed them as raw
        // bytes and confirm the decoder rejects them distinctly.
        let mut bytes = ObjectCommitmentV1::commit(ObjectKind::Model, b"x")
            .unwrap()
            .encode();
        bytes[34..36].copy_from_slice(&2u16.to_le_bytes()); // Tokenizer(2)
        assert!(matches!(
            ObjectCommitmentV1::decode_exact(&bytes),
            Err(DecodeError::ReservedEnum {
                name: "ObjectKind",
                value: 2
            })
        ));
        bytes[34..36].copy_from_slice(&8u16.to_le_bytes()); // Slot(8)
        assert!(matches!(
            ObjectCommitmentV1::decode_exact(&bytes),
            Err(DecodeError::ReservedEnum {
                name: "ObjectKind",
                value: 8
            })
        ));
    }

    #[test]
    fn single_chunk_root_is_leaf_hash() {
        let data = vec![0xabu8; CHUNK];
        let c = ObjectCommitmentV1::commit(ObjectKind::Model, &data).unwrap();
        assert_eq!(c.chunk_count(), 1);
        assert_eq!(c.merkle_root(), *blake3::hash(&data).as_bytes());
    }
}
