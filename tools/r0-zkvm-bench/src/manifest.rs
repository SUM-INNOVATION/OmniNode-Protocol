//! Output/input manifests — the adopted B0 [`OutputManifestV1`] /
//! [`InputManifestV1`] and their 85-byte slot descriptors (38-byte header).
//!
//! The bespoke OmniNode manifest / `SlotKind` / `OMNINODE.R0.OUTMANIFEST.V1`
//! domain are **deleted**. Slots are strictly ascending by `(slot_kind,
//! slot_index)`, unique, and each embedded [`ObjectCommitmentV1`] must carry the
//! object kind the slot kind maps to (`ResidualStream → ResidualState`,
//! `KvCache → KvState` for outputs; `PriorResidual/PriorKv/TokenPrefix` for
//! inputs). Canonical bytes come only through the fallible `try_encode` /
//! `try_commitment` routes, so a decoder-invalid manifest can never be hashed.
//!
//! The [`output_slot`] / [`input_slot`] helpers build a descriptor whose embedded
//! commitment is committed under exactly the slot kind's required object kind,
//! making the slot-kind ↔ object-kind relationship correct by construction.

pub use sumchain_wire::b0::enums::{InputSlotKind, SlotKind};
pub use sumchain_wire::b0::manifest::{
    InputManifestV1, InputSlotDescriptorV1, OutputManifestV1, SlotDescriptorV1,
    MANIFEST_HEADER_LEN, SLOT_DESCRIPTOR_LEN,
};

use crate::b0::codec::DecodeError;
use crate::object::ObjectCommitmentV1;

/// Build an output-manifest slot by committing `data` under the object kind the
/// `slot_kind` requires (`ResidualStream → ResidualState`, `KvCache → KvState`).
pub fn output_slot(
    slot_kind: SlotKind,
    slot_index: u32,
    data: &[u8],
) -> Result<SlotDescriptorV1, DecodeError> {
    Ok(SlotDescriptorV1 {
        slot_kind,
        slot_index,
        commitment: ObjectCommitmentV1::commit(slot_kind.object_kind(), data)?,
    })
}

/// Build an input-manifest slot by committing `data` under the object kind the
/// `slot_kind` requires (`PriorResidual`/`PriorKv`/`TokenPrefix`).
pub fn input_slot(
    slot_kind: InputSlotKind,
    slot_index: u32,
    data: &[u8],
) -> Result<InputSlotDescriptorV1, DecodeError> {
    Ok(InputSlotDescriptorV1 {
        slot_kind,
        slot_index,
        commitment: ObjectCommitmentV1::commit(slot_kind.object_kind(), data)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::ObjectKind;

    #[test]
    fn header_and_descriptor_sizes() {
        assert_eq!(MANIFEST_HEADER_LEN, 38);
        assert_eq!(SLOT_DESCRIPTOR_LEN, 85);
        let m = OutputManifestV1 {
            slots: vec![
                output_slot(SlotKind::ResidualStream, 0, b"r").unwrap(),
                output_slot(SlotKind::KvCache, 0, b"k").unwrap(),
            ],
        };
        // 38 header + 2 * 85 descriptors.
        assert_eq!(m.try_encode().unwrap().len(), 38 + 2 * 85);
    }

    #[test]
    fn output_manifest_roundtrips_and_orders() {
        let m = OutputManifestV1 {
            slots: vec![
                output_slot(SlotKind::ResidualStream, 0, b"resid").unwrap(),
                output_slot(SlotKind::KvCache, 0, b"kv").unwrap(),
            ],
        };
        let bytes = m.try_encode().unwrap();
        assert_eq!(OutputManifestV1::decode_exact(&bytes).unwrap(), m);
        // Each slot embeds its own object kind.
        assert_eq!(
            m.slots[0].commitment.object_kind(),
            ObjectKind::ResidualState
        );
        assert_eq!(m.slots[1].commitment.object_kind(), ObjectKind::KvState);
    }

    #[test]
    fn descending_slots_rejected() {
        let m = OutputManifestV1 {
            slots: vec![
                output_slot(SlotKind::KvCache, 0, b"k").unwrap(),
                output_slot(SlotKind::ResidualStream, 0, b"r").unwrap(),
            ],
        };
        assert!(matches!(
            m.try_encode(),
            Err(DecodeError::NonCanonicalOrder { .. })
        ));
    }

    #[test]
    fn wrong_slot_object_kind_rejected() {
        // A ResidualStream slot embedding a KvState commitment.
        let bad = SlotDescriptorV1 {
            slot_kind: SlotKind::ResidualStream,
            slot_index: 0,
            commitment: ObjectCommitmentV1::commit(ObjectKind::KvState, b"x").unwrap(),
        };
        let m = OutputManifestV1 { slots: vec![bad] };
        assert!(matches!(
            m.try_encode(),
            Err(DecodeError::Inconsistent {
                ctx: "SlotDescriptorV1.object_kind"
            })
        ));
    }

    #[test]
    fn input_manifest_three_slots_roundtrips() {
        let m = InputManifestV1 {
            slots: vec![
                input_slot(InputSlotKind::PriorResidual, 0, b"a").unwrap(),
                input_slot(InputSlotKind::PriorKv, 0, b"b").unwrap(),
                input_slot(InputSlotKind::TokenPrefix, 0, b"c").unwrap(),
            ],
        };
        assert_eq!(m.try_encode().unwrap().len(), 38 + 3 * 85);
        assert_eq!(
            InputManifestV1::decode_exact(&m.try_encode().unwrap()).unwrap(),
            m
        );
    }
}
