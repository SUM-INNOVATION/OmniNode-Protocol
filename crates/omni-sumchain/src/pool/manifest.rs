//! O2 / O4 (#104, #106): a unit's output manifest, and root-checked retrieval.
//!
//! * **One manifest per unit.** A transformer layer-group unit emits exactly one
//!   output manifest with two slots at the unit's position: its residual stream
//!   and its KV cache, each committed over the frozen workload encoding. This is
//!   the binding the B0 `TransformerLayerGroup` statement enforces. A
//!   SelectToken unit's output manifest is the canonical empty manifest.
//! * **Observe only on a root match.** [`verify_retrieved_output`] is the client
//!   side of #106's "retrieval verified by root recompute (not activation)".
//!   Retrieved bytes count only if the manifest recomputes to the committed
//!   identity, and every committed slot recomputes to its committed commitment.
//!   A missing, extra, duplicated or altered slot is refused, and nothing is
//!   inferred from the object being registered or activated in SNIP.

use sumchain_wire::b0::codec::DecodeError;
use sumchain_wire::b0::enums::{ObjectKind, SlotKind};
use sumchain_wire::b0::manifest::{OutputManifestV1, SlotDescriptorV1};
use sumchain_wire::b0::object_commitment::ObjectCommitmentV1;
use sumchain_wire::b0::workload::{KvPair, kv_state_bytes, residual_state_bytes};

/// A unit's output: its manifest, the manifest's canonical bytes and
/// commitment, and every slot object it commits to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitOutput {
    pub manifest: OutputManifestV1,
    pub manifest_bytes: Vec<u8>,
    pub manifest_commitment: ObjectCommitmentV1,
    /// `(slot kind, slot index, object bytes)`, in manifest order.
    pub objects: Vec<(SlotKind, u32, Vec<u8>)>,
}

/// The one output manifest of a transformer layer-group unit at `position`.
pub fn layer_group_output(
    position: u32,
    residual: &[i16; 8],
    kv: &[KvPair],
) -> Result<UnitOutput, DecodeError> {
    let residual_bytes = residual_state_bytes(residual);
    let kv_bytes = kv_state_bytes(kv);
    let manifest = OutputManifestV1 {
        slots: vec![
            SlotDescriptorV1 {
                slot_kind: SlotKind::ResidualStream,
                slot_index: position,
                commitment: ObjectCommitmentV1::commit(ObjectKind::ResidualState, &residual_bytes)?,
            },
            SlotDescriptorV1 {
                slot_kind: SlotKind::KvCache,
                slot_index: position,
                commitment: ObjectCommitmentV1::commit(ObjectKind::KvState, &kv_bytes)?,
            },
        ],
    };
    let manifest_bytes = manifest.try_encode()?;
    let manifest_commitment = manifest.try_commitment()?;
    Ok(UnitOutput {
        manifest,
        manifest_bytes,
        manifest_commitment,
        objects: vec![
            (SlotKind::ResidualStream, position, residual_bytes),
            (SlotKind::KvCache, position, kv_bytes),
        ],
    })
}

/// The canonical empty output manifest a SelectToken unit commits to.
pub fn select_token_output() -> Result<UnitOutput, DecodeError> {
    let manifest = OutputManifestV1 { slots: vec![] };
    Ok(UnitOutput {
        manifest_bytes: manifest.try_encode()?,
        manifest_commitment: manifest.try_commitment()?,
        manifest,
        objects: vec![],
    })
}

/// Why retrieved output does not prove the committed manifest.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RetrievalError {
    #[error("retrieved manifest bytes do not recompute to the committed manifest")]
    ManifestCommitment,
    #[error("committed manifest does not decode: {0}")]
    ManifestDecode(DecodeError),
    #[error("committed slot {kind:?}#{index} was not retrieved")]
    MissingSlot { kind: SlotKind, index: u32 },
    #[error("slot {kind:?}#{index} was retrieved more than once")]
    DuplicateSlot { kind: SlotKind, index: u32 },
    #[error("slot {kind:?}#{index} is not in the committed manifest")]
    UnexpectedSlot { kind: SlotKind, index: u32 },
    #[error("slot {kind:?}#{index} bytes do not recompute to its committed commitment")]
    SlotMismatch { kind: SlotKind, index: u32 },
    #[error("slot {kind:?}#{index} could not be committed: {source}")]
    SlotCommit {
        kind: SlotKind,
        index: u32,
        source: DecodeError,
    },
}

/// Verify retrieved output against the manifest commitment the chain holds.
///
/// `committed` is the manifest commitment bound on chain, `manifest_bytes` the
/// retrieved manifest object, and `retrieved` every retrieved slot object.
/// Returns the decoded manifest only if every check passes. Nothing is
/// partially accepted.
pub fn verify_retrieved_output(
    committed: &ObjectCommitmentV1,
    manifest_bytes: &[u8],
    retrieved: &[(SlotKind, u32, Vec<u8>)],
) -> Result<OutputManifestV1, RetrievalError> {
    let recomputed = ObjectCommitmentV1::commit(ObjectKind::OutputManifest, manifest_bytes)
        .map_err(RetrievalError::ManifestDecode)?;
    if &recomputed != committed {
        return Err(RetrievalError::ManifestCommitment);
    }
    let manifest =
        OutputManifestV1::decode_exact(manifest_bytes).map_err(RetrievalError::ManifestDecode)?;

    for (i, (kind, index, _)) in retrieved.iter().enumerate() {
        if retrieved[..i]
            .iter()
            .any(|(k, j, _)| k == kind && j == index)
        {
            return Err(RetrievalError::DuplicateSlot {
                kind: *kind,
                index: *index,
            });
        }
        if !manifest
            .slots
            .iter()
            .any(|s| s.slot_kind == *kind && s.slot_index == *index)
        {
            return Err(RetrievalError::UnexpectedSlot {
                kind: *kind,
                index: *index,
            });
        }
    }
    for slot in &manifest.slots {
        let (kind, index) = (slot.slot_kind, slot.slot_index);
        let bytes = retrieved
            .iter()
            .find(|(k, j, _)| *k == kind && *j == index)
            .map(|(_, _, b)| b)
            .ok_or(RetrievalError::MissingSlot { kind, index })?;
        let c = ObjectCommitmentV1::commit(kind.object_kind(), bytes).map_err(|source| {
            RetrievalError::SlotCommit {
                kind,
                index,
                source,
            }
        })?;
        if c != slot.commitment {
            return Err(RetrievalError::SlotMismatch { kind, index });
        }
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn output() -> UnitOutput {
        let kv: Vec<KvPair> = (0..3).map(|i| ([i; 8], [-i; 8])).collect();
        layer_group_output(5, &[1, -2, 3, -4, 5, -6, 7, -8], &kv).unwrap()
    }

    #[test]
    fn a_layer_group_unit_has_exactly_its_two_slots_at_its_position() {
        let out = output();
        let slots: Vec<(SlotKind, u32)> = out
            .manifest
            .slots
            .iter()
            .map(|s| (s.slot_kind, s.slot_index))
            .collect();
        assert_eq!(
            slots,
            vec![(SlotKind::ResidualStream, 5), (SlotKind::KvCache, 5)]
        );
        // The manifest bytes are canonical: they decode and re-encode to themselves.
        let back = OutputManifestV1::decode_exact(&out.manifest_bytes).unwrap();
        assert_eq!(back, out.manifest);
        assert_eq!(back.try_encode().unwrap(), out.manifest_bytes);
    }

    #[test]
    fn retrieved_output_that_matches_every_root_verifies() {
        let out = output();
        let got =
            verify_retrieved_output(&out.manifest_commitment, &out.manifest_bytes, &out.objects)
                .unwrap();
        assert_eq!(got, out.manifest);
        // Order of retrieval does not matter.
        let mut reversed = out.objects.clone();
        reversed.reverse();
        verify_retrieved_output(&out.manifest_commitment, &out.manifest_bytes, &reversed).unwrap();
    }

    #[test]
    fn any_altered_byte_in_any_slot_is_refused() {
        let out = output();
        for slot in 0..out.objects.len() {
            for byte in 0..out.objects[slot].2.len() {
                let mut objects = out.objects.clone();
                objects[slot].2[byte] ^= 0x01;
                let (kind, index) = (objects[slot].0, objects[slot].1);
                assert_eq!(
                    verify_retrieved_output(
                        &out.manifest_commitment,
                        &out.manifest_bytes,
                        &objects
                    ),
                    Err(RetrievalError::SlotMismatch { kind, index }),
                    "slot {slot} byte {byte}"
                );
            }
        }
    }

    #[test]
    fn a_missing_extra_or_duplicated_slot_is_refused() {
        let out = output();
        let c = &out.manifest_commitment;
        let m = &out.manifest_bytes;

        assert_eq!(
            verify_retrieved_output(c, m, &out.objects[..1]),
            Err(RetrievalError::MissingSlot {
                kind: SlotKind::KvCache,
                index: 5
            })
        );
        let mut extra = out.objects.clone();
        extra.push((SlotKind::KvCache, 6, vec![0; 32]));
        assert_eq!(
            verify_retrieved_output(c, m, &extra),
            Err(RetrievalError::UnexpectedSlot {
                kind: SlotKind::KvCache,
                index: 6
            })
        );
        let mut dup = out.objects.clone();
        dup.push(out.objects[0].clone());
        assert_eq!(
            verify_retrieved_output(c, m, &dup),
            Err(RetrievalError::DuplicateSlot {
                kind: SlotKind::ResidualStream,
                index: 5
            })
        );
    }

    #[test]
    fn a_manifest_that_is_not_the_committed_one_is_refused() {
        let out = output();
        let other = layer_group_output(6, &[0; 8], &[]).unwrap();
        // The right bytes against another unit's commitment.
        assert_eq!(
            verify_retrieved_output(
                &other.manifest_commitment,
                &out.manifest_bytes,
                &out.objects
            ),
            Err(RetrievalError::ManifestCommitment)
        );
        // Altered manifest bytes against the right commitment.
        let mut bytes = out.manifest_bytes.clone();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x80;
        assert_eq!(
            verify_retrieved_output(&out.manifest_commitment, &bytes, &out.objects),
            Err(RetrievalError::ManifestCommitment)
        );
    }

    #[test]
    fn the_select_token_manifest_is_the_canonical_empty_one() {
        let out = select_token_output().unwrap();
        assert!(out.manifest.slots.is_empty());
        assert!(out.objects.is_empty());
        verify_retrieved_output(&out.manifest_commitment, &out.manifest_bytes, &[]).unwrap();
    }
}
