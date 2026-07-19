//! Merkle length-ambiguity + slot / object-kind substitution tests, over the
//! adopted B0 rules and the retained host-side witness/proof helpers.

use r0_zkvm_bench::blake3_32;
use r0_zkvm_bench::manifest::{output_slot, OutputManifestV1, SlotDescriptorV1, SlotKind};
use r0_zkvm_bench::merkle::{
    check_length_commitment, chunk_count_checked, merkle_root_from_leaves, verify_chunk_witness,
    ChunkWitness, MerkleTree, WitnessError, CHUNK,
};
use r0_zkvm_bench::object::{ObjectCommitmentV1, ObjectKind};
use r0_zkvm_bench::DecodeError;
use r0_zkvm_bench::FROZEN_MAX_OBJECT_BYTES;

/// Build `full_chunks` real 1 MiB chunks plus a short final chunk of `tail` bytes.
fn object_bytes(full_chunks: usize, tail: usize) -> Vec<u8> {
    let total = full_chunks * CHUNK + tail;
    (0..total).map(|i| (i % 251) as u8).collect()
}

#[test]
fn three_leaf_and_duplicated_fourth_leaf_share_a_root() {
    // Roots alone are length-ambiguous under the B0 odd-duplication rule; the
    // ObjectCommitmentV1 length commitment is what makes them distinguishable.
    let a = blake3_32(b"a");
    let b = blake3_32(b"b");
    let c = blake3_32(b"c");
    assert_eq!(
        merkle_root_from_leaves(&[a, b, c]),
        merkle_root_from_leaves(&[a, b, c, c])
    );
}

#[test]
fn declared_chunk_count_must_match_byte_len() {
    let data = object_bytes(2, 500); // 3 chunks
    let byte_len = data.len() as u64;
    assert_eq!(chunk_count_checked(byte_len).unwrap(), 3);
    assert_eq!(
        check_length_commitment(byte_len, 4),
        Err(WitnessError::ChunkCountMismatch {
            declared: 4,
            expected: 3
        })
    );
}

#[test]
fn same_root_different_byte_len_is_caught_by_final_chunk_length() {
    let data = object_bytes(2, 100);
    let byte_len = data.len() as u64;
    let chunk_count = chunk_count_checked(byte_len).unwrap(); // 3
    let tree = MerkleTree::from_object_bytes(&data);
    let root = tree.root();

    let last_idx = chunk_count - 1;
    let last_chunk: Vec<u8> = data.chunks(CHUNK).last().unwrap().to_vec();
    let witness = ChunkWitness {
        index: last_idx,
        data: last_chunk,
    };
    let path = tree.proof(last_idx);

    assert_eq!(
        verify_chunk_witness(&witness, byte_len, chunk_count, &root, &path),
        Ok(())
    );

    // A lie about byte_len (still 3 chunks: 2 full + 200 tail) makes the final
    // chunk's required length 200, but the presented chunk is 100 bytes.
    let lying_byte_len = 2 * CHUNK as u64 + 200;
    assert_eq!(chunk_count_checked(lying_byte_len).unwrap(), 3);
    assert_eq!(
        verify_chunk_witness(&witness, lying_byte_len, chunk_count, &root, &path),
        Err(WitnessError::WrongChunkLen {
            index: 2,
            got: 100,
            expected: 200
        })
    );
}

#[test]
fn wrong_leaf_index_rejected() {
    let data = object_bytes(3, 0); // 3 full chunks
    let byte_len = data.len() as u64;
    let chunk_count = chunk_count_checked(byte_len).unwrap();
    let tree = MerkleTree::from_object_bytes(&data);
    let root = tree.root();
    let chunk0: Vec<u8> = data.chunks(CHUNK).next().unwrap().to_vec();
    let witness = ChunkWitness {
        index: 1,
        data: chunk0,
    };
    let path = tree.proof(1);
    assert_eq!(
        verify_chunk_witness(&witness, byte_len, chunk_count, &root, &path),
        Err(WitnessError::RootMismatch)
    );
}

#[test]
fn out_of_range_index_rejected() {
    let data = object_bytes(2, 10); // 3 chunks
    let byte_len = data.len() as u64;
    let chunk_count = chunk_count_checked(byte_len).unwrap();
    let tree = MerkleTree::from_object_bytes(&data);
    let root = tree.root();
    let chunk0: Vec<u8> = data.chunks(CHUNK).next().unwrap().to_vec();
    // Index 3 does not exist in a 3-chunk object (the duplicated-4th attack).
    let witness = ChunkWitness {
        index: 3,
        data: chunk0,
    };
    let path = tree.proof(0);
    assert_eq!(
        verify_chunk_witness(&witness, byte_len, chunk_count, &root, &path),
        Err(WitnessError::IndexOutOfRange {
            index: 3,
            chunk_count: 3
        })
    );
}

#[test]
fn wrong_path_depth_rejected() {
    let data = object_bytes(3, 0); // 3 chunks → depth 2
    let byte_len = data.len() as u64;
    let chunk_count = chunk_count_checked(byte_len).unwrap();
    let tree = MerkleTree::from_object_bytes(&data);
    let root = tree.root();
    let chunk0: Vec<u8> = data.chunks(CHUNK).next().unwrap().to_vec();
    let witness = ChunkWitness {
        index: 0,
        data: chunk0,
    };
    let mut path = tree.proof(0);
    path.push([0u8; 32]); // one element too many
    assert!(matches!(
        verify_chunk_witness(&witness, byte_len, chunk_count, &root, &path),
        Err(WitnessError::WrongPathDepth { .. })
    ));
}

#[test]
fn declared_byte_len_over_max_rejected_before_allocation() {
    assert_eq!(
        check_length_commitment(FROZEN_MAX_OBJECT_BYTES + 1, u32::MAX),
        Err(WitnessError::ByteLenExceedsMax {
            byte_len: FROZEN_MAX_OBJECT_BYTES + 1,
            max: FROZEN_MAX_OBJECT_BYTES,
        })
    );
    let witness = ChunkWitness {
        index: 0,
        data: vec![],
    };
    assert!(matches!(
        verify_chunk_witness(
            &witness,
            FROZEN_MAX_OBJECT_BYTES + 1,
            u32::MAX,
            &[0u8; 32],
            &[]
        ),
        Err(WitnessError::ByteLenExceedsMax { .. })
    ));
}

#[test]
fn empty_object_has_no_chunk_witnesses() {
    let witness = ChunkWitness {
        index: 0,
        data: vec![],
    };
    assert_eq!(
        verify_chunk_witness(&witness, 0, 0, &[0u8; 32], &[]),
        Err(WitnessError::EmptyObjectHasNoChunks)
    );
}

// ── Manifest slot-reorder & object-kind / embedded-commitment substitution ───

#[test]
fn manifest_slot_reorder_rejected() {
    let m = OutputManifestV1 {
        slots: vec![
            output_slot(SlotKind::KvCache, 0, b"k").unwrap(),
            output_slot(SlotKind::ResidualStream, 0, b"r").unwrap(),
        ],
    };
    // Descending (kv=1 before residual=0) is rejected by the fallible encode.
    assert!(matches!(
        m.try_encode(),
        Err(DecodeError::NonCanonicalOrder { .. })
    ));
}

#[test]
fn object_kind_substitution_changes_commitment_identity() {
    let data = b"payload";
    let as_model = ObjectCommitmentV1::commit(ObjectKind::Model, data).unwrap();
    let as_token_seq = ObjectCommitmentV1::commit(ObjectKind::TokenSeq, data).unwrap();
    // Same bytes/root but different kind → different identity.
    assert_eq!(as_model.merkle_root(), as_token_seq.merkle_root());
    assert_ne!(as_model.identity(), as_token_seq.identity());
}

#[test]
fn wrong_object_kind_in_manifest_slot_rejected() {
    // A ResidualStream slot whose embedded commitment is KvState is rejected.
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
    // The raw bytes are likewise rejected by the strict decoder. Byte-patch a
    // VALID ResidualStream/ResidualState manifest so the embedded commitment's
    // object_kind reads KvState(7) instead of ResidualState(6), leaving all other
    // invariants (chunk_count/root) consistent — the decoder still rejects the
    // slot-kind ↔ object-kind mismatch.
    let good = OutputManifestV1 {
        slots: vec![output_slot(SlotKind::ResidualStream, 0, b"r").unwrap()],
    };
    let mut bytes = good.try_encode().unwrap();
    // header(38) + slot_kind(1) + slot_index(4) + commitment tag(32) + schema(2)
    // = 77 → the embedded object_kind u16 lives at bytes [77..79].
    assert_eq!(ObjectKind::KvState.to_repr(), 7);
    bytes[77..79].copy_from_slice(&ObjectKind::KvState.to_repr().to_le_bytes());
    assert!(matches!(
        OutputManifestV1::decode_exact(&bytes),
        Err(DecodeError::Inconsistent {
            ctx: "SlotDescriptorV1.object_kind"
        })
    ));
}
