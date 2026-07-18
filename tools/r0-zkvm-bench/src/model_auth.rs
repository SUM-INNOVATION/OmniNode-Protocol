//! Model authentication — **Approach 2**, re-pointed at the B0 commitment
//! accessors.
//!
//! The public B0 [`ObjectCommitmentV1`] `model_commitment` binds the *complete*
//! model's `byte_len` / `chunk_count` / Merkle root over the model's canonical
//! serialization (see [`crate::workload::Model::canonical_bytes`]). Every weight
//! chunk the (modeled) guest consumes must be authenticated against that
//! commitment with an **index-aware Merkle path** via
//! [`crate::merkle::verify_chunk_witness`].
//!
//! This authentication cost belongs to **guest measurement**: the host never
//! re-hashes the model on the critical path. The functions here are written as
//! the guest would run them (they take a witness + path and verify), so a
//! benchmark could attribute their cost to the guest. The host-side helper
//! [`prepare_weight_witnesses`] exists only to *produce* the witnesses+paths a
//! prover would feed the guest; it is explicitly not part of the measured guest
//! work.

use crate::b0::codec::DecodeError;
use crate::merkle::{verify_chunk_witness, ChunkWitness, MerkleTree, WitnessError, CHUNK};
use crate::object::ObjectCommitmentV1;
use crate::workload::Model;

/// Authenticate a single weight chunk against the public model commitment,
/// exactly as the guest would (index-aware path, position-fixed chunk length,
/// allocation bound). Reads `byte_len` / `chunk_count` / `merkle_root` through
/// the B0 accessor methods.
pub fn authenticate_weight_chunk(
    model_commitment: &ObjectCommitmentV1,
    witness: &ChunkWitness,
    path: &[[u8; 32]],
) -> Result<(), WitnessError> {
    verify_chunk_witness(
        witness,
        model_commitment.byte_len(),
        model_commitment.chunk_count(),
        &model_commitment.merkle_root(),
        path,
    )
}

/// A weight chunk plus its authentication path, as handed to the guest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedWeightChunk {
    pub witness: ChunkWitness,
    pub path: Vec<[u8; 32]>,
}

/// **Host-side / prover-side only** — produce the witnesses+paths for every model
/// chunk. Not part of guest measurement. Propagates the fallible B0 `commit`.
pub fn prepare_weight_witnesses(
    model: &Model,
) -> Result<(ObjectCommitmentV1, Vec<AuthenticatedWeightChunk>), DecodeError> {
    let bytes = model.canonical_bytes();
    let commitment = model.commitment()?;
    let tree = MerkleTree::from_object_bytes(&bytes);
    let chunks: Vec<AuthenticatedWeightChunk> = bytes
        .chunks(CHUNK)
        .enumerate()
        .map(|(i, chunk)| AuthenticatedWeightChunk {
            witness: ChunkWitness {
                index: i as u32,
                data: chunk.to_vec(),
            },
            path: tree.proof(i as u32),
        })
        .collect();
    Ok((commitment, chunks))
}

/// Authenticate *every* consumed weight chunk (guest-modeled loop). Returns the
/// number of chunks authenticated, or the first failure.
pub fn authenticate_all(
    model_commitment: &ObjectCommitmentV1,
    chunks: &[AuthenticatedWeightChunk],
) -> Result<usize, WitnessError> {
    for c in chunks {
        authenticate_weight_chunk(model_commitment, &c.witness, &c.path)?;
    }
    Ok(chunks.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blake3_32;
    use crate::object::ObjectKind;

    fn model() -> Model {
        Model::deterministic(7, blake3_32(b"auth-model"))
    }

    #[test]
    fn every_weight_chunk_authenticates() {
        let m = model();
        let (commitment, chunks) = prepare_weight_witnesses(&m).unwrap();
        assert_eq!(authenticate_all(&commitment, &chunks), Ok(chunks.len()));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn tampered_weight_chunk_fails_authentication() {
        let m = model();
        let (commitment, mut chunks) = prepare_weight_witnesses(&m).unwrap();
        chunks[0].witness.data[0] ^= 0x01; // flip one byte of the weights
        assert_eq!(
            authenticate_all(&commitment, &chunks),
            Err(WitnessError::RootMismatch)
        );
    }

    #[test]
    fn multi_chunk_model_paths_are_index_aware() {
        // A multi-chunk object with DISTINCT content per chunk.
        let bytes: Vec<u8> = (0..(CHUNK * 3 + 123)).map(|i| (i % 251) as u8).collect();
        let commitment = ObjectCommitmentV1::commit(ObjectKind::Model, &bytes).unwrap();
        assert_eq!(commitment.chunk_count(), 4);
        let tree = MerkleTree::from_object_bytes(&bytes);
        for (i, chunk) in bytes.chunks(CHUNK).enumerate() {
            let w = ChunkWitness {
                index: i as u32,
                data: chunk.to_vec(),
            };
            let path = tree.proof(i as u32);
            assert_eq!(authenticate_weight_chunk(&commitment, &w, &path), Ok(()));
            // Replaying this chunk's data at the next index must fail.
            if i + 1 < commitment.chunk_count() as usize {
                let wrong = ChunkWitness {
                    index: (i as u32) + 1,
                    data: chunk.to_vec(),
                };
                let wrong_path = tree.proof((i as u32) + 1);
                assert!(authenticate_weight_chunk(&commitment, &wrong, &wrong_path).is_err());
            }
        }
    }
}
