//! SNIP Merkle: **root and chunk-count rules adopted from B0**, host-side
//! **witness/proof** generation and length-unambiguous verification RETAINED.
//!
//! The commitment rules (`leaf = BLAKE3(chunk)`, `parent = BLAKE3(l‖r)`, odd
//! level duplicates the last node, single leaf → root is the leaf, empty →
//! `chunk_count = 0` / zero root, `chunk_count = ceil(byte_len / CHUNK)`) come
//! from [`sumchain_wire::b0::merkle`]; [`merkle_root`], [`merkle_root_from_leaves`],
//! [`chunk_count_checked`], and [`CHUNK`] are re-exported from there, so this
//! crate has no second copy of the rules. A parity test pins
//! `MerkleTree::root() == merkle_root(data)`.
//!
//! What is RETAINED here is the tooling B0 does not provide: an in-memory tree
//! that yields **index-aware inclusion proofs**, and the guest-side, length-
//! unambiguous **chunk-witness** verifier. A byte length + root are not by
//! themselves a length commitment (a 3-leaf tree and a 4-leaf tree whose 4th leaf
//! duplicates the 3rd share a root); B0's `ObjectCommitmentV1` closes that by
//! binding `byte_len` and `chunk_count`, and [`verify_chunk_witness`] enforces the
//! same invariants on each streamed chunk before trusting it.

pub use sumchain_wire::b0::merkle::{
    chunk_count_checked, merkle_root, merkle_root_from_leaves, CHUNK,
};

use crate::blake3_32;
use crate::FROZEN_MAX_OBJECT_BYTES;

/// A binary Merkle tree over 1 MiB `BLAKE3` chunk-leaf hashes, built to produce
/// index-aware inclusion proofs. `levels[0]` holds the leaf hashes;
/// `levels[last]` holds `[root]`. Its root is byte-identical to
/// [`merkle_root`] (asserted by a parity test).
pub struct MerkleTree {
    levels: Vec<Vec<[u8; 32]>>,
}

/// Combine two 32-byte child hashes: `BLAKE3(left ‖ right)`.
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(left);
    h.update(right);
    *h.finalize().as_bytes()
}

impl MerkleTree {
    /// Build from pre-hashed leaves, duplicating the last node on odd levels.
    pub fn build(leaf_hashes: &[[u8; 32]]) -> Self {
        if leaf_hashes.is_empty() {
            return Self { levels: vec![] };
        }
        if leaf_hashes.len() == 1 {
            return Self {
                levels: vec![leaf_hashes.to_vec()],
            };
        }
        let mut levels = vec![leaf_hashes.to_vec()];
        let mut current = leaf_hashes.to_vec();
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len().div_ceil(2));
            for pair in current.chunks(2) {
                let left = &pair[0];
                let right = pair.get(1).unwrap_or(left); // duplicate last if odd
                next.push(hash_pair(left, right));
            }
            levels.push(next.clone());
            current = next;
        }
        Self { levels }
    }

    /// Build directly from raw object bytes by chunking into 1 MiB leaves.
    pub fn from_object_bytes(data: &[u8]) -> Self {
        let leaves: Vec<[u8; 32]> = if data.is_empty() {
            Vec::new()
        } else {
            data.chunks(CHUNK).map(blake3_32).collect()
        };
        Self::build(&leaves)
    }

    /// The Merkle root, or `[0u8; 32]` for the empty tree (matches [`merkle_root`]).
    pub fn root(&self) -> [u8; 32] {
        self.levels
            .last()
            .and_then(|level| level.first().copied())
            .unwrap_or([0u8; 32])
    }

    /// Number of levels above the leaves (`0` for 0- or 1-leaf trees).
    pub fn depth(&self) -> usize {
        if self.levels.len() <= 1 {
            0
        } else {
            self.levels.len() - 1
        }
    }

    /// Number of leaves.
    pub fn leaf_count(&self) -> usize {
        self.levels.first().map_or(0, |l| l.len())
    }

    /// Sibling hashes bottom-up for `leaf_index` (empty for 0/1-leaf trees).
    pub fn proof(&self, leaf_index: u32) -> Vec<[u8; 32]> {
        if self.levels.len() <= 1 {
            return vec![];
        }
        let mut proof = Vec::with_capacity(self.depth());
        let mut idx = leaf_index as usize;
        for level in 0..self.levels.len() - 1 {
            let sibling_idx = if idx.is_multiple_of(2) {
                idx + 1
            } else {
                idx - 1
            };
            let sibling = if sibling_idx < self.levels[level].len() {
                self.levels[level][sibling_idx]
            } else {
                self.levels[level][idx] // duplicated last node on an odd level
            };
            proof.push(sibling);
            idx /= 2;
        }
        proof
    }
}

/// Canonical Merkle-proof depth for `chunk_count` leaves.
///
/// * `0 | 1` → `0`
/// * `n ≥ 2` → `ceil(log2(n)) = 32 - (n-1).leading_zeros()`
pub fn expected_proof_depth(chunk_count: u32) -> usize {
    match chunk_count {
        0 | 1 => 0,
        n => (u32::BITS - (n - 1).leading_zeros()) as usize,
    }
}

/// Low-level hash-chain check: does `leaf` at `leaf_index` with `path` resolve to
/// `expected_root`? Direct port of the SNIP / SUM Chain L1 verifier.
///
/// **Does not** enforce tree-shape bounds — callers on the guest path must use
/// [`verify_chunk_witness`], which enforces length/index/depth first.
pub fn verify_merkle_path(
    leaf: &[u8; 32],
    leaf_index: u32,
    path: &[[u8; 32]],
    expected_root: &[u8; 32],
) -> bool {
    let mut current = *leaf;
    for (level, sibling) in path.iter().enumerate() {
        let mut h = blake3::Hasher::new();
        if (leaf_index >> level) & 1 == 0 {
            h.update(&current);
            h.update(sibling);
        } else {
            h.update(sibling);
            h.update(&current);
        }
        current = *h.finalize().as_bytes();
    }
    &current == expected_root
}

/// A single authenticated chunk: its leaf `index` plus the raw chunk bytes.
///
/// The `index` travels *with* the chunk so a witness can never be replayed at a
/// different position (which would otherwise let a duplicated final leaf
/// masquerade as a distinct chunk of a different length).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkWitness {
    pub index: u32,
    pub data: Vec<u8>,
}

/// Why a guest-side chunk/length check rejected a witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WitnessError {
    /// Declared `byte_len` exceeds the frozen allocation bound.
    ByteLenExceedsMax { byte_len: u64, max: u64 },
    /// The declared `byte_len` needs more than `u32::MAX` chunks (B0 rejects the
    /// count as unrepresentable before any comparison can succeed).
    UnrepresentableChunkCount { byte_len: u64 },
    /// `chunk_count != ceil(byte_len / CHUNK)`.
    ChunkCountMismatch { declared: u32, expected: u32 },
    /// Witness `index >= chunk_count`.
    IndexOutOfRange { index: u32, chunk_count: u32 },
    /// Proof `depth` does not match the canonical depth for `chunk_count`.
    WrongPathDepth { got: usize, expected: usize },
    /// The witnessed chunk's byte length is wrong for its position.
    WrongChunkLen {
        index: u32,
        got: usize,
        expected: usize,
    },
    /// Empty object (`chunk_count == 0`) has no valid chunk witnesses.
    EmptyObjectHasNoChunks,
    /// The leaf/index/path hash-chain did not resolve to the expected root.
    RootMismatch,
}

/// The B0 `ceil(byte_len / CHUNK)` mapped into a [`WitnessError`]. Fallible only,
/// exactly as B0's [`chunk_count_checked`] — no lossy infallible form exists.
fn checked_chunk_count(byte_len: u64) -> Result<u32, WitnessError> {
    chunk_count_checked(byte_len).map_err(|_| WitnessError::UnrepresentableChunkCount { byte_len })
}

/// Length-only sanity checks against a declared `(byte_len, chunk_count)` pair,
/// performed *before* any allocation. Enforces the allocation bound and the B0
/// `chunk_count == ceil(byte_len / CHUNK)` rule.
pub fn check_length_commitment(byte_len: u64, chunk_count: u32) -> Result<(), WitnessError> {
    if byte_len > FROZEN_MAX_OBJECT_BYTES {
        return Err(WitnessError::ByteLenExceedsMax {
            byte_len,
            max: FROZEN_MAX_OBJECT_BYTES,
        });
    }
    let expected = checked_chunk_count(byte_len)?;
    if chunk_count != expected {
        return Err(WitnessError::ChunkCountMismatch {
            declared: chunk_count,
            expected,
        });
    }
    Ok(())
}

/// The exact byte length the chunk at `index` must have, given the object's
/// `(byte_len, chunk_count)`. Interior chunks are full 1 MiB; the final chunk
/// carries the remainder.
fn expected_chunk_len(index: u32, byte_len: u64, chunk_count: u32) -> usize {
    debug_assert!(index < chunk_count);
    if index + 1 < chunk_count {
        CHUNK
    } else {
        let full = (chunk_count as u64 - 1) * CHUNK as u64;
        (byte_len - full) as usize
    }
}

/// Fully verify a chunk witness against a committed object, enforcing every
/// length-ambiguity defence. Returns `Ok(())` only when the chunk is
/// authenticated *and* provably occupies exactly `index` in an object of exactly
/// `byte_len` bytes / `chunk_count` chunks with root `merkle_root`.
pub fn verify_chunk_witness(
    witness: &ChunkWitness,
    byte_len: u64,
    chunk_count: u32,
    merkle_root: &[u8; 32],
    path: &[[u8; 32]],
) -> Result<(), WitnessError> {
    // Length commitment + allocation bound (before touching chunk data).
    check_length_commitment(byte_len, chunk_count)?;
    if chunk_count == 0 {
        return Err(WitnessError::EmptyObjectHasNoChunks);
    }
    // Index in range.
    if witness.index >= chunk_count {
        return Err(WitnessError::IndexOutOfRange {
            index: witness.index,
            chunk_count,
        });
    }
    // Canonical path depth for this chunk_count.
    let expected_depth = expected_proof_depth(chunk_count);
    if path.len() != expected_depth {
        return Err(WitnessError::WrongPathDepth {
            got: path.len(),
            expected: expected_depth,
        });
    }
    // This chunk's length is fixed by its position — a short final chunk cannot
    // masquerade as a full one, nor vice-versa.
    let want_len = expected_chunk_len(witness.index, byte_len, chunk_count);
    if witness.data.len() != want_len {
        return Err(WitnessError::WrongChunkLen {
            index: witness.index,
            got: witness.data.len(),
            expected: want_len,
        });
    }
    // Authenticate leaf/index/path against the root.
    let leaf = blake3_32(&witness.data);
    if chunk_count == 1 {
        // Single-leaf tree: root IS the leaf; the (mandatory-empty) path was
        // already length-checked above.
        if &leaf == merkle_root {
            Ok(())
        } else {
            Err(WitnessError::RootMismatch)
        }
    } else if verify_merkle_path(&leaf, witness.index, path, merkle_root) {
        Ok(())
    } else {
        Err(WitnessError::RootMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(data: &[u8]) -> [u8; 32] {
        blake3_32(data)
    }

    #[test]
    fn tree_root_matches_b0_merkle_root() {
        // Parity: the retained tree's root is byte-identical to the adopted B0
        // rule across empty / single / multi-chunk objects.
        for byte_len in [0usize, 10, CHUNK, CHUNK + 1, 3 * CHUNK + 7] {
            let data: Vec<u8> = (0..byte_len).map(|i| (i % 251) as u8).collect();
            assert_eq!(
                MerkleTree::from_object_bytes(&data).root(),
                merkle_root(&data)
            );
        }
    }

    #[test]
    fn empty_and_single_leaf() {
        assert_eq!(MerkleTree::build(&[]).root(), [0u8; 32]);
        assert_eq!(MerkleTree::from_object_bytes(&[]).root(), merkle_root(&[]));
        let h = leaf(b"only");
        let t = MerkleTree::build(&[h]);
        assert_eq!(t.root(), h);
        assert!(t.proof(0).is_empty());
    }

    #[test]
    fn three_leaf_odd_duplication_matches_b0() {
        let h0 = leaf(b"c0");
        let h1 = leaf(b"c1");
        let h2 = leaf(b"c2");
        let t = MerkleTree::build(&[h0, h1, h2]);
        assert_eq!(t.root(), merkle_root_from_leaves(&[h0, h1, h2]));
        assert_eq!(t.proof(2), vec![h2, hash_pair(&h0, &h1)]);
    }

    #[test]
    fn chunk_count_matches_b0() {
        assert_eq!(chunk_count_checked(0).unwrap(), 0);
        assert_eq!(chunk_count_checked(1).unwrap(), 1);
        assert_eq!(chunk_count_checked(CHUNK as u64).unwrap(), 1);
        assert_eq!(chunk_count_checked(CHUNK as u64 + 1).unwrap(), 2);
    }

    #[test]
    fn witness_roundtrip_multichunk() {
        // 2.5 MiB object → 3 chunks, final chunk = 0.5 MiB.
        let byte_len = 2 * CHUNK as u64 + CHUNK as u64 / 2;
        let data: Vec<u8> = (0..byte_len).map(|i| (i % 251) as u8).collect();
        let tree = MerkleTree::from_object_bytes(&data);
        let root = tree.root();
        let chunk_count = chunk_count_checked(byte_len).unwrap();
        assert_eq!(chunk_count, 3);
        for (idx, chunk) in data.chunks(CHUNK).enumerate() {
            let w = ChunkWitness {
                index: idx as u32,
                data: chunk.to_vec(),
            };
            let path = tree.proof(idx as u32);
            assert_eq!(
                verify_chunk_witness(&w, byte_len, chunk_count, &root, &path),
                Ok(())
            );
        }
    }

    #[test]
    fn declared_byte_len_over_max_rejected_before_alloc() {
        let err = check_length_commitment(FROZEN_MAX_OBJECT_BYTES + 1, 999);
        assert!(matches!(err, Err(WitnessError::ByteLenExceedsMax { .. })));
    }
}
