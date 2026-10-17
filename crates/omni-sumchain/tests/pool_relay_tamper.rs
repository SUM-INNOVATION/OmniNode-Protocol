//! #104: the relay cannot change results.
//!
//! #104 routes coordinator traffic through a neutral relay with no on-chain
//! role. The relay carries bytes only: a unit's output manifest and slot
//! objects, and the token sequence between SelectToken steps. What a result
//! *is* comes from the chain: the B0 statement binds the unit's
//! `output_manifest` commitment and each step's `updated_token_seq_commitment`.
//! The client checks relayed bytes against those commitments with the
//! `pool-client` code merged in #139, and nothing else.
//!
//! These tests play a dishonest relay against that code: altered, truncated or
//! extended bytes, a substituted manifest or root, reordered chunks, content or
//! slots, and swapped units. Each one must be refused, so the only output a
//! relay can get accepted is the committed one. Changing the *order of
//! delivery* is not a change of result and still verifies.
//!
//! Not covered here, because it does not exist yet: the relay transport itself
//! (no wire format is ratified), and the on-chain half, `CommitPartialProofV1`
//! verification (sum-chain #131, C3).
#![cfg(feature = "pool-client")]

use omni_sumchain::pool::manifest::{
    RetrievalError, UnitOutput, layer_group_output, select_token_output, verify_retrieved_output,
};
use omni_sumchain::pool::tokens::{Finish, TokenError, TokenSequence};
use sumchain_wire::b0::enums::SlotKind;
use sumchain_wire::b0::merkle::CHUNK;
use sumchain_wire::b0::object_commitment::ObjectCommitmentV1;
use sumchain_wire::b0::workload::KvPair;

const POSITION: u32 = 5;
const KV_PAIR_BYTES: usize = 32;

fn kv(n: usize, salt: i16) -> Vec<KvPair> {
    (0..n)
        .map(|i| {
            let x = (i as i16).wrapping_mul(31).wrapping_add(salt);
            ([x; 8], [x.wrapping_neg(); 8])
        })
        .collect()
}

/// A small unit: a 16-byte residual and three KV pairs.
fn small_unit() -> UnitOutput {
    layer_group_output(POSITION, &[1, -2, 3, -4, 5, -6, 7, -8], &kv(3, 0)).unwrap()
}

/// A unit whose KV object spans two full Merkle chunks plus a tail.
fn multi_chunk_unit() -> UnitOutput {
    let pairs = 2 * CHUNK / KV_PAIR_BYTES + 100;
    let out = layer_group_output(POSITION, &[9; 8], &kv(pairs, 7)).unwrap();
    assert_eq!(out.objects[1].0, SlotKind::KvCache);
    assert!(out.objects[1].2.len() > 2 * CHUNK);
    out
}

/// Verify what a relay delivered against the commitment the chain holds.
fn verify(
    committed: &ObjectCommitmentV1,
    manifest_bytes: &[u8],
    objects: &[(SlotKind, u32, Vec<u8>)],
) -> Result<(), RetrievalError> {
    verify_retrieved_output(committed, manifest_bytes, objects).map(|_| ())
}

fn slot_mismatch(kind: SlotKind) -> Result<(), RetrievalError> {
    Err(RetrievalError::SlotMismatch {
        kind,
        index: POSITION,
    })
}

// ── Honest delivery ───────────────────────────────────────────────────────────

#[test]
fn honest_delivery_verifies_in_any_delivery_order() {
    for out in [small_unit(), multi_chunk_unit()] {
        let c = &out.manifest_commitment;
        verify(c, &out.manifest_bytes, &out.objects).unwrap();
        let mut reversed = out.objects.clone();
        reversed.reverse();
        verify(c, &out.manifest_bytes, &reversed).unwrap();
    }
}

// ── Altered bytes ─────────────────────────────────────────────────────────────

#[test]
fn a_single_bit_flip_in_the_relayed_manifest_is_refused() {
    let out = small_unit();
    for byte in 0..out.manifest_bytes.len() {
        for bit in [0x01, 0x80] {
            let mut m = out.manifest_bytes.clone();
            m[byte] ^= bit;
            assert_eq!(
                verify(&out.manifest_commitment, &m, &out.objects),
                Err(RetrievalError::ManifestCommitment),
                "manifest byte {byte} bit {bit:#04x}"
            );
        }
    }
}

#[test]
fn a_single_bit_flip_in_any_chunk_of_a_large_object_is_refused() {
    let out = multi_chunk_unit();
    let len = out.objects[1].2.len();
    // First, last and boundary bytes of every chunk.
    let mut offsets = vec![0, len - 1];
    for k in 1..=len / CHUNK {
        offsets.extend([k * CHUNK - 1, k * CHUNK]);
    }
    for off in offsets.into_iter().filter(|&o| o < len) {
        let mut objects = out.objects.clone();
        objects[1].2[off] ^= 0x01;
        assert_eq!(
            verify(&out.manifest_commitment, &out.manifest_bytes, &objects),
            slot_mismatch(SlotKind::KvCache),
            "offset {off}"
        );
    }
}

// ── Truncated or extended output ──────────────────────────────────────────────

#[test]
fn truncated_or_extended_relayed_output_is_refused() {
    let out = small_unit();
    let c = &out.manifest_commitment;
    let m = &out.manifest_bytes;

    for slot in 0..out.objects.len() {
        let kind = out.objects[slot].0;
        let full = out.objects[slot].2.clone();
        for cut in [full.len() - 1, full.len() / 2, 0] {
            let mut objects = out.objects.clone();
            objects[slot].2.truncate(cut);
            assert_eq!(
                verify(c, m, &objects),
                slot_mismatch(kind),
                "slot {slot} cut to {cut}"
            );
        }
        let mut objects = out.objects.clone();
        objects[slot].2.push(0);
        assert_eq!(
            verify(c, m, &objects),
            slot_mismatch(kind),
            "slot {slot} extended"
        );

        // A dropped slot.
        let mut objects = out.objects.clone();
        objects.remove(slot);
        assert_eq!(
            verify(c, m, &objects),
            Err(RetrievalError::MissingSlot {
                kind,
                index: POSITION
            })
        );
    }

    // A truncated or extended manifest.
    for cut in [m.len() - 1, m.len() / 2, 0] {
        assert_eq!(
            verify(c, &m[..cut], &out.objects),
            Err(RetrievalError::ManifestCommitment),
            "manifest cut to {cut}"
        );
    }
    let mut longer = m.clone();
    longer.push(0);
    assert_eq!(
        verify(c, &longer, &out.objects),
        Err(RetrievalError::ManifestCommitment)
    );

    // Nothing at all.
    assert_eq!(verify(c, &[], &[]), Err(RetrievalError::ManifestCommitment));
}

#[test]
fn a_large_object_cut_at_a_chunk_boundary_is_refused() {
    let out = multi_chunk_unit();
    for chunks in [1, 2] {
        let mut objects = out.objects.clone();
        objects[1].2.truncate(chunks * CHUNK);
        assert_eq!(
            verify(&out.manifest_commitment, &out.manifest_bytes, &objects),
            slot_mismatch(SlotKind::KvCache),
            "cut to {chunks} chunks"
        );
    }
}

// ── Substituted manifest or root ──────────────────────────────────────────────

#[test]
fn another_units_complete_output_is_refused() {
    let out = small_unit();
    let other = layer_group_output(POSITION, &[0; 8], &kv(3, 1)).unwrap();
    assert_ne!(other.manifest_commitment, out.manifest_commitment);
    // Self-consistent, same position and shape, but not the committed unit.
    assert_eq!(
        verify(
            &out.manifest_commitment,
            &other.manifest_bytes,
            &other.objects
        ),
        Err(RetrievalError::ManifestCommitment)
    );
    // The SelectToken empty manifest: the relay claims there is no output.
    let empty = select_token_output().unwrap();
    assert_eq!(
        verify(&out.manifest_commitment, &empty.manifest_bytes, &[]),
        Err(RetrievalError::ManifestCommitment)
    );
}

#[test]
fn a_forged_manifest_over_forged_objects_is_refused() {
    let out = small_unit();
    // The relay alters the residual and rebuilds a manifest that commits to its
    // own bytes, so the forged bundle is internally consistent.
    let forged = layer_group_output(POSITION, &[1, -2, 3, -4, 5, -6, 7, -9], &kv(3, 0)).unwrap();
    assert_eq!(
        verify(
            &out.manifest_commitment,
            &forged.manifest_bytes,
            &forged.objects
        ),
        Err(RetrievalError::ManifestCommitment)
    );
    // It verifies only against its own commitment. That is why the commitment
    // must come from the chain and never from the relay.
    verify(
        &forged.manifest_commitment,
        &forged.manifest_bytes,
        &forged.objects,
    )
    .unwrap();
}

#[test]
fn a_substituted_manifest_root_is_refused() {
    let out = small_unit();
    let encoded = out.manifest_commitment.encode();
    assert_eq!(encoded.len(), ObjectCommitmentV1::LEN);
    // Same kind, length and chunk count; only the Merkle root differs.
    for byte in ObjectCommitmentV1::LEN - 32..ObjectCommitmentV1::LEN {
        let mut bytes = encoded.clone();
        bytes[byte] ^= 0x01;
        let substituted = ObjectCommitmentV1::decode_exact(&bytes).unwrap();
        assert_eq!(substituted.byte_len(), out.manifest_commitment.byte_len());
        assert_eq!(
            verify(&substituted, &out.manifest_bytes, &out.objects),
            Err(RetrievalError::ManifestCommitment),
            "root byte {byte}"
        );
    }
}

#[test]
fn a_slot_object_substituted_from_another_unit_is_refused() {
    let out = small_unit();
    let other = layer_group_output(POSITION, &[0; 8], &kv(3, 1)).unwrap();
    for slot in 0..out.objects.len() {
        // Same kind, same index, same length; other content.
        let mut objects = out.objects.clone();
        objects[slot].2 = other.objects[slot].2.clone();
        assert_eq!(objects[slot].2.len(), out.objects[slot].2.len());
        assert_eq!(
            verify(&out.manifest_commitment, &out.manifest_bytes, &objects),
            slot_mismatch(out.objects[slot].0)
        );
    }
}

// ── Reordering ────────────────────────────────────────────────────────────────

#[test]
fn reordered_chunks_of_a_large_object_are_refused() {
    let out = multi_chunk_unit();
    let original = out.objects[1].2.clone();
    // Swap the two full chunks: the length and chunk count are unchanged, so
    // only the Merkle root's order binding can catch it.
    let mut swapped = original.clone();
    swapped[..CHUNK].copy_from_slice(&original[CHUNK..2 * CHUNK]);
    swapped[CHUNK..2 * CHUNK].copy_from_slice(&original[..CHUNK]);
    assert_ne!(swapped, original);
    assert_eq!(swapped.len(), original.len());
    let mut objects = out.objects.clone();
    objects[1].2 = swapped;
    assert_eq!(
        verify(&out.manifest_commitment, &out.manifest_bytes, &objects),
        slot_mismatch(SlotKind::KvCache)
    );
}

#[test]
fn reordered_content_within_an_object_is_refused() {
    let out = small_unit();
    // Swap KV pairs 0 and 2 inside the KV object.
    let mut kv_bytes = out.objects[1].2.clone();
    let (a, b) = (0..KV_PAIR_BYTES, 2 * KV_PAIR_BYTES..3 * KV_PAIR_BYTES);
    let first = kv_bytes[a.clone()].to_vec();
    kv_bytes.copy_within(b.clone(), a.start);
    kv_bytes[b].copy_from_slice(&first);
    assert_ne!(kv_bytes, out.objects[1].2);
    let mut objects = out.objects.clone();
    objects[1].2 = kv_bytes;
    assert_eq!(
        verify(&out.manifest_commitment, &out.manifest_bytes, &objects),
        slot_mismatch(SlotKind::KvCache)
    );
}

#[test]
fn objects_swapped_between_slots_or_relabelled_are_refused() {
    let out = small_unit();
    let c = &out.manifest_commitment;
    let m = &out.manifest_bytes;

    // The residual's bytes delivered as the KV cache and the reverse.
    let mut swapped = out.objects.clone();
    let residual = swapped[0].2.clone();
    swapped[0].2 = swapped[1].2.clone();
    swapped[1].2 = residual;
    assert_eq!(
        verify(c, m, &swapped),
        slot_mismatch(SlotKind::ResidualStream)
    );

    // A slot relabelled to another position.
    let mut moved = out.objects.clone();
    moved[1].1 = POSITION + 1;
    assert_eq!(
        verify(c, m, &moved),
        Err(RetrievalError::UnexpectedSlot {
            kind: SlotKind::KvCache,
            index: POSITION + 1
        })
    );
}

#[test]
fn two_units_outputs_swapped_by_the_relay_are_both_refused() {
    let a = layer_group_output(4, &[1; 8], &kv(2, 3)).unwrap();
    let b = layer_group_output(5, &[2; 8], &kv(2, 4)).unwrap();
    assert_eq!(
        verify(&a.manifest_commitment, &b.manifest_bytes, &b.objects),
        Err(RetrievalError::ManifestCommitment)
    );
    assert_eq!(
        verify(&b.manifest_commitment, &a.manifest_bytes, &a.objects),
        Err(RetrievalError::ManifestCommitment)
    );
    // The right manifest with the other unit's objects.
    assert!(verify(&a.manifest_commitment, &a.manifest_bytes, &b.objects).is_err());
}

// ── Token sequence between SelectToken steps ─────────────────────────────────

const VOCAB: u32 = 16;
const EOS: u32 = VOCAB - 1;
const PROMPT: [u32; 3] = [1, 2, 3];
const HONEST: [(u32, u8); 3] = [(5, 0), (7, 0), (EOS, 1)];

/// Replay relayed `(selected, eos_flag)` results from `prompt` and return the
/// final `TokenSeq` commitment, or the refusal.
fn replay(prompt: &[u32], relayed: &[(u32, u8)]) -> Result<ObjectCommitmentV1, TokenError> {
    let mut seq = TokenSequence::new(prompt.to_vec(), VOCAB)?;
    for &(selected, eos_flag) in relayed {
        seq.apply_select(selected, eos_flag)?;
    }
    seq.sequence_commitment()
}

#[test]
fn a_relayed_token_sequence_other_than_the_committed_one_does_not_match() {
    // What the last step's statement binds as `updated_token_seq_commitment`.
    let committed = replay(&PROMPT, &HONEST).unwrap();

    let altered = [(6, 0), (7, 0), (EOS, 1)];
    let dropped = [(7, 0), (EOS, 1)];
    let reordered = [(7, 0), (5, 0), (EOS, 1)];
    let truncated = [(5, 0), (7, 0)];
    for (name, relayed) in [
        ("altered", &altered[..]),
        ("dropped", &dropped[..]),
        ("reordered", &reordered[..]),
        ("truncated before EOS", &truncated[..]),
    ] {
        assert_ne!(replay(&PROMPT, relayed).unwrap(), committed, "{name}");
    }
    assert_ne!(
        replay(&[1, 2, 4], &HONEST).unwrap(),
        committed,
        "altered prompt"
    );
}

#[test]
fn relayed_selections_that_break_the_statement_rules_are_refused() {
    let mut after_eos = HONEST.to_vec();
    after_eos.push((3, 0));
    assert_eq!(
        replay(&PROMPT, &after_eos),
        Err(TokenError::Finished(Finish::Eos))
    );
    assert_eq!(
        replay(&PROMPT, &[(5, 1)]),
        Err(TokenError::EosFlagMismatch {
            selected: 5,
            eos_flag: 1,
            eos: EOS
        })
    );
    assert_eq!(
        replay(&PROMPT, &[(EOS, 0)]),
        Err(TokenError::EosFlagMismatch {
            selected: EOS,
            eos_flag: 0,
            eos: EOS
        })
    );
    assert_eq!(
        replay(&PROMPT, &[(VOCAB, 0)]),
        Err(TokenError::OutOfVocabulary {
            token: VOCAB,
            vocab_size: VOCAB
        })
    );
}
