//! `pool-client` against the chain's frozen B0 official-workload vectors.
//!
//! `tests/fixtures/b0_official_workload_v1.json` is a verbatim copy of
//! sum-chain `docs/b0-pre/fixtures/workload/official.json`, unchanged since
//! fdc9aee3 and identical at main 76dfab80, sha256
//! `0273b05f869352eab8e38cb0f2be62c3ac39d94033ed72f7d703c3dab3377aa9`. It is the
//! fixture sum-chain's own guest-core `reference_agreement` test binds the
//! official statements to. These tests rebuild the chain-committed objects from
//! the fixture's raw bytes with this crate's client code, and require them to
//! equal the chain's bytes and commitments exactly.
#![cfg(feature = "pool-client")]

use omni_sumchain::pool::manifest::{
    RetrievalError, layer_group_output, select_token_output, verify_retrieved_output,
};
use omni_sumchain::pool::tokens::{Finish, TokenSequence};
use sumchain_wire::b0::consts::VOCAB_SIZE;
use sumchain_wire::b0::enums::SlotKind;
use sumchain_wire::b0::object_commitment::ObjectCommitmentV1;
use sumchain_wire::b0::workload::{decode_kv_state, decode_residual_state, decode_token_seq};

const FIXTURE: &str = include_str!("fixtures/b0_official_workload_v1.json");

fn fixture() -> serde_json::Value {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(v["fixture_name"], "official-workload-v1");
    v
}

fn hex(v: &serde_json::Value) -> Vec<u8> {
    let s = v.as_str().expect("hex string");
    assert!(s.len() % 2 == 0, "odd hex length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

fn commitment(v: &serde_json::Value) -> ObjectCommitmentV1 {
    ObjectCommitmentV1::decode_exact(&hex(v)).expect("80-byte object commitment")
}

#[test]
fn the_layer_group_output_manifest_is_byte_identical_to_the_chains() {
    let f = fixture();
    let tlg = &f["tlg"];
    let position = f["tlg_position"].as_u64().unwrap() as u32;
    let residual = decode_residual_state(&hex(&tlg["output_residual"])).unwrap();
    let kv = decode_kv_state(&hex(&tlg["output_kv"])).unwrap();

    let out = layer_group_output(position, &residual, &kv).unwrap();
    assert_eq!(out.manifest_bytes, hex(&tlg["output_manifest"]));
    assert_eq!(
        out.manifest_commitment,
        commitment(&tlg["c_output_manifest"])
    );
    assert_eq!(
        out.manifest.slots[0].commitment,
        commitment(&tlg["c_output_residual"])
    );
    assert_eq!(
        out.manifest.slots[1].commitment,
        commitment(&tlg["c_output_kv"])
    );
    assert_eq!(out.objects[0].2, hex(&tlg["output_residual"]));
    assert_eq!(out.objects[1].2, hex(&tlg["output_kv"]));
}

#[test]
fn retrieval_of_the_chains_objects_verifies_and_any_tampering_is_refused() {
    let f = fixture();
    let tlg = &f["tlg"];
    let position = f["tlg_position"].as_u64().unwrap() as u32;
    let committed = commitment(&tlg["c_output_manifest"]);
    let manifest_bytes = hex(&tlg["output_manifest"]);
    let objects = vec![
        (
            SlotKind::ResidualStream,
            position,
            hex(&tlg["output_residual"]),
        ),
        (SlotKind::KvCache, position, hex(&tlg["output_kv"])),
    ];

    let m = verify_retrieved_output(&committed, &manifest_bytes, &objects).unwrap();
    assert_eq!(m.slots.len(), 2);

    let mut tampered = objects.clone();
    tampered[1].2[0] ^= 0xFF;
    assert_eq!(
        verify_retrieved_output(&committed, &manifest_bytes, &tampered),
        Err(RetrievalError::SlotMismatch {
            kind: SlotKind::KvCache,
            index: position
        })
    );
    // The prior KV (an input, not this unit's output) is not a substitute.
    let mut wrong_object = objects.clone();
    wrong_object[1].2 = hex(&tlg["prior_kv"]);
    if wrong_object[1].2 != objects[1].2 {
        assert!(verify_retrieved_output(&committed, &manifest_bytes, &wrong_object).is_err());
    }
    // The SelectToken unit's (empty) manifest is not this unit's manifest.
    let select = &f["select"];
    assert_eq!(
        verify_retrieved_output(
            &commitment(&select["c_output_manifest_empty"]),
            &manifest_bytes,
            &objects
        ),
        Err(RetrievalError::ManifestCommitment)
    );
}

#[test]
fn the_select_token_step_reproduces_the_chains_updated_token_sequence() {
    let f = fixture();
    let s = &f["select"];
    let prefix = decode_token_seq(&hex(&s["token_prefix"])).unwrap();
    let selected = u32::from_le_bytes(hex(&s["selected_token"]).try_into().unwrap());
    let eos_flag = hex(&s["eos_flag"])[0];
    assert_eq!((selected, eos_flag), (2, 0));

    let mut seq = TokenSequence::new(prefix.clone(), VOCAB_SIZE).unwrap();
    // The statement's input prefix is a TokenPrefix object …
    assert_eq!(
        seq.prefix_commitment().unwrap(),
        commitment(&s["c_token_prefix"])
    );

    // … and the updated sequence a TokenSeq object over prefix ‖ selected.
    let updated = seq.apply_select(selected, eos_flag).unwrap();
    assert_eq!(updated, commitment(&s["c_updated_token_seq"]));
    assert_eq!(seq.sequence_commitment().unwrap(), updated);
    assert_eq!(
        decode_token_seq(&hex(&s["updated_token_seq"])).unwrap(),
        seq.tokens()
    );
    assert_eq!(seq.output_tokens(), &[selected]);
    assert!(seq.next_step_allowed());

    // A SelectToken unit commits to the canonical empty output manifest.
    let empty = select_token_output().unwrap();
    assert_eq!(
        empty.manifest_commitment,
        commitment(&s["c_output_manifest_empty"])
    );
    assert_eq!(empty.manifest_bytes, hex(&s["output_manifest"]));
}

#[test]
fn the_chains_vocabulary_ends_in_eos_and_the_stop_rule_follows() {
    // Under the frozen workload EOS is token VOCAB_SIZE − 1. A sequence that
    // selects it stops and does not count it.
    let f = fixture();
    let prefix = decode_token_seq(&hex(&f["select"]["token_prefix"])).unwrap();
    let mut seq = TokenSequence::new(prefix, VOCAB_SIZE).unwrap();
    assert_eq!(seq.eos(), VOCAB_SIZE - 1);
    seq.apply_select(VOCAB_SIZE - 1, 1).unwrap();
    assert_eq!(seq.finished(), Some(Finish::Eos));
    assert_eq!(seq.output_count(), 0);
    assert!(!seq.next_step_allowed());
}
