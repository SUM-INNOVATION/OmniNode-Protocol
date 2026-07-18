//! Frozen B0 byte sizes, strict-decoder round-trips, reserved-kind rejection,
//! and confirmation that the obsolete V1 / 1066-byte statement and the reserved
//! `Slot` / `Tokenizer` object kinds are absent from every path this crate uses.

mod common;

use common::golden;
use r0_zkvm_bench::envelope::Candidate;
use r0_zkvm_bench::fixture;
use r0_zkvm_bench::manifest::{
    output_slot, OutputManifestV1, SlotKind, MANIFEST_HEADER_LEN, SLOT_DESCRIPTOR_LEN,
};
use r0_zkvm_bench::object::{ObjectCommitmentV1, ObjectKind};
use r0_zkvm_bench::statement::{DerivedInputV1, SyntheticJournal};
use r0_zkvm_bench::DecodeError;
// The raw statement type / template encoder are NOT public in r0-zkvm-bench;
// exercise the frozen 996-byte size through the `sumchain-wire` dependency.
use sumchain_wire::b0::statement::{template_bytes, R0ComputationStatementV2};

#[test]
fn object_commitment_is_80_bytes_and_roundtrips() {
    let c = ObjectCommitmentV1::commit(ObjectKind::Model, b"weights").unwrap();
    assert_eq!(c.encode().len(), 80);
    assert_eq!(ObjectCommitmentV1::decode_exact(&c.encode()).unwrap(), c);
}

#[test]
fn manifest_header_38_and_slot_descriptor_85() {
    assert_eq!(MANIFEST_HEADER_LEN, 38);
    assert_eq!(SLOT_DESCRIPTOR_LEN, 85);
    // A one-slot manifest is 38 + 85 = 123, proving both sizes.
    let m = OutputManifestV1 {
        slots: vec![output_slot(SlotKind::ResidualStream, 0, b"r").unwrap()],
    };
    let bytes = m.try_encode().unwrap();
    assert_eq!(bytes.len(), 38 + 85);
    assert_eq!(OutputManifestV1::decode_exact(&bytes).unwrap(), m);
}

#[test]
fn derived_input_is_350_bytes_and_roundtrips() {
    let d = DerivedInputV1 {
        job_id: [1; 32],
        session_id: [2; 32],
        unit_id: [3; 32],
        generation_index: 7,
        model_id: [4; 32],
        model_commitment_identity: [5; 32],
        layer_start: 0,
        layer_end: 1,
        prior_residual_commitment_identity: [6; 32],
        prior_kv_commitment_identity: [7; 32],
        token_prefix_commitment_identity: [8; 32],
        position: 7,
        sequence_length: 8,
    };
    assert_eq!(d.encode().len(), 350);
    assert_eq!(DerivedInputV1::decode_exact(&d.encode()).unwrap(), d);
}

#[test]
fn statement_is_996_bytes_and_roundtrips() {
    let (_s, j) = golden();
    assert_eq!(j.len(), 996);
    assert_eq!(R0ComputationStatementV2::LEN, 996);
    // Round-trips: decode the journal bytes (via sumchain-wire), re-produce the
    // zero-spec template, and re-ingest through the SyntheticJournal boundary.
    let s = R0ComputationStatementV2::decode_exact(j.bytes()).unwrap();
    let bytes = template_bytes(s).unwrap();
    assert_eq!(
        SyntheticJournal::from_template_bytes(&bytes)
            .unwrap()
            .bytes(),
        j.bytes()
    );
}

#[test]
fn partial_proof_137_and_envelope_235() {
    let j = fixture::journal().unwrap();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let partial = fixture::partial_proof(&j, &allow).unwrap();
    assert_eq!(partial.encode().len(), 137);
    let env = fixture::envelope(&j, &allow, Candidate::Sp1).unwrap();
    assert_eq!(env.encode().len(), 235);
}

#[test]
fn trailing_bytes_rejected_on_object_commitment() {
    let c = ObjectCommitmentV1::commit(ObjectKind::Model, b"x").unwrap();
    let mut bytes = c.encode();
    bytes.push(0);
    assert!(matches!(
        ObjectCommitmentV1::decode_exact(&bytes),
        Err(DecodeError::TrailingBytes { .. })
    ));
}

#[test]
fn reserved_slot_and_tokenizer_object_kinds_are_rejected() {
    // The reserved kinds are not even nameable variants; feeding their raw
    // discriminants proves the decoder rejects them distinctly from unknown ones.
    assert!(matches!(
        ObjectKind::from_repr(2), // Tokenizer
        Err(DecodeError::ReservedEnum {
            name: "ObjectKind",
            value: 2
        })
    ));
    assert!(matches!(
        ObjectKind::from_repr(8), // Slot
        Err(DecodeError::ReservedEnum {
            name: "ObjectKind",
            value: 8
        })
    ));
    // And the defined kinds this crate uses all round-trip.
    for kind in [
        ObjectKind::Empty,
        ObjectKind::Model,
        ObjectKind::TokenPrefix,
        ObjectKind::InputManifest,
        ObjectKind::OutputManifest,
        ObjectKind::ResidualState,
        ObjectKind::KvState,
        ObjectKind::DerivedInput,
        ObjectKind::TokenSeq,
        ObjectKind::PriorResidual,
        ObjectKind::PriorKv,
    ] {
        assert_eq!(ObjectKind::from_repr(kind.to_repr()).unwrap(), kind);
        assert_ne!(
            kind.to_repr(),
            2,
            "no crate object kind may collide with reserved Tokenizer"
        );
        assert_ne!(
            kind.to_repr(),
            8,
            "no crate object kind may collide with reserved Slot"
        );
    }
}

#[test]
fn statement_is_the_996_byte_v2_not_the_obsolete_1066_v1() {
    // Identity reset: the crate has exactly one statement format, the 996-byte
    // B0 V2. The obsolete 1066-byte OmniNode V1 (and its OMNINODE.R0.* domain) is
    // gone; the frozen statement tag is SUMCHAIN/R0/STATEMENT/v2.
    assert_eq!(R0ComputationStatementV2::LEN, 996);
    assert_ne!(R0ComputationStatementV2::LEN, 1066);
    let (_s, j) = golden();
    assert_eq!(&j.bytes()[0..24], b"SUMCHAIN/R0/STATEMENT/v2");
}
