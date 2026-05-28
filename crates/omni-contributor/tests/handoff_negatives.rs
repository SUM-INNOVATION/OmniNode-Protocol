//! Stage 12.4 — schema / signature / hash / chunk / binding
//! negative tests for `ActivationHandoff`, `verify_activation_handoff`,
//! and `HandoffReceiver`.

use omni_contributor::{
    canonical::{
        activation_handoff_signing_input, assignment_id_hex, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, session_id_hex,
        work_assignment_signing_input,
    },
    error::SchemaError,
    result::WorkUnitKind,
    verify_activation_handoff, verify_contributor_join, verify_execution_session,
    verify_work_assignment, ActivationHandoff, ChunkOutcome, ContributorJoin,
    ContributorSigner, CoordinatorSigner, ExecutionSession, HandoffReceiver,
    HandoffVerifyOutcome, TensorDtype, WorkAssignment, WorkKind, HANDOFF_BYTE_LEN_MAX,
    HANDOFF_CHUNK_COUNT_MAX, HANDOFF_CHUNK_MAX_BYTES, HANDOFF_SCHEMA_VERSION,
    HANDOFF_SHAPE_RANK_MAX, SESSION_SCHEMA_VERSION,
};
use std::collections::HashSet;

const COORD_SEED: [u8; 32] = *b"stage12.4-coord-seed-32-byte-key";
const CONTRIB_A_SEED: [u8; 32] = *b"stage12.4-contrib-A-seed-32byte!";
const CONTRIB_B_SEED: [u8; 32] = *b"stage12.4-contrib-B-seed-32byte!";
const ROGUE_SEED: [u8; 32] = *b"stage12.4-rogue-seed-32-bytekey!";

fn build_session(coord: &CoordinatorSigner) -> ExecutionSession {
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-05-27T00:00:00Z".into(),
        expires_at_utc: "2026-05-27T01:00:00Z".into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn build_join(session: &ExecutionSession, contrib: &ContributorSigner) -> ContributorJoin {
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 100,
        max_output_tokens: 100,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-27T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

fn build_assignment(
    session: &ExecutionSession,
    contrib_pub: &str,
    stage_index: u32,
    coord: &CoordinatorSigner,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index,
        contributor_pubkey_hex: contrib_pub.to_string(),
        work_kind: WorkKind::Layers { start: 0, end: 16 },
        expected_work_units: 16,
        expected_work_unit_kind: WorkUnitKind::Layers,
        assigned_at_utc: format!("2026-05-27T00:00:0{stage_index}Z"),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

fn build_signed_handoff(
    session: &ExecutionSession,
    from: &WorkAssignment,
    to: &WorkAssignment,
    sender: &ContributorSigner,
    tensor_bytes: Vec<u8>,
) -> ActivationHandoff {
    let tensor_hash = hex_lower(blake3::hash(&tensor_bytes).as_bytes());
    let byte_len = tensor_bytes.len() as u64;
    let mut h = ActivationHandoff {
        schema_version: HANDOFF_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        from_assignment_id: from.assignment_id.clone(),
        to_assignment_id: to.assignment_id.clone(),
        from_contributor_pubkey_hex: from.contributor_pubkey_hex.clone(),
        to_contributor_pubkey_hex: to.contributor_pubkey_hex.clone(),
        dtype: TensorDtype::F16,
        shape: vec![byte_len / 2], // f16 → 2 bytes per element
        byte_len,
        tensor_hash,
        chunk_index: 0,
        chunk_count: 1,
        produced_at_utc: "2026-05-27T00:00:30Z".into(),
        tensor_chunk_bytes: tensor_bytes,
        sender_signature_hex: String::new(),
    };
    let si = activation_handoff_signing_input(&h).unwrap();
    h.sender_signature_hex = sender.sign_hex(&si);
    h
}

// Convenience: build the 12.3 session graph and return signers + envelopes.
fn graph_2_stages() -> (
    ExecutionSession,
    WorkAssignment,
    WorkAssignment,
    ContributorSigner,
    ContributorSigner,
) {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let b_signer = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = build_session(&coord);
    let _join_a = build_join(&session, &a_signer);
    let _join_b = build_join(&session, &b_signer);
    let asn_a = build_assignment(&session, &a_signer.pubkey_hex(), 0, &coord);
    let asn_b = build_assignment(&session, &b_signer.pubkey_hex(), 1, &coord);
    (session, asn_a, asn_b, a_signer, b_signer)
}

// ── Schema bounds ──────────────────────────────────────────────────────

#[test]
fn schema_shape_empty_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    h.shape = vec![];
    assert!(matches!(h.validate_schema(), Err(SchemaError::HandoffShapeEmpty)));
}

#[test]
fn schema_shape_rank_too_large_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    h.shape = vec![1; HANDOFF_SHAPE_RANK_MAX + 1];
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffShapeRankTooLarge { .. })
    ));
}

#[test]
fn schema_shape_zero_dim_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    h.shape = vec![0, 8];
    assert!(matches!(h.validate_schema(), Err(SchemaError::HandoffShapeZeroDim)));
}

#[test]
fn schema_byte_len_mismatch_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    // declared byte_len 16, shape [8] @ f16 → expected 16 — change shape to [9].
    h.shape = vec![9];
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffByteLenMismatch { .. })
    ));
}

#[test]
fn schema_byte_len_too_large_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    h.byte_len = HANDOFF_BYTE_LEN_MAX + 1;
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffByteLenTooLarge { .. })
    ));
}

#[test]
fn schema_chunk_count_zero_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    h.chunk_count = 0;
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffChunkCountZero)
    ));
}

#[test]
fn schema_chunk_count_too_large_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    h.chunk_count = HANDOFF_CHUNK_COUNT_MAX + 1;
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffChunkCountTooLarge { .. })
    ));
}

#[test]
fn schema_chunk_index_out_of_range_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    h.chunk_count = 2;
    // chunk_index 5 with count 2 → out of range.
    h.chunk_index = 5;
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffChunkIndexOutOfRange { .. })
    ));
}

#[test]
fn schema_chunk_bytes_empty_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    h.tensor_chunk_bytes.clear();
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffChunkBytesEmpty)
    ));
}

#[test]
fn schema_single_chunk_len_mismatch_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    // chunk_count == 1 but tensor_chunk_bytes shorter than byte_len.
    h.tensor_chunk_bytes = vec![0xAB; 8];
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffSingleChunkLenMismatch { .. })
    ));
}

#[test]
fn schema_chunk_bytes_too_large_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    // Falsify chunk_count so the single-chunk len check doesn't fire
    // first; we're isolating the per-chunk size bound.
    h.chunk_count = 4;
    h.chunk_index = 0;
    h.tensor_chunk_bytes = vec![0xAB; HANDOFF_CHUNK_MAX_BYTES as usize + 1];
    assert!(matches!(
        h.validate_schema(),
        Err(SchemaError::HandoffChunkBytesTooLarge { .. })
    ));
}

// ── Signature + binding ────────────────────────────────────────────────

#[test]
fn verify_happy_path() {
    let (session, a, b, sa, _sb) = graph_2_stages();
    let h = build_signed_handoff(&session, &a, &b, &sa, vec![0xAB; 16]);
    assert!(verify_activation_handoff(&session, &a, &b, &h).is_ok());
}

#[test]
fn verify_tampered_sender_signature_refused() {
    let (session, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&session, &a, &b, &sa, vec![0xAB; 16]);
    let mut sig = h.sender_signature_hex.into_bytes();
    sig[0] = if sig[0] == b'a' { b'b' } else { b'a' };
    h.sender_signature_hex = String::from_utf8(sig).unwrap();
    assert!(matches!(
        verify_activation_handoff(&session, &a, &b, &h),
        HandoffVerifyOutcome::SenderSignatureFailed
    ));
}

#[test]
fn verify_wrong_sender_pubkey_refused() {
    // Sender forges from_contributor_pubkey_hex to claim someone else
    // — sig validates against the wrong key.
    let (session, a, b, _sa, _sb) = graph_2_stages();
    let rogue = ContributorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    // Build envelope as if rogue is sending — but the verifier
    // checks from_contributor_pubkey_hex against from_assignment's
    // contributor_pubkey_hex (which is contrib-A's, not rogue's).
    let h = build_signed_handoff(&session, &a, &b, &rogue, vec![0xAB; 16]);
    // Manually re-build the envelope so its from_contributor_pubkey
    // says A but the actual signature is rogue's.
    let mut h2 = h.clone();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    h2.from_contributor_pubkey_hex = a_signer.pubkey_hex();
    // Re-sign with rogue under a's claimed pubkey.
    let si = activation_handoff_signing_input(&h2).unwrap();
    h2.sender_signature_hex = rogue.sign_hex(&si);
    // Now from_contributor_pubkey_hex (a's pubkey) doesn't verify the
    // rogue-signed bytes.
    assert!(matches!(
        verify_activation_handoff(&session, &a, &b, &h2),
        HandoffVerifyOutcome::SenderSignatureFailed
    ));
}

#[test]
fn verify_stage_order_non_adjacent_refused() {
    // to.stage_index = 2, from.stage_index = 0 → not +1.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let b_signer = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = build_session(&coord);
    let asn_a = build_assignment(&session, &a_signer.pubkey_hex(), 0, &coord);
    let asn_b = build_assignment(&session, &b_signer.pubkey_hex(), 2, &coord);
    let h = build_signed_handoff(&session, &asn_a, &asn_b, &a_signer, vec![0xAB; 16]);
    assert!(matches!(
        verify_activation_handoff(&session, &asn_a, &asn_b, &h),
        HandoffVerifyOutcome::StageOrderInvalid { .. }
    ));
}

#[test]
fn verify_session_id_drift_refused() {
    let (session, a, b, sa, _sb) = graph_2_stages();
    let mut h = build_signed_handoff(&session, &a, &b, &sa, vec![0xAB; 16]);
    h.session_id = "ff".repeat(32);
    let si = activation_handoff_signing_input(&h).unwrap();
    h.sender_signature_hex = sa.sign_hex(&si);
    assert!(matches!(
        verify_activation_handoff(&session, &a, &b, &h),
        HandoffVerifyOutcome::SessionIdMismatch
    ));
}

#[test]
fn verify_session_expired_refused() {
    let (mut session, a, b, sa, _sb) = graph_2_stages();
    // Shorten session expiry to before handoff produced_at_utc.
    session.expires_at_utc = "2026-05-27T00:00:00Z".into();
    let si = execution_session_signing_input(&session).unwrap();
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    session.session_id = session_id_hex(&session).unwrap();
    session.coordinator_signature_hex = coord.sign_hex(&si);
    let h = build_signed_handoff(&session, &a, &b, &sa, vec![0xAB; 16]);
    assert!(matches!(
        verify_activation_handoff(&session, &a, &b, &h),
        HandoffVerifyOutcome::SessionExpired { .. }
    ));
}

#[test]
fn verify_to_contributor_mismatch_refused() {
    let (session, a, b, sa, _sb) = graph_2_stages();
    let rogue = ContributorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let mut h = build_signed_handoff(&session, &a, &b, &sa, vec![0xAB; 16]);
    h.to_contributor_pubkey_hex = rogue.pubkey_hex();
    let si = activation_handoff_signing_input(&h).unwrap();
    h.sender_signature_hex = sa.sign_hex(&si);
    assert!(matches!(
        verify_activation_handoff(&session, &a, &b, &h),
        HandoffVerifyOutcome::ToContributorMismatch
    ));
}

// Sanity: the assignment + session verifier chains we lean on for
// integration still pass when used in the typical happy-path
// configuration this stage builds on.
#[test]
fn upstream_12_3_chain_still_passes() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let session = build_session(&coord);
    let join = build_join(&session, &a_signer);
    let asn = build_assignment(&session, &a_signer.pubkey_hex(), 0, &coord);
    assert!(verify_execution_session(&session).is_ok());
    assert!(verify_contributor_join(&session, &join).is_ok());
    let mut joined: HashSet<String> = HashSet::new();
    joined.insert(join.contributor_pubkey_hex.clone());
    assert!(verify_work_assignment(&session, &joined, &asn).is_ok());
}

// ── Reassembler ────────────────────────────────────────────────────────

#[test]
fn reassembler_single_chunk_happy_path() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let h = build_signed_handoff(&s, &a, &b, &sa, vec![0xAB; 16]);
    let mut r = HandoffReceiver::new();
    let out = r.feed(&h);
    match out {
        ChunkOutcome::Complete { tensor_bytes } => {
            assert_eq!(tensor_bytes.len(), 16);
        }
        other => panic!("expected Complete, got {other:?}"),
    }
    assert_eq!(r.pending_streams(), 0);
}

#[test]
fn reassembler_three_chunks_happy_path() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a_signer = ContributorSigner::from_seed_bytes(&CONTRIB_A_SEED).unwrap();
    let b_signer = ContributorSigner::from_seed_bytes(&CONTRIB_B_SEED).unwrap();
    let session = build_session(&coord);
    let asn_a = build_assignment(&session, &a_signer.pubkey_hex(), 0, &coord);
    let asn_b = build_assignment(&session, &b_signer.pubkey_hex(), 1, &coord);

    let full: Vec<u8> = (0u8..30).cycle().take(30).collect(); // 30 bytes
    let tensor_hash = hex_lower(blake3::hash(&full).as_bytes());
    let chunks = [&full[..10], &full[10..20], &full[20..]];
    let mut r = HandoffReceiver::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let mut h = ActivationHandoff {
            schema_version: HANDOFF_SCHEMA_VERSION,
            session_id: session.session_id.clone(),
            from_assignment_id: asn_a.assignment_id.clone(),
            to_assignment_id: asn_b.assignment_id.clone(),
            from_contributor_pubkey_hex: a_signer.pubkey_hex(),
            to_contributor_pubkey_hex: b_signer.pubkey_hex(),
            dtype: TensorDtype::F16,
            shape: vec![15], // 15 elements × 2 bytes = 30 bytes
            byte_len: 30,
            tensor_hash: tensor_hash.clone(),
            chunk_index: i as u32,
            chunk_count: 3,
            produced_at_utc: "2026-05-27T00:00:30Z".into(),
            tensor_chunk_bytes: chunk.to_vec(),
            sender_signature_hex: String::new(),
        };
        let si = activation_handoff_signing_input(&h).unwrap();
        h.sender_signature_hex = a_signer.sign_hex(&si);
        // For this test we feed straight into the reassembler — the
        // verifier-side checks are exercised by the verify_* tests
        // above.
        let out = r.feed(&h);
        if i < 2 {
            assert!(matches!(out, ChunkOutcome::Accepted), "{out:?}");
        } else {
            match out {
                ChunkOutcome::Complete { tensor_bytes } => {
                    assert_eq!(tensor_bytes, full);
                }
                other => panic!("final chunk expected Complete, got {other:?}"),
            }
        }
    }
    assert_eq!(r.pending_streams(), 0);
}

fn build_chunked_handoff(
    session: &ExecutionSession,
    from: &WorkAssignment,
    to: &WorkAssignment,
    sender: &ContributorSigner,
    full: &[u8],
    chunk_index: u32,
    chunk_count: u32,
    chunk_bytes: &[u8],
) -> ActivationHandoff {
    let tensor_hash = hex_lower(blake3::hash(full).as_bytes());
    let mut h = ActivationHandoff {
        schema_version: HANDOFF_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        from_assignment_id: from.assignment_id.clone(),
        to_assignment_id: to.assignment_id.clone(),
        from_contributor_pubkey_hex: from.contributor_pubkey_hex.clone(),
        to_contributor_pubkey_hex: to.contributor_pubkey_hex.clone(),
        dtype: TensorDtype::F16,
        shape: vec![(full.len() / 2) as u64],
        byte_len: full.len() as u64,
        tensor_hash,
        chunk_index,
        chunk_count,
        produced_at_utc: "2026-05-27T00:00:30Z".into(),
        tensor_chunk_bytes: chunk_bytes.to_vec(),
        sender_signature_hex: String::new(),
    };
    let si = activation_handoff_signing_input(&h).unwrap();
    h.sender_signature_hex = sender.sign_hex(&si);
    h
}

#[test]
fn reassembler_tensor_hash_mismatch_refused() {
    let (session, a, b, sa, _sb) = graph_2_stages();
    // Sign over a hash that doesn't match the bytes we then attach.
    let lying = vec![0xAB; 16];
    let mut h = build_signed_handoff(&session, &a, &b, &sa, lying.clone());
    // Now swap in different bytes (same length so single-chunk schema
    // check passes), without re-signing — the signature still covers
    // the LYING bytes' hash.
    h.tensor_chunk_bytes = vec![0xCD; 16];
    // We bypass verify_activation_handoff here (that path catches
    // signature/binding stuff). The reassembler must catch
    // tensor-bytes mismatch.
    let mut r = HandoffReceiver::new();
    let out = r.feed(&h);
    assert!(matches!(out, ChunkOutcome::TensorHashMismatch { .. }), "{out:?}");
}

#[test]
fn reassembler_duplicate_chunk_index_refused_idempotently() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let full: Vec<u8> = vec![0xAB; 20];
    let c0 = &full[..10];
    let c1 = &full[10..];
    let h0 = build_chunked_handoff(&s, &a, &b, &sa, &full, 0, 2, c0);
    let h0_dup = h0.clone();
    let h1 = build_chunked_handoff(&s, &a, &b, &sa, &full, 1, 2, c1);
    let mut r = HandoffReceiver::new();
    assert!(matches!(r.feed(&h0), ChunkOutcome::Accepted));
    // Duplicate.
    assert!(matches!(
        r.feed(&h0_dup),
        ChunkOutcome::DuplicateChunkIndex { chunk_index: 0 }
    ));
    // Stream is still alive; the second-and-final chunk completes it.
    match r.feed(&h1) {
        ChunkOutcome::Complete { tensor_bytes } => assert_eq!(tensor_bytes, full),
        other => panic!("expected Complete, got {other:?}"),
    }
}

#[test]
fn reassembler_stream_metadata_drift_refused() {
    let (s, a, b, sa, _sb) = graph_2_stages();
    let full: Vec<u8> = vec![0xAB; 20];
    let c0 = &full[..10];
    let mut h0 = build_chunked_handoff(&s, &a, &b, &sa, &full, 0, 2, c0);
    let mut h1 = build_chunked_handoff(&s, &a, &b, &sa, &full, 1, 2, &full[10..]);
    // Drift chunk_count on the second chunk.
    h1.chunk_count = 3;
    let si = activation_handoff_signing_input(&h1).unwrap();
    h1.sender_signature_hex = sa.sign_hex(&si);
    let mut r = HandoffReceiver::new();
    // Also drift produced_at so the first chunk schema passes.
    h0.produced_at_utc = "2026-05-27T00:00:30Z".into();
    let si = activation_handoff_signing_input(&h0).unwrap();
    h0.sender_signature_hex = sa.sign_hex(&si);
    assert!(matches!(r.feed(&h0), ChunkOutcome::Accepted));
    assert!(matches!(
        r.feed(&h1),
        ChunkOutcome::StreamMetadataDrift { field: "chunk_count" }
    ));
    // Drift drops the stream.
    assert_eq!(r.pending_streams(), 0);
}
