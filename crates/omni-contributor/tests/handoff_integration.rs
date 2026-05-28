//! Stage 12.4 — end-to-end integration tests using
//! `InMemoryTensorTransport` + `InMemoryRelay` + `MockSnipStore`.
//! No real networking. Exercises the full
//!   open → join → assign → send-handoff → receive-handoff → run
//! shape against the local verifier + runner trait.

mod common;

use std::collections::HashSet;

use common::MockSnipStore;
use omni_contributor::{
    canonical::{
        activation_handoff_signing_input, assignment_id_hex, contributor_join_signing_input,
        execution_session_signing_input, hex_lower, session_id_hex,
        work_assignment_signing_input,
    },
    result::WorkUnitKind,
    runner::StubRunner,
    verify_activation_handoff, verify_contributor_join, verify_execution_session,
    verify_partial_result, verify_work_assignment, ActivationHandoff, ChunkOutcome,
    ContributorJoin, ContributorSigner, CoordinatorSigner, ExecutionSession,
    HandoffReceiver, HandoffVerifyOutcome, InMemoryTensorTransport, InferenceRunner,
    TensorDtype, TensorTransport, WorkAssignment, WorkKind, HANDOFF_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.4-int-coord-seed-32byte!";
const A_SEED: [u8; 32] = *b"stage12.4-int-contrib-A-seed-32!";
const B_SEED: [u8; 32] = *b"stage12.4-int-contrib-B-seed-32!";

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
        available_ram_bytes: 32 * 1024 * 1024 * 1024,
        max_input_tokens: 200_000,
        max_output_tokens: 1_000_000,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens, WorkUnitKind::Layers],
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

// Helper: chunk + sign + send via TensorTransport. Mirrors the
// CLI's `live_send_activation` so we exercise the same shape.
fn send_chunked(
    transport: &mut impl TensorTransport,
    session: &ExecutionSession,
    from: &WorkAssignment,
    to: &WorkAssignment,
    sender: &ContributorSigner,
    full: &[u8],
    chunk_size: usize,
) {
    let byte_len = full.len() as u64;
    let tensor_hash = hex_lower(blake3::hash(full).as_bytes());
    let chunk_count = full.len().div_ceil(chunk_size) as u32;
    for chunk_index in 0..chunk_count {
        let start = chunk_index as usize * chunk_size;
        let end = ((chunk_index + 1) as usize * chunk_size).min(full.len());
        let chunk_bytes = full[start..end].to_vec();
        let mut h = ActivationHandoff {
            schema_version: HANDOFF_SCHEMA_VERSION,
            session_id: session.session_id.clone(),
            from_assignment_id: from.assignment_id.clone(),
            to_assignment_id: to.assignment_id.clone(),
            from_contributor_pubkey_hex: from.contributor_pubkey_hex.clone(),
            to_contributor_pubkey_hex: to.contributor_pubkey_hex.clone(),
            dtype: TensorDtype::F16,
            shape: vec![byte_len / 2],
            byte_len,
            tensor_hash: tensor_hash.clone(),
            chunk_index,
            chunk_count,
            produced_at_utc: "2026-05-27T00:00:30Z".into(),
            tensor_chunk_bytes: chunk_bytes,
            sender_signature_hex: String::new(),
        };
        let si = activation_handoff_signing_input(&h).unwrap();
        h.sender_signature_hex = sender.sign_hex(&si);
        h.validate_schema().unwrap();
        transport.send_handoff(None, &h).unwrap();
    }
}

// Helper: receive all envelopes from a transport, verify each, and
// drive a HandoffReceiver to a Complete outcome.
fn receive_complete(
    transport: &mut impl TensorTransport,
    session: &ExecutionSession,
    upstream: &WorkAssignment,
    this: &WorkAssignment,
) -> Vec<u8> {
    let mut receiver = HandoffReceiver::new();
    let envelopes = transport.poll_handoffs().unwrap();
    for h in envelopes {
        let v_out = verify_activation_handoff(session, upstream, this, &h);
        assert!(v_out.is_ok(), "verify failed: {v_out:?}");
        match receiver.feed(&h) {
            ChunkOutcome::Complete { tensor_bytes } => return tensor_bytes,
            ChunkOutcome::Accepted => {}
            other => panic!("unexpected chunk outcome: {other:?}"),
        }
    }
    panic!("receiver did not reach Complete");
}

// ── A → B happy path, in-memory ──────────────────────────────────────

#[test]
fn inmem_transport_single_chunk_a_to_b_happy_path() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a = ContributorSigner::from_seed_bytes(&A_SEED).unwrap();
    let b = ContributorSigner::from_seed_bytes(&B_SEED).unwrap();
    let session = build_session(&coord);
    let _join_a = build_join(&session, &a);
    let _join_b = build_join(&session, &b);
    let asn_a = build_assignment(&session, &a.pubkey_hex(), 0, &coord);
    let asn_b = build_assignment(&session, &b.pubkey_hex(), 1, &coord);

    let mut transport = InMemoryTensorTransport::new();
    let tensor: Vec<u8> = (0..32u8).cycle().take(64).collect();
    send_chunked(&mut transport, &session, &asn_a, &asn_b, &a, &tensor, 999_999);
    let received = receive_complete(&mut transport, &session, &asn_a, &asn_b);
    assert_eq!(received, tensor);
}

#[test]
fn inmem_transport_multi_chunk_a_to_b_reassembles() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a = ContributorSigner::from_seed_bytes(&A_SEED).unwrap();
    let b = ContributorSigner::from_seed_bytes(&B_SEED).unwrap();
    let session = build_session(&coord);
    let asn_a = build_assignment(&session, &a.pubkey_hex(), 0, &coord);
    let asn_b = build_assignment(&session, &b.pubkey_hex(), 1, &coord);

    let mut transport = InMemoryTensorTransport::new();
    let tensor: Vec<u8> = (0u8..200).collect();
    // Force 4 chunks of ~50 bytes each.
    send_chunked(&mut transport, &session, &asn_a, &asn_b, &a, &tensor, 50);
    let received = receive_complete(&mut transport, &session, &asn_a, &asn_b);
    assert_eq!(received, tensor);
}

// ── Runner integration (StubRunner via run_with_activations) ─────────

#[test]
fn stub_runner_default_run_with_activations_ignores_upstream() {
    // Default impl on InferenceRunner just delegates to run() and
    // returns None for output_activation — Stage 12.0/12.1/12.2/12.3
    // runners stay valid in 12.4 without code changes.
    let a = ContributorSigner::from_seed_bytes(&A_SEED).unwrap();
    let r = StubRunner::new(a.pubkey_hex(), b"hello".to_vec(), 5, 5);
    let upstream = b"upstream-activation".to_vec();
    let with_act = r
        .run_with_activations(std::path::Path::new("/dev/null"), b"", Some(&upstream))
        .unwrap();
    assert_eq!(with_act.run_output.response_bytes, b"hello".to_vec());
    assert!(with_act.output_activation.is_none());
}

// ── End-to-end: A produces activation → B receives it via transport
//    → B's runner consumes it. ──────────────────────────────────────

#[test]
fn end_to_end_a_runs_then_b_consumes_via_transport() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a = ContributorSigner::from_seed_bytes(&A_SEED).unwrap();
    let b = ContributorSigner::from_seed_bytes(&B_SEED).unwrap();
    let session = build_session(&coord);
    let asn_a = build_assignment(&session, &a.pubkey_hex(), 0, &coord);
    let asn_b = build_assignment(&session, &b.pubkey_hex(), 1, &coord);

    // A's runner produces some bytes (stub does fixed_response).
    let a_runner = StubRunner::new(a.pubkey_hex(), b"a-output-bytes".to_vec(), 10, 0);
    let a_out = a_runner
        .run_with_activations(std::path::Path::new("/dev/null"), b"", None)
        .unwrap();
    // 12.4: take the response_bytes as A's "output activation" for
    // the handoff (stub runner doesn't separate them).
    let activation_bytes = a_out.run_output.response_bytes;
    // Round it up to an even number of bytes so f16 element bound
    // holds (handoff shape canonical check).
    let activation_bytes = if activation_bytes.len() % 2 == 0 {
        activation_bytes
    } else {
        let mut v = activation_bytes;
        v.push(0);
        v
    };

    let mut transport = InMemoryTensorTransport::new();
    send_chunked(
        &mut transport,
        &session,
        &asn_a,
        &asn_b,
        &a,
        &activation_bytes,
        4096,
    );

    // B receives + verifies + reassembles.
    let upstream_for_b = receive_complete(&mut transport, &session, &asn_a, &asn_b);
    assert_eq!(upstream_for_b, activation_bytes);

    // B's runner consumes the upstream activation via the default
    // run_with_activations impl. Stub ignores it (documented), but
    // the call signature compiles + the wired path round-trips.
    let b_runner = StubRunner::new(b.pubkey_hex(), b"b-response".to_vec(), 1, 1);
    let b_out = b_runner
        .run_with_activations(
            std::path::Path::new("/dev/null"),
            b"",
            Some(&upstream_for_b),
        )
        .unwrap();
    assert_eq!(b_out.run_output.response_bytes, b"b-response".to_vec());
}

// ── Negative: transport delivers bytes whose hash disagrees with the
//    signed tensor_hash. The receiver's reassembler must catch this
//    even when the per-envelope verifier signature path is clean. ───

#[test]
fn end_to_end_tampered_chunk_bytes_caught_by_reassembler() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a = ContributorSigner::from_seed_bytes(&A_SEED).unwrap();
    let b = ContributorSigner::from_seed_bytes(&B_SEED).unwrap();
    let session = build_session(&coord);
    let asn_a = build_assignment(&session, &a.pubkey_hex(), 0, &coord);
    let asn_b = build_assignment(&session, &b.pubkey_hex(), 1, &coord);

    let tensor: Vec<u8> = vec![0xAB; 16];
    let tensor_hash = hex_lower(blake3::hash(&tensor).as_bytes());
    // Sign over the legit hash but ship different bytes.
    let mut h = ActivationHandoff {
        schema_version: HANDOFF_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        from_assignment_id: asn_a.assignment_id.clone(),
        to_assignment_id: asn_b.assignment_id.clone(),
        from_contributor_pubkey_hex: a.pubkey_hex(),
        to_contributor_pubkey_hex: b.pubkey_hex(),
        dtype: TensorDtype::F16,
        shape: vec![8],
        byte_len: 16,
        tensor_hash,
        chunk_index: 0,
        chunk_count: 1,
        produced_at_utc: "2026-05-27T00:00:30Z".into(),
        tensor_chunk_bytes: tensor.clone(),
        sender_signature_hex: String::new(),
    };
    let si = activation_handoff_signing_input(&h).unwrap();
    h.sender_signature_hex = a.sign_hex(&si);
    // Substitute bytes after signing — sig still verifies (it covers
    // tensor_hash, not the bytes), but the reassembler's hash check
    // will reject.
    h.tensor_chunk_bytes = vec![0xCD; 16];

    let mut transport = InMemoryTensorTransport::new();
    transport.send_handoff(None, &h).unwrap();
    let envelopes = transport.poll_handoffs().unwrap();
    assert_eq!(envelopes.len(), 1);
    let v_out = verify_activation_handoff(&session, &asn_a, &asn_b, &envelopes[0]);
    // Verifier OK — signature binds tensor_hash, not the (now
    // tampered) bytes.
    assert!(v_out.is_ok(), "{v_out:?}");
    let mut receiver = HandoffReceiver::new();
    let outcome = receiver.feed(&envelopes[0]);
    assert!(
        matches!(outcome, ChunkOutcome::TensorHashMismatch { .. }),
        "{outcome:?}"
    );
}

// ── Posture sanity: 12.3 chain still works ─────────────────────────────

#[test]
fn upstream_session_chain_still_works_under_12_4_deps() {
    // Make sure nothing in 12.4 broke the 12.3 verifier chain shape.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let a = ContributorSigner::from_seed_bytes(&A_SEED).unwrap();
    let session = build_session(&coord);
    let join = build_join(&session, &a);
    let asn = build_assignment(&session, &a.pubkey_hex(), 0, &coord);
    assert!(verify_execution_session(&session).is_ok());
    assert!(verify_contributor_join(&session, &join).is_ok());
    let mut joined: HashSet<String> = HashSet::new();
    joined.insert(join.contributor_pubkey_hex.clone());
    assert!(verify_work_assignment(&session, &joined, &asn).is_ok());
    let _ = (
        verify_partial_result,
        HandoffVerifyOutcome::Ok,
    ); // suppress unused
}

// MockSnipStore presence asserts the existing test infrastructure
// hasn't drifted — handoff has no SNIP dependency itself but
// integration tests typically use both.
#[test]
fn mock_snip_store_still_constructible() {
    let _store = MockSnipStore::new();
}
