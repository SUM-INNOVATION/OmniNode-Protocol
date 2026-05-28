//! Stage 12.5 — regression for the verified join loader the CLI's
//! `--joins-dir` path uses. The loader walks the watch-sessions
//! output tree (`<dir>/<session_id>/{session.json, joins/*.json}`)
//! and runs `verify_execution_session` + `verify_contributor_join`
//! on each entry. Schema-only loading was removed in Stage 12.5
//! review #2 because a forged local join file could otherwise let
//! a forged advert pass the matching-join gate.
//!
//! The CLI helper is private; this test exercises the underlying
//! Stage 12.3 verifier semantics it relies on. A "verified loader"
//! is correct iff the underlying join verifier correctly refuses
//! a join whose signature was forged. We also assert the positive
//! path so the test pins what acceptance looks like.

use omni_contributor::{
    canonical::{
        contributor_join_signing_input, execution_session_signing_input, session_id_hex,
    },
    result::WorkUnitKind,
    verify_contributor_join, verify_execution_session, ContributorJoin,
    ContributorSigner, CoordinatorSigner, ExecutionSession, SessionVerifyOutcome,
    SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.5-loader-coord-32-byte!!";
const CONTRIB_SEED: [u8; 32] = *b"stage12.5-loader-contrib-32-byt!";
const ROGUE_SEED: [u8; 32] = *b"stage12.5-loader-rogue-seed-32by";

fn signed_session(coord: &CoordinatorSigner) -> ExecutionSession {
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-05-28T00:00:00Z".into(),
        expires_at_utc: "2026-05-29T00:00:00Z".into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn signed_join(session: &ExecutionSession, contrib: &ContributorSigner) -> ContributorJoin {
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 100,
        max_output_tokens: 100,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-28T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

#[test]
fn verified_loader_accepts_legitimate_join() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session = signed_session(&coord);
    let join = signed_join(&session, &contrib);
    assert!(verify_execution_session(&session).is_ok());
    assert!(verify_contributor_join(&session, &join).is_ok());
}

#[test]
fn verified_loader_rejects_forged_join_with_attacker_signature() {
    // Attack scenario: an attacker drops a `joins/<victim_pk>.json`
    // into the CLI's --joins-dir, claiming to be the victim
    // contributor. They sign it with their own seed. The schema
    // would accept it (hex widths, RFC3339, etc.). The signature
    // does NOT verify against the claimed contributor_pubkey_hex,
    // so the Stage 12.3 verifier rejects it — which is exactly
    // what the CLI's load_verified_joins_from_dir relies on.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let rogue = ContributorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let session = signed_session(&coord);
    let mut forged = signed_join(&session, &contrib);
    // Re-sign with the rogue's key over the same canonical body.
    // The body claims contrib's pubkey but the signature is rogue's.
    let si = contributor_join_signing_input(&forged).unwrap();
    forged.contributor_signature_hex = rogue.sign_hex(&si);
    let out = verify_contributor_join(&session, &forged);
    assert!(
        matches!(out, SessionVerifyOutcome::ContributorSignatureFailed),
        "{out:?}"
    );
}

#[test]
fn verified_loader_rejects_join_for_wrong_session() {
    // Attack scenario: a legitimately-signed join for one session
    // is dropped into a directory belonging to a different session.
    // The loader pairs joins with the sibling session.json — so
    // the binding check inside `verify_contributor_join` catches
    // the mismatch.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let session_a = signed_session(&coord);
    let mut session_b = signed_session(&coord);
    // Make session_b distinct by giving it a different posted_id
    // and re-deriving the session_id + signature.
    session_b.posted_id = "ff".repeat(32);
    session_b.session_id = session_id_hex(&session_b).unwrap();
    let si = execution_session_signing_input(&session_b).unwrap();
    session_b.coordinator_signature_hex = coord.sign_hex(&si);
    let join_for_a = signed_join(&session_a, &contrib);
    let out = verify_contributor_join(&session_b, &join_for_a);
    assert!(
        matches!(out, SessionVerifyOutcome::BindingMismatch { .. }),
        "{out:?}"
    );
}
