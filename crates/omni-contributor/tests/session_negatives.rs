//! Stage 12.3 — schema + signature + drift negative tests for the
//! 5 inner session envelopes and `WorkKind`.

use std::collections::HashSet;

use omni_contributor::{
    canonical::{
        aggregated_result_signing_input, assignment_id_hex, contributor_join_signing_input,
        canonical_partial_result_bytes, execution_session_signing_input, hex_lower,
        partial_result_signing_input, session_id_hex, work_assignment_signing_input,
    },
    error::SchemaError,
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    verify_aggregated_result, verify_contributor_join, verify_execution_session,
    verify_partial_result, verify_work_assignment, AggregatedContributorResult,
    AggregatedPartialRef, ContributorJoin, ContributorSigner, CoordinatorSigner,
    ExecutionSession, PartialContributorResult, SessionVerifyOutcome, WorkAssignment,
    WorkKind, SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.3-coord-seed-32-byte-key";
const CONTRIB_SEED: [u8; 32] = *b"stage12.3-contrib-seed-32-bytes!";
const ROGUE_SEED: [u8; 32] = *b"stage12.3-rogue-seed-32-byteskey";

fn signed_session(coord: &CoordinatorSigner) -> ExecutionSession {
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
    let sig_input = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&sig_input);
    s
}

fn signed_join(session: &ExecutionSession, contrib: &ContributorSigner) -> ContributorJoin {
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 16 * 1024 * 1024 * 1024,
        max_input_tokens: 200_000,
        max_output_tokens: 1_000_000,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-27T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

fn signed_assignment(
    session: &ExecutionSession,
    join: &ContributorJoin,
    coord: &CoordinatorSigner,
) -> WorkAssignment {
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index: 0,
        contributor_pubkey_hex: join.contributor_pubkey_hex.clone(),
        work_kind: WorkKind::Prefill,
        expected_work_units: 200_000,
        expected_work_unit_kind: WorkUnitKind::PrefillTokens,
        assigned_at_utc: "2026-05-27T00:00:02Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    a
}

fn signed_partial(
    assignment: &WorkAssignment,
    contrib: &ContributorSigner,
) -> PartialContributorResult {
    let mut p = PartialContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: assignment.session_id.clone(),
        assignment_id: assignment.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        partial_artifact_snip_root: format!("0x{}", "aa".repeat(32)),
        partial_artifact_hash: hex_lower(blake3::hash(b"partial-art").as_bytes()),
        measured_accounting: MeasuredAccounting {
            tokenizer_hash: "44".repeat(32),
            input_token_count: 200_000,
            output_token_count: 0,
            total_base_units: 200_000,
            stage_contributions: vec![StageContribution {
                contributor_pubkey_hex: contrib.pubkey_hex(),
                stage_label: "prefill".into(),
                work_unit_kind: WorkUnitKind::PrefillTokens,
                work_units: 200_000,
            }],
        },
        produced_at_utc: "2026-05-27T00:00:10Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = partial_result_signing_input(&p).unwrap();
    p.contributor_signature_hex = contrib.sign_hex(&si);
    p
}

fn signed_aggregate(
    session: &ExecutionSession,
    assignments: &[WorkAssignment],
    partials: &[PartialContributorResult],
    coord: &CoordinatorSigner,
) -> AggregatedContributorResult {
    let partial_refs = assignments
        .iter()
        .zip(partials.iter())
        .map(|(a, p)| {
            let bytes = canonical_partial_result_bytes(p).unwrap();
            AggregatedPartialRef {
                assignment_id: a.assignment_id.clone(),
                stage_index: a.stage_index,
                contributor_pubkey_hex: p.contributor_pubkey_hex.clone(),
                partial_snip_root: format!("0x{}", "cc".repeat(32)),
                partial_canonical_hash: hex_lower(blake3::hash(&bytes).as_bytes()),
            }
        })
        .collect();
    let mut g = AggregatedContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        final_result_snip_root: format!("0x{}", "bb".repeat(32)),
        final_result_canonical_hash: hex_lower(blake3::hash(b"final").as_bytes()),
        partial_refs,
        aggregated_at_utc: "2026-05-27T00:00:30Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let si = aggregated_result_signing_input(&g).unwrap();
    g.coordinator_signature_hex = coord.sign_hex(&si);
    g
}

// ── WorkKind validation ─────────────────────────────────────────────

#[test]
fn workkind_custom_label_empty_refused() {
    let wk = WorkKind::Custom { label: String::new() };
    assert!(matches!(
        wk.validate_schema(),
        Err(SchemaError::WorkKindCustomEmptyLabel)
    ));
}

#[test]
fn workkind_custom_label_too_long_refused() {
    let wk = WorkKind::Custom { label: "x".repeat(65) };
    assert!(matches!(
        wk.validate_schema(),
        Err(SchemaError::WorkKindCustomLabelTooLong { .. })
    ));
}

#[test]
fn workkind_custom_label_non_ascii_refused() {
    let wk = WorkKind::Custom { label: "ka\u{00e9}".into() };
    assert!(matches!(
        wk.validate_schema(),
        Err(SchemaError::WorkKindCustomLabelNotPrintableAscii)
    ));
}

#[test]
fn workkind_custom_label_control_char_refused() {
    let wk = WorkKind::Custom { label: "ka\nshard".into() };
    assert!(matches!(
        wk.validate_schema(),
        Err(SchemaError::WorkKindCustomLabelNotPrintableAscii)
    ));
}

#[test]
fn workkind_layers_inverted_refused() {
    let wk = WorkKind::Layers { start: 10, end: 5 };
    assert!(matches!(
        wk.validate_schema(),
        Err(SchemaError::WorkKindLayersInverted { .. })
    ));
}

#[test]
fn workkind_layers_zero_width_refused() {
    let wk = WorkKind::Layers { start: 5, end: 5 };
    assert!(matches!(
        wk.validate_schema(),
        Err(SchemaError::WorkKindLayersInverted { .. })
    ));
}

// ── ExecutionSession ─────────────────────────────────────────────────

#[test]
fn session_tampered_signature_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = signed_session(&coord);
    let mut sig = s.coordinator_signature_hex.into_bytes();
    sig[0] = if sig[0] == b'a' { b'b' } else { b'a' };
    s.coordinator_signature_hex = String::from_utf8(sig).unwrap();
    let out = verify_execution_session(&s);
    assert!(matches!(out, SessionVerifyOutcome::CoordinatorSignatureFailed), "{out:?}");
}

#[test]
fn session_wrong_signer_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let mut s = signed_session(&coord);
    // Re-sign with the rogue's key — bytes-valid signature, wrong key.
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = rogue.sign_hex(&si);
    let out = verify_execution_session(&s);
    assert!(matches!(out, SessionVerifyOutcome::CoordinatorSignatureFailed), "{out:?}");
}

#[test]
fn session_id_mismatch_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = signed_session(&coord);
    s.session_id = "00".repeat(32);
    // Re-sign so the signature check passes; only session_id-mismatch trips.
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    let out = verify_execution_session(&s);
    assert!(matches!(out, SessionVerifyOutcome::SessionIdMismatch { .. }), "{out:?}");
}

#[test]
fn session_schema_bad_hex_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = signed_session(&coord);
    s.posted_id = "short".into();
    let out = verify_execution_session(&s);
    assert!(matches!(out, SessionVerifyOutcome::SchemaMalformed(_)), "{out:?}");
}

#[test]
fn session_schema_bad_timestamp_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = signed_session(&coord);
    s.created_at_utc = "not-a-timestamp".into();
    let out = verify_execution_session(&s);
    assert!(matches!(out, SessionVerifyOutcome::SchemaMalformed(_)), "{out:?}");
}

// ── ContributorJoin ─────────────────────────────────────────────────

#[test]
fn join_drift_session_id_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut j = signed_join(&s, &contrib);
    j.session_id = "ff".repeat(32);
    // Re-sign so signature is valid; binding mismatch trips.
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    let out = verify_contributor_join(&s, &j);
    assert!(matches!(out, SessionVerifyOutcome::BindingMismatch { .. }), "{out:?}");
}

#[test]
fn join_empty_work_unit_kinds_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut j = signed_join(&s, &contrib);
    j.supported_work_unit_kinds.clear();
    let out = verify_contributor_join(&s, &j);
    assert!(matches!(out, SessionVerifyOutcome::SchemaMalformed(_)), "{out:?}");
}

#[test]
fn join_empty_runner_kind_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut j = signed_join(&s, &contrib);
    j.runner_kind = String::new();
    let out = verify_contributor_join(&s, &j);
    assert!(matches!(out, SessionVerifyOutcome::SchemaMalformed(_)), "{out:?}");
}

#[test]
fn join_tampered_signature_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let mut j = signed_join(&s, &contrib);
    let mut sig = j.contributor_signature_hex.into_bytes();
    sig[5] = if sig[5] == b'1' { b'2' } else { b'1' };
    j.contributor_signature_hex = String::from_utf8(sig).unwrap();
    let out = verify_contributor_join(&s, &j);
    assert!(matches!(out, SessionVerifyOutcome::ContributorSignatureFailed), "{out:?}");
}

// ── WorkAssignment ───────────────────────────────────────────────────

#[test]
fn assignment_not_joined_contributor_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let joined_pubkeys: HashSet<String> = HashSet::new(); // empty — not joined
    let out = verify_work_assignment(&s, &joined_pubkeys, &a);
    assert!(matches!(out, SessionVerifyOutcome::BindingMismatch { .. }), "{out:?}");
}

#[test]
fn assignment_signed_by_non_coordinator_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let mut a = signed_assignment(&s, &j, &coord);
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = rogue.sign_hex(&si);
    let mut joined = HashSet::new();
    joined.insert(j.contributor_pubkey_hex.clone());
    let out = verify_work_assignment(&s, &joined, &a);
    assert!(matches!(out, SessionVerifyOutcome::CoordinatorSignatureFailed), "{out:?}");
}

#[test]
fn assignment_zero_work_units_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let mut a = signed_assignment(&s, &j, &coord);
    a.expected_work_units = 0;
    let mut joined = HashSet::new();
    joined.insert(j.contributor_pubkey_hex.clone());
    let out = verify_work_assignment(&s, &joined, &a);
    assert!(matches!(out, SessionVerifyOutcome::SchemaMalformed(_)), "{out:?}");
}

#[test]
fn assignment_drift_session_id_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let mut a = signed_assignment(&s, &j, &coord);
    a.session_id = "ee".repeat(32);
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&si);
    let mut joined = HashSet::new();
    joined.insert(j.contributor_pubkey_hex.clone());
    let out = verify_work_assignment(&s, &joined, &a);
    assert!(matches!(out, SessionVerifyOutcome::BindingMismatch { .. }), "{out:?}");
}

// ── PartialContributorResult ─────────────────────────────────────────

#[test]
fn partial_wrong_assignment_id_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let mut p = signed_partial(&a, &contrib);
    p.assignment_id = "ff".repeat(32);
    // Re-sign so signature is valid; binding mismatch trips.
    let si = partial_result_signing_input(&p).unwrap();
    p.contributor_signature_hex = contrib.sign_hex(&si);
    let out = verify_partial_result(&a, &p);
    assert!(matches!(out, SessionVerifyOutcome::BindingMismatch { .. }), "{out:?}");
}

#[test]
fn partial_wrong_contributor_pubkey_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let rogue = ContributorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let mut p = signed_partial(&a, &contrib);
    p.contributor_pubkey_hex = rogue.pubkey_hex();
    // Update the stage_contribution to match (so schema passes).
    p.measured_accounting.stage_contributions[0].contributor_pubkey_hex = rogue.pubkey_hex();
    let si = partial_result_signing_input(&p).unwrap();
    p.contributor_signature_hex = rogue.sign_hex(&si);
    let out = verify_partial_result(&a, &p);
    assert!(matches!(out, SessionVerifyOutcome::BindingMismatch { .. }), "{out:?}");
}

#[test]
fn partial_multiple_stage_contributions_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let mut p = signed_partial(&a, &contrib);
    let extra = p.measured_accounting.stage_contributions[0].clone();
    p.measured_accounting.stage_contributions.push(extra);
    let out = verify_partial_result(&a, &p);
    assert!(matches!(out, SessionVerifyOutcome::SchemaMalformed(_)), "{out:?}");
}

#[test]
fn partial_tampered_signature_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let mut p = signed_partial(&a, &contrib);
    let mut sig = p.contributor_signature_hex.into_bytes();
    sig[10] = if sig[10] == b'a' { b'b' } else { b'a' };
    p.contributor_signature_hex = String::from_utf8(sig).unwrap();
    let out = verify_partial_result(&a, &p);
    assert!(matches!(out, SessionVerifyOutcome::ContributorSignatureFailed), "{out:?}");
}

// ── AggregatedContributorResult ──────────────────────────────────────

#[test]
fn aggregate_missing_partial_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let p = signed_partial(&a, &contrib);
    // Build a SECOND assignment whose partial is missing.
    let mut a2 = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: s.session_id.clone(),
        assignment_id: String::new(),
        stage_index: 1,
        contributor_pubkey_hex: j.contributor_pubkey_hex.clone(),
        work_kind: WorkKind::Decode,
        expected_work_units: 1_000_000,
        expected_work_unit_kind: WorkUnitKind::DecodeTokens,
        assigned_at_utc: "2026-05-27T00:00:03Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a2.assignment_id = assignment_id_hex(&a2).unwrap();
    let si = work_assignment_signing_input(&a2).unwrap();
    a2.coordinator_signature_hex = coord.sign_hex(&si);
    // Aggregate references only the first assignment's partial.
    let g = signed_aggregate(&s, &[a.clone()], &[p.clone()], &coord);
    let out = verify_aggregated_result(&s, &[j], &[a, a2], &[p], &g);
    assert!(matches!(out, SessionVerifyOutcome::AggregateMissingPartialFor { .. }), "{out:?}");
}

#[test]
fn aggregate_coordinator_mismatch_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let p = signed_partial(&a, &contrib);
    let g = signed_aggregate(&s, &[a.clone()], &[p.clone()], &rogue);
    let out = verify_aggregated_result(&s, &[j], &[a], &[p], &g);
    assert!(matches!(out, SessionVerifyOutcome::AggregateCoordinatorMismatch), "{out:?}");
}

#[test]
fn aggregate_partial_ref_drift_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let p = signed_partial(&a, &contrib);
    let mut g = signed_aggregate(&s, &[a.clone()], &[p.clone()], &coord);
    // Mutate the partial_canonical_hash on the ref to something else.
    g.partial_refs[0].partial_canonical_hash = "ff".repeat(32);
    let si = aggregated_result_signing_input(&g).unwrap();
    g.coordinator_signature_hex = coord.sign_hex(&si);
    let out = verify_aggregated_result(&s, &[j], &[a], &[p], &g);
    assert!(matches!(out, SessionVerifyOutcome::AggregatePartialRefDrift { .. }), "{out:?}");
}

#[test]
fn aggregate_empty_partial_refs_refused() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let p = signed_partial(&a, &contrib);
    let mut g = signed_aggregate(&s, &[a.clone()], &[p.clone()], &coord);
    g.partial_refs.clear();
    let out = verify_aggregated_result(&s, &[j], &[a], &[p], &g);
    assert!(matches!(out, SessionVerifyOutcome::SchemaMalformed(_)), "{out:?}");
}

#[test]
fn aggregate_happy_path_verifies() {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let p = signed_partial(&a, &contrib);
    let g = signed_aggregate(&s, &[a.clone()], &[p.clone()], &coord);
    assert!(verify_execution_session(&s).is_ok());
    assert!(verify_contributor_join(&s, &j).is_ok());
    let mut joined = HashSet::new();
    joined.insert(j.contributor_pubkey_hex.clone());
    assert!(verify_work_assignment(&s, &joined, &a).is_ok());
    assert!(verify_partial_result(&a, &p).is_ok());
    let out = verify_aggregated_result(&s, &[j], &[a], &[p], &g);
    assert!(out.is_ok(), "{out:?}");
}

#[test]
fn aggregate_chain_refused_when_assignment_contributor_never_joined() {
    // The aggregate, on its own, looks clean (refs+coordinator sig
    // verify). But the assignment targets a contributor who never
    // submitted a join. The widened verifier must catch this via
    // the joined-set check inside the chain.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    let a = signed_assignment(&s, &j, &coord);
    let p = signed_partial(&a, &contrib);
    let g = signed_aggregate(&s, &[a.clone()], &[p.clone()], &coord);
    // Pass an EMPTY joins vec — assignment.contributor_pubkey_hex
    // is not in the joined set the verifier computes.
    let out = verify_aggregated_result(&s, &[], &[a], &[p], &g);
    assert!(matches!(out, SessionVerifyOutcome::BindingMismatch { .. }), "{out:?}");
}

#[test]
fn aggregate_chain_refused_when_assignment_signed_by_rogue() {
    // Aggregate's own coordinator sig verifies, but an assignment
    // inside it was signed by a different key than the session's
    // coordinator. The widened chain check must catch this even
    // when ref+aggregate-sig fields are clean.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let rogue = CoordinatorSigner::from_seed_bytes(&ROGUE_SEED).unwrap();
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIB_SEED).unwrap();
    let s = signed_session(&coord);
    let j = signed_join(&s, &contrib);
    // Assignment signed by rogue (NOT session's coordinator).
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: s.session_id.clone(),
        assignment_id: String::new(),
        stage_index: 0,
        contributor_pubkey_hex: contrib.pubkey_hex(),
        work_kind: WorkKind::Prefill,
        expected_work_units: 100,
        expected_work_unit_kind: WorkUnitKind::PrefillTokens,
        assigned_at_utc: "2026-05-27T00:00:02Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let si = work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = rogue.sign_hex(&si);
    let p = signed_partial(&a, &contrib);
    // Coordinator (legitimate) signs the aggregate over this
    // rogue-signed assignment.
    let g = signed_aggregate(&s, &[a.clone()], &[p.clone()], &coord);
    let out = verify_aggregated_result(&s, &[j], &[a], &[p], &g);
    assert!(matches!(out, SessionVerifyOutcome::CoordinatorSignatureFailed), "{out:?}");
}

// ── Time bounds ──────────────────────────────────────────────────────

#[test]
fn check_not_expired_after_expiry_refused() {
    let out = omni_contributor::check_not_expired(
        "2026-05-27T02:00:00Z",
        "2026-05-27T01:00:00Z",
    );
    assert!(matches!(out, SessionVerifyOutcome::ExpiredAtCheck { .. }), "{out:?}");
}

#[test]
fn check_not_expired_before_expiry_passes() {
    let out = omni_contributor::check_not_expired(
        "2026-05-27T00:30:00Z",
        "2026-05-27T01:00:00Z",
    );
    assert!(out.is_ok(), "{out:?}");
}
