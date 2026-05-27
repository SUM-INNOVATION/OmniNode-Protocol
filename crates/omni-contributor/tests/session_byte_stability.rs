//! Stage 12.3 — byte-stability tests for the 5 inner session
//! envelopes. Hashes are pinned so canonical bytes can't drift
//! silently. A reorder of fields on any *CanonicalBody struct in
//! `canonical.rs`, or a change to a domain separator, will trip one
//! of these.

use omni_contributor::{
    canonical::{
        assignment_id_hex, canonical_aggregated_result_bytes,
        canonical_contributor_join_bytes, canonical_execution_session_bytes,
        canonical_partial_result_bytes, canonical_work_assignment_bytes, hex_lower,
        session_id_hex,
    },
    result::{MeasuredAccounting, StageContribution, WorkUnitKind},
    AggregatedContributorResult, AggregatedPartialRef, ContributorJoin, ContributorSigner,
    CoordinatorSigner, ExecutionSession, PartialContributorResult, WorkAssignment,
    WorkKind, SESSION_SCHEMA_VERSION,
};

const COORDINATOR_SEED: [u8; 32] = *b"stage12.3-coord-seed-32-byte-key";
const CONTRIBUTOR_SEED: [u8; 32] = *b"stage12.3-contrib-seed-32-bytes!";

fn build_minimal_session() -> ExecutionSession {
    let coord = CoordinatorSigner::from_seed_bytes(&COORDINATOR_SEED).unwrap();
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
    let signing_input =
        omni_contributor::canonical::execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&signing_input);
    s
}

fn build_minimal_join(session: &ExecutionSession) -> ContributorJoin {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes: 64 * 1024 * 1024 * 1024,
        max_input_tokens: 200_000,
        max_output_tokens: 1_000_000,
        supported_work_unit_kinds: vec![WorkUnitKind::Tokens, WorkUnitKind::PrefillTokens],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-27T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let signing_input =
        omni_contributor::canonical::contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&signing_input);
    j
}

fn build_minimal_assignment(
    session: &ExecutionSession,
    join: &ContributorJoin,
) -> WorkAssignment {
    let coord = CoordinatorSigner::from_seed_bytes(&COORDINATOR_SEED).unwrap();
    let mut a = WorkAssignment {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: String::new(),
        stage_index: 0,
        contributor_pubkey_hex: join.contributor_pubkey_hex.clone(),
        work_kind: WorkKind::Layers { start: 0, end: 32 },
        expected_work_units: 1_200_000,
        expected_work_unit_kind: WorkUnitKind::Tokens,
        assigned_at_utc: "2026-05-27T00:00:02Z".into(),
        coordinator_signature_hex: String::new(),
    };
    a.assignment_id = assignment_id_hex(&a).unwrap();
    let signing_input =
        omni_contributor::canonical::work_assignment_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&signing_input);
    a
}

fn build_minimal_partial(
    session: &ExecutionSession,
    assignment: &WorkAssignment,
) -> PartialContributorResult {
    let contrib = ContributorSigner::from_seed_bytes(&CONTRIBUTOR_SEED).unwrap();
    let mut p = PartialContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        assignment_id: assignment.assignment_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        partial_artifact_snip_root: format!("0x{}", "aa".repeat(32)),
        partial_artifact_hash: hex_lower(blake3::hash(b"partial-artifact").as_bytes()),
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
    let signing_input =
        omni_contributor::canonical::partial_result_signing_input(&p).unwrap();
    p.contributor_signature_hex = contrib.sign_hex(&signing_input);
    p
}

fn build_minimal_aggregate(
    session: &ExecutionSession,
    assignment: &WorkAssignment,
    partial: &PartialContributorResult,
) -> AggregatedContributorResult {
    let coord = CoordinatorSigner::from_seed_bytes(&COORDINATOR_SEED).unwrap();
    let partial_bytes =
        omni_contributor::canonical::canonical_partial_result_bytes(partial).unwrap();
    let partial_hash = hex_lower(blake3::hash(&partial_bytes).as_bytes());
    let mut a = AggregatedContributorResult {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        posted_id: session.posted_id.clone(),
        final_result_snip_root: format!("0x{}", "bb".repeat(32)),
        final_result_canonical_hash: hex_lower(blake3::hash(b"final-canonical").as_bytes()),
        partial_refs: vec![AggregatedPartialRef {
            assignment_id: assignment.assignment_id.clone(),
            stage_index: assignment.stage_index,
            contributor_pubkey_hex: partial.contributor_pubkey_hex.clone(),
            partial_snip_root: format!("0x{}", "cc".repeat(32)),
            partial_canonical_hash: partial_hash,
        }],
        aggregated_at_utc: "2026-05-27T00:00:30Z".into(),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        coordinator_signature_hex: String::new(),
    };
    let signing_input =
        omni_contributor::canonical::aggregated_result_signing_input(&a).unwrap();
    a.coordinator_signature_hex = coord.sign_hex(&signing_input);
    a
}

#[test]
fn execution_session_canonical_bytes_blake3_is_pinned() {
    let s = build_minimal_session();
    let bytes = canonical_execution_session_bytes(&s).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "3c54834b4064905b181bac1cad746c6c60b57744f3513a33e02f0fa59a5dc227",
        "drift in canonical_execution_session_bytes — recompute and re-pin"
    );
}

#[test]
fn contributor_join_canonical_bytes_blake3_is_pinned() {
    let s = build_minimal_session();
    let j = build_minimal_join(&s);
    let bytes = canonical_contributor_join_bytes(&j).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "3315079bd970274ed610f9c0bad85aae0e115bdcfd049b439105c885f27b5391",
        "drift in canonical_contributor_join_bytes — recompute and re-pin"
    );
}

#[test]
fn work_assignment_canonical_bytes_blake3_is_pinned() {
    let s = build_minimal_session();
    let j = build_minimal_join(&s);
    let a = build_minimal_assignment(&s, &j);
    let bytes = canonical_work_assignment_bytes(&a).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "a5c45de0c9d941ffa517b8b0213140dea864014102f857210823465b1a700a19",
        "drift in canonical_work_assignment_bytes — recompute and re-pin"
    );
}

#[test]
fn partial_result_canonical_bytes_blake3_is_pinned() {
    let s = build_minimal_session();
    let j = build_minimal_join(&s);
    let a = build_minimal_assignment(&s, &j);
    let p = build_minimal_partial(&s, &a);
    let bytes = canonical_partial_result_bytes(&p).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "231280623615cb993b0c1e771ee677723ce2b7eab07e90d5067ebbab08568ece",
        "drift in canonical_partial_result_bytes — recompute and re-pin"
    );
}

#[test]
fn aggregated_result_canonical_bytes_blake3_is_pinned() {
    let s = build_minimal_session();
    let j = build_minimal_join(&s);
    let a = build_minimal_assignment(&s, &j);
    let p = build_minimal_partial(&s, &a);
    let g = build_minimal_aggregate(&s, &a, &p);
    let bytes = canonical_aggregated_result_bytes(&g).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "e02d530db5cd7eb4b779a143e086b82be2de31763418a12295fafe5d78eac12e",
        "drift in canonical_aggregated_result_bytes — recompute and re-pin"
    );
}

#[test]
fn session_id_is_derived_from_canonical_bytes() {
    let s = build_minimal_session();
    let derived = session_id_hex(&s).unwrap();
    assert_eq!(s.session_id, derived);
}

#[test]
fn assignment_id_is_derived_from_canonical_bytes() {
    let s = build_minimal_session();
    let j = build_minimal_join(&s);
    let a = build_minimal_assignment(&s, &j);
    let derived = assignment_id_hex(&a).unwrap();
    assert_eq!(a.assignment_id, derived);
}
