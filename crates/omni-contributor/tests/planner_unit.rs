//! Stage 12.8 — unit tests for the local assignment planner.
//!
//! These tests pin the determinism contract (deterministic
//! pubkey-sorted selection, no ranking, no clock dependency outside
//! the explicit `now_utc` for live-routing advert expiry), the
//! eligibility-filter posture (RAM floor pass/fail, no scoring), the
//! strategy correctness (single / sequential / round-robin), the
//! model-plan + layer-count fallback shape, and the `plan_hash`
//! stability across serde round-trips.

use omni_contributor::{
    canonical::{
        advertisement_id_hex, contributor_join_signing_input,
        execution_session_signing_input, peer_advertisement_signing_input,
        session_id_hex,
    },
    handoff::TensorDtype,
    plan_assignments, plan_hash_hex,
    planner::{ModelPlan, ModelPlanStage, PlannerInputs, PlannerStrategy},
    result::WorkUnitKind,
    session::WorkKind,
    AssignmentPlan, ContributorJoin, ContributorPeerAdvertisement, ContributorSigner,
    CoordinatorSigner, ExecutionSession, PeerCapabilities, PlannerError,
    MODEL_PLAN_SCHEMA_VERSION, PEER_ADVERTISEMENT_SCHEMA_VERSION, PLANNER_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
};

const COORD_SEED: [u8; 32] = *b"stage12.8-planner-coord-seed-32!";

fn build_session() -> ExecutionSession {
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-05-31T00:00:00Z".into(),
        expires_at_utc: "2026-06-01T00:00:00Z".into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    s
}

fn build_join(
    session: &ExecutionSession,
    seed: [u8; 32],
    available_ram_bytes: u64,
) -> ContributorJoin {
    let contrib = ContributorSigner::from_seed_bytes(&seed).unwrap();
    let mut j = ContributorJoin {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        available_ram_bytes,
        max_input_tokens: 1024,
        max_output_tokens: 1024,
        supported_work_unit_kinds: vec![WorkUnitKind::Layers],
        runner_kind: "stub".into(),
        joined_at_utc: "2026-05-31T00:00:01Z".into(),
        contributor_signature_hex: String::new(),
    };
    let si = contributor_join_signing_input(&j).unwrap();
    j.contributor_signature_hex = contrib.sign_hex(&si);
    j
}

fn build_advert(
    session: &ExecutionSession,
    contrib_seed: [u8; 32],
    expires_at_utc: &str,
    supports_live_handoff: bool,
    supported_dtypes: Vec<TensorDtype>,
) -> ContributorPeerAdvertisement {
    let contrib = ContributorSigner::from_seed_bytes(&contrib_seed).unwrap();
    let mut a = ContributorPeerAdvertisement {
        schema_version: PEER_ADVERTISEMENT_SCHEMA_VERSION,
        advertisement_id: String::new(),
        session_id: session.session_id.clone(),
        contributor_pubkey_hex: contrib.pubkey_hex(),
        libp2p_peer_id: "12D3KooWGjMyD3v8UYjK3bV7Sqg3WcWzkF7QbTtwYNRsBwCb6KpC".into(),
        listen_multiaddrs: vec![],
        capabilities: PeerCapabilities {
            supports_live_handoff,
            max_handoff_chunk_bytes: 32 * 1024 * 1024,
            supported_dtypes,
        },
        advertised_at_utc: "2026-05-31T00:00:00Z".into(),
        expires_at_utc: expires_at_utc.into(),
        contributor_signature_hex: String::new(),
    };
    a.advertisement_id = advertisement_id_hex(&a).unwrap();
    let si = peer_advertisement_signing_input(&a).unwrap();
    a.contributor_signature_hex = contrib.sign_hex(&si);
    a
}

const NOW_UTC: &str = "2026-05-31T00:30:00Z";
const FAR_FUTURE: &str = "2026-06-01T00:00:00Z";

// ── single-contributor strategy ───────────────────────────────────

#[test]
fn single_contributor_picks_one_eligible() {
    let session = build_session();
    let joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
    ];
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    assert_eq!(plan.assignments.len(), 1);
    assert_eq!(plan.assignments[0].stage_index, 0);
    assert_eq!(
        plan.assignments[0].work_kind,
        WorkKind::Layers { start: 0, end: 8 }
    );
    // First in lex order of (sorted) pubkeys wins.
    let mut pubkeys: Vec<&String> =
        joins.iter().map(|j| &j.contributor_pubkey_hex).collect();
    pubkeys.sort();
    assert_eq!(
        plan.assignments[0].contributor_pubkey_hex,
        *pubkeys[0]
    );
}

// ── sequential-layers strategy ────────────────────────────────────

#[test]
fn sequential_layers_equal_split_contiguous_non_overlapping() {
    let session = build_session();
    let joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-C-seed-32-bytes!", 1 << 30),
    ];
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(32),
        },
        PlannerStrategy::SequentialLayers,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    assert_eq!(plan.assignments.len(), 3);
    // Base = 32 / 3 = 10; last stage absorbs the remainder.
    let layers: Vec<(u32, u32)> = plan
        .assignments
        .iter()
        .map(|a| match a.work_kind {
            WorkKind::Layers { start, end } => (start, end),
            _ => panic!("expected WorkKind::Layers"),
        })
        .collect();
    assert_eq!(layers, vec![(0, 10), (10, 20), (20, 32)]);
    // Stages strictly increasing and non-overlapping.
    let mut prev_end = 0u32;
    for (start, end) in &layers {
        assert_eq!(*start, prev_end);
        assert!(end > start);
        prev_end = *end;
    }
    assert_eq!(prev_end, 32);
}

#[test]
fn sequential_layers_rejects_split_smaller_than_contributors() {
    let session = build_session();
    let joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-C-seed-32-bytes!", 1 << 30),
    ];
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(2),
        },
        PlannerStrategy::SequentialLayers,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        PlannerError::EqualSplitProducesEmptyStage {
            layer_count: 2,
            contributor_count: 3
        }
    ));
}

#[test]
fn sequential_layers_uses_model_plan_when_supplied() {
    let session = build_session();
    let joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
    ];
    let mp = ModelPlan {
        schema_version: MODEL_PLAN_SCHEMA_VERSION,
        stages: vec![
            ModelPlanStage {
                stage_index: 0,
                work_kind: WorkKind::Layers { start: 0, end: 16 },
                expected_work_units: 16,
                expected_work_unit_kind: WorkUnitKind::Layers,
            },
            ModelPlanStage {
                stage_index: 1,
                work_kind: WorkKind::Layers {
                    start: 16,
                    end: 24,
                },
                expected_work_units: 8,
                expected_work_unit_kind: WorkUnitKind::Layers,
            },
            ModelPlanStage {
                stage_index: 2,
                work_kind: WorkKind::Layers {
                    start: 24,
                    end: 32,
                },
                expected_work_units: 8,
                expected_work_unit_kind: WorkUnitKind::Layers,
            },
        ],
    };
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: None,
        },
        PlannerStrategy::SequentialLayers,
        Some(&mp),
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    assert_eq!(plan.assignments.len(), 3);
    // 3 stages over 2 contributors → stages 0 and 2 to contrib[0],
    // stage 1 to contrib[1] (sorted by pubkey).
    let mut sorted = joins.clone();
    sorted.sort_by(|a, b| a.contributor_pubkey_hex.cmp(&b.contributor_pubkey_hex));
    assert_eq!(
        plan.assignments[0].contributor_pubkey_hex,
        sorted[0].contributor_pubkey_hex
    );
    assert_eq!(
        plan.assignments[1].contributor_pubkey_hex,
        sorted[1].contributor_pubkey_hex
    );
    assert_eq!(
        plan.assignments[2].contributor_pubkey_hex,
        sorted[0].contributor_pubkey_hex
    );
    // Work-kind comes verbatim from the model-plan.
    assert_eq!(
        plan.assignments[0].work_kind,
        WorkKind::Layers { start: 0, end: 16 }
    );
}

// ── round-robin strategy ──────────────────────────────────────────

#[test]
fn round_robin_emits_deterministic_contributor_sequence() {
    let session = build_session();
    let joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
    ];
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(5),
        },
        PlannerStrategy::RoundRobin,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    assert_eq!(plan.assignments.len(), 5);
    let mut sorted = joins.clone();
    sorted.sort_by(|a, b| a.contributor_pubkey_hex.cmp(&b.contributor_pubkey_hex));
    // Stage 0 → sorted[0], 1 → sorted[1], 2 → sorted[0], ...
    for (i, a) in plan.assignments.iter().enumerate() {
        assert_eq!(a.stage_index, i as u32);
        assert_eq!(
            a.contributor_pubkey_hex,
            sorted[i % sorted.len()].contributor_pubkey_hex
        );
        assert_eq!(
            a.work_kind,
            WorkKind::Layers {
                start: i as u32,
                end: i as u32 + 1
            }
        );
    }
}

// ── Determinism / plan_hash ──────────────────────────────────────

#[test]
fn plan_deterministic_under_input_shuffle_and_hash_round_trips() {
    let session = build_session();
    let mut joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-C-seed-32-bytes!", 1 << 30),
    ];
    let p1 = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(24),
        },
        PlannerStrategy::SequentialLayers,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    // Reverse input order and replan: must yield byte-identical
    // assignments (deterministic post-filter sort).
    joins.reverse();
    let p2 = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(24),
        },
        PlannerStrategy::SequentialLayers,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    assert_eq!(p1.assignments, p2.assignments);
    assert_eq!(p1.plan_hash, p2.plan_hash);

    // Hash survives a serde JSON round-trip.
    let json = serde_json::to_vec(&p1).unwrap();
    let back: AssignmentPlan = serde_json::from_slice(&json).unwrap();
    assert_eq!(back.plan_hash, plan_hash_hex(&back));
    assert_eq!(back, p1);
    assert_eq!(back.schema_version, PLANNER_SCHEMA_VERSION);
}

// ── Eligibility filter ──────────────────────────────────────────

#[test]
fn ram_floor_filters_below_threshold_pass_fail() {
    let session = build_session();
    let joins = vec![
        build_join(
            &session,
            *b"planner-contrib-A-seed-32-bytes!",
            8 * 1024 * 1024 * 1024,
        ),
        build_join(
            &session,
            *b"planner-contrib-B-seed-32-bytes!",
            1 * 1024 * 1024 * 1024,
        ),
    ];
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 4 * 1024 * 1024 * 1024,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    // Only contributor A is above the 4 GiB floor.
    assert_eq!(plan.assignments.len(), 1);
    assert_eq!(
        plan.assignments[0].contributor_pubkey_hex,
        joins[0].contributor_pubkey_hex
    );
}

#[test]
fn empty_eligible_set_is_a_typed_error_not_a_panic() {
    let session = build_session();
    let joins = vec![build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    )];
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            // Floor higher than the only contributor.
            min_available_ram_bytes: 1 << 40,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(err, PlannerError::NoEligibleContributors { .. }));
}

// ── Live routing posture ─────────────────────────────────────────

#[test]
fn require_live_routing_filters_dtype_mismatch() {
    let session = build_session();
    let join = build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    );
    let joins = vec![join];
    // Advertise BF16 but plan requires F16.
    let advert = build_advert(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        FAR_FUTURE,
        true,
        vec![TensorDtype::Bf16],
    );
    let adverts = vec![advert];
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &adverts,
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: true,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(err, PlannerError::NoEligibleContributors { .. }));
}

#[test]
fn require_live_routing_filters_expired_advert() {
    let session = build_session();
    let join = build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    );
    let joins = vec![join];
    let advert = build_advert(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        // Expired before NOW_UTC.
        "2026-05-30T00:00:00Z",
        true,
        vec![TensorDtype::F16],
    );
    let adverts = vec![advert];
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &adverts,
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: true,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(err, PlannerError::NoEligibleContributors { .. }));
}

#[test]
fn require_live_routing_passes_when_advert_matches() {
    let session = build_session();
    let join = build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    );
    let joins = vec![join];
    let advert = build_advert(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        FAR_FUTURE,
        true,
        vec![TensorDtype::F16],
    );
    let adverts = vec![advert];
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &adverts,
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: true,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    assert_eq!(plan.assignments.len(), 1);
}

// ── Model-plan validation ────────────────────────────────────────

#[test]
fn model_plan_rejects_inverted_layers() {
    let session = build_session();
    let joins = vec![build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    )];
    let mp = ModelPlan {
        schema_version: MODEL_PLAN_SCHEMA_VERSION,
        stages: vec![ModelPlanStage {
            stage_index: 0,
            work_kind: WorkKind::Layers {
                start: 10,
                end: 5,
            },
            expected_work_units: 5,
            expected_work_unit_kind: WorkUnitKind::Layers,
        }],
    };
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: None,
        },
        PlannerStrategy::SingleContributor,
        Some(&mp),
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        PlannerError::InvalidModelPlanStage { stage_index: 0, .. }
    ));
}

#[test]
fn model_plan_rejects_out_of_order_stage_index() {
    let session = build_session();
    let joins = vec![build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    )];
    let mp = ModelPlan {
        schema_version: MODEL_PLAN_SCHEMA_VERSION,
        stages: vec![
            ModelPlanStage {
                stage_index: 1,
                work_kind: WorkKind::Prefill,
                expected_work_units: 1,
                expected_work_unit_kind: WorkUnitKind::Tokens,
            },
            ModelPlanStage {
                stage_index: 0,
                work_kind: WorkKind::Decode,
                expected_work_units: 1,
                expected_work_unit_kind: WorkUnitKind::Tokens,
            },
        ],
    };
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: None,
        },
        PlannerStrategy::SingleContributor,
        Some(&mp),
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(err, PlannerError::InvalidModelPlanStage { .. }));
}

#[test]
fn model_plan_rejects_unknown_schema_version() {
    let session = build_session();
    let joins = vec![build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    )];
    let mp = ModelPlan {
        schema_version: MODEL_PLAN_SCHEMA_VERSION + 9,
        stages: vec![ModelPlanStage {
            stage_index: 0,
            work_kind: WorkKind::Prefill,
            expected_work_units: 1,
            expected_work_unit_kind: WorkUnitKind::Tokens,
        }],
    };
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: None,
        },
        PlannerStrategy::SingleContributor,
        Some(&mp),
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        PlannerError::UnsupportedModelPlanVersion { .. }
    ));
}

// ── Caller-facing contracts ─────────────────────────────────────

#[test]
fn missing_inputs_for_strategy_are_typed_errors() {
    let session = build_session();
    let joins = vec![build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    )];
    // sequential-layers without model-plan AND without layer-count.
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: None,
        },
        PlannerStrategy::SequentialLayers,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(err, PlannerError::InconsistentInputs { .. }));
}

/// `--max-assignments` on `sequential-layers` (no model-plan) is a
/// CONTRIBUTOR cap. The strategy splits across at most N
/// contributors and the resulting layer ranges still cover the full
/// envelope.
#[test]
fn max_assignments_caps_contributor_count_for_equal_split() {
    let session = build_session();
    let joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-C-seed-32-bytes!", 1 << 30),
    ];
    let plan = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: Some(2),
            require_live_routing: false,
            layer_count: Some(32),
        },
        PlannerStrategy::SequentialLayers,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap();
    // 2 contributors → 2 assignments covering the full 0..32.
    assert_eq!(plan.assignments.len(), 2);
    let layers: Vec<(u32, u32)> = plan
        .assignments
        .iter()
        .map(|a| match a.work_kind {
            WorkKind::Layers { start, end } => (start, end),
            _ => panic!("expected WorkKind::Layers"),
        })
        .collect();
    assert_eq!(layers, vec![(0, 16), (16, 32)]);
}

/// `--max-assignments` BELOW the model-plan stage count is refused
/// (stage-count-driven strategies don't have a "drop the tail
/// silently" option).
#[test]
fn max_assignments_below_model_plan_stage_count_is_rejected() {
    let session = build_session();
    let joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
    ];
    let mp = ModelPlan {
        schema_version: MODEL_PLAN_SCHEMA_VERSION,
        stages: vec![
            ModelPlanStage {
                stage_index: 0,
                work_kind: WorkKind::Layers { start: 0, end: 8 },
                expected_work_units: 8,
                expected_work_unit_kind: WorkUnitKind::Layers,
            },
            ModelPlanStage {
                stage_index: 1,
                work_kind: WorkKind::Layers { start: 8, end: 16 },
                expected_work_units: 8,
                expected_work_unit_kind: WorkUnitKind::Layers,
            },
            ModelPlanStage {
                stage_index: 2,
                work_kind: WorkKind::Layers {
                    start: 16,
                    end: 24,
                },
                expected_work_units: 8,
                expected_work_unit_kind: WorkUnitKind::Layers,
            },
        ],
    };
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: Some(2),
            require_live_routing: false,
            layer_count: None,
        },
        PlannerStrategy::SequentialLayers,
        Some(&mp),
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        PlannerError::MaxAssignmentsTooSmall {
            strategy: "sequential-layers",
            required: 3,
            max: 2
        }
    ));
}

/// `--max-assignments` BELOW the `--layer-count` for `round-robin`
/// is also refused (one stage per layer; truncating drops layers).
#[test]
fn max_assignments_below_round_robin_layer_count_is_rejected() {
    let session = build_session();
    let joins = vec![
        build_join(&session, *b"planner-contrib-A-seed-32-bytes!", 1 << 30),
        build_join(&session, *b"planner-contrib-B-seed-32-bytes!", 1 << 30),
    ];
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: Some(3),
            require_live_routing: false,
            layer_count: Some(5),
        },
        PlannerStrategy::RoundRobin,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        PlannerError::MaxAssignmentsTooSmall {
            strategy: "round-robin",
            required: 5,
            max: 3
        }
    ));
}

/// `now_utc >= session.expires_at_utc` → planner refuses with the
/// new `SessionExpired` variant. Matches the publish-time posture so
/// `--no-prune-state-on-start` cannot reach a signed assignment.
#[test]
fn expired_session_is_a_typed_error() {
    // Build a session that expires BEFORE NOW_UTC.
    let coord = CoordinatorSigner::from_seed_bytes(&COORD_SEED).unwrap();
    let mut s = ExecutionSession {
        schema_version: SESSION_SCHEMA_VERSION,
        session_id: String::new(),
        posted_id: "11".repeat(32),
        job_hash: "22".repeat(32),
        model_hash: "33".repeat(32),
        tokenizer_hash: Some("44".repeat(32)),
        coordinator_pubkey_hex: coord.pubkey_hex(),
        created_at_utc: "2026-05-29T00:00:00Z".into(),
        expires_at_utc: "2026-05-29T12:00:00Z".into(),
        coordinator_signature_hex: String::new(),
    };
    s.session_id = session_id_hex(&s).unwrap();
    let si = execution_session_signing_input(&s).unwrap();
    s.coordinator_signature_hex = coord.sign_hex(&si);
    let joins = vec![build_join(
        &s,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    )];
    let err = plan_assignments(
        PlannerInputs {
            session: &s,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: None,
            require_live_routing: false,
            layer_count: Some(8),
        },
        PlannerStrategy::SingleContributor,
        None,
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(
        matches!(err, PlannerError::SessionExpired { .. }),
        "expected SessionExpired, got {err:?}"
    );
}

/// `--max-assignments` BELOW the model-plan stage count is refused
/// for `single-contributor` too (operator asked for one contributor
/// across N stages, and N stages must all be emitted).
#[test]
fn max_assignments_below_single_contributor_model_plan_stages_is_rejected() {
    let session = build_session();
    let joins = vec![build_join(
        &session,
        *b"planner-contrib-A-seed-32-bytes!",
        1 << 30,
    )];
    let mp = ModelPlan {
        schema_version: MODEL_PLAN_SCHEMA_VERSION,
        stages: vec![
            ModelPlanStage {
                stage_index: 0,
                work_kind: WorkKind::Prefill,
                expected_work_units: 1,
                expected_work_unit_kind: WorkUnitKind::Tokens,
            },
            ModelPlanStage {
                stage_index: 1,
                work_kind: WorkKind::Decode,
                expected_work_units: 1,
                expected_work_unit_kind: WorkUnitKind::Tokens,
            },
        ],
    };
    let err = plan_assignments(
        PlannerInputs {
            session: &session,
            joins: &joins,
            peer_adverts: &[],
            required_dtype: TensorDtype::F16,
            min_available_ram_bytes: 0,
            max_assignments: Some(1),
            require_live_routing: false,
            layer_count: None,
        },
        PlannerStrategy::SingleContributor,
        Some(&mp),
        NOW_UTC,
        NOW_UTC,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        PlannerError::MaxAssignmentsTooSmall {
            strategy: "single-contributor",
            required: 2,
            max: 1
        }
    ));
}
