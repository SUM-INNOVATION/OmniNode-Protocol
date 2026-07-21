//! Issue #83 — hermetic tests for the `settlement-read` adapter and
//! view layer. All tests use [`FakeJsonRpcTransport`] with pre-seeded
//! responses; no live network I/O.
//!
//! ## Assertions each test carries
//!
//! Every gated-method test asserts three things:
//! 1. Return value (`SettlementReadError::Dormant` variant, `Ok(...)`
//!    with expected shape, etc.).
//! 2. `fake.calls()` sequence — `chain_getChainParams` must be at
//!    index 0, `chain_getBlockHeight` at index 1, and (for dormant
//!    branches) the gated RPC MUST NOT appear.
//! 3. Params sent on the gated RPC when active — exact JSON.
//!
//! Terminology guard: reward denial is never referred to as "slashing"
//! anywhere in this test file. The verifier registry record
//! (`omninode_getVerifier` / [`VerifierRegistryRaw`]) carries no slash
//! history — slashing is a strictly different, validator-quorum-denied
//! bond-removal event the chain does not expose through this RPC.

#![cfg(feature = "settlement-read")]

use omni_sumchain::settlement::{
    view::{
        BondState, ClaimState, DisputeState, SessionLifecycle, SettlementSessionView,
    },
    wire::{
        DigestTupleRaw, InferenceClaimsRaw, InferenceConsistencyRaw,
        InferenceDisputesRaw, InferenceSessionRaw, VerifierRegistryRaw,
    },
    SettlementGate, SettlementReadError,
};
use omni_sumchain::{FakeJsonRpcTransport, SumChainClient};
use omni_zkml::ChainClientError;
use serde_json::json;

// ── Fixtures ────────────────────────────────────────────────────────────────

fn make_client() -> (SumChainClient<FakeJsonRpcTransport>, FakeJsonRpcTransport) {
    let fake = FakeJsonRpcTransport::new();
    let client = SumChainClient::with_transport([9u8; 32], fake.clone());
    (client, fake)
}

fn params_all_dormant() -> serde_json::Value {
    json!({
        "finality_depth": 12,
        "min_fee": 100,
        "chain_id": 1_800_100,
    })
}

fn params_settlement_only_active(h: u64) -> serde_json::Value {
    json!({
        "finality_depth": 12,
        "min_fee": 100,
        "chain_id": 1_800_100,
        "inference_settlement_enabled_from_height": h,
    })
}

fn params_settlement_and_consistency_active(h: u64) -> serde_json::Value {
    json!({
        "finality_depth": 12,
        "min_fee": 100,
        "chain_id": 1_800_100,
        "inference_settlement_enabled_from_height": h,
        "inference_settlement_consistency_enabled_from_height": h,
    })
}

fn params_all_gates_active(h: u64) -> serde_json::Value {
    json!({
        "finality_depth": 12,
        "min_fee": 100,
        "chain_id": 1_800_100,
        "inference_settlement_enabled_from_height": h,
        "inference_settlement_consistency_enabled_from_height": h,
        "inference_verifier_bonding_enabled_from_height": h,
    })
}

fn seed_params(fake: &FakeJsonRpcTransport, params: serde_json::Value) {
    fake.set_response("chain_getChainParams", Ok(params));
}

fn seed_head(fake: &FakeJsonRpcTransport, height: u64) {
    fake.set_response(
        "chain_getBlockHeight",
        Ok(json!({ "height": height, "finality": "latest" })),
    );
}

fn call_methods(fake: &FakeJsonRpcTransport) -> Vec<String> {
    fake.calls().into_iter().map(|(m, _)| m).collect()
}

// ── Scenario 1 — settlement gate dormant ─────────────────────────────────────

#[test]
fn scenario_1_settlement_dormant_refuses_and_skips_gated_rpc() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_dormant());
    seed_head(&fake, 200_000);

    let err = client
        .omninode_get_inference_session("session-1")
        .expect_err("expected Dormant on missing settlement gate");
    match err {
        SettlementReadError::Dormant { gate, observed, head } => {
            assert_eq!(gate, SettlementGate::Settlement);
            assert_eq!(observed, None);
            assert_eq!(head, 200_000);
        }
        other => panic!("expected Dormant(Settlement), got {other:?}"),
    }

    let methods = call_methods(&fake);
    assert_eq!(methods.first().map(String::as_str), Some("chain_getChainParams"));
    assert_eq!(methods.get(1).map(String::as_str), Some("chain_getBlockHeight"));
    assert_eq!(methods.len(), 2, "dormant path must not issue the gated RPC; calls={methods:?}");
}

// ── Scenario 2 — consistency gate dormant (settlement active) ────────────────

#[test]
fn scenario_2_consistency_dormant_refuses_consistency_rpc() {
    let (client, fake) = make_client();
    seed_params(&fake, params_settlement_only_active(0));
    seed_head(&fake, 200_000);

    let err = client
        .omninode_get_inference_consistency("session-1")
        .expect_err("expected Dormant(Consistency)");
    match err {
        SettlementReadError::Dormant { gate, observed, head } => {
            assert_eq!(gate, SettlementGate::Consistency);
            assert_eq!(observed, None);
            assert_eq!(head, 200_000);
        }
        other => panic!("expected Dormant(Consistency), got {other:?}"),
    }

    let methods = call_methods(&fake);
    assert_eq!(methods[0], "chain_getChainParams");
    assert_eq!(methods[1], "chain_getBlockHeight");
    assert!(
        !methods.iter().any(|m| m == "omninode_getInferenceConsistency"),
        "consistency RPC MUST NOT be called with dormant gate; methods={methods:?}"
    );
}

// ── Scenario 3 — bonding gate dormant ────────────────────────────────────────

#[test]
fn scenario_3_bonding_dormant_refuses_verifier_rpc() {
    let (client, fake) = make_client();
    // Settlement and consistency active but bonding absent.
    seed_params(&fake, params_settlement_and_consistency_active(0));
    seed_head(&fake, 200_000);

    let err = client
        .omninode_get_verifier("verifier-address-1")
        .expect_err("expected Dormant(Bonding)");
    match err {
        SettlementReadError::Dormant { gate, observed, head } => {
            assert_eq!(gate, SettlementGate::Bonding);
            assert_eq!(observed, None);
            assert_eq!(head, 200_000);
        }
        other => panic!("expected Dormant(Bonding), got {other:?}"),
    }

    let methods = call_methods(&fake);
    assert_eq!(methods[0], "chain_getChainParams");
    assert_eq!(methods[1], "chain_getBlockHeight");
    assert!(
        !methods.iter().any(|m| m == "omninode_getVerifier"),
        "verifier RPC MUST NOT be called with dormant bonding gate; methods={methods:?}"
    );
}

// ── Scenario 4 — active gates, empty state ───────────────────────────────────

#[test]
fn scenario_4_active_empty_state_returns_none_or_empty() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);

    // Session absent → null → Ok(None).
    fake.set_response("omninode_getInferenceSession", Ok(serde_json::Value::Null));
    // Claims empty container.
    fake.set_response(
        "omninode_getInferenceClaims",
        Ok(json!({ "session_id": "session-1", "claims": [] })),
    );
    // Disputes empty container.
    fake.set_response(
        "omninode_getInferenceDisputes",
        Ok(json!({ "session_id": "session-1", "disputes": [] })),
    );

    assert_eq!(client.omninode_get_inference_session("session-1").unwrap(), None);
    let claims = client.omninode_get_inference_claims("session-1").unwrap();
    assert_eq!(claims.claims.len(), 0);
    let disputes = client.omninode_get_inference_disputes("session-1").unwrap();
    assert_eq!(disputes.disputes.len(), 0);

    // Sanity: the gated RPCs were actually invoked with the right params.
    let calls = fake.calls();
    let session_call = calls.iter().find(|(m, _)| m == "omninode_getInferenceSession").unwrap();
    assert_eq!(session_call.1, json!(["session-1"]));
    let claims_call = calls.iter().find(|(m, _)| m == "omninode_getInferenceClaims").unwrap();
    assert_eq!(claims_call.1, json!(["session-1"]));
    let disputes_call = calls.iter().find(|(m, _)| m == "omninode_getInferenceDisputes").unwrap();
    assert_eq!(disputes_call.1, json!(["session-1"]));
}

// ── Scenario 5 — multi-verifier session view (2 verifiers) ──────────────────

#[test]
fn scenario_5_multi_verifier_view_composes_two_verifiers() {
    // Two verifiers, both with attestations and claims, no disputes.
    let session = InferenceSessionRaw {
        session_id: "session-mv".into(),
        consistency_required: false,
        bond_required: false,
        max_verifiers: 3,
        escrow_total: "1000".into(),
        escrow_remaining: "600".into(),
        claims_count: 2,
        lifecycle: "active".into(),
        created_at_height: 400_000,
        settled_at_height: None,
        refunded_at_height: None,
        ..Default::default()
    };
    let claims = InferenceClaimsRaw {
        session_id: "session-mv".into(),
        claims: vec![
            omni_sumchain::settlement::wire::InferenceClaimRaw {
                verifier_address: "v-A".into(),
                claimed_at_height: 450_000,
                reward_amount: "200".into(),
                state: "paid".into(),
                paid_at_height: Some(450_001),
                denied_at_height: None,
                denied_reason: None,
            },
            omni_sumchain::settlement::wire::InferenceClaimRaw {
                verifier_address: "v-B".into(),
                claimed_at_height: 450_050,
                reward_amount: "200".into(),
                state: "pending".into(),
                paid_at_height: None,
                denied_at_height: None,
                denied_reason: None,
            },
        ],
    };
    let disputes = InferenceDisputesRaw {
        session_id: "session-mv".into(),
        disputes: vec![],
    };
    let attestations = vec![
        omni_sumchain::InferenceAttestationInfo {
            session_id: "session-mv".into(),
            verifier_address: "v-A".into(),
            model_hash: "0xaa".into(),
            manifest_root: "0xbb".into(),
            response_hash: "0xcc".into(),
            proof_root: "0xdd".into(),
            verifier_signature: "0xsig-a".into(),
            included_at_height: 440_000,
            tx_hash: "0xtx-a".into(),
            finalized: true,
        },
        omni_sumchain::InferenceAttestationInfo {
            session_id: "session-mv".into(),
            verifier_address: "v-B".into(),
            model_hash: "0xaa".into(),
            manifest_root: "0xbb".into(),
            response_hash: "0xcc".into(),
            proof_root: "0xdd".into(),
            verifier_signature: "0xsig-b".into(),
            included_at_height: 440_500,
            tx_hash: "0xtx-b".into(),
            finalized: true,
        },
    ];

    let view = SettlementSessionView::compose(
        session,
        claims,
        disputes,
        attestations,
        None,       // no consistency data — session is not consistency-mode
        None,       // no verifier registry — session is not bond-required
        true, true, // gate flags irrelevant when not required
    )
    .expect("compose should succeed for non-consistency, non-bond session");

    assert_eq!(view.verifiers.len(), 2, "multi-verifier from day one");
    assert_eq!(view.escrow_total, 1000u128);
    assert_eq!(view.escrow_remaining, 600u128);
    assert_eq!(view.lifecycle, SessionLifecycle::Active);

    let v_a = view.verifiers.iter().find(|v| v.verifier_address == "v-A").unwrap();
    let v_b = view.verifiers.iter().find(|v| v.verifier_address == "v-B").unwrap();
    assert!(matches!(v_a.claim_state, ClaimState::Paid { .. }));
    assert!(matches!(v_b.claim_state, ClaimState::Pending { .. }));
    assert_eq!(v_a.dispute_state, DisputeState::None);
    assert_eq!(v_b.dispute_state, DisputeState::None);
    // No bond summary for non-bond session.
    assert!(v_a.bond_summary.is_none());
    assert!(v_b.bond_summary.is_none());
}

// ── Scenario 6 — consistency-mode session, view-level incomplete ────────────

#[test]
fn scenario_6_consistency_mode_view_requires_consistency_gate() {
    let session = InferenceSessionRaw {
        session_id: "session-consistency".into(),
        consistency_required: true,   // <-- session is consistency-mode
        bond_required: false,
        max_verifiers: 3,
        escrow_total: "1000".into(),
        escrow_remaining: "1000".into(),
        claims_count: 0,
        lifecycle: "active".into(),
        created_at_height: 400_000,
        settled_at_height: None,
        refunded_at_height: None,
        ..Default::default()
    };
    let claims = InferenceClaimsRaw { session_id: "session-consistency".into(), claims: vec![] };
    let disputes = InferenceDisputesRaw { session_id: "session-consistency".into(), disputes: vec![] };

    let err = SettlementSessionView::compose(
        session,
        claims,
        disputes,
        vec![],
        None,
        None,
        /* consistency_gate_active */ false,
        /* bonding_gate_active */ false,
    )
    .expect_err("expected ViewIncomplete for consistency-mode session w/ dormant gate");
    match err {
        SettlementReadError::ViewIncomplete { missing_gate, session_id, reason } => {
            assert_eq!(missing_gate, SettlementGate::Consistency);
            assert_eq!(session_id, "session-consistency");
            assert!(reason.contains("consistency"));
            assert!(
                reason.contains("gate to be active"),
                "gate-dormant reason must be distinct from missing-DTO reason; got {reason:?}"
            );
        }
        other => panic!("expected ViewIncomplete(Consistency), got {other:?}"),
    }
}

// ── Scenario 6b — consistency gate active but no consistency DTO passed ─────

#[test]
fn scenario_6b_consistency_mode_view_requires_consistency_data() {
    let session = InferenceSessionRaw {
        session_id: "session-cm-nodto".into(),
        consistency_required: true,
        bond_required: false,
        max_verifiers: 3,
        escrow_total: "1000".into(),
        escrow_remaining: "1000".into(),
        claims_count: 0,
        lifecycle: "active".into(),
        created_at_height: 400_000,
        settled_at_height: None,
        refunded_at_height: None,
        ..Default::default()
    };
    let claims = InferenceClaimsRaw {
        session_id: "session-cm-nodto".into(),
        claims: vec![],
    };
    let disputes = InferenceDisputesRaw {
        session_id: "session-cm-nodto".into(),
        disputes: vec![],
    };

    let err = SettlementSessionView::compose(
        session,
        claims,
        disputes,
        vec![],
        /* consistency */ None,   // <-- caller forgot to fetch it
        /* verifier_registry */ None,
        /* consistency_gate_active */ true,
        /* bonding_gate_active */ true,
    )
    .expect_err(
        "consistency-mode session with the gate active but no InferenceConsistencyRaw \
         passed must fail — otherwise the view silently falls back to attestation \
         digests and defeats the day-one plurality-group requirement",
    );
    match err {
        SettlementReadError::ViewIncomplete { missing_gate, session_id, reason } => {
            assert_eq!(missing_gate, SettlementGate::Consistency);
            assert_eq!(session_id, "session-cm-nodto");
            assert!(
                reason.contains("InferenceConsistencyRaw"),
                "reason should call out the missing DTO input; got {reason:?}"
            );
        }
        other => panic!("expected ViewIncomplete(Consistency, missing DTO), got {other:?}"),
    }
}

// ── Scenario 6c — bond-required session, bonding gate active, no registry ───

#[test]
fn scenario_6c_bond_required_view_requires_verifier_registry() {
    let session = InferenceSessionRaw {
        session_id: "session-bond-nodto".into(),
        consistency_required: false,
        bond_required: true,
        max_verifiers: 3,
        escrow_total: "1000".into(),
        escrow_remaining: "1000".into(),
        claims_count: 0,
        lifecycle: "active".into(),
        created_at_height: 400_000,
        settled_at_height: None,
        refunded_at_height: None,
        ..Default::default()
    };
    let claims = InferenceClaimsRaw {
        session_id: "session-bond-nodto".into(),
        claims: vec![],
    };
    let disputes = InferenceDisputesRaw {
        session_id: "session-bond-nodto".into(),
        disputes: vec![],
    };

    let err = SettlementSessionView::compose(
        session,
        claims,
        disputes,
        vec![],
        None,
        /* verifier_registry */ None, // <-- caller forgot to fetch it
        /* consistency_gate_active */ true,
        /* bonding_gate_active */ true,
    )
    .expect_err(
        "bond-required session with the bonding gate active but no verifier \
         registry passed must fail — otherwise per-verifier bond state is silently \
         absent from the view",
    );
    match err {
        SettlementReadError::ViewIncomplete { missing_gate, session_id, reason } => {
            assert_eq!(missing_gate, SettlementGate::Bonding);
            assert_eq!(session_id, "session-bond-nodto");
            assert!(
                reason.contains("VerifierRegistryRaw"),
                "reason should call out the missing DTO input; got {reason:?}"
            );
        }
        other => panic!("expected ViewIncomplete(Bonding, missing DTO), got {other:?}"),
    }
}

// ── Scenario 6d — bond-required session composes bond_summary from registry ─

#[test]
fn scenario_6d_bond_required_view_composes_bond_summary_from_registry() {
    let session = InferenceSessionRaw {
        session_id: "session-bond-ok".into(),
        consistency_required: false,
        bond_required: true,
        max_verifiers: 3,
        escrow_total: "1000".into(),
        escrow_remaining: "1000".into(),
        claims_count: 0,
        lifecycle: "active".into(),
        created_at_height: 400_000,
        settled_at_height: None,
        refunded_at_height: None,
        ..Default::default()
    };
    let claims = InferenceClaimsRaw {
        session_id: "session-bond-ok".into(),
        claims: vec![],
    };
    let disputes = InferenceDisputesRaw {
        session_id: "session-bond-ok".into(),
        disputes: vec![],
    };
    let attestations = vec![
        omni_sumchain::InferenceAttestationInfo {
            session_id: "session-bond-ok".into(),
            verifier_address: "v-bonded".into(),
            model_hash: "0xaa".into(),
            manifest_root: "0xbb".into(),
            response_hash: "0xcc".into(),
            proof_root: "0xdd".into(),
            verifier_signature: "0xsig-1".into(),
            included_at_height: 440_000,
            tx_hash: "0xtx-1".into(),
            finalized: true,
        },
        omni_sumchain::InferenceAttestationInfo {
            session_id: "session-bond-ok".into(),
            verifier_address: "v-unbonding".into(),
            model_hash: "0xaa".into(),
            manifest_root: "0xbb".into(),
            response_hash: "0xcc".into(),
            proof_root: "0xdd".into(),
            verifier_signature: "0xsig-2".into(),
            included_at_height: 440_500,
            tx_hash: "0xtx-2".into(),
            finalized: true,
        },
    ];
    // Registry covers both verifiers with distinct bond states.
    let registry = vec![
        VerifierRegistryRaw {
            verifier: "v-bonded".into(),
            bond: 10_000,
            status: "Active".into(),
            registered_at_height: 400_000,
            unbonding_started_height: None,
            unlock_height: None,
        },
        VerifierRegistryRaw {
            verifier: "v-unbonding".into(),
            bond: 5_000,
            status: "Unbonding".into(),
            registered_at_height: 400_000,
            unbonding_started_height: Some(490_000),
            unlock_height: Some(500_050),
        },
    ];

    let view = SettlementSessionView::compose(
        session,
        claims,
        disputes,
        attestations,
        None,
        Some(registry),
        /* consistency_gate_active */ true,
        /* bonding_gate_active */ true,
    )
    .expect("bond-required session with registry supplied should compose");

    assert_eq!(view.verifiers.len(), 2);
    let v_bonded = view.verifiers.iter().find(|v| v.verifier_address == "v-bonded").unwrap();
    let v_unbonding = view.verifiers.iter().find(|v| v.verifier_address == "v-unbonding").unwrap();

    let bs_bonded = v_bonded.bond_summary.as_ref().expect("bonded verifier must have bond_summary");
    assert_eq!(bs_bonded.bond_amount, 10_000u128);
    assert_eq!(bs_bonded.bond_state, BondState::Bonded);

    let bs_unb = v_unbonding.bond_summary.as_ref().expect("unbonding verifier must have bond_summary");
    assert_eq!(bs_unb.bond_amount, 5_000u128);
    assert_eq!(bs_unb.bond_state, BondState::Unbonding);
    assert_eq!(bs_unb.unbonding_since_height, Some(490_000));
    assert_eq!(bs_unb.withdrawable_at_height, Some(500_050));
}

// ── Scenario 6e — bond-required session, verifier missing from registry ────

#[test]
fn scenario_6e_bond_required_view_leaves_bond_summary_none_for_unregistered() {
    // Even with the guard satisfied (Some(...) passed), individual
    // verifiers without a registry entry get `bond_summary: None`.
    // The view still composes — the caller decides policy on the
    // missing entries (e.g. tolerate pending-registration verifiers).
    let session = InferenceSessionRaw {
        session_id: "session-bond-partial".into(),
        consistency_required: false,
        bond_required: true,
        max_verifiers: 3,
        escrow_total: "1000".into(),
        escrow_remaining: "1000".into(),
        claims_count: 0,
        lifecycle: "active".into(),
        created_at_height: 400_000,
        settled_at_height: None,
        refunded_at_height: None,
        ..Default::default()
    };
    let claims = InferenceClaimsRaw {
        session_id: "session-bond-partial".into(),
        claims: vec![],
    };
    let disputes = InferenceDisputesRaw {
        session_id: "session-bond-partial".into(),
        disputes: vec![],
    };
    let attestations = vec![omni_sumchain::InferenceAttestationInfo {
        session_id: "session-bond-partial".into(),
        verifier_address: "v-unregistered".into(),
        model_hash: "0xaa".into(),
        manifest_root: "0xbb".into(),
        response_hash: "0xcc".into(),
        proof_root: "0xdd".into(),
        verifier_signature: "0xsig".into(),
        included_at_height: 440_000,
        tx_hash: "0xtx".into(),
        finalized: true,
    }];

    let view = SettlementSessionView::compose(
        session,
        claims,
        disputes,
        attestations,
        None,
        Some(vec![]), // registry supplied but empty
        true,
        true,
    )
    .expect("empty Some(registry) satisfies the guard");
    let v = view.verifiers.iter().find(|v| v.verifier_address == "v-unregistered").unwrap();
    assert!(v.bond_summary.is_none());
}

// ── Scenario 7 — verifier registry read for bonded / unbonding / withdrawn ──

#[test]
fn scenario_7_verifier_registry_records_round_trip() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({
            "verifier": "verifier-1",
            "bond": 10_000,
            "status": "Active",
            "registered_at_height": 12_345
        })),
    );

    let v = client
        .omninode_get_verifier("verifier-1")
        .expect("bonded read should succeed")
        .expect("verifier should not be absent");
    assert_eq!(v.verifier, "verifier-1");
    assert_eq!(v.bond, 10_000);
    assert_eq!(v.status, "Active");
    assert_eq!(v.registered_at_height, 12_345);
    assert_eq!(v.unbonding_started_height, None);
    assert_eq!(v.unlock_height, None);

    // Also test unbonding + withdrawn.
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({
            "verifier": "verifier-2",
            "bond": 5_000,
            "status": "Unbonding",
            "registered_at_height": 10_000,
            "unbonding_started_height": 490_000,
            "unlock_height": 500_050
        })),
    );
    let v2 = client.omninode_get_verifier("verifier-2").unwrap().unwrap();
    assert_eq!(v2.status, "Unbonding");
    assert_eq!(v2.unbonding_started_height, Some(490_000));
    assert_eq!(v2.unlock_height, Some(500_050));

    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({
            "verifier": "verifier-3",
            "bond": 0,
            "status": "Withdrawn",
            "registered_at_height": 9_000
        })),
    );
    let v3 = client.omninode_get_verifier("verifier-3").unwrap().unwrap();
    assert_eq!(v3.status, "Withdrawn");
    assert_eq!(v3.bond, 0);

    // Ensure each call carried the right address param.
    let calls = fake.calls();
    let verifier_calls: Vec<_> = calls
        .iter()
        .filter(|(m, _)| m == "omninode_getVerifier")
        .collect();
    assert_eq!(verifier_calls.len(), 3);
    assert_eq!(verifier_calls[0].1, json!(["verifier-1"]));
    assert_eq!(verifier_calls[1].1, json!(["verifier-2"]));
    assert_eq!(verifier_calls[2].1, json!(["verifier-3"]));
}

// ── Scenario 7a — getVerifier decode: negatives, boundaries, forward-compat ─

// `null` is a successful "not registered", never a parse error.
#[test]
fn scenario_7a_null_maps_to_not_registered() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response("omninode_getVerifier", Ok(serde_json::Value::Null));
    assert!(
        client
            .omninode_get_verifier("nobody")
            .expect("null must be a successful not-registered, not a parse error")
            .is_none()
    );
}

// A legacy / mis-named envelope (no canonical `verifier`) is rejected —
// never silently decoded. Proves the stale client shape can't sneak through.
#[test]
fn scenario_7a_legacy_shape_is_rejected() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({
            "address": "legacy-v",
            "bond_amount": "10000",
            "bond_state": "bonded"
        })),
    );
    match client
        .omninode_get_verifier("legacy-v")
        .expect_err("a legacy-named response must be rejected, not decoded")
    {
        SettlementReadError::WireParse(msg) => {
            assert!(msg.contains("omninode_getVerifier"), "got {msg:?}");
        }
        other => panic!("expected WireParse, got {other:?}"),
    }
}

// A missing required field (`verifier`) is rejected.
#[test]
fn scenario_7a_missing_verifier_field_is_rejected() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({
            "bond": 10_000,
            "status": "Active",
            "registered_at_height": 1
        })),
    );
    assert!(matches!(
        client
            .omninode_get_verifier("v")
            .expect_err("missing `verifier` must reject"),
        SettlementReadError::WireParse(_)
    ));
}

// The canonical wire is a JSON number; a string bond is rejected.
#[test]
fn scenario_7a_string_bond_is_rejected() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({
            "verifier": "v",
            "bond": "10000",
            "status": "Active",
            "registered_at_height": 1
        })),
    );
    assert!(matches!(
        client
            .omninode_get_verifier("v")
            .expect_err("string bond must reject on the canonical numeric wire"),
        SettlementReadError::WireParse(_)
    ));
}

// Boundary: `u64::MAX` decodes exactly (the largest value serde_json's
// `Value` holds as a native integer even without arbitrary_precision).
#[test]
fn scenario_7a_bond_u64_max_is_accepted() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({
            "verifier": "v",
            "bond": u64::MAX,
            "status": "Active",
            "registered_at_height": 1
        })),
    );
    let v = client.omninode_get_verifier("v").unwrap().unwrap();
    assert_eq!(v.bond, u64::MAX as u128);
}

// Boundary: values ABOVE `u64::MAX` decode exactly. `arbitrary_precision`
// (issue #97) preserves the exact integer token through the `Value` stage,
// so there is no u64 ceiling — `u64::MAX + 1` and `u128::MAX` both decode
// byte-for-byte. Tokens are written as raw JSON so no Rust-side coercion
// occurs before the typed decode.
#[test]
fn scenario_7a_bond_above_u64_max_and_u128_max_decode_exactly() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    let cases: [(&str, u128); 2] = [
        ("18446744073709551616", (u64::MAX as u128) + 1),
        ("340282366920938463463374607431768211455", u128::MAX),
    ];
    for (token, expected) in cases {
        let raw = format!(
            r#"{{"verifier":"v","bond":{token},"status":"Active","registered_at_height":1}}"#
        );
        let value: serde_json::Value = serde_json::from_str(&raw).expect("raw json parses to Value");
        fake.set_response("omninode_getVerifier", Ok(value));
        let v = client.omninode_get_verifier("v").unwrap().unwrap();
        assert_eq!(v.bond, expected, "bond token {token} must decode exactly");
    }
}

// Boundary: `u128::MAX + 1` overflows `u128` and is rejected. The token is
// written directly as raw JSON — it cannot be represented in a Rust `u128`,
// so nothing truncates it before the typed decode sees the overflow.
#[test]
fn scenario_7a_bond_above_u128_max_is_rejected() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    let raw = r#"{"verifier":"v","bond":340282366920938463463374607431768211456,"status":"Active","registered_at_height":1}"#;
    let value: serde_json::Value = serde_json::from_str(raw).expect("raw json parses to Value");
    fake.set_response("omninode_getVerifier", Ok(value));
    assert!(matches!(
        client
            .omninode_get_verifier("v")
            .expect_err("bond above u128::MAX must reject as overflow"),
        SettlementReadError::WireParse(_)
    ));
}

// Invalid numeric forms are rejected under the canonical numeric contract.
#[test]
fn scenario_7a_bond_negative_is_rejected() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({"verifier": "v", "bond": -1, "status": "Active", "registered_at_height": 1})),
    );
    assert!(matches!(
        client
            .omninode_get_verifier("v")
            .expect_err("negative bond must reject"),
        SettlementReadError::WireParse(_)
    ));
}

#[test]
fn scenario_7a_bond_fractional_is_rejected() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({"verifier": "v", "bond": 1.5, "status": "Active", "registered_at_height": 1})),
    );
    assert!(matches!(
        client
            .omninode_get_verifier("v")
            .expect_err("fractional bond must reject"),
        SettlementReadError::WireParse(_)
    ));
}

// Forward-compat: an unrecognized `status` decodes at the wire level
// (never hard-fails the read); the view layer normalizes it to
// `BondState::UnknownWire` (see the signer's precheck_bond_unknown_wire_state).
#[test]
fn scenario_7a_unknown_status_decodes_verbatim() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);
    fake.set_response(
        "omninode_getVerifier",
        Ok(json!({
            "verifier": "v",
            "bond": 42,
            "status": "Frobnicated",
            "registered_at_height": 7
        })),
    );
    let v = client.omninode_get_verifier("v").unwrap().unwrap();
    assert_eq!(v.status, "Frobnicated");
}

// ── Scenario 8 — claimable reward: mature-and-claimable vs. immature ────────

#[test]
fn scenario_8_claimable_reward_mature_and_immature() {
    let (client, fake) = make_client();
    seed_params(&fake, params_all_gates_active(0));
    seed_head(&fake, 500_000);

    // Mature verifier.
    fake.set_response(
        "omninode_getClaimableReward",
        Ok(json!({
            "session_id": "s-1",
            "verifier_address": "v-mature",
            "mature": true,
            "claim_ready_block": 499_000,
            "blocks_until_ready": 0,
            "escrow_available": true,
            "cap_available": true,
            "dispute_clear": true,
            "claimable_now": true,
            "reward_amount": "200"
        })),
    );
    let mature = client
        .omninode_get_claimable_reward("s-1", "v-mature")
        .expect("mature reward read");
    assert!(mature.mature);
    assert!(mature.claimable_now);
    assert_eq!(mature.blocks_until_ready, 0);

    // Immature verifier — chain still in dispute window.
    fake.set_response(
        "omninode_getClaimableReward",
        Ok(json!({
            "session_id": "s-1",
            "verifier_address": "v-immature",
            "mature": false,
            "claim_ready_block": 550_000,
            "blocks_until_ready": 50_000,
            "escrow_available": true,
            "cap_available": true,
            "dispute_clear": true,
            "claimable_now": false,
            "reward_amount": "200"
        })),
    );
    let immature = client
        .omninode_get_claimable_reward("s-1", "v-immature")
        .expect("immature reward read");
    assert!(!immature.mature);
    assert!(!immature.claimable_now);
    assert_eq!(immature.blocks_until_ready, 50_000);
}

// ── Scenario 9 — params-first ordering invariant across every gated method ──

#[test]
fn scenario_9_params_and_head_precede_every_gated_rpc() {
    // With ALL gates dormant, every gated method must issue exactly
    // `chain_getChainParams`, then `chain_getBlockHeight`, then NOT
    // issue the gated RPC.
    struct GatedCase {
        name: &'static str,
        gated_rpc: &'static str,
        exercise: fn(&SumChainClient<FakeJsonRpcTransport>) -> Result<(), SettlementReadError>,
    }
    let cases: &[GatedCase] = &[
        GatedCase {
            name: "getInferenceSession",
            gated_rpc: "omninode_getInferenceSession",
            exercise: |c| c.omninode_get_inference_session("s").map(|_| ()),
        },
        GatedCase {
            name: "getInferenceClaims",
            gated_rpc: "omninode_getInferenceClaims",
            exercise: |c| c.omninode_get_inference_claims("s").map(|_| ()),
        },
        GatedCase {
            name: "getInferenceDisputes",
            gated_rpc: "omninode_getInferenceDisputes",
            exercise: |c| c.omninode_get_inference_disputes("s").map(|_| ()),
        },
        GatedCase {
            name: "getClaimableReward",
            gated_rpc: "omninode_getClaimableReward",
            exercise: |c| c.omninode_get_claimable_reward("s", "v").map(|_| ()),
        },
        GatedCase {
            name: "getInferenceConsistency",
            gated_rpc: "omninode_getInferenceConsistency",
            exercise: |c| c.omninode_get_inference_consistency("s").map(|_| ()),
        },
        GatedCase {
            name: "getVerifier",
            gated_rpc: "omninode_getVerifier",
            exercise: |c| c.omninode_get_verifier("a").map(|_| ()),
        },
    ];

    for case in cases {
        let (client, fake) = make_client();
        seed_params(&fake, params_all_dormant());
        seed_head(&fake, 42_000);

        let res = (case.exercise)(&client);
        assert!(res.is_err(), "[{}] dormant path must error", case.name);

        let methods = call_methods(&fake);
        assert_eq!(
            methods.first().map(String::as_str),
            Some("chain_getChainParams"),
            "[{}] chain_getChainParams must be call 0; got {methods:?}",
            case.name,
        );
        assert_eq!(
            methods.get(1).map(String::as_str),
            Some("chain_getBlockHeight"),
            "[{}] chain_getBlockHeight must be call 1; got {methods:?}",
            case.name,
        );
        assert!(
            !methods.iter().any(|m| m == case.gated_rpc),
            "[{}] {} MUST NOT be called with dormant gate; methods={methods:?}",
            case.name,
            case.gated_rpc,
        );
    }
}

// ── Sanity — RPC error propagates as SettlementReadError::Rpc ───────────────

#[test]
fn rpc_transport_error_propagates_as_rpc_variant() {
    let (client, fake) = make_client();
    fake.set_response(
        "chain_getChainParams",
        Err(ChainClientError::Other("simulated chain outage".into())),
    );
    let err = client.omninode_get_inference_session("s").expect_err("must error");
    match err {
        SettlementReadError::Rpc(ChainClientError::Other(msg)) => {
            assert!(msg.contains("simulated chain outage"));
        }
        other => panic!("expected Rpc, got {other:?}"),
    }
}

// ── Compile-time sanity — types exposed ─────────────────────────────────────

#[allow(dead_code)]
fn _type_check() {
    let _ = DigestTupleRaw {
        model_hash: String::new(),
        manifest_root: String::new(),
        response_hash: String::new(),
        proof_root: String::new(),
    };
    let _: Option<InferenceConsistencyRaw> = None;
    let _: Option<VerifierRegistryRaw> = None;
}
