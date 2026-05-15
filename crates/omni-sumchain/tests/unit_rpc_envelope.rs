//! Stage 7a — RPC envelope assembly tests.
//!
//! Construct `SumChainClient::with_transport(FakeJsonRpcTransport)`,
//! exercise each public read RPC, and assert (a) the correct JSON-RPC
//! method name and (b) the correct positional `params` array. Plus
//! integration with the Stage 5.1 `query_attestation_workflow`.

use omni_sumchain::{
    BlockFinality, FakeJsonRpcTransport, SumChainClient,
};
use omni_zkml::{
    AttestationRegistry, AttestationStatus, ChainClient, ChainClientError,
    LocalAttestationStatus, RegistryError, query_attestation_workflow,
};

fn dummy_seed() -> [u8; 32] {
    [42u8; 32]
}

fn client_with_fake() -> (FakeJsonRpcTransport, SumChainClient<FakeJsonRpcTransport>) {
    let fake = FakeJsonRpcTransport::new();
    let client = SumChainClient::with_transport(dummy_seed(), fake.clone());
    (fake, client)
}

// ── query_attestation_status (the trait method) ──────────────────────────────

#[test]
fn query_attestation_status_calls_method_and_passes_tx_id() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "sum_getInferenceAttestationStatus",
        Ok(serde_json::json!({
            "status": "finalized",
            "included_at_height": 200,
            "reason": null
        })),
    );
    let status = client.query_attestation_status("0xdeadbeef").unwrap();
    assert_eq!(status, AttestationStatus::Finalized);

    let calls = fake.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "sum_getInferenceAttestationStatus");
    assert_eq!(calls[0].1, serde_json::json!(["0xdeadbeef"]));
}

#[test]
fn query_attestation_status_maps_unknown_chain_status() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "sum_getInferenceAttestationStatus",
        Ok(serde_json::json!({
            "status": "unknown",
            "included_at_height": null,
            "reason": null
        })),
    );
    let status = client.query_attestation_status("0xbeef").unwrap();
    assert_eq!(status, AttestationStatus::Unknown);
}

#[test]
fn query_attestation_status_maps_failed_with_reason() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "sum_getInferenceAttestationStatus",
        Ok(serde_json::json!({
            "status": "failed",
            "included_at_height": 50,
            "reason": "execution reverted"
        })),
    );
    let status = client.query_attestation_status("0xface").unwrap();
    assert_eq!(
        status,
        AttestationStatus::Failed {
            reason: "execution reverted".into()
        }
    );
}

// ── Inherent helpers ────────────────────────────────────────────────────────

#[test]
fn get_chain_params_calls_correct_method_with_empty_params() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337,
            "omninode_enabled_from_height": null
        })),
    );
    let params = client.get_chain_params().unwrap();
    assert_eq!(params.finality_depth, 10);
    assert_eq!(params.min_fee, 1);
    assert_eq!(params.chain_id, 31337);
    assert_eq!(params.omninode_enabled_from_height, None);

    let calls = fake.calls();
    assert_eq!(calls[0].0, "chain_getChainParams");
    assert_eq!(calls[0].1, serde_json::json!([]));
}

#[test]
fn get_block_height_passes_finality_token() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "chain_getBlockHeight",
        Ok(serde_json::json!({"height":123,"finality":"latest"})),
    );
    let info = client.get_block_height(BlockFinality::Latest).unwrap();
    assert_eq!(info.height, 123);
    assert_eq!(info.finality, "latest");

    let calls = fake.calls();
    assert_eq!(calls[0].0, "chain_getBlockHeight");
    assert_eq!(calls[0].1, serde_json::json!(["latest"]));
}

#[test]
fn get_block_height_finalized_token() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "chain_getBlockHeight",
        Ok(serde_json::json!({"height":100,"finality":"finalized"})),
    );
    let info = client.get_block_height(BlockFinality::Finalized).unwrap();
    assert_eq!(info.finality, "finalized");

    let calls = fake.calls();
    assert_eq!(calls[0].1, serde_json::json!(["finalized"]));
}

#[test]
fn get_nonce_calls_correct_method_with_address() {
    let (fake, client) = client_with_fake();
    fake.set_response("sum_getNonce", Ok(serde_json::json!(7)));
    let nonce = client.get_nonce("addr-1").unwrap();
    assert_eq!(nonce, 7);

    let calls = fake.calls();
    assert_eq!(calls[0].0, "sum_getNonce");
    assert_eq!(calls[0].1, serde_json::json!(["addr-1"]));
}

#[test]
fn get_attestation_returns_none_for_null_response() {
    let (fake, client) = client_with_fake();
    fake.set_response("sum_getInferenceAttestation", Ok(serde_json::Value::Null));
    let result = client.get_attestation("sess-x", "addr-y").unwrap();
    assert!(result.is_none());

    let calls = fake.calls();
    assert_eq!(calls[0].0, "sum_getInferenceAttestation");
    assert_eq!(calls[0].1, serde_json::json!(["sess-x", "addr-y"]));
}

#[test]
fn get_attestation_parses_full_info() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "sum_getInferenceAttestation",
        Ok(serde_json::json!({
            "session_id": "sess-1",
            "verifier_address": "addr-1",
            "model_hash": "0xaa",
            "manifest_root": "0xbb",
            "response_hash": "0xcc",
            "proof_root": "0xdd",
            "verifier_signature": "0xee",
            "included_at_height": 42,
            "tx_hash": "0x1234",
            "finalized": true
        })),
    );
    let info = client.get_attestation("sess-1", "addr-1").unwrap().unwrap();
    assert_eq!(info.session_id, "sess-1");
    assert_eq!(info.included_at_height, 42);
    assert!(info.finalized);
}

#[test]
fn list_attestations_returns_vec() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "sum_listInferenceAttestations",
        Ok(serde_json::json!([])),
    );
    let list = client.list_attestations("sess-1").unwrap();
    assert!(list.is_empty());

    let calls = fake.calls();
    assert_eq!(calls[0].0, "sum_listInferenceAttestations");
    assert_eq!(calls[0].1, serde_json::json!(["sess-1"]));
}

// ── omninode_is_active ───────────────────────────────────────────────────────

#[test]
fn omninode_is_active_returns_false_pre_patch() {
    // Pre-patch: omninode_enabled_from_height is missing from the response.
    let (fake, client) = client_with_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337
        })),
    );
    assert!(!client.omninode_is_active().unwrap());

    // Sanity: chain_getBlockHeight should NOT have been called — the
    // pre-patch shortcut returns false without querying the head.
    let calls = fake.calls();
    let methods: Vec<&str> = calls.iter().map(|(m, _)| m.as_str()).collect();
    assert!(!methods.contains(&"chain_getBlockHeight"));
}

#[test]
fn omninode_is_active_returns_true_post_patch_when_head_ge_activation() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337,
            "omninode_enabled_from_height": 0
        })),
    );
    fake.set_response(
        "chain_getBlockHeight",
        Ok(serde_json::json!({"height": 5, "finality": "latest"})),
    );
    assert!(client.omninode_is_active().unwrap());
}

#[test]
fn omninode_is_active_returns_false_when_head_below_activation() {
    let (fake, client) = client_with_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337,
            "omninode_enabled_from_height": 100
        })),
    );
    fake.set_response(
        "chain_getBlockHeight",
        Ok(serde_json::json!({"height": 50, "finality": "latest"})),
    );
    assert!(!client.omninode_is_active().unwrap());
}

// ── submit_attestation (Stage 7b — real implementation) ────────────────────

/// Smoke: a fully-configured happy-path submit returns
/// `Ok(SubmissionReceipt)` and the fake transport saw a
/// `sum_sendRawTransaction` call. Detailed gate / RPC-shape pinning
/// lives in `tests/unit_submit_construction.rs`.
#[test]
fn submit_attestation_happy_path_returns_receipt() {
    use omni_types::phase5::{InferenceAttestation, InferenceCommitment, SnipV2ObjectId};

    // Derive the verifier address from a known seed so the
    // attestation's `verifier_address` matches (consistency gate).
    let seed = [42u8; 32];
    let verifier_address = omni_zkml::signer_chain_address_base58(&seed).unwrap();

    let fake = omni_sumchain::FakeJsonRpcTransport::new();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337,
            "omninode_enabled_from_height": 0,
            "v2_enabled_from_height": 0,
        })),
    );
    fake.set_response(
        "chain_getBlockHeight",
        Ok(serde_json::json!({"height": 5, "finality": "latest"})),
    );
    fake.set_response("sum_getNonce", Ok(serde_json::json!(0)));
    fake.set_response(
        "sum_sendRawTransaction",
        // Canonical chain response shape (sum-chain @ b586ff3f).
        Ok(serde_json::json!({ "tx_hash": "0xdeadbeefcafebabe" })),
    );
    let client = omni_sumchain::SumChainClient::with_transport(seed, fake.clone());

    let attestation = InferenceAttestation {
        commitment: InferenceCommitment {
            session_id: "sess-happy".into(),
            model_hash: "0".repeat(64),
            manifest_snip_root: SnipV2ObjectId::from_bytes([0x11u8; 32]),
            response_hash: "1".repeat(64),
            proof_snip_root: SnipV2ObjectId::from_bytes([0x22u8; 32]),
        },
        verifier_address,
        verifier_signature: "ignored-by-stage-7b".into(),
    };

    let receipt = client.submit_attestation(&attestation).unwrap();
    assert_eq!(receipt.tx_id, "0xdeadbeefcafebabe");

    let methods: Vec<String> = fake
        .calls()
        .iter()
        .map(|(m, _)| m.clone())
        .collect();
    assert!(methods.contains(&"sum_sendRawTransaction".to_string()));
}

// ── Stage 5.1 integration: Unknown is non-terminal ──────────────────────────

#[test]
fn workflow_with_sum_chain_client_leaves_submitted_unchanged_on_unknown() {
    use omni_zkml::{compute_attestation_id, SubmissionReceipt};
    use omni_types::phase5::{InferenceAttestation, InferenceCommitment, SnipV2ObjectId};

    let tmp = tempfile::tempdir().unwrap();
    let reg = AttestationRegistry::open(tmp.path().join("attestations")).unwrap();

    // Insert + mark_submitted a record so it's in the queryable state.
    let attestation = InferenceAttestation {
        commitment: InferenceCommitment {
            session_id: "sess-int".into(),
            model_hash: "a".repeat(64),
            manifest_snip_root: SnipV2ObjectId::from_bytes([0u8; 32]),
            response_hash: "b".repeat(64),
            proof_snip_root: SnipV2ObjectId::from_bytes([0u8; 32]),
        },
        verifier_address: "addr-int".into(),
        verifier_signature: "sig-int".into(),
    };
    let id = compute_attestation_id(&attestation).unwrap();
    reg.insert(attestation.clone()).unwrap();
    reg.mark_submitted(
        &id,
        SubmissionReceipt {
            tx_id: "0xchain-tx-1".into(),
            note: None,
        },
    )
    .unwrap();

    // Configure the fake to return Unknown for that tx_id.
    let (fake, client) = client_with_fake();
    fake.set_response(
        "sum_getInferenceAttestationStatus",
        Ok(serde_json::json!({
            "status": "unknown",
            "included_at_height": null,
            "reason": null
        })),
    );

    let updated = query_attestation_workflow(&reg, &client, &id).unwrap();
    assert_eq!(updated.status, LocalAttestationStatus::Submitted); // unchanged

    let calls = fake.calls();
    assert_eq!(calls[0].0, "sum_getInferenceAttestationStatus");
    assert_eq!(calls[0].1, serde_json::json!(["0xchain-tx-1"]));
}

// ── Stage 5.1 integration: RPC errors leave record unchanged ────────────────

#[test]
fn workflow_with_sum_chain_client_returns_chain_client_error_on_rpc_failure() {
    use omni_zkml::{compute_attestation_id, SubmissionReceipt};
    use omni_types::phase5::{InferenceAttestation, InferenceCommitment, SnipV2ObjectId};

    let tmp = tempfile::tempdir().unwrap();
    let reg = AttestationRegistry::open(tmp.path().join("attestations")).unwrap();

    let attestation = InferenceAttestation {
        commitment: InferenceCommitment {
            session_id: "sess-err".into(),
            model_hash: "a".repeat(64),
            manifest_snip_root: SnipV2ObjectId::from_bytes([0u8; 32]),
            response_hash: "b".repeat(64),
            proof_snip_root: SnipV2ObjectId::from_bytes([0u8; 32]),
        },
        verifier_address: "addr-err".into(),
        verifier_signature: "sig-err".into(),
    };
    let id = compute_attestation_id(&attestation).unwrap();
    reg.insert(attestation.clone()).unwrap();
    reg.mark_submitted(
        &id,
        SubmissionReceipt {
            tx_id: "0xerr".into(),
            note: None,
        },
    )
    .unwrap();

    let (fake, client) = client_with_fake();
    fake.set_response(
        "sum_getInferenceAttestationStatus",
        Err(ChainClientError::Other("rpc gone".into())),
    );

    let err = query_attestation_workflow(&reg, &client, &id).unwrap_err();
    assert!(matches!(
        err,
        RegistryError::ChainClient(ChainClientError::Other(_))
    ));
    // Record is unchanged.
    let reloaded = reg.load(&id).unwrap();
    assert_eq!(reloaded.status, LocalAttestationStatus::Submitted);
}
