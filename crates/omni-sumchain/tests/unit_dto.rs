//! Stage 7a — DTO deserialisation unit tests.
//!
//! Pin the chain-confirmed JSON shape for each read RPC's response so
//! any drift surfaces as a clear named failure.

use omni_sumchain::{
    BlockFinality, BlockHeightInfo, ChainParamsInfo, InferenceAttestationInfo,
    InferenceAttestationStatusInfo,
};

// ── InferenceAttestationStatusInfo ───────────────────────────────────────────

#[test]
fn status_info_parses_submitted_with_nulls() {
    let json = r#"{"status":"submitted","included_at_height":null,"reason":null}"#;
    let parsed: InferenceAttestationStatusInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.status, "submitted");
    assert_eq!(parsed.included_at_height, None);
    assert_eq!(parsed.reason, None);
}

#[test]
fn status_info_parses_included_with_height() {
    let json = r#"{"status":"included","included_at_height":42,"reason":null}"#;
    let parsed: InferenceAttestationStatusInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.status, "included");
    assert_eq!(parsed.included_at_height, Some(42));
    assert_eq!(parsed.reason, None);
}

#[test]
fn status_info_parses_failed_with_reason() {
    let json = r#"{"status":"failed","included_at_height":null,"reason":"execution reverted"}"#;
    let parsed: InferenceAttestationStatusInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.status, "failed");
    assert_eq!(parsed.reason, Some("execution reverted".into()));
}

#[test]
fn status_info_parses_unknown_bare() {
    let json = r#"{"status":"unknown","included_at_height":null,"reason":null}"#;
    let parsed: InferenceAttestationStatusInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.status, "unknown");
}

// ── BlockHeightInfo ──────────────────────────────────────────────────────────

#[test]
fn block_height_info_parses_latest() {
    let json = r#"{"height":123,"finality":"latest"}"#;
    let parsed: BlockHeightInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.height, 123);
    assert_eq!(parsed.finality, "latest");
}

#[test]
fn block_height_info_parses_finalized() {
    let json = r#"{"height":100,"finality":"finalized"}"#;
    let parsed: BlockHeightInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.height, 100);
    assert_eq!(parsed.finality, "finalized");
}

#[test]
fn block_finality_token_serialisation() {
    assert_eq!(BlockFinality::Latest.as_token(), "latest");
    assert_eq!(BlockFinality::Finalized.as_token(), "finalized");
}

// ── ChainParamsInfo ──────────────────────────────────────────────────────────

#[test]
fn chain_params_info_parses_post_patch_response() {
    // Post-patch: omninode_enabled_from_height present.
    let json = r#"{"finality_depth":10,"min_fee":1,"chain_id":31337,"omninode_enabled_from_height":0}"#;
    let parsed: ChainParamsInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.finality_depth, 10);
    assert_eq!(parsed.min_fee, 1);
    assert_eq!(parsed.chain_id, 31337);
    assert_eq!(parsed.omninode_enabled_from_height, Some(0));
}

#[test]
fn chain_params_info_parses_pre_patch_response() {
    // Pre-patch: omninode_enabled_from_height missing. #[serde(default)]
    // makes the parser forward-compatible — None pre-patch, Some post.
    let json = r#"{"finality_depth":10,"min_fee":1,"chain_id":31337}"#;
    let parsed: ChainParamsInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.finality_depth, 10);
    assert_eq!(parsed.min_fee, 1);
    assert_eq!(parsed.chain_id, 31337);
    assert_eq!(parsed.omninode_enabled_from_height, None);
}

#[test]
fn chain_params_info_local_mirror_defaults() {
    // Pin the documented local-mirror defaults: chain_id 31337, min_fee 1.
    let json = r#"{"finality_depth":12,"min_fee":1,"chain_id":31337,"omninode_enabled_from_height":0}"#;
    let parsed: ChainParamsInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.chain_id, 31337);
    assert_eq!(parsed.min_fee, 1);
}

// ── Stage 7b additions: v2_enabled_from_height ──────────────────────

/// Pin the Stage 7b-confirmed shape from the local mirror at chain rev
/// `d83e45a4`: both activation fields present and set to `0`
/// (activation-from-genesis).
#[test]
fn chain_params_info_parses_with_both_activation_fields() {
    let json = r#"{
        "finality_depth": 12,
        "min_fee": 1,
        "chain_id": 31337,
        "omninode_enabled_from_height": 0,
        "v2_enabled_from_height": 0
    }"#;
    let parsed: ChainParamsInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.omninode_enabled_from_height, Some(0));
    assert_eq!(parsed.v2_enabled_from_height, Some(0));
}

/// Asymmetric activation case (real branch): OmniNode subprotocol is
/// active but the V2 envelope itself is gated. Stage 7b's
/// `submit_attestation` must reject submission in this state.
#[test]
fn chain_params_info_parses_asymmetric_activation() {
    let json = r#"{
        "finality_depth": 12,
        "min_fee": 1,
        "chain_id": 31337,
        "omninode_enabled_from_height": 0,
        "v2_enabled_from_height": null
    }"#;
    let parsed: ChainParamsInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.omninode_enabled_from_height, Some(0));
    assert_eq!(parsed.v2_enabled_from_height, None);
}

// ── InferenceAttestationInfo ────────────────────────────────────────────────

#[test]
fn inference_attestation_info_parses_chain_0x_prefixed_hex() {
    let json = r#"{
        "session_id": "sess-1",
        "verifier_address": "NFG1W1iuwcHCvdFxvWNTjDgqqYgq7m155",
        "model_hash": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "manifest_root": "0x1111111111111111111111111111111111111111111111111111111111111111",
        "response_hash": "0x2222222222222222222222222222222222222222222222222222222222222222",
        "proof_root": "0x3333333333333333333333333333333333333333333333333333333333333333",
        "verifier_signature": "0x44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444",
        "included_at_height": 50,
        "tx_hash": "0xdeadbeef",
        "finalized": false
    }"#;
    let parsed: InferenceAttestationInfo = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.session_id, "sess-1");
    // 0x-prefixed hex is stored as-emitted; downstream consumers strip
    // the prefix themselves if they need bare hex.
    assert!(parsed.model_hash.starts_with("0x"));
    assert!(parsed.manifest_root.starts_with("0x"));
    assert!(parsed.response_hash.starts_with("0x"));
    assert!(parsed.proof_root.starts_with("0x"));
    assert!(parsed.verifier_signature.starts_with("0x"));
    assert!(parsed.tx_hash.starts_with("0x"));
    // included_at_height is REQUIRED on this DTO (the chain only emits
    // it for included attestations).
    assert_eq!(parsed.included_at_height, 50);
    assert!(!parsed.finalized);
}

#[test]
fn inference_attestation_info_finalized_true_round_trips() {
    let json = r#"{
        "session_id": "sess-2",
        "verifier_address": "addr-x",
        "model_hash": "0xaaaa",
        "manifest_root": "0xbbbb",
        "response_hash": "0xcccc",
        "proof_root": "0xdddd",
        "verifier_signature": "0xeeee",
        "included_at_height": 1024,
        "tx_hash": "0x12ab",
        "finalized": true
    }"#;
    let parsed: InferenceAttestationInfo = serde_json::from_str(json).unwrap();
    assert!(parsed.finalized);
    assert_eq!(parsed.included_at_height, 1024);
}
