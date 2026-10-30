//! Issue #129 — `SumChainClient::submit_digest`, hermetic.
//!
//! A caller with its own five digest fields (no OmniNode inference
//! session) submits through the same path as `submit_attestation`: same
//! gates, same transaction bytes for the same digest, same receipt.
#![cfg(feature = "submit")]

use omni_sumchain::{
    FakeJsonRpcTransport, InferenceAttestationDigest, MAX_SESSION_ID_BYTES, SumChainClient,
};
use omni_types::phase5::{InferenceAttestation, InferenceCommitment, SnipV2ObjectId};
use omni_zkml::{ChainClient, ChainClientError};

const SEED: [u8; 32] = [42u8; 32];

fn seed_address() -> String {
    omni_zkml::signer_chain_address_base58(&SEED).unwrap()
}

fn commitment() -> InferenceCommitment {
    InferenceCommitment {
        session_id: "swum/1/0011223344556677/9".into(),
        model_hash: "0a".repeat(32),
        manifest_snip_root: SnipV2ObjectId::from_bytes([0x11u8; 32]),
        response_hash: "1b".repeat(32),
        proof_snip_root: SnipV2ObjectId::from_bytes([0x22u8; 32]),
    }
}

/// The digest a caller would build itself for `commitment()`.
fn digest() -> InferenceAttestationDigest {
    InferenceAttestationDigest {
        session_id: "swum/1/0011223344556677/9".into(),
        model_hash: [0x0a; 32],
        manifest_root: [0x11; 32],
        response_hash: [0x1b; 32],
        proof_root: [0x22; 32],
    }
}

fn happy_fake() -> FakeJsonRpcTransport {
    let fake = FakeJsonRpcTransport::new();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 1337,
            "omninode_enabled_from_height": 0,
            "v2_enabled_from_height": 0,
        })),
    );
    fake.set_response(
        "chain_getBlockHeight",
        Ok(serde_json::json!({"height": 5, "finality": "latest"})),
    );
    fake.set_response("sum_getNonce", Ok(serde_json::json!(7)));
    fake.set_response(
        "sum_sendRawTransaction",
        Ok(serde_json::json!({ "tx_hash": "0xfeedface" })),
    );
    fake
}

fn methods(fake: &FakeJsonRpcTransport) -> Vec<String> {
    fake.calls().into_iter().map(|(m, _)| m).collect()
}

fn sent_hex(fake: &FakeJsonRpcTransport) -> String {
    fake.calls()
        .into_iter()
        .find(|(m, _)| m == "sum_sendRawTransaction")
        .and_then(|(_, p)| p.as_array()?.first()?.as_str().map(str::to_string))
        .expect("sum_sendRawTransaction called with a hex string")
}

#[test]
fn the_digest_a_commitment_converts_to_is_the_callers_digest() {
    assert_eq!(
        omni_zkml::commitment_to_chain_digest(&commitment()).unwrap(),
        digest()
    );
}

#[test]
fn submit_digest_sends_the_transaction_submit_attestation_sends() {
    let by_attestation = happy_fake();
    let receipt_a = SumChainClient::with_transport(SEED, by_attestation.clone())
        .submit_attestation(&InferenceAttestation {
            commitment: commitment(),
            verifier_address: seed_address(),
            verifier_signature: "unused".into(),
        })
        .unwrap();

    let by_digest = happy_fake();
    let receipt_d = SumChainClient::with_transport(SEED, by_digest.clone())
        .submit_digest(&digest())
        .unwrap();

    assert_eq!(sent_hex(&by_digest), sent_hex(&by_attestation));
    assert_eq!(methods(&by_digest), methods(&by_attestation));
    assert_eq!(receipt_d.tx_id, "0xfeedface");
    assert_eq!(receipt_d.tx_id, receipt_a.tx_id);
}

#[test]
fn submit_digest_signs_the_digest_with_the_clients_seed_as_the_sender() {
    use sumchain_primitives::{SignedTransaction, TxInner, TxPayload};

    let fake = happy_fake();
    SumChainClient::with_transport(SEED, fake.clone())
        .submit_digest(&digest())
        .unwrap();

    // Nonce read for the seed's own address.
    let nonce = fake
        .calls()
        .into_iter()
        .find(|(m, _)| m == "sum_getNonce")
        .unwrap();
    assert_eq!(nonce.1, serde_json::json!([seed_address()]));

    let signed = SignedTransaction::from_hex(&sent_hex(&fake)).unwrap();
    let TxInner::V2(tx) = signed.inner else {
        panic!("expected a V2 transaction");
    };
    assert_eq!(tx.chain_id, 1337);
    assert_eq!(tx.nonce, 7);
    assert_eq!(tx.fee, 1);
    assert_eq!(tx.from.to_base58(), seed_address());
    let TxPayload::InferenceAttestation(data) = &tx.payload else {
        panic!("not an inference attestation: {:?}", tx.payload);
    };
    assert_eq!(data.digest.session_id, digest().session_id);
    assert_eq!(data.digest.model_hash, digest().model_hash);
    assert_eq!(data.digest.manifest_root, digest().manifest_root);
    assert_eq!(data.digest.response_hash, digest().response_hash);
    assert_eq!(data.digest.proof_root, digest().proof_root);
    assert_eq!(
        data.verifier_signature,
        omni_sumchain::sign_chain_attestation_digest(&SEED, &digest()).unwrap()
    );
}

#[test]
fn submit_digest_refuses_when_omninode_is_not_active() {
    let fake = happy_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10, "min_fee": 1, "chain_id": 1337,
            "v2_enabled_from_height": 0,
        })),
    );
    let err = SumChainClient::with_transport(SEED, fake.clone())
        .submit_digest(&digest())
        .unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.contains("OmniNode subprotocol not activated"), "{msg}");
    assert!(
        !methods(&fake)
            .iter()
            .any(|m| m == "sum_getNonce" || m == "sum_sendRawTransaction")
    );
}

#[test]
fn submit_digest_refuses_when_v2_is_not_active_at_the_head() {
    let fake = happy_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10, "min_fee": 1, "chain_id": 1337,
            "omninode_enabled_from_height": 0, "v2_enabled_from_height": 100,
        })),
    );
    let err = SumChainClient::with_transport(SEED, fake.clone())
        .submit_digest(&digest())
        .unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.contains("V2 transaction envelope not activated"),
        "{msg}"
    );
    assert!(
        !methods(&fake)
            .iter()
            .any(|m| m == "sum_getNonce" || m == "sum_sendRawTransaction")
    );
}

#[test]
fn submit_digest_refuses_an_overlong_session_id_before_any_rpc() {
    let fake = happy_fake();
    let mut d = digest();
    d.session_id = "s".repeat(MAX_SESSION_ID_BYTES + 1);
    let err = SumChainClient::with_transport(SEED, fake.clone())
        .submit_digest(&d)
        .unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.contains("session_id"), "{msg}");
    assert!(fake.calls().is_empty(), "{:?}", fake.calls());

    // The bound itself is accepted.
    d.session_id = "s".repeat(MAX_SESSION_ID_BYTES);
    SumChainClient::with_transport(SEED, happy_fake())
        .submit_digest(&d)
        .unwrap();
}

#[test]
fn submit_digest_reports_a_refused_submission() {
    let fake = happy_fake();
    fake.set_response(
        "sum_sendRawTransaction",
        Ok(serde_json::json!({ "status": "ok" })),
    );
    let err = SumChainClient::with_transport(SEED, fake)
        .submit_digest(&digest())
        .unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.contains("missing 'tx_hash'"), "{msg}");
}
