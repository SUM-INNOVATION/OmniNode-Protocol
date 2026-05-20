//! Stage 9a: when `omni-sumchain` is built without the `submit`
//! feature (the default), `SumChainClient::submit_attestation` still
//! satisfies the `omni_zkml::ChainClient` trait but returns a typed
//! `ChainClientError::Other(_)` whose message names the feature.
//! Read-only consumers (poll/query workflows) continue to compile and
//! work against the same client without the vendored chain primitives
//! ever being fetched.

#![cfg(not(feature = "submit"))]

use omni_sumchain::{FakeJsonRpcTransport, SumChainClient};
use omni_types::phase5::{InferenceAttestation, InferenceCommitment, SnipV2ObjectId};
use omni_zkml::{ChainClient, ChainClientError};

#[test]
fn submit_attestation_without_feature_returns_typed_error() {
    let fake = FakeJsonRpcTransport::new();
    let client = SumChainClient::with_transport([0u8; 32], fake.clone());

    let attestation = InferenceAttestation {
        commitment: InferenceCommitment {
            session_id: "no-submit-feature-test".into(),
            model_hash: "a".repeat(64),
            manifest_snip_root: SnipV2ObjectId::from_bytes([0x11u8; 32]),
            response_hash: "b".repeat(64),
            proof_snip_root: SnipV2ObjectId::from_bytes([0x22u8; 32]),
        },
        verifier_address: "addr".into(),
        verifier_signature: "sig".into(),
    };

    let err = client.submit_attestation(&attestation).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.contains("submit"),
        "expected error message to name the `submit` feature; got: {msg}"
    );

    // Crucially: zero chain calls were made. The error is a pure
    // compile-time-driven runtime stub.
    assert!(
        fake.calls().is_empty(),
        "no-submit submit_attestation must not reach the transport; \
         got calls: {:?}",
        fake.calls()
    );
}
