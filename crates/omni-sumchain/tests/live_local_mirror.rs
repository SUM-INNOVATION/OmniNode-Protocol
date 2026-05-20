//! Stage 7a — live local-mirror tests.
//!
//! Every test in this file is `#[ignore]`'d by default and additionally
//! self-skips when `OMNINODE_SUMCHAIN_RPC_URL` is unset. CI never runs
//! these. Developers / operators run them explicitly once the chain
//! team's local-mirror branch is rebuilt with
//! `omninode_enabled_from_height: 0`:
//!
//! ```text
//! OMNINODE_SUMCHAIN_RPC_URL=http://localhost:8545 \
//!     cargo test -p omni-sumchain -- --ignored
//! ```
//!
//! See [`crates/omni-sumchain/README.md`](../README.md) for the full
//! operational setup (genesis funding via `extra-alloc.json`,
//! verifier-seed env var, etc.).

use omni_sumchain::{BlockFinality, SumChainClient};

/// Resolve the live-mirror RPC URL from the env var, or print a clear
/// skip message and return `None` so the test exits 0 instead of
/// failing.
fn live_url_or_skip(test: &str) -> Option<String> {
    match std::env::var("OMNINODE_SUMCHAIN_RPC_URL") {
        Ok(url) if !url.is_empty() => Some(url),
        _ => {
            eprintln!(
                "[skip] {test}: OMNINODE_SUMCHAIN_RPC_URL unset; live tests \
                 require an activated local-mirror endpoint"
            );
            None
        }
    }
}

fn dummy_seed() -> [u8; 32] {
    // Stage 7a live tests are read-only and never sign anything — any
    // 32 bytes are fine. Stage 7b live submit tests will load a real
    // funded seed from `OMNINODE_VERIFIER_SEED_HEX`.
    [0u8; 32]
}

/// Call `chain_getChainParams` against the live mirror; assert it
/// parses cleanly. Once the chain follow-up patch lands, also asserts
/// `omninode_enabled_from_height == Some(0)`. Pre-patch the field is
/// `None` and we surface that fact for the developer running the test.
#[test]
#[ignore]
fn live_get_chain_params() {
    let Some(url) = live_url_or_skip("live_get_chain_params") else { return };
    let client = SumChainClient::new(url, dummy_seed());

    let params = client.get_chain_params().expect("chain_getChainParams must parse");
    eprintln!(
        "[live] chain_getChainParams: finality_depth={} min_fee={} chain_id={} \
         omninode_enabled_from_height={:?}",
        params.finality_depth,
        params.min_fee,
        params.chain_id,
        params.omninode_enabled_from_height,
    );

    // Local mirror default sanity. Soft-asserted via eprintln + a
    // single assert_eq on chain_id (the most stable identifier) so
    // operators on a non-default port can override the URL but a
    // wrong chain entirely fails loud.
    assert_eq!(
        params.chain_id, 31337,
        "live mirror chain_id should be the documented local-mirror default 31337"
    );
}

/// Call `chain_getBlockHeight` for both finality tokens. Sanity-checks
/// the live mirror's responsiveness to both shapes.
#[test]
#[ignore]
fn live_get_block_height_both_finalities() {
    let Some(url) = live_url_or_skip("live_get_block_height_both_finalities") else { return };
    let client = SumChainClient::new(url, dummy_seed());

    let latest = client
        .get_block_height(BlockFinality::Latest)
        .expect("chain_getBlockHeight(latest) must parse");
    let finalized = client
        .get_block_height(BlockFinality::Finalized)
        .expect("chain_getBlockHeight(finalized) must parse");

    assert_eq!(latest.finality, "latest");
    assert_eq!(finalized.finality, "finalized");
    // Finalized must not exceed latest.
    assert!(finalized.height <= latest.height);
}

/// Query a tx_id the chain has never seen. Expected:
/// `AttestationStatus::Unknown`. Confirms the read path end-to-end.
#[test]
#[ignore]
fn live_query_unknown_tx_hash_returns_unknown() {
    use omni_zkml::{AttestationStatus, ChainClient};

    let Some(url) = live_url_or_skip("live_query_unknown_tx_hash_returns_unknown") else { return };
    let client = SumChainClient::new(url, dummy_seed());

    // 32 zero bytes hex-encoded with chain's `0x` prefix — guaranteed
    // never to be a real tx hash on a fresh local mirror.
    let bogus_tx = "0x0000000000000000000000000000000000000000000000000000000000000000";
    let status = client
        .query_attestation_status(bogus_tx)
        .expect("query for an unknown tx_hash must produce a parseable response");
    assert_eq!(status, AttestationStatus::Unknown);
}

/// After the chain follow-up patch lands, the local mirror will return
/// `omninode_enabled_from_height: Some(0)` and the mirror's head will
/// be >= 0 by construction, so `omninode_is_active` returns `true`.
/// Pre-patch this test would fail; that failure is the intended
/// signal that the chain patch hasn't landed yet on this mirror.
#[test]
#[ignore]
fn live_omninode_is_active_after_chain_patch() {
    let Some(url) = live_url_or_skip("live_omninode_is_active_after_chain_patch") else { return };
    let client = SumChainClient::new(url, dummy_seed());

    let active = client
        .omninode_is_active()
        .expect("omninode_is_active must succeed against an activated mirror");
    assert!(
        active,
        "expected omninode_is_active() == true on an activated local mirror; \
         if false, the chain follow-up patch exposing \
         `omninode_enabled_from_height` in chain_getChainParams has not yet \
         landed on this mirror"
    );
}

// ── Stage 7b — live submit roundtrip ────────────────────────────────────────

/// Submit a synthesized `InferenceAttestation` against the local
/// mirror and poll `query_attestation_status` until the lifecycle
/// progresses to `Included` or `Finalized`. Requires:
///
/// - `OMNINODE_SUMCHAIN_RPC_URL` — local-mirror endpoint.
/// - `OMNINODE_VERIFIER_SEED_HEX` — 64 hex chars (32-byte Ed25519 seed)
///   for a verifier whose derived address has been pre-funded via the
///   chain's `extra-alloc.json` before `docker-compose up`.
///
/// Skipped (with a clear message) if either env var is unset, so
/// `cargo test -- --ignored` without the env vars exits 0.
///
/// Stage 9a: `#[cfg(feature = "submit")]` — the submit path itself
/// only exists with the feature. The other live tests in this file
/// (chain params, block height, unknown tx) are read-only and stay
/// default-on.
#[cfg(feature = "submit")]
#[test]
#[ignore]
fn live_submit_roundtrip() {
    use std::time::{Duration, Instant};

    use omni_sumchain::BlockFinality;
    use omni_types::phase5::{
        InferenceAttestation, InferenceCommitment, SnipV2ObjectId,
    };
    use omni_zkml::{AttestationStatus, ChainClient};

    let Some(url) = live_url_or_skip("live_submit_roundtrip") else { return };
    let seed_hex = match std::env::var("OMNINODE_VERIFIER_SEED_HEX") {
        Ok(v) if !v.is_empty() => v,
        _ => {
            eprintln!(
                "[skip] live_submit_roundtrip: OMNINODE_VERIFIER_SEED_HEX unset; \
                 Stage 7b live submit needs a funded verifier seed"
            );
            return;
        }
    };

    // Parse the seed: 64 hex chars → [u8; 32].
    assert_eq!(
        seed_hex.len(),
        64,
        "OMNINODE_VERIFIER_SEED_HEX must be exactly 64 hex chars (32 bytes)"
    );
    let mut seed = [0u8; 32];
    for i in 0..32 {
        let pair = &seed_hex[i * 2..i * 2 + 2];
        seed[i] = u8::from_str_radix(pair, 16)
            .expect("OMNINODE_VERIFIER_SEED_HEX must be valid lowercase hex");
    }

    let client = SumChainClient::new(url, seed);

    // Guardrail: refuse to run against anything other than the local
    // mirror's documented chain_id.
    let params = client.get_chain_params().expect("chain_getChainParams must succeed");
    assert_eq!(
        params.chain_id, 31337,
        "expected local-mirror chain_id 31337; got {}. Refusing to submit \
         against a non-local-mirror endpoint",
        params.chain_id
    );

    // Both activation gates must be live.
    assert!(
        client.omninode_is_active().expect("omninode_is_active"),
        "omninode_is_active is false; chain follow-up patch may not have landed"
    );
    assert!(
        client.v2_is_active().expect("v2_is_active"),
        "v2_is_active is false; V2 envelope not yet activated"
    );

    // Synthesise a unique attestation per run so reruns don't collide
    // with a prior on-chain entry under the same (session_id, verifier)
    // de-dup key.
    let address = omni_zkml::signer_chain_address_base58(&seed).unwrap();
    let head = client
        .get_block_height(BlockFinality::Latest)
        .expect("chain_getBlockHeight")
        .height;
    let session_id = format!("omninode-stage7b-live-{}-{}", address, head);

    let attestation = InferenceAttestation {
        commitment: InferenceCommitment {
            session_id,
            model_hash: "a".repeat(64),
            manifest_snip_root: SnipV2ObjectId::from_bytes([0x11u8; 32]),
            response_hash: "b".repeat(64),
            proof_snip_root: SnipV2ObjectId::from_bytes([0x22u8; 32]),
        },
        verifier_address: address.clone(),
        verifier_signature: "live-stage7b-roundtrip".into(),
    };

    let receipt = client
        .submit_attestation(&attestation)
        .expect("live submit_attestation must succeed against a funded verifier");
    eprintln!("[live] submitted; tx_hash = {}", receipt.tx_id);

    // Poll status until Included / Finalized, with a generous timeout.
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut last_status = None;
    while Instant::now() < deadline {
        let status = client
            .query_attestation_status(&receipt.tx_id)
            .expect("query_attestation_status must succeed");
        match &status {
            AttestationStatus::Included | AttestationStatus::Finalized => {
                eprintln!("[live] tx reached {:?}", status);
                return;
            }
            AttestationStatus::Failed { reason } => {
                panic!("live submit progressed to Failed: {reason}");
            }
            other => {
                last_status = Some(other.clone());
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
    panic!(
        "live submit did not reach Included/Finalized within 30s; last seen \
         status: {last_status:?}"
    );
}
