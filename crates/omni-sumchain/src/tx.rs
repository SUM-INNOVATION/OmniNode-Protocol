//! Phase 5 Stage 7b — outer `SignedTransaction` construction & submission.
//!
//! Implements the chain-confirmed submit flow against the vendored
//! `sumchain-primitives` types at chain rev `d83e45a4`. Called by
//! [`crate::SumChainClient::submit_attestation`].
//!
//! # Flow
//!
//! Gates (no `sum_getNonce` or `sum_sendRawTransaction` is reached
//! unless all four pass):
//!
//! 1. `self.omninode_is_active()` — chain accepts the OmniNode subprotocol.
//! 2. `self.v2_is_active()` — chain accepts V2 envelope.
//! 3. `attestation.verifier_address == self.derived_verifier_address()`
//!    — sender == verifier per chain rule; refusing to submit on
//!    mismatch saves the operator from burning a fee on a tx the
//!    chain would reject anyway.
//!
//! Inner pipeline (Stage 6, reused unchanged):
//!
//! 4. `omni_zkml::commitment_to_chain_digest(&attestation.commitment)`
//!    → local `InferenceAttestationDigest`.
//! 5. `omni_zkml::sign_chain_attestation_digest(&seed, &digest)` →
//!    64-byte inner Ed25519 signature over
//!    `DOMAIN_TAG || bincode_1_3(digest)`.
//!
//! Local → vendored conversion (Stage 7b boundary):
//!
//! 6. Field-by-field copy into
//!    `sumchain_primitives::InferenceAttestationDigest`. Parity test
//!    [`tests/parity_vendored_primitives.rs`] proves this is byte-
//!    preserving under bincode 1.3.
//! 7. Wrap as `sumchain_primitives::InferenceAttestationTxData
//!    { digest, verifier_signature: inner_sig }`.
//!
//! Outer transaction (vendored chain types):
//!
//! 8. Read nonce: `self.get_nonce(&derived_address)`.
//! 9. Build `TransactionV2 { chain_id, from, fee, nonce, payload }`.
//!    `chain_id` and `min_fee` come from `chain_getChainParams`
//!    (already read for the gates above; reused without a second RPC
//!    call). `fee` widens `params.min_fee: u64 as u128` to fit the
//!    chain's `Balance` type. `from = Address::from_public_key(pubkey)`.
//! 10. Outer sign via [`crate::outer_sign::outer_sign_transaction_v2`]:
//!     `outer_hash = TransactionV2::signing_hash()` (BLAKE3 of bincode
//!     1.3 of the tx); `outer_sig = Ed25519_sign(outer_hash.as_bytes())`;
//!     `SignedTransaction::new_v2(tx, sig_bytes, pubkey_bytes)`.
//! 11. Submit: `signed_tx.to_hex()` produces **bare hex** (no `0x`
//!     prefix); `sum_sendRawTransaction([hex])` accepts bare hex; the
//!     response is a `0x`-prefixed tx hash and is propagated unchanged
//!     into `SubmissionReceipt::tx_id`.

use omni_types::phase5::InferenceAttestation;
use omni_zkml::{ChainClientError, SubmissionReceipt};
use sumchain_primitives::{
    inference_attestation::{
        InferenceAttestationDigest as ChainDigest, InferenceAttestationTxData,
    },
    Address, TransactionV2, TxPayload,
};

use crate::client::SumChainClient;
use crate::dto::BlockFinality;
use crate::outer_sign::outer_sign_transaction_v2;
use crate::rpc::JsonRpcTransport;

/// Stage 7b real implementation. Stage 5.1 contract preserved: any
/// error from this function surfaces through the submit workflow as
/// `RegistryError::ChainClient(_)` and leaves the local record at its
/// pre-submit state (`Pending` or `Dropped`).
pub(crate) fn build_and_submit_signed_transaction<T: JsonRpcTransport>(
    client: &SumChainClient<T>,
    attestation: &InferenceAttestation,
) -> std::result::Result<SubmissionReceipt, ChainClientError> {
    // ── Step 1: fetch chain params once; reuse across gates 1+2+9 ────
    let params = client.get_chain_params()?;

    // Fetch `chain_getBlockHeight` at most once, only if at least one
    // activation field is `Some(_)`. Cached `Option<u64>` shared across
    // both gates; gates that don't need it (field is `None`) don't
    // force the call.
    let head = if params.omninode_enabled_from_height.is_some()
        || params.v2_enabled_from_height.is_some()
    {
        Some(client.get_block_height(BlockFinality::Latest)?.height)
    } else {
        None
    };

    let activation_ok = |activation: Option<u64>, head: Option<u64>| -> bool {
        match (activation, head) {
            (Some(h), Some(current)) => current >= h,
            _ => false,
        }
    };

    // ── Step 2: OmniNode activation gate ─────────────────────────────
    if !activation_ok(params.omninode_enabled_from_height, head) {
        return Err(ChainClientError::Other(format!(
            "OmniNode subprotocol not activated on the connected chain: \
             chain_getChainParams.omninode_enabled_from_height = {:?}. \
             Refusing to submit — sum_sendRawTransaction would be \
             rejected by the chain.",
            params.omninode_enabled_from_height,
        )));
    }

    // ── Step 3: V2 envelope activation gate ──────────────────────────
    if !activation_ok(params.v2_enabled_from_height, head) {
        return Err(ChainClientError::Other(format!(
            "V2 transaction envelope not activated on the connected \
             chain: chain_getChainParams.v2_enabled_from_height = {:?}. \
             Refusing to submit — the chain would reject the outer \
             SignedTransaction format.",
            params.v2_enabled_from_height,
        )));
    }

    // ── Step 4: verifier address consistency gate (no RPC) ───────────
    let derived = client.derived_verifier_address()?;
    if derived != attestation.verifier_address {
        return Err(ChainClientError::Other(format!(
            "verifier address mismatch: attestation was built for \
             verifier_address = {}, but this SumChainClient is \
             configured with a seed deriving to {}. The chain enforces \
             sender == verifier; refusing to submit (no sum_getNonce / \
             sum_sendRawTransaction call made).",
            attestation.verifier_address, derived
        )));
    }

    // [Gates passed — sum_getNonce and sum_sendRawTransaction unlocked]

    // ── Step 5: Stage 6 inner pipeline ───────────────────────────────
    let digest_local = omni_zkml::commitment_to_chain_digest(&attestation.commitment)
        .map_err(|e| {
            ChainClientError::Other(format!(
                "Stage 6 commitment_to_chain_digest failed: {e}"
            ))
        })?;
    let inner_sig = omni_zkml::sign_chain_attestation_digest(
        client.seed(),
        &digest_local,
    )
    .map_err(|e| {
        ChainClientError::Other(format!(
            "Stage 6 sign_chain_attestation_digest failed: {e}"
        ))
    })?;

    // ── Step 6: convert local → vendored (parity-tested byte-preserving) ─
    let digest_chain = ChainDigest {
        session_id: digest_local.session_id.clone(),
        model_hash: digest_local.model_hash,
        manifest_root: digest_local.manifest_root,
        response_hash: digest_local.response_hash,
        proof_root: digest_local.proof_root,
    };

    // ── Step 7: wrap as vendored inner payload ───────────────────────
    let tx_data = InferenceAttestationTxData {
        digest: digest_chain,
        verifier_signature: inner_sig,
    };

    // ── Step 8: fetch nonce for the (now confirmed) verifier ─────────
    let nonce = client.get_nonce(&derived)?;

    // ── Step 9: assemble TransactionV2 ───────────────────────────────
    let pubkey = omni_zkml::signer_pubkey_bytes(client.seed()).map_err(|e| {
        ChainClientError::Other(format!("signer_pubkey_bytes failed: {e}"))
    })?;
    let from = Address::from_public_key(&pubkey);

    let tx = TransactionV2 {
        chain_id: params.chain_id,
        from,
        fee: params.min_fee as u128, // widen u64 → Balance (= u128)
        nonce,
        payload: TxPayload::InferenceAttestation(tx_data),
    };

    // ── Step 10: outer sign via sumchain-crypto ──────────────────────
    let signed = outer_sign_transaction_v2(client.seed(), &tx)?;

    // ── Step 11: submit ──────────────────────────────────────────────
    // `to_hex()` returns BARE hex (no `0x` prefix); the chain accepts
    // both. Chain response is a `0x`-prefixed tx hash, propagated
    // unchanged into `SubmissionReceipt::tx_id`.
    let hex = signed.to_hex();
    let result = client
        .transport()
        .call("sum_sendRawTransaction", serde_json::json!([hex]))?;
    let tx_hash = result.as_str().ok_or_else(|| {
        ChainClientError::Other(format!(
            "sum_sendRawTransaction returned non-string result: {result}"
        ))
    })?;

    tracing::info!(
        tx_hash = %tx_hash,
        chain_id = params.chain_id,
        fee = params.min_fee,
        "submitted InferenceAttestation to SUM Chain"
    );

    Ok(SubmissionReceipt {
        tx_id: tx_hash.to_string(),
        note: None,
    })
}
