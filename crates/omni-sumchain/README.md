# omni-sumchain

Phase 5 Stage 7a — SUM Chain adapter for `omni-zkml::ChainClient`.

**Stage 7a is read/query only.** `submit_attestation` returns a typed
`ChainClientError::Other(_)` describing exactly why; the real submit
path lands in Stage 7b after the chain team confirms the primitive
vendoring strategy. Do not wire `SumChainClient` into a production
submit workflow before Stage 7b.

## What ships in Stage 7a

- `SumChainClient<T: JsonRpcTransport>` (defaults to `UreqTransport`):
  - **`ChainClient::query_attestation_status(tx_id)`** — implemented; maps
    the chain `InferenceAttestationStatusInfo` into
    `omni_zkml::AttestationStatus`.
  - **`ChainClient::submit_attestation(...)`** — typed-error stub.
  - **Inherent read helpers** — `query_attestation_status_full`,
    `get_attestation`, `list_attestations`, `get_chain_params`,
    `get_block_height(BlockFinality)`, `get_nonce`, `omninode_is_active`.
- **Read DTOs**: `InferenceAttestationStatusInfo`,
  `InferenceAttestationInfo`, `BlockHeightInfo`, `ChainParamsInfo`.
- **`JsonRpcTransport`** trait, plus production `UreqTransport` (sync
  HTTP via `ureq`) and `FakeJsonRpcTransport` (hermetic test fixture
  with an `Arc<Mutex<_>>` state).

Default `cargo test` is fully hermetic — production `UreqTransport` is
exercised only by `#[ignore]`'d live tests gated on an env var.

## Stage 5.1 contract reaffirmation

- Chain `"unknown"` maps to `AttestationStatus::Unknown` and is
  **non-terminal**. The Stage 5.1 `query_attestation_workflow` leaves
  the record unchanged and logs a warning.
- The chain never returns `Dropped`. Local
  `LocalAttestationStatus::Dropped` is set only by Stage 5.2 staleness
  detection (planned, not in this stage).
- `query_attestation_status` is keyed by `tx_id: &str`; the workflow
  extracts `record.receipt.tx_id` before calling. If a queryable
  record's receipt is missing, the workflow returns
  `RegistryError::SubmittedRecordMissingReceipt` and never reaches
  the chain.

## Running tests

```bash
# Hermetic (default) — runs in CI; no network.
cargo test -p omni-sumchain

# Live read tests against an activated local mirror (developer-only).
OMNINODE_SUMCHAIN_RPC_URL=http://localhost:8545 \
    cargo test -p omni-sumchain -- --ignored
```

`#[ignore]`'d live tests self-skip when the env var is unset, so
`cargo test -- --ignored` without the URL still exits 0.

## Operational setup for live tests

1. **Generate an Ed25519 seed** locally (32 bytes). Never commit it.
2. **Derive the verifier address** with
   `omni_zkml::signer_chain_address_base58(&seed)` (Stage 6 helper).
3. **Fund the address** by adding it with a balance to the chain's
   `extra-alloc.json` *before* the first `docker-compose up`. Per the
   chain team, no private seeds are committed to the chain repo —
   each OmniNode operator owns their own.
4. **Bring up the local mirror.** Once the chain follow-up patch is in
   place, the mirror's config sets
   `omninode_enabled_from_height: 0`.
5. **Run live tests** with `OMNINODE_SUMCHAIN_RPC_URL=http://localhost:8545`.
   Stage 7b live submit tests will additionally read
   `OMNINODE_VERIFIER_SEED_HEX=<64 hex chars>`.

The local mirror's documented defaults are:

| | Value |
|---|---|
| RPC URL | `http://localhost:8545` |
| `chain_id` | `31337` |
| `min_fee` | `1` |

Live tests assert `chain_id == 31337` to fail loud when pointed at the
wrong endpoint.

## Stage 7b — what's coming

The construction sequence is documented in `src/tx.rs`'s module rustdoc.
Outline:

1. Stage 6 inner pipeline → `InferenceAttestationDigest` + 64-byte
   `verifier_signature`.
2. Wrap as the chain inner payload (`InferenceAttestationTxData`).
3. Build `TransactionV2 { chain_id, from, fee, nonce, payload }`:
   - `chain_id` from `get_chain_params()?.chain_id` (= `31337` on
     local mirror).
   - `from = Address::from_public_key(verifier_pubkey)`.
   - `fee` defaults to `params.min_fee` unless the caller overrides.
   - `nonce` from `get_nonce(&verifier_address)?`.
4. Outer canonical bytes = raw bincode 1.3 of `TransactionV2`
   (no domain tag; `chain_id` provides replay protection).
5. Outer hash = BLAKE3 of the canonical bytes. Outer signature =
   Ed25519 over the 32-byte hash (note: this is different from Stage 6's
   inner pattern, which signs `DOMAIN_TAG || canonical` bytes directly).
6. `SignedTransaction::new_v2(tx, outer_signature, verifier_pubkey)`.
7. `signed_tx.to_hex()` returns **bare hex** (no `0x` prefix);
   `sum_sendRawTransaction(hex)` accepts bare hex.
8. Response `tx_hash` is `0x`-prefixed and propagates to
   `SubmissionReceipt::tx_id` as-is.

Stage 7b's only remaining decision is the chain primitive vendoring
strategy:

- **Preferred:** vendor `TransactionV2`, `TxPayload`,
  `SignedTransaction`, `Address` from a publishable chain primitives
  crate (or pinned git dep).
- **Fallback:** mirror them locally with a parity fixture
  (`SignedTransaction` hex blob from chain code, committed under
  `tests/fixtures/`).

See the workspace [README](../../README.md) Stage 7 section and the
chain-team open questions for the activation patch timing.
