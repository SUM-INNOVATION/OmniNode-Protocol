# omni-sumchain

Phase 5 Stage 7a + 7b — SUM Chain adapter for `omni-zkml::ChainClient`.

**Stage 7a (read/query):** `query_attestation_status`,
`get_attestation`, `list_attestations`, `get_chain_params`,
`get_block_height(BlockFinality)`, `get_nonce`, `omninode_is_active`,
`v2_is_active`.

**Stage 7b (submit):** `submit_attestation` is fully implemented using
vendored chain primitives (`sumchain-primitives` + `sumchain-crypto`
pinned to chain rev `d83e45a4`). Flow: four pre-flight gates →
Stage 6 inner pipeline → local-to-vendored conversion (parity-tested
under bincode 1.3) → outer-tx assembly → outer `BLAKE3+Ed25519` sign
via `sumchain-crypto` → bare-hex `sum_sendRawTransaction`.

## What ships

### Stage 7a (read/query)

- `SumChainClient<T: JsonRpcTransport>` (defaults to `UreqTransport`):
  - **`ChainClient::query_attestation_status(tx_id)`** — maps
    `InferenceAttestationStatusInfo` into
    `omni_zkml::AttestationStatus` via strict variant matching.
  - **Inherent read helpers** — `query_attestation_status_full`,
    `get_attestation`, `list_attestations`, `get_chain_params`,
    `get_block_height(BlockFinality)`, `get_nonce`,
    `omninode_is_active`, `v2_is_active`.
- **Read DTOs**: `InferenceAttestationStatusInfo`,
  `InferenceAttestationInfo`, `BlockHeightInfo`, `ChainParamsInfo`
  (with `omninode_enabled_from_height` and `v2_enabled_from_height`
  both `#[serde(default)] Option<u64>` for forward-compat).
- **`JsonRpcTransport`** trait, plus production `UreqTransport` (sync
  HTTP via `ureq`) and `FakeJsonRpcTransport` (hermetic test fixture
  with an `Arc<Mutex<_>>` state).

### Stage 7b (submit)

- **`ChainClient::submit_attestation(...)`** — real implementation. Four
  pre-flight gates (omninode-active, v2-active, verifier-address
  consistency), then Stage 6 inner pipeline, then conversion to
  vendored `sumchain_primitives` types, then outer-tx assembly + sign,
  then bare-hex `sum_sendRawTransaction`. Submission flow detailed in
  [`src/tx.rs`](src/tx.rs).
- **Vendored chain primitives** at rev `d83e45a4`:
  `sumchain-primitives` (TransactionV2, TxPayload, SignedTransaction,
  Address) and `sumchain-crypto` (Ed25519 sign/verify, key derivation).
- **Parity-verified** local-to-vendored byte equivalence under bincode
  1.3 (3 tests in `tests/parity_vendored_primitives.rs`).

Default `cargo test` is fully hermetic — `UreqTransport` is exercised
only by `#[ignore]`'d live tests gated on env vars.

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
# Requires GitHub credentials for SUM-INNOVATION/sum-chain on first
# build (vendored chain primitives) — see "Auth setup" below.
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo test -p omni-sumchain

# Live read tests against an activated local mirror (developer-only).
OMNINODE_SUMCHAIN_RPC_URL=http://localhost:8545 \
    cargo test -p omni-sumchain -- --ignored

# Live Stage 7b submit roundtrip (developer-only; requires a funded
# verifier address pre-allocated via extra-alloc.json).
OMNINODE_SUMCHAIN_RPC_URL=http://localhost:8545 \
OMNINODE_VERIFIER_SEED_HEX=<64 hex chars> \
    cargo test -p omni-sumchain -- --ignored
```

`#[ignore]`'d live tests self-skip when the required env vars are
unset, so `cargo test -- --ignored` without them still exits 0.

## Auth setup (vendored chain deps)

The chain repo `SUM-INNOVATION/sum-chain` is currently private. The
workspace pins the chain primitives via Cargo git deps at rev
`d83e45a4`; first `cargo fetch` / `cargo build` against the workspace
needs git auth that has read access to the chain repo.

Easiest path: configure GitHub auth via the `gh` CLI or an SSH key,
then set `CARGO_NET_GIT_FETCH_WITH_CLI=true` so cargo defers to git's
native auth (which respects `gh auth`, SSH keys, OS keychain, etc.):

```bash
export CARGO_NET_GIT_FETCH_WITH_CLI=true
cargo fetch  # one-time; lock file pins the chain commit
```

CI invocations should set the same env var and either ship a deploy
key or use a PAT in a credential helper.

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

## Stage 7b — submission flow (shipped)

The full chain-confirmed construction sequence is implemented in
[`src/tx.rs::build_and_submit_signed_transaction`](src/tx.rs):

1. **Cache `chain_getChainParams`** once (reused across both
   activation gates and the `TransactionV2` build).
2. **Cache `chain_getBlockHeight(Latest)`** at most once (skipped if
   neither activation field is `Some(_)`).
3. **OmniNode activation gate** — `omninode_enabled_from_height` must
   be `Some(h)` with `head >= h`; otherwise typed error.
4. **V2 envelope activation gate** — `v2_enabled_from_height` must be
   `Some(h)` with `head >= h`; otherwise typed error.
5. **Verifier-address consistency** —
   `omni_zkml::signer_chain_address_base58(&seed)` must equal
   `attestation.verifier_address`; refuses to submit otherwise (no
   nonce or send RPC reached).
6. **Stage 6 inner pipeline** — `commitment_to_chain_digest` →
   `sign_chain_attestation_digest` (inner Ed25519 over `DOMAIN_TAG ||
   bincode_1_3(digest)`).
7. **Local → vendored conversion** — field-by-field copy of
   `omni_zkml::InferenceAttestationDigest` →
   `sumchain_primitives::InferenceAttestationDigest`; parity-proven
   byte-equivalent under bincode 1.3.
8. **`sum_getNonce(verifier_address)`** — fetched only after gates pass.
9. **`TransactionV2 { chain_id, from, fee: min_fee as u128, nonce,
   payload }`** — `chain_id` from params, `from =
   Address::from_public_key(pubkey)`, fee widened from the DTO's `u64`
   to the chain's `Balance` (= `u128`), payload =
   `TxPayload::InferenceAttestation(tx_data)`.
10. **Outer signing** via `sumchain-crypto`:
    `outer_hash = TransactionV2::signing_hash()` (BLAKE3 of bincode 1.3
    of the tx); `outer_sig = sumchain_crypto::sign(outer_hash.as_bytes(),
    &PrivateKey)`; `SignedTransaction::new_v2(tx, sig_bytes,
    pubkey_bytes)`.
11. **Submission** — `signed.to_hex()` produces **bare hex** (no `0x`
    prefix); `sum_sendRawTransaction([hex])` accepts bare. Response
    `tx_hash` is `0x`-prefixed and propagates verbatim into
    `SubmissionReceipt::tx_id`.

The runtime invariant is: `sum_getNonce` and `sum_sendRawTransaction`
are **never** reached on any gate failure. Pinned by
[`tests/unit_submit_construction.rs`](tests/unit_submit_construction.rs).
