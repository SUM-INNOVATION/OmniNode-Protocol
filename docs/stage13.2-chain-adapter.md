# Stage 13.2 — Real SUM Chain Adapter for Integrity-Evidence Anchors

**Status:** implemented. The Stage 13.0 stub remains the default-mode client for local-only flows; Stage 13.2 adds a real `omni-sumchain` `EvidenceAnchorChainClient` adapter that activates when operators supply `--rpc-url` + `--expect-chain-id`. **Stage 13.0 wire / schema / domain / reason tags / same-key model unchanged.**

**Co-references:**
- [`docs/stage13-evidence-anchor-spec.md`](stage13-evidence-anchor-spec.md) — frozen Stage 13.0 wire spec (the byte contract Stage 13.2 implements against).
- [`docs/stage13.1-chain-adapter-review.md`](stage13.1-chain-adapter-review.md) — chain-team review packet; historical, since chain-team review concluded.
- [`docs/operator-runbook.md`](operator-runbook.md) §Stage 13.0 — operator workflow, real-RPC flag table, chain-mode gate ordering, refusal taxonomy.
- `crates/omni-sumchain/src/anchor_dto.rs` — Stage 13.2 RPC DTOs (OmniNode-owned).
- `crates/omni-sumchain/src/anchor_tx.rs` — adapter call sites (submit + status query).
- `crates/omni-sumchain/src/client.rs` — `impl EvidenceAnchorChainClient for SumChainClient<T>`.
- `crates/omni-sumchain/src/rpc.rs` — `error_prefixes::*` constants + `ChainErrorCategory` + `classify_chain_client_error`.

## What Stage 13.2 ships

- **`impl EvidenceAnchorChainClient for SumChainClient<T>`** in `omni-sumchain`. Trait already existed (Stage 13.0); this is the real implementation.
- **No `TransactionV2` envelope.** Stage 13.2 uses a dedicated chain RPC (`sum_submitIntegrityEvidenceAnchor`) that accepts the 148-byte `IntegrityEvidenceAnchorTxData` directly. **No dependency on `sumchain-primitives` anchor types**; OmniNode owns its own DTOs and treats SUM Chain purely as an external JSON-RPC service.
- **Adapter-layer defense-in-depth gates** (`SumChainClient::submit_anchor`, before any RPC):
  1. **Activation gate** via `SumChainClient::integrity_evidence_anchor_is_active()`. Independent of `omninode_is_active` / `v2_is_active` (Blocker C resolved at Option C-new-flag).
  2. **Same-key submitter gate** — `self.seed()` must derive `tx_data.digest.signer_pubkey`. Stage 13.0 workflow already enforces this; the adapter re-checks at the boundary so non-CLI callers cannot bypass.
- **No adapter-layer chain-id gate.** The trait carries no `expected_chain_id` and `SumChainClient` stores only `(seed, transport)`. Chain-id enforcement lives at the CLI preflight layer, where `--expect-chain-id` is available.
- **CLI preflight (Stage 9a-style, before the workflow runs):**
  1. Fetch chain params once via `chain_getChainParams`.
  2. **`chain_id` sanity check** → `chain_id_mismatch` on mismatch.
  3. **Activation check, mainnet-aware tagging** (the locked corrected boundary):
     - `chain_id == 1 ∧ None` → `mainnet_policy_unresolved`
     - `chain_id == 1 ∧ Some(h) ∧ head < h` → `not_activated` (scheduled, not reached)
     - non-mainnet ∧ (`None` ∨ `Some(h)` with `head < h`) → `not_activated`
     - active → proceed
  4. Operator opt-ins (`--allow-submit`; mainnet additionally requires `--allow-mainnet-submit`).
  5. Call `submit_evidence_anchor_workflow(registry, &real_client, digest, &seed)`.
- **CLI subcommands gain chain-mode flags:**
  - `submit-integrity-evidence-anchor` — adds `--rpc-url`, `--expect-chain-id`, `--allow-submit`, `--allow-mainnet-submit`. Without `--rpc-url`, falls back to the Stage 13.0 stub path. Gated behind `--features submit`.
  - `query-integrity-evidence-anchor` — adds `--rpc-url`, `--expect-chain-id`. Without `--rpc-url`, registry-only (Stage 13.0). With `--rpc-url`: chain-read-only, local-registry-mutating.
  - `reconcile-integrity-evidence-anchor` — **new**. Required flags `--anchor-registry-dir`, `--rpc-url`, `--expect-chain-id`. Sweeps `Submitted` / `Included` records, queries chain per record, applies status transitions. Chain-read-only, local-registry-mutating. Mirrors Stage 5.3 `poll_attestations_workflow` semantics verbatim.
  - `verify-integrity-evidence-anchor` + `verify-integrity-evidence-anchor-file` — **unchanged**. Local-only.
- **Reason-tag mapping via typed classifier:**
  - `omni-sumchain` exposes `error_prefixes::*` constants (the literal strings) + `pub enum ChainErrorCategory` + `pub fn classify_chain_client_error(&ChainClientError) -> ChainErrorCategory`.
  - The CLI consumes the typed enum, **never inspects raw error strings**. The `crates/omni-sumchain/tests/error_prefix_classification_is_stable.rs` regression pins the literal prefix values; bumping any of them is a coordinated `omni-sumchain` + CLI change.
- **Six new closed-set `EvidenceAnchorError` variants** (Stage 13.0 tags unchanged):
  - `ChainIdMismatch { expected, actual }` → `reason=chain_id_mismatch`
  - `NotActivated { chain_id, activation_status }` → `reason=not_activated`
  - `MainnetPolicyUnresolved` → `reason=mainnet_policy_unresolved` (Stage 13.1 reserved this tag; Stage 13.2 wires it in)
  - `ChainRpc(String)` → `reason=chain_rpc`
  - `ChainSubmitRefused(String)` → `reason=chain_submit_refused`
  - `ChainResponseMalformed(String)` → `reason=chain_response_malformed`
- **`ChainParamsInfo` additively extended** with `integrity_evidence_anchor_enabled_from_height: Option<u64>` (the Q3 locked field name), `#[serde(default)]` so missing-field tolerance is preserved. Mirrors the two existing activation flags.

## What did NOT change (locked invariants)

- Stage 13.0 wire schema (`INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION = 1`, 84-byte canonical, 148-byte tx_data, `OMNINODE-INTEGRITY-EVIDENCE-ANCHOR:v1:` domain).
- Stage 13.0 same-key submitter rule.
- Stage 13.0 `AnchorStatus` enum (Q5: `included_at_height` parsed-but-dropped; no enum extension).
- Stage 12 schemas and verifier behavior.
- `verify-integrity-evidence-anchor` + `-file`: still local-only, still chain-untouched.
- Submit path still feature-gated behind `--features submit`.

## Chain-team review answers (per Stage 13.1 R-packet)

| Blocker | Resolution | Stage 13.2 wiring |
| --- | --- | --- |
| A — `TxPayload::IntegrityEvidenceAnchor` variant | **Not required.** Chain exposes dedicated RPC `sum_submitIntegrityEvidenceAnchor` accepting the 148-byte `IntegrityEvidenceAnchorTxData` directly; no `TransactionV2` envelope construction client-side. | OmniNode owns its own DTOs; zero vendored-anchor-type dependency. |
| B — Status RPC | `sum_getIntegrityEvidenceAnchorStatus(tx_hash)` with `{ "status": "...", "included_at_height": u64\|null, "reason": string\|null }`. By-artifact-hash deferred. | `omni_sumchain::anchor_tx::query_anchor_status` maps lowercase status string → `AnchorStatus`. |
| C — Activation flag | Option C-new-flag: chain adds `integrity_evidence_anchor_enabled_from_height: Option<u64>` to `chain_getChainParams`, independent of `omninode_is_active` / `v2_is_active`. | `ChainParamsInfo.integrity_evidence_anchor_enabled_from_height`; `SumChainClient::integrity_evidence_anchor_is_active`. |
| D — Mainnet policy | Mainnet remains **dormant by default**. Activation is a separate governance step. No assumed mainnet permission. | CLI preflight refuses mainnet `chain_id=1 ∧ None` with `mainnet_policy_unresolved`. Once mainnet sets `Some(h)`, refusal drops to `not_activated` until `head >= h`. |
| E — Fee schedule | Flat declared fee, min-fee floor, charged only on successful inclusion; pre-success rejects consume no fee, no nonce, no state. | OmniNode does not currently surface fee in the CLI — the chain RPC handles fee accounting server-side. Future enhancement if operators need declared-fee control. |

## Submit RPC contract (locked Stage 13.2 selected assumptions)

Per Stage 13.2 Q1/Q2 — implemented against these recommended shapes; if chain team revises later, the DTO + call site change in one place each.

- **Param shape (Q1):** single positional `["0x<bincode1_hex>"]`. Mirrors `sum_sendRawTransaction([hex])`.
- **Response shape (Q2):** lenient parser accepting either `{ "tx_hash": "0x..." }` (preferred) OR a bare `"0x..."` string (backwards-compat). Anything else → `chain_response_malformed`.

## Reason-tag mapper rule (closed)

`ChainClientError::Other(_)` produced by `omni-sumchain` routes through `classify_chain_client_error()` → `ChainErrorCategory` → CLI reason tag:

| `ChainErrorCategory` | CLI tag |
| --- | --- |
| `Transport` (HTTP / body read / serialize / non-JSON / missing `result`) | `chain_rpc` |
| `JsonRpcError` (chain returned a JSON-RPC `error` object) | `chain_submit_refused` |
| `Malformed` (response shape parse failure OR unrecognized status enum) | `chain_response_malformed` |
| `AdapterNotActivated` (`integrity_evidence_anchor not activated`) | `not_activated` |
| `AdapterSameKeyFail` (`same-key submitter check: …`) | `chain_rpc` (catch-all; defense-in-depth) |
| `Unknown` (string not recognized by classifier) | `chain_rpc` |

The CLI never inspects raw prefix strings. Adding a new error category is a coordinated `omni-sumchain` + CLI patch that the CLI's exhaustive `match` surfaces.

## Test inventory

In `crates/omni-sumchain/tests/`:
- `error_prefix_classification_is_stable.rs` — pins literal prefix values + classifier mapping for every category (13 tests).
- `anchor_adapter_integration.rs` (`--features submit`) — adapter activation gate (dormant + scheduled), same-key gate, submit happy path with 148-byte payload assertion, lenient response parser (object + bare-string), malformed response refusal, JSON-RPC error propagation, status mapping (each variant + foreign tx_hash + missing field + unrecognized status + optional fields default-None) (16 tests).
- `anchor_submit_feature_gate.rs` (default build only) — `submit_anchor` without `--features submit` refuses cleanly without reaching any chain RPC (1 test).

In `crates/omni-zkml/tests/`:
- `evidence_anchor_reconcile_integration.rs` — reconcile sweep semantics: empty registry, Submitted → Included → Finalized progression, chain `Unknown` observation-only, per-record RPC failure containment, non-Submitted/Non-Included skipping (5 tests).

All hermetic. **No live-chain tests.**

## Operator workflow examples

### Stage 13.0 stub mode (default — local testing, regen, no chain)

```sh
# Submit (stub client, in-memory tx_id).
omni-node operator evidence-anchor submit-integrity-evidence-anchor \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --submitter-seed     /etc/omni-node/chain-report-signer.seed \
  --anchor-registry-dir /var/omni-anchors
# event=integrity_evidence_anchor_submit_ok ... tx_id=anchor-00000000-<hash-prefix>
```

### Stage 13.2 chain mode (real SUM Chain)

```sh
# Submit (real RPC).
omni-node operator evidence-anchor submit-integrity-evidence-anchor \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --submitter-seed     /etc/omni-node/chain-report-signer.seed \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url            https://rpc.sumchain.example.com \
  --expect-chain-id    42 \
  --allow-submit
# event=integrity_evidence_anchor_submit_ok ... tx_id=0x<chain_tx_hash>

# Query (chain mode — chain-read-only, local-registry-mutating).
omni-node operator evidence-anchor query-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --tx-id              0x<chain_tx_hash> \
  --rpc-url            https://rpc.sumchain.example.com \
  --expect-chain-id    42

# Reconcile sweep (chain-read-only, local-registry-mutating).
omni-node operator evidence-anchor reconcile-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url            https://rpc.sumchain.example.com \
  --expect-chain-id    42

# Verify (UNCHANGED — local-only).
omni-node operator evidence-anchor verify-integrity-evidence-anchor \
  --signed-chain-report /var/omni-evidence/2026-06-15.chain.signed.json \
  --anchor-registry-dir /var/omni-anchors
```

### Mainnet (chain_id 1)

Submission requires:
- chain reports `integrity_evidence_anchor_enabled_from_height: Some(h)` and `head >= h`,
- `--allow-submit` AND `--allow-mainnet-submit`,
- `--expect-chain-id 1`.

If mainnet has `integrity_evidence_anchor_enabled_from_height = None`, submission refuses with `mainnet_policy_unresolved` regardless of opt-ins — chain governance must set activation first. If mainnet has `Some(h)` but `head < h`, refusal drops to `not_activated`.

## Out of scope (Stage 13.3+ candidates)

- Block-aware staleness sweep for anchors (analog of Stage 5.2 `StalenessPolicy`).
- Surface `included_at_height` through `AnchorStatus::Included { at_height: Option<u64> }` (Stage 13.0 enum bump, requires re-spec).
- `sum_getIntegrityEvidenceAnchorByHash(artifact_hash_hex)` chain RPC — deferred Stage 13.1 B4.
- Aggregation forms (anchor bundles, anchor chains).
- Separate-submitter / relay flows (would require an `anchor_schema_version` bump).
- Operator monitoring loop subcommand (`watch-anchor-activation` analog of `operator watch-activation`).
