# Stage 13.9 — SUM Chain read/reconcile integration for integrity-evidence anchors

## Stage scope

Stage 13.9 integrates the filled SUM Chain read/reconcile contract for `IntegrityEvidenceAnchor` into OmniNode's read-only anchor lifecycle. It adds:

- **Batch status RPC** (`sum_getIntegrityEvidenceAnchorStatusBatch`) — up to 100 tx_ids per call; ordered response with per-item `result` / `error` containment.
- **By-tuple lookup RPC** (`sum_getIntegrityEvidenceAnchorByTuple`) — 5-tuple lookup yielding the chain's canonical tx_hash. **Durable; never pruned.**
- **Richer single-status response** — `code: Option<u32>` parsed and surfaced. Stable on `failed`; opaque on other statuses.
- **`AnchorStatusReport` DTO** carrying `included_at_height` / `code` / opaque `reason` alongside the closed `AnchorStatus` enum.
- **`reconcile_evidence_anchors_workflow` rewrite** using batch (chunks of 100) with chunk-level fail-fast fan-out and the locked Stage 13.9 transition table.
- **`lookup-integrity-evidence-anchor-by-tuple` CLI** — operator-facing chain query that derives the tuple from a local record.
- **Stage 13.9 read-path mapper** that NEVER produces `chain_submit_refused` for JSON-RPC errors on read flows.

This is the last stage in the 13.x evidence-anchor track. Stage 13.8's local consistency report is the recommended preflight before invoking any Stage 13.9 chain reconcile.

## What Stage 13.9 does NOT ship

- ❌ No Stage 13.0 wire / schema / domain / canonical-bytes / signing changes.
- ❌ No private chain repo / `sumchain-primitives` dep.
- ❌ No submit-path changes — the existing `submit_anchor` semantics are untouched.
- ❌ No live-chain tests. All tests `FakeJsonRpcTransport`-based.
- ❌ No artifact-hash-only lookup (chain contract doesn't offer it).
- ❌ No reorg-aware status downgrade — `Submitted` from chain is **observation-only** regardless of prior local state.
- ❌ No registry mutation on by-tuple lookup — strictly read-only on the local registry (Q3 lock).
- ❌ No new `EvidenceAnchorError` variants. No new closed `reason=` tag strings. `error.rs` unchanged.

## Locked decisions (Q&A + v2 REJECT findings)

- ✅ **Q1 — extend the existing trait with default-impl'd methods.** No separate extension trait. Stubs gain the new methods transparently.
- ✅ **Q2 — by-tuple CLI command** `lookup-integrity-evidence-anchor-by-tuple`. Inputs come from local record selectors (`--artifact-hash-hex` / `--tx-id`), NEVER raw tuple flags (operator-footgun guard).
- ✅ **Q3 — by-tuple is read-only on the local registry.** Future stage may add gated `--apply-canonical-tx-id`.
- ✅ **Q4 — `AnchorStatus` enum unchanged.** New `AnchorStatusReport` DTO carries `included_at_height` / `code` / opaque `reason` without growing the closed status enum.
- ✅ **Q5 — chunk-level batch failure fans out to per-record `Err` entries** for every tx_id in the chunk. Sweep continues with the next chunk.
- ✅ **v2 Finding 1 — batch params wire shape is `[[tx_hash, ...]]`** (single positional param, the array of hashes). Serde shape: `params: (Vec<String>,)`.
- ✅ **v2 Finding 2 — by-tuple params wire shape is `[u32, u32, u32, "0x...", "0x..."]`** (5-element positional array). `artifact_kind` is numeric `0` for v1, not the snake_case string.
- ✅ **v2 Finding 3 — default batch fallback fails fast on first transport/client error.** Symmetric with real chunk-level batch failure; reconcile fans out per-record from the `Err`.
- ✅ **v2 Finding 4 — no `chain_submit_refused` in Stage 13.9 read flows.** New `read_rpc_error_to_evidence_anchor_error` mapper routes JSON-RPC errors on reads to `chain_rpc`, not `chain_submit_refused`.
- ✅ **v2 Finding 5 — by-tuple `null` outcome is informational.** Event line `event=integrity_evidence_anchor_tuple_lookup_no_chain_anchor ...` carries NO `reason=` key. Exit 0 — successful read returning "no chain anchor for tuple."
- ✅ **v2 Finding 6 — `query_anchor_status_report` single-record path** added to the trait with default wrapping `query_anchor_status` (None fields). Real adapter overrides.
- ✅ **v2 Finding 7 — `Submitted` from chain is observation-only.** No reorg downgrade in 13.x; documented closed transition table.
- ✅ **v2 Finding 8 — by-tuple CLI uses read preflight only.** `run_chain_read_preflight` (chain_id check only). Activation / mainnet gates do NOT apply — read RPCs work during dormant.
- ✅ **v2 Finding 9 — schema invariance.** No changes to `AnchorStatus`, `IntegrityEvidenceAnchorTxData`, or anchor wire schema.
- ✅ **Implementation lock 1 — canonical tx-hash normalization for order verification.** `canonicalize_tx_hash` strips `0x` and lowercases. Cosmetic differences between request and chain echo do NOT cause `chain_response_malformed`. Malformed inputs (operator typos) echoed back verbatim ALSO pass the order check so the chain's per-item `error` becomes the right surface.
- ✅ **Implementation lock 2 — `failed` with `reason: null` fallback.** `FAILED_REASON_NULL_FALLBACK = "chain returned failed with no reason"` substitutes for `reason: null` so `AnchorStatus::Failed { reason: String }` stays backward-compatible. The chain's `code` surfaces independently via `AnchorStatusReport.code`.

## Chain JSON-RPC wire shapes

### `sum_getIntegrityEvidenceAnchorStatus` (single)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "sum_getIntegrityEvidenceAnchorStatus",
  "params": ["0x<tx_hash>"]
}
```

Response:
```json
{
  "result": {
    "status": "finalized",
    "included_at_height": 4807033,
    "code": null,
    "reason": null
  },
  "id": 1
}
```

`code` is parsed; stable only when `status == "failed"`.

### `sum_getIntegrityEvidenceAnchorStatusBatch` (batch — v2 Finding 1)

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "sum_getIntegrityEvidenceAnchorStatusBatch",
  "params": [["0x<tx_hash_1>", "0x<tx_hash_2>"]]
}
```

Single positional param: **an array of hashes**, max **100**. Oversize is rejected at the client side before the RPC (`ANCHOR_STATUS_BATCH_MAX = 100`).

Response (order matches request):
```json
{
  "result": [
    { "tx_hash": "0x<echo>", "result": { ...status... }, "error": null },
    { "tx_hash": "bad",      "result": null,           "error": "Invalid hash: ..." }
  ]
}
```

Per-item error containment: a malformed hash in the request becomes ONE per-item `error`; the whole batch still returns success status.

### `sum_getIntegrityEvidenceAnchorByTuple` (by-tuple — v2 Finding 2)

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "sum_getIntegrityEvidenceAnchorByTuple",
  "params": [
    1,
    0,
    1,
    "0x<artifact_hash>",
    "0x<signer_pubkey>"
  ]
}
```

Positional 5-tuple: `(anchor_schema_version, artifact_kind_tag, artifact_schema_version, artifact_hash_0x_hex, signer_pubkey_0x_hex)`. The `artifact_kind_tag` is the bincode-1 enum discriminant — `0` for v1 `SignedIntegrityEvidenceChainReport`.

Response:
```json
{ "result": { "tx_hash": "0x<canonical>", "included_at_height": 120 } }
```
or `{ "result": null }` for not-found.

## Library surface

```rust
pub const ANCHOR_STATUS_BATCH_MAX: usize = 100;
pub const FAILED_REASON_NULL_FALLBACK: &str = "chain returned failed with no reason";

pub fn canonicalize_tx_hash(s: &str) -> Option<String>;

pub struct AnchorStatusReport {
    pub status: AnchorStatus,            // unchanged closed enum
    pub included_at_height: Option<u64>,
    pub code: Option<u32>,
    pub reason: Option<String>,
}

pub struct BatchStatusItem {
    pub tx_hash: String,
    pub result: Option<AnchorStatusReport>,
    pub error: Option<String>,
}

pub struct TupleLookupResult {
    pub tx_hash: String,                  // canonical 0x-prefixed lowercase 32-byte hex
    pub included_at_height: u64,
}

pub enum TupleLookupOutcome {
    Found {
        canonical_tx_hash: String,
        included_at_height: u64,
        local_record_tx_id: String,
    },
    NotFound,
}

// EvidenceAnchorChainClient gains 3 default-impl'd methods:
//   - query_anchor_status_report(tx_id) -> AnchorStatusReport
//   - query_anchor_status_batch(tx_ids) -> Vec<BatchStatusItem>
//   - lookup_anchor_by_tuple(... 5-tuple ...) -> Option<TupleLookupResult>

pub fn lookup_anchor_by_tuple_workflow<C: EvidenceAnchorChainClient>(
    registry: &LocalEvidenceAnchorRegistry,
    client: &C,
    selector: AnchorSelector<'_>,
) -> EvidenceAnchorResult<TupleLookupOutcome>;
```

The reconcile workflow signature is unchanged; `QueryAnchorOutcome` gains two new `Option<...>` fields (`included_at_height`, `code`) backward-compatibly.

## Closed transition table (Stage 13.9 lock)

| Chain says | Local was | Local becomes | Why |
| --- | --- | --- | --- |
| `Unknown` | any | unchanged | observation-only (Stage 13.0 + 5.1 contract) |
| `Submitted` | any | unchanged | **no reorg downgrade** (v2 Finding 7 lock) |
| `Included` | `Submitted` | `Included` | normal forward transition |
| `Included` | else | unchanged | no reorg downgrade |
| `Finalized` | `Submitted` / `Included` | `Finalized` | normal forward transition |
| `Finalized` | else | unchanged | no reorg downgrade |
| `Failed` | `Submitted` / `Included` | `Failed{reason}` | normal forward transition |
| `Failed` | else | unchanged | no overwrite |

`Failed{reason}` substitutes `FAILED_REASON_NULL_FALLBACK` when the chain's `reason` is `null`; the chain's `code` is preserved independently on the `QueryAnchorOutcome` / `AnchorStatusReport`.

## Read-path reason-tag mapper (v2 Finding 4)

```text
ChainErrorCategory::Transport         → chain_rpc
ChainErrorCategory::JsonRpcError      → chain_rpc       (NOT chain_submit_refused)
ChainErrorCategory::Malformed         → chain_response_malformed
ChainErrorCategory::AdapterNotActivated → chain_rpc     (defense-in-depth on read paths)
ChainErrorCategory::AdapterSameKeyFail  → chain_rpc
ChainErrorCategory::Unknown             → chain_rpc
```

The Stage 13.2 submit-path mapper (`chain_client_error_to_evidence_anchor_error`) stays as-is for the submit flow. Stage 13.9 read paths (`run_lookup_by_tuple`, future migrations) exclusively use `read_rpc_error_to_evidence_anchor_error`.

## By-tuple `null` outcome (v2 Finding 5)

When the chain returns `result: null` on `sum_getIntegrityEvidenceAnchorByTuple`:

- Library: `TupleLookupOutcome::NotFound`.
- CLI: informational event line, **no `reason=` key**, **exit 0**:

```text
event=integrity_evidence_anchor_tuple_lookup_no_chain_anchor \
  anchor_registry_dir=<DIR> rpc_url=<URL>
```

This distinguishes "local anchor found, chain tuple lookup returned null" from `anchor_not_found` (which means local selector miss).

## Read preflight scope (v2 Finding 8)

The Stage 13.9 by-tuple CLI uses **`run_chain_read_preflight(client, expected_chain_id)`** only. Chain-id check ONLY. NO mainnet / activation gates — read RPCs work during dormant per the chain contract. The submit-path `run_chain_submit_preflight` is reserved for the submit flow.

## Reason-tag taxonomy delta — ZERO new tags

| Outcome | Tag (reused) |
| --- | --- |
| Per-item batch error | `chain_rpc` |
| Whole-chunk transport / fail-fast | `chain_rpc` |
| Whole-chunk malformed (length / order / shape) | `chain_response_malformed` |
| JSON-RPC error on a read RPC | `chain_rpc` (**not** `chain_submit_refused`) |
| By-tuple `result: null` | **no `reason=` key** (informational event) |
| Local FS / registry IO | `io` |

`crates/omni-zkml/src/error.rs` is unchanged. The exhaustive `every_variant_has_a_stable_tag` mapper test is unchanged.

## Operator-runbook recipes

```sh
# Periodic chain catch-up — auto-batched against the real chain.
omni-node operator evidence-anchor reconcile-integrity-evidence-anchor \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1

# By-tuple cross-check (read-only). Operator picks a local record;
# the command extracts the 5-tuple from that record's digest and
# asks the chain for the canonical anchor.
omni-node operator evidence-anchor lookup-integrity-evidence-anchor-by-tuple \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1 \
  --artifact-hash-hex   <64-lower-hex>

# Same, by tx_id (when the operator has the receipt but not the hash).
omni-node operator evidence-anchor lookup-integrity-evidence-anchor-by-tuple \
  --anchor-registry-dir /var/omni-anchors \
  --rpc-url             https://rpc.sumchain.io \
  --expect-chain-id     1 \
  --tx-id               <chain-tx_id>
```

**Recommended preflight before any chain reconcile**: run Stage 13.8's `report-integrity-evidence-anchor-consistency`. Local-side problems (malformed records, tx_index drift, archive-hot collisions) should be resolved before talking to the chain.

## Failure code 60–63 operator guide

The chain contract documents these stable codes on `status == "failed"`:

| `code` | Meaning | Operator response |
| --- | --- | --- |
| `60` | not activated | Submit RPC was rejected because chain hadn't activated anchor support at the time. Re-submit later if needed. |
| `61` | duplicate 5-tuple | First-wins. The operator's record may carry a non-canonical `tx_id`. Use `lookup-integrity-evidence-anchor-by-tuple` to find the canonical one. |
| `62` | invalid submitter signature | Local record was tampered or submit was buggy. Investigate Stage 13.0 verifier output. |
| `63` | `tx.from != address(signer_pubkey)` | Submit-side configuration error — investigate the submit recipe. |
| other | parse `code` numerically; treat `reason` as opaque | — |

The `reason` field is **opaque** — operators read it as documentation, NOT as a parsed token. Log scrapers should match on `code` for closed-set routing.

## Long-term identity check via by-tuple

The chain contract notes that **`tx_hash` status may become `unknown`** if pruning is enabled in the future. **By-tuple lookup is durable and never pruned.** If a local record's `tx_id` falls out of `getStatus` (returns `unknown`), the operator can run by-tuple to confirm the chain still has the anchor under the locked 5-tuple identity. This is the recommended long-term identity check.

## Test inventory

In `crates/omni-sumchain/tests/anchor_stage_13_9_batch_and_tuple.rs` (12 tests, hermetic):
- Wire-shape pins (3): batch single-positional-array, by-tuple 5-positional-array, artifact_kind v1 = 0.
- Batch happy / order / containment / malformed (5).
- By-tuple null + happy + canonical (2).
- Canonicalizer behavior + reason-tag classification (2).

In `crates/omni-zkml/tests/evidence_anchor_stage_13_9.rs` (15 tests, hermetic):
- Default trait impl fail-fast + report wrapper + by-tuple "not supported" (3).
- Reconcile batch path + chunking at 100 + Submitted observation-only + Unknown observation-only + Failed code (5).
- Real-batch per-item containment vs. fallback fail-fast (1).
- By-tuple workflow Found / NotFound / no-mutation / selector miss (4).
- Implementation locks: `FAILED_REASON_NULL_FALLBACK` stable + back-compat compile-time pin (2).

In `crates/omni-node/src/evidence_anchor_cli.rs::stage_13_9_cli_tests` (14 tests default build / 16 with `--features submit`):
- Selector mutex (no, both, hash-only, tx_id-only, message stability) (5).
- Read-path mapper exclusion of `chain_submit_refused` for JSON-RPC errors (1).
- Read-path mapper routing of transport / malformed (2).
- Raw tuple flags compile-time pin (1).
- REJECT-fix v2 (Q5) — `_no_chain_anchor` informational event has no `reason=` key, uses the informational event name (NOT `_failed`), and CLI returns `Ok(())` for exit 0 (3).
- REJECT-fix v2 (Q8) — `lookup_by_tuple_preflight` accepts mainnet+dormant and pre-activation chain states under default build (2); under `--features submit`, the same fake-client states are refused by `run_chain_submit_preflight` with `MainnetPolicyUnresolved` / `NotActivated`, proving the two preflights are not aliased (2).

Plus `crates/omni-zkml/tests/evidence_anchor_reconcile_integration.rs` — existing 5 tests updated to reflect the Stage 13.9 chunk-level fail-fast semantic; all pass.

**Exhaustive `every_variant_has_a_stable_tag` mapper test unchanged** — Stage 13.9 introduces ZERO new variants.

## Forward outlook — what comes after Stage 13.9

The Stage 13.x evidence-anchor track closes with Stage 13.9. Future evolution typically falls into one of three families:

1. **Operator UX over the existing surface** — e.g. opt-in `--apply-canonical-tx-id` repair flag on the by-tuple CLI, periodic Stage 13.8 + Stage 13.9 schedules, dashboard surfacing.
2. **Reorg-aware downgrade semantics** — would require a model for how local records react to chain-side reversions; out of scope for 13.x.
3. **Multi-chain / multi-instance** — multiple SUM Chain endpoints, replicated registries, etc.

None are in scope for Stage 13.9.
