# Stage 13.3 — Operator-Facing Hardening for Anchor Lifecycle

**Status:** operator-side hardening only. Stage 13.0 wire / schema / domain / reason tags / same-key submitter model **unchanged**. Stage 13.2 adapter **unchanged** — Stage 13.3 consumes the existing `reconcile_evidence_anchors_workflow` verbatim each watch tick.

**Co-references:**
- [`docs/stage13-evidence-anchor-spec.md`](stage13-evidence-anchor-spec.md) — frozen Stage 13.0 wire spec.
- [`docs/stage13.2-chain-adapter.md`](stage13.2-chain-adapter.md) — Stage 13.2 real SUM Chain adapter (reused by `watch-integrity-evidence-anchors`).
- [`docs/operator-runbook.md`](operator-runbook.md) §Stage 13.3 — operator workflow, flag table, refusal taxonomy.
- `crates/omni-zkml/src/evidence_anchor/operations.rs` — library helpers.
- `crates/omni-node/src/evidence_anchor_cli.rs` — `run_summary` / `run_watch` / `emit_*` helpers.

## What Stage 13.3 ships

- **Two new CLI subcommands** under `omni-node operator evidence-anchor`:
  - `summary-integrity-evidence-anchors` — fully local registry snapshot.
  - `watch-integrity-evidence-anchors` — chain-read-only periodic monitor.
- **Three new library helpers** in `omni-zkml::evidence_anchor`:
  - `list_evidence_anchors_by_status(registry) -> EvidenceAnchorRegistrySummary`
  - `check_evidence_anchor_registry_health(registry) -> EvidenceAnchorRegistryHealth`
  - `list_stale_submitted_or_included(registry, now_utc, threshold_secs) -> Vec<StaleAnchorInfo>`
- **Three new serde-friendly structs**: `EvidenceAnchorRegistrySummary`, `EvidenceAnchorRegistryHealth`, `StaleAnchorInfo`.
- **Informational event taxonomy** for summary, stale rows, health, watch lifecycle. **Zero new closed reason tags** — refusals continue to route through Stage 13.0 / 13.2 tags.

## What Stage 13.3 does NOT ship

- ❌ No wire / schema / domain / canonical-bytes / signing changes.
- ❌ No new chain RPCs. The watch loop's chain interaction is exactly:
  - **Once at startup (CLI preflight):** `chain_getChainParams` for the `chain_id` sanity check (reused from Stage 13.2's `run_chain_read_preflight`).
  - **Per tick (inside `reconcile_evidence_anchors_workflow`):** one `sum_getIntegrityEvidenceAnchorStatus(tx_hash)` per `Submitted` / `Included` record. Both RPCs already exist in Stage 13.2; Stage 13.3 calls them at no higher frequency than the operator-invoked Stage 13.2 `reconcile-integrity-evidence-anchor` command already does.
- ❌ No `AnchorRecord` shape change. Time-based staleness uses the existing `submitted_at` field. Block-based staleness deferred.
- ❌ No `--rpc-url` flag on `summary` — summary stays fully local.
- ❌ No submit / retry path in watch. Watch is monitor-only.
- ❌ No new closed-set reason tags. Watch-stop events use the new `cause=` key, deliberately separated from the refusal `reason=` taxonomy.
- ❌ No automatic cleanup of orphan files. Health diagnostic is read-only; operators decide what to delete.

## Locked decisions (post-REJECT)

- ✅ Q1 — Time-based staleness only. No `submitted_at_block` field in Stage 13.3.
- ✅ Q2 — Single `summary-integrity-evidence-anchors` command with `--include-health` flag; no separate `health-*` subcommand.
- ✅ Q3 — Watch reuses `reconcile_evidence_anchors_workflow` each tick (one chain query per `Submitted`/`Included` record).
- ✅ Q4 — Watch stops use `cause=ctrl_c` / `cause=max_ticks`. The `reason=` key stays reserved for refusal/failure taxonomy.
- ✅ Q5 — Watch stays read-only on chain; never invokes submit / never retries.
- ✅ Finding 1 — `AnchorRecord` shape unchanged.
- ✅ Finding 2 — Summary is fully local; no `--rpc-url` flag.
- ✅ Finding 3 — `cause=` key locked across CLI, tests, and docs.

## Time-based staleness rationale

Stage 5.2 attestations use **block-based** staleness because attestations have a `submitted_at_block: Option<u64>` field captured at submit time. The chain-head block height is a more accurate "elapsed time" measure than wall-clock seconds during chain stalls / restarts.

Stage 13.3 anchors use **time-based** staleness because:

1. Stage 13.0's `AnchorRecord` doesn't carry a block-height field — adding one would be a Stage 13.3 local-schema bump.
2. Anchors are anchored to chain confirmation events rather than ongoing on-chain state; the block height at submit isn't load-bearing for the anchor's correctness.
3. Operators primarily care about "this anchor has been Submitted for N hours without progress" — a wall-clock signal.

Block-based staleness is deferred. If a future Stage 13.x needs it (e.g. for an SLA framework), the path is:

1. Add `submitted_at_block: Option<u64>` to `AnchorRecord` with `#[serde(default)]`.
2. Capture chain head at submit time in Stage 13.2's adapter (or via a separate `submit-integrity-evidence-anchor-with-block` flag).
3. Add an `AnchorStalenessPolicy { threshold_blocks: u64 }` analog of Stage 5.2.
4. Wire alongside the time-based detection — operator picks one or both.

## Event taxonomy

All Stage 13.3 events are **informational** (no `reason=` field) unless they route through a Stage 13.0 / 13.2 refusal. The new event keys:

| Event | Emitted by | Fields |
| --- | --- | --- |
| `event=integrity_evidence_anchor_summary` | `summary`, `watch` (per tick) | `total=`, `submitted=`, `included=`, `finalized=`, `failed=` |
| `event=integrity_evidence_anchor_stale` | `summary --stale-threshold-secs`, `watch --stale-threshold-secs` (per stale record) | `artifact_hash_hex=`, `tx_id=`, `status=`, `age_secs=`, `threshold_secs=` |
| `event=integrity_evidence_anchor_stale_summary` | After all stale rows for a given invocation/tick | `count=`, `threshold_secs=` |
| `event=integrity_evidence_anchor_health` | `summary --include-health` | `records=`, `malformed_records=`, `orphan_tx_index_entries=`, `orphan_tmp_files=` |
| `event=integrity_evidence_anchor_watch_started` | `watch` (once, at start) | `anchor_registry_dir=`, `rpc_url=`, `expect_chain_id=`, `poll_interval_secs=` |
| `event=integrity_evidence_anchor_watch_tick` | `watch` (per tick) | `tick=N` |
| `event=integrity_evidence_anchor_watch_stopped` | `watch` (once, at exit) | **`cause=ctrl_c`** or **`cause=max_ticks`**, `ticks=N` |
| `event=integrity_evidence_anchor_watch_tick_failed` | `watch` (per-tick orchestration failure — sweep abort, summary refusal) | `reason=<closed-set tag>`, `detail=` |

## `cause=` vs `reason=` invariant

The `reason=` key is reserved for the **closed Stage 13.0 / 13.2 refusal taxonomy**. The `cause=` key is reserved for **informational watch stops**. This split is enforced by:

- The `format_watch_stop_event(cause, tick)` helper in `evidence_anchor_cli.rs` — the only function that emits the `watch_stopped` event.
- Pinned by `watch_stop_event_uses_cause_key_not_reason_key` test which asserts the line contains `cause=` and explicitly NOT `reason=`.
- Operator log-scraping convention:
  - `grep 'reason='` returns ONLY refusals.
  - `grep 'cause='` returns ONLY informational stops.

Adding a future variant (e.g. `cause=watchdog`) is a CLI-only change; the closed-set refusal taxonomy stays untouched.

## Library helper API

```rust
// Stage 13.3 — counts-by-status snapshot.
pub struct EvidenceAnchorRegistrySummary {
    pub total: u64,
    pub submitted: u64,
    pub included: u64,
    pub finalized: u64,
    pub failed: u64,
}
pub fn list_evidence_anchors_by_status(
    registry: &LocalEvidenceAnchorRegistry,
) -> EvidenceAnchorResult<EvidenceAnchorRegistrySummary>;

// Stage 13.3 — registry-health diagnostic. Read-only; counts only.
pub struct EvidenceAnchorRegistryHealth {
    pub records: u64,
    pub malformed_records: u64,
    pub orphan_tx_index_entries: u64,
    pub orphan_tmp_files: u64,
}
pub fn check_evidence_anchor_registry_health(
    registry: &LocalEvidenceAnchorRegistry,
) -> EvidenceAnchorResult<EvidenceAnchorRegistryHealth>;

// Stage 13.3 — time-based stale detection.
pub struct StaleAnchorInfo {
    pub artifact_hash_hex: String,
    pub tx_id: String,
    pub status: LocalAnchorStatus,
    pub submitted_at: DateTime<Utc>,
    pub age_secs: u64,
}
pub fn list_stale_submitted_or_included(
    registry: &LocalEvidenceAnchorRegistry,
    now_utc: DateTime<Utc>,
    threshold_secs: u64,
) -> EvidenceAnchorResult<Vec<StaleAnchorInfo>>;
```

All three helpers are **pure reads** — they do not mutate the registry. Pinned by `summary_helper_does_not_mutate_registry` integration test (file-content hash comparison before / after).

## CLI surface

```text
omni-node operator evidence-anchor summary-integrity-evidence-anchors
    --anchor-registry-dir <DIR>             [required]
    [--stale-threshold-secs <U64>]          (time-based stale detection)
    [--include-health]                      (read-only health diagnostic)

omni-node operator evidence-anchor watch-integrity-evidence-anchors
    --anchor-registry-dir <DIR>             [required]
    --rpc-url <URL>                         [required]
    --expect-chain-id <U64>                 [required]
    [--poll-interval-secs <U64>]            (default 30)
    [--max-ticks <U64>]                     (test-injection / scripted-stop)
    [--stale-threshold-secs <U64>]          (per-tick stale rows)
```

`watch` runs CLI preflight (chain-id check) once at startup. The watch loop itself does NOT re-check chain-id mid-run; operators reconfiguring an RPC endpoint should restart the watch.

`watch` does NOT enforce anchor activation. A previously-active chain that has since gone dormant should still permit status reads — refusing to watch would block exactly the operator who needs visibility.

## Test inventory

In `crates/omni-zkml/tests/evidence_anchor_summary_and_health.rs` (new, library layer, 11 tests):

| Test | Pins |
| --- | --- |
| `summary_counts_records_by_status` | Counts per status across all `LocalAnchorStatus` variants |
| `summary_handles_empty_registry` | Zero-everywhere on empty dir |
| `stale_detection_finds_records_past_threshold` | Backdated record reports stale; fresh does not |
| `stale_detection_skips_finalized_and_failed_records` | Terminal states never stale |
| `stale_detection_handles_threshold_zero_gracefully` | All open records report at threshold=0 |
| `stale_detection_skips_future_dated_submitted_at` | Clock-skew underflow guard |
| `health_reports_orphan_tx_index_entries` | tx_index entry with no record file |
| `health_reports_orphan_tmp_files` | `.tmp` files counted, not deleted |
| `health_reports_malformed_records` | 64-hex `.json` that doesn't parse |
| `summary_helper_does_not_mutate_registry` | File-content hash pre/post invariant |
| `health_empty_registry_returns_zero_counts` | Sanity baseline |

In `crates/omni-node/src/evidence_anchor_cli.rs` `stage_13_3_cli_tests` (in-bin, 7 tests):

| Test | Pins |
| --- | --- |
| `watch_stop_event_uses_cause_key_not_reason_key` | Locked `cause=` invariant — refusal taxonomy isolation |
| `watch_stop_event_emits_ticks_count` | `ticks=N` field shape |
| `summary_helper_returns_typed_counts` | `emit_summary_event` returns correct counts |
| `stale_helper_returns_rows_above_threshold_only` | Time-based detection through the CLI emitter |
| `stale_helper_emits_empty_summary_when_no_rows` | Empty case still emits `stale_summary count=0` |
| `health_helper_returns_typed_counts` | `emit_health_event` returns correct counts |
| `health_helper_does_not_delete_orphan_tmp_files` | Read-only invariant pinned at CLI emitter level |

**All hermetic. No live-chain tests.**

## Deferred (not Stage 13.3)

- Block-based staleness (would add `submitted_at_block` to `AnchorRecord` + an `AnchorStalenessPolicy` analog of Stage 5.2).
- Active cleanup of orphan files (operator-driven for now).
- Watch-driven submit / retry (locked Q5: never).
- Stale-anchor quarantine / archive (operator decides).
- Aggregation forms (anchor bundles, anchor chains).
- Separate-submitter / relay flows.

> **Post-Stage-13.4 note:** Stage 13.4 turns Stage 13.3's health/stale findings into a planned cleanup (plan → dry-run → apply → restore). Stage 13.3 detection helpers (`check_evidence_anchor_registry_health`, `list_stale_submitted_or_included`) are **unchanged**; Stage 13.4 calls them verbatim. See [`docs/stage13.4-anchor-cleanup.md`](stage13.4-anchor-cleanup.md).
