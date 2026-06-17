# Stage 13.4 — Anchor-Registry Cleanup with Quarantine

**Status:** local-only operator hardening. Stage 13.0 wire / schema / domain / canonical bytes / signing model **unchanged**. Stage 13.2 adapter **unchanged**. Stage 13.3 detection surface **unchanged** — Stage 13.4 consumes `check_evidence_anchor_registry_health` + `list_stale_submitted_or_included` verbatim.

**Co-references:**
- [`docs/stage13-evidence-anchor-spec.md`](stage13-evidence-anchor-spec.md) — frozen Stage 13.0 wire spec.
- [`docs/stage13.3-anchor-operations.md`](stage13.3-anchor-operations.md) — summary / watch / detection helpers.
- [`docs/operator-runbook.md`](operator-runbook.md) §Stage 13.4 — operator workflow.
- `crates/omni-zkml/src/evidence_anchor/cleanup.rs` — library helpers.
- `crates/omni-node/src/evidence_anchor_cli.rs` — `run_cleanup` + `check_cleanup_mutex_rules`.

## What Stage 13.4 ships

- **One new CLI subcommand** under `omni-node operator evidence-anchor`: `cleanup-integrity-evidence-anchor-registry`, with three phases (plan → apply → restore) selected by closed mutex rules.
- **Three library helpers** in `omni-zkml::evidence_anchor::cleanup`:
  - `plan_anchor_cleanup(registry, opts) -> AnchorCleanupPlan`
  - `apply_anchor_cleanup(plan, opts) -> AnchorCleanupReport`
  - `restore_anchor_cleanup_quarantine(manifest, opts) -> AnchorQuarantineRestoreReport`
- **Closed action taxonomy** (`AnchorCleanupActionKind`): three Tier A + one Tier B variant cover every Stage 13.3 finding.
- **Persisted JSON forms** with schema constants (`ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION = 1`, `ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION = 1`).
- **Seven new `EvidenceAnchorError` variants** + seven new `evidence_anchor_reason_tag` mapper arms. **Three new tag strings** (`cleanup_drift`, `cleanup_invalid_path`, `unsupported_cleanup_plan_schema_version`); four reused tag strings (`cleanup_plan_hash_mismatch`, `gate_required`, `quarantine_blake3_mismatch`, `restore_target_exists`). The two extra new tags were added during the REJECT-fix loop to refuse operator-supplied JSON that contains absolute paths, `..` traversal, or wrong per-kind path shapes, and to refuse plans whose `schema_version` is not the locked `ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION`.

## What Stage 13.4 does NOT ship

- ❌ No SUM Chain RPC interaction at all. Fully local.
- ❌ No `AnchorRecord` shape change.
- ❌ No `LocalAnchorStatus` enum change.
- ❌ No submit / retry path.
- ❌ No automatic delete of Tier B candidates — Tier B always quarantines first.
- ❌ No `--purge-stray` flag (removed from the v1 plan per REJECT review).
- ❌ No process lock; concurrent operators trip `cleanup_drift` and re-plan.

## Locked decisions (post-REJECT loop)

- ✅ Q1 — Stale records require TWO opt-ins: `--stale-threshold-secs` at plan time AND `--allow-stale-quarantine` at apply time.
- ✅ Q2 — Tag strings reused from Stage 12.17/12.18; anchor-side `EvidenceAnchorError` variants and mapper arms added so the closed surface stays exhaustive.
- ✅ Q3 — `registry_state_hash` includes `status`, excludes `updated_at`.
- ✅ Q4 — Quarantine layout `<quarantine-dir>/<plan_id>/<source_relative>`.
- ✅ Q5 — `RemoveOrphanTxIndexEntry` does NOT quarantine.
- ✅ Q6 — Apply uses `--apply` confirmation flag (not `--no-dry-run`).
- ✅ Q7 — No process lock; drift refusal is the contract.
- ✅ Finding 1 — Reused tag strings each get their own anchor-side error variant.
- ✅ Finding 2 — Closed mutex/required-with preflight rules table (see CLI section).
- ✅ Finding 3 — `--purge-stray` removed; Tier A defaults to dry-run unless `--apply`.
- ✅ Finding 4 — Restore derives quarantine root from `manifest_path.parent()`.
- ✅ REJECT-fix path validation — Apply and restore refuse any `source_relative` / `quarantine_relative` that is absolute, contains `..`, contains path separators, or violates the per-kind shape (root-level `*.tmp` for `RemoveOrphanTmpFile`, exactly `tx_index.json` for `RemoveOrphanTxIndexEntry`, `<64-lower-hex>.json` for `QuarantineMalformedRecord` / `QuarantineStaleOpenRecord`). Routed through `CleanupInvalidPath { action_kind, source_relative, reason }` → `reason=cleanup_invalid_path`.
- ✅ REJECT-fix durability ordering — Apply executes in two passes: (1) Tier B quarantine copies + accumulated manifest entries with all source removals deferred; (1.5) atomic manifest write as the durability fence; (2) source removals + `tx_index.json` rewrite. A manifest write failure leaves the original registry bytes intact.
- ✅ REJECT-fix restore idempotency — `restore_re_adds_tx_index_entry` runs in BOTH the `restored` and the `skipped_already_restored` branch, so a stale record whose file is already back is still re-registered in `tx_index.json` if the entry is missing.
- ✅ REJECT-fix schema-version refusal — Apply checks `plan.schema_version` BEFORE the plan-hash check and refuses with `reason=unsupported_cleanup_plan_schema_version` when the value is not the locked schema version. Future-schema plans cannot apply against the current implementation.

## Closed action taxonomy

| Variant | Tier | Source finding | Apply does |
| --- | --- | --- | --- |
| `RemoveOrphanTmpFile` | A | health: `.tmp` in registry root | Delete in place (under `--apply`). No quarantine. |
| `RemoveOrphanTxIndexEntry` | A | health: `tx_index` entry maps to absent record | Atomic-rewrite `tx_index.json` minus the entry (under `--apply`). No quarantine (Q5). |
| `QuarantineMalformedRecord` | B | health: `<64-hex>.json` doesn't parse | Copy to quarantine, append manifest entry, then delete source. |
| `QuarantineStaleOpenRecord` | B (gated) | stale: `Submitted`/`Included` past threshold | Copy to quarantine, append manifest entry, then delete record + remove `tx_index.json` entry. **Requires `--allow-stale-quarantine`** AND `--stale-threshold-secs` at plan time. |

Tier B writes go through `.tmp + rename`. The manifest is written once atomically after all Tier B actions complete.

## Drift / plan-hash / gate recipes

- **`registry_state_hash`** (Q3 locked): BLAKE3 of canonical JSON `{ records: sorted_by(artifact_hash_hex) [{ artifact_hash_hex, status, submitted_at_unix }], tx_index_entries: sorted_by(tx_id) [{ tx_id, artifact_hash_hex }] }`. Includes `status` so chain-driven transitions between plan and apply trip drift correctly. Excludes `updated_at` so no-op `update_status` calls don't trip drift.
- **`cleanup_plan_hash`**: BLAKE3 of canonical JSON of the plan with `cleanup_plan_hash` blanked. Pinned by `plan_hash_is_byte_stable_across_recomputation`.
- **`plan_id`**: lowercase-hex BLAKE3 of `(anchor_registry_dir || registry_state_hash || created_at_utc)`, first 16 chars. Scopes the quarantine subtree.
- **Gate check**: any `QuarantineStaleOpenRecord` in the plan requires `--allow-stale-quarantine` at apply time. Refused with `EvidenceAnchorError::CleanupGateRequired` → `reason=gate_required`.

## Quarantine layout

```text
<quarantine-dir>/
    <plan_id>/
        quarantine_manifest.json     ← always at the plan-id root
        aa…aa.json                   ← source paths mirrored verbatim
        bb…bb.json
```

Restore derives the quarantine root from `manifest_path.parent()` (Finding 4). The library API takes `quarantine_dir` and `anchor_registry_dir` as explicit args; the CLI computes them from the manifest path + `--anchor-registry-dir`.

## CLI surface

```text
omni-node operator evidence-anchor cleanup-integrity-evidence-anchor-registry
    --anchor-registry-dir <DIR>          [required in all three phases]

    # Phase selectors (mutually exclusive)
    [--apply-plan <PATH>]                (apply mode)
    [--restore-manifest <PATH>]          (restore mode)
    # neither flag → plan mode (default)

    # Plan-mode flags (refused outside plan mode)
    [--plan-out <PATH>]                  (default: stdout)
    [--stale-threshold-secs <U64>]       (no stale-cleanup without it)

    # Apply-mode flags (refused outside apply mode)
    [--quarantine-dir <DIR>]             (required iff plan has Tier B)
    [--apply]                            (without it: dry-run)
    [--allow-stale-quarantine]           (gate for QuarantineStaleOpenRecord)
```

### Closed mutex / required-with preflight rules

The `check_cleanup_mutex_rules` helper returns one of seven `CleanupMutexViolation` values; the CLI maps each to a stable operator-facing message via clap's usage-error channel (these refusals exit non-zero but do NOT emit `reason=…` lines — they're argument-parse-layer concerns, not the closed refusal taxonomy):

| Condition | Violation | Message |
| --- | --- | --- |
| `--apply-plan` AND `--restore-manifest` | `ApplyAndRestoreConflict` | `--apply-plan and --restore-manifest are mutually exclusive` |
| `--plan-out` AND (apply mode OR restore mode) | `PlanOutOutsidePlanMode` | `--plan-out is plan-mode only …` |
| `--stale-threshold-secs` AND (apply mode OR restore mode) | `StaleThresholdOutsidePlanMode` | `--stale-threshold-secs is plan-mode only …` |
| `--apply` AND NOT `--apply-plan` | `ApplyFlagWithoutApplyPlan` | `--apply requires --apply-plan` |
| `--quarantine-dir` AND NOT `--apply-plan` | `QuarantineDirOutsideApplyMode` | `--quarantine-dir is apply-mode only (requires --apply-plan)` |
| `--allow-stale-quarantine` AND NOT `--apply-plan` | `AllowStaleQuarantineOutsideApplyMode` | `--allow-stale-quarantine is apply-mode only (requires --apply-plan)` |
| Plan has Tier B AND NOT `--quarantine-dir` | `QuarantineDirRequiredForTierB` | `--quarantine-dir required: plan contains Tier B actions` |

Pinned by `mutex_refuses_*` tests; messages pinned by `violation_messages_are_stable_strings`.

## Reason-tag taxonomy

Stage 13.4 adds **seven new `EvidenceAnchorError` variants** and **three new tag strings**. The other four tag strings are reused from the Stage 12.17/12.18 cleanup taxonomy so operators reading logs across stages don't have to learn two names for the same condition.

| Variant | Tag string | Source |
| --- | --- | --- |
| `CleanupDrift { computed, expected }` | `cleanup_drift` | **NEW** (Stage 13.4) |
| `CleanupInvalidPath { action_kind, source_relative, reason }` | `cleanup_invalid_path` | **NEW** (Stage 13.4 REJECT-fix) |
| `CleanupPlanSchemaUnsupported { got, expected }` | `unsupported_cleanup_plan_schema_version` | **NEW** (Stage 13.4 REJECT-fix) |
| `CleanupPlanHashMismatch { computed, expected }` | `cleanup_plan_hash_mismatch` | Reused (Stage 12.17) |
| `CleanupGateRequired { action_kind, gate_flag }` | `gate_required` | Reused (Stage 12.17) |
| `QuarantineBlake3Mismatch { source_relative, computed, expected }` | `quarantine_blake3_mismatch` | Reused (Stage 12.18) |
| `RestoreTargetExists { target_path }` | `restore_target_exists` | Reused (Stage 12.18) |

The exhaustive `every_variant_has_a_stable_tag` test extends to cover all seven — adding a variant without a mapper arm is a compile error.

## Informational event taxonomy

| Event | When | Fields |
| --- | --- | --- |
| `event=integrity_evidence_anchor_cleanup_plan_started` | Plan startup | `anchor_registry_dir=` |
| `event=integrity_evidence_anchor_cleanup_plan_written` | Plan written to `--plan-out` | `path=` |
| `event=integrity_evidence_anchor_cleanup_plan_summary` | Plan complete | `plan_id=`, `actions=`, `tier_a=`, `tier_b=`, `gated=` |
| `event=integrity_evidence_anchor_cleanup_apply_started` | Apply startup | `anchor_registry_dir=`, `plan=`, `mode=apply|dry_run` |
| `event=integrity_evidence_anchor_cleanup_apply_action_outcome` | Per action | `action_index=`, `kind=`, `source_relative=`, `status=applied|would_apply|skipped_missing` |
| `event=integrity_evidence_anchor_cleanup_apply_summary` | Apply complete | `plan_id=`, `mode=`, `actions_applied=`, `actions_dry_run=`, `actions_skipped=`, `quarantine_dir=`, `quarantine_manifest_relative=` |
| `event=integrity_evidence_anchor_cleanup_restore_started` | Restore startup | `anchor_registry_dir=`, `manifest=` |
| `event=integrity_evidence_anchor_cleanup_restore_outcome` | Per entry | `source_relative=`, `status=restored|would_restore|skipped_already_restored` |
| `event=integrity_evidence_anchor_cleanup_restore_summary` | Restore complete | `plan_id=`, `mode=`, `restored=`, `skipped=` |
| `event=integrity_evidence_anchor_cleanup_failed` | Any phase failure | `reason=<closed-set tag>`, `detail=` |

No new `cause=` keys. Refusals route through `reason=`.

## Apply preflight ordering

Apply runs preflight checks in this fixed order, refusing on the first failure with the indicated `reason=` tag:

1. **Schema version** — `plan.schema_version == ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION`. Refuses with `reason=unsupported_cleanup_plan_schema_version`. Runs before hash/drift so a future-schema plan whose hash field was re-computed cannot smuggle itself through.
2. **Plan hash** — `compute_cleanup_plan_hash(plan) == plan.cleanup_plan_hash`. Refuses with `reason=cleanup_plan_hash_mismatch`.
3. **Drift** — `compute_registry_state_hash(registry) == plan.registry_state_hash`. Refuses with `reason=cleanup_drift`.
4. **Gate** — any Tier B gated action without the matching `--allow-*` flag refuses with `reason=gate_required`.
5. **Per-action path validation** — every `action.source_relative` is validated via `validate_source_relative_for_kind`:
   - Reject empty, absolute, separator-containing, or `..`-traversing paths.
   - `RemoveOrphanTmpFile` requires a root-level `*.tmp` filename.
   - `RemoveOrphanTxIndexEntry` requires exactly `tx_index.json`.
   - `QuarantineMalformedRecord` / `QuarantineStaleOpenRecord` require a `<64-lower-hex>.json` filename.
   - All failures refuse with `reason=cleanup_invalid_path`.

## Apply durability ordering

Apply executes Tier B / Tier A actions in two passes around a manifest-write durability fence so a crash between FS operations leaves the original registry bytes intact.

- **Pass 1** — For each Tier B action: read source bytes; compute BLAKE3; `mkdir -p` quarantine dir; atomic write `<quarantine-dir>/<plan_id>/<source_relative>.tmp` then rename; accumulate a manifest entry. Source removal is deferred. For Tier A actions, the `RemoveOrphanTxIndexEntry` entries are accumulated for later atomic rewrite.
- **Pass 1.5 (durability fence)** — Atomic-write `quarantine_manifest.json.tmp` then rename. Until this succeeds, no source file has been removed and no `tx_index.json` rewrite has happened. A panic/disk-full here leaves the original registry intact.
- **Pass 2** — Only after the manifest is durable: for each deferred Tier B and Tier A source removal, `fs::remove_file(<source>)`; then atomic-rewrite `tx_index.json.tmp + rename` with the accumulated entries removed.

## Exact FS operations per action

### `RemoveOrphanTmpFile`
1. (under `--apply`) `fs::remove_file(<orphan.tmp>)` in Pass 2. Idempotent missing → `skipped_missing`.

### `RemoveOrphanTxIndexEntry`
1. (under `--apply`) Queue the `tx_id` in Pass 1 for end-of-apply rewrite.
2. After Pass 2 source removals, atomic-rewrite `tx_index.json.tmp + rename` with the queued entries removed.

### `QuarantineMalformedRecord`
1. (under `--apply`, Pass 1) Read source bytes; compute BLAKE3.
2. `mkdir -p <quarantine-dir>/<plan_id>/`.
3. Write `<quarantine-dir>/<plan_id>/<hash>.json.tmp` then rename.
4. Accumulate a manifest entry.
5. Pass 1.5: atomic manifest write.
6. Pass 2: `fs::remove_file(<source>)`.

### `QuarantineStaleOpenRecord`
1. (under `--apply` AND `--allow-stale-quarantine`, Pass 1) Same steps 1-4 as malformed-record quarantine, with the record's `tx_id` captured in the manifest entry.
2. Queue the `tx_id` for end-of-apply `tx_index.json` rewrite (entry removed).
3. Pass 1.5: atomic manifest write.
4. Pass 2: source removal + `tx_index.json` rewrite.

### Restore (per manifest entry)
1. Validate the entry's `source_relative` AND `quarantine_relative` via the same `validate_source_relative_for_kind` rules. Refuse with `reason=cleanup_invalid_path` on failure. Q4 layout lock: `quarantine_relative == source_relative` is enforced.
2. Read quarantined file bytes; recompute BLAKE3. Refuse with `quarantine_blake3_mismatch` on drift.
3. If target exists: refuse with `restore_target_exists` UNLESS the target's BLAKE3 equals the manifest's recorded hash (idempotent already-restored → `skipped_already_restored`).
4. `mkdir -p` target parent; atomic copy via `.tmp + rename`.
5. For stale-record entries: re-add the `tx_id → artifact_hash_hex` mapping to `tx_index.json` via atomic rewrite. **This step runs in BOTH the `restored` and the `skipped_already_restored` branch** so a stale record whose file is already back but whose index entry is missing is still re-registered.

## Test inventory

In `crates/omni-zkml/tests/evidence_anchor_cleanup.rs` (33 library integration tests):

| # | Test | Pins |
| - | --- | --- |
| 1 | `plan_handles_empty_registry` | Empty plan, valid hash |
| 2 | `plan_detects_orphan_tmp_files` | `RemoveOrphanTmpFile` |
| 3 | `plan_detects_orphan_tx_index_entries` | `RemoveOrphanTxIndexEntry` |
| 4 | `plan_detects_malformed_records` | `QuarantineMalformedRecord` |
| 5 | `plan_detects_stale_open_records_when_threshold_given` | `QuarantineStaleOpenRecord` |
| 6 | `plan_skips_stale_records_when_threshold_omitted` | Q1 invariant |
| 7 | `plan_hash_is_byte_stable_across_recomputation` | `cleanup_plan_hash` byte-stable |
| 8 | `plan_registry_state_hash_changes_when_status_changes` | Q3: status drifts |
| 9 | `plan_registry_state_hash_unchanged_when_only_updated_at_changes` | Q3: `updated_at` excluded |
| 10 | `apply_dry_run_emits_would_apply_for_every_action` | Dry-run no-mutation |
| 11 | `apply_real_run_quarantines_malformed_then_deletes_source` | Tier B happy path |
| 12 | `apply_real_run_removes_orphan_tx_index_entry` | Tier A happy path |
| 13 | `apply_refuses_on_cleanup_drift` | Drift refusal |
| 14 | `apply_refuses_on_cleanup_plan_hash_mismatch` | Plan-hash refusal |
| 15 | `apply_refuses_gated_action_without_allow_stale_quarantine` | Gate refusal |
| 16 | `apply_idempotent_skipped_missing_for_already_removed_file` | Idempotent |
| 17 | `apply_does_not_quarantine_orphan_tx_index_entry` | Q5 invariant |
| 18 | `restore_round_trips_quarantined_malformed_record` | Restore happy path |
| 19 | `restore_refuses_on_quarantine_blake3_mismatch` | Tampered quarantine |
| 20 | `restore_refuses_when_target_exists` | No-clobber |
| 21 | `restore_idempotent_skipped_already_restored_for_matching_target` | Idempotent restore |
| 22 | `restore_re_adds_tx_index_entry_for_stale_records` | Symmetric to apply removal |
| 23 | `apply_refuses_action_with_absolute_path` | REJECT Finding 1 (path traversal) |
| 24 | `apply_refuses_action_with_parent_traversal` | REJECT Finding 1 (`..` traversal) |
| 25 | `apply_refuses_action_with_wrong_per_kind_shape` | REJECT Finding 1 (per-kind shape) |
| 26 | `restore_refuses_manifest_entry_with_absolute_path` | REJECT Finding 1 on restore side |
| 27 | `restore_refuses_manifest_entry_with_parent_traversal` | REJECT Finding 1 on restore side |
| 28 | `restore_refuses_manifest_entry_where_quarantine_relative_differs_from_source` | Q4 layout lock |
| 29 | `apply_refuses_unsupported_plan_schema_version` | REJECT Finding 4 (schema-version refusal) |
| 30 | `apply_schema_version_check_fires_before_plan_hash_check` | REJECT Finding 4 (ordering) |
| 31 | `apply_does_not_delete_tier_b_source_before_manifest_lands` | REJECT Finding 2 (durability) |
| 32 | `apply_with_manifest_write_failure_leaves_source_intact` (cfg-unix) | REJECT Finding 2 (manifest-write failure) |
| 33 | `restore_idempotent_re_adds_tx_index_entry_when_file_already_back` | REJECT Finding 3 (stale idempotency) |

In `crates/omni-node/src/evidence_anchor_cli.rs` `stage_13_4_cli_tests` (18 CLI tests):

- 8 mutex-rule refusal pins.
- 3 mutex-rule pass pins (default plan, valid apply combo, restore mode).
- 1 violation-message stability pin.
- 6 reason-tag mapper routing pins (`cleanup_drift`, `gate_required`, `quarantine_blake3_mismatch`, `restore_target_exists`, `cleanup_invalid_path`, `unsupported_cleanup_plan_schema_version`).
- 1 Finding-4 derivation pin (restore quarantine root from `manifest_path.parent()`).

All hermetic. No network.

## Out of scope (Stage 13.5+ candidates)

- Active chain-side action (no submit / no retry).
- Cleanup of `Finalized` / `Failed` terminal records (archival concern, operator-driven).
- Multi-host quarantine sync.
- Cleanup of standalone-JSON anchor files outside the registry.
- Block-based "reorged out" detection.
- Process locks (drift refusal is the contract).
