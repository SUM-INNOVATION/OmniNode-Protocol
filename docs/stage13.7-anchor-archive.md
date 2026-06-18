# Stage 13.7 — local terminal-anchor archive / restore

## Stage scope

Stage 13.7 moves valid **terminal** anchor records (`Finalized` / `Failed`) out of the hot anchor registry into a byte-preserving archive subtree, plus a symmetric restore path. Fully local — no SUM Chain RPCs, no `omni-sumchain` types, no private chain repo deps. Stage 13.0 wire / domain / canonical-bytes / signing is **read-only** — not a byte is re-signed.

This is the last purely-local lifecycle stage before Stage 13.9's comprehensive chain read / reconcile work. Stage 13.8 (forward outlook) is a local registry consistency report.

## What Stage 13.7 ships

- **Library helpers** in `omni-zkml::evidence_anchor::archive`:
  - `plan_anchor_archive(registry, opts) -> AnchorArchivePlan`
  - `apply_anchor_archive(plan, opts) -> AnchorArchiveReport`
  - `restore_anchor_archive(manifest, opts) -> AnchorArchiveRestoreReport`
- **New CLI subcommand** on `omni-node operator evidence-anchor`:
  - `archive-integrity-evidence-anchors` (plan / apply / restore modes via mutex)
- **Two new schema constants**: `ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION = 1`, `ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION = 1`.
- **6 new `EvidenceAnchorError` variants** + 6 new closed reason-tag strings (see §7).
- **Reused tag strings** (no surface growth): `anchor_not_found` (selector miss), `malformed_json`, `io`, `submitter_signature_invalid`, `unsupported_anchor_schema_version`.

## What Stage 13.7 does NOT ship

- ❌ No SUM Chain RPC interaction.
- ❌ No `AnchorRecord` shape change. Records are archived and restored byte-for-byte.
- ❌ No wire / domain / canonical-bytes / signing changes.
- ❌ No submit / retry / re-signing path.
- ❌ Not eligible: `Submitted` / `Included`. The clap layer refuses non-terminal `--status` values.
- ❌ No `--force` flag for restore over-clobber. Operators rename the conflicting record before retrying.
- ❌ No archive-of-archive. No automatic / scheduled archival. No multi-archive merge.

## Why local-only / terminal-only

- **Why local-only**: archive shrinks the hot registry. The chain doesn't care; the chain is the source of truth for whether an anchor is finalized. Moving terminal records to an archive subtree doesn't touch the chain.
- **Why terminal-only**: archiving a `Submitted` / `Included` record would archive a record whose lifecycle is in flight. Stage 13.2 reconcile would then silently no-op when trying to apply a chain-returned status update to a record no longer in the hot registry. That's a correctness footgun.

## Locked decisions (Q&A + REJECT findings)

- ✅ Q1 — `Failed` opt-in via repeatable `--status FAILED`. No separate `--include-failed` boolean. Default `[Finalized]`.
- ✅ Q2 — **6 new tag strings** with semantic separation from Stage 13.4 cleanup. The Stage 13.6 single-variant + `field=` discriminator pattern is reused for `archive_target_exists`.
- ✅ Q3 — `registry_state_hash` includes `artifact_hash_hex` / `status` / `tx_id` / `updated_at_unix` per record. **Deliberately includes `updated_at_unix`** (divergence from Stage 13.4) because the `--before` selector reads it.
- ✅ Q4 — Non-terminal `--status` is a clap-level refusal. Selector misses (after the terminal filter OR `--before` filter) refuse with `anchor_not_found` and operator-readable detail.
- ✅ REJECT-fix Finding 1 — `--apply` gates BOTH apply mode and restore mode. Restore is dry-run by default. Mutex rule: `--apply requires --apply-plan or --restore-manifest`.
- ✅ REJECT-fix Finding 2 — honest two-phase durability contract. **Phase 1** (before manifest lands): zero hot-registry mutation. **Phase 2** (after manifest lands): destructive phase may be partially applied on IO failure; the manifest is the recovery source. **Restore is the official recovery path** for partial Phase-2 apply.
- ✅ REJECT-fix Finding 3 — removed `skipped_missing`. Closed apply outcome set is exactly `archived | would_archive`. Missing planned sources surface via drift (at preflight) or `io` (TOCTOU between preflight and Pass 1).
- ✅ REJECT-fix Finding 4 — plan-vs-manifest portability is explicit. Plan carries `anchor_registry_dir` (local replay). Archive manifest does NOT carry host-local paths (portable handoff).
- ✅ REJECT-fix Finding 5 — `--before` selector miss routes through `anchor_not_found` with operator-readable detail.
- ✅ Post-implementation REJECT Finding 1 — CLI `--anchor-registry-dir` MUST match the plan's `anchor_registry_dir`. Without this check, an operator could pass `--anchor-registry-dir B --apply-plan plan-for-A.json` and silently mutate registry A while the event log says B. Comparison strategy: try `canonicalize` on both paths and accept equality, fall back to lexical equality if canonicalize fails (e.g. plan dir was relocated). Refused via clap-level `bail!` BEFORE any FS write, with stable message `--anchor-registry-dir does not match the plan's anchor_registry_dir; refusing before any archive write`. Pinned by `cli_check_refuses_when_anchor_registry_dir_does_not_match_plan_dir` + `cli_check_accepts_lexically_equal_dirs_when_paths_dont_exist` + `cli_check_accepts_canonically_equal_dirs_via_tempdir`.
- ✅ Post-implementation REJECT Finding 2 — `--archive-dir` is required ONLY when the plan has ≥1 action. A zero-action plan can apply (dry-run or real-run) without a destination. The library accepts an empty `archive_dir` path on zero-action plans and never reads or writes it. Pinned by `resolve_archive_dir_required_when_plan_has_actions` + `resolve_archive_dir_optional_when_plan_has_zero_actions` (CLI) and `apply_zero_action_plan_succeeds_in_dry_run_with_empty_archive_dir` + `apply_zero_action_plan_succeeds_in_real_run_with_empty_archive_dir` (library).

## CLI

### `archive-integrity-evidence-anchors`

```
omni-node operator evidence-anchor archive-integrity-evidence-anchors

# Common
  --anchor-registry-dir <DIR>                          (required in all three modes)

# Phase selector — plan mode is the default
  [--apply-plan          <PATH>]                       (apply mode)
  [--restore-manifest    <PATH>]                       (restore mode)

# Plan-mode flags
  [--plan-out            <PATH>]                       (stdout if omitted)
  [--status              <FINALIZED|FAILED>]           (repeatable; default = [FINALIZED])
  [--before              <RFC3339>]                    (filter on updated_at < this)
  [--tx-id               <HEX>]                        (repeatable selector)
  [--artifact-hash-hex   <64-lower-hex>]               (repeatable selector)

# Apply-mode flags
  [--archive-dir         <DIR>]                        (required when the plan has ≥1 action)

# Mutation gate — valid in BOTH apply and restore modes
  [--apply]                                            (explicit mutation gate; default dry-run)
```

#### Clap-level mutex / required-with rules

| Rule | Refusal message |
| --- | --- |
| `--apply-plan` and `--restore-manifest` are mutually exclusive | `--apply-plan and --restore-manifest are mutually exclusive` |
| `--plan-out` plan-mode only | `--plan-out is plan-mode only (omit --apply-plan / --restore-manifest)` |
| `--status` / `--before` plan-mode only | `<flag> is plan-mode only` |
| `--tx-id` / `--artifact-hash-hex` plan-mode only | `--tx-id / --artifact-hash-hex are plan-mode only` |
| `--archive-dir` apply-mode only | `--archive-dir is apply-mode only (requires --apply-plan)` |
| **`--apply` requires `--apply-plan` OR `--restore-manifest`** (Finding 1) | `--apply requires --apply-plan or --restore-manifest` |
| `--status` value is one of `finalized | failed` (case-insensitive) | `--status must be one of finalized \| failed (archive operates only on terminal records)` |
| `--artifact-hash-hex` is exactly 64 lower-hex chars | `--artifact-hash-hex must be exactly 64 lowercase hex characters` |
| `--before` parses as RFC 3339 | `--before must be an RFC 3339 timestamp` |
| **`--anchor-registry-dir` must match the plan's `anchor_registry_dir`** (post-implementation Finding 1) | `--anchor-registry-dir does not match the plan's anchor_registry_dir; refusing before any archive write` |
| **`--archive-dir` is required when the plan has ≥1 action** (post-implementation Finding 2) | `--archive-dir is required when the plan has one or more actions` |

Pinned by `stage_13_7_cli_tests::mutex_*` tests. Messages pinned by `violation_messages_are_stable_strings`.

## Plan / manifest schemas

### `AnchorArchivePlan` (LOCAL — not portable; Finding 4)

```json
{
  "schema_version": 1,
  "plan_id": "<16-lower-hex>",
  "created_at_utc": "2026-06-18T00:00:00Z",
  "anchor_registry_dir": "/var/omni-anchors",
  "registry_state_hash": "<64-lower-hex>",
  "omni_zkml_version": "...",
  "actions": [
    {
      "kind": "archive_terminal_record",
      "artifact_hash_hex": "<64-lower-hex>",
      "tx_id": "<string>",
      "status": "finalized" | "failed",
      "source_relative": "<artifact_hash_hex>.json"
    }
  ],
  "archive_plan_hash": "<64-lower-hex>"
}
```

### `AnchorArchiveManifest` (PORTABLE — Finding 4; no `anchor_registry_dir`)

```json
{
  "schema_version": 1,
  "plan_id": "<16-lower-hex>",
  "created_at_utc": "2026-06-18T00:00:00Z",
  "entries": [
    {
      "artifact_hash_hex": "<64-lower-hex>",
      "tx_id": "<string>",
      "status": "finalized" | "failed",
      "archive_relative": "anchors/<artifact_hash_hex>.json",
      "blake3_hex": "<64-lower-hex>",
      "bytes": 1234
    }
  ]
}
```

### Recipes

- `plan_id = lower_hex(BLAKE3(anchor_registry_dir || "||" || registry_state_hash || "||" || created_at_utc))[..16]`.
- `archive_plan_hash = lower_hex(BLAKE3(canonical JSON of plan with this field blanked))`.
- `registry_state_hash = BLAKE3(canonical JSON of { records: sorted_by(artifact_hash_hex) [{ artifact_hash_hex, status, tx_id, updated_at_unix }], tx_index_entries: sorted_by(tx_id) [{ tx_id, artifact_hash_hex }] })`. **Includes `updated_at_unix`** — Q3 lock.

## File layout

```
<archive-dir>/
└── <plan_id>/
    ├── archive_manifest.json
    └── anchors/
        ├── <artifact_hash_hex>.json    # copied verbatim from the hot registry
        └── ...
```

The hot registry is left flat (Stage 13.0 convention); archived bytes go under the `anchors/` subdir per the portable Stage 13.5-style layout.

## Apply preflight ordering (fixed)

1. **Schema-version** → `unsupported_archive_plan_schema_version`. Runs FIRST.
2. **Plan-hash** → `archive_plan_hash_mismatch`.
3. **Drift** → `archive_drift`.
4. **Per-action path validation** → `archive_invalid_path`.

## Apply durability — honest two-phase contract

### Phase 1 — before manifest lands

- **Pass 1** (non-destructive): for each action, read source bytes via `std::fs::read`, compute BLAKE3, atomic-write to `<archive-dir>/<plan_id>/anchors/<hash>.json` (`.tmp + rename`). Accumulate manifest entry. **No hot-registry mutation.**
- **Pass 1.5** (durability fence): atomic-write `<archive-dir>/<plan_id>/archive_manifest.json` (`.tmp + rename`).

**Guarantee on a Phase-1 failure:** the hot registry is byte-identical to its pre-apply state. Operator can re-run apply from scratch. Pinned by `apply_writes_manifest_before_deleting_any_source_record` (Unix-only sabotage test forcing manifest-write failure).

### Phase 2 — after manifest lands

- **Pass 2** (destructive): for each action, `fs::remove_file(<registry>/<hash>.json)`; accumulate `tx_id` removals.
- **Pass 2.5**: atomic-rewrite `<registry>/tx_index.json` with accumulated removals (merge-style; preserves unrelated entries).

**Guarantee on a Phase-2 failure:** the hot registry MAY be partially mutated (some sources deleted, some still there; `tx_index.json` may be the pre-merge or post-merge version). The archive subtree IS complete. **Restore is the official recovery path** — running restore against the manifest re-establishes any source record that Phase 2 successfully deleted; row 2 (byte-equal target) idempotent-skips for any record Phase 2 didn't reach. Pinned by `restore_via_manifest_recovers_archived_records_after_apply`.

## Restore conflict matrix

Mirrors Stage 13.6 import.

| # | `<registry>/<hash>.json`? | target BLAKE3 vs manifest | `tx_index.json` has `tx_id`? | tx_index → | apply outcome | dry-run outcome | refuse |
| - | --- | --- | --- | --- | --- | --- | --- |
| 1 | no  | n/a   | no  | n/a   | `restored`                   | `would_restore`                 | — |
| 2 | yes | equal | yes | same  | `skipped_already_restored`   | `skipped_already_restored`      | — |
| 3 | yes | equal | no  | n/a   | `re_added_tx_index_entry`    | `would_re_add_tx_index_entry`   | — |
| 4 | yes | diff  | any | any   | refuse                       | refuse                          | `archive_target_exists` field=artifact_hash |
| 5 | any | n/a   | yes | diff  | refuse                       | refuse                          | `archive_target_exists` field=tx_id |

Row precedence: **5 → 4 → 1/2/3** (tx_id collision is the highest-precedence failure mode).

Pre-row: archived-bytes integrity check — refuse with `archive_blake3_mismatch` when `blake3(<archive>/<plan_id>/anchors/<hash>.json) != manifest.entries[i].blake3_hex`.

Defense-in-depth: per-entry `verify_anchor_tx_data` runs on the parsed archived record. A hand-edited archive file with a tampered signature refuses with the Stage 13.0 `submitter_signature_invalid` tag.

Preflight-all-before-mutate: classify EVERY entry first; refuse on any conflict with zero writes. Row-3 does NOT rewrite the byte-equal record file (mtime preservation test).

## Closed reason-tag taxonomy (Q2 lock)

**NEW (6):**

| Variant | Tag |
| --- | --- |
| `ArchivePlanSchemaUnsupported { got, expected }` | `unsupported_archive_plan_schema_version` |
| `ArchivePlanHashMismatch { computed, expected }` | `archive_plan_hash_mismatch` |
| `ArchiveDrift { computed, expected }` | `archive_drift` |
| `ArchiveInvalidPath { source_relative, reason }` | `archive_invalid_path` |
| `ArchiveBlake3Mismatch { archive_relative, computed, expected }` | `archive_blake3_mismatch` |
| `ArchiveTargetExists { field, artifact_hash_hex, tx_id }` | `archive_target_exists` |

**REUSED (no surface growth):** `anchor_not_found`, `malformed_json`, `io`, `submitter_signature_invalid`, `unsupported_anchor_schema_version`.

Each new variant has a mapper arm + an entry in the exhaustive `every_variant_has_a_stable_tag` test. Adding a variant without a mapper arm is a compile error.

## Informational event taxonomy

| Event | When | Fields |
| --- | --- | --- |
| `event=integrity_evidence_anchor_archive_plan_started` | Plan startup | `anchor_registry_dir=` |
| `event=integrity_evidence_anchor_archive_plan_written` | Plan written to `--plan-out` | `path=` |
| `event=integrity_evidence_anchor_archive_plan_summary` | Plan complete | `plan_id=`, `actions=` |
| `event=integrity_evidence_anchor_archive_apply_started` | Apply startup | `anchor_registry_dir=`, `plan=`, `mode=apply|dry_run` |
| `event=integrity_evidence_anchor_archive_apply_summary` | Apply complete | `plan_id=`, `mode=`, `actions_archived=`, `actions_would_archive=`, `archive_dir=`, `archive_manifest_relative=` |
| `event=integrity_evidence_anchor_archive_restore_started` | Restore startup | `anchor_registry_dir=`, `manifest=`, `mode=apply|dry_run` |
| `event=integrity_evidence_anchor_archive_restore_summary` | Restore complete | `plan_id=`, `mode=`, `restored=`, `would_restore=`, `skipped_already_restored=`, `re_added_tx_index_entry=`, `would_re_add_tx_index_entry=` |
| `event=integrity_evidence_anchor_archive_failed` | Refusal | `reason=<closed-set tag>`, `detail=` |

No new `cause=` keys. Refusals route through `reason=`.

## Operator-visible implications of byte-preserve

Archived records carry the source registry's `submitted_at` / `updated_at` verbatim. When restored, those timestamps are HISTORICAL — exactly as in Stage 13.6 import. Stage 13.3 stale-age views may report historical ages on restored records by design.

## Test inventory

In `crates/omni-zkml/tests/evidence_anchor_archive.rs` (38 library integration tests, hermetic):

- **Plan / selection** (5): default `[Finalized]`, explicit `FAILED` opt-in, selector miss, non-terminal-record detail, `--before` exclusion detail.
- **Plan integrity** (5): plan-hash byte-stable, plan-hash blanks self, state-hash changes on status / `updated_at` / unrelated tx_index changes.
- **Apply preflight** (4): unsupported schema, ordering pin, plan-hash mismatch, drift.
- **Apply path validation** (3): absolute path, parent traversal, wrong per-kind shape.
- **Apply happy paths + durability** (5): dry-run no-mutation, real-run copy-then-delete, **manifest-before-delete (cfg-unix sabotage)**, tx_index removal, unrelated tx_index preservation.
- **Manifest portability** (1): no `anchor_registry_dir` field in the on-disk manifest JSON.
- **Phase-2 recovery via restore** (1): apply-then-restore round-trips records and tx_index.
- **Restore happy paths** (3): byte-preserve round-trip, tx_index re-add, unrelated tx_index preservation.
- **Restore conflict matrix** (5): row 2 idempotent skip, row 3 no-rewrite mtime pin, row 4 `field=artifact_hash`, row 5 `field=tx_id`, pre-row BLAKE3 mismatch.
- **Restore path validation** (2): absolute path, parent traversal.
- **Restore dry-run** (1): `would_restore` outcome and no FS mutation.
- **Preflight-all-before-mutate** (1): clean entry does NOT land when later entry refuses.
- **Defense-in-depth** (1): tampered archived signature refuses via Stage 13.0 verifier.
- **Plan round-trip** (1): plan written to disk then re-loaded re-applies cleanly.

In `crates/omni-node/src/evidence_anchor_cli.rs::stage_13_7_cli_tests` (27 CLI tests):

- 10 mutex pass/refuse pins (including the **REJECT-fix Finding 1 pin**: `mutex_passes_for_restore_mode_with_apply`).
- 4 status closed-set pins (terminal-only refusals + case-insensitive accept + unknown refusal).
- 2 `--before` parse pins.
- 1 `--artifact-hash-hex` shape pin.
- 1 violation-message stability pin.
- 7 reason-tag mapper routing pins (one per new variant + the `field=tx_id` case).
- 1 selector-miss `anchor_not_found` reuse pin.
- 1 mutex pass for default plan mode.

All hermetic. No network.

## Out of scope (Stage 13.8+ candidates)

- Chain-side cross-check of archived `tx_id`s (Stage 13.9).
- Automatic / scheduled archival (operator-driven only).
- Multi-archive merge into a single subtree.
- Compression / encryption of the archive subtree (operator-host concern).
- A `--force` flag for restore over-clobber (intentionally absent).
- An archive-of-archive (recursion is out of scope).
- A `--prune-archive` flag (Stage 13.8+; old archives are operator-managed by `rm`).

## Forward outlook

- **Stage 13.8** — local registry consistency report. Cross-checks hot registry + archive subtrees + `tx_index.json` integrity in a single read-only sweep. Reports drift, dangling references, orphan archives. No chain interaction.
- **Stage 13.9** — comprehensive chain read / reconcile support. Operator-driven chain catch-up that bridges archived records back into reconcile flows when needed (e.g. an archived `Failed` record that the chain later re-finalized).

> **Post-Stage-13.8 note:** Stage 13.8 turns Stage 13.7's archive subtrees into one of three first-class inspection surfaces (hot, archive, export) in a single read-only consistency report. Stage 13.7's archive shape / restore path / two-phase durability contract is **unchanged**; Stage 13.8 reads archives without ever invoking restore. A Phase-2 partial-apply state surfaces as `archive_hot_collision_same_bytes` (warning) findings on records Phase 2 didn't reach. See [`docs/stage13.8-anchor-consistency-report.md`](stage13.8-anchor-consistency-report.md).
