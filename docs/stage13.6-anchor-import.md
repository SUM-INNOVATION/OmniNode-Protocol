# Stage 13.6 — local-only integrity-evidence-anchor export import / registry restore

## Stage scope

Stage 13.6 adds an operator-facing import path that restores selected `anchor_record` entries from a Stage 13.5 export into a target local anchor registry. Fully local — no SUM Chain RPCs, no `omni-sumchain` types, no private-chain repo deps. Stage 13.0 wire / domain / canonical-bytes / signing is **read-only** — not a byte is re-signed.

## What Stage 13.6 ships

- **Library helpers** in `omni-zkml::evidence_anchor::import`:
  - `plan_anchor_export_import(export_dir, target_registry, opts) -> AnchorImportPlan`
  - `apply_anchor_export_import(export_dir, target_registry, opts) -> AnchorImportReport`
- **New CLI subcommand** on `omni-node operator evidence-anchor`:
  - `import-integrity-evidence-anchor-export`
- **ONE new `EvidenceAnchorError` variant** + one new mapper arm. **One new tag string**: `import_target_exists`. Disambiguator on the variant: `field: "artifact_hash" | "tx_id"`.
- **Reused tag strings** (no surface growth): `anchor_not_found` (selector miss), `export_entry_metadata_mismatch` (manifest-vs-record drift), `malformed_json`, `io`, `submitter_signature_invalid`, `unsupported_anchor_schema_version`, plus the full Stage 13.5 `export_*` set propagated from `verify_anchor_export`.

## What Stage 13.6 does NOT ship

- ❌ No SUM Chain RPC interaction. Fully local.
- ❌ No `AnchorRecord` shape change. Records imported byte-for-byte.
- ❌ No wire / domain / canonical-bytes / signing changes.
- ❌ No submit / retry / re-signing path.
- ❌ No source-registry mutation. The export is read-only.
- ❌ No target-registry record overwrite. Conflicts refuse via `import_target_exists`; same-bytes targets are idempotent.
- ❌ No artifact_bytes / signed_chain_report restoration. Stage 13.6 only restores the `anchor_record` slice of an export.
- ❌ No `--force-overwrite-tx-id-collision`, no `--re-stamp-updated-at-on-import`. Operators handle conflicts out-of-band.

## Locked decisions (Q&A + reviewer findings)

- ✅ D1 — Idempotent re-import is a `skipped_already_imported` outcome, NOT a refusal.
- ✅ D2 — Apply preflights ALL classifications BEFORE any FS mutation (implementation note 1). A conflict on action N must refuse with zero writes from actions 1..N-1.
- ✅ D3 — `tx_index.json` is **merged** (load existing → add new → atomic rewrite), not rebuilt. Unrelated entries are preserved verbatim.
- ✅ D4 — `--strict` calls `verify_anchor_export(strict=true)` on the WHOLE export tree, not just the records being imported.
- ✅ D5 — Record bytes are preserved verbatim. `submitted_at` / `updated_at` are HISTORICAL (export-time facts). Stage 13.3 stale-age views may show historical ages on imported records by design.
- ✅ D6 — No-selector default imports all `anchor_record` entries from the manifest. Asymmetric to Stage 13.5 export's required-selector (export is already a curated subset).
- ✅ D7 — `export_entry_metadata_mismatch` (Stage 13.5) is reused for manifest-vs-record drift. No `import_manifest_metadata_mismatch` tag added.
- ✅ D8 — Selector misses refuse with **reused** `anchor_not_found` (correct existing tag).
- ✅ D9 — Stage 13.2 reconcile / query and Stage 13.3 watch / summary continue to work on imported records unchanged. The chain is the source of truth for status drift after import.
- ✅ Implementation note 1 — Apply runs preflight-all-before-mutate. Pinned by `apply_does_not_modify_target_when_any_action_refuses_with_target_exists`.
- ✅ Implementation note 2 — Row 3 (`re_added_tx_index_entry`) does NOT rewrite the byte-equal record file; only `tx_index.json` changes. Pinned by `import_re_adds_missing_tx_index_entry_when_record_file_already_present` (mtime preservation check).

## CLI

### `import-integrity-evidence-anchor-export`

```
omni-node operator evidence-anchor import-integrity-evidence-anchor-export
  --export-dir          <DIR>                                          (required)
  --anchor-registry-dir <DIR>                                          (required; target)
  [--apply]                                                            (explicit mutation gate; default dry-run)
  [--strict]                                                           (passes through to verify_anchor_export)
  [--artifact-hash-hex  <64-lower-hex>]                                (repeatable selector)
  [--tx-id              <hex>]                                         (repeatable selector)
  [--status             <SUBMITTED|INCLUDED|FINALIZED|FAILED>]         (repeatable selector)
```

#### Clap-level mutex / required-with rules

| Rule | Refusal message |
| --- | --- |
| `--status` value is one of the four closed kinds (case-insensitive) | `--status must be one of submitted \| included \| finalized \| failed` |
| `--artifact-hash-hex` is exactly 64 lower-hex chars | `--artifact-hash-hex must be exactly 64 lowercase hex characters` |

No required-selector rule (D6 — intentional asymmetry vs Stage 13.5 export). Pinned by `mutex_passes_with_no_selectors_for_d6_lock`. Messages pinned by `violation_messages_are_stable_strings`.

## Library surface

```rust
pub struct AnchorImportOptions<'a> {
    pub dry_run: bool,                          // default true at the CLI layer
    pub strict: bool,                           // passed through to verify_anchor_export
    pub selection: &'a AnchorImportSelection,
    pub now_utc: &'a str,                       // RFC 3339; folded into event line
}

pub struct AnchorImportSelection {
    pub statuses: Vec<LocalAnchorStatus>,
    pub tx_ids: Vec<String>,
    pub artifact_hashes: Vec<String>,
}

pub struct AnchorImportPlan {
    pub export_id: String,
    pub created_at_utc: String,
    pub actions: Vec<PlannedImportAction>,  // sorted by artifact_hash_hex asc
}

pub struct PlannedImportAction {
    pub artifact_hash_hex: String,
    pub tx_id: String,
    pub status: String,                     // closed-set: submitted | included | finalized | failed
    pub relative_path: String,              // anchors/<hash>.json
    pub bytes: u64,
    pub pre_action_outcome: String,         // closed-set; see §6
}

pub struct AnchorImportActionOutcome {
    pub artifact_hash_hex: String,
    pub tx_id: String,
    pub status: String,
    pub outcome: String,                    // closed-set; see §6
}

pub struct AnchorImportReport {
    pub export_id: String,
    pub mode: String,                       // "apply" | "dry_run"
    pub actions_imported: u32,
    pub actions_would_import: u32,
    pub actions_skipped_already_imported: u32,
    pub actions_re_added_tx_index_entry: u32,
    pub actions_would_re_add_tx_index_entry: u32,
    pub outcomes: Vec<AnchorImportActionOutcome>,
}
```

## Conflict matrix (D1 locked)

For each selected `anchor_record` entry, classify against the target registry's current state:

| # | `<registry>/<hash>.json` exists? | Target file BLAKE3 vs manifest `blake3_hex` | `tx_index.json` has `tx_id`? | tx_index → maps to | Outcome (apply) | Outcome (dry-run) | Refuse? |
| - | --- | --- | --- | --- | --- | --- | --- |
| 1 | no  | n/a   | no  | n/a   | `imported`                   | `would_import`                  | — |
| 2 | yes | equal | yes | same  | `skipped_already_imported`   | `skipped_already_imported`      | — |
| 3 | yes | equal | no  | n/a   | `re_added_tx_index_entry`    | `would_re_add_tx_index_entry`   | — |
| 4 | yes | diff  | any | any   | refuse                       | refuse                          | **`import_target_exists` field=artifact_hash** |
| 5 | any | n/a   | yes | diff  | refuse                       | refuse                          | **`import_target_exists` field=tx_id** |

Row precedence: 5 → 4 → 1/2/3. Row 5 (tx_id collision) is the highest-precedence failure mode and short-circuits first — a row-5 collision wins over a row-1 fresh-import even when the artifact-hash file is absent. Row 4 fires next when row 5 doesn't apply. Rows 1/2/3 are the non-refusal outcomes selected on remaining state.

Closed outcome strings (the only ones the report may carry):

- `imported`
- `would_import`
- `skipped_already_imported` *(same string in apply and dry-run — non-mutating in both)*
- `re_added_tx_index_entry`
- `would_re_add_tx_index_entry`

## Import phases

1. **Verify** — `verify_anchor_export(--strict?)`. Refuses with the Stage 13.5 tag set on any failure. **Apply re-runs verify as a durability fence** so plan-then-tamper-then-apply is caught.
2. **Select** — filter manifest `anchor_record` entries via AND-across-kinds / OR-within-kind. No-selector default imports all (D6). Selector misses refuse with `anchor_not_found` (D8).
3. **Cross-check** — manifest's per-entry `artifact_hash_hex` / `tx_id` / `status` must match the record file's own fields. Refuses with `export_entry_metadata_mismatch` on drift (D7 — reused tag, not a new one).
4. **Classify all** — every selected action gets a §6 outcome. Implementation note 1: ALL classifications run BEFORE any FS write.
5. **Refuse-first** — any `import_target_exists` outcome refuses with zero writes.
6. **Apply** (only when `dry_run = false`):
   - **Pass 1** — write each `imported` action's bytes via atomic temp+rename to `<registry>/<hash>.json`. Bytes are read verbatim from `<export-dir>/anchors/<hash>.json` — no serde round-trip (D5). `re_added_tx_index_entry` actions do NOT touch the record file (note 2). `skipped_already_imported` does nothing.
   - **Pass 1.5 (durability fence)** — merge `tx_index.json`: load existing → add the imported actions' `(tx_id → artifact_hash_hex)` mappings → atomic temp+rename. Preserves every unrelated entry (D3).
7. **Return** the typed report.

## Reason-tag taxonomy

**NEW (1 tag — minimal footprint):**

| Variant | Tag string |
| --- | --- |
| `EvidenceAnchorError::ImportTargetExists { field, artifact_hash_hex, tx_id }` | `import_target_exists` |

The `field: &'static str` discriminator carries `"artifact_hash"` or `"tx_id"`. One variant + one tag covers both conflict cases.

**Reused (no surface growth):**

| Tag | Source | When |
| --- | --- | --- |
| `anchor_not_found` | Stage 13.0 | selector miss (D8) |
| `export_entry_metadata_mismatch` | Stage 13.5 | manifest claim doesn't match record's actual fields (D7) |
| `malformed_json` | Stage 13.0 | manifest / record parse failure |
| `io` | Stage 13.0 | FS errors |
| `submitter_signature_invalid` | Stage 13.0 | `verify_anchor_tx_data` failure (defense-in-depth on the import path too) |
| `unsupported_anchor_schema_version` | Stage 13.0 | same |
| `unsupported_export_manifest_schema_version` | Stage 13.5 | `verify_anchor_export` step 2 |
| `export_manifest_hash_mismatch` | Stage 13.5 | `verify_anchor_export` step 3 |
| `export_invalid_path` | Stage 13.5 | `verify_anchor_export` step 4 |
| `export_blake3_mismatch` | Stage 13.5 | `verify_anchor_export` step 6 |
| `export_strict_mode_artifact_bytes_missing` | Stage 13.5 | `--strict` passthrough |

The exhaustive `every_variant_has_a_stable_tag` test extends to cover `ImportTargetExists` — adding a variant without a mapper arm is a compile error.

## Informational event taxonomy

| Event | When | Fields |
| --- | --- | --- |
| `event=integrity_evidence_anchor_import_started` | Import startup | `export_dir=`, `anchor_registry_dir=`, `apply=`, `strict=` |
| `event=integrity_evidence_anchor_import_ok` | Import complete | `export_id=`, `mode=`, `actions_imported=`, `actions_would_import=`, `actions_skipped_already_imported=`, `actions_re_added_tx_index_entry=`, `actions_would_re_add_tx_index_entry=` |
| `event=integrity_evidence_anchor_import_failed` | Refusal | `reason=<closed-set tag>`, `detail=` |

No new `cause=` keys. Refusals route through `reason=`.

## Operator-visible implications of byte-preserve (D5)

Imported records carry the source registry's `submitted_at` / `updated_at` timestamps verbatim. This means:

- Stage 13.3 `summary-integrity-evidence-anchors --stale-threshold-secs <N>` will report imported `Submitted` / `Included` records as stale if their **source-registry** `submitted_at` is past the threshold — even if they were imported moments ago on this host.
- Stage 13.3 `summary --include-health` will count these records normally (the health check only looks at JSON shape + tx_index integrity).
- Stage 13.2 reconcile will query the chain for each imported record's `tx_id` and apply any chain-returned status transition — this is the recommended post-import step to bring the imported records' status up to current chain truth.
- The file's mtime on the target host reflects the IMPORT time (the only "this registry first saw the record" trace).

This is the **intentional** consequence of forensic fidelity. Operators wanting "when did this host first see the record" timestamps consult the import event line + file mtime.

## Test inventory

In `crates/omni-zkml/tests/evidence_anchor_import.rs` (23 library integration tests, hermetic):

- **Plan / selection** (5): no-selector (D6), status-only, tx_id-only, artifact_hash-only, status×artifact_hash intersection.
- **Verify-first invariant** (2): plan refuses on `export_blake3_mismatch`; apply re-verifies and refuses on post-plan tamper.
- **Selector miss** (1): `anchor_not_found` (D8).
- **Byte-preserve** (1): source bytes == target bytes after import (D5).
- **Conflict matrix** (5): rows 1 (fresh import), 2 (skip), 3 (re-add tx_index), 4 (artifact-hash collision), 5 (tx_id collision).
- **tx_index merge** (1): unrelated entries preserved (D3).
- **Dry-run** (3): `would_import`, `would_re_add_tx_index_entry`, `skipped_already_imported` (no `would_` prefix).
- **Strict mode passthrough** (1): `export_strict_mode_artifact_bytes_missing` propagates.
- **Metadata mismatch tag reuse** (1): `export_entry_metadata_mismatch` (D7).
- **`field=` disambiguator** (2): artifact_hash vs tx_id.
- **Preflight-all-before-mutate** (1): clean action A does NOT land when later action B refuses (implementation note 1).

In `crates/omni-node/src/evidence_anchor_cli.rs::stage_13_6_cli_tests` (10 CLI tests):

- 2 mutex refusal pins (`--status` closed set, `--artifact-hash-hex` shape).
- 1 status case-insensitive pin (all four kinds).
- 1 no-selector mutex-pass pin (D6 lock).
- 1 violation-message stability pin.
- 2 reason-tag mapper routing pins for `ImportTargetExists` (both `field=` values).
- 1 selector-miss `anchor_not_found` reuse pin.
- 1 `--apply` opt-in semantics pin.

All hermetic. No network.

## Out of scope (Stage 13.7+ candidates)

- Importing artifact_bytes or signed_chain_report files. Stage 13.6 only restores the `anchor_record` slice.
- Re-running Stage 13.0 submit on imported records.
- Cross-host export sync over the wire (operator's transport concern).
- Conflict resolution beyond refuse-or-skip — no `--force-overwrite-tx-id-collision`, no `--re-stamp-updated-at-on-import`.
- Multi-export merge into a single registry (operators run multiple imports sequentially).
- Importing into an in-memory registry surface.
