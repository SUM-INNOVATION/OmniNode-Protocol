# Stage 13.8 — local integrity-evidence-anchor consistency report

## Stage scope

Stage 13.8 adds a strictly local, strictly **read-only** consistency report that inspects the hot integrity-evidence-anchor registry plus optional Stage 13.5 exports and Stage 13.7 archives in a single sweep. Fully local — no SUM Chain RPCs, no `omni-sumchain` types, no private chain repo deps. Stage 13.0 wire / domain / canonical-bytes / signing is **read-only**.

This is the last purely-local lifecycle stage. Stage 13.9 (forward outlook) is the chain-facing comprehensive read / reconcile stage.

## What Stage 13.8 ships

- Library helper in `omni-zkml::evidence_anchor::consistency`:
  - `build_anchor_consistency_report(opts) -> AnchorConsistencyReport`
- One new schema constant: `ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION = 1`.
- One new CLI subcommand on `omni-node operator evidence-anchor`:
  - `report-integrity-evidence-anchor-consistency`
- **ZERO new `EvidenceAnchorError` variants. ZERO new closed `reason=` tag strings. ZERO modifications to `crates/omni-zkml/src/error.rs`.**

## What Stage 13.8 does NOT ship

- ❌ No mutation. No registry, export, archive, or `tx_index.json` write helper is called from the consistency module.
- ❌ No SUM Chain RPC interaction.
- ❌ No `EvidenceAnchorError` taxonomy growth — findings are typed report data on the report, not refusals.
- ❌ No `--include-ok` flag — deferred per v1 Q4 lock.
- ❌ No `--strict-exports` flag — Stage 13.5's strict mode is not propagated.
- ❌ No archive deletion / pruning / repair. Operators run Stage 13.4 cleanup / Stage 13.7 restore separately when findings demand it.

## Locked decisions

- ✅ Q1 — `--archive-dir` accepts both concrete `<archive-root>/<plan_id>` dirs AND archive roots containing multiple `<plan_id>/archive_manifest.json` subtrees. Detection rule: if `<dir>/archive_manifest.json` exists → concrete; else scan immediate children for `<child>/archive_manifest.json` → root; else emit `ArchiveDirNoManifest` (warning).
- ✅ Q2 — coarse `ExportVerifyFailed`. Delegates to Stage 13.5 `verify_anchor_export(strict=false)`; the failure's Stage 13.5 reason tag is carried in the `detail` string. No per-export-error-kind findings; avoids duplicating Stage 13.5 taxonomy.
- ✅ Q3 — severity matrix locked in §6. Cross-surface "expected duplicate" overlaps moved to summary counters, NOT per-item findings (v2 Finding 4).
- ✅ Q4 — `--include-ok` deferred. The summary's per-status counts already give operators total-record visibility.
- ✅ Q5 — optional path unreadability is a `warning` finding, NOT a command failure. Required `--anchor-registry-dir` unreadability bubbles up as `EvidenceAnchorError::Io` (existing tag).
- ✅ v2 Finding 1 — `ArchiveDirUnreadable` AND `ExportDirUnreadable` are BOTH `warning` severity. The locked principle: **optional path can't be opened OR no manifest found → `warning`**; **readable AND claims to be an archive/export but fails integrity/schema → `error`**.
- ✅ v2 Finding 2 — no brittle exhaustive-count test on `EvidenceAnchorError`. Stage 13.8's contract is that `error.rs` is not modified; the review confirms via file diff.
- ✅ v2 Finding 4 — `HotAndExportDuplicate` / `ArchiveAndExportDuplicate` dropped from v1 enum. Overlap information lives in `summary.hot_export_overlaps` and `summary.archive_export_overlaps` (unique-set semantics so duplicate manifest entries don't inflate).
- ✅ v2 Finding 5 — `ArchiveDirNoManifest` (warning) is distinct from `ArchiveDirUnreadable` (warning). The former fires when the path exists and is readable but contains no manifest; the latter fires when `read_dir()` itself fails.
- ✅ v2 Finding 6 — bad `--json-out` parent is a CLI usage error (`bail!`) emitted BEFORE the library is invoked. Pinned by `json_out_with_missing_parent_fails_at_cli_boundary`.

## CLI

```
omni-node operator evidence-anchor report-integrity-evidence-anchor-consistency
  --anchor-registry-dir       <DIR>                     (required; hot registry)
  [--archive-dir              <DIR>]                    (repeatable; concrete or root)
  [--export-dir               <DIR>]                    (repeatable; Stage 13.5 export dir)
  [--stale-threshold-secs     <U64>]                    (optional)
  [--json-out                 <PATH>]                   (optional; atomic .tmp + rename; parent must exist)
```

## Library surface

```rust
pub const ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION: u32 = 1;

pub enum AnchorConsistencySeverity { Info, Warning, Error }

pub enum AnchorConsistencyFindingKind { /* 24 variants — see §6 */ }

pub struct AnchorConsistencyFinding {
    pub severity: AnchorConsistencySeverity,
    pub kind: AnchorConsistencyFindingKind,
    pub location: String,
    pub artifact_hash_hex: Option<String>,
    pub tx_id: Option<String>,
    pub detail: String,
    pub suggested_action: Option<String>,
}

pub struct AnchorConsistencySummary {
    pub hot_total: u64,
    pub hot_submitted: u64,
    pub hot_included: u64,
    pub hot_finalized: u64,
    pub hot_failed: u64,
    pub hot_malformed_records: u64,
    pub tx_index_entries: u64,
    pub tx_index_orphans: u64,
    pub archive_manifests_scanned: u64,
    pub archive_entries_scanned: u64,
    pub export_manifests_scanned: u64,
    pub export_entries_scanned: u64,
    pub hot_export_overlaps: u64,
    pub archive_export_overlaps: u64,
    pub findings_by_severity_info: u64,
    pub findings_by_severity_warning: u64,
    pub findings_by_severity_error: u64,
}

pub struct AnchorConsistencyReport {
    pub schema_version: u32,
    pub created_at_utc: String,
    pub anchor_registry_dir: String,
    pub archive_dirs: Vec<String>,
    pub export_dirs: Vec<String>,
    pub stale_threshold_secs: Option<u64>,
    pub summary: AnchorConsistencySummary,
    pub findings: Vec<AnchorConsistencyFinding>,
}

pub struct AnchorConsistencyOptions<'a> {
    pub anchor_registry_dir: &'a Path,
    pub archive_dirs: &'a [PathBuf],
    pub export_dirs: &'a [PathBuf],
    pub stale_threshold_secs: Option<u64>,
    pub now_utc: &'a str,
}

pub fn build_anchor_consistency_report(
    opts: &AnchorConsistencyOptions<'_>,
) -> EvidenceAnchorResult<AnchorConsistencyReport>;
```

## Finding taxonomy (24 closed kinds)

**Hot registry (9):**
`HotRecordMalformed`, `HotRecordSignatureInvalid`, `HotRecordSchemaUnsupported`, `HotFilenameHashMismatch`, `HotTxIndexOrphan`, `HotTxIndexMismatch`, `HotTxIdDuplicate`, `HotStaleOpenRecord`, `HotTmpOrphan`.

**Archive (13):**
`ArchiveDirUnreadable`, `ArchiveDirNoManifest`, `ArchiveManifestMalformed`, `ArchiveManifestSchemaUnsupported`, `ArchiveEntryInvalidPath`, `ArchiveEntryMissingFile`, `ArchiveEntryBlake3Mismatch`, `ArchiveEntryRecordMalformed`, `ArchiveEntrySignatureInvalid`, `ArchiveEntryMetadataMismatch`, `ArchiveHotCollisionSameBytes`, `ArchiveHotCollisionDifferentBytes`, `ArchiveTxIdCollision`.

**Export (2):**
`ExportDirUnreadable`, `ExportVerifyFailed`.

## Severity matrix

| Finding | Severity | Notes |
| --- | --- | --- |
| `HotRecordMalformed` / `HotRecordSignatureInvalid` / `HotRecordSchemaUnsupported` / `HotFilenameHashMismatch` / `HotTxIndexMismatch` / `HotTxIdDuplicate` | `error` | Hot-registry integrity broken; run Stage 13.4 cleanup or investigate manually. |
| `HotTxIndexOrphan` / `HotStaleOpenRecord` / `HotTmpOrphan` | `warning` | Cleanup target; operator decides timing. |
| `ArchiveDirUnreadable` / `ArchiveDirNoManifest` | `warning` | Optional path; report continues. **v2 Finding 5 lock.** |
| `ArchiveManifestMalformed` / `ArchiveManifestSchemaUnsupported` / `ArchiveEntry*` | `error` | Archive readable-but-invalid; restore would refuse. |
| `ArchiveHotCollisionSameBytes` | `warning` | Likely partial Phase-2 archive state; investigate via Stage 13.7 logs. |
| `ArchiveHotCollisionDifferentBytes` / `ArchiveTxIdCollision` | `error` | Real divergence between archive and hot. Stage 13.7 restore would refuse with `archive_target_exists`. |
| `ExportDirUnreadable` | `warning` | **v2 Finding 1 lock** — was `error` in v1, now consistent with `ArchiveDirUnreadable`. Optional path; report continues. |
| `ExportVerifyFailed` | `error` | Readable-but-invalid export. `detail` carries the Stage 13.5 reason tag (`unsupported_export_manifest_schema_version`, `export_manifest_hash_mismatch`, etc.). |

**The locked severity principle (v2 Finding 1 + 3):**
- Optional path cannot be opened OR no manifest found at the expected location → **`warning`**.
- Optional path is readable AND claims to be an archive/export but fails integrity/schema checks → **`error`**.

## How cross-surface duplicates are reported (v2 Finding 4 lock)

Stage 13.8 does NOT emit per-item findings for byte-equal hot↔export or archive↔export overlaps. Instead:

- `summary.hot_export_overlaps: u64` — count of unique `artifact_hash_hex` values present in BOTH the hot registry and at least one export's `anchor_record` entries.
- `summary.archive_export_overlaps: u64` — count of unique `artifact_hash_hex` values present in BOTH at least one archive and at least one export.

**Set semantics**: each unique `artifact_hash_hex` contributes at most 1 to each counter, regardless of how many manifest entries reference it. A 10k-record export overlapping with a 10k-record hot registry contributes at most 10k to `hot_export_overlaps` — never more, even if a single export contains duplicate manifest entries (which Stage 13.5 itself refuses).

Operators inspecting `hot_export_overlaps=N` see at a glance that the export covers N records of the hot registry without scrolling through N info findings.

The archive↔hot byte-equal duplicate IS emitted as `ArchiveHotCollisionSameBytes` (warning) because it indicates a likely partial-Phase-2 archive state, not a normal copy-by-design relationship — operationally distinct from the export case.

## Hot registry checks

The hot scan does ONE filesystem walk + ONE tx_index.json read.

For each `<64-lower-hex>.json`:
1. Read file bytes → populate `<filename_stem> → blake3` map (used for cross-surface collision detection even if the record is malformed).
2. Parse as `AnchorRecord` → if fails: `HotRecordMalformed`.
3. Filename stem == record's `artifact_hash_hex` → if fails: `HotFilenameHashMismatch`.
4. `verify_anchor_tx_data(&record.tx_data)` → routes to `HotRecordSignatureInvalid` or `HotRecordSchemaUnsupported` based on the Stage 13.0 error.
5. Status counter increment + `(tx_id → hash)` collected for cross-record tx_id duplicate detection.

For `.tmp` files: `HotTmpOrphan`.

For `tx_index.json`: read + parse. The tx_index is the CANONICAL source of truth for the hot `(tx_id → artifact_hash)` mapping (used for archive tx_id collision detection):
- For each `(tx_id, hash)` entry:
  - Update `summary.tx_index_entries`.
  - Populate `hot_tx_id_to_hash[tx_id] = hash`.
  - If no `<hash>.json` file exists → `HotTxIndexOrphan` (warning).
  - **`HotTxIndexMismatch` (error) fires in either of two cases**, at most one finding per `tx_index` entry:
    - **(A) Forward** — the record at `<hash>.json` exists but its own `receipt.tx_id` differs from the tx_index entry's `tx_id`. Lookup by `tx_id` would route to a record that doesn't claim it.
    - **(B) Reverse** — a record claiming `receipt.tx_id == tx_id` exists under a DIFFERENT artifact hash. The tx_index points at a wrong record.
    - If both cases apply, the forward finding wins (it's the more direct statement of the problem).

Stale detection runs only when `--stale-threshold-secs` is set: walks well-formed Submitted/Included records and emits `HotStaleOpenRecord` per stale entry.

## Archive checks (v2 Finding 5 distinction)

For each `--archive-dir`:

1. If `<dir>/archive_manifest.json` exists → treat as concrete plan dir; scan its manifest + entries.
2. Else try `read_dir(<dir>)`. If `read_dir()` fails → `ArchiveDirUnreadable` (warning); continue.
3. Else scan immediate children for `<child>/archive_manifest.json`. If any exist → scan each. If NONE exist → `ArchiveDirNoManifest` (warning); continue.

Per-manifest checks: `ArchiveManifestMalformed`, `ArchiveManifestSchemaUnsupported`.

Per-entry checks:
1. Path shape (must be `anchors/<64-lower-hex>.json`) → `ArchiveEntryInvalidPath`.
2. File present → `ArchiveEntryMissingFile`.
3. Computed BLAKE3 + length match manifest → `ArchiveEntryBlake3Mismatch`.
4. Parse as `AnchorRecord` → `ArchiveEntryRecordMalformed`.
5. `verify_anchor_tx_data` → `ArchiveEntrySignatureInvalid`.
6. Metadata cross-check (manifest's `artifact_hash_hex` / `tx_id` / `status` match the record's fields) → `ArchiveEntryMetadataMismatch`.
7. Cross-surface vs hot:
   - Hot has byte-equal record at the same hash → `ArchiveHotCollisionSameBytes` (warning).
   - Hot has byte-different record at the same hash → `ArchiveHotCollisionDifferentBytes` (error).
   - Hot's `tx_index.json` maps the archive's tx_id to a different artifact hash → `ArchiveTxIdCollision` (error).

## Export checks (coarse — Q2 lock)

For each `--export-dir`:
1. If `<dir>` is not a directory → `ExportDirUnreadable` (warning). Report continues.
2. Else invoke `verify_anchor_export(strict=false)` (Stage 13.5).
3. On any Stage 13.5 typed error → `ExportVerifyFailed` (error). `detail` string carries the Stage 13.5 reason tag verbatim.
4. On verify success → re-read the manifest; increment `summary.export_manifests_scanned` and `export_entries_scanned`. Anchor-record artifact_hashes contribute to the cross-surface `export_hash_union` for `hot_export_overlaps` / `archive_export_overlaps` counters.

## Event taxonomy — NO `reason=` for findings

```
event=integrity_evidence_anchor_consistency_started anchor_registry_dir=<DIR> archive_dirs=<N> export_dirs=<M> stale_threshold_secs=<S|absent>
event=integrity_evidence_anchor_consistency_finding severity=<info|warning|error> kind=<snake_case_kind> location=<path-or-id> [artifact_hash_hex=<HEX>] [tx_id=<HEX>] detail="<quoted>"
event=integrity_evidence_anchor_consistency_summary hot_total=<N> ... findings_by_severity_error=<N>
event=integrity_evidence_anchor_consistency_report_written path=<PATH>           # only when --json-out is set
```

The only `reason=` line the command can emit is `integrity_evidence_anchor_consistency_failed reason=io detail=…`, fired only when the required `--anchor-registry-dir` is itself unreadable at startup.

The `--json-out` parent-directory boundary refuses via `bail!` (clap-level usage error) BEFORE the library is invoked; that refusal does NOT carry `reason=io` (v2 Finding 6 lock).

## Test inventory

In `crates/omni-zkml/tests/evidence_anchor_consistency.rs` (32 library integration tests, hermetic):

- **Hot registry** (11): empty, by-status counts, malformed, tampered signature, schema-unsupported, filename-hash mismatch, tx_index orphan, tx_index mismatch (reverse direction), tx_index mismatch (forward direction — REJECT-fix regression), tx_id duplicate, .tmp orphan, stale-with-threshold.
- **Archive** (10): concrete dir + root scan (Q1), no-manifest (Finding 5), unreadable path, manifest-malformed, manifest-schema-unsupported, BLAKE3 mismatch, missing file, hot-collision-same-bytes, hot-collision-different-bytes, tx_id collision.
- **Export** (3): verify-failed-carries-Stage-13.5-tag, valid export contributes to summary, unreadable yields **warning** (Finding 1).
- **Cross-surface counters** (2): `hot_export_overlaps` counter, no per-item info findings (Finding 4).
- **Schema + serde** (3): `report_schema_version_is_one`, finding-kind serde round-trip (24 variants), severity serde round-trip.
- **Findings-by-severity** (1): summary counts match emitted findings.
- **Boundary** (1): unreadable required `--anchor-registry-dir` returns `EvidenceAnchorError::Io`.
- **Helpers** included for archive scan via Stage 13.7 apply.

In `crates/omni-node/src/evidence_anchor_cli.rs::stage_13_8_cli_tests` (10 CLI tests):

- `--json-out` parent boundary (4 cases — present, missing, CWD, absent) — pins Finding 6.
- Repeatable `--archive-dir` and `--export-dir` flags.
- `--stale-threshold-secs` parses as `u64`.
- Schema-version constant re-export.
- Event-line format pin: no `reason=` substring on findings.

Exhaustive `every_variant_has_a_stable_tag` mapper test in `mod.rs` is **unchanged** — Stage 13.8 introduces zero new `EvidenceAnchorError` variants and zero modifications to `crates/omni-zkml/src/error.rs`.

## Why local-only / forward outlook

- **Why local-only**: a consistency report inspects ON-DISK state across hot / archive / export surfaces. The chain is the source of truth for status; Stage 13.8 doesn't ask "is this anchor finalized on chain?" — it asks "is my local state internally consistent?" Pulling chain RPCs into this stage would mix two operator concerns.
- **Why last local stage**: this completes the local-side preflight surface. Stages 13.0–13.7 added the building blocks; 13.8 reads them. Stage 13.9 — comprehensive chain read / reconcile support — needs a clean local state to reason about, and `report-integrity-evidence-anchor-consistency` is how operators confirm that.

**Stage 13.9 (forward outlook)**: comprehensive chain read / reconcile support. Operator-driven, chain-facing. Cross-checks hot registry tx_ids against the chain's recorded anchors, drives reconcile sweeps over backlogs, and bridges archived records back into reconcile flows when needed. Stage 13.8's report is the recommended preflight before any 13.9 invocation.

## Out of scope (Stage 13.9+ candidates)

- No chain interaction.
- No `--include-ok` flag (deferred per Q4).
- No `--strict-exports` flag.
- No mutation helpers.
- No multi-host correlation.
- No JSON report signing (operators wrap with Stage 12.25 separately if desired).
- No interactive paginated review / HTML / Markdown rendering.
