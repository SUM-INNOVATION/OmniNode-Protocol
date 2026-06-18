# Stage 13.5 — local-only integrity-evidence-anchor export / verify

## Stage scope

Stage 13.5 adds a portable handoff form for anchor records — a JSON manifest plus a small subtree of copied bytes — and a pure-read verify command that re-checks every byte against the manifest. The flow is fully local: no SUM Chain RPCs, no `omni-sumchain` types pulled in for either subcommand. Stage 13.0 wire / domain / canonical-bytes / signing is **read only**; not a byte is re-signed.

## What Stage 13.5 ships

- **Library helpers** in `omni-zkml::evidence_anchor::export`:
  - `plan_anchor_export(registry, opts) -> AnchorExportPlan`
  - `apply_anchor_export(registry, opts) -> AnchorExportReport`
  - `verify_anchor_export(opts) -> AnchorExportVerifyReport`
- **Closed entry-kind taxonomy** (`AnchorExportEntryKind`): `anchor_record` | `artifact_bytes` | `signed_chain_report`.
- **Persisted JSON form** with schema constant (`EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION = 1`).
- **Six new `EvidenceAnchorError` variants** + six new `evidence_anchor_reason_tag` mapper arms. Three new tag strings (`unsupported_export_manifest_schema_version`, `export_manifest_hash_mismatch`, `export_invalid_path`) + three new tag strings (`export_blake3_mismatch`, `export_entry_metadata_mismatch`, `export_strict_mode_artifact_bytes_missing`).
- **Reused tag strings** (no surface growth): `anchor_not_found` (selector miss — corrected from `record_not_found` per REJECT Finding 3), `malformed_json`, `io`, `unsupported_anchor_schema_version`, `submitter_signature_invalid`.

## What Stage 13.5 does NOT ship

- ❌ No SUM Chain RPC interaction. Fully local.
- ❌ No `AnchorRecord` shape change. Records copied verbatim.
- ❌ No wire / domain / canonical-bytes / signing changes.
- ❌ No registry mutation. Both subcommands open the registry read-only.
- ❌ No `--force` on `--export-out` (Q8 — operator picks a fresh path).
- ❌ No "export everything" default (Q4 — at least one selector required).
- ❌ No Stage 12.25 signed-chain-report own-signature verification (Q7 — out of scope; operator runs Stage 12.25 verify separately).
- ❌ No wrapper-signer binding from artifact bytes alone (REJECT Finding 4 — that's a Stage 12.25 + chain-report concern, not Stage 13.5).

## Locked decisions (Q&A)

- ✅ Q1 — 5 new export-side tag strings on the closed taxonomy, plus 1 strict-mode tag.
- ✅ Q2 — `--strict` enforcement happens at verify time, not at the clap layer.
- ✅ Q3 — `--include-artifact-bytes <PATH>:<HASH>` pair form. Export-time BLAKE3 must equal the claimed hash.
- ✅ Q4 — at least one of `--status`, `--tx-id`, `--artifact-hash-hex` is required. No accidental export-all.
- ✅ Q5 — `export_id` derivation: `BLAKE3(record_set_hash \|\| "||" \|\| created_at_utc)[..16]`. Timestamp included; tests inject `created_at_utc` for determinism.
- ✅ Q6 — `anchor_registry_dir` is **not** in the manifest (REJECT Finding 2 — keeps the handoff path-minimal). Operator provenance lives in `label` / `notes`.
- ✅ Q7 — Stage 12.25 signed-chain-report own-signature verification is out of scope.
- ✅ Q8 — `--export-out` no-clobber. No `--force`.
- ✅ Q9 — paired artifact-hash binding refuses with the existing `export_entry_metadata_mismatch` tag (Q9 fold, not a separate `export_artifact_hash_binding_mismatch` string).
- ✅ REJECT-fix Finding 1 — `--strict` is a verification-time refusal routed through `export_strict_mode_artifact_bytes_missing`, not a clap-level usage error.
- ✅ REJECT-fix Finding 2 — `anchor_registry_dir` dropped from manifest.
- ✅ REJECT-fix Finding 3 — selector miss reuses `anchor_not_found` (not a new `record_not_found`).
- ✅ REJECT-fix Finding 4 — Stage 13.5 verify reuses `verify_anchor_tx_data` for submitter-signature checks, but **does not** call `verify_anchor_file_against_artifact_bytes` — passing `record.tx_data.digest.signer_pubkey` back as the expected wrapper signer would make the signer check tautological. The private helper `verify_anchor_record_with_artifact_bytes` runs only the honest checks (artifact-hash binding + `verify_anchor_tx_data`).
- ✅ REJECT-fix Finding 5 — duplicate `relative_path` entries (e.g. two `--include-signed-chain-report` paths with the same basename) would collide on disk during apply yet record two different `blake3_hex`s in the manifest, producing an export that fails its own verifier with `export_blake3_mismatch`. Plan-side dedup catches it at plan time; verify-side dedup catches hand-edited manifests before any FS read. Both routed through the existing `ExportInvalidPath` variant with `reason: "duplicate relative_path"` (`reason=export_invalid_path`).

## CLI

### `export-integrity-evidence-anchors`

```
omni-node operator evidence-anchor export-integrity-evidence-anchors
  --anchor-registry-dir <DIR>                                       (required)
  --export-out          <DIR>                                       (required; no-clobber)
  [--status                          <SUBMITTED|INCLUDED|FINALIZED|FAILED>]  (repeatable)
  [--tx-id                           <hex>]                                  (repeatable)
  [--artifact-hash-hex               <64-lower-hex>]                         (repeatable)
  [--include-artifact-bytes          <FILE_PATH>:<ARTIFACT_HASH_HEX>]        (repeatable)
  [--include-signed-chain-report     <FILE_PATH>]                            (repeatable)
  [--label <STRING>] [--notes <STRING>]                                      (provenance)
```

#### Clap-level mutex / required-with rules

| Rule | Refusal message |
| --- | --- |
| At least one of `--status`, `--tx-id`, `--artifact-hash-hex` is required | `at least one of --status, --tx-id, --artifact-hash-hex is required` |
| `--status` value must be one of `submitted` / `included` / `finalized` / `failed` (case-insensitive) | `--status must be one of submitted \| included \| finalized \| failed` |
| `--artifact-hash-hex` must be exactly 64 lower-hex chars | `--artifact-hash-hex must be exactly 64 lowercase hex characters` |
| `--include-artifact-bytes` must be `<PATH>:<64-lower-hex>` | `--include-artifact-bytes expects <path>:<64-lower-hex>` |

Pinned by `mutex_refuses_*` tests; messages pinned by `violation_messages_are_stable_strings`. These refuse via `bail!`, NOT via the closed `reason=…` taxonomy.

### `verify-integrity-evidence-anchor-export`

```
omni-node operator evidence-anchor verify-integrity-evidence-anchor-export
  --export-dir <DIR>                                  (required)
  [--strict]                                          (require artifact_bytes for every anchor_record)
```

`--strict` enforcement runs at verify time, not at clap. A miss surfaces as `event=integrity_evidence_anchor_verify_export_failed reason=export_strict_mode_artifact_bytes_missing`.

## Manifest schema v1 (locked)

```json
{
  "schema_version": 1,
  "export_id": "<16-lower-hex>",
  "created_at_utc": "2026-06-17T22:00:00Z",
  "label": "",
  "notes": "",
  "entries": [
    {
      "kind": "anchor_record",
      "relative_path": "anchors/<64-lower-hex>.json",
      "blake3_hex": "<64-lower-hex>",
      "bytes": 1234,
      "artifact_hash_hex": "<64-lower-hex>",
      "tx_id": "<string>",
      "status": "submitted | included | finalized | failed"
    }
  ],
  "export_manifest_hash": "<64-lower-hex>"
}
```

Optional fields (`artifact_hash_hex`, `tx_id`, `status`, `source_basename`) populate per-kind:

- `anchor_record` → all three of `artifact_hash_hex`, `tx_id`, `status`.
- `artifact_bytes` → `artifact_hash_hex` only.
- `signed_chain_report` → `source_basename` only.

`export_id = BLAKE3(record_set_hash || "||" || created_at_utc)[..16]`.

`record_set_hash = BLAKE3(canonical JSON of selected_records.map(r => { artifact_hash_hex, tx_id, status }) sorted by artifact_hash_hex ascending)`.

`export_manifest_hash = BLAKE3(canonical JSON of manifest with this field blanked)`.

## Exact file layout

```
<export-out>/
├── evidence_anchor_export_manifest.json    # written LAST (durability fence)
├── anchors/
│   ├── <artifact_hash_hex>.json            # copied verbatim from <registry>/<artifact_hash_hex>.json
│   └── ...
├── artifacts/                              # only if --include-artifact-bytes
│   ├── <artifact_hash_hex>                 # raw bytes, no extension
│   └── ...
└── signed_chain_reports/                   # only if --include-signed-chain-report
    └── <source_basename>                   # copied verbatim
```

`tx_index.json` is **NOT** exported — Stage 13.5 is a record-subset export, not a registry mirror.

Atomic temp+rename per file. Manifest written **after** all data files land.

## Selection rules

Selectors combine AND across kinds, OR within a kind.

- `--status` (repeatable): record's `status` kind must match one of the supplied values.
- `--tx-id` (repeatable): record's `receipt.tx_id` must equal one of the supplied values.
- `--artifact-hash-hex` (repeatable): record's `artifact_hash_hex` must equal one of the supplied values.

A `--tx-id` or `--artifact-hash-hex` selector pointing at a non-existent record refuses with **reused** `anchor_not_found` (Stage 13.0 tag; corrected from v1 plan's `record_not_found` per Finding 3).

If no selectors are supplied: clap-level usage error (no accidental "export everything").

## Verify preflight ordering (fixed)

Run in order; first failure stops with `reason=<tag>`.

| # | Check | Refusal | Tag | Source |
| - | --- | --- | --- | --- |
| 1 | parse manifest as JSON | existing `malformed_json` | `malformed_json` | Reused |
| 2 | `manifest.schema_version == 1` | `ExportManifestSchemaUnsupported { got, expected }` | `unsupported_export_manifest_schema_version` | **NEW** |
| 3 | recomputed `export_manifest_hash` (canonical JSON with field blanked) == declared | `ExportManifestHashMismatch { computed, expected }` | `export_manifest_hash_mismatch` | **NEW** |
| 4 | per-entry `relative_path` shape (universal + per-kind) | `ExportInvalidPath { entry_kind, relative_path, reason }` | `export_invalid_path` | **NEW** |
| 4b | duplicate `relative_path` defense — refuse if any two entries share a destination, before any FS read | `ExportInvalidPath { …, reason: "duplicate relative_path" }` | `export_invalid_path` | (Reused — NEW above) |
| 5 | per-entry file exists at `<export-dir>/<relative_path>` | existing `io` | `io` | Reused |
| 6 | per-entry BLAKE3(bytes) == declared `blake3_hex` AND length == declared `bytes` | `ExportBlake3Mismatch { relative_path, computed, expected }` | `export_blake3_mismatch` | **NEW** |
| 7a | `anchor_record` entries — parse bytes as `AnchorRecord` | existing `malformed_json` | `malformed_json` | Reused |
| 7b | `anchor_record` entries — `verify_anchor_tx_data(&record.tx_data)` (Stage 13.0 verifier, verbatim) | `unsupported_anchor_schema_version` \| `submitter_signature_invalid` | (both) | Reused |
| 8 | `anchor_record` entries — `record.artifact_hash_hex == entry.artifact_hash_hex` AND `record.receipt.tx_id == entry.tx_id` AND `record.status.kind == entry.status` | `ExportEntryMetadataMismatch { relative_path, field, computed, manifest }` | `export_entry_metadata_mismatch` | **NEW** |
| 9 | `artifact_bytes` entries — declared `artifact_hash_hex == blake3_hex` (an artifact_bytes file's BLAKE3 IS its claimed hash; step 6 already proved BLAKE3) | `ExportEntryMetadataMismatch { …, field: "artifact_hash_hex", … }` | `export_entry_metadata_mismatch` | (Reused — NEW above) |
| 10 | paired `(anchor_record, artifact_bytes)` for same `artifact_hash_hex` — `hex_lower(record.tx_data.digest.artifact_hash) == bytes_entry.artifact_hash_hex`. Honest check via `verify_anchor_record_with_artifact_bytes`. **NOT** `verify_anchor_file_against_artifact_bytes` (REJECT Finding 4). | `ExportEntryMetadataMismatch { …, field: "artifact_hash_binding", … }` | `export_entry_metadata_mismatch` | (Reused — NEW above) |
| 11 | `signed_chain_report` entries — BLAKE3 only (step 6). **Stage 12.25 own-signature verification out of scope** (Q7). | — | — | — |
| 12 | `--strict` — every `anchor_record` entry must have a paired `artifact_bytes` for the same `artifact_hash_hex` | `ExportStrictModeArtifactBytesMissing { anchor_record_relative_path, artifact_hash_hex }` | `export_strict_mode_artifact_bytes_missing` | **NEW** |

## What Stage 13.5 verify proves

- The export tree's bytes match the manifest (steps 1–6).
- Each anchor record's submitter signature is valid under its embedded `digest.signer_pubkey`, and its wire format is the locked v1 (step 7b).
- Per-record metadata on the manifest matches the record's own fields (step 8).
- For paired `artifact_bytes`: the bytes' BLAKE3 equals `record.tx_data.digest.artifact_hash` — the **artifact-hash binding**, not wrapper-signer binding (step 10).

## What Stage 13.5 verify does NOT prove

- The Stage 12.25 wrapper signer. That binding requires the wrapper's own signature, which lives in a signed-chain-report. Stage 13.5 verifies BLAKE3 of any included signed-chain-report but does **not** re-verify its own signature (Q7). Operator runs Stage 12.25 verify on the signed-chain-report separately.
- That the anchor is finalized on chain. That requires a Stage 13.2 / 13.3 chain RPC call. Out of scope.

## Closed reason-tag taxonomy delta

Stage 13.5 adds **six new `EvidenceAnchorError` variants** and **six new tag strings**:

| Variant | Tag string |
| --- | --- |
| `ExportManifestSchemaUnsupported { got, expected }` | `unsupported_export_manifest_schema_version` |
| `ExportManifestHashMismatch { computed, expected }` | `export_manifest_hash_mismatch` |
| `ExportInvalidPath { entry_kind, relative_path, reason }` | `export_invalid_path` |
| `ExportBlake3Mismatch { relative_path, computed, expected }` | `export_blake3_mismatch` |
| `ExportEntryMetadataMismatch { relative_path, field, computed, manifest }` | `export_entry_metadata_mismatch` |
| `ExportStrictModeArtifactBytesMissing { anchor_record_relative_path, artifact_hash_hex }` | `export_strict_mode_artifact_bytes_missing` |

Reused tags (no surface growth): `anchor_not_found`, `malformed_json`, `io`, `submitter_signature_invalid`, `unsupported_anchor_schema_version`.

The exhaustive `every_variant_has_a_stable_tag` test extends to cover all six — adding a variant without a mapper arm is a compile error.

## Informational event taxonomy

| Event | When | Fields |
| --- | --- | --- |
| `event=integrity_evidence_anchor_export_started` | Export startup | `anchor_registry_dir=`, `export_out=` |
| `event=integrity_evidence_anchor_export_ok` | Export complete | `export_id=`, `anchors_written=`, `artifact_bytes_written=`, `signed_chain_reports_written=`, `manifest_relative_path=` |
| `event=integrity_evidence_anchor_export_failed` | Export refusal | `reason=<closed-set tag>`, `detail=` |
| `event=integrity_evidence_anchor_verify_export_started` | Verify startup | `export_dir=`, `strict=` |
| `event=integrity_evidence_anchor_verify_export_ok` | Verify complete | `export_id=`, `entries_verified=`, `anchor_records_verified=`, `artifact_bytes_verified=`, `signed_chain_reports_verified=`, `pairings_artifact_hash_bound=`, `strict=` |
| `event=integrity_evidence_anchor_verify_export_failed` | Verify refusal | `reason=<closed-set tag>`, `detail=` |

No new `cause=` keys. Refusals route through `reason=`.

## Test inventory

In `crates/omni-zkml/tests/evidence_anchor_export.rs` (43 library integration tests, hermetic):

- **Selection** (6): selector by status / tx_id / artifact_hash; intersection; selector-miss → `anchor_not_found`; no-clobber refusal.
- **Manifest invariants** (5): schema_version locked; manifest-hash byte-stable; portable (no `anchor_registry_dir`); `export_id` deterministic; entries sorted.
- **Layout** (2): anchors subdir, no `tx_index.json`.
- **Optional include** (3): artifact_bytes + signed_chain_report copy; export refuses on claimed-hash mismatch.
- **Verify happy paths** (4): records-only, paired binding, signed-chain-report BLAKE3, all three kinds.
- **Verify refusals (reused tags)** (4): tampered signature, malformed JSON, missing file, manifest parse failure.
- **Verify refusals (new tags)** (9): schema-version drift, manifest-hash drift, BLAKE3 drift, absolute path, parent traversal, per-kind shape, artifact_hash / tx_id / status metadata mismatch.
- **Ordering pin** (1): schema-version refusal fires before manifest-hash refusal.
- **Strict mode** (2): refusal + happy path.
- **Honesty-of-binding pins** (2): paired binding proves artifact-hash, not wrapper-signer; bytes-diverging-from-anchor refuses with metadata mismatch.
- **Duplicate relative_path defense** (4): export refuses on duplicate signed-chain-report basenames; export refuses on duplicate `--include-artifact-bytes` target hashes; verify refuses on hand-edited manifest with duplicate signed_chain_report path; verify refuses on hand-edited manifest with duplicate anchor_record path.

In `crates/omni-node/src/evidence_anchor_cli.rs::stage_13_5_cli_tests` (20 CLI tests):

- 4 mutex-rule refusal pins (no-selector, unknown status, malformed hash length, malformed hash case).
- 3 mutex-rule pass pins (status / tx_id / artifact_hash alone).
- 1 status case-insensitive pin (all four kinds).
- 4 pair-form parser pins (happy path, missing colon, short hash, uppercase hash).
- 1 violation-message stability pin.
- 6 reason-tag mapper routing pins (one per new Stage 13.5 variant).
- 1 selector-miss tag pin (`anchor_not_found` remains the correct tag).

All hermetic. No network.

## Out of scope (Stage 13.6+ candidates)

- Verifying the Stage 12.25 signed-chain-report's own signature inside Stage 13.5 verify (Q7).
- Wrapper-signer binding (would require including + verifying a signed-chain-report — see above).
- Cross-checking exported `tx_id`s against live SUM Chain state (would need chain RPC).
- Importing exports back into a registry (would mutate the registry).
- Multi-host export sync / scheduled archival.
- Encrypted transport of the export tree (operator's transport concern).

> **Post-Stage-13.6 note:** Stage 13.6 lifts the "importing exports back into a registry" item by adding `import-integrity-evidence-anchor-export` — a verify-first, default-dry-run, byte-preserve restore path into a target local registry. Stage 13.5's export and verify surfaces are **unchanged**; Stage 13.6 ships its own one-tag delta (`import_target_exists`) on a distinct concept (import-side refusals). See [`docs/stage13.6-anchor-import.md`](stage13.6-anchor-import.md).
