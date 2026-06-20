# Stage 13.10 — operator acceptance engineering doc

**Status:** delivered as acceptance + docs only. Zero production code changes. The 13.x integrity-evidence-anchor track closes with Stage 13.9; Stage 13.10 is the release-hardening checkpoint.

## Scope

Stage 13.10 is **not a new capability stage.** Goal: validate the full 13.x operator workflow end-to-end, tighten docs and tests around real CLI behavior, and pin operator-acceptance scenarios as a regression net for future maintenance work.

### Scope constraints (carried verbatim from the APPROVE'd plan)

- No Stage 13.0 wire / schema / domain / canonical-bytes / signing changes.
- No new SUM Chain RPCs.
- No private chain repo or `sumchain-primitives` dependency.
- No submit/retry behavior changes unless an acceptance bug proves a regression.
- Zero new `EvidenceAnchorError` variants. Zero new `reason=` tag strings.
- Tests remain hermetic. No live-chain CI tests.
- No new operator features.

### Out of scope (re-stated)

- New chain RPCs.
- Archive pruning / compression / encryption.
- Import of artifact bytes or signed reports.
- Mutation from by-tuple lookup (operator decides what to do; Stage 13.x does not auto-repair).
- Live-chain acceptance tests.
- Fixing pre-existing clippy warnings (deferred to a separate housekeeping PR; inventory below).

### Halt-rule check

| Signal | State at delivery |
| --- | --- |
| Stages 13.0–13.9 merged on `main` | ✅ via PR #55 (`2e073b6`) |
| CI green on Stage 13.9 merge | ✅ 11/11 checks passed |
| Open incidents touching the 13.x surface | none |
| New chain contract changes pending | none |
| Wire / schema / domain version pressure | none — `INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION` stable |
| Pre-existing clippy warnings on 13.x crates | inventoried; none introduced by 13.0–13.9 |

Halt rule satisfied. Stage 13.10 proceeded as planned.

## End-to-end workflow matrix

Each row: stage → CLI subcommand → acceptance signal asserted by the umbrella suite. Subcommand names match [`crates/omni-node/src/evidence_anchor_cli.rs`](../crates/omni-node/src/evidence_anchor_cli.rs).

| Stage | Subcommand | Acceptance signal |
| --- | --- | --- |
| 13.0 | `submit-integrity-evidence-anchor` (submit feature) | local record landed with `Submitted` status + `tx_id` derived from canonical digest |
| 13.0 | `query-integrity-evidence-anchor` | record fields surface in event line |
| 13.0 | `verify-integrity-evidence-anchor` / `…-file` | signature verifies against canonical digest |
| 13.3 | `summary-integrity-evidence-anchors` | counts per status; oldest-submitted-age line |
| 13.3 | `watch-integrity-evidence-anchors` | event names stable (smoke only — no long-poll) |
| 13.2 / 13.9 | `reconcile-integrity-evidence-anchor` | batch path used when supported; chunks at 100; per-item containment |
| 13.4 | `cleanup-integrity-evidence-anchor-registry` | dry-run produces plan; apply quarantines; restore round-trips |
| 13.5 | `export-integrity-evidence-anchors` + `verify-integrity-evidence-anchor-export` | export manifest verifies; tamper detected |
| 13.6 | `import-integrity-evidence-anchor-export` | dry-run shows plan; apply idempotent on rerun |
| 13.7 | `archive-integrity-evidence-anchors` | dry-run plan; apply; restore-from-manifest |
| 13.8 | `report-integrity-evidence-anchor-consistency` | local-only findings; no chain RPC |
| 13.9 | `lookup-integrity-evidence-anchor-by-tuple` | NotFound emits informational event without `reason=`; Found surfaces canonical_tx_hash |

## Operator acceptance scenarios

Nine scenarios; one `#[test]` each (no mega-test). Implementation in [`crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs`](../crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs).

| # | Scenario | Test function |
| --- | --- | --- |
| 1 | Happy-path full lifecycle: submit → reconcile (batch) → summary → consistency-derived health → export → verify → import to second registry → archive → restore. Single test exercising every stage. | `scenario_1_happy_path_full_lifecycle_seed_reconcile_summary_export_import_archive_restore` |
| 2 | Corrupted hot registry record: Stage 13.8 flags `HotRecordMalformed`; Stage 13.4 plans `QuarantineMalformedRecord`; apply moves to quarantine; restore round-trips byte-equal. | `scenario_2_corrupted_hot_record_consistency_then_cleanup_quarantine_then_restore_roundtrip` |
| 3 | Stale / open submitted record: Stage 13.3 stale-list surfaces it; Stage 13.9 reconcile against chain reporting `Unknown` is observation-only (Q7 lock — no downgrade). | `scenario_3_stale_open_submitted_summary_then_reconcile_observation_only_for_unknown_chain_state` |
| 4 | Orphan `tx_index.json` entry: Stage 13.8 flags `HotTxIndexOrphan`; Stage 13.4 plans `RemoveOrphanTxIndexEntry`. | `scenario_4_orphan_tx_index_entry_consistency_flags_and_cleanup_plans_removal` |
| 5 | Export tamper detected: byte-flip inside an exported anchor record → Stage 13.5 verify refuses with a stable reason tag; Stage 13.6 import refuses identically. | `scenario_5_export_tamper_detected_by_verify_before_import_is_attempted` |
| 6 | Import idempotent rerun: byte-equal target → second run does not mutate hot bytes; if policy refuses, refusal surfaces a stable tag. | `scenario_6_import_idempotent_rerun_against_already_imported_target` |
| 7 | **Archive partial-state recovery (black-box per APPROVE scope lock).** Seed the operator-observable state directly: durable archive manifest exists with byte-equal record bytes, one hot record is missing. `restore-anchor-archive` is idempotent: missing record materialised from manifest, tx_index re-added, byte-equal record left untouched. Second restore is a no-op. **No internal phase hooks exercised.** | `scenario_7_archive_partial_state_restore_from_manifest_after_one_hot_record_disappears` |
| 8 | Consistency report before chain reconcile: Stage 13.8 makes zero chain RPCs; orphans surface; reconcile afterward uses the batch path; orphans remain visible (reconcile does not touch them). | `scenario_8_consistency_report_runs_without_chain_calls_before_reconcile` |
| 9 | By-tuple canonical tx mismatch surfaced read-only: chain returns canonical_tx_hash ≠ local tx_id; `tx_id_matches_canonical=false` surfaces; **registry is byte-identical before/after.** | `scenario_9_by_tuple_canonical_tx_mismatch_surfaced_without_mutating_registry` |

Plus three structural / smoke tests:
- `parse_event_line_extracts_known_keys_and_skips_malformed_tokens` — pins the structured event-line parser used by scenario assertions.
- `scenario_9b_by_tuple_not_found_event_line_has_no_reason_key` — companion structural pin on the Stage 13.9 `_no_chain_anchor` event-line shape (no `reason=` key, exact event-name string).
- `scenario_smoke_error_taxonomy_stable_for_acceptance_paths` — sanity check that error paths used by acceptance scenarios all produce stable `reason=` tags.

**Total: 12 tests, all hermetic.**

## Test strategy

### Q1: single umbrella file vs per-stage modules?

**Single umbrella file** — `crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs`. Per-stage modules already exist ([`evidence_anchor_cleanup.rs`](../crates/omni-zkml/tests/evidence_anchor_cleanup.rs), [`evidence_anchor_export.rs`](../crates/omni-zkml/tests/evidence_anchor_export.rs), …). The Stage 13.10 acceptance value is in the **stitching** — scenarios where one stage's output is another's input. That signal would be lost if scenarios were scattered. Each scenario is still its own `#[test]` for failure isolation.

### Q2: transcript pins — exact strings or structured parsing?

**Structured parsing + small exact-string anchor pins.** The umbrella file ships a tiny `parse_event_line(&str) -> BTreeMap<String, String>` helper. Scenario assertions use `m["event"] == "…"` and `assert!(!m.contains_key("reason"))` rather than full transcript snapshots. Exact-string anchor pins live on event names already pinned by Stage 13.9 CLI tests (e.g. `integrity_evidence_anchor_tuple_lookup_no_chain_anchor`). This avoids brittle full-line snapshots while still pinning regressions in event-name or key-name drift.

### Q3: pre-existing clippy in Stage 13.10?

**No.** Quarantined to a separate housekeeping PR. Inventory (none introduced by 13.0–13.9):

| Crate | Surface | Lints |
| --- | --- | --- |
| `omni-zkml` | lib | doc-list-indent (×4), explicit `.into_iter()` in fn arg, `sort_by_key`, redundant closure (7 total) |
| `omni-zkml` | tests | `assert_eq!` with bool literal in [`proof.rs`](../crates/omni-zkml/src/proof.rs) (×3), `into_iter` calls |
| `omni-sumchain` | tests | unused `JsonRpcTransport` import in [`anchor_submit_feature_gate.rs:8`](../crates/omni-sumchain/tests/anchor_submit_feature_gate.rs#L8) (since Stage 13.2) |
| `omni-node` | bins | items after a test module; returning the result of a `let` binding (×3); too many arguments (×2); literal with empty format string; boolean expression can be simplified |

Proposed follow-up: **"Stage 13.x housekeeping: pre-existing clippy cleanup (no behavior change)"**.

### Q4: minimum acceptance matrix without a fragile mega-test?

**Nine scenarios → 12 total tests.** Scenario 1 is the only multi-stage test (full lifecycle). Scenarios 2–9 are 2-stage at most. Plus 3 structural / smoke tests. Each scenario is its own `#[test]`, so failure-isolation is preserved.

## Scope lock on scenario 7 (carried verbatim from APPROVE)

> Scenario 7 is allowed, but keep it black-box at the archive/restore boundary. Do not add production hooks or refactor archive internals just to force a Phase-2 failure. If the existing public helpers can create the partial state, test it; otherwise model the operator-observable state directly: durable archive manifest exists, one or more hot records are missing or duplicated, then restore from manifest recovers idempotently.

**How scenario 7 was implemented:** the test calls `apply_anchor_archive` normally (no internal hooks), then **manually rewrites a single record back into the hot registry** to simulate the post-Phase-2 partial state (an operator copy-paste, or a half-recovered Phase-2 failure). It then asserts `restore_anchor_archive` is idempotent. Zero internal phase hooks exercised.

## Verification matrix at delivery

| Suite | Tests | Pass | Notes |
| --- | --- | --- | --- |
| [`crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs`](../crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs) | 12 | 12 | hermetic; all scenarios green |
| All prior Stage 13.0–13.9 test suites | (unchanged) | green | no regressions |
| `git diff main..HEAD -- crates/omni-zkml/src/error.rs` | 0 LOC | ✅ | schema invariance |
| `git diff main..HEAD -- crates/omni-zkml/src/evidence_anchor/wire.rs` | 0 LOC | ✅ | wire invariance |
| `git diff main..HEAD -- crates/omni-zkml/src/` (excluding tests) | 0 LOC | ✅ | **zero production code changes** |
| Pre-existing clippy warning count | unchanged | ✅ | no new warnings |
| Submit-path semantic changes | none | ✅ | scope held |
| New chain RPCs | none | ✅ | scope held |
| New operator features | none | ✅ | scope held |

## Files delivered

- New: `crates/omni-zkml/tests/evidence_anchor_stage_13_10_acceptance.rs` — 12 acceptance tests (~700 LOC).
- New: `docs/stage13.10-acceptance.md` — this engineering doc.
- Modified: `docs/operator-runbook.md` — Stage 13.10 section added with full lifecycle recipe + recovery playbook.

**No production crate sources modified.**

## Future outlook

Stage 13.10 is the release-hardening checkpoint for the 13.x track. Future evolution typically falls into:

1. **Operator UX over the existing surface** — e.g. opt-in `--apply-canonical-tx-id` repair flag on by-tuple CLI, periodic schedules, dashboard surfacing.
2. **Reorg-aware downgrade semantics** — would require a chain-side reversion model; out of scope for the 13.x track.
3. **Multi-chain / multi-instance** — multiple SUM Chain endpoints, replicated registries, etc.

None are in scope here. The 13.x integrity-evidence-anchor track closes with this acceptance round.
