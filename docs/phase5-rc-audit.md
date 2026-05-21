# Phase 5 — Release Candidate Audit (2026-05-21)

Point-in-time release-candidate audit for Phase 5 after Stage 10a. This document is
**immutable** once landed — future audits should be new dated files (`phase5-rc-audit-YYYY-MM-DD.md`),
not edits to this one. The convention mirrors [`docs/mainnet-smoke-audit.md`](mainnet-smoke-audit.md).

This audit is a verification pass, not a feature stage. The only repo change that
accompanies it is a small README status-table correction described in §8 below
("Inconsistencies found and corrected").

---

## 1. Status

| Item | Value |
|---|---|
| HEAD commit | `fcde900690abb8ee3a2926dcdcbcaa919fe3b61b` |
| HEAD subject | `Merge pull request #4 from SUM-INNOVATION/stage10a-operator-observability` |
| Branch under audit | `main` |
| Working tree | clean (no uncommitted changes) |
| Latest CI run on `main` | run `26202634481`, conclusion `success`, headSha `fcde900690abb8ee3a2926dcdcbcaa919fe3b61b`, created `2026-05-21T02:54:09Z` |
| CI ↔ HEAD match | **yes** — the green run is against the exact audit commit |

CI is green for the SHA being audited. (Per the audit plan: if the latest run had
been for a different SHA, this section would say so explicitly and would not call
CI green for `fcde900`.)

---

## 2. Stage rows present in [README.md](../README.md) status table

After the §8 correction (Stage 5.2 + 5.3 row addition), the Phase 5 status table
lists every implemented sub-stage with its own milestone section:

| Stage | Row label (truncated) | Status |
|---|---|---|
| 1 | SNIP V2 types + CLI adapter | Complete |
| 2 | SNIP-backed model artifacts | Complete |
| 3 | Proof artifact flow | Complete |
| 4 | Local verifier attestation envelope | Complete |
| 5 | Chain client abstraction + offline attestation registry | Complete |
| 5.2 | Client-local staleness / retry policy | Complete |
| 5.3 | End-to-end attestation orchestration | Complete |
| 6 | Chain wire fixture & signing-spec deliverables | Complete |
| 7a | SUM Chain adapter (read/query) | Complete |
| 7b | SUM Chain submit path | Complete |
| 8a | Operator activation / smoke / lifecycle loop | Complete |
| 8b | Mainnet activation smoke hardening | Complete |
| 8c | Post-mainnet-smoke hardening | Complete |
| 9a | Production runbook + build decoupling | Complete |
| 9b | CI matrix (superseded by 9c) | Superseded by 9c |
| 9c | Public SUM Chain crates + CI gate restoration | Complete |
| 9c.1 | Chain-produced signed-transaction fixture | Complete |
| 10a | Operator observability + release readiness | Complete |
| 10b | Release artifact workflow + signing | Planned |
| 8+ | zkML proof generation + tokenomics | Planned |

Stage 5.1 was a chain-abstraction alignment fix (commit `2fee875`, no dedicated
milestone section); it intentionally has no status-table row.

---

## 3. Test matrix results

### 3a. Default workspace (no `--features submit`)

Command: `cargo test -p omni-net -p omni-pipeline -p omni-store -p omni-types
-p omni-zkml -p omni-sumchain -p omni-node`.

| Crate / target | Tests passed |
|---|---|
| `omni-net` (unit) | 8 |
| `omni-node` (unit) | 43 |
| `omni-pipeline` (unit) | 27 |
| `omni-store` (unit) | 60 |
| `omni-types` (unit) | 29 |
| `omni-zkml` (unit) | 142 |
| `omni-zkml` `tests/chain_attestation_vectors.rs` | 1 |
| `omni-sumchain` `tests/no_submit_feature.rs` | 1 |
| `omni-sumchain` `tests/stage6_wire_parity.rs` | 1 |
| `omni-sumchain` `tests/unit_dto.rs` | 14 |
| `omni-sumchain` `tests/unit_rpc_envelope.rs` | 16 |
| `omni-sumchain` `tests/unit_status_mapping.rs` | 10 |
| `omni-sumchain` `tests/live_local_mirror.rs` | 0 passed, 4 ignored (env-gated; auto-skip without `OMNINODE_SUMCHAIN_RPC_URL`) |

`tests/chain_produced_fixture.rs`, `tests/parity_vendored_primitives.rs`, and
`tests/unit_submit_construction.rs` correctly report `0 passed` under default
features because they're file-level `#![cfg(feature = "submit")]`-gated.

### 3b. `omni-sumchain` with `--features submit`

Command: `cargo test -p omni-sumchain --features submit`.

| Target | Tests passed |
|---|---|
| `tests/chain_produced_fixture.rs` (Stage 9c.1) | 5 |
| `tests/parity_vendored_primitives.rs` (Stage 7b + 9c additions) | 5 |
| `tests/stage6_wire_parity.rs` | 1 |
| `tests/unit_dto.rs` | 14 |
| `tests/unit_rpc_envelope.rs` | 17 |
| `tests/unit_status_mapping.rs` | 10 |
| `tests/unit_submit_construction.rs` | 15 |
| `tests/live_local_mirror.rs` | 0 passed, 5 ignored (env-gated) |
| Subtotal | **67 pass**, 5 ignored (live, env-gated) |

### 3c. `omni-node` with `--features submit`

Command: `cargo test -p omni-node --features submit`.

| Target | Tests passed |
|---|---|
| `omni-node` (unit, including cfg-gated `smoke` / retry-loop suites) | **56** |

---

## 4. Dependency-tree assertions

| Assertion | Result |
|---|---|
| `cargo tree -p omni-node \| grep -E 'sumchain-(crypto\|primitives)'` | empty — Stage 9a feature gating intact; default builds do not pull the chain crates |
| `cargo tree -p omni-node --features submit \| grep -E 'sumchain-(crypto\|primitives)'` | `sumchain-crypto v0.1.0` + `sumchain-primitives v0.1.0` both present — Stage 9c public crates resolve from crates.io |

The corresponding CI hard-gate jobs (`default-tree-check`, `submit-tree-check`)
encoded these same assertions in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
and are green on the audit commit.

---

## 5. Stage 6 byte-stability

The Stage 6 chain-wire fixture and `chain_wire.rs` are the canonical byte-stable
anchor that every post-Stage-6 work item must preserve.

### 5a. Commit-history evidence (`git log --oneline -- <path>`)

```
crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json
  509f7fd  fix(phase5): chain-wire canonical bytes use bincode 1.3 (Stage 6 fix)
  c91eb5d  feat(phase5): chain wire fixture + signing-spec deliverables (Stage 6)

crates/omni-zkml/src/chain_wire.rs
  509f7fd  fix(phase5): chain-wire canonical bytes use bincode 1.3 (Stage 6 fix)
  c91eb5d  feat(phase5): chain wire fixture + signing-spec deliverables (Stage 6)
```

Both files have **only** Stage 6 commits in their history — no post-Stage-6
edits at any point. Every stage from 7a through 10a has preserved the contract.

### 5b. Diff-against-baseline evidence

`509f7fd` is the canonical Stage 6 byte-stability baseline (the post-bincode-1.3-fix
state of the fixture and `chain_wire.rs`). Diff against HEAD is the strict invariant
test:

```
$ git diff --stat 509f7fd..HEAD -- \
    crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json \
    crates/omni-zkml/src/chain_wire.rs
# (empty output)
```

Empty. Byte-stability is intact from Stage 6 baseline → Stage 10a HEAD.

The CI hard-gate `stage6-fixture-check` enforces this on every PR / push and is
green on the audit commit.

---

## 6. On-chain reference

The 2026-05-19 mainnet smoke proved end-to-end submit on chain:

- Tx hash: `0x3a9cbf85945136e55a3ab8bb04a09d406d52438d9c2fa1f77850a706a1c32a56`
- Included at block height: 6049201
- Status: Finalized
- Verifier address: `2mvPk4h883B7DrcZvwy7yWKXyGYHuVzGP`

Pinned in-repo by:

| Anchor | Path | Status |
|---|---|---|
| Mainnet smoke audit (immutable) | [`docs/mainnet-smoke-audit.md`](mainnet-smoke-audit.md) | exists; single commit in history (`18a125a` — Stage 8c PR that introduced it); no subsequent edits |
| Chain-produced signed-tx fixture | [`crates/omni-sumchain/tests/fixtures/chain_produced_signed_tx.json`](../crates/omni-sumchain/tests/fixtures/chain_produced_signed_tx.json) | exists (3,371 bytes); chain-team provenance preserved |
| Chain-produced fixture test (Stage 9c.1) | [`crates/omni-sumchain/tests/chain_produced_fixture.rs`](../crates/omni-sumchain/tests/chain_produced_fixture.rs) | exists; **5/5 pass** under `--features submit`, including `fixture_signed_tx_hash_matches_chain_produced_tx_hash` which pins `SignedTransaction::hash().to_hex() == fixture.tx_hash` against the authoritative on-chain bytes |

---

## 7. Operator surface (Stage 9a + 10a)

[`docs/operator-runbook.md`](operator-runbook.md) is the canonical operator
reference. The Stage 10a additions are all present (14 grep matches across the
named Stage 10a markers):

| Stage 10a deliverable | Location |
|---|---|
| `operator registry summary` subcommand | [`docs/operator-runbook.md §6`](operator-runbook.md) (docs) + [`crates/omni-node/src/operator.rs`](../crates/omni-node/src/operator.rs) (code; 9 hermetic tests) |
| Stable `event=` marker contract (`startup`, `chain_params`, `activation_state`, `registry_summary`) | [`docs/operator-runbook.md §7a`](operator-runbook.md) |
| Free-form Stage 9a markers (still in place) | [`docs/operator-runbook.md §7b`](operator-runbook.md) |
| Failure-triage matrix (11 concrete failure states) | [`docs/operator-runbook.md §11a`](operator-runbook.md) |
| Escalation packet | [`docs/operator-runbook.md §11b`](operator-runbook.md) |
| `--help` capture + diff workflow | [`docs/operator-runbook.md §1a`](operator-runbook.md) |
| Release-readiness checklist (10 items) | [`docs/operator-runbook.md §14`](operator-runbook.md) |

The `event="chain_params"` and `event="activation_state"` markers fire on **every
loop tick**, including monitor-only mode (confirmed by the
`loop_monitor_only_reads_activation_state_every_tick` test added in the Stage 10a
review fix).

---

## 8. Inconsistencies found and corrected

While cross-checking the [README.md](../README.md) status table against the
merged-stage history, this audit found one real inconsistency:

- Sub-stage milestone sections existed for **Stage 5.2** ([README.md:932](../README.md#L932))
  and **Stage 5.3** ([README.md:1200](../README.md#L1200)) — both with full body
  descriptions of work that landed on `main` — but **neither sub-stage had a
  status-table row**. By contrast, the equally fine-grained Stage 9c.1 and Stage 10a
  sub-stages did have rows.

**Correction landed in the same commit as this audit doc** ([README.md:101-102](../README.md#L101-L102)):
two new rows added between Stage 5 and Stage 6, both labelled **Complete**, with
short descriptions matching the pattern used by the surrounding rows:

```
| Phase 5 Stage 5.2 — Client-local staleness / retry policy …  | omni-zkml | Complete |
| Phase 5 Stage 5.3 — End-to-end attestation orchestration …  | omni-zkml | Complete |
```

No other inconsistencies surfaced.

---

## 9. CI workflow structure

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) defines exactly the
five jobs Stage 9c specified:

1. `default-build-test`
2. `default-tree-check` (hard gate)
3. `submit-build-test`
4. `submit-tree-check` (hard gate)
5. `stage6-fixture-check` (hard gate)

All five run unconditionally on every push to `main` and every PR targeting
`main`; no secret-gating, no skip branch, no deploy-key / PAT machinery.

---

## 10. Known deferrals (intentional)

These are documented absences at this audit point, not gaps to fix here:

| Deferral | Reason | Next checkpoint |
|---|---|---|
| Stage 10b: release-artifact GitHub workflow + signing | No external operator audience expecting downloadable binaries yet; Stage 10a kept release prep docs-only by design | Stage 10b plan when operator demand materialises |
| Stage 8+: zkML proof generation + SUM Chain tokenomics | Out of scope for Phase 5 RC; the proof-generation / staking-economy half of the protocol begins after Phase 5 closes | Stage 8+ plan when Phase 5 is fully closed |
| Typed `ChainClientError` taxonomy | [`crates/omni-zkml/src/error.rs`](../crates/omni-zkml/src/error.rs) currently has a single `Other(String)` variant; fee / balance / transport / signature failures all funnel through one opaque string. Stage 10a documented this honestly in [runbook §11a](operator-runbook.md) rather than silently fixing it | Candidate for a follow-up stage if operator feedback shows the parsing burden is real |
| Metrics backend (Prometheus / OTLP) | Existing tracing fields are already key=value structured; no proven operator need for a metrics endpoint yet | Stage 10b+ consideration |
| `--json` output mode for operator commands | Same reasoning as metrics; doubles the contract surface for unproven value | Stage 10b+ consideration |
| Daemonization / systemd code | Sample systemd unit lives in [`docs/operator-runbook.md §12`](operator-runbook.md) as docs only; OmniNode ships no installer / service-manager binary | No checkpoint planned |

---

## 11. What this audit explicitly does NOT change

Per the approved audit plan (verbatim non-edit list):

- [`docs/mainnet-smoke-audit.md`](mainnet-smoke-audit.md) — immutability rule.
- [`crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json`](../crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json) — Stage 6 fixture.
- [`crates/omni-zkml/src/chain_wire.rs`](../crates/omni-zkml/src/chain_wire.rs) — Stage 6 wire bytes.
- [`crates/omni-sumchain/src/tx.rs`](../crates/omni-sumchain/src/tx.rs) — Stage 7b transaction construction.
- [`crates/omni-sumchain/src/outer_sign.rs`](../crates/omni-sumchain/src/outer_sign.rs) — Stage 7b outer-sign helper.

Additionally, this audit makes **no protocol-level changes**: no chain RPC
surface change, no `ChainClientError` taxonomy change, no behaviour change
anywhere. The single in-PR repo edit is the Stage 5.2 / 5.3 status-table row
correction documented in §8.

---

## 12. Audit summary

Phase 5 is in a release-candidate-ready state at commit `fcde900`:

- Mainnet submit path proven end-to-end against real on-chain bytes (Stage 9c.1
  hash gate green).
- Build-risk loop closed (Stage 9c public crates; fresh source builds need no
  credentials).
- CI gates restored and green on every push / PR (five hard gates).
- Operator surface usable (Stage 10a: summary subcommand, stable log markers,
  failure-triage matrix, release-readiness checklist).
- Stage 6 byte-stability preserved from baseline `509f7fd` to HEAD `fcde900`.

After this audit lands, the next implementation choice is between **Stage 10b**
(release-artifact automation + signing — if operator demand for downloadable
binaries materialises) and **Stage 8+** (zkML proof generation + SUM Chain
tokenomics — the second half of the protocol). Both are intentionally out of
scope for Phase 5 RC.
