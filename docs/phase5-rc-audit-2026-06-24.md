# Phase 5 — Release Candidate Audit (2026-06-24)

Point-in-time release-candidate audit for Phase 5 after Stage 14.8. This
document is **immutable** once landed — future audits are new dated files
(`phase5-rc-audit-YYYY-MM-DD.md`), not edits to this one. The convention
mirrors the prior audit at [`docs/phase5-rc-audit.md`](phase5-rc-audit.md)
(2026-05-21).

This audit is a **verification + gap-analysis pass**, not a feature stage.
No source code, CI workflow, `Cargo.toml`, schema, enum, or proof artifact
is modified by Stage 15.1. The 2026-05-21 audit is **not** edited; this is
its successor, written from scratch for the current HEAD.

---

## 1. Status

| Item | Value |
|---|---|
| HEAD commit | `2356e0cf4cf485a60a559dd0091c3cd6cbd25316` |
| HEAD subject | `Merge pull request #65 from SUM-INNOVATION/stage14.8-proof-generation-readiness` |
| Branch under audit | `main` |
| Working tree | clean at audit start |
| Latest CI run on `main` | run `28127489830`, conclusion `success`, headSha `2356e0c…`, created `2026-06-24T20:27:42Z` |
| CI ↔ HEAD match | **yes** — the green run is against the exact audit commit |
| Stages closed since prior audit | Stage 11d.2 (production-MLP class), Stage 11d.3A/B, Stage 12.16 → 12.26, Stage 13.0 → 13.10 + 13.x housekeeping, Stage 14.1 → 14.8 |
| Workspace version | `0.1.0` (`[workspace.package].version` in root `Cargo.toml`; unchanged since 2026-05-21) |
| Rust toolchain pin | `[toolchain] channel = "stable"`, components = `rustfmt`, `clippy`, `rust-src` ([`rust-toolchain.toml`](../rust-toolchain.toml)) |

CI is green for the SHA being audited. The audit covers the 135 commits
between `fcde900` (prior audit's HEAD) and `2356e0c` inclusive.

---

## 2. Scope

This audit rates each of the **nine production-readiness categories** from
the Stage 15.0 plan against the live state of `main` at `2356e0c`, plus a
short **delta table** for major stages added since the 2026-05-21 audit
(per Q2 of the Stage 15.0 approval).

| Category | Rating |
|---|---|
| 3a. Reproducible builds | **Shipped** (toolchain pin + locked Cargo + dep-tree gates) |
| 3b. Artifact signing / provenance | **Missing** — Stage 10b deferred, no release workflow |
| 3c. Release bundle contents | **Partial** — manual checklist in runbook §14; no committed release artifact spec |
| 3d. Operator config surface | **Partial** — env-var + CLI-flag contract documented; no committed example config |
| 3e. Proof-generation runtime requirements | **Shipped** (feature gates + `RUST_MIN_STACK` documented; per-stage docs + benchmark record) |
| 3f. Observability | **Shipped (tracing baseline)** — Stage 10a `event=` markers + Stage 12.x event-stream commands; no metrics, no JSON output |
| 3g. Upgrade / rollback | **Missing** — undocumented today |
| 3h. Evidence bundle index | **Missing** — per-stage docs exist; no single per-tag index |
| 3i. Failure-mode inventory | **Partial** — typed-error catalog in runbook §11; not packaged as operator-symptom-keyed index |

Net: **3 Shipped, 3 Partial, 3 Missing.** The three Missing items are the
natural Stage 15.2+ scope — see §6.

---

## 3. Category audit

### 3a. Reproducible builds — **Shipped**

| Sub-item | State | Evidence |
|---|---|---|
| Rust toolchain pinned | yes | [`rust-toolchain.toml`](../rust-toolchain.toml) — `channel = "stable"` (the CI runners honour this via `Show toolchain (honours rust-toolchain.toml)`) |
| Workspace `Cargo.lock` committed | yes | Tracked at workspace root |
| Default-build dep tree clean of prover deps | yes | CI tree-check `default tree — must NOT contain sumchain-(crypto\|primitives)` + a positive gate for `--features submit` + four prove/verify-tree positive gates assert pull-ins; default tree carries **zero** halo2 / pasta / `omni-proofs-halo2-*` / direct `rand_chacha` (re-confirmed `cargo tree -p omni-node --edges normal \| grep -E 'halo2_proofs\|pasta_curves\|omni-proofs-halo2'` → empty) |
| Build matrix documented per feature | yes | [`docs/operator-runbook.md`](operator-runbook.md) §1d, §14; per-stage docs (14.1, 14.5) document `--features halo2-reference-{verify,prove}`, `--features stage11d-production-{verify,prove}` |
| Determinism across host re-builds | yes (Stage 6 + Stage 11a + Stage 11d.2 fixture byte-stability gates) | `Stage 6 chain-wire + fixture — byte-stable` CI job + `Stage 11a proof pipeline fixture — byte-stable` CI job + Stage 11d.2 fixture commit |
| Workspace crates accounted for | yes — 11 crates: `omni-bridge`, `omni-contributor`, `omni-net`, `omni-node`, `omni-pipeline`, `omni-proofs-halo2-production-mlp`, `omni-proofs-halo2-reference`, `omni-store`, `omni-sumchain`, `omni-types`, `omni-zkml` | `grep -E '^name\s*=' crates/*/Cargo.toml` |

**Finding.** Reproducible-build hygiene is sound. No Stage 15.x action required here.

### 3b. Artifact signing / provenance — **Missing**

| Sub-item | State |
|---|---|
| GitHub Actions release workflow | none |
| `cargo-dist` / `goreleaser` / equivalent | none |
| SHA-256 published per release | manual — runbook §14 step 9 prescribes `sha256sum target/release/omni-node` pasted into release notes |
| Cryptographic signature (cosign / minisign / gpg / sigstore) | none — explicitly deferred at Stage 10a §14 ("Signing is deferred to Stage 10b") |
| SLSA-style build-provenance attestation | none |
| `SHA256SUMS` / `SHA256SUMS.sig` artifact | not produced |
| Release-tag → CI run → checksum linkage | manual via release-notes copy-paste |

**Finding.** This is the canonical Stage 10b gap. The 2026-05-21 audit
predicted it ("Stage 10b — Release artifact workflow + signing | Planned"
in [`README.md`](../README.md) row 55); Stage 14.x did not touch it.
Stage 15.2 candidate.

### 3c. Release bundle contents — **Partial**

| Sub-item | State |
|---|---|
| Per-tag binary catalog (which feature flags ship?) | implicit — runbook §14 covers default + `--features submit`; the Stage 14.x prove/verify feature builds are documented per-stage but not packaged into a release-bundle inventory |
| Per-binary `--version` capture | manual — runbook §14 step 8 |
| Per-binary `--help` snapshots | manual — runbook §1a |
| Fixtures hash record | partial — Stage 6 + Stage 11a + Stage 11d.2 fixtures pinned by gate; no per-release fixture-hash report |
| Runbook diff vs previous tag | not produced |
| Mainnet eligibility delta | not produced (audit-time; not release-time) |

**Finding.** A committed `docs/release-bundle-template.md` would close the
ambiguity around "what is a release". Sized after Stage 10b. Stage 15.3 or
later candidate.

### 3d. Operator config surface — **Partial**

| Sub-item | State |
|---|---|
| Env-var contract documented | yes — runbook §12 (sample systemd unit) enumerates `OMNINODE_SUMCHAIN_RPC_URL`, `OMNINODE_VERIFIER_SEED_HEX`, `RUST_LOG`; runbook §7a lists structured `event=` markers and required fields |
| CLI flag contract documented | yes — per-stage docs (14.1 / 14.5 / 14.6 / 14.7 / 14.8) + the Stage 14.8 CLI-drift-safety test pins the documented long-names by introspecting the `clap` tree |
| Committed example config (`.toml` / `.env` template) | **none on disk** — the runbook §12 systemd unit is markdown only; no `examples/operator-production.env` or equivalent |
| Precedence rule (env vs flag vs file) | not explicitly documented; in practice clap flags override env, and the runbook examples mix the two without a single source of truth |
| Verifier-seed handling guidance | yes — runbook §12 calls out `EnvironmentFile=-/etc/omninode/verifier-seed.env` over inline env var to keep the seed out of `ps aux` |

**Finding.** A short Stage 15.x slice could ship `examples/operator-production.env`
+ a runbook §14 cross-link without touching code. Low-risk; bounded scope.
Candidate for Stage 15.3.

### 3e. Proof-generation runtime requirements — **Shipped**

| Sub-item | State |
|---|---|
| Feature-flag matrix documented | yes — [`docs/stage14.8-proof-generation-readiness.md`](stage14.8-proof-generation-readiness.md) §2 |
| `RUST_MIN_STACK=67108864` requirement for production prover | yes — Stage 14.5 + 14.8 docs; CI `stage11d-production-prove-build-test` job exports it |
| Pinned `halo2_proofs = 0.3.2` | yes — [`crates/omni-proofs-halo2-production-mlp/Cargo.toml`](../crates/omni-proofs-halo2-production-mlp/Cargo.toml#L62) (also [`omni-proofs-halo2-reference/Cargo.toml`](../crates/omni-proofs-halo2-reference/Cargo.toml)) |
| Prover RNG seeds pinned per crate | yes — `PROVER_RNG_SEED` in each crate; byte-determinism tests pinned by `prove_canonical_is_byte_deterministic` per crate |
| Verifier drift detection at construction | yes — `Halo2ProductionMlpVerifier::from_embedded_fixtures` re-derives `circuit_id_hex` + `verification_key_hash_hex` and refuses on drift ([`crates/omni-proofs-halo2-production-mlp/src/verifier.rs:169-184`](../crates/omni-proofs-halo2-production-mlp/src/verifier.rs#L169-L184)) |
| Per-host benchmark | yes — [`docs/stage11d.2-benchmark-record.md`](stage11d.2-benchmark-record.md) for the production circuit |
| Mainnet refusal posture documented | yes — Stage 14.8 §2 table: reference 1+3+6, production 6-only, both empty-registry-gated |
| Stage 14.8 CLI-drift-safety test | yes — `operator::tests::stage_14_8_runbook_documented_operator_cli_flags_remain_registered` (introduced PR #65) |

**Finding.** Closed. No action required.

### 3f. Observability — **Shipped (tracing baseline); metrics/JSON deferred**

| Sub-item | State |
|---|---|
| Structured `event=` markers (Stage 10a contract) | yes — runbook §7a documents `startup`, `chain_params`, `activation_state`, `registry_summary` |
| Free-form markers (Stage 9a precedent) | yes — runbook §7b |
| Stage 12.x event-stream commands | yes — runbook §"Stage 12 — production posture" |
| Stage 13.x anchor-lifecycle observability | yes — runbook §13.3, §13.7, §13.8, §13.9 |
| Default `RUST_LOG` recipe | yes — runbook §7 |
| JSON log output | **not shipped** — Stage 10a says "tracing-only; JSON output is deferred" |
| Metrics emission (Prometheus / OpenMetrics) | **not shipped** |
| Dashboard recipes / alert thresholds | **not documented** |

**Finding.** Baseline is sound for shell-and-grep operations. JSON +
metrics + dashboards are a possible Stage 15.x slice but not a blocker —
operators today can run with `journalctl -u omninode-operator-loop -f`
+ `grep 'event="…"'`.

### 3g. Upgrade / rollback — **Missing**

| Sub-item | State |
|---|---|
| Upgrade runbook (`v0.1.0 → v0.x.y`) | **none** — workspace is `0.1.0` so no upgrade has happened in repo history; no documented procedure |
| Forward-compat: on-disk state survival across binary versions | implicit (Stage 13.x anchor registry has consistency reports; Stage 12.x contributor state-dir has auto-prune-on-open) but not collected into one upgrade procedure |
| Rollback procedure if v0.x.y regresses | **none** |
| Migration scripts | **none** |
| Schema-versioning policy | partial — proof artifact + contributor result use serde with `Option`-backwards-compat; no documented policy |

**Finding.** A non-trivial gap once v0.2.0 looms. Today, with a single
v0.1.0 in repo, the practical impact is zero. Stage 15.x candidate.

### 3h. Evidence bundle index — **Missing**

| Sub-item | State |
|---|---|
| Per-tag evidence-bundle index | **none** |
| Per-dimension evidence map (proof gen, anchor registry, contributor protocol, observability, RC audit) | partial — readers must walk per-stage docs |
| Stage 14.8 §1 inventory | yes (as in-stage doc), but not a tag-level rollup |
| Stage 11d.3A evidence bundle | yes (proof eligibility dimension only) |
| RC audit | yes — this doc + 2026-05-21 doc |

**Finding.** A small `docs/evidence-bundle-index.md` would resolve this
without touching code. Candidate for Stage 15.x.

### 3i. Failure-mode inventory — **Partial**

| Sub-item | State |
|---|---|
| Typed `OperatorError` catalog | yes — runbook §11 documents each variant + remediation |
| Layered mainnet refusal table | yes — runbook §11a covers all six `MainnetRefusalReason` layers |
| Operator-symptom-keyed lookup (logs → cause → fix) | **partial** — the per-error remediation column is symptom-adjacent but not symptom-keyed |
| Contributor subsystem failure modes | partial — runbook §"Stage 12.x" + per-stage 12.x docs |
| Anchor-registry failure modes | yes — runbook §13.x sections cover consistency-report findings |
| Proof-generation failure modes | yes — Stage 14.7 acceptance umbrella tests pin the cross-family refusal posture |

**Finding.** Close. A "what does an operator see, and what should they
do?" rewrite of runbook §11 keyed by stdout/log signature rather than
typed-error name would close it. Low-priority; no code action.

---

## 4. Delta table — major stages added since 2026-05-21

Reference: the 2026-05-21 audit's stage-row table at
[`docs/phase5-rc-audit.md`](phase5-rc-audit.md) §2 records the state at
`fcde900`. The table below adds the rows that have **landed since** at
`2356e0c`. Stages already present in the prior audit and not modified are
not repeated.

| Stage | Capability delivered | Status at `2356e0c` | Evidence doc |
|---|---|---|---|
| 11d.1 | Structured `AllowlistEntry` eligibility registry schema + empty-by-design constants | Complete (was Complete pre-audit too; cited here because it's a load-bearing dependency for the 11d.3 work below) | inline in [`crates/omni-zkml/src/proof.rs`](../crates/omni-zkml/src/proof.rs) + [`docs/mainnet-eligibility-criteria.md`](mainnet-eligibility-criteria.md) |
| 11d.2 | Production fixed-point MLP proof class + verifier + canonical spec + benchmark | Complete | [`docs/stage11d.2-review-packet-production-fixedpoint-mlp.md`](stage11d.2-review-packet-production-fixedpoint-mlp.md), [`docs/stage11d.2-benchmark-record.md`](stage11d.2-benchmark-record.md) |
| 11d.3A | Production proof eligibility evidence bundle (chain-team review packet) | Complete; dormant | [`docs/stage11.d.3A-production-proof-eligibility-evidence.md`](stage11.d.3A-production-proof-eligibility-evidence.md) |
| 11d.3B | Proof Eligibility Registry terminology + license normalization | Complete | [`docs/stage11.d.3B-proof-eligibility-registry-alignment.md`](stage11.d.3B-proof-eligibility-registry-alignment.md) |
| 11d.3C | OmniNode-side consumption of chain registry state | **Not started** — blocked on chain-side `CandidateRefused` record landing | — |
| 12.16–12.26 | Signed integrity baseline / diff / evidence bundle / evidence chain / signed reports / docs prefix | Complete | folded into [`docs/stage12-contributor-protocol.md`](stage12-contributor-protocol.md) (~1900 lines) |
| 13.0 | Chain anchoring for integrity evidence (stub + wire spec) | Complete | [`docs/stage13-evidence-anchor-spec.md`](stage13-evidence-anchor-spec.md) |
| 13.1 | Chain-team review packet + wire fixtures | Complete | [`docs/stage13.1-chain-adapter-review.md`](stage13.1-chain-adapter-review.md) |
| 13.2 | Real SUM Chain adapter for integrity-evidence anchors | Complete | [`docs/stage13.2-chain-adapter.md`](stage13.2-chain-adapter.md) |
| 13.3 | Operator hardening for anchor lifecycle (summary, watch) | Complete | [`docs/stage13.3-anchor-operations.md`](stage13.3-anchor-operations.md) |
| 13.4 | Anchor-registry cleanup with quarantine and restore | Complete | [`docs/stage13.4-anchor-cleanup.md`](stage13.4-anchor-cleanup.md) |
| 13.5 | Local-only anchor export with portable manifest | Complete | [`docs/stage13.5-anchor-export.md`](stage13.5-anchor-export.md) |
| 13.6 | Anchor export import / registry restore | Complete | [`docs/stage13.6-anchor-import.md`](stage13.6-anchor-import.md) |
| 13.7 | Local terminal-anchor archive and restore | Complete | [`docs/stage13.7-anchor-archive.md`](stage13.7-anchor-archive.md) |
| 13.8 | Local integrity-evidence-anchor consistency report | Complete | [`docs/stage13.8-anchor-consistency-report.md`](stage13.8-anchor-consistency-report.md) |
| 13.9 | SUM Chain read/reconcile integration | Complete | [`docs/stage13.9-chain-read-reconcile.md`](stage13.9-chain-read-reconcile.md) |
| 13.10 | Operator acceptance umbrella for the 13.x track | Complete | [`docs/stage13.10-acceptance.md`](stage13.10-acceptance.md) |
| 13.x housekeeping | Pre-existing clippy cleanup + Stage 6 byte-stability honoring | Complete | merged via PR #57 (commits `8cf6bbe`, `603f6a3`) |
| 14.1 | Halo2 reference operator prover | Complete | [`docs/stage14.1-halo2-reference-prove.md`](stage14.1-halo2-reference-prove.md) |
| 14.2 | Reference contributor StubRunner sidecar | Complete | [`docs/stage14.2-contributor-halo2-reference-proof.md`](stage14.2-contributor-halo2-reference-proof.md) |
| 14.3 | Reference contributor ExternalCommandRunner sidecar | Complete | [`docs/stage14.3-external-command-runner-halo2-reference-proof.md`](stage14.3-external-command-runner-halo2-reference-proof.md) |
| 14.4 | EZKL feasibility + **rejection** (license / supply-chain) | Complete; rejected | inline in [`docs/stage14.8-proof-generation-readiness.md`](stage14.8-proof-generation-readiness.md) §1 |
| 14.5 | Halo2 production-MLP operator prover | Complete | [`docs/stage14.5-halo2-production-mlp-prove.md`](stage14.5-halo2-production-mlp-prove.md) |
| 14.6 | Production-MLP contributor sidecar (Stub + External) | Complete | [`docs/stage14.6-contributor-production-mlp-proof.md`](stage14.6-contributor-production-mlp-proof.md) |
| 14.7 | Cross-family acceptance hardening (umbrella) | Complete | [`docs/stage14.7-proof-generation-acceptance-hardening.md`](stage14.7-proof-generation-acceptance-hardening.md) |
| 14.8 | Proof-generation track closure / readiness packet | Complete | [`docs/stage14.8-proof-generation-readiness.md`](stage14.8-proof-generation-readiness.md) |
| 15.0 | Phase 5 production packaging / RC audit discovery (plan-only) | Plan accepted; implementation in this doc | this doc |
| 15.1 | Refreshed Phase 5 RC audit (this doc) | **In flight — this doc** | this doc |

**Stages explicitly NOT shipped since the prior audit:**
- Stage 10b (release artifact workflow + signing) — still Planned per README row 55.
- Stage 11d.3C (Proof Eligibility Registry consumption) — blocked on the chain-side `CandidateRefused` record landing first.
- Chain-side SNARK verification — out of scope for v1 (register-only architecture per Stage 11d.3A §8).
- Economics / staking / slashing / rewards — paper track only per Stage 15.0 §5.

---

## 5. Test matrix results

Captured locally on the audit host on 2026-06-24 against `2356e0c`.
Counts are sums of every `test result: ok. <N> passed` line in each run
(unit tests + integration tests + doctests counted; ignored doctests
counted as 0). All runs **passed** with zero failures.

| Crate / feature flags | Total passed |
|---|---|
| `omni-zkml` (default) | 477 |
| `omni-contributor` (default) | 618 |
| `omni-proofs-halo2-reference` (`--features verify --features prove`) | 85 |
| `omni-proofs-halo2-production-mlp` (`--features verify --features prove`, `RUST_MIN_STACK=67108864`) | 58 |
| `omni-node` (default) | 197 |
| `omni-node` (`--features submit`) | 216 |
| `omni-node` (`--features halo2-reference-prove`, `RUST_MIN_STACK=67108864`) | 227 |
| `omni-node` (`--features stage11d-production-prove`, `RUST_MIN_STACK=67108864`) | 217 |

CI-side matrix (run `28127489830` on `main` HEAD `2356e0c`) passed 15
distinct jobs (verbatim from `gh pr checks 65`):
- `default (read-only) — build + test`
- `--features submit — build + test`
- `--features halo2-reference-verify — verifier-only build + test`
- `--features halo2-reference-prove — prover-and-verifier build + test`
- `--features stage11d-production-verify — verifier-only build + test`
- `--features stage11d-production-prove — prover-and-verifier build + test`
- `omni-contributor — default-features build + test`
- `default tree — must NOT contain sumchain-(crypto|primitives)`
- `submit tree — MUST contain sumchain-(crypto|primitives)`
- `halo2-reference-verify tree — must contain halo2_proofs`
- `halo2-reference-prove tree — must contain halo2_proofs + rand_chacha`
- `stage11d-production-verify tree — must contain halo2_proofs + production-mlp`
- `stage11d-production-prove tree — must contain halo2_proofs + rand_chacha + production-mlp`
- `Stage 11a proof pipeline fixture — byte-stable`
- `Stage 6 chain-wire + fixture — byte-stable`

---

## 6. Findings → Stage 15.x sizing

Summary of the gaps surfaced by §3:

1. **Stage 10b (release artifact workflow + signing)** — Missing. Direct gap that fixes §3b in full and §3c substantially. Sized at "new `.github/workflows/release.yml` + new `docs/stage10b-release-artifact-workflow.md` + small runbook §14 cross-link". Signing scheme TBD (cosign keyless OIDC vs minisign vs gpg). Candidate Stage 15.2.
2. **Upgrade / rollback runbook** — Missing (§3g). Sized at "new runbook section + cross-link from §14 release checklist". Low-priority while workspace is single-version. Candidate Stage 15.4 or later.
3. **Evidence bundle index** — Missing (§3h). Sized at a single new `docs/evidence-bundle-index.md`. Candidate Stage 15.5.
4. **Operator config template** — Partial (§3d). Sized at `examples/operator-production.env` + runbook §12 cross-link. Candidate Stage 15.3.
5. **Failure-mode inventory rewrite** — Partial (§3i). Sized at a runbook §11 rewrite keyed by operator symptom rather than typed-error name. Low-priority; backlog candidate.
6. **Observability extension** — Shipped today (§3f); JSON/metrics/dashboards are a *possible* slice but not blocking.

The audit does **not** prescribe a specific Stage 15.2 — the recommendation
sits in the Stage 15.0 plan (§6 of [`docs/stage15.x` — see runbook for the
pointer once 15.0 lands as docs]). Most likely Stage 15.2 = Stage 10b
revival per (1) above.

---

## 7. On-chain reference

This audit is **off-chain**. No chain RPC call was issued during the audit
run. No Proof Eligibility Registry record was consulted, mirrored, or
queried. The auditor verified the empty-by-design constants exist:

```rust
// crates/omni-zkml/src/proof.rs (verified by inspection at 2356e0c)
pub const MAINNET_APPROVED_PROOF_SYSTEMS: &[ProofSystem] = &[];
pub const MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES: &[AllowlistEntry] = &[];
```

Both constants remain empty at audit HEAD, consistent with Stage 11d.3A's
recorded mainnet posture and Stage 14.8's readiness checklist.

---

## 8. Inconsistencies found

None.

The prior audit (2026-05-21 §8) recorded one inconsistency
("Stage 5.2 + 5.3 row addition" to the README status table) and corrected
it. This audit found no analogous discrepancy: the README, the runbook,
the per-stage Stage 13.x and 14.x docs, and the Stage 11d.3A/B evidence
docs all agree on:

- Empty `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` and
  `MAINNET_APPROVED_PROOF_SYSTEMS`.
- Reference family permanent testnet/dev posture.
- Production family layer-6-only mainnet refusal.
- EZKL rejection (Stage 14.4).
- Stage 10b deferral.
- Stage 11d.3C blocking dependency on chain-side `CandidateRefused`
  record.

No correction PR is produced by this audit.

---

## 9. CI workflow structure

Single file: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
Jobs (run on push + PR to `main`):

- `default-build-test` — workspace + omni-node default build + tests.
- `default-tree-check` — `cargo tree` assertion that the default graph
  pulls no `sumchain-(crypto|primitives)` and no
  `halo2_proofs` / `pasta_curves` / `omni-proofs-halo2-*` / direct prover
  `rand_chacha`.
- `submit-build-test` — `--features submit` build + tests.
- `submit-tree-check` — `cargo tree` positive gate for submit deps.
- `contributor-build-test` — `omni-contributor` default-features.
- `halo2-reference-verify` build + test + tree-check.
- `halo2-reference-prove` build + test + tree-check.
- `stage11d-production-verify` build + test + tree-check.
- `stage11d-production-prove` build + test + tree-check (`RUST_MIN_STACK=67108864`).
- `Stage 6 chain-wire + fixture — byte-stable` — `git diff` gate on
  the Stage 6 fixture.
- `Stage 11a proof pipeline fixture — byte-stable` — `git diff` gate on
  the Stage 11a fixture.

**No release workflow.** **No publish/tag automation.** **No
provenance / signing steps.** (Stage 10b gap, surfaced in §3b.)

---

## 10. Known deferrals (intentional)

| Deferral | Reason | Stage that owns it |
|---|---|---|
| Stage 10b release artifact + signing | Plan deferred; not yet sized | Stage 15.2 candidate |
| Proof Eligibility Registry consumption | Blocked on chain-side `CandidateRefused` record | Stage 11d.3C+ |
| Chain-side SNARK verification | v1 register-only architecture per Stage 11d.3A §8 | Possibly never, possibly later stage |
| Economics / staking / slashing / rewards implementation | Paper track only | Stage 15.x paper track |
| EZKL revisit | License rejected per Stage 14.4 | None until upstream license changes |
| JSON log output | Stage 10a deferred | Not yet sized |
| Metrics emission | Not in scope for Phase 5 baseline | Not yet sized |
| Upgrade / rollback runbook | Not needed at workspace v0.1.0 | Stage 15.x candidate (post-Stage-10b) |
| README status table for Stage 15.0 | Stage 15.0 is plan-only per Q6 of approval | Stage 15.1 landed → consider adding row for 15.0 + 15.1 in a future cleanup |

---

## 11. What this audit explicitly does NOT change

- No source code modified.
- No CI workflow modified.
- No `Cargo.toml` or `Cargo.lock` modified.
- No schema / enum / proof artifact / contributor schema modified.
- No `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` mutation; no eligibility activation; no `Active` record.
- No chain-side change.
- The 2026-05-21 audit at [`docs/phase5-rc-audit.md`](phase5-rc-audit.md) is **not edited** (per Stage 15.0 approval Q6: "no edits to the old 2026-05-21 audit except maybe a pointer if absolutely necessary; prefer leaving it untouched"). This audit links **to** the prior one; the prior one does not need to link back.
- No README status-table row for Stage 15.0 / 15.1 (Q6).
- No new test, no new fixture, no new dependency.
- No EZKL revisit.

---

## 12. Audit summary

**Verdict: Phase 5 surface is functionally complete for off-chain proof
generation and verification.** The Stage 14.x track is closed and the
Stage 11d.3A/B evidence trail is landed. The three Missing categories
(§3b, §3g, §3h) are all packaging / release-readiness concerns rather
than functional gaps. Stage 10b (release artifact signing) is the
single highest-leverage Stage 15.2 candidate.

**Verdict: Empty registries remain empty.** No Stage 14.x or Stage
11d.3A/B work mutated the eligibility registry. Production proof family
remains dormant at layer 6 of `check_mainnet_eligible`; reference family
remains testnet/dev only at layers 1+3+6.

**Verdict: CI ↔ HEAD match holds.** The audit's own SHA `2356e0c` has a
green CI run `28127489830`.

**Verdict: No discrepancies surfaced; no correction PR produced.**
