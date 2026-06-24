# Stage 11d.3B — Proof Eligibility Registry terminology + dormant-policy alignment

**Status: docs + cosmetic rename only.** No behavior change. No eligibility
activation. No `Active` record anywhere. `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`
and `MAINNET_APPROVED_PROOF_SYSTEMS` remain empty (`&[]`). All Stage 11d.x and
Stage 14.x acceptance tests pass byte-equivalent.

This stage updates project terminology to match the SUM Chain
**Proof Eligibility Registry** framing that chain team locked in at
`sum-chain#21`, and normalizes the copyright attribution on
`LICENSE-APACHE` / `LICENSE-MIT`.

## Chain-team confirmation summary

Chain team landed a dormant subprotocol design package at `sum-chain#21`:

- [`docs/SUBPROTOCOLS/PROOF-ELIGIBILITY-REGISTRY.md`](https://github.com/SUM-INNOVATION/sum-chain/blob/main/docs/SUBPROTOCOLS/PROOF-ELIGIBILITY-REGISTRY.md)
- [`docs/SUBPROTOCOLS/PROOF-ELIGIBILITY-REGISTRY-ACTIVATION.md`](https://github.com/SUM-INNOVATION/sum-chain/blob/main/docs/SUBPROTOCOLS/PROOF-ELIGIBILITY-REGISTRY-ACTIVATION.md)

Locked facts:

- **Use Proof Eligibility Registry, not allowlist.** The chain-side mechanism
  is a registry of proof-profile records, not a binary allow-list.
- **Append-only and superseding.** Records are never mutated in place;
  deactivation is a superseding record.
- **First `Stage11dProductionFixedPointMlp` record must be `CandidateRefused`,
  not `Active`.** Recording the candidate identity tuple creates the
  review trail without granting eligibility.
- **v1 is register-only.** SUM Chain admits/refuses by exact
  proof-profile identity match only. The chain does **not** verify SNARK
  proof correctness for this profile in v1.
- **OmniNode owns proof correctness.** Proof generation
  (`Halo2ProductionMlpProofBackend`) and proof verification
  (`Halo2ProductionMlpVerifier`, with construction-time VK drift
  detection and per-artifact metadata enforcement) remain on the OmniNode
  side. Validators do not run halo2 verification in v1.
- **`proof_eligibility_enabled_from_height: Option<u64> = None` is dormant
  by default.** No merge-time activation. No `Active` record exists at
  Stage 11d.3B close, on either side.

## Register-only v1 model

Recall the layered local refusal model in
[`crates/omni-zkml/src/proof.rs`](../crates/omni-zkml/src/proof.rs)
`check_mainnet_eligible` (six layers, layer 6 is the eligibility-registry
match on `(proof_system, circuit_id_hex, model_hash)`). The register-only v1
contract:

| Side | Responsibility |
| --- | --- |
| OmniNode | Generate proofs; verify proofs; enforce metadata contract; apply six-layer refusal locally; refuse anything not matching an eligible profile. |
| SUM Chain | Hold the **Proof Eligibility Registry** — append-only records of `(proof_system, circuit_id_hex, model_hash)` profiles with a lifecycle (`CandidateRefused` → `CandidateApproved` → `Active` → `Superseded`). Admit / refuse profiles by exact identity match against the live registry. **Does not verify SNARK proofs in v1.** |

Activation is a chain-side record-state flip plus a height (governed by
`proof_eligibility_enabled_from_height`), not a SNARK verifier added to
consensus. Adding chain-side verification, if ever chosen, is a separate
stage with its own threat-model / validator-cost / audit requirements (see
Stage 11d.3A §8).

## CandidateRefused-first requirement

The first chain-side record for `Stage11dProductionFixedPointMlp` must be a
`CandidateRefused` record carrying the candidate identity tuple from
Stage 11d.3A §2. This intentionally:

- Creates an on-chain review trail without granting eligibility.
- Pins the candidate identity tuple in chain history for the inevitable
  comparison against any future `CandidateApproved` record.
- Matches OmniNode's current local posture: both sides refuse, both sides
  record the candidate.
- Decouples evidence-gathering from activation. Operator and chain-team
  reviewers can examine the on-chain record alongside the off-chain
  evidence bundle without anyone needing to flip eligibility live.

OmniNode-side does not consume the chain registry yet; consumption is the
Stage 11d.3C dependency below. At Stage 11d.3B close, the local-side
`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` constant remains empty regardless of
the chain-side record state.

## No chain-side proof verification in v1

Out of scope for the whole Stage 11d.3 family. The chain treats proof
artifacts opaquely; OmniNode's `Halo2ProductionMlpVerifier` is the
cryptographic enforcement point. Adding chain-side verification would
require:

- Third-party cryptographic audit of the halo2 circuit + canonical spec.
- Validator execution cost analysis on real validator hardware.
- A separate chain-side verifier crate with no `omni-zkml` dependency.
- A consensus-level commitment to `halo2_proofs = 0.3.2` or an audited
  equivalent.

None of those are required by the register-only v1 path Stage 11d.3 takes.

## Stage 11d.3C dependency

Stage 11d.3C will land OmniNode-side **consumption** of chain registry state:
either mirroring the registry into the local
`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`, or live-reading it at refusal
check-time. That work is blocked on the chain-side `CandidateRefused` record
existing first — without a record, there is nothing for OmniNode to consume.
Stage 11d.3B does not write any consumption code.

A future Stage 11d.3D-or-later may optionally rename the local-side
`MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` / `AllowlistEntry` /
`MainnetRefusalReason::NotInMainnetAllowlist` identifiers to match the
Proof Eligibility Registry terminology. Stage 11d.3B does **not** rename
those public-API surfaces — see "Terminology canonicalization" below.

## Terminology canonicalization

| Context | Canonical term |
| --- | --- |
| Named chain-side subprotocol | **Proof Eligibility Registry** (title case) |
| Local-side mirror / generic prose | **eligibility registry** (lowercase) |
| Public API identifiers (grandfathered) | `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`, `MAINNET_APPROVED_PROOF_SYSTEMS`, `AllowlistEntry`, `MainnetRefusalReason::NotInMainnetAllowlist`, `matches_structured_allowlist` — kept verbatim |
| Historical citations | "Stage 11b.0 allowlist empty by design" etc. — preserved as history; not retermed |
| Unrelated subsystem | `env_allowlist` (environment-variable passthrough for `ExternalCommandRunner`) — fully unrelated to mainnet eligibility; **untouched** |

Each definition site of the four grandfathered identifiers gains a one-line
doc-comment pointer to this document so future readers can navigate from the
old name to the registry framing without an identifier rename.

## License normalization

Canonical copyright attribution is `SUM INNOVATION INC` — all caps, no
period. Two files updated:

- [`LICENSE-APACHE`](../LICENSE-APACHE) line 178: `Copyright 2026 SUM Innovation Inc.` → `Copyright 2026 SUM INNOVATION INC`
- [`LICENSE-MIT`](../LICENSE-MIT) line 3: `Copyright (c) 2026 SUM Innovation Inc.` → `Copyright (c) 2026 SUM INNOVATION INC`

No other variants exist in the workspace. No `NOTICE` / `AUTHORS` file is
created. No per-file source-header copyright notices are added. Workspace
`license = "MIT OR Apache-2.0"` SPDX identifier is unchanged.

## Surface map

| Bucket | Files | Edit class |
| --- | --- | --- |
| License attribution | [`LICENSE-APACHE`](../LICENSE-APACHE), [`LICENSE-MIT`](../LICENSE-MIT) | one-line copyright update each |
| Cargo.toml header comments | [`crates/omni-contributor/Cargo.toml`](../crates/omni-contributor/Cargo.toml), [`crates/omni-proofs-halo2-production-mlp/Cargo.toml`](../crates/omni-proofs-halo2-production-mlp/Cargo.toml) | comment prose only |
| Test fn renames + grandfathered-ident pointers | [`crates/omni-zkml/src/proof.rs`](../crates/omni-zkml/src/proof.rs) | 5 test fn renames + 4 doc-comment pointers at definition sites |
| Source-comment cleanup | files under `crates/omni-zkml/`, `crates/omni-node/`, `crates/omni-proofs-halo2-{reference,production-mlp}/`, `crates/omni-contributor/`, `tools/halo2_production_mlp_regen/` | comment + doc-comment prose only; identifier citations preserved verbatim |
| Operator runbook | [`docs/operator-runbook.md`](operator-runbook.md) | prose reterm; pointer section pointing at this doc |
| Per-stage docs | `docs/mainnet-eligibility-criteria.md`, `docs/stage11d-mainnet-eligibility-FAQ.md`, `docs/stage11d-review-packet.md`, `docs/stage11d.2-review-packet-production-fixedpoint-mlp.md`, `docs/stage14.{1,2,3,5,6,7}-*.md`, `docs/stage12-contributor-protocol.md`, `docs/stage13.1-chain-adapter-review.md` | prose reterm; historical citations preserved |
| Stage 11d.3A bundle | [`docs/stage11.d.3A-…`](stage11.d.3A-production-proof-eligibility-evidence.md) | prose reterm in §2 / §6 / §8 / §10; code-identifier citations verbatim |
| This doc | new | engineering doc |

## Tests

No new behavior tests. 5 fn renames in
[`crates/omni-zkml/src/proof.rs`](../crates/omni-zkml/src/proof.rs) `mod tests`
(see surface map). Diff is `s/allowlist/eligibility_registry/` in fn names;
assertions unchanged. `cargo test -p omni-zkml` passes byte-equivalent.

Re-run unchanged CI gates:

- `cargo test -p omni-zkml`
- `cargo test -p omni-node --features halo2-reference-prove --bins`
- `cargo test -p omni-node --features stage11d-production-prove --bins`
- default-build dep-tree isolation gates

## Out of scope

- Activation of any proof system. No `Active` record.
- Adding a chain-side SNARK verifier.
- OmniNode-side consumption of chain registry state (deferred to
  Stage 11d.3C, blocked on the chain-side `CandidateRefused` record).
- Renaming the grandfathered public-API identifiers (deferred).
- Changing proof artifact metadata, public-inputs shape, or verifier /
  prover behavior.
- Changing contributor schemas, runner traits, or CLI contracts.
- Cargo dependency edits, CI workflow edits, feature-flag edits.
- Adding `NOTICE` / `AUTHORS` files or per-file source-header notices.
- EZKL revisit (still license-blocked per Stage 14.4).
- Reference proof family mainnet eligibility (still dev/testnet only).
- Touching unrelated `env_allowlist` semantics in
  [`crates/omni-contributor/src/runner.rs`](../crates/omni-contributor/src/runner.rs).

## Future outlook

- **Stage 11d.3C** — OmniNode-side consumption of chain registry state.
  Blocked on chain-side `CandidateRefused` record existing first.
- **Stage 11d.3D (optional housekeeping)** — rename the four grandfathered
  identifiers to match registry terminology. Touches every match site in
  operator.rs CLI tests + Stage 14.x acceptance tests + the Stage 11d.3A
  evidence bundle; cost-benefit reviewed separately.
- **A potential future stage (not numbered)** — chain-side proof
  verification, if ever chosen, with the audit / cost / format-commitment
  prerequisites enumerated above and in Stage 11d.3A §8.

Stage 11d.3B is intentionally minimal: terminology + license normalization +
a single new doc + comment cleanups. Behavior is byte-equivalent. The
chain-team review trail (the eventual `CandidateRefused` record + the
Stage 11d.3A evidence bundle) is the load-bearing artifact for any future
activation; this stage just makes the local-side prose consistent with the
chain-side framing.
