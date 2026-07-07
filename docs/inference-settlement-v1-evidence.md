# InferenceSettlement v1 — Evidence Bundle and Terminology

Chain-team-facing evidence bundle + canonical OmniNode terminology for the
dormant `InferenceSettlement` v1 subprotocol landed in `sum-chain#76`
(dispute nuance from `sum-chain#86` folded in).

**Docs-only.** Zero source / Cargo / `Cargo.lock` / CI /
release-workflow / schema / enum / proof-artifact / contributor-schema /
`error.rs` changes. No `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` mutation.
No activation params. Immutable audit docs
([`phase5-rc-audit.md`](phase5-rc-audit.md),
[`phase5-rc-audit-2026-06-24.md`](phase5-rc-audit-2026-06-24.md)) are
not edited.

Owns the InferenceSettlement track's terminology contract. Every
downstream OmniNode issue in the track (see §9) links here as the
canonical source of the words we use for the concepts.

---

## 1. Dormant posture (up front)

- **SUM Chain `InferenceSettlement` v1 is a dormant / activation-gated
  subprotocol.** It exists on chain in a dormant state and does not
  clear rewards, disburse escrow, or run dispute resolution until
  chain-team activation.
- **Exact activation parameter (verbatim):**
  ```
  inference_settlement_enabled_from_height: Option<u64>
  ```
  The subprotocol becomes active at the block height inside the
  `Some(_)` when the current head reaches that height. Any `None`
  reading or any `Some(N)` with `head < N` is **dormant**.
- **Do not claim mainnet-active settlement.** No OmniNode doc, error
  message, README section, release note, or dashboard may state or
  imply that settlement is active on mainnet unless
  `inference_settlement_enabled_from_height` on the mainnet chain is
  `Some(<past-height>)` and OmniNode has independently observed a head
  ≥ that height. **Dormancy detection is enforced locally** by the
  OmniNode client against chain params — chain read RPCs may still
  return empty / none state while the subprotocol is dormant; those
  empty results are not evidence of activation. See Issue #83.

---

## 2. Chain-side settlement v1 summary

- **Separate subprotocol from `InferenceAttestation` v1.**
  `InferenceSettlement` is a distinct on-chain subprotocol shipped by
  `sum-chain#76`. It does **not** live inside the attestation
  subprotocol; it references attestations but does not own them.
- **No mutation of the attestation schema.** In particular:
  - `InferenceAttestationDigest` — unchanged.
  - `InferenceAttestationTxData` — unchanged.
  - Attestation storage — unchanged.
  Attestations remain immutable once included on chain; settlement
  reads them by identity.
- **Attestation identity used by settlement:** the tuple
  `(session_id, verifier_address)`. This tuple uniquely identifies the
  attestation the verifier claim will reference.
- **Escrow-funded and supply-conserving.** Each settlement session is
  backed by a funder-provided escrow deposit. Rewards are paid **out
  of the session escrow**; nothing is minted; total supply is
  conserved. Unclaimed escrow returns to the funder via the refund
  path (see §3 eligibility rules and §9 open questions on the
  funder-refund race).
- **No unbounded rewards.** Every reward disbursement is bounded by
  the escrow balance remaining for the session; the runtime rejects
  claims that would over-draw.

---

## 3. Claim eligibility rules

A verifier claim (`ClaimReward` transaction) is accepted only when
**every** rule below holds. Any single failure fails the tx with a
typed reason; there is no partial claim.

1. **Settlement enabled.**
   `inference_settlement_enabled_from_height` is `Some(N)` and
   current head ≥ `N`.
2. **Session exists and is open.** The session referenced by the
   claim tx is registered on chain and is not in a closed / settled /
   fully-refunded terminal state.
3. **Matching attestation exists.** An `InferenceAttestation` on chain
   matches the tuple `(session_id, verifier_address)` where
   `verifier_address` equals the claim tx signer.
4. **Claim signer is the verifier address.** The transaction is signed
   by the same chain address that holds the attestation
   (`verifier_address` from the previous rule). **Chain v1 is
   verifier self-claim.** No coordinator-claim; no permissionless
   claim-on-behalf; no contributor-triggered tx path in v1. See
   Issue #79 for the OmniNode-side design of the claim flow and
   Issue #80 for the `ContributorResult`-signer ↔ chain-verifier
   identity mapping.
5. **Maturity — the claim window has opened.** Use either form
   verbatim:
   ```
   claim_ready_block = attestation.included_at_height
                     + finality_depth
                     + dispute_window_blocks
   ```
   or equivalently (finality already applied in the base term):
   ```
   claim_ready_block = finalized_at_height
                     + dispute_window_blocks
   ```
   The claim is accepted only when `head ≥ claim_ready_block`. Never
   write `tx_finalized_block + finality_depth + …` — that
   double-counts finality if `tx_finalized_block` already denotes
   finalized height.
6. **No duplicate claim.** A given verifier may claim once per
   `(session_id, verifier_address)`. A second submission from the
   same verifier for the same session is rejected as a duplicate.
7. **No unresolved / open or denied dispute.** If any dispute against
   the attestation is in an unresolved (open) state, the claim is
   refused pending resolution. If a dispute has been resolved
   **against** the claim (denied), the claim is permanently refused
   for that verifier — this is the **reward denial** path (§7). No
   bonded stake is taken.
8. **`claims_count < max_verifiers`.** The session's cap on
   simultaneously-eligible verifiers has not been reached. A claim
   arriving after the cap is refused even if all other rules pass.
9. **`remaining_escrow` is sufficient.** The session's escrow has
   enough balance to cover the reward this claim would disburse.
   Escrow-exhaustion is a refusal reason distinct from all the
   above — it is a **funder** condition, not a verifier fault.

---

## 4. Dispute nuance (from `sum-chain#86`)

`sum-chain#86` sharpened the dispute-resolution authority model.
OmniNode terminology + documentation must reflect the sharpening:

- **Disputes are validator-quorum authorized, not controlled by one
  resolver key.** There is no single "dispute resolver address"
  whose signature alone can flip a dispute outcome. Any earlier draft
  language that implied a solo resolver is superseded by this section.
- **`ResolveDispute` requires validator approvals reaching
  `inference_settlement_dispute_threshold_bps` over the active PoA
  validator set.** The threshold is expressed in basis points (bps)
  over the active validator set at the block the resolution is
  applied. Reaching threshold requires enough distinct validator
  approvals.
- **Non-signing validators count in the denominator as abstentions.**
  The threshold is computed against the full active validator set,
  not against a subset of "engaged" validators. Non-signing
  validators do not count as approvals and do not count as
  affirmative opposition or vetoes. Because the threshold is
  computed over the full active validator set, enough abstentions
  can still prevent quorum from being reached.
- **`tx.from` is the fee payer, not the authority.** The transaction
  submitter pays the fee to broadcast the `ResolveDispute` tx; the
  authority comes from the aggregated validator signatures the tx
  carries. A tx submitted by any address can succeed if it carries
  sufficient validator approvals; a tx submitted by a "resolver
  address" without sufficient validator approvals cannot succeed.

OmniNode consequences (informational; owned by Issue #81):

- OmniNode does **not** operate a solo resolver key in v1. There is
  no operational role for OmniNode to "be the resolver."
- If OmniNode ever writes to the dispute-resolution path, it is as
  one validator among many in a PoA set — governed by the
  validator-registration process, not by an OmniNode-owned
  configuration flag.
- Runbook language must reflect the quorum model. Any earlier
  runbook or design draft that named a resolver key is superseded
  when Issue #81 lands.

---

## 5. OmniNode stance today

- **No OmniNode source / client code consumes settlement yet.** The
  chain-side subprotocol is dormant; the OmniNode-side surface is
  intentionally empty until Issue #83 lands the read-only client.
- **No OmniNode code signs settlement claim txs yet.** No key
  material — new or existing — is wired to sign `ClaimReward`, and
  no operator-facing surface exposes such a signing path today. Issue
  #87 lands the write path behind `--features settlement-submit`,
  activation-gated, and only after Issues #79 + #80 have merged with
  chain-team acks.
- **No OmniNode claim submission today.** `omni-node operator …`
  has no settlement subcommand tree today; Issue #84 lands the
  read-only tree behind `--features settlement-read`.
- **No OmniNode resolver operation.** Following §4, OmniNode does
  not run a solo resolver key. Issue #81 records the resolver-ops
  posture (yes / no / undecided) — implementation, if any, is
  separate work.
- **No OmniNode dashboard support yet.** Issue #85 lands the
  baseline `event=` markers + runbook §7 recipe.
- **This issue (evidence + terminology) is docs-only.** No code, no
  Cargo, no CI, no release-workflow, no schema mutation. The
  contract this doc establishes is the terminology + posture that
  every downstream OmniNode issue in the track must honor.

---

## 6. Canonical terminology

Every downstream OmniNode issue, doc, error message, and code
comment in the InferenceSettlement track uses these words for these
concepts. Verbatim.

| Term | Definition (canonical) |
| --- | --- |
| **Session** | A settlement scope on chain. Registered by the funder with an escrow deposit. Identified by `session_id`. References attestations by tuple `(session_id, verifier_address)`. |
| **Contributor** | The role that performs the inference work off-chain and produces a `ContributorResult`. In OmniNode today the `ContributorResult` is signed by the contributor's seed. The chain-side identity mapping to a `verifier_address` is designed in Issue #80. |
| **Verifier** | The chain-address role that holds an `InferenceAttestation` and is authorized to submit a `ClaimReward` for the corresponding `(session_id, verifier_address)`. **Chain v1 is verifier self-claim** — only the verifier signs the claim tx. |
| **Reward** | The payout to a verifier for a valid claim against an attested session. Paid out of the session escrow. Supply-conserving. Bounded by `remaining_escrow`. |
| **Claim** | The `ClaimReward` transaction a verifier submits to receive their reward. Subject to the nine eligibility rules in §3. |
| **Dispute** | A challenge lodged against an attestation. Resolution is validator-quorum authorized per §4 — not a solo-resolver decision. |
| **Resolver / validator-quorum dispute authority** | The authority that resolves disputes. **In v1 this is not a single address.** It is validator quorum over the active PoA validator set, weighted by `inference_settlement_dispute_threshold_bps`. Every OmniNode use of the word "resolver" must be qualified with "quorum" or "validator-quorum" so the solo-resolver mental model is never inadvertently reintroduced. |
| **Funder** | The address that opens a session by depositing escrow. Rewards are paid from this escrow; the funder may reclaim unspent escrow via **refund** (subject to whatever expiry policy chain-team chooses; see §9). |
| **Refund** | The path by which unspent session escrow returns to the funder. **Open funder-refund vs. unclaimed-mature-reward race** owned by Issue #79. |
| **`included_at_height`** | The chain height at which the attestation tx was included in a block. Base term of the two-part maturity formula. |
| **`finality_depth`** | The number of blocks after `included_at_height` at which the attestation is considered finalized under the chain's finality rules. |
| **`finalized_at_height`** | Equivalent to `included_at_height + finality_depth`. When docs use this term, they must not also add `finality_depth` again — that double-counts finality. |
| **`dispute_window_blocks`** | The number of blocks after finalization during which disputes can be lodged. The claim cannot mature before this window closes. |
| **`claim_ready_block`** | The first block at which a claim is eligible. `attestation.included_at_height + finality_depth + dispute_window_blocks`, or equivalently `finalized_at_height + dispute_window_blocks`. Never `tx_finalized_block + finality_depth + …`. |
| **Reward denial** | The outcome when a verifier's claim is refused — because a dispute resolved against the claim, or because the verifier failed to submit before some deadline, or because any §3 rule failed. **No bonded stake is taken.** The verifier simply does not receive the reward for that claim. |
| **Slashing** | Taking **bonded stake** from a registered participant as penalty. **Distinct from reward denial.** Requires a bond to have been posted; v1 has no bonds. See §7 and Issue #82 tracker. |

---

## 7. Reward denial ≠ slashing (explicit)

- **Reward denial is not slashing.** Reward denial is inherent to v1:
  a verifier's claim can fail any §3 eligibility rule, or a dispute
  can resolve against them — in either case the reward for that
  session's attestation is not paid. **No stake is taken. No bond
  is confiscated.** The verifier's balance is not reduced.
- **v1 has no verifier bond.** There is no on-chain bonded stake tied
  to a verifier identity in v1. There is no verifier registry in v1.
- **Bonded slashing is v1.1 / v2 and requires verifier bonds +
  verifier registry.** Any future proposal to take stake as
  penalty — whether via `sum-chain` or OmniNode-side tooling —
  requires **both** a verifier registry and posted verifier bonds
  to exist first. This is enforced as a track-wide invariant:
  - No OmniNode doc, error message, or code path in v1 uses the
    word "slashing" to describe a reward denial outcome.
  - No OmniNode code path in v1 exposes any "slash verifier"
    capability. Any such surface, when it eventually exists, is
    gated on the verifier-registry and verifier-bond issues in
    Issue #82's deferred tracker.

Track-wide constraint: this distinction is enforceable via grep on
OmniNode-side PR bodies + docs. If a PR touching the settlement
track uses "slashing" without also introducing a verifier bond
surface, it is not merged.

---

## 8. Reviewer-flagged deferred non-goals

The following items are **explicitly not in v1** and are tracked in
Issue #82 (Settlement: deferred v1.1 / v2 design tracker). They are
listed here so every downstream OmniNode issue in the track knows
what it must not silently claim to deliver.

- **Consistency / plurality reward mode** (v2 candidate). Multi-verifier
  sessions and multi-contributor reward-split rules.
- **Verifier registry** (v1.1 candidate). On-chain verifier identities
  + eligibility.
- **Verifier bonds** (v1.1 candidate). Bonded stake tied to registered
  verifiers. Prerequisite for bonded slashing.
- **Bonded slashing** (v1.1 candidate). Taking bonded stake as penalty
  — distinct from reward denial. See §7.
- **Sponsored attestation** (v2 candidate). Third-party funding of
  settlement sessions. Requires race + griefing analysis first.
- **Permissionless settle-on-behalf** (v2 candidate). Any party can
  trigger settle for any session. Requires race + griefing analysis
  first.
- **Auto-claim daemon / safe-expiry policy** (v1.1 candidate). Ties to
  Issue #79's funder-refund vs. unclaimed-mature-reward race.

---

## 9. Open questions for downstream issues

Each of these is owned by a specific downstream issue; this doc surfaces
them but does not resolve them.

- **Claim-flow design / funder-refund vs. unclaimed-mature-reward
  race.** How does the funder-refund path interact with an eligible
  verifier's claim window? Does the chain provide a "safe expiry" that
  blocks refund until eligible claims are drained? If not, does
  OmniNode ship an auto-claim daemon (v1.1 candidate — Issue #82) or
  document the race and rely on manual verifier action? **Owned by
  Issue #79.**
- **`ContributorResult` signer vs. chain verifier identity
  mapping.** Are the two the same key (Case A) or distinct (Case B)?
  If distinct, how does OmniNode resolve a `ContributorResult` to
  the chain `verifier_address` that must sign the claim? **Owned by
  Issue #80.**
- **Dispute operations / runbook.** Given the validator-quorum model
  (§4), OmniNode does not run a solo resolver key. Does OmniNode
  operate as a PoA validator? Does OmniNode publish a runbook for
  operators who participate in dispute resolution as validators?
  **Owned by Issue #81.**
- **Read-only client integration.** How does the OmniNode client
  fetch `inference_settlement_enabled_from_height`, session state,
  claim state, and dispute state — with locally-enforced dormancy
  detection? **Owned by Issue #83.**
- **Operator CLI integration.** How does `omni-node operator
  settlement …` expose the read-only surface? What subcommand shape?
  What structured `event=` markers? **Owned by Issue #84.**

Additional cross-cutting downstream issues that touch these open
questions: Issue #85 (observability recipe), Issue #86 (runbook +
README rollup), Issue #87 (verifier self-claim submission — blocked
on #79 + #80 + chain-team activation).

---

## 10. Out of scope

Explicitly excluded from this issue and this doc:

- **No source changes.** Zero Rust code, no PyO3 code, no shell.
- **No `Cargo.toml` / `Cargo.lock` / `rust-toolchain.toml` changes.**
- **No CI changes.** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
  is not touched.
- **No release-workflow changes.**
  [`.github/workflows/release.yml`](../.github/workflows/release.yml)
  is not touched.
- **No immutable audit doc edits.**
  [`docs/phase5-rc-audit.md`](phase5-rc-audit.md) (2026-05-21) and
  [`docs/phase5-rc-audit-2026-06-24.md`](phase5-rc-audit-2026-06-24.md)
  are not edited.
- **No `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` mutation.** `&[]`
  preserved.
- **No activation params.** OmniNode does not set, propose, or
  advocate for a specific value of
  `inference_settlement_enabled_from_height` in this doc. Activation
  is a chain-team governance decision.
- **No `omni-zkml::error::*` mutation.**
- **No mutation of the ProofArtifactBody / ContributorResult /
  ContributorJob / InferenceAttestationDigest schemas.**
- **No cutting of a release tag.**
- **No creation of additional GitHub issues.** Every referenced issue
  above (#79, #80, #81, #82, #83, #84, #85, #86, #87) already exists
  from the approved plan.
- **No mutation of the InferenceSettlement track's existing filed
  issue bodies** beyond routine cross-linking (which is not done in
  this doc — this doc is standalone).

---

## Cross-references

- SUM Chain `sum-chain#76` (dormant `InferenceSettlement` v1).
- SUM Chain `sum-chain#86` (dispute nuance folded into §4 here).
- Stage 11d.3A / 11d.3B evidence bundle precedent:
  [`docs/stage11.d.3A-production-proof-eligibility-evidence.md`](stage11.d.3A-production-proof-eligibility-evidence.md),
  [`docs/stage11.d.3B-proof-eligibility-registry-alignment.md`](stage11.d.3B-proof-eligibility-registry-alignment.md).
- Stage 14.8 proof-generation readiness (precedent for locking
  dormant boundary language):
  [`docs/stage14.8-proof-generation-readiness.md`](stage14.8-proof-generation-readiness.md).
- OmniNode operator runbook:
  [`docs/operator-runbook.md`](operator-runbook.md).
